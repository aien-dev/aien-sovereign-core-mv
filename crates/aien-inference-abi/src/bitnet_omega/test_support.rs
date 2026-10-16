//! A tiny BitNet checkpoint (in memory or written as one `model.safetensors`), a recording
//! uploader with host matrices and its deliberate mutants, and the byte test, so the loader runs
//! end to end without a chip.

use std::path::Path;
use std::sync::{Arc, Mutex};

use super::*;
use crate::qwen3_dense_omega::test_support::RecordedShape;
use crate::qwen3_dense_omega::HostFp8Gemv;
use crate::qwen3_moe::{f32_to_bf16, fp8_e4m3fn_to_f32};
use crate::qwen3_moe_omega::test_support::{prand, MemorySource};

/// 2 layers, hidden 256 as 4 query heads of 64 over 2 kv heads (kv_dim 128, so k and v are
/// smaller than q and a k|v swap moves rows), intermediate 384, vocab 256.
pub fn tiny_config() -> BitNetConfig {
    BitNetConfig {
        layers: 2,
        hidden: 256,
        q_heads: 4,
        kv_heads: 2,
        head_dim: 64,
        intermediate: 384,
        vocab: 256,
        rms_eps: 1e-5,
        rope_theta: 500000.0,
        max_position: 4096,
        bos_token_id: 1,
        eos_token_ids: vec![2, 3],
    }
}

/// The Hugging Face `config.json` text of `c`, with the keys the real BitNet 2B-4T carries.
pub fn config_json(c: &BitNetConfig) -> String {
    serde_json::json!({
        "architectures": ["BitNetForCausalLM"],
        "auto_map": {
            "AutoConfig": "configuration_bitnet.BitNetConfig",
            "AutoModelForCausalLM": "modeling_bitnet.BitNetForCausalLM"
        },
        "bos_token_id": c.bos_token_id,
        "eos_token_id": c.eos_token_ids[0],
        "hidden_act": "relu2",
        "hidden_size": c.hidden,
        "initializer_range": 0.02,
        "intermediate_size": c.intermediate,
        "max_position_embeddings": c.max_position,
        "model_type": "bitnet",
        "rms_norm_eps": c.rms_eps,
        "num_attention_heads": c.q_heads,
        "num_hidden_layers": c.layers,
        "num_key_value_heads": c.kv_heads,
        "rope_theta": c.rope_theta,
        "tie_word_embeddings": true,
        "torch_dtype": "bfloat16",
        "use_cache": true,
        "vocab_size": c.vocab,
        "quantization_config": {
            "quant_method": "bitnet",
            "linear_class": "autobitlinear",
            "quantization_mode": "online"
        }
    })
    .to_string()
}

/// One tensor of the tiny checkpoint: name, safetensors dtype, shape, little-endian bytes.
pub type TinyTensor = (String, &'static str, Vec<usize>, Vec<u8>);

fn bf16_tensor(name: String, shape: Vec<usize>, seed: u64, f: impl Fn(f32) -> f32) -> TinyTensor {
    let n: usize = shape.iter().product();
    let bytes = (0..n)
        .flat_map(|i| f32_to_bf16(f(prand(seed, i))).to_le_bytes())
        .collect();
    (name, "BF16", shape, bytes)
}

/// Every tensor of the tiny checkpoint, in the real checkpoint's order, each with its own seed
/// (rows of different tensors differ). Projections are latent weights of different spreads per
/// part, so each part gets its own scale.
pub fn tensors(c: &BitNetConfig) -> Vec<TinyTensor> {
    let (h, i, qd, kvd) = (c.hidden, c.intermediate, c.q_dim(), c.kv_dim());
    let mut all = vec![bf16_tensor(
        "model.embed_tokens.weight".into(),
        vec![c.vocab, h],
        1,
        |x| x * 0.5,
    )];
    for l in 0..c.layers {
        let p = format!("model.layers.{l}");
        let seed = 100 + 1000 * l as u64;
        let norm = |x: f32| 1.0 + 0.25 * x;
        all.push(bf16_tensor(
            format!("{p}.input_layernorm.weight"),
            vec![h],
            seed,
            norm,
        ));
        for (j, (part, rows, cols)) in [
            (TernPart::Q, qd, h),
            (TernPart::K, kvd, h),
            (TernPart::V, kvd, h),
            (TernPart::O, h, qd),
        ]
        .into_iter()
        .enumerate()
        {
            let spread = 0.5 + 0.25 * j as f32;
            all.push(bf16_tensor(
                part.tensor_name(l),
                vec![rows, cols],
                seed + 1 + j as u64,
                move |x| x * spread,
            ));
        }
        all.push(bf16_tensor(
            format!("{p}.self_attn.attn_sub_norm.weight"),
            vec![qd],
            seed + 5,
            norm,
        ));
        all.push(bf16_tensor(
            format!("{p}.post_attention_layernorm.weight"),
            vec![h],
            seed + 6,
            norm,
        ));
        for (j, (part, rows, cols)) in [
            (TernPart::Gate, i, h),
            (TernPart::Up, i, h),
            (TernPart::Down, h, i),
        ]
        .into_iter()
        .enumerate()
        {
            let spread = 1.0 + 0.5 * j as f32;
            all.push(bf16_tensor(
                part.tensor_name(l),
                vec![rows, cols],
                seed + 7 + j as u64,
                move |x| x * spread,
            ));
        }
        all.push(bf16_tensor(
            format!("{p}.mlp.ffn_sub_norm.weight"),
            vec![i],
            seed + 10,
            norm,
        ));
    }
    all.push(bf16_tensor("model.norm.weight".into(), vec![h], 9, |x| {
        1.0 + 0.25 * x
    }));
    all
}

pub fn memory_source(c: &BitNetConfig) -> MemorySource {
    memory_source_of(tensors(c))
}

pub fn memory_source_of(tensors: Vec<TinyTensor>) -> MemorySource {
    let mut m = MemorySource::default();
    for (name, dtype, shape, bytes) in tensors {
        m.tensors.insert(name, (dtype.into(), shape, bytes));
    }
    m
}

/// Writes `config.json`, `generation_config.json` and one `model.safetensors` (header metadata
/// `{"format":"pt"}`, as the real master).
pub fn write_checkpoint(dir: &Path, c: &BitNetConfig) -> Result<(), String> {
    write_checkpoint_of(dir, c, tensors(c))
}

/// [`write_checkpoint`] with the given tensors.
pub fn write_checkpoint_of(
    dir: &Path,
    c: &BitNetConfig,
    all: Vec<TinyTensor>,
) -> Result<(), String> {
    let io = |e: std::io::Error| format!("{}: {e}", dir.display());
    std::fs::create_dir_all(dir).map_err(io)?;
    std::fs::write(dir.join("config.json"), config_json(c)).map_err(io)?;
    std::fs::write(
        dir.join("generation_config.json"),
        serde_json::json!({ "bos_token_id": c.bos_token_id, "eos_token_id": c.eos_token_ids })
            .to_string(),
    )
    .map_err(io)?;
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".into(), serde_json::json!({"format": "pt"}));
    let mut payload = Vec::new();
    for (name, dtype, shape, bytes) in &all {
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape,
                "data_offsets": [payload.len(), payload.len() + bytes.len()]}),
        );
        payload.extend_from_slice(bytes);
    }
    let header = serde_json::to_vec(&header).map_err(|e| e.to_string())?;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&header);
    file.extend_from_slice(&payload);
    std::fs::write(dir.join("model.safetensors"), file).map_err(io)
}

/// What a record-only ternary upload hands back: the shape and the scales, no weights.
#[derive(Debug)]
pub struct RecordedTern {
    pub n: usize,
    pub k: usize,
    pub segments: Vec<TernSegment>,
}

impl TernMatrix for RecordedTern {
    fn n(&self) -> usize {
        self.n
    }
    fn k(&self) -> usize {
        self.k
    }
    fn segments(&self) -> &[TernSegment] {
        &self.segments
    }
    fn gemv_i8(&self, _: &[i8], _: &mut [i32]) -> Result<(), String> {
        Err("record-only upload keeps no weights".into())
    }
}

/// A deliberately wrong uploader, for the negative controls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TernMutant {
    /// Stores q|k|v with the k rows and the v rows exchanged (segments as given).
    SwapKvRows { q_rows: usize, kv_rows: usize },
    /// Stores q|k|v with the k and v segment scales exchanged (rows right).
    SwapKvScales,
    /// Stores every word with the plus and minus planes exchanged (every trit negated).
    PlaneSwap,
    /// Stores the word table with one byte changed (one e4m3 code moved by one step).
    FlipTableByte,
}

/// What the recording uploader keeps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Keep {
    /// Shapes, byte counts and scales only (the real-checkpoint run).
    #[default]
    Nothing,
    /// The packed matrices and the e4m3 table, as given (byte tests and host GEMVs).
    Packed,
    /// An f32 image of every matrix (trit times scale) and of the table: the full host copy a
    /// naive reference would hold. The RSS gate's control.
    F32Image,
    /// An f32 image of layer 0's matrix of one kind only: the stray host copy of a single
    /// matrix the tightened RSS bound must still catch.
    F32CopyOf(TernTensor),
}

/// One upload as the recording fake saw it.
#[derive(Debug)]
pub struct TernRecord {
    pub layer: Option<usize>,
    /// `None` for the word table.
    pub what: Option<TernTensor>,
    pub n: usize,
    pub k: usize,
    /// Bytes handed over (words plus segment scales; table bytes plus block scales).
    pub bytes: usize,
    pub segments: Vec<TernSegment>,
    pub matrix: Option<Arc<HostTernMatrix>>,
    pub table: Option<Arc<HostFp8Gemv>>,
    pub f32_image: Option<Vec<f32>>,
}

/// Records every upload. Like a device upload it checks every word first
/// ([`check_ternary_upload`], [`check_table_upload`]).
#[derive(Debug, Default)]
pub struct RecordingTernUploader {
    pub keep: Keep,
    pub mutant: Option<TernMutant>,
    pub records: Mutex<Vec<TernRecord>>,
}

impl RecordingTernUploader {
    pub fn record_only() -> Self {
        Self::default()
    }
    pub fn keeping() -> Self {
        Self {
            keep: Keep::Packed,
            ..Self::default()
        }
    }
    pub fn keeping_f32_image() -> Self {
        Self {
            keep: Keep::F32Image,
            ..Self::default()
        }
    }
    pub fn with(mutant: TernMutant) -> Self {
        Self {
            keep: Keep::Packed,
            mutant: Some(mutant),
            ..Self::default()
        }
    }
    pub fn swapping_kv_rows(c: &BitNetConfig) -> Self {
        Self::with(TernMutant::SwapKvRows {
            q_rows: c.q_dim(),
            kv_rows: c.kv_dim(),
        })
    }
}

fn dequant(words: &[u32], k: usize, segments: &[TernSegment]) -> Vec<f32> {
    let mut out = Vec::with_capacity(words.len() * TRITS_PER_WORD);
    let wpr = k / TRITS_PER_WORD;
    let mut row = 0usize;
    for s in segments {
        let scale = bf16_to_f32(s.scale_bf16);
        for &w in &words[row * wpr..(row + s.rows) * wpr] {
            for i in 0..TRITS_PER_WORD {
                let code = (w >> (2 * i)) & 3;
                out.push(((code & 1) as f32 - (code >> 1) as f32) * scale);
            }
        }
        row += s.rows;
    }
    out
}

impl TernUploader for RecordingTernUploader {
    fn upload_ternary(
        &self,
        layer: usize,
        what: TernTensor,
        n: usize,
        k: usize,
        words: &[u32],
        segments: &[TernSegment],
    ) -> Result<Arc<dyn TernMatrix>, String> {
        check_ternary_upload(n, k, words, segments)?;
        // The words are copied only when something keeps them, so a record-only load holds
        // nothing beyond the loader's own buffers.
        let copy = match self.keep {
            Keep::Nothing => false,
            Keep::F32CopyOf(kind) => kind == what && layer == 0,
            Keep::Packed | Keep::F32Image => true,
        };
        let (mut w, mut segs) = (
            if copy { words.to_vec() } else { Vec::new() },
            segments.to_vec(),
        );
        let wpr = k / TRITS_PER_WORD;
        match (self.mutant, what) {
            _ if !copy => {}
            (Some(TernMutant::SwapKvRows { q_rows, kv_rows }), TernTensor::Qkv) => {
                let (lo, hi) = w.split_at_mut((q_rows + kv_rows) * wpr);
                lo[q_rows * wpr..].swap_with_slice(&mut hi[..kv_rows * wpr]);
            }
            (Some(TernMutant::SwapKvScales), TernTensor::Qkv) => {
                let (a, b) = (segs[1].scale_bf16, segs[2].scale_bf16);
                segs[1].scale_bf16 = b;
                segs[2].scale_bf16 = a;
            }
            (Some(TernMutant::PlaneSwap), _) => {
                for x in &mut w {
                    *x = ((*x & 0x5555_5555) << 1) | ((*x >> 1) & 0x5555_5555);
                }
            }
            _ => {}
        }
        let bytes = words.len() * 4 + 2 * segments.len();
        let (matrix, f32_image) = match self.keep {
            Keep::Nothing => (None, None),
            Keep::Packed => (
                Some(Arc::new(HostTernMatrix::new(n, k, w, segs.clone())?)),
                None,
            ),
            Keep::F32Image => (None, Some(dequant(&w, k, &segs))),
            Keep::F32CopyOf(kind) if kind == what && layer == 0 => {
                (None, Some(dequant(&w, k, &segs)))
            }
            Keep::F32CopyOf(_) => (None, None),
        };
        self.records.lock().unwrap().push(TernRecord {
            layer: Some(layer),
            what: Some(what),
            n,
            k,
            bytes,
            segments: segs.clone(),
            matrix: matrix.clone(),
            table: None,
            f32_image,
        });
        Ok(match matrix {
            Some(m) => m,
            None => Arc::new(RecordedTern {
                n,
                k,
                segments: segs,
            }),
        })
    }

    fn upload_table_fp8(
        &self,
        n: usize,
        k: usize,
        w: &[u8],
        scales: &[u16],
    ) -> Result<Arc<dyn DenseGemv>, String> {
        check_table_upload(n, k, w, scales)?;
        // Copied only when kept, so a record-only load holds no second table.
        let stored = || {
            let mut s = w.to_vec();
            if self.mutant == Some(TernMutant::FlipTableByte) {
                // Byte 1 one code step away; bit 1 instead where bit 0 would make the NaN code.
                let b = s[1] ^ 1;
                s[1] = if b & 0x7f == 0x7f { s[1] ^ 2 } else { b };
            }
            s
        };
        let (table, f32_image) = match self.keep {
            Keep::Nothing | Keep::F32CopyOf(_) => (None, None),
            Keep::Packed => (
                Some(Arc::new(HostFp8Gemv::new(n, k, stored(), scales.to_vec())?)),
                None,
            ),
            Keep::F32Image => {
                let kb = k / FP8_BLOCK;
                let img = w
                    .iter()
                    .enumerate()
                    .map(|(i, &b)| {
                        let (r, c) = (i / k, i % k);
                        fp8_e4m3fn_to_f32(b)
                            * bf16_to_f32(scales[(r / FP8_BLOCK) * kb + c / FP8_BLOCK])
                    })
                    .collect();
                (None, Some(img))
            }
        };
        self.records.lock().unwrap().push(TernRecord {
            layer: None,
            what: None,
            n,
            k,
            bytes: w.len() + 2 * scales.len(),
            segments: Vec::new(),
            matrix: None,
            table: table.clone(),
            f32_image,
        });
        Ok(match table {
            Some(t) => t,
            None => Arc::new(RecordedShape { n, k }),
        })
    }
}

/// The reference scale of one projection, computed apart from the loader: one sequential f64
/// sum, then PyTorch's bf16 steps.
pub fn reference_scale(bits: &[u16]) -> (f32, f32) {
    let bf = |x: f32| bf16_to_f32(f32_to_bf16(x));
    let sum: f64 = bits.iter().map(|&b| bf16_to_f32(b).abs() as f64).sum();
    let mean = bf((sum / bits.len() as f64) as f32).max(bf(1e-5));
    let s = bf(1.0 / mean);
    (s, bf(1.0 / s))
}

/// The reference trit: PyTorch's `round` (ties to even) of the bf16 product, clamped.
pub fn reference_trit(w: u16, s: f32) -> i8 {
    let p = bf16_to_f32(f32_to_bf16(bf16_to_f32(w) * s));
    p.round_ties_even().clamp(-1.0, 1.0) as i8
}

/// Row `r` of a stored matrix, decoded plane by plane (independently of the loader's decoder).
pub fn stored_row(m: &HostTernMatrix, r: usize) -> Vec<i8> {
    let wpr = m.k() / TRITS_PER_WORD;
    m.words()[r * wpr..(r + 1) * wpr]
        .iter()
        .flat_map(|&w| {
            (0..TRITS_PER_WORD)
                .map(move |i| ((w >> (2 * i)) & 1) as i8 - ((w >> (2 * i + 1)) & 1) as i8)
        })
        .collect()
}

/// The byte test: the uploads must be, in order, each layer's q|k|v, o, gate|up, down, then the
/// word table, with the right shapes and byte counts; every segment must name its part, row
/// count and scale (recomputed from the checkpoint); every stored row must be the trits of the
/// checkpoint row it stands for (read again from `source`); the table must be the e4m3 blocks
/// of the embedding. The first wrong row or scale is named.
pub fn check_upload_bytes(
    c: &BitNetConfig,
    source: &mut dyn TensorSource,
    records: &[TernRecord],
) -> Result<(), String> {
    let expected = c.layers * TernTensor::ALL.len() + 1;
    if records.len() != expected {
        return Err(format!("{} uploads, expected {expected}", records.len()));
    }
    let read = |source: &mut dyn TensorSource, name: &str, shape: &[usize]| {
        let mut bytes = vec![0u8; 2 * shape.iter().product::<usize>()];
        source.read_into(name, "BF16", shape, &mut bytes)?;
        Ok::<Vec<u16>, String>(
            bytes
                .chunks_exact(2)
                .map(|b| u16::from_le_bytes([b[0], b[1]]))
                .collect(),
        )
    };
    for (idx, rec) in records[..expected - 1].iter().enumerate() {
        let (layer, what) = (idx / 4, TernTensor::ALL[idx % 4]);
        let (n, k) = c.shape(what);
        let at = format!("upload {idx} (layer {layer} {})", what.name());
        let bytes = c.ternary_bytes(what);
        if (rec.layer, rec.what, rec.n, rec.k, rec.bytes) != (Some(layer), Some(what), n, k, bytes)
        {
            return Err(format!(
                "{at}: got {:?} {:?} [{}][{}] {} bytes, expected [{n}][{k}] {bytes}",
                rec.layer, rec.what, rec.n, rec.k, rec.bytes
            ));
        }
        let m = rec
            .matrix
            .as_ref()
            .ok_or_else(|| format!("{at}: not kept"))?;
        let parts = c.parts(what);
        if rec.segments.len() != parts.len() {
            return Err(format!(
                "{at}: {} segments, expected {}",
                rec.segments.len(),
                parts.len()
            ));
        }
        let mut row0 = 0usize;
        for (seg, (part, rows)) in rec.segments.iter().zip(parts) {
            let name = part.tensor_name(layer);
            let src = read(source, &name, &[rows, k])?;
            let (s, w_scale) = reference_scale(&src);
            if (seg.part, seg.rows, seg.scale_bf16) != (part, rows, f32_to_bf16(w_scale)) {
                return Err(format!(
                    "{at}: segment {:?} rows {} scale {} is not {name} ({rows} rows, scale {w_scale})",
                    seg.part,
                    seg.rows,
                    bf16_to_f32(seg.scale_bf16)
                ));
            }
            for r in 0..rows {
                let want: Vec<i8> = src[r * k..(r + 1) * k]
                    .iter()
                    .map(|&w| reference_trit(w, s))
                    .collect();
                if stored_row(m, row0 + r) != want {
                    return Err(format!(
                        "{at}: stored row {} is not {name} row {r}",
                        row0 + r
                    ));
                }
            }
            row0 += rows;
        }
    }
    let rec = &records[expected - 1];
    let (n, h, b) = (c.vocab, c.hidden, FP8_BLOCK);
    if (rec.layer, rec.what, rec.n, rec.k, rec.bytes) != (None, None, n, h, c.table_bytes()) {
        return Err(format!(
            "last upload is not the [{n}][{h}] word table: {:?} {:?} [{}][{}]",
            rec.layer, rec.what, rec.n, rec.k
        ));
    }
    let t = rec.table.as_ref().ok_or("word table not kept")?;
    let embed = read(source, "model.embed_tokens.weight", &[n, h])?;
    for bi in 0..n / b {
        for bj in 0..h / b {
            let at = |r: usize, cc: usize| embed[(bi * b + r) * h + bj * b + cc];
            let amax = (0..b * b)
                .map(|j| bf16_to_f32(at(j / b, j % b)).abs())
                .fold(0.0f32, f32::max);
            let sb = f32_to_bf16(if amax > 0.0 { amax / 448.0 } else { 1.0 });
            if t.scales()[bi * (h / b) + bj] != sb {
                return Err(format!("word table block ({bi}, {bj}) scale differs"));
            }
            for j in 0..b * b {
                let (r, cc) = (j / b, j % b);
                let code = f32_to_fp8_e4m3fn(bf16_to_f32(at(r, cc)) / bf16_to_f32(sb));
                let code = if code == 0x80 { 0 } else { code };
                if t.weights()[(bi * b + r) * h + bj * b + cc] != code {
                    return Err(format!(
                        "word table row {} column {} differs",
                        bi * b + r,
                        bj * b + cc
                    ));
                }
            }
        }
    }
    Ok(())
}

/// The host half of the byte test: each layer's four norms and the final norm must be the
/// checkpoint's (read again from `source`), each in its own slot, so an input/post norm swap is
/// named.
pub fn check_host_norms(
    c: &BitNetConfig,
    source: &mut dyn TensorSource,
    w: &BitNetOmegaWeights,
) -> Result<(), String> {
    let mut check = |name: String, len: usize, got: &[f32]| {
        let mut bytes = vec![0u8; 2 * len];
        source.read_into(&name, "BF16", &[len], &mut bytes)?;
        let want: Vec<f32> = bytes
            .chunks_exact(2)
            .map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]])))
            .collect();
        if got != want.as_slice() {
            return Err(format!("host norm {name} is not the checkpoint's"));
        }
        Ok::<(), String>(())
    };
    if w.layers.len() != c.layers {
        return Err(format!(
            "{} layers held, expected {}",
            w.layers.len(),
            c.layers
        ));
    }
    for (l, layer) in w.layers.iter().enumerate() {
        let p = format!("model.layers.{l}");
        check(
            format!("{p}.input_layernorm.weight"),
            c.hidden,
            &layer.input_norm,
        )?;
        check(
            format!("{p}.self_attn.attn_sub_norm.weight"),
            c.q_dim(),
            &layer.attn_sub_norm,
        )?;
        check(
            format!("{p}.post_attention_layernorm.weight"),
            c.hidden,
            &layer.post_norm,
        )?;
        check(
            format!("{p}.mlp.ffn_sub_norm.weight"),
            c.intermediate,
            &layer.ffn_sub_norm,
        )?;
    }
    check("model.norm.weight".into(), c.hidden, &w.final_norm)
}
