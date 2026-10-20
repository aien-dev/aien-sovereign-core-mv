//! The MXFP4 store of Qwen3.8-27B (L125 SPEED-1): every decode projection and the lm_head converted from the FP8
//! checkpoint once, ahead of time (`qwen38_w4_store`), so a W4-only load reads 4-bit bytes and uploads them: no FP8
//! read, no FP8 copy on the device, no conversion at start.
//!
//! File: `AIENW4S1`, then per tensor the e2m1 nibbles `[n][k/2]` and the E8M0 scale bytes `[n][k/32]` (omega
//! `omega_gpu_w4_tensor_upload` layout, each part starting on a 4 KiB boundary), then the header (JSON), its length
//! (u64 little endian) and `AIENW4S1` again. Written to `<out>.tmp` and renamed, so a cut write never looks whole.
//!
//! What can never load silently: a store of another checkpoint (the header's source fingerprint covers config.json,
//! the shard index, and every shard's length and safetensors header), a store whose bytes changed (a 64-bit check per
//! tensor, verified on every read), a store from a converter that rounds differently (a fixed synthetic FP8 block is
//! converted again at open and must give the stored canary bytes), and a missing or misshapen tensor.
use std::fs::File;
use std::io::{BufWriter, Read, Seek, SeekFrom, Write};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

use crate::qwen38::{read_matrix, text_prefix, Matrix, Qwen38Config};
use crate::qwen3_moe_omega::{ShardSource, TensorSource};

pub const MAGIC: &[u8; 8] = b"AIENW4S1";
pub const FORMAT: u32 = 1;
const ALIGN: u64 = 4096;

/// One stored tensor: nibbles at `w_off` (`n * k / 2` bytes), scales at `e_off` (`n * k / 32`).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
pub struct Entry {
    pub name: String,
    pub n: usize,
    pub k: usize,
    pub w_off: u64,
    pub e_off: u64,
    /// [`check64`] of the nibbles then the scales, hex.
    pub check: String,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct Header {
    pub format: u32,
    /// MXFP4 group-scale rule the converter used (`ocp`, `fit`, `best`).
    pub rule: String,
    /// omega converter entry and the lm_head path (bf16 -> FP8 -> MXFP4).
    pub converter: String,
    /// [`source_fingerprint`] of the checkpoint, hex.
    pub source: String,
    pub layers: usize,
    pub tensors: Vec<Entry>,
    /// The converter on [`canary_fp8`]: nibbles then scales, hex.
    pub canary: String,
    /// Converter tally over every tensor (omega `OmegaW4Stats`).
    pub weights: u64,
    pub saturated: u64,
    pub rel_rms: f64,
    /// The same tally over the projections only (no head): the number the host survey reports (`qwen38_ab w4err`).
    pub proj_rel_rms: f64,
}

/// A 64-bit check of `parts` in order (8-byte words, multiply-rotate mix; tail bytes zero padded). Detects changed
/// bytes, not an adversary.
pub fn check64(parts: &[&[u8]]) -> u64 {
    let mut h: u64 = 0x243f_6a88_85a3_08d3;
    let mut len = 0u64;
    for p in parts {
        let mut c = p.chunks_exact(8);
        for w in &mut c {
            let v = u64::from_le_bytes(w.try_into().expect("8 bytes"));
            h = (h.rotate_left(23) ^ v).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
        let r = c.remainder();
        if !r.is_empty() {
            let mut b = [0u8; 8];
            b[..r.len()].copy_from_slice(r);
            h = (h.rotate_left(23) ^ u64::from_le_bytes(b)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        }
        len += p.len() as u64;
    }
    (h ^ len).wrapping_mul(0xff51_afd7_ed55_8ccd) ^ (h >> 29)
}

fn read_file(p: &Path) -> Result<Vec<u8>, String> {
    std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))
}

/// Identity of a checkpoint directory without reading its weights: config.json, model.safetensors.index.json, and
/// for every shard the index names (sorted) its file length and its safetensors header.
pub fn source_fingerprint(dir: &Path) -> Result<u64, String> {
    let config = read_file(&dir.join("config.json"))?;
    let index = read_file(&dir.join("model.safetensors.index.json"))?;
    let v: serde_json::Value =
        serde_json::from_slice(&index).map_err(|e| format!("shard index: {e}"))?;
    let mut shards: Vec<String> = v["weight_map"]
        .as_object()
        .ok_or("shard index: no weight_map")?
        .values()
        .filter_map(|s| s.as_str().map(str::to_owned))
        .collect();
    shards.sort();
    shards.dedup();
    let mut parts: Vec<Vec<u8>> = vec![config, index];
    for s in &shards {
        let p = dir.join(s);
        let mut f = File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
        let len = f
            .metadata()
            .map_err(|e| format!("{}: {e}", p.display()))?
            .len();
        let mut n8 = [0u8; 8];
        f.read_exact(&mut n8)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        let hl = u64::from_le_bytes(n8);
        if hl > 100 << 20 || hl + 8 > len {
            return Err(format!("{}: safetensors header length {hl}", p.display()));
        }
        let mut hdr = vec![0u8; hl as usize];
        f.read_exact(&mut hdr)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        let mut rec = s.as_bytes().to_vec();
        rec.extend_from_slice(&len.to_le_bytes());
        rec.extend_from_slice(&hdr);
        parts.push(rec);
    }
    let refs: Vec<&[u8]> = parts.iter().map(Vec::as_slice).collect();
    Ok(check64(&refs))
}

/// A checkpoint tensor `(name, n, k)`.
pub type Part = (String, usize, usize);
/// A store tensor: its name and the checkpoint parts stacked by rows.
pub type Proj = (String, Vec<Part>);

/// The four decode projections of model layer `l` in the graph's order (mix, o, gate_up, down): the store name (the
/// one the loader reports) and the checkpoint tensors stacked by rows, each `(name, n, k)`.
pub fn layer_projs(c: &Qwen38Config, prefix: &str, l: usize, attention: bool) -> [Proj; 4] {
    let h = c.hidden;
    let p = format!("{prefix}layers.{l}");
    let m = format!("{p}.mlp");
    let (mix, o) = if attention {
        let a = format!("{p}.self_attn");
        (
            (
                format!("{a} q|k|v"),
                vec![
                    (format!("{a}.q_proj.weight"), 2 * c.heads * c.head_dim, h),
                    (format!("{a}.k_proj.weight"), c.kv_dim(), h),
                    (format!("{a}.v_proj.weight"), c.kv_dim(), h),
                ],
            ),
            (
                format!("{a}.o_proj"),
                vec![(format!("{a}.o_proj.weight"), h, c.heads * c.head_dim)],
            ),
        )
    } else {
        let a = format!("{p}.linear_attn");
        (
            (
                format!("{a} qkv|z"),
                vec![
                    (format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h),
                    (format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h),
                ],
            ),
            (
                format!("{a}.out_proj"),
                vec![(format!("{a}.out_proj.weight"), h, c.lin_value_dim())],
            ),
        )
    };
    [
        mix,
        o,
        (
            format!("{m} gate|up"),
            vec![
                (format!("{m}.gate_proj.weight"), c.intermediate, h),
                (format!("{m}.up_proj.weight"), c.intermediate, h),
            ],
        ),
        (
            format!("{m}.down_proj"),
            vec![(format!("{m}.down_proj.weight"), h, c.intermediate)],
        ),
    ]
}

/// The lm_head's store name.
pub const LM_HEAD: &str = "lm_head";

/// FP8 parts stacked by rows: `(n, k, weights, scale_inv)`.
pub fn read_fused(
    source: &mut dyn TensorSource,
    store_name: &str,
    parts: &[Part],
) -> Result<(usize, usize, Vec<u8>, Vec<u16>), String> {
    let (mut n, mut k0) = (0usize, None);
    let (mut w, mut s) = (Vec::new(), Vec::new());
    for (name, pn, pk) in parts {
        match read_matrix(source, name, *pn, *pk)? {
            Matrix::Fp8 {
                n: rn,
                k,
                w: pw,
                scales,
            } => {
                if *k0.get_or_insert(k) != k {
                    return Err(format!("{store_name}: parts with different k"));
                }
                n += rn;
                w.extend_from_slice(&pw);
                s.extend_from_slice(&scales);
            }
            Matrix::Bf16 { .. } => {
                return Err(format!(
                    "{store_name}: a bf16 part (the store converts FP8 projections)"
                ))
            }
        }
    }
    Ok((n, k0.ok_or(format!("{store_name}: no parts"))?, w, s))
}

/// A fixed FP8 block `[128][128]` (every exponent, both signs, zeros; two scales) for the converter canary.
pub fn canary_fp8() -> (Vec<u8>, Vec<u16>) {
    let w = (0..128 * 128u32)
        .map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8 & 0xfe)
        .collect();
    (w, vec![0x3c00])
}

/// The linked converter on [`canary_fp8`] with `rule`, hex.
pub fn canary(rule: i32) -> Result<String, String> {
    let (w, s) = canary_fp8();
    let (w4, e8, _) = aien_omega_q38::w4_from_fp8(128, 128, &w, &s, 1, rule)
        .map_err(|e| format!("w4 canary: {e}"))?;
    Ok([w4, e8]
        .concat()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

pub fn rule_name(rule: i32) -> &'static str {
    match rule {
        aien_omega_q38::W4_RULE_FIT => "fit",
        aien_omega_q38::W4_RULE_BEST => "best",
        _ => "ocp",
    }
}

fn align(f: &mut BufWriter<File>, pos: &mut u64) -> Result<(), String> {
    let pad = pos.next_multiple_of(ALIGN) - *pos;
    f.write_all(&vec![0u8; pad as usize])
        .map_err(|e| format!("store write: {e}"))?;
    *pos += pad;
    Ok(())
}

/// Converts the checkpoint in `dir` (all layers, then the lm_head through FP8 like `AIEN_Q38_W4_HEAD=1`) with
/// `rule` on `threads` threads and writes the store to `out`. `log` gets one line per layer.
pub fn write_store(
    dir: &Path,
    out: &Path,
    rule: i32,
    threads: usize,
    log: &mut dyn FnMut(&str),
) -> Result<Header, String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let source_fp = source_fingerprint(dir)?;
    let tmp = out.with_extension("tmp");
    let mut f = BufWriter::with_capacity(
        8 << 20,
        File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?,
    );
    f.write_all(MAGIC)
        .map_err(|e| format!("store write: {e}"))?;
    let mut pos = MAGIC.len() as u64;
    let mut tensors = Vec::new();
    let mut tally = aien_omega_q38::W4Stats::default();
    let mut put = |name: String,
                   n: usize,
                   k: usize,
                   w8: &[u8],
                   s: &[u16],
                   f: &mut BufWriter<File>,
                   pos: &mut u64|
     -> Result<aien_omega_q38::W4Stats, String> {
        let (w4, e8, st) = aien_omega_q38::w4_from_fp8(n, k, w8, s, threads, rule)
            .map_err(|e| format!("{name}: {e}"))?;
        align(f, pos)?;
        let w_off = *pos;
        f.write_all(&w4).map_err(|e| format!("store write: {e}"))?;
        *pos += w4.len() as u64;
        align(f, pos)?;
        let e_off = *pos;
        f.write_all(&e8).map_err(|e| format!("store write: {e}"))?;
        *pos += e8.len() as u64;
        tensors.push(Entry {
            name,
            n,
            k,
            w_off,
            e_off,
            check: format!("{:016x}", check64(&[&w4, &e8])),
        });
        Ok(st)
    };
    for l in 0..c.layers {
        for (name, parts) in layer_projs(&c, &prefix, l, c.full_attention[l]) {
            let (n, k, w, s) = read_fused(&mut source, &name, &parts)?;
            tally.add(&put(name, n, k, &w, &s, &mut f, &mut pos)?);
        }
        log(&format!("w4 store: layer {l} of {} converted", c.layers));
    }
    let proj_rel_rms = tally.rel_rms();
    match read_matrix(&mut source, "lm_head.weight", c.vocab, c.hidden)? {
        Matrix::Bf16 { n, k, w } => {
            let (w8, s) = aien_omega_q38::quantize_fp8_bf16(n, k, &w, threads)
                .map_err(|e| format!("lm_head fp8 quantise: {e}"))?;
            tally.add(&put(LM_HEAD.into(), n, k, &w8, &s, &mut f, &mut pos)?);
        }
        Matrix::Fp8 { .. } => return Err("lm_head: FP8 in the checkpoint (expected bf16)".into()),
    }
    let header = Header {
        format: FORMAT,
        rule: rule_name(rule).into(),
        converter:
            "omega omega_gpu_w4_from_fp8_rule; lm_head bf16 -> omega_gpu_fp8_quantize_bf16 -> MXFP4"
                .into(),
        source: format!("{source_fp:016x}"),
        layers: c.layers,
        tensors,
        canary: canary(rule)?,
        weights: tally.weights,
        saturated: tally.saturated,
        rel_rms: tally.rel_rms(),
        proj_rel_rms,
    };
    let js = serde_json::to_vec(&header).map_err(|e| format!("store header: {e}"))?;
    f.write_all(&js)
        .and_then(|_| f.write_all(&(js.len() as u64).to_le_bytes()))
        .and_then(|_| f.write_all(MAGIC))
        .map_err(|e| format!("store write: {e}"))?;
    let file = f.into_inner().map_err(|e| format!("store write: {e}"))?;
    file.sync_all().map_err(|e| format!("store sync: {e}"))?;
    drop(file);
    // Read back as the loader will before the name exists: a store that does not open and check never appears.
    let back = W4Store::open(&tmp, dir)?;
    for t in &back.header.tensors {
        back.read(&t.name, t.n, t.k)?;
    }
    std::fs::rename(&tmp, out).map_err(|e| format!("{}: {e}", out.display()))?;
    Ok(header)
}

/// An opened store, checked against its checkpoint and the linked converter.
pub struct W4Store {
    file: File,
    pub path: PathBuf,
    pub header: Header,
}

impl std::fmt::Debug for W4Store {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "W4Store {{ {} }}", self.path.display())
    }
}

impl W4Store {
    /// Opens `path` and refuses it unless it is a whole store of the checkpoint in `dir` made by a converter that
    /// rounds like the linked one.
    pub fn open(path: &Path, dir: &Path) -> Result<Self, String> {
        let e = |m: String| format!("w4 store {}: {m}", path.display());
        let mut file = File::open(path).map_err(|x| e(x.to_string()))?;
        let len = file.metadata().map_err(|x| e(x.to_string()))?.len();
        let mut head = [0u8; 8];
        file.read_exact(&mut head).map_err(|x| e(x.to_string()))?;
        let mut tail = [0u8; 16];
        if len < 24 {
            return Err(e("too short".into()));
        }
        file.seek(SeekFrom::Start(len - 16))
            .and_then(|_| file.read_exact(&mut tail))
            .map_err(|x| e(x.to_string()))?;
        if &head != MAGIC || &tail[8..] != MAGIC {
            return Err(e("not a whole AIENW4S1 store".into()));
        }
        let hl = u64::from_le_bytes(tail[..8].try_into().expect("8 bytes"));
        if hl + 24 > len {
            return Err(e(format!("header length {hl}")));
        }
        let mut js = vec![0u8; hl as usize];
        file.read_exact_at(&mut js, len - 16 - hl)
            .map_err(|x| e(x.to_string()))?;
        let header: Header = serde_json::from_slice(&js).map_err(|x| e(x.to_string()))?;
        if header.format != FORMAT {
            return Err(e(format!(
                "format {} (this build reads {FORMAT})",
                header.format
            )));
        }
        let fp = format!("{:016x}", source_fingerprint(dir)?);
        if header.source != fp {
            return Err(e(format!(
                "made from another checkpoint (store {}, {} is {fp})",
                header.source,
                dir.display()
            )));
        }
        let rule = match header.rule.as_str() {
            "ocp" => aien_omega_q38::W4_RULE_OCP,
            "fit" => aien_omega_q38::W4_RULE_FIT,
            "best" => aien_omega_q38::W4_RULE_BEST,
            r => return Err(e(format!("rule {r}"))),
        };
        if aien_omega_q38::w4_native() && canary(rule)? != header.canary {
            return Err(e(
                "the linked MXFP4 converter rounds differently from the one that wrote this store (canary)".into(),
            ));
        }
        for t in &header.tensors {
            let end = t.e_off + (t.n * t.k / 32) as u64;
            if t.w_off + (t.n * t.k / 2) as u64 > t.e_off || end > len - 16 - hl {
                return Err(e(format!("{}: offsets outside the payload", t.name)));
            }
        }
        Ok(Self {
            file,
            path: path.to_owned(),
            header,
        })
    }

    /// Nibbles and scales of `name`, shape and check verified.
    pub fn read(&self, name: &str, n: usize, k: usize) -> Result<(Vec<u8>, Vec<u8>), String> {
        let t = self
            .header
            .tensors
            .iter()
            .find(|t| t.name == name)
            .ok_or_else(|| format!("w4 store {}: no tensor {name}", self.path.display()))?;
        if (t.n, t.k) != (n, k) {
            return Err(format!(
                "w4 store: {name} is [{}][{}], the model wants [{n}][{k}]",
                t.n, t.k
            ));
        }
        let mut w4 = vec![0u8; n * k / 2];
        let mut e8 = vec![0u8; n * k / 32];
        self.file
            .read_exact_at(&mut w4, t.w_off)
            .and_then(|_| self.file.read_exact_at(&mut e8, t.e_off))
            .map_err(|x| format!("w4 store {}: {name}: {x}", self.path.display()))?;
        if format!("{:016x}", check64(&[&w4, &e8])) != t.check {
            return Err(format!(
                "w4 store {}: {name}: bytes changed since written (check)",
                self.path.display()
            ));
        }
        Ok((w4, e8))
    }
}

/// The store a load of `dir` uses: env `AIEN_Q38_W4_STORE` = a path, or `off` (FP8 as before); unset = the default
/// `<dir>.mxfp4/aien-w4.store` when that file exists (the served default), else none.
pub fn resolve(dir: &Path) -> Result<Option<PathBuf>, String> {
    match std::env::var("AIEN_Q38_W4_STORE").as_deref() {
        Ok("off") | Ok("0") => Ok(None),
        Ok(p) if !p.is_empty() => {
            let p = PathBuf::from(p);
            if p.is_file() {
                Ok(Some(p))
            } else {
                Err(format!("AIEN_Q38_W4_STORE={}: no such file", p.display()))
            }
        }
        _ => {
            let p = default_path(dir);
            Ok(p.is_file().then_some(p))
        }
    }
}

/// `<dir>.mxfp4/aien-w4.store` (beside the checkpoint, never inside it).
pub fn default_path(dir: &Path) -> PathBuf {
    let name = dir
        .file_name()
        .map_or_else(|| "model".into(), |n| n.to_string_lossy().into_owned());
    dir.with_file_name(format!("{name}.mxfp4"))
        .join("aien-w4.store")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn check64_sees_one_bit_and_the_split() {
        let a = vec![7u8; 1001];
        let mut b = a.clone();
        b[500] ^= 1;
        assert_ne!(check64(&[&a]), check64(&[&b]));
        assert_eq!(check64(&[&a]), check64(&[&a]));
        let short = vec![7u8; 1000];
        assert_ne!(check64(&[&a]), check64(&[&short]));
    }

    #[test]
    fn default_path_sits_beside_the_checkpoint() {
        assert_eq!(
            default_path(Path::new("/m/qwen3.8-27b-fp8")),
            PathBuf::from("/m/qwen3.8-27b-fp8.mxfp4/aien-w4.store")
        );
        assert_eq!(
            default_path(Path::new("/m/qwen3.8-27b-fp8/")),
            PathBuf::from("/m/qwen3.8-27b-fp8.mxfp4/aien-w4.store")
        );
    }
}
