//! READ DRUM (sc #677, FORTRAN idea 5): a dense model bigger than its memory budget, run in layer
//! groups that take turns on the chip while the NVMe streams the next group.
//!
//! The 1956 FORTRAN manual's answer to a program too large for core was READ DRUM and "the
//! programmer must subdivide the program". Here the core is the memory budget and the drum is the
//! NVMe:
//!
//! - [`build`] writes every layer's chip-ready projections (the same fused, shifted bytes
//!   [`super::Qwen38Omega`] uploads) once, as one contiguous 4 KiB-aligned image per layer, plus a
//!   JSON manifest with each layer's sha256. Small per-layer parameters (norms, conv, A_log,
//!   dt_bias, q/k norms) stay resident: they are read from the checkpoint at open.
//! - [`plan`] splits the layers into groups and pins as many whole groups as the budget holds after
//!   the fixed part (embedding, lm_head, conversation state) and the streaming slots (two host
//!   staging buffers plus one group on the device).
//! - [`DrumModel`] runs a token: pinned groups straight away; every streamed group waits for its
//!   O_DIRECT read (a reader thread fills the other staging buffer meanwhile), is uploaded, its
//!   graphs re-pointed, run for every conversation of the batch, and dropped. Per group and
//!   conversation one [`Q38Graph`] with its own DeltaNet state and KV cache; the residual stream
//!   goes from group to group through the host as f32 (exact).
//!
//! Entry names follow the FORTRAN idea 4 layout pack (`layers.N.mix.w`, `layers.N.mix.s`,
//! `layers.N.ab`, ...; 4 KiB aligned): [`DrumIndex`] only needs each layer's byte range and its
//! entries' offsets, so it can be built from that pack's manifest instead of this file's.
use std::collections::BTreeMap;
use std::fs::File;
use std::io::Write;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

use aien_omega_bf16::Bf16Tensor;
use aien_omega_fp8::Fp8Tensor;
use aien_omega_q38::{Attn256Cache, Gdn, GraphConfig, Q38Graph};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use super::{ab_rows, fold, gpu, LayerTensors, EPS_L2};
use crate::qwen38::{read_bf16_words, read_matrix, read_vec, text_prefix, Matrix, Qwen38Config};
use crate::qwen3_moe::bf16_to_f32;
use crate::qwen3_moe_omega::{ShardSource, TensorSource};

/// O_DIRECT alignment of every image, offset and length.
pub const ALIGN: usize = 4096;
/// Manifest format tag.
pub const FORMAT: &str = "aien-q38-drum/1";
/// Bytes one reader thread asks for per `pread`.
pub const READ_CHUNK: usize = 8 << 20;
/// Rows of the stand-in head that every group but the last carries (the graph always ends in a
/// head; only the last group's head is the model's).
pub const STAND_IN_VOCAB: usize = 2048;

fn round_up(v: usize, a: usize) -> usize {
    v.div_ceil(a) * a
}

/// One piece of a layer image: byte range in the drum file and its matrix shape.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrumEntry {
    pub name: String,
    pub offset: u64,
    pub bytes: u64,
    /// Rows and columns (FP8 weight `[n][k]`; scales `[n/128][k/128]`; ab `[96][5120]`).
    pub n: usize,
    pub k: usize,
}

/// One layer's contiguous image.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrumLayer {
    pub layer: usize,
    pub attention: bool,
    pub start: u64,
    /// Multiple of [`ALIGN`].
    pub len: u64,
    /// sha256 of the `len` bytes at `start` (padding included).
    pub sha256: String,
    pub entries: Vec<DrumEntry>,
}

/// What the drum file holds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DrumIndex {
    pub format: String,
    /// The checkpoint directory name it was built from (`qwen3.8-27b-fp8-017b9c7a`).
    pub checkpoint: String,
    pub layers: Vec<DrumLayer>,
}

impl DrumIndex {
    pub fn load(path: &Path) -> Result<Self, String> {
        let s = std::fs::read_to_string(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let ix: Self = serde_json::from_str(&s).map_err(|e| format!("{}: {e}", path.display()))?;
        if ix.format != FORMAT {
            return Err(format!(
                "{}: format {} (want {FORMAT})",
                path.display(),
                ix.format
            ));
        }
        ix.check()?;
        Ok(ix)
    }

    /// Every layer aligned, in order, entries inside their layer and not overlapping.
    pub fn check(&self) -> Result<(), String> {
        let a = ALIGN as u64;
        let mut end = 0u64;
        for (i, l) in self.layers.iter().enumerate() {
            if l.layer != i {
                return Err(format!("drum: layer {} at index {i}", l.layer));
            }
            if !l.start.is_multiple_of(a) || !l.len.is_multiple_of(a) || l.start < end {
                return Err(format!("drum: layer {i} range not aligned or overlapping"));
            }
            let mut e_end = l.start;
            for e in &l.entries {
                if !e.offset.is_multiple_of(a)
                    || e.offset < e_end
                    || e.offset + e.bytes > l.start + l.len
                {
                    return Err(format!(
                        "drum: layer {i} entry {} outside its layer",
                        e.name
                    ));
                }
                e_end = e.offset + e.bytes;
            }
            end = l.start + l.len;
        }
        Ok(())
    }

    /// Image bytes per layer (what one pass streams for that layer).
    pub fn layer_bytes(&self) -> Vec<u64> {
        self.layers.iter().map(|l| l.len).collect()
    }
}

/// A 4 KiB-aligned heap buffer (O_DIRECT reads land here).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
}

// SAFETY: owns its allocation; handed between threads, never shared.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize) -> Self {
        let len = round_up(len.max(ALIGN), ALIGN);
        let layout = std::alloc::Layout::from_size_align(len, ALIGN).expect("aligned layout");
        // SAFETY: non-zero size.
        let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
        assert!(!ptr.is_null(), "AlignedBuf: out of memory ({len} B)");
        Self { ptr, len }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: ptr..ptr+len is our allocation.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, uniquely borrowed.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = std::alloc::Layout::from_size_align(self.len, ALIGN).expect("aligned layout");
        // SAFETY: allocated with this layout in `new`.
        unsafe { std::alloc::dealloc(self.ptr, layout) }
    }
}

/// The drum file opened for O_DIRECT reads (the page cache is bypassed, so every byte comes off
/// the NVMe).
pub struct DrumFile {
    file: File,
    threads: usize,
}

impl DrumFile {
    pub fn open(path: &Path, threads: usize) -> Result<Self, String> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
            .map_err(|e| format!("{} (O_DIRECT): {e}", path.display()))?;
        Ok(Self {
            file,
            threads: threads.max(1),
        })
    }

    /// Reads `len` bytes at `start` (both aligned) into `buf[..len]` with the reader threads, each
    /// taking every `threads`-th chunk.
    pub fn read(&self, start: u64, len: usize, buf: &mut AlignedBuf) -> Result<(), String> {
        if !start.is_multiple_of(ALIGN as u64) || !len.is_multiple_of(ALIGN) || len > buf.len() {
            return Err(format!(
                "drum read: {len} B at {start} not aligned or too big"
            ));
        }
        let chunks: Vec<(usize, &mut [u8])> = buf.as_mut_slice()[..len]
            .chunks_mut(READ_CHUNK)
            .enumerate()
            .map(|(i, c)| (i * READ_CHUNK, c))
            .collect();
        let mut lanes: Vec<Vec<(usize, &mut [u8])>> =
            (0..self.threads).map(|_| Vec::new()).collect();
        for (i, c) in chunks.into_iter().enumerate() {
            lanes[i % self.threads].push(c);
        }
        let file = &self.file;
        std::thread::scope(|s| {
            let hs: Vec<_> = lanes
                .into_iter()
                .map(|lane| {
                    s.spawn(move || -> Result<(), String> {
                        for (off, dst) in lane {
                            let mut done = 0;
                            while done < dst.len() {
                                let n = file
                                    .read_at(&mut dst[done..], start + (off + done) as u64)
                                    .map_err(|e| format!("drum pread: {e}"))?;
                                if n == 0 {
                                    return Err("drum pread: end of file".into());
                                }
                                done += n;
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            hs.into_iter()
                .try_for_each(|h| h.join().map_err(|_| "drum reader panicked".to_string())?)
        })
    }
}

/// sha256 as lowercase hex.
pub fn sha256_hex(b: &[u8]) -> String {
    Sha256::digest(b)
        .iter()
        .map(|x| format!("{x:02x}"))
        .collect()
}

/// FP8 parts stacked by rows (the host half of [`super::fuse_fp8`]): `(n, k, w, scales)`.
fn concat_fp8(name: &str, parts: Vec<Matrix>) -> Result<(usize, usize, Vec<u8>, Vec<u16>), String> {
    let (mut n, mut k0) = (0usize, None);
    let (mut w, mut scales) = (Vec::new(), Vec::new());
    for m in parts {
        match m {
            Matrix::Fp8 {
                n: pn,
                k,
                w: pw,
                scales: ps,
            } => {
                if *k0.get_or_insert(k) != k {
                    return Err(format!("{name}: parts with different k"));
                }
                n += pn;
                w.extend_from_slice(&pw);
                scales.extend_from_slice(&ps);
            }
            Matrix::Bf16 { .. } => return Err(format!("{name}: a bf16 part")),
        }
    }
    let k = k0.ok_or_else(|| format!("{name}: no parts"))?;
    Ok((n, k, w, scales))
}

/// One staged piece: `(name, n, k, bytes)`.
type Piece = (String, usize, usize, Vec<u8>);

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn bytes_u16(b: &[u8]) -> Vec<u16> {
    b.chunks_exact(2)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect()
}

/// One layer's pieces as the graph takes them: `(name, n, k, bytes)` in image order.
fn stage_layer(
    source: &mut dyn TensorSource,
    c: &Qwen38Config,
    prefix: &str,
    l: usize,
) -> Result<Vec<Piece>, String> {
    let h = c.hidden;
    let p = format!("{prefix}layers.{l}");
    let name = format!("layers.{l}");
    let rm = |s: &mut dyn TensorSource, t: String, n, k| read_matrix(s, &t, n, k);
    let mut out = Vec::new();
    let push_fp8 =
        |out: &mut Vec<_>, key: &str, (n, k, w, s): (usize, usize, Vec<u8>, Vec<u16>)| {
            out.push((format!("{name}.{key}.w"), n, k, w));
            out.push((format!("{name}.{key}.s"), n / 128, k / 128, u16_bytes(&s)));
        };
    if c.full_attention[l] {
        let a = format!("{p}.self_attn");
        let mix = concat_fp8(
            &format!("{a} q|k|v"),
            vec![
                rm(
                    source,
                    format!("{a}.q_proj.weight"),
                    2 * c.heads * c.head_dim,
                    h,
                )?,
                rm(source, format!("{a}.k_proj.weight"), c.kv_dim(), h)?,
                rm(source, format!("{a}.v_proj.weight"), c.kv_dim(), h)?,
            ],
        )?;
        push_fp8(&mut out, "mix", mix);
        let o = concat_fp8(
            &format!("{a}.o_proj"),
            vec![rm(
                source,
                format!("{a}.o_proj.weight"),
                h,
                c.heads * c.head_dim,
            )?],
        )?;
        push_fp8(&mut out, "o", o);
    } else {
        let a = format!("{p}.linear_attn");
        let mix = concat_fp8(
            &format!("{a} qkv|z"),
            vec![
                rm(source, format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h)?,
                rm(
                    source,
                    format!("{a}.in_proj_z.weight"),
                    c.lin_value_dim(),
                    h,
                )?,
            ],
        )?;
        push_fp8(&mut out, "mix", mix);
        let ia = rm(source, format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?;
        let ib = rm(source, format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?;
        let (an, ak, aw) = ab_rows(&ia, &ib).map_err(|e| format!("{a}: {e}"))?;
        out.push((format!("{name}.ab"), an, ak, u16_bytes(&aw)));
        let o = concat_fp8(
            &format!("{a}.out_proj"),
            vec![rm(
                source,
                format!("{a}.out_proj.weight"),
                h,
                c.lin_value_dim(),
            )?],
        )?;
        push_fp8(&mut out, "o", o);
    }
    let m = format!("{p}.mlp");
    let gu = concat_fp8(
        &format!("{m} gate|up"),
        vec![
            rm(source, format!("{m}.gate_proj.weight"), c.intermediate, h)?,
            rm(source, format!("{m}.up_proj.weight"), c.intermediate, h)?,
        ],
    )?;
    push_fp8(&mut out, "gate_up", gu);
    let dn = concat_fp8(
        &format!("{m}.down_proj"),
        vec![rm(
            source,
            format!("{m}.down_proj.weight"),
            h,
            c.intermediate,
        )?],
    )?;
    push_fp8(&mut out, "down", dn);
    Ok(out)
}

/// Lays one layer's pieces out from `start`: (entries, image bytes with padding).
fn image_of(layer: usize, start: u64, pieces: Vec<Piece>) -> (Vec<DrumEntry>, Vec<u8>) {
    let mut img = Vec::new();
    let mut entries = Vec::new();
    for (name, n, k, b) in pieces {
        entries.push(DrumEntry {
            name,
            offset: start + img.len() as u64,
            bytes: b.len() as u64,
            n,
            k,
        });
        img.extend_from_slice(&b);
        img.resize(round_up(img.len(), ALIGN), 0);
    }
    let _ = layer;
    (entries, img)
}

/// Writes the drum file `out` and its manifest `out.json` from checkpoint `dir`; returns the index.
pub fn build(dir: &Path, out: &Path) -> Result<DrumIndex, String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let mut f = File::create(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let mut layers = Vec::with_capacity(c.layers);
    let mut start = 0u64;
    for l in 0..c.layers {
        let pieces = stage_layer(&mut source, &c, &prefix, l)?;
        let (entries, img) = image_of(l, start, pieces);
        f.write_all(&img)
            .map_err(|e| format!("{}: {e}", out.display()))?;
        layers.push(DrumLayer {
            layer: l,
            attention: c.full_attention[l],
            start,
            len: img.len() as u64,
            sha256: sha256_hex(&img),
            entries,
        });
        start += img.len() as u64;
    }
    f.sync_all()
        .map_err(|e| format!("{}: {e}", out.display()))?;
    let ix = DrumIndex {
        format: FORMAT.into(),
        checkpoint: dir
            .file_name()
            .map(|s| s.to_string_lossy().into_owned())
            .unwrap_or_default(),
        layers,
    };
    ix.check()?;
    let js = serde_json::to_string_pretty(&ix).map_err(|e| e.to_string())?;
    std::fs::write(manifest_path(out), js).map_err(|e| format!("manifest: {e}"))?;
    Ok(ix)
}

/// `<drum>.json`.
pub fn manifest_path(drum: &Path) -> PathBuf {
    let mut p = drum.as_os_str().to_owned();
    p.push(".json");
    PathBuf::from(p)
}

/// Re-reads every layer (O_DIRECT) and compares its sha256 with the manifest: the layers that
/// differ (empty when the file is intact).
pub fn verify(drum: &Path, ix: &DrumIndex, threads: usize) -> Result<Vec<usize>, String> {
    let f = DrumFile::open(drum, threads)?;
    let max = ix.layers.iter().map(|l| l.len).max().unwrap_or(0) as usize;
    let mut buf = AlignedBuf::new(max);
    let mut bad = Vec::new();
    for l in &ix.layers {
        f.read(l.start, l.len as usize, &mut buf)?;
        if sha256_hex(&buf.as_slice()[..l.len as usize]) != l.sha256 {
            bad.push(l.layer);
        }
    }
    Ok(bad)
}

/// Uploads one layer from its image (`img[0]` is the byte at `layer.start`).
fn upload_layer(layer: &DrumLayer, img: &[u8]) -> Result<LayerTensors, String> {
    let ent: BTreeMap<&str, &DrumEntry> = layer
        .entries
        .iter()
        .map(|e| {
            (
                e.name
                    .rsplit_once(&format!("layers.{}.", layer.layer))
                    .map_or(e.name.as_str(), |x| x.1),
                e,
            )
        })
        .collect();
    let get = |k: &str| -> Result<(&DrumEntry, &[u8]), String> {
        let e = ent
            .get(k)
            .ok_or_else(|| format!("drum layer {}: no {k}", layer.layer))?;
        let o = (e.offset - layer.start) as usize;
        Ok((e, &img[o..o + e.bytes as usize]))
    };
    let fp8 = |key: &str| -> Result<Fp8Tensor, String> {
        let (w, wb) = get(&format!("{key}.w"))?;
        let (_, sb) = get(&format!("{key}.s"))?;
        Fp8Tensor::upload(1, w.n, w.k, wb, &bytes_u16(sb)).map_err(gpu("drum fp8 upload"))
    };
    let ab = if layer.attention {
        None
    } else {
        let (e, b) = get("ab")?;
        Some(Bf16Tensor::upload(e.n, e.k, &bytes_u16(b)).map_err(gpu("drum ab upload"))?)
    };
    Ok(LayerTensors {
        mix: fp8("mix")?,
        ab,
        o: fp8("o")?,
        gate_up: fp8("gate_up")?,
        down: fp8("down")?,
    })
}

/// How the layers are split under a budget.
#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct DrumPlan {
    pub budget: u64,
    /// Fixed part: everything that is not a layer image (embedding, heads, state, margin).
    pub fixed: u64,
    pub groups: Vec<(usize, usize)>,
    /// The first `pinned` groups stay resident.
    pub pinned: usize,
    pub pinned_bytes: u64,
    /// Largest streamed group (one staging buffer, one device copy).
    pub slot_bytes: u64,
    /// Bytes read from the NVMe per pass (every streamed group once).
    pub streamed_bytes: u64,
    /// fixed + pinned + 3 slots (2 staging + 1 on the device) when anything streams.
    pub peak: u64,
}

/// Groups of `group` consecutive layers; pins whole groups from the first while
/// `fixed + pinned + 3 * slot <= budget`. Refuses when even nothing pinned does not fit.
pub fn plan(
    layer_bytes: &[u64],
    group: usize,
    budget: u64,
    fixed: u64,
) -> Result<DrumPlan, String> {
    if group == 0 || layer_bytes.is_empty() {
        return Err("drum plan: no layers or group 0".into());
    }
    let groups: Vec<(usize, usize)> = (0..layer_bytes.len())
        .step_by(group)
        .map(|a| (a, (a + group).min(layer_bytes.len())))
        .collect();
    let gb = |&(a, b): &(usize, usize)| layer_bytes[a..b].iter().sum::<u64>();
    let all: u64 = layer_bytes.iter().sum();
    if fixed + all <= budget {
        return Ok(DrumPlan {
            budget,
            fixed,
            pinned: groups.len(),
            groups,
            pinned_bytes: all,
            slot_bytes: 0,
            streamed_bytes: 0,
            peak: fixed + all,
        });
    }
    let mut pinned = 0usize;
    let mut pinned_bytes = 0u64;
    loop {
        let slot = groups[pinned + 1..].iter().map(gb).max().unwrap_or(0);
        let g = gb(&groups[pinned]);
        if pinned + 1 < groups.len() && fixed + pinned_bytes + g + 3 * slot <= budget {
            pinned_bytes += g;
            pinned += 1;
        } else {
            break;
        }
    }
    let slot = groups[pinned..].iter().map(gb).max().unwrap_or(0);
    let peak = fixed + pinned_bytes + 3 * slot;
    if peak > budget {
        return Err(format!(
            "drum plan: {peak} B needed with nothing more pinned, budget {budget} B (smaller groups?)"
        ));
    }
    Ok(DrumPlan {
        budget,
        fixed,
        streamed_bytes: all - pinned_bytes,
        pinned,
        groups,
        pinned_bytes,
        slot_bytes: slot,
        peak,
    })
}

/// A negative control (the run must stop matching the reference).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    None,
    /// One byte of the first streamed group's gate_up weight flipped after every read.
    FlipByte,
    /// The second streamed group runs on the first streamed group's tensors (a stale slot).
    StaleSlot,
}

/// One pass's costs (ns) and bytes.
#[derive(Clone, Debug, Default, Serialize)]
pub struct PassStats {
    pub wall_ns: u64,
    /// Main thread blocked waiting for a staging buffer to be full.
    pub wait_ns: u64,
    pub upload_ns: u64,
    /// Graph tokens and residual reads, all conversations.
    pub compute_ns: u64,
    /// Sum of the chip time the graphs report.
    pub chip_ns: u64,
    /// Reader thread time spent in reads for the groups this pass consumed.
    pub read_ns: u64,
    pub read_bytes: u64,
}

/// Per layer what stays resident: norms (and the state objects take the rest at open).
struct Small {
    in_norm: Vec<f32>,
    post_norm: Vec<f32>,
}

/// One conversation's per-group state and graphs (graphs first: drop order).
struct Conversation {
    graphs: Vec<Q38Graph>,
    _gdn: Vec<Option<Gdn>>,
    _attn: Vec<Option<Attn256Cache>>,
}

type Filled = Result<(usize, AlignedBuf, u64), String>;
/// What [`DrumModel::step`] returns: tokens, logits (when asked), costs.
pub type StepOut = (Vec<u32>, Vec<Vec<f32>>, PassStats);

/// The model run from the drum under a budget, `batch` conversations in lockstep.
pub struct DrumModel {
    convs: Vec<Conversation>,
    pinned: Vec<LayerTensors>,
    _lm_head: Bf16Tensor,
    _stand_in: Bf16Tensor,
    small: Vec<Small>,
    pub config: Qwen38Config,
    pub plan: DrumPlan,
    index: DrumIndex,
    embed: Vec<u16>,
    pub pos: usize,
    max_pos: usize,
    control: Control,
    free_tx: Option<mpsc::Sender<AlignedBuf>>,
    full_rx: Option<mpsc::Receiver<Filled>>,
    reader: Option<std::thread::JoinHandle<()>>,
}

impl std::fmt::Debug for DrumModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "DrumModel {{ plan: {:?}, pos: {} }}",
            self.plan, self.pos
        )
    }
}

/// DeltaNet state bytes per layer (48 heads x 128 x 128 f32 plus the conv tail) and KV bytes per
/// attention layer at `max_pos`: the conversation's share of the fixed part.
pub fn state_bytes(c: &Qwen38Config, max_pos: usize) -> u64 {
    let n_attn = c.full_attention.iter().filter(|&&a| a).count();
    let n_gdn = c.layers - n_attn;
    let gdn =
        c.lin_v_heads * c.lin_k_dim * c.lin_v_dim * 4 + c.conv_dim() * (c.conv_kernel - 1) * 4;
    (n_gdn * gdn + n_attn * Attn256Cache::layer_bytes(max_pos)) as u64
}

impl DrumModel {
    /// Opens checkpoint `dir` (resident parts) and drum `drum` (layer images); plans the groups
    /// under `budget` with `margin` bytes for graphs and code, and starts the reader.
    #[allow(clippy::too_many_arguments)]
    pub fn open(
        dir: &Path,
        drum: &Path,
        budget: u64,
        group: usize,
        batch: usize,
        max_pos: usize,
        margin: u64,
        threads: usize,
        control: Control,
    ) -> Result<Self, String> {
        let c = Qwen38Config::from_checkpoint(dir)?;
        let index = DrumIndex::load(&manifest_path(drum))?;
        if index.layers.len() != c.layers {
            return Err(format!(
                "drum: {} layers, checkpoint {}",
                index.layers.len(),
                c.layers
            ));
        }
        let (v, h) = (c.vocab, c.hidden);
        let fixed = (2 * v * h * 2 + STAND_IN_VOCAB * h * 2) as u64
            + batch as u64 * state_bytes(&c, max_pos)
            + margin;
        let plan = plan(&index.layer_bytes(), group, budget, fixed)?;

        let mut source = ShardSource::open(dir)?;
        let prefix = text_prefix(&mut source)?;
        let embed = read_bf16_words(
            &mut source,
            &format!("{prefix}embed_tokens.weight"),
            &[v, h],
        )?;
        let final_norm = fold(read_vec(&mut source, &format!("{prefix}norm.weight"), h)?);
        let lm_head = match read_matrix(&mut source, "lm_head.weight", v, h)? {
            Matrix::Bf16 { n, k, w } => Bf16Tensor::upload(n, k, &w).map_err(gpu("lm_head"))?,
            Matrix::Fp8 { .. } => return Err("lm_head: FP8".into()),
        };
        let stand_in = Bf16Tensor::upload(STAND_IN_VOCAB, h, &vec![0u16; STAND_IN_VOCAB * h])
            .map_err(gpu("stand-in head"))?;

        // Small per-layer parameters, read once.
        struct GdnP {
            conv: Vec<f32>,
            a_log: Vec<f32>,
            dt_bias: Vec<f32>,
            norm: Vec<f32>,
        }
        let mut small = Vec::with_capacity(c.layers);
        let mut gdn_p: Vec<Option<GdnP>> = Vec::with_capacity(c.layers);
        let mut qk: Vec<Option<(Vec<f32>, Vec<f32>)>> = Vec::with_capacity(c.layers);
        for l in 0..c.layers {
            let p = format!("{prefix}layers.{l}");
            small.push(Small {
                in_norm: fold(read_vec(
                    &mut source,
                    &format!("{p}.input_layernorm.weight"),
                    h,
                )?),
                post_norm: fold(read_vec(
                    &mut source,
                    &format!("{p}.post_attention_layernorm.weight"),
                    h,
                )?),
            });
            if c.full_attention[l] {
                let a = format!("{p}.self_attn");
                qk.push(Some((
                    read_vec(&mut source, &format!("{a}.q_norm.weight"), c.head_dim)?,
                    read_vec(&mut source, &format!("{a}.k_norm.weight"), c.head_dim)?,
                )));
                gdn_p.push(None);
            } else {
                let a = format!("{p}.linear_attn");
                gdn_p.push(Some(GdnP {
                    conv: read_bf16_words(
                        &mut source,
                        &format!("{a}.conv1d.weight"),
                        &[c.conv_dim(), 1, c.conv_kernel],
                    )?
                    .into_iter()
                    .map(bf16_to_f32)
                    .collect(),
                    a_log: read_vec(&mut source, &format!("{a}.A_log"), c.lin_v_heads)?,
                    dt_bias: read_vec(&mut source, &format!("{a}.dt_bias"), c.lin_v_heads)?,
                    norm: read_vec(&mut source, &format!("{a}.norm.weight"), c.lin_v_dim)?,
                }));
                qk.push(None);
            }
        }
        drop(source);

        // Per conversation and group: state objects and a graph.
        let last = plan.groups.len() - 1;
        let mut convs = Vec::with_capacity(batch);
        for _ in 0..batch {
            let (mut graphs, mut gdns, mut attns) = (Vec::new(), Vec::new(), Vec::new());
            for (gi, &(a, b)) in plan.groups.iter().enumerate() {
                let n_attn = (a..b).filter(|&l| c.full_attention[l]).count();
                let n_gdn = (b - a) - n_attn;
                let mut gdn = if n_gdn > 0 {
                    Some(Gdn::alloc(n_gdn).map_err(gpu("gdn alloc"))?)
                } else {
                    None
                };
                let mut attn = if n_attn > 0 {
                    Some(
                        Attn256Cache::alloc(n_attn, max_pos, c.rope_theta, c.rotary_dim)
                            .map_err(gpu("attn256 alloc"))?,
                    )
                } else {
                    None
                };
                let (mut gx, mut ax, mut mask) = (0usize, 0usize, 0u64);
                for l in a..b {
                    if let Some((q, k)) = &qk[l] {
                        attn.as_mut()
                            .expect("attn")
                            .set_norms(ax, q, k)
                            .map_err(gpu("set_norms"))?;
                        ax += 1;
                        mask |= 1 << (l - a);
                    } else {
                        let g = gdn_p[l].as_ref().expect("gdn params");
                        gdn.as_mut()
                            .expect("gdn")
                            .set_layer(gx, &g.conv, &g.a_log, &g.dt_bias, &g.norm)
                            .map_err(gpu("gdn set_layer"))?;
                        gx += 1;
                    }
                }
                let head = if gi == last { &lm_head } else { &stand_in };
                let cfg = GraphConfig {
                    layers: (b - a) as u32,
                    attn_mask: mask,
                    vocab: head.n() as u32,
                    eps: c.eps,
                    eps_l2: EPS_L2,
                    gdn: gdn.as_ref().map_or(std::ptr::null_mut(), Gdn::as_raw),
                    attn: attn
                        .as_ref()
                        .map_or(std::ptr::null_mut(), Attn256Cache::as_raw),
                    lm_head: head.as_raw(),
                };
                // SAFETY: gdn, attn and the heads live in the returned model, which drops the
                // graphs first (field order: convs, then pinned/heads; the state objects sit
                // after the graphs inside Conversation).
                let mut graph =
                    unsafe { Q38Graph::create(&cfg) }.map_err(gpu("q38 graph create"))?;
                graph
                    .set_final_norm(&final_norm)
                    .map_err(gpu("set_final_norm"))?;
                graphs.push(graph);
                gdns.push(gdn);
                attns.push(attn);
            }
            convs.push(Conversation {
                graphs,
                _gdn: gdns,
                _attn: attns,
            });
        }

        // Pinned groups: read through the same O_DIRECT path, uploaded once, graphs pointed.
        let file = DrumFile::open(drum, threads)?;
        let mut pinned = Vec::new();
        if plan.pinned > 0 {
            let max = index.layers.iter().map(|l| l.len).max().unwrap_or(0) as usize;
            let mut buf = AlignedBuf::new(max);
            for l in 0..plan.groups[plan.pinned - 1].1 {
                let dl = &index.layers[l];
                file.read(dl.start, dl.len as usize, &mut buf)?;
                pinned.push(upload_layer(dl, &buf.as_slice()[..dl.len as usize])?);
            }
        }
        let mut m = Self {
            convs,
            pinned,
            _lm_head: lm_head,
            _stand_in: stand_in,
            small,
            config: c,
            plan,
            index,
            embed,
            pos: 0,
            max_pos,
            control,
            free_tx: None,
            full_rx: None,
            reader: None,
        };
        for gi in 0..m.plan.pinned {
            let (a, b) = m.plan.groups[gi];
            for conv in 0..m.convs.len() {
                for l in a..b {
                    // SAFETY: pinned tensors live as long as the model and outlive the graphs.
                    unsafe { m.point(conv, gi, l, l - a, None)? };
                }
            }
        }
        if m.plan.pinned < m.plan.groups.len() {
            m.start_reader(file)?;
        }
        Ok(m)
    }

    /// Points conversation `conv`'s graph `gi` layer `slot` at model layer `l`'s tensors
    /// (`t`, or the pinned copy when `None`).
    ///
    /// # Safety
    /// The tensors must stay alive while the graph can run with them.
    unsafe fn point(
        &mut self,
        conv: usize,
        gi: usize,
        l: usize,
        slot: usize,
        t: Option<&LayerTensors>,
    ) -> Result<(), String> {
        let t = t.unwrap_or(&self.pinned[l]);
        let s = &self.small[l];
        let ab =
            t.ab.as_ref()
                .map_or(std::ptr::null_mut(), Bf16Tensor::as_raw);
        self.convs[conv].graphs[gi]
            .set_layer(
                slot,
                t.mix.as_raw(),
                ab,
                t.o.as_raw(),
                t.gate_up.as_raw(),
                t.down.as_raw(),
                &s.in_norm,
                &s.post_norm,
            )
            .map_err(gpu("q38 graph set_layer"))
    }

    fn start_reader(&mut self, file: DrumFile) -> Result<(), String> {
        let streamed: Vec<(usize, u64, usize)> = (self.plan.pinned..self.plan.groups.len())
            .map(|gi| {
                let (a, b) = self.plan.groups[gi];
                let start = self.index.layers[a].start;
                let end = self.index.layers[b - 1].start + self.index.layers[b - 1].len;
                (gi, start, (end - start) as usize)
            })
            .collect();
        let (free_tx, free_rx) = mpsc::channel::<AlignedBuf>();
        let (full_tx, full_rx) = mpsc::sync_channel::<Filled>(2);
        for _ in 0..2 {
            free_tx
                .send(AlignedBuf::new(self.plan.slot_bytes as usize))
                .map_err(|_| "drum: reader channel")?;
        }
        let reader = std::thread::Builder::new()
            .name("drum-reader".into())
            .spawn(move || 'outer: loop {
                for &(gi, start, len) in &streamed {
                    let Ok(mut buf) = free_rx.recv() else {
                        break 'outer;
                    };
                    let t = Instant::now();
                    let r = file
                        .read(start, len, &mut buf)
                        .map(|()| (gi, buf, t.elapsed().as_nanos() as u64));
                    let failed = r.is_err();
                    if full_tx.send(r).is_err() || failed {
                        break 'outer;
                    }
                }
            })
            .map_err(|e| format!("drum reader: {e}"))?;
        self.free_tx = Some(free_tx);
        self.full_rx = Some(full_rx);
        self.reader = Some(reader);
        Ok(())
    }

    /// Conversations in the batch.
    pub fn batch(&self) -> usize {
        self.convs.len()
    }

    /// One pass: feeds `tokens[i]` to conversation `i` at the next position; returns each
    /// conversation's greedy next token, its logits when `want_logits`, and the pass costs.
    pub fn step(&mut self, tokens: &[u32], want_logits: bool) -> Result<StepOut, String> {
        let (b, h) = (self.convs.len(), self.config.hidden);
        if tokens.len() != b {
            return Err(format!(
                "drum step: {} tokens for {b} conversations",
                tokens.len()
            ));
        }
        if self.pos >= self.max_pos {
            return Err(format!(
                "drum step: the KV cache holds {} positions",
                self.max_pos
            ));
        }
        let wall = Instant::now();
        let mut st = PassStats::default();
        let mut xs: Vec<Vec<f32>> = tokens
            .iter()
            .map(|&t| {
                let t = t as usize;
                if t >= self.config.vocab {
                    return Err(format!("drum step: token {t} outside the vocab"));
                }
                Ok(self.embed[t * h..(t + 1) * h]
                    .iter()
                    .map(|&w| bf16_to_f32(w))
                    .collect())
            })
            .collect::<Result<_, String>>()?;
        let mut out = vec![0u32; b];
        let mut logits = Vec::new();
        let last = self.plan.groups.len() - 1;
        let mut held: Option<Vec<LayerTensors>> = None;
        for gi in 0..self.plan.groups.len() {
            let (a, e) = self.plan.groups[gi];
            let mut cur: Option<Vec<LayerTensors>> = None;
            if gi >= self.plan.pinned {
                let t = Instant::now();
                let rx = self.full_rx.as_ref().expect("reader running");
                let (got, mut buf, read_ns) = rx.recv().map_err(|_| "drum reader stopped")??;
                st.wait_ns += t.elapsed().as_nanos() as u64;
                st.read_ns += read_ns;
                if got != gi {
                    return Err(format!("drum: reader delivered group {got}, wanted {gi}"));
                }
                let base = self.index.layers[a].start;
                if self.control == Control::FlipByte && gi == self.plan.pinned {
                    let gu = self.index.layers[a]
                        .entries
                        .iter()
                        .find(|x| x.name.ends_with(".gate_up.w"))
                        .ok_or("drum: no gate_up")?;
                    let o = (gu.offset - base) as usize + 12345;
                    buf.as_mut_slice()[o] ^= 0x01;
                }
                let t = Instant::now();
                let stale = self.control == Control::StaleSlot && gi == self.plan.pinned + 1;
                let mut ts = Vec::with_capacity(e - a);
                if !stale {
                    for l in a..e {
                        let dl = &self.index.layers[l];
                        let o = (dl.start - base) as usize;
                        st.read_bytes += dl.len;
                        ts.push(upload_layer(dl, &buf.as_slice()[o..o + dl.len as usize])?);
                    }
                } else {
                    st.read_bytes += (a..e).map(|l| self.index.layers[l].len).sum::<u64>();
                }
                st.upload_ns += t.elapsed().as_nanos() as u64;
                self.free_tx
                    .as_ref()
                    .expect("reader running")
                    .send(buf)
                    .map_err(|_| "drum reader stopped")?;
                let use_ts = if stale {
                    held.take().ok_or("stale control: nothing held")?
                } else {
                    ts
                };
                for conv in 0..b {
                    for l in a..e {
                        // SAFETY: `use_ts` lives until after this group's tokens below.
                        unsafe { self.point(conv, gi, l, l - a, Some(&use_ts[l - a]))? };
                    }
                }
                cur = Some(use_ts);
            }
            let t = Instant::now();
            for (conv, x) in xs.iter_mut().enumerate() {
                let g = &mut self.convs[conv].graphs[gi];
                let (tok, gs) = g.token(x, self.pos).map_err(gpu("q38 token"))?;
                st.chip_ns += gs.chip_ns;
                if gi == last {
                    out[conv] = tok;
                    if want_logits {
                        let mut l = vec![0f32; self.config.vocab];
                        g.read(None, None, Some(&mut l))
                            .map_err(gpu("read logits"))?;
                        logits.push(l);
                    }
                } else {
                    g.read(Some(x), None, None).map_err(gpu("read residual"))?;
                }
            }
            st.compute_ns += t.elapsed().as_nanos() as u64;
            // The group's streamed tensors go now (before the next upload); the stale control
            // keeps the first streamed group's for the second.
            if self.control == Control::StaleSlot && gi == self.plan.pinned {
                held = cur.take();
            }
            drop(cur);
        }
        self.pos += 1;
        st.wall_ns = wall.elapsed().as_nanos() as u64;
        Ok((out, logits, st))
    }
}

impl Drop for DrumModel {
    fn drop(&mut self) {
        // Close both channels so the reader stops, then wait for it.
        self.free_tx.take();
        self.full_rx.take();
        if let Some(r) = self.reader.take() {
            let _ = r.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: u64 = 383_760_000; // DeltaNet layer image (calc:29339f1d)
    const A: u64 = 372_289_920; // attention layer image (calc:52d19dc7)

    fn q38_layers() -> Vec<u64> {
        (0..64).map(|l| if l % 4 == 3 { A } else { G }).collect()
    }

    #[test]
    fn plan_pins_all_when_it_fits() {
        let p = plan(&q38_layers(), 4, 64 << 30, 6 << 30).unwrap();
        assert_eq!(p.pinned, 16);
        assert_eq!(p.streamed_bytes, 0);
        assert_eq!(p.peak, (6 << 30) + q38_layers().iter().sum::<u64>());
    }

    #[test]
    fn plan_under_16_gib_streams_and_stays_under_budget() {
        let budget = 16u64 << 30;
        let p = plan(&q38_layers(), 4, budget, 6 << 30).unwrap();
        let group = 2 * G + G + A;
        assert!(p.pinned > 0 && p.pinned < 16, "{p:?}");
        assert_eq!(p.slot_bytes, group);
        assert_eq!(p.pinned_bytes, p.pinned as u64 * group);
        assert_eq!(p.streamed_bytes, (16 - p.pinned as u64) * group);
        assert!(p.peak <= budget);
        // One more pinned group would break the budget.
        assert!(p.fixed + p.pinned_bytes + group + 3 * group > budget);
    }

    #[test]
    fn plan_refuses_when_slots_do_not_fit() {
        assert!(plan(&q38_layers(), 16, 8 << 30, 6 << 30).is_err());
    }

    #[test]
    fn index_check_catches_overlap_and_misalignment() {
        let e = |name: &str, offset, bytes| DrumEntry {
            name: name.into(),
            offset,
            bytes,
            n: 1,
            k: 1,
        };
        let mut ix = DrumIndex {
            format: FORMAT.into(),
            checkpoint: "x".into(),
            layers: vec![
                DrumLayer {
                    layer: 0,
                    attention: false,
                    start: 0,
                    len: 8192,
                    sha256: String::new(),
                    entries: vec![e("a", 0, 100), e("b", 4096, 4096)],
                },
                DrumLayer {
                    layer: 1,
                    attention: true,
                    start: 8192,
                    len: 4096,
                    sha256: String::new(),
                    entries: vec![e("a", 8192, 10)],
                },
            ],
        };
        ix.check().unwrap();
        ix.layers[1].start = 4096;
        assert!(ix.check().is_err(), "overlap must be refused");
        ix.layers[1].start = 8192;
        ix.layers[0].entries[1].offset = 4000;
        assert!(ix.check().is_err(), "misaligned entry must be refused");
    }

    #[test]
    fn image_layout_is_aligned_and_round_trips() {
        let pieces = vec![
            ("layers.0.mix.w".to_string(), 2, 3, vec![1u8; 6]),
            ("layers.0.mix.s".to_string(), 1, 1, vec![2u8; 2]),
        ];
        let (ents, img) = image_of(0, 8192, pieces);
        assert_eq!(img.len(), 2 * ALIGN);
        assert_eq!(ents[0].offset, 8192);
        assert_eq!(ents[1].offset, 8192 + ALIGN as u64);
        assert_eq!(&img[..6], &[1u8; 6]);
        assert_eq!(&img[ALIGN..ALIGN + 2], &[2u8; 2]);
        assert_eq!(
            bytes_u16(&u16_bytes(&[0x1234, 0xabcd])),
            vec![0x1234, 0xabcd]
        );
    }

    /// The reader returns the file's bytes exactly, across chunk and thread boundaries, and a
    /// changed byte changes the layer's sha256 (what [`verify`] reports).
    #[test]
    fn direct_reader_round_trips_and_verify_catches_a_flip() {
        let dir = std::env::temp_dir().join(format!("drum-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("t.drum");
        let len = READ_CHUNK * 2 + 3 * ALIGN;
        let data: Vec<u8> = (0..len).map(|i| (i * 7 % 251) as u8).collect();
        std::fs::write(&path, &data).unwrap();
        let ix = DrumIndex {
            format: FORMAT.into(),
            checkpoint: "t".into(),
            layers: vec![DrumLayer {
                layer: 0,
                attention: false,
                start: 0,
                len: len as u64,
                sha256: sha256_hex(&data),
                entries: vec![],
            }],
        };
        let f = match DrumFile::open(&path, 3) {
            Ok(f) => f,
            // tmpfs refuses O_DIRECT: nothing to test on such a machine.
            Err(e) if e.contains("Invalid argument") => return,
            Err(e) => panic!("{e}"),
        };
        let mut buf = AlignedBuf::new(len);
        f.read(0, len, &mut buf).unwrap();
        assert!(buf.as_slice()[..len] == data[..]);
        assert!(verify(&path, &ix, 3).unwrap().is_empty());
        let mut bad = data.clone();
        bad[READ_CHUNK + 5] ^= 1;
        std::fs::write(&path, &bad).unwrap();
        assert_eq!(verify(&path, &ix, 3).unwrap(), vec![0]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
