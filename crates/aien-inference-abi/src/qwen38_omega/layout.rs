//! The resident layout of Qwen3.8-27B on the Omega engine (sc #676, FORTRAN idea 4: "families of
//! programs which deal with the same data").
//!
//! The 1956 FORTRAN manual: two programs with identical DIMENSION and EQUIVALENCE statements "will
//! have this region allocated identically". Here the region is the set of device images the engine
//! uploads: fused FP8 projections and their scale grids, the interleaved and 2^-64-shifted
//! `in_proj_a|b`, its transposed matmul copy, the folded norms, lm_head and the host embedding.
//!
//! - [`plan`] is the one ordered list of those images (name, element type, dims) for a stack of
//!   layers: the family's DIMENSION statements. Every tool builds from it.
//! - [`stage`] makes the images from a checkpoint, in plan order (every transform the engine
//!   applies lives here).
//! - A pack file (`<layout id>.q38layout`) holds the images exactly as uploaded, each at a 4 KiB
//!   boundary, plus a manifest with one fingerprint per image ([`entry_sha256`]). The layout id is
//!   the sha256 of the manifest bytes, so it names the content: the same checkpoint and the same
//!   transforms give the same id whoever builds it.
//! - [`Pack::open`] refuses a pack whose id is not the one the tool expects, and [`Pack::take`]
//!   refuses an image whose bytes do not hash to its manifest entry. Both refusals carry
//!   [`LAYOUT_REFUSAL_PREFIX`].
//!
//! A tool loading from the checkpoint computes the same fingerprints on the images it stages, so
//! equal ids from a checkpoint load and a pack load mean the two uploaded the same bytes, from
//! independent inputs.
//!
//! Threat model: the hash is taken on the exact slice handed to the upload, in the same thread,
//! right before the call. The pack is mapped read-only and written 0444; changing the file while a
//! tool has it open is outside what this checks (the embedding is read from the mapping for the
//! tool's lifetime after its one check at open).

use std::fs::File;
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sha2_11::{Digest, Sha256};

use crate::qwen38::{read_bf16_words, read_matrix, read_vec, text_prefix, Matrix, Qwen38Config};
use crate::qwen3_moe::bf16_to_f32;
use crate::qwen3_moe_omega::TensorSource;

#[cfg(not(target_endian = "little"))]
compile_error!("the q38 layout stores little-endian words and reads them in place");

/// Tag every layout refusal carries, so logs and receipts can be grepped.
pub const LAYOUT_REFUSAL_PREFIX: &str = "LAYOUT_REFUSED";

/// Pack file format (header, alignment, manifest encoding).
pub const FORMAT: &str = "aien-q38-layout/1";

/// What [`stage`] does to the checkpoint. Change it whenever a transform changes: the id moves
/// with it, and every tool pinned to the old id refuses the new pack.
pub const RULES: &str = "qwen38-omega/1: q|k|v, qkv|z, gate|up fused by rows (FP8 + bf16 scale \
grid); in_proj_a|b rows interleaved times 2^-64 (bf16) plus the plain [k][n] matmul copy; \
input/post/final norms folded to 1+w (f32); q/k norms, conv1d, A_log, dt_bias, gated norm as \
stored (f32); embed and lm_head bf16 as stored";

/// The layout of `~/models/qwen3.8-27b-fp8-017b9c7a` (Qwen/Qwen3.8-27B-FP8 at 017b9c7a), all 64
/// layers, under [`FORMAT`] and [`RULES`]. Built by `qwen38_layout build`; receipt in
/// `evidence/f4-layout-families-*/`.
pub const QWEN38_27B_FP8_017B9C7A: &str =
    "a6edf91182a0bd38827b94121f5e11cc8318a5f0aeb420e76e98217ff84d27a0";

/// Every image starts at a multiple of this (the page size: the mapped slices are word-aligned).
pub const ALIGN: u64 = 4096;

/// Fingerprint chunk: the bytes hash in parallel, one sha256 per chunk.
pub const CHUNK: usize = 1 << 20;

const MAGIC: &[u8; 8] = b"AIENQ38L";

fn refuse(what: impl std::fmt::Display) -> String {
    format!("{LAYOUT_REFUSAL_PREFIX}: {what}")
}

/// Element type of an image's bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    /// e4m3 FP8 weights.
    U8,
    /// bf16 words (weights, embed, FP8 scale grids).
    U16,
    /// f32 host vectors (norms, DeltaNet constants).
    F32,
}

impl Dtype {
    pub fn size(self) -> usize {
        match self {
            Dtype::U8 => 1,
            Dtype::U16 => 2,
            Dtype::F32 => 4,
        }
    }
}

/// One image of the layout.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EntrySpec {
    pub name: String,
    pub dtype: Dtype,
    pub dims: Vec<usize>,
}

impl EntrySpec {
    pub fn bytes(&self) -> usize {
        self.dims.iter().product::<usize>() * self.dtype.size()
    }
}

const FP8_BLOCK: usize = 128;

/// The ordered images of `c.layers` layers starting at model layer `first` (names carry the model
/// layer index, so a window's images are a subset of the whole model's), then nothing else: the
/// whole model is `plan(full, 0)`.
pub fn plan(c: &Qwen38Config, first: usize) -> Vec<EntrySpec> {
    let mut out = Vec::new();
    let mut e = |name: String, dtype, dims: Vec<usize>| out.push(EntrySpec { name, dtype, dims });
    let h = c.hidden;
    e("embed".into(), Dtype::U16, vec![c.vocab, h]);
    e("final_norm".into(), Dtype::F32, vec![h]);
    e("lm_head".into(), Dtype::U16, vec![c.vocab, h]);
    let fp8 = |e: &mut dyn FnMut(String, Dtype, Vec<usize>), name: String, n: usize, k: usize| {
        e(format!("{name}.w"), Dtype::U8, vec![n, k]);
        e(
            format!("{name}.s"),
            Dtype::U16,
            vec![n / FP8_BLOCK, k / FP8_BLOCK],
        );
    };
    for l in 0..c.layers {
        let p = format!("layers.{}", first + l);
        if c.full_attention[l] {
            let mix = 2 * c.heads * c.head_dim + 2 * c.kv_dim();
            fp8(&mut e, format!("{p}.mix"), mix, h);
            fp8(&mut e, format!("{p}.o"), h, c.heads * c.head_dim);
            e(format!("{p}.q_norm"), Dtype::F32, vec![c.head_dim]);
            e(format!("{p}.k_norm"), Dtype::F32, vec![c.head_dim]);
        } else {
            let mix = c.conv_dim() + c.lin_value_dim();
            fp8(&mut e, format!("{p}.mix"), mix, h);
            e(format!("{p}.ab"), Dtype::U16, vec![2 * c.lin_v_heads, h]);
            e(format!("{p}.ab_mm"), Dtype::U16, vec![h, 2 * c.lin_v_heads]);
            fp8(&mut e, format!("{p}.o"), h, c.lin_value_dim());
            e(
                format!("{p}.conv"),
                Dtype::F32,
                vec![c.conv_dim(), c.conv_kernel],
            );
            e(format!("{p}.a_log"), Dtype::F32, vec![c.lin_v_heads]);
            e(format!("{p}.dt_bias"), Dtype::F32, vec![c.lin_v_heads]);
            e(format!("{p}.gnorm"), Dtype::F32, vec![c.lin_v_dim]);
        }
        fp8(&mut e, format!("{p}.gate_up"), 2 * c.intermediate, h);
        fp8(&mut e, format!("{p}.down"), h, c.intermediate);
        e(format!("{p}.in_norm"), Dtype::F32, vec![h]);
        e(format!("{p}.post_norm"), Dtype::F32, vec![h]);
    }
    out
}

/// A read-only mapping of a pack file (pages shared with every other process mapping it).
pub struct Mapping {
    ptr: *mut libc::c_void,
    len: usize,
}

// SAFETY: the mapping is read-only for its whole life and unmapped only on drop.
unsafe impl Send for Mapping {}
unsafe impl Sync for Mapping {}

impl Mapping {
    fn open(file: &File, len: usize) -> Result<Self, String> {
        use std::os::fd::AsRawFd;
        if len == 0 {
            return Err("empty file".into());
        }
        // SAFETY: a fresh read-only private mapping of `len` bytes of an open file; checked below.
        let ptr = unsafe {
            libc::mmap(
                std::ptr::null_mut(),
                len,
                libc::PROT_READ,
                libc::MAP_PRIVATE,
                file.as_raw_fd(),
                0,
            )
        };
        if ptr == libc::MAP_FAILED {
            return Err(format!("mmap: {}", std::io::Error::last_os_error()));
        }
        Ok(Self { ptr, len })
    }

    fn bytes(&self) -> &[u8] {
        // SAFETY: ptr maps len readable bytes until drop.
        unsafe { std::slice::from_raw_parts(self.ptr as *const u8, self.len) }
    }

    /// Drops this process's page mappings of `off .. off + len` (the page cache keeps the file;
    /// a later read maps it again). Used after an image is on the device.
    fn release(&self, off: usize, len: usize) {
        self.advise(off, len, libc::MADV_DONTNEED);
    }

    /// Asks the kernel to start reading `off .. off + len` into the page cache now (no wait).
    fn prefetch(&self, off: usize, len: usize) {
        self.advise(off, len, libc::MADV_WILLNEED);
    }

    fn advise(&self, off: usize, len: usize, advice: libc::c_int) {
        let start = off & !(ALIGN as usize - 1);
        let end = (off + len).min(self.len);
        if end > start {
            // SAFETY: the range lies inside the mapping. On a private read-only mapping
            // MADV_DONTNEED only drops page table entries (the bytes read back unchanged) and
            // MADV_WILLNEED only starts readahead.
            unsafe {
                libc::madvise(
                    (self.ptr as *mut u8).add(start) as *mut libc::c_void,
                    end - start,
                    advice,
                );
            }
        }
    }
}

impl Drop for Mapping {
    fn drop(&mut self) {
        // SAFETY: ptr/len came from a successful mmap.
        unsafe {
            libc::munmap(self.ptr, self.len);
        }
    }
}

/// An image's bytes: made by [`stage`] or a slice of a pack mapping.
pub enum ImageData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
    Mapped {
        map: Arc<Mapping>,
        off: usize,
        len: usize,
    },
}

/// One image: its spec, its bytes and (from a pack) the fingerprint they were checked against.
pub struct Image {
    pub spec: EntrySpec,
    data: ImageData,
    sha256: Option<String>,
}

impl std::fmt::Debug for Image {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Image({} {:?} {:?})",
            self.spec.name, self.spec.dtype, self.spec.dims
        )
    }
}

impl Image {
    /// Checks the bytes against `spec` (length; type and alignment for a mapped slice).
    pub fn new(spec: EntrySpec, data: ImageData) -> Result<Self, String> {
        let (len, ok_type) = match &data {
            ImageData::U8(v) => (v.len(), spec.dtype == Dtype::U8),
            ImageData::U16(v) => (2 * v.len(), spec.dtype == Dtype::U16),
            ImageData::F32(v) => (4 * v.len(), spec.dtype == Dtype::F32),
            ImageData::Mapped { map, off, len } => (
                *len,
                off.is_multiple_of(spec.dtype.size())
                    && off + len <= map.len
                    && (map.ptr as usize).is_multiple_of(8),
            ),
        };
        if !ok_type || len != spec.bytes() {
            return Err(format!(
                "image {}: {len} bytes as {:?}, the layout says {:?} {:?} ({} bytes)",
                spec.name,
                data_kind(&data),
                spec.dtype,
                spec.dims,
                spec.bytes()
            ));
        }
        Ok(Self {
            spec,
            data,
            sha256: None,
        })
    }

    /// The bytes exactly as they go to the device (little-endian words).
    pub fn bytes(&self) -> &[u8] {
        match &self.data {
            ImageData::U8(v) => v,
            // SAFETY: plain little-endian words viewed as their bytes (target_endian checked).
            ImageData::U16(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, 2 * v.len())
            },
            ImageData::F32(v) => unsafe {
                std::slice::from_raw_parts(v.as_ptr() as *const u8, 4 * v.len())
            },
            ImageData::Mapped { map, off, len } => &map.bytes()[*off..*off + *len],
        }
    }

    pub fn u8(&self) -> &[u8] {
        debug_assert_eq!(self.spec.dtype, Dtype::U8);
        self.bytes()
    }

    pub fn u16(&self) -> &[u16] {
        debug_assert_eq!(self.spec.dtype, Dtype::U16);
        let b = self.bytes();
        // SAFETY: Image::new checked the type, the length and (mapped) the alignment.
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u16, b.len() / 2) }
    }

    pub fn f32(&self) -> &[f32] {
        debug_assert_eq!(self.spec.dtype, Dtype::F32);
        let b = self.bytes();
        // SAFETY: as for u16.
        unsafe { std::slice::from_raw_parts(b.as_ptr() as *const f32, b.len() / 4) }
    }

    /// The fingerprint a pack checked these bytes against, if any.
    pub fn checked_sha256(&self) -> Option<&str> {
        self.sha256.as_deref()
    }

    /// Lets go of this process's page mappings of a pack image (after its upload).
    pub fn release(&self) {
        if let ImageData::Mapped { map, off, len } = &self.data {
            map.release(*off, *len);
        }
    }

    /// Owned copy with one byte changed (the negative control's corrupted pack, made in memory so
    /// the file stays as built).
    pub fn flipped(&self) -> Result<Self, String> {
        let mut b = self.bytes().to_vec();
        let i = b.len() / 2;
        b[i] ^= 0x01;
        let data = match self.spec.dtype {
            Dtype::U8 => ImageData::U8(b),
            Dtype::U16 => ImageData::U16(
                b.chunks_exact(2)
                    .map(|c| u16::from_le_bytes([c[0], c[1]]))
                    .collect(),
            ),
            Dtype::F32 => ImageData::F32(
                b.chunks_exact(4)
                    .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                    .collect(),
            ),
        };
        Image::new(self.spec.clone(), data)
    }
}

fn data_kind(d: &ImageData) -> &'static str {
    match d {
        ImageData::U8(_) => "u8",
        ImageData::U16(_) => "u16",
        ImageData::F32(_) => "f32",
        ImageData::Mapped { .. } => "mapped",
    }
}

/// Fingerprint of an image: sha256 over (format tag, length, the sha256 of every [`CHUNK`]-byte
/// piece in order). The pieces hash in parallel.
pub fn entry_sha256(bytes: &[u8]) -> String {
    let pieces: Vec<[u8; 32]> = bytes
        .par_chunks(CHUNK)
        .map(|c| Sha256::digest(c).into())
        .collect();
    let mut h = Sha256::new();
    h.update(FORMAT.as_bytes());
    h.update([0u8]);
    h.update((bytes.len() as u64).to_le_bytes());
    for p in &pieces {
        h.update(p);
    }
    hex(&h.finalize())
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// One manifest line: where an image sits in the pack and what its bytes hash to.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ManifestEntry {
    pub name: String,
    pub dtype: Dtype,
    pub dims: Vec<usize>,
    pub offset: u64,
    pub bytes: u64,
    pub sha256: String,
}

/// The shape facts a tool checks before trusting a pack for its checkpoint.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ShapeFacts {
    pub hidden: usize,
    pub vocab: usize,
    pub intermediate: usize,
    pub first_layer: usize,
    pub layers: usize,
    /// One letter per layer: `a` full attention, `g` Gated DeltaNet.
    pub layer_kinds: String,
}

impl ShapeFacts {
    pub fn of(c: &Qwen38Config, first: usize) -> Self {
        Self {
            hidden: c.hidden,
            vocab: c.vocab,
            intermediate: c.intermediate,
            first_layer: first,
            layers: c.layers,
            layer_kinds: c
                .full_attention
                .iter()
                .map(|&a| if a { 'a' } else { 'g' })
                .collect(),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    pub format: String,
    pub rules: String,
    pub chunk_bytes: usize,
    pub shape: ShapeFacts,
    pub entries: Vec<ManifestEntry>,
}

impl Manifest {
    pub fn new(shape: ShapeFacts) -> Self {
        Self {
            format: FORMAT.into(),
            rules: RULES.into(),
            chunk_bytes: CHUNK,
            shape,
            entries: Vec::new(),
        }
    }

    /// Appends `spec` with fingerprint `sha256` at the next aligned offset (the same offsets a
    /// pack writer gives, so a manifest recorded during a checkpoint load equals the pack's).
    pub fn push(&mut self, spec: &EntrySpec, sha256: String) -> &ManifestEntry {
        let offset = self
            .entries
            .last()
            .map_or(ALIGN, |e| (e.offset + e.bytes).div_ceil(ALIGN) * ALIGN);
        self.entries.push(ManifestEntry {
            name: spec.name.clone(),
            dtype: spec.dtype,
            dims: spec.dims.clone(),
            offset,
            bytes: spec.bytes() as u64,
            sha256,
        });
        self.entries.last().expect("just pushed")
    }

    /// The manifest as stored (and hashed for the id).
    pub fn encode(&self) -> Vec<u8> {
        serde_json::to_vec_pretty(self).expect("a manifest always encodes")
    }

    /// The layout id: sha256 of [`Self::encode`].
    pub fn id(&self) -> String {
        hex(&Sha256::digest(self.encode()))
    }

    pub fn entry(&self, name: &str) -> Option<&ManifestEntry> {
        self.entries.iter().find(|e| e.name == name)
    }

    /// Every entry of `other` (a window recorded by a checkpoint load) is in `self` with the same
    /// type, dims and fingerprint. Returns how many matched, or the first difference.
    pub fn covers(&self, other: &Manifest) -> Result<usize, String> {
        for e in &other.entries {
            let mine = self
                .entry(&e.name)
                .ok_or_else(|| format!("{}: not in this layout", e.name))?;
            if (mine.dtype, &mine.dims, &mine.sha256) != (e.dtype, &e.dims, &e.sha256) {
                return Err(format!(
                    "{}: {:?} {:?} {} here, {:?} {:?} {} there",
                    e.name, mine.dtype, mine.dims, mine.sha256, e.dtype, e.dims, e.sha256
                ));
            }
        }
        Ok(other.entries.len())
    }
}

/// Test switch for the wrong-transform negative control: `stage` skips the `1 + w` fold of the
/// image named here (only `qwen38_layout build` reads it, from `AIEN_Q38_LAYOUT_MUTANT_NOFOLD`).
#[derive(Clone, Debug, Default)]
pub struct StageMutant {
    pub no_fold: Option<String>,
}

/// `1 + w`, the zero-centred RMSNorm weight the graph's norms multiply by.
pub(crate) fn fold(w: Vec<f32>) -> Vec<f32> {
    w.into_iter().map(|v| 1.0 + v).collect()
}

/// FP8 projections stacked by rows (every part `[n_i][k]`, whole 128-row blocks, so the scale
/// grids concatenate too): `(n, k, weights, scale grid)`.
fn fuse_fp8(name: &str, parts: Vec<Matrix>) -> Result<(usize, usize, Vec<u8>, Vec<u16>), String> {
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
                if pn % FP8_BLOCK != 0 {
                    return Err(format!(
                        "{name}: a part of {pn} rows (not whole 128 blocks)"
                    ));
                }
                n += pn;
                if w.is_empty() {
                    w = pw;
                    scales = ps;
                } else {
                    w.extend_from_slice(&pw);
                    scales.extend_from_slice(&ps);
                }
            }
            Matrix::Bf16 { .. } => {
                return Err(format!(
                    "{name}: a bf16 part (the graph takes FP8 projections)"
                ))
            }
        }
    }
    let k = k0.ok_or_else(|| format!("{name}: no parts"))?;
    Ok((n, k, w, scales))
}

/// `in_proj_a` / `in_proj_b` interleaved by rows and scaled by 2^-64 (exact for every normal
/// bf16: only the exponent moves; the checkpoint's values sit far above 2^-62).
pub(crate) fn ab_rows(a: &Matrix, b: &Matrix) -> Result<(usize, usize, Vec<u16>), String> {
    let (
        Matrix::Bf16 { n, k, w: wa },
        Matrix::Bf16 {
            n: nb,
            k: kb,
            w: wb,
        },
    ) = (a, b)
    else {
        return Err("in_proj_a / in_proj_b: expected bf16".into());
    };
    if (n, k) != (nb, kb) {
        return Err("in_proj_a / in_proj_b: shapes differ".into());
    }
    let scale = (2.0f32).powi(-64);
    let shift = |v: u16| -> Result<u16, String> {
        let x = bf16_to_f32(v) * scale;
        let bits = x.to_bits();
        if x != 0.0 && (bits & 0xffff) != 0 {
            return Err(format!(
                "in_proj_a/b: {} * 2^-64 is not exact in bf16",
                bf16_to_f32(v)
            ));
        }
        Ok((bits >> 16) as u16)
    };
    let mut out = Vec::with_capacity(2 * n * k);
    for h in 0..*n {
        for row in [&wa[h * k..(h + 1) * k], &wb[h * k..(h + 1) * k]] {
            for &v in row {
                out.push(shift(v)?);
            }
        }
    }
    Ok((2 * n, *k, out))
}

/// [`ab_rows`]' `[n][k]` words back at the checkpoint's values (times 2^64, exact: the reverse
/// exponent shift) and transposed to `[k][n]`, the resident matmul's B layout.
pub(crate) fn ab_plain_transposed(n: usize, k: usize, w: &[u16]) -> Vec<u16> {
    let unscale = |v: u16| ((bf16_to_f32(v) * aien_omega_q38::ROW_SCALE).to_bits() >> 16) as u16;
    let mut t = vec![0u16; n * k];
    for (j, row) in w.chunks_exact(k).enumerate() {
        for (i, &v) in row.iter().enumerate() {
            t[i * n + j] = unscale(v);
        }
    }
    t
}

/// Hands images to the consumer in [`plan`] order, each checked against its spec.
struct Emitter<'a> {
    next: std::vec::IntoIter<EntrySpec>,
    emit: &'a mut dyn FnMut(Image) -> Result<(), String>,
}

impl Emitter<'_> {
    fn send(&mut self, data: ImageData) -> Result<(), String> {
        let spec = self
            .next
            .next()
            .ok_or_else(|| "stage: more images than the plan".to_string())?;
        (self.emit)(Image::new(spec, data)?)
    }

    fn fp8(&mut self, name: &str, parts: Vec<Matrix>) -> Result<(), String> {
        let (_, _, w, s) = fuse_fp8(name, parts)?;
        self.send(ImageData::U8(w))?;
        self.send(ImageData::U16(s))
    }
}

/// Runs `f`, adding its seconds to `acc`.
fn timed<T>(acc: &mut f64, f: impl FnOnce() -> T) -> T {
    let t = std::time::Instant::now();
    let out = f();
    *acc += t.elapsed().as_secs_f64();
    out
}

/// Makes the images of `c.layers` layers from model layer `first` out of `source`, in [`plan`]
/// order, handing each to `emit` as soon as it exists (one projection in memory at a time).
/// Returns the seconds spent reading the checkpoint (the rest is transforms and `emit`).
pub fn stage(
    c: &Qwen38Config,
    first: usize,
    source: &mut dyn TensorSource,
    mutant: &StageMutant,
    emit: &mut dyn FnMut(Image) -> Result<(), String>,
) -> Result<f64, String> {
    let mut out = Emitter {
        next: plan(c, first).into_iter(),
        emit,
    };
    let mut rs = 0f64;
    let fold_unless = |name: &str, w: Vec<f32>| {
        if mutant.no_fold.as_deref() == Some(name) {
            w
        } else {
            fold(w)
        }
    };
    let prefix = text_prefix(source)?;
    let (v, h) = (c.vocab, c.hidden);
    let embed = timed(&mut rs, || {
        read_bf16_words(source, &format!("{prefix}embed_tokens.weight"), &[v, h])
    })?;
    out.send(ImageData::U16(embed))?;
    let final_norm = timed(&mut rs, || {
        read_vec(source, &format!("{prefix}norm.weight"), h)
    })?;
    out.send(ImageData::F32(fold_unless("final_norm", final_norm)))?;
    match timed(&mut rs, || read_matrix(source, "lm_head.weight", v, h))? {
        Matrix::Bf16 { w, .. } => out.send(ImageData::U16(w))?,
        Matrix::Fp8 { .. } => return Err("lm_head: FP8 (the graph takes a bf16 head)".into()),
    }
    for l in 0..c.layers {
        let lp = format!("layers.{}", first + l);
        let p = format!("{prefix}layers.{}", first + l);
        let mut rm = |name: String, n: usize, k: usize| {
            timed(&mut rs, || read_matrix(&mut *source, &name, n, k))
        };
        if c.full_attention[l] {
            let a = format!("{p}.self_attn");
            let parts = vec![
                rm(format!("{a}.q_proj.weight"), 2 * c.heads * c.head_dim, h)?,
                rm(format!("{a}.k_proj.weight"), c.kv_dim(), h)?,
                rm(format!("{a}.v_proj.weight"), c.kv_dim(), h)?,
            ];
            out.fp8(&format!("{a} q|k|v"), parts)?;
            let o = rm(format!("{a}.o_proj.weight"), h, c.heads * c.head_dim)?;
            out.fp8(&format!("{a}.o_proj"), vec![o])?;
            let (q_norm, k_norm) = timed(&mut rs, || -> Result<_, String> {
                Ok((
                    read_vec(source, &format!("{a}.q_norm.weight"), c.head_dim)?,
                    read_vec(source, &format!("{a}.k_norm.weight"), c.head_dim)?,
                ))
            })?;
            out.send(ImageData::F32(q_norm))?;
            out.send(ImageData::F32(k_norm))?;
        } else {
            let a = format!("{p}.linear_attn");
            let parts = vec![
                rm(format!("{a}.in_proj_qkv.weight"), c.conv_dim(), h)?,
                rm(format!("{a}.in_proj_z.weight"), c.lin_value_dim(), h)?,
            ];
            out.fp8(&format!("{a} qkv|z"), parts)?;
            let ia = rm(format!("{a}.in_proj_a.weight"), c.lin_v_heads, h)?;
            let ib = rm(format!("{a}.in_proj_b.weight"), c.lin_v_heads, h)?;
            let o = rm(format!("{a}.out_proj.weight"), h, c.lin_value_dim())?;
            let (an, ak, aw) = ab_rows(&ia, &ib).map_err(|e| format!("{a}: {e}"))?;
            let ab_mm = ab_plain_transposed(an, ak, &aw);
            out.send(ImageData::U16(aw))?;
            out.send(ImageData::U16(ab_mm))?;
            out.fp8(&format!("{a}.out_proj"), vec![o])?;
            let (conv, a_log, dt_bias, norm) = timed(&mut rs, || -> Result<_, String> {
                let conv: Vec<f32> = read_bf16_words(
                    source,
                    &format!("{a}.conv1d.weight"),
                    &[c.conv_dim(), 1, c.conv_kernel],
                )?
                .into_iter()
                .map(bf16_to_f32)
                .collect();
                Ok((
                    conv,
                    read_vec(source, &format!("{a}.A_log"), c.lin_v_heads)?,
                    read_vec(source, &format!("{a}.dt_bias"), c.lin_v_heads)?,
                    read_vec(source, &format!("{a}.norm.weight"), c.lin_v_dim)?,
                ))
            })?;
            out.send(ImageData::F32(conv))?;
            out.send(ImageData::F32(a_log))?;
            out.send(ImageData::F32(dt_bias))?;
            out.send(ImageData::F32(norm))?;
        }
        let m = format!("{p}.mlp");
        let mut rm = |name: String, n: usize, k: usize| {
            timed(&mut rs, || read_matrix(&mut *source, &name, n, k))
        };
        let parts = vec![
            rm(format!("{m}.gate_proj.weight"), c.intermediate, h)?,
            rm(format!("{m}.up_proj.weight"), c.intermediate, h)?,
        ];
        out.fp8(&format!("{m} gate|up"), parts)?;
        let d = rm(format!("{m}.down_proj.weight"), h, c.intermediate)?;
        out.fp8(&format!("{m}.down_proj"), vec![d])?;
        let (in_norm, post_norm) = timed(&mut rs, || -> Result<_, String> {
            Ok((
                read_vec(source, &format!("{p}.input_layernorm.weight"), h)?,
                read_vec(source, &format!("{p}.post_attention_layernorm.weight"), h)?,
            ))
        })?;
        out.send(ImageData::F32(fold_unless(
            &format!("{lp}.in_norm"),
            in_norm,
        )))?;
        out.send(ImageData::F32(fold_unless(
            &format!("{lp}.post_norm"),
            post_norm,
        )))?;
    }
    if out.next.next().is_some() {
        return Err("stage: fewer images than the plan".into());
    }
    Ok(rs)
}

/// An open pack: its mapping, its manifest and its id.
pub struct Pack {
    pub path: PathBuf,
    pub id: String,
    pub manifest: Manifest,
    map: Arc<Mapping>,
}

impl std::fmt::Debug for Pack {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Pack({}, {})", self.path.display(), self.id)
    }
}

/// Header: magic, format version 1 (u32), 0 (u32), manifest offset (u64), manifest length (u64).
fn header(manifest_off: u64, manifest_len: u64) -> Vec<u8> {
    let mut h = vec![0u8; ALIGN as usize];
    h[..8].copy_from_slice(MAGIC);
    h[8..12].copy_from_slice(&1u32.to_le_bytes());
    h[16..24].copy_from_slice(&manifest_off.to_le_bytes());
    h[24..32].copy_from_slice(&manifest_len.to_le_bytes());
    h
}

impl Pack {
    /// Maps `path` and refuses it unless its layout id is `expect`, its format and rules are this
    /// build's, and every entry lies inside the file at an aligned offset. No image is read yet.
    pub fn open(path: &Path, expect: &str) -> Result<Self, String> {
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let len = file
            .metadata()
            .map_err(|e| format!("{}: {e}", path.display()))?
            .len() as usize;
        if len < ALIGN as usize {
            return Err(refuse(format!("{}: shorter than a header", path.display())));
        }
        let map = Mapping::open(&file, len).map_err(|e| format!("{}: {e}", path.display()))?;
        let b = map.bytes();
        if &b[..8] != MAGIC || u32::from_le_bytes(b[8..12].try_into().unwrap()) != 1 {
            return Err(refuse(format!("{}: not a q38 layout pack", path.display())));
        }
        let moff = u64::from_le_bytes(b[16..24].try_into().unwrap()) as usize;
        let mlen = u64::from_le_bytes(b[24..32].try_into().unwrap()) as usize;
        if moff.checked_add(mlen).is_none_or(|end| end > len) {
            return Err(refuse(format!(
                "{}: manifest outside the file",
                path.display()
            )));
        }
        let mbytes = &b[moff..moff + mlen];
        let id = hex(&Sha256::digest(mbytes));
        if id != expect {
            return Err(refuse(format!(
                "{}: layout id {id}, this tool expects {expect}",
                path.display()
            )));
        }
        let manifest: Manifest = serde_json::from_slice(mbytes)
            .map_err(|e| refuse(format!("{}: manifest: {e}", path.display())))?;
        if manifest.format != FORMAT || manifest.rules != RULES || manifest.chunk_bytes != CHUNK {
            return Err(refuse(format!(
                "{}: format/rules of another build ({} / {})",
                path.display(),
                manifest.format,
                manifest.rules
            )));
        }
        for e in &manifest.entries {
            let spec = EntrySpec {
                name: e.name.clone(),
                dtype: e.dtype,
                dims: e.dims.clone(),
            };
            if e.offset % ALIGN != 0
                || e.bytes != spec.bytes() as u64
                || e.offset
                    .checked_add(e.bytes)
                    .is_none_or(|end| end > moff as u64)
            {
                return Err(refuse(format!(
                    "{}: entry {} out of place",
                    path.display(),
                    e.name
                )));
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            id,
            manifest,
            map: Arc::new(map),
        })
    }

    /// Starts reading the named entries into the page cache (readahead; returns at once).
    pub fn prefetch(&self, names: &[&str]) {
        for n in names {
            if let Some(e) = self.manifest.entry(n) {
                self.map.prefetch(e.offset as usize, e.bytes as usize);
            }
        }
    }

    /// The image for `spec` from the pack, its bytes hashed and held to the manifest. `flip`
    /// corrupts one byte first (the negative control).
    pub fn take(&self, spec: &EntrySpec, flip: bool) -> Result<Image, String> {
        let e = self
            .manifest
            .entry(&spec.name)
            .ok_or_else(|| refuse(format!("{}: no entry {}", self.path.display(), spec.name)))?;
        if e.dtype != spec.dtype || e.dims != spec.dims {
            return Err(refuse(format!(
                "{}: entry {} is {:?} {:?}, this tool's layout says {:?} {:?}",
                self.path.display(),
                e.name,
                e.dtype,
                e.dims,
                spec.dtype,
                spec.dims
            )));
        }
        let mut img = Image::new(
            spec.clone(),
            ImageData::Mapped {
                map: self.map.clone(),
                off: e.offset as usize,
                len: e.bytes as usize,
            },
        )?;
        if flip {
            img = img.flipped()?;
        }
        let got = entry_sha256(img.bytes());
        if got != e.sha256 && !mutant_skip_verify() {
            return Err(refuse(format!(
                "{}: entry {} hashes to {got}, its manifest says {}",
                self.path.display(),
                e.name,
                e.sha256
            )));
        }
        img.sha256 = Some(got);
        Ok(img)
    }
}

/// Mutant switch for the unit tests' red run: `AIEN_Q38_LAYOUT_MUTANT_SKIP_VERIFY=1` makes
/// [`Pack::take`] accept a mismatched image. A check that cannot fail proves nothing; this shows
/// the tamper tests fail without it.
fn mutant_skip_verify() -> bool {
    std::env::var_os("AIEN_Q38_LAYOUT_MUTANT_SKIP_VERIFY").is_some_and(|v| v == "1")
}

/// Writes a pack: images arrive in plan order and go straight to disk (never the whole model in
/// memory). [`PackWriter::finish`] names the file by its id and makes it read-only.
pub struct PackWriter {
    dir: PathBuf,
    tmp: PathBuf,
    file: std::io::BufWriter<File>,
    manifest: Manifest,
    pos: u64,
}

impl PackWriter {
    pub fn create(dir: &Path, shape: ShapeFacts) -> Result<Self, String> {
        std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        let tmp = dir.join(format!(".building-{}.q38layout", std::process::id()));
        let file = File::create(&tmp).map_err(|e| format!("{}: {e}", tmp.display()))?;
        let mut file = std::io::BufWriter::with_capacity(8 << 20, file);
        file.write_all(&header(0, 0))
            .map_err(|e| format!("{}: {e}", tmp.display()))?;
        Ok(Self {
            dir: dir.to_path_buf(),
            tmp,
            file,
            manifest: Manifest::new(shape),
            pos: ALIGN,
        })
    }

    fn pad_to(&mut self, off: u64) -> Result<(), String> {
        let pad = (off - self.pos) as usize;
        self.file
            .write_all(&vec![0u8; pad])
            .map_err(|e| format!("{}: {e}", self.tmp.display()))?;
        self.pos = off;
        Ok(())
    }

    pub fn add(&mut self, img: &Image) -> Result<(), String> {
        let sha = entry_sha256(img.bytes());
        let (off, len) = {
            let e = self.manifest.push(&img.spec, sha);
            (e.offset, e.bytes)
        };
        self.pad_to(off)?;
        self.file
            .write_all(img.bytes())
            .map_err(|e| format!("{}: {e}", self.tmp.display()))?;
        self.pos += len;
        Ok(())
    }

    /// Writes the manifest and header, syncs, renames to `<id>.q38layout` (mode 0444) and writes
    /// `<id>.manifest.json` beside it. Returns the pack path and id.
    pub fn finish(mut self) -> Result<(PathBuf, String), String> {
        let io = |e: std::io::Error| e.to_string();
        let moff = self.pos.div_ceil(ALIGN) * ALIGN;
        self.pad_to(moff)?;
        let m = self.manifest.encode();
        self.file.write_all(&m).map_err(io)?;
        let mut file = self.file.into_inner().map_err(|e| e.to_string())?;
        file.seek(SeekFrom::Start(0)).map_err(io)?;
        file.write_all(&header(moff, m.len() as u64)).map_err(io)?;
        file.sync_all().map_err(io)?;
        drop(file);
        let id = self.manifest.id();
        let path = self.dir.join(format!("{id}.q38layout"));
        std::fs::rename(&self.tmp, &path).map_err(io)?;
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o444)).map_err(io)?;
        std::fs::write(self.dir.join(format!("{id}.manifest.json")), &m).map_err(io)?;
        Ok((path, id))
    }
}

/// Reads a pack's id without trusting it (for `qwen38_layout show` and receipts).
pub fn peek_id(path: &Path) -> Result<String, String> {
    use std::io::Read;
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = vec![0u8; ALIGN as usize];
    f.read_exact(&mut h)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    if &h[..8] != MAGIC {
        return Err(refuse(format!("{}: not a q38 layout pack", path.display())));
    }
    let moff = u64::from_le_bytes(h[16..24].try_into().unwrap());
    let mlen = u64::from_le_bytes(h[24..32].try_into().unwrap()) as usize;
    f.seek(SeekFrom::Start(moff))
        .map_err(|e| format!("{}: {e}", path.display()))?;
    let mut m = vec![0u8; mlen];
    f.read_exact(&mut m)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(hex(&Sha256::digest(&m)))
}

/// The layout id a tool expects: [`QWEN38_27B_FP8_017B9C7A`], unless `AIEN_Q38_LAYOUT_EXPECT`
/// names another one (said out loud on stderr; for a rebuilt layout under test).
pub fn expected_id() -> String {
    match std::env::var("AIEN_Q38_LAYOUT_EXPECT") {
        Ok(id) if !id.is_empty() => {
            eprintln!(
                "layout: AIEN_Q38_LAYOUT_EXPECT={id} replaces the pinned {QWEN38_27B_FP8_017B9C7A}"
            );
            id
        }
        _ => QWEN38_27B_FP8_017B9C7A.to_string(),
    }
}

/// Takes `--layout <pack>` out of a tool's arguments.
pub fn take_layout_arg(args: &mut Vec<String>) -> Result<Option<PathBuf>, String> {
    match args.iter().position(|a| a == "--layout") {
        None => Ok(None),
        Some(i) if i + 1 < args.len() => {
            let p = PathBuf::from(args.remove(i + 1));
            args.remove(i);
            Ok(Some(p))
        }
        Some(_) => Err("--layout needs a pack path".into()),
    }
}

/// Opens a pack for a tool against [`expected_id`], with one line on stderr.
pub fn open_for_tool(path: &Path) -> Result<Pack, String> {
    let t = std::time::Instant::now();
    let pack = Pack::open(path, &expected_id())?;
    eprintln!(
        "layout: pack {} id {} ({} images) opened in {:.3} s",
        path.display(),
        pack.id,
        pack.manifest.entries.len(),
        t.elapsed().as_secs_f64()
    );
    Ok(pack)
}

/// The negative control's corrupted image: `AIEN_Q38_LAYOUT_FLIP=<entry name>`.
pub fn flip_from_env() -> Option<String> {
    std::env::var("AIEN_Q38_LAYOUT_FLIP")
        .ok()
        .filter(|s| !s.is_empty())
}

/// This process's resident pages by where they come from (`/proc/self/smaps`): layout packs,
/// the driver's device mappings (`/dev/nvidia*`, where omega's buffers show on the Spark), other
/// files, and anonymous memory (heap).
#[derive(Clone, Copy, Debug, Default, Serialize)]
pub struct ResidentSplit {
    pub pack: u64,
    pub driver: u64,
    pub other_file: u64,
    pub anon: u64,
}

pub fn resident_split() -> Result<ResidentSplit, String> {
    let s = std::fs::read_to_string("/proc/self/smaps").map_err(|e| format!("smaps: {e}"))?;
    let mut out = ResidentSplit::default();
    let mut kind = 3u8;
    for line in s.lines() {
        let first = line.split_whitespace().next().unwrap_or("");
        if first.contains('-') && first.chars().all(|c| c.is_ascii_hexdigit() || c == '-') {
            let path = line.split_whitespace().nth(5).unwrap_or("");
            kind = if path.ends_with(".q38layout") {
                0
            } else if path.starts_with("/dev/nvidia") {
                1
            } else if path.starts_with('/') {
                2
            } else {
                3
            };
        } else if let Some(v) = line.strip_prefix("Rss:") {
            let kb: u64 = v
                .trim()
                .trim_end_matches("kB")
                .trim()
                .parse()
                .map_err(|_| format!("smaps: {line}"))?;
            let b = kb * 1024;
            match kind {
                0 => out.pack += b,
                1 => out.driver += b,
                2 => out.other_file += b,
                _ => out.anon += b,
            }
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen38::tests::{tiny, tiny_source};

    fn staged(c: &Qwen38Config, first: usize, mutant: &StageMutant) -> Vec<Image> {
        let mut src = tiny_source(&tiny());
        let mut out = Vec::new();
        stage(c, first, &mut src, mutant, &mut |img| {
            out.push(img);
            Ok(())
        })
        .unwrap();
        out
    }

    fn scratch(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("q38layout-{tag}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    fn build(dir: &Path, c: &Qwen38Config, mutant: &StageMutant) -> (PathBuf, String) {
        let mut w = PackWriter::create(dir, ShapeFacts::of(c, 0)).unwrap();
        for img in staged(c, 0, mutant) {
            w.add(&img).unwrap();
        }
        w.finish().unwrap()
    }

    #[test]
    fn stage_follows_the_plan() {
        let c = tiny();
        let imgs = staged(&c, 0, &StageMutant::default());
        let p = plan(&c, 0);
        assert_eq!(imgs.len(), p.len());
        for (i, s) in imgs.iter().zip(&p) {
            assert_eq!(&i.spec, s);
        }
        // 3 globals + 3 DeltaNet layers of 16 images + 1 attention layer of 12.
        assert_eq!(p.len(), 3 + 3 * 16 + 12);
    }

    #[test]
    fn a_window_is_a_subset_of_the_whole() {
        let c = tiny();
        let whole = plan(&c, 0);
        let win = plan(&c.window(2, 2).unwrap(), 2);
        for s in &win {
            assert!(whole.contains(s), "{} missing from the whole plan", s.name);
        }
    }

    #[test]
    fn pack_round_trip_gives_the_staged_bytes() {
        let c = tiny();
        let dir = scratch("rt");
        let (path, id) = build(&dir, &c, &StageMutant::default());
        let pack = Pack::open(&path, &id).unwrap();
        let mut recorded = Manifest::new(ShapeFacts::of(&c, 0));
        for (img, spec) in staged(&c, 0, &StageMutant::default())
            .iter()
            .zip(plan(&c, 0))
        {
            let got = pack.take(&spec, false).unwrap();
            assert_eq!(got.bytes(), img.bytes(), "{}", spec.name);
            recorded.push(&spec, entry_sha256(img.bytes()));
        }
        // A checkpoint load records the same manifest, so it names the same id.
        assert_eq!(recorded.id(), id);
        // Two builds from the same checkpoint name the same file.
        let (_, id2) = build(&scratch("rt2"), &c, &StageMutant::default());
        assert_eq!(id2, id);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_wrong_id_is_refused_before_any_image() {
        let c = tiny();
        let dir = scratch("id");
        let (path, id) = build(&dir, &c, &StageMutant::default());
        let mut other = id.clone().into_bytes();
        other[0] = if other[0] == b'0' { b'1' } else { b'0' };
        let err = Pack::open(&path, std::str::from_utf8(&other).unwrap()).unwrap_err();
        assert!(err.starts_with(LAYOUT_REFUSAL_PREFIX), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_corrupted_image_is_refused() {
        let c = tiny();
        let dir = scratch("flip");
        let (path, id) = build(&dir, &c, &StageMutant::default());
        let pack = Pack::open(&path, &id).unwrap();
        for spec in plan(&c, 0) {
            let err = pack.take(&spec, true).unwrap_err();
            assert!(err.starts_with(LAYOUT_REFUSAL_PREFIX), "{err}");
            assert!(err.contains(&spec.name), "{err}");
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_changed_transform_changes_the_id() {
        let c = tiny();
        let (_, good) = build(&scratch("t1"), &c, &StageMutant::default());
        let bad = StageMutant {
            no_fold: Some("layers.0.in_norm".into()),
        };
        let (path, bad_id) = build(&scratch("t2"), &c, &bad);
        assert_ne!(good, bad_id);
        // The wrong-transform pack is internally consistent: only the pinned id catches it.
        let pack = Pack::open(&path, &bad_id).unwrap();
        for spec in plan(&c, 0) {
            pack.take(&spec, false).unwrap();
        }
        assert!(Pack::open(&path, &good).is_err());
    }

    #[test]
    fn the_tiny_layout_id_is_pinned() {
        // Any change to plan, stage or the pack format moves this id: bump RULES (or FORMAT) and
        // re-pin on purpose, never by accident.
        let (_, id) = build(&scratch("pin"), &tiny(), &StageMutant::default());
        assert_eq!(id, TINY_LAYOUT_ID);
    }

    const TINY_LAYOUT_ID: &str = "9d27663a225b922d2597b4fe9bff4489790d12a7cb33fb35780889b85b8229a4";
}
