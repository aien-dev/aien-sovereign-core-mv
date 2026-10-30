//! Canonical byte formats: the expert artifact and the versioned index.
//!
//! Artifact (all integers little endian):
//!
//! ```text
//! 0..4096                 header region (fields below, every other byte zero)
//! 4096..4096+W            FP8 E4M3 weight bytes, row major [rows][cols], as in the checkpoint
//! S..S+L                  scale bytes [ceil(rows/128)][ceil(cols/128)], as in the checkpoint
//!                         (S = round_up(4096 + W, 4096))
//! S+L..object_len         zero padding to a 4096 multiple
//! ```
//!
//! Header fields: magic `AIENXPRT` (0..8), version (8..12), header_len 4096
//! (12..16), model digest (16..48), layer (48..52), expert (52..56),
//! projection (56), quantization (57), weight dtype (58, 1 = F8_E4M3), scale
//! dtype (59), block (60..64), rows (64..72), cols (72..80), scale rows
//! (80..88), scale cols (88..96), weight offset (96..104), weight length
//! (104..112), scale offset (112..120), scale length (120..128), object length
//! (128..136).
//!
//! The content address is SHA-256 of the whole artifact. Because the digest
//! covers the header, the payload and the padding, it binds the weight bytes
//! themselves, not only an index of them (sc#338).
//!
//! Index: magic `AIENXIDX` (0..8), version (8..12), entry length 84 (12..16),
//! entry count (16..24), the entries in strictly ascending key order (so a
//! duplicate key is a format error), then SHA-256 of everything before it.

use crate::error::StoreError;
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

pub const OBJECT_MAGIC: [u8; 8] = *b"AIENXPRT";
pub const INDEX_MAGIC: [u8; 8] = *b"AIENXIDX";
pub const FORMAT_VERSION: u32 = 1;
/// Size of the artifact header region; the weight bytes start here.
pub const HEADER_REGION: usize = 4096;
/// Alignment of every payload offset and of the artifact length (direct I/O).
pub const ALIGN: u64 = 4096;
/// FP8 quantization block edge (rows and columns).
pub const FP8_BLOCK: u64 = 128;
/// Bytes of one index entry.
pub const INDEX_ENTRY_LEN: usize = 84;
const INDEX_PREFIX_LEN: usize = 24;
const HEADER_FIELDS_END: usize = 136;
const WEIGHT_DTYPE_F8_E4M3: u8 = 1;

/// Which projection of an expert's MLP.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum Projection {
    Gate = 1,
    Up = 2,
    Down = 3,
}

impl Projection {
    pub const ALL: [Projection; 3] = [Projection::Gate, Projection::Up, Projection::Down];

    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Gate),
            2 => Some(Self::Up),
            3 => Some(Self::Down),
            _ => None,
        }
    }

    /// The checkpoint tensor name component (`gate_proj`, `up_proj`, `down_proj`).
    pub fn checkpoint_name(self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }
}

/// Quantization format of the weight bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum QuantFormat {
    /// FP8 E4M3 weights with one scale per 128 x 128 block.
    Fp8E4m3Block128 = 1,
}

impl QuantFormat {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Fp8E4m3Block128),
            _ => None,
        }
    }
}

/// Element type of the stored scale bytes (kept exactly as the checkpoint has them).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ScaleDtype {
    Bf16 = 1,
    F32 = 2,
}

impl ScaleDtype {
    pub fn from_u8(v: u8) -> Option<Self> {
        match v {
            1 => Some(Self::Bf16),
            2 => Some(Self::F32),
            _ => None,
        }
    }

    pub fn from_safetensors(s: &str) -> Option<Self> {
        match s {
            "BF16" => Some(Self::Bf16),
            "F32" => Some(Self::F32),
            _ => None,
        }
    }

    pub fn safetensors_name(self) -> &'static str {
        match self {
            Self::Bf16 => "BF16",
            Self::F32 => "F32",
        }
    }

    pub fn size(self) -> u64 {
        match self {
            Self::Bf16 => 2,
            Self::F32 => 4,
        }
    }
}

/// The identity of one stored tensor pair. Ordered field by field, in
/// declaration order (the index order).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertKey {
    pub model_digest: [u8; 32],
    pub layer: u32,
    pub expert: u32,
    pub projection: Projection,
    pub quant: QuantFormat,
}

/// One index record: the key, the content address and the artifact length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct IndexEntry {
    pub key: ExpertKey,
    pub data_digest: [u8; 32],
    pub object_len: u64,
}

/// The decoded artifact header. Every field after `scale_dtype` is derived
/// from `(rows, cols, scale_dtype)`; `decode` recomputes and compares them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ArtifactHeader {
    pub key: ExpertKey,
    pub rows: u64,
    pub cols: u64,
    pub scale_dtype: ScaleDtype,
    pub scale_rows: u64,
    pub scale_cols: u64,
    pub weight_offset: u64,
    pub weight_len: u64,
    pub scale_offset: u64,
    pub scale_len: u64,
    pub object_len: u64,
}

fn round_up(v: u64, align: u64) -> Option<u64> {
    v.checked_add(align - 1).map(|x| x / align * align)
}

fn rd_u32(b: &[u8], at: usize) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[at..at + 4]);
    u32::from_le_bytes(a)
}

fn rd_u64(b: &[u8], at: usize) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[at..at + 8]);
    u64::from_le_bytes(a)
}

impl ArtifactHeader {
    /// The canonical layout for a `[rows][cols]` FP8 weight with block-128 scales.
    pub fn new(
        key: ExpertKey,
        rows: u64,
        cols: u64,
        scale_dtype: ScaleDtype,
    ) -> Result<Self, StoreError> {
        if rows == 0 || cols == 0 {
            return Err(StoreError::BadHeader("zero rows or cols"));
        }
        let ov = StoreError::Overflow;
        let weight_len = rows.checked_mul(cols).ok_or(ov.clone())?;
        let scale_rows = rows.div_ceil(FP8_BLOCK);
        let scale_cols = cols.div_ceil(FP8_BLOCK);
        let scale_len = scale_rows
            .checked_mul(scale_cols)
            .and_then(|n| n.checked_mul(scale_dtype.size()))
            .ok_or(ov.clone())?;
        let weight_offset = HEADER_REGION as u64;
        let scale_offset = weight_offset
            .checked_add(weight_len)
            .and_then(|e| round_up(e, ALIGN))
            .ok_or(ov.clone())?;
        let object_len = scale_offset
            .checked_add(scale_len)
            .and_then(|e| round_up(e, ALIGN))
            .ok_or(ov)?;
        Ok(Self {
            key,
            rows,
            cols,
            scale_dtype,
            scale_rows,
            scale_cols,
            weight_offset,
            weight_len,
            scale_offset,
            scale_len,
            object_len,
        })
    }

    /// The 4096-byte canonical header region.
    pub fn encode(&self) -> Vec<u8> {
        let mut h = vec![0u8; HEADER_REGION];
        h[0..8].copy_from_slice(&OBJECT_MAGIC);
        h[8..12].copy_from_slice(&FORMAT_VERSION.to_le_bytes());
        h[12..16].copy_from_slice(&(HEADER_REGION as u32).to_le_bytes());
        h[16..48].copy_from_slice(&self.key.model_digest);
        h[48..52].copy_from_slice(&self.key.layer.to_le_bytes());
        h[52..56].copy_from_slice(&self.key.expert.to_le_bytes());
        h[56] = self.key.projection as u8;
        h[57] = self.key.quant as u8;
        h[58] = WEIGHT_DTYPE_F8_E4M3;
        h[59] = self.scale_dtype as u8;
        h[60..64].copy_from_slice(&(FP8_BLOCK as u32).to_le_bytes());
        let fields = [
            self.rows,
            self.cols,
            self.scale_rows,
            self.scale_cols,
            self.weight_offset,
            self.weight_len,
            self.scale_offset,
            self.scale_len,
            self.object_len,
        ];
        for (i, v) in fields.iter().enumerate() {
            let at = 64 + i * 8;
            h[at..at + 8].copy_from_slice(&v.to_le_bytes());
        }
        h
    }

    /// Parses and fully re-derives a header region; any disagreement is refused.
    pub fn decode(region: &[u8]) -> Result<Self, StoreError> {
        if region.len() < HEADER_REGION {
            return Err(StoreError::BadHeader("short header region"));
        }
        let h = &region[..HEADER_REGION];
        if h[0..8] != OBJECT_MAGIC {
            return Err(StoreError::BadMagic);
        }
        if rd_u32(h, 8) != FORMAT_VERSION {
            return Err(StoreError::UnsupportedVersion);
        }
        if rd_u32(h, 12) != HEADER_REGION as u32 {
            return Err(StoreError::BadHeader("header_len"));
        }
        let projection = Projection::from_u8(h[56]).ok_or(StoreError::BadHeader("projection"))?;
        let quant = QuantFormat::from_u8(h[57]).ok_or(StoreError::BadHeader("quant"))?;
        if h[58] != WEIGHT_DTYPE_F8_E4M3 {
            return Err(StoreError::BadHeader("weight dtype"));
        }
        let scale_dtype = ScaleDtype::from_u8(h[59]).ok_or(StoreError::BadHeader("scale dtype"))?;
        if u64::from(rd_u32(h, 60)) != FP8_BLOCK {
            return Err(StoreError::BadHeader("block"));
        }
        if h[HEADER_FIELDS_END..].iter().any(|&b| b != 0) {
            return Err(StoreError::BadHeader("nonzero reserved bytes"));
        }
        let mut model_digest = [0u8; 32];
        model_digest.copy_from_slice(&h[16..48]);
        let key = ExpertKey {
            model_digest,
            layer: rd_u32(h, 48),
            expert: rd_u32(h, 52),
            projection,
            quant,
        };
        let want = Self::new(key, rd_u64(h, 64), rd_u64(h, 72), scale_dtype)?;
        if want.encode() != h {
            return Err(StoreError::BadHeader("derived fields disagree"));
        }
        Ok(want)
    }

    /// The weight byte range inside the artifact.
    pub fn weight_range(&self) -> core::ops::Range<usize> {
        self.weight_offset as usize..(self.weight_offset + self.weight_len) as usize
    }

    /// The scale byte range inside the artifact.
    pub fn scale_range(&self) -> core::ops::Range<usize> {
        self.scale_offset as usize..(self.scale_offset + self.scale_len) as usize
    }
}

/// SHA-256 of `bytes`.
pub fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

/// Constant-time equality of two digests.
pub fn digest_eq(a: &[u8; 32], b: &[u8; 32]) -> bool {
    a.ct_eq(b).into()
}

/// Builds the canonical artifact bytes for one projection. `weights` and
/// `scales` are copied verbatim (checkpoint byte order is preserved).
pub fn build_artifact(
    key: ExpertKey,
    rows: u64,
    cols: u64,
    scale_dtype: ScaleDtype,
    weights: &[u8],
    scales: &[u8],
    max_object_bytes: u64,
) -> Result<(ArtifactHeader, Vec<u8>), StoreError> {
    let hdr = ArtifactHeader::new(key, rows, cols, scale_dtype)?;
    if weights.len() as u64 != hdr.weight_len {
        return Err(StoreError::PayloadLength {
            expected: hdr.weight_len,
            found: weights.len() as u64,
        });
    }
    if scales.len() as u64 != hdr.scale_len {
        return Err(StoreError::PayloadLength {
            expected: hdr.scale_len,
            found: scales.len() as u64,
        });
    }
    if hdr.object_len > max_object_bytes {
        return Err(StoreError::ObjectTooLarge {
            len: hdr.object_len,
            max: max_object_bytes,
        });
    }
    let len = usize::try_from(hdr.object_len).map_err(|_| StoreError::Overflow)?;
    let mut bytes = vec![0u8; len];
    bytes[..HEADER_REGION].copy_from_slice(&hdr.encode());
    bytes[hdr.weight_range()].copy_from_slice(weights);
    bytes[hdr.scale_range()].copy_from_slice(scales);
    Ok((hdr, bytes))
}

/// Encodes an index. Entries must be in strictly ascending key order.
pub fn encode_index(entries: &[IndexEntry]) -> Result<Vec<u8>, StoreError> {
    if entries.windows(2).any(|w| w[0].key >= w[1].key) {
        return Err(StoreError::BadIndex("entries not strictly ascending"));
    }
    let mut out = Vec::with_capacity(INDEX_PREFIX_LEN + entries.len() * INDEX_ENTRY_LEN + 32);
    out.extend_from_slice(&INDEX_MAGIC);
    out.extend_from_slice(&FORMAT_VERSION.to_le_bytes());
    out.extend_from_slice(&(INDEX_ENTRY_LEN as u32).to_le_bytes());
    out.extend_from_slice(&(entries.len() as u64).to_le_bytes());
    for e in entries {
        out.extend_from_slice(&e.key.model_digest);
        out.extend_from_slice(&e.key.layer.to_le_bytes());
        out.extend_from_slice(&e.key.expert.to_le_bytes());
        out.push(e.key.projection as u8);
        out.push(e.key.quant as u8);
        out.extend_from_slice(&[0, 0]);
        out.extend_from_slice(&e.data_digest);
        out.extend_from_slice(&e.object_len.to_le_bytes());
    }
    let trailer = sha256(&out);
    out.extend_from_slice(&trailer);
    Ok(out)
}

/// The largest index file `decode_index` accepts for `max_entries`.
pub fn max_index_bytes(max_entries: usize) -> u64 {
    (INDEX_PREFIX_LEN + 32) as u64 + max_entries as u64 * INDEX_ENTRY_LEN as u64
}

/// Decodes and validates an index: size, trailer (constant time), version,
/// strict key order (no duplicates), reserved bytes, enums and object lengths.
pub fn decode_index(bytes: &[u8], max_entries: usize) -> Result<Vec<IndexEntry>, StoreError> {
    if bytes.len() < INDEX_PREFIX_LEN + 32 {
        return Err(StoreError::BadIndex("short index"));
    }
    if bytes[0..8] != INDEX_MAGIC {
        return Err(StoreError::BadMagic);
    }
    let (body, trailer) = bytes.split_at(bytes.len() - 32);
    let mut t = [0u8; 32];
    t.copy_from_slice(trailer);
    if !digest_eq(&sha256(body), &t) {
        return Err(StoreError::IndexChecksum);
    }
    if rd_u32(body, 8) != FORMAT_VERSION {
        return Err(StoreError::UnsupportedVersion);
    }
    if rd_u32(body, 12) as usize != INDEX_ENTRY_LEN {
        return Err(StoreError::BadIndex("entry length"));
    }
    let count = rd_u64(body, 16);
    if count > max_entries as u64 {
        return Err(StoreError::TooManyEntries);
    }
    let records = &body[INDEX_PREFIX_LEN..];
    if records.len() as u64 != count * INDEX_ENTRY_LEN as u64 {
        return Err(StoreError::BadIndex("size disagrees with count"));
    }
    let mut out: Vec<IndexEntry> = Vec::with_capacity(count as usize);
    let (chunks, rest) = records.as_chunks::<INDEX_ENTRY_LEN>();
    debug_assert!(rest.is_empty());
    for r in chunks {
        let mut model_digest = [0u8; 32];
        model_digest.copy_from_slice(&r[0..32]);
        let projection = Projection::from_u8(r[40]).ok_or(StoreError::BadIndex("projection"))?;
        let quant = QuantFormat::from_u8(r[41]).ok_or(StoreError::BadIndex("quant"))?;
        if r[42] != 0 || r[43] != 0 {
            return Err(StoreError::BadIndex("nonzero reserved bytes"));
        }
        let mut data_digest = [0u8; 32];
        data_digest.copy_from_slice(&r[44..76]);
        let object_len = rd_u64(r, 76);
        if object_len <= HEADER_REGION as u64 || !object_len.is_multiple_of(ALIGN) {
            return Err(StoreError::BadIndex("object length"));
        }
        let e = IndexEntry {
            key: ExpertKey {
                model_digest,
                layer: rd_u32(r, 32),
                expert: rd_u32(r, 36),
                projection,
                quant,
            },
            data_digest,
            object_len,
        };
        if let Some(prev) = out.last() {
            if prev.key >= e.key {
                return Err(StoreError::BadIndex("entries not strictly ascending"));
            }
        }
        out.push(e);
    }
    Ok(out)
}

/// Lowercase hex of a digest (the object file name).
pub fn hex(d: &[u8; 32]) -> String {
    const H: &[u8; 16] = b"0123456789abcdef";
    let mut s = String::with_capacity(64);
    for b in d {
        s.push(H[(b >> 4) as usize] as char);
        s.push(H[(b & 15) as usize] as char);
    }
    s
}
