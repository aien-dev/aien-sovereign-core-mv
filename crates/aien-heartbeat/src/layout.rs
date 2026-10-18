//! Qwen3-MoE FP8 expert layout (Expert Ribbon P1, sc#629): which bytes of a verified expert payload are the
//! gate|up and down projections and their block-128 scales, checked against the model shape before any byte
//! is handed to a GPU slot.
//!
//! A payload is the expert's six tensors in file order (manifest [`crate::manifest::TensorRange`] offsets).
//! The chip wants, per expert, exactly what the resident loader uploads (`qwen3_moe_omega.rs`):
//! * gate|up weights `[2i][h]` e4m3 = `gate_proj.weight [i][h]` then `up_proj.weight [i][h]`, scales
//!   `[2i/128][h/128]` bf16 = gate scales then up scales;
//! * down weights `[h][i]` e4m3, scales `[h/128][i/128]` bf16.
//!
//! Gate and up are adjacent in the checkpoint (safetensors orders tensors by name within a dtype), so the
//! gate|up weights are normally one borrowed slice; when they are not adjacent they are copied, never assumed.

use std::borrow::Cow;

use crate::error::HeartbeatError;
use crate::manifest::{ExpertEntry, TensorRange};

/// Quantisation format name recorded with every expert identity.
pub const FORMAT_FP8_BLOCK128: &str = "qwen3-fp8-e4m3/bf16-scale-inv/block128";
pub const FP8_BLOCK: u64 = 128;

/// Model dimensions the layout is checked against (config.json `hidden_size`, `moe_intermediate_size`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExpertShape {
    pub hidden: u64,
    pub inter: u64,
}

impl ExpertShape {
    pub fn gate_up_weight_bytes(&self) -> usize {
        (2 * self.inter * self.hidden) as usize
    }
    pub fn down_weight_bytes(&self) -> usize {
        (self.hidden * self.inter) as usize
    }
    pub fn gate_up_scales(&self) -> usize {
        (2 * (self.inter / FP8_BLOCK) * (self.hidden / FP8_BLOCK)) as usize
    }
    pub fn down_scales(&self) -> usize {
        ((self.hidden / FP8_BLOCK) * (self.inter / FP8_BLOCK)) as usize
    }
}

/// Payload ranges `(offset, len)` of the six tensors of one expert, validated.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExpertLayout {
    pub gate_w: (u64, u64),
    pub up_w: (u64, u64),
    pub down_w: (u64, u64),
    pub gate_s: (u64, u64),
    pub up_s: (u64, u64),
    pub down_s: (u64, u64),
}

fn find<'a>(e: &'a ExpertEntry, suffix: &str) -> Result<&'a TensorRange, HeartbeatError> {
    let want = format!("model.layers.{}.mlp.experts.{}.{suffix}", e.layer, e.expert);
    let mut hits = e.tensors.iter().filter(|t| t.name == want);
    let t = hits.next().ok_or_else(|| {
        HeartbeatError::Manifest(format!("expert ({},{}): missing {want}", e.layer, e.expert))
    })?;
    if hits.next().is_some() {
        return Err(HeartbeatError::Manifest(format!(
            "expert ({},{}): duplicate {want}",
            e.layer, e.expert
        )));
    }
    Ok(t)
}

fn check(
    t: &TensorRange,
    dtype: &str,
    shape: [u64; 2],
    elem: u64,
) -> Result<(u64, u64), HeartbeatError> {
    if t.dtype != dtype || t.shape != shape || t.len != shape[0] * shape[1] * elem {
        return Err(HeartbeatError::Manifest(format!(
            "{}: dtype {} shape {:?} len {} (want {dtype} {shape:?} len {})",
            t.name,
            t.dtype,
            t.shape,
            t.len,
            shape[0] * shape[1] * elem
        )));
    }
    Ok((t.offset, t.len))
}

impl ExpertLayout {
    /// Validates the six tensors of `e` (names, dtypes, shapes, lengths; no extra tensors) for `shape`.
    pub fn for_entry(e: &ExpertEntry, shape: ExpertShape) -> Result<Self, HeartbeatError> {
        let (h, i) = (shape.hidden, shape.inter);
        if h % FP8_BLOCK != 0 || i % FP8_BLOCK != 0 {
            return Err(HeartbeatError::Manifest(format!(
                "shape {h}x{i} is not block-128"
            )));
        }
        if e.tensors.len() != 6 {
            return Err(HeartbeatError::Manifest(format!(
                "expert ({},{}): {} tensors, want 6",
                e.layer,
                e.expert,
                e.tensors.len()
            )));
        }
        let (hb, ib) = (h / FP8_BLOCK, i / FP8_BLOCK);
        let l = Self {
            gate_w: check(find(e, "gate_proj.weight")?, "F8_E4M3", [i, h], 1)?,
            up_w: check(find(e, "up_proj.weight")?, "F8_E4M3", [i, h], 1)?,
            down_w: check(find(e, "down_proj.weight")?, "F8_E4M3", [h, i], 1)?,
            gate_s: check(find(e, "gate_proj.weight_scale_inv")?, "BF16", [ib, hb], 2)?,
            up_s: check(find(e, "up_proj.weight_scale_inv")?, "BF16", [ib, hb], 2)?,
            down_s: check(find(e, "down_proj.weight_scale_inv")?, "BF16", [hb, ib], 2)?,
        };
        let end = e.len;
        for (o, n) in [l.gate_w, l.up_w, l.down_w, l.gate_s, l.up_s, l.down_s] {
            if o + n > end {
                return Err(HeartbeatError::Manifest(format!(
                    "expert ({},{}): tensor past payload end",
                    e.layer, e.expert
                )));
            }
        }
        Ok(l)
    }
}

/// An expert payload as its segments in order (a ring lease, or a read adapter result).
pub struct Payload<'a> {
    segs: Vec<&'a [u8]>,
}

impl<'a> Payload<'a> {
    pub fn new(segs: Vec<&'a [u8]>) -> Self {
        Self { segs }
    }

    pub fn len(&self) -> usize {
        self.segs.iter().map(|s| s.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// `len` bytes at payload offset `off`, borrowed when they lie in one segment.
    pub fn bytes(&self, off: u64, len: u64) -> Result<Cow<'a, [u8]>, HeartbeatError> {
        let (off, len) = (off as usize, len as usize);
        if off.checked_add(len).is_none_or(|end| end > self.len()) {
            return Err(HeartbeatError::Manifest(format!(
                "payload range {off}+{len} past the {}-byte payload",
                self.len()
            )));
        }
        let mut base = 0usize;
        for s in &self.segs {
            if off >= base && off + len <= base + s.len() {
                return Ok(Cow::Borrowed(&s[off - base..off - base + len]));
            }
            base += s.len();
        }
        // Straddles a segment boundary (never for whole tensors, but correct anyway): copy.
        let flat: Vec<u8> = self.segs.iter().flat_map(|s| s.iter().copied()).collect();
        Ok(Cow::Owned(flat[off..off + len].to_vec()))
    }
}

/// What a GPU slot pair is loaded from: gate|up and down weights (e4m3) and scales (bf16 bits).
#[derive(Debug)]
pub struct ExpertParts<'a> {
    pub gate_up_w: Cow<'a, [u8]>,
    pub gate_up_s: Vec<u16>,
    pub down_w: Cow<'a, [u8]>,
    pub down_s: Vec<u16>,
}

fn bf16_words(b: &[u8]) -> Vec<u16> {
    b.as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect()
}

impl ExpertLayout {
    /// Splits a verified payload into the chip layout.
    pub fn parts<'a>(&self, p: &Payload<'a>) -> Result<ExpertParts<'a>, HeartbeatError> {
        let gate_up_w = if self.up_w.0 == self.gate_w.0 + self.gate_w.1 {
            p.bytes(self.gate_w.0, self.gate_w.1 + self.up_w.1)?
        } else {
            let mut v = p.bytes(self.gate_w.0, self.gate_w.1)?.into_owned();
            v.extend_from_slice(&p.bytes(self.up_w.0, self.up_w.1)?);
            Cow::Owned(v)
        };
        let mut gate_up_s = bf16_words(&p.bytes(self.gate_s.0, self.gate_s.1)?);
        gate_up_s.extend(bf16_words(&p.bytes(self.up_s.0, self.up_s.1)?));
        Ok(ExpertParts {
            gate_up_w,
            gate_up_s,
            down_w: p.bytes(self.down_w.0, self.down_w.1)?,
            down_s: bf16_words(&p.bytes(self.down_s.0, self.down_s.1)?),
        })
    }
}
