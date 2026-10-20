//! Qwen3-Coder-30B-A3B routed expert layer on GB10.
//!
//! Three pieces share one tensor contract:
//! - sharded FP8 checkpoint ingest into a [`ModelCapsule`] (one layer, no FP32 copy),
//! - the batch routing plan contract shared with the Omega GPU path,
//! - an FP32 CPU oracle that dequantizes only the experts a token selected.
//!
//! Capsule tensors, all row-major:
//! `router` BF16 [128, H]; `gate_up` FP8 E4M3 [128, 2I, H] (gate rows then up rows);
//! `gate_up_scale_inv` BF16 [128, 2I/128, H/128]; `down` FP8 E4M3 [128, H, I];
//! `down_scale_inv` BF16 [128, H/128, I/128].

use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use crate::capsule::{
    capsule_digest, row_major_strides, source_digest, CapsuleTensor, ModelCapsule,
    QuantizationDescriptor, TensorLayout, WeightDType,
};
use crate::moe_plan::{MoeBatchPlan, MoePlanError, QWEN3_CODER_A3B_EXPERTS};

pub const QWEN3_A3B_HIDDEN: usize = 2048;
pub const QWEN3_A3B_INTERMEDIATE: usize = 768;
pub const QWEN3_A3B_LAYERS: usize = 48;
const BLOCK: usize = 128;
pub(crate) const FP8_BLOCK: usize = BLOCK;
const EXPERTS: usize = QWEN3_CODER_A3B_EXPERTS;
/// Largest safetensors JSON header accepted; real Qwen shards use about 2 MB.
const MAX_HEADER_BYTES: u64 = 256 << 20;

pub const ROUTER: &str = "router";
pub const GATE_UP: &str = "gate_up";
pub const GATE_UP_SCALE_INV: &str = "gate_up_scale_inv";
pub const DOWN: &str = "down";
pub const DOWN_SCALE_INV: &str = "down_scale_inv";

#[derive(Debug, Clone, PartialEq)]
pub enum Qwen3MoeError {
    Io(String),
    Index(String),
    Contract(String),
    Plan(MoePlanError),
}

impl fmt::Display for Qwen3MoeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(detail) => write!(f, "checkpoint I/O: {detail}"),
            Self::Index(detail) => write!(f, "checkpoint index: {detail}"),
            Self::Contract(detail) => write!(f, "tensor contract: {detail}"),
            Self::Plan(error) => write!(f, "routing plan: {error}"),
        }
    }
}

impl std::error::Error for Qwen3MoeError {}

impl From<MoePlanError> for Qwen3MoeError {
    fn from(error: MoePlanError) -> Self {
        Self::Plan(error)
    }
}

/// One routed expert layer held in a capsule. Accessors borrow typed views of it.
#[derive(Debug, Clone)]
pub struct Qwen3MoeLayer {
    pub hidden: usize,
    pub intermediate: usize,
    pub capsule: ModelCapsule,
}

fn fp8_block_quant() -> Option<QuantizationDescriptor> {
    Some(QuantizationDescriptor {
        scheme: "fp8-e4m3fn-block128x128-bf16-scale-inv".to_string(),
        axis: 0,
        group_size: BLOCK,
        scales: Vec::new(),
        zero_points: Vec::new(),
        packing: "none".to_string(),
    })
}

/// (name, dtype, shape, element bytes) for every capsule tensor, in payload order.
fn layer_tensor_specs(
    hidden: usize,
    intermediate: usize,
) -> [(&'static str, WeightDType, Vec<usize>, usize); 5] {
    let (h, i) = (hidden, intermediate);
    [
        (ROUTER, WeightDType::Bf16, vec![EXPERTS, h], 2),
        (GATE_UP, WeightDType::Fp8E4M3Fn, vec![EXPERTS, 2 * i, h], 1),
        (
            GATE_UP_SCALE_INV,
            WeightDType::Bf16,
            vec![EXPERTS, 2 * i / BLOCK, h / BLOCK],
            2,
        ),
        (DOWN, WeightDType::Fp8E4M3Fn, vec![EXPERTS, h, i], 1),
        (
            DOWN_SCALE_INV,
            WeightDType::Bf16,
            vec![EXPERTS, h / BLOCK, i / BLOCK],
            2,
        ),
    ]
}

fn as_u16(bytes: &[u8]) -> &[u16] {
    // SAFETY: u16 has no invalid bit patterns; alignment is checked below.
    let (prefix, values, suffix) = unsafe { bytes.align_to::<u16>() };
    assert!(
        prefix.is_empty() && suffix.is_empty(),
        "BF16 capsule view misaligned"
    );
    values
}

impl Qwen3MoeLayer {
    /// Packs raw tensors into a capsule. BF16 tensors are raw bits; FP8 tensors are raw E4M3 bytes.
    pub fn from_parts(
        hidden: usize,
        intermediate: usize,
        router: &[u16],
        gate_up: &[u8],
        gate_up_scale_inv: &[u16],
        down: &[u8],
        down_scale_inv: &[u16],
    ) -> Result<Self, Qwen3MoeError> {
        Self::build(hidden, intermediate, |name, dst| {
            let src: &[u8] = match name {
                ROUTER => bytes_of(router),
                GATE_UP => gate_up,
                GATE_UP_SCALE_INV => bytes_of(gate_up_scale_inv),
                DOWN => down,
                _ => bytes_of(down_scale_inv),
            };
            if src.len() != dst.len() {
                return Err(Qwen3MoeError::Contract(format!(
                    "{name}: {} bytes supplied, {} expected",
                    src.len(),
                    dst.len()
                )));
            }
            dst.copy_from_slice(src);
            Ok(())
        })
    }

    /// Allocates the capsule payload once and lets `fill` write each tensor in place.
    fn build(
        hidden: usize,
        intermediate: usize,
        mut fill: impl FnMut(&str, &mut [u8]) -> Result<(), Qwen3MoeError>,
    ) -> Result<Self, Qwen3MoeError> {
        if hidden == 0
            || intermediate == 0
            || !hidden.is_multiple_of(BLOCK)
            || !intermediate.is_multiple_of(BLOCK)
        {
            return Err(Qwen3MoeError::Contract(format!(
                "hidden {hidden} and intermediate {intermediate} must be positive multiples of {BLOCK}"
            )));
        }
        let specs = layer_tensor_specs(hidden, intermediate);
        let mut ranges = Vec::with_capacity(specs.len());
        let mut cursor = 0usize;
        for (_, _, shape, element_bytes) in &specs {
            // 256-byte alignment keeps every view aligned for vector loads.
            let start = cursor.div_ceil(256) * 256;
            let end = start + shape.iter().product::<usize>() * element_bytes;
            ranges.push(start..end);
            cursor = end;
        }
        let mut payload = vec![0u8; cursor];
        for ((name, ..), range) in specs.iter().zip(&ranges) {
            fill(name, &mut payload[range.clone()])?;
        }
        let bytes: Arc<[u8]> = Arc::from(payload);
        let mut tensors = HashMap::new();
        for ((name, dtype, shape, _), range) in specs.into_iter().zip(ranges) {
            let quant = if dtype == WeightDType::Fp8E4M3Fn {
                fp8_block_quant()
            } else {
                None
            };
            tensors.insert(
                name.to_string(),
                CapsuleTensor {
                    dtype,
                    layout: TensorLayout::RowMajor,
                    logical_strides: row_major_strides(&shape),
                    shape,
                    byte_range: range,
                    quant,
                },
            );
        }
        Ok(Self {
            hidden,
            intermediate,
            capsule: ModelCapsule {
                source_digest: source_digest(&bytes),
                capsule_digest: capsule_digest(&bytes, &tensors),
                bytes,
                tensors,
            },
        })
    }

    fn view(&self, name: &str) -> &[u8] {
        self.capsule
            .tensor_bytes(name)
            .expect("Qwen3MoeLayer capsule always holds its five tensors")
    }

    pub fn router_bf16(&self) -> &[u16] {
        as_u16(self.view(ROUTER))
    }
    pub fn gate_up_fp8(&self) -> &[u8] {
        self.view(GATE_UP)
    }
    pub fn gate_up_scale_inv_bf16(&self) -> &[u16] {
        as_u16(self.view(GATE_UP_SCALE_INV))
    }
    pub fn down_fp8(&self) -> &[u8] {
        self.view(DOWN)
    }
    pub fn down_scale_inv_bf16(&self) -> &[u16] {
        as_u16(self.view(DOWN_SCALE_INV))
    }

    /// Bytes resident on the device for this layer's weights.
    pub fn resident_weight_bytes(&self) -> usize {
        [ROUTER, GATE_UP, GATE_UP_SCALE_INV, DOWN, DOWN_SCALE_INV]
            .iter()
            .map(|name| self.view(name).len())
            .sum()
    }
}

fn bytes_of(values: &[u16]) -> &[u8] {
    // SAFETY: any u16 slice is a valid byte slice of twice the length.
    unsafe { std::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * 2) }
}

/// Resolves a shard filename from the index. Only a single plain path component
/// is accepted, so an index entry cannot point outside `checkpoint_dir`.
pub(crate) fn shard_path(checkpoint_dir: &Path, filename: &str) -> Result<PathBuf, Qwen3MoeError> {
    let mut components = Path::new(filename).components();
    match (components.next(), components.next()) {
        (Some(Component::Normal(name)), None) if name == filename => Ok(checkpoint_dir.join(name)),
        _ => Err(Qwen3MoeError::Index(format!(
            "shard filename {filename:?} is not a plain file name inside the checkpoint directory"
        ))),
    }
}

pub(crate) struct Shard {
    file: File,
    payload_start: u64,
    file_len: u64,
    header: serde_json::Map<String, serde_json::Value>,
}

impl Shard {
    pub(crate) fn open(path: &Path) -> Result<Self, Qwen3MoeError> {
        let io = |e: std::io::Error| Qwen3MoeError::Io(format!("{}: {e}", path.display()));
        let file = File::open(path).map_err(io)?;
        let file_len = file.metadata().map_err(io)?.len();
        let mut prefix = [0u8; 8];
        file.read_exact_at(&mut prefix, 0).map_err(io)?;
        let header_len = u64::from_le_bytes(prefix);
        if header_len > MAX_HEADER_BYTES || 8 + header_len > file_len {
            return Err(Qwen3MoeError::Contract(format!(
                "{}: header length {header_len} is invalid for a {file_len}-byte file",
                path.display()
            )));
        }
        let mut header = vec![0u8; header_len as usize];
        file.read_exact_at(&mut header, 8).map_err(io)?;
        let header: serde_json::Value = serde_json::from_slice(&header).map_err(|e| {
            Qwen3MoeError::Contract(format!("{}: header JSON: {e}", path.display()))
        })?;
        let header = match header {
            serde_json::Value::Object(map) => map,
            _ => {
                return Err(Qwen3MoeError::Contract(format!(
                    "{}: header is not a JSON object",
                    path.display()
                )))
            }
        };
        Ok(Self {
            file,
            payload_start: 8 + header_len,
            file_len,
            header,
        })
    }

    /// The header's dtype of `name`, `None` when this shard does not hold it.
    pub(crate) fn dtype(&self, name: &str) -> Option<&str> {
        self.header.get(name)?.get("dtype")?.as_str()
    }

    /// Reads `dst.len()` bytes of tensor `name` starting `offset` bytes into its data, after checking dtype and shape and that the
    /// range lies inside the tensor (one row of a table too big to read whole).
    pub(crate) fn read_range(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        offset: u64,
        dst: &mut [u8],
    ) -> Result<(), Qwen3MoeError> {
        let contract = |detail: String| Qwen3MoeError::Contract(format!("{name}: {detail}"));
        let entry = self
            .header
            .get(name)
            .ok_or_else(|| contract("missing from shard header".to_string()))?;
        if entry.get("dtype").and_then(|v| v.as_str()) != Some(dtype) {
            return Err(contract(format!("dtype {:?}, expected {dtype}", entry.get("dtype"))));
        }
        let actual_shape: Option<Vec<usize>> = entry
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|dims| dims.iter().filter_map(|d| d.as_u64().map(|d| d as usize)).collect());
        if actual_shape.as_deref() != Some(shape) {
            return Err(contract(format!("shape {actual_shape:?}, expected {shape:?}")));
        }
        let offsets: Vec<u64> = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|o| o.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        let [start, end] = offsets[..] else {
            return Err(contract("data_offsets must hold two integers".to_string()));
        };
        let need = offset
            .checked_add(dst.len() as u64)
            .ok_or_else(|| contract("range overflows".to_string()))?;
        if start > end || need > end - start || self.payload_start + end > self.file_len {
            return Err(contract(format!(
                "range {offset}+{} is outside the {} byte tensor",
                dst.len(),
                end.saturating_sub(start)
            )));
        }
        self.file
            .read_exact_at(dst, self.payload_start + start + offset)
            .map_err(|e| Qwen3MoeError::Io(format!("{name}: {e}")))
    }

    /// Reads one tensor into `dst` after checking dtype, shape and byte range.
    pub(crate) fn read(
        &self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        dst: &mut [u8],
    ) -> Result<(), Qwen3MoeError> {
        let contract = |detail: String| Qwen3MoeError::Contract(format!("{name}: {detail}"));
        let entry = self
            .header
            .get(name)
            .ok_or_else(|| contract("missing from shard header".to_string()))?;
        if entry.get("dtype").and_then(|v| v.as_str()) != Some(dtype) {
            return Err(contract(format!(
                "dtype {:?}, expected {dtype}",
                entry.get("dtype")
            )));
        }
        let actual_shape: Option<Vec<usize>> =
            entry.get("shape").and_then(|v| v.as_array()).map(|dims| {
                dims.iter()
                    .filter_map(|d| d.as_u64().map(|d| d as usize))
                    .collect()
            });
        if actual_shape.as_deref() != Some(shape) {
            return Err(contract(format!(
                "shape {actual_shape:?}, expected {shape:?}"
            )));
        }
        let offsets: Vec<u64> = entry
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .map(|o| o.iter().filter_map(|v| v.as_u64()).collect())
            .unwrap_or_default();
        let [start, end] = offsets[..] else {
            return Err(contract("data_offsets must hold two integers".to_string()));
        };
        if start > end
            || end - start != dst.len() as u64
            || self.payload_start + end > self.file_len
        {
            return Err(contract(format!(
                "byte range {start}..{end} does not hold {} bytes inside the shard",
                dst.len()
            )));
        }
        self.file
            .read_exact_at(dst, self.payload_start + start)
            .map_err(|e| Qwen3MoeError::Io(format!("{name}: {e}")))
    }
}

/// Loads one official `Qwen3-Coder-30B-A3B-Instruct-FP8` expert layer into a capsule.
///
/// Only the shards holding that layer are opened, and only the layer's tensors are
/// read. Expert weights stay FP8; nothing is decoded.
pub fn load_qwen3_fp8_moe_layer(
    checkpoint_dir: &Path,
    layer: usize,
) -> Result<Qwen3MoeLayer, Qwen3MoeError> {
    load_qwen3_fp8_moe_layer_tapped(checkpoint_dir, layer, &mut |_, _| {})
}

/// As [`load_qwen3_fp8_moe_layer`], handing every tensor's raw checkpoint bytes to `tap` (name, bytes)
/// right after they are read, so a second consumer (the heartbeat manifest digests, L190) needs no
/// second read.
pub fn load_qwen3_fp8_moe_layer_tapped(
    checkpoint_dir: &Path,
    layer: usize,
    tap: &mut dyn FnMut(&str, &[u8]),
) -> Result<Qwen3MoeLayer, Qwen3MoeError> {
    load_moe_layer(
        checkpoint_dir,
        layer,
        QWEN3_A3B_HIDDEN,
        QWEN3_A3B_INTERMEDIATE,
        tap,
    )
}

fn load_moe_layer(
    checkpoint_dir: &Path,
    layer: usize,
    h: usize,
    i: usize,
    tap: &mut dyn FnMut(&str, &[u8]),
) -> Result<Qwen3MoeLayer, Qwen3MoeError> {
    if layer >= QWEN3_A3B_LAYERS {
        return Err(Qwen3MoeError::Contract(format!(
            "layer {layer} outside [0, {QWEN3_A3B_LAYERS})"
        )));
    }
    let index_path = checkpoint_dir.join("model.safetensors.index.json");
    let index: serde_json::Value = serde_json::from_slice(
        &std::fs::read(&index_path)
            .map_err(|e| Qwen3MoeError::Io(format!("{}: {e}", index_path.display())))?,
    )
    .map_err(|e| Qwen3MoeError::Index(format!("{}: {e}", index_path.display())))?;
    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .ok_or_else(|| Qwen3MoeError::Index("missing weight_map object".to_string()))?;

    let prefix = format!("model.layers.{layer}.mlp");
    let mut shards: HashMap<String, Shard> = HashMap::new();
    let mut read =
        |name: &str, dtype: &str, shape: &[usize], dst: &mut [u8]| -> Result<(), Qwen3MoeError> {
            let filename = weight_map
                .get(name)
                .and_then(|v| v.as_str())
                .ok_or_else(|| Qwen3MoeError::Index(format!("no shard for {name}")))?;
            if !shards.contains_key(filename) {
                let shard = Shard::open(&shard_path(checkpoint_dir, filename)?)?;
                shards.insert(filename.to_string(), shard);
            }
            shards[filename].read(name, dtype, shape, dst)?;
            tap(name, dst);
            Ok(())
        };

    Qwen3MoeLayer::build(h, i, |tensor, dst| match tensor {
        ROUTER => read(&format!("{prefix}.gate.weight"), "BF16", &[EXPERTS, h], dst),
        GATE_UP | GATE_UP_SCALE_INV => {
            let scales = tensor == GATE_UP_SCALE_INV;
            let per_expert = dst.len() / EXPERTS;
            for (expert, chunk) in dst.chunks_exact_mut(per_expert).enumerate() {
                let (gate, up) = chunk.split_at_mut(per_expert / 2);
                for (projection, half) in [("gate_proj", gate), ("up_proj", up)] {
                    let base = format!("{prefix}.experts.{expert}.{projection}");
                    if scales {
                        read(
                            &format!("{base}.weight_scale_inv"),
                            "BF16",
                            &[i / BLOCK, h / BLOCK],
                            half,
                        )?;
                    } else {
                        read(&format!("{base}.weight"), "F8_E4M3", &[i, h], half)?;
                    }
                }
            }
            Ok(())
        }
        _ => {
            let scales = tensor == DOWN_SCALE_INV;
            let per_expert = dst.len() / EXPERTS;
            for (expert, chunk) in dst.chunks_exact_mut(per_expert).enumerate() {
                let base = format!("{prefix}.experts.{expert}.down_proj");
                if scales {
                    read(
                        &format!("{base}.weight_scale_inv"),
                        "BF16",
                        &[h / BLOCK, i / BLOCK],
                        chunk,
                    )?;
                } else {
                    read(&format!("{base}.weight"), "F8_E4M3", &[h, i], chunk)?;
                }
            }
            Ok(())
        }
    })
}

// ---------------------------------------------------------------------------
// Numeric helpers and the FP32 CPU oracle.
// ---------------------------------------------------------------------------

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// Round-to-nearest-even FP32 to BF16. NaN stays NaN.
pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x40) as u16;
    }
    let rounding = 0x7fff + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// E4M3FN decode: bias 7, no infinities, 0x7f/0xff are NaN.
pub fn fp8_e4m3fn_to_f32(bits: u8) -> f32 {
    let sign = if bits & 0x80 != 0 { -1.0 } else { 1.0 };
    let exponent = ((bits >> 3) & 0x0f) as i32;
    let mantissa = (bits & 0x07) as f32;
    if exponent == 0x0f && bits & 0x07 == 0x07 {
        return f32::NAN;
    }
    if exponent == 0 {
        sign * mantissa * 2f32.powi(-9)
    } else {
        sign * (1.0 + mantissa / 8.0) * 2f32.powi(exponent - 7)
    }
}

pub(crate) fn fp8_table() -> [f32; 256] {
    std::array::from_fn(|bits| fp8_e4m3fn_to_f32(bits as u8))
}

/// Round-to-nearest-even FP32 to E4M3FN, saturating at +/-448. NaN maps to 0x7f.
pub fn f32_to_fp8_e4m3fn(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80u8 } else { 0 };
    let magnitude = value.abs();
    // Positive codes 0x00..=0x7e are monotonically increasing in value.
    let (mut lo, mut hi) = (0u8, 0x7eu8);
    if magnitude >= fp8_e4m3fn_to_f32(hi) {
        return sign | hi;
    }
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if fp8_e4m3fn_to_f32(mid) <= magnitude {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let (below, above) = (fp8_e4m3fn_to_f32(lo), fp8_e4m3fn_to_f32(hi));
    let code = match (magnitude - below).partial_cmp(&(above - magnitude)) {
        Some(std::cmp::Ordering::Less) => lo,
        Some(std::cmp::Ordering::Greater) => hi,
        _ => {
            if lo & 1 == 0 {
                lo
            } else {
                hi
            }
        }
    };
    sign | code
}

/// How the oracle treats activations entering each expert GEMM.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleActivations {
    /// Exact FP32 activations: measures the device's end-to-end numeric quality.
    Fp32,
    /// Quantize each row's 128-channel blocks to E4M3 exactly as the device does
    /// (scale = max|v| / 448). Isolates indexing and accumulation from rounding.
    Fp8Block128,
}

fn quantize_block128_in_place(values: &mut [f32]) {
    for block in values.chunks_mut(BLOCK) {
        let maximum = block.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let scale = if maximum > 0.0 { maximum / 448.0 } else { 1.0 };
        for v in block.iter_mut() {
            *v = fp8_e4m3fn_to_f32(f32_to_fp8_e4m3fn(*v / scale)) * scale;
        }
    }
}

/// FP32 router logits [tokens, 128] from BF16-representable inputs.
pub fn reference_router_logits(layer: &Qwen3MoeLayer, x: &[f32], tokens: usize) -> Vec<f32> {
    reference_router_logits_bf16(layer.router_bf16(), layer.hidden, x, tokens)
}

/// [`reference_router_logits`] from the router's raw BF16 words `[EXPERTS][h]` (no packed layer needed).
pub fn reference_router_logits_bf16(router_bf16: &[u16], h: usize, x: &[f32], tokens: usize) -> Vec<f32> {
    let router: Vec<f32> = router_bf16
        .iter()
        .map(|&b| bf16_to_f32(b))
        .collect();
    let mut logits = vec![0.0f32; tokens * EXPERTS];
    logits
        .par_chunks_mut(EXPERTS)
        .enumerate()
        .for_each(|(t, row)| {
            let xt = &x[t * h..(t + 1) * h];
            for (e, logit) in row.iter_mut().enumerate() {
                *logit = xt
                    .iter()
                    .zip(&router[e * h..(e + 1) * h])
                    .map(|(a, b)| a * b)
                    .sum();
            }
        });
    logits
}

/// out[r] = sum_k W[r, k] * v[k] with W an FP8 [rows, cols] matrix and 128x128 BF16 inverse scales.
pub(crate) fn dequant_matvec(
    weights: &[u8],
    scales: &[u16],
    rows: usize,
    cols: usize,
    v: &[f32],
    lut: &[f32; 256],
) -> Vec<f32> {
    let k_blocks = cols / BLOCK;
    (0..rows)
        .map(|r| {
            let row = &weights[r * cols..(r + 1) * cols];
            (0..k_blocks)
                .map(|b| {
                    let dot: f32 = row[b * BLOCK..(b + 1) * BLOCK]
                        .iter()
                        .zip(&v[b * BLOCK..(b + 1) * BLOCK])
                        .map(|(&w, &x)| lut[w as usize] * x)
                        .sum();
                    dot * bf16_to_f32(scales[(r / BLOCK) * k_blocks + b])
                })
                .sum()
        })
        .collect()
}

/// Independent FP32 layer output [tokens, H] for a routing plan.
///
/// Weights are dequantized exactly from FP8; activations are not quantized, so the
/// difference to the device measures the device's activation quantization and
/// accumulation order.
pub fn reference_moe_forward(layer: &Qwen3MoeLayer, x: &[f32], plan: &MoeBatchPlan) -> Vec<f32> {
    reference_moe_forward_with(layer, x, plan, OracleActivations::Fp32)
}

pub fn reference_moe_forward_with(
    layer: &Qwen3MoeLayer,
    x: &[f32],
    plan: &MoeBatchPlan,
    activations: OracleActivations,
) -> Vec<f32> {
    let (h, i) = (layer.hidden, layer.intermediate);
    let gate_up = layer.gate_up_fp8();
    let gate_up_scales = layer.gate_up_scale_inv_bf16();
    let down = layer.down_fp8();
    let down_scales = layer.down_scale_inv_bf16();
    let gu_scale_len = (2 * i / BLOCK) * (h / BLOCK);
    let down_scale_len = (h / BLOCK) * (i / BLOCK);
    reference_moe_forward_experts(h, i, x, plan, activations, &|e| ExpertView {
        gate_up: &gate_up[e * 2 * i * h..(e + 1) * 2 * i * h],
        gate_up_scales: &gate_up_scales[e * gu_scale_len..(e + 1) * gu_scale_len],
        down: &down[e * h * i..(e + 1) * h * i],
        down_scales: &down_scales[e * down_scale_len..(e + 1) * down_scale_len],
    })
}

/// One routed expert's bytes as the reference reads them: FP8 E4M3 weights (gate rows then up rows for
/// `gate_up`, `[2i][h]`; `down` `[h][i]`) and their BF16 block-128 inverse scales.
#[derive(Clone, Copy)]
pub struct ExpertView<'a> {
    pub gate_up: &'a [u8],
    pub gate_up_scales: &'a [u16],
    pub down: &'a [u8],
    pub down_scales: &'a [u16],
}

/// [`reference_moe_forward_with`] with each expert's bytes from `expert(e)` instead of a packed layer, so
/// a caller can hold only the routed experts (L190: the 30B drafter with its experts streamed by the
/// heartbeat ring). The arithmetic and its order are the same, so the result is bit for bit the same.
pub fn reference_moe_forward_experts<'a>(
    h: usize,
    i: usize,
    x: &[f32],
    plan: &MoeBatchPlan,
    activations: OracleActivations,
    expert: &(dyn Fn(usize) -> ExpertView<'a> + Sync),
) -> Vec<f32> {
    let quantize = activations == OracleActivations::Fp8Block128;
    let lut = fp8_table();
    let mut output = vec![0.0f32; plan.tokens * h];
    output.par_chunks_mut(h).enumerate().for_each(|(t, out)| {
        let mut xt = x[t * h..(t + 1) * h].to_vec();
        if quantize {
            quantize_block128_in_place(&mut xt);
        }
        for slot in 0..plan.top_k {
            let assignment = t * plan.top_k + slot;
            let e = plan.topk_experts[assignment] as usize;
            let weight = plan.topk_weights[assignment];
            let ev = expert(e);
            let projected = dequant_matvec(
                ev.gate_up,
                ev.gate_up_scales,
                2 * i,
                h,
                &xt,
                &lut,
            );
            let mut hidden: Vec<f32> = (0..i)
                .map(|j| {
                    let gate = projected[j];
                    gate / (1.0 + (-gate).exp()) * projected[i + j]
                })
                .collect();
            if quantize {
                quantize_block128_in_place(&mut hidden);
            }
            let expert_out = dequant_matvec(
                ev.down,
                ev.down_scales,
                h,
                i,
                &hidden,
                &lut,
            );
            for (o, v) in out.iter_mut().zip(expert_out) {
                *o += weight * v;
            }
        }
    });
    output
}

/// Agreement between a device output and the oracle.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParityStats {
    pub cosine: f64,
    pub max_abs: f32,
    /// 95th percentile of |a - e| / max(|e|, 0.05).
    pub p95_rel: f32,
    pub max_abs_expected: f32,
    pub non_finite: usize,
}

pub fn parity_stats(actual: &[f32], expected: &[f32]) -> ParityStats {
    assert_eq!(actual.len(), expected.len());
    let (mut dot, mut na, mut ne) = (0.0f64, 0.0f64, 0.0f64);
    let mut max_abs = 0.0f32;
    let mut max_abs_expected = 0.0f32;
    let mut non_finite = 0;
    let mut relative = Vec::with_capacity(actual.len());
    for (&a, &e) in actual.iter().zip(expected) {
        if !a.is_finite() {
            non_finite += 1;
            continue;
        }
        dot += a as f64 * e as f64;
        na += a as f64 * a as f64;
        ne += e as f64 * e as f64;
        let diff = (a - e).abs();
        max_abs = max_abs.max(diff);
        max_abs_expected = max_abs_expected.max(e.abs());
        relative.push(diff / e.abs().max(0.05));
    }
    relative.sort_by(f32::total_cmp);
    let p95_rel = relative
        .get((relative.len().saturating_sub(1) * 95).div_ceil(100))
        .copied()
        .unwrap_or(f32::NAN);
    ParityStats {
        cosine: dot / (na.sqrt() * ne.sqrt()),
        max_abs,
        p95_rel,
        max_abs_expected,
        non_finite,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// L190: the per-expert path (views copied out of the packed layer, as the ring hands them over) is
    /// bit for bit the packed reference; a view of the wrong expert is caught.
    #[test]
    fn experts_view_path_equals_packed_reference() {
        let (h, i) = (128usize, 128usize);
        let mut s = 0x9e37_79b9_7f4a_7c15u64;
        let mut rnd = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // finite E4M3 bytes only (0x7f / 0xff are NaN)
        let mut fp8 = |n: usize| -> Vec<u8> { (0..n).map(|_| (rnd() as u8) & 0x7e).collect() };
        let gate_up = fp8(EXPERTS * 2 * i * h);
        let down = fp8(EXPERTS * h * i);
        let mut s2 = 7u32;
        let mut words = |n: usize, base: u16| -> Vec<u16> {
            (0..n)
                .map(|_| {
                    s2 = s2.wrapping_mul(1_103_515_245).wrapping_add(12_345);
                    base + ((s2 >> 16) as u16 & 0x7f)
                })
                .collect()
        };
        let gus = words(EXPERTS * 2 * (i / BLOCK) * (h / BLOCK), 0x3b00);
        let ds = words(EXPERTS * (h / BLOCK) * (i / BLOCK), 0x3b00);
        let router = words(EXPERTS * h, 0x3c00);
        let layer = Qwen3MoeLayer::from_parts(h, i, &router, &gate_up, &gus, &down, &ds).unwrap();
        let x: Vec<f32> = (0..h).map(|j| ((j * 37 % 101) as f32 - 50.0) / 64.0).collect();
        let logits = reference_router_logits(&layer, &x, 1);
        assert_eq!(logits, reference_router_logits_bf16(&router, h, &x, 1));
        let plan = MoeBatchPlan::qwen3_coder_a3b(&logits, 1).unwrap();
        let want = reference_moe_forward(&layer, &x, &plan);
        let (gul, dl) = (2 * (i / BLOCK) * (h / BLOCK), (h / BLOCK) * (i / BLOCK));
        let owned: Vec<(Vec<u8>, Vec<u16>, Vec<u8>, Vec<u16>)> = (0..EXPERTS)
            .map(|e| {
                (
                    gate_up[e * 2 * i * h..(e + 1) * 2 * i * h].to_vec(),
                    gus[e * gul..(e + 1) * gul].to_vec(),
                    down[e * h * i..(e + 1) * h * i].to_vec(),
                    ds[e * dl..(e + 1) * dl].to_vec(),
                )
            })
            .collect();
        let view = |e: usize| ExpertView {
            gate_up: &owned[e].0,
            gate_up_scales: &owned[e].1,
            down: &owned[e].2,
            down_scales: &owned[e].3,
        };
        let got = reference_moe_forward_experts(h, i, &x, &plan, OracleActivations::Fp32, &view);
        assert_eq!(
            got.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            want.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        // negative control: hand over the neighbour expert
        let wrong = |e: usize| view((e + 1) % EXPERTS);
        let bad = reference_moe_forward_experts(h, i, &x, &plan, OracleActivations::Fp32, &wrong);
        assert_ne!(bad, want, "a view of the wrong expert must change the output");
    }

    #[test]
    fn fp8_e4m3fn_decodes_known_values() {
        assert_eq!(fp8_e4m3fn_to_f32(0x38), 1.0);
        assert_eq!(fp8_e4m3fn_to_f32(0x30), 0.5);
        assert_eq!(fp8_e4m3fn_to_f32(0xc0), -2.0);
        assert_eq!(fp8_e4m3fn_to_f32(0x7e), 448.0);
        assert_eq!(fp8_e4m3fn_to_f32(0x01), 2f32.powi(-9));
        assert!(fp8_e4m3fn_to_f32(0x7f).is_nan());
        assert!(fp8_e4m3fn_to_f32(0xff).is_nan());
    }

    #[test]
    fn fp8_encode_rounds_to_nearest_even_and_saturates() {
        for bits in 0u8..=0x7e {
            for code in [bits, bits | 0x80] {
                assert_eq!(
                    f32_to_fp8_e4m3fn(fp8_e4m3fn_to_f32(code)),
                    code,
                    "exact {code:#x}"
                );
            }
        }
        // 1.0625 lies halfway between 1.0 (0x38, even) and 1.125 (0x39).
        assert_eq!(f32_to_fp8_e4m3fn(1.0625), 0x38);
        // 1.1875 lies halfway between 1.125 (0x39) and 1.25 (0x3a, even).
        assert_eq!(f32_to_fp8_e4m3fn(1.1875), 0x3a);
        assert_eq!(f32_to_fp8_e4m3fn(1.07), 0x39);
        assert_eq!(f32_to_fp8_e4m3fn(1000.0), 0x7e);
        assert_eq!(f32_to_fp8_e4m3fn(-1000.0), 0xfe);
        assert_eq!(f32_to_fp8_e4m3fn(-0.0), 0x80);
    }

    #[test]
    fn bf16_rounds_to_nearest_even() {
        assert_eq!(f32_to_bf16(1.0), 0x3f80);
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0 + 1.0 / 256.0)), 1.0);
        assert_eq!(
            bf16_to_f32(f32_to_bf16(1.0 + 3.0 / 256.0)),
            1.0 + 4.0 / 256.0
        );
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
    }

    #[test]
    fn shard_names_cannot_escape_the_checkpoint_directory() {
        let dir = Path::new("/ckpt");
        assert_eq!(
            shard_path(dir, "model-00001-of-00004.safetensors").unwrap(),
            PathBuf::from("/ckpt/model-00001-of-00004.safetensors")
        );
        for bad in [
            "../secret",
            "/etc/passwd",
            "sub/model.safetensors",
            "..",
            ".",
            "",
            "./x",
        ] {
            assert!(shard_path(dir, bad).is_err(), "{bad:?} must be rejected");
        }
    }

    #[test]
    fn capsule_views_round_trip_and_digest_covers_payload() {
        let (h, i) = (128, 128);
        let router: Vec<u16> = (0..EXPERTS * h).map(|v| v as u16).collect();
        let gate_up = vec![0x38u8; EXPERTS * 2 * i * h];
        let gate_up_scales = vec![0x3f80u16; EXPERTS * 2];
        let down = vec![0x30u8; EXPERTS * h * i];
        let down_scales = vec![0x4000u16; EXPERTS];
        let layer = Qwen3MoeLayer::from_parts(
            h,
            i,
            &router,
            &gate_up,
            &gate_up_scales,
            &down,
            &down_scales,
        )
        .unwrap();
        assert_eq!(layer.router_bf16(), &router[..]);
        assert_eq!(layer.gate_up_fp8(), &gate_up[..]);
        assert_eq!(layer.gate_up_scale_inv_bf16(), &gate_up_scales[..]);
        assert_eq!(layer.down_fp8(), &down[..]);
        assert_eq!(layer.down_scale_inv_bf16(), &down_scales[..]);
        assert_eq!(
            layer.capsule.tensor(GATE_UP).unwrap().dtype,
            WeightDType::Fp8E4M3Fn
        );
        assert!(layer.capsule.tensor(GATE_UP).unwrap().quant.is_some());

        let mut changed_down = down.clone();
        changed_down[7] = 0x38;
        let changed = Qwen3MoeLayer::from_parts(
            h,
            i,
            &router,
            &gate_up,
            &gate_up_scales,
            &changed_down,
            &down_scales,
        )
        .unwrap();
        assert_ne!(layer.capsule.capsule_digest, changed.capsule.capsule_digest);

        assert!(Qwen3MoeLayer::from_parts(
            h,
            i,
            &router,
            &gate_up,
            &gate_up_scales,
            &down[1..],
            &down_scales
        )
        .is_err());
        assert!(Qwen3MoeLayer::from_parts(
            100,
            i,
            &router,
            &gate_up,
            &gate_up_scales,
            &down,
            &down_scales
        )
        .is_err());
    }

    #[test]
    fn loader_reads_one_layer_from_a_sharded_checkpoint_and_rejects_bad_contracts() {
        let dir = std::env::temp_dir().join(format!("aien-qwen3-loader-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let (h, i) = (256, 128);
        let prefix = "model.layers.3.mlp";
        // Two shards: router and experts 0..63 in the first, the rest in the second.
        // (name, dtype, shape, payload) per shard.
        type ShardTensors<'a> = Vec<(String, &'a str, Vec<usize>, Vec<u8>)>;
        let mut shard_tensors: [ShardTensors; 2] = [Vec::new(), Vec::new()];
        shard_tensors[0].push((
            format!("{prefix}.gate.weight"),
            "BF16",
            vec![EXPERTS, h],
            vec![0x11; EXPERTS * h * 2],
        ));
        for expert in 0..EXPERTS {
            let shard = usize::from(expert >= 64);
            let base = format!("{prefix}.experts.{expert}");
            let tag = expert as u8;
            shard_tensors[shard].push((
                format!("{base}.gate_proj.weight"),
                "F8_E4M3",
                vec![i, h],
                vec![tag; i * h],
            ));
            shard_tensors[shard].push((
                format!("{base}.up_proj.weight"),
                "F8_E4M3",
                vec![i, h],
                vec![tag ^ 0x80; i * h],
            ));
            shard_tensors[shard].push((
                format!("{base}.down_proj.weight"),
                "F8_E4M3",
                vec![h, i],
                vec![tag ^ 0x40; h * i],
            ));
            for projection in ["gate_proj", "up_proj"] {
                shard_tensors[shard].push((
                    format!("{base}.{projection}.weight_scale_inv"),
                    "BF16",
                    vec![i / BLOCK, h / BLOCK],
                    vec![tag; (i / BLOCK) * (h / BLOCK) * 2],
                ));
            }
            shard_tensors[shard].push((
                format!("{base}.down_proj.weight_scale_inv"),
                "BF16",
                vec![h / BLOCK, i / BLOCK],
                vec![tag; (h / BLOCK) * (i / BLOCK) * 2],
            ));
        }
        let mut weight_map = serde_json::Map::new();
        for (index, tensors) in shard_tensors.iter().enumerate() {
            let filename = format!("shard-{index}.safetensors");
            let mut header = serde_json::Map::new();
            let mut payload = Vec::new();
            for (name, dtype, shape, bytes) in tensors {
                header.insert(name.clone(), serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [payload.len(), payload.len() + bytes.len()]}));
                payload.extend_from_slice(bytes);
                weight_map.insert(name.clone(), serde_json::json!(filename));
            }
            let header = serde_json::to_vec(&header).unwrap();
            let mut file = (header.len() as u64).to_le_bytes().to_vec();
            file.extend_from_slice(&header);
            file.extend_from_slice(&payload);
            std::fs::write(dir.join(&filename), file).unwrap();
        }
        let write_index = |map: &serde_json::Map<String, serde_json::Value>| {
            std::fs::write(
                dir.join("model.safetensors.index.json"),
                serde_json::to_vec(&serde_json::json!({"weight_map": map})).unwrap(),
            )
            .unwrap();
        };
        write_index(&weight_map);

        let layer = load_moe_layer(&dir, 3, h, i, &mut |_, _| {}).unwrap();
        let per_expert = 2 * i * h;
        for expert in [0usize, 63, 64, 127] {
            let tag = expert as u8;
            assert_eq!(layer.gate_up_fp8()[expert * per_expert], tag);
            assert_eq!(layer.gate_up_fp8()[expert * per_expert + i * h], tag ^ 0x80);
            assert_eq!(layer.down_fp8()[expert * h * i], tag ^ 0x40);
        }
        assert!(layer.router_bf16().iter().all(|&v| v == 0x1111));
        assert!(
            load_moe_layer(&dir, 4, h, i, &mut |_, _| {}).is_err(),
            "layer 4 is not in the checkpoint"
        );
        assert!(load_qwen3_fp8_moe_layer(&dir, QWEN3_A3B_LAYERS).is_err());

        let mut escaping = weight_map.clone();
        escaping.insert(
            format!("{prefix}.gate.weight"),
            serde_json::json!("../shard-0.safetensors"),
        );
        write_index(&escaping);
        assert!(matches!(
            load_moe_layer(&dir, 3, h, i, &mut |_, _| {}),
            Err(Qwen3MoeError::Index(_))
        ));

        let mut wrong_shard = weight_map.clone();
        wrong_shard.insert(
            format!("{prefix}.experts.5.down_proj.weight"),
            serde_json::json!("shard-1.safetensors"),
        );
        write_index(&wrong_shard);
        assert!(matches!(
            load_moe_layer(&dir, 3, h, i, &mut |_, _| {}),
            Err(Qwen3MoeError::Contract(_))
        ));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn oracle_matches_hand_computed_identity_expert() {
        let (h, i) = (128, 128);
        let mut gate_up = vec![0u8; EXPERTS * 2 * i * h];
        let mut down = vec![0u8; EXPERTS * h * i];
        for e in 0..EXPERTS {
            for c in 0..128 {
                gate_up[e * 2 * i * h + c * h + c] = 0x38;
                gate_up[e * 2 * i * h + (i + c) * h + c] = 0x40; // 2.0
                down[e * h * i + c * i + c] = 0x38;
            }
        }
        let layer = Qwen3MoeLayer::from_parts(
            h,
            i,
            &vec![0u16; EXPERTS * h],
            &gate_up,
            &vec![0x3f80; EXPERTS * 2],
            &down,
            &vec![0x3f80; EXPERTS],
        )
        .unwrap();
        let x = vec![1.0f32; h];
        let plan =
            MoeBatchPlan::qwen3_coder_a3b(&reference_router_logits(&layer, &x, 1), 1).unwrap();
        let out = reference_moe_forward(&layer, &x, &plan);
        let expected = 2.0 / (1.0 + (-1.0f32).exp());
        assert!(
            out.iter().all(|v| (v - expected).abs() < 1e-5),
            "{:?}",
            &out[..4]
        );
    }
}
