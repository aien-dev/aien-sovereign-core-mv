// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! GB10 weight ownership (sovereign-core #277, cut C; design: docs/design/gb10-weight-ownership.md).
//!
//! On the GB10 production path a matmul weight lives on the device only: it is streamed from
//! the checkpoint's bf16 bytes straight into a resident tensor, and the process never holds an
//! f32 copy of it. [`MatrixWeight`] is the one type that says which of the two a weight is.
//!
//! * `Host`: the f32 values, for the CPU reference backend, dev builds, tests and every model
//!   that is not on the GB10 resident path. Unchanged behaviour.
//! * `Resident`: a handle to the device copy. There are no host values; asking for them is a
//!   named refusal (`RESIDENT_WEIGHT_REFUSED`), never a silent CPU computation.

use crate::checkpoint::{decode_bf16_to_fp32, CheckpointError, StreamedCheckpoint};
use crate::weights::{TransformerLayerWeights, TransformerWeights};
use aien_abi_core::ModelConfig;
use std::sync::{Arc, Mutex};

/// Tag every refusal about a resident weight carries, so logs and receipts can be grepped.
pub const RESIDENT_REFUSAL_PREFIX: &str = "RESIDENT_WEIGHT_REFUSED";

/// Panics with the named refusal. A resident weight has no host values, so there is nothing
/// a fallback could compute with; the failure must be loud.
#[cold]
pub(crate) fn refuse_resident(what: &str) -> ! {
    panic!(
        "{RESIDENT_REFUSAL_PREFIX}: {what}; the weight is resident on the device and has no host copy, so there is no CPU fallback"
    )
}

/// The device side of a resident weight. The omega implementation wraps a `ResidentTensor`
/// behind a lock; tests use a fake that never touches a device.
pub trait ResidentHandle: Send + Sync + std::fmt::Debug {
    /// `c[m x n] = a[m x k] * W`, f32 `a`. Returns the chip time in nanoseconds.
    fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String>;
}

impl ResidentHandle for Mutex<aien_omega_gpu::ResidentTensor> {
    fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String> {
        let guard = self
            .lock()
            .map_err(|_| "resident tensor lock poisoned".to_string())?;
        guard
            .matmul_f32(m, a, c)
            .map(|info| info.raw.elapsed_ns)
            .map_err(|e| e.to_string())
    }
}

/// A matmul weight that lives on the device: `k` is the input width, `n` the output width.
/// Cloning shares the device copy; it is freed when the last clone drops.
#[derive(Clone, Debug)]
pub struct ResidentWeight {
    k: usize,
    n: usize,
    handle: Arc<dyn ResidentHandle>,
}

impl ResidentWeight {
    pub fn new(k: usize, n: usize, handle: Arc<dyn ResidentHandle>) -> Self {
        Self { k, n, handle }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    pub fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String> {
        self.handle.matmul_f32(m, a, c)
    }
}

/// A matmul weight, `[out_dim, in_dim]` row-major when on the host.
#[derive(Clone, Debug)]
pub enum MatrixWeight {
    Host(Vec<f32>),
    Resident(ResidentWeight),
}

/// A borrowed [`MatrixWeight`], as the backend sees it.
#[derive(Clone, Copy, Debug)]
pub enum MatrixRef<'a> {
    Host(&'a [f32]),
    Resident(&'a ResidentWeight),
}

impl<'a> MatrixRef<'a> {
    /// The host values, or the named refusal when the weight is resident.
    pub fn host_or_refuse(self, op: &str) -> &'a [f32] {
        match self {
            Self::Host(v) => v,
            Self::Resident(r) => refuse_resident(&format!(
                "{op} needs host f32 values of a {}x{} weight",
                r.k, r.n
            )),
        }
    }
}

impl MatrixWeight {
    pub fn as_ref(&self) -> MatrixRef<'_> {
        match self {
            Self::Host(v) => MatrixRef::Host(v),
            Self::Resident(r) => MatrixRef::Resident(r),
        }
    }

    pub fn is_resident(&self) -> bool {
        matches!(self, Self::Resident(_))
    }
}

impl From<Vec<f32>> for MatrixWeight {
    fn from(v: Vec<f32>) -> Self {
        Self::Host(v)
    }
}

/// Host-slice view for the CPU reference code and tests. A resident weight is refused by name.
impl std::ops::Deref for MatrixWeight {
    type Target = [f32];
    fn deref(&self) -> &[f32] {
        self.as_ref().host_or_refuse("host slice access")
    }
}

/// What omega's `omega_gpu_tensor_upload_f32` would store for the bf16 pattern `bits` after the
/// loader widened it to f32: its f32-to-bf16 step is `(__bf16)f` (src/omega_blackwell_matmul.c:98
/// at omega.lock), which rounds to nearest even and sets the quiet bit of a NaN. Widening a bf16
/// is exact, so every pattern comes back unchanged except the 126 signaling NaNs (exponent all
/// ones, quiet bit clear, mantissa non-zero), which come back with the quiet bit set. Checked
/// against the compiler's own cast for all 65536 patterns on this aarch64 host (see the test).
#[inline]
pub fn bf16_bits_as_uploaded_today(bits: u16) -> u16 {
    let is_nan = bits & 0x7f80 == 0x7f80 && bits & 0x007f != 0;
    if is_nan {
        bits | 0x0040
    } else {
        bits
    }
}

/// Transposes bf16 weight bytes `[n][k]` (little endian, as safetensors stores `[out][in]`) into
/// the `[k][n]` bf16 layout the engine wants, applying [`bf16_bits_as_uploaded_today`]. `dst` is
/// resized to `k * n` and reused across calls.
pub fn transpose_bf16_for_upload(raw: &[u8], n: usize, k: usize, dst: &mut Vec<u16>) {
    debug_assert_eq!(raw.len(), n * k * 2);
    dst.clear();
    dst.resize(n * k, 0);
    const TILE: usize = 64;
    for o0 in (0..n).step_by(TILE) {
        let o1 = (o0 + TILE).min(n);
        for i0 in (0..k).step_by(TILE) {
            let i1 = (i0 + TILE).min(k);
            for o in o0..o1 {
                let row = &raw[o * k * 2..(o + 1) * k * 2];
                for i in i0..i1 {
                    let bits = u16::from_le_bytes([row[i * 2], row[i * 2 + 1]]);
                    dst[i * n + o] = bf16_bits_as_uploaded_today(bits);
                }
            }
        }
    }
}

/// Makes a `[k][n]` bf16 matrix resident.
pub trait ResidentUploader {
    fn upload_bf16(&self, k: usize, n: usize, bits: &[u16]) -> Result<ResidentWeight, String>;
}

/// The real uploader: omega's `omega_gpu_tensor_upload_bf16`.
pub struct OmegaUploader;

impl ResidentUploader for OmegaUploader {
    fn upload_bf16(&self, k: usize, n: usize, bits: &[u16]) -> Result<ResidentWeight, String> {
        let tensor =
            aien_omega_gpu::ResidentTensor::upload_bf16(k, n, bits).map_err(|e| e.to_string())?;
        Ok(ResidentWeight::new(k, n, Arc::new(Mutex::new(tensor))))
    }
}

/// Whether the daemon loads weights resident: the native engine is linked, the build is
/// production-strict (no CPU fallback to need host values for), the GB10 Qwen3 path
/// is enabled (see `gb10_qwen3_enabled`), and the model is one the engine accepts. Anything else keeps the host f32 load.
pub fn resident_load_wanted(
    native_linked: bool,
    strict: bool,
    qwen3_enabled: bool,
    config: &ModelConfig,
) -> bool {
    native_linked
        && strict
        && config.qk_norm
        && crate::omega_backend::omega_model_refusal_with(config, qwen3_enabled).is_none()
}

/// Streams a checkpoint straight to the device. For each of the seven matmul weights of each
/// layer (and the tied or separate output head) it reads that tensor's bf16 bytes, transposes
/// them into `[k][n]` bf16 and uploads. All uploads happen BEFORE the f32 embedding is decoded,
/// so the driver's pinned-memory requests land while the process holds almost nothing. The
/// returned weights hold resident matrices, the f32 embedding and the small f32 norm vectors.
///
/// `progress` is called once per uploaded matrix with its tensor name.
pub fn load_resident_weights<P: AsRef<std::path::Path>>(
    path: P,
    config: &ModelConfig,
    uploader: &dyn ResidentUploader,
    progress: &mut dyn FnMut(&str),
) -> Result<TransformerWeights, CheckpointError> {
    let catalog = crate::checkpoint::llama_catalog(config);
    let ckpt = StreamedCheckpoint::open(path, &catalog)?;
    let mut raw: Vec<u8> = Vec::new();
    let mut part: Vec<u8> = Vec::new();
    let mut bits: Vec<u16> = Vec::new();
    // One device matrix from one tensor, or from several `[n_i, k]` tensors stacked row-wise
    // (the fused q/k/v and gate/up matrices, sc#384): bf16 rows are contiguous, so stacking is
    // appending bytes; `n` is the total row count.
    let mut upload =
        |names: &[&str], n: usize, k: usize| -> Result<MatrixWeight, CheckpointError> {
            raw.clear();
            for name in names {
                ckpt.read_bf16_bytes(name, &mut part)?;
                raw.extend_from_slice(&part);
            }
            let label = names.join("+");
            if raw.len() != n * k * 2 {
                return Err(CheckpointError::ResidentUpload {
                    tensor: label,
                    message: format!(
                        "{} bf16 bytes read, {} expected for {n}x{k}",
                        raw.len(),
                        n * k * 2
                    ),
                });
            }
            transpose_bf16_for_upload(&raw, n, k, &mut bits);
            let w = uploader.upload_bf16(k, n, &bits).map_err(|message| {
                CheckpointError::ResidentUpload {
                    tensor: label.clone(),
                    message,
                }
            })?;
            progress(&label);
            Ok(MatrixWeight::Resident(w))
        };
    let hidden = config.hidden_dim();
    let inter = config.intermediate_dim();
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;

    // Pass 1: every matmul weight to the device. Nothing large is held on the host.
    let mut mats = Vec::with_capacity(config.num_layers);
    for idx in 0..config.num_layers {
        let p = format!("model.layers.{idx}");
        let q = format!("{p}.self_attn.q_proj.weight");
        let k = format!("{p}.self_attn.k_proj.weight");
        let v = format!("{p}.self_attn.v_proj.weight");
        let gate = format!("{p}.mlp.gate_proj.weight");
        let up = format!("{p}.mlp.up_proj.weight");
        mats.push([
            upload(&[&q], q_dim, hidden)?,
            upload(&[&k], kv_dim, hidden)?,
            upload(&[&v], kv_dim, hidden)?,
            upload(&[&format!("{p}.self_attn.o_proj.weight")], hidden, q_dim)?,
            upload(&[&gate], inter, hidden)?,
            upload(&[&up], inter, hidden)?,
            upload(&[&format!("{p}.mlp.down_proj.weight")], hidden, inter)?,
            // The fused matrices the forward paths prefer (one launch each): q|k|v rows and
            // gate|up rows. The single matrices above stay for every other reader.
            upload(&[&q, &k, &v], q_dim + 2 * kv_dim, hidden)?,
            upload(&[&gate, &up], 2 * inter, hidden)?,
        ]);
    }
    let vocab = config.vocab_size();
    // Tied models project through the embedding: its own device copy, transposed.
    let head_name = if config.tie_word_embeddings {
        "model.embed_tokens.weight"
    } else {
        "lm_head.weight"
    };
    let head = upload(&[head_name], vocab, hidden)?;
    // Release the big transient buffers before the f32 embedding is allocated.
    raw = Vec::new();
    drop(bits);

    // Pass 2: the host-side f32 data the host really reads (embedding rows, norms).
    let decode = |name: &str, raw: &mut Vec<u8>| -> Result<Vec<f32>, CheckpointError> {
        ckpt.read_bf16_bytes(name, raw)?;
        Ok(decode_bf16_to_fp32(raw))
    };
    let mut layers = Vec::with_capacity(config.num_layers);
    for (idx, m) in mats.into_iter().enumerate() {
        let p = format!("model.layers.{idx}");
        let [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj, qkv_proj, gate_up_proj] =
            m;
        let (q_norm, k_norm) = if config.qk_norm {
            (
                Some(decode(&format!("{p}.self_attn.q_norm.weight"), &mut raw)?),
                Some(decode(&format!("{p}.self_attn.k_norm.weight"), &mut raw)?),
            )
        } else {
            (None, None)
        };
        layers.push(TransformerLayerWeights {
            input_layernorm: decode(&format!("{p}.input_layernorm.weight"), &mut raw)?,
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            post_attention_layernorm: decode(
                &format!("{p}.post_attention_layernorm.weight"),
                &mut raw,
            )?,
            gate_proj,
            up_proj,
            down_proj,
            qkv_proj: Some(qkv_proj),
            gate_up_proj: Some(gate_up_proj),
            q_norm,
            k_norm,
        });
    }
    let final_norm = decode("model.norm.weight", &mut raw)?;
    let embed_tokens = decode("model.embed_tokens.weight", &mut raw)?;
    Ok(TransformerWeights {
        config: config.clone(),
        embed_tokens,
        layers,
        final_norm,
        lm_head: Some(head),
    })
}
