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

//! Strict Safetensors checkpoint loader with loud shape and dtype validation.
//! Ingests BF16 weights into one `ModelCapsule`. FP32 is decoded on demand.

use std::collections::HashMap;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::Arc;

/// Loud error enum describing checkpoint parsing and validation failures.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointError {
    /// A required tensor in the model catalog was not found in the checkpoint.
    MissingTensor(String),
    /// A tensor's shape in the checkpoint does not match the canonical model architecture.
    ShapeMismatch {
        tensor: String,
        expected: Vec<usize>,
        actual: Vec<usize>,
    },
    /// A tensor's dtype is not BF16.
    DtypeMismatch {
        tensor: String,
        expected: String,
        actual: String,
    },
    /// A tensor's byte range exceeds the binary payload or does not match shape element count.
    OffsetOutOfBounds {
        tensor: String,
        offset: usize,
        buffer_len: usize,
    },
    /// Safetensors binary header is malformed, truncated, or invalid JSON.
    InvalidHeader(String),
    /// A capsule tensor could not be decoded.
    Capsule(String),
    /// A weight could not be made resident on the device (named, never a silent host copy).
    ResidentUpload { tensor: String, message: String },
}

impl fmt::Display for CheckpointError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingTensor(name) => {
                write!(f, "Missing required tensor in checkpoint: {}", name)
            }
            Self::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Shape mismatch for tensor {}: expected {:?}, got {:?}",
                    tensor, expected, actual
                )
            }
            Self::DtypeMismatch {
                tensor,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "Dtype mismatch for tensor {}: expected {}, got {}",
                    tensor, expected, actual
                )
            }
            Self::OffsetOutOfBounds {
                tensor,
                offset,
                buffer_len,
            } => {
                write!(
                    f,
                    "Offset out of bounds for tensor {}: offset {} exceeds buffer length {}",
                    tensor, offset, buffer_len
                )
            }
            Self::InvalidHeader(err) => {
                write!(f, "Invalid safetensors header: {}", err)
            }
            Self::Capsule(err) => write!(f, "Invalid capsule tensor: {}", err),
            Self::ResidentUpload { tensor, message } => {
                write!(
                    f,
                    "Resident upload of tensor {} failed: {}",
                    tensor, message
                )
            }
        }
    }
}

impl std::error::Error for CheckpointError {}

/// Checkpoint stored as one capsule. Tensor bytes stay in `capsule.bytes`.
/// FP32 is produced by `decode_fp32` and is not retained.
#[derive(Debug, Clone)]
pub struct LoadedCheckpoint {
    pub capsule: crate::capsule::ModelCapsule,
}

impl Default for LoadedCheckpoint {
    fn default() -> Self {
        Self::new()
    }
}

impl LoadedCheckpoint {
    /// Empty capsule: empty byte buffer, empty tensor map, digests of that empty input.
    pub fn new() -> Self {
        Self::from_bf16_tensors(Vec::new())
    }

    /// Builds a checkpoint from BF16 payloads without a safetensors file.
    /// Payloads are concatenated into one `Arc<[u8]>`. Each tensor records its range.
    pub fn from_bf16_tensors(items: Vec<(String, Vec<usize>, Vec<u8>)>) -> Self {
        let mut payload = Vec::new();
        let mut tensors = HashMap::with_capacity(items.len());
        for (name, shape, raw) in items {
            let start = payload.len();
            payload.extend_from_slice(&raw);
            let end = payload.len();
            let logical_strides = crate::capsule::row_major_strides(&shape);
            tensors.insert(
                name,
                crate::capsule::CapsuleTensor {
                    dtype: crate::capsule::WeightDType::Bf16,
                    layout: crate::capsule::TensorLayout::RowMajor,
                    shape,
                    logical_strides,
                    byte_range: start..end,
                    quant: None,
                },
            );
        }
        Self::from_owned_bytes(Arc::from(payload), tensors)
    }

    pub fn tensor_count(&self) -> usize {
        self.capsule.tensors.len()
    }

    pub fn contains_tensor(&self, name: &str) -> bool {
        self.capsule.tensors.contains_key(name)
    }

    pub fn decode_fp32(&self, name: &str) -> Result<Vec<f32>, crate::capsule::CapsuleError> {
        self.capsule.decode_fp32(name)
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], crate::capsule::CapsuleError> {
        self.capsule.tensor_bytes(name)
    }

    pub fn get_shape(&self, name: &str) -> Option<&Vec<usize>> {
        self.capsule.tensors.get(name).map(|tensor| &tensor.shape)
    }

    fn from_owned_bytes(
        bytes: Arc<[u8]>,
        tensors: HashMap<String, crate::capsule::CapsuleTensor>,
    ) -> Self {
        let source_digest = crate::capsule::source_digest(&bytes);
        let capsule_digest = crate::capsule::capsule_digest(&bytes, &tensors);
        Self {
            capsule: crate::capsule::ModelCapsule {
                bytes,
                source_digest,
                tensors,
                capsule_digest,
            },
        }
    }
}

/// Decodes little-endian bfloat16 raw bytes to single-precision f32.
/// Each 16-bit BF16 word occupies the upper 16 bits of the IEEE-754 32-bit float.
pub fn decode_bf16_to_fp32(bf16_bytes: &[u8]) -> Vec<f32> {
    let count = bf16_bytes.len() / 2;
    let mut out = Vec::with_capacity(count);
    for chunk in bf16_bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f32::from_bits((bits as u32) << 16));
    }
    out
}

/// Encodes single-precision f32 to little-endian bfloat16 raw bytes by truncation of lower 16 bits.
pub fn encode_fp32_to_bf16(floats: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(floats.len() * 2);
    for &val in floats {
        let bits = val.to_bits();
        let bf16_bits = (bits >> 16) as u16;
        out.extend_from_slice(&bf16_bits.to_le_bytes());
    }
    out
}

/// Generates the canonical catalog of all 201 TinyLlama-1.1B tensors with their expected shapes.
pub fn tinyllama_catalog() -> Vec<(String, Vec<usize>)> {
    let mut catalog = Vec::with_capacity(201);

    catalog.push(("model.embed_tokens.weight".to_string(), vec![32000, 2048]));

    for layer in 0..22 {
        let prefix = format!("model.layers.{}", layer);
        catalog.push((format!("{}.input_layernorm.weight", prefix), vec![2048]));
        catalog.push((
            format!("{}.self_attn.q_proj.weight", prefix),
            vec![2048, 2048],
        ));
        catalog.push((
            format!("{}.self_attn.k_proj.weight", prefix),
            vec![256, 2048],
        ));
        catalog.push((
            format!("{}.self_attn.v_proj.weight", prefix),
            vec![256, 2048],
        ));
        catalog.push((
            format!("{}.self_attn.o_proj.weight", prefix),
            vec![2048, 2048],
        ));
        catalog.push((
            format!("{}.post_attention_layernorm.weight", prefix),
            vec![2048],
        ));
        catalog.push((format!("{}.mlp.gate_proj.weight", prefix), vec![5632, 2048]));
        catalog.push((format!("{}.mlp.up_proj.weight", prefix), vec![5632, 2048]));
        catalog.push((format!("{}.mlp.down_proj.weight", prefix), vec![2048, 5632]));
    }

    catalog.push(("model.norm.weight".to_string(), vec![2048]));
    catalog.push(("lm_head.weight".to_string(), vec![32000, 2048]));

    catalog
}

/// The tensor catalog of a Llama-architecture checkpoint (`LlamaForCausalLM`, or Qwen3 with
/// per-layer `q_norm`/`k_norm` when `config.qk_norm`) described by
/// `config`: names and shapes of every tensor the forward pass reads. `lm_head.weight` is
/// left out when the model ties its output projection to `model.embed_tokens.weight`.
/// For `ModelConfig::tinyllama_1_1b()` this is exactly [`tinyllama_catalog`].
pub fn llama_catalog(config: &aien_abi_core::ModelConfig) -> Vec<(String, Vec<usize>)> {
    let hidden = config.hidden_dim();
    let inter = config.intermediate_dim();
    let vocab = config.vocab_size();
    let q_dim = config.num_heads * config.head_dim;
    let kv_dim = config.num_kv_heads * config.head_dim;
    let mut catalog = Vec::with_capacity(3 + 11 * config.num_layers);
    catalog.push(("model.embed_tokens.weight".to_string(), vec![vocab, hidden]));
    for layer in 0..config.num_layers {
        let prefix = format!("model.layers.{}", layer);
        catalog.push((format!("{}.input_layernorm.weight", prefix), vec![hidden]));
        catalog.push((
            format!("{}.self_attn.q_proj.weight", prefix),
            vec![q_dim, hidden],
        ));
        catalog.push((
            format!("{}.self_attn.k_proj.weight", prefix),
            vec![kv_dim, hidden],
        ));
        catalog.push((
            format!("{}.self_attn.v_proj.weight", prefix),
            vec![kv_dim, hidden],
        ));
        catalog.push((
            format!("{}.self_attn.o_proj.weight", prefix),
            vec![hidden, q_dim],
        ));
        catalog.push((
            format!("{}.post_attention_layernorm.weight", prefix),
            vec![hidden],
        ));
        catalog.push((
            format!("{}.mlp.gate_proj.weight", prefix),
            vec![inter, hidden],
        ));
        catalog.push((
            format!("{}.mlp.up_proj.weight", prefix),
            vec![inter, hidden],
        ));
        catalog.push((
            format!("{}.mlp.down_proj.weight", prefix),
            vec![hidden, inter],
        ));
        if config.qk_norm {
            for name in ["q_norm", "k_norm"] {
                catalog.push((
                    format!("{}.self_attn.{}.weight", prefix, name),
                    vec![config.head_dim],
                ));
            }
        }
    }
    catalog.push(("model.norm.weight".to_string(), vec![hidden]));
    if !config.tie_word_embeddings {
        catalog.push(("lm_head.weight".to_string(), vec![vocab, hidden]));
    }
    catalog
}

/// Parses a safetensors binary buffer and strictly validates against a specified tensor catalog.
/// The file is copied once into the capsule. Tensor ranges point into that buffer.
pub fn parse_safetensors_with_catalog(
    bytes: &[u8],
    catalog: &[(String, Vec<usize>)],
) -> Result<LoadedCheckpoint, CheckpointError> {
    let bytes: Arc<[u8]> = Arc::from(bytes.to_vec());
    parse_safetensors_arc(bytes, catalog)
}

fn parse_safetensors_arc(
    bytes: Arc<[u8]>,
    catalog: &[(String, Vec<usize>)],
) -> Result<LoadedCheckpoint, CheckpointError> {
    let whole = 0..bytes.len();
    parse_shards(bytes, &[whole], catalog)
}

/// Validates `catalog` against one or more safetensors files laid end to end in `bytes`
/// (`shards` are their byte ranges). Every catalog tensor must appear in exactly one shard.
fn parse_shards(
    bytes: Arc<[u8]>,
    shards: &[std::ops::Range<usize>],
    catalog: &[(String, Vec<usize>)],
) -> Result<LoadedCheckpoint, CheckpointError> {
    let mut tensors = HashMap::with_capacity(catalog.len());
    for shard in shards {
        index_shard(&bytes, shard.clone(), catalog, &mut tensors)?;
    }
    if let Some((name, _)) = catalog.iter().find(|(name, _)| !tensors.contains_key(name)) {
        return Err(CheckpointError::MissingTensor(name.clone()));
    }
    Ok(LoadedCheckpoint::from_owned_bytes(bytes, tensors))
}

/// Indexes the catalog tensors present in the safetensors file at `bytes[shard]`.
fn index_shard(
    all_bytes: &[u8],
    shard: std::ops::Range<usize>,
    catalog: &[(String, Vec<usize>)],
    tensors: &mut HashMap<String, crate::capsule::CapsuleTensor>,
) -> Result<(), CheckpointError> {
    let base = shard.start;
    let bytes = &all_bytes[shard];
    if bytes.len() < 8 {
        return Err(CheckpointError::InvalidHeader(
            "Buffer smaller than 8-byte header prefix".to_string(),
        ));
    }

    let header_len_raw =
        u64::from_le_bytes(bytes[0..8].try_into().map_err(|_| {
            CheckpointError::InvalidHeader("Failed to read header size".to_string())
        })?);
    let header_len = usize::try_from(header_len_raw).map_err(|_| {
        CheckpointError::InvalidHeader("Header length exceeds address space".to_string())
    })?;

    let header_end = 8usize.checked_add(header_len).ok_or_else(|| {
        CheckpointError::InvalidHeader("Header length arithmetic overflow".to_string())
    })?;

    if bytes.len() < header_end {
        return Err(CheckpointError::InvalidHeader(format!(
            "Header length {} exceeds total buffer size {}",
            header_len,
            bytes.len()
        )));
    }

    let header_str = std::str::from_utf8(&bytes[8..header_end]).map_err(|e| {
        CheckpointError::InvalidHeader(format!("Header JSON is not valid UTF-8: {}", e))
    })?;

    let header: serde_json::Value = serde_json::from_str(header_str).map_err(|e| {
        CheckpointError::InvalidHeader(format!("Failed to parse header JSON: {}", e))
    })?;

    let header_obj = header.as_object().ok_or_else(|| {
        CheckpointError::InvalidHeader("Header JSON must be an object".to_string())
    })?;

    index_header(
        header_obj,
        bytes.len() - header_end,
        base + header_end,
        catalog,
        tensors,
    )
}

/// Validates the catalog tensors named in one safetensors header against their declared shape,
/// dtype and offsets. `data_len` is the size of the data blob after the header; `data_start` is
/// the absolute position of that blob in whatever buffer or file the ranges refer to.
fn index_header(
    header_obj: &serde_json::Map<String, serde_json::Value>,
    data_len: usize,
    data_start: usize,
    catalog: &[(String, Vec<usize>)],
    tensors: &mut HashMap<String, crate::capsule::CapsuleTensor>,
) -> Result<(), CheckpointError> {
    for (name, expected_shape) in catalog {
        let Some(info) = header_obj.get(name) else {
            continue;
        };
        if tensors.contains_key(name) {
            return Err(CheckpointError::InvalidHeader(format!(
                "tensor {} appears in more than one shard",
                name
            )));
        }

        // 1. Verify dtype is BF16. This catalog parser does not accept other dtypes.
        let dtype = info
            .get("dtype")
            .and_then(|v| v.as_str())
            .ok_or_else(|| CheckpointError::MissingTensor(format!("{}.dtype", name)))?;

        if dtype != "BF16" {
            return Err(CheckpointError::DtypeMismatch {
                tensor: name.clone(),
                expected: "BF16".to_string(),
                actual: dtype.to_string(),
            });
        }

        // 2. Verify shape matches canonical architecture
        let shape_arr = info
            .get("shape")
            .and_then(|v| v.as_array())
            .ok_or_else(|| CheckpointError::MissingTensor(format!("{}.shape", name)))?;

        let actual_shape: Vec<usize> = shape_arr
            .iter()
            .map(|v| {
                v.as_u64()
                    .and_then(|n| usize::try_from(n).ok())
                    .ok_or_else(|| {
                        CheckpointError::InvalidHeader(format!(
                            "Invalid dimension in shape for {}",
                            name
                        ))
                    })
            })
            .collect::<Result<Vec<usize>, CheckpointError>>()?;

        if actual_shape != *expected_shape {
            return Err(CheckpointError::ShapeMismatch {
                tensor: name.clone(),
                expected: expected_shape.clone(),
                actual: actual_shape,
            });
        }

        // 3. Verify offsets and byte length. Safetensors offsets are relative to the data blob.
        let offsets_arr = info
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .ok_or_else(|| CheckpointError::MissingTensor(format!("{}.data_offsets", name)))?;

        if offsets_arr.len() != 2 {
            return Err(CheckpointError::InvalidHeader(format!(
                "data_offsets for {} must have exactly 2 elements",
                name
            )));
        }

        let start = offsets_arr[0]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                CheckpointError::InvalidHeader(format!("Invalid start offset for {}", name))
            })?;
        let end = offsets_arr[1]
            .as_u64()
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| {
                CheckpointError::InvalidHeader(format!("Invalid end offset for {}", name))
            })?;
        let expected_bytes = expected_shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .and_then(|n| n.checked_mul(2))
            .ok_or_else(|| {
                CheckpointError::InvalidHeader(format!("Tensor {} byte count overflows", name))
            })?;

        if start > end || (end - start) != expected_bytes || end > data_len {
            return Err(CheckpointError::OffsetOutOfBounds {
                tensor: name.clone(),
                offset: end,
                buffer_len: data_len,
            });
        }

        // Ranges are absolute inside the whole file, including the 8-byte length and JSON header.
        let abs_start = data_start.checked_add(start).ok_or_else(|| {
            CheckpointError::InvalidHeader(format!("Tensor {} offset overflows", name))
        })?;
        let abs_end = data_start.checked_add(end).ok_or_else(|| {
            CheckpointError::InvalidHeader(format!("Tensor {} offset overflows", name))
        })?;
        let logical_strides = crate::capsule::row_major_strides(&actual_shape);
        tensors.insert(
            name.clone(),
            crate::capsule::CapsuleTensor {
                dtype: crate::capsule::WeightDType::Bf16,
                layout: crate::capsule::TensorLayout::RowMajor,
                shape: actual_shape,
                logical_strides,
                byte_range: abs_start..abs_end,
                quant: None,
            },
        );
    }

    Ok(())
}

/// Loads a TinyLlama safetensors checkpoint from an owned byte buffer.
/// The `Vec` is moved into one `Arc` and is not copied per tensor.
pub fn load_safetensors_from_bytes(bytes: Vec<u8>) -> Result<LoadedCheckpoint, CheckpointError> {
    let catalog = tinyllama_catalog();
    parse_safetensors_arc(Arc::from(bytes), &catalog)
}

/// Name of the shard index Hugging Face writes beside sharded safetensors files.
pub const SAFETENSORS_INDEX: &str = "model.safetensors.index.json";

// Bounds apply to disk input, not to in-memory tensor/capsule construction.
// Large BF16 checkpoints may span many shards; keep the aggregate limit above
// production model sizes while refusing absurd metadata and sparse file lengths.
const MAX_INDEX_BYTES: u64 = 16 << 20;
const MAX_HEADER_BYTES: u64 = 256 << 20;
const MAX_SHARD_BYTES: u64 = 256 << 30;
const MAX_CHECKPOINT_BYTES: u64 = 1 << 40;

fn invalid_file(path: &Path, reason: impl fmt::Display) -> CheckpointError {
    CheckpointError::InvalidHeader(format!("{}: {reason}", path.display()))
}

/// Reject special files before open; O_NONBLOCK and O_NOFOLLOW guard against
/// swaps to FIFOs and symlinks at open. Retain this fd through reads.
fn open_regular(path: &Path, limit: u64) -> Result<(File, usize), CheckpointError> {
    let candidate = std::fs::symlink_metadata(path).map_err(|e| invalid_file(path, e))?;
    if !candidate.file_type().is_file() {
        return Err(invalid_file(path, "symlink or nonregular checkpoint file"));
    }
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NONBLOCK | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|e| invalid_file(path, e))?;
    let meta = file.metadata().map_err(|e| invalid_file(path, e))?;
    if !meta.is_file() {
        return Err(invalid_file(path, "not a regular file"));
    }
    if meta.len() > limit {
        return Err(invalid_file(path, format!("file exceeds {limit} bytes")));
    }
    let len = usize::try_from(meta.len())
        .map_err(|_| invalid_file(path, "file length exceeds address space"))?;
    Ok((file, len))
}

fn read_header(
    file: &File,
    path: &Path,
    file_len: usize,
    data_base: usize,
    catalog: &[(String, Vec<usize>)],
    tensors: &mut HashMap<String, crate::capsule::CapsuleTensor>,
) -> Result<(), CheckpointError> {
    if file_len < 8 {
        return Err(invalid_file(path, "Buffer smaller than 8-byte header prefix"));
    }
    let mut prefix = [0u8; 8];
    file.read_exact_at(&mut prefix, 0)
        .map_err(|e| invalid_file(path, e))?;
    let header_len = u64::from_le_bytes(prefix);
    if header_len > MAX_HEADER_BYTES || header_len > (file_len - 8) as u64 {
        return Err(invalid_file(path, format!("invalid header length {header_len}")));
    }
    let header_len = usize::try_from(header_len)
        .map_err(|_| invalid_file(path, "header length exceeds address space"))?;
    let header_end = 8 + header_len;
    let mut header = vec![0u8; header_len];
    file.read_exact_at(&mut header, 8)
        .map_err(|e| invalid_file(path, e))?;
    let header: serde_json::Value = serde_json::from_slice(&header)
        .map_err(|e| invalid_file(path, format!("invalid header JSON: {e}")))?;
    let object = header
        .as_object()
        .ok_or_else(|| invalid_file(path, "header JSON must be an object"))?;
    index_header(
        object,
        file_len - header_end,
        data_base.checked_add(header_end)
            .ok_or_else(|| invalid_file(path, "checkpoint byte offset overflow"))?,
        catalog,
        tensors,
    )
}

fn validate_catalog(
    tensors: &HashMap<String, crate::capsule::CapsuleTensor>,
    catalog: &[(String, Vec<usize>)],
) -> Result<(), CheckpointError> {
    if let Some((name, _)) = catalog.iter().find(|(name, _)| !tensors.contains_key(name)) {
        return Err(CheckpointError::MissingTensor(name.clone()));
    }
    Ok(())
}

fn shard_path(dir: &Path, name: &str) -> Result<PathBuf, CheckpointError> {
    let mut components = Path::new(name).components();
    if !matches!(
        (components.next(), components.next()),
        (Some(Component::Normal(component)), None) if component == name
    ) {
        return Err(CheckpointError::InvalidHeader(format!(
            "shard name {name:?} must be a plain file name"
        )));
    }
    let path = dir.join(name);
    // A symlink inside the directory can itself lead outside. Reject it,
    // including when a caller only asks for checkpoint_files/digest inputs.
    let meta = std::fs::symlink_metadata(&path).map_err(|e| invalid_file(&path, e))?;
    if !meta.file_type().is_file() {
        return Err(invalid_file(&path, "shard is a symlink or not a regular file"));
    }
    Ok(path)
}

/// Digest the same bounded, regular checkpoint file that a loader may open.
/// Keep the descriptor pinned so a path swap cannot change the file mid-read.
pub fn checkpoint_sha256(path: &Path) -> Result<[u8; 32], CheckpointError> {
    use sha2::{Digest, Sha256};
    let limit = if path.file_name().and_then(|n| n.to_str()) == Some(SAFETENSORS_INDEX) {
        MAX_INDEX_BYTES
    } else {
        MAX_SHARD_BYTES
    };
    let (file, len) = open_regular(path, limit)?;
    let mut hasher = Sha256::new();
    let mut buf = vec![0u8; len.min(1 << 20)];
    let mut offset = 0usize;
    while offset < len {
        let count = (len - offset).min(buf.len());
        file.read_exact_at(&mut buf[..count], offset as u64)
            .map_err(|e| invalid_file(path, e))?;
        hasher.update(&buf[..count]);
        offset += count;
    }
    if file.metadata().map_err(|e| invalid_file(path, e))?.len() != len as u64 {
        return Err(invalid_file(path, "file length changed during digest"));
    }
    Ok(hasher.finalize().into())
}


/// The files a checkpoint path names.
enum ResolvedFiles {
    /// One `.safetensors` file.
    Single(PathBuf),
    /// The `model.safetensors.index.json` and the shard files it names, as `(name in the
    /// index, path)`, sorted by name and de-duplicated.
    Sharded {
        index: PathBuf,
        shards: Vec<(String, PathBuf)>,
    },
}

impl ResolvedFiles {
    fn into_paths(self) -> Vec<PathBuf> {
        match self {
            Self::Single(p) => vec![p],
            Self::Sharded { shards, .. } => shards.into_iter().map(|(_, p)| p).collect(),
        }
    }
}

/// The files a checkpoint path names, exactly as the loaders read them (sc#338): what the
/// daemon digests so its record binds the weight bytes, not only the path it was given.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointFiles {
    /// The `model.safetensors.index.json` read; None for a single file.
    pub index: Option<PathBuf>,
    /// Every weight file as `(name, path)`, sorted by name and de-duplicated: the name the
    /// index's `weight_map` gives a shard, or the file name of a single file.
    pub weights: Vec<(String, PathBuf)>,
}

/// Resolves `path` like [`load_checkpoint_with_catalog`] and [`StreamedCheckpoint::open`]
/// do: a `.safetensors` file, a `model.safetensors.index.json`, a model directory holding
/// either, or a shard file whose directory has an index (the whole sharded set).
pub fn checkpoint_files<P: AsRef<Path>>(path: P) -> Result<CheckpointFiles, CheckpointError> {
    Ok(match resolve_checkpoint_files(path.as_ref())? {
        ResolvedFiles::Single(p) => {
            open_regular(&p, MAX_SHARD_BYTES)?;
            CheckpointFiles {
                index: None,
                weights: vec![(
                    p.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    p,
                )],
            }
        }
        ResolvedFiles::Sharded { index, shards } => CheckpointFiles {
            index: Some(index),
            weights: shards,
        },
    })
}

/// Decides which files `path` names: a `.safetensors` file, a `model.safetensors.index.json`
/// (sharded), or a model directory holding either. A shard file whose directory has an index
/// names the whole sharded set.
fn resolve_checkpoint_files(p: &Path) -> Result<ResolvedFiles, CheckpointError> {
    let index = if p.is_dir() {
        let single = p.join("model.safetensors");
        if single.is_file() {
            return Ok(ResolvedFiles::Single(single));
        }
        p.join(SAFETENSORS_INDEX)
    } else if p.file_name().and_then(|n| n.to_str()) == Some(SAFETENSORS_INDEX) {
        p.to_path_buf()
    } else {
        let sibling = p.with_file_name(SAFETENSORS_INDEX);
        let is_shard = p.file_name().and_then(|n| n.to_str()) != Some("model.safetensors");
        if !(is_shard && sibling.is_file()) {
            return Ok(ResolvedFiles::Single(p.to_path_buf()));
        }
        sibling
    };
    let (index_file, index_len) = open_regular(&index, MAX_INDEX_BYTES)?;
    let mut index_bytes = vec![0u8; index_len];
    index_file
        .read_exact_at(&mut index_bytes, 0)
        .map_err(|e| invalid_file(&index, e))?;
    let index_json: serde_json::Value = serde_json::from_slice(&index_bytes)
        .map_err(|e| invalid_file(&index, format!("invalid index JSON: {e}")))?;
    let weight_map = index_json
        .get("weight_map")
        .and_then(|m| m.as_object())
        .ok_or_else(|| invalid_file(&index, "missing weight_map"))?;
    let mut files: Vec<&str> = weight_map
        .values()
        .map(|v| v.as_str().ok_or_else(|| invalid_file(&index, "non-string shard name")))
        .collect::<Result<_, _>>()?;
    files.sort_unstable();
    files.dedup();
    let dir = index.parent().unwrap_or(Path::new("."));
    let mut shards = Vec::with_capacity(files.len());
    let mut total = 0u64;
    for name in files {
        let path = shard_path(dir, name)?;
        // checkpoint_files callers hash these paths, so reject nonregular or
        // impossible disk lengths before handing any path to a caller.
        let (_, len) = open_regular(&path, MAX_SHARD_BYTES)?;
        total = total
            .checked_add(len as u64)
            .filter(|n| *n <= MAX_CHECKPOINT_BYTES)
            .ok_or_else(|| invalid_file(&path, "checkpoint exceeds aggregate byte limit"))?;
        shards.push((name.to_string(), path));
    }
    Ok(ResolvedFiles::Sharded { index, shards })
}

/// Loads and validates a checkpoint against `catalog`. `path` is a `.safetensors` file, a
/// `model.safetensors.index.json` (sharded), or a model directory holding either. A shard
/// file whose directory has an index loads the whole sharded set.
pub fn load_checkpoint_with_catalog<P: AsRef<Path>>(
    path: P,
    catalog: &[(String, Vec<usize>)],
) -> Result<LoadedCheckpoint, CheckpointError> {
    load_paths(resolve_checkpoint_files(path.as_ref())?.into_paths(), catalog)
}

fn load_paths(
    paths: Vec<PathBuf>,
    catalog: &[(String, Vec<usize>)],
) -> Result<LoadedCheckpoint, CheckpointError> {
    let mut total = 0usize;
    let mut tensors = HashMap::with_capacity(catalog.len());
    let mut opened = Vec::with_capacity(paths.len());
    let mut ranges = Vec::with_capacity(paths.len());
    for path in &paths {
        let (file, len) = open_regular(path, MAX_SHARD_BYTES)?;
        let end = total
            .checked_add(len)
            .filter(|n| (*n as u64) <= MAX_CHECKPOINT_BYTES)
            .ok_or_else(|| invalid_file(path, "checkpoint exceeds aggregate byte limit"))?;
        read_header(&file, path, len, total, catalog, &mut tensors)?;
        ranges.push(total..end);
        opened.push(file);
        total = end;
    }
    validate_catalog(&tensors, catalog)?;
    // A small catalog must not authorize a terabyte-sized sparse payload.
    // Allow one header budget per shard and room for extra checkpoint tensors,
    // but scale the eager read with the model this caller actually expects.
    let expected = catalog
        .iter()
        .try_fold(0u64, |sum, (_, shape)| {
            shape
                .iter()
                .try_fold(2u64, |bytes, dim| bytes.checked_mul(*dim as u64))
                .and_then(|bytes| sum.checked_add(bytes))
        })
        .ok_or_else(|| CheckpointError::InvalidHeader("catalog byte count overflows".into()))?;
    let overhead = (paths.len() as u64).saturating_mul(MAX_HEADER_BYTES + 8);
    let budget = expected
        .saturating_mul(2)
        .saturating_add(overhead)
        .min(MAX_CHECKPOINT_BYTES);
    if (total as u64) > budget {
        return Err(CheckpointError::InvalidHeader(format!(
            "checkpoint exceeds catalog byte budget {budget}"
        )));
    }
    // Nothing above has allocated a full shard: all required tensor headers
    // and spans have been checked first, including cross-shard duplicates.
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(total)
        .map_err(|e| CheckpointError::InvalidHeader(format!("checkpoint allocation: {e}")))?;
    bytes.resize(total, 0);
    for ((path, file), range) in paths.iter().zip(opened.iter()).zip(ranges.iter()) {
        file.read_exact_at(&mut bytes[range.clone()], 0)
            .map_err(|e| invalid_file(path, e))?;
        if file.metadata().map_err(|e| invalid_file(path, e))?.len() != range.len() as u64 {
            return Err(invalid_file(path, "shard length changed during read"));
        }
    }
    // Check headers again in case an open file was modified between preflight
    // and payload read. The capsule must index exactly the bytes it owns.
    parse_shards(Arc::from(bytes), &ranges, catalog)
}

/// A checkpoint indexed but not read: the catalog is validated against each shard's header
/// (the same checks as [`load_checkpoint_with_catalog`]) and tensor bytes are read one tensor
/// at a time with positioned reads, so no whole-file copy is ever held in process memory.
/// The shard pages stay in the kernel's page cache, which the kernel can reclaim.
#[derive(Debug)]
pub struct StreamedCheckpoint {
    files: Vec<(PathBuf, File)>,
    tensors: HashMap<String, (usize, crate::capsule::CapsuleTensor)>,
}

impl StreamedCheckpoint {
    /// Indexes `path` (file, shard index, or model directory) against `catalog`. Reads only
    /// each shard's header. Every catalog tensor must appear in exactly one shard.
    pub fn open<P: AsRef<Path>>(
        path: P,
        catalog: &[(String, Vec<usize>)],
    ) -> Result<Self, CheckpointError> {
        let paths = resolve_checkpoint_files(path.as_ref())?.into_paths();
        let mut files = Vec::with_capacity(paths.len());
        let mut tensors = HashMap::with_capacity(catalog.len());
        let mut total = 0usize;
        for path in paths {
            let (file, len) = open_regular(&path, MAX_SHARD_BYTES)?;
            total = total
                .checked_add(len)
                .filter(|n| (*n as u64) <= MAX_CHECKPOINT_BYTES)
                .ok_or_else(|| invalid_file(&path, "checkpoint exceeds aggregate byte limit"))?;
            let mut shard_tensors = HashMap::new();
            read_header(&file, &path, len, 0, catalog, &mut shard_tensors)?;
            for (name, tensor) in shard_tensors {
                if tensors.insert(name.clone(), (files.len(), tensor)).is_some() {
                    return Err(CheckpointError::InvalidHeader(format!(
                        "tensor {} appears in more than one shard",
                        name
                    )));
                }
            }
            files.push((path, file));
        }
        if let Some((name, _)) = catalog.iter().find(|(name, _)| !tensors.contains_key(name)) {
            return Err(CheckpointError::MissingTensor(name.clone()));
        }
        Ok(Self { files, tensors })
    }

    /// Shape of `name` as the checkpoint declares it.
    pub fn shape(&self, name: &str) -> Option<&[usize]> {
        self.tensors.get(name).map(|(_, t)| t.shape.as_slice())
    }

    /// Reads the raw little-endian bf16 bytes of `name` into `buf` (cleared first; its
    /// capacity is reused across calls). Returns the number of bf16 values.
    pub fn read_bf16_bytes(&self, name: &str, buf: &mut Vec<u8>) -> Result<usize, CheckpointError> {
        let (idx, tensor) = self
            .tensors
            .get(name)
            .ok_or_else(|| CheckpointError::MissingTensor(name.to_string()))?;
        let range = tensor.byte_range.clone();
        buf.clear();
        buf.try_reserve(range.len())
            .map_err(|e| CheckpointError::InvalidHeader(format!("tensor allocation: {e}")))?;
        buf.resize(range.len(), 0);
        let (path, file) = &self.files[*idx];
        file.read_exact_at(buf, range.start as u64)
            .map_err(|e| invalid_file(path, format!("failed to read tensor {name}: {e}")))?;
        Ok(range.len() / 2)
    }
}

/// Loads and validates a TinyLlama safetensors checkpoint from a filesystem path.
pub fn load_safetensors_checkpoint<P: AsRef<Path>>(
    path: P,
) -> Result<LoadedCheckpoint, CheckpointError> {
    load_paths(vec![path.as_ref().to_path_buf()], &tinyllama_catalog())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    struct TestDir(PathBuf);

    impl TestDir {
        fn new() -> Self {
            static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
            let path = std::env::temp_dir().join(format!(
                "aien-checkpoint-security-{}-{}",
                std::process::id(),
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn index(&self, name: &str) {
            let text = serde_json::json!({"weight_map": {"w": name}}).to_string();
            std::fs::write(self.0.join(SAFETENSORS_INDEX), text).unwrap();
        }
    }

    impl Drop for TestDir {
        fn drop(&mut self) {
            std::fs::remove_dir_all(&self.0).unwrap();
        }
    }

    fn security_catalog() -> Vec<(String, Vec<usize>)> {
        vec![("w".to_string(), vec![1])]
    }

    fn security_loaders_reject(path: &Path) {
        assert!(matches!(
            load_checkpoint_with_catalog(path, &security_catalog()),
            Err(CheckpointError::InvalidHeader(_))
        ));
        assert!(matches!(
            StreamedCheckpoint::open(path, &security_catalog()),
            Err(CheckpointError::InvalidHeader(_))
        ));
    }

    fn security_files_reject(path: &Path) {
        security_loaders_reject(path);
        assert!(matches!(
            checkpoint_files(path),
            Err(CheckpointError::InvalidHeader(_))
        ));
    }

    #[test]
    fn single_checkpoint_rejects_nonregular_input() {
        let dir = TestDir::new();
        let fifo = dir.0.join("model.safetensors");
        let name = std::ffi::CString::new({
            use std::os::unix::ffi::OsStrExt;
            fifo.as_os_str().as_bytes()
        }).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        security_files_reject(&fifo);
        std::fs::remove_file(&fifo).unwrap();
        std::os::unix::fs::symlink("/dev/zero", &fifo).unwrap();
        security_files_reject(&fifo);
    }

    #[test]
    fn shard_index_rejects_parent_absolute_and_symlink_escape() {
        let dir = TestDir::new();
        let outside = dir.0.join("outside.safetensors");
        let model = dir.0.join("model");
        std::fs::create_dir(&model).unwrap();
        std::fs::write(&outside, build_test_safetensors(&[("w", "BF16", &[1], &[0, 0])]))
            .unwrap();
        let index = model.join(SAFETENSORS_INDEX);

        std::fs::write(
            &index,
            r#"{"weight_map":{"w":"../outside.safetensors"}}"#,
        )
        .unwrap();
        security_files_reject(&index);
        std::fs::write(
            &index,
            serde_json::json!({"weight_map": {"w": outside.to_str().unwrap()}}).to_string(),
        )
        .unwrap();
        security_files_reject(&index);
        std::os::unix::fs::symlink(&outside, model.join("link.safetensors")).unwrap();
        std::fs::write(&index, r#"{"weight_map":{"w":"link.safetensors"}}"#).unwrap();
        security_files_reject(&index);
        assert!(matches!(
            checkpoint_sha256(&model.join("link.safetensors")),
            Err(CheckpointError::InvalidHeader(_))
        ));
    }

    #[test]
    fn shard_index_rejects_nonregular_and_oversized_inputs_before_reading() {
        let dir = TestDir::new();
        let index = dir.0.join(SAFETENSORS_INDEX);
        dir.index("directory.safetensors");
        std::fs::create_dir(dir.0.join("directory.safetensors")).unwrap();
        security_files_reject(&index);

        // A FIFO would block on open without O_NONBLOCK; never open an endless device.
        let fifo = dir.0.join("fifo.safetensors");
        let name = std::ffi::CString::new({
            use std::os::unix::ffi::OsStrExt;
            fifo.as_os_str().as_bytes()
        }).unwrap();
        assert_eq!(unsafe { libc::mkfifo(name.as_ptr(), 0o600) }, 0);
        dir.index("fifo.safetensors");
        security_files_reject(&index);

        let oversized_index = std::fs::OpenOptions::new().write(true).open(&index).unwrap();
        oversized_index.set_len(MAX_INDEX_BYTES + 1).unwrap();
        security_files_reject(&index);

        dir.index("large.safetensors");
        let shard = dir.0.join("large.safetensors");
        let large = std::fs::File::create(&shard).unwrap();
        large.set_len(MAX_SHARD_BYTES + 1).unwrap();
        security_files_reject(&index);
        let mut header = std::fs::File::create(&shard).unwrap();
        header.write_all(&(MAX_HEADER_BYTES + 1).to_le_bytes()).unwrap();
        security_loaders_reject(&index);
    }

    #[test]
    fn sharded_loader_validates_before_allocating_and_preserves_valid_payload() {
        let dir = TestDir::new();
        let index = dir.0.join(SAFETENSORS_INDEX);
        dir.index("a.safetensors");
        let file = dir.0.join("a.safetensors");
        let valid = build_test_safetensors(&[("w", "BF16", &[1], &[0, 0x3f])]);
        std::fs::write(&file, &valid).unwrap();
        use sha2::Digest;
        assert_eq!(
            checkpoint_sha256(&file).unwrap().as_slice(),
            sha2::Sha256::digest(&valid).as_slice()
        );
        assert_eq!(
            load_checkpoint_with_catalog(&index, &security_catalog())
                .unwrap()
                .tensor_bytes("w")
                .unwrap(),
            &[0, 0x3f]
        );
        let streamed = StreamedCheckpoint::open(&index, &security_catalog()).unwrap();
        let mut raw = Vec::new();
        streamed.read_bf16_bytes("w", &mut raw).unwrap();
        assert_eq!(raw, &[0, 0x3f]);

        // A sparse file advertises a large length, but an invalid required span must
        // be rejected from its header, before the whole-file loader allocates it.
        let invalid = br#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,999999]}}"#;
        let mut sparse = std::fs::File::create(&file).unwrap();
        sparse.write_all(&(invalid.len() as u64).to_le_bytes()).unwrap();
        sparse.write_all(invalid).unwrap();
        sparse.set_len(MAX_SHARD_BYTES).unwrap();
        assert!(matches!(
            load_checkpoint_with_catalog(&index, &security_catalog()),
            Err(CheckpointError::OffsetOutOfBounds { .. })
        ));
        // Even a plausible required span cannot authorize a giant unaccounted
        // payload when this catalog expects only one BF16 element.
        std::fs::write(&file, &valid).unwrap();
        std::fs::OpenOptions::new()
            .write(true)
            .open(&file)
            .unwrap()
            .set_len(MAX_SHARD_BYTES)
            .unwrap();
        assert!(matches!(
            load_checkpoint_with_catalog(&index, &security_catalog()),
            Err(CheckpointError::InvalidHeader(_))
        ));
    }

    #[test]
    fn sharded_loader_rejects_aggregate_sparse_length_before_payload_read() {
        let dir = TestDir::new();
        let mut map = serde_json::Map::new();
        for n in 0..=(MAX_CHECKPOINT_BYTES / MAX_SHARD_BYTES) {
            let name = format!("{n}.safetensors");
            let path = dir.0.join(&name);
            let mut file = std::fs::File::create(&path).unwrap();
            let header = if n == 0 {
                br#"{"w":{"dtype":"BF16","shape":[1],"data_offsets":[0,2]}}"#.as_slice()
            } else {
                b"{}".as_slice()
            };
            file.write_all(&(header.len() as u64).to_le_bytes()).unwrap();
            file.write_all(header).unwrap();
            file.set_len(MAX_SHARD_BYTES).unwrap();
            map.insert(format!("tensor{n}"), serde_json::Value::String(name));
        }
        std::fs::write(
            dir.0.join(SAFETENSORS_INDEX),
            serde_json::json!({"weight_map": map}).to_string(),
        ).unwrap();
        let index = dir.0.join(SAFETENSORS_INDEX);
        security_files_reject(&index);
    }

    /// Constructs a minimal valid safetensors binary buffer for testing.
    fn build_test_safetensors(tensors: &[(&str, &str, &[usize], &[u8])]) -> Vec<u8> {
        let mut header_map = serde_json::Map::new();
        let mut data_payload = Vec::new();

        for (name, dtype, shape, payload) in tensors {
            let start = data_payload.len();
            data_payload.extend_from_slice(payload);
            let end = data_payload.len();

            let mut tensor_info = serde_json::Map::new();
            tensor_info.insert(
                "dtype".to_string(),
                serde_json::Value::String(dtype.to_string()),
            );
            let shape_val = shape
                .iter()
                .map(|&d| serde_json::Value::Number(serde_json::Number::from(d)))
                .collect();
            tensor_info.insert("shape".to_string(), serde_json::Value::Array(shape_val));
            let offsets_val = vec![
                serde_json::Value::Number(serde_json::Number::from(start)),
                serde_json::Value::Number(serde_json::Number::from(end)),
            ];
            tensor_info.insert(
                "data_offsets".to_string(),
                serde_json::Value::Array(offsets_val),
            );

            header_map.insert(name.to_string(), serde_json::Value::Object(tensor_info));
        }

        let header_json = serde_json::Value::Object(header_map).to_string();
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut buffer = Vec::new();
        buffer.extend_from_slice(&header_len.to_le_bytes());
        buffer.extend_from_slice(header_bytes);
        buffer.extend_from_slice(&data_payload);
        buffer
    }

    #[test]
    fn test_tinyllama_catalog_size_and_shapes() {
        let catalog = tinyllama_catalog();
        assert_eq!(catalog.len(), 201);

        assert_eq!(catalog[0].0, "model.embed_tokens.weight");
        assert_eq!(catalog[0].1, vec![32000, 2048]);

        assert_eq!(catalog[200].0, "lm_head.weight");
        assert_eq!(catalog[200].1, vec![32000, 2048]);

        assert_eq!(catalog[199].0, "model.norm.weight");
        assert_eq!(catalog[199].1, vec![2048]);
    }

    #[test]
    fn test_bf16_fp32_codec_roundtrip() {
        let floats = vec![0.0f32, 1.0f32, -1.0f32, 0.5f32, 64.0f32];
        let bf16_bytes = encode_fp32_to_bf16(&floats);
        let decoded = decode_bf16_to_fp32(&bf16_bytes);

        assert_eq!(decoded.len(), floats.len());
        assert_eq!(decoded[0], 0.0);
        assert_eq!(decoded[1], 1.0);
        assert_eq!(decoded[2], -1.0);
        assert_eq!(decoded[3], 0.5);
        assert_eq!(decoded[4], 64.0);
    }

    #[test]
    fn test_valid_custom_catalog_parsing() {
        let floats = vec![1.5f32, 2.5f32, -0.5f32, 4.0f32];
        let bf16_bytes = encode_fp32_to_bf16(&floats);
        let buffer = build_test_safetensors(&[("test.weight", "BF16", &[2, 2], &bf16_bytes)]);

        let catalog = vec![("test.weight".to_string(), vec![2, 2])];
        let loaded = parse_safetensors_with_catalog(&buffer, &catalog).unwrap();

        assert_eq!(loaded.tensor_count(), 1);
        assert_eq!(loaded.capsule.tensors["test.weight"].shape, vec![2, 2]);
        assert_eq!(loaded.get_shape("test.weight").unwrap(), &vec![2, 2]);
        let fp32 = loaded.decode_fp32("test.weight").unwrap();
        assert_eq!(fp32.len(), 4);
        assert_eq!(fp32[0], 1.5);
        assert_eq!(fp32[1], 2.5);
        assert_eq!(loaded.tensor_bytes("test.weight").unwrap(), &bf16_bytes[..]);
        assert!(matches!(
            loaded.capsule.tensors["test.weight"].dtype,
            crate::capsule::WeightDType::Bf16
        ));
        assert!(loaded.capsule.tensors["test.weight"].quant.is_none());
    }

    #[test]
    fn test_capsule_owns_file_bytes_and_byte_range_points_at_payload() {
        let floats = vec![1.0f32, -1.0f32, 0.5f32];
        let bf16_bytes = encode_fp32_to_bf16(&floats);
        let file_bytes = build_test_safetensors(&[("w", "BF16", &[3], &bf16_bytes)]);
        let catalog = vec![("w".to_string(), vec![3])];
        let loaded = parse_safetensors_with_catalog(&file_bytes, &catalog).unwrap();

        assert_eq!(loaded.capsule.owned_payload_len(), file_bytes.len());
        assert_eq!(loaded.capsule.bytes.as_ref(), file_bytes.as_slice());
        assert_eq!(
            loaded.capsule.source_digest,
            crate::capsule::source_digest(&file_bytes)
        );

        let range = loaded.capsule.tensors["w"].byte_range.clone();
        assert!(range.start >= 8);
        assert_eq!(range.end - range.start, bf16_bytes.len());
        assert_eq!(&loaded.capsule.bytes[range], bf16_bytes.as_slice());
        assert_eq!(loaded.tensor_bytes("w").unwrap(), bf16_bytes.as_slice());
        let decoded = loaded.decode_fp32("w").unwrap();
        assert_eq!(decoded, decode_bf16_to_fp32(&bf16_bytes));
    }

    #[test]
    fn test_from_bf16_tensors_one_buffer() {
        let a = encode_fp32_to_bf16(&[1.0, 2.0]);
        let b = encode_fp32_to_bf16(&[0.5]);
        let loaded = LoadedCheckpoint::from_bf16_tensors(vec![
            ("a".to_string(), vec![2], a.clone()),
            ("b".to_string(), vec![1], b.clone()),
        ]);

        assert_eq!(loaded.capsule.owned_payload_len(), a.len() + b.len());
        assert_eq!(loaded.tensor_bytes("a").unwrap(), a.as_slice());
        assert_eq!(loaded.tensor_bytes("b").unwrap(), b.as_slice());
        assert_eq!(loaded.decode_fp32("a").unwrap()[0], 1.0);
        assert_eq!(loaded.decode_fp32("b").unwrap()[0], 0.5);
        assert_eq!(
            loaded.capsule.tensors["a"].logical_strides,
            crate::capsule::row_major_strides(&[2])
        );
        assert!(loaded.capsule.tensors["b"].quant.is_none());
        assert_eq!(
            loaded.capsule.source_digest,
            crate::capsule::source_digest(&loaded.capsule.bytes)
        );
        assert_eq!(
            loaded.capsule.capsule_digest,
            crate::capsule::capsule_digest(&loaded.capsule.bytes, &loaded.capsule.tensors)
        );

        let empty = LoadedCheckpoint::new();
        assert_eq!(empty.tensor_count(), 0);
        assert_eq!(empty.capsule.owned_payload_len(), 0);
        assert!(empty.capsule.tensors.is_empty());
        assert_eq!(
            empty.capsule.source_digest,
            crate::capsule::source_digest(&[])
        );
    }

    #[test]
    fn test_missing_tensor_loud_error() {
        let floats = vec![1.0f32, 2.0f32];
        let bf16_bytes = encode_fp32_to_bf16(&floats);
        let buffer = build_test_safetensors(&[("present.weight", "BF16", &[2], &bf16_bytes)]);

        let catalog = vec![
            ("present.weight".to_string(), vec![2]),
            ("missing.weight".to_string(), vec![2]),
        ];

        let err = parse_safetensors_with_catalog(&buffer, &catalog).unwrap_err();
        assert_eq!(
            err,
            CheckpointError::MissingTensor("missing.weight".to_string())
        );
    }

    #[test]
    fn test_shape_mismatch_loud_error() {
        let floats = vec![1.0f32, 2.0f32];
        let bf16_bytes = encode_fp32_to_bf16(&floats);
        let buffer = build_test_safetensors(&[("layer.weight", "BF16", &[2], &bf16_bytes)]);

        let catalog = vec![("layer.weight".to_string(), vec![4])];

        let err = parse_safetensors_with_catalog(&buffer, &catalog).unwrap_err();
        match err {
            CheckpointError::ShapeMismatch {
                tensor,
                expected,
                actual,
            } => {
                assert_eq!(tensor, "layer.weight");
                assert_eq!(expected, vec![4]);
                assert_eq!(actual, vec![2]);
            }
            _ => panic!("Expected ShapeMismatch"),
        }
    }

    #[test]
    fn test_dtype_mismatch_loud_error() {
        let raw_f32_bytes = vec![0u8; 16]; // 4 floats
        let buffer = build_test_safetensors(&[("layer.weight", "F32", &[4], &raw_f32_bytes)]);

        let catalog = vec![("layer.weight".to_string(), vec![4])];

        let err = parse_safetensors_with_catalog(&buffer, &catalog).unwrap_err();
        match err {
            CheckpointError::DtypeMismatch {
                tensor,
                expected,
                actual,
            } => {
                assert_eq!(tensor, "layer.weight");
                assert_eq!(expected, "BF16");
                assert_eq!(actual, "F32");
            }
            _ => panic!("Expected DtypeMismatch"),
        }
    }

    #[test]
    fn test_offset_out_of_bounds_loud_error() {
        // Create corrupted header where end offset exceeds payload
        let header_json = r#"{"weight":{"dtype":"BF16","shape":[4],"data_offsets":[0,100]}}"#;
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut buffer = Vec::new();
        buffer.extend_from_slice(&header_len.to_le_bytes());
        buffer.extend_from_slice(header_bytes);
        buffer.extend_from_slice(&[0u8; 8]); // only 8 bytes payload, offset requires 100

        let catalog = vec![("weight".to_string(), vec![4])];
        let err = parse_safetensors_with_catalog(&buffer, &catalog).unwrap_err();
        match err {
            CheckpointError::OffsetOutOfBounds {
                tensor,
                offset,
                buffer_len,
            } => {
                assert_eq!(tensor, "weight");
                assert_eq!(offset, 100);
                assert_eq!(buffer_len, 8);
            }
            _ => panic!("Expected OffsetOutOfBounds"),
        }
    }

    #[test]
    fn test_invalid_header_truncated_buffer() {
        let buffer = vec![1, 2, 3];
        let catalog = vec![("weight".to_string(), vec![2])];
        let err = parse_safetensors_with_catalog(&buffer, &catalog).unwrap_err();
        match err {
            CheckpointError::InvalidHeader(_) => {}
            _ => panic!("Expected InvalidHeader"),
        }
    }

    #[test]
    fn test_header_len_arithmetic_overflow() {
        let mut buffer = Vec::new();
        buffer.extend_from_slice(&u64::MAX.to_le_bytes());
        buffer.extend_from_slice(b"payload");
        let catalog = vec![("weight".to_string(), vec![2])];
        let err = parse_safetensors_with_catalog(&buffer, &catalog).unwrap_err();
        match err {
            CheckpointError::InvalidHeader(_) => {}
            _ => panic!("Expected InvalidHeader on arithmetic overflow"),
        }
    }

    /// sc#338: the files the loader reads, named for the daemon's digest: the index and
    /// every shard it names (sorted, de-duplicated, by the index's name); a single file
    /// has no index.
    #[test]
    fn checkpoint_files_names_the_index_and_every_shard() {
        let dir = std::env::temp_dir().join(format!("aien-ckfiles-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let index = dir.join(SAFETENSORS_INDEX);
        std::fs::write(
            &index,
            r#"{"weight_map":{"x":"b.safetensors","y":"a.safetensors","z":"b.safetensors"}}"#,
        )
        .unwrap();
        std::fs::write(dir.join("a.safetensors"), b"").unwrap();
        std::fs::write(dir.join("b.safetensors"), b"").unwrap();
        let files = checkpoint_files(&index).unwrap();
        assert_eq!(files.index.as_deref(), Some(index.as_path()));
        assert_eq!(
            files.weights,
            vec![
                ("a.safetensors".to_string(), dir.join("a.safetensors")),
                ("b.safetensors".to_string(), dir.join("b.safetensors")),
            ]
        );
        // A shard file beside an index names the whole set, like the loader.
        assert_eq!(checkpoint_files(dir.join("a.safetensors")).unwrap(), files);
        let single = dir.join("model.safetensors");
        std::fs::write(&single, b"").unwrap();
        let files = checkpoint_files(&single).unwrap();
        assert_eq!(files.index, None);
        assert_eq!(
            files.weights,
            vec![("model.safetensors".to_string(), single.clone())]
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
