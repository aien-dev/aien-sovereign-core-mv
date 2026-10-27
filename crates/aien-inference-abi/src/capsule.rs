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

//! Weight capsule: raw payload bytes plus a tensor index and canonical digests.

use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use crate::checkpoint::decode_bf16_to_fp32;

/// Element type stored in a capsule tensor payload.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum WeightDType {
    F32,
    Bf16,
    Fp16,
    Fp8E4M3Fn,
    Fp8E5M2,
    Int8,
    Int4,
}

impl WeightDType {
    fn label(self) -> &'static str {
        match self {
            WeightDType::F32 => "f32",
            WeightDType::Bf16 => "bf16",
            WeightDType::Fp16 => "fp16",
            WeightDType::Fp8E4M3Fn => "fp8e4m3fn",
            WeightDType::Fp8E5M2 => "fp8e5m2",
            WeightDType::Int8 => "int8",
            WeightDType::Int4 => "int4",
        }
    }
}

/// Logical tensor layout, independent of packed byte encoding.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorLayout {
    RowMajor,
    ColumnMajor,
    Strided,
}

impl TensorLayout {
    fn label(self) -> &'static str {
        match self {
            Self::RowMajor => "row-major",
            Self::ColumnMajor => "column-major",
            Self::Strided => "strided",
        }
    }
}

/// Quantization metadata. IEEE dtypes use packing `"none"`.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizationDescriptor {
    pub scheme: String,
    pub axis: usize,
    pub group_size: usize,
    pub scales: Vec<f32>,
    pub zero_points: Vec<i32>,
    pub packing: String, // "none" for IEEE dtypes; INT4 uses an explicit nibble order and sign
}

/// One named tensor view into [`ModelCapsule::bytes`].
#[derive(Debug, Clone, PartialEq)]
pub struct CapsuleTensor {
    pub dtype: WeightDType,
    pub layout: TensorLayout,
    pub shape: Vec<usize>,
    pub logical_strides: Vec<usize>, // element strides for the selected layout
    pub byte_range: Range<usize>,    // into ModelCapsule::bytes, the whole file
    pub quant: Option<QuantizationDescriptor>,
}

/// Immutable model payload plus its tensor index.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCapsule {
    pub bytes: Arc<[u8]>,
    pub source_digest: String, // lowercase hex sha256 of bytes
    pub tensors: HashMap<String, CapsuleTensor>,
    pub capsule_digest: String,
}

/// Failures when reading a tensor out of a capsule.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CapsuleError {
    MissingTensor(String),
    UnsupportedDtype(String),
    BadRange(String),
}

impl fmt::Display for CapsuleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CapsuleError::MissingTensor(name) => write!(f, "missing tensor {name}"),
            CapsuleError::UnsupportedDtype(dtype) => write!(f, "unsupported dtype {dtype}"),
            CapsuleError::BadRange(detail) => write!(f, "bad range {detail}"),
        }
    }
}

impl std::error::Error for CapsuleError {}

impl ModelCapsule {
    pub fn tensor(&self, name: &str) -> Result<&CapsuleTensor, CapsuleError> {
        self.tensors
            .get(name)
            .ok_or_else(|| CapsuleError::MissingTensor(name.to_string()))
    }

    pub fn tensor_bytes(&self, name: &str) -> Result<&[u8], CapsuleError> {
        let tensor = self.tensor(name)?;
        let range = &tensor.byte_range;
        if range.start > range.end || range.end > self.bytes.len() {
            return Err(CapsuleError::BadRange(format!(
                "{name} range {}-{} outside payload length {}",
                range.start,
                range.end,
                self.bytes.len()
            )));
        }
        Ok(&self.bytes[range.start..range.end])
    }

    /// Allocates. BF16 uses the existing decode_bf16_to_fp32 from checkpoint.rs (crate::checkpoint::decode_bf16_to_fp32). F32 interprets little-endian f32. Other dtypes return UnsupportedDtype. Do not store a decoded copy on the struct.
    pub fn decode_fp32(&self, name: &str) -> Result<Vec<f32>, CapsuleError> {
        let tensor = self.tensor(name)?;
        let raw = self.tensor_bytes(name)?;
        match tensor.dtype {
            WeightDType::F32 => {
                if raw.len() % 4 != 0 {
                    return Err(CapsuleError::BadRange(format!(
                        "{name} f32 payload length {} is not a multiple of 4",
                        raw.len()
                    )));
                }
                let mut out = Vec::with_capacity(raw.len() / 4);
                for chunk in raw.chunks_exact(4) {
                    let bits = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                    out.push(f32::from_bits(bits));
                }
                Ok(out)
            }
            WeightDType::Bf16 => Ok(decode_bf16_to_fp32(raw)),
            other => Err(CapsuleError::UnsupportedDtype(other.label().to_string())),
        }
    }

    pub fn owned_payload_len(&self) -> usize {
        self.bytes.len()
    }
}

fn sha256_hex(parts: &[&[u8]]) -> String {
    let mut hasher = Sha256::new();
    for part in parts {
        hasher.update(part);
    }
    let digest = hasher.finalize();
    let mut hex = String::with_capacity(digest.len() * 2);
    for byte in digest {
        hex.push_str(&format!("{byte:02x}"));
    }
    hex
}

/// Lowercase hex SHA-256 of `bytes`.
pub fn source_digest(bytes: &[u8]) -> String {
    sha256_hex(&[bytes])
}

/// Canonical capsule digest over the stored bytes and every tensor descriptor.
/// The length-prefixed binary manifest is sorted by tensor name and versioned.
pub fn capsule_digest(bytes: &[u8], tensors: &HashMap<String, CapsuleTensor>) -> String {
    let mut names: Vec<&String> = tensors.keys().collect();
    names.sort();
    let mut manifest = Vec::new();
    manifest.extend_from_slice(b"AIEN-CAPSULE-V1\0");
    push_len(&mut manifest, names.len());
    for name in names {
        let tensor = &tensors[name];
        push_bytes(&mut manifest, name.as_bytes());
        push_bytes(&mut manifest, tensor.dtype.label().as_bytes());
        push_bytes(&mut manifest, tensor.layout.label().as_bytes());
        push_len(&mut manifest, tensor.shape.len());
        for dim in &tensor.shape {
            push_len(&mut manifest, *dim);
        }
        push_len(&mut manifest, tensor.logical_strides.len());
        for stride in &tensor.logical_strides {
            push_len(&mut manifest, *stride);
        }
        push_len(&mut manifest, tensor.byte_range.start);
        push_len(&mut manifest, tensor.byte_range.end);
        match &tensor.quant {
            None => manifest.push(0),
            Some(quant) => {
                manifest.push(1);
                push_bytes(&mut manifest, quant.scheme.as_bytes());
                push_len(&mut manifest, quant.axis);
                push_len(&mut manifest, quant.group_size);
                push_len(&mut manifest, quant.scales.len());
                for scale in &quant.scales {
                    manifest.extend_from_slice(&scale.to_bits().to_le_bytes());
                }
                push_len(&mut manifest, quant.zero_points.len());
                for zero_point in &quant.zero_points {
                    manifest.extend_from_slice(&zero_point.to_le_bytes());
                }
                push_bytes(&mut manifest, quant.packing.as_bytes());
            }
        }
    }
    sha256_hex(&[&manifest, bytes])
}

fn push_len(buf: &mut Vec<u8>, value: usize) {
    buf.extend_from_slice(&(value as u64).to_le_bytes());
}

fn push_bytes(buf: &mut Vec<u8>, value: &[u8]) {
    push_len(buf, value.len());
    buf.extend_from_slice(value);
}

/// Element strides for row-major (C) layout. The last axis has stride 1.
pub fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let rank = shape.len();
    let mut strides = vec![0; rank];
    if rank == 0 {
        return strides;
    }
    strides[rank - 1] = 1;
    for index in (0..rank - 1).rev() {
        strides[index] = strides[index + 1] * shape[index + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_digest_of_abc_matches_well_known_sha256() {
        let digest = source_digest(b"abc");
        assert_eq!(
            digest,
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }

    fn sample_tensor(
        dtype: WeightDType,
        shape: Vec<usize>,
        byte_range: Range<usize>,
    ) -> CapsuleTensor {
        let logical_strides = row_major_strides(&shape);
        CapsuleTensor {
            dtype,
            layout: TensorLayout::RowMajor,
            shape,
            logical_strides,
            byte_range,
            quant: None,
        }
    }

    #[test]
    fn capsule_digest_is_stable_across_hashmap_insertion_order() {
        let bytes = b"raw-payload-bytes";
        let tensor_a = sample_tensor(WeightDType::Bf16, vec![2], 0..2);
        let tensor_b = sample_tensor(WeightDType::F32, vec![1, 1], 2..6);

        let mut first = HashMap::new();
        first.insert("zeta".to_string(), tensor_a.clone());
        first.insert("alpha".to_string(), tensor_b.clone());

        let mut second = HashMap::new();
        second.insert("alpha".to_string(), tensor_b.clone());
        second.insert("zeta".to_string(), tensor_a.clone());

        let digest_first = capsule_digest(bytes, &first);
        let digest_second = capsule_digest(bytes, &second);
        assert_eq!(digest_first, digest_second);

        let mut changed = second.clone();
        changed.get_mut("alpha").unwrap().logical_strides = vec![2, 1];
        assert_ne!(digest_first, capsule_digest(bytes, &changed));

        let mut changed = second;
        changed.get_mut("zeta").unwrap().quant = Some(QuantizationDescriptor {
            scheme: "test".to_string(),
            axis: 0,
            group_size: 2,
            scales: vec![1.0],
            zero_points: vec![0],
            packing: "none".to_string(),
        });
        assert_ne!(digest_first, capsule_digest(bytes, &changed));
    }

    #[test]
    fn decode_fp32_bf16_one_and_owned_len_and_missing_tensor() {
        let raw = [0x80u8, 0x3f];
        let bytes: Arc<[u8]> = Arc::from(raw.as_slice());
        let mut tensors = HashMap::new();
        tensors.insert(
            "w".to_string(),
            sample_tensor(WeightDType::Bf16, vec![1], 0..2),
        );
        let capsule = ModelCapsule {
            source_digest: source_digest(&bytes),
            capsule_digest: capsule_digest(&bytes, &tensors),
            bytes: Arc::clone(&bytes),
            tensors,
        };

        let decoded = capsule.decode_fp32("w").unwrap();
        assert_eq!(decoded.len(), 1);
        assert!((decoded[0] - 1.0).abs() < 1e-6);
        assert_eq!(capsule.owned_payload_len(), capsule.bytes.len());
        assert_eq!(capsule.owned_payload_len(), 2);

        match capsule.tensor("missing") {
            Err(CapsuleError::MissingTensor(name)) => assert_eq!(name, "missing"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
        match capsule.decode_fp32("missing") {
            Err(CapsuleError::MissingTensor(name)) => assert_eq!(name, "missing"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
        match capsule.tensor_bytes("missing") {
            Err(CapsuleError::MissingTensor(name)) => assert_eq!(name, "missing"),
            other => panic!("expected MissingTensor, got {other:?}"),
        }
    }

    #[test]
    fn row_major_strides_match_c_order() {
        assert_eq!(row_major_strides(&[]), Vec::<usize>::new());
        assert_eq!(row_major_strides(&[4]), vec![1]);
        assert_eq!(row_major_strides(&[2, 3, 4]), vec![12, 4, 1]);
    }
}
