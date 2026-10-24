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

//! C-compatible typed tensor boundary for capsule storage and accelerator residency.

use crate::capsule::{ModelCapsule, TensorLayout, WeightDType};
use std::ffi::c_void;
use std::marker::PhantomData;

pub const MAX_TENSOR_RANK: usize = 4;

/// Numeric values are part of the C ABI of this boundary (`#[repr]` below); keep them stable.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiDType {
    F32 = 1,
    Bf16 = 2,
    Fp16 = 3,
    Fp8E4M3Fn = 4,
    Fp8E5M2 = 5,
    Int8 = 6,
    Int4 = 7,
}

impl From<WeightDType> for AbiDType {
    fn from(dtype: WeightDType) -> Self {
        match dtype {
            WeightDType::F32 => Self::F32,
            WeightDType::Bf16 => Self::Bf16,
            WeightDType::Fp16 => Self::Fp16,
            WeightDType::Fp8E4M3Fn => Self::Fp8E4M3Fn,
            WeightDType::Fp8E5M2 => Self::Fp8E5M2,
            WeightDType::Int8 => Self::Int8,
            WeightDType::Int4 => Self::Int4,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiLayout {
    RowMajor = 1,
    ColumnMajor = 2,
    Strided = 3,
}

impl From<TensorLayout> for AbiLayout {
    fn from(layout: TensorLayout) -> Self {
        match layout {
            TensorLayout::RowMajor => Self::RowMajor,
            TensorLayout::ColumnMajor => Self::ColumnMajor,
            TensorLayout::Strided => Self::Strided,
        }
    }
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiResidency {
    Host = 1,
    Device = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiQuantScheme {
    None = 0,
    Symmetric = 1,
    Affine = 2,
}

#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AbiPacking {
    None = 0,
    Int4LowNibbleFirstSigned = 1,
    Int4HighNibbleFirstSigned = 2,
    Int4LowNibbleFirstUnsigned = 3,
    Int4HighNibbleFirstUnsigned = 4,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AbiQuantization {
    pub scheme: AbiQuantScheme,
    pub axis: i32,
    pub group_size: u32,
    pub packing: AbiPacking,
    pub scales: *const f32,
    pub scale_count: u64,
    pub zero_points: *const i32,
    pub zero_point_count: u64,
}

impl Default for AbiQuantization {
    fn default() -> Self {
        Self {
            scheme: AbiQuantScheme::None,
            axis: -1,
            group_size: 0,
            packing: AbiPacking::None,
            scales: std::ptr::null(),
            scale_count: 0,
            zero_points: std::ptr::null(),
            zero_point_count: 0,
        }
    }
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorDescriptor {
    pub dtype: AbiDType,
    pub layout: AbiLayout,
    pub rank: u32,
    pub residency: AbiResidency,
    pub shape: [u64; MAX_TENSOR_RANK],
    pub strides: [u64; MAX_TENSOR_RANK],
    pub byte_len: u64,
    pub quant: AbiQuantization,
}

#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct TensorView {
    pub address: *const c_void,
    pub descriptor: TensorDescriptor,
}

/// The device allocation owns `address`, `quant.scales`, and `quant.zero_points`.
/// Whoever allocates it on a device must release all three.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ResidentTensor {
    pub address: *mut c_void,
    pub descriptor: TensorDescriptor,
}

/// Keeps the host bytes and quantization arrays alive during an FFI upload.
pub struct BorrowedTensorView<'a> {
    view: TensorView,
    _source: PhantomData<&'a [u8]>,
}

impl BorrowedTensorView<'_> {
    pub fn as_abi(&self) -> &TensorView {
        &self.view
    }
}

impl TensorView {
    pub fn f32_contiguous<'a>(
        data: &'a [f32],
        shape: &[usize],
    ) -> Result<BorrowedTensorView<'a>, String> {
        let byte_len = data
            .len()
            .checked_mul(4)
            .ok_or("F32 byte length overflow")?;
        let descriptor = descriptor(
            AbiDType::F32,
            AbiLayout::RowMajor,
            shape,
            &row_major_strides(shape)?,
            byte_len,
            AbiQuantization::default(),
        )?;
        Ok(BorrowedTensorView {
            view: Self {
                address: data.as_ptr().cast(),
                descriptor,
            },
            _source: PhantomData,
        })
    }

    pub fn from_capsule<'a>(
        capsule: &'a ModelCapsule,
        name: &str,
    ) -> Result<BorrowedTensorView<'a>, String> {
        let tensor = capsule.tensor(name).map_err(|err| err.to_string())?;
        let bytes = capsule.tensor_bytes(name).map_err(|err| err.to_string())?;
        let quant = match &tensor.quant {
            None => AbiQuantization::default(),
            Some(quant) => AbiQuantization {
                scheme: match quant.scheme.as_str() {
                    "symmetric" => AbiQuantScheme::Symmetric,
                    "affine" => AbiQuantScheme::Affine,
                    other => return Err(format!("unsupported quantization scheme {other}")),
                },
                axis: i32::try_from(quant.axis).map_err(|_| "quantization axis overflow")?,
                group_size: u32::try_from(quant.group_size).map_err(|_| "group size overflow")?,
                packing: match quant.packing.as_str() {
                    "none" => AbiPacking::None,
                    "int4-low-signed" => AbiPacking::Int4LowNibbleFirstSigned,
                    "int4-high-signed" => AbiPacking::Int4HighNibbleFirstSigned,
                    "int4-low-unsigned" => AbiPacking::Int4LowNibbleFirstUnsigned,
                    "int4-high-unsigned" => AbiPacking::Int4HighNibbleFirstUnsigned,
                    other => return Err(format!("unsupported packing {other}")),
                },
                scales: quant.scales.as_ptr(),
                scale_count: quant.scales.len() as u64,
                zero_points: quant.zero_points.as_ptr(),
                zero_point_count: quant.zero_points.len() as u64,
            },
        };
        let descriptor = descriptor(
            tensor.dtype.into(),
            tensor.layout.into(),
            &tensor.shape,
            &tensor.logical_strides,
            bytes.len(),
            quant,
        )?;
        Ok(BorrowedTensorView {
            view: Self {
                address: bytes.as_ptr().cast(),
                descriptor,
            },
            _source: PhantomData,
        })
    }
}

fn descriptor(
    dtype: AbiDType,
    layout: AbiLayout,
    shape: &[usize],
    strides: &[usize],
    byte_len: usize,
    quant: AbiQuantization,
) -> Result<TensorDescriptor, String> {
    if shape.len() > MAX_TENSOR_RANK || shape.len() != strides.len() {
        return Err("tensor rank or stride count invalid".to_string());
    }
    if shape.is_empty() || shape.contains(&0) {
        return Err("tensor dimensions must be nonzero".to_string());
    }
    let elements = shape
        .iter()
        .try_fold(1usize, |count, dim| count.checked_mul(*dim))
        .ok_or("tensor element count overflow")?;
    let expected = match dtype {
        AbiDType::F32 => elements.checked_mul(4),
        AbiDType::Bf16 | AbiDType::Fp16 => elements.checked_mul(2),
        AbiDType::Fp8E4M3Fn | AbiDType::Fp8E5M2 | AbiDType::Int8 => Some(elements),
        AbiDType::Int4 => elements.checked_add(1).map(|count| count / 2),
    }
    .ok_or("tensor byte length overflow")?;
    if layout == AbiLayout::RowMajor && strides != row_major_strides(shape)? {
        return Err("row-major strides do not match shape".to_string());
    }
    if layout == AbiLayout::RowMajor && byte_len != expected {
        return Err(format!(
            "tensor byte length {byte_len} does not match {expected}"
        ));
    }
    if dtype == AbiDType::Int4 && quant.packing == AbiPacking::None {
        return Err("INT4 requires explicit nibble packing".to_string());
    }
    if quant.scheme != AbiQuantScheme::None && quant.scale_count == 0 {
        return Err("quantized tensor needs scales".to_string());
    }
    if quant.scheme == AbiQuantScheme::None
        && (quant.scale_count != 0 || quant.zero_point_count != 0)
    {
        return Err("unquantized tensor has quantization arrays".to_string());
    }
    if quant.scale_count > 0 && quant.scales.is_null() {
        return Err("quantization scales missing".to_string());
    }
    if quant.zero_point_count > 0 && quant.zero_points.is_null() {
        return Err("quantization zero points missing".to_string());
    }
    let mut abi_shape = [0; MAX_TENSOR_RANK];
    let mut abi_strides = [0; MAX_TENSOR_RANK];
    for (index, value) in shape.iter().enumerate() {
        abi_shape[index] = *value as u64;
        abi_strides[index] = strides[index] as u64;
    }
    Ok(TensorDescriptor {
        dtype,
        layout,
        rank: shape.len() as u32,
        residency: AbiResidency::Host,
        shape: abi_shape,
        strides: abi_strides,
        byte_len: byte_len as u64,
        quant,
    })
}

fn row_major_strides(shape: &[usize]) -> Result<Vec<usize>, String> {
    let mut strides = vec![1usize; shape.len()];
    for index in (0..shape.len().saturating_sub(1)).rev() {
        strides[index] = strides[index + 1]
            .checked_mul(shape[index + 1])
            .ok_or("tensor stride overflow")?;
    }
    Ok(strides)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capsule::{capsule_digest, source_digest, CapsuleTensor, QuantizationDescriptor};
    use std::collections::HashMap;
    use std::sync::Arc;

    #[test]
    fn f32_view_records_shape_stride_and_host_residency() {
        let data = [1.0f32; 6];
        let view = TensorView::f32_contiguous(&data, &[2, 3]).unwrap();
        let descriptor = view.as_abi().descriptor;
        assert_eq!(descriptor.dtype, AbiDType::F32);
        assert_eq!(descriptor.residency, AbiResidency::Host);
        assert_eq!(descriptor.shape[..2], [2, 3]);
        assert_eq!(descriptor.strides[..2], [3, 1]);
        assert_eq!(descriptor.byte_len, 24);
        assert!(TensorView::f32_contiguous(&data, &[3, 3]).is_err());
    }

    #[test]
    fn capsule_views_preserve_typed_payload_and_quantization() {
        let cases = [
            (WeightDType::Bf16, AbiDType::Bf16, vec![0_u8; 8], None),
            (WeightDType::Fp16, AbiDType::Fp16, vec![0_u8; 8], None),
            (
                WeightDType::Fp8E4M3Fn,
                AbiDType::Fp8E4M3Fn,
                vec![0_u8; 4],
                None,
            ),
            (WeightDType::Fp8E5M2, AbiDType::Fp8E5M2, vec![0_u8; 4], None),
            (
                WeightDType::Int4,
                AbiDType::Int4,
                vec![0_u8; 2],
                Some(QuantizationDescriptor {
                    scheme: "symmetric".to_string(),
                    axis: 0,
                    group_size: 4,
                    scales: vec![0.5],
                    zero_points: vec![],
                    packing: "int4-low-signed".to_string(),
                }),
            ),
        ];
        for (dtype, expected_dtype, bytes, quant) in cases {
            let mut tensors = HashMap::new();
            tensors.insert(
                "w".to_string(),
                CapsuleTensor {
                    dtype,
                    layout: TensorLayout::RowMajor,
                    shape: vec![2, 2],
                    logical_strides: vec![2, 1],
                    byte_range: 0..bytes.len(),
                    quant,
                },
            );
            let capsule = ModelCapsule {
                source_digest: source_digest(&bytes),
                capsule_digest: capsule_digest(&bytes, &tensors),
                bytes: Arc::from(bytes),
                tensors,
            };
            let view = TensorView::from_capsule(&capsule, "w").unwrap();
            assert_eq!(view.as_abi().descriptor.dtype, expected_dtype);
            assert_eq!(
                view.as_abi().descriptor.byte_len as usize,
                capsule.bytes.len()
            );
            assert_eq!(view.as_abi().address as *const u8, capsule.bytes.as_ptr());
            if dtype == WeightDType::Int4 {
                assert_eq!(view.as_abi().descriptor.quant.scale_count, 1);
                assert_eq!(
                    view.as_abi().descriptor.quant.packing,
                    AbiPacking::Int4LowNibbleFirstSigned
                );
            }
        }
    }

    #[test]
    fn c_abi_layout_is_stable() {
        assert_eq!(std::mem::size_of::<AbiQuantization>(), 48);
        assert_eq!(std::mem::size_of::<TensorDescriptor>(), 136);
        assert_eq!(std::mem::size_of::<TensorView>(), 144);
        assert_eq!(std::mem::size_of::<ResidentTensor>(), 144);
    }
}
