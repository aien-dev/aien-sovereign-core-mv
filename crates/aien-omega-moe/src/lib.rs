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

//! Safe Rust wrapper over omega's column-window matmul for mixture-of-experts weights
//! (`src/omega_gpu_moe_api.h`, OWN-30B cut 2, no CUDA).
//!
//! One resident bf16 tensor holds every expert's projection side by side along N; a call
//! multiplies the activation rows by the column windows of the selected experts only, with the
//! same chip-proven kernel as the whole-tensor matmul (B is tile-major, so a window that starts on
//! a column tile is the kernel with its B base advanced). Conventions as in `aien-omega-gpu`:
//! row-major, `C[m x cols] = A[m x k] * B[:, col0 .. col0 + cols)`, f32 out, `a` rounded to bf16.
//!
//! Alignment rule (omega refuses otherwise, `BAD_ARGS`): `col0 % 8 == 0`, `1 <= cols`,
//! `col0 + cols <= n`, `1 <= m <= 8192`, at most [`MAX_WINDOWS`] windows per call.
//! Without the native library (`is_native()` false) every call returns `Unavailable`.
#[cfg(has_omega_moe)]
use aien_omega_gpu::OmegaGpuMatmulInfo;
use aien_omega_gpu::{rc_name, MatmulInfo, OmegaGpuError};
use std::fmt;

/// Column tile width: a window must start on a multiple of this (`OMEGA_GPU_MATMUL_TILE_N`).
pub const TILE_N: usize = 8;
/// Most windows one `windows_f32` call takes (`OMEGA_GPU_MOE_MAX_WINDOWS`).
pub const MAX_WINDOWS: usize = 64;
/// Row limit of one call (`OMEGA_BW_MATMUL_MAX_M`).
pub const MAX_ROWS: usize = 8192;

/// True when the window API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_moe)
}

#[cfg(has_omega_moe)]
mod ffi {
    use aien_omega_gpu::OmegaGpuMatmulInfo;
    #[repr(C)]
    pub struct OmegaGpuMoeTensor {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_moe_tensor_upload_bf16(
            k: u32,
            n: u32,
            b: *const u16,
            out: *mut *mut OmegaGpuMoeTensor,
        ) -> i32;
        pub fn omega_gpu_moe_tensor_free(t: *mut OmegaGpuMoeTensor);
        pub fn omega_gpu_matmul_resident_window_f32(
            m: u32,
            a: *const f32,
            t: *const OmegaGpuMoeTensor,
            col0: u32,
            cols: u32,
            c: *mut f32,
            info: *mut OmegaGpuMatmulInfo,
        ) -> i32;
        pub fn omega_gpu_matmul_resident_windows_f32(
            m: u32,
            a: *const f32,
            t: *const OmegaGpuMoeTensor,
            nwin: u32,
            col0: *const u32,
            cols: u32,
            c: *mut f32,
            info: *mut OmegaGpuMatmulInfo,
        ) -> i32;
        pub fn omega_gpu_moe_reserve(
            max_rows: u32,
            max_k: u32,
            max_cols: u32,
            kernel_slots: u32,
        ) -> i32;
        pub fn omega_gpu_moe_prepare(m: u32, k: u32, cols: u32) -> i32;
        pub fn omega_gpu_moe_seal(on: i32);
        pub fn omega_gpu_moe_unreserve();
        pub fn omega_gpu_moe_cache_clear();
        pub fn omega_gpu_moe_set_cta_budget(ctas: u32);
        pub fn omega_gpu_moe_cta_budget() -> u32;
        pub fn omega_gpu_moe_window_byte_offset(kp: u32, col0: u32) -> u64;
        pub fn omega_gpu_moe_test_set_mutant(on: i32);
        pub fn omega_gpu_moe_test_set_window_skew(tiles: i32);
    }
}

fn rc_err(rc: i32) -> OmegaGpuError {
    OmegaGpuError::Rc {
        rc,
        name: rc_name(rc),
    }
}

fn dim(v: usize) -> Result<u32, OmegaGpuError> {
    u32::try_from(v).map_err(|_| rc_err(aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_TOO_LARGE))
}

fn check_len(what: &'static str, expected: usize, got: usize) -> Result<(), OmegaGpuError> {
    if expected == got {
        Ok(())
    } else {
        Err(OmegaGpuError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

/// Checks the window alignment rule on the host, before any chip call, so a wrong offset is
/// refused by name even in the stub.
pub fn check_window(n: usize, col0: usize, cols: usize) -> Result<(), OmegaGpuError> {
    if !col0.is_multiple_of(TILE_N) || cols == 0 || col0.checked_add(cols).is_none_or(|end| end > n)
    {
        return Err(OmegaGpuError::Rc {
            rc: aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_BAD_ARGS,
            name: format!(
                "window col0={col0} cols={cols} refused: col0 must be a multiple of {TILE_N}, cols >= 1, col0 + cols <= n={n}"
            ),
        });
    }
    Ok(())
}

/// A `k x n` bf16 tensor resident on the device until dropped, addressable by column window.
/// Not `Sync` (as `ResidentTensor`): callers serialise use behind a lock.
pub struct MoeTensor {
    #[cfg(has_omega_moe)]
    ptr: *mut ffi::OmegaGpuMoeTensor,
    k: usize,
    n: usize,
}

// SAFETY: an owning handle to a device buffer; moved between threads, never shared unlocked.
unsafe impl Send for MoeTensor {}

impl fmt::Debug for MoeTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "MoeTensor {{ k: {}, n: {} }}", self.k, self.n)
    }
}

impl MoeTensor {
    /// Upload a row-major `k x n` matrix of bf16 bit patterns (no rounding).
    pub fn upload_bf16(k: usize, n: usize, b: &[u16]) -> Result<Self, OmegaGpuError> {
        let elems = k
            .checked_mul(n)
            .ok_or_else(|| rc_err(aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_TOO_LARGE))?;
        check_len("b", elems, b.len())?;
        let (ku, nu) = (dim(k)?, dim(n)?);
        #[cfg(has_omega_moe)]
        {
            let mut ptr: *mut ffi::OmegaGpuMoeTensor = std::ptr::null_mut();
            // SAFETY: b has k*n elements (checked); ptr is a valid out-pointer.
            let rc = unsafe { ffi::omega_gpu_moe_tensor_upload_bf16(ku, nu, b.as_ptr(), &mut ptr) };
            if rc != aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_OK || ptr.is_null() {
                return Err(rc_err(if rc == 0 {
                    aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_BAD_ARGS
                } else {
                    rc
                }));
            }
            Ok(Self { ptr, k, n })
        }
        #[cfg(not(has_omega_moe))]
        {
            let _ = (ku, nu);
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `c[m x cols] = a[m x k] * self[:, col0 .. col0 + cols)`.
    pub fn window_f32(
        &self,
        m: usize,
        a: &[f32],
        col0: usize,
        cols: usize,
        c: &mut [f32],
    ) -> Result<MatmulInfo, OmegaGpuError> {
        check_window(self.n, col0, cols)?;
        check_len("a", m * self.k, a.len())?;
        check_len("c", m * cols, c.len())?;
        let (mu, c0, cw) = (dim(m)?, dim(col0)?, dim(cols)?);
        #[cfg(has_omega_moe)]
        {
            let mut info = OmegaGpuMatmulInfo::zeroed();
            // SAFETY: lengths checked; self.ptr is a live handle owned by self.
            let rc = unsafe {
                ffi::omega_gpu_matmul_resident_window_f32(
                    mu,
                    a.as_ptr(),
                    self.ptr,
                    c0,
                    cw,
                    c.as_mut_ptr(),
                    &mut info,
                )
            };
            finish(rc, info)
        }
        #[cfg(not(has_omega_moe))]
        {
            let _ = (mu, c0, cw);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// `windows.len()` windows of one width in one call: `c` holds the results back to back,
    /// window `w` at `c[w * m * cols ..][.. m * cols]`. A is staged once.
    pub fn windows_f32(
        &self,
        m: usize,
        a: &[f32],
        col0s: &[u32],
        cols: usize,
        c: &mut [f32],
    ) -> Result<MatmulInfo, OmegaGpuError> {
        if col0s.is_empty() || col0s.len() > MAX_WINDOWS {
            return Err(OmegaGpuError::Rc {
                rc: aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_BAD_ARGS,
                name: format!("{} windows, allowed 1..={MAX_WINDOWS}", col0s.len()),
            });
        }
        for &c0 in col0s {
            check_window(self.n, c0 as usize, cols)?;
        }
        check_len("a", m * self.k, a.len())?;
        check_len("c", col0s.len() * m * cols, c.len())?;
        let (mu, nw, cw) = (dim(m)?, dim(col0s.len())?, dim(cols)?);
        #[cfg(has_omega_moe)]
        {
            let mut info = OmegaGpuMatmulInfo::zeroed();
            // SAFETY: lengths checked; col0s has nw entries; self.ptr is live.
            let rc = unsafe {
                ffi::omega_gpu_matmul_resident_windows_f32(
                    mu,
                    a.as_ptr(),
                    self.ptr,
                    nw,
                    col0s.as_ptr(),
                    cw,
                    c.as_mut_ptr(),
                    &mut info,
                )
            };
            finish(rc, info)
        }
        #[cfg(not(has_omega_moe))]
        {
            let _ = (mu, nw, cw);
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for MoeTensor {
    fn drop(&mut self) {
        #[cfg(has_omega_moe)]
        // SAFETY: ptr came from a successful upload and is freed exactly once.
        unsafe {
            ffi::omega_gpu_moe_tensor_free(self.ptr)
        }
    }
}

#[cfg(has_omega_moe)]
fn finish(rc: i32, info: OmegaGpuMatmulInfo) -> Result<MatmulInfo, OmegaGpuError> {
    if rc == aien_omega_gpu::ffi::OMEGA_GPU_MATMUL_OK {
        Ok(MatmulInfo { raw: info })
    } else {
        Err(rc_err(rc))
    }
}

/// Serving reservation for the window API (its own staging buffers and kernel cache; a server
/// reserves the whole-tensor API too). Stub: `Unavailable`.
pub fn reserve(
    max_rows: u32,
    max_k: u32,
    max_cols: u32,
    kernel_slots: u32,
) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_moe)]
    {
        // SAFETY: plain integers; omega takes its own lock.
        let rc = unsafe { ffi::omega_gpu_moe_reserve(max_rows, max_k, max_cols, kernel_slots) };
        if rc == 0 {
            Ok(())
        } else {
            Err(rc_err(rc))
        }
    }
    #[cfg(not(has_omega_moe))]
    {
        let _ = (max_rows, max_k, max_cols, kernel_slots);
        Err(OmegaGpuError::Unavailable)
    }
}

/// Build and pin the kernel a call of `m` rows over `k` x `cols` windows uses.
pub fn prepare(m: u32, k: u32, cols: u32) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_moe)]
    {
        // SAFETY: plain integers; omega takes its own lock.
        let rc = unsafe { ffi::omega_gpu_moe_prepare(m, k, cols) };
        if rc == 0 {
            Ok(())
        } else {
            Err(rc_err(rc))
        }
    }
    #[cfg(not(has_omega_moe))]
    {
        let _ = (m, k, cols);
        Err(OmegaGpuError::Unavailable)
    }
}

/// Sealed: an unprepared kernel is refused, never built. No-op in the stub.
pub fn seal(on: bool) {
    #[cfg(has_omega_moe)]
    // SAFETY: no pointers; omega takes its own lock.
    unsafe {
        ffi::omega_gpu_moe_seal(on as i32)
    }
    #[cfg(not(has_omega_moe))]
    let _ = on;
}

/// Lift the reservation. No-op in the stub.
pub fn unreserve() {
    #[cfg(has_omega_moe)]
    // SAFETY: no arguments.
    unsafe {
        ffi::omega_gpu_moe_unreserve()
    }
}

/// Drop every cached window kernel. No-op in the stub.
pub fn cache_clear() {
    #[cfg(has_omega_moe)]
    // SAFETY: no arguments.
    unsafe {
        ffi::omega_gpu_moe_cache_clear()
    }
}

/// CTA budget per window launch (0 restores omega's default; clears the kernel cache).
pub fn set_cta_budget(ctas: u32) {
    #[cfg(has_omega_moe)]
    // SAFETY: plain integer.
    unsafe {
        ffi::omega_gpu_moe_set_cta_budget(ctas)
    }
    #[cfg(not(has_omega_moe))]
    let _ = ctas;
}

pub fn cta_budget() -> Option<u32> {
    #[cfg(has_omega_moe)]
    // SAFETY: reads one value.
    return Some(unsafe { ffi::omega_gpu_moe_cta_budget() });
    #[cfg(not(has_omega_moe))]
    None
}

/// Byte offset of column tile `col0 / 8` inside a packed tensor of padded depth `kp`: omega's
/// own arithmetic when native, the same formula (`(col0 / 8) * kp * 8 * 2`) otherwise.
pub fn window_byte_offset(kp: u32, col0: u32) -> u64 {
    #[cfg(has_omega_moe)]
    // SAFETY: pure host arithmetic.
    return unsafe { ffi::omega_gpu_moe_window_byte_offset(kp, col0) };
    #[cfg(not(has_omega_moe))]
    {
        (col0 as u64 / TILE_N as u64) * kp as u64 * TILE_N as u64 * 2
    }
}

/// TEST ONLY: deliberately wrong window kernels (last K pass dropped). No-op in the stub.
pub fn test_set_mutant(on: bool) {
    #[cfg(has_omega_moe)]
    // SAFETY: plain integer.
    unsafe {
        ffi::omega_gpu_moe_test_set_mutant(on as i32)
    }
    #[cfg(not(has_omega_moe))]
    let _ = on;
}

/// TEST ONLY: move every window's B base by `tiles` column tiles (the wrong-window control).
pub fn test_set_window_skew(tiles: i32) {
    #[cfg(has_omega_moe)]
    // SAFETY: plain integer.
    unsafe {
        ffi::omega_gpu_moe_test_set_window_skew(tiles)
    }
    #[cfg(not(has_omega_moe))]
    let _ = tiles;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_rule_is_checked_on_the_host() {
        assert!(check_window(262144, 0, 2048).is_ok());
        assert!(check_window(262144, 127 * 2048, 2048).is_ok());
        assert!(check_window(196608, 5 * 1536, 1536).is_ok());
        assert!(
            check_window(196608, 4, 8).is_err(),
            "col0 not a tile multiple"
        );
        assert!(check_window(196608, 0, 0).is_err(), "empty window");
        assert!(check_window(196608, 196608 - 8, 16).is_err(), "past n");
    }

    #[test]
    fn stub_reports_unavailable() {
        if !is_native() {
            assert!(matches!(
                MoeTensor::upload_bf16(8, 8, &[0; 64]),
                Err(OmegaGpuError::Unavailable)
            ));
            assert_eq!(cta_budget(), None);
        }
    }

    #[test]
    fn byte_offset_formula() {
        // tile-major: column tile t of a tensor with padded depth kp starts at t * kp * 8 elements.
        assert_eq!(window_byte_offset(2048, 0), 0);
        assert_eq!(window_byte_offset(2048, 1536), (1536 / 8) * 2048 * 8 * 2);
    }
}
