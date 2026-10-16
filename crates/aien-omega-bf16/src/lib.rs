//! Safe Rust wrapper over omega's bf16 streaming GEMV (`src/omega_gpu_bf16_gemv_api.h`,
//! OWN-30B lm_head cut on the omega side; no CUDA).
//!
//! A tensor is an `[n][k]` bf16 matrix exactly as the checkpoint stores it (the lm_head is
//! `[vocab = 151936][hidden = 2048]`, no transpose), resident on the chip. One call is
//! `y[r] = sum_c W[r][c] * x[c]` in f32 with `x` f32 or bf16: one launch, deterministic
//! summation order (so the same input always gives the same logits).
//!
//! Rules omega refuses otherwise: `n >= 1`, `k % 256 == 0`, `256 <= k <= 16384`, and a NaN or
//! infinite bf16 weight is refused at upload. Without the native library (`is_native()` false)
//! every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;
use std::fmt;

/// Narrowest and widest `k` the kernel accepts, and its granule.
pub const MIN_K: usize = 256;
pub const MAX_K: usize = 16384;
pub const K_GRANULE: usize = 256;

/// True when the bf16 GEMV API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_bf16)
}

/// Mirror of `OmegaGpuBf16Info`: what the chip did for one call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bf16Info {
    pub n: u32,
    pub k: u32,
    pub rows_per_warp: u32,
    pub warps_per_cta: u32,
    pub depth: u32,
    pub grid_x: u32,
    pub steps: u32,
    /// bf16 bytes the launch reads (`n * k * 2`).
    pub weight_bytes: u64,
    /// Chip submit-to-second-marker.
    pub elapsed_ns: u64,
    /// Whole API call.
    pub call_ns: u64,
    pub completion_marker: u32,
    pub kernel_cache_hit: bool,
    pub target_chip: [u8; 64],
    pub sm_architecture: u32,
}

impl Default for Bf16Info {
    fn default() -> Self {
        // SAFETY: a plain C struct of integers, a bool and a byte array; all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(has_omega_bf16)]
mod ffi {
    use super::Bf16Info;
    #[repr(C)]
    pub struct OmegaGpuBf16Tensor {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_bf16_tensor_upload(
            n: u32,
            k: u32,
            w_bf16: *const u16,
            out: *mut *mut OmegaGpuBf16Tensor,
        ) -> i32;
        pub fn omega_gpu_bf16_tensor_free(t: *mut OmegaGpuBf16Tensor);
        pub fn omega_gpu_bf16_gemv_f32(
            t: *const OmegaGpuBf16Tensor,
            x: *const f32,
            y: *mut f32,
            info: *mut Bf16Info,
        ) -> i32;
        pub fn omega_gpu_bf16_gemv_bf16(
            t: *const OmegaGpuBf16Tensor,
            x: *const u16,
            y: *mut f32,
            info: *mut Bf16Info,
        ) -> i32;
        pub fn omega_gpu_bf16_prepare(k: u32) -> i32;
        pub fn omega_gpu_bf16_rc_name(rc: i32) -> *const std::os::raw::c_char;
        pub fn omega_gpu_bf16_last_error() -> *const std::os::raw::c_char;
        pub fn omega_gpu_bf16_test_set_mutant(mutant: i32);
    }
}

#[cfg(has_omega_bf16)]
fn rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: both return static NUL-terminated strings (or NULL for last_error).
    let name = unsafe {
        let n = ffi::omega_gpu_bf16_rc_name(rc);
        let n = if n.is_null() {
            String::from("?")
        } else {
            std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned()
        };
        let e = ffi::omega_gpu_bf16_last_error();
        if e.is_null() {
            n
        } else {
            format!("{n}: {}", std::ffi::CStr::from_ptr(e).to_string_lossy())
        }
    };
    OmegaGpuError::Rc { rc, name }
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

fn bad(name: String) -> OmegaGpuError {
    OmegaGpuError::Rc { rc: -1, name }
}

/// Host check of the upload shape rule, before any chip call (also in the stub).
pub fn check_shape(n: usize, k: usize) -> Result<(), OmegaGpuError> {
    if n == 0 || !k.is_multiple_of(K_GRANULE) || !(MIN_K..=MAX_K).contains(&k) {
        return Err(bad(format!("bf16 gemv shape n={n} k={k} refused: n >= 1, k a multiple of {K_GRANULE} in {MIN_K}..={MAX_K}")));
    }
    Ok(())
}

/// An `[n][k]` bf16 matrix resident on the device until dropped. Not `Sync`: callers
/// serialise use behind a lock.
pub struct Bf16Tensor {
    #[cfg(has_omega_bf16)]
    ptr: *mut ffi::OmegaGpuBf16Tensor,
    n: usize,
    k: usize,
}

// SAFETY: an owning handle to a device buffer; moved between threads, never shared unlocked.
unsafe impl Send for Bf16Tensor {}

impl fmt::Debug for Bf16Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Bf16Tensor {{ n: {}, k: {} }}", self.n, self.k)
    }
}

impl Bf16Tensor {
    /// The C handle (`OmegaGpuBf16Tensor *`) for another omega API that takes this object (the decode
    /// graph, OWN-30B cut 2). Null in the stub. The handle stays owned by `self`: the caller must
    /// not free it, must not use it after `self` is dropped, and must serialise its use with
    /// `self`'s own calls.
    // SAFETY: C writes through this handle, so it is only handed out under `&mut self`.
    pub fn as_raw(&mut self) -> *mut std::ffi::c_void {
        #[cfg(has_omega_bf16)]
        {
            self.ptr as *mut std::ffi::c_void
        }
        #[cfg(not(has_omega_bf16))]
        {
            std::ptr::null_mut()
        }
    }

    /// Upload `n * k` bf16 bit patterns, row-major `[n][k]` (the checkpoint's layout).
    pub fn upload(n: usize, k: usize, w_bf16: &[u16]) -> Result<Self, OmegaGpuError> {
        check_shape(n, k)?;
        check_len("bf16 gemv weight words", n * k, w_bf16.len())?;
        #[cfg(has_omega_bf16)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: shape and length checked; omega copies the bytes before returning.
            let rc = unsafe {
                ffi::omega_gpu_bf16_tensor_upload(n as u32, k as u32, w_bf16.as_ptr(), &mut ptr)
            };
            if rc != 0 || ptr.is_null() {
                return Err(rc_err(rc));
            }
            Ok(Self { ptr, n, k })
        }
        #[cfg(not(has_omega_bf16))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn n(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }

    /// Bytes the device holds for this tensor.
    pub fn resident_bytes(&self) -> usize {
        self.n * self.k * 2
    }

    /// One launch: `y[n] = W * x[k]`, `x` f32.
    pub fn gemv_f32(&self, x: &[f32], y: &mut [f32]) -> Result<Bf16Info, OmegaGpuError> {
        check_len("bf16 gemv x", self.k, x.len())?;
        check_len("bf16 gemv y", self.n, y.len())?;
        #[cfg(has_omega_bf16)]
        {
            let mut info = Bf16Info::default();
            // SAFETY: lengths checked; the tensor is live for &self.
            let rc = unsafe {
                ffi::omega_gpu_bf16_gemv_f32(self.ptr, x.as_ptr(), y.as_mut_ptr(), &mut info)
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_bf16))]
        {
            let _ = (x, y);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// As [`Self::gemv_f32`] with `x` given as bf16 bits.
    pub fn gemv_bf16(&self, x: &[u16], y: &mut [f32]) -> Result<Bf16Info, OmegaGpuError> {
        check_len("bf16 gemv x", self.k, x.len())?;
        check_len("bf16 gemv y", self.n, y.len())?;
        #[cfg(has_omega_bf16)]
        {
            let mut info = Bf16Info::default();
            // SAFETY: lengths checked; the tensor is live for &self.
            let rc = unsafe {
                ffi::omega_gpu_bf16_gemv_bf16(self.ptr, x.as_ptr(), y.as_mut_ptr(), &mut info)
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_bf16))]
        {
            let _ = (x, y);
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for Bf16Tensor {
    fn drop(&mut self) {
        #[cfg(has_omega_bf16)]
        // SAFETY: ptr came from upload and is freed once.
        unsafe {
            ffi::omega_gpu_bf16_tensor_free(self.ptr)
        }
    }
}

/// Serving: build and pin the kernel for width `k`.
pub fn prepare(k: usize) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_bf16)]
    {
        // SAFETY: plain integer argument.
        let rc = unsafe { ffi::omega_gpu_bf16_prepare(k as u32) };
        if rc != 0 {
            return Err(rc_err(rc));
        }
        Ok(())
    }
    #[cfg(not(has_omega_bf16))]
    {
        let _ = k;
        Err(OmegaGpuError::Unavailable)
    }
}

/// TEST ONLY: deliberately wrong kernels 1, 2, 3 (0 off); clears the kernel cache.
pub fn test_set_mutant(mutant: i32) {
    #[cfg(has_omega_bf16)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_bf16_test_set_mutant(mutant)
    }
    #[cfg(not(has_omega_bf16))]
    let _ = mutant;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_rule_is_checked_on_the_host() {
        assert!(check_shape(151_936, 2048).is_ok());
        assert!(check_shape(1, 256).is_ok());
        assert!(check_shape(0, 2048).is_err());
        assert!(check_shape(8, 100).is_err());
        assert!(check_shape(8, 128).is_err());
        assert!(check_shape(8, MAX_K + 256).is_err());
    }

    #[test]
    fn lengths_are_checked_before_the_chip() {
        let r = Bf16Tensor::upload(4, 256, &[0u16; 100]);
        assert!(matches!(
            r,
            Err(OmegaGpuError::ShapeMismatch {
                what: "bf16 gemv weight words",
                ..
            })
        ));
    }

    #[test]
    fn info_layout_matches_the_c_struct() {
        // Measured with cc against omega_gpu_bf16_gemv_api.h (own30b/integration 3c12866a):
        // sizeof 136, weight_bytes at 32, elapsed_ns at 40, call_ns at 48, completion_marker at
        // 56, kernel_cache_hit (bool) at 60, target_chip at 61, sm_architecture at 128.
        assert_eq!(std::mem::size_of::<Bf16Info>(), 136);
        assert_eq!(std::mem::offset_of!(Bf16Info, grid_x), 20);
        assert_eq!(std::mem::offset_of!(Bf16Info, steps), 24);
        assert_eq!(std::mem::offset_of!(Bf16Info, weight_bytes), 32);
        assert_eq!(std::mem::offset_of!(Bf16Info, elapsed_ns), 40);
        assert_eq!(std::mem::offset_of!(Bf16Info, call_ns), 48);
        assert_eq!(std::mem::offset_of!(Bf16Info, completion_marker), 56);
        assert_eq!(std::mem::offset_of!(Bf16Info, kernel_cache_hit), 60);
        assert_eq!(std::mem::offset_of!(Bf16Info, target_chip), 61);
        assert_eq!(std::mem::offset_of!(Bf16Info, sm_architecture), 128);
    }
}
