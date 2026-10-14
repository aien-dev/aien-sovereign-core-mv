//! Safe Rust wrapper over omega's FP8 e4m3 block-scaled GEMV (`src/omega_gpu_fp8_api.h`,
//! OWN-30B cuts 5-6, no CUDA).
//!
//! A tensor holds `experts` projections of shape `[n][k]` (n output rows, k input columns) as
//! the checkpoint's e4m3 bytes plus bf16 `weight_scale_inv` in 128 x 128 blocks, exactly the
//! safetensors layout, experts concatenated. One call computes
//! `y[g][r] = sum_c W[id_g][r][c] * x[row(g)][c] * scale[id_g][r/128][c/128]` in f32 for up to
//! [`MAX_GROUPS`] experts in one launch, with one shared activation row (`x_rows == 1`) or one
//! row per group (`x_rows == groups`).
//!
//! Rules omega refuses otherwise (`BAD_ARGS` / `TOO_LARGE`): `n % 128 == 0`, `k % 128 == 0`,
//! `128 <= k <= 16384`, `1 <= groups <= 64`, every id `< experts`; any e4m3 NaN code in the
//! weights is refused at upload (`NAN_WEIGHT`). Without the native library (`is_native()`
//! false) every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;
use std::fmt;

/// Scale block edge (`weight_scale_inv` covers 128 x 128 weights).
pub const BLOCK: usize = 128;
/// Most experts one grouped call takes (`OMEGA_GPU_FP8_MAX_GROUPS`).
pub const MAX_GROUPS: usize = 64;
/// Widest `k` the kernel accepts.
pub const MAX_K: usize = 16384;

/// True when the FP8 API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_fp8)
}

/// True when the device-table entry points (`omega_gpu_fp8_gemv_grouped_dev`,
/// `omega_gpu_fp8_tensor_entry`, OWN-30B router cut) are linked.
pub const fn has_device_tables() -> bool {
    cfg!(has_omega_fp8_dev)
}

/// True when the prefill grouped GEMM (`omega_gpu_fp8gemm_f32`, `src/omega_gpu_fp8gemm_api.h`) is linked.
pub const fn has_gemm() -> bool {
    cfg!(has_omega_fp8gemm)
}

/// Most rows one GEMM launch takes (`OMEGA_GPU_FP8GEMM_MAX_ROWS`).
pub const GEMM_MAX_ROWS: usize = 1024;
/// Most distinct experts (segments) one GEMM launch takes (`OMEGA_GPU_FP8GEMM_MAX_SEGMENTS`).
pub const GEMM_MAX_SEGMENTS: usize = 256;

/// Mirror of `OmegaGpuFp8GemmInfo`: what the chip did for one GEMM call.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GemmInfo {
    pub rows: u32,
    pub segments: u32,
    pub n: u32,
    pub k: u32,
    pub mt: u32,
    pub warps_per_cta: u32,
    pub lane_bytes: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub passes_max: u32,
    /// e4m3 bytes of the distinct experts, one pass each.
    pub weight_bytes: u64,
    /// Bytes the launch reads from the weights (sum over segments of passes x n x k).
    pub streamed_bytes: u64,
    /// Chip submit-to-second-marker.
    pub elapsed_ns: u64,
    /// Whole API call (host permute, launch, copy back).
    pub call_ns: u64,
    pub completion_marker: u32,
    pub kernel_cache_hit: bool,
}

/// Mirror of `OmegaGpuFp8Info`: what the chip did for one call.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Fp8Info {
    pub experts: u32,
    pub n: u32,
    pub k: u32,
    pub groups: u32,
    pub lane_bytes: u32,
    pub rows_per_warp: u32,
    pub warps_per_cta: u32,
    pub depth: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub steps: u32,
    pub weight_bytes: u64,
    pub scale_bytes: u64,
    pub elapsed_ns: u64,
    pub call_ns: u64,
    pub completion_marker: u32,
    pub kernel_cache_hit: bool,
    pub target_chip: [u8; 64],
    pub sm_architecture: u32,
}

impl Default for Fp8Info {
    fn default() -> Self {
        // SAFETY: a plain C struct of integers, a bool and a byte array; all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

#[cfg(has_omega_fp8)]
mod ffi {
    use super::Fp8Info;
    #[repr(C)]
    pub struct OmegaGpuFp8Tensor {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_fp8_tensor_upload(
            experts: u32,
            n: u32,
            k: u32,
            w: *const u8,
            scale_inv_bf16: *const u16,
            out: *mut *mut OmegaGpuFp8Tensor,
        ) -> i32;
        pub fn omega_gpu_fp8_tensor_free(t: *mut OmegaGpuFp8Tensor);
        pub fn omega_gpu_fp8_gemv_f32(
            t: *const OmegaGpuFp8Tensor,
            expert: u32,
            x: *const f32,
            y: *mut f32,
            info: *mut Fp8Info,
        ) -> i32;
        pub fn omega_gpu_fp8_gemv_grouped_f32(
            t: *const OmegaGpuFp8Tensor,
            groups: u32,
            expert_ids: *const u32,
            x: *const f32,
            x_rows: u32,
            y: *mut f32,
            info: *mut Fp8Info,
        ) -> i32;
    }
    #[cfg(has_omega_fp8_dev)]
    extern "C" {
        pub fn omega_gpu_fp8_gemv_grouped_dev(
            t: *const OmegaGpuFp8Tensor,
            groups: u32,
            tab_va: u64,
            x_va: u64,
            y: *mut f32,
            info: *mut Fp8Info,
        ) -> i32;
        pub fn omega_gpu_fp8_tensor_entry(t: *const OmegaGpuFp8Tensor, expert: u32, w_va: *mut u64, s_va: *mut u64) -> i32;
    }
    #[cfg(has_omega_fp8gemm)]
    extern "C" {
        pub fn omega_gpu_fp8gemm_f32(
            t: *const OmegaGpuFp8Tensor,
            m: u32,
            expert_ids: *const u32,
            token_rows: *const u32,
            x: *const f32,
            x_rows: u32,
            y: *mut f32,
            info: *mut super::GemmInfo,
        ) -> i32;
        pub fn omega_gpu_fp8gemm_prepare(k: u32) -> i32;
        pub fn omega_gpu_fp8gemm_rc_name(rc: i32) -> *const std::os::raw::c_char;
        pub fn omega_gpu_fp8gemm_last_error() -> *const std::os::raw::c_char;
        pub fn omega_gpu_fp8gemm_test_set_mutant(mutant: i32);
    }
    extern "C" {
        pub fn omega_gpu_fp8_prepare(k: u32) -> i32;
        pub fn omega_gpu_fp8_seal(on: i32);
        pub fn omega_gpu_fp8_rc_name(rc: i32) -> *const std::os::raw::c_char;
        pub fn omega_gpu_fp8_last_error() -> *const std::os::raw::c_char;
        pub fn omega_gpu_fp8_cache_clear();
        pub fn omega_gpu_fp8_set_shape(lane_bytes: u32, rows_per_warp: u32, warps_per_cta: u32);
        pub fn omega_gpu_fp8_set_cta_budget(ctas: u32);
        pub fn omega_gpu_fp8_test_set_mutant(mutant: i32);
        pub fn omega_gpu_fp8_test_set_no_fold(on: i32);
    }
}

#[cfg(has_omega_fp8)]
fn rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: both return static NUL-terminated strings (or NULL for last_error).
    let name = unsafe {
        let n = ffi::omega_gpu_fp8_rc_name(rc);
        let n = if n.is_null() { String::from("?") } else { std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned() };
        let e = ffi::omega_gpu_fp8_last_error();
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
        Err(OmegaGpuError::ShapeMismatch { what, expected, got })
    }
}

fn bad(name: String) -> OmegaGpuError {
    OmegaGpuError::Rc { rc: -1, name }
}

/// Host check of the upload shape rule, before any chip call (also in the stub).
pub fn check_shape(experts: usize, n: usize, k: usize) -> Result<(), OmegaGpuError> {
    if experts == 0 || n == 0 || !n.is_multiple_of(BLOCK) || !k.is_multiple_of(BLOCK) || !(BLOCK..=MAX_K).contains(&k) {
        return Err(bad(format!(
            "fp8 shape experts={experts} n={n} k={k} refused: experts >= 1, n and k multiples of {BLOCK}, {BLOCK} <= k <= {MAX_K}"
        )));
    }
    Ok(())
}

/// `experts` FP8 projections of `[n][k]`, resident on the device until dropped. Not `Sync`:
/// callers serialise use behind a lock.
pub struct Fp8Tensor {
    #[cfg(has_omega_fp8)]
    ptr: *mut ffi::OmegaGpuFp8Tensor,
    experts: usize,
    n: usize,
    k: usize,
}

// SAFETY: an owning handle to a device buffer; moved between threads, never shared unlocked.
unsafe impl Send for Fp8Tensor {}

impl fmt::Debug for Fp8Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Fp8Tensor {{ experts: {}, n: {}, k: {} }}", self.experts, self.n, self.k)
    }
}

impl Fp8Tensor {
    /// The C handle (`OmegaGpuFp8Tensor *`) for another omega API that takes this object (the decode
    /// graph, OWN-30B cut 2). Null in the stub. The handle stays owned by `self`: the caller must
    /// not free it, must not use it after `self` is dropped, and must serialise its use with
    /// `self`'s own calls.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        #[cfg(has_omega_fp8)]
        {
            self.ptr as *mut std::ffi::c_void
        }
        #[cfg(not(has_omega_fp8))]
        {
            std::ptr::null_mut()
        }
    }

    /// Upload `experts * n * k` e4m3 bytes (`[expert][n][k]`) and
    /// `experts * (n/128) * (k/128)` bf16 scale bits (`[expert][n/128][k/128]`).
    pub fn upload(experts: usize, n: usize, k: usize, w: &[u8], scale_inv_bf16: &[u16]) -> Result<Self, OmegaGpuError> {
        check_shape(experts, n, k)?;
        check_len("fp8 weight bytes", experts * n * k, w.len())?;
        check_len("fp8 scale words", experts * (n / BLOCK) * (k / BLOCK), scale_inv_bf16.len())?;
        #[cfg(has_omega_fp8)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: lengths checked above; omega copies the bytes before returning.
            let rc = unsafe {
                ffi::omega_gpu_fp8_tensor_upload(experts as u32, n as u32, k as u32, w.as_ptr(), scale_inv_bf16.as_ptr(), &mut ptr)
            };
            if rc != 0 || ptr.is_null() {
                return Err(rc_err(rc));
            }
            Ok(Self { ptr, experts, n, k })
        }
        #[cfg(not(has_omega_fp8))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn experts(&self) -> usize {
        self.experts
    }
    pub fn n(&self) -> usize {
        self.n
    }
    pub fn k(&self) -> usize {
        self.k
    }

    /// Bytes the device holds for this tensor (weights plus f32 scales).
    pub fn resident_bytes(&self) -> usize {
        self.experts * self.n * self.k + self.experts * (self.n / BLOCK) * (self.k / BLOCK) * 4
    }

    /// One projection, one launch: `y[n] = W[expert] * x[k]`.
    pub fn gemv_f32(&self, expert: usize, x: &[f32], y: &mut [f32]) -> Result<Fp8Info, OmegaGpuError> {
        check_len("fp8 x", self.k, x.len())?;
        check_len("fp8 y", self.n, y.len())?;
        if expert >= self.experts {
            return Err(bad(format!("fp8 expert {expert} >= experts {}", self.experts)));
        }
        #[cfg(has_omega_fp8)]
        {
            let mut info = Fp8Info::default();
            // SAFETY: lengths checked; the tensor is live for &self.
            let rc = unsafe { ffi::omega_gpu_fp8_gemv_f32(self.ptr, expert as u32, x.as_ptr(), y.as_mut_ptr(), &mut info) };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_fp8))]
        {
            let _ = (x, y);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Device addresses `(weights, scales)` of one expert, for a directory the router copies
    /// group-table entries from (`omega_gpu_fp8_tensor_entry`).
    pub fn entry(&self, expert: usize) -> Result<(u64, u64), OmegaGpuError> {
        if expert >= self.experts {
            return Err(bad(format!("fp8 expert {expert} >= experts {}", self.experts)));
        }
        #[cfg(has_omega_fp8_dev)]
        {
            let (mut w, mut s) = (0u64, 0u64);
            // SAFETY: expert is in range; both out pointers are valid for a write.
            let rc = unsafe { ffi::omega_gpu_fp8_tensor_entry(self.ptr, expert as u32, &mut w, &mut s) };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok((w, s))
        }
        #[cfg(not(has_omega_fp8_dev))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// `groups` experts in one launch whose group table and single activation row are already
    /// on the device (written by the router launch): `tab_va` holds `groups` `{w_va, s_va}`
    /// entries, `x_va` the f32 row already multiplied by 2^64. No host copy of either.
    pub fn gemv_grouped_dev(&self, groups: usize, tab_va: u64, x_va: u64, y: &mut [f32]) -> Result<Fp8Info, OmegaGpuError> {
        if groups == 0 || groups > MAX_GROUPS {
            return Err(bad(format!("fp8 groups {groups} outside 1..={MAX_GROUPS}")));
        }
        if tab_va == 0 || x_va == 0 {
            return Err(bad("fp8 device table or row address is 0".into()));
        }
        check_len("fp8 y (groups*n)", groups * self.n, y.len())?;
        #[cfg(has_omega_fp8_dev)]
        {
            let mut info = Fp8Info::default();
            // SAFETY: y length checked; the addresses come from the router that wrote them; the
            // tensor is live for &self.
            let rc = unsafe { ffi::omega_gpu_fp8_gemv_grouped_dev(self.ptr, groups as u32, tab_va, x_va, y.as_mut_ptr(), &mut info) };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_fp8_dev))]
        {
            let _ = y;
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// `groups = ids.len()` experts in one launch. `x` is `k` values shared by every group, or
    /// `groups * k` values (one row per group). `y` is `groups * n`, group-major.
    pub fn gemv_grouped_f32(&self, ids: &[u32], x: &[f32], y: &mut [f32]) -> Result<Fp8Info, OmegaGpuError> {
        let groups = ids.len();
        if groups == 0 || groups > MAX_GROUPS {
            return Err(bad(format!("fp8 groups {groups} outside 1..={MAX_GROUPS}")));
        }
        if let Some(&id) = ids.iter().find(|&&id| id as usize >= self.experts) {
            return Err(bad(format!("fp8 expert id {id} >= experts {}", self.experts)));
        }
        let x_rows = if x.len() == self.k { 1 } else { groups };
        check_len("fp8 x (k or groups*k)", x_rows * self.k, x.len())?;
        check_len("fp8 y (groups*n)", groups * self.n, y.len())?;
        #[cfg(has_omega_fp8)]
        {
            let mut info = Fp8Info::default();
            // SAFETY: lengths checked; ids are in range; the tensor is live for &self.
            let rc = unsafe {
                ffi::omega_gpu_fp8_gemv_grouped_f32(self.ptr, groups as u32, ids.as_ptr(), x.as_ptr(), x_rows as u32, y.as_mut_ptr(), &mut info)
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_fp8))]
        {
            let _ = (x_rows, y);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Prefill grouped GEMM, one launch (`omega_gpu_fp8gemm_f32`). Row `i` computes expert
    /// `expert_ids[i]` on activation row `x[token_rows[i]]` (`x` is `x_rows * k` values);
    /// `expert_ids` must be non-decreasing (sorted by expert). `y` is `M * n`, in row-table
    /// order, bit-identical to calling the GEMV once per row. Dense projections are tensors
    /// with `experts == 1`.
    pub fn gemm_f32(&self, expert_ids: &[u32], token_rows: &[u32], x: &[f32], y: &mut [f32]) -> Result<GemmInfo, OmegaGpuError> {
        let m = expert_ids.len();
        if m == 0 || m > GEMM_MAX_ROWS {
            return Err(bad(format!("fp8 gemm rows {m} outside 1..={GEMM_MAX_ROWS}")));
        }
        check_len("fp8 gemm token rows", m, token_rows.len())?;
        if !x.len().is_multiple_of(self.k) || x.is_empty() {
            return Err(bad(format!("fp8 gemm x length {} is not a positive multiple of k={}", x.len(), self.k)));
        }
        let x_rows = x.len() / self.k;
        if let Some(&id) = expert_ids.iter().find(|&&id| id as usize >= self.experts) {
            return Err(bad(format!("fp8 gemm expert id {id} >= experts {}", self.experts)));
        }
        if expert_ids.windows(2).any(|w| w[0] > w[1]) {
            return Err(bad("fp8 gemm expert ids must be sorted (non-decreasing)".into()));
        }
        let mut segments = 1;
        for w in expert_ids.windows(2) {
            if w[0] != w[1] {
                segments += 1;
            }
        }
        if segments > GEMM_MAX_SEGMENTS {
            return Err(bad(format!("fp8 gemm segments {segments} > {GEMM_MAX_SEGMENTS}")));
        }
        if let Some(&r) = token_rows.iter().find(|&&r| r as usize >= x_rows) {
            return Err(bad(format!("fp8 gemm token row {r} >= x rows {x_rows}")));
        }
        check_len("fp8 gemm y (rows*n)", m * self.n, y.len())?;
        #[cfg(has_omega_fp8gemm)]
        {
            let mut info = GemmInfo::default();
            // SAFETY: lengths, ranges and sort order checked above; the tensor is live for &self.
            let rc = unsafe {
                ffi::omega_gpu_fp8gemm_f32(
                    self.ptr,
                    m as u32,
                    expert_ids.as_ptr(),
                    token_rows.as_ptr(),
                    x.as_ptr(),
                    x_rows as u32,
                    y.as_mut_ptr(),
                    &mut info,
                )
            };
            if rc != 0 {
                return Err(gemm_rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_fp8gemm))]
        {
            let _ = (x_rows, y);
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for Fp8Tensor {
    fn drop(&mut self) {
        #[cfg(has_omega_fp8)]
        // SAFETY: ptr came from upload and is freed once.
        unsafe {
            ffi::omega_gpu_fp8_tensor_free(self.ptr)
        }
    }
}

#[cfg(has_omega_fp8gemm)]
fn gemm_rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: both return static NUL-terminated strings (or NULL for last_error).
    let name = unsafe {
        let n = ffi::omega_gpu_fp8gemm_rc_name(rc);
        let n = if n.is_null() { String::from("?") } else { std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned() };
        let e = ffi::omega_gpu_fp8gemm_last_error();
        if e.is_null() {
            n
        } else {
            format!("{n}: {}", std::ffi::CStr::from_ptr(e).to_string_lossy())
        }
    };
    OmegaGpuError::Rc { rc, name }
}

/// Serving: build and pin the GEMM kernel for width `k`.
pub fn gemm_prepare(k: usize) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_fp8gemm)]
    {
        // SAFETY: plain integer argument.
        let rc = unsafe { ffi::omega_gpu_fp8gemm_prepare(k as u32) };
        if rc != 0 {
            return Err(gemm_rc_err(rc));
        }
        Ok(())
    }
    #[cfg(not(has_omega_fp8gemm))]
    {
        let _ = k;
        Err(OmegaGpuError::Unavailable)
    }
}

/// TEST ONLY: deliberately wrong GEMM kernel (0 off; 1 scale dropped; 2 scale pointer never
/// advances; 3 neighbour row's x). Clears the GEMM kernel cache.
pub fn gemm_test_set_mutant(mutant: i32) {
    #[cfg(has_omega_fp8gemm)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_fp8gemm_test_set_mutant(mutant)
    }
    #[cfg(not(has_omega_fp8gemm))]
    let _ = mutant;
}

/// Serving: build and pin the kernel for width `k`.
pub fn prepare(k: usize) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_fp8)]
    {
        // SAFETY: plain integer argument.
        let rc = unsafe { ffi::omega_gpu_fp8_prepare(k as u32) };
        if rc != 0 {
            return Err(rc_err(rc));
        }
        Ok(())
    }
    #[cfg(not(has_omega_fp8))]
    {
        let _ = k;
        Err(OmegaGpuError::Unavailable)
    }
}

/// Seal on: a call whose kernel is not cached is refused instead of built.
pub fn seal(on: bool) {
    #[cfg(has_omega_fp8)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_fp8_seal(on as i32)
    }
    #[cfg(not(has_omega_fp8))]
    let _ = on;
}

pub fn cache_clear() {
    #[cfg(has_omega_fp8)]
    // SAFETY: no arguments.
    unsafe {
        ffi::omega_gpu_fp8_cache_clear()
    }
}

/// Kernel shape override (0 = automatic); clears the kernel cache.
pub fn set_shape(lane_bytes: u32, rows_per_warp: u32, warps_per_cta: u32) {
    #[cfg(has_omega_fp8)]
    // SAFETY: plain integer arguments.
    unsafe {
        ffi::omega_gpu_fp8_set_shape(lane_bytes, rows_per_warp, warps_per_cta)
    }
    #[cfg(not(has_omega_fp8))]
    let _ = (lane_bytes, rows_per_warp, warps_per_cta);
}

/// CTAs per group per launch (0 = automatic: 256 single, 6 grouped).
pub fn set_cta_budget(ctas: u32) {
    #[cfg(has_omega_fp8)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_fp8_set_cta_budget(ctas)
    }
    #[cfg(not(has_omega_fp8))]
    let _ = ctas;
}

/// TEST ONLY: deliberately wrong kernel (0 off; 1 scale dropped; 2 scale block never advances;
/// 3 group table entry skewed). A parity test must catch each.
pub fn test_set_mutant(mutant: i32) {
    #[cfg(has_omega_fp8)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_fp8_test_set_mutant(mutant)
    }
    #[cfg(not(has_omega_fp8))]
    let _ = mutant;
}

/// TEST ONLY: upload scales without the 2^56 fold (host-side mutant).
pub fn test_set_no_fold(on: bool) {
    #[cfg(has_omega_fp8)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_fp8_test_set_no_fold(on as i32)
    }
    #[cfg(not(has_omega_fp8))]
    let _ = on;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_rule_is_checked_on_the_host() {
        assert!(check_shape(1, 128, 128).is_ok());
        assert!(check_shape(128, 1536, 2048).is_ok());
        assert!(check_shape(0, 128, 128).is_err());
        assert!(check_shape(1, 100, 128).is_err());
        assert!(check_shape(1, 128, 64).is_err());
        assert!(check_shape(1, 128, MAX_K + 128).is_err());
    }

    #[test]
    fn lengths_are_checked_before_the_chip() {
        let r = Fp8Tensor::upload(1, 128, 128, &[0u8; 100], &[0u16; 1]);
        assert!(matches!(r, Err(OmegaGpuError::ShapeMismatch { what: "fp8 weight bytes", .. })));
        let r = Fp8Tensor::upload(1, 128, 128, &[0u8; 128 * 128], &[0u16; 2]);
        assert!(matches!(r, Err(OmegaGpuError::ShapeMismatch { what: "fp8 scale words", .. })));
    }

    #[test]
    fn info_layout_matches_the_c_struct() {
        // Measured with cc against omega_gpu_fp8_api.h (own30b/integration da9a3d67):
        // sizeof 160, weight_bytes at 48, kernel_cache_hit at 84, sm_architecture at 152.
        assert_eq!(std::mem::size_of::<Fp8Info>(), 160);
        assert_eq!(std::mem::offset_of!(Fp8Info, weight_bytes), 48);
        assert_eq!(std::mem::offset_of!(Fp8Info, kernel_cache_hit), 84);
        assert_eq!(std::mem::offset_of!(Fp8Info, sm_architecture), 152);
    }

    #[test]
    fn gemm_info_layout_matches_the_c_struct() {
        // Measured with cc against omega_gpu_fp8gemm_api.h (own30b/integration 967b14c5):
        // sizeof 80, passes_max at 36, weight_bytes at 40, streamed_bytes at 48, elapsed_ns at 56,
        // call_ns at 64, completion_marker at 72, kernel_cache_hit at 76.
        assert_eq!(std::mem::size_of::<GemmInfo>(), 80);
        assert_eq!(std::mem::offset_of!(GemmInfo, passes_max), 36);
        assert_eq!(std::mem::offset_of!(GemmInfo, weight_bytes), 40);
        assert_eq!(std::mem::offset_of!(GemmInfo, streamed_bytes), 48);
        assert_eq!(std::mem::offset_of!(GemmInfo, elapsed_ns), 56);
        assert_eq!(std::mem::offset_of!(GemmInfo, call_ns), 64);
        assert_eq!(std::mem::offset_of!(GemmInfo, completion_marker), 72);
        assert_eq!(std::mem::offset_of!(GemmInfo, kernel_cache_hit), 76);
    }
}
