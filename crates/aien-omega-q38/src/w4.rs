//! MXFP4 (4-bit) tensors for the Qwen3.6-35B-A3B graph (L355): the host converter (omega
//! `omega_gpu_w4_from_fp8_rule`, the chip's exact data), device tensors and expert slabs
//! (omega `omega_gpu_w4_api.h`, from qwen38/w4-gemv + speed1/w4-default, grouped launches l355/w4-experts).
//!
//! `cfg(has_omega_w4)`: the archive defines the converter and the W4 GEMV. Without it every call
//! returns [`OmegaGpuError::Unavailable`].
use crate::{bad, check_len, OmegaGpuError};
use std::ffi::c_void;
use std::fmt;

#[cfg(has_omega_w4)]
pub(crate) mod ffi {
    use std::ffi::c_char;

    #[repr(C)]
    pub struct OmegaGpuW4Tensor {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_w4_tensor_upload(n: u32, k: u32, w4: *const u8, e8: *const u8, out: *mut *mut OmegaGpuW4Tensor) -> i32;
        pub fn omega_gpu_w4_tensor_free(t: *mut OmegaGpuW4Tensor);
        pub fn omega_gpu_w4_tensor_bytes(t: *const OmegaGpuW4Tensor) -> usize;
        pub fn omega_gpu_w4_last_error() -> *const c_char;
        pub fn omega_gpu_w4_from_fp8_rule(
            n: u32,
            k: u32,
            w8: *const u8,
            scale_bf16: *const u16,
            row0: u32,
            rows: u32,
            rule: i32,
            w4: *mut u8,
            e8: *mut u8,
            stats: *mut super::W4Stats,
        ) -> i32;
    }
}

/// Scale rules of the converter (omega `OMEGA_W4_RULE_*`): OCP = the f7 rule (clips amax / X in [6, 8));
/// FIT = one power of two up where a group would clip; BEST = the smaller squared error per group.
pub const W4_RULE_OCP: i32 = 0;
pub const W4_RULE_FIT: i32 = 1;
pub const W4_RULE_BEST: i32 = 2;

/// The converter's error tally (omega `OmegaW4Stats`): relative RMS error = sqrt(sq_err / sq_ref).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct W4Stats {
    pub weights: u64,
    pub groups: u64,
    pub zero_groups: u64,
    pub saturated: u64,
    pub sq_err: f64,
    pub sq_ref: f64,
}

impl W4Stats {
    pub fn add(&mut self, o: &W4Stats) {
        self.weights += o.weights;
        self.groups += o.groups;
        self.zero_groups += o.zero_groups;
        self.saturated += o.saturated;
        self.sq_err += o.sq_err;
        self.sq_ref += o.sq_ref;
    }
    pub fn rel_rms(&self) -> f64 {
        if self.sq_ref > 0.0 {
            (self.sq_err / self.sq_ref).sqrt()
        } else {
            0.0
        }
    }
}

pub fn native() -> bool {
    cfg!(has_omega_w4)
}

#[cfg(has_omega_w4)]
fn w4_err(what: &str, rc: i32) -> OmegaGpuError {
    // SAFETY: no arguments; returns a static NUL-terminated string or NULL.
    crate::rc_err(what, rc, crate::c_text(unsafe { ffi::omega_gpu_w4_last_error() }))
}

/// A device-resident MXFP4 tensor `[n][k]`: e2m1 nibbles `[n][k/2]` plus one E8M0 scale per 32 weights `[n][k/32]`.
/// An expert slab is one tensor of `count` blocks (`n = count * block_rows`).
pub struct W4Tensor {
    #[cfg(has_omega_w4)]
    ptr: *mut ffi::OmegaGpuW4Tensor,
    n: usize,
    k: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for W4Tensor {}

impl fmt::Debug for W4Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "W4Tensor {{ n: {}, k: {} }}", self.n, self.k)
    }
}

impl W4Tensor {
    /// Uploads packed nibbles (`n * k / 2` bytes) and scale bytes (`n * k / 32`).
    pub fn upload(n: usize, k: usize, w4: &[u8], e8: &[u8]) -> Result<Self, OmegaGpuError> {
        check_len("w4 nibbles", n * k / 2, w4.len())?;
        check_len("w4 scales", n * k / 32, e8.len())?;
        #[cfg(has_omega_w4)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: the slices hold the lengths checked above; out is a valid pointer.
            let rc = unsafe { ffi::omega_gpu_w4_tensor_upload(n as u32, k as u32, w4.as_ptr(), e8.as_ptr(), &mut ptr) };
            if rc != 0 || ptr.is_null() {
                return Err(w4_err("w4 upload", rc));
            }
            Ok(Self { ptr, n, k })
        }
        #[cfg(not(has_omega_w4))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn shape(&self) -> (usize, usize) {
        (self.n, self.k)
    }

    /// Device bytes (nibbles + scales).
    pub fn resident_bytes(&self) -> usize {
        #[cfg(has_omega_w4)]
        {
            // SAFETY: ptr is live.
            unsafe { ffi::omega_gpu_w4_tensor_bytes(self.ptr) }
        }
        #[cfg(not(has_omega_w4))]
        0
    }

    pub fn as_raw(&self) -> *mut c_void {
        #[cfg(has_omega_w4)]
        {
            self.ptr.cast()
        }
        #[cfg(not(has_omega_w4))]
        std::ptr::null_mut()
    }
}

impl Drop for W4Tensor {
    fn drop(&mut self) {
        #[cfg(has_omega_w4)]
        // SAFETY: ptr came from omega_gpu_w4_tensor_upload and is freed once.
        unsafe {
            ffi::omega_gpu_w4_tensor_free(self.ptr)
        }
    }
}

/// FP8 `[n][k]` e4m3 with bf16 scale_inv `[n/128][k/128]` (the checkpoint layout) to MXFP4 by the omega converter
/// (`rule` one of `W4_RULE_*`) on `threads` threads in 128-row stripes: `(nibbles, scale bytes, stats)`.
pub fn from_fp8(
    n: usize,
    k: usize,
    w8: &[u8],
    scales: &[u16],
    threads: usize,
    rule: i32,
) -> Result<(Vec<u8>, Vec<u8>, W4Stats), OmegaGpuError> {
    const B: usize = 128;
    if n == 0 || k == 0 || !n.is_multiple_of(B) || !k.is_multiple_of(B) {
        return Err(bad(format!("w4 from fp8: [{n}][{k}] is not whole 128 blocks")));
    }
    check_len("w4 from fp8 weights", n * k, w8.len())?;
    check_len("w4 from fp8 scales", (n / B) * (k / B), scales.len())?;
    let mut w4 = vec![0u8; n * k / 2];
    let mut e8 = vec![0u8; n * k / 32];
    #[cfg(has_omega_w4)]
    {
        let per = (n / B).div_ceil(threads.max(1)) * B;
        // The converter addresses the whole output by row (rows [row0, row0 + rows) of w4 / e8), so every thread gets
        // the base pointers and a disjoint row range.
        let (pw, pe) = (w4.as_mut_ptr() as usize, e8.as_mut_ptr() as usize);
        let outs: Vec<(i32, W4Stats)> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..n)
                .step_by(per)
                .map(|row0| {
                    sc.spawn(move || {
                        let mut st = W4Stats::default();
                        let rows = per.min(n - row0);
                        // SAFETY: w8 / scales hold the whole tensor and w4 / e8 the whole output (checked / sized
                        // above, alive for the scope); the threads write disjoint row ranges.
                        let rc = unsafe {
                            ffi::omega_gpu_w4_from_fp8_rule(
                                n as u32,
                                k as u32,
                                w8.as_ptr(),
                                scales.as_ptr(),
                                row0 as u32,
                                rows as u32,
                                rule,
                                pw as *mut u8,
                                pe as *mut u8,
                                &mut st,
                            )
                        };
                        (rc, st)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or((-1, W4Stats::default()))).collect()
        });
        let mut total = W4Stats::default();
        for (rc, st) in &outs {
            if *rc != 0 {
                return Err(w4_err("w4 from fp8", *rc));
            }
            total.add(st);
        }
        Ok((w4, e8, total))
    }
    #[cfg(not(has_omega_w4))]
    {
        let _ = (&mut w4, &mut e8, threads, rule);
        Err(OmegaGpuError::Unavailable)
    }
}

/// e2m1 code (sign bit 3) to its value: 0, 0.5, 1, 1.5, 2, 3, 4, 6 (omega `omega_w4_e2m1_ref`).
pub fn e2m1(nib: u8) -> f32 {
    const M: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let v = M[(nib & 7) as usize];
    if nib & 8 != 0 {
        -v
    } else {
        v
    }
}

/// Row `r` of a packed tensor as f32 (exact: e2m1 * 2^(E - 127)) into `out` (`k` values). Host oracle and the
/// host reference's 4-bit weights; the nibble order is omega's `OMEGA_W4_NIBBLE` (low nibble = even column).
pub fn widen_row(w4: &[u8], e8: &[u8], k: usize, r: usize, out: &mut [f32]) {
    let rb = &w4[r * k / 2..(r + 1) * k / 2];
    let sb = &e8[r * k / 32..(r + 1) * k / 32];
    for (gi, (o, &e)) in out[..k].chunks_exact_mut(32).zip(sb).enumerate() {
        let s = 2f32.powi(e as i32 - 127); // exact (a power of two, e <= 192 by the upload rule)
        for (j, o) in o.iter_mut().enumerate() {
            let c = gi * 32 + j;
            *o = e2m1((rb[c >> 1] >> ((c & 1) * 4)) & 15) * s;
        }
    }
}
