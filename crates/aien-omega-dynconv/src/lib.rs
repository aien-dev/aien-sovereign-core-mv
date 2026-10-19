//! FFI to omega's DFlash 2 two-tap grouped dynamic causal conv (`omega_gpu_dynconv2_f32`,
//! omega branch cascade/dflash-drafter). The maths and its exact f32 order are in
//! aien-inference-abi `qwen38_dflash::dynconv2` (the host reference); the chip result is the same
//! bits.
use aien_omega_gpu::OmegaGpuError;

/// Elementwise API info block (`OmegaGpuEwInfo`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EwInfo {
    pub chip_calls: u32,
    pub elapsed_ns: u64,
    pub completion_marker: u32,
    pub unwritten_words: u32,
    pub kernel_cache_hit: bool,
    pub gpr_count: u32,
    pub insn_count: u32,
    pub threads_per_cta: u32,
    pub ctas_last_launch: u32,
    pub target_chip: [u8; 64],
    pub sm_architecture: u32,
    pub call_ns: u64,
}

impl Default for EwInfo {
    fn default() -> Self {
        // SAFETY: a plain C struct of integers, a bool and a byte array; all-zero is valid.
        unsafe { std::mem::zeroed() }
    }
}

/// The op number of the kernel (`OMEGA_GPU_EW_DYNCONV2`) for the test mutant switch.
pub const OP_DYNCONV2: i32 = 24;

#[cfg(has_omega_dynconv)]
mod ffi {
    use super::EwInfo;
    extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub fn omega_gpu_dynconv2_f32(
            rows: u32,
            hidden: u32,
            group: u32,
            v: *const f32,
            dynk: *const f32,
            base: *const f32,
            out: *mut f32,
            info: *mut EwInfo,
        ) -> i32;
        pub fn omega_gpu_elementwise_rc_name(rc: i32) -> *const std::os::raw::c_char;
        pub fn omega_gpu_elementwise_last_error() -> *const std::os::raw::c_char;
        pub fn omega_gpu_elementwise_test_set_mutant(op: i32);
    }
}

/// True when the chip kernel is linked.
pub const fn is_native() -> bool {
    cfg!(has_omega_dynconv)
}

/// `out = dynconv2(v, dynk, base)` on the chip: `v` `[rows][hidden]`, `dynk`
/// `[rows][2][hidden / 16]`, `base` `[2][hidden]`. Group must be 16.
pub fn dynconv2_f32(
    rows: usize,
    hidden: usize,
    group: usize,
    v: &[f32],
    dynk: &[f32],
    base: &[f32],
    out: &mut [f32],
) -> Result<EwInfo, OmegaGpuError> {
    let bad = |name: String| OmegaGpuError::Rc { rc: -1, name };
    if group != 16 || hidden == 0 || rows == 0 {
        return Err(bad(format!(
            "dynconv2: rows {rows} hidden {hidden} group {group}"
        )));
    }
    if v.len() != rows * hidden
        || out.len() != rows * hidden
        || dynk.len() != rows * 2 * (hidden / group)
        || base.len() != 2 * hidden
    {
        return Err(bad("dynconv2: buffer lengths".into()));
    }
    #[cfg(has_omega_dynconv)]
    {
        let mut info = EwInfo::default();
        // SAFETY: every length checked above; the call copies through its own buffers.
        let rc = unsafe {
            ffi::omega_gpu_dynconv2_f32(
                rows as u32,
                hidden as u32,
                group as u32,
                v.as_ptr(),
                dynk.as_ptr(),
                base.as_ptr(),
                out.as_mut_ptr(),
                &mut info,
            )
        };
        if rc != 0 {
            // SAFETY: static NUL-terminated strings or NULL.
            let name = unsafe {
                let n = ffi::omega_gpu_elementwise_rc_name(rc);
                let e = ffi::omega_gpu_elementwise_last_error();
                let s = |p: *const std::os::raw::c_char| {
                    if p.is_null() {
                        String::new()
                    } else {
                        std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
                    }
                };
                format!("{} {}", s(n), s(e))
            };
            return Err(OmegaGpuError::Rc { rc, name });
        }
        Ok(info)
    }
    #[cfg(not(has_omega_dynconv))]
    {
        Err(OmegaGpuError::Unavailable)
    }
}

/// TEST ONLY: `true` builds the kernel's mutant (base taps swapped) for the next calls.
pub fn test_set_mutant(on: bool) {
    #[cfg(has_omega_dynconv)]
    // SAFETY: a process-wide switch; clears the kernel cache.
    unsafe {
        ffi::omega_gpu_elementwise_test_set_mutant(if on { OP_DYNCONV2 } else { 0 })
    }
    #[cfg(not(has_omega_dynconv))]
    let _ = on;
}

#[cfg(test)]
mod tests {
    #[test]
    fn info_block_matches_the_c_layout() {
        // sizeof(OmegaGpuEwInfo) 120, offsetof(call_ns) 112 on aarch64 (omega header).
        assert_eq!(std::mem::size_of::<super::EwInfo>(), 120);
        assert_eq!(std::mem::offset_of!(super::EwInfo, call_ns), 112);
    }

    #[test]
    fn shape_errors_before_any_chip_call() {
        let v = vec![0f32; 256];
        let mut o = vec![0f32; 256];
        assert!(super::dynconv2_f32(1, 256, 8, &v, &[0.0; 32], &[0.0; 512], &mut o).is_err());
        assert!(super::dynconv2_f32(1, 256, 16, &v, &[0.0; 31], &[0.0; 512], &mut o).is_err());
    }
}
