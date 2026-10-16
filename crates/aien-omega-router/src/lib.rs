//! Safe Rust wrapper over omega's one-launch MoE router for Qwen3-30B-A3B
//! (`src/omega_gpu_router_api.h`, OWN-30B router cut on the omega side; no CUDA).
//!
//! One call per layer per token does what the host did between attention and the experts:
//! RMSNorm of the residual row with the layer's `post_attention_layernorm` weight, the 128
//! router logits over bf16 weights, softmax, top-8 and (when `norm_topk_prob`) renormalised
//! weights. Besides the ids and weights it hands the host, the launch leaves `x' =
//! RMSNorm(x) * out_scale` and the two grouped-GEMV group tables on the device, at the
//! addresses [`Router::device_outputs`] returns, so the FP8 gate|up launch
//! (`Fp8Tensor::gemv_grouped_dev`) reads them with no host round trip.
//!
//! Shapes are fixed by the kernel: 128 experts, hidden 2048, top-8, at most [`MAX_LAYERS`]
//! layers. Without the native library (`is_native()` false) every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;
use std::fmt;

/// Experts the router scores (`OMEGA_ROUTER_EXPERTS`).
pub const EXPERTS: usize = 128;
/// Hidden width (`OMEGA_ROUTER_HIDDEN`).
pub const HIDDEN: usize = 2048;
/// Experts selected per token (`OMEGA_ROUTER_TOPK`).
pub const TOP_K: usize = 8;
/// Most layers one router holds (`OMEGA_GPU_ROUTER_MAX_LAYERS`).
pub const MAX_LAYERS: usize = 64;
/// The x' scale that gives the FP8 grouped GEMV its activation row (2^64).
pub const OUT_SCALE_FP8: f32 = 18_446_744_073_709_551_616.0;

/// True when the router API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_router)
}

/// Mirror of `OmegaGpuRouterInfo`: what the chip did for one call.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct RouterInfo {
    pub layer: u32,
    pub warps: u32,
    pub depth: u32,
    pub threads: u32,
    /// bf16 router bytes the launch streams: `128 * 2048 * 2`.
    pub weight_bytes: u64,
    /// Marker-to-marker chip time.
    pub elapsed_ns: u64,
    /// Whole call.
    pub call_ns: u64,
    pub completion_marker: u32,
    /// C `int`: non-zero when the kernel came from the cache.
    pub kernel_cache_hit: i32,
}

/// Device addresses of the last call's outputs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct DeviceOutputs {
    /// `x' = RMSNorm(x) * out_scale`, f32 `[2048]`.
    pub xout_va: u64,
    /// Group table for the gate|up launch (8 entries of `{w_va, s_va}`).
    pub tab_gu_va: u64,
    /// Group table for the down launch.
    pub tab_dn_va: u64,
}

#[cfg(has_omega_router)]
mod ffi {
    use super::RouterInfo;
    #[repr(C)]
    pub struct OmegaGpuRouter {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_router_alloc(layers: u32, out: *mut *mut OmegaGpuRouter) -> i32;
        pub fn omega_gpu_router_free(r: *mut OmegaGpuRouter);
        pub fn omega_gpu_router_set_layer(
            r: *mut OmegaGpuRouter,
            layer: u32,
            gate_bf16: *const u16,
            norm_w: *const f32,
        ) -> i32;
        pub fn omega_gpu_router_set_directory(
            r: *mut OmegaGpuRouter,
            layer: u32,
            gu: *const u64,
            dn: *const u64,
        ) -> i32;
        pub fn omega_gpu_router(
            r: *mut OmegaGpuRouter,
            layer: u32,
            x: *const f32,
            eps: f32,
            norm_topk: i32,
            out_scale: f32,
            ids: *mut u32,
            weights: *mut f32,
            info: *mut RouterInfo,
        ) -> i32;
        pub fn omega_gpu_router_device_outputs(
            r: *const OmegaGpuRouter,
            xout_va: *mut u64,
            tab_gu_va: *mut u64,
            tab_dn_va: *mut u64,
        );
        pub fn omega_gpu_router_read_back(
            r: *mut OmegaGpuRouter,
            xout: *mut f32,
            tab_gu: *mut u64,
            tab_dn: *mut u64,
        ) -> i32;
        pub fn omega_gpu_router_last_error() -> *const std::os::raw::c_char;
        pub fn omega_gpu_router_rc_name(rc: i32) -> *const std::os::raw::c_char;
        pub fn omega_gpu_router_test_set_mutant(mutant: i32);
    }
}

#[cfg(has_omega_router)]
fn rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: both return static NUL-terminated strings (or NULL for last_error).
    let name = unsafe {
        let n = ffi::omega_gpu_router_rc_name(rc);
        let n = if n.is_null() {
            String::from("?")
        } else {
            std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned()
        };
        let e = ffi::omega_gpu_router_last_error();
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

/// Host check of the allocation rule, before any chip call (also in the stub).
pub fn check_layers(layers: usize) -> Result<(), OmegaGpuError> {
    if layers == 0 || layers > MAX_LAYERS {
        return Err(bad(format!(
            "router layers={layers} refused: 1 <= layers <= {MAX_LAYERS}"
        )));
    }
    Ok(())
}

/// The device-resident router weights, norm weights and expert directories of every layer plus
/// the launch buffers. Not `Sync`: callers serialise use behind a lock.
pub struct Router {
    #[cfg(has_omega_router)]
    ptr: *mut ffi::OmegaGpuRouter,
    layers: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for Router {}

impl fmt::Debug for Router {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Router {{ layers: {} }}", self.layers)
    }
}

impl Router {
    /// The C handle (`OmegaGpuRouter *`) for another omega API that takes this object (the decode
    /// graph, OWN-30B cut 2). Null in the stub. The handle stays owned by `self`: the caller must
    /// not free it, must not use it after `self` is dropped, and must serialise its use with
    /// `self`'s own calls.
    // SAFETY: C writes through this handle, so it is only handed out under `&mut self`.
    pub fn as_raw(&mut self) -> *mut std::ffi::c_void {
        #[cfg(has_omega_router)]
        {
            self.ptr as *mut std::ffi::c_void
        }
        #[cfg(not(has_omega_router))]
        {
            std::ptr::null_mut()
        }
    }

    /// Allocates the per-layer tables and launch buffers on the chip.
    pub fn alloc(layers: usize) -> Result<Self, OmegaGpuError> {
        check_layers(layers)?;
        #[cfg(has_omega_router)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: layers checked; omega writes the handle on success.
            let rc = unsafe { ffi::omega_gpu_router_alloc(layers as u32, &mut ptr) };
            if rc != 0 || ptr.is_null() {
                return Err(rc_err(rc));
            }
            Ok(Self { ptr, layers })
        }
        #[cfg(not(has_omega_router))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    fn check_layer(&self, layer: usize) -> Result<(), OmegaGpuError> {
        if layer >= self.layers {
            return Err(bad(format!(
                "router layer {layer} >= layers {}",
                self.layers
            )));
        }
        Ok(())
    }

    /// The layer's `mlp.gate.weight` (`[128][2048]` bf16 bits) and `post_attention_layernorm`
    /// weight (`[2048]` f32, widened by the caller).
    pub fn set_layer(
        &mut self,
        layer: usize,
        gate_bf16: &[u16],
        norm_w: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("router gate words", EXPERTS * HIDDEN, gate_bf16.len())?;
        check_len("router norm weight", HIDDEN, norm_w.len())?;
        self.check_layer(layer)?;
        #[cfg(has_omega_router)]
        {
            // SAFETY: lengths and layer checked; omega copies before returning.
            let rc = unsafe {
                ffi::omega_gpu_router_set_layer(
                    self.ptr,
                    layer as u32,
                    gate_bf16.as_ptr(),
                    norm_w.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_router))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// The layer's expert directories: `gu` and `dn` are `[128][2]` `{w_va, s_va}` (flat, 256
    /// words each), one pair per expert of the resident gate|up and down FP8 tensors.
    pub fn set_directory(
        &mut self,
        layer: usize,
        gu: &[u64],
        dn: &[u64],
    ) -> Result<(), OmegaGpuError> {
        check_len("router gate|up directory", EXPERTS * 2, gu.len())?;
        check_len("router down directory", EXPERTS * 2, dn.len())?;
        self.check_layer(layer)?;
        #[cfg(has_omega_router)]
        {
            // SAFETY: lengths and layer checked; omega copies before returning.
            let rc = unsafe {
                ffi::omega_gpu_router_set_directory(
                    self.ptr,
                    layer as u32,
                    gu.as_ptr(),
                    dn.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_router))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// One launch for `layer`: routes the residual row `x` (`[2048]` f32). Writes the 8 chosen
    /// expert ids (descending by logit) and their weights to the host and `x' = RMSNorm(x) *
    /// out_scale` plus both group tables to the device ([`Self::device_outputs`]).
    #[allow(clippy::too_many_arguments)]
    pub fn route(
        &mut self,
        layer: usize,
        x: &[f32],
        eps: f32,
        norm_topk: bool,
        out_scale: f32,
        ids: &mut [u32],
        weights: &mut [f32],
    ) -> Result<RouterInfo, OmegaGpuError> {
        check_len("router x", HIDDEN, x.len())?;
        check_len("router ids", TOP_K, ids.len())?;
        check_len("router weights", TOP_K, weights.len())?;
        self.check_layer(layer)?;
        #[cfg(has_omega_router)]
        {
            let mut info = RouterInfo::default();
            // SAFETY: lengths and layer checked; omega fills ids and weights before returning.
            let rc = unsafe {
                ffi::omega_gpu_router(
                    self.ptr,
                    layer as u32,
                    x.as_ptr(),
                    eps,
                    norm_topk as i32,
                    out_scale,
                    ids.as_mut_ptr(),
                    weights.as_mut_ptr(),
                    &mut info,
                )
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_router))]
        {
            let _ = (eps, norm_topk, out_scale);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Device addresses of the outputs of the last [`Self::route`] (fixed for the router's life).
    pub fn device_outputs(&self) -> DeviceOutputs {
        #[cfg(has_omega_router)]
        {
            let mut o = DeviceOutputs::default();
            // SAFETY: the router is live for &self; the three out pointers are valid.
            unsafe {
                ffi::omega_gpu_router_device_outputs(
                    self.ptr,
                    &mut o.xout_va,
                    &mut o.tab_gu_va,
                    &mut o.tab_dn_va,
                )
            };
            o
        }
        #[cfg(not(has_omega_router))]
        {
            DeviceOutputs::default()
        }
    }

    /// Test and trace helper: copies `x'` of the last call to the host.
    pub fn read_xout(&mut self, xout: &mut [f32]) -> Result<(), OmegaGpuError> {
        check_len("router xout", HIDDEN, xout.len())?;
        #[cfg(has_omega_router)]
        {
            // SAFETY: length checked; the two table pointers are NULL (not wanted).
            let rc = unsafe {
                ffi::omega_gpu_router_read_back(
                    self.ptr,
                    xout.as_mut_ptr(),
                    std::ptr::null_mut(),
                    std::ptr::null_mut(),
                )
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_router))]
        {
            let _ = xout;
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for Router {
    fn drop(&mut self) {
        #[cfg(has_omega_router)]
        // SAFETY: ptr came from alloc and is freed once.
        unsafe {
            ffi::omega_gpu_router_free(self.ptr)
        }
    }
}

/// TEST ONLY: deliberately wrong router kernel (0 off; 1 RMSNorm without the weight; 2 the
/// mask after each argmax round removes expert `id ^ 1`; 3 the directory entries copied are
/// those of `id ^ 1`; 4 the weights are not normalised). A parity test must catch each.
/// Clears the kernel cache.
pub fn test_set_mutant(mutant: i32) {
    #[cfg(has_omega_router)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_router_test_set_mutant(mutant)
    }
    #[cfg(not(has_omega_router))]
    let _ = mutant;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_rule_is_checked_on_the_host() {
        assert!(check_layers(48).is_ok());
        assert!(check_layers(0).is_err());
        assert!(check_layers(MAX_LAYERS + 1).is_err());
    }

    #[test]
    fn lengths_are_checked_before_the_chip() {
        // Without a chip alloc is Unavailable or refused by shape, never a C call.
        assert!(Router::alloc(0).is_err());
        assert_eq!(EXPERTS * HIDDEN * 2, 524_288);
        assert_eq!(OUT_SCALE_FP8, 2f32.powi(64));
    }

    #[test]
    fn info_layout_matches_the_c_struct() {
        // Measured with cc against omega_gpu_router_api.h (own30b/integration 3c12866a):
        // sizeof 48, warps at 4, depth at 8, threads at 12, weight_bytes at 16, elapsed_ns at 24,
        // call_ns at 32, completion_marker at 40, kernel_cache_hit (int) at 44.
        assert_eq!(std::mem::size_of::<RouterInfo>(), 48);
        assert_eq!(std::mem::offset_of!(RouterInfo, warps), 4);
        assert_eq!(std::mem::offset_of!(RouterInfo, depth), 8);
        assert_eq!(std::mem::offset_of!(RouterInfo, threads), 12);
        assert_eq!(std::mem::offset_of!(RouterInfo, weight_bytes), 16);
        assert_eq!(std::mem::offset_of!(RouterInfo, elapsed_ns), 24);
        assert_eq!(std::mem::offset_of!(RouterInfo, call_ns), 32);
        assert_eq!(std::mem::offset_of!(RouterInfo, completion_marker), 40);
        assert_eq!(std::mem::offset_of!(RouterInfo, kernel_cache_hit), 44);
    }
}
