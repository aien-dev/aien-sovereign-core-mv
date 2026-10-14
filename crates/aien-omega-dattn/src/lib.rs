//! Safe Rust wrapper over omega's one-launch decode attention for Qwen3-30B-A3B
//! (`src/omega_gpu_decode_attn_api.h`, OWN-30B cut 8 on the omega side; no CUDA).
//!
//! One call per layer per token does what the split path did in one launch plus host work:
//! per-head RMSNorm of q and k, RoPE from a chip-resident table, the append of the new K/V
//! (bf16) into a GPU-resident cache, and the softmax attention over the whole context. The
//! cache ([`DecodeAttnCache`]) owns the KV of every layer; the caller keeps positions in
//! order (`0..pos` must be in the cache for that layer before the call at `pos`).
//!
//! Shapes are fixed by the kernel: 32 query heads, 4 KV heads, head_dim 128, so the input row
//! is [`QKV_WORDS`] f32 (`q[32][128] | k[4][128] | v[4][128]`, exactly what the fused q/k/v
//! GEMV writes) and the output is `32 * 128` f32. `max_pos` must be a multiple of 256 and at
//! most [`MAX_POS`]; `layers` at most [`MAX_LAYERS`]. Without the native library
//! (`is_native()` false) every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;
use std::fmt;
pub mod raw;

/// Query heads the kernel is built for (`OMEGA_GPU_DATTN_Q_HEADS`).
pub const Q_HEADS: usize = 32;
/// KV heads (`OMEGA_GPU_DATTN_KV_HEADS`).
pub const KV_HEADS: usize = 4;
/// Head width.
pub const HEAD_DIM: usize = 128;
/// f32 words of one input row (`OMEGA_GPU_DATTN_QKV_WORDS`).
pub const QKV_WORDS: usize = Q_HEADS * HEAD_DIM + 2 * KV_HEADS * HEAD_DIM;
/// f32 words of one output row.
pub const OUT_WORDS: usize = Q_HEADS * HEAD_DIM;
/// Positions per split CTA (`OMEGA_DATTN_CHUNK`); `max_pos` is rounded to this.
pub const POS_ROUND: usize = 256;
/// Most positions one cache can hold (`OMEGA_GPU_DATTN_MAX_POS`).
pub const MAX_POS: usize = 65536;
/// Most layers one cache can hold (`OMEGA_GPU_DATTN_MAX_LAYERS`).
pub const MAX_LAYERS: usize = 64;

/// True when the decode attention API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_dattn)
}

/// Mirror of `OmegaGpuDecodeAttnInfo`: what the chip did for one call.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct DecodeAttnInfo {
    pub layer: u32,
    pub pos: u32,
    pub splits: u32,
    pub per_split: u32,
    pub depth: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub threads: u32,
    /// K + V bytes the launch streams: `2 * 4 * (pos + 1) * 256`.
    pub kv_bytes: u64,
    /// Marker-to-marker chip time.
    pub elapsed_ns: u64,
    /// Whole call.
    pub call_ns: u64,
    pub completion_marker: u32,
    /// C `int`: non-zero when the kernel came from the cache.
    pub kernel_cache_hit: i32,
}

#[cfg(has_omega_dattn)]
mod ffi {
    use super::DecodeAttnInfo;
    #[repr(C)]
    pub struct OmegaGpuDecodeAttnCache {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_decode_attn_cache_alloc(
            layers: u32,
            max_pos: u32,
            rope_theta: f64,
            out: *mut *mut OmegaGpuDecodeAttnCache,
        ) -> i32;
        pub fn omega_gpu_decode_attn_cache_free(c: *mut OmegaGpuDecodeAttnCache);
        pub fn omega_gpu_decode_attn_set_norms(
            c: *mut OmegaGpuDecodeAttnCache,
            layer: u32,
            q_norm: *const f32,
            k_norm: *const f32,
        ) -> i32;
        pub fn omega_gpu_decode_attn(
            c: *mut OmegaGpuDecodeAttnCache,
            layer: u32,
            pos: u32,
            qkv: *const f32,
            eps: f32,
            out_scale: f32,
            out: *mut f32,
            info: *mut DecodeAttnInfo,
        ) -> i32;
        pub fn omega_gpu_decode_attn_last_error() -> *const std::os::raw::c_char;
        pub fn omega_gpu_decode_attn_rc_name(rc: i32) -> *const std::os::raw::c_char;
        pub fn omega_gpu_decode_attn_set_depth(depth: u32);
        pub fn omega_gpu_decode_attn_set_per_split(per_split: u32);
        pub fn omega_gpu_decode_attn_test_set_mutant(mutant: i32);
    }
}

#[cfg(has_omega_dattn)]
fn rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: both return static NUL-terminated strings (or NULL for last_error).
    let name = unsafe {
        let n = ffi::omega_gpu_decode_attn_rc_name(rc);
        let n = if n.is_null() { String::from("?") } else { std::ffi::CStr::from_ptr(n).to_string_lossy().into_owned() };
        let e = ffi::omega_gpu_decode_attn_last_error();
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

/// Host check of the allocation rule, before any chip call (also in the stub).
pub fn check_shape(layers: usize, max_pos: usize) -> Result<(), OmegaGpuError> {
    if layers == 0 || layers > MAX_LAYERS || max_pos == 0 || max_pos > MAX_POS || !max_pos.is_multiple_of(POS_ROUND) {
        return Err(bad(format!(
            "decode attn cache layers={layers} max_pos={max_pos} refused: 1 <= layers <= {MAX_LAYERS}, max_pos a multiple of {POS_ROUND} up to {MAX_POS}"
        )));
    }
    Ok(())
}

/// Rounds a context length up to the cache granule, capped at [`MAX_POS`].
pub fn round_max_pos(max_context: usize) -> usize {
    max_context.div_ceil(POS_ROUND).clamp(1, MAX_POS / POS_ROUND) * POS_ROUND
}

/// Bytes the chip holds for a cache of `layers` x `max_pos` (K and V planes, without the
/// small RoPE / norm / partial buffers).
pub fn kv_bytes(layers: usize, max_pos: usize) -> usize {
    layers * 2 * KV_HEADS * max_pos * HEAD_DIM * 2
}

/// The GPU-resident KV of every layer plus the RoPE table and norm weights. Not `Sync`:
/// callers serialise use behind a lock.
pub struct DecodeAttnCache {
    #[cfg(has_omega_dattn)]
    ptr: *mut ffi::OmegaGpuDecodeAttnCache,
    layers: usize,
    max_pos: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for DecodeAttnCache {}

impl fmt::Debug for DecodeAttnCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DecodeAttnCache {{ layers: {}, max_pos: {} }}", self.layers, self.max_pos)
    }
}

impl DecodeAttnCache {
    /// The C handle (`OmegaGpuDecodeAttnCache *`) for another omega API that takes this object (the decode
    /// graph, OWN-30B cut 2). Null in the stub. The handle stays owned by `self`: the caller must
    /// not free it, must not use it after `self` is dropped, and must serialise its use with
    /// `self`'s own calls.
    pub fn as_raw(&self) -> *mut std::ffi::c_void {
        #[cfg(has_omega_dattn)]
        {
            self.ptr as *mut std::ffi::c_void
        }
        #[cfg(not(has_omega_dattn))]
        {
            std::ptr::null_mut()
        }
    }

    /// Allocates the cache on the chip and builds the RoPE table for `rope_theta`.
    pub fn alloc(layers: usize, max_pos: usize, rope_theta: f64) -> Result<Self, OmegaGpuError> {
        check_shape(layers, max_pos)?;
        if rope_theta.partial_cmp(&0.0) != Some(std::cmp::Ordering::Greater) {
            return Err(bad(format!("decode attn rope_theta {rope_theta} must be positive")));
        }
        #[cfg(has_omega_dattn)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: arguments checked above; omega writes the handle on success.
            let rc = unsafe { ffi::omega_gpu_decode_attn_cache_alloc(layers as u32, max_pos as u32, rope_theta, &mut ptr) };
            if rc != 0 || ptr.is_null() {
                return Err(rc_err(rc));
            }
            Ok(Self { ptr, layers, max_pos })
        }
        #[cfg(not(has_omega_dattn))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }
    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// The per-head RMSNorm weights of `layer` (`q_norm`, `k_norm`: 128 f32 each).
    pub fn set_norms(&mut self, layer: usize, q_norm: &[f32], k_norm: &[f32]) -> Result<(), OmegaGpuError> {
        check_len("decode attn q_norm", HEAD_DIM, q_norm.len())?;
        check_len("decode attn k_norm", HEAD_DIM, k_norm.len())?;
        if layer >= self.layers {
            return Err(bad(format!("decode attn layer {layer} >= layers {}", self.layers)));
        }
        #[cfg(has_omega_dattn)]
        {
            // SAFETY: lengths checked; the cache is live for &mut self.
            let rc = unsafe { ffi::omega_gpu_decode_attn_set_norms(self.ptr, layer as u32, q_norm.as_ptr(), k_norm.as_ptr()) };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_dattn))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// One layer step for the token at `pos`: appends its K/V and writes the attention output
    /// (`out[32 * 128] * out_scale`). `qkv` is the raw fused projection row (pre-norm, pre-rope).
    pub fn attend(
        &mut self,
        layer: usize,
        pos: usize,
        qkv: &[f32],
        eps: f32,
        out_scale: f32,
        out: &mut [f32],
    ) -> Result<DecodeAttnInfo, OmegaGpuError> {
        check_len("decode attn qkv", QKV_WORDS, qkv.len())?;
        check_len("decode attn out", OUT_WORDS, out.len())?;
        if layer >= self.layers {
            return Err(bad(format!("decode attn layer {layer} >= layers {}", self.layers)));
        }
        if pos >= self.max_pos {
            return Err(bad(format!("decode attn pos {pos} >= max_pos {}", self.max_pos)));
        }
        #[cfg(has_omega_dattn)]
        {
            let mut info = DecodeAttnInfo::default();
            // SAFETY: lengths and ranges checked; the cache is live for &mut self.
            let rc = unsafe {
                ffi::omega_gpu_decode_attn(self.ptr, layer as u32, pos as u32, qkv.as_ptr(), eps, out_scale, out.as_mut_ptr(), &mut info)
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            Ok(info)
        }
        #[cfg(not(has_omega_dattn))]
        {
            let _ = (eps, out_scale, out);
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for DecodeAttnCache {
    fn drop(&mut self) {
        #[cfg(has_omega_dattn)]
        // SAFETY: ptr came from alloc and is freed once.
        unsafe {
            ffi::omega_gpu_decode_attn_cache_free(self.ptr)
        }
    }
}

/// Prefetch depth of the K and V streams, 1..4 (0 = the default); clears the kernel cache.
pub fn set_depth(depth: u32) {
    #[cfg(has_omega_dattn)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_decode_attn_set_depth(depth)
    }
    #[cfg(not(has_omega_dattn))]
    let _ = depth;
}

/// Positions per split CTA, a multiple of 256 (0 = the default).
pub fn set_per_split(per_split: u32) {
    #[cfg(has_omega_dattn)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_decode_attn_set_per_split(per_split)
    }
    #[cfg(not(has_omega_dattn))]
    let _ = per_split;
}

/// TEST ONLY: deliberately wrong kernel (0 off; 1 no RoPE on q and k; 2 V read one position
/// late; 3 the new K/V not written to the cache; 4 cross-split combine ignores the max). A
/// parity test must catch the one it sets. Clears the kernel cache.
pub fn test_set_mutant(mutant: i32) {
    #[cfg(has_omega_dattn)]
    // SAFETY: plain integer argument.
    unsafe {
        ffi::omega_gpu_decode_attn_test_set_mutant(mutant)
    }
    #[cfg(not(has_omega_dattn))]
    let _ = mutant;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_rule_is_checked_on_the_host() {
        assert!(check_shape(48, 32768).is_ok());
        assert!(check_shape(0, 256).is_err());
        assert!(check_shape(65, 256).is_err());
        assert!(check_shape(1, 100).is_err());
        assert!(check_shape(1, MAX_POS + 256).is_err());
        assert_eq!(round_max_pos(1), 256);
        assert_eq!(round_max_pos(8192), 8192);
        assert_eq!(round_max_pos(8193), 8448);
        assert_eq!(round_max_pos(1 << 20), MAX_POS);
    }

    #[test]
    fn lengths_are_checked_before_the_chip() {
        // Without a chip the constructor is Unavailable or refused by shape, never a C call.
        assert!(DecodeAttnCache::alloc(1, 100, 1e7).is_err());
        assert_eq!(QKV_WORDS, 5120);
        assert_eq!(OUT_WORDS, 4096);
        assert_eq!(kv_bytes(48, 32768), 48 * 2 * 4 * 32768 * 256);
    }

    #[test]
    fn info_layout_matches_the_c_struct() {
        // Measured with cc against omega_gpu_decode_attn_api.h (own30b/integration 831001cc):
        // sizeof 64, kv_bytes at 32, elapsed_ns at 40, call_ns at 48, completion_marker at 56,
        // kernel_cache_hit (int) at 60.
        assert_eq!(std::mem::size_of::<DecodeAttnInfo>(), 64);
        assert_eq!(std::mem::offset_of!(DecodeAttnInfo, kv_bytes), 32);
        assert_eq!(std::mem::offset_of!(DecodeAttnInfo, elapsed_ns), 40);
        assert_eq!(std::mem::offset_of!(DecodeAttnInfo, call_ns), 48);
        assert_eq!(std::mem::offset_of!(DecodeAttnInfo, completion_marker), 56);
        assert_eq!(std::mem::offset_of!(DecodeAttnInfo, kernel_cache_hit), 60);
    }
}
