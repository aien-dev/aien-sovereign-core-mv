//! Safe-ish Rust wrapper over omega's one-command-list decode token for Qwen3-30B-A3B
//! (`src/omega_gpu_decode_graph.h`, OWN-30B L2 cut 2 on the omega side; no CUDA).
//!
//! One call ([`DecodeGraph::token`]) records every layer (input norm, q|k|v FP8, decode
//! attention, o FP8, residual add, router, gate|up FP8, SwiGLU, down FP8, combine) plus the
//! final norm, the bf16 lm_head and a two-stage argmax into ONE omega command list and submits
//! it once. The residual row stays on the device for the whole token; the host writes the
//! token's embedding row before and reads the greedy token after.
//!
//! The graph does not own the weights, the router or the KV cache: it takes the C handles of
//! objects the caller already uploaded (`as_raw()` of `aien_omega_fp8::Fp8Tensor`,
//! `aien_omega_router::Router`, `aien_omega_dattn::DecodeAttnCache`,
//! `aien_omega_bf16::Bf16Tensor`). Those objects must outlive the graph and must not be used
//! concurrently with a [`DecodeGraph::token`] call. Without the native library
//! (`is_native()` false) every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;
use std::ffi::c_void;
use std::fmt;

/// Residual width (`OMEGA_GPU_DGRAPH_HIDDEN`).
pub const HIDDEN: usize = 2048;
/// Most layers (`OMEGA_GPU_DGRAPH_MAX_LAYERS`).
pub const MAX_LAYERS: usize = 64;
/// Largest vocabulary the two-stage argmax covers (`2048 * 2048`).
pub const MAX_VOCAB: usize = 2048 * 2048;

/// True when the decode graph API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_dgraph)
}

/// Launch kinds of the timeline (`OMEGA_GPU_DGRAPH_K_*`), in enum order.
pub const KIND_NAMES: [&str; 12] = ["norm", "qkv", "attn", "o_proj", "res_add", "router", "gate_up", "swiglu", "down", "combine_norm", "lm_head", "argmax"];

/// omega's own name for a launch kind (`omega_gpu_decode_graph_kind_name`); the table above
/// in the stub.
pub fn kind_name(kind: u8) -> String {
    #[cfg(has_omega_dgraph)]
    {
        // SAFETY: returns a static NUL-terminated string or NULL.
        let p = unsafe { ffi::omega_gpu_decode_graph_kind_name(kind) };
        if !p.is_null() {
            // SAFETY: non-null static string.
            return unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned();
        }
    }
    KIND_NAMES.get(kind as usize).map_or_else(|| format!("kind{kind}"), |s| s.to_string())
}

/// Mirror of `OmegaGpuDecodeGraphStats`: what one token cost.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct GraphStats {
    /// Launches recorded for the token.
    pub launches: u32,
    /// Submissions (1 normally).
    pub submits: u32,
    /// Chip time of the submission(s): first submit to last marker.
    pub chip_ns: u64,
    /// Host: begin_token to the submit (recording).
    pub record_ns: u64,
    /// Host: begin_token to end_token return.
    pub total_ns: u64,
    /// Launch semaphores found DONE.
    pub sems_done: u32,
}

/// Mirror of `OmegaGpuDecodeGraphConfig`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GraphConfig {
    pub layers: u32,
    pub vocab: u32,
    pub eps: f32,
    pub norm_topk: i32,
    pub router: *mut c_void,
    pub attn: *mut c_void,
    pub lm_head: *const c_void,
}

#[cfg(has_omega_dgraph)]
mod ffi {
    use super::{GraphConfig, GraphStats};
    use std::ffi::c_void;
    #[repr(C)]
    pub struct OmegaGpuDecodeGraph {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_decode_graph_create(cfg: *const GraphConfig, out: *mut *mut OmegaGpuDecodeGraph) -> i32;
        pub fn omega_gpu_decode_graph_free(g: *mut OmegaGpuDecodeGraph);
        pub fn omega_gpu_decode_graph_set_layer(
            g: *mut OmegaGpuDecodeGraph,
            layer: u32,
            qkv: *const c_void,
            o: *const c_void,
            gate_up: *const c_void,
            down: *const c_void,
            input_norm: *const f32,
        ) -> i32;
        pub fn omega_gpu_decode_graph_set_final_norm(g: *mut OmegaGpuDecodeGraph, final_norm: *const f32) -> i32;
        pub fn omega_gpu_decode_graph_token(
            g: *mut OmegaGpuDecodeGraph,
            x_embed: *const f32,
            pos: u32,
            token: *mut u32,
            st: *mut GraphStats,
        ) -> i32;
        pub fn omega_gpu_decode_graph_read(g: *mut OmegaGpuDecodeGraph, x: *mut f32, hidden: *mut f32, logits: *mut f32) -> i32;
        pub fn omega_gpu_decode_graph_set_flush_each(g: *mut OmegaGpuDecodeGraph, on: i32);
        pub fn omega_gpu_decode_graph_test_set_mutant(g: *mut OmegaGpuDecodeGraph, mutant: i32);
        pub fn omega_gpu_decode_graph_set_stamps(g: *mut OmegaGpuDecodeGraph, on: i32);
        pub fn omega_gpu_decode_graph_stamps(g: *const OmegaGpuDecodeGraph, stamps: *mut u64, kinds: *mut u8, cap: u32) -> i32;
        pub fn omega_gpu_decode_graph_kind_name(kind: u8) -> *const std::os::raw::c_char;
        pub fn omega_gpu_session_last_error() -> *const std::os::raw::c_char;
    }
}

#[cfg(has_omega_dgraph)]
fn rc_err(stage: &str, rc: i32) -> OmegaGpuError {
    let name = match rc {
        -1 => "BAD_ARGS",
        -3 => "CODEGEN_FAIL",
        -4 => "CHIP_FAIL",
        _ => "?",
    };
    // SAFETY: returns a static NUL-terminated string or NULL.
    let detail = unsafe {
        let e = ffi::omega_gpu_session_last_error();
        if e.is_null() {
            String::new()
        } else {
            std::ffi::CStr::from_ptr(e).to_string_lossy().into_owned()
        }
    };
    OmegaGpuError::Rc { rc, name: format!("decode graph {stage}: {name}: {detail}") }
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

/// Host check of the configuration, before any chip call (also in the stub).
pub fn check_config(cfg: &GraphConfig) -> Result<(), OmegaGpuError> {
    if cfg.layers == 0 || cfg.layers as usize > MAX_LAYERS {
        return Err(bad(format!("decode graph: {} layers, 1..={MAX_LAYERS} supported", cfg.layers)));
    }
    if cfg.vocab == 0 || cfg.vocab as usize > MAX_VOCAB {
        return Err(bad(format!("decode graph: vocab {} outside 1..={MAX_VOCAB}", cfg.vocab)));
    }
    if cfg.router.is_null() || cfg.attn.is_null() || cfg.lm_head.is_null() {
        return Err(bad("decode graph: the router, the KV cache and the lm_head must all be device objects".into()));
    }
    Ok(())
}

/// One recorded-per-token decode graph. Not `Sync`: callers serialise use behind a lock.
pub struct DecodeGraph {
    #[cfg(has_omega_dgraph)]
    ptr: *mut ffi::OmegaGpuDecodeGraph,
    layers: usize,
    vocab: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for DecodeGraph {}

impl fmt::Debug for DecodeGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "DecodeGraph {{ layers: {}, vocab: {} }}", self.layers, self.vocab)
    }
}

impl DecodeGraph {
    /// Allocates the graph's device buffers.
    ///
    /// # Safety
    /// `cfg.router`, `cfg.attn` and `cfg.lm_head` are live omega handles (`as_raw()` of the
    /// matching crates' objects) that outlive the graph.
    pub unsafe fn create(cfg: &GraphConfig) -> Result<Self, OmegaGpuError> {
        check_config(cfg)?;
        #[cfg(has_omega_dgraph)]
        {
            let mut ptr = std::ptr::null_mut();
            let rc = ffi::omega_gpu_decode_graph_create(cfg, &mut ptr);
            if rc != 0 || ptr.is_null() {
                return Err(rc_err("create", rc));
            }
            Ok(Self { ptr, layers: cfg.layers as usize, vocab: cfg.vocab as usize })
        }
        #[cfg(not(has_omega_dgraph))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// Binds one layer's FP8 projections (omega checks the shapes) and copies its
    /// `input_layernorm` weight (f32 `[HIDDEN]`).
    ///
    /// # Safety
    /// The four pointers are live `OmegaGpuFp8Tensor` handles that outlive the graph.
    pub unsafe fn set_layer(
        &mut self,
        layer: usize,
        qkv: *const c_void,
        o: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        input_norm: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("decode graph input_norm", HIDDEN, input_norm.len())?;
        if layer >= self.layers {
            return Err(bad(format!("decode graph: layer {layer} of {}", self.layers)));
        }
        if qkv.is_null() || o.is_null() || gate_up.is_null() || down.is_null() {
            return Err(bad(format!("decode graph layer {layer}: a projection has no device handle")));
        }
        #[cfg(has_omega_dgraph)]
        {
            let rc = ffi::omega_gpu_decode_graph_set_layer(self.ptr, layer as u32, qkv, o, gate_up, down, input_norm.as_ptr());
            if rc != 0 {
                return Err(rc_err("set_layer", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_dgraph))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Copies the final RMSNorm weight (f32 `[HIDDEN]`).
    pub fn set_final_norm(&mut self, final_norm: &[f32]) -> Result<(), OmegaGpuError> {
        check_len("decode graph final_norm", HIDDEN, final_norm.len())?;
        #[cfg(has_omega_dgraph)]
        {
            // SAFETY: ptr is live; the slice holds HIDDEN f32 (checked).
            let rc = unsafe { ffi::omega_gpu_decode_graph_set_final_norm(self.ptr, final_norm.as_ptr()) };
            if rc != 0 {
                return Err(rc_err("set_final_norm", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_dgraph))]
        Err(OmegaGpuError::Unavailable)
    }

    /// One whole token as one command list: `x_embed` (the token's embedding row, f32
    /// `[HIDDEN]`) at position `pos` -> the greedy token (lowest index of the max logit) and
    /// what it cost. The KV cache gets this position's K/V for every layer.
    pub fn token(&mut self, x_embed: &[f32], pos: usize) -> Result<(u32, GraphStats), OmegaGpuError> {
        check_len("decode graph x_embed", HIDDEN, x_embed.len())?;
        let pos = u32::try_from(pos).map_err(|_| bad(format!("decode graph: pos {pos} too large")))?;
        #[cfg(has_omega_dgraph)]
        {
            let mut token = 0u32;
            let mut st = GraphStats::default();
            // SAFETY: ptr is live; x_embed holds HIDDEN f32 (checked); token and st are local.
            let rc = unsafe { ffi::omega_gpu_decode_graph_token(self.ptr, x_embed.as_ptr(), pos, &mut token, &mut st) };
            if rc != 0 {
                return Err(rc_err("token", rc));
            }
            Ok((token, st))
        }
        #[cfg(not(has_omega_dgraph))]
        {
            let _ = pos;
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Copies the last token's device residual `x`, final-normed `hidden` (both `[HIDDEN]`)
    /// and `logits` (`[vocab]`); any of them may be `None`. Costs a device read.
    pub fn read(&mut self, x: Option<&mut [f32]>, hidden: Option<&mut [f32]>, logits: Option<&mut [f32]>) -> Result<(), OmegaGpuError> {
        if let Some(x) = &x {
            check_len("decode graph read x", HIDDEN, x.len())?;
        }
        if let Some(h) = &hidden {
            check_len("decode graph read hidden", HIDDEN, h.len())?;
        }
        if let Some(l) = &logits {
            check_len("decode graph read logits", self.vocab, l.len())?;
        }
        #[cfg(has_omega_dgraph)]
        {
            let p = |s: Option<&mut [f32]>| s.map_or(std::ptr::null_mut(), |s| s.as_mut_ptr());
            // SAFETY: ptr is live; each non-null buffer has the length omega copies (checked).
            let rc = unsafe { ffi::omega_gpu_decode_graph_read(self.ptr, p(x), p(hidden), p(logits)) };
            if rc != 0 {
                return Err(rc_err("read", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_dgraph))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Timeline (measurement): when on, every later token also records a GPU timestamp before
    /// the first launch and after each launch. Call between tokens. Off by default.
    pub fn set_stamps(&mut self, on: bool) {
        #[cfg(has_omega_dgraph)]
        // SAFETY: ptr is live; plain integer argument.
        unsafe {
            ffi::omega_gpu_decode_graph_set_stamps(self.ptr, on as i32)
        }
        #[cfg(not(has_omega_dgraph))]
        let _ = on;
    }

    /// The last token's timeline: `launches + 1` GPU timestamps (ns) and one kind per launch
    /// (launch `i` took `stamps[i + 1] - stamps[i]`). Refused when stamps are off or the token failed.
    pub fn stamps(&self) -> Result<(Vec<u64>, Vec<u8>), OmegaGpuError> {
        #[cfg(has_omega_dgraph)]
        {
            // 9 per layer + 4, with room to spare.
            let cap = (MAX_LAYERS * 16 + 16) as u32;
            let mut stamps = vec![0u64; cap as usize + 1];
            let mut kinds = vec![0u8; cap as usize + 1];
            // SAFETY: ptr is live; both buffers hold cap + 1 entries.
            let n = unsafe { ffi::omega_gpu_decode_graph_stamps(self.ptr, stamps.as_mut_ptr(), kinds.as_mut_ptr(), cap + 1) };
            if n < 0 {
                return Err(rc_err("stamps", n));
            }
            let n = n as usize;
            stamps.truncate(n + 1);
            kinds.truncate(n);
            Ok((stamps, kinds))
        }
        #[cfg(not(has_omega_dgraph))]
        Err(OmegaGpuError::Unavailable)
    }

    /// TEST ONLY: submit after every launch (the batching comparison).
    pub fn set_flush_each(&mut self, on: bool) {
        #[cfg(has_omega_dgraph)]
        // SAFETY: ptr is live; plain integer argument.
        unsafe {
            ffi::omega_gpu_decode_graph_set_flush_each(self.ptr, on as i32)
        }
        #[cfg(not(has_omega_dgraph))]
        let _ = on;
    }

    /// TEST ONLY: omega's deliberate wrong kernels (0 off; 1 SwiGLU reads the wrong half;
    /// 2 combine skips expert 7; 3 attention written at scale 1 instead of 2^64). A token
    /// check must catch the one it sets.
    pub fn test_set_mutant(&mut self, mutant: i32) {
        #[cfg(has_omega_dgraph)]
        // SAFETY: ptr is live; plain integer argument.
        unsafe {
            ffi::omega_gpu_decode_graph_test_set_mutant(self.ptr, mutant)
        }
        #[cfg(not(has_omega_dgraph))]
        let _ = mutant;
    }
}

impl Drop for DecodeGraph {
    fn drop(&mut self) {
        #[cfg(has_omega_dgraph)]
        // SAFETY: ptr came from create and is freed once.
        unsafe {
            ffi::omega_gpu_decode_graph_free(self.ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(layers: u32, vocab: u32, live: bool) -> GraphConfig {
        let p = if live { 8usize as *mut c_void } else { std::ptr::null_mut() };
        GraphConfig { layers, vocab, eps: 1e-6, norm_topk: 1, router: p, attn: p, lm_head: p }
    }

    #[test]
    fn config_is_checked_on_the_host() {
        assert!(check_config(&cfg(48, 151936, true)).is_ok());
        assert!(check_config(&cfg(0, 151936, true)).is_err());
        assert!(check_config(&cfg(65, 151936, true)).is_err());
        assert!(check_config(&cfg(48, 0, true)).is_err());
        assert!(check_config(&cfg(48, (MAX_VOCAB + 1) as u32, true)).is_err());
        assert!(check_config(&cfg(48, 151936, false)).is_err());
    }

    #[test]
    fn create_refuses_null_handles_before_any_c_call() {
        // SAFETY: the null handles are refused by check_config before the C call.
        let r = unsafe { DecodeGraph::create(&cfg(48, 151936, false)) };
        assert!(r.is_err());
    }

    #[test]
    fn layouts_match_the_c_structs() {
        // Measured with cc against omega_gpu_decode_graph.h (omega d3f465dc): config sizeof 40,
        // layers 0 vocab 4 eps 8 norm_topk 12 router 16 attn 24 lm_head 32; stats sizeof 40,
        // launches 0 submits 4 chip_ns 8 record_ns 16 total_ns 24 sems_done 32.
        assert_eq!(std::mem::size_of::<GraphConfig>(), 40);
        assert_eq!(std::mem::offset_of!(GraphConfig, eps), 8);
        assert_eq!(std::mem::offset_of!(GraphConfig, norm_topk), 12);
        assert_eq!(std::mem::offset_of!(GraphConfig, router), 16);
        assert_eq!(std::mem::offset_of!(GraphConfig, attn), 24);
        assert_eq!(std::mem::offset_of!(GraphConfig, lm_head), 32);
        assert_eq!(std::mem::size_of::<GraphStats>(), 40);
        assert_eq!(std::mem::offset_of!(GraphStats, submits), 4);
        assert_eq!(std::mem::offset_of!(GraphStats, chip_ns), 8);
        assert_eq!(std::mem::offset_of!(GraphStats, record_ns), 16);
        assert_eq!(std::mem::offset_of!(GraphStats, total_ns), 24);
        assert_eq!(std::mem::offset_of!(GraphStats, sems_done), 32);
    }

    /// Native binding smoke without the chip: omega's own argument checks answer BAD_ARGS
    /// before touching any handle or the device.
    #[cfg(has_omega_dgraph)]
    #[test]
    fn native_symbols_link_and_refuse_bad_args_without_the_chip() {
        let c = cfg(0, 151936, true);
        let mut out = std::ptr::null_mut();
        // SAFETY: layers == 0 is refused by omega before any handle is read (omega_gpu_decode_graph.c create).
        let rc = unsafe { ffi::omega_gpu_decode_graph_create(&c, &mut out) };
        assert_eq!(rc, -1);
        assert!(out.is_null());
        let norm = [0.0f32; HIDDEN];
        // SAFETY: a NULL graph is refused first.
        assert_eq!(unsafe { ffi::omega_gpu_decode_graph_set_final_norm(std::ptr::null_mut(), norm.as_ptr()) }, -1);
        let e = rc_err("smoke", -1).to_string();
        assert!(e.contains("BAD_ARGS"), "{e}");
        // omega's kind names are the table's, in enum order.
        for (k, want) in KIND_NAMES.iter().enumerate() {
            assert_eq!(kind_name(k as u8), *want);
        }
        assert_eq!(kind_name(12), "?");
    }

    #[test]
    fn native_flag_matches_the_cfg() {
        assert_eq!(is_native(), cfg!(has_omega_dgraph));
    }
}
