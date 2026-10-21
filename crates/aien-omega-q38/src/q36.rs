//! Qwen3.6-35B-A3B decode graph (omega `src/omega_gpu_q36_graph.h`, sc L306): the 27B graph's
//! mixers at the 35B shapes (DeltaNet ratio 2, attention 16 q / 2 kv heads) and the 30B's sparse
//! block with the shared expert as the ninth group. One whole token per [`Q36Graph::token`].
//!
//! As [`crate::Q38Graph`], the graph owns neither the weights nor the state: it takes the C
//! handles of a [`crate::Gdn`] (ratio 2), an [`crate::Attn256Cache`] (shape 16 / 2 / 4), an
//! `aien_omega_router::Router` (shape 256 + 1 shared, every layer and directory set) and the FP8 /
//! bf16 tensors, which must outlive the graph and not be used concurrently with a token.
use crate::{bad, check_len, session_err, GraphStats};
use aien_omega_gpu::OmegaGpuError;
use std::ffi::c_void;
use std::fmt;

/// Residual width (`OMEGA_GPU_Q36_HIDDEN`).
pub const HIDDEN: usize = 2048;
/// Routed and shared expert width (`OMEGA_GPU_Q36_EXPERT_INTER`).
pub const EXPERT_INTER: usize = 512;
/// Expert groups per token: 8 routed + the shared one.
pub const GROUPS: usize = 9;
/// Experts in each layer's gate_up / down tensor: 256 routed, then the shared expert.
pub const EXPERT_ROWS: usize = 257;
/// DeltaNet fused projection rows: in_proj_qkv 8192 | in_proj_z 4096.
pub const GDN_MIX: usize = 12288;
/// DeltaNet in_proj_qkv rows (the conv width).
pub const GDN_QKV: usize = 8192;
/// DeltaNet a|b rows: 32 heads x {a, b}, interleaved.
pub const AB: usize = 64;
/// Attention fused projection rows: q_proj 8192 (16 x [q 256 | gate 256]) | k 512 | v 512.
pub const ATT_MIX: usize = 9216;
/// Mixer output width (DeltaNet 32 x 128 = attention 16 x 256): the o_proj input.
pub const MIXER_OUT: usize = 4096;
/// Attention heads: query, key/value, queries per CTA row (`omega_gpu_decode_attn256_cache_alloc_shape`).
pub const Q_HEADS: usize = 16;
pub const KV_HEADS: usize = 2;
pub const Q_PER_CTA: usize = 4;
/// DeltaNet value heads per key head.
pub const GDN_RATIO: usize = 2;

/// The C config (`OmegaGpuQ36GraphConfig`).
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Q36Config {
    pub layers: u32,
    pub attn_mask: u64,
    pub vocab: u32,
    pub eps: f32,
    pub eps_l2: f32,
    pub gdn: *mut c_void,
    pub attn: *mut c_void,
    pub router: *mut c_void,
    pub lm_head: *const c_void,
}

#[cfg(has_omega_q36)]
mod ffi {
    use super::{GraphStats, Q36Config};
    use std::ffi::c_void;
    #[repr(C)]
    pub struct OmegaGpuQ36Graph {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_q36_graph_create(cfg: *const Q36Config, out: *mut *mut OmegaGpuQ36Graph) -> i32;
        pub fn omega_gpu_q36_graph_free(g: *mut OmegaGpuQ36Graph);
        pub fn omega_gpu_q36_graph_set_layer(
            g: *mut OmegaGpuQ36Graph,
            layer: u32,
            mix: *const c_void,
            ab: *const c_void,
            o: *const c_void,
            gate_up: *const c_void,
            down: *const c_void,
            in_norm: *const f32,
        ) -> i32;
        pub fn omega_gpu_q36_graph_set_final_norm(g: *mut OmegaGpuQ36Graph, final_norm: *const f32) -> i32;
        pub fn omega_gpu_q36_graph_token(
            g: *mut OmegaGpuQ36Graph,
            x_embed: *const f32,
            pos: u32,
            token: *mut u32,
            st: *mut GraphStats,
        ) -> i32;
        pub fn omega_gpu_q36_graph_read(
            g: *mut OmegaGpuQ36Graph,
            x: *mut f32,
            hidden: *mut f32,
            logits: *mut f32,
        ) -> i32;
        pub fn omega_gpu_q36_graph_read_buf(g: *mut OmegaGpuQ36Graph, which: i32, out: *mut f32, n: u32) -> i32;
        pub fn omega_gpu_q36_graph_set_flush_each(g: *mut OmegaGpuQ36Graph, on: i32);
        pub fn omega_gpu_q36_graph_test_set_mutant(g: *mut OmegaGpuQ36Graph, mutant: i32);
        pub fn omega_gpu_q36_graph_set_exact_head(g: *mut OmegaGpuQ36Graph, screen: *const c_void, coef: *const f32) -> i32;
        pub fn omega_gpu_q36_graph_set_exact_head_on(g: *mut OmegaGpuQ36Graph, on: i32);
        pub fn omega_gpu_q36_graph_exact_head_active(g: *const OmegaGpuQ36Graph) -> i32;
    }
}

/// True when the 35B graph is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_q36)
}

/// Intermediates [`Q36Graph::read_buf`] can copy (`OMEGA_GPU_Q36_BUF_*`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Q36Buf {
    X = 0,
    Xq,
    Mix,
    Ab,
    Agg,
    Att,
    O,
    Gu,
    Act,
    Hidden,
    Dn,
}

pub struct Q36Graph {
    #[cfg(has_omega_q36)]
    ptr: *mut ffi::OmegaGpuQ36Graph,
    layers: usize,
    attn_mask: u64,
    vocab: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for Q36Graph {}

impl fmt::Debug for Q36Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Q36Graph {{ layers: {}, attn_mask: {:#x}, vocab: {} }}",
            self.layers, self.attn_mask, self.vocab
        )
    }
}

impl Q36Graph {
    /// # Safety
    /// Every handle in `cfg` must be live, of the shape the header names, and outlive the graph.
    pub unsafe fn create(cfg: &Q36Config) -> Result<Self, OmegaGpuError> {
        if cfg.layers == 0 || cfg.layers > 64 || cfg.vocab == 0 {
            return Err(bad(format!("q36 graph: layers {} vocab {}", cfg.layers, cfg.vocab)));
        }
        if cfg.gdn.is_null() || cfg.attn.is_null() || cfg.router.is_null() || cfg.lm_head.is_null() {
            return Err(bad("q36 graph: a state or weight handle is null".into()));
        }
        #[cfg(has_omega_q36)]
        {
            let mut ptr = std::ptr::null_mut();
            let rc = ffi::omega_gpu_q36_graph_create(cfg, &mut ptr);
            if rc != 0 || ptr.is_null() {
                return Err(session_err("q36 graph create", rc));
            }
            Ok(Self { ptr, layers: cfg.layers as usize, attn_mask: cfg.attn_mask, vocab: cfg.vocab as usize })
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    pub fn is_attention(&self, l: usize) -> bool {
        l < 64 && self.attn_mask >> l & 1 == 1
    }

    /// # Safety
    /// The handles must be live FP8 / bf16 tensors of the header's shapes and outlive the graph.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn set_layer(
        &mut self,
        layer: usize,
        mix: *const c_void,
        ab: *const c_void,
        o: *const c_void,
        gate_up: *const c_void,
        down: *const c_void,
        in_norm: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("q36 graph in_norm", HIDDEN, in_norm.len())?;
        if layer >= self.layers {
            return Err(bad(format!("q36 graph: layer {layer} of {}", self.layers)));
        }
        if mix.is_null() || o.is_null() || gate_up.is_null() || down.is_null() {
            return Err(bad(format!("q36 graph layer {layer}: a projection has no device handle")));
        }
        if ab.is_null() != self.is_attention(layer) {
            return Err(bad(format!("q36 graph layer {layer}: the a|b tensor is for DeltaNet layers only")));
        }
        #[cfg(has_omega_q36)]
        {
            let rc = ffi::omega_gpu_q36_graph_set_layer(self.ptr, layer as u32, mix, ab, o, gate_up, down, in_norm.as_ptr());
            if rc != 0 {
                return Err(session_err("q36 graph set_layer", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn set_final_norm(&mut self, final_norm: &[f32]) -> Result<(), OmegaGpuError> {
        check_len("q36 graph final_norm", HIDDEN, final_norm.len())?;
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live; the slice holds HIDDEN f32 (checked).
            let rc = unsafe { ffi::omega_gpu_q36_graph_set_final_norm(self.ptr, final_norm.as_ptr()) };
            if rc != 0 {
                return Err(session_err("q36 graph set_final_norm", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// One whole token: the embedding row in, the greedy token out (position `pos`).
    pub fn token(&mut self, x_embed: &[f32], pos: usize) -> Result<(u32, GraphStats), OmegaGpuError> {
        check_len("q36 graph x_embed", HIDDEN, x_embed.len())?;
        let pos = u32::try_from(pos).map_err(|_| bad(format!("q36 graph: pos {pos} too large")))?;
        #[cfg(has_omega_q36)]
        {
            let mut token = 0u32;
            let mut st = GraphStats::default();
            // SAFETY: ptr is live; x_embed holds HIDDEN f32 (checked); token and st are local.
            let rc = unsafe { ffi::omega_gpu_q36_graph_token(self.ptr, x_embed.as_ptr(), pos, &mut token, &mut st) };
            if rc != 0 {
                return Err(session_err("q36 graph token", rc));
            }
            Ok((token, st))
        }
        #[cfg(not(has_omega_q36))]
        {
            let _ = pos;
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// The last token's residual, final hidden and logits (any may be `None`).
    pub fn read(
        &mut self,
        x: Option<&mut [f32]>,
        hidden: Option<&mut [f32]>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), OmegaGpuError> {
        if let Some(x) = &x {
            check_len("q36 graph read x", HIDDEN, x.len())?;
        }
        if let Some(h) = &hidden {
            check_len("q36 graph read hidden", HIDDEN, h.len())?;
        }
        if let Some(l) = &logits {
            check_len("q36 graph read logits", self.vocab, l.len())?;
        }
        #[cfg(has_omega_q36)]
        {
            let p = |s: Option<&mut [f32]>| s.map_or(std::ptr::null_mut(), |s| s.as_mut_ptr());
            // SAFETY: ptr is live; every non-null pointer holds the length checked above.
            let rc = unsafe { ffi::omega_gpu_q36_graph_read(self.ptr, p(x), p(hidden), p(logits)) };
            if rc != 0 {
                return Err(session_err("q36 graph read", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// One intermediate of the last token (the first `out.len()` values).
    pub fn read_buf(&mut self, which: Q36Buf, out: &mut [f32]) -> Result<(), OmegaGpuError> {
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live; omega refuses n above the buffer's capacity.
            let rc = unsafe { ffi::omega_gpu_q36_graph_read_buf(self.ptr, which as i32, out.as_mut_ptr(), out.len() as u32) };
            if rc != 0 {
                return Err(session_err("q36 graph read_buf", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        {
            let _ = (which, out);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// HUM L314 exact two-pass head (omega `omega_gpu_q36_graph_set_exact_head`): `screen` is the
    /// fp8b copy of the bf16 lm_head (`[vocab][2048]`, one bf16 scale per 128 x 128 block),
    /// `coef` the Row bound per row (`exact_head::Screen::row_coef`). The greedy token stays the
    /// plain head's by construction; the logits read back are -inf off the candidates, so turn it
    /// off ([`Self::set_exact_head_on`]) for any token that is sampled. Null `screen` removes it.
    ///
    /// # Safety
    /// `screen` is null or a live `OmegaGpuFp8Tensor` that outlives every later token of this graph.
    pub unsafe fn set_exact_head(&mut self, screen: *mut c_void, coef: &[f32]) -> Result<(), OmegaGpuError> {
        if !screen.is_null() {
            check_len("q36 graph exact head coef", self.vocab, coef.len())?;
        }
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live; screen per the caller's contract; coef holds vocab f32 (checked).
            let rc = unsafe { ffi::omega_gpu_q36_graph_set_exact_head(self.ptr, screen, coef.as_ptr()) };
            if rc != 0 {
                return Err(session_err("q36 graph set_exact_head", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        {
            let _ = (screen, coef);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Use the exact two-pass head for the next tokens (greedy) or the plain head (sampled).
    pub fn set_exact_head_on(&mut self, on: bool) {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_set_exact_head_on(self.ptr, on as i32)
        }
        #[cfg(not(has_omega_q36))]
        let _ = on;
    }

    /// The next token runs the exact two-pass head (set, on, no FP8 lm_head).
    pub fn exact_head_active(&self) -> bool {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_exact_head_active(self.ptr) != 0
        }
        #[cfg(not(has_omega_q36))]
        false
    }
    /// 1: submit after every launch (the comparison path).
    pub fn set_flush_each(&mut self, on: bool) {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_set_flush_each(self.ptr, on as i32)
        }
        #[cfg(not(has_omega_q36))]
        let _ = on;
    }

    /// TEST ONLY (negative controls): 1 SWIGLU up half off by one, 2 GATHER index + 1, 3 GATE
    /// dropped, 4 COMBINE skips the shared expert, 5 the exact head's MASK inverted, 6 its bound
    /// dropped. 0 none.
    pub fn test_set_mutant(&mut self, mutant: i32) {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_test_set_mutant(self.ptr, mutant)
        }
        #[cfg(not(has_omega_q36))]
        let _ = mutant;
    }
}

impl Drop for Q36Graph {
    fn drop(&mut self) {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr came from create and is freed once.
        unsafe {
            ffi::omega_gpu_q36_graph_free(self.ptr)
        }
    }
}
