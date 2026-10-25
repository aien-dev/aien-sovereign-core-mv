//! Qwen3.6-35B-A3B decode graph (omega `src/omega_gpu_q36_graph.h`, sc L306): the 27B graph's
//! mixers at the 35B shapes (DeltaNet ratio 2, attention 16 q / 2 kv heads) and the 30B's sparse
//! block with the shared expert as the ninth group. One whole token per [`Q36Graph::token`].
//!
//! As [`crate::Q38Graph`], the graph owns neither the weights nor the state: it takes the C
//! handles of a [`crate::Gdn`] (ratio 2), an [`crate::Attn256Cache`] (shape 16 / 2 / 4), an
//! `aien_omega_router::Router` (shape 256 + 1 shared, every layer and directory set) and the FP8 /
//! bf16 tensors, which must outlive the graph and not be used concurrently with a token.
#[cfg(has_omega_q36)]
use crate::session_err;
use crate::{bad, check_len, GraphStats};
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
        pub fn omega_gpu_q36_graph_set_final_norm(
            g: *mut OmegaGpuQ36Graph,
            final_norm: *const f32,
        ) -> i32;
        pub fn omega_gpu_q36_graph_create_heads(
            cfg: *const Q36Config,
            heads: u32,
            out: *mut *mut OmegaGpuQ36Graph,
        ) -> i32;
        pub fn omega_gpu_q36_graph_set_head_state(
            g: *mut OmegaGpuQ36Graph,
            head: u32,
            gdn: *mut c_void,
            attn: *mut c_void,
        ) -> i32;
        pub fn omega_gpu_q36_graph_stroke(
            g: *mut OmegaGpuQ36Graph,
            x_embed: *const f32,
            pos: *const u32,
            heads: u32,
            tokens: *mut u32,
            st: *mut GraphStats,
        ) -> i32;
        pub fn omega_gpu_q36_graph_prefill_alloc(g: *mut OmegaGpuQ36Graph, cap: u32) -> i32;
        pub fn omega_gpu_q36_graph_prefill_cap(g: *const OmegaGpuQ36Graph) -> u32;
        pub fn omega_gpu_q36_graph_prefill_streamed(g: *const OmegaGpuQ36Graph) -> u64;
        pub fn omega_gpu_q36_graph_prefill(
            g: *mut OmegaGpuQ36Graph,
            head: u32,
            x_embed: *const f32,
            n: u32,
            pos0: u32,
            want_token: i32,
            token: *mut u32,
            st: *mut GraphStats,
        ) -> i32;
        pub fn omega_gpu_q36_graph_read_head(
            g: *mut OmegaGpuQ36Graph,
            head: u32,
            x: *mut f32,
            hidden: *mut f32,
            logits: *mut f32,
        ) -> i32;
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
        pub fn omega_gpu_q36_graph_read_buf(
            g: *mut OmegaGpuQ36Graph,
            which: i32,
            out: *mut f32,
            n: u32,
        ) -> i32;
        pub fn omega_gpu_q36_graph_set_flush_each(g: *mut OmegaGpuQ36Graph, on: i32);
        pub fn omega_gpu_q36_graph_set_batch(g: *mut OmegaGpuQ36Graph, on: i32) -> i32;
        pub fn omega_gpu_q36_graph_batch(g: *const OmegaGpuQ36Graph) -> i32;
        pub fn omega_gpu_q36_graph_set_stamps(g: *mut OmegaGpuQ36Graph, on: i32);
        pub fn omega_gpu_q36_graph_stamps(
            g: *const OmegaGpuQ36Graph,
            stamps: *mut u64,
            kinds: *mut u8,
            cap: u32,
        ) -> i32;
        pub fn omega_gpu_q36_graph_kind_name(kind: u8) -> *const std::ffi::c_char;
        pub fn omega_gpu_q36_graph_test_set_mutant(g: *mut OmegaGpuQ36Graph, mutant: i32);
        pub fn omega_gpu_q36_graph_set_exact_head(
            g: *mut OmegaGpuQ36Graph,
            screen: *const c_void,
            coef: *const f32,
        ) -> i32;
        pub fn omega_gpu_q36_graph_set_exact_head_on(g: *mut OmegaGpuQ36Graph, on: i32);
        pub fn omega_gpu_q36_graph_exact_head_active(g: *const OmegaGpuQ36Graph) -> i32;
        pub fn omega_gpu_q36_graph_set_head_exact_head_on(
            g: *mut OmegaGpuQ36Graph,
            head: u32,
            on: i32,
        );
        pub fn omega_gpu_q36_graph_exact_head_active_head(
            g: *const OmegaGpuQ36Graph,
            head: u32,
        ) -> i32;
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

/// Heads per stroke the chip library lays out at most (one 4096-launch command list holds eight 40-layer tokens of 444 launches).
pub const MAX_HEADS: usize = 8;
/// Most tokens one prefill chunk carries (`OMEGA_GPU_Q36_PREFILL_MAX_ROWS`).
pub const PREFILL_MAX_ROWS: usize = 128;

pub struct Q36Graph {
    #[cfg(has_omega_q36)]
    ptr: *mut ffi::OmegaGpuQ36Graph,
    layers: usize,
    attn_mask: u64,
    vocab: usize,
    heads: usize,
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
        // SAFETY: the caller's contract, passed through.
        unsafe { Self::create_heads(cfg, 1) }
    }

    /// A graph with `heads` slots (1..=MAX_HEADS): head 0 uses `cfg.gdn` / `cfg.attn`, the others take
    /// their state objects from [`Self::set_head_state`] before a stroke.
    ///
    /// # Safety
    /// As [`Self::create`].
    pub unsafe fn create_heads(cfg: &Q36Config, heads: usize) -> Result<Self, OmegaGpuError> {
        if heads == 0 || heads > MAX_HEADS {
            return Err(bad(format!("q36 graph: heads {heads} (1..={MAX_HEADS})")));
        }
        if cfg.layers == 0 || cfg.layers > 64 || cfg.vocab == 0 {
            return Err(bad(format!(
                "q36 graph: layers {} vocab {}",
                cfg.layers, cfg.vocab
            )));
        }
        if cfg.gdn.is_null() || cfg.attn.is_null() || cfg.router.is_null() || cfg.lm_head.is_null()
        {
            return Err(bad("q36 graph: a state or weight handle is null".into()));
        }
        #[cfg(has_omega_q36)]
        {
            let mut ptr = std::ptr::null_mut();
            let rc = ffi::omega_gpu_q36_graph_create_heads(cfg, heads as u32, &mut ptr);
            if rc != 0 || ptr.is_null() {
                return Err(session_err("q36 graph create", rc));
            }
            Ok(Self {
                ptr,
                layers: cfg.layers as usize,
                attn_mask: cfg.attn_mask,
                vocab: cfg.vocab as usize,
                heads,
            })
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn heads(&self) -> usize {
        self.heads
    }

    /// Point head `head` at a DeltaNet state and an attention cache of the configured shapes (head swap).
    ///
    /// # Safety
    /// Both handles must be live and outlive every stroke that uses the head.
    pub unsafe fn set_head_state(
        &mut self,
        head: usize,
        gdn: *mut c_void,
        attn: *mut c_void,
    ) -> Result<(), OmegaGpuError> {
        if head >= self.heads || gdn.is_null() || attn.is_null() {
            return Err(bad(format!(
                "q36 graph: head {head} of {} or a null state handle",
                self.heads
            )));
        }
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live; the handles are the caller's contract.
            let rc = unsafe {
                ffi::omega_gpu_q36_graph_set_head_state(self.ptr, head as u32, gdn, attn)
            };
            if rc != 0 {
                return Err(session_err("q36 graph set_head_state", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// One stroke: `pos.len()` heads through one command list (`x_embed` holds HIDDEN values per head,
    /// head h at position `pos[h]`), one submit; the greedy token of every head.
    pub fn stroke(
        &mut self,
        x_embed: &[f32],
        pos: &[usize],
    ) -> Result<(Vec<u32>, GraphStats), OmegaGpuError> {
        let heads = pos.len();
        if heads == 0 || heads > self.heads {
            return Err(bad(format!(
                "q36 graph: stroke of {heads} heads on a graph of {}",
                self.heads
            )));
        }
        check_len("q36 graph stroke x_embed", HIDDEN * heads, x_embed.len())?;
        let pos: Vec<u32> = pos
            .iter()
            .map(|&p| u32::try_from(p).map_err(|_| bad(format!("q36 graph: pos {p} too large"))))
            .collect::<Result<_, _>>()?;
        #[cfg(has_omega_q36)]
        {
            let mut tokens = vec![0u32; heads];
            let mut st = GraphStats::default();
            // SAFETY: ptr is live; x_embed holds HIDDEN * heads f32 and pos/tokens heads u32 (checked).
            let rc = unsafe {
                ffi::omega_gpu_q36_graph_stroke(
                    self.ptr,
                    x_embed.as_ptr(),
                    pos.as_ptr(),
                    heads as u32,
                    tokens.as_mut_ptr(),
                    &mut st,
                )
            };
            if rc != 0 {
                return Err(session_err("q36 graph stroke", rc));
            }
            Ok((tokens, st))
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// L369 prefill: the chunk size the buffers hold (0 before [`Self::prefill_alloc`]).
    pub fn prefill_cap(&self) -> usize {
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live.
            unsafe { ffi::omega_gpu_q36_graph_prefill_cap(self.ptr) as usize }
        }
        #[cfg(not(has_omega_q36))]
        0
    }

    /// L369 prefill: fp8 GEMM weight bytes every prefill call so far streamed (a running count; the
    /// difference around a call over its time against the memory bandwidth is the ceiling receipt).
    pub fn prefill_streamed(&self) -> u64 {
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live.
            unsafe { ffi::omega_gpu_q36_graph_prefill_streamed(self.ptr) }
        }
        #[cfg(not(has_omega_q36))]
        0
    }

    /// L369 prefill: sizes the chunk buffers for up to `cap` tokens (1..=[`PREFILL_MAX_ROWS`]).
    pub fn prefill_alloc(&mut self, cap: usize) -> Result<(), OmegaGpuError> {
        if cap == 0 || cap > PREFILL_MAX_ROWS {
            return Err(bad(format!(
                "q36 graph: prefill cap {cap} (1..={PREFILL_MAX_ROWS})"
            )));
        }
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live.
            let rc = unsafe { ffi::omega_gpu_q36_graph_prefill_alloc(self.ptr, cap as u32) };
            if rc != 0 {
                return Err(session_err("q36 graph prefill_alloc", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// L369 prefill: `n` prompt tokens (embeddings `x_embed`, `[n][HIDDEN]`) of the conversation in
    /// head slot `head` at positions `pos0..pos0 + n`, each as if it went through a token step in
    /// turn (the decode step's kernels, so the state matches token by token bit for bit). With
    /// `want_token` the last token's greedy token (and [`Self::read_head`] for its logits).
    pub fn prefill(
        &mut self,
        head: usize,
        x_embed: &[f32],
        pos0: usize,
        want_token: bool,
    ) -> Result<(Option<u32>, GraphStats), OmegaGpuError> {
        let n = x_embed.len() / HIDDEN;
        if head >= self.heads || n == 0 {
            return Err(bad(format!(
                "q36 graph: prefill head {head} of {}, {n} tokens",
                self.heads
            )));
        }
        check_len("q36 graph prefill x_embed", HIDDEN * n, x_embed.len())?;
        let pos0 =
            u32::try_from(pos0).map_err(|_| bad(format!("q36 graph: pos {pos0} too large")))?;
        #[cfg(has_omega_q36)]
        {
            let mut token = 0u32;
            let mut st = GraphStats::default();
            // SAFETY: ptr is live; x_embed holds HIDDEN * n f32 (checked).
            let rc = unsafe {
                ffi::omega_gpu_q36_graph_prefill(
                    self.ptr,
                    head as u32,
                    x_embed.as_ptr(),
                    n as u32,
                    pos0,
                    want_token as i32,
                    &mut token,
                    &mut st,
                )
            };
            if rc != 0 {
                return Err(session_err("q36 graph prefill", rc));
            }
            Ok((want_token.then_some(token), st))
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// As [`Self::read`], for head `head` of the last stroke.
    pub fn read_head(
        &mut self,
        head: usize,
        x: Option<&mut [f32]>,
        hidden: Option<&mut [f32]>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), OmegaGpuError> {
        if head >= self.heads {
            return Err(bad(format!("q36 graph: head {head} of {}", self.heads)));
        }
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
            let rc = unsafe {
                ffi::omega_gpu_q36_graph_read_head(
                    self.ptr,
                    head as u32,
                    p(x),
                    p(hidden),
                    p(logits),
                )
            };
            if rc != 0 {
                return Err(session_err("q36 graph read_head", rc));
            }
            Ok(())
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
            return Err(bad(format!(
                "q36 graph layer {layer}: a projection has no device handle"
            )));
        }
        if ab.is_null() != self.is_attention(layer) {
            return Err(bad(format!(
                "q36 graph layer {layer}: the a|b tensor is for DeltaNet layers only"
            )));
        }
        #[cfg(has_omega_q36)]
        {
            let rc = ffi::omega_gpu_q36_graph_set_layer(
                self.ptr,
                layer as u32,
                mix,
                ab,
                o,
                gate_up,
                down,
                in_norm.as_ptr(),
            );
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
            let rc =
                unsafe { ffi::omega_gpu_q36_graph_set_final_norm(self.ptr, final_norm.as_ptr()) };
            if rc != 0 {
                return Err(session_err("q36 graph set_final_norm", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
    }

    /// One whole token: the embedding row in, the greedy token out (position `pos`).
    pub fn token(
        &mut self,
        x_embed: &[f32],
        pos: usize,
    ) -> Result<(u32, GraphStats), OmegaGpuError> {
        check_len("q36 graph x_embed", HIDDEN, x_embed.len())?;
        let pos = u32::try_from(pos).map_err(|_| bad(format!("q36 graph: pos {pos} too large")))?;
        #[cfg(has_omega_q36)]
        {
            let mut token = 0u32;
            let mut st = GraphStats::default();
            // SAFETY: ptr is live; x_embed holds HIDDEN f32 (checked); token and st are local.
            let rc = unsafe {
                ffi::omega_gpu_q36_graph_token(self.ptr, x_embed.as_ptr(), pos, &mut token, &mut st)
            };
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
            let rc = unsafe {
                ffi::omega_gpu_q36_graph_read_buf(
                    self.ptr,
                    which as i32,
                    out.as_mut_ptr(),
                    out.len() as u32,
                )
            };
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
    pub unsafe fn set_exact_head(
        &mut self,
        screen: *mut c_void,
        coef: &[f32],
    ) -> Result<(), OmegaGpuError> {
        if !screen.is_null() {
            check_len("q36 graph exact head coef", self.vocab, coef.len())?;
        }
        #[cfg(has_omega_q36)]
        {
            // SAFETY: ptr is live; screen per the caller's contract; coef holds vocab f32 (checked).
            let rc =
                unsafe { ffi::omega_gpu_q36_graph_set_exact_head(self.ptr, screen, coef.as_ptr()) };
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

    /// The batch (v2, L312): each layer recorded for every conversation at once, one weight read per layer.
    /// Pins the GEMM kernels on the session; `Err` carries the chip rc.
    pub fn set_batch(&mut self, on: bool) -> Result<(), i32> {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            let rc = ffi::omega_gpu_q36_graph_set_batch(self.ptr, on as i32);
            if rc == 0 {
                Ok(())
            } else {
                Err(rc)
            }
        }
        #[cfg(not(has_omega_q36))]
        {
            let _ = on;
            Err(-1)
        }
    }
    /// Whether the batch is on (also set by `AIEN_OMEGA_Q36_BATCH=1` at create).
    pub fn batch(&self) -> bool {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_batch(self.ptr) != 0
        }
        #[cfg(not(has_omega_q36))]
        false
    }
    /// Strokes: the exact head on / off for one head slot (a stroke may mix greedy and sampled
    /// rows). Call between tokens; out-of-range heads are ignored.
    pub fn set_head_exact_head_on(&mut self, head: usize, on: bool) {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live; plain integer arguments.
        unsafe {
            ffi::omega_gpu_q36_graph_set_head_exact_head_on(self.ptr, head as u32, on as i32)
        }
        #[cfg(not(has_omega_q36))]
        let _ = (head, on);
    }

    /// The next stroke runs the exact two-pass head on this head slot.
    pub fn exact_head_active_head(&self, head: usize) -> bool {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_exact_head_active_head(self.ptr, head as u32) != 0
        }
        #[cfg(not(has_omega_q36))]
        {
            let _ = head;
            false
        }
    }

    /// Chip timestamps around every launch of the next tokens (a timeline; off by default).
    pub fn set_stamps(&mut self, on: bool) {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live.
        unsafe {
            ffi::omega_gpu_q36_graph_set_stamps(self.ptr, on as i32);
        }
        #[cfg(not(has_omega_q36))]
        let _ = on;
    }
    /// The last token's timeline: (kind, chip ns) per launch, in launch order. Needs `set_stamps(true)`
    /// before that token; Err(rc) when there is none.
    pub fn timeline(&self) -> Result<Vec<(u8, u64)>, i32> {
        #[cfg(has_omega_q36)]
        // SAFETY: ptr is live; the buffers hold the most launches one command list carries, plus one.
        unsafe {
            const CAP: usize = 4097;
            let mut stamps = vec![0u64; CAP];
            let mut kinds = vec![0u8; CAP];
            let n = ffi::omega_gpu_q36_graph_stamps(
                self.ptr,
                stamps.as_mut_ptr(),
                kinds.as_mut_ptr(),
                CAP as u32,
            );
            if n < 0 {
                return Err(n);
            }
            let n = n as usize;
            Ok((0..n)
                .map(|i| (kinds[i], stamps[i + 1].wrapping_sub(stamps[i])))
                .collect())
        }
        #[cfg(not(has_omega_q36))]
        Err(-1)
    }
    /// omega's own name for a launch kind of the 35B graph.
    pub fn kind_name(kind: u8) -> String {
        #[cfg(has_omega_q36)]
        // SAFETY: the library returns a static NUL-terminated string for every kind value.
        unsafe {
            let p = ffi::omega_gpu_q36_graph_kind_name(kind);
            if p.is_null() {
                return "?".into();
            }
            std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
        }
        #[cfg(not(has_omega_q36))]
        {
            let _ = kind;
            "?".into()
        }
    }

    /// The last token's timeline: `launches + 1` GPU timestamps (ns) and one kind per launch
    /// (launch `i` took `stamps[i + 1] - stamps[i]`). Refused when stamps are off or the token
    /// failed.
    pub fn stamps(&self) -> Result<(Vec<u64>, Vec<u8>), OmegaGpuError> {
        #[cfg(has_omega_q36)]
        {
            // 11 per layer + 4, with room to spare.
            let cap = (self.layers * 16 + 16) as u32;
            let mut stamps = vec![0u64; cap as usize + 1];
            let mut kinds = vec![0u8; cap as usize + 1];
            // SAFETY: ptr is live; both buffers hold cap + 1 entries.
            let n = unsafe {
                ffi::omega_gpu_q36_graph_stamps(
                    self.ptr,
                    stamps.as_mut_ptr(),
                    kinds.as_mut_ptr(),
                    cap + 1,
                )
            };
            if n < 0 {
                return Err(session_err("q36 graph stamps", n));
            }
            let n = n as usize;
            stamps.truncate(n + 1);
            kinds.truncate(n);
            Ok((stamps, kinds))
        }
        #[cfg(not(has_omega_q36))]
        Err(OmegaGpuError::Unavailable)
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

/// omega's own name for a 35B launch kind (`omega_gpu_q36_graph_kind_name`).
pub fn kind_name(kind: u8) -> String {
    #[cfg(has_omega_q36)]
    {
        // SAFETY: returns a static NUL-terminated string or NULL.
        let p = unsafe { ffi::omega_gpu_q36_graph_kind_name(kind) };
        if !p.is_null() {
            // SAFETY: non-null static string.
            return unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
        }
    }
    format!("kind{kind}")
}
