//! Safe-ish Rust wrapper over omega's Qwen3.8-27B decode pieces (sc #642; no CUDA):
//!
//! * [`Gdn`]: the device-resident Gated DeltaNet parameters, conv tails and recurrent states of
//!   every DeltaNet layer (`src/omega_gpu_gdn_api.h`).
//! * [`Attn256Cache`]: the bf16 KV cache, RoPE table and q/k norms of every full-attention layer
//!   (`src/omega_gpu_decode_attn256_api.h`, head_dim 256, partial RoPE by permutation).
//! * [`Q38Graph`]: one whole decode token as ONE command list (`src/omega_gpu_q38_graph.h`): per
//!   layer the fused FP8 mixer projection, the DeltaNet step or the attention, o_proj, the
//!   residual adds and norms, the FP8 MLP; then the bf16 lm_head and a two-stage argmax.
//!
//! The graph does not own the weights or the state: it takes the C handles of objects the caller
//! already uploaded (`as_raw()` of `aien_omega_fp8::Fp8Tensor`, `aien_omega_bf16::Bf16Tensor`,
//! [`Gdn`], [`Attn256Cache`]). Those objects must outlive the graph and must not be used
//! concurrently with a [`Q38Graph::token`] call. Without the native library (`is_native()` false)
//! every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;

pub mod q36;
use std::ffi::c_void;
use std::fmt;

/// Residual width (`OMEGA_GPU_Q38_HIDDEN`).
pub const HIDDEN: usize = 5120;
/// MLP width (`OMEGA_GPU_Q38_INTER`).
pub const INTER: usize = 17408;
/// DeltaNet fused projection rows: in_proj_qkv 10240 | in_proj_z 6144.
pub const GDN_MIX: usize = 16384;
/// DeltaNet in_proj_qkv rows (the conv width).
pub const GDN_QKV: usize = 10240;
/// DeltaNet a|b rows: 48 heads x {a, b}, interleaved.
pub const AB: usize = 96;
/// Attention fused projection rows: q_proj 12288 (24 x [q 256 | gate 256]) | k 1024 | v 1024.
pub const ATT_MIX: usize = 14336;
/// Mixer output width (DeltaNet 48 x 128 = attention 24 x 256): the o_proj input.
pub const MIXER_OUT: usize = 6144;
/// DeltaNet value heads.
pub const GDN_HEADS: usize = 48;
/// DeltaNet conv kernel.
pub const GDN_CONV_KERNEL: usize = 4;
/// DeltaNet per-head value width (the gated norm's weight length).
pub const GDN_HEAD_DIM: usize = 128;
/// One DeltaNet layer's conv tail in the reference layout, `[3][10240]` oldest first
/// (`OMEGA_GPU_GDN_TAIL_FLOATS`).
pub const GDN_TAIL_FLOATS: usize = (GDN_CONV_KERNEL - 1) * GDN_QKV;
/// One DeltaNet layer's recurrent state, `[48][128 k][128 v]` f32 (`OMEGA_GPU_GDN_STATE_FLOATS`).
pub const GDN_STATE_FLOATS: usize = GDN_HEADS * GDN_HEAD_DIM * GDN_HEAD_DIM;
/// Attention head width.
pub const HEAD_DIM: usize = 256;
/// Most layers (`OMEGA_GPU_Q38_MAX_LAYERS`).
pub const MAX_LAYERS: usize = 64;
/// Largest vocabulary the two-stage argmax covers (`2048 * 2048`).
pub const MAX_VOCAB: usize = 2048 * 2048;
/// Longest KV cache (`OMEGA_GPU_DATTN256_MAX_POS`).
pub const MAX_POS: usize = 131072;
/// 2^64: the FP8 GEMV row scale (the graph keeps its activation rows at this scale).
pub const ROW_SCALE: f32 = 18_446_744_073_709_551_616.0;

/// The attention kernel's q|k|v row: q 24 x 256 (permuted) | k 4 x 256 (permuted) | v 4 x 256.
pub const ATT_QKV: usize = 8192;
/// Fewest tokens one DeltaNet prefill call takes (omega#543).
pub const GDN_PF_MIN_T: usize = 3;
/// Most tokens one DeltaNet prefill call takes (`OMEGA_GPU_GDN_PF_MAX_T`).
pub const GDN_PF_MAX_T: usize = 65536;
/// Most tokens one attention prefill call takes (`OMEGA_GPU_PATTN256_MAX_M`).
pub const PATTN_MAX_M: usize = 32768;

/// True when the Qwen3.8 decode API is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_q38)
}

/// True when the prefill entries ([`Gdn::prefill`], [`Attn256Cache::prefill`]) are linked.
pub const fn has_prefill() -> bool {
    cfg!(has_omega_q38_prefill)
}

/// Source index in the attention projection row (`[ATT_MIX]`: q_proj 24 x [q 256 | gate 256],
/// k_proj, v_proj) of every slot of the gathered row: q permuted | k permuted | v (the first
/// [`ATT_QKV`], the attention kernel's row) | gate (`[MIXER_OUT]`). Mirror of
/// `omega_q38_gather_table` (the decode graph's GATHER).
pub fn gather_table(rotary: usize) -> Vec<u32> {
    let qp = 24 * 512;
    let kp = qp + 4 * 256;
    (0..ATT_MIX)
        .map(|n| {
            let s = if n < 6144 {
                (n / 256) * 512 + perm_slot(n % 256, rotary)
            } else if n < 7168 {
                qp + ((n - 6144) / 256) * 256 + perm_slot(n % 256, rotary)
            } else if n < ATT_QKV {
                kp + (n - 7168)
            } else {
                ((n - ATT_QKV) / 256) * 512 + 256 + (n - ATT_QKV) % 256
            };
            s as u32
        })
        .collect()
}

/// What one prefill call cost: the chip time of its launches and the whole call (host copies
/// included), both from omega's info record.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PrefillCost {
    pub chip_ns: u64,
    pub call_ns: u64,
}

/// Mirror of `OmegaGpuGdnPfInfo`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct GdnPfInfo {
    pub layer: u32,
    pub tokens: u32,
    pub insns: [u32; 3],
    pub gprs: [u32; 3],
    pub elapsed_ns: [u64; 3],
    pub call_ns: u64,
    pub kernel_cache_hits: u32,
}

/// Mirror of `OmegaGpuPrefillAttn256Info`.
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct Attn256PfInfo {
    pub layer: u32,
    pub pos0: u32,
    pub m: u32,
    pub grid_x: [u32; 2],
    pub insns: [u32; 2],
    pub gprs: [u32; 2],
    pub elapsed_ns: [u64; 2],
    pub call_ns: u64,
    pub kernel_cache_hits: u32,
}

/// Launch kinds of the timeline (`OMEGA_GPU_Q38_K_*`), in enum order.
pub const KIND_NAMES: [&str; 14] = [
    "norm", "mix", "ab", "gdn", "gather", "attn", "gate", "o_proj", "res", "gate_up", "swiglu",
    "down", "lm_head", "argmax",
];

/// Intermediates [`Q38Graph::read_buf`] can copy (`OMEGA_GPU_Q38_BUF_*`) and their lengths.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Buf {
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
}

impl Buf {
    /// The buffer's capacity in f32.
    pub const fn capacity(self) -> usize {
        match self {
            Self::X | Self::Xq | Self::O | Self::Hidden => HIDDEN,
            Self::Mix => GDN_MIX,
            Self::Ab => AB,
            Self::Agg => ATT_MIX,
            Self::Att => MIXER_OUT,
            Self::Gu => 2 * INTER,
            Self::Act => INTER,
        }
    }
}

/// Slot `i` of the attention kernel's head holds the model's dim `perm_slot(i, rotary)`
/// (`omega_dattn256_perm_slot`, design decision 1 of omega#544): the kernel's rotate-half pairs
/// `(i, i + 128)`, `i < rotary / 2`, are the model's pairs `(j, j + rotary / 2)`.
pub const fn perm_slot(i: usize, rotary: usize) -> usize {
    let p = rotary / 2;
    if i < p {
        i
    } else if i < 128 {
        rotary + (i - p)
    } else if i < 128 + p {
        p + (i - 128)
    } else {
        i
    }
}

/// `dst[i] = src[perm_slot(i)]` over one 256-wide head (q, k and their norm weights).
pub fn perm_vec(src: &[f32], rotary: usize) -> Vec<f32> {
    assert_eq!(src.len(), HEAD_DIM, "perm_vec: one head of 256");
    (0..HEAD_DIM).map(|i| src[perm_slot(i, rotary)]).collect()
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

/// Mirror of `OmegaGpuQ38GraphConfig`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct GraphConfig {
    /// 1..=64 (a model prefix in tests).
    pub layers: u32,
    /// Bit `l` set: model layer `l` is full attention, else DeltaNet.
    pub attn_mask: u64,
    /// lm_head rows.
    pub vocab: u32,
    /// rms_norm_eps.
    pub eps: f32,
    /// DeltaNet q/k l2norm eps.
    pub eps_l2: f32,
    /// [`Gdn::as_raw`]: one entry per DeltaNet layer, in model order.
    pub gdn: *mut c_void,
    /// [`Attn256Cache::as_raw`]: one entry per attention layer, in model order; rotary 64.
    pub attn: *mut c_void,
    /// `Bf16Tensor::as_raw`, `[vocab][HIDDEN]`.
    pub lm_head: *const c_void,
}

#[cfg(has_omega_q38)]
mod ffi {
    use super::{GraphConfig, GraphStats};
    use std::ffi::c_void;
    use std::os::raw::c_char;
    #[repr(C)]
    pub struct OmegaGpuQ38Graph {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct OmegaGpuGdn {
        _private: [u8; 0],
    }
    #[repr(C)]
    pub struct OmegaGpuDecodeAttn256Cache {
        _private: [u8; 0],
    }
    extern "C" {
        pub fn omega_gpu_gdn_alloc_ratio(
            layers: u32,
            ratio: u32,
            out: *mut *mut OmegaGpuGdn,
        ) -> i32;
        pub fn omega_gpu_gdn_free(g: *mut OmegaGpuGdn);
        pub fn omega_gpu_gdn_set_layer(
            g: *mut OmegaGpuGdn,
            layer: u32,
            conv_w: *const f32,
            a_log: *const f32,
            dt_bias: *const f32,
            norm_w: *const f32,
        ) -> i32;
        pub fn omega_gpu_gdn_reset(g: *mut OmegaGpuGdn) -> i32;
        pub fn omega_gpu_gdn_set_state(
            g: *mut OmegaGpuGdn,
            layer: u32,
            tail: *const f32,
            state: *const f32,
        ) -> i32;
        pub fn omega_gpu_gdn_get_state(
            g: *mut OmegaGpuGdn,
            layer: u32,
            tail: *mut f32,
            state: *mut f32,
        ) -> i32;

        pub fn omega_gpu_decode_attn256_cache_alloc_shape(
            layers: u32,
            max_pos: u32,
            rope_theta: f64,
            rotary_dim: u32,
            q_heads: u32,
            kv_heads: u32,
            q_per_cta: u32,
            out: *mut *mut OmegaGpuDecodeAttn256Cache,
        ) -> i32;
        pub fn omega_gpu_decode_attn256_cache_free(c: *mut OmegaGpuDecodeAttn256Cache);
        pub fn omega_gpu_decode_attn256_set_norms(
            c: *mut OmegaGpuDecodeAttn256Cache,
            layer: u32,
            q_norm: *const f32,
            k_norm: *const f32,
        ) -> i32;
        /// L803 (2026-10-24): positions per CTA of the single-row decode attention launch (a multiple of 256; 0 =
        /// compiled default 256). Experiments only: the multi-row launch refuses any value but 256.
        pub fn omega_gpu_decode_attn256_set_per_split(per_split: u32);
        pub fn omega_gpu_decode_attn_last_error() -> *const c_char;

        pub fn omega_gpu_q38_graph_create(
            cfg: *const GraphConfig,
            out: *mut *mut OmegaGpuQ38Graph,
        ) -> i32;
        pub fn omega_gpu_q38_graph_free(g: *mut OmegaGpuQ38Graph);
        pub fn omega_gpu_q38_graph_set_layer(
            g: *mut OmegaGpuQ38Graph,
            layer: u32,
            mix: *const c_void,
            ab: *const c_void,
            o: *const c_void,
            gate_up: *const c_void,
            down: *const c_void,
            in_norm: *const f32,
            post_norm: *const f32,
        ) -> i32;
        pub fn omega_gpu_q38_graph_set_final_norm(
            g: *mut OmegaGpuQ38Graph,
            final_norm: *const f32,
        ) -> i32;
        pub fn omega_gpu_q38_graph_set_lm_head_fp8(
            g: *mut OmegaGpuQ38Graph,
            head: *const c_void,
        ) -> i32;
        pub fn omega_gpu_fp8_quantize_bf16(
            n: u32,
            k: u32,
            w: *const u16,
            w8: *mut u8,
            s: *mut u16,
        ) -> i32;
        pub fn omega_gpu_q38_graph_token(
            g: *mut OmegaGpuQ38Graph,
            x_embed: *const f32,
            pos: u32,
            token: *mut u32,
            st: *mut GraphStats,
        ) -> i32;
        pub fn omega_gpu_q38_graph_read(
            g: *mut OmegaGpuQ38Graph,
            x: *mut f32,
            hidden: *mut f32,
            logits: *mut f32,
        ) -> i32;
        pub fn omega_gpu_q38_graph_read_buf(
            g: *mut OmegaGpuQ38Graph,
            which: i32,
            out: *mut f32,
            n: u32,
        ) -> i32;
        pub fn omega_gpu_q38_graph_set_stamps(g: *mut OmegaGpuQ38Graph, on: i32);
        pub fn omega_gpu_q38_graph_stamps(
            g: *const OmegaGpuQ38Graph,
            stamps: *mut u64,
            kinds: *mut u8,
            cap: u32,
        ) -> i32;
        pub fn omega_gpu_q38_graph_kind_name(kind: u8) -> *const c_char;
        pub fn omega_gpu_q38_graph_set_flush_each(g: *mut OmegaGpuQ38Graph, on: i32);
        pub fn omega_gpu_session_last_error() -> *const c_char;
        pub fn omega_q38_gather_table(tab: *mut u32);
    }

    #[cfg(has_omega_q38_prefill)]
    extern "C" {
        pub fn omega_gpu_gdn_prefill(
            g: *mut OmegaGpuGdn,
            layer: u32,
            tokens: u32,
            qkv: *const f32,
            z: *const f32,
            ab: *const f32,
            eps_l2: f32,
            eps_norm: f32,
            in_scale: f32,
            out_scale: f32,
            out: *mut f32,
            info: *mut super::GdnPfInfo,
        ) -> i32;
        pub fn omega_gpu_gdn_last_error() -> *const c_char;
        pub fn omega_gpu_prefill_attn256(
            c: *mut OmegaGpuDecodeAttn256Cache,
            layer: u32,
            pos0: u32,
            m: u32,
            qkv: *const f32,
            eps: f32,
            out_scale: f32,
            out: *mut f32,
            info: *mut super::Attn256PfInfo,
        ) -> i32;
    }
}

/// omega's own gather table (`omega_q38_gather_table`), for checking [`gather_table`].
/// L803 (2026-10-24): positions per CTA of the single-row decode attention launch, process-wide (0 = the compiled
/// default, 256 = one chunk per CTA). Call before the first token. Any value that is not a positive multiple of 256
/// makes every attention launch fail with BAD_ARGS, and the multi-row (batched) launch refuses anything but 256:
/// a diagnostic for the attention block-count experiment, never a server setting.
pub fn decode_attn256_set_per_split(per_split: u32) {
    // SAFETY: a plain setter that takes omega's session lock; no pointers cross.
    unsafe { ffi::omega_gpu_decode_attn256_set_per_split(per_split) }
}

pub fn gather_table_c() -> Result<Vec<u32>, OmegaGpuError> {
    #[cfg(has_omega_q38)]
    {
        let mut t = vec![0u32; ATT_MIX];
        // SAFETY: omega writes exactly OMEGA_GPU_Q38_ATT_MIX entries.
        unsafe { ffi::omega_q38_gather_table(t.as_mut_ptr()) };
        Ok(t)
    }
    #[cfg(not(has_omega_q38))]
    Err(OmegaGpuError::Unavailable)
}

/// omega's own name for a launch kind (`omega_gpu_q38_graph_kind_name`); the table above in
/// the stub.
pub fn kind_name(kind: u8) -> String {
    #[cfg(has_omega_q38)]
    {
        // SAFETY: returns a static NUL-terminated string or NULL.
        let p = unsafe { ffi::omega_gpu_q38_graph_kind_name(kind) };
        if !p.is_null() {
            // SAFETY: non-null static string.
            return unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
        }
    }
    KIND_NAMES
        .get(kind as usize)
        .map_or_else(|| format!("kind{kind}"), |s| s.to_string())
}

#[cfg(has_omega_q38)]
fn c_text(p: *const std::os::raw::c_char) -> String {
    if p.is_null() {
        String::new()
    } else {
        // SAFETY: omega returns static or library-owned NUL-terminated strings.
        unsafe { std::ffi::CStr::from_ptr(p) }
            .to_string_lossy()
            .into_owned()
    }
}

#[cfg(has_omega_q38)]
fn rc_err(what: &str, rc: i32, detail: String) -> OmegaGpuError {
    let name = match rc {
        -1 => "BAD_ARGS",
        -2 => "TOO_LARGE",
        -3 => "CODEGEN_FAIL",
        -4 => "CHIP_FAIL",
        _ => "?",
    };
    OmegaGpuError::Rc {
        rc,
        name: format!("{what}: {name}: {detail}"),
    }
}

/// An error from a session-wide API (DeltaNet, the graph): the detail is the session's last error.
#[cfg(has_omega_q38)]
fn session_err(what: &str, rc: i32) -> OmegaGpuError {
    // SAFETY: no arguments; returns a static NUL-terminated string or NULL.
    rc_err(
        what,
        rc,
        c_text(unsafe { ffi::omega_gpu_session_last_error() }),
    )
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

// ---------------------------------------------------------------------------
// Gated DeltaNet state
// ---------------------------------------------------------------------------

/// Every DeltaNet layer's device-resident parameters, conv tail and recurrent state.
pub struct Gdn {
    #[cfg(has_omega_q38)]
    ptr: *mut ffi::OmegaGpuGdn,
    layers: usize,
    /// DeltaNet value heads per key head (16 key heads): 3 = Qwen3.8-27B, 2 = Qwen3.6-35B-A3B.
    ratio: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for Gdn {}

impl fmt::Debug for Gdn {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Gdn {{ layers: {}, ratio: {} }}",
            self.layers, self.ratio
        )
    }
}

impl Gdn {
    /// Allocates `layers` DeltaNet entries (tails and states zeroed).
    pub fn alloc(layers: usize) -> Result<Self, OmegaGpuError> {
        Self::alloc_ratio(layers, 3)
    }

    /// The same for another model: `ratio` value heads per key head (3 = Qwen3.8-27B, 2 =
    /// Qwen3.6-35B-A3B; omega `omega_gpu_gdn_alloc_ratio`). Every length the other calls check
    /// follows the ratio.
    pub fn alloc_ratio(layers: usize, ratio: usize) -> Result<Self, OmegaGpuError> {
        if ratio != 2 && ratio != 3 {
            return Err(bad(format!("gdn: ratio {ratio}, 2 or 3 supported")));
        }
        if layers == 0 || layers > MAX_LAYERS {
            return Err(bad(format!(
                "gdn: {layers} layers, 1..={MAX_LAYERS} supported"
            )));
        }
        #[cfg(has_omega_q38)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: out pointer is local.
            let rc =
                unsafe { ffi::omega_gpu_gdn_alloc_ratio(layers as u32, ratio as u32, &mut ptr) };
            if rc != 0 || ptr.is_null() {
                return Err(session_err("gdn alloc", rc));
            }
            Ok(Self { ptr, layers, ratio })
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Value heads (16 key heads x ratio).
    pub fn heads(&self) -> usize {
        16 * self.ratio
    }

    /// in_proj_qkv rows, the conv width: q and k 16 x 128 each, then heads x 128 of v.
    pub fn qkv(&self) -> usize {
        (32 + self.heads()) * GDN_HEAD_DIM
    }

    /// One layer's conv tail, `[3][qkv]` oldest first.
    pub fn tail_floats(&self) -> usize {
        (GDN_CONV_KERNEL - 1) * self.qkv()
    }

    /// One layer's recurrent state, `[heads][128 k][128 v]` f32.
    pub fn state_floats(&self) -> usize {
        self.heads() * GDN_HEAD_DIM * GDN_HEAD_DIM
    }

    /// The C handle (`OmegaGpuGdn *`) for [`GraphConfig::gdn`]; null in the stub. Stays owned by
    /// `self`.
    pub fn as_raw(&self) -> *mut c_void {
        #[cfg(has_omega_q38)]
        {
            self.ptr.cast()
        }
        #[cfg(not(has_omega_q38))]
        {
            std::ptr::null_mut()
        }
    }

    /// One layer's parameters: `conv_w` `[10240][4]` (conv1d.weight widened to f32), `a_log` and
    /// `dt_bias` `[48]`, `norm_w` `[128]` (linear_attn.norm.weight, used as is).
    pub fn set_layer(
        &mut self,
        layer: usize,
        conv_w: &[f32],
        a_log: &[f32],
        dt_bias: &[f32],
        norm_w: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("gdn conv_w", self.qkv() * GDN_CONV_KERNEL, conv_w.len())?;
        check_len("gdn a_log", self.heads(), a_log.len())?;
        check_len("gdn dt_bias", self.heads(), dt_bias.len())?;
        check_len("gdn norm_w", GDN_HEAD_DIM, norm_w.len())?;
        if layer >= self.layers {
            return Err(bad(format!("gdn: layer {layer} of {}", self.layers)));
        }
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; every slice has the length omega reads (checked).
            let rc = unsafe {
                ffi::omega_gpu_gdn_set_layer(
                    self.ptr,
                    layer as u32,
                    conv_w.as_ptr(),
                    a_log.as_ptr(),
                    dt_bias.as_ptr(),
                    norm_w.as_ptr(),
                )
            };
            if rc != 0 {
                return Err(session_err("gdn set_layer", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Every layer's tail and state to zero (a new sequence).
    pub fn reset(&mut self) -> Result<(), OmegaGpuError> {
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live.
            let rc = unsafe { ffi::omega_gpu_gdn_reset(self.ptr) };
            if rc != 0 {
                return Err(session_err("gdn reset", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Layer `layer`'s conv tail ([`GDN_TAIL_FLOATS`]) and recurrent state ([`GDN_STATE_FLOATS`])
    /// in the reference layouts (`omega_gpu_gdn_get_state`): what a later [`Self::set_state`]
    /// puts back so the sequence continues from here.
    pub fn get_state(
        &mut self,
        layer: usize,
        tail: &mut [f32],
        state: &mut [f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("gdn tail", self.tail_floats(), tail.len())?;
        check_len("gdn state", self.state_floats(), state.len())?;
        if layer >= self.layers {
            return Err(bad(format!(
                "gdn get_state: layer {layer} of {}",
                self.layers
            )));
        }
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; both buffers hold the lengths checked above.
            let rc = unsafe {
                ffi::omega_gpu_gdn_get_state(
                    self.ptr,
                    layer as u32,
                    tail.as_mut_ptr(),
                    state.as_mut_ptr(),
                )
            };
            if rc != 0 {
                // SAFETY: no arguments.
                let detail = c_text(unsafe { ffi::omega_gpu_gdn_last_error() });
                return Err(rc_err("gdn get_state", rc, detail));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Writes layer `layer`'s conv tail and recurrent state ([`Self::get_state`]'s layouts).
    pub fn set_state(
        &mut self,
        layer: usize,
        tail: &[f32],
        state: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("gdn tail", self.tail_floats(), tail.len())?;
        check_len("gdn state", self.state_floats(), state.len())?;
        if layer >= self.layers {
            return Err(bad(format!(
                "gdn set_state: layer {layer} of {}",
                self.layers
            )));
        }
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; both buffers hold the lengths checked above.
            let rc = unsafe {
                ffi::omega_gpu_gdn_set_state(self.ptr, layer as u32, tail.as_ptr(), state.as_ptr())
            };
            if rc != 0 {
                // SAFETY: no arguments.
                let detail = c_text(unsafe { ffi::omega_gpu_gdn_last_error() });
                return Err(rc_err("gdn set_state", rc, detail));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// `tokens` new tokens of DeltaNet layer `layer` (its index among the DeltaNet layers) in one
    /// call (`omega_gpu_gdn_prefill`), continuing from the layer's tail and state exactly as
    /// `tokens` decode steps would and leaving them for the next call or decode step. Plain
    /// values in and out (scales 1): `qkv` `[tokens][10240]`, `z` `[tokens][6144]`, `ab`
    /// `[tokens][96]` ({a, b} per head), `out` `[tokens][6144]` (the out_proj input).
    #[allow(clippy::too_many_arguments)]
    pub fn prefill(
        &mut self,
        layer: usize,
        tokens: usize,
        qkv: &[f32],
        z: &[f32],
        ab: &[f32],
        eps_l2: f32,
        eps_norm: f32,
        out: &mut [f32],
    ) -> Result<PrefillCost, OmegaGpuError> {
        if !(GDN_PF_MIN_T..=GDN_PF_MAX_T).contains(&tokens) {
            return Err(bad(format!(
                "gdn prefill: {tokens} tokens, {GDN_PF_MIN_T}..={GDN_PF_MAX_T} supported"
            )));
        }
        check_len("gdn prefill qkv", tokens * GDN_QKV, qkv.len())?;
        check_len("gdn prefill z", tokens * MIXER_OUT, z.len())?;
        check_len("gdn prefill ab", tokens * AB, ab.len())?;
        check_len("gdn prefill out", tokens * MIXER_OUT, out.len())?;
        if layer >= self.layers {
            return Err(bad(format!(
                "gdn prefill: layer {layer} of {}",
                self.layers
            )));
        }
        #[cfg(has_omega_q38_prefill)]
        {
            let mut info = GdnPfInfo::default();
            // SAFETY: ptr is live; every slice has the length omega reads or writes (checked).
            let rc = unsafe {
                ffi::omega_gpu_gdn_prefill(
                    self.ptr,
                    layer as u32,
                    tokens as u32,
                    qkv.as_ptr(),
                    z.as_ptr(),
                    ab.as_ptr(),
                    eps_l2,
                    eps_norm,
                    1.0,
                    1.0,
                    out.as_mut_ptr(),
                    &mut info,
                )
            };
            if rc != 0 {
                // SAFETY: no arguments.
                let detail = c_text(unsafe { ffi::omega_gpu_gdn_last_error() });
                return Err(rc_err("gdn prefill", rc, detail));
            }
            Ok(PrefillCost {
                chip_ns: info.elapsed_ns.iter().sum(),
                call_ns: info.call_ns,
            })
        }
        #[cfg(not(has_omega_q38_prefill))]
        {
            let _ = (eps_l2, eps_norm);
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for Gdn {
    fn drop(&mut self) {
        #[cfg(has_omega_q38)]
        // SAFETY: ptr came from omega_gpu_gdn_alloc and is freed once.
        unsafe {
            ffi::omega_gpu_gdn_free(self.ptr)
        }
    }
}

// ---------------------------------------------------------------------------
// head_dim-256 KV cache
// ---------------------------------------------------------------------------

/// Every attention layer's bf16 KV cache, RoPE table and q/k norm weights.
pub struct Attn256Cache {
    #[cfg(has_omega_q38)]
    ptr: *mut ffi::OmegaGpuDecodeAttn256Cache,
    layers: usize,
    max_pos: usize,
    rotary: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for Attn256Cache {}

impl fmt::Debug for Attn256Cache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Attn256Cache {{ layers: {}, max_pos: {}, rotary: {} }}",
            self.layers, self.max_pos, self.rotary
        )
    }
}

impl Attn256Cache {
    /// Bytes one layer's cache holds: K and V, 4 KV heads x `max_pos` x 256 bf16.
    pub const fn layer_bytes(max_pos: usize) -> usize {
        2 * 4 * max_pos * HEAD_DIM * 2
    }

    /// Allocates `layers` attention entries of `max_pos` positions; RoPE with `theta` on the
    /// first `rotary` dims (even, 2..=256).
    pub fn alloc(
        layers: usize,
        max_pos: usize,
        theta: f64,
        rotary: usize,
    ) -> Result<Self, OmegaGpuError> {
        Self::alloc_shape(layers, max_pos, theta, rotary, 0, 0, 0)
    }

    /// The same for another head shape (omega `omega_gpu_decode_attn256_cache_alloc_shape`):
    /// `q_heads` / `kv_heads` / `q_per_cta` (0, 0, 0 = the 27B's 24 / 4 / 6; Qwen3.6-35B-A3B: 16, 2, 4).
    #[allow(clippy::too_many_arguments)]
    pub fn alloc_shape(
        layers: usize,
        max_pos: usize,
        theta: f64,
        rotary: usize,
        q_heads: usize,
        kv_heads: usize,
        q_per_cta: usize,
    ) -> Result<Self, OmegaGpuError> {
        if layers == 0 || layers > MAX_LAYERS {
            return Err(bad(format!(
                "attn256: {layers} layers, 1..={MAX_LAYERS} supported"
            )));
        }
        if max_pos == 0 || max_pos > MAX_POS {
            return Err(bad(format!(
                "attn256: max_pos {max_pos} outside 1..={MAX_POS}"
            )));
        }
        if !(2..=HEAD_DIM).contains(&rotary) || !rotary.is_multiple_of(2) {
            return Err(bad(format!("attn256: rotary {rotary} (even, 2..=256)")));
        }
        #[cfg(has_omega_q38)]
        {
            let mut ptr = std::ptr::null_mut();
            // SAFETY: out pointer is local; plain arguments.
            let rc = unsafe {
                ffi::omega_gpu_decode_attn256_cache_alloc_shape(
                    layers as u32,
                    max_pos as u32,
                    theta,
                    rotary as u32,
                    q_heads as u32,
                    kv_heads as u32,
                    q_per_cta as u32,
                    &mut ptr,
                )
            };
            if rc != 0 || ptr.is_null() {
                // SAFETY: no arguments.
                let detail = c_text(unsafe { ffi::omega_gpu_decode_attn_last_error() });
                return Err(rc_err("attn256 alloc", rc, detail));
            }
            Ok(Self {
                ptr,
                layers,
                max_pos,
                rotary,
            })
        }
        #[cfg(not(has_omega_q38))]
        {
            let _ = (theta, q_heads, kv_heads, q_per_cta);
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn max_pos(&self) -> usize {
        self.max_pos
    }

    /// The C handle (`OmegaGpuDecodeAttn256Cache *`) for [`GraphConfig::attn`]; null in the
    /// stub. Stays owned by `self`.
    pub fn as_raw(&self) -> *mut c_void {
        #[cfg(has_omega_q38)]
        {
            self.ptr.cast()
        }
        #[cfg(not(has_omega_q38))]
        {
            std::ptr::null_mut()
        }
    }

    /// One layer's q_norm / k_norm as the checkpoint stores them (`[256]` each, zero-centred):
    /// folded to `1 + w` and permuted here.
    pub fn set_norms(
        &mut self,
        layer: usize,
        q_norm: &[f32],
        k_norm: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("attn256 q_norm", HEAD_DIM, q_norm.len())?;
        check_len("attn256 k_norm", HEAD_DIM, k_norm.len())?;
        if layer >= self.layers {
            return Err(bad(format!("attn256: layer {layer} of {}", self.layers)));
        }
        let fold =
            |w: &[f32]| perm_vec(&w.iter().map(|v| 1.0 + v).collect::<Vec<_>>(), self.rotary);
        let (q, k) = (fold(q_norm), fold(k_norm));
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; q and k hold 256 f32.
            let rc = unsafe {
                ffi::omega_gpu_decode_attn256_set_norms(
                    self.ptr,
                    layer as u32,
                    q.as_ptr(),
                    k.as_ptr(),
                )
            };
            if rc != 0 {
                // SAFETY: no arguments.
                let detail = c_text(unsafe { ffi::omega_gpu_decode_attn_last_error() });
                return Err(rc_err("attn256 set_norms", rc, detail));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        {
            let _ = (q, k);
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Causal prefill attention for attention layer `layer` (its index among the attention
    /// layers), `omega_gpu_prefill_attn256`: `m` tokens at positions `pos0 .. pos0 + m` (positions
    /// before `pos0` already in the cache). `qkv` `[m][8192]` in the decode call's row layout
    /// ([`gather_table`]'s first [`ATT_QKV`] slots, raw: the q/k norms and RoPE are applied
    /// here); appends their K and V and writes `out` `[m][6144]` (head h at h * 256, before the
    /// output gate).
    pub fn prefill(
        &mut self,
        layer: usize,
        pos0: usize,
        m: usize,
        qkv: &[f32],
        eps: f32,
        out: &mut [f32],
    ) -> Result<PrefillCost, OmegaGpuError> {
        if !(1..=PATTN_MAX_M).contains(&m) {
            return Err(bad(format!(
                "attn256 prefill: {m} tokens, 1..={PATTN_MAX_M} supported"
            )));
        }
        if pos0 + m > self.max_pos {
            return Err(bad(format!(
                "attn256 prefill: positions {pos0} + {m} > cache of {}",
                self.max_pos
            )));
        }
        check_len("attn256 prefill qkv", m * ATT_QKV, qkv.len())?;
        check_len("attn256 prefill out", m * MIXER_OUT, out.len())?;
        if layer >= self.layers {
            return Err(bad(format!(
                "attn256 prefill: layer {layer} of {}",
                self.layers
            )));
        }
        #[cfg(has_omega_q38_prefill)]
        {
            let mut info = Attn256PfInfo::default();
            // SAFETY: ptr is live; qkv and out have the lengths omega reads and writes (checked).
            let rc = unsafe {
                ffi::omega_gpu_prefill_attn256(
                    self.ptr,
                    layer as u32,
                    pos0 as u32,
                    m as u32,
                    qkv.as_ptr(),
                    eps,
                    1.0,
                    out.as_mut_ptr(),
                    &mut info,
                )
            };
            if rc != 0 {
                // SAFETY: no arguments.
                let detail = c_text(unsafe { ffi::omega_gpu_decode_attn_last_error() });
                return Err(rc_err("attn256 prefill", rc, detail));
            }
            Ok(PrefillCost {
                chip_ns: info.elapsed_ns.iter().sum(),
                call_ns: info.call_ns,
            })
        }
        #[cfg(not(has_omega_q38_prefill))]
        {
            let _ = eps;
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for Attn256Cache {
    fn drop(&mut self) {
        #[cfg(has_omega_q38)]
        // SAFETY: ptr came from omega_gpu_decode_attn256_cache_alloc and is freed once.
        unsafe {
            ffi::omega_gpu_decode_attn256_cache_free(self.ptr)
        }
    }
}

// ---------------------------------------------------------------------------
// The one-list decode token
// ---------------------------------------------------------------------------

/// Host check of the configuration, before any chip call (also in the stub).
pub fn check_config(cfg: &GraphConfig) -> Result<(), OmegaGpuError> {
    if cfg.layers == 0 || cfg.layers as usize > MAX_LAYERS {
        return Err(bad(format!(
            "q38 graph: {} layers, 1..={MAX_LAYERS} supported",
            cfg.layers
        )));
    }
    if cfg.vocab == 0 || cfg.vocab as usize > MAX_VOCAB {
        return Err(bad(format!(
            "q38 graph: vocab {} outside 1..={MAX_VOCAB}",
            cfg.vocab
        )));
    }
    let mask = if cfg.layers == 64 {
        u64::MAX
    } else {
        (1u64 << cfg.layers) - 1
    };
    let attn = (cfg.attn_mask & mask).count_ones();
    if (attn > 0 && cfg.attn.is_null()) || (attn < cfg.layers && cfg.gdn.is_null()) {
        return Err(bad(
            "q38 graph: a layer kind in use has no device state (gdn or attn handle)".into(),
        ));
    }
    if cfg.lm_head.is_null() {
        return Err(bad("q38 graph: the lm_head must be a device tensor".into()));
    }
    Ok(())
}

/// One recorded-per-token Qwen3.8 decode graph. Not `Sync`: callers serialise use behind a lock.
pub struct Q38Graph {
    #[cfg(has_omega_q38)]
    ptr: *mut ffi::OmegaGpuQ38Graph,
    layers: usize,
    attn_mask: u64,
    vocab: usize,
}

// SAFETY: an owning handle to device buffers; moved between threads, never shared unlocked.
unsafe impl Send for Q38Graph {}

impl fmt::Debug for Q38Graph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Q38Graph {{ layers: {}, attn_mask: {:#x}, vocab: {} }}",
            self.layers, self.attn_mask, self.vocab
        )
    }
}

impl Q38Graph {
    /// Allocates the graph's device buffers.
    ///
    /// # Safety
    /// `cfg.gdn`, `cfg.attn` and `cfg.lm_head` are live omega handles (`as_raw()` of the matching
    /// objects, null only for a kind no layer uses) that outlive the graph.
    pub unsafe fn create(cfg: &GraphConfig) -> Result<Self, OmegaGpuError> {
        check_config(cfg)?;
        #[cfg(has_omega_q38)]
        {
            let mut ptr = std::ptr::null_mut();
            let rc = ffi::omega_gpu_q38_graph_create(cfg, &mut ptr);
            if rc != 0 || ptr.is_null() {
                return Err(session_err("q38 graph create", rc));
            }
            Ok(Self {
                ptr,
                layers: cfg.layers as usize,
                attn_mask: cfg.attn_mask,
                vocab: cfg.vocab as usize,
            })
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn vocab(&self) -> usize {
        self.vocab
    }

    /// True when model layer `l` is full attention.
    pub fn is_attention(&self, l: usize) -> bool {
        l < 64 && self.attn_mask >> l & 1 == 1
    }

    /// Binds one layer's projections (omega checks the shapes) and copies its norms (f32
    /// `[HIDDEN]` each, already folded to `1 + w`). DeltaNet: `mix` `[16384][5120]`, `ab` bf16
    /// `[96][5120]` (a/b interleaved, each value times 2^-64). Attention: `mix` `[14336][5120]`,
    /// `ab` null. Both: `o` `[5120][6144]`, `gate_up` `[34816][5120]`, `down` `[5120][17408]`.
    ///
    /// # Safety
    /// The handles are live `OmegaGpuFp8Tensor` / `OmegaGpuBf16Tensor` pointers that outlive the
    /// graph.
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
        post_norm: &[f32],
    ) -> Result<(), OmegaGpuError> {
        check_len("q38 graph in_norm", HIDDEN, in_norm.len())?;
        check_len("q38 graph post_norm", HIDDEN, post_norm.len())?;
        if layer >= self.layers {
            return Err(bad(format!("q38 graph: layer {layer} of {}", self.layers)));
        }
        if mix.is_null() || o.is_null() || gate_up.is_null() || down.is_null() {
            return Err(bad(format!(
                "q38 graph layer {layer}: a projection has no device handle"
            )));
        }
        if ab.is_null() != self.is_attention(layer) {
            return Err(bad(format!(
                "q38 graph layer {layer}: the a|b tensor is for DeltaNet layers only"
            )));
        }
        #[cfg(has_omega_q38)]
        {
            let rc = ffi::omega_gpu_q38_graph_set_layer(
                self.ptr,
                layer as u32,
                mix,
                ab,
                o,
                gate_up,
                down,
                in_norm.as_ptr(),
                post_norm.as_ptr(),
            );
            if rc != 0 {
                return Err(session_err("q38 graph set_layer", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Uses an FP8 lm_head (`[vocab][HIDDEN]`, one expert) for every later token instead of the bf16 one
    /// (omega `omega_gpu_q38_graph_set_lm_head_fp8`, qwen38/decode-speed); null goes back to bf16.
    ///
    /// # Safety
    /// `head` is null or a live `OmegaGpuFp8Tensor` that outlives every later token of this graph.
    pub unsafe fn set_lm_head_fp8(&mut self, head: *mut c_void) -> Result<(), OmegaGpuError> {
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; head per the caller's contract.
            let rc = unsafe { ffi::omega_gpu_q38_graph_set_lm_head_fp8(self.ptr, head) };
            if rc != 0 {
                return Err(session_err("q38 graph set_lm_head_fp8", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        {
            let _ = head;
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Copies the final RMSNorm weight (f32 `[HIDDEN]`, folded to `1 + w`).
    pub fn set_final_norm(&mut self, final_norm: &[f32]) -> Result<(), OmegaGpuError> {
        check_len("q38 graph final_norm", HIDDEN, final_norm.len())?;
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; the slice holds HIDDEN f32 (checked).
            let rc =
                unsafe { ffi::omega_gpu_q38_graph_set_final_norm(self.ptr, final_norm.as_ptr()) };
            if rc != 0 {
                return Err(session_err("q38 graph set_final_norm", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// One whole token as one command list: `x_embed` (the token's embedding row, f32
    /// `[HIDDEN]`) at position `pos` -> the greedy token (lowest index of the max logit) and what
    /// it cost. Attention layers append K/V at `pos`; DeltaNet layers advance their state.
    pub fn token(
        &mut self,
        x_embed: &[f32],
        pos: usize,
    ) -> Result<(u32, GraphStats), OmegaGpuError> {
        check_len("q38 graph x_embed", HIDDEN, x_embed.len())?;
        let pos = u32::try_from(pos).map_err(|_| bad(format!("q38 graph: pos {pos} too large")))?;
        #[cfg(has_omega_q38)]
        {
            let mut token = 0u32;
            let mut st = GraphStats::default();
            // SAFETY: ptr is live; x_embed holds HIDDEN f32 (checked); token and st are local.
            let rc = unsafe {
                ffi::omega_gpu_q38_graph_token(self.ptr, x_embed.as_ptr(), pos, &mut token, &mut st)
            };
            if rc != 0 {
                return Err(session_err("q38 graph token", rc));
            }
            Ok((token, st))
        }
        #[cfg(not(has_omega_q38))]
        {
            let _ = pos;
            Err(OmegaGpuError::Unavailable)
        }
    }

    /// Copies the last token's device residual `x`, final-normed `hidden` (both `[HIDDEN]`) and
    /// `logits` (`[vocab]`); any may be `None`. Costs a device read.
    pub fn read(
        &mut self,
        x: Option<&mut [f32]>,
        hidden: Option<&mut [f32]>,
        logits: Option<&mut [f32]>,
    ) -> Result<(), OmegaGpuError> {
        if let Some(x) = &x {
            check_len("q38 graph read x", HIDDEN, x.len())?;
        }
        if let Some(h) = &hidden {
            check_len("q38 graph read hidden", HIDDEN, h.len())?;
        }
        if let Some(l) = &logits {
            check_len("q38 graph read logits", self.vocab, l.len())?;
        }
        #[cfg(has_omega_q38)]
        {
            let p = |s: Option<&mut [f32]>| s.map_or(std::ptr::null_mut(), |s| s.as_mut_ptr());
            // SAFETY: ptr is live; each non-null buffer has the length omega copies (checked).
            let rc = unsafe { ffi::omega_gpu_q38_graph_read(self.ptr, p(x), p(hidden), p(logits)) };
            if rc != 0 {
                return Err(session_err("q38 graph read", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Copies the first `out.len()` f32 of one intermediate of the last token (test and trace).
    pub fn read_buf(&mut self, which: Buf, out: &mut [f32]) -> Result<(), OmegaGpuError> {
        if out.len() > which.capacity() {
            return Err(bad(format!(
                "q38 graph read_buf {which:?}: {} values, the buffer holds {}",
                out.len(),
                which.capacity()
            )));
        }
        #[cfg(has_omega_q38)]
        {
            // SAFETY: ptr is live; out holds out.len() <= the buffer's capacity f32 (checked).
            let rc = unsafe {
                ffi::omega_gpu_q38_graph_read_buf(
                    self.ptr,
                    which as i32,
                    out.as_mut_ptr(),
                    out.len() as u32,
                )
            };
            if rc != 0 {
                return Err(session_err("q38 graph read_buf", rc));
            }
            Ok(())
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// Timeline (measurement): when on, every later token also records a GPU timestamp before
    /// the first launch and after each launch. Call between tokens. Off by default.
    pub fn set_stamps(&mut self, on: bool) {
        #[cfg(has_omega_q38)]
        // SAFETY: ptr is live; plain integer argument.
        unsafe {
            ffi::omega_gpu_q38_graph_set_stamps(self.ptr, on as i32)
        }
        #[cfg(not(has_omega_q38))]
        let _ = on;
    }

    /// The last token's timeline: `launches + 1` GPU timestamps (ns) and one kind per launch
    /// (launch `i` took `stamps[i + 1] - stamps[i]`). Refused when stamps are off or the token
    /// failed.
    pub fn stamps(&self) -> Result<(Vec<u64>, Vec<u8>), OmegaGpuError> {
        #[cfg(has_omega_q38)]
        {
            // 10 per layer + 4, with room to spare.
            let cap = (MAX_LAYERS * 16 + 16) as u32;
            let mut stamps = vec![0u64; cap as usize + 1];
            let mut kinds = vec![0u8; cap as usize + 1];
            // SAFETY: ptr is live; both buffers hold cap + 1 entries.
            let n = unsafe {
                ffi::omega_gpu_q38_graph_stamps(
                    self.ptr,
                    stamps.as_mut_ptr(),
                    kinds.as_mut_ptr(),
                    cap + 1,
                )
            };
            if n < 0 {
                return Err(session_err("q38 graph stamps", n));
            }
            let n = n as usize;
            stamps.truncate(n + 1);
            kinds.truncate(n);
            Ok((stamps, kinds))
        }
        #[cfg(not(has_omega_q38))]
        Err(OmegaGpuError::Unavailable)
    }

    /// TEST ONLY: submit after every launch (the batching comparison).
    pub fn set_flush_each(&mut self, on: bool) {
        #[cfg(has_omega_q38)]
        // SAFETY: ptr is live; plain integer argument.
        unsafe {
            ffi::omega_gpu_q38_graph_set_flush_each(self.ptr, on as i32)
        }
        #[cfg(not(has_omega_q38))]
        let _ = on;
    }
}

impl Drop for Q38Graph {
    fn drop(&mut self) {
        #[cfg(has_omega_q38)]
        // SAFETY: ptr came from omega_gpu_q38_graph_create and is freed once.
        unsafe {
            ffi::omega_gpu_q38_graph_free(self.ptr)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gather_table_is_a_permutation_and_matches_omega() {
        let t = gather_table(64);
        let mut seen = vec![false; ATT_MIX];
        for &s in &t {
            assert!(!std::mem::replace(&mut seen[s as usize], true), "{s} twice");
        }
        assert!(seen.iter().all(|&s| s));
        // q head 0 dim 0 comes from row 0; the gate of head 0 from row 256; v from after k.
        assert_eq!((t[0], t[ATT_QKV], t[7168]), (0, 256, 24 * 512 + 4 * 256));
        if is_native() {
            assert_eq!(gather_table_c().unwrap(), t);
        }
    }

    #[test]
    fn perm_slot_matches_the_documented_table() {
        // rotary 64: P = 32.
        assert_eq!(perm_slot(0, 64), 0);
        assert_eq!(perm_slot(31, 64), 31);
        assert_eq!(perm_slot(32, 64), 64);
        assert_eq!(perm_slot(127, 64), 159);
        assert_eq!(perm_slot(128, 64), 32);
        assert_eq!(perm_slot(159, 64), 63);
        assert_eq!(perm_slot(160, 64), 160);
        assert_eq!(perm_slot(255, 64), 255);
        // A permutation: every model dim exactly once.
        let mut seen = [false; 256];
        for i in 0..256 {
            assert!(!seen[perm_slot(i, 64)]);
            seen[perm_slot(i, 64)] = true;
        }
        // Full RoPE is the identity.
        assert!((0..256).all(|i| perm_slot(i, 256) == i));
    }

    #[test]
    fn kernel_pairs_are_the_model_pairs() {
        for i in 0..32 {
            assert_eq!(perm_slot(i + 128, 64), perm_slot(i, 64) + 32);
        }
    }

    #[test]
    fn config_check_refuses_missing_state() {
        let mut lm = 0u8;
        let cfg = GraphConfig {
            layers: 4,
            attn_mask: 0x8,
            vocab: 248320,
            eps: 1e-6,
            eps_l2: 1e-6,
            gdn: std::ptr::null_mut(),
            attn: std::ptr::null_mut(),
            lm_head: (&mut lm as *mut u8).cast(),
        };
        assert!(check_config(&cfg).is_err());
        let ok = GraphConfig {
            gdn: (&mut lm as *mut u8).cast(),
            attn: (&mut lm as *mut u8).cast(),
            ..cfg
        };
        assert!(check_config(&ok).is_ok());
        // Only DeltaNet layers in the prefix: no attention state needed.
        let gdn_only = GraphConfig {
            layers: 3,
            attn: std::ptr::null_mut(),
            ..ok
        };
        assert!(check_config(&gdn_only).is_ok());
        assert!(check_config(&GraphConfig { layers: 65, ..ok }).is_err());
    }
}

/// bf16 `[n][k]` -> e4m3 `[n][k]` + bf16 scale_inv `[n/128][k/128]` (omega `omega_gpu_fp8_quantize_bf16`: per 128 x 128
/// block, scale = max|w| / 448 rounded up to bf16, round to nearest even). Stripes of 128 rows run on `threads` threads.
pub fn quantize_fp8_bf16(
    n: usize,
    k: usize,
    w: &[u16],
    threads: usize,
) -> Result<(Vec<u8>, Vec<u16>), OmegaGpuError> {
    const B: usize = 128;
    if n == 0 || k == 0 || !n.is_multiple_of(B) || !k.is_multiple_of(B) {
        return Err(OmegaGpuError::Unavailable);
    }
    check_len("fp8 quantise bf16 words", n * k, w.len())?;
    let mut w8 = vec![0u8; n * k];
    let mut s = vec![0u16; (n / B) * (k / B)];
    #[cfg(has_omega_q38)]
    {
        let stripes = n / B;
        let per = stripes.div_ceil(threads.max(1));
        let rcs: Vec<i32> = std::thread::scope(|sc| {
            let hs: Vec<_> = w
                .chunks(per * B * k)
                .zip(w8.chunks_mut(per * B * k))
                .zip(s.chunks_mut(per * (k / B)))
                .map(|((wi, oi), si)| {
                    sc.spawn(move || {
                        // SAFETY: wi holds rows * k words, oi rows * k bytes, si (rows / 128) * (k / 128) words.
                        unsafe {
                            ffi::omega_gpu_fp8_quantize_bf16(
                                (wi.len() / k) as u32,
                                k as u32,
                                wi.as_ptr(),
                                oi.as_mut_ptr(),
                                si.as_mut_ptr(),
                            )
                        }
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(-1)).collect()
        });
        if let Some(&rc) = rcs.iter().find(|&&rc| rc != 0) {
            return Err(session_err("fp8 quantise", rc));
        }
        Ok((w8, s))
    }
    #[cfg(not(has_omega_q38))]
    {
        let _ = (&mut w8, &mut s, threads);
        Err(OmegaGpuError::Unavailable)
    }
}
