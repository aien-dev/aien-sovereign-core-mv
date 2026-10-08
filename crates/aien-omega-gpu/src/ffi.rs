//! Raw declarations, written by hand from omega `src/omega_gpu_matmul_api.h`
//! at the commit pinned in `omega.lock` (e5593ae; matmul header unchanged since 6b940fa, attention header
//! extended by OM-2 e5593ae: staging counters appended to `OmegaGpuAttnInfo`). Nothing generated.
//! Includes the resident-weights handles added by omega FB-1 cut 1b.
use std::os::raw::{c_char, c_int};

pub const OMEGA_GPU_MATMUL_OK: c_int = 0;
pub const OMEGA_GPU_MATMUL_BAD_ARGS: c_int = -1;
pub const OMEGA_GPU_MATMUL_TOO_LARGE: c_int = -2;
pub const OMEGA_GPU_MATMUL_CODEGEN_FAIL: c_int = -3;
pub const OMEGA_GPU_MATMUL_CHIP_FAIL: c_int = -4;
pub const OMEGA_GPU_MATMUL_PARITY_FAIL: c_int = -5;

pub const OMEGA_GPU_MATMUL_TILE_M: u32 = 16;
pub const OMEGA_GPU_MATMUL_TILE_N: u32 = 8;
pub const OMEGA_GPU_MATMUL_TILE_K: u32 = 16;
pub const OMEGA_GPU_MATMUL_MAX_CTAS: u32 = 64;

/// Mirrors `OmegaGpuMatmulInfo` field for field, same order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OmegaGpuMatmulInfo {
    pub m: u32,
    pub k: u32,
    pub n: u32,
    pub padded_m: u32,
    pub padded_n: u32,
    pub k_slices: u32,
    pub chip_calls: u32,
    pub rows_per_call: u32,
    pub elapsed_ns: u64,
    pub max_abs_err: f32,
    pub max_rel_err: f32,
    pub mismatch_count: usize,
    pub parity_verified: bool,
    pub completion_marker: u32,
    pub kernel_cache_hit: bool,
    pub target_chip: [c_char; 64],
    pub sm_architecture: u32,
    // cut 1b additions (appended in the C header, same order)
    pub grid_x: u32,
    pub grid_y: u32,
    pub padded_k: u32,
    pub resident: bool,
    pub oracle_ran: bool,
    pub call_ns: u64,
}

impl OmegaGpuMatmulInfo {
    pub fn zeroed() -> Self {
        Self {
            m: 0,
            k: 0,
            n: 0,
            padded_m: 0,
            padded_n: 0,
            k_slices: 0,
            chip_calls: 0,
            rows_per_call: 0,
            elapsed_ns: 0,
            max_abs_err: 0.0,
            max_rel_err: 0.0,
            mismatch_count: 0,
            parity_verified: false,
            completion_marker: 0,
            kernel_cache_hit: false,
            target_chip: [0; 64],
            sm_architecture: 0,
            grid_x: 0,
            grid_y: 0,
            padded_k: 0,
            resident: false,
            oracle_ran: false,
            call_ns: 0,
        }
    }
}

/// Opaque resident weight tensor (omega `OmegaGpuTensor`).
#[repr(C)]
pub struct OmegaGpuTensor {
    _private: [u8; 0],
}
#[cfg(has_omega_gpu)]
/// omega `OmegaGpuServingBounds` (`src/omega_gpu_serving.h`), field for field.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OmegaGpuServingBounds {
    pub max_context: u32,
    pub max_seqs: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub kv_block_size: u32,
    pub max_rows: u32,
    pub max_k: u32,
    pub max_n: u32,
    pub max_n_one_row: u32,
    pub kernel_slots: u32,
}

#[cfg(has_omega_gpu)]
extern "C" {
    pub fn omega_gpu_matmul_bf16(
        m: u32,
        k: u32,
        n: u32,
        a: *const u16,
        b: *const u16,
        c: *mut f32,
        info: *mut OmegaGpuMatmulInfo,
    ) -> c_int;
    pub fn omega_gpu_matmul_f32(
        m: u32,
        k: u32,
        n: u32,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        info: *mut OmegaGpuMatmulInfo,
    ) -> c_int;
    pub fn omega_gpu_matmul_rc_name(rc: c_int) -> *const c_char;
    pub fn omega_gpu_matmul_cache_clear();
    pub fn omega_gpu_tensor_upload_bf16(
        k: u32,
        n: u32,
        b: *const u16,
        out: *mut *mut OmegaGpuTensor,
    ) -> c_int;
    pub fn omega_gpu_tensor_upload_f32(
        k: u32,
        n: u32,
        b: *const f32,
        out: *mut *mut OmegaGpuTensor,
    ) -> c_int;
    pub fn omega_gpu_tensor_free(t: *mut OmegaGpuTensor);
    pub fn omega_gpu_tensor_shape(t: *const OmegaGpuTensor, k: *mut u32, n: *mut u32);
    pub fn omega_gpu_matmul_resident_bf16(
        m: u32,
        a: *const u16,
        b: *const OmegaGpuTensor,
        c: *mut f32,
        info: *mut OmegaGpuMatmulInfo,
    ) -> c_int;
    pub fn omega_gpu_matmul_resident_f32(
        m: u32,
        a: *const f32,
        b: *const OmegaGpuTensor,
        c: *mut f32,
        info: *mut OmegaGpuMatmulInfo,
    ) -> c_int;
    pub fn omega_gpu_matmul_last_error() -> *const c_char;
    pub fn omega_gpu_matmul_is_blocked() -> c_int;
    pub fn omega_gpu_device_close();
    // omega c0369e6 `src/omega_gpu_matmul_api.h:130-133`: CTA budget per launch
    // (default OMEGA_GPU_MATMUL_MAX_CTAS; 0 restores it). Clears the kernel cache.
    pub fn omega_gpu_matmul_set_cta_budget(ctas: u32);
    pub fn omega_gpu_matmul_cta_budget() -> u32;
    // omega d6d82f/session-spin-default `src/omega_gpu_session.h`: process-wide marker-wait spin
    // window in us for launches whose own spin_us is 0 (0 = sleep-poll, the default);
    // -1 above OMEGA_GPU_SESSION_MAX_SPIN_US with the setting unchanged (omega#328).
    pub fn omega_gpu_session_set_spin_us(us: u32) -> c_int;
    pub fn omega_gpu_session_spin_us() -> u32;
    // omega c1/lib-serving `src/omega_gpu_serving.h` (omega#333 reservation, shipped in libomega_gpu.a by
    // omega#338): one all-or-nothing call; 0 on success, else the first failing OMEGA_GPU_MATMUL_* /
    // OMEGA_GPU_ATTN_* code (omega_gpu_matmul_last_error() has the text).
    pub fn omega_gpu_reserve_serving(b: *const OmegaGpuServingBounds) -> c_int;
    pub fn omega_gpu_serving_release();
}

// ---- omega `src/omega_gpu_elementwise_api.h` (FB-1 cut 4, pinned 2636409) ----
pub const OMEGA_GPU_EW_OK: c_int = 0;
pub const OMEGA_GPU_EW_BAD_ARGS: c_int = -1;
pub const OMEGA_GPU_EW_TOO_LARGE: c_int = -2;
pub const OMEGA_GPU_EW_CODEGEN_FAIL: c_int = -3;
pub const OMEGA_GPU_EW_CHIP_FAIL: c_int = -4;
pub const OMEGA_GPU_EW_UNWRITTEN: c_int = -5;

/// Mirrors `OmegaGpuEwInfo` field for field, same order.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OmegaGpuEwInfo {
    pub chip_calls: u32,
    pub elapsed_ns: u64,
    pub completion_marker: u32,
    pub unwritten_words: u32,
    pub kernel_cache_hit: bool,
    pub gpr_count: u32,
    pub insn_count: u32,
    pub threads_per_cta: u32,
    pub ctas_last_launch: u32,
    pub target_chip: [c_char; 64],
    pub sm_architecture: u32,
    pub call_ns: u64,
}

impl OmegaGpuEwInfo {
    pub fn zeroed() -> Self {
        Self {
            chip_calls: 0,
            elapsed_ns: 0,
            completion_marker: 0,
            unwritten_words: 0,
            kernel_cache_hit: false,
            gpr_count: 0,
            insn_count: 0,
            threads_per_cta: 0,
            ctas_last_launch: 0,
            target_chip: [0; 64],
            sm_architecture: 0,
            call_ns: 0,
        }
    }
}

#[cfg(has_omega_gpu)]
extern "C" {
    pub fn omega_gpu_rmsnorm_f32(
        rows: u32,
        dim: u32,
        x: *const f32,
        weight: *const f32,
        eps: f32,
        out: *mut f32,
        info: *mut OmegaGpuEwInfo,
    ) -> c_int;
    pub fn omega_gpu_rope_f32(
        heads: u32,
        head_dim: u32,
        v: *const f32,
        cos_half: *const f32,
        sin_half: *const f32,
        out: *mut f32,
        info: *mut OmegaGpuEwInfo,
    ) -> c_int;
    pub fn omega_gpu_swiglu_f32(
        n: u32,
        gate: *const f32,
        up: *const f32,
        out: *mut f32,
        info: *mut OmegaGpuEwInfo,
    ) -> c_int;
    pub fn omega_gpu_elementwise_rc_name(rc: c_int) -> *const c_char;
}

// ---- omega `src/omega_gpu_attention_api.h` (FB-1 cut 5 + 4b + OM-1/OM-2, pinned e5593ae) ----
pub const OMEGA_GPU_ATTN_OK: c_int = 0;
pub const OMEGA_GPU_ATTN_BAD_ARGS: c_int = -1;
pub const OMEGA_GPU_ATTN_TOO_LARGE: c_int = -2;
pub const OMEGA_GPU_ATTN_CODEGEN_FAIL: c_int = -3;
pub const OMEGA_GPU_ATTN_CHIP_FAIL: c_int = -4;
pub const OMEGA_GPU_ATTN_UNWRITTEN: c_int = -5;
/// `OmegaGpuAttnInfo::kv_source`: KV reached the kernel through the per-call staging copy.
pub const OMEGA_GPU_ATTN_KV_STAGED: u32 = 0;

/// Mirrors `OmegaGpuAttnInfo` field for field, same order (note `call_ns` is third).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct OmegaGpuAttnInfo {
    pub chip_calls: u32,
    pub elapsed_ns: u64,
    pub call_ns: u64,
    pub completion_marker: u32,
    pub unwritten_words: u32,
    pub gpr_count: u32,
    pub insn_count: u32,
    pub kernel_cache_hit: bool,
    pub ctas_last_launch: u32,
    pub threads_per_cta: u32,
    pub target_chip: [c_char; 64],
    pub sm_architecture: u32,
    // OM-2 (omega e5593ae): appended after sm_architecture, ABI order preserved.
    /// Bytes of KV copied into the staging buffer by this call (one copy per physical block per launch).
    pub kv_bytes_staged: u64,
    /// Bytes per-reference staging would have copied (every block-table entry once).
    pub kv_bytes_naive: u64,
    /// Block-table entries walked.
    pub kv_blocks_logical: u32,
    /// Distinct physical blocks actually copied.
    pub kv_blocks_unique: u32,
    pub q_bytes: u64,
    pub tab_bytes: u64,
    pub out_bytes: u64,
    /// `OMEGA_GPU_ATTN_KV_STAGED` (0): the only source in this cut; there is no resident KV.
    pub kv_source: u32,
}

impl OmegaGpuAttnInfo {
    pub fn zeroed() -> Self {
        Self {
            chip_calls: 0,
            elapsed_ns: 0,
            call_ns: 0,
            completion_marker: 0,
            unwritten_words: 0,
            gpr_count: 0,
            insn_count: 0,
            kernel_cache_hit: false,
            ctas_last_launch: 0,
            threads_per_cta: 0,
            target_chip: [0; 64],
            sm_architecture: 0,
            kv_bytes_staged: 0,
            kv_bytes_naive: 0,
            kv_blocks_logical: 0,
            kv_blocks_unique: 0,
            q_bytes: 0,
            tab_bytes: 0,
            out_bytes: 0,
            kv_source: 0,
        }
    }
}

// Field-for-field mirror guard against omega `src/omega_gpu_attention_api.h` at e5593ae: the C
// struct is 176 bytes and these are its offsets (checked with _Static_assert against the header on
// 2026-10-04; e5593ae is the squash-merge of OM-2 (f777036 was cd80bf4 rebased onto omega main 55d05d6, same attention sources)). A header change that moves a field fails this build instead of misreading a counter.
const _: () = assert!(std::mem::size_of::<OmegaGpuAttnInfo>() == 176);
const _: () = assert!(std::mem::offset_of!(OmegaGpuAttnInfo, sm_architecture) == 116);
const _: () = assert!(std::mem::offset_of!(OmegaGpuAttnInfo, kv_bytes_staged) == 120);
const _: () = assert!(std::mem::offset_of!(OmegaGpuAttnInfo, kv_blocks_logical) == 136);
const _: () = assert!(std::mem::offset_of!(OmegaGpuAttnInfo, q_bytes) == 144);
const _: () = assert!(std::mem::offset_of!(OmegaGpuAttnInfo, kv_source) == 168);

/// Mirrors `OmegaGpuKvLayout` (itself a mirror of aien-kv-cache `KvLayoutDesc`),
/// field for field, strides in bytes.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct OmegaGpuKvLayout {
    pub block_stride_bytes: u64,
    pub layer_stride_bytes: u64,
    pub kv_plane_stride_bytes: u64,
    pub token_stride_bytes: u64,
    pub head_stride_bytes: u64,
    pub pool_bytes: u64,
    pub num_blocks: u32,
    pub num_layers: u32,
    pub block_size: u32,
}

#[cfg(has_omega_gpu)]
extern "C" {
    pub fn omega_gpu_gqa_attention_f32(
        q: *const f32,
        k_cache: *const f32,
        v_cache: *const f32,
        seq_len: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        out: *mut f32,
        info: *mut OmegaGpuAttnInfo,
    ) -> c_int;
    pub fn omega_gpu_paged_attention_bf16(
        q: *const f32,
        pool: *const u8,
        layout: *const OmegaGpuKvLayout,
        block_ids: *const u32,
        num_block_ids: u32,
        context_len: u32,
        layer_idx: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        out: *mut f32,
        info: *mut OmegaGpuAttnInfo,
    ) -> c_int;
    pub fn omega_gpu_paged_attention_batch_bf16(
        q: *const f32,
        pool: *const u8,
        layout: *const OmegaGpuKvLayout,
        block_tables: *const i32,
        context_lens: *const i32,
        max_blocks_per_seq: u32,
        num_seqs: u32,
        layer_idx: u32,
        num_q_heads: u32,
        num_kv_heads: u32,
        head_dim: u32,
        out: *mut f32,
        info: *mut OmegaGpuAttnInfo,
    ) -> c_int;
    pub fn omega_gpu_attention_rc_name(rc: c_int) -> *const c_char;
    pub fn omega_gpu_attention_last_error() -> *const c_char;
}
