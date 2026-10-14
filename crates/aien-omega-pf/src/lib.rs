//! Safe Rust wrapper over omega's device-resident prefill API (`src/omega_gpu_prefill.h`, W-L1G
//! cut 1e76e728) and the causal prefill attention (`src/omega_gpu_prefill_attn_api.h`, W-L1A).
//! No CUDA: every launch is omega's own Blackwell code generator.
//!
//! What is bound:
//!  * buffers: `omega_gpu_pf_alloc_dev` (GPU-cached, device-private: the CPU never touches them)
//!    and `omega_gpu_pf_alloc_host` (GPU-uncached session memory the CPU reads and writes).
//!  * `omega_gpu_pf_quant`: f32 or bf16 rows of `k` -> e4m3 rows + f32 power-of-two scales per
//!    128-wide group (`omega_blackwell_codegen_qg.h`).
//!  * `omega_gpu_pf_qgemm`: grouped GEMM on QMMA, weights read IN PLACE from FP8 GEMV tensors
//!    through a tile table of per-expert `(w_va, s_va)`; layouts [`Layout::Plain`] (f32 rows of n)
//!    and [`Layout::SwigluNat`] (bf16 act rows of n/2, act = silu(gate) * up).
//!  * `omega_gpu_prefill_attn` (+ `_dev`): one call for all `m` prompt tokens of a layer.
//!
//! The two traits [`PrefillDevice`] and [`PrefillAttention`] are the seams the whole-prompt
//! driver (`aien-inference-abi`, `qwen3_moe_omega_prefill`) is written against, so host tests use
//! fakes. Without the native library every call returns `Unavailable`.
use aien_omega_gpu::OmegaGpuError;

/// Scale group width of the quantiser and the GEMM (k blocks of 128).
pub const BLOCK: usize = 128;
/// Widest row the quantiser takes (`omega_pq_spec_ok`: 128 <= k <= 4096).
pub const QUANT_MAX_K: usize = 4096;
/// Widest `k` the QMMA GEMM takes (`omega_qg_spec_ok`).
pub const GEMM_MAX_K: usize = 16384;
/// Bytes of one tile-table entry (`OMEGA_QG_TILE_BYTES`).
pub const TILE_BYTES: usize = 32;
/// Most tiles one GEMM launch takes (grid y, `omega_gpu_pf_qgemm` refuses more with TOO_LARGE).
pub const MAX_TILES: usize = 65535;
/// Rows per tile at the default shape (16 x rt 4); the device reports the live value.
pub const DEFAULT_BM: usize = 64;
/// GEMM output columns one CTA covers at the default 8 warps (`n` must be a multiple).
pub const DEFAULT_SPAN: usize = 128;

/// True when the quantiser / QMMA GEMM / pf buffers are linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_pf)
}
/// True when `omega_gpu_prefill_attn` (host-pointer entry) is linked.
pub const fn has_attention() -> bool {
    cfg!(has_omega_pattn)
}
/// True when `omega_gpu_prefill_attn_dev` (device-pointer entry) is linked.
pub const fn has_attention_dev() -> bool {
    cfg!(has_omega_pattn_dev)
}

/// GEMM output layout (`OMEGA_GPU_PF_PLAIN` / `OMEGA_GPU_PF_SWIGLU_NAT`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Layout {
    /// `y` f32 rows of `n`.
    Plain = 0,
    /// Tensor `[E][2I][k]` (gate rows then up rows, checkpoint order); `y` bf16 act rows of `I = n / 2`.
    SwigluNat = 2,
}

/// Quantiser input element type (`OMEGA_QG_IN_F32` / `OMEGA_QG_IN_BF16`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum QuantIn {
    F32 = 0,
    Bf16 = 1,
}

/// One tile-table entry, byte-exact with the C `Tile` (`{ w_va u64, s_va u64, row0 u32, cnt u32, pad u64 }`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[repr(C)]
pub struct Tile {
    pub w_va: u64,
    pub s_va: u64,
    pub row0: u32,
    pub cnt: u32,
    pub pad: u64,
}

const _: () = assert!(std::mem::size_of::<Tile>() == TILE_BYTES);

impl Tile {
    /// Little-endian bytes as the kernel reads them.
    pub fn to_bytes(&self) -> [u8; TILE_BYTES] {
        let mut b = [0u8; TILE_BYTES];
        b[0..8].copy_from_slice(&self.w_va.to_le_bytes());
        b[8..16].copy_from_slice(&self.s_va.to_le_bytes());
        b[16..20].copy_from_slice(&self.row0.to_le_bytes());
        b[20..24].copy_from_slice(&self.cnt.to_le_bytes());
        b[24..32].copy_from_slice(&self.pad.to_le_bytes());
        b
    }
    pub fn from_bytes(b: &[u8]) -> Self {
        let u64_at = |o: usize| u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes"));
        let u32_at = |o: usize| u32::from_le_bytes(b[o..o + 4].try_into().expect("4 bytes"));
        Self { w_va: u64_at(0), s_va: u64_at(8), row0: u32_at(16), cnt: u32_at(20), pad: u64_at(24) }
    }
}

/// The tile table for rows grouped by expert (`counts[e]` rows of expert `e`, experts in order),
/// a Rust mirror of `omega_gpu_pf_qgemm_tiles_va`: experts with no rows get no tile, each tile
/// covers up to `bm` rows of one expert, `row0` counts rows over all experts.
pub fn build_tiles(bm: usize, w_va: &[u64], s_va: &[u64], counts: &[u32]) -> Result<Vec<Tile>, OmegaGpuError> {
    if bm == 0 || w_va.len() != counts.len() || s_va.len() != counts.len() {
        return Err(bad(format!(
            "pf tiles: bm {bm}, {} weight / {} scale addresses for {} experts",
            w_va.len(),
            s_va.len(),
            counts.len()
        )));
    }
    let mut tiles = Vec::new();
    let mut row = 0u32;
    for (e, &c) in counts.iter().enumerate() {
        let mut r = 0u32;
        while r < c {
            let cnt = (c - r).min(bm as u32);
            tiles.push(Tile { w_va: w_va[e], s_va: s_va[e], row0: row + r, cnt, pad: 0 });
            r += cnt;
        }
        row += c;
    }
    if tiles.len() > MAX_TILES {
        return Err(bad(format!("pf tiles: {} tiles > {MAX_TILES}", tiles.len())));
    }
    Ok(tiles)
}

/// Host check of the quantiser rule (`omega_pq_spec_ok`), before any chip call.
pub fn check_quant(k: usize) -> Result<(), OmegaGpuError> {
    if !k.is_multiple_of(BLOCK) || !(BLOCK..=QUANT_MAX_K).contains(&k) {
        return Err(bad(format!("pf quant k={k} refused: multiple of {BLOCK}, {BLOCK} <= k <= {QUANT_MAX_K}")));
    }
    Ok(())
}

/// Host check of the GEMM rule at the default shape (rt 4, 8 warps; `omega_qg_spec_ok`).
pub fn check_qgemm(n: usize, k: usize, layout: Layout, ntiles: usize) -> Result<(), OmegaGpuError> {
    if !k.is_multiple_of(BLOCK) || !(BLOCK..=GEMM_MAX_K).contains(&k) {
        return Err(bad(format!("pf qgemm k={k} refused: multiple of {BLOCK}, {BLOCK} <= k <= {GEMM_MAX_K}")));
    }
    if n == 0 || !n.is_multiple_of(DEFAULT_SPAN) || n * 4 * DEFAULT_BM >= 1 << 23 {
        return Err(bad(format!("pf qgemm n={n} refused: positive multiple of {DEFAULT_SPAN}, n * 256 < 2^23")));
    }
    if layout == Layout::SwigluNat && !n.is_multiple_of(2 * BLOCK) {
        return Err(bad(format!("pf qgemm SwiGLU n={n} refused: multiple of 256 (gate and up halves on scale blocks)")));
    }
    if ntiles > MAX_TILES {
        return Err(bad(format!("pf qgemm {ntiles} tiles > {MAX_TILES}")));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// The seams the driver is written against.
// ---------------------------------------------------------------------------

/// A device buffer: its GPU address, size, and (host buffers only) the CPU view.
pub trait PfBuffer {
    fn va(&self) -> u64;
    fn bytes(&self) -> usize;
    /// The CPU view of a HOST buffer; `None` for device-private buffers (the CPU never touches those).
    fn host(&self) -> Option<&[u8]>;
    fn host_mut(&mut self) -> Option<&mut [u8]>;
}

/// One grouped GEMM launch (`omega_gpu_pf_qgemm` arguments).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QgemmCall {
    pub n: usize,
    pub k: usize,
    pub layout: Layout,
    pub tiles_va: u64,
    pub ntiles: usize,
    /// 0 = direct rows (output row r reads activation row r); else u32 `[rows + bm]` source rows.
    pub rowidx_va: u64,
    pub xq_va: u64,
    pub xs_va: u64,
    pub y_va: u64,
}

/// omega's prefill kernels on the shared session. Every call is synchronous (it returns after
/// the chip finished, with the chip nanoseconds). Real: [`OmegaPfDevice`].
pub trait PrefillDevice {
    type Buf: PfBuffer;
    /// GPU-cached, device-private buffer.
    fn alloc_dev(&mut self, bytes: usize) -> Result<Self::Buf, String>;
    /// GPU-uncached buffer the host reads and writes.
    fn alloc_host(&mut self, bytes: usize) -> Result<Self::Buf, String>;
    /// Rows per GEMM tile (the GEMM reads activation rows up to `row0 + bm - 1` of every tile).
    fn qgemm_bm(&self) -> usize;
    /// Activation quantiser over `rows` rows of `k`.
    fn quant(&mut self, k: usize, input: QuantIn, rows: usize, x_va: u64, q_va: u64, s_va: u64) -> Result<u64, String>;
    /// One grouped GEMM launch.
    fn qgemm(&mut self, call: &QgemmCall) -> Result<u64, String>;
    /// Device-to-device copy by address (16-byte multiples). Default: refused.
    fn copy_va(&mut self, _dst_va: u64, _src_va: u64, _bytes: usize) -> Result<(), String> {
        Err("pf copy: not offered by this device".into())
    }
    /// Rows per CTA of the device router-logits launch (W-L1G-D), or `None` when this device has
    /// no router-logits / combine kernels. The router reads and writes `ceil(rows / t) * t` rows.
    fn router_rows(&self) -> Option<usize> {
        None
    }
    /// Router logits for `rows` rows of `k` f32 at `x_va` against the packed weights at `wq_va`
    /// ([`pack_router`]) into f32 rows of `e` at `logits_va`; bit-identical to [`router_logits_ref`].
    fn router_logits(&mut self, _k: usize, _e: usize, _rows: usize, _x_va: u64, _wq_va: u64, _logits_va: u64) -> Result<u64, String> {
        Err("pf router logits: not offered by this device".into())
    }
    /// MoE weighted combine ([`moe_combine_ref`]): `out[r] = sum_slot wt[r][slot] * down[inv[r][slot]]`.
    #[allow(clippy::too_many_arguments)]
    fn moe_combine(&mut self, _h: usize, _top_k: usize, _rows: usize, _down_va: u64, _inv_va: u64, _wt_va: u64, _out_va: u64) -> Result<u64, String> {
        Err("pf moe combine: not offered by this device".into())
    }
}

/// The attention call boundary: `omega_gpu_prefill_attn` exactly. One call does layer `layer`
/// for the `m` tokens at positions `pos0 .. pos0 + m - 1` (positions `0 .. pos0 - 1` already in
/// the KV cache): `qkv` f32 `[m][5120]` raw fused projection rows (q 32 x 128 | k 4 x 128 |
/// v 4 x 128), appends their K/V, writes `out` f32 `[m][4096]` (token-major, head h at h * 128)
/// multiplied by `out_scale`. Returns chip nanoseconds.
pub trait PrefillAttention {
    #[allow(clippy::too_many_arguments)]
    fn attend(&mut self, layer: u32, pos0: u32, m: u32, qkv: &[f32], eps: f32, out_scale: f32, out: &mut [f32]) -> Result<u64, String>;
    /// Whether [`Self::attend_dev`] is available.
    fn has_dev(&self) -> bool {
        false
    }
    /// The device-pointer variant (`omega_gpu_prefill_attn_dev`): `qkv_va` f32 `[m][5120]` and
    /// `out_va` f32 `[m][4096]` rows in session memory (no host copies). `None`: not offered.
    #[allow(clippy::too_many_arguments)]
    fn attend_dev(&mut self, layer: u32, pos0: u32, m: u32, qkv_va: u64, eps: f32, out_scale: f32, out_va: u64) -> Option<Result<u64, String>> {
        let _ = (layer, pos0, m, qkv_va, eps, out_scale, out_va);
        None
    }
}

/// Query heads x head dim of the attention kernel's fixed shape (Qwen3-30B-A3B).
pub const ATTN_OUT_WORDS: usize = 32 * 128;
/// Fused q|k|v row of the attention kernel's fixed shape.
pub const ATTN_QKV_WORDS: usize = (32 + 2 * 4) * 128;

// ---------------------------------------------------------------------------
// FFI.
// ---------------------------------------------------------------------------

/// Mirror of `OmegaGpuMem` (omega_gpu_session.h): handle {index, generation}, kind, cpu, va, size.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct OmegaGpuMem {
    pub handle_index: u32,
    pub handle_generation: u32,
    pub kind: u32,
    pub cpu: *mut std::ffi::c_void,
    pub va: u64,
    pub size: u64,
}

impl Default for OmegaGpuMem {
    fn default() -> Self {
        Self { handle_index: 0, handle_generation: 0, kind: 0, cpu: std::ptr::null_mut(), va: 0, size: 0 }
    }
}

/// Mirror of `OmegaGpuPrefillAttnInfo` (omega_gpu_prefill_attn_api.h).
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
pub struct PrefillAttnInfo {
    pub layer: u32,
    pub pos0: u32,
    pub m: u32,
    pub chunks: u32,
    pub grid_x: u32,
    pub grid_y: u32,
    pub threads: u32,
    pub splits: u32,
    pub per_split: u32,
    pub elapsed_ns: u64,
    pub call_ns: u64,
    pub kernel_cache_hit: i32,
}

#[cfg(has_omega_pf)]
mod ffi {
    use super::OmegaGpuMem;
    use std::os::raw::{c_char, c_void};
    extern "C" {
        pub fn omega_gpu_pf_rc_name(rc: i32) -> *const c_char;
        pub fn omega_gpu_session_last_error() -> *const c_char;
        pub fn omega_gpu_pf_alloc_dev(bytes: usize, out: *mut OmegaGpuMem) -> i32;
        pub fn omega_gpu_pf_alloc_host(bytes: usize, out: *mut OmegaGpuMem) -> i32;
        pub fn omega_gpu_pf_free(m: *mut OmegaGpuMem);
        pub fn omega_gpu_pf_copy_va(dst_va: u64, src_va: u64, bytes: usize) -> i32;
        pub fn omega_gpu_pf_stats(chip_ns: *mut u64, launches: *mut u64);
        pub fn omega_gpu_pf_stats_reset();
        pub fn omega_gpu_pf_quant(k: u32, in_type: u32, rows: u32, x_va: u64, q_va: u64, s_va: u64, ns: *mut u64) -> i32;
        pub fn omega_gpu_pf_qgemm_bm() -> u32;
        pub fn omega_gpu_pf_qgemm_tiles_va(experts: u32, w_va: *const u64, s_va: *const u64, counts: *const u32, tab: *mut c_void, max: u32) -> u32;
        pub fn omega_gpu_pf_qgemm(
            n: u32,
            k: u32,
            layout: u32,
            tiles_va: u64,
            ntiles: u32,
            rowidx_va: u64,
            xq_va: u64,
            xs_va: u64,
            y_va: u64,
            ns: *mut u64,
        ) -> i32;
    }
}

/// Router logits and MoE combine kernels (W-L1G-D, omega_gpu_pfhost.h).
#[cfg(has_omega_pfhost)]
mod ffi_host {
    extern "C" {
        pub fn omega_gpu_pf_router_rows() -> u32;
        pub fn omega_gpu_pf_router_pack(e: u32, k: u32, w: *const f32, wq: *mut f32);
        pub fn omega_gpu_pf_router_logits(k: u32, e: u32, rows: u32, x_va: u64, wq_va: u64, logits_va: u64, ns: *mut u64) -> i32;
        pub fn omega_gpu_pf_moe_combine(h: u32, top_k: u32, rows: u32, down_va: u64, inv_va: u64, wt_va: u64, out_va: u64, ns: *mut u64) -> i32;
    }
}

/// True when omega's router-logits and combine kernels (omega_gpu_pfhost.h) are linked.
pub const fn has_pfhost() -> bool {
    cfg!(has_omega_pfhost)
}

/// f32 count of the packed router layout R4 plus its pad group: `(k / 4 + 1) * e * 4`.
pub fn router_packed_len(e: usize, k: usize) -> usize {
    (k / 4 + 1) * e * 4
}

/// Packs a router `w` (f32 `[e][k]`, row = expert) into the kernel's layout R4:
/// `wq[g][ex][q] = w[ex][4g + q]`, then one zero pad group (omega_gpu_pf_router_pack).
pub fn pack_router(e: usize, k: usize, w: &[f32]) -> Vec<f32> {
    assert!(k.is_multiple_of(4) && w.len() == e * k, "pack_router: w must be e x k with k a multiple of 4");
    let mut wq = vec![0.0f32; router_packed_len(e, k)];
    for g in 0..k / 4 {
        for ex in 0..e {
            wq[(g * e + ex) * 4..(g * e + ex) * 4 + 4].copy_from_slice(&w[ex * k + 4 * g..ex * k + 4 * g + 4]);
        }
    }
    wq
}

/// The C packer (omega_gpu_pf_router_pack), for checking [`pack_router`] against it.
pub fn pack_router_c(e: usize, k: usize, w: &[f32]) -> Result<Vec<f32>, OmegaGpuError> {
    if !k.is_multiple_of(4) || w.len() != e * k {
        return Err(bad("pack_router_c: w must be e x k with k a multiple of 4".into()));
    }
    #[cfg(has_omega_pfhost)]
    {
        let mut wq = vec![f32::NAN; router_packed_len(e, k)];
        // SAFETY: w has e * k floats, wq router_packed_len(e, k) (the C packer's size).
        unsafe { ffi_host::omega_gpu_pf_router_pack(e as u32, k as u32, w.as_ptr(), wq.as_mut_ptr()) };
        Ok(wq)
    }
    #[cfg(not(has_omega_pfhost))]
    {
        Err(OmegaGpuError::Unavailable)
    }
}

/// Host mirror of the router-logits kernel: `out[r][ex]` = the sequential f32 sum from -0.0 of
/// `w[ex][j] * x[r][j]` (Rust's `map(a * b).sum::<f32>()`), read from the packed layout.
pub fn router_logits_ref(k: usize, e: usize, rows: usize, x: &[f32], wq: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        for ex in 0..e {
            let mut acc = -0.0f32;
            for j in 0..k {
                acc += wq[((j / 4) * e + ex) * 4 + j % 4] * x[r * k + j];
            }
            out[r * e + ex] = acc;
        }
    }
}

/// Host mirror of the combine kernel: `out[r][c]` = the sum from +0.0, in slot order, of
/// `wt[r][slot] * down[inv[r][slot]][c]` (the driver's combine accumulator).
pub fn moe_combine_ref(h: usize, top_k: usize, rows: usize, down: &[f32], inv: &[u32], wt: &[f32], out: &mut [f32]) {
    for r in 0..rows {
        for c in 0..h {
            let mut acc = 0.0f32;
            for s in 0..top_k {
                acc += wt[r * top_k + s] * down[inv[r * top_k + s] as usize * h + c];
            }
            out[r * h + c] = acc;
        }
    }
}
#[cfg(has_omega_pattn)]
mod ffi_attn {
    use super::PrefillAttnInfo;
    use std::os::raw::{c_char, c_void};
    extern "C" {
        pub fn omega_gpu_prefill_attn(
            c: *mut c_void,
            layer: u32,
            pos0: u32,
            m: u32,
            qkv: *const f32,
            eps: f32,
            out_scale: f32,
            out: *mut f32,
            info: *mut PrefillAttnInfo,
        ) -> i32;
        pub fn omega_gpu_decode_attn_rc_name(rc: i32) -> *const c_char;
        pub fn omega_gpu_decode_attn_last_error() -> *const c_char;
    }
    #[cfg(has_omega_pattn_dev)]
    extern "C" {
        pub fn omega_gpu_prefill_attn_dev(
            c: *mut c_void,
            layer: u32,
            pos0: u32,
            m: u32,
            qkv_va: u64,
            eps: f32,
            out_scale: f32,
            out_va: u64,
            info: *mut PrefillAttnInfo,
        ) -> i32;
    }
}

fn bad(name: String) -> OmegaGpuError {
    OmegaGpuError::Rc { rc: -1, name }
}

#[cfg(any(has_omega_pf, has_omega_pattn))]
fn cstr(p: *const std::os::raw::c_char) -> Option<String> {
    // SAFETY: omega returns static NUL-terminated strings or NULL.
    (!p.is_null()).then(|| unsafe { std::ffi::CStr::from_ptr(p) }.to_string_lossy().into_owned())
}

#[cfg(has_omega_pf)]
fn rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: both functions take no pointers and return static strings or NULL.
    let name = cstr(unsafe { ffi::omega_gpu_pf_rc_name(rc) }).unwrap_or_else(|| "?".into());
    let name = match cstr(unsafe { ffi::omega_gpu_session_last_error() }) {
        Some(e) => format!("{name}: {e}"),
        None => name,
    };
    OmegaGpuError::Rc { rc, name }
}

#[cfg(has_omega_pattn)]
fn attn_rc_err(rc: i32) -> OmegaGpuError {
    // SAFETY: as rc_err.
    let name = cstr(unsafe { ffi_attn::omega_gpu_decode_attn_rc_name(rc) }).unwrap_or_else(|| "?".into());
    let name = match cstr(unsafe { ffi_attn::omega_gpu_decode_attn_last_error() }) {
        Some(e) => format!("{name}: {e}"),
        None => name,
    };
    OmegaGpuError::Rc { rc, name }
}

/// An owned pf buffer (freed on drop). Host buffers expose their CPU view.
#[derive(Debug)]
pub struct PfMem {
    mem: OmegaGpuMem,
    bytes: usize,
    host: bool,
}

// SAFETY: an owning handle to session memory; moved between threads, never shared unlocked.
unsafe impl Send for PfMem {}

impl PfMem {
    fn alloc(bytes: usize, host: bool) -> Result<Self, OmegaGpuError> {
        if bytes == 0 {
            return Err(bad("pf alloc of 0 bytes".into()));
        }
        #[cfg(has_omega_pf)]
        {
            let mut mem = OmegaGpuMem::default();
            // SAFETY: `mem` is a valid out pointer for the call.
            let rc = unsafe {
                if host {
                    ffi::omega_gpu_pf_alloc_host(bytes, &mut mem)
                } else {
                    ffi::omega_gpu_pf_alloc_dev(bytes, &mut mem)
                }
            };
            if rc != 0 {
                return Err(rc_err(rc));
            }
            if mem.va == 0 || (mem.size as usize) < bytes || (host && mem.cpu.is_null()) {
                // SAFETY: the buffer came from the alloc above.
                unsafe { ffi::omega_gpu_pf_free(&mut mem) };
                return Err(bad(format!("pf alloc of {bytes} bytes returned no usable view")));
            }
            Ok(Self { mem, bytes, host })
        }
        #[cfg(not(has_omega_pf))]
        {
            let _ = host;
            Err(OmegaGpuError::Unavailable)
        }
    }
    /// GPU-cached device-private buffer (`omega_gpu_pf_alloc_dev`).
    pub fn alloc_dev(bytes: usize) -> Result<Self, OmegaGpuError> {
        Self::alloc(bytes, false)
    }
    /// GPU-uncached host-visible buffer (`omega_gpu_pf_alloc_host`).
    pub fn alloc_host(bytes: usize) -> Result<Self, OmegaGpuError> {
        Self::alloc(bytes, true)
    }
}

impl PfBuffer for PfMem {
    fn va(&self) -> u64 {
        self.mem.va
    }
    fn bytes(&self) -> usize {
        self.bytes
    }
    fn host(&self) -> Option<&[u8]> {
        // SAFETY: a host buffer's CPU view covers `bytes` and lives until drop; launches are
        // synchronous, so the GPU does not write it while this borrow is in use.
        self.host.then(|| unsafe { std::slice::from_raw_parts(self.mem.cpu as *const u8, self.bytes) })
    }
    fn host_mut(&mut self) -> Option<&mut [u8]> {
        // SAFETY: as host(); &mut self makes the view exclusive.
        self.host.then(|| unsafe { std::slice::from_raw_parts_mut(self.mem.cpu as *mut u8, self.bytes) })
    }
}

impl Drop for PfMem {
    fn drop(&mut self) {
        #[cfg(has_omega_pf)]
        // SAFETY: the buffer came from alloc and is freed once.
        unsafe {
            ffi::omega_gpu_pf_free(&mut self.mem)
        }
    }
}

/// omega's prefill kernels (the real [`PrefillDevice`]).
#[derive(Debug, Default)]
pub struct OmegaPfDevice;

impl OmegaPfDevice {
    /// Accumulated chip time and launches of the pf module since the last reset.
    pub fn stats(&self) -> (u64, u64) {
        #[cfg(has_omega_pf)]
        {
            let (mut ns, mut n) = (0u64, 0u64);
            // SAFETY: two valid out pointers.
            unsafe { ffi::omega_gpu_pf_stats(&mut ns, &mut n) };
            (ns, n)
        }
        #[cfg(not(has_omega_pf))]
        {
            (0, 0)
        }
    }
    pub fn stats_reset(&self) {
        #[cfg(has_omega_pf)]
        // SAFETY: no arguments.
        unsafe {
            ffi::omega_gpu_pf_stats_reset()
        }
    }
    /// The C tile builder (`omega_gpu_pf_qgemm_tiles_va`), for checking [`build_tiles`] against it.
    pub fn tiles_va_c(&self, w_va: &[u64], s_va: &[u64], counts: &[u32]) -> Result<Vec<Tile>, OmegaGpuError> {
        if w_va.len() != counts.len() || s_va.len() != counts.len() || counts.is_empty() {
            return Err(bad("pf tiles_va: one weight and scale address per expert".into()));
        }
        #[cfg(has_omega_pf)]
        {
            let mut tab = vec![Tile::default(); MAX_TILES];
            // SAFETY: arrays have `counts.len()` entries; tab holds MAX_TILES entries of 32 bytes.
            let n = unsafe {
                ffi::omega_gpu_pf_qgemm_tiles_va(
                    counts.len() as u32,
                    w_va.as_ptr(),
                    s_va.as_ptr(),
                    counts.as_ptr(),
                    tab.as_mut_ptr().cast(),
                    MAX_TILES as u32,
                )
            };
            if n == 0 && counts.iter().any(|&c| c > 0) {
                return Err(bad("pf tiles_va: table overflow or bad arguments".into()));
            }
            tab.truncate(n as usize);
            Ok(tab)
        }
        #[cfg(not(has_omega_pf))]
        {
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl PrefillDevice for OmegaPfDevice {
    type Buf = PfMem;
    fn router_rows(&self) -> Option<usize> {
        #[cfg(has_omega_pfhost)]
        {
            // SAFETY: no arguments.
            Some(unsafe { ffi_host::omega_gpu_pf_router_rows() } as usize)
        }
        #[cfg(not(has_omega_pfhost))]
        {
            None
        }
    }
    fn router_logits(&mut self, k: usize, e: usize, rows: usize, x_va: u64, wq_va: u64, logits_va: u64) -> Result<u64, String> {
        if x_va == 0 || wq_va == 0 || logits_va == 0 || rows > u32::MAX as usize || k > u32::MAX as usize || e > u32::MAX as usize {
            return Err(format!("pf router logits: null address or {rows} rows"));
        }
        #[cfg(has_omega_pfhost)]
        {
            let mut ns = 0u64;
            // SAFETY: plain values and one out pointer; the addresses are session buffers the caller owns.
            let rc = unsafe { ffi_host::omega_gpu_pf_router_logits(k as u32, e as u32, rows as u32, x_va, wq_va, logits_va, &mut ns) };
            if rc != 0 {
                return Err(rc_err(rc).to_string());
            }
            Ok(ns)
        }
        #[cfg(not(has_omega_pfhost))]
        {
            Err(OmegaGpuError::Unavailable.to_string())
        }
    }
    fn moe_combine(&mut self, h: usize, top_k: usize, rows: usize, down_va: u64, inv_va: u64, wt_va: u64, out_va: u64) -> Result<u64, String> {
        if down_va == 0 || inv_va == 0 || wt_va == 0 || out_va == 0 || rows > u32::MAX as usize || h > u32::MAX as usize || top_k > u32::MAX as usize {
            return Err(format!("pf moe combine: null address or {rows} rows"));
        }
        #[cfg(has_omega_pfhost)]
        {
            let mut ns = 0u64;
            // SAFETY: plain values and one out pointer; the addresses are session buffers the caller owns.
            let rc = unsafe { ffi_host::omega_gpu_pf_moe_combine(h as u32, top_k as u32, rows as u32, down_va, inv_va, wt_va, out_va, &mut ns) };
            if rc != 0 {
                return Err(rc_err(rc).to_string());
            }
            Ok(ns)
        }
        #[cfg(not(has_omega_pfhost))]
        {
            Err(OmegaGpuError::Unavailable.to_string())
        }
    }
    fn copy_va(&mut self, dst_va: u64, src_va: u64, bytes: usize) -> Result<(), String> {
        if dst_va == 0 || src_va == 0 || dst_va % 16 != 0 || src_va % 16 != 0 || bytes % 16 != 0 {
            return Err(format!("pf copy: addresses and {bytes} bytes must be non-zero 16-byte multiples"));
        }
        #[cfg(has_omega_pf)]
        {
            // SAFETY: plain values; both ranges are session buffers the caller owns.
            let rc = unsafe { ffi::omega_gpu_pf_copy_va(dst_va, src_va, bytes) };
            if rc != 0 {
                return Err(rc_err(rc).to_string());
            }
            Ok(())
        }
        #[cfg(not(has_omega_pf))]
        {
            Err(OmegaGpuError::Unavailable.to_string())
        }
    }
    fn alloc_dev(&mut self, bytes: usize) -> Result<PfMem, String> {
        PfMem::alloc_dev(bytes).map_err(|e| e.to_string())
    }
    fn alloc_host(&mut self, bytes: usize) -> Result<PfMem, String> {
        PfMem::alloc_host(bytes).map_err(|e| e.to_string())
    }
    fn qgemm_bm(&self) -> usize {
        #[cfg(has_omega_pf)]
        {
            // SAFETY: no arguments.
            unsafe { ffi::omega_gpu_pf_qgemm_bm() as usize }
        }
        #[cfg(not(has_omega_pf))]
        {
            DEFAULT_BM
        }
    }
    fn quant(&mut self, k: usize, input: QuantIn, rows: usize, x_va: u64, q_va: u64, s_va: u64) -> Result<u64, String> {
        check_quant(k).map_err(|e| e.to_string())?;
        if x_va == 0 || q_va == 0 || s_va == 0 || rows > u32::MAX as usize {
            return Err(format!("pf quant: null address or {rows} rows"));
        }
        #[cfg(has_omega_pf)]
        {
            let mut ns = 0u64;
            // SAFETY: plain values and one out pointer; the addresses are session buffers the caller owns.
            let rc = unsafe { ffi::omega_gpu_pf_quant(k as u32, input as u32, rows as u32, x_va, q_va, s_va, &mut ns) };
            if rc != 0 {
                return Err(rc_err(rc).to_string());
            }
            Ok(ns)
        }
        #[cfg(not(has_omega_pf))]
        {
            let _ = input;
            Err(OmegaGpuError::Unavailable.to_string())
        }
    }
    fn qgemm(&mut self, c: &QgemmCall) -> Result<u64, String> {
        check_qgemm(c.n, c.k, c.layout, c.ntiles).map_err(|e| e.to_string())?;
        if c.tiles_va == 0 || c.xq_va == 0 || c.xs_va == 0 || c.y_va == 0 {
            return Err("pf qgemm: null address".into());
        }
        #[cfg(has_omega_pf)]
        {
            let mut ns = 0u64;
            // SAFETY: plain values and one out pointer; the addresses are session buffers the caller owns.
            let rc = unsafe {
                ffi::omega_gpu_pf_qgemm(
                    c.n as u32,
                    c.k as u32,
                    c.layout as u32,
                    c.tiles_va,
                    c.ntiles as u32,
                    c.rowidx_va,
                    c.xq_va,
                    c.xs_va,
                    c.y_va,
                    &mut ns,
                )
            };
            if rc != 0 {
                return Err(rc_err(rc).to_string());
            }
            Ok(ns)
        }
        #[cfg(not(has_omega_pf))]
        {
            Err(OmegaGpuError::Unavailable.to_string())
        }
    }
}

/// omega's prefill attention on a decode attention cache (the real [`PrefillAttention`]).
/// The KV cache is shared with decode: after a call, decode continues at `pos0 + m`.
pub struct OmegaPrefillAttention<'a> {
    cache: &'a mut aien_omega_dattn::DecodeAttnCache,
}

impl std::fmt::Debug for OmegaPrefillAttention<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "OmegaPrefillAttention {{ layers: {}, max_pos: {} }}", self.cache.layers(), self.cache.max_pos())
    }
}

impl<'a> OmegaPrefillAttention<'a> {
    pub fn new(cache: &'a mut aien_omega_dattn::DecodeAttnCache) -> Self {
        Self { cache }
    }
    fn check(&self, layer: u32, pos0: u32, m: u32) -> Result<(), String> {
        if m == 0 || layer as usize >= self.cache.layers() || pos0 as usize + m as usize > self.cache.max_pos() {
            return Err(format!(
                "prefill attn layer {layer} pos0 {pos0} m {m} refused: m >= 1, layer < {}, pos0 + m <= {}",
                self.cache.layers(),
                self.cache.max_pos()
            ));
        }
        Ok(())
    }
}

impl PrefillAttention for OmegaPrefillAttention<'_> {
    fn attend(&mut self, layer: u32, pos0: u32, m: u32, qkv: &[f32], eps: f32, out_scale: f32, out: &mut [f32]) -> Result<u64, String> {
        self.check(layer, pos0, m)?;
        let mu = m as usize;
        if qkv.len() != mu * ATTN_QKV_WORDS || out.len() != mu * ATTN_OUT_WORDS {
            return Err(format!(
                "prefill attn: qkv {} / out {} words, want {} / {}",
                qkv.len(),
                out.len(),
                mu * ATTN_QKV_WORDS,
                mu * ATTN_OUT_WORDS
            ));
        }
        #[cfg(has_omega_pattn)]
        {
            let c = self.cache.raw_handle();
            if c.is_null() {
                return Err("prefill attn: the decode attention cache is a stub".into());
            }
            let mut info = PrefillAttnInfo::default();
            // SAFETY: lengths and ranges checked; the cache is exclusively borrowed for the call.
            let rc = unsafe { ffi_attn::omega_gpu_prefill_attn(c, layer, pos0, m, qkv.as_ptr(), eps, out_scale, out.as_mut_ptr(), &mut info) };
            if rc != 0 {
                return Err(attn_rc_err(rc).to_string());
            }
            Ok(info.elapsed_ns)
        }
        #[cfg(not(has_omega_pattn))]
        {
            let _ = (eps, out_scale);
            Err(OmegaGpuError::Unavailable.to_string())
        }
    }
    fn has_dev(&self) -> bool {
        has_attention_dev()
    }
    fn attend_dev(&mut self, layer: u32, pos0: u32, m: u32, qkv_va: u64, eps: f32, out_scale: f32, out_va: u64) -> Option<Result<u64, String>> {
        #[cfg(has_omega_pattn_dev)]
        {
            if let Err(e) = self.check(layer, pos0, m) {
                return Some(Err(e));
            }
            if qkv_va == 0 || out_va == 0 || qkv_va % 16 != 0 || out_va % 16 != 0 {
                return Some(Err("prefill attn dev: addresses must be non-zero and 16-byte aligned".into()));
            }
            let c = self.cache.raw_handle();
            if c.is_null() {
                return Some(Err("prefill attn: the decode attention cache is a stub".into()));
            }
            let mut info = PrefillAttnInfo::default();
            // SAFETY: ranges checked; the addresses are session buffers the caller owns; the cache is
            // exclusively borrowed for the call.
            let rc = unsafe { ffi_attn::omega_gpu_prefill_attn_dev(c, layer, pos0, m, qkv_va, eps, out_scale, out_va, &mut info) };
            Some(if rc != 0 { Err(attn_rc_err(rc).to_string()) } else { Ok(info.elapsed_ns) })
        }
        #[cfg(not(has_omega_pattn_dev))]
        {
            let _ = (layer, pos0, m, qkv_va, eps, out_scale, out_va);
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tile_bytes_match_the_c_layout() {
        let t = Tile { w_va: 0x1122_3344_5566_7788, s_va: 0x99aa_bbcc_ddee_ff00, row0: 7, cnt: 64, pad: 0 };
        let b = t.to_bytes();
        assert_eq!(&b[0..8], &0x1122_3344_5566_7788u64.to_le_bytes());
        assert_eq!(&b[16..20], &7u32.to_le_bytes());
        assert_eq!(&b[20..24], &64u32.to_le_bytes());
        assert_eq!(Tile::from_bytes(&b), t);
        assert_eq!(std::mem::offset_of!(Tile, row0), 16);
        assert_eq!(std::mem::offset_of!(Tile, cnt), 20);
    }

    #[test]
    fn tiles_follow_the_c_builder_rules() {
        // expert 0: 130 rows (3 tiles: 64, 64, 2), expert 1: none, expert 2: 64 rows (1 tile)
        let t = build_tiles(64, &[10, 20, 30], &[11, 21, 31], &[130, 0, 64]).unwrap();
        let rows: Vec<(u64, u32, u32)> = t.iter().map(|t| (t.w_va, t.row0, t.cnt)).collect();
        assert_eq!(rows, vec![(10, 0, 64), (10, 64, 64), (10, 128, 2), (30, 130, 64)]);
        assert!(t.iter().all(|t| t.s_va == t.w_va + 1 && t.pad == 0));
        assert!(build_tiles(64, &[1], &[1, 2], &[3]).is_err());
        assert!(build_tiles(0, &[1], &[1], &[3]).is_err());
    }

    #[test]
    fn host_checks_mirror_the_spec_rules() {
        assert!(check_quant(2048).is_ok() && check_quant(4096).is_ok() && check_quant(768).is_ok());
        assert!(check_quant(8192).is_err() && check_quant(100).is_err());
        assert!(check_qgemm(5120, 2048, Layout::Plain, 57).is_ok());
        assert!(check_qgemm(2048, 4096, Layout::Plain, 1).is_ok());
        assert!(check_qgemm(1536, 2048, Layout::SwigluNat, 1).is_ok());
        assert!(check_qgemm(2048, 768, Layout::Plain, 1).is_ok());
        assert!(check_qgemm(384, 2048, Layout::SwigluNat, 1).is_err(), "SwiGLU n must be a multiple of 256");
        assert!(check_qgemm(100, 2048, Layout::Plain, 1).is_err());
        assert!(check_qgemm(32768, 2048, Layout::Plain, 1).is_err(), "n * 256 >= 2^23");
        assert!(check_qgemm(128, 2048, Layout::Plain, MAX_TILES + 1).is_err());
    }

    #[test]
    fn ffi_struct_layouts() {
        assert_eq!(std::mem::size_of::<OmegaGpuMem>(), 40);
        assert_eq!(std::mem::offset_of!(OmegaGpuMem, cpu), 16);
        assert_eq!(std::mem::offset_of!(OmegaGpuMem, va), 24);
        assert_eq!(std::mem::offset_of!(PrefillAttnInfo, elapsed_ns), 40);
        assert_eq!(std::mem::size_of::<PrefillAttnInfo>(), 64);
    }

    /// Host only (no session, no chip): the Rust tile builder equals omega's C builder.
    #[cfg(has_omega_pf)]
    #[test]
    fn rust_tiles_equal_the_c_builder() {
        let d = OmegaPfDevice;
        let bm = d.qgemm_bm();
        assert_eq!(bm, DEFAULT_BM);
        let counts = [130u32, 0, 64, 1, 0, 200];
        let w: Vec<u64> = (0..6).map(|e| 0x1000 * (e + 1)).collect();
        let s: Vec<u64> = (0..6).map(|e| 0x10_0000 + e).collect();
        assert_eq!(build_tiles(bm, &w, &s, &counts).unwrap(), d.tiles_va_c(&w, &s, &counts).unwrap());
    }

    #[cfg(not(has_omega_pf))]
    #[test]
    fn stub_refuses() {
        let mut d = OmegaPfDevice;
        assert!(matches!(PfMem::alloc_dev(64), Err(OmegaGpuError::Unavailable)));
        assert!(d.quant(2048, QuantIn::F32, 1, 1, 1, 1).unwrap_err().contains("unavailable"));
        assert_eq!(d.qgemm_bm(), DEFAULT_BM);
    }

    // ---- W-L1G-D: router logits + combine host mirrors ----
    fn lcg_vec(seed: &mut u64, n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
            })
            .collect()
    }

    #[test]
    fn router_mirror_equals_the_rust_sum_bit_for_bit() {
        let (e, k, rows) = (16usize, 64usize, 5usize);
        let mut s = 7u64;
        let w = lcg_vec(&mut s, e * k, 0.1);
        let mut x = lcg_vec(&mut s, rows * k, 2.0);
        x[2 * k..3 * k].fill(-0.0); // signed-zero products: the sign of the sum depends on the -0.0 start
        let wq = pack_router(e, k, &w);
        assert_eq!(wq.len(), router_packed_len(e, k));
        assert!(wq[(k / 4) * e * 4..].iter().all(|&v| v.to_bits() == 0), "pad group is zero");
        let mut got = vec![0.0f32; rows * e];
        router_logits_ref(k, e, rows, &x, &wq, &mut got);
        for r in 0..rows {
            for ex in 0..e {
                let want: f32 = w[ex * k..(ex + 1) * k].iter().zip(&x[r * k..(r + 1) * k]).map(|(a, b)| a * b).sum();
                assert_eq!(got[r * e + ex].to_bits(), want.to_bits(), "row {r} expert {ex}");
            }
        }
    }

    #[test]
    fn combine_mirror_accumulates_from_zero_in_slot_order() {
        let (h, top_k, rows) = (8usize, 4usize, 3usize);
        let mut s = 11u64;
        let down = lcg_vec(&mut s, rows * top_k * h, 1.0);
        let wt = lcg_vec(&mut s, rows * top_k, 0.5);
        let inv: Vec<u32> = (0..(rows * top_k) as u32).rev().collect();
        let mut got = vec![0.0f32; rows * h];
        moe_combine_ref(h, top_k, rows, &down, &inv, &wt, &mut got);
        for r in 0..rows {
            for c in 0..h {
                let mut acc = 0.0f32;
                for sl in 0..top_k {
                    acc += wt[r * top_k + sl] * down[inv[r * top_k + sl] as usize * h + c];
                }
                assert_eq!(got[r * h + c].to_bits(), acc.to_bits());
            }
        }
    }

    #[cfg(has_omega_pfhost)]
    #[test]
    fn rust_router_pack_equals_the_c_packer() {
        let (e, k) = (128usize, 2048usize);
        let mut s = 3u64;
        let w = lcg_vec(&mut s, e * k, 0.1);
        let a = pack_router(e, k, &w);
        let b = pack_router_c(e, k, &w).unwrap();
        assert!(a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()) && a.len() == b.len());
        assert_eq!(OmegaPfDevice.router_rows(), Some(16));
    }

    #[cfg(not(has_omega_pfhost))]
    #[test]
    fn pfhost_stub_offers_no_router() {
        assert_eq!(OmegaPfDevice.router_rows(), None);
        assert!(pack_router_c(4, 4, &[0.0; 16]).is_err());
    }
}
