//! Safe Rust wrapper over omega's native GPU matmul (no CUDA).
//!
//! Built against the omega commit in `omega.lock`. With `cfg(has_omega_gpu)` the
//! real library is linked; otherwise (`AIEN_FORCE_CPU_STUB=1`, or no omega checkout
//! given) every chip call returns [`OmegaGpuError::Unavailable`], so CI passes
//! without the chip. Only the API on omega main at the pinned commit is wrapped.
//!
//! Conventions: row-major, `C[m x n] = A[m x k] * B[k x n]`, f32 output.
//! Inputs are bf16 (`matmul_bf16`) or f32 rounded to bf16 on the host (`matmul_f32`).
//! Limits (omega FB-1 cut 1b: M <= 8192, K <= 16384, N <= 262144) are enforced by omega
//! and surface as `Rc { rc: -2 (TOO_LARGE) }`.
pub mod ffi;

use std::fmt;

pub use ffi::OmegaGpuMatmulInfo;

/// Omega commit this crate was built against (from `omega.lock`).
pub const PINNED_OMEGA_SHA: &str = env!("AIEN_OMEGA_PINNED_SHA");

/// True when the real native library is linked (not the stub).
pub const fn is_native() -> bool {
    cfg!(has_omega_gpu)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OmegaGpuError {
    /// Stub build: no native library linked.
    Unavailable,
    /// A slice length does not match the stated shape (caught before the C call).
    ShapeMismatch {
        what: &'static str,
        expected: usize,
        got: usize,
    },
    /// Omega returned a non-zero code; `name` is `omega_gpu_matmul_rc_name(rc)`.
    Rc { rc: i32, name: String },
}

impl fmt::Display for OmegaGpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Unavailable => write!(f, "omega gpu unavailable (stub build)"),
            Self::ShapeMismatch {
                what,
                expected,
                got,
            } => write!(f, "{what}: expected {expected} elements, got {got}"),
            Self::Rc { rc, name } => write!(f, "omega_gpu rc={rc} ({name})"),
        }
    }
}

impl std::error::Error for OmegaGpuError {}

/// Name of a return code; the C function when native, the same table otherwise.
pub fn rc_name(rc: i32) -> String {
    #[cfg(has_omega_gpu)]
    {
        // SAFETY: returns a pointer to a static NUL-terminated string.
        let p = unsafe { ffi::omega_gpu_matmul_rc_name(rc) };
        if !p.is_null() {
            // SAFETY: non-null, static, NUL-terminated.
            return unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
        }
    }
    match rc {
        ffi::OMEGA_GPU_MATMUL_OK => "OK",
        ffi::OMEGA_GPU_MATMUL_BAD_ARGS => "BAD_ARGS",
        ffi::OMEGA_GPU_MATMUL_TOO_LARGE => "TOO_LARGE",
        ffi::OMEGA_GPU_MATMUL_CODEGEN_FAIL => "CODEGEN_FAIL",
        ffi::OMEGA_GPU_MATMUL_CHIP_FAIL => "CHIP_FAIL",
        ffi::OMEGA_GPU_MATMUL_PARITY_FAIL => "PARITY_FAIL",
        _ => "UNKNOWN",
    }
    .to_string()
}

/// Per-call receipt from omega (copy of `OmegaGpuMatmulInfo`).
#[derive(Debug, Clone)]
pub struct MatmulInfo {
    pub raw: OmegaGpuMatmulInfo,
}

impl MatmulInfo {
    pub fn target_chip(&self) -> String {
        let bytes: Vec<u8> = self
            .raw
            .target_chip
            .iter()
            .take_while(|&&c| c != 0)
            .map(|c| c.to_ne_bytes()[0])
            .collect();
        String::from_utf8_lossy(&bytes).into_owned()
    }
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

fn dims(m: usize, k: usize, n: usize) -> Result<(u32, u32, u32), OmegaGpuError> {
    let conv = |v: usize| {
        u32::try_from(v).map_err(|_| OmegaGpuError::Rc {
            rc: ffi::OMEGA_GPU_MATMUL_TOO_LARGE,
            name: rc_name(ffi::OMEGA_GPU_MATMUL_TOO_LARGE),
        })
    };
    Ok((conv(m)?, conv(k)?, conv(n)?))
}

#[cfg(has_omega_gpu)]
fn finish(rc: i32, info: OmegaGpuMatmulInfo) -> Result<MatmulInfo, OmegaGpuError> {
    if rc == ffi::OMEGA_GPU_MATMUL_OK {
        Ok(MatmulInfo { raw: info })
    } else {
        Err(OmegaGpuError::Rc {
            rc,
            name: rc_name(rc),
        })
    }
}

/// `c = a * b`, bf16 inputs (raw bits), f32 output.
pub fn matmul_bf16(
    m: usize,
    k: usize,
    n: usize,
    a: &[u16],
    b: &[u16],
    c: &mut [f32],
) -> Result<MatmulInfo, OmegaGpuError> {
    check_len("a", m * k, a.len())?;
    check_len("b", k * n, b.len())?;
    check_len("c", m * n, c.len())?;
    let (mu, ku, nu) = dims(m, k, n)?;
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuMatmulInfo::zeroed();
        // SAFETY: lengths checked above match m*k, k*n, m*n; pointers are valid
        // for the call; info is a valid out-pointer.
        let rc = unsafe {
            ffi::omega_gpu_matmul_bf16(
                mu,
                ku,
                nu,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                &mut info,
            )
        };
        finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (mu, ku, nu);
        Err(OmegaGpuError::Unavailable)
    }
}

/// `c = a * b`, f32 inputs rounded to bf16 on the host, f32 output.
pub fn matmul_f32(
    m: usize,
    k: usize,
    n: usize,
    a: &[f32],
    b: &[f32],
    c: &mut [f32],
) -> Result<MatmulInfo, OmegaGpuError> {
    check_len("a", m * k, a.len())?;
    check_len("b", k * n, b.len())?;
    check_len("c", m * n, c.len())?;
    let (mu, ku, nu) = dims(m, k, n)?;
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuMatmulInfo::zeroed();
        // SAFETY: as in matmul_bf16.
        let rc = unsafe {
            ffi::omega_gpu_matmul_f32(
                mu,
                ku,
                nu,
                a.as_ptr(),
                b.as_ptr(),
                c.as_mut_ptr(),
                &mut info,
            )
        };
        finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (mu, ku, nu);
        Err(OmegaGpuError::Unavailable)
    }
}

/// Drop every cached kernel (tests; otherwise never needed). No-op in the stub.
pub fn cache_clear() {
    #[cfg(has_omega_gpu)]
    // SAFETY: takes no arguments; omega guards its cache with a mutex.
    unsafe {
        ffi::omega_gpu_matmul_cache_clear()
    }
}

/// A weight matrix (k x n, bf16, zero-padded by omega) resident on the device
/// until dropped. Not `Sync`: omega's resident-call thread safety is not
/// documented, so callers must serialize use (the backend holds it in a Mutex).
pub struct ResidentTensor {
    #[cfg(has_omega_gpu)]
    ptr: *mut ffi::OmegaGpuTensor,
    k: usize,
    n: usize,
}

// SAFETY: the handle is an owning pointer to a device buffer with no thread
// affinity that we know of (UNVERIFIED in omega docs); we only ever move it
// between threads and never share it without external locking.
unsafe impl Send for ResidentTensor {}

impl fmt::Debug for ResidentTensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ResidentTensor {{ k: {}, n: {} }}", self.k, self.n)
    }
}

impl ResidentTensor {
    /// Upload a row-major k x n f32 matrix (rounded to bf16 on the host by omega).
    pub fn upload_f32(k: usize, n: usize, b: &[f32]) -> Result<Self, OmegaGpuError> {
        check_len("b", k * n, b.len())?;
        let (_, ku, nu) = dims(1, k, n)?;
        #[cfg(has_omega_gpu)]
        {
            let mut ptr: *mut ffi::OmegaGpuTensor = std::ptr::null_mut();
            // SAFETY: b has k*n elements (checked); ptr is a valid out-pointer.
            let rc = unsafe { ffi::omega_gpu_tensor_upload_f32(ku, nu, b.as_ptr(), &mut ptr) };
            if rc != ffi::OMEGA_GPU_MATMUL_OK || ptr.is_null() {
                return Err(rc_error(if rc == 0 {
                    ffi::OMEGA_GPU_MATMUL_BAD_ARGS
                } else {
                    rc
                }));
            }
            Ok(Self { ptr, k, n })
        }
        #[cfg(not(has_omega_gpu))]
        {
            let _ = (ku, nu);
            Err(OmegaGpuError::Unavailable)
        }
    }

    pub fn k(&self) -> usize {
        self.k
    }

    pub fn n(&self) -> usize {
        self.n
    }

    /// `c[m x n] = a[m x k] * self`, f32 `a` rounded to bf16 on the host, no host oracle.
    pub fn matmul_f32(
        &self,
        m: usize,
        a: &[f32],
        c: &mut [f32],
    ) -> Result<MatmulInfo, OmegaGpuError> {
        check_len("a", m * self.k, a.len())?;
        check_len("c", m * self.n, c.len())?;
        let (mu, _, _) = dims(m, self.k, self.n)?;
        #[cfg(has_omega_gpu)]
        {
            let mut info = OmegaGpuMatmulInfo::zeroed();
            // SAFETY: lengths checked; self.ptr is a live handle owned by self.
            let rc = unsafe {
                ffi::omega_gpu_matmul_resident_f32(
                    mu,
                    a.as_ptr(),
                    self.ptr,
                    c.as_mut_ptr(),
                    &mut info,
                )
            };
            finish(rc, info)
        }
        #[cfg(not(has_omega_gpu))]
        {
            let _ = mu;
            Err(OmegaGpuError::Unavailable)
        }
    }
}

impl Drop for ResidentTensor {
    fn drop(&mut self) {
        #[cfg(has_omega_gpu)]
        // SAFETY: ptr came from a successful upload and is freed exactly once.
        unsafe {
            ffi::omega_gpu_tensor_free(self.ptr)
        }
    }
}

#[cfg(has_omega_gpu)]
fn rc_error(rc: i32) -> OmegaGpuError {
    OmegaGpuError::Rc {
        rc,
        name: rc_name(rc),
    }
}

/// Stage name of the most recent chip failure ("" if none; always "" in the stub).
pub fn last_error() -> String {
    #[cfg(has_omega_gpu)]
    {
        // SAFETY: returns a pointer to a static or internal NUL-terminated string.
        let p = unsafe { ffi::omega_gpu_matmul_last_error() };
        if !p.is_null() {
            // SAFETY: non-null and NUL-terminated.
            return unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
        }
    }
    String::new()
}

/// True once an uncertain completion has latched the process (every later chip call fails).
pub fn is_blocked() -> bool {
    #[cfg(has_omega_gpu)]
    // SAFETY: no arguments, returns an int.
    return unsafe { ffi::omega_gpu_matmul_is_blocked() } != 0;
    #[cfg(not(has_omega_gpu))]
    false
}

// ---- matmul launch budget (omega c0369e6 `omega_gpu_matmul_api.h:130-133`) ----

/// Daemon setting that picks the matmul CTA budget per launch. Unset or empty
/// means the AIEN daemon default ([`DAEMON_DEFAULT_CTA_BUDGET`], 256); omega's
/// own library default ([`ffi::OMEGA_GPU_MATMUL_MAX_CTAS`], 64) is not changed.
pub const CTA_BUDGET_ENV: &str = "AIEN_OMEGA_CTA_BUDGET";

/// Largest budget [`parse_cta_budget`] accepts. 64 (omega's default) and 256 are
/// the only budgets run on the GB10 (T4 measurement 2026-10-06 for the decode
/// shapes, t4fix verification for the prefill shapes; parity held at both);
/// 512 and 1024 were never chip-tested, so a larger value is refused.
pub const CTA_BUDGET_MAX_MEASURED: u32 = 256;

/// A validated CTA budget (1..=[`CTA_BUDGET_MAX_MEASURED`]). Only
/// [`parse_cta_budget`] and [`DAEMON_DEFAULT_CTA_BUDGET`] build one, so
/// [`set_cta_budget`] never sees 0 (which omega reads as "restore the
/// default") or an unmeasured size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CtaBudget(u32);

impl CtaBudget {
    pub const fn get(self) -> u32 {
        self.0
    }
}

/// The AIEN daemon's budget when [`CTA_BUDGET_ENV`] is unset: 256. Sealed GB10
/// evidence in [`DAEMON_DEFAULT_CTA_BUDGET_EVIDENCE`]: decode 89.7 ms/token at
/// 256 vs 184.7 at 64, T4 S3 23 678 ms (< B = 29 000 ms), prefill shapes at 256
/// parity ok (repeat_mismatch 0).
pub const DAEMON_DEFAULT_CTA_BUDGET: CtaBudget = CtaBudget(256);

/// Repository path of the evidence behind [`DAEMON_DEFAULT_CTA_BUDGET`].
pub const DAEMON_DEFAULT_CTA_BUDGET_EVIDENCE: &str = "docs/inference/evidence/t4fix-20261006T2254Z";

/// Parse the [`CTA_BUDGET_ENV`] setting. `None` or blank: `Ok(None)` (the caller
/// picks its default). Otherwise a plain decimal integer in
/// 1..=[`CTA_BUDGET_MAX_MEASURED`]; anything else is an error naming the value.
pub fn parse_cta_budget(raw: Option<&str>) -> Result<Option<CtaBudget>, String> {
    let Some(text) = raw.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let refuse = |why: &str| {
        Err(format!(
            "{CTA_BUDGET_ENV}={text:?} refused: {why} (allowed 1..={CTA_BUDGET_MAX_MEASURED}; unset uses the AIEN daemon default {})",
            DAEMON_DEFAULT_CTA_BUDGET.get()
        ))
    };
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return refuse("not a plain decimal integer");
    }
    match text.parse::<u32>() {
        Ok(0) => refuse("0 would silently restore omega's default"),
        Ok(v) if v <= CTA_BUDGET_MAX_MEASURED => Ok(Some(CtaBudget(v))),
        Ok(_) | Err(_) => refuse("larger than any budget measured on the GB10"),
    }
}

/// Set omega's per-launch CTA budget. Omega clears its kernel cache (kernels
/// bake in the grid), so call it before the first matmul. Stub: `Unavailable`.
pub fn set_cta_budget(budget: CtaBudget) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_gpu)]
    {
        // SAFETY: plain u32 argument; omega takes its session lock inside.
        unsafe { ffi::omega_gpu_matmul_set_cta_budget(budget.0) };
        Ok(())
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = budget;
        Err(OmegaGpuError::Unavailable)
    }
}

/// The CTA budget omega will use for the next launch; `None` in the stub.
pub fn cta_budget() -> Option<u32> {
    #[cfg(has_omega_gpu)]
    // SAFETY: no arguments; reads one u32 field, does not open the device.
    return Some(unsafe { ffi::omega_gpu_matmul_cta_budget() });
    #[cfg(not(has_omega_gpu))]
    None
}

/// Environment setting for omega's marker-wait spin window, in microseconds
/// (omega#328). Unset or blank means the AIEN daemon default
/// ([`DAEMON_DEFAULT_SPIN_US`]); 0 restores omega's own default, the 50 us
/// sleep-poll that timer slack rounds to about 102 us per launch.
pub const SPIN_US_ENV: &str = "AIEN_OMEGA_SPIN_US";

/// The AIEN daemon's window when [`SPIN_US_ENV`] is unset: 2000 us. Sealed GB10
/// evidence in [`DAEMON_DEFAULT_SPIN_US_EVIDENCE`]: Qwen3-4B decode 310 -> 268 ms/token
/// with identical replies (two alternating pairs).
pub const DAEMON_DEFAULT_SPIN_US: u32 = 2000;

/// Repository path of the evidence behind [`DAEMON_DEFAULT_SPIN_US`].
pub const DAEMON_DEFAULT_SPIN_US_EVIDENCE: &str = "docs/inference/evidence/spin-20261007T0726Z";

/// Largest window omega accepts (`OMEGA_GPU_SESSION_MAX_SPIN_US`).
pub const SPIN_US_MAX: u32 = 1_000_000;

// The daemon default must be a window omega accepts (checked at build time).
const _: () = assert!(DAEMON_DEFAULT_SPIN_US <= SPIN_US_MAX);

/// Parse the [`SPIN_US_ENV`] setting. `None` or blank: `Ok(None)` (the caller
/// picks its default). Otherwise a plain decimal integer in 0..=[`SPIN_US_MAX`];
/// anything else is an error naming the value.
pub fn parse_spin_us(raw: Option<&str>) -> Result<Option<u32>, String> {
    let Some(text) = raw.map(str::trim).filter(|t| !t.is_empty()) else {
        return Ok(None);
    };
    let refuse = |why: &str| {
        Err(format!(
            "{SPIN_US_ENV}={text:?} refused: {why} (allowed 0..={SPIN_US_MAX}; unset uses the AIEN daemon default {DAEMON_DEFAULT_SPIN_US})"
        ))
    };
    if !text.bytes().all(|b| b.is_ascii_digit()) {
        return refuse("not a plain decimal integer");
    }
    match text.parse::<u32>() {
        Ok(v) if v <= SPIN_US_MAX => Ok(Some(v)),
        Ok(_) | Err(_) => refuse("larger than omega's maximum"),
    }
}

/// Set omega's marker-wait spin window. Stub: `Unavailable`.
pub fn set_spin_us(us: u32) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_gpu)]
    {
        // SAFETY: plain u32 argument; omega stores it atomically and does not open the device.
        let rc = unsafe { ffi::omega_gpu_session_set_spin_us(us) };
        if rc == 0 {
            Ok(())
        } else {
            Err(OmegaGpuError::Rc {
                rc,
                name: format!("spin window {us} us refused (maximum {SPIN_US_MAX})"),
            })
        }
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = us;
        Err(OmegaGpuError::Unavailable)
    }
}

/// The spin window omega will use for the next launch; `None` in the stub.
pub fn spin_us() -> Option<u32> {
    #[cfg(has_omega_gpu)]
    // SAFETY: no arguments; reads one u32, does not open the device.
    return Some(unsafe { ffi::omega_gpu_session_spin_us() });
    #[cfg(not(has_omega_gpu))]
    None
}

// ---- native elementwise ops (omega FB-1 cut 4) ----

pub use ffi::OmegaGpuEwInfo;

/// Name of an elementwise return code (omega's table when native).
pub fn ew_rc_name(rc: i32) -> String {
    #[cfg(has_omega_gpu)]
    {
        // SAFETY: returns a pointer to a static NUL-terminated string.
        let p = unsafe { ffi::omega_gpu_elementwise_rc_name(rc) };
        if !p.is_null() {
            // SAFETY: non-null, static, NUL-terminated.
            return unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
        }
    }
    match rc {
        ffi::OMEGA_GPU_EW_OK => "OK",
        ffi::OMEGA_GPU_EW_BAD_ARGS => "BAD_ARGS",
        ffi::OMEGA_GPU_EW_TOO_LARGE => "TOO_LARGE",
        ffi::OMEGA_GPU_EW_CODEGEN_FAIL => "CODEGEN_FAIL",
        ffi::OMEGA_GPU_EW_CHIP_FAIL => "CHIP_FAIL",
        ffi::OMEGA_GPU_EW_UNWRITTEN => "UNWRITTEN",
        _ => "UNKNOWN",
    }
    .to_string()
}

#[cfg(has_omega_gpu)]
fn ew_finish(rc: i32, info: OmegaGpuEwInfo) -> Result<OmegaGpuEwInfo, OmegaGpuError> {
    if rc == ffi::OMEGA_GPU_EW_OK {
        Ok(info)
    } else {
        Err(OmegaGpuError::Rc {
            rc,
            name: ew_rc_name(rc),
        })
    }
}

fn ew_dim(v: usize) -> Result<u32, OmegaGpuError> {
    u32::try_from(v).map_err(|_| OmegaGpuError::Rc {
        rc: ffi::OMEGA_GPU_EW_TOO_LARGE,
        name: ew_rc_name(ffi::OMEGA_GPU_EW_TOO_LARGE),
    })
}

/// `out[r*dim+i] = x[r*dim+i] * scale_r * weight[i]`, `scale_r = 1/sqrt(mean(x^2)+eps)`.
/// omega requires `dim % 128 == 0` and `dim <= 16384`; otherwise `Rc { rc: -1 | -2 }`.
pub fn rmsnorm_f32(
    rows: usize,
    dim: usize,
    x: &[f32],
    weight: &[f32],
    eps: f32,
    out: &mut [f32],
) -> Result<OmegaGpuEwInfo, OmegaGpuError> {
    check_len("x", rows * dim, x.len())?;
    check_len("weight", dim, weight.len())?;
    check_len("out", rows * dim, out.len())?;
    let (ru, du) = (ew_dim(rows)?, ew_dim(dim)?);
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuEwInfo::zeroed();
        // SAFETY: lengths checked above; pointers valid for the call; info is an out-pointer.
        let rc = unsafe {
            ffi::omega_gpu_rmsnorm_f32(
                ru,
                du,
                x.as_ptr(),
                weight.as_ptr(),
                eps,
                out.as_mut_ptr(),
                &mut info,
            )
        };
        ew_finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (ru, du, eps);
        Err(OmegaGpuError::Unavailable)
    }
}

/// Rotate `heads` heads of `head_dim` values (rotate_half pairing `i, i+head_dim/2`) with a
/// host-provided table of `head_dim/2` cos and sin values (same for every head).
/// `out` may not alias `v` in safe Rust; omega itself copies through its own buffers.
pub fn rope_f32(
    heads: usize,
    head_dim: usize,
    v: &[f32],
    cos_half: &[f32],
    sin_half: &[f32],
    out: &mut [f32],
) -> Result<OmegaGpuEwInfo, OmegaGpuError> {
    check_len("v", heads * head_dim, v.len())?;
    check_len("cos_half", head_dim / 2, cos_half.len())?;
    check_len("sin_half", head_dim / 2, sin_half.len())?;
    check_len("out", heads * head_dim, out.len())?;
    let (hu, du) = (ew_dim(heads)?, ew_dim(head_dim)?);
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuEwInfo::zeroed();
        // SAFETY: lengths checked above; pointers valid for the call.
        let rc = unsafe {
            ffi::omega_gpu_rope_f32(
                hu,
                du,
                v.as_ptr(),
                cos_half.as_ptr(),
                sin_half.as_ptr(),
                out.as_mut_ptr(),
                &mut info,
            )
        };
        ew_finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (hu, du);
        Err(OmegaGpuError::Unavailable)
    }
}

/// `out[i] = silu(gate[i]) * up[i]` for `i < n` (`n = gate.len()`, all three equal length).
pub fn swiglu_f32(
    gate: &[f32],
    up: &[f32],
    out: &mut [f32],
) -> Result<OmegaGpuEwInfo, OmegaGpuError> {
    check_len("up", gate.len(), up.len())?;
    check_len("out", gate.len(), out.len())?;
    let nu = ew_dim(gate.len())?;
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuEwInfo::zeroed();
        // SAFETY: lengths checked equal; pointers valid for the call.
        let rc = unsafe {
            ffi::omega_gpu_swiglu_f32(nu, gate.as_ptr(), up.as_ptr(), out.as_mut_ptr(), &mut info)
        };
        ew_finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = nu;
        Err(OmegaGpuError::Unavailable)
    }
}

// ---- native decode attention (omega FB-1 cut 5, persistent session cut 4b) ----

pub use ffi::{OmegaGpuAttnInfo, OmegaGpuKvLayout};

/// Name of an attention return code (omega's table when native).
pub fn attn_rc_name(rc: i32) -> String {
    #[cfg(has_omega_gpu)]
    {
        // SAFETY: returns a pointer to a static NUL-terminated string.
        let p = unsafe { ffi::omega_gpu_attention_rc_name(rc) };
        if !p.is_null() {
            // SAFETY: non-null, static, NUL-terminated.
            return unsafe { std::ffi::CStr::from_ptr(p) }
                .to_string_lossy()
                .into_owned();
        }
    }
    match rc {
        ffi::OMEGA_GPU_ATTN_OK => "OK",
        ffi::OMEGA_GPU_ATTN_BAD_ARGS => "BAD_ARGS",
        ffi::OMEGA_GPU_ATTN_TOO_LARGE => "TOO_LARGE",
        ffi::OMEGA_GPU_ATTN_CODEGEN_FAIL => "CODEGEN_FAIL",
        ffi::OMEGA_GPU_ATTN_CHIP_FAIL => "CHIP_FAIL",
        ffi::OMEGA_GPU_ATTN_UNWRITTEN => "UNWRITTEN",
        _ => "UNKNOWN",
    }
    .to_string()
}

#[cfg(has_omega_gpu)]
fn attn_finish(rc: i32, info: OmegaGpuAttnInfo) -> Result<OmegaGpuAttnInfo, OmegaGpuError> {
    if rc == ffi::OMEGA_GPU_ATTN_OK {
        Ok(info)
    } else {
        Err(OmegaGpuError::Rc {
            rc,
            name: attn_rc_name(rc),
        })
    }
}

fn attn_dim(v: usize) -> Result<u32, OmegaGpuError> {
    u32::try_from(v).map_err(|_| OmegaGpuError::Rc {
        rc: ffi::OMEGA_GPU_ATTN_TOO_LARGE,
        name: attn_rc_name(ffi::OMEGA_GPU_ATTN_TOO_LARGE),
    })
}

/// Decode attention over contiguous f32 KV (`TensorBackend::gqa_attention`).
/// `q`, `out`: `[num_q_heads][head_dim]`; `k_cache`, `v_cache`: `[seq_len][num_kv_heads][head_dim]`.
/// omega requires `head_dim` 64 or 128 and `num_q_heads <= 64`; otherwise `Rc { rc: -1 | -2 }`.
#[allow(clippy::too_many_arguments)]
pub fn gqa_attention_f32(
    q: &[f32],
    k_cache: &[f32],
    v_cache: &[f32],
    seq_len: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<OmegaGpuAttnInfo, OmegaGpuError> {
    check_len("q", num_q_heads * head_dim, q.len())?;
    check_len("k_cache", seq_len * num_kv_heads * head_dim, k_cache.len())?;
    check_len("v_cache", seq_len * num_kv_heads * head_dim, v_cache.len())?;
    check_len("out", num_q_heads * head_dim, out.len())?;
    let (su, qh, kh, hd) = (
        attn_dim(seq_len)?,
        attn_dim(num_q_heads)?,
        attn_dim(num_kv_heads)?,
        attn_dim(head_dim)?,
    );
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuAttnInfo::zeroed();
        // SAFETY: lengths checked above; pointers valid for the call; info is an out-pointer.
        let rc = unsafe {
            ffi::omega_gpu_gqa_attention_f32(
                q.as_ptr(),
                k_cache.as_ptr(),
                v_cache.as_ptr(),
                su,
                qh,
                kh,
                hd,
                out.as_mut_ptr(),
                &mut info,
            )
        };
        attn_finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (su, qh, kh, hd);
        Err(OmegaGpuError::Unavailable)
    }
}

/// Decode attention over the bf16 paged KV pool (`TensorBackend::paged_attention`).
/// `pool` is the whole pool (at least `layout.pool_bytes` bytes); `block_ids` lists the
/// sequence's blocks in token order. `q`, `out`: `[num_q_heads][head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_bf16(
    q: &[f32],
    pool: &[u8],
    layout: &OmegaGpuKvLayout,
    block_ids: &[u32],
    context_len: usize,
    layer_idx: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<OmegaGpuAttnInfo, OmegaGpuError> {
    check_len("q", num_q_heads * head_dim, q.len())?;
    check_len("out", num_q_heads * head_dim, out.len())?;
    check_pool(pool, layout)?;
    let (nb, cl, li, qh, kh, hd) = (
        attn_dim(block_ids.len())?,
        attn_dim(context_len)?,
        attn_dim(layer_idx)?,
        attn_dim(num_q_heads)?,
        attn_dim(num_kv_heads)?,
        attn_dim(head_dim)?,
    );
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuAttnInfo::zeroed();
        // SAFETY: q/out lengths checked; pool covers layout.pool_bytes; block_ids has nb entries.
        let rc = unsafe {
            ffi::omega_gpu_paged_attention_bf16(
                q.as_ptr(),
                pool.as_ptr(),
                layout,
                block_ids.as_ptr(),
                nb,
                cl,
                li,
                qh,
                kh,
                hd,
                out.as_mut_ptr(),
                &mut info,
            )
        };
        attn_finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (nb, cl, li, qh, kh, hd);
        Err(OmegaGpuError::Unavailable)
    }
}

/// Batched decode attention (`TensorBackend::paged_attention_batch`).
/// `block_tables`: `[num_seqs][max_blocks_per_seq]` (negative = none), `context_lens`: `[num_seqs]`.
/// `q`, `out`: `[num_seqs][num_q_heads][head_dim]`.
#[allow(clippy::too_many_arguments)]
pub fn paged_attention_batch_bf16(
    q: &[f32],
    pool: &[u8],
    layout: &OmegaGpuKvLayout,
    block_tables: &[i32],
    context_lens: &[i32],
    max_blocks_per_seq: usize,
    num_seqs: usize,
    layer_idx: usize,
    num_q_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    out: &mut [f32],
) -> Result<OmegaGpuAttnInfo, OmegaGpuError> {
    check_len("q", num_seqs * num_q_heads * head_dim, q.len())?;
    check_len("out", num_seqs * num_q_heads * head_dim, out.len())?;
    check_len("context_lens", num_seqs, context_lens.len())?;
    check_len(
        "block_tables",
        num_seqs * max_blocks_per_seq,
        block_tables.len(),
    )?;
    check_pool(pool, layout)?;
    let (mb, ns, li, qh, kh, hd) = (
        attn_dim(max_blocks_per_seq)?,
        attn_dim(num_seqs)?,
        attn_dim(layer_idx)?,
        attn_dim(num_q_heads)?,
        attn_dim(num_kv_heads)?,
        attn_dim(head_dim)?,
    );
    #[cfg(has_omega_gpu)]
    {
        let mut info = OmegaGpuAttnInfo::zeroed();
        // SAFETY: every slice length checked against the stated shape; pool covers the layout.
        let rc = unsafe {
            ffi::omega_gpu_paged_attention_batch_bf16(
                q.as_ptr(),
                pool.as_ptr(),
                layout,
                block_tables.as_ptr(),
                context_lens.as_ptr(),
                mb,
                ns,
                li,
                qh,
                kh,
                hd,
                out.as_mut_ptr(),
                &mut info,
            )
        };
        attn_finish(rc, info)
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = (mb, ns, li, qh, kh, hd);
        Err(OmegaGpuError::Unavailable)
    }
}

fn check_pool(pool: &[u8], layout: &OmegaGpuKvLayout) -> Result<(), OmegaGpuError> {
    let need = usize::try_from(layout.pool_bytes).unwrap_or(usize::MAX);
    if pool.len() < need {
        return Err(OmegaGpuError::ShapeMismatch {
            what: "pool",
            expected: need,
            got: pool.len(),
        });
    }
    Ok(())
}

// ---- opt-in serving reservation (omega b980783, omega#333; sovereign-core#277) ----

/// Declared serving bounds, the Rust twin of omega's `OmegaGpuServingBounds`
/// (`src/omega_gpu_serving.h`). Every field is a bound the daemon declares at start-up;
/// omega reserves the driver memory for them once and refuses a call past them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingBounds {
    /// Longest sequence (tokens) one attention call may cover.
    pub max_context: u32,
    /// Sequences per attention call (the GB10 backend calls attention one sequence at a time: 1).
    pub max_seqs: u32,
    pub num_q_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    /// Paged KV block size in tokens (power of two).
    pub kv_block_size: u32,
    /// Most rows one matmul call may carry.
    pub max_rows: u32,
    /// Widest weight input dimension.
    pub max_k: u32,
    /// Widest weight output dimension for calls of up to `max_rows` rows.
    pub max_n: u32,
    /// Widest output dimension for calls of at most 16 rows (the logits projection).
    pub max_n_one_row: u32,
    /// Matmul kernel cache slots (omega allows at most 32).
    pub kernel_slots: u32,
}

/// Driver bytes a reservation asks for, computed with omega's own formulas
/// (`omega_gpu_attention_reserve`, attention_api.c:664-681; `omega_gpu_matmul_reserve`,
/// matmul_api.c:130-150, at b980783). Reported in the daemon log so a chip run can compare
/// it with the driver's allocation counters.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServingBytes {
    pub attention_pool: u64,
    pub attention_q_out: u64,
    pub attention_table: u64,
    pub matmul_activation: u64,
    pub matmul_result: u64,
}

impl ServingBytes {
    pub fn total(&self) -> u64 {
        self.attention_pool
            + 2 * self.attention_q_out
            + self.attention_table
            + self.matmul_activation
            + self.matmul_result
    }
}

fn round_up(v: u64, to: u64) -> u64 {
    v.div_ceil(to) * to
}

impl ServingBounds {
    /// Bytes omega will reserve for these bounds. Tiles: omega `OMEGA_GPU_MATMUL_TILE_M` 16,
    /// `_TILE_N` 8, `_TILE_K` 16 (omega_gpu_matmul_api.h:46-48).
    pub fn bytes(&self) -> ServingBytes {
        // Each buffer is rounded to a 4 KiB page by omega (checked against omega's fake driver
        // layer: 194170880 bytes for the Qwen3-4B daemon bounds, see the unit test).
        let (ctx, seqs) = (self.max_context as u64, self.max_seqs.max(1) as u64);
        let (kvh, hd) = (self.num_kv_heads as u64, self.head_dim as u64);
        let mut per_seq = 2 * ctx * kvh * hd * 4;
        if self.kv_block_size != 0 {
            let bs = self.kv_block_size as u64;
            per_seq = per_seq.max(round_up(ctx, bs) * kvh * hd * 4);
        }
        let qo = seqs * self.num_q_heads as u64 * hd * 4;
        let table = round_up(((seqs * ctx + seqs) * 4).max(8), 4096);
        let mp = round_up(self.max_rows as u64, 16);
        let kp = round_up(self.max_k as u64, 16);
        let mut c = mp * round_up(self.max_n as u64, 8) * 4;
        if self.max_n_one_row != 0 {
            c = c.max(16 * round_up(self.max_n_one_row as u64, 8) * 4);
        }
        ServingBytes {
            attention_pool: round_up(per_seq * seqs, 4096),
            attention_q_out: round_up(qo, 4096),
            attention_table: table,
            matmul_activation: round_up(mp * kp * 2, 4096),
            matmul_result: round_up(c, 4096),
        }
    }
}

/// Reserve the serving buffers once (opens the device if needed): omega's own all-or-nothing
/// `omega_gpu_reserve_serving` (`src/omega_gpu_serving.c`, in libomega_gpu.a since omega#338).
/// On refusal the error carries omega's return-code name; callers add `last_error()` for the
/// stage text (for example "serving reservation exceeded" or the driver's allocation failure).
/// Stub: `Unavailable`.
pub fn reserve_serving(b: &ServingBounds) -> Result<(), OmegaGpuError> {
    #[cfg(has_omega_gpu)]
    {
        let c = ffi::OmegaGpuServingBounds {
            max_context: b.max_context,
            max_seqs: b.max_seqs,
            num_q_heads: b.num_q_heads,
            num_kv_heads: b.num_kv_heads,
            head_dim: b.head_dim,
            kv_block_size: b.kv_block_size,
            max_rows: b.max_rows,
            max_k: b.max_k,
            max_n: b.max_n,
            max_n_one_row: b.max_n_one_row,
            kernel_slots: b.kernel_slots,
        };
        // SAFETY: `c` is a valid, fully initialised repr(C) struct that omega only reads.
        let rc = unsafe { ffi::omega_gpu_reserve_serving(&c) };
        if rc == 0 {
            Ok(())
        } else {
            // matmul and attention codes share the numbering 0/-1..-5; the matmul table names them.
            Err(rc_error(rc))
        }
    }
    #[cfg(not(has_omega_gpu))]
    {
        let _ = b;
        Err(OmegaGpuError::Unavailable)
    }
}

/// Lift the reservation (buffers stay, growth is allowed again). No-op in the stub.
pub fn release_serving() {
    #[cfg(has_omega_gpu)]
    // SAFETY: no arguments; omega takes its own lock.
    unsafe {
        ffi::omega_gpu_serving_release();
    }
}

#[cfg(test)]
mod cta_budget_tests {
    use super::*;

    #[test]
    fn unset_or_blank_parses_to_none_caller_default() {
        assert_eq!(parse_cta_budget(None), Ok(None));
        assert_eq!(parse_cta_budget(Some("")), Ok(None));
        assert_eq!(parse_cta_budget(Some("  ")), Ok(None));
    }

    #[test]
    fn measured_budgets_parse() {
        for (raw, v) in [("64", 64), ("256", 256), (" 256\n", 256), ("1", 1)] {
            assert_eq!(
                parse_cta_budget(Some(raw)).unwrap().map(CtaBudget::get),
                Some(v)
            );
        }
    }

    #[test]
    fn invalid_values_are_refused() {
        for raw in [
            "0",
            "-1",
            "+64",
            "abc",
            "64.0",
            "0x40",
            "257",
            "512",
            "1024",
            "99999999999",
        ] {
            let err = parse_cta_budget(Some(raw)).expect_err(raw);
            assert!(err.contains(CTA_BUDGET_ENV), "{raw}: {err}");
        }
    }

    #[test]
    fn default_budget_is_omega_max_ctas() {
        assert_eq!(ffi::OMEGA_GPU_MATMUL_MAX_CTAS, 64);
        // Reading the budget never opens the device (omega_gpu_matmul_api.c:121).
        if is_native() {
            assert_eq!(cta_budget(), Some(ffi::OMEGA_GPU_MATMUL_MAX_CTAS));
        } else {
            assert_eq!(cta_budget(), None);
            let b = parse_cta_budget(Some("256")).unwrap().unwrap();
            assert_eq!(set_cta_budget(b), Err(OmegaGpuError::Unavailable));
        }
    }

    #[test]
    fn daemon_default_is_256_and_inside_the_measured_range() {
        assert_eq!(DAEMON_DEFAULT_CTA_BUDGET.get(), 256);
        assert!(DAEMON_DEFAULT_CTA_BUDGET.get() <= CTA_BUDGET_MAX_MEASURED);
        // The same value as an explicit setting parses to the same budget.
        assert_eq!(
            parse_cta_budget(Some("256")),
            Ok(Some(DAEMON_DEFAULT_CTA_BUDGET))
        );
        // Omega's own library default is untouched by the AIEN default.
        assert_eq!(ffi::OMEGA_GPU_MATMUL_MAX_CTAS, 64);
        assert!(DAEMON_DEFAULT_CTA_BUDGET_EVIDENCE.starts_with("docs/inference/evidence/"));
        let doc = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(DAEMON_DEFAULT_CTA_BUDGET_EVIDENCE)
            .join("FINDINGS.md");
        assert!(doc.is_file(), "evidence missing: {}", doc.display());
    }
}

#[cfg(test)]
mod spin_us_tests {
    use super::*;

    #[test]
    fn unset_or_blank_keeps_omega_default() {
        assert_eq!(parse_spin_us(None), Ok(None));
        assert_eq!(parse_spin_us(Some("")), Ok(None));
        assert_eq!(parse_spin_us(Some(" \n")), Ok(None));
    }

    #[test]
    fn in_range_values_parse() {
        for (raw, v) in [
            ("0", 0),
            ("2000", 2000),
            (" 2000\n", 2000),
            ("1000000", SPIN_US_MAX),
        ] {
            assert_eq!(parse_spin_us(Some(raw)), Ok(Some(v)), "{raw}");
        }
    }

    #[test]
    fn invalid_values_are_refused() {
        for raw in [
            "-1",
            "+2000",
            "abc",
            "2000.0",
            "0x10",
            "1000001",
            "99999999999",
        ] {
            let err = parse_spin_us(Some(raw)).expect_err(raw);
            assert!(err.contains(SPIN_US_ENV), "{raw}: {err}");
        }
    }

    #[test]
    fn daemon_default_is_2000_with_sealed_evidence() {
        assert_eq!(DAEMON_DEFAULT_SPIN_US, 2000);
        assert_eq!(
            parse_spin_us(Some("2000")),
            Ok(Some(DAEMON_DEFAULT_SPIN_US))
        );
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../..")
            .join(DAEMON_DEFAULT_SPIN_US_EVIDENCE);
        for file in ["FINDINGS.md", "SHA256SUMS", "run-ab.sh"] {
            assert!(
                dir.join(file).is_file(),
                "evidence missing: {}",
                dir.join(file).display()
            );
        }
    }

    fn qwen3_4b_bounds() -> ServingBounds {
        ServingBounds {
            max_context: 4096,
            max_seqs: 1,
            num_q_heads: 32,
            num_kv_heads: 8,
            head_dim: 128,
            kv_block_size: 16,
            max_rows: 256,
            max_k: 9728,
            max_n: 151936,
            max_n_one_row: 151936,
            kernel_slots: 32,
        }
    }

    #[test]
    fn serving_bytes_follow_omegas_formulas() {
        // omega attention_api.c:664-681 and matmul_api.c:130-150 at b980783, Qwen3-4B shape.
        let y = qwen3_4b_bounds().bytes();
        assert_eq!(y.attention_pool, 2 * 4096 * 8 * 128 * 4); // 32 MiB
        assert_eq!(y.attention_q_out, 16384);
        assert_eq!(y.attention_table, 20480); // (4096 + 1) * 4 rounded up to a page
                                              // Measured, not derived: omega_gpu_reserve_serving on its fake driver layer with these
                                              // bounds asked the driver for exactly 194170880 bytes (6 buffers).
        assert_eq!(y.total(), 194_170_880);
        assert_eq!(y.matmul_activation, 256 * 9728 * 2);
        assert_eq!(y.matmul_result, 256 * 151936 * 4);
        // under half a block of context the paged path stages a whole block, the larger one
        let mut b = qwen3_4b_bounds();
        (b.max_context, b.kv_block_size) = (16, 64);
        assert_eq!(b.bytes().attention_pool, 64 * 8 * 128 * 4);
    }

    #[test]
    fn reserving_without_the_engine_is_unavailable() {
        if !is_native() {
            assert_eq!(
                reserve_serving(&qwen3_4b_bounds()),
                Err(OmegaGpuError::Unavailable)
            );
            release_serving();
        }
    }

    #[test]
    fn reading_the_window_matches_the_build() {
        if is_native() {
            assert!(spin_us().is_some_and(|us| us <= SPIN_US_MAX));
        } else {
            assert_eq!(spin_us(), None);
            assert_eq!(set_spin_us(2000), Err(OmegaGpuError::Unavailable));
        }
    }
}
