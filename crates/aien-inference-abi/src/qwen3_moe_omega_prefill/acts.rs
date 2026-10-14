//! W8A16 whole-prompt activations (OWN-30B W-L1H, 2026-10-14).
//!
//! The default whole-prompt path (W8A8) rounds every GEMM input to e4m3 with a per-token, per-128 power-of-two
//! scale (4 significant bits). W8A16 keeps the same FP8 weights and the same block scales but feeds the GEMMs bf16
//! activations (8 significant bits): omega's `hconv` turns f32 rows into bf16 rows, the tensor-core GEMM
//! (`hgemm`) dequantises the weight tiles to bf16 in registers and accumulates in f32, and the MoE SwiGLU runs as
//! its own kernel (`hswiglu`) on the f32 gate|up rows, writing the down GEMM's bf16 input directly.
//!
//! The switch is [`super::WholePromptPrefill::set_acts`]; [`PrefillActs::W8A8`] stays the default. The three
//! hooks below are the only places the two modes differ; each call site in `run` / `run_dev` calls them instead
//! of `quant` / `qgemm`.

use aien_omega_pf::{HgemmCall, Layout, PfBuffer, PrefillDevice, QgemmCall, QuantIn};

use super::Bufs;

/// GEMM activation precision of the whole-prompt prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PrefillActs {
    /// e4m3 activations (per-token, per-128 power-of-two scales): the shipped path.
    #[default]
    W8A8,
    /// bf16 activations (FP8 weights dequantised to bf16 on the tensor cores, f32 accumulate).
    W8A16,
}

impl PrefillActs {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "a8" | "w8a8" => Ok(Self::W8A8),
            "a16" | "w8a16" => Ok(Self::W8A16),
            other => Err(format!("prefill activations {other:?}: expected a8 or a16")),
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::W8A8 => "a8",
            Self::W8A16 => "a16",
        }
    }
}

/// The extra W8A16 buffers of one sizing.
pub(super) struct A16Bufs<B> {
    /// HOST u32 [g + bm]: identity row index (the down GEMM reads sorted row r at r; W8A8 uses its direct mode).
    pub ident: B,
    /// DEV f32 [g][2I]: gate|up GEMM rows before the SwiGLU kernel.
    pub gu: B,
}

impl<B: PfBuffer> Bufs<B> {
    /// GEMM input rows into `xq` (W8A8: e4m3 + scales in `xs`; W8A16: bf16 in the hconv form).
    pub(super) fn act_in<D: PrefillDevice<Buf = B>>(&self, dev: &mut D, k: usize, input: QuantIn, rows: usize, x_va: u64) -> Result<u64, String> {
        match &self.a16 {
            None => dev.quant(k, input, rows, x_va, self.xq.va(), self.xs.va()),
            Some(_) => dev.hconv(k, input, rows, x_va, self.xq.va()),
        }
    }

    /// One GEMM. W8A16: `xq_va` is read as bf16 rows, `xs_va` is unused, a direct row read (`rowidx_va` 0) uses the
    /// identity index, and the SwiGLU layout writes the f32 gate|up rows to the `gu` scratch (see [`Self::act_mid`]).
    pub(super) fn gemm<D: PrefillDevice<Buf = B>>(&self, dev: &mut D, c: &QgemmCall) -> Result<u64, String> {
        let Some(a) = &self.a16 else {
            return dev.qgemm(c);
        };
        let rowidx_va = if c.rowidx_va == 0 { a.ident.va() } else { c.rowidx_va };
        let y_va = match c.layout {
            Layout::Plain => c.y_va,
            Layout::SwigluNat => a.gu.va(),
        };
        dev.hgemm(&HgemmCall { n: c.n, k: c.k, tiles_va: c.tiles_va, ntiles: c.ntiles, rowidx_va, xh_va: c.xq_va, y_va })
    }

    /// The step between gate|up and down for `g` sorted rows of width `i`. W8A8: quantise the bf16 act rows into
    /// `aq` / `a_s`. W8A16: SwiGLU from the f32 gate|up rows straight into `aq` (bf16, hconv form).
    pub(super) fn act_mid<D: PrefillDevice<Buf = B>>(&self, dev: &mut D, i: usize, g: usize) -> Result<u64, String> {
        match &self.a16 {
            None => dev.quant(i, QuantIn::Bf16, g, self.act.va(), self.aq.va(), self.a_s.va()),
            Some(a) => dev.hswiglu(i, g, a.gu.va(), self.aq.va()),
        }
    }
}
