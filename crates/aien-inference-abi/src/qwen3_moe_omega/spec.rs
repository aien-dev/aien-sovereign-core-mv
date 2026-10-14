//! OWN-30B speculative decoding, cut 1 (greedy only, prompt-lookup drafting, no draft model).
//!
//! A child module of `qwen3_moe_omega` (it needs the engine's private state) and all new code:
//! nothing in the existing decode or prefill paths is edited. Default OFF: the door only calls
//! [`Qwen3MoeOmegaEngine::generate_spec_with`] when `AIEN_OWN30B_SPEC=1` and the request is greedy.
//!
//! Round: the pending hidden state gives token `t0` exactly as plain decode does. The drafter looks
//! for the most recent earlier occurrence of the last n tokens (n = 3, 2, 1) in prompt plus output
//! and proposes the tokens that followed. `[t0, drafts..]` run through ONE batched pass
//! ([`Qwen3MoeOmegaEngine::step_verify`]); each position's hidden state goes through the same lm_head
//! launch and the same sampler as plain decode; drafts are accepted while they equal the sampler's
//! token, then the sampler's token at the first mismatch (or the bonus after a full match) becomes
//! the next pending state. The KV is rewound to the last accepted position (`rewind`, which is what
//! prefix reuse already relies on: the pool keeps its bytes and later appends overwrite them).
//!
//! Same-numerics rule (the chunk-N versus chunk-1 router trap): `step_verify` routes with the
//! device router and feeds the experts the device's own normed row (`read_post`), exactly as
//! `step` does, whenever `step` would; it never uses the host f32 router unless `step` does too.

use super::*;

pub mod suffix;
use suffix::{SuffixConfig, SuffixDrafter};

/// Where drafts come from (W-L4c).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum DraftSource {
    /// [`find_draft`]: most recent earlier occurrence of the last 1..3 tokens, fixed `max_draft`.
    #[default]
    PromptLookup,
    /// [`suffix::SuffixDrafter`]: suffix index over this request and past ones, cost-aware length.
    Suffix,
}

/// Speculative decoding settings.
#[derive(Clone, Debug)]
pub struct SpecConfig {
    pub enabled: bool,
    /// Most draft tokens per round.
    pub max_draft: usize,
    /// Longest and shortest suffix the drafter matches.
    pub max_ngram: usize,
    pub min_ngram: usize,
    /// NEGATIVE CONTROL ONLY: also accepts the first draft that does not match. Output must then
    /// differ from spec-off; a check that does not notice is broken.
    pub mutant_accept_extra: bool,
    /// NEGATIVE CONTROL ONLY (W-L4d, W-L4e): a rounding-class change inside the `Exact` verify that the
    /// token comparison must catch. See [`NumericsMutant`].
    pub mutant_numerics: NumericsMutant,
    /// Which pass checks the drafts (W-L4b). See [`VerifyPath`].
    pub verify: VerifyPath,
    /// W-L4b back-off: a round that accepts fewer than `min_accept` drafts makes the next rounds
    /// skip drafting (plain steps) for 1, 2, 4 .. 16 steps; a good round resets it. 0 = off. Only
    /// WHICH steps get a draft changes, never how a token is chosen.
    pub min_accept: usize,
    /// W-L4e cut 2: the measured break-even back-off ([`BreakEven`]). Off in `default()` (unit tests on host fakes);
    /// on in `from_env()` unless `AIEN_OWN30B_SPEC_BREAK_EVEN=0`, and on in the server.
    pub break_even: bool,
    /// RC3: the sampler is the plain argmax (greedy), so a one-list decode step's own chip argmax may stand in for
    /// reading its logits back (the same rule: lowest index of the max). Plain steps and the spec-off fallback then
    /// run exactly like the server's greedy decode. Default false.
    pub chip_argmax: bool,
    /// W-L4c: draft source and the suffix drafter's settings.
    pub source: DraftSource,
    pub suffix: SuffixConfig,
}

/// NEGATIVE CONTROL ONLY: a deliberate rounding error inside the exact verify pass (never set in serving).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum NumericsMutant {
    #[default]
    None,
    /// W-L4d: attention outputs truncated to bf16 before o_proj. Too weak: on 3 real layers it moves the logits
    /// by at most 0.036 (W-L4e probe), and the 2026-10-14 full-model gate never saw it change a token.
    AttnBf16,
    /// W-L4e: the residual stream kept in bf16, as a verify kernel with bf16 activations would: after each
    /// residual add (attention and MoE) every word of x is truncated to bf16. x is never quantised again
    /// before the device router, the norms and the next layer, so the error reaches the routing (expert
    /// choice), the KV written for accepted positions and the final hidden state, and compounds over the layers.
    ResidualBf16,
    /// W-L4e: W8A8 activations, the rounding class of the whole-prompt path: every GEMM input row of the verify
    /// (input-normed row, attention output, post-normed row, SwiGLU output rounded to bf16 first) goes through the
    /// prefill quantiser (`fake_quant_row`: e4m3, power-of-two scale per 128 block) before the FP8 GEMM. The
    /// whole-prompt verify, which has exactly these numerics, diverged from spec-off on 6 of 10 full-model runs
    /// (W-L4c cut 2 receipt), so the token check is known to see this class on the real model.
    ActFp8,
}

impl NumericsMutant {
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" => Ok(Self::None),
            "attn-bf16" => Ok(Self::AttnBf16),
            "residual-bf16" => Ok(Self::ResidualBf16),
            "act-fp8" => Ok(Self::ActFp8),
            other => Err(format!("numerics mutant {other:?}: none, attn-bf16, residual-bf16 or act-fp8")),
        }
    }
}

/// Truncates every word to bf16 (keeps the top 16 bits).
fn truncate_bf16(v: &mut [f32]) {
    v.iter_mut().for_each(|x| *x = f32::from_bits(x.to_bits() & 0xffff_0000));
}

/// W8A8 fake quantisation of each `width`-wide row (the prefill quantiser).
fn fake_quant_rows(v: &mut [f32], width: usize, bf16_first: bool) {
    use crate::qwen3_moe_omega_prefill::fake_quant::{bf16_round_row, fake_quant_row};
    for row in v.chunks_mut(width) {
        if bf16_first {
            bf16_round_row(row);
        }
        fake_quant_row(row);
    }
}

/// The pass that runs `[t0, drafts..]` in one go.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum VerifyPath {
    /// [`Qwen3MoeOmegaEngine::step_verify`]: GEMM batched, f32 activations, device router per token:
    /// the same rounding as plain decode (token-identical on the chip, receipt own30b-l4-spec-chip-20261013T2158Z).
    #[default]
    Decode,
    /// [`Qwen3MoeOmegaEngine::step_verify_wp`]: the whole-prompt prefill driver (fixed launch count
    /// per layer, FP8 activations, host router). Faster, but rounds differently from plain decode,
    /// so accepted tokens follow the W8A8 numerics. Needs `set_whole_prompt(true)`.
    WholePrompt,
    /// W-L4d: [`Qwen3MoeOmegaEngine::step_verify_exact`]: the same numerics as `Decode` (every row
    /// bit-identical to a plain `step`, probe receipt r4-rows-attn-probe) with serial host loops and
    /// omega's multi-row decode attention (two launches per layer for all rows). Needs the device
    /// router and the fused KV; `AIEN_OWN30B_SPEC_VERIFY=exact`.
    Exact,
}

pub const SPEC_ENV: &str = "AIEN_OWN30B_SPEC";

impl Default for SpecConfig {
    fn default() -> Self {
        Self { enabled: false, max_draft: 8, max_ngram: 3, min_ngram: 1, mutant_accept_extra: false, mutant_numerics: NumericsMutant::None, verify: VerifyPath::Decode, min_accept: 0, break_even: false, chip_argmax: false, source: DraftSource::PromptLookup, suffix: SuffixConfig::default() }
    }
}

impl SpecConfig {
    /// `AIEN_OWN30B_SPEC=1|on|true` turns it on; `AIEN_OWN30B_SPEC_K`, `_MAX_N`, `_MIN_N` tune it.
    pub fn from_env() -> Self {
        let get = |k: &str| std::env::var(k).ok();
        let num = |k: &str, d: usize| get(k).and_then(|v| v.trim().parse::<usize>().ok()).unwrap_or(d);
        let d = Self::default();
        Self {
            enabled: matches!(get(SPEC_ENV).as_deref().map(str::trim), Some("1") | Some("on") | Some("true")),
            max_draft: num("AIEN_OWN30B_SPEC_K", d.max_draft),
            max_ngram: num("AIEN_OWN30B_SPEC_MAX_N", d.max_ngram),
            min_ngram: num("AIEN_OWN30B_SPEC_MIN_N", d.min_ngram),
            mutant_accept_extra: false,
            mutant_numerics: NumericsMutant::None,
            min_accept: num("AIEN_OWN30B_SPEC_MIN_ACCEPT", d.min_accept),
            break_even: !matches!(get("AIEN_OWN30B_SPEC_BREAK_EVEN").as_deref().map(str::trim), Some("0") | Some("off")),
            chip_argmax: false,
            // AIEN_OWN30B_SPEC_VERIFY=wp: verify through the whole-prompt driver (W-L4b).
            verify: match get("AIEN_OWN30B_SPEC_VERIFY").as_deref().map(str::trim) {
                Some("wp") | Some("whole-prompt") => VerifyPath::WholePrompt,
                // W-L4d: AIEN_OWN30B_SPEC_VERIFY=exact: decode numerics, faster pass (step_verify_exact).
                Some("exact") => VerifyPath::Exact,
                _ => VerifyPath::Decode,
            },
            // W-L4c: AIEN_OWN30B_SPEC_DRAFTER=suffix, tuned by AIEN_OWN30B_SPEC_SUFFIX_* (see SuffixConfig::from_env).
            source: match get("AIEN_OWN30B_SPEC_DRAFTER").as_deref().map(str::trim) {
                Some("suffix") => DraftSource::Suffix,
                _ => DraftSource::PromptLookup,
            },
            suffix: SuffixConfig::from_env(),
        }
    }
}

/// W-L4d: one operation's totals in the exact verify pass.
#[derive(Clone, Debug, Default)]
pub struct OpProf {
    /// Wall time of the calls (host API time included).
    pub wall_ns: u64,
    /// Chip time the kernels reported.
    pub chip_ns: u64,
    pub calls: u64,
}

impl OpProf {
    fn add(&mut self, since: Instant, chip_ns: u64) {
        self.add_n(since, chip_ns, 1);
    }
    fn add_n(&mut self, since: Instant, chip_ns: u64, calls: u64) {
        self.wall_ns += since.elapsed().as_nanos() as u64;
        self.chip_ns += chip_ns;
        self.calls += calls;
    }
}

/// W-L4d: per-operation profile of [`Qwen3MoeOmegaEngine::step_verify_exact_prof`] (summed over calls).
#[derive(Clone, Debug, Default)]
pub struct ExactProf {
    pub calls: u64,
    pub rows: u64,
    pub qkv: OpProf,
    pub attn: OpProf,
    pub o: OpProf,
    pub route: OpProf,
    pub gate_up: OpProf,
    pub down: OpProf,
    /// Host norms, residual adds, sort, SwiGLU, combine.
    pub host_ns: u64,
}

impl ExactProf {
    /// One line, milliseconds per call: `op wall/chip` for each operation.
    pub fn line(&self) -> String {
        let n = self.calls.max(1) as f64;
        let f = |o: &OpProf| format!("{:.3}/{:.3}", o.wall_ns as f64 / n / 1e6, o.chip_ns as f64 / n / 1e6);
        format!(
            "qkv {} attn {} o {} route {} gate_up {} down {} host {:.3} (ms per call, wall/chip)",
            f(&self.qkv),
            f(&self.attn),
            f(&self.o),
            f(&self.route),
            f(&self.gate_up),
            f(&self.down),
            self.host_ns as f64 / n / 1e6
        )
    }
}

/// Counters of one speculative generation.
#[derive(Clone, Debug, Default)]
pub struct SpecStats {
    /// Rounds that verified a draft in one batched pass.
    pub rounds: usize,
    /// Rounds with no draft (plain single step).
    pub plain_steps: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// Rounds verified through the whole-prompt driver (the rest used `step_verify`).
    pub wp_rounds: usize,
    /// Plain steps taken because of the back-off (also counted in `plain_steps`).
    pub backoff_steps: usize,
    /// W-L4c: wall time of the verify passes (ns), so verify ms per round can be reported.
    pub verify_ns: u64,
    /// Rounds whose draft came from the global (past requests) suffix tree.
    pub global_rounds: usize,
    /// W-L4e: the break-even state at the end (measured costs), when it was on.
    pub break_even: Option<BreakEven>,
}

impl SpecStats {
    pub fn acceptance(&self) -> f64 {
        if self.drafted == 0 {
            0.0
        } else {
            self.accepted as f64 / self.drafted as f64
        }
    }
}


/// W-L4e cut 2: the break-even back-off. Measures what a plain token costs (step + lm_head) and what a
/// speculative round costs and yields (verify wall + lm_head per evaluated row; tokens = accepted + 1), as
/// running averages over the recent rounds. When the recent rounds yield fewer tokens per ms than plain decode,
/// drafting stops for `16, 32, .. max_backoff` steps; the round after a pause is a probe with a short draft
/// (at most `probe_draft` tokens, the cheapest verify), so a probe that fails costs little. A round that
/// pays clears the pause. Only WHICH steps draft changes, never how a token is chosen.
#[derive(Clone, Debug)]
pub struct BreakEven {
    /// Running averages (weight `alpha` on the newest sample).
    pub plain_ms: f64,
    pub round_ms: f64,
    pub tokens_per_round: f64,
    /// Measured verify wall / the cost model's prediction for the same rows (scales the drafter's costs).
    pub verify_scale: f64,
    pub plain_seen: usize,
    pub rounds_seen: usize,
    skip: usize,
    backoff: usize,
    probing: bool,
    pub alpha: f64,
    /// Rounds measured before the first decision.
    pub min_rounds: usize,
    pub max_backoff: usize,
    pub probe_draft: usize,
}

impl BreakEven {
    /// `plain_ms_seed`: the plain token cost until one is measured (the cost model's `plain_ms`).
    pub fn new(plain_ms_seed: f64) -> Self {
        Self {
            plain_ms: plain_ms_seed,
            round_ms: 0.0,
            tokens_per_round: 0.0,
            verify_scale: 1.0,
            plain_seen: 0,
            rounds_seen: 0,
            skip: 0,
            backoff: 0,
            probing: false,
            alpha: 0.5,
            min_rounds: 1,
            max_backoff: 128,
            probe_draft: 1,
        }
    }

    fn ema(old: f64, new: f64, n_before: usize, alpha: f64) -> f64 {
        if n_before == 0 {
            new
        } else {
            old + alpha * (new - old)
        }
    }

    /// A plain token took `ms` (step + lm_head).
    pub fn observe_plain(&mut self, ms: f64) {
        self.plain_ms = Self::ema(self.plain_ms, ms, self.plain_seen, self.alpha);
        self.plain_seen += 1;
    }

    /// Before a step: `None` = no draft this step (paused), `Some(cap)` = draft at most `cap` tokens.
    pub fn draft_cap(&mut self, k: usize) -> Option<usize> {
        if self.skip > 0 {
            self.skip -= 1;
            return None;
        }
        Some(if self.probing { k.min(self.probe_draft) } else { k })
    }

    /// Recent rounds' tokens per ms vs plain decode's.
    pub fn pays(&self) -> bool {
        self.round_ms > 0.0 && self.tokens_per_round / self.round_ms >= 1.0 / self.plain_ms
    }

    /// A round ran: `ms` wall (verify + lm_head of the evaluated rows), `tokens` emitted by it
    /// (accepted + 1), `verify_ms` of the verify pass alone and `predicted_verify_ms` from the cost model.
    pub fn observe_round(&mut self, ms: f64, tokens: usize, verify_ms: f64, predicted_verify_ms: f64) {
        let n = self.rounds_seen;
        // A probe after a pause starts the averages afresh (the text may have changed).
        let fresh = if self.probing { 0 } else { n };
        self.round_ms = Self::ema(self.round_ms, ms, fresh, self.alpha);
        self.tokens_per_round = Self::ema(self.tokens_per_round, tokens as f64, fresh, self.alpha);
        if predicted_verify_ms > 0.0 {
            self.verify_scale = Self::ema(self.verify_scale, verify_ms / predicted_verify_ms, n, self.alpha);
        }
        self.rounds_seen += 1;
        if self.rounds_seen < self.min_rounds && !self.probing {
            return;
        }
        if self.pays() {
            self.backoff = 0;
            self.probing = false;
        } else {
            self.backoff = (self.backoff * 2).clamp(16, self.max_backoff.max(16));
            self.skip = self.backoff;
            self.probing = true;
        }
    }

    /// The cost model the drafter should weigh: measured plain cost, verify scaled by what was measured.
    pub fn cost(&self, base: &suffix::CostModel) -> suffix::CostModel {
        let s = self.verify_scale as f32;
        suffix::CostModel {
            verify_base_ms: base.verify_base_ms * s,
            verify_row_ms: base.verify_row_ms * s,
            lm_ms: base.lm_ms,
            // Text with no measured round yet, or a probe after a pause, must look clearly better than plain
            // decode (plain cost seen as 1/1.2 of itself): an unproven round is where the back-off loses time.
            plain_ms: (if self.plain_seen > 0 { self.plain_ms as f32 } else { base.plain_ms }) / if self.rounds_seen == 0 || self.probing { 1.2 } else { 1.0 },
        }
    }
}

/// Prompt-lookup draft: the tokens that followed the most recent earlier occurrence of the last
/// n tokens of `ctx` (n from `max_n` down to `min_n`), at most `k`. Empty when nothing matches.
pub fn find_draft(ctx: &[u32], max_n: usize, min_n: usize, k: usize) -> Vec<u32> {
    let len = ctx.len();
    if k == 0 || min_n == 0 {
        return Vec::new();
    }
    for n in (min_n..=max_n.min(len.saturating_sub(1))).rev() {
        let suffix = &ctx[len - n..];
        // An earlier occurrence starting at s has at least one following token: s + n < len.
        for s in (0..len - n).rev() {
            if &ctx[s..s + n] == suffix {
                let end = (s + n + k).min(len);
                return ctx[s + n..end].to_vec();
            }
        }
    }
    Vec::new()
}

/// Whether the model's token `pred` at a verify position confirms the draft there. A stop id never
/// counts as a confirmed draft (the caller finishes on it as the next pending token).
pub fn draft_confirmed(draft: Option<u32>, pred: u32, stops: &[u32]) -> bool {
    matches!(draft, Some(d) if d == pred) && !stops.contains(&pred)
}

/// Longest accepted prefix of `drafts` given the model's token at every verify position
/// (`preds[j]` follows inputs `0..=j`; `preds.len() == drafts.len() + 1`).
pub fn accepted_prefix(drafts: &[u32], preds: &[u32], stops: &[u32]) -> usize {
    let mut j = 0;
    while j < drafts.len() && draft_confirmed(drafts.get(j).copied(), preds[j], stops) {
        j += 1;
    }
    j
}

impl Qwen3MoeOmegaEngine {
    /// Whether [`Self::step_verify`] can run: FP8 attention and experts with the prefill GEMM
    /// everywhere, no tracing. Otherwise speculative decoding falls back to plain decode.
    pub fn spec_possible(&self) -> bool {
        self.trace_layers.is_empty() && gemm_everywhere(&self.weights)
    }

    /// Whether the verify pass `path` can run on this engine. `WholePrompt` needs the whole-prompt
    /// switch on (`set_whole_prompt(true)`: fused KV, prefill kernels linked), no tracing and no
    /// routing mutant (the same conditions as `whole_prompt_prefill`).
    pub fn spec_verify_possible(&self, path: VerifyPath) -> bool {
        match path {
            VerifyPath::Decode => self.spec_possible(),
            VerifyPath::Exact => self.spec_possible() && self.exact_verify_possible(),
            VerifyPath::WholePrompt => self.whole_prompt.is_some() && self.trace_layers.is_empty() && !self.routing_mutant,
        }
    }

    /// W-L4b: runs `tokens` (consecutive positions from `pos`) through the whole-prompt driver in
    /// ONE call (fixed launch count per layer) and returns every position's final-normed hidden
    /// state. Writes their KV into the fused cache, advances `pos` and `history`; the caller
    /// rewinds what it rejects. Numerics are the whole-prompt ones (FP8 activations, host router).
    pub fn step_verify_wp(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let m = tokens.len();
        if m == 0 {
            return Err("step_verify_wp: empty batch".into());
        }
        if !self.spec_verify_possible(VerifyPath::WholePrompt) {
            return Err("step_verify_wp: whole-prompt switch off, or tracing / routing mutant on".into());
        }
        let h = self.weights.config.hidden;
        let pos0 = self.pos;
        let t = Instant::now();
        let wp = self.whole_prompt.as_mut().expect("checked above");
        let fused = self.kv.fused_kv().ok_or("step_verify_wp: KV is not omega's fused cache")?;
        if pos0 + m > fused.capacity() {
            return Err(format!("context full: {pos0} + {m} of {}", fused.capacity()));
        }
        let out = wp.run_rows(&self.weights, &mut fused.prefill_attention(), tokens, pos0);
        let rows = match out {
            Ok(r) => r,
            Err(e) => {
                self.rewind(pos0);
                return Err(format!("step_verify_wp x{m}: {e}"));
            }
        };
        fused.mark_filled(pos0 + m);
        self.pos = pos0 + m;
        self.history.extend_from_slice(tokens);
        let st = &wp.stats;
        self.last_timing = StepTiming {
            total_ns: t.elapsed().as_nanos() as u64,
            attention_ns: st.attn_ns,
            experts_ns: 0,
            lm_head_ns: 0,
            host_ns: st.host_ns,
            chip_ns: st.chip_ns + st.attn_ns,
            launches: st.launches() as u64,
            route_ns: 0,
            moe_host_ns: 0,
            ..StepTiming::default()
        };
        Ok(rows.chunks(h).map(<[f32]>::to_vec).collect())
    }

    /// Runs `tokens` (consecutive positions, at most 128) in one batched pass and returns the
    /// final-normed hidden state of EVERY position. Layer-major like `step_batch_gemm`, but
    /// routing and the experts' input row follow `step` (see the module note). Advances `pos` and
    /// `history` by `tokens.len()`; the caller rewinds what it rejects.
    pub fn step_verify(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let m = tokens.len();
        if m == 0 {
            return Err("step_verify: empty batch".into());
        }
        if !self.spec_possible() {
            return Err("step_verify: needs FP8 attention and experts with the prefill GEMM, and no tracing".into());
        }
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let top_k = c.top_k;
        if m > aien_omega_fp8::GEMM_MAX_ROWS / top_k {
            return Err(format!("step_verify: {m} tokens exceed one GEMM launch"));
        }
        let (h, qd, kvd, hd, eps) = (c.hidden, c.q_dim(), c.kv_dim(), c.head_dim, c.rms_eps);
        let (e, i, qkv_n) = (c.experts, c.moe_intermediate, c.qkv_n());
        let pos0 = self.pos;
        if pos0 + m > self.kv.capacity() {
            return Err(format!("context full: {pos0} + {m} of {}", self.kv.capacity()));
        }
        let g = m * top_k;
        let mut x: Vec<f32> = tokens.iter().flat_map(|&t| self.weights.embed_row(t)).collect();
        let mut x_norm = vec![0.0f32; m * h];
        let mut qkv_all = vec![0.0f32; m * qkv_n];
        let mut qn = vec![0.0f32; qd];
        let mut kn = vec![0.0f32; kvd];
        let mut attn_all = vec![0.0f32; m * qd];
        let mut proj_all = vec![0.0f32; m * h];
        let mut post = vec![0.0f32; m * h];
        let mut logits = vec![0.0f32; m * e];
        let mut ids = vec![0u32; g];
        let mut wts = vec![0.0f32; g];
        let mut sorted_expert = vec![0u32; g];
        let mut sorted_token = vec![0u32; g];
        let mut inv = vec![0u32; g];
        let mut gu_all = vec![0.0f32; g * 2 * i];
        let mut act_all = vec![0.0f32; g * i];
        let mut down_all = vec![0.0f32; g * h];
        let dense_ids = vec![0u32; m];
        let rows_m: Vec<u32> = (0..m as u32).collect();
        let rows_g: Vec<u32> = (0..g as u32).collect();
        // Exactly `step`'s condition for the device router (every layer here has FP8 experts).
        let device_router = match &self.weights.device_router {
            Some(r) if self.device_routing && !self.routing_mutant => Some(r.clone()),
            _ => None,
        };

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) = (&layer.attn, &layer.experts) else {
                return Err(format!("layer {layer_idx}: step_verify needs FP8 attention and experts"));
            };
            let t0 = Instant::now();
            for r in 0..m {
                rmsnorm(&x[r * h..(r + 1) * h], &layer.input_norm, eps, &mut x_norm[r * h..(r + 1) * h]);
            }
            timing.host_ns += t0.elapsed().as_nanos() as u64;
            let t1 = Instant::now();
            timing.chip_ns += qkv.gemm_f32(&dense_ids, &rows_m, &x_norm, &mut qkv_all).map_err(|e| format!("layer {layer_idx} qkv verify x{m}: {e}"))?;
            timing.launches += 1;
            for r in 0..m {
                let pos = pos0 + r;
                let qkv_r = &qkv_all[r * qkv_n..(r + 1) * qkv_n];
                let out_r = &mut attn_all[r * qd..(r + 1) * qd];
                match self.kv.fused_attend(layer_idx, pos, qkv_r, eps, out_r) {
                    Some(res) => timing.chip_ns += res?,
                    None => {
                        rmsnorm_per_head(&qkv_r[..qd], &layer.q_norm, eps, c.q_heads, hd, &mut qn);
                        rmsnorm_per_head(&qkv_r[qd..qd + kvd], &layer.k_norm, eps, c.kv_heads, hd, &mut kn);
                        apply_rope(&mut qn, &mut kn, pos, c.q_heads, c.kv_heads, hd, c.rope_theta);
                        self.kv.append(layer_idx, pos, &kn, &qkv_r[qd + kvd..])?;
                        timing.chip_ns += self.kv.attend(layer_idx, &qn, pos + 1, out_r)?;
                    }
                }
                timing.launches += 1;
            }
            timing.chip_ns += o.gemm_f32(&dense_ids, &rows_m, &attn_all, &mut proj_all).map_err(|e| format!("layer {layer_idx} o verify x{m}: {e}"))?;
            timing.launches += 1;
            timing.attention_ns += t1.elapsed().as_nanos() as u64;

            let t3 = Instant::now();
            for r in 0..m {
                let xr = &mut x[r * h..(r + 1) * h];
                for (xi, p) in xr.iter_mut().zip(&proj_all[r * h..(r + 1) * h]) {
                    *xi += p;
                }
            }
            timing.host_ns += t3.elapsed().as_nanos() as u64;
            let t4 = Instant::now();
            let tr = Instant::now();
            if let Some(router) = &device_router {
                // Per token: the device router on the residual row, then the device's own normed
                // row back (what decode's grouped_dev launch reads), so routing AND the experts'
                // input are the decode values.
                for r in 0..m {
                    let (a, b) = (r * top_k, (r + 1) * top_k);
                    let (ns, _) = router
                        .route(layer_idx, &x[r * h..(r + 1) * h], eps, &mut ids[a..b], &mut wts[a..b])
                        .map_err(|e| format!("layer {layer_idx} device router (verify token {r} of {m}): {e}"))?;
                    timing.chip_ns += ns;
                    timing.launches += 1;
                    router.read_post(&mut post[r * h..(r + 1) * h]).map_err(|e| format!("layer {layer_idx} router readback: {e}"))?;
                }
            } else {
                for r in 0..m {
                    rmsnorm(&x[r * h..(r + 1) * h], &layer.post_norm, eps, &mut post[r * h..(r + 1) * h]);
                }
                router_logits(&layer.router, &post, h, e, &mut logits);
                let plan = MoeBatchPlan::from_logits(&logits, m, e, top_k).map_err(|e| format!("layer {layer_idx} routing: {e}"))?;
                for (dst, &ex) in ids.iter_mut().zip(&plan.topk_experts) {
                    *dst = if self.routing_mutant { (ex + 1) % e as u32 } else { ex };
                }
                wts.copy_from_slice(&plan.topk_weights);
            }
            timing.route_ns += tr.elapsed().as_nanos() as u64;

            let mut start = vec![0u32; e + 1];
            for &ex in &ids {
                start[ex as usize + 1] += 1;
            }
            for ex in 0..e {
                start[ex + 1] += start[ex];
            }
            for (gi, &ex) in ids.iter().enumerate() {
                let slot = &mut start[ex as usize];
                sorted_expert[*slot as usize] = ex;
                sorted_token[*slot as usize] = (gi / top_k) as u32;
                inv[gi] = *slot;
                *slot += 1;
            }
            timing.chip_ns += gate_up
                .gemm_f32(&sorted_expert, &sorted_token, &post, &mut gu_all)
                .map_err(|e| format!("layer {layer_idx} gate_up verify x{g}: {e}"))?;
            timing.launches += 1;
            swiglu_rows(&gu_all, &mut act_all, i);
            timing.chip_ns += down
                .gemm_f32(&sorted_expert, &rows_g, &act_all, &mut down_all)
                .map_err(|e| format!("layer {layer_idx} down verify x{g}: {e}"))?;
            timing.launches += 1;
            combine_rows(&mut x, &down_all, &wts, &inv, h, top_k);
            timing.experts_ns += t4.elapsed().as_nanos() as u64;
        }
        let t5 = Instant::now();
        let mut hiddens = Vec::with_capacity(m);
        for r in 0..m {
            let mut hid = vec![0.0f32; h];
            rmsnorm(&x[r * h..(r + 1) * h], &self.weights.final_norm, eps, &mut hid);
            hiddens.push(hid);
        }
        timing.host_ns += t5.elapsed().as_nanos() as u64;
        self.pos = pos0 + m;
        self.history.extend_from_slice(tokens);
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        self.last_timing = timing;
        Ok(hiddens)
    }

    /// W-L4d: the exact verify pass (every row's hidden state bit-identical to a plain `step` on that
    /// token), built for speed. See [`Self::step_verify_exact_prof`].
    pub fn step_verify_exact(&mut self, tokens: &[u32]) -> Result<Vec<Vec<f32>>, String> {
        let mut prof = ExactProf::default();
        self.step_verify_exact_prof(tokens, &mut prof)
    }

    /// [`Self::step_verify_exact`] with a per-operation profile (wall ns of each call and the chip ns
    /// the kernels report), added into `prof`.
    ///
    /// Same operations, order and numerics as [`Self::step_verify`] (each row: host input norm, the
    /// projections through omega's FP8 GEMM which is bit-identical to decode's GEMV per row, decode's
    /// fused attention per row in position order, the device router per row and its own normed row as
    /// the experts' input, host SwiGLU, combine in slot order from zero), with these speed changes:
    ///   * host loops serial (no rayon wake-ups for at most a few rows).
    pub fn step_verify_exact_prof(&mut self, tokens: &[u32], prof: &mut ExactProf) -> Result<Vec<Vec<f32>>, String> {
        self.step_verify_exact_inner(tokens, prof, NumericsMutant::None)
    }

    /// Whether [`Self::step_verify_exact`] can run here: the device router (decode routes on the
    /// device) and the fused KV (decode's attention kernel).
    pub fn exact_verify_possible(&self) -> bool {
        self.weights.device_router.is_some() && self.device_routing && !self.routing_mutant && self.kv.path_name() == "fused"
    }

    /// `mutant`: NEGATIVE CONTROL ONLY (see [`NumericsMutant`]); `None` in every real use.
    pub fn step_verify_exact_inner(&mut self, tokens: &[u32], prof: &mut ExactProf, mutant: NumericsMutant) -> Result<Vec<Vec<f32>>, String> {
        let m = tokens.len();
        if m == 0 {
            return Err("step_verify_exact: empty batch".into());
        }
        if !self.spec_possible() {
            return Err("step_verify_exact: needs FP8 attention and experts with the prefill GEMM, and no tracing".into());
        }
        let t_step = Instant::now();
        let mut timing = StepTiming::default();
        let c = &self.weights.config;
        let top_k = c.top_k;
        if m > aien_omega_fp8::GEMM_MAX_ROWS / top_k {
            return Err(format!("step_verify_exact: {m} tokens exceed one GEMM launch"));
        }
        let (h, qd, eps) = (c.hidden, c.q_dim(), c.rms_eps);
        let (e, i, qkv_n) = (c.experts, c.moe_intermediate, c.qkv_n());
        let pos0 = self.pos;
        if pos0 + m > self.kv.capacity() {
            return Err(format!("context full: {pos0} + {m} of {}", self.kv.capacity()));
        }
        let g = m * top_k;
        let mut x: Vec<f32> = tokens.iter().flat_map(|&t| self.weights.embed_row(t)).collect();
        let mut x_norm = vec![0.0f32; m * h];
        let mut qkv_all = vec![0.0f32; m * qkv_n];
        let mut attn_all = vec![0.0f32; m * qd];
        let mut proj_all = vec![0.0f32; m * h];
        let mut post = vec![0.0f32; m * h];
        let mut ids = vec![0u32; g];
        let mut wts = vec![0.0f32; g];
        let mut sorted_expert = vec![0u32; g];
        let mut sorted_token = vec![0u32; g];
        let mut inv = vec![0u32; g];
        let mut gu_all = vec![0.0f32; g * 2 * i];
        let mut act_all = vec![0.0f32; g * i];
        let mut down_all = vec![0.0f32; g * h];
        let mut acc = vec![0.0f32; h];
        let dense_ids = vec![0u32; m];
        let rows_m: Vec<u32> = (0..m as u32).collect();
        let rows_g: Vec<u32> = (0..g as u32).collect();
        let router = match &self.weights.device_router {
            Some(r) if self.device_routing && !self.routing_mutant => r.clone(),
            _ => return Err("step_verify_exact: needs the device router (decode routes on the device)".into()),
        };
        let mut start = vec![0u32; e + 1];
        // A/B switch for the probe: AIEN_OWN30B_EXACT_ROWS_ATTN=0 keeps one attention launch per row.
        let rows_attn = std::env::var("AIEN_OWN30B_EXACT_ROWS_ATTN").as_deref() != Ok("0");
        prof.calls += 1;
        prof.rows += m as u64;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let (AttnWeights::Fp8 { qkv, o }, ExpertWeights::Fp8 { gate_up, down }) = (&layer.attn, &layer.experts) else {
                return Err(format!("layer {layer_idx}: step_verify_exact needs FP8 attention and experts"));
            };
            let t = Instant::now();
            for r in 0..m {
                rmsnorm(&x[r * h..(r + 1) * h], &layer.input_norm, eps, &mut x_norm[r * h..(r + 1) * h]);
            }
            prof.host_ns += t.elapsed().as_nanos() as u64;

            let t = Instant::now();
            if mutant == NumericsMutant::ActFp8 {
                fake_quant_rows(&mut x_norm, h, false);
            }
            let ns = qkv.gemm_f32(&dense_ids, &rows_m, &x_norm, &mut qkv_all).map_err(|e| format!("layer {layer_idx} qkv exact x{m}: {e}"))?;
            prof.qkv.add(t, ns);
            let t = Instant::now();
            let mut chip = 0u64;
            // All rows in two launches (omega's rows attention, bit-identical to the per-row calls), else per row.
            let rows_done = if m > 1 && rows_attn {
                match self.kv.fused_attend_rows(layer_idx, pos0, m, &qkv_all, eps, &mut attn_all) {
                    Some(res) => {
                        chip += res?;
                        true
                    }
                    None => false,
                }
            } else {
                false
            };
            if !rows_done {
                for r in 0..m {
                    let qkv_r = &qkv_all[r * qkv_n..(r + 1) * qkv_n];
                    let out_r = &mut attn_all[r * qd..(r + 1) * qd];
                    match self.kv.fused_attend(layer_idx, pos0 + r, qkv_r, eps, out_r) {
                        Some(res) => chip += res?,
                        None => return Err("step_verify_exact: needs the fused attention kernel (decode's)".into()),
                    }
                }
            }
            prof.attn.add_n(t, chip, if rows_done { 2 } else { m as u64 });
            let t = Instant::now();
            if mutant == NumericsMutant::AttnBf16 {
                truncate_bf16(&mut attn_all);
            }
            if mutant == NumericsMutant::ActFp8 {
                fake_quant_rows(&mut attn_all, qd, false);
            }
            let ns = o.gemm_f32(&dense_ids, &rows_m, &attn_all, &mut proj_all).map_err(|e| format!("layer {layer_idx} o exact x{m}: {e}"))?;
            prof.o.add(t, ns);

            let t = Instant::now();
            for (xi, p) in x.iter_mut().zip(&proj_all) {
                *xi += p;
            }
            if mutant == NumericsMutant::ResidualBf16 {
                truncate_bf16(&mut x);
            }
            prof.host_ns += t.elapsed().as_nanos() as u64;
            // Per token: the device router on the residual row, then the device's own normed row back.
            let t = Instant::now();
            let mut chip = 0u64;
            for r in 0..m {
                let (a, b) = (r * top_k, (r + 1) * top_k);
                let (ns, _) = router
                    .route(layer_idx, &x[r * h..(r + 1) * h], eps, &mut ids[a..b], &mut wts[a..b])
                    .map_err(|e| format!("layer {layer_idx} device router (exact token {r} of {m}): {e}"))?;
                chip += ns;
                router.read_post(&mut post[r * h..(r + 1) * h]).map_err(|e| format!("layer {layer_idx} router readback: {e}"))?;
            }
            prof.route.add_n(t, chip, m as u64);

            let t = Instant::now();
            start.iter_mut().for_each(|s| *s = 0);
            for &ex in &ids {
                start[ex as usize + 1] += 1;
            }
            for ex in 0..e {
                start[ex + 1] += start[ex];
            }
            for (gi, &ex) in ids.iter().enumerate() {
                let slot = &mut start[ex as usize];
                sorted_expert[*slot as usize] = ex;
                sorted_token[*slot as usize] = (gi / top_k) as u32;
                inv[gi] = *slot;
                *slot += 1;
            }
            prof.host_ns += t.elapsed().as_nanos() as u64;
            let t = Instant::now();
            if mutant == NumericsMutant::ActFp8 {
                fake_quant_rows(&mut post, h, false);
            }
            let ns = gate_up.gemm_f32(&sorted_expert, &sorted_token, &post, &mut gu_all).map_err(|e| format!("layer {layer_idx} gate_up exact x{g}: {e}"))?;
            prof.gate_up.add(t, ns);
            let t = Instant::now();
            for (a, gu) in act_all.chunks_mut(i).zip(gu_all.chunks(2 * i)) {
                swiglu(gu, i, a);
            }
            prof.host_ns += t.elapsed().as_nanos() as u64;
            let t = Instant::now();
            if mutant == NumericsMutant::ActFp8 {
                fake_quant_rows(&mut act_all, i, true);
            }
            let ns = down.gemm_f32(&sorted_expert, &rows_g, &act_all, &mut down_all).map_err(|e| format!("layer {layer_idx} down exact x{g}: {e}"))?;
            prof.down.add(t, ns);
            // Combine per token in slot order from zero, then the residual add (decode's order).
            let t = Instant::now();
            for r in 0..m {
                acc.iter_mut().for_each(|a| *a = 0.0);
                for slot in 0..top_k {
                    let gi = r * top_k + slot;
                    let w = wts[gi];
                    let row = inv[gi] as usize;
                    for (a, d) in acc.iter_mut().zip(&down_all[row * h..(row + 1) * h]) {
                        *a += w * d;
                    }
                }
                for (xi, a) in x[r * h..(r + 1) * h].iter_mut().zip(&acc) {
                    *xi += a;
                }
            }
            if mutant == NumericsMutant::ResidualBf16 {
                truncate_bf16(&mut x);
            }
            prof.host_ns += t.elapsed().as_nanos() as u64;
        }
        let t = Instant::now();
        let mut hiddens = Vec::with_capacity(m);
        for r in 0..m {
            let mut hid = vec![0.0f32; h];
            rmsnorm(&x[r * h..(r + 1) * h], &self.weights.final_norm, eps, &mut hid);
            hiddens.push(hid);
        }
        prof.host_ns += t.elapsed().as_nanos() as u64;
        self.pos = pos0 + m;
        self.history.extend_from_slice(tokens);
        let ops = [&prof.qkv, &prof.attn, &prof.o, &prof.route, &prof.gate_up, &prof.down];
        timing.chip_ns = ops.iter().map(|o| o.chip_ns).sum();
        timing.total_ns = t_step.elapsed().as_nanos() as u64;
        timing.attention_ns = prof.qkv.wall_ns + prof.attn.wall_ns + prof.o.wall_ns;
        timing.experts_ns = prof.route.wall_ns + prof.gate_up.wall_ns + prof.down.wall_ns;
        timing.route_ns = prof.route.wall_ns;
        timing.host_ns = prof.host_ns;
        self.last_timing = timing;
        Ok(hiddens)
    }

    /// Greedy generation with speculative decoding. Same contract and same emitted tokens as
    /// [`Self::generate_with`] when the sampler is greedy; falls back to it (stats all zero) when
    /// `cfg.enabled` is false or [`Self::spec_possible`] is false. Prefill is shared code.
    pub fn generate_spec_with(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        cfg: &SpecConfig,
        sampler: &mut dyn FnMut(&[f32], usize) -> u32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(GenerateOutcome, SpecStats), String> {
        // A suffix drafter that lives for this request only (no past requests).
        let mut local = (cfg.source == DraftSource::Suffix).then(|| SuffixDrafter::new(SuffixConfig { use_global: false, ..cfg.suffix.clone() }));
        self.generate_spec_drafter(prompt, params, cfg, local.as_mut(), sampler, on_token)
    }

    /// [`Self::generate_spec_with`] with a caller-owned suffix drafter (W-L4c): it keeps its global
    /// tree across requests (past conversations) and gets this request's prompt and output. With
    /// `drafter` `None` the drafts are prompt-lookup ([`find_draft`]).
    pub fn generate_spec_drafter(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        cfg: &SpecConfig,
        mut drafter: Option<&mut SuffixDrafter>,
        sampler: &mut dyn FnMut(&[f32], usize) -> u32,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<(GenerateOutcome, SpecStats), String> {
        if !cfg.enabled || !self.spec_verify_possible(cfg.verify) {
            return self.generate_inner(prompt, params, sampler, cfg.chip_argmax, on_token).map(|o| (o, SpecStats::default()));
        }
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() >= self.kv.capacity() {
            return Err(format!("prompt of {} tokens, context is {}", prompt.len(), self.kv.capacity()));
        }
        let common = self.history.iter().zip(prompt).take_while(|(a, b)| a == b).count();
        let reused = common.min(prompt.len() - 1);
        if reused > 0 {
            self.rewind(reused);
        } else {
            self.reset();
        }
        let t0 = Instant::now();
        let mut pending = Vec::new();
        let mut prefill_timing = StepTiming::default();
        // Same prefill as `generate_with` (whole-prompt when its switch is on), so spec off and on start
        // from the same KV and the same first hidden state.
        let whole = self.whole_prompt_prefill(&prompt[reused..], &mut pending, &mut prefill_timing)?;
        for chunk in prompt[reused + whole..].chunks(self.prefill_chunk.max(1)) {
            pending = self.step_batch(chunk)?;
            prefill_timing.add(&self.last_timing);
        }
        let prefill_ms = t0.elapsed().as_millis() as u64;
        self.last_timing = StepTiming::default();
        let t1 = Instant::now();
        let mut decode_timing = StepTiming::default();
        let mut stats = SpecStats::default();
        let mut ctx: Vec<u32> = prompt.to_vec();
        if let Some(d) = drafter.as_deref_mut() {
            d.begin(prompt);
        }
        let mut emitted = 0usize;
        let mut finish = FinishReason::Length;
        let mut step = 0usize;
        let mut pending_logits: Option<Vec<f32>> = None;
        // RC3: a one-list decode step's greedy token (chip argmax), when `cfg.chip_argmax`.
        let mut pending_token: Option<u32> = None;
        let (mut skip, mut backoff) = (0usize, 0usize);
        let mut be = cfg.break_even.then(|| BreakEven::new(cfg.suffix.cost.plain_ms as f64));
        // lm_head launch with its timing folded into decode_timing.
        macro_rules! lm {
            ($self:ident, $h:expr) => {{
                $self.last_timing = StepTiming::default();
                let lg = $self.logits($h)?;
                decode_timing.add(&$self.last_timing);
                lg
            }};
        }
        'outer: loop {
            if emitted >= params.max_new_tokens {
                break;
            }
            let t_first = match pending_token.take() {
                Some(t) => t,
                None => {
                    let logits = match pending_logits.take() {
                        Some(l) => l,
                        None => lm!(self, &pending),
                    };
                    sampler(&logits, step)
                }
            };
            step += 1;
            if params.stop_token_ids.contains(&t_first) {
                finish = FinishReason::Stop;
                break;
            }
            emitted += 1;
            ctx.push(t_first);
            if let Some(d) = drafter.as_deref_mut() {
                d.push(t_first);
            }
            if !on_token(t_first) {
                finish = FinishReason::Cancelled;
                break;
            }
            if emitted >= params.max_new_tokens || self.pos >= self.kv.capacity() {
                break;
            }
            // Room: inputs t_first + drafts must fit the cache; at most max_new - emitted more tokens
            // can be emitted after t_first's step, drafts included.
            let room = self.kv.capacity() - self.pos;
            let k = cfg
                .max_draft
                .min(room.saturating_sub(1))
                .min(params.max_new_tokens - emitted)
                .min(aien_omega_fp8::GEMM_MAX_ROWS / self.weights.config.top_k - 1);
            // W-L4e: the break-even pause (None) or a draft cap (a short probe after a pause).
            let cap = if skip > 0 { None } else { be.as_mut().map_or(Some(k), |b| b.draft_cap(k)) };
            let drafts = if skip > 0 || cap.is_none() {
                skip = skip.saturating_sub(1);
                stats.backoff_steps += 1;
                Vec::new()
            } else if let Some(d) = drafter.as_deref() {
                let k = cap.unwrap_or(k);
                let dr = match &be {
                    Some(b) => d.draft_with_cost(&ctx, k, &b.cost(&d.cfg.cost)),
                    None => d.draft(&ctx, k),
                };
                if dr.from_global && !dr.tokens.is_empty() {
                    stats.global_rounds += 1;
                }
                dr.tokens
            } else {
                find_draft(&ctx, cfg.max_ngram, cfg.min_ngram, cap.unwrap_or(k))
            };
            if drafts.is_empty() {
                stats.plain_steps += 1;
                let t_plain = Instant::now();
                self.last_timing = StepTiming::default();
                // RC3: the plain step is the one-list decode token when its switch is on (else `step`).
                pending = self.decode_step(t_first)?;
                decode_timing.add(&self.last_timing);
                // The next token's lm_head now (the loop top takes it), so the plain token's full cost is measured.
                match self.dgraph_take_greedy(cfg.chip_argmax) {
                    Some(t) => pending_token = Some(t),
                    None => pending_logits = Some(lm!(self, &pending)),
                }
                if let Some(b) = be.as_mut() {
                    b.observe_plain(t_plain.elapsed().as_secs_f64() * 1e3);
                }
                continue;
            }
            stats.rounds += 1;
            stats.drafted += drafts.len();
            let pos_before = self.pos;
            let mut inputs = Vec::with_capacity(drafts.len() + 1);
            inputs.push(t_first);
            inputs.extend_from_slice(&drafts);
            let t_verify = Instant::now();
            let hiddens = match cfg.verify {
                VerifyPath::Decode => self.step_verify(&inputs)?,
                VerifyPath::Exact => {
                    let mut prof = ExactProf::default();
                    self.step_verify_exact_inner(&inputs, &mut prof, cfg.mutant_numerics)?
                }
                VerifyPath::WholePrompt => {
                    stats.wp_rounds += 1;
                    self.step_verify_wp(&inputs)?
                }
            };
            let verify_ms = t_verify.elapsed().as_secs_f64() * 1e3;
            stats.verify_ns += t_verify.elapsed().as_nanos() as u64;
            decode_timing.add(&self.last_timing);
            let mut acc = 0usize;
            let mut mutant_pending = cfg.mutant_accept_extra;
            loop {
                let lg = lm!(self, &hiddens[acc]);
                let pred = sampler(&lg, step);
                let mut confirmed = draft_confirmed(drafts.get(acc).copied(), pred, &params.stop_token_ids);
                if !confirmed && mutant_pending && acc < drafts.len() && !params.stop_token_ids.contains(&drafts[acc]) {
                    // Negative control: take the wrong draft anyway.
                    mutant_pending = false;
                    confirmed = true;
                }
                if !confirmed {
                    // `pred` is the next token (correction, bonus, or a stop id): it becomes the pending state.
                    pending = hiddens[acc].clone();
                    pending_logits = Some(lg);
                    break;
                }
                step += 1;
                acc += 1;
                emitted += 1;
                stats.accepted += 1;
                let tok = drafts[acc - 1];
                ctx.push(tok);
                if let Some(d) = drafter.as_deref_mut() {
                    d.push(tok);
                }
                if !on_token(tok) {
                    finish = FinishReason::Cancelled;
                    self.rewind(pos_before + 1 + acc);
                    break 'outer;
                }
                if emitted >= params.max_new_tokens {
                    self.rewind(pos_before + 1 + acc);
                    break 'outer;
                }
            }
            // Keep t_first and the accepted drafts; drop the rejected inputs' KV.
            self.rewind(pos_before + 1 + acc);
            if let Some(b) = be.as_mut() {
                let predicted = cfg.suffix.cost.verify_ms(inputs.len()) as f64;
                b.observe_round(t_verify.elapsed().as_secs_f64() * 1e3, acc + 1, verify_ms, predicted);
            }
            if cfg.min_accept > 0 {
                if acc < cfg.min_accept {
                    backoff = (backoff * 2).clamp(1, 16);
                    skip = backoff;
                } else {
                    backoff = 0;
                }
            }
        }
        if let Some(d) = drafter {
            d.finish(&ctx);
        }
        stats.break_even = be;
        Ok((
            GenerateOutcome {
                emitted_tokens: emitted,
                finish,
                prefill_ms,
                decode_ms: t1.elapsed().as_millis() as u64,
                prompt_tokens: prompt.len(),
                reused_prompt_tokens: reused,
                decode_timing,
                prefill_timing,
            },
            stats,
        ))
    }
}

/// RC3: speculative decoding as the HTTP door runs it (`--spec on`): the suffix drafter with its global tree
/// (past requests, kept across requests), the fast exact verify, the break-even back-off. Greedy requests only;
/// plain steps are the one-list decode token when its switch is on.
#[derive(Clone, Debug)]
pub struct SpecServe {
    pub cfg: SpecConfig,
    pub drafter: SuffixDrafter,
}

impl SpecServe {
    /// The served configuration: tuning from the `AIEN_OWN30B_SPEC_*` environment (draft length, suffix and cost
    /// knobs, `AIEN_OWN30B_SPEC_BREAK_EVEN`), but always enabled, `Exact` verify, suffix drafter, no mutants.
    pub fn served() -> Self {
        let env = SpecConfig::from_env();
        let cfg = SpecConfig {
            enabled: true,
            verify: VerifyPath::Exact,
            source: DraftSource::Suffix,
            mutant_accept_extra: false,
            mutant_numerics: NumericsMutant::None,
            chip_argmax: false,
            ..env
        };
        let drafter = SuffixDrafter::new(cfg.suffix.clone());
        Self { cfg, drafter }
    }
}

impl Qwen3MoeOmegaEngine {
    /// Why `--spec on` cannot run on this load (`None`: it can). The exact verify needs FP8 attention and experts,
    /// the device router and omega's fused KV (the same objects the one-list decode uses).
    pub fn spec_serve_refusal(&self) -> Option<String> {
        if self.spec_verify_possible(VerifyPath::Exact) {
            return None;
        }
        Some(format!(
            "speculative decoding needs the exact verify: --weights fp8, the device router and omega's fused attention (this load: attention={}, device_routing={}, spec_possible={})",
            self.kv.path_name(),
            self.device_routing,
            self.spec_possible()
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

    #[test]
    fn numerics_mutant_names_parse_and_typos_are_refused() {
        assert_eq!(NumericsMutant::parse("act-fp8"), Ok(NumericsMutant::ActFp8));
        assert_eq!(NumericsMutant::parse("residual-bf16"), Ok(NumericsMutant::ResidualBf16));
        assert_eq!(NumericsMutant::parse("attn-bf16"), Ok(NumericsMutant::AttnBf16));
        assert_eq!(NumericsMutant::parse("none"), Ok(NumericsMutant::None));
        assert!(NumericsMutant::parse("act_fp8").is_err());
        assert_eq!(SpecConfig::default().mutant_numerics, NumericsMutant::None, "never on by default");
        // The W8A8 fake quantiser really moves values (else the control could not change anything).
        let mut v: Vec<f32> = (0..256).map(|j| (j as f32 * 0.37).sin() * 3.1).collect();
        let before = v.clone();
        fake_quant_rows(&mut v, 128, true);
        assert!(v.iter().zip(&before).any(|(a, b)| a != b));
        let mut w = before.clone();
        truncate_bf16(&mut w);
        assert!(w.iter().zip(&before).all(|(a, b)| (a - b).abs() <= b.abs() / 128.0));
    }

    #[test]
    fn find_draft_takes_the_most_recent_match_longest_suffix_first() {
        // suffix [7,8,9] occurs earlier at 0 and 5: the most recent one (5) wins, followed by 1,2,3.
        let ctx = [7, 8, 9, 4, 4, 7, 8, 9, 1, 2, 3, 6, 7, 8, 9];
        assert_eq!(find_draft(&ctx, 3, 1, 8), vec![1, 2, 3, 6, 7, 8, 9]);
        assert_eq!(find_draft(&ctx, 3, 1, 2), vec![1, 2]);
        // a 3-gram that never repeats falls back to the 2-gram, then the 1-gram
        let ctx = [1, 2, 3, 9, 5, 6, 2, 3];
        assert_eq!(find_draft(&ctx, 3, 1, 4), vec![9, 5, 6, 2]);
        let ctx = [1, 2, 3, 9, 5, 6, 8, 3];
        assert_eq!(find_draft(&ctx, 3, 1, 3), vec![9, 5, 6]);
        assert!(find_draft(&ctx, 3, 2, 3).is_empty(), "min_n 2: no 2-gram repeats");
        assert!(find_draft(&[1, 2, 3, 4], 3, 1, 4).is_empty());
        assert!(find_draft(&[5], 3, 1, 4).is_empty());
        assert!(find_draft(&[], 3, 1, 4).is_empty());
        assert!(find_draft(&ctx, 3, 1, 0).is_empty());
        // the continuation may run into the suffix itself but never past the end
        assert_eq!(find_draft(&[4, 4, 4], 1, 1, 8), vec![4]);
    }

    /// Cases for the acceptance rule: (drafts, preds, stops, accepted prefix).
    #[allow(clippy::type_complexity)]
    fn cases() -> Vec<(Vec<u32>, Vec<u32>, Vec<u32>, usize)> {
        vec![
            (vec![1, 2, 3], vec![1, 2, 3, 9], vec![], 3),
            (vec![1, 2, 3], vec![1, 2, 7, 9], vec![], 2),
            (vec![1, 2, 3], vec![5, 2, 3, 9], vec![], 0),
            (vec![1, 2, 3], vec![1, 7, 3, 9], vec![], 1),
            (vec![], vec![4], vec![], 0),
            (vec![1, 2], vec![1, 99, 3], vec![99], 1),
            (vec![1, 99], vec![1, 99, 3], vec![99], 1),
        ]
    }

    /// Ok when `f` returns the right prefix on every case and never accepts a wrong token.
    fn check_accept(f: impl Fn(&[u32], &[u32], &[u32]) -> usize) -> Result<(), String> {
        for (d, p, s, want) in cases() {
            let got = f(&d, &p, &s);
            if got != want {
                return Err(format!("drafts {d:?} preds {p:?} stops {s:?}: accepted {got}, want {want}"));
            }
        }
        Ok(())
    }

    #[test]
    fn acceptance_takes_exactly_the_matching_prefix_and_the_check_catches_an_extra_wrong_token() {
        check_accept(accepted_prefix).unwrap();
        // Mutant: accepts one extra (wrong) token after the first mismatch.
        let mutant = |d: &[u32], p: &[u32], s: &[u32]| (accepted_prefix(d, p, s) + 1).min(d.len());
        let err = check_accept(mutant).expect_err("the extra-token mutant must be caught");
        assert!(err.contains("accepted"), "{err}");
        // Mutant: ignores stop ids.
        let no_stop = |d: &[u32], p: &[u32], _s: &[u32]| accepted_prefix(d, p, &[]);
        assert!(check_accept(no_stop).is_err());
    }

    fn argmax(lg: &[f32]) -> u32 {
        lg.iter().enumerate().max_by(|a, b| a.1.partial_cmp(b.1).unwrap()).unwrap().0 as u32
    }

    fn engine(device: bool) -> Qwen3MoeOmegaEngine {
        let c = tiny_config();
        let mut src = MemorySource::for_config(&c);
        let up = RecordingUploader { device_fakes: device, gemm_fakes: true, ..Default::default() };
        let paths = if device { DevicePaths::default() } else { DevicePaths { router: PathChoice::Host, lm_head: PathChoice::Host } };
        let (w, _) = load_qwen3_moe_omega_from_paths(&c, &mut src, &up, WeightFormat::Fp8All, paths, &mut |_| {}).unwrap();
        Qwen3MoeOmegaEngine::new(w, Box::new(HostKv::new(&c, 160)))
    }

    fn run(eng: &mut Qwen3MoeOmegaEngine, prompt: &[u32], new: usize, cfg: &SpecConfig, stops: Vec<u32>) -> (Vec<u32>, SpecStats, FinishReason) {
        let p = GenerateParams::greedy(new, stops);
        let mut got = Vec::new();
        let (o, st) = eng
            .generate_spec_with(prompt, &p, cfg, &mut |lg, _| argmax(lg), &mut |t| {
                got.push(t);
                true
            })
            .unwrap();
        assert_eq!(o.emitted_tokens, got.len());
        (got, st, o.finish)
    }

    /// Engine-level identity on the host fakes (where batched and per-token maths are bit-equal):
    /// spec-on tokens, and the stored history, equal spec-off, with real drafting, real rejections.
    #[test]
    fn spec_tokens_equal_plain_decode_with_real_acceptances_and_rejections() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        for device in [true, false] {
            let mut plain = engine(device);
            let want = run(&mut plain, &prompt, 60, &SpecConfig::default(), vec![]).0;
            assert_eq!(want.len(), 60);
            let on = SpecConfig { enabled: true, ..Default::default() };
            let mut spec = engine(device);
            assert!(spec.spec_possible());
            let (got, st, fin) = run(&mut spec, &prompt, 60, &on, vec![]);
            assert_eq!(got, want, "device={device}: spec output differs from plain decode");
            assert!(matches!(fin, FinishReason::Length));
            assert!(st.rounds > 0 && st.accepted > 0, "no speculation happened: {st:?}");
            assert!(st.drafted > st.accepted, "no rejection exercised: {st:?}");
            assert_eq!(spec.history(), &plain.history()[..spec.history().len()].to_vec()[..], "history is a faithful prefix");
            // cache state stays valid: a second generation reusing the prefix still matches
            let again = run(&mut spec, &prompt, 20, &on, vec![]).0;
            assert_eq!(again, want[..20].to_vec());
            // other draft widths
            for k in [1usize, 3, 8] {
                let cfg = SpecConfig { enabled: true, max_draft: k, ..Default::default() };
                let mut e = engine(device);
                assert_eq!(run(&mut e, &prompt, 40, &cfg, vec![]).0, want[..40].to_vec(), "k={k}");
            }
        }
    }

    /// W-L4c: the suffix drafter (request-local and with a caller-owned global tree that has seen
    /// an earlier request) changes only which tokens are proposed: output equals plain decode.
    #[test]
    fn suffix_drafter_keeps_plain_decode_tokens_and_uses_the_global_tree() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        // Optimistic costs so the fake model's short runs still draft.
        let suffix = suffix::SuffixConfig { cost: suffix::CostModel { verify_base_ms: 1.0, verify_row_ms: 0.1, lm_ms: 0.0, plain_ms: 1.0 }, ..Default::default() };
        let on = SpecConfig { enabled: true, max_draft: 12, source: DraftSource::Suffix, suffix: suffix.clone(), ..Default::default() };
        for device in [true, false] {
            let mut plain = engine(device);
            let want = run(&mut plain, &prompt, 60, &SpecConfig::default(), vec![]).0;
            let mut spec = engine(device);
            let (got, st, _) = run(&mut spec, &prompt, 60, &on, vec![]);
            assert_eq!(got, want, "device={device}: suffix-drafted output differs from plain decode");
            assert!(st.rounds > 0 && st.accepted > 0, "no speculation happened: {st:?}");
            // Caller-owned drafter: request 1 fills the global tree, request 2 (a different prompt
            // that runs into the same text) drafts from it.
            let mut d = suffix::SuffixDrafter::new(suffix.clone());
            let p = GenerateParams::greedy(40, vec![]);
            let mut e = engine(device);
            let mut first = Vec::new();
            e.generate_spec_drafter(&prompt, &p, &on, Some(&mut d), &mut |lg, _| argmax(lg), &mut |t| {
                first.push(t);
                true
            })
            .unwrap();
            assert_eq!(first, want[..40].to_vec());
            let mut e2 = engine(device);
            let prompt2: Vec<u32> = [1u32, 2].iter().copied().chain(prompt.iter().copied()).collect();
            let want2 = run(&mut engine(device), &prompt2, 40, &SpecConfig::default(), vec![]).0;
            let mut second = Vec::new();
            let (_, st2) = e2
                .generate_spec_drafter(&prompt2, &p, &on, Some(&mut d), &mut |lg, _| argmax(lg), &mut |t| {
                    second.push(t);
                    true
                })
                .unwrap();
            assert_eq!(second, want2, "device={device}: second request");
            assert!(st2.global_rounds > 0, "the global tree was never used: {st2:?}");
        }
    }

    #[test]
    fn spec_off_and_unsupported_engines_run_plain_decode() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42];
        let mut a = engine(true);
        let (t, st, _) = run(&mut a, &prompt, 10, &SpecConfig::default(), vec![]);
        assert_eq!((st.rounds, st.drafted), (0, 0));
        assert_eq!(t.len(), 10);
        std::env::remove_var(SPEC_ENV);
        assert!(!SpecConfig::from_env().enabled, "default is OFF");
    }

    #[test]
    fn back_off_skips_drafting_after_poor_rounds_and_keeps_the_tokens() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        for device in [true, false] {
            let want = run(&mut engine(device), &prompt, 60, &SpecConfig::default(), vec![]).0;
            let free = run(&mut engine(device), &prompt, 60, &SpecConfig { enabled: true, ..Default::default() }, vec![]).1;
            // min_accept above any possible acceptance: every round is "poor", so back-off must kick in.
            let cfg = SpecConfig { enabled: true, min_accept: 9, ..Default::default() };
            let (got, st, _) = run(&mut engine(device), &prompt, 60, &cfg, vec![]);
            assert_eq!(got, want, "device={device}: back-off changed the tokens");
            assert!(st.backoff_steps > 0 && st.rounds > 0, "{st:?}");
            assert!(st.rounds < free.rounds, "back-off did not reduce rounds: {st:?} vs {free:?}");
        }
        std::env::set_var("AIEN_OWN30B_SPEC_MIN_ACCEPT", "2");
        assert_eq!(SpecConfig::from_env().min_accept, 2);
        std::env::remove_var("AIEN_OWN30B_SPEC_MIN_ACCEPT");
        assert_eq!(SpecConfig::from_env().min_accept, 0, "default: no back-off");
    }

    #[test]
    fn whole_prompt_verify_without_its_switch_runs_plain_decode() {
        // W-L4b: verify=WholePrompt needs set_whole_prompt(true); without it the engine must not
        // half-run a speculative round, it decodes plainly (stats zero), same tokens as spec off.
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42];
        let mut a = engine(true);
        assert!(!a.spec_verify_possible(VerifyPath::WholePrompt));
        assert!(a.step_verify_wp(&[5, 17]).is_err());
        let (plain, _, _) = run(&mut a, &prompt, 10, &SpecConfig::default(), vec![]);
        let cfg = SpecConfig { enabled: true, verify: VerifyPath::WholePrompt, ..SpecConfig::default() };
        let (t, st, _) = run(&mut a, &prompt, 10, &cfg, vec![]);
        assert_eq!((st.rounds, st.wp_rounds, st.drafted), (0, 0, 0));
        assert_eq!(t, plain);
        std::env::set_var("AIEN_OWN30B_SPEC_VERIFY", "wp");
        assert_eq!(SpecConfig::from_env().verify, VerifyPath::WholePrompt);
        std::env::remove_var("AIEN_OWN30B_SPEC_VERIFY");
        assert_eq!(SpecConfig::from_env().verify, VerifyPath::Decode, "default verify is the decode-rounding path");
    }

    #[test]
    fn stop_id_inside_a_draft_and_the_token_cap_end_generation_exactly_like_plain_decode() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        let mut plain = engine(true);
        let free = run(&mut plain, &prompt, 40, &SpecConfig::default(), vec![]).0;
        // Stop on a token that appears mid-stream; the token cap case uses odd caps.
        for &idx in &[3usize, 10, 25] {
            let stop = free[idx];
            let first = free.iter().position(|&t| t == stop).unwrap();
            let mut p = engine(true);
            let want = run(&mut p, &prompt, 40, &SpecConfig::default(), vec![stop]);
            let mut s = engine(true);
            let got = run(&mut s, &prompt, 40, &SpecConfig { enabled: true, ..Default::default() }, vec![stop]);
            assert_eq!(got.0, want.0);
            assert_eq!(got.0.len(), first);
            assert!(matches!((&got.2, &want.2), (FinishReason::Stop, FinishReason::Stop)));
        }
        for cap in [1usize, 2, 5, 9, 13] {
            let mut s = engine(true);
            let got = run(&mut s, &prompt, cap, &SpecConfig { enabled: true, ..Default::default() }, vec![]);
            assert_eq!(got.0, free[..cap].to_vec(), "cap {cap}");
        }
    }

    /// W-L4e cut 2: simulated generation of `n` tokens with measured-like costs. Each draft token is accepted
    /// with probability `p` (fixed LCG, so the run is repeatable) until the first rejection. Returns (total ms,
    /// rounds). `be` None = always draft (no back-off).
    fn simulate(mut be: Option<BreakEven>, n: usize, p: f64, draft_len: usize, plain_ms: f64) -> (f64, usize) {
        let mut seed = 12345u64;
        let mut rnd = || {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 53) as f64
        };
        let (mut t, mut done, mut rounds) = (0.0, 0usize, 0usize);
        while done < n {
            let cap = match be.as_mut() {
                Some(b) => b.draft_cap(draft_len),
                None => Some(draft_len),
            };
            match cap {
                None | Some(0) => {
                    t += plain_ms;
                    done += 1;
                    if let Some(b) = be.as_mut() {
                        b.observe_plain(plain_ms);
                    }
                }
                Some(k) => {
                    let mut acc = 0;
                    while acc < k && rnd() < p {
                        acc += 1;
                    }
                    // Exact verify on the chip (W-L4d gate): ~30 ms at 2 rows, ~3 ms per extra row; lm_head 2.9 ms per evaluated row.
                    let verify = 30.0 + 3.0 * (k as f64 - 1.0);
                    let ms = verify + 2.9 * (acc as f64 + 1.0);
                    t += ms;
                    done += acc + 1;
                    rounds += 1;
                    if let Some(b) = be.as_mut() {
                        b.observe_round(ms, acc + 1, verify, verify);
                    }
                }
            }
        }
        (t, rounds)
    }

    #[test]
    fn break_even_stops_drafting_where_it_loses_and_keeps_it_where_it_pays() {
        // p4-long on the full model (W-L4d gate): acceptance 0.15, plain decode 27.8 ms/token.
        let plain = 27.8;
        let n = 64;
        let (always, r_always) = simulate(None, n, 0.15, 2, plain);
        let (gov, r_gov) = simulate(Some(BreakEven::new(28.3)), n, 0.15, 2, plain);
        let off = n as f64 * plain;
        assert!(always > off * 1.05, "without back-off p4-like text must lose ({always:.0} vs {off:.0} ms)");
        assert!(gov <= off * 1.02, "with back-off it must not lose more than 2% ({gov:.0} vs {off:.0} ms)");
        assert!(r_gov < r_always / 3, "back-off must cut the rounds: {r_gov} vs {r_always}");
        // With the one-list decode (plain ~22.5 ms) the same text pauses too.
        let (gov_g, _) = simulate(Some(BreakEven::new(28.3)), n, 0.15, 2, 22.5);
        // Every unproven round costs ~10 ms against a 22.5 ms plain token: at most 3% over 64 tokens, and it
        // amortises (256 tokens: at most 1.5%).
        assert!(gov_g <= n as f64 * 22.5 * 1.03, "graph plain cost: {gov_g:.0}");
        let (gov_g256, _) = simulate(Some(BreakEven::new(28.3)), 256, 0.15, 2, 22.5);
        assert!(gov_g256 <= 256.0 * 22.5 * 1.015, "graph plain cost, 256 tokens: {gov_g256:.0}");
        // code/JSON (acceptance 0.86, long drafts): keeps drafting and wins.
        let (json, r_json) = simulate(Some(BreakEven::new(28.3)), 256, 0.86, 8, plain);
        assert!(json < 256.0 * plain * 0.8, "code/JSON must keep its gain: {json:.0} vs {:.0}", 256.0 * plain);
        assert!(r_json > 30, "{r_json}");
        // A switch from repetitive to fresh text and back: the probes find the good stretch again.
        let mut b = BreakEven::new(28.3);
        for _ in 0..6 {
            if b.draft_cap(8).is_some() {
                b.observe_round(45.0, 1, 33.0, 33.0);
            } else {
                b.observe_plain(plain);
            }
        }
        assert!(!b.pays());
        let mut drafted_again = false;
        for _ in 0..80 {
            match b.draft_cap(8) {
                Some(k) => {
                    assert!(k <= 8);
                    b.observe_round(50.0, k + 1, 36.0, 36.0);
                    drafted_again = true;
                }
                None => b.observe_plain(plain),
            }
        }
        assert!(drafted_again && b.pays(), "a paying stretch must turn drafting back on: {b:?}");
        assert_eq!(b.draft_cap(8), Some(8), "full drafts after a paying probe");
    }

    #[test]
    fn break_even_on_keeps_plain_decode_tokens() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        for device in [true, false] {
            let want = run(&mut engine(device), &prompt, 60, &SpecConfig::default(), vec![]).0;
            for source in [DraftSource::PromptLookup, DraftSource::Suffix] {
                let cfg = SpecConfig { enabled: true, break_even: true, source, ..Default::default() };
                let (got, st, _) = run(&mut engine(device), &prompt, 60, &cfg, vec![]);
                assert_eq!(got, want, "device={device} {source:?}: break-even changed the tokens");
                let b = st.break_even.expect("break-even state reported");
                assert!(b.plain_seen + b.rounds_seen > 0, "{b:?}");
            }
        }
        std::env::remove_var("AIEN_OWN30B_SPEC_BREAK_EVEN");
        assert!(SpecConfig::from_env().break_even, "from_env: on by default");
        std::env::set_var("AIEN_OWN30B_SPEC_BREAK_EVEN", "0");
        assert!(!SpecConfig::from_env().break_even);
        std::env::remove_var("AIEN_OWN30B_SPEC_BREAK_EVEN");
    }

    /// RC3: the door's switch. On is refused on a load without the exact verify (host fakes: no fused KV) and
    /// stays off; off runs plain decode (the env is not touched here: other tests read it concurrently).
    #[test]
    fn served_spec_switch_refuses_a_load_without_exact_verify_and_stays_off() {
        let report = LoadReport { resident_bytes: 0, format: WeightFormat::Fp8All, host_bytes: 0, peak_rss_bytes: None, peak_rss_before_bytes: None, seconds: 0.0, window_api: false };
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        let want = run(&mut engine(true), &prompt, 20, &SpecConfig::default(), vec![]).0;
        let shared = OmegaEngine::from_engine(engine(true), report);
        assert!(!shared.spec_on());
        let err = shared.set_spec(true).unwrap_err();
        assert!(err.contains("exact verify"), "{err}");
        assert!(!shared.spec_on() && shared.spec_config().is_none(), "a refused on leaves it off");
        shared.set_spec(false).unwrap();
        let mut got = Vec::new();
        let p = GenerateParams::greedy(20, vec![]);
        shared.generate_with(&prompt, &p, &mut |lg, _| argmax(lg), &mut |t| {
            got.push(t);
            true
        })
        .unwrap();
        assert_eq!(got, want);
        assert!(shared.take_last_spec_stats().is_none(), "door said off: plain decode");
        // The served configuration: exact verify, suffix drafter, break-even on, no mutants.
        let s = SpecServe::served();
        assert!(s.cfg.enabled && s.cfg.verify == VerifyPath::Exact && s.cfg.source == DraftSource::Suffix);
        assert!(s.cfg.break_even && !s.cfg.mutant_accept_extra && s.cfg.mutant_numerics == NumericsMutant::None);
    }

    #[test]
    fn extra_accept_mutant_changes_the_output_so_the_identity_check_catches_it() {
        let prompt = [5u32, 17, 42, 3, 9, 60, 5, 17, 42, 3, 9, 60, 5, 17, 42];
        let mut plain = engine(true);
        let want = run(&mut plain, &prompt, 60, &SpecConfig::default(), vec![]).0;
        let mut m = engine(true);
        let cfg = SpecConfig { enabled: true, mutant_accept_extra: true, ..Default::default() };
        let got = run(&mut m, &prompt, 60, &cfg, vec![]).0;
        assert_ne!(got, want, "a wrong extra accepted token must change the output");
    }
}
