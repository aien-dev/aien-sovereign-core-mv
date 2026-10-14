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
    /// Which pass checks the drafts (W-L4b). See [`VerifyPath`].
    pub verify: VerifyPath,
    /// W-L4b back-off: a round that accepts fewer than `min_accept` drafts makes the next rounds
    /// skip drafting (plain steps) for 1, 2, 4 .. 16 steps; a good round resets it. 0 = off. Only
    /// WHICH steps get a draft changes, never how a token is chosen.
    pub min_accept: usize,
    /// W-L4c: draft source and the suffix drafter's settings.
    pub source: DraftSource,
    pub suffix: SuffixConfig,
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
}

pub const SPEC_ENV: &str = "AIEN_OWN30B_SPEC";

impl Default for SpecConfig {
    fn default() -> Self {
        Self { enabled: false, max_draft: 8, max_ngram: 3, min_ngram: 1, mutant_accept_extra: false, verify: VerifyPath::Decode, min_accept: 0, source: DraftSource::PromptLookup, suffix: SuffixConfig::default() }
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
            min_accept: num("AIEN_OWN30B_SPEC_MIN_ACCEPT", d.min_accept),
            // AIEN_OWN30B_SPEC_VERIFY=wp: verify through the whole-prompt driver (W-L4b).
            verify: match get("AIEN_OWN30B_SPEC_VERIFY").as_deref().map(str::trim) {
                Some("wp") | Some("whole-prompt") => VerifyPath::WholePrompt,
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
            return self.generate_with(prompt, params, sampler, on_token).map(|o| (o, SpecStats::default()));
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
        let (mut skip, mut backoff) = (0usize, 0usize);
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
            let logits = match pending_logits.take() {
                Some(l) => l,
                None => lm!(self, &pending),
            };
            let t_first = sampler(&logits, step);
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
            let drafts = if skip > 0 {
                skip -= 1;
                stats.backoff_steps += 1;
                Vec::new()
            } else if let Some(d) = drafter.as_deref() {
                let dr = d.draft(&ctx, k);
                if dr.from_global && !dr.tokens.is_empty() {
                    stats.global_rounds += 1;
                }
                dr.tokens
            } else {
                find_draft(&ctx, cfg.max_ngram, cfg.min_ngram, k)
            };
            if drafts.is_empty() {
                stats.plain_steps += 1;
                self.last_timing = StepTiming::default();
                pending = self.step(t_first)?;
                decode_timing.add(&self.last_timing);
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
                VerifyPath::WholePrompt => {
                    stats.wp_rounds += 1;
                    self.step_verify_wp(&inputs)?
                }
            };
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

#[cfg(test)]
mod tests {
    use super::super::test_support::*;
    use super::*;

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
