//! The generation controller and the forward-pass seam (ADR 0004 sections 2.2 to 2.5).
//!
//! [`GenerationController`] owns the sequence ledger (history, and so the position), prefix
//! reuse, rewind, KV export and restore, sampling, constraints, termination and streaming. It is
//! written once and drives any model through [`ForwardPass`]. A forward pass owns its weights,
//! its maths and its KV cache, and lends the controller only the model-independent half of the
//! cache ([`KvStore`]).
//!
//! Step 2 of the ADR's migration: this module is pure addition. No production caller uses it
//! yet; the 30B engine keeps its own loop until step 4. The host fake `generation::echo_chain` (feature
//! `host-test-engine`) is the only forward pass so far.

use std::sync::Mutex;
use std::time::Instant;

use crate::qwen3_moe_omega::{
    sample_next, FinishReason, GenerateOutcome, GenerateParams, KvStore, StepTiming,
};
use crate::tensor::sample_argmax;

#[cfg(any(test, feature = "host-test-engine"))]
pub mod echo_chain;

#[cfg(test)]
pub(crate) mod tests;

/// What the controller wants from a head.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Want {
    /// The plain argmax: the model may answer with a token it already holds.
    Greedy,
    /// The full logits, always.
    Logits,
}

/// One head's answer.
#[derive(Debug, PartialEq)]
pub enum Next<'a> {
    /// The argmax the model already held (only for [`Want::Greedy`], under the
    /// [`sample_argmax`] tie rule: lowest index of the maximum).
    Token(u32),
    /// `vocab()` logits.
    Logits(&'a [f32]),
}

/// Every setting of a forward pass that can change a K/V byte, as named entries, plus its kind.
/// Built by exhaustive destructuring inside the adapter (no `..`), so a new switch cannot be
/// forgotten.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct KvIdentityInputs {
    pub kind: &'static str,
    pub entries: Vec<(&'static str, String)>,
}

/// One model implementation behind the seam (ADR 0004 section 2.2). Invariants I1 to I6 of the
/// ADR hold for every adapter; the contract suite in this module's tests checks them.
pub trait ForwardPass: Send {
    /// "qwen3-moe-omega", "qwen3-dense-omega", "echo-chain".
    fn kind(&self) -> &'static str;
    fn vocab(&self) -> usize;
    /// The cache, lent as its model-independent half. Safe to truncate, export and import
    /// between forward calls; the model holds no pending state across them.
    fn kv(&mut self) -> &mut dyn KvStore;
    /// Draft tokens one `decode` call may verify at once. 1 until speculative decode lands.
    fn max_decode_batch(&self) -> usize {
        1
    }
    /// Prompt phase. Writes K/V for positions `pos..pos + tokens.len()` in every layer and
    /// leaves ONE head (head 0, the last token). The model chooses its prefill strategy.
    fn prefill(
        &mut self,
        pos: usize,
        tokens: &[u32],
        timing: &mut StepTiming,
    ) -> Result<(), String>;
    /// Generation phase. `1 <= tokens.len() <= max_decode_batch()`. Writes `tokens.len()`
    /// positions and leaves that many heads; head `j` predicts the token after `tokens[j]`.
    fn decode(&mut self, pos: usize, tokens: &[u32], timing: &mut StepTiming)
        -> Result<(), String>;
    /// Reads head `head` once. [`Want::Greedy`]: `Token` only if the model already holds the
    /// argmax, else `Logits` (the controller then takes the argmax itself). [`Want::Logits`]:
    /// always `Logits`. A host output head adds its own time to `timing`.
    fn next(
        &mut self,
        head: usize,
        want: Want,
        timing: &mut StepTiming,
    ) -> Result<Next<'_>, String>;
    /// Sparse output head: the logits of head `head` for `rows` only, into
    /// `out[..rows.len()]`. The default is [`gather_rows`] (read the full logits, then pick the
    /// rows). Never called when no constraint is active.
    fn next_rows(
        &mut self,
        head: usize,
        rows: &[u32],
        out: &mut [f32],
        timing: &mut StepTiming,
    ) -> Result<(), String> {
        gather_rows(self, head, rows, out, timing)
    }
    /// Forgets every unread head. Called by the controller on rewind, a forced token, an error
    /// and at the start of every generation.
    fn clear_pending(&mut self);
    /// The sequence starts over: forget heads and any per-sequence diagnostics. Called by the
    /// controller's `reset` (and so by `restore_kv`).
    fn begin_sequence(&mut self) {
        self.clear_pending()
    }
    /// The K/V identity entries, or `Err(reason)` when K/V from this run must never be stored.
    fn kv_identity(&self) -> Result<KvIdentityInputs, String>;
    /// Migration only (ADR 0004 steps 3 and 4; step 5 deletes it): the controller shows the
    /// model its ledger before every forward call, so a model that still keeps its own copy of
    /// the history (the 30B's mirror) can take the tokens of a restore or import it did not see
    /// (I1). Default: nothing.
    fn mirror_ledger(&mut self, _history: &[u32]) {}
}

/// The default sparse output head: `next(head, Logits)`, then the rows gathered into
/// `out[..rows.len()]`. Public so an adapter that instruments `next_rows` can still run it.
pub fn gather_rows<M: ForwardPass + ?Sized>(
    model: &mut M,
    head: usize,
    rows: &[u32],
    out: &mut [f32],
    timing: &mut StepTiming,
) -> Result<(), String> {
    if out.len() < rows.len() {
        return Err(format!(
            "next_rows: {} rows into a buffer of {}",
            rows.len(),
            out.len()
        ));
    }
    match model.next(head, Want::Logits, timing)? {
        Next::Logits(l) => {
            for (o, &r) in out.iter_mut().zip(rows) {
                *o = *l.get(r as usize).ok_or_else(|| {
                    format!(
                        "next_rows: row {r} is outside the vocabulary of {}",
                        l.len()
                    )
                })?;
            }
            Ok(())
        }
        Next::Token(_) => Err("next_rows: the model answered Want::Logits with a token".into()),
    }
}

/// Picks the next token from logits.
pub trait Sampler {
    /// `false`: the plain argmax; on an unconstrained step `pick` is never called and the chip's
    /// own argmax may stand in. Under an `Allow` rule `pick` is always called, on the masked
    /// buffer.
    fn wants_logits(&self) -> bool;
    fn pick(&mut self, logits: &[f32], step: usize) -> u32;
}

/// The door's sampling rule ([`sample_next`]) as a [`Sampler`]: what today's `generate`
/// convenience runs. `wants_logits()` is `temperature > 0.001`, today's `argmax_sampler` flag
/// inverted.
#[derive(Debug, Clone)]
pub struct SampleNext(pub GenerateParams);

impl Sampler for SampleNext {
    fn wants_logits(&self) -> bool {
        self.0.temperature > 0.001
    }
    fn pick(&mut self, logits: &[f32], step: usize) -> u32 {
        sample_next(logits, &self.0, step)
    }
}

/// What a constraint allows at one decode step.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StepRule {
    /// No constraint this step: the unconstrained path, byte for byte.
    Any,
    /// The legal token ids (strictly ascending, non-empty); the sampler sees only these.
    Allow(Vec<u32>),
    /// Exactly one legal token: emitted without reading a head.
    Force(u32),
    /// The output is complete: stop with [`FinishReason::Constraint`].
    Finish,
}

/// A per-step rule source applied between the head and the sampler (ADR 0004 section 2.3).
pub trait Constraint {
    /// Called once per step before the sampler, with the prompt and the tokens generated so far.
    fn rule(&mut self, prompt: &[u32], generated: &[u32]) -> Result<StepRule, String>;
}

/// The one generation loop for every forward pass (ADR 0004 section 2.4). Between calls every
/// layer of the model's cache holds exactly positions `0..position()`.
pub struct GenerationController<M: ForwardPass> {
    /// Owns the cache; declared first, drops first.
    model: M,
    /// Positions `0..history.len()` are stored in every layer.
    history: Vec<u32>,
    /// The cache's capacity, read once: the cache is fixed for the adapter's lifetime (I6).
    capacity: usize,
    /// Sparse-head output, `rows.len()` long; allocated on the first `Allow`.
    rows_buf: Vec<f32>,
    /// Vocab-sized, negative infinity except the allowed rows; allocated on the first `Allow`.
    masked: Vec<f32>,
}

impl<M: ForwardPass> GenerationController<M> {
    /// Takes the model with an empty sequence (its cache is emptied).
    pub fn new(mut model: M) -> Self {
        let capacity = model.kv().capacity();
        let mut c = Self {
            model,
            history: Vec::new(),
            capacity,
            rows_buf: Vec::new(),
            masked: Vec::new(),
        };
        c.reset();
        c
    }

    pub fn max_context_tokens(&self) -> usize {
        self.capacity
    }

    pub fn position(&self) -> usize {
        self.history.len()
    }

    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Empties the sequence: the model begins a new one, the cache and the ledger are emptied.
    pub fn reset(&mut self) {
        self.model.begin_sequence();
        self.model.kv().truncate(0);
        self.history.clear();
    }

    /// Keeps positions `0..len` (forgets unread heads).
    pub fn rewind(&mut self, len: usize) {
        self.model.clear_pending();
        self.model.kv().truncate(len);
        self.history.truncate(len);
    }

    /// Tokens `generate` would keep from the stored history for `prompt`: the longest common
    /// prefix, capped at `prompt.len() - 1` (the last prompt token is always prefilled).
    pub fn reusable_prefix(&self, prompt: &[u32]) -> usize {
        let common = self
            .history
            .iter()
            .zip(prompt)
            .take_while(|(a, b)| a == b)
            .count();
        common.min(prompt.len().saturating_sub(1))
    }

    /// The K/V of stored positions `start..start + len` as the `HostPagedPoolV1` payload
    /// (`start` a multiple of the block size, `start + len <= position()`).
    pub fn export_kv(&mut self, start: usize, len: usize) -> Result<Vec<u8>, String> {
        if start + len > self.history.len() {
            return Err(format!(
                "kv export {start}+{len}: only {} positions stored",
                self.history.len()
            ));
        }
        let kv = self.model.kv();
        kv.export_blocks(start, len).unwrap_or_else(|| {
            Err(format!(
                "the {} attention cache cannot export its KV",
                kv.path_name()
            ))
        })
    }

    /// Replaces the sequence by `tokens` with the K/V of `payload` (positions from 0). On
    /// success the next `generate` reuses them by the usual prefix rule. On any error the
    /// sequence is reset (empty ledger, empty cache).
    pub fn restore_kv(&mut self, tokens: &[u32], payload: &[u8]) -> Result<(), String> {
        self.reset();
        if tokens.is_empty() || tokens.len() >= self.capacity {
            return Err(format!(
                "kv restore of {} tokens, context is {}",
                tokens.len(),
                self.capacity
            ));
        }
        let kv = self.model.kv();
        let r = kv
            .import_blocks(0, tokens.len(), payload)
            .unwrap_or_else(|| {
                Err(format!(
                    "the {} attention cache cannot import KV",
                    kv.path_name()
                ))
            });
        match r {
            Ok(()) => {
                self.history = tokens.to_vec();
                Ok(())
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }

    pub fn kv_identity(&self) -> Result<KvIdentityInputs, String> {
        self.model.kv_identity()
    }

    /// Generates from `prompt`: prefix reuse, one prefill of the un-reused suffix, then one
    /// token at a time. `on_token` sees each emitted token and may return `false` to cancel.
    /// Stop ids are consumed, never emitted. With `constraint` `None` (or a constraint that
    /// always answers [`StepRule::Any`]) this is today's loop, byte for byte.
    pub fn generate(
        &mut self,
        prompt: &[u32],
        params: &GenerateParams,
        sampler: &mut dyn Sampler,
        mut constraint: Option<&mut dyn Constraint>,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, String> {
        self.model.clear_pending();
        if prompt.is_empty() {
            return Err("empty prompt".into());
        }
        if prompt.len() >= self.capacity {
            return Err(format!(
                "prompt of {} tokens, context is {}",
                prompt.len(),
                self.capacity
            ));
        }
        let reused = self.reusable_prefix(prompt);
        if reused > 0 {
            self.rewind(reused);
        } else {
            self.reset();
        }
        let t0 = Instant::now();
        let mut prefill_timing = StepTiming::default();
        let suffix = &prompt[reused..];
        // I5: the whole un-reused suffix in one call; the model slices it as it must.
        self.forward(suffix, |m, pos| m.prefill(pos, suffix, &mut prefill_timing))?;
        let prefill_ms = t0.elapsed().as_millis() as u64;
        let t1 = Instant::now();
        let mut decode_timing = StepTiming::default();
        let mut emitted = 0usize;
        let mut finish = FinishReason::Length;
        let mut step = 0usize;
        while emitted < params.max_new_tokens {
            let next = match self.choose(prompt, sampler, &mut constraint, step, &mut decode_timing)
            {
                Ok(Some(t)) => t,
                Ok(None) => {
                    finish = FinishReason::Constraint;
                    break;
                }
                Err(e) => {
                    self.rollback();
                    return Err(e);
                }
            };
            step += 1;
            if params.stop_token_ids.contains(&next) {
                // Consumed, never emitted, never stepped.
                finish = FinishReason::Stop;
                break;
            }
            emitted += 1;
            if !on_token(next) {
                // Emitted, not stepped.
                finish = FinishReason::Cancelled;
                break;
            }
            if emitted >= params.max_new_tokens || self.position() >= self.capacity {
                break;
            }
            let tok = [next];
            self.forward(&tok, |m, pos| m.decode(pos, &tok, &mut decode_timing))?;
        }
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms,
            decode_ms: t1.elapsed().as_millis() as u64,
            prompt_tokens: prompt.len(),
            reused_prompt_tokens: reused,
            decode_timing,
            prefill_timing,
        })
    }

    pub fn model(&self) -> &M {
        &self.model
    }

    /// The side door (traces, mutants, switches). Anything that touches the cache through it
    /// must be followed by [`Self::reset`].
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    /// One step's token: `Some(t)` to emit (or stop on), `None` when a constraint finished.
    fn choose(
        &mut self,
        prompt: &[u32],
        sampler: &mut dyn Sampler,
        constraint: &mut Option<&mut dyn Constraint>,
        step: usize,
        timing: &mut StepTiming,
    ) -> Result<Option<u32>, String> {
        let rule = match constraint.as_deref_mut() {
            Some(c) => c.rule(prompt, &self.history[prompt.len()..])?,
            None => StepRule::Any,
        };
        let vocab = self.model.vocab();
        Ok(Some(match rule {
            StepRule::Finish => return Ok(None),
            StepRule::Force(t) => {
                if t as usize >= vocab {
                    return Err(format!(
                        "constraint forced token {t}, vocabulary is {vocab}"
                    ));
                }
                self.model.clear_pending();
                t
            }
            StepRule::Allow(rows) => {
                check_allowed(&rows, vocab)?;
                self.rows_buf.resize(rows.len(), 0.0);
                self.model.next_rows(0, &rows, &mut self.rows_buf, timing)?;
                // The constrained argmax is the argmax over the allowed set: never the chip's.
                self.masked.clear();
                self.masked.resize(vocab, f32::NEG_INFINITY);
                for (&r, &v) in rows.iter().zip(&self.rows_buf) {
                    self.masked[r as usize] = v;
                }
                let t = sampler.pick(&self.masked, step);
                // All-negative-infinity or NaN logits make an argmax fall back to token 0;
                // whatever the sampler did, a token outside the set never leaves this step.
                if rows.binary_search(&t).is_err() {
                    return Err(format!(
                        "the sampler picked token {t}, outside the {} allowed",
                        rows.len()
                    ));
                }
                t
            }
            StepRule::Any => {
                let wants = sampler.wants_logits();
                let want = if wants { Want::Logits } else { Want::Greedy };
                match self.model.next(0, want, timing)? {
                    Next::Token(t) if !wants => t,
                    Next::Token(_) => {
                        return Err("the model answered Want::Logits with a token".into())
                    }
                    Next::Logits(l) if wants => sampler.pick(l, step),
                    // Today's host path: the plain argmax over the host lm_head.
                    Next::Logits(l) => sample_argmax(l).0,
                }
            }
        }))
    }

    /// Runs one forward call at the ledger's position. On `Ok` the ledger takes `tokens`; on
    /// `Err` the cache is cut back to the ledger (I3) and every head is forgotten.
    fn forward(
        &mut self,
        tokens: &[u32],
        f: impl FnOnce(&mut M, usize) -> Result<(), String>,
    ) -> Result<(), String> {
        let pos = self.history.len();
        self.model.mirror_ledger(&self.history);
        match f(&mut self.model, pos) {
            Ok(()) => {
                self.history.extend_from_slice(tokens);
                Ok(())
            }
            Err(e) => {
                self.rollback();
                Err(e)
            }
        }
    }

    /// Every layer back to exactly `0..position()`, no unread head.
    fn rollback(&mut self) {
        let n = self.history.len();
        self.model.kv().truncate(n);
        self.model.clear_pending();
    }
}

/// An `Allow` set must be non-empty, strictly ascending and inside the vocabulary.
fn check_allowed(rows: &[u32], vocab: usize) -> Result<(), String> {
    if rows.is_empty() {
        return Err("constraint allowed no token".into());
    }
    if rows.windows(2).any(|w| w[0] >= w[1]) {
        return Err("constraint's allowed tokens are not strictly ascending".into());
    }
    if let Some(&r) = rows.iter().find(|&&r| r as usize >= vocab) {
        return Err(format!(
            "constraint allowed token {r}, vocabulary is {vocab}"
        ));
    }
    Ok(())
}

/// One generation at a time behind one lock (today's `OmegaEngine`). `Send + Sync`.
pub struct Locked<M: ForwardPass>(Mutex<GenerationController<M>>);

impl<M: ForwardPass> Locked<M> {
    pub fn new(controller: GenerationController<M>) -> Self {
        Self(Mutex::new(controller))
    }

    /// Runs `f` on the controller under the lock. Model-only calls go through
    /// `controller.model()` / `model_mut()`.
    pub fn with_controller<R>(
        &self,
        f: impl FnOnce(&mut GenerationController<M>) -> R,
    ) -> Result<R, String> {
        let mut g = self
            .0
            .lock()
            .map_err(|_| "generation controller lock poisoned".to_string())?;
        Ok(f(&mut g))
    }
}
