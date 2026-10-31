// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Whole-prompt prefill wired into [`Qwen3MoeOmegaEngine`] (W-L1G-W, 2026-10-14). New file, a
//! child module of `qwen3_moe_omega` so it can reach the engine's private `kv`, `pos`, `history`.
//!
//! Switch: the engine field `whole_prompt` (OFF by default: `None`). [`Qwen3MoeOmegaEngine::set_whole_prompt`]
//! builds or drops the driver. When it is set, `generate_with` hands the not-yet-stored part of
//! the prompt to [`WholePromptPrefill::run`] in ONE call (a fixed launch count per layer) instead
//! of the `prefill_chunk` loop, provided: no layer is traced, the routing mutant is off, and the
//! KV is omega's fused cache ([`KvAttention::fused_kv`](super::KvAttention::fused_kv)).
//! Otherwise the chunk loop runs as before.
//! Prefix reuse is kept: the driver starts at `pos0 = self.pos` after `rewind` / `reset`.
use std::time::Instant;

use super::{KvStore, Qwen3MoeOmegaEngine, StepTiming};
use crate::qwen3_moe_omega_prefill::WholePromptPrefill;

/// The real driver type the engine holds.
pub type EngineWholePrompt = WholePromptPrefill<aien_omega_pf::OmegaPfDevice>;

impl Qwen3MoeOmegaEngine {
    /// Turns the whole-prompt prefill on (builds the driver over the resident FP8 weights, read
    /// in place) or off. Returns why it cannot be turned on: the prefill kernels or the device
    /// attention entry not linked, the KV not omega's fused cache, or weights not FP8 everywhere.
    pub fn set_whole_prompt(&mut self, on: bool) -> Result<(), String> {
        if !on {
            self.whole_prompt = None;
            return Ok(());
        }
        if self.whole_prompt.is_some() {
            return Ok(());
        }
        if !aien_omega_pf::is_native() || !aien_omega_pf::has_attention() {
            return Err("whole-prompt prefill: omega's prefill GEMM / prefill attention are not linked in this build".into());
        }
        if self.kv.fused_kv().is_none() {
            return Err(format!(
                "whole-prompt prefill needs omega's fused KV cache, attention path is {}",
                self.kv.path_name()
            ));
        }
        self.whole_prompt = Some(Box::new(WholePromptPrefill::new(
            aien_omega_pf::OmegaPfDevice,
            &self.weights,
        )?));
        Ok(())
    }

    /// Gives the whole-prompt GEMMs their own GPU-cached copy of every FP8 weight (about the
    /// model's FP8 size again, held until the switch is turned off). Returns the bytes copied.
    pub fn whole_prompt_cache_weights(&mut self) -> Result<usize, String> {
        let c = self.weights.config.clone();
        self.whole_prompt
            .as_mut()
            .ok_or("whole-prompt prefill is off")?
            .cache_weights(&c)
    }

    /// Moves the whole-prompt router logits and MoE combine to the chip (W-L1G-E; OFF by default,
    /// and dropped with the driver when the whole-prompt switch goes off). Needs the switch on and
    /// omega's router / combine kernels linked; the result is bit-identical to the host path.
    pub fn set_whole_prompt_device_host(&mut self, on: bool) -> Result<(), String> {
        let wp = self
            .whole_prompt
            .as_mut()
            .ok_or("whole-prompt prefill is off")?;
        wp.set_device_host_work(&self.weights, on)
    }

    /// Whether the whole-prompt path runs its router logits and combine on the chip.
    pub fn whole_prompt_device_host_on(&self) -> bool {
        self.whole_prompt
            .as_ref()
            .is_some_and(|w| w.device_host_work())
    }

    /// What the whole-prompt path holds now, copied out so a caller can price the next prefill
    /// ([`WholePromptFootprint::growth`]) without waiting for the engine lock during a generation.
    pub fn whole_prompt_footprint(&mut self) -> WholePromptFootprint {
        let kv_bytes = self.kv.fused_kv().map_or(0, |f| f.kv_bytes() as u64);
        WholePromptFootprint {
            on: self.whole_prompt.is_some(),
            mirror_live: self.whole_prompt_mirror_live,
            scratch_tokens: self.whole_prompt.as_ref().map_or(0, |w| w.scratch_tokens()),
            kv_bytes,
            chunk: self.whole_prompt_chunk,
        }
    }

    /// Tokens per whole-prompt pass (0 = the whole prompt in one pass). Takes effect on the next
    /// prompt; scratch already sized for a longer pass stays until the switch is turned off.
    pub fn set_whole_prompt_chunk(&mut self, tokens: usize) {
        self.whole_prompt_chunk = tokens;
    }

    /// Tokens per whole-prompt pass (0 = unbounded).
    pub fn whole_prompt_chunk(&self) -> usize {
        self.whole_prompt_chunk
    }

    /// Whether `generate_with` will take the whole-prompt path.
    pub fn whole_prompt_on(&self) -> bool {
        self.whole_prompt.is_some()
    }

    /// Correctness gate (W-L1G-C): installs or removes the recorder on the whole-prompt driver.
    pub fn whole_prompt_set_trace(
        &mut self,
        trace: Option<crate::qwen3_moe_omega_prefill::WpTrace>,
    ) -> Result<(), String> {
        self.whole_prompt
            .as_mut()
            .ok_or("whole-prompt prefill is off")?
            .trace = trace;
        Ok(())
    }

    /// Correctness gate (W-L1G-C): takes the recorder (and what it recorded) off the driver.
    pub fn whole_prompt_take_trace(&mut self) -> Option<crate::qwen3_moe_omega_prefill::WpTrace> {
        self.whole_prompt.as_mut().and_then(|w| w.trace.take())
    }

    /// Correctness gate (W-L1G-C): sets the negative-control mutant of the whole-prompt driver.
    pub fn whole_prompt_set_mutant(
        &mut self,
        mutant: crate::qwen3_moe_omega_prefill::WpMutant,
    ) -> Result<(), String> {
        self.whole_prompt
            .as_mut()
            .ok_or("whole-prompt prefill is off")?
            .mutant = mutant;
        Ok(())
    }

    /// The last whole-prompt call's counters (launches, chip and host time).
    pub fn whole_prompt_stats(&self) -> Option<&crate::qwen3_moe_omega_prefill::PrefillStats> {
        self.whole_prompt.as_ref().map(|w| &w.stats)
    }

    /// Called by `generate_with` before its chunk loop. When the switch is on and the path
    /// applies, runs `tokens` (positions `self.pos ..`) as whole-prompt passes of at most
    /// `whole_prompt_chunk` tokens each, stores them as history, sets `hidden` and adds the
    /// timing; returns how many tokens it consumed (all or 0).
    ///
    /// The passes are sequential and each one sees the KV of the earlier ones, exactly like a
    /// prefix-reuse start, so the result matches one pass over the whole prompt. What changes is
    /// memory: the driver's scratch is sized by the tokens of one pass (about 0.22 MB per token on
    /// the 30B), so a 66k-token prompt no longer makes it hold 14 GB until the process exits.
    pub(super) fn whole_prompt_prefill(
        &mut self,
        tokens: &[u32],
        hidden: &mut Vec<f32>,
        timing: &mut StepTiming,
    ) -> Result<usize, String> {
        if tokens.is_empty() || !self.trace_layers.is_empty() || self.routing_mutant {
            return Ok(0);
        }
        let chunk = self.whole_prompt_chunk;
        let Some(wp) = self.whole_prompt.as_mut() else {
            return Ok(0);
        };
        let Some(fused) = self.kv.fused_kv() else {
            return Ok(0);
        };
        let (start, m) = (self.pos, tokens.len());
        if start + m > fused.capacity() {
            return Err(format!(
                "whole-prompt prefill: {start} + {m} tokens > context {}",
                fused.capacity()
            ));
        }
        for (from, to) in chunk_spans(m, chunk) {
            let pos0 = start + from;
            let t = Instant::now();
            let out = wp.run(
                &self.weights,
                &mut fused.prefill_attention(),
                &tokens[from..to],
                pos0,
            );
            let wall = t.elapsed().as_nanos() as u64;
            // On failure the cache rows past `start` are unspecified: forget them before reporting.
            let h = match out {
                Ok(h) => h,
                Err(e) => {
                    self.rewind(start);
                    return Err(e);
                }
            };
            fused.mark_filled(start + to);
            self.whole_prompt_mirror_live = true;
            self.pos = start + to;
            self.history.extend_from_slice(&tokens[from..to]);
            let st = &wp.stats;
            timing.add(&StepTiming {
                total_ns: wall,
                attention_ns: st.attn_ns,
                experts_ns: 0,
                lm_head_ns: 0,
                host_ns: st.host_ns,
                chip_ns: st.chip_ns + st.attn_ns,
                launches: st.launches() as u64,
                route_ns: 0,
                moe_host_ns: 0,
                // Decode-graph counters (decode-mega): the prefill never runs the one-list decode.
                ..StepTiming::default()
            });
            *hidden = h;
        }
        Ok(m)
    }
}

/// Default tokens per whole-prompt pass. The driver's scratch is sized by the pass (measured on
/// the live 30B, ledger L884: 5.9 GB for a 26,793-token pass, about 0.22 MB per token, and it
/// was kept after the request). 8192 tokens bounds it near 1.8 GB and still gives the prefill
/// GEMMs thousands of rows each.
pub const DEFAULT_CHUNK: usize = 8192;

/// `(from, to)` token ranges of the passes for `m` tokens: `chunk` tokens each, the last one
/// shorter; `chunk == 0` means one pass over everything.
pub(super) fn chunk_spans(m: usize, chunk: usize) -> impl Iterator<Item = (usize, usize)> {
    let step = if chunk == 0 { m.max(1) } else { chunk };
    (0..m)
        .step_by(step)
        .map(move |from| (from, (from + step).min(m)))
}

#[cfg(test)]
mod tests {
    use super::{chunk_spans, estimate_growth, PrefillGrowth, SCRATCH_BYTES_PER_TOKEN};

    #[test]
    fn spans_cover_every_token_once_in_order() {
        for (m, chunk) in [
            (1, 8),
            (8, 8),
            (9, 8),
            (26_793, 8192),
            (66_000, 8192),
            (5, 1),
        ] {
            let spans: Vec<_> = chunk_spans(m, chunk).collect();
            assert_eq!(spans.first().map(|s| s.0), Some(0), "m {m} chunk {chunk}");
            assert_eq!(spans.last().map(|s| s.1), Some(m), "m {m} chunk {chunk}");
            assert!(spans.windows(2).all(|w| w[0].1 == w[1].0), "contiguous");
            assert!(
                spans.iter().all(|&(a, b)| a < b && b - a <= chunk),
                "bounded"
            );
            assert_eq!(spans.len(), m.div_ceil(chunk), "m {m} chunk {chunk}");
        }
    }

    #[test]
    fn zero_chunk_is_one_pass_and_empty_input_has_no_pass() {
        assert_eq!(
            chunk_spans(30_000, 0).collect::<Vec<_>>(),
            vec![(0, 30_000)]
        );
        assert_eq!(chunk_spans(0, 8192).count(), 0);
        assert_eq!(chunk_spans(0, 0).count(), 0);
    }

    const KV_128K: u64 = 48 * 256 * 1024 * 1024 + 48 * 20 * 1024;

    #[test]
    fn first_prefill_pays_the_mirror_and_one_pass_of_scratch() {
        let g = estimate_growth(false, KV_128K, 0, 66_000, 8192);
        assert_eq!(g.mirror_bytes, KV_128K);
        assert_eq!(g.scratch_bytes, 8192 * SCRATCH_BYTES_PER_TOKEN);
        assert_eq!(g.total(), g.mirror_bytes + g.scratch_bytes);
    }

    #[test]
    fn a_66k_prompt_costs_no_more_than_an_8k_one_once_chunked() {
        let long = estimate_growth(true, KV_128K, 0, 66_000, 8192);
        let short = estimate_growth(true, KV_128K, 0, 8192, 8192);
        assert_eq!(long, short);
        // Unchunked it is the 14+ GB the live server grew by (ledger L884).
        let unchunked = estimate_growth(true, KV_128K, 0, 66_000, 0);
        assert!(unchunked.scratch_bytes > 15_000_000_000);
    }

    #[test]
    fn warm_driver_grows_only_past_its_high_water_mark() {
        assert_eq!(
            estimate_growth(true, KV_128K, 8192, 30_000, 8192),
            PrefillGrowth::default()
        );
        let g = estimate_growth(true, KV_128K, 100, 200, 8192);
        assert_eq!(g.scratch_bytes, 100 * SCRATCH_BYTES_PER_TOKEN);
        assert_eq!(g.mirror_bytes, 0);
    }
}

/// Bytes of prompt-sized scratch per token of a pass, 30B device-host path: measured on the live
/// server (ledger L884) as 5.9 GB for 26,793 tokens = 236 KB; the buffers add up to 231 KB
/// (qkv/out/oproj/xin/xq 74 KB, MoE act/aq 18 KB, `down` 64 KB twice-ish, router/combine 17 KB).
pub const SCRATCH_BYTES_PER_TOKEN: u64 = 240 * 1024;

/// Memory a prefill adds to the process beyond the load footprint.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefillGrowth {
    /// omega's KV mirror (one cache-sized copy of the KV, GPU-cached), allocated by the first
    /// pass of the process and kept. Measured on the live 30B: 48 x 256 MiB at 131,072 positions.
    pub mirror_bytes: u64,
    /// Prompt-sized scratch beyond what the driver already holds.
    pub scratch_bytes: u64,
}

impl PrefillGrowth {
    pub fn total(self) -> u64 {
        self.mirror_bytes + self.scratch_bytes
    }
}

/// What a prefill of `new_tokens` uncached tokens allocates: the mirror if no pass has run yet,
/// and scratch for one pass (`chunk` tokens at most, 0 = all) above the `held_tokens` the driver
/// is already sized for.
pub fn estimate_growth(
    mirror_live: bool,
    kv_bytes: u64,
    held_tokens: usize,
    new_tokens: usize,
    chunk: usize,
) -> PrefillGrowth {
    let pass = if chunk == 0 {
        new_tokens
    } else {
        new_tokens.min(chunk)
    };
    PrefillGrowth {
        mirror_bytes: if mirror_live { 0 } else { kv_bytes },
        scratch_bytes: pass.saturating_sub(held_tokens) as u64 * SCRATCH_BYTES_PER_TOKEN,
    }
}

/// The whole-prompt path's resident memory state, see `Qwen3MoeOmegaEngine::whole_prompt_footprint`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WholePromptFootprint {
    pub on: bool,
    /// The KV mirror exists (a pass has run).
    pub mirror_live: bool,
    /// Tokens the prompt-sized scratch is sized for.
    pub scratch_tokens: usize,
    /// Bytes of the KV cache (the mirror is a copy of it).
    pub kv_bytes: u64,
    /// Tokens per pass (0 = unbounded).
    pub chunk: usize,
}

impl WholePromptFootprint {
    /// Memory a prefill of `new_tokens` uncached tokens adds on top of this footprint.
    pub fn growth(&self, new_tokens: usize) -> PrefillGrowth {
        if !self.on {
            return PrefillGrowth::default();
        }
        estimate_growth(
            self.mirror_live,
            self.kv_bytes,
            self.scratch_tokens,
            new_tokens,
            self.chunk,
        )
    }
}
