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

//! One conversation on the resident model (sc #642 serving): prompts fed through the prefill,
//! replies through the decode graph, and the next turn resumed from what the device holds.
//!
//! The DeltaNet layers carry a recurrent state that cannot be rewound, so a turn can only reuse
//! a state whose tokens are an exact prefix of the new prompt. Two such states are kept:
//!
//! - live: everything fed so far (the last prompt and the reply tokens fed back). It matches the
//!   next prompt only when the client returns the reply byte for byte, thinking included.
//! - snapshot: the DeltaNet tails and states copied to host memory just before the prompt's last
//!   `<|im_start|>` (the header of the turn being answered). The chat template renders every
//!   earlier turn the same way on the next request, so the next prompt starts with these tokens
//!   and only the new turns are fed. The attention KV needs no copy: the rows past the snapshot
//!   are rewritten before anything reads them.
//!
//! The longer of the two that is a strict prefix of the prompt wins; otherwise the state is reset
//! and the whole prompt is fed (cold).
use std::time::Instant;

use super::{PrefillStats, Qwen38Omega, Qwen38Snapshot};
use aien_omega_q38 as q;

/// `<|im_start|>` in the Qwen3.8 vocabulary (tokenizer.json added token 248045).
pub const IM_START: u32 = 248045;

/// What the next prompt continues from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reuse {
    /// Reset and feed the whole prompt.
    Cold,
    /// Continue the live state, which holds this many prompt tokens.
    Live(usize),
    /// Restore the snapshot, which holds this many prompt tokens.
    Snapshot(usize),
}

impl Reuse {
    pub fn tokens(self) -> usize {
        match self {
            Self::Cold => 0,
            Self::Live(n) | Self::Snapshot(n) => n,
        }
    }
}

/// The longer of `live` and `snapshot` that is a strict prefix of `prompt` (at least one prompt
/// token must be fed: its logits give the first reply token). Live wins a tie (no copy).
pub fn plan_reuse(live: &[u32], snapshot: Option<&[u32]>, prompt: &[u32]) -> Reuse {
    let fits = |s: &[u32]| !s.is_empty() && s.len() < prompt.len() && prompt.starts_with(s);
    let live_n = if fits(live) { live.len() } else { 0 };
    let snap_n = snapshot.filter(|s| fits(s)).map_or(0, <[u32]>::len);
    if live_n == 0 && snap_n == 0 {
        Reuse::Cold
    } else if live_n >= snap_n {
        Reuse::Live(live_n)
    } else {
        Reuse::Snapshot(snap_n)
    }
}

/// What feeding one prompt cost.
#[derive(Debug, Clone, Default)]
pub struct FeedStats {
    pub reuse: Option<Reuse>,
    /// Prompt tokens fed this turn (prompt length minus the reused prefix).
    pub fed: usize,
    /// Of those, fed one at a time through the decode graph (pieces shorter than the prefill's
    /// minimum).
    pub stepped: usize,
    pub prefill: PrefillStats,
    pub restore_ns: u64,
    pub snapshot_ns: u64,
    pub wall_ns: u64,
}

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

fn add(sum: &mut PrefillStats, s: &PrefillStats) {
    sum.tokens += s.tokens;
    sum.chunks += s.chunks;
    sum.wall_ns += s.wall_ns;
    sum.gemm_chip_ns += s.gemm_chip_ns;
    sum.gemm_call_ns += s.gemm_call_ns;
    sum.gemm_launches += s.gemm_launches;
    sum.ab_call_ns += s.ab_call_ns;
    sum.gdn_chip_ns += s.gdn_chip_ns;
    sum.gdn_call_ns += s.gdn_call_ns;
    sum.attn_chip_ns += s.attn_chip_ns;
    sum.attn_call_ns += s.attn_call_ns;
    sum.host_ns += s.host_ns;
    sum.head_ns += s.head_ns;
}

/// The resident model plus the token record of its live state and of its snapshot.
#[derive(Debug)]
pub struct Qwen38Session {
    model: Qwen38Omega,
    chunk: usize,
    /// Tokens the live state holds (`model.pos == history.len()` whenever this is trusted).
    history: Vec<u32>,
    snap: Qwen38Snapshot,
    /// Tokens the snapshot holds; `None` until one is taken (or after a failed copy).
    snap_tokens: Option<Vec<u32>>,
}

impl Qwen38Session {
    /// `chunk`: the prefill's chunk length (`6 ..= 32768`).
    pub fn new(model: Qwen38Omega, chunk: usize) -> Self {
        Self {
            model,
            chunk,
            history: Vec::new(),
            snap: Qwen38Snapshot::default(),
            snap_tokens: None,
        }
    }

    pub fn model(&self) -> &Qwen38Omega {
        &self.model
    }

    pub fn model_mut(&mut self) -> &mut Qwen38Omega {
        &mut self.model
    }

    /// Tokens the live state holds.
    pub fn history(&self) -> &[u32] {
        &self.history
    }

    /// Tokens the snapshot holds.
    pub fn snapshot_tokens(&self) -> Option<&[u32]> {
        self.snap_tokens.as_deref()
    }

    /// Positions the KV cache holds: prompt plus reply must stay below this.
    pub fn max_pos(&self) -> usize {
        self.model.max_pos()
    }

    /// Drops both reuse candidates: the next prompt is fed cold.
    pub fn forget(&mut self) {
        self.history.clear();
        self.snap_tokens = None;
    }

    /// Feeds `prompt` (reusing what [`plan_reuse`] allows) and returns the logits after its last
    /// token. On any error the live state is dropped (the snapshot, a host copy, stays valid).
    pub fn start(&mut self, prompt: &[u32]) -> Result<(Vec<f32>, FeedStats), String> {
        let r = self.start_inner(prompt);
        if r.is_err() {
            self.history.clear();
        }
        r
    }

    fn start_inner(&mut self, prompt: &[u32]) -> Result<(Vec<f32>, FeedStats), String> {
        let t0 = Instant::now();
        if prompt.is_empty() {
            return Err("qwen38 session: empty prompt".into());
        }
        if prompt.len() >= self.model.max_pos() {
            return Err(format!(
                "qwen38 session: prompt of {} tokens, the KV cache holds {}",
                prompt.len(),
                self.model.max_pos()
            ));
        }
        let reuse = plan_reuse(&self.history, self.snap_tokens.as_deref(), prompt);
        let mut st = FeedStats {
            reuse: Some(reuse),
            ..Default::default()
        };
        match reuse {
            Reuse::Live(_) => {}
            Reuse::Snapshot(n) => {
                let t = Instant::now();
                self.history.clear();
                self.model.restore(&self.snap)?;
                self.history.extend_from_slice(&prompt[..n]);
                st.restore_ns = ns(t);
            }
            Reuse::Cold => {
                self.history.clear();
                self.model.reset()?;
            }
        }
        let from = self.history.len();
        st.fed = prompt.len() - from;
        // The snapshot point: the last turn header not yet inside the reused prefix.
        if let Some(k) = prompt
            .iter()
            .rposition(|&t| t == IM_START)
            .filter(|&k| k >= from)
        {
            self.feed_rows(&prompt[from..k], &mut st)?;
            if self.snap_tokens.as_deref() != Some(&prompt[..k]) && k > 0 {
                let t = Instant::now();
                self.snap_tokens = None;
                self.model.snapshot(&mut self.snap)?;
                self.snap_tokens = Some(prompt[..k].to_vec());
                st.snapshot_ns = ns(t);
            }
        }
        let rest = &prompt[self.history.len()..];
        let logits = self.feed_last(rest, &mut st)?;
        st.wall_ns = ns(t0);
        Ok((logits, st))
    }

    /// Feeds `tokens` without the head (the residual rows are dropped).
    fn feed_rows(&mut self, tokens: &[u32], st: &mut FeedStats) -> Result<(), String> {
        if tokens.len() >= q::GDN_PF_MIN_T {
            let (_, s) = self.model.prefill_rows(tokens, self.chunk, false)?;
            add(&mut st.prefill, &s);
        } else {
            for &t in tokens {
                self.model.step(t)?;
            }
            st.stepped += tokens.len();
        }
        self.history.extend_from_slice(tokens);
        Ok(())
    }

    /// Feeds `tokens` (at least one) and returns the logits after the last.
    fn feed_last(&mut self, tokens: &[u32], st: &mut FeedStats) -> Result<Vec<f32>, String> {
        if tokens.is_empty() {
            return Err("qwen38 session: nothing to feed".into());
        }
        let logits = if tokens.len() >= q::GDN_PF_MIN_T {
            let (_, logits, s) = self.model.prefill(tokens, self.chunk)?;
            add(&mut st.prefill, &s);
            logits
        } else {
            for &t in tokens {
                self.model.step(t)?;
            }
            st.stepped += tokens.len();
            self.model.logits()?
        };
        self.history.extend_from_slice(tokens);
        Ok(logits)
    }

    /// Feeds one reply token through the decode graph: the logits after it and the graph's
    /// chip time for the step (ns).
    pub fn advance(&mut self, token: u32) -> Result<(Vec<f32>, u64), String> {
        let r = self.model.step(token).and_then(|(_, stats)| {
            let logits = self.model.logits()?;
            Ok((logits, stats.chip_ns))
        });
        match r {
            Ok(v) => {
                self.history.push(token);
                Ok(v)
            }
            Err(e) => {
                self.history.clear();
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reuse_takes_the_longest_strict_prefix() {
        let p = [1, 2, 3, 4, 5, 6];
        assert_eq!(plan_reuse(&[], None, &p), Reuse::Cold);
        assert_eq!(plan_reuse(&[1, 2, 3], None, &p), Reuse::Live(3));
        assert_eq!(
            plan_reuse(&[1, 2], Some(&[1, 2, 3, 4]), &p),
            Reuse::Snapshot(4)
        );
        assert_eq!(
            plan_reuse(&[1, 2, 3, 4, 5], Some(&[1, 2]), &p),
            Reuse::Live(5)
        );
        // Tie: live (no copy).
        assert_eq!(plan_reuse(&[1, 2], Some(&[1, 2]), &p), Reuse::Live(2));
    }

    #[test]
    fn reuse_never_takes_a_whole_or_diverging_state() {
        let p = [1, 2, 3];
        // Equal to the prompt: nothing left to feed, so not reusable.
        assert_eq!(plan_reuse(&[1, 2, 3], None, &p), Reuse::Cold);
        // Longer than the prompt, or diverging inside it.
        assert_eq!(plan_reuse(&[1, 2, 3, 4], Some(&[1, 9]), &p), Reuse::Cold);
        assert_eq!(plan_reuse(&[9], Some(&[1, 2, 3, 4]), &p), Reuse::Cold);
        // A diverging live state does not hide a valid snapshot.
        assert_eq!(
            plan_reuse(&[1, 2, 9], Some(&[1, 2]), &p),
            Reuse::Snapshot(2)
        );
    }
}
