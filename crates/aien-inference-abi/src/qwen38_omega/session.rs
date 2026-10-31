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
    /// Of those, fed one at a time through the decode graph: pieces shorter than the 27B
    /// prefill's minimum, and on the 35B only with `AIEN_Q36_PREFILL_CHUNK=0` or a last piece
    /// shorter than [`q::GDN_PF_MIN_T`] (DRA-65: the 35B's rows before the snapshot point go in
    /// as chip prefill chunks at any length).
    pub stepped: usize,
    pub prefill: PrefillStats,
    pub restore_ns: u64,
    pub snapshot_ns: u64,
    pub wall_ns: u64,
}

fn ns(t: Instant) -> u64 {
    t.elapsed().as_nanos() as u64
}

/// The snapshot point of `prompt` when the first `from` tokens are already fed: its last
/// `<|im_start|>` (the header of the turn being answered) at or past `from`. The session feeds
/// `prompt[from..k]` without the head, snapshots there, then feeds `prompt[k..]` with the head;
/// `None`: no such header, the whole rest goes in with the head and no snapshot is taken.
pub fn snapshot_point(prompt: &[u32], from: usize) -> Option<usize> {
    prompt
        .iter()
        .rposition(|&t| t == IM_START)
        .filter(|&k| k >= from)
}

/// Whether a row piece of `len` tokens (no head) is stepped token by token rather than fed by
/// [`Qwen38Omega::prefill_state`]: only the 27B's pieces under its prefill minimum; the 35B's
/// chip prefill takes any length (DRA-65).
pub fn rows_by_steps(len: usize, moe: bool) -> bool {
    !moe && len < q::GDN_PF_MIN_T
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

/// Books one prefill call: the 35B with `AIEN_Q36_PREFILL_CHUNK=0` reports no chunks for tokens
/// it stepped through the decode graph, which count as stepped.
fn count(st: &mut FeedStats, s: &PrefillStats) {
    if s.chunks == 0 && s.tokens > 0 {
        st.stepped += s.tokens;
    } else {
        add(&mut st.prefill, s);
    }
}

/// What a prompt's feed hands back after its last token.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Head {
    /// The logits (`[vocab]`): the reply's first token is drawn from them on the host.
    Logits(Vec<f32>),
    /// The graph's greedy token (DRA-87: the chip's argmax, no logits read back).
    Token(u32),
}

/// What a session feeds a prompt into: the model, or a recording fake in the tests.
pub(crate) trait PromptFeed {
    fn is_moe(&self) -> bool;
    /// One decode step: the graph's greedy token after it.
    fn step_token(&mut self, t: u32) -> Result<u32, String>;
    /// The rows before the snapshot point, for their state only ([`Qwen38Omega::prefill_state`]).
    fn feed_state(&mut self, tokens: &[u32], chunk: usize) -> Result<PrefillStats, String>;
    /// At least [`q::GDN_PF_MIN_T`] tokens with the head ([`Qwen38Omega::prefill`]): the logits.
    fn feed_head(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(Vec<f32>, PrefillStats), String>;
    /// As [`Self::feed_head`], the greedy token instead of the logits
    /// ([`Qwen38Omega::prefill_token`]).
    fn feed_head_token(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(u32, PrefillStats), String>;
    fn last_logits(&mut self) -> Result<Vec<f32>, String>;
}

impl PromptFeed for Qwen38Omega {
    fn is_moe(&self) -> bool {
        Qwen38Omega::is_moe(self)
    }

    fn step_token(&mut self, t: u32) -> Result<u32, String> {
        self.step(t).map(|(next, _)| next)
    }

    fn feed_state(&mut self, tokens: &[u32], chunk: usize) -> Result<PrefillStats, String> {
        self.prefill_state(tokens, chunk)
    }

    fn feed_head(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(Vec<f32>, PrefillStats), String> {
        self.prefill(tokens, chunk)
            .map(|(_, logits, s)| (logits, s))
    }

    fn feed_head_token(
        &mut self,
        tokens: &[u32],
        chunk: usize,
    ) -> Result<(u32, PrefillStats), String> {
        self.prefill_token(tokens, chunk)
    }

    fn last_logits(&mut self) -> Result<Vec<f32>, String> {
        self.logits()
    }
}

/// Feeds `prompt[from..]` into a state holding `prompt[..from]`: the rows before the snapshot point
/// without the head, then `snap(m, k)` at that point (its ns, if it took one), then the rest with
/// the head. Returns the logits after the last token, or with `want_token` the graph's greedy
/// token after it.
fn feed_prompt<M: PromptFeed>(
    m: &mut M,
    prompt: &[u32],
    from: usize,
    chunk: usize,
    want_token: bool,
    st: &mut FeedStats,
    mut snap: impl FnMut(&mut M, usize) -> Result<Option<u64>, String>,
) -> Result<Head, String> {
    let mut at = from;
    // The snapshot point: the last turn header not yet inside the reused prefix.
    if let Some(k) = snapshot_point(prompt, from) {
        feed_rows(m, &prompt[from..k], chunk, st)?;
        at = k;
        if let Some(t) = snap(m, k)? {
            st.snapshot_ns = t;
        }
    }
    feed_last(m, &prompt[at..], chunk, want_token, st)
}

/// Feeds `tokens` without the head (no residual read back).
fn feed_rows<M: PromptFeed>(
    m: &mut M,
    tokens: &[u32],
    chunk: usize,
    st: &mut FeedStats,
) -> Result<(), String> {
    if rows_by_steps(tokens.len(), m.is_moe()) {
        for &t in tokens {
            m.step_token(t)?;
        }
        st.stepped += tokens.len();
    } else {
        let s = m.feed_state(tokens, chunk)?;
        count(st, &s);
    }
    Ok(())
}

/// Feeds `tokens` (at least one) and returns the logits after the last, or with `want_token` the
/// graph's greedy token after it (no logits read back).
fn feed_last<M: PromptFeed>(
    m: &mut M,
    tokens: &[u32],
    chunk: usize,
    want_token: bool,
    st: &mut FeedStats,
) -> Result<Head, String> {
    if tokens.is_empty() {
        return Err("qwen38 session: nothing to feed".into());
    }
    if tokens.len() >= q::GDN_PF_MIN_T {
        let (head, s) = if want_token {
            let (t, s) = m.feed_head_token(tokens, chunk)?;
            (Head::Token(t), s)
        } else {
            let (logits, s) = m.feed_head(tokens, chunk)?;
            (Head::Logits(logits), s)
        };
        count(st, &s);
        return Ok(head);
    }
    let mut next = 0;
    for &t in tokens {
        next = m.step_token(t)?;
    }
    st.stepped += tokens.len();
    if want_token {
        Ok(Head::Token(next))
    } else {
        m.last_logits().map(Head::Logits)
    }
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
        match self.start_head(prompt, false)? {
            (Head::Logits(logits), st) => Ok((logits, st)),
            (Head::Token(_), _) => {
                Err("qwen38 session: a token where the logits were asked".into())
            }
        }
    }

    /// [`Self::start`] for a greedy reply (DRA-87): the graph's greedy token after the prompt's
    /// last token (on the 35B the chip's argmax of the head, the host rule: the largest logit,
    /// the first index on a tie) instead of its logits, so no logits are read back and no host
    /// argmax runs for the reply's first token.
    pub fn start_greedy(&mut self, prompt: &[u32]) -> Result<(u32, FeedStats), String> {
        match self.start_head(prompt, true)? {
            (Head::Token(t), st) => Ok((t, st)),
            (Head::Logits(_), _) => Err("qwen38 session: logits where the token was asked".into()),
        }
    }

    fn start_head(
        &mut self,
        prompt: &[u32],
        want_token: bool,
    ) -> Result<(Head, FeedStats), String> {
        let r = self.start_inner(prompt, want_token);
        if r.is_err() {
            self.history.clear();
        }
        r
    }

    fn start_inner(
        &mut self,
        prompt: &[u32],
        want_token: bool,
    ) -> Result<(Head, FeedStats), String> {
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
        let (snap, snap_tokens) = (&mut self.snap, &mut self.snap_tokens);
        let head = feed_prompt(
            &mut self.model,
            prompt,
            from,
            self.chunk,
            want_token,
            &mut st,
            |m, k| {
                if k == 0 || snap_tokens.as_deref() == Some(&prompt[..k]) {
                    return Ok(None);
                }
                let t = Instant::now();
                *snap_tokens = None;
                m.snapshot(snap)?;
                *snap_tokens = Some(prompt[..k].to_vec());
                Ok(Some(ns(t)))
            },
        )?;
        self.history.extend_from_slice(&prompt[from..]);
        st.wall_ns = ns(t0);
        Ok((head, st))
    }

    /// Feeds one reply token through the decode graph without reading the logits: the graph's
    /// greedy token after it (chosen on the chip) and the step's chip time (ns). For greedy
    /// replies (L356: no 1 MB logits copy and host argmax per token).
    pub fn advance_greedy(&mut self, token: u32) -> Result<(u32, u64), String> {
        match self.model.step(token) {
            Ok((next, stats)) => {
                self.history.push(token);
                Ok((next, stats.chip_ns))
            }
            Err(e) => {
                self.history.clear();
                Err(e)
            }
        }
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
    use crate::qwen38_omega::prefill::{chip_argmax_model, q36_chunk_walk, q36_state, Q36Feed};

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

    /// A rendered single-turn prompt of `len` tokens whose answer header (`<|im_start|>`
    /// assistant ... , 7 tokens with thinking off) closes it, plus an earlier user header at 0.
    fn rendered(len: usize) -> Vec<u32> {
        let mut p: Vec<u32> = (0..len as u32).map(|i| 1000 + i).collect();
        p[0] = IM_START;
        if len > 7 {
            p[len - 7] = IM_START;
        }
        p
    }

    /// DRA-65: where start_inner splits a prompt at the lengths the chunk tests use: the rows
    /// before the answer header (fed without the head, as 35B prefill chunks at any length),
    /// the snapshot at the header, the header and what follows fed with the head.
    #[test]
    fn snapshot_point_splits_rows_and_last_piece() {
        let cap = q::q36::PREFILL_MAX_ROWS;
        for len in [1, cap - 1, cap, cap + 1, 2048, 8192] {
            let p = rendered(len);
            let k = snapshot_point(&p, 0).expect("a header");
            let want = if len > 7 { len - 7 } else { 0 };
            assert_eq!(k, want, "len {len}");
            assert_eq!(p[k], IM_START);
            // The rows piece and the last piece cover the prompt once, the last starts at the header.
            let (rows, last) = p.split_at(k);
            assert_eq!(rows.len() + last.len(), len);
            assert!(!last.is_empty() && last[0] == IM_START);
            // The 35B never steps the rows piece; its chunks cover it.
            assert!(!rows_by_steps(rows.len(), true));
            let c = super::super::prefill::q36_chunk_lengths(rows.len(), cap);
            assert_eq!(c.iter().sum::<usize>(), rows.len(), "len {len}");
        }
        // 2048: 2041 rows = 15 chunks of 128 + 121; the answer header's 7 tokens go in with the head.
        let c = super::super::prefill::q36_chunk_lengths(2048 - 7, cap);
        assert_eq!((c.len(), c[15]), (16, 121));
    }

    #[test]
    fn snapshot_point_respects_the_reused_prefix() {
        // Turn 2: the reused snapshot ends at turn 1's answer header (k1); the new answer header
        // is past it, so the rows between are fed and the snapshot moves there.
        let mut p = rendered(40);
        let k1 = 33;
        p.extend([2000, 2001, 2002, IM_START, 2003, 2004]);
        assert_eq!(snapshot_point(&p, k1), Some(43));
        // The header right at the reused end: an empty rows piece, the same snapshot.
        assert_eq!(snapshot_point(&p, 43), Some(43));
        // A live state past the last header: no snapshot, the rest goes in with the head.
        assert_eq!(snapshot_point(&p, 44), None);
        // No header at all.
        assert_eq!(snapshot_point(&[5, 6, 7], 0), None);
    }

    #[test]
    fn only_short_27b_row_pieces_are_stepped() {
        for len in [0, 1, 2, 3, 127, 128, 129, 8192] {
            assert!(!rows_by_steps(len, true), "35B len {len}");
        }
        assert!(rows_by_steps(2, false));
        assert!(!rows_by_steps(q::GDN_PF_MIN_T, false));
    }

    /// Records every token the feed path hands over, with the position it lands at and how
    /// (`s` decode step, `r` state chunk, `h` chunk with the head), through the same routing as
    /// the 35B: [`q36_state`] for the rows, [`q36_chunk_walk`] for every chunk run. Its "model":
    /// the logits after the last token fed are a function of every (position, token) fed so far,
    /// over 97 tokens and 5 levels (so the largest logit is always tied); the graph's greedy
    /// token is the CPU model of the chip's two-stage ARGMAX over them ([`chip_argmax_model`]).
    struct Fake {
        c: usize,
        pos: usize,
        fed: Vec<(usize, u32, char)>,
        chunks: Vec<usize>,
        hash: u64,
        /// Head reads: `L` logits after a head chunk, `T` token after a head chunk, `l` the last
        /// step's logits.
        heads: Vec<char>,
    }

    const FAKE_VOCAB: usize = 97;

    impl Fake {
        fn new(c: usize, pos: usize) -> Self {
            Self {
                c,
                pos,
                fed: Vec::new(),
                chunks: Vec::new(),
                hash: 0x5eed,
                heads: Vec::new(),
            }
        }

        fn take(&mut self, tokens: &[u32], pos: usize, how: char) -> Result<(), String> {
            if pos != self.pos {
                return Err(format!("fed at {pos}, the state is at {}", self.pos));
            }
            for &t in tokens {
                self.fed.push((self.pos, t, how));
                self.hash = (self.hash ^ ((self.pos as u64) << 32 | u64::from(t)))
                    .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                    .rotate_left(29);
                self.pos += 1;
            }
            Ok(())
        }

        fn walk(&mut self, tokens: &[u32], chunk: usize, how: char) -> Result<usize, String> {
            let pos0 = self.pos;
            let (n, end) = q36_chunk_walk(tokens, chunk, pos0, |c, pos, _| {
                self.chunks.push(c.len());
                self.take(c, pos, how)
            })?;
            assert_eq!(end, self.pos);
            Ok(n)
        }

        fn logits(&self) -> Vec<f32> {
            (0..FAKE_VOCAB as u64)
                .map(|i| {
                    let x = (self.hash ^ i.wrapping_mul(0xff51_afd7_ed55_8ccd))
                        .wrapping_mul(0xc4ce_b9fe_1a85_ec53);
                    ((x >> 59) % 5) as f32
                })
                .collect()
        }

        fn chip_token(&self) -> u32 {
            chip_argmax_model(&self.logits(), 8)
        }

        fn head(&mut self, tokens: &[u32]) -> Result<PrefillStats, String> {
            let mut chunks = 0;
            if self.c == 0 {
                for &t in tokens {
                    self.q36_step(t)?;
                }
            } else {
                let c = self.c;
                chunks = self.walk(tokens, c, 'h')?;
            }
            Ok(PrefillStats {
                tokens: tokens.len(),
                chunks,
                ..Default::default()
            })
        }
    }

    impl Q36Feed for Fake {
        fn q36_step(&mut self, t: u32) -> Result<(), String> {
            let p = self.pos;
            self.take(&[t], p, 's')
        }

        fn q36_chunks(&mut self, tokens: &[u32], chunk: usize) -> Result<usize, String> {
            self.walk(tokens, chunk, 'r')
        }
    }

    impl PromptFeed for Fake {
        fn is_moe(&self) -> bool {
            true
        }

        fn step_token(&mut self, t: u32) -> Result<u32, String> {
            self.q36_step(t)?;
            Ok(self.chip_token())
        }

        fn feed_state(&mut self, tokens: &[u32], _chunk: usize) -> Result<PrefillStats, String> {
            let c = self.c;
            let chunks = q36_state(self, tokens, c)?;
            Ok(PrefillStats {
                tokens: tokens.len(),
                chunks,
                ..Default::default()
            })
        }

        fn feed_head(
            &mut self,
            tokens: &[u32],
            _chunk: usize,
        ) -> Result<(Vec<f32>, PrefillStats), String> {
            let s = self.head(tokens)?;
            self.heads.push('L');
            Ok((self.logits(), s))
        }

        fn feed_head_token(
            &mut self,
            tokens: &[u32],
            _chunk: usize,
        ) -> Result<(u32, PrefillStats), String> {
            let s = self.head(tokens)?;
            self.heads.push('T');
            Ok((self.chip_token(), s))
        }

        fn last_logits(&mut self) -> Result<Vec<f32>, String> {
            self.heads.push('l');
            Ok(self.logits())
        }
    }

    /// DRA-65: through the session's feed path every prompt token past the reused prefix is fed
    /// exactly once, in order, at its own position, across chunk boundaries and from a cached
    /// prefix that ends mid-chunk; the snapshot is taken right after the rows; with chunking on
    /// no row is stepped, with `AIEN_Q36_PREFILL_CHUNK=0` every token is.
    #[test]
    fn every_prompt_token_is_fed_once_in_order() {
        let cap = q::q36::PREFILL_MAX_ROWS;
        for c in [cap, 0] {
            for len in [1, cap - 1, cap, cap + 1, 2 * cap + 3] {
                // 61: a cached prefix ending mid-chunk; len - 3: a live state past the header.
                for (from, want_token) in [0, 61, len.saturating_sub(3)]
                    .into_iter()
                    .flat_map(|f| [(f, false), (f, true)])
                {
                    if from >= len {
                        continue;
                    }
                    let p = rendered(len);
                    let mut f = Fake::new(c, from);
                    let mut st = FeedStats::default();
                    let mut snaps = Vec::new();
                    let r = feed_prompt(&mut f, &p, from, 512, want_token, &mut st, |m, k| {
                        snaps.push((k, m.pos));
                        Ok(None)
                    });
                    let at = format!("c {c} len {len} from {from} token {want_token}");
                    r.unwrap_or_else(|e| panic!("{at}: {e}"));
                    let got: Vec<(usize, u32)> = f.fed.iter().map(|&(i, t, _)| (i, t)).collect();
                    let want: Vec<(usize, u32)> = (from..len).map(|i| (i, p[i])).collect();
                    assert_eq!(got, want, "{at}");
                    let k = snapshot_point(&p, from);
                    assert_eq!(
                        snaps,
                        k.map(|k| (k, k)).into_iter().collect::<Vec<_>>(),
                        "{at}"
                    );
                    assert!(f.chunks.iter().all(|&m| m >= 1 && m <= cap), "{at}");
                    if c == 0 {
                        assert!(f.fed.iter().all(|&(_, _, how)| how == 's'), "{at}");
                        assert_eq!(st.stepped, len - from, "{at}");
                    } else if let Some(k) = k {
                        let stepped_row = f.fed.iter().any(|&(i, _, how)| i < k && how == 's');
                        assert!(!stepped_row, "{at}");
                    }
                }
            }
        }
    }

    /// Feeds `p[from..]` into a fresh fake: what came back, the stats and the fake.
    fn feed(p: &[u32], from: usize, c: usize, want_token: bool) -> (Head, FeedStats, Fake) {
        let mut f = Fake::new(c, from);
        let mut st = FeedStats::default();
        let h = feed_prompt(&mut f, p, from, 512, want_token, &mut st, |_, _| Ok(None))
            .unwrap_or_else(|e| panic!("from {from} c {c} token {want_token}: {e}"));
        (h, st, f)
    }

    /// DRA-87, the single door: `start_greedy`'s feed (`want_token`) gives as the reply's first
    /// token exactly what `start`'s logits give the host's greedy draw (`sample_argmax`, the
    /// first index on a tie; the fake's largest logit is always tied), with the same tokens fed
    /// the same way and the same stats, and reads no logits: the head chunk returns its token
    /// (`T`, never `L`), a short tail returns its last step's token (never `l`). `start` (the
    /// flag off) is unchanged: it reads the logits once, exactly where it did.
    #[test]
    fn greedy_first_token_is_the_host_argmax_without_logits() {
        let cap = q::q36::PREFILL_MAX_ROWS;
        let min = q::GDN_PF_MIN_T;
        let headerless =
            |len: usize| -> Vec<u32> { (0..len as u32).map(|i| 3000 + i * 7).collect() };
        let mut prompts: Vec<(String, Vec<u32>)> = Vec::new();
        for len in [
            1,
            2,
            min - 1,
            min,
            min + 1,
            40,
            cap - 1,
            cap,
            cap + 1,
            2 * cap + 3,
        ] {
            prompts.push((format!("rendered {len}"), rendered(len)));
            prompts.push((format!("headerless {len}"), headerless(len)));
        }
        let mut ties = 0;
        for c in [cap, 0] {
            for (name, p) in &prompts {
                for from in [0, 61, p.len().saturating_sub(3)] {
                    if from >= p.len() {
                        continue;
                    }
                    let at = format!("{name} c {c} from {from}");
                    let (off, st_off, f_off) = feed(p, from, c, false);
                    let (on, st_on, f_on) = feed(p, from, c, true);
                    let Head::Logits(logits) = off else {
                        panic!("{at}: start gave a token")
                    };
                    let Head::Token(tok) = on else {
                        panic!("{at}: start_greedy gave logits")
                    };
                    let host = crate::tensor::sample_argmax(&logits).0;
                    assert_eq!(tok, host, "{at}");
                    let top = logits.iter().cloned().fold(f32::MIN, f32::max);
                    if logits.iter().filter(|&&x| x == top).count() > 1 {
                        ties += 1;
                    }
                    assert_eq!(f_on.fed, f_off.fed, "{at}");
                    assert_eq!(f_on.chunks, f_off.chunks, "{at}");
                    assert_eq!(
                        (st_on.stepped, &st_on.prefill),
                        (st_off.stepped, &st_off.prefill),
                        "{at}"
                    );
                    assert!(
                        f_on.heads.iter().all(|&h| h == 'T'),
                        "{at}: {:?}",
                        f_on.heads
                    );
                    assert_eq!(f_off.heads.len(), 1, "{at}: {:?}", f_off.heads);
                    assert_eq!(f_on.heads.len(), usize::from(f_off.heads[0] == 'L'), "{at}");
                }
            }
        }
        assert!(ties > 0, "the cases must hold ties at the largest logit");
    }
}
