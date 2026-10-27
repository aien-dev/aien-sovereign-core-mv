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

//! [`OmegeseConstraint`]: the chat framing (forced fence and header) and the per-step legal-token
//! mask from an [`ItemOracle`], as a [`Constraint`] for the generation controller.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aien_inference_abi::generation::{Constraint, StepRule};

use crate::oracle::{Answer, ItemOracle};
use crate::signature::Signature;
use crate::table::{Shape, TokenTable};
use crate::word::{current_word, ItemSet};

/// How the reply is framed and which parts are forced or masked. Each switch is independent, so
/// an experiment can run, for example, an unmasked arm with only the forced fence.
#[derive(Debug, Clone)]
pub struct Framing {
    /// Forced first (e.g. "```\n"). `None`: no fence, and `eof` maps to the stop token alone.
    pub open_fence: Option<String>,
    /// Forced after the fence: the entry function's header line.
    pub header: Option<Signature>,
    /// Mask the program with the oracle. Off: after the forced parts every step is `Any`.
    pub mask_body: bool,
    /// Written when the oracle says `eof` and a fence was opened (after a newline if the program
    /// does not end with one), e.g. "```".
    pub close_fence: String,
    /// Ends the reply (`<|im_end|>`): forced after the closing fence.
    pub stop_token: u32,
}

/// Counters for the experiment's records.
#[derive(Debug, Clone, Default)]
pub struct Stats {
    /// Body steps that asked the oracle.
    pub masked_steps: u64,
    /// Steps answered `Force` (prelude, closing, or a mask with one legal token).
    pub forced_steps: u64,
    pub oracle_calls: u64,
    /// Wall time inside oracle calls.
    pub oracle_time: Duration,
    /// Each masked step's wall time: its oracle calls plus the vocabulary scan.
    pub step_times: Vec<Duration>,
    /// Tokens refused only because they cross two word boundaries, summed over masked steps
    /// (counted when the part before their second boundary passed the cheap filter).
    pub multi_boundary_refused: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Phase {
    /// Forcing `prelude[i]`.
    Prelude(usize),
    Body,
    /// Forcing the rest of the closing fence, then the stop token.
    Closing(VecDeque<u32>),
    Done,
}

/// The Omegese guardrail (ADR 0004 section 2.3, Drake's guardrail decision): forced fence, forced
/// header, the body masked to tokens that keep the program a prefix of one `omegese emit`
/// accepts, and `eof` mapped to the closing fence plus the stop token.
///
/// Mapping rule, per masked step: one oracle call on the program so far gives the items. A token
/// of word bytes only is legal when the current word followed by it is still the start of some
/// item. A token with one run of separators (optional word bytes, separators, optional word
/// bytes) is legal when its word bytes before the separators complete an item (or it starts with
/// a separator run whose canonical first separator is printed), each of its separators is offered by the
/// answer for the text before it, and its word bytes after them start an item of the answer
/// after them. Blank runs are read in [`canonical_seps`] form, so all the whitespace tokens share
/// a few answers: one cached oracle call per distinct (word bytes, canonical separators) prefix
/// per step; a lone separator token needs none. A token with two separator runs is refused (sound, slightly incomplete; see
/// [`Stats::multi_boundary_refused`]). Only bytes Omegese can hold (letters, digits, `_ - =`,
/// space, tab, newline and brackets) are ever legal in the program.
pub struct OmegeseConstraint<O: ItemOracle> {
    table: Arc<TokenTable>,
    oracle: O,
    framing: Framing,
    prelude: Vec<u32>,
    header_text: Vec<u8>,
    /// The generated tokens this constraint has accounted for.
    seen: Vec<u32>,
    phase: Phase,
    program: Vec<u8>,
    /// The rule answered for the step whose token arrives next.
    last: Option<StepRule>,
    stats: Stats,
}

/// A byte Omegese text may hold (omegese.c refuses every other one).
fn omegese_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric()
        || matches!(b, b'_' | b'-' | b'=' | b' ' | b'\t' | b'\n' | b'(' | b')')
}

impl<O: ItemOracle> OmegeseConstraint<O> {
    pub fn new(table: Arc<TokenTable>, oracle: O, framing: Framing) -> Result<Self, String> {
        if table.text(framing.stop_token).is_some() {
            return Err(format!(
                "the stop token {} is ordinary text, not a special token",
                framing.stop_token
            ));
        }
        if framing.stop_token as usize >= table.len() {
            return Err(format!(
                "the stop token {} is outside the vocabulary of {}",
                framing.stop_token,
                table.len()
            ));
        }
        if framing.open_fence.is_some() && !framing.close_fence.contains('`') {
            return Err("the closing fence must contain a backtick, the eof marker".into());
        }
        let mut prelude = Vec::new();
        if let Some(f) = &framing.open_fence {
            prelude.extend(table.encode(f)?);
        }
        let header_text = framing
            .header
            .as_ref()
            .map(Signature::header)
            .unwrap_or_default();
        if !header_text.is_empty() {
            prelude.extend(table.encode(&header_text)?);
        }
        let mut c = Self {
            table,
            oracle,
            framing,
            prelude,
            header_text: header_text.into_bytes(),
            seen: Vec::new(),
            phase: Phase::Done,
            program: Vec::new(),
            last: None,
            stats: Stats::default(),
        };
        c.restart();
        if c.framing.mask_body && !c.header_text.is_empty() {
            // Refuse a header the oracle refuses now, before the fence and header are written.
            let h = c.header_text.clone();
            if let Answer::Refused(line) = c.query(&h)? {
                return Err(format!(
                    "the oracle refuses the header ({line}): {:?}",
                    String::from_utf8_lossy(&h)
                ));
            }
        }
        Ok(c)
    }

    /// The program text written so far (the header included, the fences not).
    pub fn program(&self) -> &[u8] {
        &self.program
    }

    pub fn stats(&self) -> &Stats {
        &self.stats
    }

    pub fn oracle(&self) -> &O {
        &self.oracle
    }

    /// True once the reply is complete: only the stop token is left to force (the controller
    /// consumes it without showing it back) or it was emitted. Without a fence the stop token is
    /// usually one choice among others at eof, and a consumed stop looks the same to the
    /// constraint as a cut by length after another token; there completion is
    /// `FinishReason::Stop`, and this is true only when the stop was forced.
    pub fn finished(&self) -> bool {
        let stop = self.framing.stop_token;
        match &self.phase {
            Phase::Done => true,
            Phase::Closing(q) => q.len() == 1 && q[0] == stop,
            Phase::Body if self.framing.mask_body && self.framing.open_fence.is_none() => {
                match &self.last {
                    Some(StepRule::Force(t)) => *t == stop,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    fn restart(&mut self) {
        self.seen.clear();
        self.last = None;
        self.program = self.header_text.clone();
        self.phase = if self.prelude.is_empty() {
            Phase::Body
        } else {
            Phase::Prelude(0)
        };
    }

    /// The text `eof` writes before the stop token.
    fn closing_text(&self) -> Vec<u8> {
        if self.framing.open_fence.is_none() {
            return Vec::new();
        }
        let mut c = Vec::new();
        if self.program.last().is_some_and(|&b| b != b'\n') {
            c.push(b'\n');
        }
        c.extend_from_slice(self.framing.close_fence.as_bytes());
        c
    }

    /// The closing tokens after `written` bytes of the closing text, then the stop token.
    fn closing_queue(&self, written: usize) -> Result<VecDeque<u32>, String> {
        let c = self.closing_text();
        let mut q = VecDeque::new();
        if written < c.len() {
            let rest = std::str::from_utf8(&c[written..]).map_err(|e| e.to_string())?;
            q.extend(self.table.encode(rest)?);
        }
        q.push_back(self.framing.stop_token);
        Ok(q)
    }

    /// Accounts for the token the controller emitted after the last rule.
    fn advance(&mut self, tok: u32) -> Result<(), String> {
        let ok = match &self.last {
            Some(StepRule::Force(t)) => *t == tok,
            Some(StepRule::Allow(set)) => set.binary_search(&tok).is_ok(),
            Some(StepRule::Any) => true,
            Some(StepRule::Finish) | None => false,
        };
        if !ok {
            return Err(format!(
                "token {tok} was emitted against the rule {:?}",
                self.last
            ));
        }
        self.seen.push(tok);
        match &mut self.phase {
            Phase::Prelude(i) => {
                *i += 1;
                if *i == self.prelude.len() {
                    self.phase = Phase::Body;
                }
            }
            Phase::Body if !self.framing.mask_body => {
                if let Some(t) = self.table.text(tok) {
                    self.program.extend_from_slice(t);
                }
            }
            Phase::Body if tok == self.framing.stop_token => self.phase = Phase::Done,
            Phase::Body => {
                let t = self
                    .table
                    .text(tok)
                    .ok_or_else(|| format!("special token {tok} inside the program"))?;
                if t.contains(&b'`') {
                    // The eof marker: the model began the closing fence.
                    let c = self.closing_text();
                    let written = if c.starts_with(t) { t.len() } else { c.len() };
                    self.phase = Phase::Closing(self.closing_queue(written)?);
                } else {
                    self.program.extend_from_slice(t);
                }
            }
            Phase::Closing(q) => {
                q.pop_front();
                if q.is_empty() {
                    self.phase = Phase::Done;
                }
            }
            Phase::Done => return Err(format!("token {tok} after the reply finished")),
        }
        Ok(())
    }

    fn query(&mut self, prefix: &[u8]) -> Result<Answer, String> {
        let t = Instant::now();
        let r = self.oracle.next(prefix);
        self.stats.oracle_calls += 1;
        self.stats.oracle_time += t.elapsed();
        r
    }

    /// The items after `program` + `key` (a key ending in a separator run is in canonical form),
    /// one oracle call per distinct key per step; `None` when the oracle refuses it.
    fn state<'c>(
        &mut self,
        cache: &'c mut HashMap<Vec<u8>, Option<ItemSet>>,
        program: &[u8],
        key: &[u8],
    ) -> Result<Option<&'c ItemSet>, String> {
        if !cache.contains_key(key) {
            let mut p = program.to_vec();
            p.extend_from_slice(key);
            let set = match self.query(&p)? {
                Answer::Items(i) => Some(ItemSet::new(&i, current_word(&p))?),
                Answer::Refused(_) => None,
            };
            cache.insert(key.to_vec(), set);
        }
        Ok(cache[key].as_ref())
    }

    /// One masked step.
    fn mask(&mut self) -> Result<StepRule, String> {
        let t0 = Instant::now();
        let program = self.program.clone();
        let items = match self.query(&program)? {
            Answer::Items(i) => i,
            Answer::Refused(line) => {
                return Err(format!(
                    "the oracle refused the program so far ({line}); program: {:?}",
                    String::from_utf8_lossy(&program)
                ))
            }
        };
        let here = ItemSet::new(&items, current_word(&program))?;
        // Text up to the end of a token's separators -> the items after it (None: refused).
        let mut after: HashMap<Vec<u8>, Option<ItemSet>> = HashMap::new();
        let table = Arc::clone(&self.table);
        let mut allowed = Vec::new();
        let mut multi = 0u64;
        for id in 0..table.len() as u32 {
            let (Some(t), shape) = (table.text(id), table.shape(id)) else {
                continue;
            };
            if !t.iter().all(|&b| omegese_byte(b)) {
                continue;
            }
            match shape {
                Shape::Special => {}
                Shape::Word => {
                    if here.extends(t) {
                        allowed.push(id);
                    }
                }
                Shape::Multi => {
                    let a_end = t.iter().position(|&b| crate::word::is_sep(b)).unwrap_or(0);
                    let a = &t[..a_end];
                    if (a.is_empty() && here.sep_offered(t[0]))
                        || (!a.is_empty() && here.completes(a))
                    {
                        multi += 1;
                    }
                }
                Shape::OneBoundary { a_end, sep_end } => {
                    let (a, b) = (&t[..a_end], &t[sep_end..]);
                    // Canonical first: a blank run holding a newline reads as that newline, so a
                    // token like ` \n` must find the newline offered, not the space.
                    let seps = canonical_seps(&t[a_end..sep_end]);
                    let first_ok = if a.is_empty() {
                        here.sep_offered(seps[0])
                    } else {
                        here.completes(a)
                    };
                    if !first_ok {
                        continue;
                    }
                    if a.is_empty() && seps.len() == 1 && b.is_empty() {
                        // The separator's own item: exact by section 6.1.
                        allowed.push(id);
                        continue;
                    }
                    // Walk the separators one at a time: each must be offered by the state
                    // before it (the state at the program itself is `here`).
                    let mut key = a.to_vec();
                    let mut ok = true;
                    for (i, &c) in seps.iter().enumerate() {
                        if !(a.is_empty() && i == 0) {
                            let st = self.state(&mut after, &program, &key)?;
                            if !st.is_some_and(|s| s.sep_offered(c)) {
                                ok = false;
                                break;
                            }
                        }
                        key.push(c);
                    }
                    if !ok {
                        continue;
                    }
                    if b.is_empty()
                        || self
                            .state(&mut after, &program, &key)?
                            .is_some_and(|s| s.extends(b))
                    {
                        allowed.push(id);
                    }
                }
            }
        }
        let mut rule = None;
        if here.eof() {
            match &self.framing.open_fence {
                None => allowed.push(self.framing.stop_token),
                Some(_) => {
                    let c = self.closing_text();
                    for id in 0..table.len() as u32 {
                        let Some(t) = table.text(id) else { continue };
                        let fits = c.starts_with(t)
                            || t.strip_prefix(c.as_slice())
                                .is_some_and(|r| r.iter().all(u8::is_ascii_whitespace));
                        if fits && t.contains(&b'`') {
                            allowed.push(id);
                        }
                    }
                    if allowed.is_empty() {
                        // Eof is the only item and no token both starts the closing text and
                        // marks it with a backtick: force the closing. (When some body token is
                        // legal but no marking token fits, eof waits for a later step.)
                        let q = self.closing_queue(0)?;
                        let first = q[0];
                        self.phase = Phase::Closing(q);
                        rule = Some(StepRule::Force(first));
                    }
                }
            }
        }
        allowed.sort_unstable();
        allowed.dedup();
        self.stats.masked_steps += 1;
        self.stats.multi_boundary_refused += multi;
        self.stats.step_times.push(t0.elapsed());
        if let Some(r) = rule {
            return Ok(r);
        }
        match allowed.as_slice() {
            [] => Err(format!(
                "no token is legal after {:?} although the oracle printed {} items (a single-byte \
                 token should always continue an item)",
                String::from_utf8_lossy(&program),
                items.len()
            )),
            [t] => Ok(StepRule::Force(*t)),
            _ => Ok(StepRule::Allow(allowed)),
        }
    }
}

impl<O: ItemOracle> Constraint for OmegeseConstraint<O> {
    fn rule(&mut self, _prompt: &[u32], generated: &[u32]) -> Result<StepRule, String> {
        if generated.is_empty() {
            self.restart();
        } else if generated.len() == self.seen.len() + 1 && generated.starts_with(&self.seen) {
            self.advance(generated[generated.len() - 1])?;
        } else {
            return Err(format!(
                "the constraint saw {} tokens and is now shown {} that do not extend them",
                self.seen.len(),
                generated.len()
            ));
        }
        let rule = match &self.phase {
            Phase::Prelude(i) => StepRule::Force(self.prelude[*i]),
            Phase::Body if !self.framing.mask_body => StepRule::Any,
            Phase::Body => self.mask()?,
            Phase::Closing(q) => StepRule::Force(q[0]),
            Phase::Done => StepRule::Finish,
        };
        if matches!(rule, StepRule::Force(_)) {
            self.stats.forced_steps += 1;
        }
        self.last = Some(rule.clone());
        Ok(rule)
    }
}

/// A separator run with every maximal run of blanks (space, tab, newline) written as one
/// newline if it holds one, else one space; brackets kept. Blank space between words is free in
/// Omegese and blank lines are empty lines, so the text after either form has the same items
/// (pinned against the real oracle by `real_oracle_blank_runs_are_interchangeable`).
pub fn canonical_seps(seps: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(seps.len());
    let mut blank: Option<u8> = None;
    for &c in seps {
        match c {
            b' ' | b'\t' => {
                blank.get_or_insert(b' ');
            }
            b'\n' => blank = Some(b'\n'),
            _ => {
                out.extend(blank.take());
                out.push(c);
            }
        }
    }
    out.extend(blank);
    out
}

#[cfg(test)]
mod tests {
    use super::canonical_seps;

    #[test]
    fn blank_runs_shrink_to_one_byte() {
        for (run, want) in [
            ("  ", " "),
            ("\t \t", " "),
            (" \n\t\n ", "\n"),
            (")\n\n", ")\n"),
            (" ( ", " ( "),
            ("((", "(("),
            ("\t)\t)\n ", " ) )\n"),
        ] {
            assert_eq!(canonical_seps(run.as_bytes()), want.as_bytes(), "{run:?}");
        }
    }
}
