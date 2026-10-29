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

//! The learner protocol (v1): the only channel between the sealed side and a learner.
//!
//! The learner runs as a separate operating-system process. It never shares an
//! address space with sealed data; everything it can know arrives through the
//! frames below, and every frame type is listed here. There is no frame that
//! can carry a sealed field, a held-out example, a label or an explanation.
//!
//! Frame: `type u8 | len u32 | payload`.
//!
//! Sealed side -> learner:
//! - `0x01 HELLO   `: protocol u16, library_enabled u8, SearchConfig (6 x u32)
//! - `0x02 CRUMB   `: canonical Crumb v1 visible bytes
//! - `0x03 VERDICT `: one byte (1 accept, 2 reject, 3 budget exhausted)
//! - `0x04 ADMIT   `: op_ref u32, scope_bits u8, CPG1 program
//! - `0x05 SHUTDOWN`: empty
//!
//! Learner -> sealed side:
//! - `0x81 READY   `: protocol u16
//! - `0x82 BANK    `: n u32, n x (origin u8, op_ref u32, CPG1 program)   (per crumb)
//! - `0x83 SUBMIT  `: node u32, flags u8, hypotheses u8, CPG1 program
//! - `0x84 EVENTS  `: n u32, n x 20-byte search event
//! - `0x85 DONE    `: outcome u8, solution_node u32, 12 x u32 counters, 2 x u64 counters
//!
//! Any other frame from the learner aborts the session; nothing is sent in reply.

use crate::canon::{Dec, DecodeError, Enc};
use crate::program::Program;
use crate::verify::Verdict;
use crate::visible::VisibleCrumb;
use serde::{Deserialize, Serialize};
use std::io::{Read, Write};
use std::process::{Child, ChildStdin, ChildStdout, Command, Stdio};

pub const PROTOCOL_VERSION: u16 = 1;
pub const MAX_FRAME: usize = 1 << 24;

pub mod ft {
    pub const HELLO: u8 = 0x01;
    pub const CRUMB: u8 = 0x02;
    pub const VERDICT: u8 = 0x03;
    pub const ADMIT: u8 = 0x04;
    pub const SHUTDOWN: u8 = 0x05;
    pub const READY: u8 = 0x81;
    pub const BANK: u8 = 0x82;
    pub const SUBMIT: u8 = 0x83;
    pub const EVENTS: u8 = 0x84;
    pub const DONE: u8 = 0x85;
}

/// Search limits handed to the learner. Identical in every experimental condition.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SearchConfig {
    pub max_candidates: u32,
    pub max_depth: u32,
    pub frontier_cap: u32,
    pub max_oracle_queries: u32,
    pub ambiguity_scan: u32,
    pub robust_min_pct: u32,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            max_candidates: 20_000,
            max_depth: 6,
            frontier_cap: 4_000,
            max_oracle_queries: 4,
            ambiguity_scan: 400,
            robust_min_pct: 75,
        }
    }
}

/// Learner search-event kinds.
pub mod ek {
    pub const EXPAND: u8 = 1;
    pub const SUBMIT: u8 = 2;
}

/// One observable search step, as reported by the learner. Numbers only.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearnerEvent {
    pub kind: u8,
    pub prune: u8,
    pub verify: u8,
    pub fit: u8,
    pub op_index: u16,
    pub parent: u32,
    pub child: u32,
    pub exec_cost: u32,
    pub oracle_index: u16,
}

pub const EVENT_BYTES: usize = 20;

impl LearnerEvent {
    fn decode(d: &mut Dec<'_>) -> Result<Self, DecodeError> {
        Ok(LearnerEvent {
            kind: d.u8()?,
            prune: d.u8()?,
            verify: d.u8()?,
            fit: d.u8()?,
            op_index: d.u16()?,
            parent: d.u32()?,
            child: d.u32()?,
            exec_cost: d.u32()?,
            oracle_index: d.u16()?,
        })
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BankEntry {
    /// 0 = learner base vocabulary, 1 = admitted library operation.
    pub origin: u8,
    /// The ADMIT op_ref this entry came from (0 for base operations).
    pub op_ref: u32,
    pub program: Program,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Submission {
    pub node: u32,
    pub flags: u8,
    pub hypotheses: u8,
    pub program: Program,
    pub verdict: Verdict,
}

pub const SUBMIT_ROBUST: u8 = 1;

/// The learner's own end-of-crumb counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct LearnerStats {
    pub outcome: u8,
    pub solution_node: u32,
    pub generated: u32,
    pub evaluated: u32,
    pub expansions: u32,
    pub pruned_equiv: u32,
    pub pruned_cost: u32,
    pub pruned_cap: u32,
    pub visible_fits: u32,
    pub verify_failures: u32,
    pub oracle_queries: u32,
    pub oracle_rejections: u32,
    pub candidates_to_solution: u32,
    pub hypotheses_at_first_submit: u32,
    pub exec_count: u64,
    pub wall_ns: u64,
}

pub const OUTCOME_SOLVED: u8 = 1;
pub const OUTCOME_UNSOLVED: u8 = 2;
pub const OUTCOME_UNSUPPORTED: u8 = 3;

/// Everything the learner produced for one crumb.
#[derive(Clone, Debug, Default)]
pub struct CrumbSession {
    pub bank: Vec<BankEntry>,
    pub events: Vec<LearnerEvent>,
    pub submissions: Vec<Submission>,
    pub stats: LearnerStats,
}

#[derive(Debug)]
pub enum ProtocolError {
    Io,
    Malformed,
    Violation,
    Version,
}

impl std::fmt::Display for ProtocolError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "learner protocol error: {self:?}")
    }
}

impl std::error::Error for ProtocolError {}

impl From<std::io::Error> for ProtocolError {
    fn from(_: std::io::Error) -> Self {
        ProtocolError::Io
    }
}

impl From<DecodeError> for ProtocolError {
    fn from(_: DecodeError) -> Self {
        ProtocolError::Malformed
    }
}

pub fn write_frame(w: &mut dyn Write, t: u8, payload: &[u8]) -> std::io::Result<()> {
    let mut hdr = [0u8; 5];
    hdr[0] = t;
    hdr[1..].copy_from_slice(&(payload.len() as u32).to_le_bytes());
    w.write_all(&hdr)?;
    w.write_all(payload)?;
    w.flush()
}

pub fn read_frame(r: &mut dyn Read) -> Result<(u8, Vec<u8>), ProtocolError> {
    let mut hdr = [0u8; 5];
    r.read_exact(&mut hdr)?;
    let len = u32::from_le_bytes(hdr[1..].try_into().unwrap()) as usize;
    if len > MAX_FRAME {
        return Err(ProtocolError::Malformed);
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf)?;
    Ok((hdr[0], buf))
}

/// A learner running as a child process.
pub struct LearnerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: ChildStdout,
    /// Every byte sent to the learner, kept for leak audits when enabled.
    pub transcript: Option<Vec<u8>>,
}

impl LearnerProcess {
    pub fn spawn(
        path: &str,
        args: &[&str],
        record_transcript: bool,
    ) -> Result<Self, ProtocolError> {
        let mut child = Command::new(path)
            .args(args)
            .env_clear()
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()?;
        let stdin = child.stdin.take().ok_or(ProtocolError::Io)?;
        let stdout = child.stdout.take().ok_or(ProtocolError::Io)?;
        Ok(LearnerProcess {
            child,
            stdin,
            stdout,
            transcript: record_transcript.then(Vec::new),
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    fn send(&mut self, t: u8, payload: &[u8]) -> Result<(), ProtocolError> {
        if let Some(tr) = self.transcript.as_mut() {
            tr.push(t);
            tr.extend_from_slice(&(payload.len() as u32).to_le_bytes());
            tr.extend_from_slice(payload);
        }
        write_frame(&mut self.stdin, t, payload)?;
        Ok(())
    }

    pub fn hello(
        &mut self,
        cfg: &SearchConfig,
        library_enabled: bool,
    ) -> Result<(), ProtocolError> {
        let mut e = Enc::new();
        e.u16(PROTOCOL_VERSION).u8(library_enabled as u8);
        for v in [
            cfg.max_candidates,
            cfg.max_depth,
            cfg.frontier_cap,
            cfg.max_oracle_queries,
            cfg.ambiguity_scan,
            cfg.robust_min_pct,
        ] {
            e.u32(v);
        }
        self.send(ft::HELLO, &e.finish())?;
        let (t, p) = read_frame(&mut self.stdout)?;
        if t != ft::READY {
            return Err(ProtocolError::Violation);
        }
        let mut d = Dec::new(&p);
        if d.u16()? != PROTOCOL_VERSION {
            return Err(ProtocolError::Version);
        }
        Ok(())
    }

    /// Admit a verified library operation (sent only in conditions that allow reuse).
    pub fn admit(
        &mut self,
        op_ref: u32,
        scope_bits: u8,
        program: &Program,
    ) -> Result<(), ProtocolError> {
        let mut e = Enc::new();
        e.u32(op_ref).u8(scope_bits).bytes(&program.to_bytes());
        self.send(ft::ADMIT, &e.finish())
    }

    /// Present one crumb. `oracle` is the sealed evaluator; it sees each
    /// submission and returns the verdict byte the learner will receive.
    pub fn run_crumb(
        &mut self,
        visible: &VisibleCrumb,
        oracle: &mut dyn FnMut(&Submission) -> Verdict,
    ) -> Result<CrumbSession, ProtocolError> {
        self.send(ft::CRUMB, &visible.to_bytes())?;
        let mut s = CrumbSession::default();
        loop {
            let (t, p) = read_frame(&mut self.stdout)?;
            let mut d = Dec::new(&p);
            match t {
                ft::BANK => {
                    let n = d.u32()? as usize;
                    for _ in 0..n.min(4096) {
                        let origin = d.u8()?;
                        let op_ref = d.u32()?;
                        let program = Program::decode(&mut d)?;
                        s.bank.push(BankEntry {
                            origin,
                            op_ref,
                            program,
                        });
                    }
                    d.finish()?;
                }
                ft::EVENTS => {
                    let n = d.u32()? as usize;
                    if n * EVENT_BYTES + 4 != p.len() {
                        return Err(ProtocolError::Malformed);
                    }
                    for _ in 0..n {
                        s.events.push(LearnerEvent::decode(&mut d)?);
                    }
                }
                ft::SUBMIT => {
                    let node = d.u32()?;
                    let flags = d.u8()?;
                    let hypotheses = d.u8()?;
                    let program = Program::decode(&mut d)?;
                    d.finish()?;
                    let mut sub = Submission {
                        node,
                        flags,
                        hypotheses,
                        program,
                        verdict: Verdict::Reject,
                    };
                    sub.verdict = oracle(&sub);
                    self.send(ft::VERDICT, &[sub.verdict as u8])?;
                    s.submissions.push(sub);
                }
                ft::DONE => {
                    let st = &mut s.stats;
                    st.outcome = d.u8()?;
                    st.solution_node = d.u32()?;
                    for f in [
                        &mut st.generated,
                        &mut st.evaluated,
                        &mut st.expansions,
                        &mut st.pruned_equiv,
                        &mut st.pruned_cost,
                        &mut st.pruned_cap,
                        &mut st.visible_fits,
                        &mut st.verify_failures,
                        &mut st.oracle_queries,
                        &mut st.oracle_rejections,
                        &mut st.candidates_to_solution,
                        &mut st.hypotheses_at_first_submit,
                    ] {
                        *f = d.u32()?;
                    }
                    st.exec_count = d.u64()?;
                    st.wall_ns = d.u64()?;
                    d.finish()?;
                    return Ok(s);
                }
                _ => return Err(ProtocolError::Violation),
            }
        }
    }

    pub fn shutdown(mut self) -> Result<std::process::ExitStatus, ProtocolError> {
        let _ = self.send(ft::SHUTDOWN, &[]);
        drop(self.stdin);
        Ok(self.child.wait()?)
    }

    /// Kill without ceremony (used after a protocol violation).
    pub fn kill(mut self) {
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
