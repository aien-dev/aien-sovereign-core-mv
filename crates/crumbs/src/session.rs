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

//! A curriculum session: the sealed side driving one learner process.
//!
//! For every crumb: generate (visible + sealed), present the visible bytes,
//! evaluate each submission against the sealed held-outs, derive the trace,
//! climb the promotion ladder, admit operations to the learner (only when the
//! condition allows reuse), and append a Crumbline ledger record.

use crate::digest::{digest, CrumbOccurrenceId, Digest, Domain};
use crate::gen::{self, FamilySpec};
use crate::ledger::{CrumblineRecord, Ledger};
use crate::promotion::{Admission, Ladder, PromotionConfig};
use crate::protocol::{LearnerProcess, LearnerStats, ProtocolError, SearchConfig, SUBMIT_ROBUST};
use crate::provenance::Lane;
use crate::sealed::{DecoyStatus, Population, SealedCrumb};
use crate::trace::{self, TraceSummary};
use crate::verify::{evaluate, verifier_digest, EvalClass, SealedEvaluation, Verdict};
use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::PathBuf;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SessionConfig {
    pub condition: String,
    pub learner: String,
    pub run_id: String,
    pub search: SearchConfig,
    /// Admitted operations are sent to the learner only when true.
    pub library_enabled: bool,
    pub promotion: PromotionConfig,
    pub record_transcript: bool,
    /// Directory for the trace corpus, ledger and evidence (None = in memory only).
    pub out_dir: Option<PathBuf>,
}

/// Per-crumb outcome (sealed-side bookkeeping; never sent to a learner).
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct CrumbOutcome {
    pub seq: u64,
    pub phase: String,
    pub family_id: u32,
    pub family_key: String,
    pub mechanism: String,
    pub population: Population,
    pub decoy: DecoyStatus,
    pub crumb_digest: Digest,
    pub supported: bool,
    pub accepted: bool,
    pub verified: bool,
    pub ambiguous_accept: bool,
    pub evaluations: Vec<SealedEvaluation>,
    pub stats: LearnerStats,
    pub trace: TraceSummary,
    pub library_ops_available: u32,
    /// The operation bank the learner declared for this crumb (learner-neutral CPG1).
    pub declared_bank: Vec<crate::program::Program>,
    pub admissions_after: Vec<Admission>,
    pub ledger_root: Digest,
}

impl CrumbOutcome {
    pub fn falsified_hidden(&self) -> usize {
        self.evaluations
            .iter()
            .filter(|e| e.class == EvalClass::FalsifiedHidden)
            .count()
    }
    pub fn first_class(&self) -> Option<EvalClass> {
        self.evaluations.first().map(|e| e.class)
    }
    pub fn reused_library(&self) -> bool {
        self.accepted && !self.trace.library_refs_in_solution.is_empty()
    }
}

pub struct Session {
    pub cfg: SessionConfig,
    learner: Option<LearnerProcess>,
    pub ladder: Ladder,
    pub ledger: Ledger,
    pub outcomes: Vec<CrumbOutcome>,
    pub learner_digest: Digest,
    seq: u64,
    admissions_frozen: bool,
    trace_chain: Digest,
    sample_chain: Digest,
}

fn file_digest(path: &str) -> Digest {
    std::fs::read(path)
        .map(|b| digest(Domain::Config, &b))
        .unwrap_or(Digest::ZERO)
}

impl Session {
    pub fn start(cfg: SessionConfig) -> Result<Self, ProtocolError> {
        let mut learner = LearnerProcess::spawn(&cfg.learner, &[], cfg.record_transcript)?;
        learner.hello(&cfg.search, cfg.library_enabled)?;
        if let Some(d) = &cfg.out_dir {
            std::fs::create_dir_all(d).map_err(|_| ProtocolError::Io)?;
        }
        Ok(Session {
            learner_digest: file_digest(&cfg.learner),
            ladder: Ladder::new(cfg.promotion.clone()),
            ledger: Ledger::new(Lane::Main, &cfg.run_id),
            learner: Some(learner),
            cfg,
            outcomes: vec![],
            seq: 0,
            admissions_frozen: false,
            trace_chain: Digest::ZERO,
            sample_chain: Digest::ZERO,
        })
    }

    pub fn learner_pid(&self) -> Option<u32> {
        self.learner.as_ref().map(|l| l.pid())
    }

    pub fn transcript(&self) -> Option<&[u8]> {
        self.learner.as_ref().and_then(|l| l.transcript.as_deref())
    }

    /// Stop admitting operations (used to freeze the library for evaluation).
    pub fn freeze_admissions(&mut self) {
        self.admissions_frozen = true;
    }

    fn append_file(&self, name: &str, bytes: &[u8]) {
        if let Some(d) = &self.cfg.out_dir {
            if let Ok(mut f) = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(d.join(name))
            {
                let _ = f.write_all(bytes);
            }
        }
    }

    pub fn present(
        &mut self,
        fam: &FamilySpec,
        seed: u64,
        population: Population,
        phase: &str,
    ) -> Result<CrumbOutcome, ProtocolError> {
        let g = gen::generate(fam, seed);
        self.present_sealed(&g.visible, &g.sealed, population, phase)
    }

    /// Present a crumb from an explicit visible/sealed pair (used by gates that
    /// construct canary sealed records).
    pub fn present_with(
        &mut self,
        visible: &crate::visible::VisibleCrumb,
        sealed: &SealedCrumb,
    ) -> Result<CrumbOutcome, ProtocolError> {
        self.present_sealed(visible, sealed, Population::Fundamental, "gates")
    }

    fn present_sealed(
        &mut self,
        visible: &crate::visible::VisibleCrumb,
        sealed: &SealedCrumb,
        population: Population,
        phase: &str,
    ) -> Result<CrumbOutcome, ProtocolError> {
        let seq = self.seq;
        self.seq += 1;
        let max_q = self.cfg.search.max_oracle_queries as usize;
        let mut evaluations: Vec<SealedEvaluation> = Vec::new();
        let learner = self.learner.as_mut().ok_or(ProtocolError::Io)?;
        let ladder = &mut self.ladder;
        let session = learner.run_crumb(visible, &mut |sub| {
            if evaluations.len() >= max_q {
                return Verdict::BudgetExhausted;
            }
            let ev = evaluate(
                visible,
                sealed,
                &sub.program,
                sub.hypotheses,
                sub.flags & SUBMIT_ROBUST != 0,
            );
            match ev.class {
                EvalClass::Verified => ladder.observe_accept(seq, sealed, &ev, &sub.program),
                EvalClass::FalsifiedHidden => ladder.observe_reject(&ev, &sub.program),
                _ => {}
            }
            let v = ev.verdict;
            evaluations.push(ev);
            v
        })?;

        let tr = trace::derive(visible, &session);
        let mut rec_chain = self.trace_chain;
        let tbytes = trace::encode_records(&tr.records, &mut rec_chain);
        let mut s_chain = self.sample_chain;
        let sbytes = trace::encode_samples(&tr.samples, &mut s_chain);
        self.trace_chain = rec_chain;
        self.sample_chain = s_chain;
        self.append_file("trace.ctr", &tbytes);
        self.append_file("samples.cts", &sbytes);

        let admissions = if self.admissions_frozen {
            vec![]
        } else {
            self.ladder.advance(seq)
        };
        if self.cfg.library_enabled {
            let learner = self.learner.as_mut().ok_or(ProtocolError::Io)?;
            for a in &admissions {
                learner.admit(a.op_ref, a.scope_bits, &a.program)?;
            }
        }

        let summary = tr.summary.clone().expect("derive always summarizes");
        let rec = CrumblineRecord {
            seq,
            lane: Lane::Main,
            occurrence: CrumbOccurrenceId::new(),
            condition: self.cfg.condition.clone(),
            population,
            crumb_digest: visible.digest(),
            generator_instance_digest: sealed.generator_instance_digest,
            sealed_digest: sealed.digest(),
            verifier_digest: verifier_digest(),
            learner_digest: self.learner_digest,
            evaluations: evaluations.iter().map(|e| e.digest()).collect(),
            trace_stream_digest: summary.stream_digest,
            promotions: admissions.iter().map(|a| a.promotion_digest).collect(),
            prev_root: Digest::ZERO,
            root: Digest::ZERO,
        };
        let root = self
            .ledger
            .append(rec, &sealed.provenance)
            .expect("main lane accepts every provenance");
        if let Some(r) = self.ledger.records.last() {
            self.append_file(
                "ledger.jsonl",
                format!("{}\n", serde_json::to_string(r).unwrap()).as_bytes(),
            );
        }
        for e in &evaluations {
            self.append_file(
                "evaluations.jsonl",
                format!("{}\n", serde_json::to_string(e).unwrap()).as_bytes(),
            );
        }

        let accepted = evaluations.iter().any(|e| e.verdict == Verdict::Accept);
        let out = CrumbOutcome {
            seq,
            phase: phase.to_string(),
            family_id: sealed.family_id,
            family_key: sealed.family_key.clone(),
            mechanism: sealed.mechanism.clone(),
            population,
            decoy: sealed.decoy_status,
            crumb_digest: visible.digest(),
            supported: session.stats.outcome != crate::protocol::OUTCOME_UNSUPPORTED,
            accepted,
            verified: evaluations.iter().any(|e| e.class == EvalClass::Verified),
            ambiguous_accept: evaluations
                .iter()
                .any(|e| e.class == EvalClass::AmbiguousUnderdetermined),
            library_ops_available: session.bank.iter().filter(|b| b.origin == 1).count() as u32,
            declared_bank: session.bank.iter().map(|b| b.program.clone()).collect(),
            evaluations,
            stats: session.stats,
            trace: summary,
            admissions_after: admissions,
            ledger_root: root,
        };
        self.outcomes.push(out.clone());
        Ok(out)
    }

    pub fn finish(mut self) -> Result<(Ledger, Vec<CrumbOutcome>, Ladder), ProtocolError> {
        if let Some(l) = self.learner.take() {
            l.shutdown()?;
        }
        if let Some(d) = &self.cfg.out_dir {
            let _ = std::fs::write(
                d.join("promotion.json"),
                serde_json::to_string_pretty(&self.ladder.ops.values().collect::<Vec<_>>())
                    .unwrap(),
            );
        }
        Ok((self.ledger, self.outcomes, self.ladder))
    }
}
