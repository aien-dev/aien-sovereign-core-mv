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

//! Correlation tap on the scheduler's admit/preempt decision (#398).
//!
//! Implements the scheduler's read-only `DecisionObserver`. For each
//! `AdmitPreemptRecord` it emits `sequence_admitted` for every queue head the
//! scheduler admitted (`AdmittedPrefill` or `AdmittedDecode` in
//! `alternatives.admission_examined`) and `sequence_preempted` for every id in
//! `choice.preempted`. Only sequences registered through
//! [`TraceDecisionObserver::register`] produce events; unknown sequences are
//! skipped without allocating. Observational only: it never changes a
//! scheduling decision, and it copies ids, never prompt or token data.

use aien_scheduler::dual_observer::{AdmissionVerdict, AdmitPreemptRecord, DecisionObserver};
use aien_trace::{
    now_unix_ms, CorrelationIds, EventKind, EventStatus, EvidenceRefs, TraceEvent, TraceId,
    TraceSink,
};
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

pub struct TraceDecisionObserver {
    sink: Arc<dyn TraceSink>,
    /// sequence (packed u64) -> (trace id, parent event id)
    trace_for_sequence: Mutex<HashMap<u64, (TraceId, u64)>>,
}

impl TraceDecisionObserver {
    pub fn new(sink: Arc<dyn TraceSink>) -> Self {
        Self {
            sink,
            trace_for_sequence: Mutex::new(HashMap::new()),
        }
    }

    /// Opt a sequence in. `parent` is the event its scheduler events hang under.
    pub fn register(&self, seq: u64, trace: TraceId, parent: u64) {
        self.trace_for_sequence.lock().insert(seq, (trace, parent));
    }

    /// Forget a sequence (after it finished or was cancelled).
    pub fn unregister(&self, seq: u64) {
        self.trace_for_sequence.lock().remove(&seq);
    }

    fn emit(&self, seq: u64, kind: EventKind, step_id: u64) {
        let Some((trace_id, parent)) = self.trace_for_sequence.lock().get(&seq).copied() else {
            return;
        };
        self.sink.emit(TraceEvent {
            trace_id,
            event_id: self.sink.next_event_id(),
            parent_event_id: Some(parent),
            at_unix_ms: now_unix_ms(),
            kind,
            status: EventStatus::Ok,
            effect_certainty: None,
            ids: CorrelationIds {
                sequence_id: Some(seq),
                step_id: Some(step_id),
                ..Default::default()
            },
            refs: EvidenceRefs::default(),
            note: None,
        });
    }
}

impl DecisionObserver for TraceDecisionObserver {
    fn observe_admit_preempt(&self, record: &AdmitPreemptRecord) {
        if !self.sink.enabled() {
            return;
        }
        for c in &record.alternatives.admission_examined {
            if matches!(
                c.verdict,
                AdmissionVerdict::AdmittedDecode | AdmissionVerdict::AdmittedPrefill { .. }
            ) {
                self.emit(c.seq.to_u64(), EventKind::SequenceAdmitted, record.step_id);
            }
        }
        for s in &record.choice.preempted {
            self.emit(s.to_u64(), EventKind::SequencePreempted, record.step_id);
        }
    }
}
