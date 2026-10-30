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

//! Per-task timing from the scheduler's monotonic clock, for the M6 A/B
//! analysis. Times are clock ticks as returned by the injected clock (the
//! default clock counts nanoseconds). Every value is derived from event
//! times recorded in the journal, so a replay reproduces the trace exactly.

use crate::execution::TaskState;
use serde::Serialize;

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize)]
pub struct TaskTiming {
    /// Ticks spent `Blocked` (including waiting in the ready queue).
    pub blocked: u64,
    /// Ticks spent `Provisional` (running, holding permits).
    pub provisional: u64,
    /// Ticks spent `AwaitingFinalProof`.
    pub awaiting_final_proof: u64,
    pub registered_at: u64,
    pub first_dispatch_at: Option<u64>,
    pub committable_at: Option<u64>,
    pub dispatches: u32,
    pub revocations: u32,
    /// Start of the current state.
    pub since: u64,
}

impl TaskTiming {
    pub fn new(at: u64) -> Self {
        TaskTiming {
            registered_at: at,
            since: at,
            ..TaskTiming::default()
        }
    }

    /// Charge the time since the last transition to `state`.
    pub fn close(&mut self, state: TaskState, at: u64) {
        let spent = at.saturating_sub(self.since);
        match state {
            TaskState::Blocked => self.blocked += spent,
            TaskState::Provisional => self.provisional += spent,
            TaskState::AwaitingFinalProof => self.awaiting_final_proof += spent,
            TaskState::Committable
            | TaskState::Aborted
            | TaskState::Committing
            | TaskState::Committed
            | TaskState::CommitUncertain => {}
        }
        self.since = at;
    }

    /// Totals as of `now`, counting the open interval in the current state.
    pub fn as_of(&self, state: TaskState, now: u64) -> TaskTiming {
        let mut t = self.clone();
        t.close(state, now);
        t
    }
}
