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

pub type SequenceId = u64;
pub type Ticks = u64;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Priority {
    Background = 0,
    Normal = 1,
    Interactive = 2,
    Realtime = 3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelHandle(pub u32);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvHandle(pub u64);

/// The fundamental scheduling unit of SparkOS: tokens are the workload, not processes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InferenceWork {
    pub sequence: SequenceId,
    pub model: ModelHandle,
    pub kv: KvHandle,
    pub priority: Priority,
    pub deadline: Option<Ticks>,
    pub branch_parent: Option<SequenceId>,
    pub next_token_budget: u32,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QueueError {
    Full,
    Empty,
    InvalidSequence(SequenceId),
}

impl core::fmt::Display for QueueError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Full => write!(f, "WorkQueue error: queue capacity exceeded"),
            Self::Empty => write!(f, "WorkQueue error: queue empty"),
            Self::InvalidSequence(id) => write!(f, "WorkQueue error: invalid sequence id {}", id),
        }
    }
}

pub trait WorkQueue {
    fn push(&self, work: InferenceWork) -> Result<(), QueueError>;
    fn pop(&self) -> Option<InferenceWork>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}
