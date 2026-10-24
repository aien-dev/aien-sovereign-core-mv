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

//! Jobs: what the operator wants done.

use serde::Deserialize;
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum JobKind {
    Code,
    Review,
    Research,
}

impl JobKind {
    pub const ALL: [JobKind; 3] = [JobKind::Code, JobKind::Review, JobKind::Research];

    pub fn as_str(self) -> &'static str {
        match self {
            JobKind::Code => "code",
            JobKind::Review => "review",
            JobKind::Research => "research",
        }
    }
}

impl fmt::Display for JobKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Rough size of the job. v1 routing does not use it yet; it is carried so
/// the type is stable when size-aware rules arrive.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum JobSize {
    Small,
    Medium,
    Large,
}

/// `needs` (what a job requires of an account) is deliberately absent: no
/// first-party source on this machine says what each program can or cannot
/// do, so adding it now would be a guess. UNKNOWN, tracked for a later cut.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Job {
    pub kind: JobKind,
    pub size: JobSize,
}

impl Job {
    pub fn new(kind: JobKind, size: JobSize) -> Self {
        Job { kind, size }
    }
}
