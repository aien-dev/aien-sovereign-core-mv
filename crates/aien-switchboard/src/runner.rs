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

//! Runners start one job on one account. This cut ships only the scripted
//! fake; adapters for the real programs come later and must keep this shape.

use crate::account::{Account, AccountId};
use crate::job::Job;
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum RunResult {
    Output(String),
    /// The account hit its usage limit. `resets_at` is Unix seconds when the
    /// program said when it refills, else `None`.
    LimitHit {
        resets_at: Option<u64>,
    },
}

pub trait Runner {
    fn run(&mut self, account: &Account, job: &Job, prompt: &str) -> RunResult;
}

/// Scripted runner for tests: answers per account from a queue, then falls
/// back to a canned output. Records every call.
#[derive(Debug, Default)]
pub struct FakeRunner {
    scripts: BTreeMap<AccountId, VecDeque<RunResult>>,
    pub calls: Vec<AccountId>,
}

impl FakeRunner {
    pub fn new() -> Self {
        FakeRunner::default()
    }

    /// Queue the next answer for `id`.
    pub fn push(&mut self, id: &AccountId, result: RunResult) -> &mut Self {
        self.scripts
            .entry(id.clone())
            .or_default()
            .push_back(result);
        self
    }
}

impl Runner for FakeRunner {
    fn run(&mut self, account: &Account, _job: &Job, prompt: &str) -> RunResult {
        self.calls.push(account.id.clone());
        match self
            .scripts
            .get_mut(&account.id)
            .and_then(|q| q.pop_front())
        {
            Some(r) => r,
            None => RunResult::Output(format!("fake output from {}: {prompt}", account.id)),
        }
    }
}
