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

//! Route, run, and fail over on a limit hit.

use crate::account::AccountId;
use crate::gauge::Gauges;
use crate::job::Job;
use crate::router::route;
use crate::rules::RuleTable;
use crate::runner::{RunResult, Runner};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Completed {
    pub account: AccountId,
    pub output: String,
    /// Every account tried, in order, ending with `account`.
    pub tried: Vec<AccountId>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DispatchError {
    /// Even the local account reported a limit hit.
    Exhausted { tried: Vec<AccountId> },
}

impl fmt::Display for DispatchError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DispatchError::Exhausted { tried } => {
                let t: Vec<&str> = tried.iter().map(|i| i.as_str()).collect();
                write!(f, "every account is out of room (tried: {})", t.join(", "))
            }
        }
    }
}

impl std::error::Error for DispatchError {}

/// Run `job` on the best account; on a limit hit, cool that account down and
/// route again. Each account is tried at most once, so this always ends.
pub fn run_job(
    job: &Job,
    prompt: &str,
    rules: &RuleTable,
    gauges: &mut Gauges,
    runner: &mut dyn Runner,
    now: u64,
) -> Result<Completed, DispatchError> {
    let mut tried: Vec<AccountId> = Vec::new();
    loop {
        let d = route(job, rules, gauges, now);
        if tried.contains(&d.account) {
            return Err(DispatchError::Exhausted { tried });
        }
        tried.push(d.account.clone());
        // `route` only returns ids from the rule table.
        let account = match rules.account(&d.account) {
            Some(a) => a,
            None => return Err(DispatchError::Exhausted { tried }),
        };
        match runner.run(account, job, prompt) {
            RunResult::Output(output) => {
                return Ok(Completed {
                    account: d.account,
                    output,
                    tried,
                })
            }
            RunResult::LimitHit { resets_at } => {
                gauges.on_limit_hit(&d.account, resets_at, now, rules.default_backoff_secs());
                if d.account == *rules.local() {
                    return Err(DispatchError::Exhausted { tried });
                }
            }
        }
    }
}
