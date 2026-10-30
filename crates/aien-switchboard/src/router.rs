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

//! The routing decision: pure, deterministic, no clock reads, no I/O.

use crate::account::AccountId;
use crate::gauge::Gauges;
use crate::job::Job;
use crate::rules::RuleTable;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decision {
    pub account: AccountId,
    /// Plain-words explanation, safe to show the operator.
    pub reason: String,
    /// Later accounts in the same route that also have room, in order. The
    /// local account is always the last entry unless it was chosen.
    pub alternates: Vec<AccountId>,
}

/// Why an account has no room right now, or `None` if it has room.
fn no_room(id: &AccountId, rules: &RuleTable, gauges: &Gauges, now: u64) -> Option<String> {
    let g = gauges.get(id)?;
    if let Some(t) = g.cooled_until.filter(|t| *t > now) {
        return Some(format!(
            "{id} is cooling down after hitting its limit (back in {} s)",
            t - now
        ));
    }
    g.full_window(rules.ceiling_percent(), now).map(|w| {
        format!(
            "{id} is {:.0}% used in its {} window (ceiling {:.0}%)",
            w.used_percent,
            w.name,
            rules.ceiling_percent()
        )
    })
}

/// Pick the first account in the job's route that has room. The local
/// account is the last entry of every route and is never skipped, so a
/// decision always exists and AIEN works offline.
pub fn route(job: &Job, rules: &RuleTable, gauges: &Gauges, now: u64) -> Decision {
    let order = rules.route_for(job.kind);
    let mut skipped: Vec<String> = Vec::new();
    let mut chosen: Option<(usize, bool)> = None; // (index, had room)

    for (i, id) in order.iter().enumerate() {
        if id == rules.local() {
            chosen = Some((i, false));
            break;
        }
        match no_room(id, rules, gauges, now) {
            Some(why) => skipped.push(why),
            None => {
                chosen = Some((i, true));
                break;
            }
        }
    }
    // The route always ends with local, so `chosen` is always set.
    let (idx, _) = chosen.unwrap_or((order.len() - 1, false));
    let account = order[idx].clone();

    let mut reason = String::new();
    if account == *rules.local() {
        if skipped.is_empty() {
            reason.push_str(&format!(
                "{account} is the first choice for {} jobs.",
                job.kind
            ));
        } else {
            reason.push_str(&format!(
                "No outside account has room for this {} job, so it stays on the local model {account}. ",
                job.kind
            ));
            reason.push_str(&format!("Skipped: {}.", skipped.join("; ")));
        }
    } else {
        reason.push_str(&format!(
            "{account} is the first account in the {} route with room.",
            job.kind
        ));
        if !skipped.is_empty() {
            reason.push_str(&format!(" Skipped: {}.", skipped.join("; ")));
        }
    }

    let mut alternates = Vec::new();
    for id in &order[idx + 1..] {
        if id == rules.local() || no_room(id, rules, gauges, now).is_none() {
            alternates.push(id.clone());
        }
    }

    Decision {
        account,
        reason,
        alternates,
    }
}
