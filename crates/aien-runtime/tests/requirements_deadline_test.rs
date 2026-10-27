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

//! A document task whose first attempt fails a stated requirement and whose
//! second attempt passes: both attempts happen inside the ONE per-task document
//! deadline (a short injected budget), and the bytes saved are the bytes approved.
//! Own test binary: it sets a process-global environment variable.
//!
//! Driven through `ComposeBridge::run_task` (tests/support/compose_harness.rs):
//! the limits handed to the document proposer, the attempts and the report.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::check_file_proposal;
use aien_runtime::compose::compose_budgets_from_env;
use aien_runtime::requirements::extract;
use compose_harness::{reply, Asked, Harness};
use std::sync::Arc;
use std::time::{Duration, Instant};

const DOC_BUDGET_ENV: &str = "AIEN_COMPOSE_DOC_BUDGET_MS";

fn doc(n: usize) -> String {
    let mut s = String::from("filename: DOC.md\n");
    for i in 0..n {
        s.push_str(&format!("line {i}\n"));
    }
    s
}

/// Clears the budget variable when dropped (also on a failed assertion).
struct DocBudget;

impl Drop for DocBudget {
    fn drop(&mut self) {
        std::env::remove_var(DOC_BUDGET_ENV);
    }
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn requirement_retry_stays_inside_the_single_document_deadline_and_saves_approved_bytes() {
    let _home = home_guard::home_slot();
    std::env::set_var(DOC_BUDGET_ENV, "6000");
    let _clear = DocBudget;
    let budget = compose_budgets_from_env().unwrap().doc;
    assert_eq!(budget, Duration::from_millis(6000));
    // The per-attempt budget of this document task: min(30 s, 6 s / 2) = 3 s.
    let reqs = extract("write DOC.md in at least 20 lines");
    assert_eq!(reqs.len(), 1);

    // The bridge reads the budget when it opens its home, at the first run.
    let h = Harness::new(Arc::new(|n, _: &str, _: Duration| {
        std::thread::sleep(Duration::from_millis(300));
        // First attempt: 5 lines (unmet). Second: 25 lines.
        reply(&doc(if n == 1 { 5 } else { 25 }))
    }));
    let t0 = Instant::now();
    let r = h.run("write DOC.md in at least 20 lines").unwrap();
    assert!(t0.elapsed() < budget, "both attempts inside one deadline");

    // The shared deadline, observed from the document proposer.
    let c = h.calls();
    assert!(c.iter().all(|c| c.asked == Asked::Document), "{c:?}");
    let limits: Vec<Duration> = c.iter().map(|c| c.limit).collect();
    assert_eq!(limits.len(), 2);
    assert!(limits[0] <= budget, "{limits:?}");
    assert!(
        limits[1] < limits[0],
        "retry gets what is left, not a fresh budget: {limits:?}"
    );
    // Attempt 2 started, so at least the 3 s per-attempt budget was left.
    assert!(limits[1] >= Duration::from_secs(3), "{limits:?}");
    let attempts = &r.proposal_attempts;
    assert_eq!(attempts.len(), 2);
    assert_eq!(attempts[0].outcome, "refused");
    assert_eq!(attempts[0].unmet_requirements.len(), 1);
    assert_eq!(attempts[1].outcome, "parsed");

    // The real run: committed, saved == approved.
    assert!(r.committed, "{r:?}");
    assert_eq!(attempts[1].aegis.as_deref(), Some("pass"));
    let saved = check_file_proposal(&r.proposal.clone().unwrap()).unwrap();
    assert_eq!(
        r.proposal_content_sha256.clone().unwrap(),
        aien_omega_compose::hex(&<sha2::Sha256 as sha2::Digest>::digest(
            saved.content.as_bytes()
        ))
    );
    assert_eq!(
        saved.content,
        doc(25).strip_prefix("filename: DOC.md\n").unwrap()
    );
}
