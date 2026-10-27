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

//! The bridge's time rules under short injected budgets, driven through
//! `ComposeBridge::run_task` (tests/support/compose_harness.rs).
//!
//! Own test binary: it sets the process-global budget variables. Every test
//! takes the compose-home slot first and sets and clears the variables while
//! holding it; the bridge reads them when it opens its home, which also
//! happens under the slot.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use compose_harness::{reply, Asked, Harness, Script};
use std::sync::Arc;
use std::time::Duration;

const EDIT_BUDGET_ENV: &str = "AIEN_COMPOSE_EDIT_BUDGET_MS";
const DOC_BUDGET_ENV: &str = "AIEN_COMPOSE_DOC_BUDGET_MS";

/// Sets the budget variables for one bridge and clears them when dropped.
struct Budgets;

impl Budgets {
    fn set(edit_ms: Option<u64>, doc_ms: Option<u64>) -> Self {
        for (k, v) in [(EDIT_BUDGET_ENV, edit_ms), (DOC_BUDGET_ENV, doc_ms)] {
            match v {
                Some(ms) => std::env::set_var(k, ms.to_string()),
                None => std::env::remove_var(k),
            }
        }
        Budgets
    }
}

impl Drop for Budgets {
    fn drop(&mut self) {
        std::env::remove_var(EDIT_BUDGET_ENV);
        std::env::remove_var(DOC_BUDGET_ENV);
    }
}

/// A harness whose bridge opened its home under `budgets` (the variables are
/// read once, at the first run).
fn harness(budgets: Budgets, script: Script) -> (Budgets, Harness) {
    (budgets, Harness::new(script))
}

// NEG (corrective_retry_test): the budget rule is unchanged; attempt 2 does
// not start when less than the per-attempt budget is left, feedback or not.
// Document budget 1000 ms: the per-attempt budget is half of it, 500 ms.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn budget_rule_is_unchanged() {
    let _home = home_guard::home_slot();
    let (_b, h) = harness(
        Budgets::set(None, Some(1000)),
        Arc::new(|_, _, _| {
            std::thread::sleep(Duration::from_millis(600));
            reply("no header")
        }),
    );
    let r = h.run("write a short note into NOTES.md").unwrap();
    assert_eq!(r.proposal_attempts.len(), 1, "{r:?}");
    assert!(
        r.proposer_error
            .as_deref()
            .unwrap()
            .contains("no attempt 2"),
        "{r:?}"
    );
}

// compose_task_test::retry_is_capped_and_recorded, its budget part: attempt 2
// does not start when less than the per-attempt budget is left. Document
// budget 1000 ms: the per-attempt budget is half of it, 500 ms.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn retry_is_capped_and_recorded() {
    let _home = home_guard::home_slot();
    let (_b, h) = harness(
        Budgets::set(None, Some(1000)),
        Arc::new(|_, _, _| {
            std::thread::sleep(Duration::from_millis(600));
            reply("Sure!")
        }),
    );
    let r = h.run("write a short note into NOTES.md").unwrap();
    assert_eq!(h.calls().len(), 1);
    assert_eq!(r.proposal_attempts.len(), 1, "{r:?}");
    assert!(
        r.proposer_error
            .as_deref()
            .unwrap()
            .contains("no attempt 2"),
        "{r:?}"
    );
}

// compose_task_test::kind_selects_budget, its per-attempt part: an edit keeps
// the measured 12 s attempt budget whatever its task budget; a document needs
// about the shortest measured full document (30 s), capped at half a small
// custom budget. The 30 s cap takes about 32 s of model time to show.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn kind_selects_budget() {
    let _home = home_guard::home_slot();
    let refused = |sleep_ms: u64| -> Script {
        Arc::new(move |_, _, _| {
            std::thread::sleep(Duration::from_millis(sleep_ms));
            reply("Sure!")
        })
    };
    // edit, budget 1 s: 12 000 ms per attempt (not half the budget), so no attempt 2
    {
        let (_b, h) = harness(Budgets::set(Some(1000), None), refused(0));
        h.write("notes.md", "# notes\n");
        let r = h.run("add a line to notes.md").unwrap();
        assert_eq!(h.calls().len(), 1, "{r:?}");
        assert_eq!(h.calls()[0].asked, Asked::Edit);
        let why = r.proposer_error.unwrap();
        assert!(
            why.contains("no attempt 2") && why.contains("< 12000 ms per-attempt budget"),
            "{why}"
        );
    }
    // document, budget 1 s: half of it, 500 ms
    {
        let (_b, h) = harness(Budgets::set(None, Some(1000)), refused(600));
        let r = h.run("write a new file faq.md").unwrap();
        assert_eq!(h.calls().len(), 1, "{r:?}");
        assert_eq!(h.calls()[0].asked, Asked::Document);
        let why = r.proposer_error.unwrap();
        assert!(
            why.contains("no attempt 2") && why.contains("< 500 ms per-attempt budget"),
            "{why}"
        );
    }
    // document, budget 61 s: 30 000 ms (the cap), not half the budget (30 500 ms)
    {
        let (_b, h) = harness(Budgets::set(None, Some(61_000)), refused(31_500));
        let r = h.run("write a new file faq.md").unwrap();
        assert_eq!(h.calls().len(), 1, "{r:?}");
        assert_eq!(h.calls()[0].asked, Asked::Document);
        let why = r.proposer_error.unwrap();
        assert!(
            why.contains("no attempt 2") && why.contains("< 30000 ms per-attempt budget"),
            "{why}"
        );
    }
}

// requirements_test::deadline_is_shared_across_attempts. No tight margins:
// every assertion is a one-sided bound that sleeping longer than planned (a
// loaded machine) can only make easier to meet, except that attempt 1 must end
// within 1 000 ms. Document budget 2000 ms: the per-attempt budget is 1000 ms.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn deadline_is_shared_across_attempts() {
    let _home = home_guard::home_slot();
    let doc1 = "filename: DOC.md\nline 0\n";
    let (_b, h) = harness(
        Budgets::set(None, Some(2000)),
        Arc::new(move |_, _, _| {
            std::thread::sleep(Duration::from_millis(600));
            reply(doc1)
        }),
    );
    let r = h.run("write DOC.md in at least 20 lines").unwrap();
    assert!(r.proposal.is_none(), "{r:?}");
    let seen: Vec<Duration> = h.calls().iter().map(|c| c.limit).collect();
    let a = &r.proposal_attempts;
    assert_eq!(seen.len(), a.len());
    // A per-attempt reset would allow all 3 attempts. One shared deadline
    // stops attempt k > 1 unless 1000 ms are left, and every earlier attempt
    // took at least 600 ms, so k - 1 <= (2000 - 1000) / 600: at most 2.
    assert_eq!(a.len(), 2, "attempts {a:?}");
    // Each attempt's limit is what is left of ONE budget: attempt k starts
    // after k - 1 sleeps of at least 600 ms, so it is at least that far
    // below the first limit.
    for k in 1..seen.len() {
        assert!(
            seen[k] + Duration::from_millis(590 * k as u64) <= seen[0],
            "attempt {} limit {:?} vs first {:?}",
            k + 1,
            seen[k],
            seen[0]
        );
    }
    assert!(seen[0] <= Duration::from_millis(2000));
    assert!(a.iter().all(|x| x.outcome == "refused"));
    assert!(r.proposer_error.unwrap().contains("per-attempt budget"));
}
