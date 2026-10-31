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

//! sc#349: `one line saying hello` / `one line that says hello` (no quotes).
//! A short plain phrase that runs to the end of its sentence (or to a trailing
//! destination file) is a required phrase; anything wider is UNCERTAIN, never
//! dropped. The one-line count is kept in every case.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::control::ControlResponse;
use aien_runtime::requirements::{analyze, refusal_reason, Requirement};
use aien_runtime::{ComposeBridge, ComposeProposer, Generation};
use compose_harness::{always, Harness};
use std::sync::Arc;
use std::time::Duration;

use Requirement::*;

fn read(goal: &str) -> Vec<Requirement> {
    let ex = analyze(goal);
    assert!(ex.uncertain.is_empty(), "{goal:?}: {:?}", ex.uncertain);
    ex.requirements
}

fn one_line_with(goal: &str, phrase: &str) {
    let r = read(goal);
    assert!(
        r.iter().any(|q| matches!(
            q,
            AddedLines {
                n: 1,
                exact: true,
                ..
            }
        )),
        "{goal}: {r:?}"
    );
    assert!(
        r.contains(&RequiredPhrases(vec![phrase.to_string()])),
        "{goal}: {r:?}"
    );
}

#[test]
fn a_plain_unquoted_phrase_is_required() {
    one_line_with("Add a single line saying hello to a.md", "hello");
    one_line_with("Create a.txt containing one line that says hello.", "hello");
    one_line_with("Add one line to notes/TODO.md saying buy milk", "buy milk");
    one_line_with("Add one line saying Good Morning.", "Good Morning");
    one_line_with("Add one line to a.md that says hello world!", "hello world");
}

#[test]
fn an_ambiguous_unquoted_phrase_is_uncertain_never_dropped() {
    for goal in [
        "Add a single line saying hello and then save it to a.md",
        "Add a single line saying something nice to a.md",
        "Add one line saying hello, then stop",
        "Add one line saying it to a.md",
        "Create a.txt containing one line that says the project keeps every change inside its workspace.",
        "Add one line saying",
        "Add one line saying save the file to a.md",
        "Add one line saying hello to the end of a.md",
    ] {
        let ex = analyze(goal);
        assert!(!ex.uncertain.is_empty(), "{goal:?} should be uncertain: {:?}", ex.requirements);
    }
}

#[test]
fn unrelated_numbers_and_words_are_unchanged() {
    assert!(read("Fix 2 typos in README.md").is_empty());
    assert!(read("Add a line to a.md").is_empty());
}

#[test]
fn the_check_refuses_a_line_without_the_phrase() {
    let reqs = analyze("Add a single line saying hello to a.md")
        .resolved("")
        .requirements;
    assert!(refusal_reason(&reqs, "goodbye\n").is_some());
    assert!(refusal_reason(&reqs, "hello\n").is_none());
    assert!(refusal_reason(&reqs, "hello\nhello again\n").is_some());
}

#[test]
fn linked_compose_does_not_commit_a_line_without_the_phrase() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let run = |reply: &'static str, sub: &str| {
        let proposer: ComposeProposer = Arc::new(move |_: &str, _: Duration| {
            Ok(Generation {
                text: reply.to_string(),
                tokens: 5,
                finish_reason: Some("eos".into()),
                ..Default::default()
            })
        });
        let bridge = ComposeBridge::new(tmp.path().join(sub), proposer, "test:said");
        bridge.run_task(
            "Create a.md containing one line saying hello.",
            ws.to_str().unwrap(),
        )
    };
    let resp = run("filename: a.md\ngoodbye\n", "c1");
    if !aien_omega_compose::LINKED {
        assert!(matches!(resp, ControlResponse::Error(_)));
        return;
    }
    let ControlResponse::ComposeTaskResult(r) = resp else {
        panic!("{resp:?}")
    };
    assert!(!r.committed, "{r:?}");
    assert!(r.proposal.is_none());
    assert!(
        r.proposer_error
            .as_deref()
            .is_some_and(|e| e.contains("hello")),
        "{r:?}"
    );
    assert_eq!(r.proposal_attempts.len(), 3, "{:?}", r.proposal_attempts);
    assert!(!ws.join("a.md").exists());
    let ControlResponse::ComposeTaskResult(ok) = run("filename: a.md\nhello\n", "c2") else {
        panic!()
    };
    assert!(ok.committed, "{ok:?}");
    // committed = the proposal is approved for write authorization with these exact bytes
    assert_eq!(ok.proposal.as_deref(), Some("filename: a.md\nhello\n"));
}

/// The compose verification boundary through `ComposeBridge::run_task`: a
/// reply whose one line lacks the phrase is never approved.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn the_proposal_boundary_approves_only_the_line_with_the_phrase() {
    let _home = home_guard::home_slot();
    let goal = "Create a.md containing one line saying hello.";
    let go = |text: &str| {
        let h = Harness::new(always(text));
        let r = h.run(goal).unwrap();
        r.proposal
            .ok_or_else(|| r.proposer_error.unwrap_or_default())
    };
    assert!(go("filename: a.md\ngoodbye\n").is_err());
    assert!(go("filename: a.md\nhello\n").is_ok());
}

/// A saying clause on a line is never silently dropped, whatever the count
/// wording: it is read (one line, short plain phrase) or uncertain.
#[test]
fn a_saying_clause_on_any_line_wording_is_read_or_uncertain() {
    // not exact ("write" means at least) but a single line with a plain phrase: read
    for (goal, phrase) in [
        ("Write one line saying hello to a.md", "hello"),
        ("Write a line saying hello to a.md", "hello"),
        ("Add a line that says hello.", "hello"),
        ("Write a line saying \"hello there\"", "hello there"),
    ] {
        let r = read(goal);
        assert!(
            r.contains(&RequiredPhrases(vec![phrase.to_string()])),
            "{goal}: {r:?}"
        );
    }
    // another count, or a bound, cannot say what each line holds: uncertain
    for goal in [
        "Add 2 lines saying hi to a.md",
        "Write three lines saying hello",
        "Add 2 lines that say hi",
        "Add at least 3 lines saying hi to a.md",
        "Add at least one line saying hi to a.md",
        "Add one line at least saying hi to a.md",
        "Write a line saying hello and then save it",
        "Write a line saying something nice",
        "Add one line saying",
    ] {
        let ex = analyze(goal);
        assert!(
            !ex.uncertain.is_empty(),
            "{goal:?} must be uncertain, got {:?}",
            ex.requirements
        );
    }
    // unrelated numbers and words stay as they were
    assert!(read("Fix 2 typos in README.md").is_empty());
    assert!(read("Add a line to a.md").is_empty());
}
