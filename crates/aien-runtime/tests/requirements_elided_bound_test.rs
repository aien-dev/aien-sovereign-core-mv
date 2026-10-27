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

//! #344: a second bound whose noun is left out ("at least 20 lines and no more
//! than 40") is read with the earlier noun, or reported UNCERTAIN. It is never
//! dropped silently.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::control::ControlResponse;
use aien_runtime::requirements::{analyze, refusal_reason, Requirement};
use aien_runtime::{check_file_proposal, ComposeBridge, ComposeProposer, Generation};
use compose_harness::{always, Harness};
use std::sync::Arc;
use std::time::Duration;

use Requirement::*;

fn read(goal: &str) -> Vec<Requirement> {
    let ex = analyze(goal);
    assert!(ex.uncertain.is_empty(), "{goal:?}: {:?}", ex.uncertain);
    ex.requirements
}

#[test]
fn elided_max_after_min_is_read_with_the_earlier_noun() {
    let want = [MinLines(20), MaxLines(40)];
    for g in [
        "Write at least 20 lines and no more than 40.",
        "Write at least 20 lines and no more than 40",
        "Write at least 20 lines, and no more than 40.",
        "Write at least 20 lines but at most 40.",
        "Write no fewer than 20 lines and up to 40.",
        "Create docs/a.md with at least 20 lines and at most 40 in total.",
    ] {
        let r = read(g);
        for w in &want {
            assert!(r.contains(w), "{g:?} -> {r:?}");
        }
    }
    let r = read("Write at least 100 words but at most 200.");
    assert!(
        r.contains(&MinWords(100)) && r.contains(&MaxWords(200)),
        "{r:?}"
    );
    // strict cue keeps its adjustment: fewer than 40 is at most 39
    let r = read("Write at least 20 lines and fewer than 40.");
    assert!(r.contains(&MaxLines(39)), "{r:?}");
}

#[test]
fn unsupported_noun_or_direction_is_uncertain_not_silent() {
    // paragraphs have no upper bound in the model
    let ex = analyze("Write at least 3 paragraphs and no more than 5.");
    assert!(!ex.uncertain.is_empty(), "{ex:?}");
    // items: no upper bound either
    let ex = analyze("List at least 3 items and no more than 5.");
    assert!(!ex.uncertain.is_empty(), "{ex:?}");
    // a cue with no earlier noun to carry over
    let ex = analyze("Write at least 20 lines and then keep it under, no more than 40.");
    assert!(!ex.uncertain.is_empty(), "{ex:?}");
}

#[test]
fn ordinary_numbers_stay_silent() {
    for g in [
        "Write at least 20 lines and fix 2 typos.",
        "Fix 2 typos in README.md",
        "Update section 3 of docs/a.md",
        "Write at least 20 lines and update section 3.",
    ] {
        let r = analyze(g);
        assert!(r.uncertain.is_empty(), "{g:?}: {:?}", r.uncertain);
        assert!(
            !r.requirements.iter().any(|x| matches!(x, MaxLines(_))),
            "{g:?}: {:?}",
            r.requirements
        );
    }
    assert!(analyze("Fix 2 typos").requirements.is_empty());
    assert!(analyze("Update section 3").requirements.is_empty());
}

fn gen(text: &str) -> Result<Generation, String> {
    Ok(Generation {
        text: text.to_string(),
        tokens: 5,
        finish_reason: Some("eos".into()),
        ..Default::default()
    })
}

fn reply(lines: usize) -> String {
    let mut s = String::from("filename: docs/NOTES.md\n");
    for i in 0..lines {
        s.push_str(&format!("line number {i}\n"));
    }
    s
}

const GOAL: &str = "Create the file docs/NOTES.md with at least 20 lines and no more than 40.";

#[test]
fn bytes_over_the_second_bound_are_refused_and_within_bounds_pass() {
    let reqs = analyze(GOAL).requirements;
    let over = check_file_proposal(&reply(45)).unwrap().content;
    let why = refusal_reason(&reqs, &over).expect("45 lines must be refused");
    assert!(why.contains("40"), "{why}");
    for ok in [20usize, 30, 40] {
        let c = check_file_proposal(&reply(ok)).unwrap().content;
        assert!(refusal_reason(&reqs, &c).is_none(), "{ok} lines");
    }
}

fn bridge(tmp: &std::path::Path, lines: usize) -> ComposeBridge {
    let text = reply(lines);
    let proposer: ComposeProposer = Arc::new(move |_: &str, _: Duration| gen(&text));
    ComposeBridge::new(tmp.join("compose"), proposer, "test:elided")
}

fn report(resp: ControlResponse) -> Option<aien_runtime::control::ComposeTaskReport> {
    if !aien_omega_compose::LINKED {
        assert!(matches!(resp, ControlResponse::Error(_)));
        return None;
    }
    let ControlResponse::ComposeTaskResult(r) = resp else {
        panic!("{resp:?}")
    };
    Some(*r)
}

#[test]
fn linked_compose_does_not_commit_45_lines_but_commits_30() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let r = report(bridge(tmp.path(), 45).run_task(GOAL, ws.to_str().unwrap()));
    if let Some(r) = r {
        assert!(!r.committed, "{r:?}");
        assert!(r.proposal.is_none());
        assert_eq!(r.proposal_attempts.len(), 3);
        assert_eq!(std::fs::read_dir(&ws).unwrap().count(), 0);
    }
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    if let Some(r) = report(bridge(tmp.path(), 30).run_task(GOAL, ws.to_str().unwrap())) {
        assert!(r.committed, "{r:?}");
        let p = check_file_proposal(r.proposal.as_deref().unwrap()).unwrap();
        assert_eq!(p.content, check_file_proposal(&reply(30)).unwrap().content);
    }
}

/// The compose verification boundary through `ComposeBridge::run_task`: the
/// 45 line reply is refused on every attempt and nothing is approved; the 30
/// line reply is approved byte for byte.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn compose_boundary_refuses_45_lines_and_approves_30() {
    let _home = home_guard::home_slot();
    let ex = analyze(GOAL);
    assert!(ex.uncertain.is_empty());
    let run = |lines: usize| {
        let h = Harness::new(always(&reply(lines)));
        h.run(GOAL).unwrap()
    };
    let r = run(45);
    assert!(
        r.proposal.is_none() && r.uncommitted_proposal.is_none(),
        "{r:?}"
    );
    let why = r.proposer_error.unwrap();
    assert!(why.contains("40"), "{why}");
    let attempts = &r.proposal_attempts;
    assert_eq!(attempts.len(), 3);
    assert!(attempts
        .iter()
        .all(|a| a.outcome == "refused" && a.aegis.is_none()));
    let r = run(30);
    assert_eq!(r.proposal.as_deref(), Some(reply(30).as_str()), "{r:?}");
    assert_eq!(r.proposal_attempts.len(), 1);
}

#[test]
fn bound_cue_with_bare_number_and_no_antecedent_is_uncertain() {
    for g in [
        "Keep it to no more than 40",
        "Keep it to no more than 40.",
        "Write at least 20",
        "Write at least 20.",
        "Use up to 12.",
        "Make it fewer than 40.",
        "Make it more than 10.",
        "Write a maximum of 30.",
        "Write a minimum of 30.",
        "Write no fewer than 5.",
    ] {
        let ex = analyze(g);
        assert!(!ex.uncertain.is_empty(), "{g:?} silent: {ex:?}");
    }
    for g in [
        "Fix 2 typos",
        "Update section 3",
        "Use Python 3",
        "Use Python 3.",
    ] {
        let ex = analyze(g);
        assert!(ex.uncertain.is_empty(), "{g:?}: {:?}", ex.uncertain);
        assert!(ex.requirements.is_empty(), "{g:?}: {:?}", ex.requirements);
    }
}

#[test]
fn every_bound_cue_with_a_number_is_read_or_uncertain() {
    for g in [
        "Keep it at most 40 please",
        "Keep it to no more than 40, thanks.",
        "Include at least 2 of them",
        "Fix at least 2 typos in README.md",
        "Write at least 20 lines and fix at least 2 typos.",
        "Use not more than 3 colours.",
        "Use not less than 3 colours.",
    ] {
        let ex = analyze(g);
        assert!(!ex.uncertain.is_empty(), "{g:?} silent: {ex:?}");
    }
    // still read, not uncertain
    assert_eq!(read("Write at most 40 lines please."), [MaxLines(40)]);
    for g in [
        "Fix 2 typos in README.md",
        "Update section 3",
        "Use Python 3",
    ] {
        let ex = analyze(g);
        assert!(
            ex.uncertain.is_empty() && ex.requirements.is_empty(),
            "{g:?}"
        );
    }
}
