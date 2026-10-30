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

//! sc#383 item 1: the WHOLE-SYSTEM-E2E contract objective ("write a Markdown
//! report under 200 words named report.md in the outbox folder") was refused
//! as an uncertain requirement before any model call, because only the
//! `at most` family was read. `under|below N words|lines` now reads as fewer
//! than N and `within N words|lines` as at most N; other nouns and negated
//! forms stay UNCERTAIN, never dropped.
use aien_runtime::requirements::{analyze, Requirement};

use Requirement::*;

fn reqs(goal: &str) -> Vec<Requirement> {
    let ex = analyze(goal);
    assert!(
        ex.uncertain.is_empty(),
        "{goal:?} should be read, uncertain: {:?}",
        ex.uncertain
    );
    ex.requirements
}

fn uncertain(goal: &str) {
    let ex = analyze(goal);
    assert!(
        !ex.uncertain.is_empty(),
        "{goal:?} must stay UNCERTAIN, read as {:?}",
        ex.requirements
    );
}

#[test]
fn the_contract_objective_word_bound_is_read() {
    // RUN-1-dry-20261010T133749Z row E3, the refused clause, verbatim.
    let g = "Write a Markdown report under 200 words named report.md in the outbox folder.";
    assert_eq!(reqs(g), [MaxWords(199)]);
}

#[test]
fn under_and_below_are_strict_upper_bounds() {
    assert_eq!(reqs("Keep it under 200 words."), [MaxWords(199)]);
    assert_eq!(reqs("Write a poem below twelve lines."), [MaxLines(11)]);
    assert_eq!(reqs("Write a note under 40 words long."), [MaxWords(39)]);
    // the same bound as the wording that was already read
    assert_eq!(
        reqs("Keep it under 200 words."),
        reqs("Keep it fewer than 200 words.")
    );
}

#[test]
fn within_is_an_inclusive_upper_bound() {
    assert_eq!(reqs("Summarise it within 120 words."), [MaxWords(120)]);
    assert_eq!(
        reqs("Summarise it within 120 words."),
        reqs("Summarise it in no more than 120 words.")
    );
}

#[test]
fn other_nouns_and_negations_stay_uncertain() {
    // a structure, not a bound
    uncertain("Write the report under three sections.");
    // a measure no requirement checks
    uncertain("Keep each line under 80 characters.");
    // negated
    uncertain("The summary must not be under 200 words.");
    // a bound of nothing
    uncertain("Keep it under 1 words.");
}
