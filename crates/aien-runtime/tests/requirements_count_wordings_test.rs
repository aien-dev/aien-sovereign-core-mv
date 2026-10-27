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

//! sc#331: ordinary count wordings of the v5 goals were refused as uncertain
//! before any model call: an inclusive range ("between 16 and 30 lines"), the
//! adjective "non-empty" before lines, a counted title list after the
//! preposition "under", and a single title given as "one section must be
//! titled". Each is now read; the near misses stay UNCERTAIN.
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

fn heads(t: &[&str]) -> Requirement {
    RequiredHeadings(t.iter().map(|x| x.to_string()).collect())
}

#[test]
fn the_v5_d1_count_sentence_is_read() {
    // v5 D1, second sentence, verbatim.
    let g = "It should have at least 22 non-empty lines and three sections titled \"Gather Your Tools\", \"Sharpen the Edge\" and \"Test and Store\".";
    assert_eq!(
        reqs(g),
        [
            MinLines(22),
            heads(&["Gather Your Tools", "Sharpen the Edge", "Test and Store"])
        ]
    );
}

#[test]
fn the_v5_d5_range_and_sections_are_read() {
    // v5 D5, second sentence, verbatim.
    assert_eq!(
        reqs("Keep it between 16 and 30 lines long."),
        [MinLines(16), MaxLines(30)]
    );
    // v5 D5, last sentence up to its word clause (that clause is sc#333).
    assert_eq!(
        reqs("Organise it under three sections titled \"Documents\", \"Clothes\" and \"Electronics\"."),
        [heads(&["Documents", "Clothes", "Electronics"])]
    );
}

#[test]
fn the_v5_d6_single_title_is_read() {
    // v5 D6, the clause after its paragraph count (that count is sc#335).
    let g = "Please write a warm essay about adopting a rescue cat, and one section must be titled \"Bringing Her Home\".";
    assert_eq!(reqs(g), [heads(&["Bringing Her Home"])]);
}

#[test]
fn related_wordings_are_read_the_same_way() {
    for (goal, want) in [
        (
            "Write between 100 and 200 words.",
            vec![MinWords(100), MaxWords(200)],
        ),
        (
            "Keep it between 5 and 5 lines.",
            vec![MinLines(5), MaxLines(5)],
        ),
        ("Use at least 10 nonempty lines.", vec![MinLines(10)]),
        ("Use no fewer than 10 non-blank lines.", vec![MinLines(10)]),
        (
            "The section should be titled \"Costs\".",
            vec![heads(&["Costs"])],
        ),
        ("A section is titled \"Costs\".", vec![heads(&["Costs"])]),
        // round 3: a hedge word inside the quoted title is part of the title
        (
            "One section must be titled \"Water When Dry\".",
            vec![heads(&["Water When Dry"])],
        ),
        (
            "A section is titled \"What If We Wait\".",
            vec![heads(&["What If We Wait"])],
        ),
        // a hedge in another sentence does not reach this one
        (
            "You may write more. One section must be titled \"Costs\".",
            vec![heads(&["Costs"])],
        ),
        (
            "Keep it between 16 and 30 lines and between 100 and 300 words.",
            vec![MinLines(16), MaxLines(30), MinWords(100), MaxWords(300)],
        ),
        (
            "Two sections must be titled \"Costs\" and \"Risks\".",
            vec![heads(&["Costs", "Risks"])],
        ),
    ] {
        assert_eq!(reqs(goal), want, "{goal}");
    }
}

#[test]
fn near_misses_stay_uncertain() {
    for goal in [
        // a reversed range
        "Keep it between 30 and 16 lines.",
        // no upper bound exists for items
        "Use between 3 and 5 items.",
        // a qualified range
        "Keep it between 16 and 30 lines of code.",
        "Keep it between 16 and 30 lines per section.",
        // negated
        "Do not keep it between 16 and 30 lines.",
        // other range forms
        "Write 2-3 lines.",
        "Write from 16 to 30 lines.",
        // non-empty before another noun
        "Use at least 22 non-empty words.",
        // the count differs from the list
        "Organise it under two sections titled \"Documents\", \"Clothes\" and \"Electronics\".",
        "One section must be titled \"A\" and \"B\".",
        // "under N sections" with no titles is a bound
        "Keep it under 3 sections.",
        // negated title
        "One section must not be titled \"Intro\".",
        // the aux forms need quoted titles
        "One section must be titled with the trip name.",
        // sc#343 review: a shared title, a conditional title
        "Each section is titled \"X\".",
        "Every section must be titled \"X\".",
        "A section is titled \"X\" if needed.",
        "Optionally one section must be titled \"X\".",
        // sc#343 review: a range on a part of the document, or approximate
        "Keep the gap between 10 and 20 lines.",
        "Keep each paragraph between 2 and 4 lines.",
        "Make the second paragraph between 16 and 30 words.",
        "Keep it between 16 and 30 lines or so.",
        // sc#343 review: "under" is a preposition only after an organising verb
        "Keep it under 3 sections titled \"A\", \"B\" and \"C\".",
        // sc#343 review round 2: a hedge anywhere in the sentence
        "A section is titled \"X\", if you can.",
        "One section must be titled \"X\", unless it is short.",
        "One section must be titled \"X\"; maybe add more.",
        "If needed, one section must be titled \"X\".",
        "Please write an essay, and one section must be titled \"A\", when relevant.",
        // round 2: a part of the document anywhere earlier in the sentence
        "Write the essay so that every single long paragraph stays between 16 and 30 words.",
        "Write the poem so that its stanzas are between 16 and 30 words.",
        "Write the poem; the stanzas should be between 16 and 30 words.",
        "Keep it between 16 and 30 lines long, or so.",
        "Write between 16 and 30 lines, roughly.",
        "Write between 16 and 30 lines, give or take.",
        "Write between 16 and 30 lines approximately.",
    ] {
        let ex = analyze(goal);
        assert!(
            !ex.uncertain.is_empty(),
            "{goal:?} should stay uncertain, got {:?}",
            ex.requirements
        );
    }
}
