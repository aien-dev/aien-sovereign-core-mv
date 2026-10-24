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

//! sc#333: required-word wordings were either refused as uncertain before any
//! model call ("Make sure the words whetstone and angle both appear") or, worse,
//! silently not read ("use the exact words greens and browns", "must name
//! Priya, Tomas and Wen"): the product would approve a document without them.
//! They are now `RequiredWords`; near misses stay UNCERTAIN, never silent. A
//! hyphenated required word ("first-aid") is matched as a whole word.
use aien_runtime::requirements::{analyze, refusal_reason, Requirement};

use Requirement::*;

fn words(w: &[&str]) -> Requirement {
    RequiredWords(w.iter().map(|x| x.to_string()).collect())
}

fn reqs(goal: &str) -> Vec<Requirement> {
    let ex = analyze(goal);
    assert!(
        ex.uncertain.is_empty(),
        "{goal:?} should be read, uncertain: {:?}",
        ex.uncertain
    );
    ex.requirements
}

#[test]
fn the_exact_words_are_read_not_dropped() {
    // tasks-oq3-v4 W1, last sentence, verbatim.
    assert_eq!(
        reqs("Somewhere in the guide use the exact words greens and browns."),
        [words(&["greens", "browns"])]
    );
    // tasks-oq3-v4 W5, last sentence, verbatim.
    let r = reqs("Use sections titled \"Shelter\", \"Food and water\" and \"Safety\", and use the exact words headlamp and first-aid somewhere in the list.");
    assert!(r.contains(&words(&["headlamp", "first-aid"])), "{r:?}");
}

#[test]
fn named_people_are_read_not_dropped() {
    // tasks-oq3-v4 W4, verbatim.
    let w4 = "Create the file docs/WEEK-SUMMARY.md summarising notes/standup-log.txt in at least 14 lines. It needs a section titled \"Highlights\" and a section titled \"Open questions\", and it must name Priya, Tomas and Wen.";
    assert!(reqs(w4).contains(&words(&["Priya", "Tomas", "Wen"])));
    // v5 D4 (sc#339), verbatim.
    let d4 = "There is a file at notes/choir-rehearsal.txt with my rough notes from choir practice. Please create a new file called docs/choir-summary.md that summarises it, and leave the original alone. The summary needs at least 13 lines, two sections titled \"Key Points\" and \"Next Steps\", and must mention Marta, Owen and Priya by name.";
    assert!(reqs(d4).contains(&words(&["Marta", "Owen", "Priya"])));
}

#[test]
fn the_words_that_must_appear_are_read() {
    // v5 D1, last sentence, verbatim.
    assert_eq!(
        reqs("Make sure the words whetstone and angle both appear somewhere in it."),
        [words(&["whetstone", "angle"])]
    );
    // v5 D5, the word clause (the rest of D5 is sc#331 and sc#334).
    assert_eq!(
        reqs("Make me a checklist, and be sure the words passport and charger show up."),
        [words(&["passport", "charger"])]
    );
    for (goal, w) in [
        (
            "Ensure the words red and blue are used.",
            vec!["red", "blue"],
        ),
        (
            "Check that the terms alpha, beta and gamma all appear in the file.",
            vec!["alpha", "beta", "gamma"],
        ),
        ("Make sure the word kettle appears.", vec!["kettle"]),
    ] {
        assert_eq!(reqs(goal), [words(&w)], "{goal}");
    }
}

#[test]
fn a_hyphenated_word_is_matched_whole() {
    let r = [words(&["first-aid", "headlamp"])];
    assert!(refusal_reason(&r, "Pack the First-Aid kit and a headlamp.").is_none());
    for bad in [
        "Pack the first aid kit and a headlamp.",
        "Ask a first-aider; bring a headlamp.",
        "Pack a headlamp.",
    ] {
        let why = refusal_reason(&r, bad).expect("unmet");
        assert!(why.contains("first-aid"), "{bad}: {why}");
    }
}

#[test]
fn a_hyphenated_word_has_punctuation_boundaries() {
    let r = [words(&["first-aid"])];
    for good in [
        "Bring the first-aid-kit.",
        "Bring (first-aid).",
        "FIRST-AID.",
    ] {
        assert!(refusal_reason(&r, good).is_none(), "{good}");
    }
    let x = [words(&["x-ray"])];
    assert!(refusal_reason(&x, "A box-ray is not one.").is_some());
    assert!(refusal_reason(&x, "Get an x-ray.").is_none());
}

#[test]
fn a_list_followed_by_a_new_clause_is_still_read() {
    for (goal, w) in [
        (
            "It must name Priya, Tomas and Wen, and the file needs at least 14 lines.",
            vec!["Priya", "Tomas", "Wen"],
        ),
        (
            "Make sure the words whetstone and angle appear, and keep it short.",
            vec!["whetstone", "angle"],
        ),
        (
            "It must name Priya, Tomas and Wen, and I want it short.",
            vec!["Priya", "Tomas", "Wen"],
        ),
        (
            "Make sure the words whetstone and angle appear, and then be at least 5 lines long.",
            vec!["whetstone", "angle"],
        ),
        (
            "Use the exact words greens and browns somewhere in the file.",
            vec!["greens", "browns"],
        ),
    ] {
        let r = reqs(goal);
        assert!(r.contains(&words(&w)), "{goal}: {r:?}");
    }
}

#[test]
fn near_misses_stay_uncertain() {
    for goal in [
        // negated
        "Make sure the words whetstone and angle never appear.",
        "Make sure the words whetstone and angle do not appear.",
        // a scope the check cannot measure
        "Make sure the words whetstone and angle appear in every section.",
        "Use the exact words greens and browns somewhere in each section.",
        // a count on the words
        "Use the exact words greens and browns twice.",
        // not single words
        "Make sure the words sharp knife and angle appear.",
        // a partly named list
        "It must name Priya, Tomas and the dog.",
        // sc#345 review: a title with a period, a list cut by ";"
        "It must mention Dr. Smith.",
        "It must mention Mr. Smith and Priya.",
        "It must name Priya; and Tomas.",
        "It must name Priya. And Tomas.",
        // sc#345 review: a count or narrowing after the list
        "It must mention Priya at least twice.",
        "It must mention Priya, Tomas and Wen at least once each.",
        "Use the exact words greens and browns somewhere in the list at least twice.",
        "Use the exact words greens and browns somewhere in the list, but only in the summary.",
        "Make sure the words whetstone and angle appear somewhere, but only in the summary.",
        "Make sure the words whetstone and angle appear, and only in the summary.",
        // sc#345 review: a place that may be a section
        "Ensure the words red and blue appear in the summary.",
        "Ensure the words red and blue appear in the body.",
        "Use the exact words greens and browns somewhere in the notes.",
        // sc#345 review: a choice is not "all of them"
        "It must mention Priya or Tomas.",
        "Make sure the words red or blue appear.",
        "Use the exact words greens or browns.",
        // round 2: a second requirement after ", and"
        "It must name Priya and Tomas, and also Wen.",
        "Make sure the words whetstone and angle appear, and kettle appears twice.",
        "It must name Priya, Tomas and Wen, and nothing else.",
        "It must name Priya, Tomas and Wen, and they must each appear in a different section.",
        "It must name Priya and Tomas, and the dog.",
        // round 3: a name or word hidden in the new clause
        "It must name Priya and Tomas, and I want Wen too.",
        "It must name Priya and Tomas, and I need Wen to appear.",
        "It must name Priya and Tomas, and the one called Wen must appear twice.",
        "It must name Priya and Tomas, and the Wen section needs a heading.",
        "It must name Priya and Tomas, and write about Wen.",
        "It must name Priya and Tomas, and add Wen.",
        "It must name Priya and Tomas, and put Wen first.",
        "It must name Priya and Tomas, and give Wen a section.",
        "It must name Priya and Tomas, and end with Wen.",
        "It must name Priya and Tomas, and keep Wen in the intro.",
        "It must name Priya and Tomas, and then Wen.",
        "Make sure the words whetstone and angle appear, and add kettle too.",
        // negated forms
        "Make sure none of the words whetstone and angle appear.",
        "Make sure the words whetstone and angle don't appear.",
    ] {
        let ex = analyze(goal);
        assert!(
            !ex.uncertain.is_empty(),
            "{goal:?} should stay uncertain, got {:?}",
            ex.requirements
        );
    }
}

#[test]
fn wordings_without_a_word_list_stay_silent() {
    for goal in [
        "Name the file notes.md.",
        "It should mention safety and comfort.",
        "Please name it after the river.",
    ] {
        let ex = analyze(goal);
        assert!(
            ex.uncertain.is_empty() && ex.requirements.is_empty(),
            "{goal:?}: {ex:?}"
        );
    }
}
