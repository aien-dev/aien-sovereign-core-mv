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

//! sc#334: an item count that points back at a line prefix the same sentence
//! defines ("Write every item as a line starting with "- [ ]" and include at
//! least twelve of them.") was refused as uncertain before any model call. It
//! is now read as `MinPrefixedLines`: at least N lines that start with exactly
//! that quoted prefix. Without such a referent the count stays UNCERTAIN.
use aien_runtime::requirements::{analyze, refusal_reason, Requirement};

use Requirement::*;

/// tasks-oq3-v4 W5, verbatim.
const V4_W5: &str = "Make a new file called docs/CAMPING-LIST.md, a packing checklist for a weekend camping trip. It should run to a minimum of 16 lines and no more than 45 lines. Write every item as a line starting with \"- [ ]\" and include at least twelve of them. Use sections titled \"Shelter\", \"Food and water\" and \"Safety\", and use the exact words headlamp and first-aid somewhere in the list.";

/// tasks-oq3-v5 D5, the sentence under test (the rest of D5 is sc#331 and sc#333).
const V5_D5: &str = "Make me a carry-on packing checklist in docs/packing-carry-on.md. Every item must be its own line starting with \"- [ ]\" and I want at least 12 of those items.";

fn prefixed(n: usize) -> Requirement {
    MinPrefixedLines {
        prefix: "- [ ]".into(),
        n,
    }
}

#[test]
fn the_v4_w5_goal_reads_the_item_count() {
    let ex = analyze(V4_W5);
    assert!(ex.uncertain.is_empty(), "uncertain: {:?}", ex.uncertain);
    assert!(ex.requirements.contains(&prefixed(12)), "{ex:?}");
    assert!(ex.requirements.contains(&MinLines(16)));
    assert!(ex.requirements.contains(&MaxLines(45)));
}

#[test]
fn the_v5_d5_sentence_reads_the_item_count() {
    let ex = analyze(V5_D5);
    assert!(ex.uncertain.is_empty(), "uncertain: {:?}", ex.uncertain);
    assert_eq!(ex.requirements, [prefixed(12)]);
}

#[test]
fn other_referents_and_wordings_read_the_same_way() {
    for (goal, n) in [
        (
            "Put each task on a line beginning with \"TODO:\" and write at least 5 of them.",
            5,
        ),
        (
            "Each entry is a line that starts with \"* \"; add no fewer than three of these.",
            3,
        ),
        (
            "Every step is its own line starting with \"- \" and I want more than 4 of them.",
            5,
        ),
        // curly quotes (sc#340 review: a curly closing quote panicked the reader)
        (
            "Write every item as a line starting with \u{201C}- [ ]\u{201D} and include at least 4 of them.",
            4,
        ),
        (
            "Write lines starting with \u{201C}- [ ]\u{201D} and include at least 6 of those lines.",
            6,
        ),
    ] {
        let ex = analyze(goal);
        assert!(ex.uncertain.is_empty(), "{goal}: {:?}", ex.uncertain);
        assert_eq!(ex.requirements.len(), 1, "{goal}: {ex:?}");
        assert!(
            matches!(&ex.requirements[0], MinPrefixedLines { n: m, .. } if *m == n),
            "{goal}: {ex:?}"
        );
    }
}

#[test]
fn a_trailing_description_after_the_count_is_not_a_count() {
    // "each with a date" is not measurable; the prefix count is what is checked.
    let ex = analyze(
        "Write every item as a line starting with \"- [ ]\" and include at least 4 of them, each with a date.",
    );
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    assert_eq!(ex.requirements, [prefixed(4)]);
}

#[test]
fn only_lines_with_the_exact_prefix_count() {
    let r = prefixed(3);
    let good = "## Shelter\n- [ ] tent\n- [ ] pegs\n  - [ ] mallet\n";
    assert!(refusal_reason(std::slice::from_ref(&r), good).is_none());
    // Plain list lines, checked boxes and fenced lines are not "- [ ]" lines.
    let bad =
        "- tent\n- [x] pegs\n* [ ] stove\n```\n- [ ] in code\n```\n- [ ] mallet\n- [ ] rope\n";
    let why = refusal_reason(std::slice::from_ref(&r), bad).expect("unmet");
    assert!(
        why.contains("at least 3 lines starting with \"- [ ]\", found 2"),
        "{why}"
    );
}

#[test]
fn a_count_without_a_stated_referent_stays_uncertain() {
    for goal in [
        // no prefix in the sentence
        "Write a packing list and include at least twelve of them.",
        // the prefix is in an earlier sentence
        "Every item starts with \"- [ ]\". Include at least twelve of them.",
        // the prefix is not a line prefix
        "Use the word \"tent\" and include at least twelve of them.",
        // an upper bound is not read
        "Write every item as a line starting with \"- [ ]\" and include at most twelve of them.",
        // no bound cue
        "Write every item as a line starting with \"- [ ]\" and include twelve of them.",
        // a count that qualifies the referent further
        "Write every item as a line starting with \"- [ ]\" and include at least twelve of them per section.",
        // negated
        "Write every item as a line starting with \"- [ ]\" but do not include at least twelve of them.",
        // two prefixes: which one is meant is unclear
        "Write each item as a line starting with \"- [ ]\" or \"* [ ]\" and include at least twelve of them.",
        // sc#340 review cases: a curly quote with another clause before the count
        "Each line starting with \u{201C}- [ ]\u{201D} counts; include at least 4 of them.",
        // a negator before the referent
        "Avoid lines starting with \"- [ ]\" and include at least five of them.",
        "Do not write any line starting with \"TODO\" and include at least five of them.",
        // a single line, not every item
        "Start the file with a line starting with \"# Title\" and include at least 5 of those items.",
        // a per-section intent
        "Each section has lines starting with \"- [ ]\", and include at least 3 of them.",
        // another condition on the lines
        "Write lines starting with \"- \" and ending with \"!\" and include at least 4 of them.",
        // the referent is too far from the count
        "Write lines starting with \"- \" for the shopping, the cleaning, the garden and the car, and include at least 4 of them.",
        // a count the bound cue does not lead
        "Write lines starting with \"- [ ]\" and include 12 or more of them.",
        // round 2: a line break ends the sentence
        "Write lines starting with \"- [ ]\"\nInclude at least 12 of them.",
        // round 2: only the listed joining words
        "Write lines starting with \"- [ ]\" and make at least 12 of them.",
    ] {
        let ex = analyze(goal);
        assert!(
            !ex.uncertain.is_empty(),
            "{goal:?} should stay uncertain, got {:?}",
            ex.requirements
        );
        assert!(
            !ex.requirements
                .iter()
                .any(|r| matches!(r, MinPrefixedLines { .. })),
            "{goal:?}: {ex:?}"
        );
    }
}
