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

//! A goal that asks the document to start with a markdown heading
//! (Requirement::FirstLineHeading). The ALLEN end-to-end demo v2 request was
//! refused as uncertain because no recognizer read this wording (arch#162).
//! Recognizing it must not loosen any other uncertain wording.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::check_file_proposal;
use aien_runtime::control::ComposeTaskReport;
use aien_runtime::requirements::{analyze, refusal_reason, Requirement};
use compose_harness::Harness;

use Requirement::*;

/// The S3 request of docs/campaigns/allen-e2e/DEMO-v2.md, verbatim.
const DEMO_V2_GOAL: &str = "Write garden.md in the workspace, a garden plan. Start with a Markdown heading line that begins with \"# \". Use at most 200 words.";

/// One compose task through `ComposeBridge::run_task`: the model answers call
/// k with `replies[k - 1]` (the last one repeats). Returns the report and the
/// prompts the model was handed.
fn run(goal: &str, replies: Vec<String>) -> (ComposeTaskReport, Vec<String>) {
    let ex = analyze(goal);
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    let texts: Vec<&str> = replies.iter().map(String::as_str).collect();
    let h = Harness::new(compose_harness::replies(&texts));
    let r = h.run(goal).unwrap();
    let p = h.calls().into_iter().map(|c| c.prompt).collect();
    (r, p)
}

const NO_HEADING: &str =
    "filename: garden.md\nGarden plan\n\nPlant tomatoes in May and water them weekly.\n";
const GOOD: &str =
    "filename: garden.md\n# Garden plan\n\nPlant tomatoes in May and water them weekly.\n";

#[test]
fn demo_v2_goal_is_fully_recognized() {
    let ex = analyze(DEMO_V2_GOAL);
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    assert_eq!(
        ex.requirements,
        [MaxWords(200), FirstLineHeading { level: Some(1) }]
    );
    // The S8 request differs only in the file name.
    let s8 = DEMO_V2_GOAL.replace("garden.md", "garden-b.md");
    assert_eq!(analyze(&s8), ex);
}

#[test]
fn first_heading_phrasings() {
    for (g, want) in [
        (
            "Start with a Markdown heading line that begins with \"# \".",
            Some(1),
        ),
        (
            "start with a markdown heading line that starts with \"## \"",
            Some(2),
        ),
        ("Begin with a heading.", None),
        ("Start with a Markdown heading.", None),
        ("It must start with a level-two heading.", Some(2)),
        ("Open with a markdown heading line.", None),
        (
            "Start with a heading that begins with \u{201C}# \u{201D}.",
            Some(1),
        ),
    ] {
        let ex = analyze(g);
        assert!(ex.uncertain.is_empty(), "{g}: {:?}", ex.uncertain);
        assert_eq!(ex.requirements, [FirstLineHeading { level: want }], "{g}");
    }
}

#[test]
fn near_misses_stay_uncertain() {
    for g in [
        // negated: never inverted
        "Don't start with a Markdown heading.",
        // a marker that is not a heading marker on its own
        "Start with a Markdown heading line that begins with \"#\".",
        // the level adjective and the marker disagree
        "Start with a level-one heading line that begins with \"## \".",
        // seven hashes is not a heading
        "Start with a heading line that begins with \"####### \".",
        // emphasis around the wording
        "Start with a **Markdown heading**.",
        "Start with a **heading**.",
        // another word before "heading"
        "Begin with a short heading.",
        // a qualifier after the heading
        "Start with a Markdown heading about tomatoes.",
        "Start with a Markdown heading line that begins with \"# \" in bold.",
        // no marker after "begins with"
        "Start with a Markdown heading line that begins with a hash.",
        // a sibling constraint in the same sentence is never swallowed (R319 MUST 1)
        "Begin with a markdown heading and keep it brief",
        "Begin with a markdown heading and lowercase only",
        "Begin with a markdown heading and end with a signature",
        "Begin with a markdown heading, then a table",
        "Begin with a markdown heading and a few sections",
        "Open with a level-2 heading line and a short list",
        "Open with a markdown heading line, and keep it short.",
        "Start with a heading line that begins with \"# \" and at most 5 items.",
        "Start with a heading, ####### markdown.",
        // other heading cues the safety net already refused
        "The page needs a Summary heading.",
        "Include headings for setup and use.",
    ] {
        let ex = analyze(g);
        assert!(!ex.uncertain.is_empty(), "{g}: should be uncertain");
        assert!(
            !ex.requirements
                .iter()
                .any(|r| matches!(r, FirstLineHeading { .. })),
            "{g}: {:?}",
            ex.requirements
        );
    }
}

#[test]
fn the_first_line_is_checked() {
    let one = FirstLineHeading { level: Some(1) };
    let any = FirstLineHeading { level: None };
    for (doc, ok1, ok_any) in [
        ("# Garden plan\nbody\n", true, true),
        ("# Garden plan\r\nbody\r\n", true, true),
        ("# Garden plan", true, true),
        ("## Garden plan\n", false, true),
        ("###### Six\n", false, true),
        ("####### Seven\n", false, false),
        ("Garden plan\n# Later\n", false, false),
        ("\n# Garden plan\n", false, false),
        (" # Indented\n", false, false),
        ("#Garden\n", false, false),
        ("# \nbody\n", false, false),
        ("#\n", false, false),
        ("", false, false),
        ("\u{feff}# Garden plan\n", false, false),
    ] {
        assert_eq!(one.check(doc).is_none(), ok1, "level 1 on {doc:?}");
        assert_eq!(any.check(doc).is_none(), ok_any, "any level on {doc:?}");
    }
    let why = one.check("Garden plan\n").unwrap();
    assert!(why.contains("first line"), "{why}");
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn a_reply_without_the_heading_is_refused_then_a_good_one_passes() {
    let _home = home_guard::home_slot();
    let ex = analyze(DEMO_V2_GOAL);
    let content = check_file_proposal(NO_HEADING).unwrap().content;
    let why = refusal_reason(&ex.requirements, &content).unwrap();
    assert!(why.contains("first line"), "{why}");
    let (r, prompts) = run(DEMO_V2_GOAL, vec![NO_HEADING.into(), GOOD.into()]);
    assert_eq!(r.proposal.as_deref(), Some(GOOD), "{r:?}");
    let a = &r.proposal_attempts;
    assert_eq!(a[0].outcome, "refused");
    assert_eq!(a[1].outcome, "parsed");
    // the retry prompt names what was missed
    assert!(prompts[1].contains("first line"), "{}", prompts[1]);
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn a_reply_that_never_has_the_heading_exhausts_with_nothing_approved() {
    let _home = home_guard::home_slot();
    let (r, _) = run(DEMO_V2_GOAL, vec![NO_HEADING.into()]);
    assert!(
        r.proposal.is_none() && r.uncommitted_proposal.is_none(),
        "{r:?}"
    );
    assert!(r.proposer_error.unwrap().contains("first line"));
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), 3);
    assert!(a
        .iter()
        .all(|x| x.outcome == "refused" && x.aegis.is_none()));
}
