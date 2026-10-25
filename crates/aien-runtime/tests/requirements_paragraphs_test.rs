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

//! sc#335: a paragraph count ("at least twelve paragraphs") was refused as
//! uncertain before any model call: there was no paragraph requirement. It is
//! now read as `MinParagraphs` for a minimum bound; an upper bound, an exact or
//! approximate count and a bare count stay UNCERTAIN.
use aien_runtime::requirements::{analyze, count_paragraphs, refusal_reason, Requirement};

use Requirement::*;

/// tasks-oq3-v4 N2, verbatim.
const V4_N2: &str = "Create the file docs/HISTORY.md with a long account of the project history that has at least twelve paragraphs.";
/// tasks-oq3-v3 N2, verbatim.
const V3_N2: &str = "Create the file docs/OVERVIEW.md with a detailed overview of the project that has at least ten paragraphs.";
/// v5 N2 (sc#339), verbatim.
const V5_N2: &str = "I would like a long, detailed history of the marathon as a race, from the legend of ancient Greece to modern city events, saved as docs/marathon-story.md. Please make it at least twelve paragraphs long.";
/// v5 D6 (sc#339), the sentence under test up to its comma (the section clause is sc#331).
const V5_D6: &str = "Please write a warm, longer essay about adopting a rescue cat and save it as docs/rescue-cat-essay.md. It should run to at least 7 paragraphs.";

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
fn the_campaign_paragraph_goals_are_read() {
    assert_eq!(reqs(V4_N2), [MinParagraphs(12)]);
    assert_eq!(reqs(V3_N2), [MinParagraphs(10)]);
    assert_eq!(reqs(V5_N2), [MinParagraphs(12)]);
    assert_eq!(reqs(V5_D6), [MinParagraphs(7)]);
}

#[test]
fn the_usual_minimum_wordings_are_read() {
    for (goal, n) in [
        ("Write at least 3 paragraphs about bees.", 3),
        ("Write 4 or more paragraphs about bees.", 4),
        ("Write 5+ paragraphs about bees.", 5),
        ("Use a minimum of six paragraphs.", 6),
        ("Use no fewer than 2 paragraphs.", 2),
        ("Write more than 3 paragraphs.", 4),
        ("It must be 8 paragraphs or more.", 8),
    ] {
        assert_eq!(reqs(goal), [MinParagraphs(n)], "{goal}");
    }
}

#[test]
fn other_paragraph_wordings_stay_uncertain() {
    for goal in [
        "Write at most 3 paragraphs.",
        "Write no more than 3 paragraphs.",
        "Write exactly 3 paragraphs.",
        "Write about 5 paragraphs.",
        "Write between 3 and 5 paragraphs.",
        "Write three paragraphs.",
        "Write at least 3 paragraphs per section.",
        "Write at least 3 paragraphs of code.",
        "Do not write at least 3 paragraphs.",
        "Write 2-3 paragraphs.",
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
                .any(|r| matches!(r, MinParagraphs(_))),
            "{goal:?}: {ex:?}"
        );
    }
}

#[test]
fn paragraphs_are_counted_by_the_documented_rule() {
    for (doc, n) in [
        ("", 0),
        ("one line\n", 1),
        ("one\ntwo lines, still one paragraph\n", 1),
        ("first\n\nsecond\n\n\nthird\n", 3),
        // a heading ends a paragraph and is not one
        ("# Title\nfirst\n## Next\nsecond\n", 2),
        // list items, table rows and breaks are not prose and end a paragraph
        ("intro\n- a\n- b\n\nafter the list\n", 2),
        // a line right after an item continues the item (sc#341 review)
        ("intro\n- a\n- b\nafter the list\n", 1),
        ("- a\n  more\n- b\n  more\n- c\n  more\n", 0),
        // indented code never starts a paragraph, but an indented line continues one
        ("    code1\n    code2\n\n    code3\n", 0),
        ("text\n    still the same paragraph\n", 1),
        // HTML, setext underlines and lines without letters or digits
        ("<div>\n\n<div>\n", 0),
        ("Title\n=====\nbody\n", 1),
        // round 2: a `---` underline is a setext heading too
        ("Title\n---\n\nTitle2\n---\n\nTitle3\n---\n", 0),
        ("Title\n-\nbody\n", 1),
        ("above\n\n---\n\nbelow\n", 2),
        // round 2: text indented under an item after a blank line belongs to it
        ("- item\n\n  indented para\n\n  another\n", 0),
        ("1. a\n\n   para under item\n", 0),
        ("- item\n\nnot indented, a paragraph\n", 1),
        // round 2: HTML or a pipe inside a run does not split it
        ("one\n<span>x</span>\ntwo\n", 1),
        ("one\n|two\nthree\n", 1),
        ("-\n\n1.\n\n***\n\n* * *\n\n...\n", 0),
        // blockquotes are prose; CRLF and non-ASCII text are fine
        ("> q1\n\n> q2\n", 2),
        ("a\r\n\r\nb\r\n", 2),
        ("# h\u{e9}llo\n\n\u{fc}\u{fc}\n", 1),
        // an unclosed fence hides the rest of the file
        ("before\n\n```\nafter\n\nmore\n", 1),
        ("1. step\n2. step\n", 0),
        ("| a | b |\n|---|---|\n| 1 | 2 |\n", 0),
        ("above\n***\nbelow\n", 2),
        // fenced code is not prose and ends a paragraph
        ("before\n```\ncode\n\nmore code\n```\nafter\n", 2),
        ("~~~\nonly code\n~~~\n", 0),
        // "#tag" without a space is text, not a heading
        ("#tag line\n", 1),
    ] {
        assert_eq!(count_paragraphs(doc), n, "{doc:?}");
    }
}

#[test]
fn the_check_reports_what_it_found() {
    let r = [MinParagraphs(3)];
    assert!(refusal_reason(&r, "a\n\nb\n\nc\n").is_none());
    let why = refusal_reason(&r, "a\nb\nc\n").expect("one paragraph only");
    assert!(why.contains("at least 3 paragraphs, found 1"), "{why}");
}
