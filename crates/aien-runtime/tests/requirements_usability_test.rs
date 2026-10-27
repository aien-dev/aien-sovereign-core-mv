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

//! Ordinary counted requests and the wider wordings (crate::requirements_extract),
//! PR round 2: "Add 2 lines to README.md" is read, not refused; every goal of the
//! campaign task files is classified (recognized, silent or uncertain) and the
//! refused ones are listed with their reason.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::requirements::{analyze, refusal_reason, unmet, ItemKind, Requirement};
use compose_harness::{always, Asked, Harness};

use Requirement::*;

fn added(n: usize, exact: bool) -> Requirement {
    AddedLines {
        n,
        exact,
        prior: None,
    }
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

fn uncertain(goal: &str) -> Vec<String> {
    let ex = analyze(goal);
    assert!(
        !ex.uncertain.is_empty(),
        "{goal:?} should be refused as uncertain, got {:?}",
        ex.requirements
    );
    ex.uncertain
}

#[test]
fn counted_requests_are_read_by_the_documented_rule() {
    // add|append|insert|with N lines (and "N-line"): exactly N added lines.
    for (goal, n) in [
        ("Add 2 lines to README.md", 2),
        ("Append 3 lines to LOG.md", 3),
        ("Add a one-line note to README.md", 1),
        ("Insert two lines under the intro in docs/a.md", 2),
        ("Add exactly 4 lines to docs/a.md.", 4),
        ("Write a two-line poem", 2),
        (
            "Create the file docs/CONTACT.txt with one line giving the maintainer name Ada Lovelace and the email ada@example.org.",
            1,
        ),
    ] {
        assert_eq!(reqs(goal), [added(n, true)], "{goal}");
    }
    // sc#349: the plain phrase after "saying" is required as well
    assert_eq!(
        reqs("Add one line to notes/TODO.md saying buy milk"),
        [
            added(1, true),
            RequiredPhrases(vec!["buy milk".to_string()])
        ]
    );
    // write|put|create|... N lines: at least N added lines.
    for (goal, n) in [
        ("Write 20 lines about X to docs/x.md", 20),
        ("Please write five lines on composting in docs/c.md", 5),
        ("Draft 12 lines for the changelog", 12),
    ] {
        assert_eq!(reqs(goal), [added(n, false)], "{goal}");
    }
}

#[test]
fn bare_counts_with_a_qualifier_or_no_verb_stay_uncertain() {
    for goal in [
        "Add 3 lines of context to README.md",
        "Write 10 lines per section in docs/a.md",
        "Keep it to 10 lines",
        "Do not add 2 lines to README.md",
        "The note should be 5 lines",
        // sc#336: an addition may sit in a longer file; a new one-sentence
        // file ("with a one-sentence greeting") is read as SingleSentence.
        "Add a one-sentence summary to README.md",
    ] {
        uncertain(goal);
    }
}

#[test]
fn wider_wordings_are_read() {
    assert_eq!(
        reqs("Write a new file docs/B.md. A minimum of 20 lines is needed."),
        [MinLines(20)]
    );
    assert_eq!(reqs("At least 20 lines are required."), [MinLines(20)]);
    assert_eq!(reqs("It needs 20 lines minimum."), [MinLines(20)]);
    assert_eq!(
        reqs("Finish with a section titled \"Restore notes\" that holds at least 15 words of plain text."),
        [
            SectionMinWords {
                title: Some("Restore notes".into()),
                n: 15
            },
            RequiredHeadings(vec!["Restore notes".into()]),
        ]
    );
    assert_eq!(
        reqs("The last section needs at least 15 words of plain text."),
        [SectionMinWords { title: None, n: 15 }]
    );
    assert_eq!(
        reqs("Use three level-two sections titled \"Building\", \"Testing\" and \"Packaging\", then add notes."),
        [LevelHeadings {
            level: 2,
            titles: vec!["Building".into(), "Testing".into(), "Packaging".into()]
        }]
    );
    assert_eq!(
        reqs("Include three separate fenced code examples."),
        [MinItems(ItemKind::CodeBlocks, 3)]
    );
}

#[test]
fn unscoped_word_counts_and_topic_lists_stay_uncertain() {
    uncertain("Write it with at least 15 words of plain text.");
    // sc#332: a counted topic list is read when the count matches the list.
    uncertain("Cover two topics: drying, labelling and storage.");
    uncertain("Put one fenced shell code example in each of them.");
    uncertain("Each section has two sentences.");
}

#[test]
fn the_new_checks_measure_the_right_thing() {
    let doc = "# A\nprose\n\n## Building\none two three\n## Testing\nx\n## Packaging\ny\n";
    let lv = LevelHeadings {
        level: 2,
        titles: vec!["Building".into(), "Testing".into(), "Packaging".into()],
    };
    assert!(lv.check(doc).is_none());
    // Wrong level is missing; extra ## sections are fine.
    let deep = doc.replace("## Testing", "### Testing");
    assert!(lv.check(&deep).unwrap().contains("\"Testing\""));
    assert!(lv.check(&format!("{doc}## Summary\nz\n")).is_none());
    let last = SectionMinWords { title: None, n: 3 };
    assert!(last.check("# T\n## End\nonly two\n").is_some());
    assert!(last.check("# T\n## End\nnow three words\n").is_none());
    // Code lines and headings are not plain text.
    let named = SectionMinWords {
        title: Some("Restore notes".into()),
        n: 4,
    };
    assert!(named
        .check("## Restore notes\n```sh\none two three four five\n```\nshort\n")
        .is_some());
    assert!(named
        .check("## Restore notes\nfour plain words here\n")
        .is_none());
    assert!(named.check("## Other\nfour plain words here\n").is_some());
    let blocks = MinItems(ItemKind::CodeBlocks, 2);
    assert!(blocks.check("```sh\na\n```\ntext\n~~~\nb\n~~~\n").is_none());
    assert!(blocks.check("```sh\na\n```\n").is_some());
}

/// One compose task through `ComposeBridge::run_task` with `prior` as the
/// content of `path` (None: `path` does not exist, a new document). The model
/// always answers `reply`. Ok = the approved proposal, Err = the refusal.
fn run_edit(goal: &str, path: &str, prior: Option<&str>, reply: &str) -> Result<String, String> {
    let ex = analyze(goal);
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    let h = Harness::new(always(reply));
    if let Some(p) = prior {
        h.write(path, p);
    }
    let r = h.run(goal).unwrap();
    let want = if prior.is_some() {
        Asked::Edit
    } else {
        Asked::Document
    };
    assert!(h.calls().iter().all(|c| c.asked == want), "{goal}");
    r.proposal
        .ok_or_else(|| r.proposer_error.unwrap_or_default())
}

const PRIOR: &str = "# Notes\n\nfirst\nsecond\n";

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn counted_edits_are_judged_on_the_diff_of_the_exact_proposed_bytes() {
    let _home = home_guard::home_slot();
    let g = "Add 2 lines to README.md";
    let two = "filename: README.md\n# Notes\n\nfirst\nnew a\nnew b\nsecond\n";
    let one = "filename: README.md\n# Notes\n\nfirst\nnew a\nsecond\n";
    let three = "filename: README.md\n# Notes\n\nfirst\nnew a\nnew b\nnew c\nsecond\n";
    assert!(run_edit(g, "README.md", Some(PRIOR), two).is_ok());
    let e = run_edit(g, "README.md", Some(PRIOR), one).unwrap_err();
    assert!(
        e.contains("exactly 2 added non-empty lines, found 1"),
        "{e}"
    );
    let e = run_edit(g, "README.md", Some(PRIOR), three).unwrap_err();
    assert!(e.contains("found 3"), "{e}");
    // Blank lines are not lines.
    let blanks = "filename: README.md\n# Notes\n\nfirst\n\nnew a\n\nnew b\nsecond\n";
    assert!(run_edit(g, "README.md", Some(PRIOR), blanks).is_ok());
    // "write N lines": at least N added; a new file has no prior lines.
    let w = "Write 3 lines about bread to README.md";
    let four = "filename: README.md\n# Notes\n\nfirst\na\nb\nc\nd\nsecond\n";
    assert!(run_edit(w, "README.md", Some(PRIOR), four).is_ok());
    assert!(run_edit(w, "README.md", Some(PRIOR), two).is_err());
    let fresh = "filename: NEW.md\nl1\nl2\n";
    assert!(run_edit("Add 2 lines to NEW.md", "NEW.md", None, fresh).is_ok());
    assert!(run_edit("Add 3 lines to NEW.md", "NEW.md", None, fresh).is_err());
    // Whole-file bytes that rewrite prior lines count the rewritten ones as added.
    let rewritten = "filename: README.md\n# Notes\n\nfirst CHANGED\nsecond\nnew\n";
    let e = run_edit(g, "README.md", Some(PRIOR), rewritten);
    assert!(e.is_ok(), "{e:?}");
}

#[test]
fn an_unresolved_added_line_count_fails_closed() {
    let ex = analyze("Add 2 lines to README.md");
    let u = unmet(&ex.requirements, "a\nb\n");
    assert_eq!(u.len(), 1);
    assert!(u[0].contains("cannot be measured"), "{u:?}");
    assert!(refusal_reason(&ex.requirements, "a\nb\n").is_some());
    // Resolved against the file the bytes replace, the same bytes pass.
    let ok = ex.resolved("");
    assert!(unmet(&ok.requirements, "a\nb\n").is_empty());
}

#[test]
fn a_machine_goal_for_an_approved_proposal_states_no_requirement() {
    // The goal the hook builds is "apply approved proposal for <path> (request <id>)".
    for path in [
        "docs/10 lines.md",
        "docs/add 2 lines.md",
        "notes/at least 5 lines.md",
        "write 20 lines.txt",
    ] {
        let goal = format!("apply approved proposal for {path} (request req-1)");
        let ex = analyze(&goal);
        assert!(
            ex.requirements.is_empty() && ex.uncertain.is_empty(),
            "{goal}: {ex:?}"
        );
    }
}

// ---- the campaign goals: recognized, silent or uncertain ----

const CORPUS: &str = include_str!("fixtures/goal_corpus.tsv");

/// Goals that state no measurable requirement: nothing to enforce, nothing refused.
const SILENT: [&str; 12] = [
    "dryrun-tasks-v2/N2",
    "dryrun-tasks-v2/R1",
    "dryrun-tasks-v2/T5",
    "dryrun-tasks-v2/T7",
    "tasks-oq3-v3/E1",
    "tasks-oq3-v3/E2",
    "tasks-oq3-v4/U1",
    "tasks-oq3-v4/U2",
    "tasks-v5/T1",
    "tasks-v5/T3",
    "tasks-v6/T5",
    "tasks-v8/T7",
];

/// Goals whose explicit requirements are all read and enforced.
const RECOGNIZED: [&str; 21] = [
    "dryrun-tasks-v2/N1",
    "tasks-oq3-v3/N1",
    "tasks-oq3-v4/N1",
    "tasks-v6/N1",
    "dryrun-tasks-v2/T4",
    "tasks-oq3-v3/G2",
    "tasks-oq3-v3/G3",
    // sc#336: a file of one sentence.
    "tasks-oq3-v3/R1",
    "tasks-oq3-v4/R1",
    // sc#335: "at least ten|twelve paragraphs" is MinParagraphs.
    "tasks-oq3-v3/N2",
    "tasks-oq3-v4/N2",
    "tasks-oq3-v4/RT4",
    "tasks-oq3-v4/RT6",
    "tasks-oq3-v4/W1",
    "tasks-oq3-v4/W2",
    "tasks-oq3-v4/W3",
    "tasks-oq3-v4/W4",
    // sc#334: "at least twelve of them" after `a line starting with "- [ ]"`.
    "tasks-oq3-v4/W5",
    "tasks-v5/T2",
    "tasks-v8/D3",
    "tasks-v8/D4",
];

/// Goals still refused as uncertain, each with the span that is reported and why.
const REFUSED: [(&str, &str, &str); 3] = [
    (
        "dryrun-tasks-v2/T6",
        "one section per symptom",
        "a section count that depends on the content",
    ),
    (
        "tasks-oq3-v3/G1",
        "one fenced shell code example in each of them",
        "a per-section code example count",
    ),
    (
        "tasks-v8/D5",
        "10 frequently asked questions and answers",
        "a count with a long adjective phrase",
    ),
];

#[test]
fn every_campaign_goal_is_recognized_silent_or_refused_for_a_listed_reason() {
    let mut seen = 0;
    for line in CORPUS.lines() {
        let (id, goal) = line.split_once('\t').expect("id TAB goal");
        let ex = analyze(goal);
        seen += 1;
        if SILENT.contains(&id) {
            assert!(
                ex.requirements.is_empty() && ex.uncertain.is_empty(),
                "{id} should be silent: {ex:?}"
            );
        } else if RECOGNIZED.contains(&id) {
            assert!(
                !ex.requirements.is_empty() && ex.uncertain.is_empty(),
                "{id} should be recognized: {ex:?}"
            );
        } else if let Some((_, span, _why)) = REFUSED.iter().find(|r| r.0 == id) {
            assert!(
                ex.uncertain.iter().any(|u| u.contains(span)),
                "{id} should be refused on {span:?}: {:?}",
                ex.uncertain
            );
        } else {
            panic!("{id} is in no list: {ex:?}");
        }
    }
    // Every listed id exists in the corpus.
    assert_eq!(seen, SILENT.len() + RECOGNIZED.len() + REFUSED.len());
}

#[test]
fn the_w2_goal_is_fully_enforced() {
    let goal = CORPUS
        .lines()
        .find(|l| l.starts_with("tasks-oq3-v4/W2\t"))
        .and_then(|l| l.split_once('\t'))
        .map(|x| x.1)
        .unwrap();
    let ex = analyze(goal);
    assert!(ex.uncertain.is_empty());
    assert!(ex.requirements.contains(&MinLines(20)));
    assert!(ex.requirements.contains(&MinItems(ItemKind::CodeBlocks, 3)));
    assert!(ex.requirements.contains(&SectionMinWords {
        title: Some("Restore notes".into()),
        n: 15
    }));
    let mut good = String::new();
    for t in ["Create the archive", "Copy it offsite", "Verify the copy"] {
        good.push_str(&format!(
            "## {t}\n```sh\necho {}\n```\nsome words about it\n\n",
            t.len()
        ));
    }
    good.push_str("## Restore notes\nrestore the newest archive first then check the files one by one afterwards carefully\n");
    for i in 0..10 {
        good.push_str(&format!("closing line {i}\n"));
    }
    // The long closing lines sit in the last section; keep the heading text rule honest.
    let good = good.replace("closing line 0\n", "");
    assert!(
        refusal_reason(&ex.requirements, &good).is_none(),
        "{:?}",
        refusal_reason(&ex.requirements, &good)
    );
    let short = good.replace(
        "restore the newest archive first then check the files one by one afterwards carefully\n",
        "restore it\n",
    );
    let short = short
        .lines()
        .filter(|l| !l.starts_with("closing line"))
        .collect::<Vec<_>>()
        .join("\n");
    assert!(refusal_reason(&ex.requirements, &short)
        .unwrap()
        .contains("words of plain text in the section \"Restore notes\""));
}

// ---- round 3 ----

fn added_check(goal: &str, prior: &str, content: &str) -> Option<String> {
    let ex = analyze(goal).resolved(prior);
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    refusal_reason(&ex.requirements, content)
}

#[test]
fn an_add_request_may_not_delete_or_rewrite_existing_lines() {
    let prior = "a\nb\nc\nd\n";
    let g = "Add 2 lines to README.md";
    for bad in ["x\ny\n", "a\nb\nx\ny\n", "a\nb\nC\nD\n", "a\nb\nc\nd\nx\n"] {
        assert!(
            added_check(g, prior, bad).is_some(),
            "{bad:?} must be refused"
        );
    }
    let e = added_check(g, prior, "a\nb\nC\nD\n").unwrap();
    assert!(e.contains("changed or removed"), "{e}");
    // Inserts in the middle or at the end, blank lines, and trailing-space-only
    // edits to existing lines are allowed (trailing whitespace is not content).
    for ok in [
        "a\nb\nx\nc\ny\nd\n",
        "a\nb\nc\nd\nx\ny\n",
        "x\ny\na\nb\nc\nd\n",
        "a  \nb\nc\t\nd\n\nx\n\ny\n",
    ] {
        assert!(added_check(g, prior, ok).is_none(), "{ok:?} must pass");
    }
    // The same holds for "write N lines" (at least N).
    let w = "Write 2 lines about bread to README.md";
    assert!(added_check(w, prior, "x\ny\nz\n").is_some());
    assert!(added_check(w, prior, "a\nb\nc\nd\nx\ny\nz\n").is_none());
}

#[test]
fn vague_or_odd_counts_are_uncertain_not_silent() {
    for goal in [
        "Add 2-3 lines to README.md",
        "Write a dozen lines about bread",
        "The line count should be at least 40.",
        "Write a script with 2 shell commands.",
        "Use \"A\", \"B\" and \"C\" as level-two headings.",
        "Use level-two headings \"A\", \"B\" and \"C\".",
        "Write three-plus sections about bread.",
        "Use a section titled Intro has 80 words of plain text.",
        "Add 2 lines at most to README.md",
    ] {
        uncertain(goal);
    }
}

#[test]
fn lower_bound_after_the_noun_and_single_line() {
    assert_eq!(
        reqs("Write forty lines at least to docs/a.md"),
        [added(40, false)]
    );
    assert_eq!(
        reqs("Add a single line saying hello to a.md"),
        [added(1, true), RequiredPhrases(vec!["hello".to_string()])]
    );
    // No explicit count: nothing is measured (documented): "a line" is not a number word.
    // sc#349: no count is measured, but what the line says is still required
    assert_eq!(
        reqs("Add a line saying hello to a.md"),
        [RequiredPhrases(vec!["hello".to_string()])]
    );
    assert!(reqs("Add a couple of lines or a few to a.md").is_empty());
}
