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

//! Explicit document requirements survive from the goal to the commit
//! (crate::requirements, crate::requirements_extract). Covers the two
//! open-model-qwen3 v3 gaps (G3 "with" after a count, G2 "covers" topics),
//! the uncertain-span refusal and the missing-record refusal (#289).
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::control::{ComposeTaskReport, ControlResponse};
use aien_runtime::requirements::{analyze, extract, refusal_reason, ItemKind, Requirement};
use aien_runtime::{check_file_proposal, ComposeBridge, ComposeProposer, Generation};
use compose_harness::Harness;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use Requirement::*;

const G3_GOAL: &str = "Create the file docs/ONBOARDING.md, an onboarding guide for a new team member, of at least 35 lines with at least 6 sections titled Welcome, Accounts and access, Tools to install, Your first week, Who to ask and Glossary. Write at least two sentences in every section.";
const G2_GOAL: &str = "Create the file docs/RELEASE-CHECKLIST.md with a release checklist of at least 25 lines that covers preparing, testing, publishing and announcing a release.";
// The recorded v3 replies (campaign receipts 35f00c9f and 10b9db39), with
// the dash characters replaced because the repo style check bans them.
const G3_V3_REPLY: &str = include_str!("fixtures/v3_replies/g3_onboarding_v3.txt");
const G2_V3_REPLY: &str = include_str!("fixtures/v3_replies/g2_release_checklist_v3.txt");

fn gen(text: &str) -> Result<Generation, String> {
    Ok(Generation {
        text: text.to_string(),
        tokens: 5,
        finish_reason: Some("eos".into()),
        ..Default::default()
    })
}

fn content_of(reply: &str) -> String {
    check_file_proposal(reply).unwrap().content
}

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

/// A reply that meets every requirement of the G3 goal.
fn g3_good() -> String {
    let mut s = String::from("filename: docs/ONBOARDING.md\n");
    for t in [
        "Welcome",
        "Accounts and access",
        "Tools to install",
        "Your first week",
        "Who to ask",
        "Glossary",
    ] {
        s.push_str(&format!("# {t}\n\n"));
        s.push_str("This section has a first sentence. It also has a second sentence.\n");
        s.push_str("- one more line about it\n- and another line about it\n- a third line about it\n- a fourth line about it\n\n");
    }
    s
}

/// A reply that meets every requirement of the G2 goal.
fn g2_good() -> String {
    let mut s = String::from("filename: docs/RELEASE-CHECKLIST.md\n# Release checklist\n");
    for i in 0..8 {
        s.push_str(&format!("- [ ] Prepare step {i}: confirm the version\n"));
    }
    for i in 0..6 {
        s.push_str(&format!("- [ ] Testing step {i}: run the suite\n"));
    }
    for i in 0..6 {
        s.push_str(&format!(
            "- [ ] Publish step {i}: publishing the packages\n"
        ));
    }
    for i in 0..6 {
        s.push_str(&format!(
            "- [ ] Announce step {i}: announcing the release\n"
        ));
    }
    s
}

// ---- extraction ----

#[test]
fn g3_goal_is_fully_recognized() {
    let ex = analyze(G3_GOAL);
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    assert_eq!(
        ex.requirements,
        [
            MinLines(35),
            MinItems(ItemKind::Sections, 6),
            MinSentencesPerSection(2),
            RequiredHeadings(
                [
                    "Welcome",
                    "Accounts and access",
                    "Tools to install",
                    "Your first week",
                    "Who to ask",
                    "Glossary"
                ]
                .map(String::from)
                .to_vec()
            ),
        ]
    );
}

#[test]
fn g2_goal_is_fully_recognized() {
    let ex = analyze(G2_GOAL);
    assert!(ex.uncertain.is_empty(), "{:?}", ex.uncertain);
    assert_eq!(
        ex.requirements,
        [
            MinLines(25),
            RequiredTopics(
                ["preparing", "testing", "publishing", "announcing a release"]
                    .map(String::from)
                    .to_vec()
            )
        ]
    );
}

#[test]
fn line_count_phrasings() {
    for (g, want) in [
        ("at least 20 lines", MinLines(20)),
        ("at least twenty lines", MinLines(20)),
        ("a document of at least 20 lines with a title", MinLines(20)),
        ("a note of at least 35 lines long", MinLines(35)),
        ("no fewer than 20 lines", MinLines(20)),
        ("not fewer than 20 lines.", MinLines(20)),
        ("20 or more lines", MinLines(20)),
        ("20+ lines", MinLines(20)),
        ("a minimum of 12 lines", MinLines(12)),
        ("more than 20 lines", MinLines(21)),
        ("at most 8 lines", MaxLines(8)),
        ("no more than eight lines", MaxLines(8)),
        ("8 or fewer lines", MaxLines(8)),
        ("up to 8 lines", MaxLines(8)),
        ("fewer than 9 lines", MaxLines(8)),
        ("at least twenty-five lines", MinLines(25)),
        ("at least 200 words", MinWords(200)),
        ("at most 50 words.", MaxWords(50)),
        ("at least 3 headings", MinItems(ItemKind::Sections, 3)),
    ] {
        let ex = analyze(g);
        assert!(ex.uncertain.is_empty(), "{g}: {:?}", ex.uncertain);
        assert_eq!(ex.requirements, [want], "{g}");
    }
}

#[test]
fn uncertain_spans_are_reported_not_dropped() {
    for (g, span) in [
        (
            "show at least 3 lines of context",
            "at least 3 lines of context",
        ),
        (
            "at least 5 items per category",
            "at least 5 items per category",
        ),
        ("at most 5 items", "at most 5 items"),
        ("at most 3 paragraphs", "at most 3 paragraphs"),
        ("exactly 20 lines", "exactly 20 lines"),
        ("about 30 lines", "about 30 lines"),
        ("between 10 and 20 items", "between 10 and 20 items"),
        ("write **at least 20 lines**", "at least 20 lines"),
        ("at least 20 long lines", "at least 20 long lines"),
        ("do not write at least 20 lines", "at least 20 lines"),
        ("a 35-line document", "35-line document"),
        ("write at least 2 sentences", "at least 2 sentences"),
        (
            "do not include the phrase \"X\"",
            "include the phrase \"X\"",
        ),
        (
            "include the phrase \"X\" in the title",
            "include the phrase",
        ),
        ("with 5 sections titled A, B", "sections titled A, B"),
        (
            "include the words alpha beta gamma",
            "include the words alpha",
        ),
    ] {
        let ex = analyze(g);
        assert!(
            ex.uncertain.iter().any(|u| u.contains(span)),
            "{g}: {:?}",
            ex.uncertain
        );
        let why = ex.refusal().unwrap();
        assert!(why.starts_with("uncertain requirement"), "{why}");
    }
}

#[test]
fn clean_goals_have_no_uncertain_spans() {
    for g in [
        "write a long document",
        "do not use tables. Write at least 20 lines.",
        "Add the line \"- add dark mode\" under the \"## Planned\" heading in ROADMAP.md.",
        "list three reasons",
    ] {
        assert!(analyze(g).uncertain.is_empty(), "{g}: {:?}", analyze(g));
    }
}

#[test]
fn headings_words_phrases_and_topics_extract() {
    assert_eq!(
        extract("a guide with sections named Intro, Usage and FAQ."),
        [RequiredHeadings(vec![
            "Intro".into(),
            "Usage".into(),
            "FAQ".into()
        ])]
    );
    assert_eq!(
        extract("3 sections called \"Alpha beta\", \"Gamma\" and \"Delta\""),
        [RequiredHeadings(vec![
            "Alpha beta".into(),
            "Gamma".into(),
            "Delta".into()
        ])]
    );
    assert_eq!(
        extract("The text must include the words alpha, beta and gamma."),
        [RequiredWords(vec![
            "alpha".into(),
            "beta".into(),
            "gamma".into()
        ])]
    );
    assert_eq!(
        extract("use the words \"red\" and \"blue\""),
        [RequiredWords(vec!["red".into(), "blue".into()])]
    );
    assert_eq!(
        extract("include the phrases \"good day\", \"all clear\""),
        [RequiredPhrases(vec!["good day".into(), "all clear".into()])]
    );
    assert_eq!(
        extract("a guide that should cover setup, usage and cleanup in at least 30 lines"),
        [
            MinLines(30),
            RequiredTopics(vec!["setup".into(), "usage".into(), "cleanup".into()])
        ]
    );
    assert_eq!(
        extract("In every section, write at least three sentences."),
        [MinSentencesPerSection(3)]
    );
    // "a cover letter" is not a topic list
    assert!(analyze("write a cover letter").requirements.is_empty());
}

// ---- checks ----

#[test]
fn headings_must_be_heading_lines_and_ignore_numbering_case_and_punctuation() {
    let r = RequiredHeadings(vec!["Your first week".into(), "Glossary".into()]);
    let ok = "# 1. Your First Week:\ntext\n## **glossary**\nx\n";
    assert!(r.check(ok).is_none());
    // a plain line, or a heading inside a code fence, is not a heading
    let bad = "Your first week\n```\n# Glossary\n```\n";
    let why = r.check(bad).unwrap();
    assert!(
        why.contains("missing \"Your first week\", \"Glossary\""),
        "{why}"
    );
}

#[test]
fn required_words_are_whole_words_case_insensitive() {
    let r = RequiredWords(vec!["Blue".into(), "red".into()]);
    assert!(r.check("A RED car, a blue one.").is_none());
    let why = r.check("bluebird and reddish").unwrap();
    assert!(why.contains("missing \"Blue\", \"red\""), "{why}");
}

#[test]
fn topic_word_forms_are_conservative() {
    let r = RequiredTopics(vec![
        "preparing".into(),
        "publishing".into(),
        "announcing a release".into(),
    ]);
    // different suffixes agree
    assert!(r
        .check("Prepare it. Publish it. We announce this release.")
        .is_none());
    // nearby words that do not share the stem do not
    let why = r.check("Prepare it. Make it public. The announcement of a release.");
    let why = why.unwrap();
    assert!(why.contains("not covered: publishing"), "{why}");
    // code blocks are not prose
    let why = RequiredTopics(vec!["testing".into()])
        .check("```\ntesting\n```\nnothing here\n")
        .unwrap();
    assert!(why.contains("testing"), "{why}");
}

#[test]
fn sentences_per_section_counts_prose_under_each_heading() {
    let r = MinSentencesPerSection(2);
    let ok =
        "# Title\n## A\nOne sentence here. Two sentences here.\n## B\nFirst one! Second one?\n";
    assert!(r.check(ok).is_none(), "{:?}", r.check(ok));
    let bad =
        "## A\nOne sentence here. Two sentences here.\n## B\nOnly one sentence here.\n## C\n\n";
    let why = r.check(bad).unwrap();
    assert!(
        why.contains("B (1)") && why.contains("C (0)") && !why.contains("A ("),
        "{why}"
    );
    assert!(r
        .check("no headings at all. really.\n")
        .unwrap()
        .contains("no headings"));
    // decimals and version numbers do not end a sentence
    assert!(MinSentencesPerSection(2)
        .check("## A\nUse python 3.10 now. Then 2.5 more.\n")
        .is_none());
}

#[test]
fn word_counts_and_multiple_code_blocks() {
    assert!(MinWords(5).check("a b c d e").is_none());
    assert!(MinWords(6).check("a b c d e").unwrap().contains("found 5"));
    // headings after code blocks, and text after several code blocks, still count
    let doc = "# One\n```rust\n# not a heading\n```\ntext\n```\n## also not\n```\n## Real\ntail text here. And more.\n";
    assert!(RequiredHeadings(vec!["One".into(), "Real".into()])
        .check(doc)
        .is_none());
    assert!(RequiredHeadings(vec!["not a heading".into()])
        .check(doc)
        .is_some());
    assert!(MinItems(ItemKind::Sections, 2).check(doc).is_none());
    assert!(MinItems(ItemKind::Sections, 3)
        .check(doc)
        .unwrap()
        .contains("found 2"));
    assert!(RequiredTopics(vec!["tail".into()]).check(doc).is_none());
}

// ---- the recorded v3 replies ----

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn recorded_g3_reply_is_refused_and_a_good_reply_passes() {
    let _home = home_guard::home_slot();
    let ex = analyze(G3_GOAL);
    let why = refusal_reason(&ex.requirements, &content_of(G3_V3_REPLY)).unwrap();
    assert!(
        why.contains("at least 35 non-empty lines, found 13"),
        "{why}"
    );
    let (r, prompts) = run(G3_GOAL, vec![G3_V3_REPLY.into(), g3_good()]);
    assert_eq!(r.proposal.as_deref(), Some(g3_good().as_str()), "{r:?}");
    let a = &r.proposal_attempts;
    assert_eq!(a[0].outcome, "refused");
    assert!(a[0].unmet_requirements[0].contains("found 13"));
    assert_eq!(a[1].outcome, "parsed");
    // the retry prompt names what was missed
    assert!(
        prompts[1].contains("at least 35 non-empty lines, found 13"),
        "{}",
        prompts[1]
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn recorded_g3_reply_exhausts_with_nothing_approved() {
    let _home = home_guard::home_slot();
    let (r, _) = run(G3_GOAL, vec![G3_V3_REPLY.into()]);
    assert!(
        r.proposal.is_none() && r.uncommitted_proposal.is_none(),
        "{r:?}"
    );
    let why = r.proposer_error.unwrap();
    assert!(why.contains("found 13"), "{why}");
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), 3);
    assert!(a
        .iter()
        .all(|x| x.outcome == "refused" && x.aegis.is_none()));
}

#[test]
fn g3_wrong_headings_and_short_sections_are_named() {
    let mut s = String::from("filename: docs/ONBOARDING.md\n");
    for t in ["Welcome", "Setup", "Glossary"] {
        s.push_str(&format!("# {t}\n\nOnly one sentence here.\n"));
    }
    for i in 0..30 {
        s.push_str(&format!("filler line {i}\n"));
    }
    let why = refusal_reason(&analyze(G3_GOAL).requirements, &content_of(&s)).unwrap();
    assert!(why.contains("at least 6 headings, found 3"), "{why}");
    assert!(why.contains("missing \"Accounts and access\""), "{why}");
    assert!(why.contains("too few in: Welcome (1)"), "{why}");
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn recorded_g2_reply_is_refused_and_a_good_reply_passes() {
    let _home = home_guard::home_slot();
    let ex = analyze(G2_GOAL);
    let why = refusal_reason(&ex.requirements, &content_of(G2_V3_REPLY)).unwrap();
    assert!(why.contains("not covered: publishing"), "{why}");
    let (r, prompts) = run(G2_GOAL, vec![G2_V3_REPLY.into(), g2_good()]);
    assert_eq!(r.proposal.as_deref(), Some(g2_good().as_str()), "{r:?}");
    assert_eq!(r.proposal_attempts[0].outcome, "refused");
    assert!(
        prompts[1].contains("not covered: publishing"),
        "{}",
        prompts[1]
    );
    // the recorded reply is exhausted when the model keeps repeating it
    let (r, _) = run(G2_GOAL, vec![G2_V3_REPLY.into()]);
    assert!(r.proposal.is_none() && r.proposer_error.is_some(), "{r:?}");
    assert_eq!(r.proposal_attempts.len(), 3);
}

// ---- through the real compose run (linked build only) ----

fn bridge_with(
    tmp: &std::path::Path,
    f: impl Fn(&str) -> Result<Generation, String> + Send + Sync + 'static,
) -> ComposeBridge {
    let proposer: ComposeProposer = Arc::new(move |p: &str, _: Duration| f(p));
    ComposeBridge::new(tmp.join("compose"), proposer, "test:enforce")
}

fn result(resp: ControlResponse) -> Option<aien_runtime::control::ComposeTaskReport> {
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
fn linked_uncertain_goal_is_refused_before_any_model_call() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let calls = Arc::new(Mutex::new(0usize));
    let c = calls.clone();
    let bridge = bridge_with(tmp.path(), move |_| {
        *c.lock().unwrap() += 1;
        gen("filename: DOC.md\nhello\n")
    });
    let resp = bridge.run_task(
        "write DOC.md with at least 3 lines of context",
        ws.to_str().unwrap(),
    );
    let Some(r) = result(resp) else { return };
    assert_eq!(*calls.lock().unwrap(), 0);
    assert!(!r.committed, "{r:?}");
    assert!(r.proposal.is_none());
    assert_eq!(r.requirements_uncertain, ["at least 3 lines of context"]);
    assert!(r
        .proposer_error
        .unwrap()
        .starts_with("uncertain requirement"));
    assert_eq!(std::fs::read_dir(&ws).unwrap().count(), 0);
}

#[test]
fn linked_g3_original_reply_is_refused_then_a_good_reply_commits_exact_bytes() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let calls = Arc::new(Mutex::new(0usize));
    let c = calls.clone();
    let bridge = bridge_with(tmp.path(), move |_| {
        let mut n = c.lock().unwrap();
        *n += 1;
        let good = g3_good();
        gen(if *n == 1 { G3_V3_REPLY } else { &good })
    });
    let resp = bridge.run_task(G3_GOAL, ws.to_str().unwrap());
    let Some(r) = result(resp) else { return };
    assert!(r.requirements_uncertain.is_empty());
    assert_eq!(r.proposal_attempts.len(), 2, "{r:?}");
    assert_eq!(r.proposal_attempts[0].outcome, "refused");
    assert!(r.committed, "{r:?}");
    let p = check_file_proposal(r.proposal.as_deref().unwrap()).unwrap();
    assert_eq!(p.content, content_of(&g3_good()));
    assert!(refusal_reason(&analyze(G3_GOAL).requirements, &p.content).is_none());
}

#[test]
fn linked_g2_original_reply_exhausts_and_nothing_is_written() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let bridge = bridge_with(tmp.path(), |_| gen(G2_V3_REPLY));
    let resp = bridge.run_task(G2_GOAL, ws.to_str().unwrap());
    let Some(r) = result(resp) else { return };
    assert!(!r.committed, "{r:?}");
    assert!(r.proposal.is_none());
    assert_eq!(r.proposal_attempts.len(), 3);
    assert!(r
        .proposer_error
        .unwrap()
        .contains("not covered: publishing"));
    assert_eq!(std::fs::read_dir(&ws).unwrap().count(), 0);
}

// ---- review round: nothing that looks like a requirement passes silently ----

#[test]
fn sixteen_review_goals_are_recognized_or_uncertain_never_silent() {
    for g in [
        "run to 30 lines or more",
        "25 lines minimum",
        "Keep it to 10 lines",
        "no longer than 40 lines",
        "Limit it to 80 words",
        "Use each of these words: alpha, beta",
        "Include headings for A and B",
        "needs a Summary heading and a Risks heading",
        "three sections",
        "ten bullet points",
        "Give it two code blocks",
        "One code block only",
        "Three paragraphs",
        "Cover at least 4 topics: a, b, c, d",
        "Each section needs 2 sentences",
        "say the word 'canary' at least twice",
    ] {
        let ex = analyze(g);
        assert!(
            !ex.requirements.is_empty() || !ex.uncertain.is_empty(),
            "silent: {g}"
        );
    }
    // the common ones are recognized properly
    assert_eq!(extract("run to 30 lines or more"), [MinLines(30)]);
    assert_eq!(extract("25 lines minimum"), [MinLines(25)]);
    assert_eq!(extract("no longer than 40 lines"), [MaxLines(40)]);
}

#[test]
fn false_positive_guards_stay_silent() {
    for g in [
        "Fix the 2 typos",
        "version 3.2 notes",
        "10 minute setup guide",
        "5 tests",
        "Add the line \"- x\" under the \"## Planned\" heading in ROADMAP.md.",
        "Update section 2 of the README",
        "Create 2024-report.md with a short summary",
    ] {
        let ex = analyze(g);
        assert!(
            ex.requirements.is_empty() && ex.uncertain.is_empty(),
            "{g}: {ex:?}"
        );
    }
}

#[test]
fn a_count_after_covering_is_a_count_or_uncertain_never_a_topic() {
    let ex = analyze("Write GUIDE.md covering 5 steps.");
    assert!(ex.requirements.is_empty(), "{ex:?}");
    assert!(!ex.uncertain.is_empty());
}

#[test]
fn trailing_courtesy_words_are_tolerated() {
    assert_eq!(extract("At most 15 lines please."), [MaxLines(15)]);
    assert!(analyze("At most 15 lines please.").uncertain.is_empty());
    assert_eq!(extract("at least 12 lines, thanks"), [MinLines(12)]);
}
