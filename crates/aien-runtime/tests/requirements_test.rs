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

//! Requirement validation for composed documents (crate::requirements).
//! The retry loop is driven through `ComposeBridge::run_task`
//! (tests/support/compose_harness.rs).
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::approved::{approved_proposal_sha256, proposal_text};
use aien_runtime::control::ControlResponse;
use aien_runtime::effects;
use aien_runtime::requirements::{extract, refusal_reason, unmet, ItemKind, Requirement};
use aien_runtime::{check_file_proposal, ComposeBridge, ComposeProposer, Generation};
use compose_harness::{always, replies, Asked, Harness};
use std::sync::{Arc, Mutex};
use std::time::Duration;

fn doc(n: usize) -> String {
    let mut s = String::from("filename: DOC.md\n");
    for i in 0..n {
        s.push_str(&format!("line {i}\n"));
    }
    s
}

fn gen(text: String) -> Result<Generation, String> {
    Ok(Generation {
        text,
        tokens: 5,
        finish_reason: Some("eos".into()),
        ..Default::default()
    })
}

#[test]
fn extracts_each_supported_pattern() {
    use ItemKind::*;
    use Requirement::*;
    assert_eq!(
        extract("Write a note in at least 20 lines."),
        [MinLines(20)]
    );
    assert_eq!(extract("keep it to AT MOST 8 lines"), [MaxLines(8)]);
    assert_eq!(extract("give at least 5 items"), [MinItems(Items, 5)]);
    assert_eq!(extract("give at least 1 item"), [MinItems(Items, 1)]);
    assert_eq!(
        extract("list at least 3 steps, please"),
        [MinItems(Steps, 3)]
    );
    assert_eq!(extract("with at least 4 sections"), [MinItems(Sections, 4)]);
    assert_eq!(
        extract("ask at least 6 questions."),
        [MinItems(Questions, 6)]
    );
    assert_eq!(
        extract("include the phrase \"Hello World\" and contain the phrase \"x\""),
        [RequiredPhrases(vec!["Hello World".into(), "x".into()])]
    );
    assert_eq!(
        extract("at least 20 lines and at most 40 lines"),
        [MinLines(20), MaxLines(40)]
    );
}

#[test]
fn unsupported_phrasing_is_not_extracted() {
    for g in [
        "write a long document",
        "at least 20 long lines",
        "at most 20 paragraphs",
        "at most 5 items",
        "at least 0 lines",
        "at least -3 lines",
        "at least 2.5 lines",
        "at least 99999999999999999999999 lines",
        "include the phrase hello",
        "include the phrase \"\"",
        "mention hello in the text",
        "exactly 20 lines",
    ] {
        assert_eq!(extract(g), vec![], "{g}");
    }
}

#[test]
fn checks_pass_and_fail_with_exact_counts() {
    use ItemKind::*;
    use Requirement::*;
    let thirteen: String = (0..13).map(|i| format!("l{i}\n\n")).collect();
    assert_eq!(non_empty_blank_safe(&thirteen), 13);
    assert!(MinLines(13).check(&thirteen).is_none());
    assert_eq!(
        MinLines(20).check(&thirteen).unwrap(),
        "at least 20 non-empty lines, found 13"
    );
    assert!(MaxLines(13).check(&thirteen).is_none());
    assert!(MaxLines(12).check(&thirteen).is_some());
    let md = "# One\n## Two\n#NoSpace\n####### seven\n- a\n  * b\n+ c\n1. d\n2) e\n3.f\nWhy?\nreally ?\n";
    assert!(MinItems(Sections, 2).check(md).is_none());
    assert!(MinItems(Sections, 3).check(md).is_some());
    assert!(MinItems(Items, 5).check(md).is_none());
    assert!(MinItems(Items, 6).check(md).is_some());
    assert!(MinItems(Steps, 2).check(md).is_none());
    assert!(MinItems(Steps, 3).check(md).is_some());
    assert!(MinItems(Questions, 2).check(md).is_none());
    assert!(MinItems(Questions, 3).check(md).is_some());
    let p = RequiredPhrases(vec!["hello world".into(), "Bye".into()]);
    assert_eq!(
        p.check("HeLLo WoRLD only").unwrap(),
        "the phrase \"hello world\", \"Bye\", missing \"Bye\""
    );
    assert!(p.check("Hello World and BYE").is_none());
    assert_eq!(unmet(&[], "x"), Vec::<String>::new());
    assert_eq!(refusal_reason(&[], "x"), None);
}

fn non_empty_blank_safe(s: &str) -> usize {
    s.lines().filter(|l| !l.trim().is_empty()).count()
}

/// The goal of the document tests below: a new DOC.md, at least 20 lines.
const DOC_GOAL: &str = "write DOC.md in at least 20 lines";

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn retry_names_the_unmet_requirement_and_second_attempt_passes() {
    let _home = home_guard::home_slot();
    let h = Harness::new(replies(&[&doc(13), &doc(25)]));
    let r = h.run(DOC_GOAL).unwrap();
    assert_eq!(r.proposal.as_deref(), Some(doc(25).as_str()), "{r:?}");
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), 2);
    assert_eq!(a[0].outcome, "refused");
    assert_eq!(
        a[0].reason.as_deref(),
        Some("unmet requirement: at least 20 non-empty lines, found 13")
    );
    assert_eq!(
        a[0].unmet_requirements,
        ["at least 20 non-empty lines, found 13"]
    );
    assert_eq!(a[1].outcome, "parsed");
    assert!(a[1].unmet_requirements.is_empty());
    let p: Vec<String> = h.calls().into_iter().map(|c| c.prompt).collect();
    assert!(!p[0].contains("unmet requirement"));
    assert!(
        p[1].contains("refused before anything was written (unmet requirement: at least 20 non-empty lines, found 13)"),
        "{}",
        p[1]
    );
    // sc#384 v8: the refused reply itself is quoted back for correction.
    assert!(
        p[1].contains(&format!("<<<REFUSED ANSWER\n{}", doc(13))),
        "{}",
        p[1]
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn exhaustion_returns_no_proposal() {
    let _home = home_guard::home_slot();
    let h = Harness::new(always(&doc(13)));
    let r = h.run(DOC_GOAL).unwrap();
    assert!(
        r.proposal.is_none() && r.uncommitted_proposal.is_none(),
        "{r:?}"
    );
    let why = r.proposer_error.unwrap();
    assert!(
        why.contains("at least 20 non-empty lines, found 13"),
        "{why}"
    );
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), 3);
    assert!(a
        .iter()
        .all(|x| x.outcome == "refused" && x.aegis.is_none()));
    assert!(a.iter().all(|x| x.unmet_requirements.len() == 1));
}

// deadline_is_shared_across_attempts needs a short document budget: it is in
// compose_budget_test.rs (same name).

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn validated_document_is_the_one_approval_binds() {
    let _home = home_guard::home_slot();
    let goal = "write DOC.md with at least 4 lines";
    let reqs = extract("at least 4 lines");
    assert_eq!(extract(goal), reqs);
    let h = Harness::new(always("filename: DOC.md\na\nb\nc\nd\n"));
    let r = h.run(goal).unwrap();
    let text = r.proposal.unwrap();
    let saved = check_file_proposal(&text).unwrap();
    // The content that passed is the content that would be written ...
    assert!(refusal_reason(&reqs, &saved.content).is_none());
    // ... and it reads back byte for byte through the approval template, so
    // the digest an approval binds is the digest of the saved bytes.
    assert_eq!(proposal_text(&saved.path, &saved.content).unwrap(), text);
    let bound = approved_proposal_sha256(&saved.path, &saved.content);
    let again = check_file_proposal(&proposal_text(&saved.path, &saved.content).unwrap()).unwrap();
    assert_eq!(approved_proposal_sha256(&again.path, &again.content), bound);
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn edit_mode_validates_the_merged_file_not_the_reply() {
    let _home = home_guard::home_slot();
    // The prior file has 3 lines; the reply adds 2: the saved file has 5.
    let h = Harness::new(always("filename: N.md\n# T\na\nnew1\nnew2\nb\n"));
    h.write("N.md", "# T\na\nb\n");
    let goal = |n: usize| format!("Edit N.md so that it has at least {n} lines.");
    assert_eq!(extract(&goal(5)), [Requirement::MinLines(5)]);
    assert_eq!(extract(&goal(6)), [Requirement::MinLines(6)]);
    let r = h.run(&goal(5)).unwrap();
    assert_eq!(
        check_file_proposal(&r.proposal.unwrap()).unwrap().content,
        "# T\na\nnew1\nnew2\nb\n"
    );
    let r = h.run(&goal(6)).unwrap();
    assert!(r.proposal.is_none() && r.proposer_error.is_some(), "{r:?}");
    assert_eq!(
        r.proposal_attempts[0].unmet_requirements,
        ["at least 6 non-empty lines, found 5"]
    );
    assert!(h.calls().iter().all(|c| c.asked == Asked::Edit));
}

#[test]
fn unmet_requirement_leaves_nothing_committed_or_written() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let proposer: ComposeProposer = Arc::new(|_: &str, _: Duration| gen(doc(13)));
    let bridge = ComposeBridge::new(tmp.path().join("compose"), proposer, "test:short")
        .with_authorize_requires_desk(false);
    let resp = bridge.run_task("write DOC.md in at least 20 lines", ws.to_str().unwrap());
    if !aien_omega_compose::LINKED {
        assert!(matches!(resp, ControlResponse::Error(_)));
        return;
    }
    let ControlResponse::ComposeTaskResult(r) = resp else {
        panic!("{resp:?}")
    };
    assert!(!r.committed, "{r:?}");
    assert!(r.proposal.is_none());
    assert_eq!(r.requirements_recognized, ["at least 20 non-empty lines"]);
    assert_eq!(r.proposal_attempts.len(), 3);
    assert!(r.proposer_error.unwrap().contains("found 13"));
    assert_eq!(std::fs::read_dir(&ws).unwrap().count(), 0);
}

// ---- linked proofs (real omega compose library; skipped in the CPU stub) ----

fn sha256_hex(b: &[u8]) -> String {
    aien_omega_compose::hex(&<sha2::Sha256 as sha2::Digest>::digest(b))
}

fn recalled(b: &ComposeBridge) -> aien_runtime::control::ComposeRecallReport {
    match b.recall(&[], None) {
        ControlResponse::ComposeRecalled(r) => *r,
        other => panic!("recall: {other:?}"),
    }
}

fn dir_names(d: &std::path::Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(d)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// Every attempt of a real compose run fails the requirement: nothing is
/// proposed, approved, written, committed or recorded as an effect.
#[test]
fn linked_exhaustion_leaves_ledger_state_and_files_untouched() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let home = tmp.path().join("compose");
    let calls = Arc::new(Mutex::new(0usize));
    let c2 = calls.clone();
    let proposer: ComposeProposer = Arc::new(move |_: &str, _: Duration| {
        *c2.lock().unwrap() += 1;
        gen(doc(13))
    });
    let bridge = ComposeBridge::new(home.clone(), proposer, "test:short")
        .with_authorize_requires_desk(false);
    let resp = bridge.run_task("write DOC.md in at least 20 lines", ws.to_str().unwrap());
    if !aien_omega_compose::LINKED {
        assert!(matches!(resp, ControlResponse::Error(_)));
        return;
    }
    let ControlResponse::ComposeTaskResult(r) = resp else {
        panic!("{resp:?}")
    };
    // The real model Skill ran 3 times (the attempt cap), each refused.
    assert_eq!(*calls.lock().unwrap(), 3);
    assert!(r
        .proposal_attempts
        .iter()
        .all(|a| a.outcome == "refused" && a.aegis.is_none()));
    // No committed proposal, no winner, no AEGIS pass, no promotion record.
    assert!(!r.committed, "{r:?}");
    assert_eq!(r.outcome, 2, "RXC_OUT_NO_WINNER: {r:?}");
    assert!(r.proposal.is_none() && r.proposal_sha256.is_none());
    assert!(r.proposal_path.is_none() && r.proposal_content_sha256.is_none());
    assert_eq!((r.winner, r.aegis_pass_mask), (None, 0), "{r:?}");
    assert_eq!(r.cx_promotion, 0, "{r:?}");
    // No approval request, no effect intent: the host ledger holds no
    // authorization or effect note, and rebuilds to an empty ledger.
    let rec = recalled(&bridge);
    assert!(
        rec.host
            .iter()
            .all(|h| !matches!(h.note.as_deref(), Some("authorization") | Some("effect"))),
        "{:?}",
        rec.host
    );
    let ledger = effects::Ledger::from_records(&rec.host).unwrap();
    assert!(ledger.grants.is_empty() && ledger.intents.is_empty());
    assert!(ledger.approved_by_claim.is_empty() && ledger.spent.is_empty());
    // No file was written and the destination is absent.
    assert!(!ws.join("DOC.md").exists());
    assert_eq!(dir_names(&ws), Vec::<String>::new());
    // The compose home holds only composition files.
    for n in dir_names(&home) {
        assert!(
            n == "machine.id" || n == "cortex.cx" || n.starts_with("jspace"),
            "{n}"
        );
    }
    // Compose state is unchanged by the failed task: a second identical task
    // adds no promotion and the committed World has no new document.
    let before = recalled(&bridge).records_total;
    let again = bridge.run_task("write DOC.md in at least 20 lines", ws.to_str().unwrap());
    let ControlResponse::ComposeTaskResult(r2) = again else {
        panic!("{again:?}")
    };
    assert!(!r2.committed && r2.cx_promotion == 0, "{r2:?}");
    assert!(!ws.join("DOC.md").exists());
    let after = recalled(&bridge);
    assert!(after.records_total >= before);
    assert!(after
        .host
        .iter()
        .all(|h| !matches!(h.note.as_deref(), Some("authorization") | Some("effect"))));
}

/// A document that passes validation commits through the real compose run,
/// is approved by an authorization bound to its digest, and is saved with
/// sha256(disk) == the approved content sha256.
#[test]
fn linked_passing_document_is_approved_and_saved_byte_for_byte() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let ws = std::fs::canonicalize(&ws).unwrap();
    let proposer: ComposeProposer = Arc::new(|_: &str, _: Duration| gen(doc(25)));
    let bridge = ComposeBridge::new(tmp.path().join("compose"), proposer, "test:long")
        .with_authorize_requires_desk(false);
    let resp = bridge.run_task("write DOC.md in at least 20 lines", ws.to_str().unwrap());
    if !aien_omega_compose::LINKED {
        assert!(matches!(resp, ControlResponse::Error(_)));
        return;
    }
    let ControlResponse::ComposeTaskResult(r) = resp else {
        panic!("{resp:?}")
    };
    assert!(r.committed, "{r:?}");
    assert_eq!(r.proposal_attempts.len(), 1);
    assert_eq!(r.proposal_attempts[0].aegis.as_deref(), Some("pass"));
    let proposal = r.proposal.clone().unwrap();
    let saved = check_file_proposal(&proposal).unwrap();
    assert!(refusal_reason(&extract("at least 20 lines"), &saved.content).is_none());
    let content_sha = r.proposal_content_sha256.clone().unwrap();
    assert_eq!(content_sha, sha256_hex(saved.content.as_bytes()));
    assert!(!ws.join("DOC.md").exists(), "composing writes nothing");
    // Approval (the authorization note) binds the digest; the effect writes.
    let target = ws.join(&saved.path);
    // sovereign-core #261: the daemon mints the grant from its own commit record.
    let a = match effects::mint_grant(
        &bridge,
        &effects::MintRequest {
            cx_promotion: r.cx_promotion,
            proposal_sha256: r.proposal_sha256.clone().unwrap(),
            workspace: ws.display().to_string(),
            approver: "drake".into(),
            constraints: vec![],
        },
    ) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("authorize: {other:?}"),
    };
    let (pid, start) = effects::self_executor();
    let i = match effects::open_intent(
        &bridge,
        &effects::IntentRequest {
            authorization: a,
            proposal_sha256: r.proposal_sha256.clone().unwrap(),
            path: saved.path.clone(),
            target: target.display().to_string(),
            content_sha256: content_sha.clone(),
            executor_pid: pid,
            executor_start: start,
        },
    ) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("intent: {other:?}"),
    };
    std::fs::write(&target, &saved.content).unwrap();
    assert!(matches!(
        effects::ack(
            &bridge,
            i,
            &serde_json::json!({"written_sha256": content_sha})
        ),
        ControlResponse::ComposeNoted(_)
    ));
    assert_eq!(sha256_hex(&std::fs::read(&target).unwrap()), content_sha);
    let ledger = effects::Ledger::from_records(&recalled(&bridge).host).unwrap();
    assert_eq!(ledger.intents.len(), 1);
}

// ---- review fixes (#287) ----

#[test]
fn count_followed_by_a_qualifier_is_not_extracted() {
    // The noun must END the clause: "N lines of context" states something
    // other than "the document has N lines", so it gets no check.
    for g in [
        "show a diff with at least 3 lines of context",
        "write at least 2 sections of the config",
        "at least 5 items per category",
        "at least 3 steps for each stage",
        "at least 4 questions in the survey",
        "at least 4 questions from the list",
        "at least 6 lines for the intro",
        "at most 10 lines in each file",
    ] {
        assert_eq!(extract(g), vec![], "{g}");
    }
    use Requirement::*;
    // Clause-ending punctuation, end of goal, and the joining words are fine.
    for (g, want) in [
        ("at least 20 lines", MinLines(20)),
        ("at least 20 lines.", MinLines(20)),
        ("at least 20 lines, please", MinLines(20)),
        ("at least 20 lines and a title", MinLines(20)),
        ("at least 20 lines or more", MinLines(20)),
        ("at least 20 lines that explain it", MinLines(20)),
        ("at most 8 lines which are short", MaxLines(8)),
        ("at least 6 lines with comments", MinLines(6)),
        ("at least 6 lines about cats", MinLines(6)),
        ("at least 20 lines long", MinLines(20)),
    ] {
        assert_eq!(extract(g), [want], "{g}");
    }
}

#[test]
fn covering_after_a_count_adds_a_topic_requirement() {
    assert_eq!(
        extract("at least 20 lines covering the topic"),
        [
            Requirement::MinLines(20),
            Requirement::RequiredTopics(vec!["the topic".into()])
        ]
    );
}

#[test]
fn phrase_followed_by_a_qualifier_is_not_extracted() {
    // "in the title" would make the whole-document check claim more than it covers.
    assert_eq!(extract("include the phrase \"X\" in the title"), vec![]);
    assert_eq!(
        extract("include the phrase \"X\" in the title and contain the phrase \"Y\"."),
        [Requirement::RequiredPhrases(vec!["Y".into()])]
    );
}

#[test]
fn negated_wording_is_not_extracted() {
    for g in [
        "do not include the phrase \"X\"",
        "Do NOT include the phrase \"X\"",
        "never contain the phrase \"X\"",
        "the file must not include the phrase \"X\"",
        "don't include the phrase \"X\"",
        "it doesn't contain the phrase \"X\"",
        "without ever including; never include the phrase \"X\"",
        "not at least 20 lines",
        "do not write at least 20 lines",
        "never write at least 3 sections",
        "it must not have at most 5 lines",
        "not at most 10 lines",
        "without at least 4 items",
    ] {
        assert_eq!(extract(g), vec![], "{g}");
    }
    // A negation in an EARLIER clause does not poison a later requirement.
    assert_eq!(
        extract("do not use tables. Write at least 20 lines."),
        [Requirement::MinLines(20)]
    );
    assert_eq!(
        extract("do not hedge, include the phrase \"X\""),
        [Requirement::RequiredPhrases(vec!["X".into()])]
    );
}

#[test]
fn emphasis_and_quoted_forms_are_unrecognized() {
    // Documented limitation: markdown emphasis or quotes around the wording
    // break the exact word sequence, so nothing is extracted (and nothing is
    // claimed as covered).
    for g in [
        "write **at least 20 lines**",
        "write \"at least 20 lines\"",
        "write 'at least 20 lines'",
        "write at least **20** lines",
        "write at least 20 *lines*",
        "write at least 20 `lines`",
    ] {
        assert_eq!(extract(g), vec![], "{g}");
    }
}

#[test]
fn fenced_code_is_not_counted_as_headings_questions_items_or_steps() {
    use ItemKind::*;
    let d = "# Real\n- a\n1. s\nWhy?\n```bash\n# not a heading\n- not an item\n1. not a step\nnot a question?\n```\n~~~\n## also not\n- nor this\nq?\n~~~\n";
    assert_eq!(count(Sections, d), 1);
    assert_eq!(count(Items, d), 2);
    assert_eq!(count(Steps, d), 1);
    assert_eq!(count(Questions, d), 1);
    // CommonMark close rule: a shorter or info-carrying fence does not close.
    let d2 = "````\n```\n# inside\n```bash\n- inside\n````\n# outside\n";
    assert_eq!(count(Sections, d2), 1);
    // ~~~ is not closed by ``` and vice versa.
    let d3 = "~~~\n```\n# inside\n~~~\n# outside\n";
    assert_eq!(count(Sections, d3), 1);
    // An unclosed fence runs to the end of the document.
    assert_eq!(count(Sections, "# a\n```\n# b\n# c\n"), 1);
    // A backtick fence whose info string holds a backtick is not a fence.
    assert_eq!(count(Sections, "``` `x`\n# real\n"), 1);
}

fn count(k: ItemKind, s: &str) -> usize {
    // found N appears in the unmet sentence; ask for an impossible minimum.
    let m = Requirement::MinItems(k, usize::MAX).check(s).unwrap();
    m.rsplit("found ").next().unwrap().parse().unwrap()
}

#[test]
fn line_limits_count_every_non_empty_line_including_code_and_fences() {
    // Documented: MinLines/MaxLines count lines of the file, so fence marker
    // lines and code lines count; blank lines do not.
    let d = "intro\n```rust\nlet x = 1;\n\nlet y = 2;\n```\nend\n";
    assert_eq!(Requirement::MinLines(6).check(d), None);
    assert_eq!(
        Requirement::MinLines(7).check(d).unwrap(),
        "at least 7 non-empty lines, found 6"
    );
    assert_eq!(Requirement::MaxLines(6).check(d), None);
    assert!(Requirement::MaxLines(5).check(d).is_some());
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn edit_whose_prior_file_already_breaks_a_max_is_refused_before_any_attempt() {
    let _home = home_guard::home_slot();
    let prior: String = (0..10).map(|i| format!("l{i}\n")).collect();
    let h = Harness::new(always("filename: N.md\nl0\nnew\n"));
    h.write("N.md", &prior);
    let goal = |n: usize| format!("Add a line to N.md and keep it to at most {n} lines.");
    assert_eq!(extract(&goal(5)), [Requirement::MaxLines(5)]);
    assert_eq!(extract(&goal(12)), [Requirement::MaxLines(12)]);
    let r = h.run(&goal(5)).unwrap();
    assert_eq!(h.calls().len(), 0, "no attempt may be spent");
    assert!(r.proposal.is_none() && !r.committed, "{r:?}");
    let why = r.proposer_error.unwrap();
    assert!(
        why.contains("N.md") && why.contains("at most 5 non-empty lines, found 10"),
        "{why}"
    );
    assert!(why.contains("an edit only adds lines"), "{why}");
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), 1);
    assert_eq!(a[0].outcome, "refused");
    assert_eq!(a[0].reason.as_deref(), Some(why.as_str()));
    assert_eq!(
        a[0].unmet_requirements,
        ["at most 5 non-empty lines, found 10"]
    );
    // Min requirements and a prior file within the maximum are unaffected.
    let r = h.run(&goal(12)).unwrap();
    assert!(r.proposal.is_some(), "{r:?}");
    assert_eq!(h.calls().len(), 1);
}
