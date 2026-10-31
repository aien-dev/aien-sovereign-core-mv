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

//! NEXT-PHASE-1 v6 runtime cuts (ACCEPTANCE-v6 Section 6): the length-cut
//! refusal (N2), the token ids in the attempt record (R1) and the edit-mode
//! prompt block (T5). A deterministic scripted proposer; no model.
//!
//! Every test drives `ComposeBridge::run_task` (tests/support/compose_harness.rs)
//! and reads the prompts the scripted model was handed, which proposer was
//! asked, the refusal text of a goal refused before the model, and the run's
//! attempt record.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::{token_ids_sha256, Generation};
use compose_harness::{silent, Asked, Call, Harness};
use std::sync::Arc;

/// A plain new-document goal whose destination is NOTES.md.
const GOAL: &str = "write a short note into NOTES.md";
/// Proposals per task.
const MAX_ATTEMPTS: usize = 3;
/// Largest existing file the edit block carries, in bytes.
const EDIT_MAX_BYTES: usize = 8192;

/// The NEXT-PHASE-1 seed workspace: README.md and docs/plan.txt.
fn seed(h: &Harness) {
    h.write(
        "README.md",
        "# Demo project\n\nA small local project used by the NEXT-PHASE-1 campaign.\n",
    );
    h.write("docs/plan.txt", "Plan: keep notes short and local.\n");
}

/// The fixed proposal template every prompt starts with (the product's text,
/// verbatim; the workspace is named by its logical label).
fn template(goal: &str) -> String {
    format!(
        "Goal: {goal}\nAuthorized workspace: workspace\n\
         Propose exactly one file change inside the workspace.\n\
         Answer in exactly this format and nothing else:\n\
         filename: <relative path>\n\
         <the complete new file content>"
    )
}

/// The new-document block (the product's text, verbatim).
fn new_block(path: &str) -> String {
    format!("\nWrite the new file {path}. The filename line of your answer must be exactly: filename: {path}")
}

/// The edit block (the product's text, verbatim).
fn edit_block(path: &str, content: &str) -> String {
    let nl = if content.ends_with('\n') { "" } else { "\n" };
    format!(
        "\nThe file {path} already exists. Its current content is:\n{content}{nl}\
         Write the complete new content of {path}: keep every existing line and make the requested change."
    )
}

/// The first model call of a run of `goal`, or the refusal text when the
/// bridge refused the goal before any model call.
fn first(h: &Harness, goal: &str) -> Result<Call, String> {
    h.clear();
    match h.run(goal) {
        Err(e) => {
            assert!(
                h.calls().is_empty(),
                "{goal}: a refused goal reached the model"
            );
            Err(e)
        }
        Ok(r) => Ok(h
            .calls()
            .into_iter()
            .next()
            .unwrap_or_else(|| panic!("{goal}: no model call: {r:?}"))),
    }
}

/// N2: a reply that stopped at the token limit is refused even when it
/// parses; every attempt is recorded with its stop reason and nothing passes.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn length_cut_reply_is_refused_even_when_it_parses() {
    let _home = home_guard::home_slot();
    // Calls 1..=3 (the first run) stop at the token limit; call 4 (the second
    // run) at eos; call 5 (the third run) with an unknown stop reason.
    let h = Harness::new(Arc::new(|n, _, _| {
        Ok(Generation {
            text: "filename: NOTES.md\nkeep every change\n".into(),
            tokens: 4,
            finish_reason: match n {
                1..=3 => Some("max_tokens".into()),
                4 => Some("eos".into()),
                _ => None,
            },
            ..Default::default()
        })
    }));
    let r = h.run(GOAL).unwrap();
    let why = r.proposer_error.clone().unwrap();
    assert!(why.contains("finish_reason max_tokens"), "{why}");
    assert_eq!((r.proposal, r.uncommitted_proposal), (None, None));
    let a = &r.proposal_attempts;
    assert_eq!(a.len(), MAX_ATTEMPTS);
    for x in a {
        assert_eq!(x.outcome, "refused");
        assert_eq!(x.finish_reason.as_deref(), Some("max_tokens"));
        assert!(x.reason.as_deref().unwrap().contains("token limit"));
    }
    // eos and an unknown stop reason are not length cuts: the same reply parses.
    for want in [Some("eos"), None] {
        let r = h.run(GOAL).unwrap();
        let a = &r.proposal_attempts;
        assert_eq!(a.len(), 1, "{a:?}");
        assert_eq!(a[0].outcome, "parsed", "{a:?}");
        assert_eq!(a[0].reason, None);
        assert_eq!(a[0].finish_reason.as_deref(), want);
        assert_eq!(r.proposer_error, None);
    }
    assert_eq!(h.calls().len(), MAX_ATTEMPTS + 2);
}

/// R1: the generated ids, the prompt id count and the prompt ids digest
/// reach the attempt record unchanged; an older record without them reads.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn token_ids_reach_the_attempt_record() {
    let _home = home_guard::home_slot();
    let ids = vec![63045u32, 22030, 198];
    let p = vec![128000u32, 128006, 882];
    let (ids2, p2) = (ids.clone(), p.clone());
    let h = Harness::new(Arc::new(move |_, _, _| {
        Ok(Generation {
            text: "filename: NOTES.md\nok\n".into(),
            tokens: 3,
            finish_reason: Some("eos".into()),
            token_ids: Some(ids2.clone()),
            prompt_tokens: Some(p2.len()),
            prompt_ids_sha256: Some(token_ids_sha256(&p2)),
            decoding: None,
            ops: None,
        })
    }));
    let r = h.run(GOAL).unwrap();
    assert_eq!(r.proposer_error, None);
    let a = &r.proposal_attempts;
    assert_eq!(a[0].outcome, "parsed");
    assert_eq!(a[0].token_ids.as_deref(), Some(ids.as_slice()));
    assert_eq!(a[0].prompt_tokens, Some(3));
    assert_eq!(
        a[0].prompt_ids_sha256.as_deref(),
        Some(token_ids_sha256(&p).as_str())
    );
    // 4 little-endian bytes per id.
    let mut bytes = Vec::new();
    for x in &p {
        bytes.extend_from_slice(&x.to_le_bytes());
    }
    use sha2::Digest;
    let want: String = sha2::Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(token_ids_sha256(&p), want);
    let old = r#"{"attempt":1,"ms":1,"tokens":48,"outcome":"parsed","reason":null,
        "text_sha256":null,"text":null,"aegis":null,"finish_reason":"eos"}"#;
    let r: aien_runtime::control::ProposalAttempt = serde_json::from_str(old).unwrap();
    assert_eq!(
        (r.token_ids, r.prompt_tokens, r.prompt_ids_sha256),
        (None, None, None)
    );
}

/// T5: the v5 goals on the v5 seed keep the v5 prompt byte for byte; a goal
/// naming an existing file gets the block with that file's content.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn edit_block_only_when_the_goal_names_an_existing_file() {
    let _home = home_guard::home_slot();
    let h = Harness::new(silent());
    seed(&h);
    // A new-document goal keeps the v5 prompt and gains the destination line
    // (issue #288); an unsafe destination is refused before any model call.
    for (goal, dest) in [
        ("Create the file NOTES.md with a short plain-text note that says the project keeps every change inside its workspace.", Some("NOTES.md")),
        ("Create the file docs/CONTACT.txt with one line giving the maintainer name Ada Lovelace and the email ada@example.org.", Some("docs/CONTACT.txt")),
        ("Create the file TODO.md listing three tasks: write tests, update the changelog, tag the release.", Some("TODO.md")),
        ("Create the file ../outside.txt with the single line: written outside the workspace.", None),
        ("Create the file docs/GUIDE.md about docs and plan.txt.", Some("docs/GUIDE.md")),
    ] {
        let Some(d) = dest else {
            let e = first(&h, goal).expect_err(goal);
            assert!(e.contains("destination ../outside.txt"), "{goal}: {e}");
            continue;
        };
        let c = first(&h, goal).unwrap_or_else(|e| panic!("{goal}: refused: {e}"));
        assert_eq!(c.asked, Asked::Document, "{goal}");
        assert_eq!(c.prompt, format!("{}{}", template(goal), new_block(d)), "{goal}");
    }
    let log = "# Changelog\n\n## 0.1.0\n- initial release\n";
    h.write("CHANGELOG.md", log);
    let goal =
        "Add the line \"- add contact file\" under the \"## 0.1.0\" heading in CHANGELOG.md.";
    let c = first(&h, goal).unwrap();
    assert_eq!(c.asked, Asked::Edit);
    let p = c.prompt;
    assert_eq!(
        p,
        format!("{}{}", template(goal), edit_block("CHANGELOG.md", log))
    );
    assert!(p.contains("\nThe file CHANGELOG.md already exists. Its current content is:\n# Changelog\n\n## 0.1.0\n- initial release\nWrite the complete new content of CHANGELOG.md"));
    // Quoted and nested paths are found; the first existing one wins.
    let c = first(&h, "Edit `docs/plan.txt`, then README.md").unwrap();
    assert_eq!(c.asked, Asked::Edit);
    assert!(
        c.prompt.ends_with(&edit_block(
            "docs/plan.txt",
            "Plan: keep notes short and local.\n"
        )),
        "{}",
        c.prompt
    );
}

/// T5 boundary: the block never reads outside the workspace, a directory,
/// a symlink leaving the workspace, a non-UTF-8 file or an oversized file.
/// The offline re-derivation (`compose::rederive::task_prompt`, used by
/// examples/np1_reference.rs) keeps the plain v5 prompt, byte for byte, for a
/// goal whose destination is unsafe. The bridge itself refuses that goal
/// before any model call (`edit_block_only_when_the_goal_names_an_existing_file`).
#[test]
fn rederive_keeps_the_plain_prompt_for_an_unsafe_destination() {
    use aien_runtime::compose::rederive::{proposal_prompt, task_prompt};
    let d = tempfile::tempdir().unwrap();
    let goal =
        "Create the file ../outside.txt with the single line: written outside the workspace.";
    assert_eq!(proposal_prompt(goal), template(goal));
    assert_eq!(task_prompt(goal, d.path()), template(goal));
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn edit_block_never_reads_outside_or_oversized() {
    let _home = home_guard::home_slot();
    let h = Harness::new(silent());
    seed(&h);
    let d = h.tmp.path();
    std::fs::write(d.join("secret.txt"), "outside\n").unwrap();
    std::os::unix::fs::symlink(d.join("secret.txt"), h.ws().join("link.txt")).unwrap();
    std::fs::write(h.ws().join("bin.dat"), [0xffu8, 0xfe, 0x00]).unwrap();
    h.write("big.txt", &"x".repeat(EDIT_MAX_BYTES + 1));
    for goal in [
        "Edit ../secret.txt now",
        &format!("Edit {}", d.join("secret.txt").display()),
        "Edit link.txt now",
        "Edit docs now",
        "Edit bin.dat now",
        "Edit big.txt now",
        "Edit ~/secret.txt now",
    ] {
        // Refused before the model, or a new document: never an edit target.
        match first(&h, goal) {
            Ok(c) => {
                assert_eq!(c.asked, Asked::Document, "{goal}");
                assert!(!c.prompt.contains("already exists"), "{goal}: {}", c.prompt);
            }
            Err(e) => assert!(e.starts_with("RunComposeTask: destination "), "{goal}: {e}"),
        }
    }
}
