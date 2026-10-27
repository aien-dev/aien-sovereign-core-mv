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

//! sc#382 and sc#383 items 2 and 3 (WHOLE-SYSTEM-E2E, arch#190): input files the
//! goal points at are shown to the model and never written, and "named X in the
//! Y folder" resolves to `Y/X`. Pre-registered in
//! `docs/campaigns/whole-system-e2e/REGRESSION-sc382-sc383.md`.
//!
//! Every test drives `ComposeBridge::run_task` (tests/support/compose_harness.rs):
//! the destination is the new-document or edit block the prompt ends with,
//! the inputs are the prompt's input block and the report's `inputs` and
//! `inputs_omitted`, and a refused goal is the bridge's refusal text.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::control::{ComposeTaskReport, ControlResponse};
use aien_runtime::Generation;
use compose_harness::{silent, Asked, Harness, Script};
use sha2::{Digest, Sha256};
use std::sync::Arc;

// The three objective texts of scripts/whole_system_e2e.sh, verbatim.
const OBJ: &str = "Read the three text files in the inbox folder and write a Markdown report named report.md in the outbox folder. Start with a Markdown heading line that begins with \"# \". Use at most 200 words. Mention each file by name.";
const OBJ6: &str = "Write a file named done.md in the outbox folder containing one line: finished.";
const OBJC: &str = "Write a file named ctrl.md in the outbox folder containing one line: control.";
// The first mechanics objective of harness v1 (sc#383 item 2).
const OBJ_NAMED: &str = "Write a Markdown report named report.md about the three inbox files named harbour-notes.txt, orchard-ledger.txt and windmill-log.txt. Start with a Markdown heading line that begins with \"# \". Use at most 200 words. Mention each file by name.";

/// Total bytes of input text a prompt carries (the product's limit).
const INPUTS_MAX_BYTES: usize = 16384;

const FILES: [(&str, &str); 3] = [
    (
        "harbour-notes.txt",
        "Harbour notes. Three boats came in before noon.\n",
    ),
    ("orchard-ledger.txt", "Orchard ledger. Forty apple trees.\n"),
    (
        "windmill-log.txt",
        "Windmill log. The brake band was replaced.\n",
    ),
];

/// A harness whose model runs `script` and whose workspace holds the inbox
/// and an empty outbox.
fn inbox(script: Script) -> Harness {
    let h = Harness::new(script);
    std::fs::create_dir_all(h.ws().join("inbox")).unwrap();
    std::fs::create_dir_all(h.ws().join("outbox")).unwrap();
    // Written out of order on purpose: the order must come from the names.
    for (n, c) in FILES.iter().rev() {
        h.write(&format!("inbox/{n}"), c);
    }
    h
}

/// `inbox` with a silent model.
fn ws_with_inbox() -> Harness {
    inbox(silent())
}

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// The first prompt of a run of `goal` and the run's report.
fn first(h: &Harness, goal: &str) -> (String, ComposeTaskReport) {
    h.clear();
    let r = h
        .run(goal)
        .unwrap_or_else(|e| panic!("{goal}: refused: {e}"));
    let c = h.calls();
    let p = c
        .first()
        .unwrap_or_else(|| panic!("{goal}: no model call: {r:?}"))
        .prompt
        .clone();
    (p, r)
}

/// The bridge's refusal of `goal`; no model call was made.
fn refused(h: &Harness, goal: &str) -> String {
    h.clear();
    let e = h.run(goal).expect_err(goal);
    assert!(
        h.calls().is_empty(),
        "{goal}: a refused goal reached the model"
    );
    e
}

/// The new-document block the prompt ends with when the goal names a
/// destination that does not exist yet (the product's text, verbatim).
fn new_block(path: &str) -> String {
    format!("\nWrite the new file {path}. The filename line of your answer must be exactly: filename: {path}")
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

/// Asserts the destination of `goal` is the new document `want`.
fn assert_new_dest(h: &Harness, goal: &str, want: &str) {
    let (p, _) = first(h, goal);
    assert!(p.ends_with(&new_block(want)), "{goal}: {p}");
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn folder_phrase_resolves_done_ctrl_and_report_destinations() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    assert_new_dest(&h, OBJ6, "outbox/done.md");
    assert_new_dest(&h, OBJC, "outbox/ctrl.md");
    assert_new_dest(&h, OBJ, "outbox/report.md");
    // The model is shown the resolved path.
    let (prompt, _) = first(&h, OBJC);
    assert!(prompt.contains("filename: outbox/ctrl.md"), "{prompt}");
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn folder_phrase_works_without_the_article_and_for_directory() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    assert_new_dest(
        &h,
        "Write a file named a.md in outbox folder containing x.",
        "outbox/a.md",
    );
    assert_new_dest(
        &h,
        "Write a file named b.md in the outbox directory containing x.",
        "outbox/b.md",
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn input_names_are_not_write_targets() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    assert_new_dest(&h, OBJ_NAMED, "report.md");
}

#[test]
fn two_distinct_write_targets_stay_ambiguous() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    let e = refused(&h, "Create alpha.md and beta.md about the weather.");
    assert!(e.contains("ambiguous destination"), "{e}");
    // A name that does not exist is not an input: two new files are still two targets.
    let e = refused(
        &h,
        "Write a report named one.md and write another named two.md about the inbox folder.",
    );
    assert!(e.contains("ambiguous destination"), "{e}");
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn existing_destination_is_still_an_edit() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    h.write("README.md", "# r\n");
    let c = {
        h.clear();
        h.run("Append a line to README.md").unwrap();
        h.calls().remove(0)
    };
    // The edit target, and so the destination, is README.md.
    assert_eq!(c.asked, Asked::Edit);
    assert!(
        c.prompt
            .contains("\nThe file README.md already exists. Its current content is:\n# r\n"),
        "{}",
        c.prompt
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn folder_source_phrase_includes_inputs_sorted_with_digests() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    let (prompt, r) = first(&h, OBJ);
    let (inputs, omitted) = (r.inputs, r.inputs_omitted);
    assert!(prompt.ends_with(&new_block("outbox/report.md")), "{prompt}");
    assert!(omitted.is_empty(), "{omitted:?}");
    let names: Vec<&str> = inputs.iter().map(|i| i.name.as_str()).collect();
    assert_eq!(
        names,
        [
            "inbox/harbour-notes.txt",
            "inbox/orchard-ledger.txt",
            "inbox/windmill-log.txt"
        ]
    );
    let mut at = 0;
    for ((n, c), i) in FILES.iter().zip(&inputs) {
        assert_eq!((i.sha256.clone(), i.bytes), (sha(c), c.len()));
        let p = prompt[at..]
            .find(c)
            .unwrap_or_else(|| panic!("{n} missing: {prompt}"));
        at += p;
        assert!(prompt.contains(&format!("inbox/{n}")));
    }
    // The inputs are not the file to write.
    assert!(prompt.contains("filename: outbox/report.md"));
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn named_files_become_inputs_but_destination_is_not() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    // The goal names the files bare and says where they live.
    let goal = "Write a Markdown report named report.md in the outbox folder about harbour-notes.txt, orchard-ledger.txt and windmill-log.txt in the inbox folder.";
    let (prompt, r) = first(&h, goal);
    assert!(prompt.ends_with(&new_block("outbox/report.md")), "{prompt}");
    assert_eq!(r.inputs.len(), 3, "{:?}", r.inputs);
    assert!(prompt.contains("Windmill log. The brake band was replaced."));
    // A bare name that exists at the workspace root is an input too.
    h.write("facts.txt", "Fact one.\n");
    let (prompt, r) = first(&h, "Write summary.md based on the notes named facts.txt");
    assert!(prompt.ends_with(&new_block("summary.md")), "{prompt}");
    assert_eq!(
        r.inputs.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
        ["facts.txt"]
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn inputs_are_capped_and_omissions_are_listed() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    let big = "x".repeat(INPUTS_MAX_BYTES / 2 + 1);
    h.write("inbox/big1.txt", &big);
    h.write("inbox/big2.txt", &big);
    let (_, r) = first(&h, OBJ);
    let (inputs, omitted) = (r.inputs, r.inputs_omitted);
    let total: usize = inputs.iter().map(|i| i.bytes).sum();
    assert!(total <= INPUTS_MAX_BYTES, "{total}");
    assert!(inputs.iter().any(|i| i.name == "inbox/big1.txt"));
    assert!(
        omitted.iter().any(|o| o.name == "inbox/big2.txt"),
        "{omitted:?}"
    );
    // Deterministic: the same goal gives the same lists.
    let (_, again) = first(&h, OBJ);
    assert_eq!((again.inputs, again.inputs_omitted), (inputs, omitted));
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn non_utf8_and_non_regular_inputs_are_omitted_not_read() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    std::fs::write(h.ws().join("inbox").join("bin.dat"), [0xff, 0xfe, 0x00]).unwrap();
    std::fs::create_dir(h.ws().join("inbox").join("sub")).unwrap();
    let out = tempfile::tempdir().unwrap();
    std::fs::write(out.path().join("secret.txt"), "TOPSECRET").unwrap();
    std::os::unix::fs::symlink(
        out.path().join("secret.txt"),
        h.ws().join("inbox").join("leak.txt"),
    )
    .unwrap();
    let (prompt, r) = first(&h, OBJ);
    assert!(!prompt.contains("TOPSECRET"));
    assert!(r
        .inputs
        .iter()
        .all(|i| i.name != "inbox/bin.dat" && i.name != "inbox/leak.txt"));
    for n in ["inbox/bin.dat", "inbox/leak.txt"] {
        assert!(
            r.inputs_omitted.iter().any(|o| o.name == n),
            "{n}: {:?}",
            r.inputs_omitted
        );
    }
}

// Refused before any model call: runs in the stub build too.
#[test]
fn named_input_escaping_workspace_is_refused_before_model() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    std::fs::write(h.tmp.path().join("outside.txt"), "TOPSECRET").unwrap();
    std::os::unix::fs::symlink(h.tmp.path().join("outside.txt"), h.ws().join("link.txt")).unwrap();
    for goal in [
        "Write report.md about ../outside.txt",
        "Write report.md about /etc/hostname",
        "Write report.md about link.txt",
    ] {
        let e = refused(&h, goal);
        assert!(
            e.contains("RunComposeTask") && e.contains("input"),
            "{goal}: {e}"
        );
    }
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn goal_without_inputs_keeps_prompt_byte_for_byte() {
    let _home = home_guard::home_slot();
    let h = ws_with_inbox();
    let goal = "Write a file named notes.md containing one line: hello.";
    let (prompt, r) = first(&h, goal);
    assert!(r.inputs.is_empty() && r.inputs_omitted.is_empty());
    assert!(!prompt.contains("Input files"));
    assert_eq!(
        prompt,
        format!("{}{}", template(goal), new_block("notes.md"))
    );
}

/// A model reply to the e2e objective: a report naming the three inputs.
const REPORT: &str = "filename: outbox/report.md\n# Report\nharbour-notes.txt, orchard-ledger.txt, windmill-log.txt\n";

#[test]
fn compose_propose_report_lists_inputs_with_digests() {
    let _home = home_guard::home_slot();
    let h = inbox(Arc::new(|_, _, _| {
        Ok(Generation {
            text: REPORT.into(),
            tokens: 9,
            finish_reason: None,
            ..Default::default()
        })
    }));
    let resp = h.bridge.run_task(OBJ, h.ws().to_str().unwrap());
    if !aien_omega_compose::LINKED {
        match resp {
            ControlResponse::Error(e) => assert!(e.contains("not linked"), "{e}"),
            other => panic!("stub build must refuse, got {other:?}"),
        }
        return;
    }
    let ControlResponse::ComposeTaskResult(r) = resp else {
        panic!("{resp:?}")
    };
    assert!(h.calls()[0].prompt.contains("Windmill log."));
    assert_eq!(r.inputs.len(), 3);
    for ((n, c), i) in FILES.iter().zip(&r.inputs) {
        assert_eq!(i.name, format!("inbox/{n}"));
        assert_eq!(i.sha256, sha(c));
    }
}
