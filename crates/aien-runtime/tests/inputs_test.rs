//! sc#382 and sc#383 items 2 and 3 (WHOLE-SYSTEM-E2E, arch#190): input files the
//! goal points at are shown to the model and never written, and "named X in the
//! Y folder" resolves to `Y/X`. Pre-registered in
//! `docs/campaigns/whole-system-e2e/REGRESSION-sc382-sc383.md`.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::control::ControlResponse;
use aien_runtime::spine::{
    task_decision, task_decision_with_inputs, ComposeBridge, ComposeProposer, Generation,
    COMPOSE_INPUTS_MAX_BYTES,
};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// The three objective texts of scripts/whole_system_e2e.sh, verbatim.
const OBJ: &str = "Read the three text files in the inbox folder and write a Markdown report named report.md in the outbox folder. Start with a Markdown heading line that begins with \"# \". Use at most 200 words. Mention each file by name.";
const OBJ6: &str = "Write a file named done.md in the outbox folder containing one line: finished.";
const OBJC: &str = "Write a file named ctrl.md in the outbox folder containing one line: control.";
// The first mechanics objective of harness v1 (sc#383 item 2).
const OBJ_NAMED: &str = "Write a Markdown report named report.md about the three inbox files named harbour-notes.txt, orchard-ledger.txt and windmill-log.txt. Start with a Markdown heading line that begins with \"# \". Use at most 200 words. Mention each file by name.";

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

fn ws_with_inbox() -> (tempfile::TempDir, PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let ws = d.path().join("ws");
    std::fs::create_dir_all(ws.join("inbox")).unwrap();
    std::fs::create_dir_all(ws.join("outbox")).unwrap();
    // Written out of order on purpose: the order must come from the names.
    for (n, c) in FILES.iter().rev() {
        std::fs::write(ws.join("inbox").join(n), c).unwrap();
    }
    let ws = std::fs::canonicalize(ws).unwrap();
    (d, ws)
}

fn sha(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

fn dest_of(goal: &str, ws: &Path) -> Option<String> {
    task_decision(goal, ws).unwrap().1
}

#[test]
fn folder_phrase_resolves_done_ctrl_and_report_destinations() {
    let (_d, ws) = ws_with_inbox();
    assert_eq!(dest_of(OBJ6, &ws).as_deref(), Some("outbox/done.md"));
    assert_eq!(dest_of(OBJC, &ws).as_deref(), Some("outbox/ctrl.md"));
    assert_eq!(dest_of(OBJ, &ws).as_deref(), Some("outbox/report.md"));
    // The model is shown the resolved path.
    let ((prompt, _, _), _) = task_decision(OBJC, &ws).unwrap();
    assert!(prompt.contains("filename: outbox/ctrl.md"), "{prompt}");
}

#[test]
fn folder_phrase_works_without_the_article_and_for_directory() {
    let (_d, ws) = ws_with_inbox();
    assert_eq!(
        dest_of(
            "Write a file named a.md in outbox folder containing x.",
            &ws
        )
        .as_deref(),
        Some("outbox/a.md")
    );
    assert_eq!(
        dest_of(
            "Write a file named b.md in the outbox directory containing x.",
            &ws
        )
        .as_deref(),
        Some("outbox/b.md")
    );
}

#[test]
fn input_names_are_not_write_targets() {
    let (_d, ws) = ws_with_inbox();
    let (_, dest, _, _) = task_decision_with_inputs(OBJ_NAMED, &ws).unwrap();
    assert_eq!(dest.as_deref(), Some("report.md"));
}

#[test]
fn two_distinct_write_targets_stay_ambiguous() {
    let (_d, ws) = ws_with_inbox();
    let e = task_decision("Create alpha.md and beta.md about the weather.", &ws).unwrap_err();
    assert!(e.contains("ambiguous destination"), "{e}");
    // A name that does not exist is not an input: two new files are still two targets.
    let e = task_decision(
        "Write a report named one.md and write another named two.md about the inbox folder.",
        &ws,
    )
    .unwrap_err();
    assert!(e.contains("ambiguous destination"), "{e}");
}

#[test]
fn existing_destination_is_still_an_edit() {
    let (_d, ws) = ws_with_inbox();
    std::fs::write(ws.join("README.md"), "# r\n").unwrap();
    let ((prompt, target, _), dest) = task_decision("Append a line to README.md", &ws).unwrap();
    assert_eq!(dest.as_deref(), Some("README.md"));
    assert!(target.is_some() && prompt.contains("already exists"));
}

#[test]
fn folder_source_phrase_includes_inputs_sorted_with_digests() {
    let (_d, ws) = ws_with_inbox();
    let (((prompt, _, _), dest), inputs, omitted) = {
        let (p, d, i, o) = task_decision_with_inputs(OBJ, &ws).unwrap();
        ((p, d), i, o)
    };
    assert_eq!(dest.as_deref(), Some("outbox/report.md"));
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
fn named_files_become_inputs_but_destination_is_not() {
    let (_d, ws) = ws_with_inbox();
    // The goal names the files bare and says where they live.
    let goal = "Write a Markdown report named report.md in the outbox folder about harbour-notes.txt, orchard-ledger.txt and windmill-log.txt in the inbox folder.";
    let (((prompt, _, _), dest), inputs, _) = {
        let (p, d, i, o) = task_decision_with_inputs(goal, &ws).unwrap();
        ((p, d), i, o)
    };
    assert_eq!(dest.as_deref(), Some("outbox/report.md"));
    assert_eq!(inputs.len(), 3, "{inputs:?}");
    assert!(prompt.contains("Windmill log. The brake band was replaced."));
    // A bare name that exists at the workspace root is an input too.
    std::fs::write(ws.join("facts.txt"), "Fact one.\n").unwrap();
    let (_, dest, inputs, _) =
        task_decision_with_inputs("Write summary.md based on the notes named facts.txt", &ws)
            .unwrap();
    assert_eq!(dest.as_deref(), Some("summary.md"));
    assert_eq!(
        inputs.iter().map(|i| i.name.as_str()).collect::<Vec<_>>(),
        ["facts.txt"]
    );
}

#[test]
fn inputs_are_capped_and_omissions_are_listed() {
    let (_d, ws) = ws_with_inbox();
    let big = "x".repeat(COMPOSE_INPUTS_MAX_BYTES / 2 + 1);
    std::fs::write(ws.join("inbox").join("big1.txt"), &big).unwrap();
    std::fs::write(ws.join("inbox").join("big2.txt"), &big).unwrap();
    let (_, _, inputs, omitted) = task_decision_with_inputs(OBJ, &ws).unwrap();
    let total: usize = inputs.iter().map(|i| i.bytes).sum();
    assert!(total <= COMPOSE_INPUTS_MAX_BYTES, "{total}");
    assert!(inputs.iter().any(|i| i.name == "inbox/big1.txt"));
    assert!(
        omitted.iter().any(|o| o.name == "inbox/big2.txt"),
        "{omitted:?}"
    );
    // Deterministic: the same goal gives the same lists.
    let again = task_decision_with_inputs(OBJ, &ws).unwrap();
    assert_eq!((again.2, again.3), (inputs, omitted));
}

#[test]
fn non_utf8_and_non_regular_inputs_are_omitted_not_read() {
    let (_d, ws) = ws_with_inbox();
    std::fs::write(ws.join("inbox").join("bin.dat"), [0xff, 0xfe, 0x00]).unwrap();
    std::fs::create_dir(ws.join("inbox").join("sub")).unwrap();
    let out = tempfile::tempdir().unwrap();
    std::fs::write(out.path().join("secret.txt"), "TOPSECRET").unwrap();
    std::os::unix::fs::symlink(
        out.path().join("secret.txt"),
        ws.join("inbox").join("leak.txt"),
    )
    .unwrap();
    let ((prompt, _, _), _, inputs, omitted) = task_decision_with_inputs(OBJ, &ws).unwrap();
    assert!(!prompt.contains("TOPSECRET"));
    assert!(inputs
        .iter()
        .all(|i| i.name != "inbox/bin.dat" && i.name != "inbox/leak.txt"));
    for n in ["inbox/bin.dat", "inbox/leak.txt"] {
        assert!(omitted.iter().any(|o| o.name == n), "{n}: {omitted:?}");
    }
}

#[test]
fn named_input_escaping_workspace_is_refused_before_model() {
    let (d, ws) = ws_with_inbox();
    std::fs::write(d.path().join("outside.txt"), "TOPSECRET").unwrap();
    std::os::unix::fs::symlink(d.path().join("outside.txt"), ws.join("link.txt")).unwrap();
    for goal in [
        "Write report.md about ../outside.txt",
        "Write report.md about /etc/hostname",
        "Write report.md about link.txt",
    ] {
        let e = task_decision_with_inputs(goal, &ws).unwrap_err();
        assert!(
            e.contains("RunComposeTask") && e.contains("input"),
            "{goal}: {e}"
        );
    }
}

#[test]
fn goal_without_inputs_keeps_prompt_byte_for_byte() {
    let (_d, ws) = ws_with_inbox();
    let goal = "Write a file named notes.md containing one line: hello.";
    let a = task_decision(goal, &ws).unwrap();
    let b = task_decision_with_inputs(goal, &ws).unwrap();
    assert_eq!(a.0, b.0);
    assert!(b.2.is_empty() && b.3.is_empty());
    assert!(!a.0 .0.contains("Input files"));
    let w = ws.display().to_string();
    let want = format!(
        "{}{}",
        aien_runtime::spine::proposal_prompt(goal, &w),
        aien_runtime::spine::new_document_block("notes.md")
    );
    assert_eq!(a.0 .0, want);
}

fn scripted(seen: Arc<std::sync::Mutex<String>>) -> ComposeProposer {
    Arc::new(move |prompt: &str, _l: std::time::Duration| {
        *seen.lock().unwrap() = prompt.to_string();
        Ok(Generation {
            text: "filename: outbox/report.md\n# Report\nharbour-notes.txt, orchard-ledger.txt, windmill-log.txt\n".into(),
            tokens: 9,
            finish_reason: None,
            ..Default::default()
        })
    })
}

#[test]
fn compose_propose_report_lists_inputs_with_digests() {
    let _home = home_guard::home_slot();
    let (d, ws) = ws_with_inbox();
    let seen = Arc::new(std::sync::Mutex::new(String::new()));
    let bridge = ComposeBridge::new(
        d.path().join("compose"),
        scripted(seen.clone()),
        "test:scripted",
    );
    let resp = bridge.run_task(OBJ, ws.to_str().unwrap());
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
    assert!(seen.lock().unwrap().contains("Windmill log."));
    assert_eq!(r.inputs.len(), 3);
    for ((n, c), i) in FILES.iter().zip(&r.inputs) {
        assert_eq!(i.name, format!("inbox/{n}"));
        assert_eq!(i.sha256, sha(c));
    }
}
