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

//! The omega settle wait really set before each run, and the target-kind
//! decision (existing path => Edit/refused, missing path => Document).
//! One test binary, one wait test: `LAST_WAIT_MS` is process-global.
//!
//! Every test drives `ComposeBridge::run_task` (tests/support/compose_harness.rs)
//! and reads which proposer was asked, the prompt and limit it was handed, and
//! the run's report. A goal refused before any model call comes back as `Err`.
#[path = "support/compose_harness.rs"]
mod compose_harness;
#[path = "support/home_guard.rs"]
mod home_guard;

use compose_harness::{first_call, reply, Asked, Harness};
use std::sync::atomic::Ordering;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Linked build only: run_task_inner tells omega budget + 1000 ms for each
/// kind before compose.run (observed inside the proposer, i.e. during the run).
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn run_task_sets_omega_wait_to_budget_plus_margin_for_both_kinds() {
    let _home = home_guard::home_slot();
    let seen: Arc<Mutex<Vec<u32>>> = Arc::default();
    let s = seen.clone();
    let h = Harness::new(Arc::new(move |_, _: &str, _: Duration| {
        s.lock()
            .unwrap()
            .push(aien_omega_compose::LAST_WAIT_MS.load(Ordering::SeqCst));
        reply("filename: faq.md\nconstraint: x\n")
    }));
    h.write("notes.md", "# notes\n");
    for goal in ["add a line to notes.md", "write a new file faq.md"] {
        h.run(goal).unwrap();
    }
    let seen = seen.lock().unwrap().clone();
    assert_eq!(
        seen,
        vec![30_000, 121_000],
        "wait seen during each run: edit 30000, document 121000"
    );
    let asked: Vec<Asked> = h.calls().iter().map(|c| c.asked).collect();
    assert_eq!(asked, [Asked::Edit, Asked::Document]);
}

/// One deadline across attempts: each retry is given only what is left of
/// the document budget, never a fresh budget.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn one_deadline_is_shared_across_attempts() {
    let _home = home_guard::home_slot();
    let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
        std::thread::sleep(Duration::from_millis(60));
        reply("not a proposal")
    }));
    let r = h.run("write a new file faq.md").unwrap();
    assert!(r.proposal.is_none() && r.proposer_error.is_some(), "{r:?}");
    let c = h.calls();
    assert!(c.iter().all(|c| c.asked == Asked::Document));
    let l: Vec<Duration> = c.iter().map(|c| c.limit).collect();
    assert!(l.len() >= 2, "{l:?}");
    // the default document budget, 120 s
    assert!(l[0] <= Duration::from_secs(120));
    for w in l.windows(2) {
        assert!(w[1] + Duration::from_millis(55) <= w[0], "{l:?}");
    }
    assert_eq!(r.proposal_attempts.len(), l.len());
}

fn ws_with(files: &[(&str, &[u8])]) -> (tempfile::TempDir, std::path::PathBuf) {
    let d = tempfile::tempdir().unwrap();
    let ws = d.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    for (n, c) in files {
        std::fs::write(ws.join(n), c).unwrap();
    }
    let ws = std::fs::canonicalize(ws).unwrap();
    (d, ws)
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn missing_file_is_document() {
    let _home = home_guard::home_slot();
    let (_d, ws) = ws_with(&[]);
    let (c, _) = first_call("write faq.md", &ws).unwrap();
    // a new document: the document proposer, no edit block, the new-file block
    assert_eq!(c.asked, Asked::Document);
    assert!(!c.prompt.contains("already exists"), "{}", c.prompt);
    assert!(
        c.prompt
            .contains("Write the new file faq.md. The filename line of your answer must be exactly: filename: faq.md"),
        "{}",
        c.prompt
    );
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn small_existing_file_is_edit() {
    let _home = home_guard::home_slot();
    let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
        reply("filename: notes.md\n# notes\nnew line\n")
    }));
    h.write("notes.md", "# notes\n");
    let r = h.run("add a line to notes.md").unwrap();
    let c = h.calls();
    assert_eq!(c[0].asked, Asked::Edit);
    assert!(c[0].prompt.contains("already exists"), "{}", c[0].prompt);
    // the edit target: notes.md with exactly its content, shown and merged into
    assert!(
        c[0].prompt
            .contains("The file notes.md already exists. Its current content is:\n# notes\nWrite"),
        "{}",
        c[0].prompt
    );
    assert_eq!(
        r.proposal.as_deref(),
        Some("filename: notes.md\n# notes\nnew line\n"),
        "{r:?}"
    );
}

#[test]
fn large_existing_file_is_refused_not_document() {
    let _home = home_guard::home_slot();
    let big = vec![b'a'; 11 * 1024];
    let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
        reply("filename: big.md\nx\n")
    }));
    std::fs::write(h.ws().join("big.md"), &big).unwrap();
    let e = h.run("rewrite big.md").unwrap_err();
    assert!(e.contains("big.md") && e.contains("already exists"), "{e}");
    // refused, never a new document: no proposer was asked
    assert!(h.calls().is_empty());
    assert_eq!(std::fs::read(h.ws().join("big.md")).unwrap(), big);
}

#[test]
fn non_utf8_existing_file_is_refused() {
    let _home = home_guard::home_slot();
    let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
        reply("filename: bin.md\nx\n")
    }));
    std::fs::write(h.ws().join("bin.md"), [0xff, 0xfe, 0x00, 0x80]).unwrap();
    let e = h.run("rewrite bin.md").unwrap_err();
    assert!(e.contains("not UTF-8"), "{e}");
    assert!(h.calls().is_empty());
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn outward_symlink_is_refused_and_directory_is_skipped() {
    let _home = home_guard::home_slot();
    let h = Harness::new(Arc::new(|_, _: &str, _: Duration| {
        reply("filename: docs/faq.md\nq and a\n")
    }));
    let outside = h.tmp.path().join("secret.txt");
    std::fs::write(&outside, "s\n").unwrap();
    std::fs::create_dir_all(h.ws().join("docs")).unwrap();
    std::os::unix::fs::symlink(&outside, h.ws().join("link.md")).unwrap();
    assert!(h.run("rewrite link.md").is_err());
    assert!(h.calls().is_empty());
    // the directory `docs` is not the destination: a new document
    h.run("put it in docs faq.md").unwrap();
    let c = h.calls();
    assert!(!c.is_empty());
    assert_eq!(c[0].asked, Asked::Document);
    assert!(!c[0].prompt.contains("already exists"), "{}", c[0].prompt);
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "s\n");
}
