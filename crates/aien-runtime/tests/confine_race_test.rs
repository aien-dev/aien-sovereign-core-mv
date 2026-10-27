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

//! sovereign-core #267 follow-up: the REAL `mint_grant` (authorize) and
//! `open_intent` calls, with a swap fired INSIDE the call between the
//! confinement check and the read. Each asserts an exact outcome: the recorded
//! prior is the hash of the file the confinement check saw, or a specific
//! refusal. Needs the linked composition archive; IGNORED in a stub build.
//! branch must give the identical kind for every row.
//!
//! Needs the linked composition archive; IGNORED in a stub build.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ComposeNoteReport, ComposeTaskReport, ControlResponse};
use aien_runtime::effects::test_hooks;
use aien_runtime::effects::{self, IntentRequest, MintRequest};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::Arc;

const PROPOSAL: &str = "filename: d/NOTES.md\nkeep every change inside the workspace\n";
const CONTENT: &str = "keep every change inside the workspace\n";

fn proposer() -> ComposeProposer {
    Arc::new(|_p: &str, _l: std::time::Duration| {
        Ok(Generation {
            text: PROPOSAL.to_string(),
            tokens: 8,
            finish_reason: Some("eos".into()),
            ..Default::default()
        })
    })
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

static ONE_HOME: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Fx {
    _turn: std::sync::MutexGuard<'static, ()>,
    _tmp: tempfile::TempDir,
    root: PathBuf,
    ws: PathBuf,
    b: ComposeBridge,
    report: ComposeTaskReport,
}

impl Fx {
    fn new() -> Option<Fx> {
        let turn = ONE_HOME.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        let ws = root.join("ws");
        std::fs::create_dir_all(ws.join("d")).unwrap();
        let b = ComposeBridge::new(root.join("compose"), proposer(), "test:fixed-proposer")
            .with_authorize_requires_desk(false);
        let run = b.run_task("write the note", ws.to_str().unwrap());
        if !aien_omega_compose::LINKED {
            return None;
        }
        let ControlResponse::ComposeTaskResult(report) = run else {
            panic!("run: {run:?}")
        };
        assert!(report.committed, "{report:?}");
        Some(Fx {
            _turn: turn,
            _tmp: tmp,
            root,
            ws,
            b,
            report: *report,
        })
    }
    fn file(&self) -> PathBuf {
        self.ws.join("d/NOTES.md")
    }
    fn mint_req(&self) -> MintRequest {
        MintRequest {
            cx_promotion: self.report.cx_promotion,
            proposal_sha256: sha(PROPOSAL.as_bytes()),
            workspace: self.ws.display().to_string(),
            approver: "drake".into(),
            constraints: vec![],
        }
    }
    fn req(&self, a: u64, wrong_grant: bool) -> IntentRequest {
        let (pid, start) = effects::self_executor();
        IntentRequest {
            authorization: a,
            proposal_sha256: sha(PROPOSAL.as_bytes()),
            path: "d/NOTES.md".into(),
            target: self.file().display().to_string(),
            content_sha256: if wrong_grant {
                sha(b"other")
            } else {
                sha(CONTENT.as_bytes())
            },
            executor_pid: pid,
            executor_start: start,
        }
    }
}

fn id(r: ControlResponse) -> u64 {
    match r {
        ControlResponse::ComposeNoted(ComposeNoteReport { id, .. }) => id,
        other => panic!("expected a note, got {other:?}"),
    }
}

fn text(fx: &Fx, id: u64) -> serde_json::Value {
    match fx.b.recall(&[id], None) {
        ControlResponse::ComposeRecalled(r) => {
            serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap()
        }
        other => panic!("recall: {other:?}"),
    }
}

/// Move the real `d` aside and put a symlink to `outside` in its place.
fn swap_in_symlink(ws: PathBuf, outside: PathBuf) {
    std::fs::rename(ws.join("d"), ws.join("d.held")).unwrap();
    std::os::unix::fs::symlink(&outside, ws.join("d")).unwrap();
}

/// authorize: `d` is swapped for a symlink to outside bytes after the
/// confinement check and before the read. The grant must record the hash of the
/// inside file (what confinement approved), never the outside bytes.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
fn mint_records_the_prior_of_the_confined_file() {
    let _home = home_guard::home_slot();
    let Some(fx) = Fx::new() else { return };
    let outside = fx.root.join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(fx.file(), "inside").unwrap();
    std::fs::write(outside.join("NOTES.md"), "expected").unwrap();
    let (ws, o) = (fx.ws.clone(), outside.clone());
    test_hooks::set_pause(move || swap_in_symlink(ws, o));
    let g = id(effects::mint_grant(&fx.b, &fx.mint_req()));
    assert!(
        fx.ws.join("d").is_symlink(),
        "the swap hook did not fire inside the call"
    );
    assert_eq!(
        text(&fx, g)["prior_sha256"],
        serde_json::json!(sha(b"inside")),
        "the grant recorded bytes that confinement did not check"
    );
}

/// open_intent: the grant was minted on "old\n". `d` is then a symlink to an
/// outside directory that holds the SAME bytes, while the real `d` (now
/// holding different bytes) is parked. The swap back to the real directory
/// fires at the start of the confined open. The read must see the real file,
/// find it changed, and refuse as Stale with no intent recorded. A by-path read
/// done before the swap would read the outside "old\n", match the grant and
/// record an intent.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
fn open_intent_reads_the_confined_file_not_the_path() {
    let _home = home_guard::home_slot();
    let Some(fx) = Fx::new() else { return };
    let outside = fx.root.join("outside");
    std::fs::create_dir_all(&outside).unwrap();
    std::fs::write(fx.file(), "old\n").unwrap();
    let g = id(effects::mint_grant(&fx.b, &fx.mint_req()));
    std::fs::write(outside.join("NOTES.md"), "old\n").unwrap();
    std::fs::write(fx.file(), "changed\n").unwrap();
    swap_in_symlink(fx.ws.clone(), outside);
    let ws = fx.ws.clone();
    test_hooks::set_before(move || {
        std::fs::remove_file(ws.join("d")).unwrap();
        std::fs::rename(ws.join("d.held"), ws.join("d")).unwrap();
    });
    let r = effects::open_intent(&fx.b, &fx.req(g, false));
    assert!(
        !fx.ws.join("d").is_symlink(),
        "the swap hook did not fire inside the call"
    );
    match &r {
        ControlResponse::Error(e) => assert!(
            e.starts_with("EFFECT_REFUSED Stale"),
            "wanted Stale, got {e}"
        ),
        other => panic!("wanted a Stale refusal, got {other:?}"),
    }
}
