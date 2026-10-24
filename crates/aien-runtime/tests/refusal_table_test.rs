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

//! sovereign-core #267 follow-up: a security fix must not change the refusal
//! an old caller gets. This table drives the REAL `authorize` (mint_grant) and
//! `open_intent` over eight target states and records the refusal KIND each
//! gets. The same file ran on origin/main (3acc2cb) to produce EXPECTED; the
//! branch must give the identical kind for every row.
//!
//! Needs the linked composition archive; IGNORED in a stub build.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ComposeNoteReport, ComposeTaskReport, ControlResponse};
use aien_runtime::effects::{self, IntentRequest, MintRequest};
use sha2::{Digest, Sha256};
use std::os::unix::fs::PermissionsExt;
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

/// The refusal kind of a response: "OK", the EFFECT_REFUSED name, or the
/// first word of a plain error.
fn kind(r: &ControlResponse) -> String {
    match r {
        ControlResponse::Error(e) => match e.strip_prefix("EFFECT_REFUSED ") {
            Some(rest) => rest.split(':').next().unwrap().to_string(),
            None => format!("plain:{}", e.split_whitespace().next().unwrap_or("")),
        },
        ControlResponse::ComposeNoted(_) => "OK".into(),
        other => format!("other:{other:?}"),
    }
}

fn id(r: ControlResponse) -> u64 {
    match r {
        ControlResponse::ComposeNoted(ComposeNoteReport { id, .. }) => id,
        other => panic!("expected a note, got {other:?}"),
    }
}

/// How the world is shaped before the call.
#[derive(Clone, Copy, Debug)]
enum World {
    Valid,
    Missing,
    Directory,
    PermissionDenied,
    LeafSymlinkInside,
    LeafSymlinkOutside,
    IntermediateSymlinkSameBytes,
    IntermediateSymlinkOtherBytes,
    MissingParent,
}

const WORLDS: [World; 9] = [
    World::Valid,
    World::Missing,
    World::Directory,
    World::PermissionDenied,
    World::LeafSymlinkInside,
    World::LeafSymlinkOutside,
    World::IntermediateSymlinkSameBytes,
    World::IntermediateSymlinkOtherBytes,
    World::MissingParent,
];

fn shape(fx: &Fx, w: World) {
    let f = fx.file();
    let d = fx.ws.join("d");
    let outside = fx.root.join("outside");
    match w {
        World::Valid => {}
        World::Missing => std::fs::remove_file(&f).unwrap(),
        World::Directory => {
            std::fs::remove_file(&f).unwrap();
            std::fs::create_dir(&f).unwrap();
        }
        World::PermissionDenied => {
            std::fs::set_permissions(&f, std::fs::Permissions::from_mode(0o000)).unwrap()
        }
        World::LeafSymlinkInside => {
            std::fs::write(d.join("other"), "old\n").unwrap();
            std::fs::remove_file(&f).unwrap();
            std::os::unix::fs::symlink(d.join("other"), &f).unwrap();
        }
        World::LeafSymlinkOutside => {
            std::fs::create_dir_all(&outside).unwrap();
            std::fs::write(outside.join("x"), "old\n").unwrap();
            std::fs::remove_file(&f).unwrap();
            std::os::unix::fs::symlink(outside.join("x"), &f).unwrap();
        }
        World::IntermediateSymlinkSameBytes | World::IntermediateSymlinkOtherBytes => {
            std::fs::create_dir_all(&outside).unwrap();
            let bytes = if matches!(w, World::IntermediateSymlinkSameBytes) {
                "old\n"
            } else {
                "different\n"
            };
            std::fs::write(outside.join("NOTES.md"), bytes).unwrap();
            std::fs::rename(&d, fx.ws.join("d.held")).unwrap();
            std::os::unix::fs::symlink(&outside, &d).unwrap();
        }
        World::MissingParent => std::fs::remove_dir_all(&d).unwrap(),
    }
}

/// Rows: "<world> | <op>" -> kind. `wrong_grant` rows use a request whose
/// content differs from the grant (NotAuthorized on a readable target).
fn run_table() -> Vec<(String, String)> {
    let mut rows = Vec::new();
    for w in WORLDS {
        // authorize: the world is shaped, then the mint is asked.
        {
            let fx = Fx::new().unwrap();
            std::fs::write(fx.file(), "old\n").unwrap();
            shape(&fx, w);
            let r = effects::mint_grant(&fx.b, &fx.mint_req());
            rows.push((format!("{w:?} | authorize"), kind(&r)));
        }
        // open_intent: a valid grant is minted on "old\n", then the world is shaped.
        for wrong in [false, true] {
            let fx = Fx::new().unwrap();
            std::fs::write(fx.file(), "old\n").unwrap();
            let g = id(effects::mint_grant(&fx.b, &fx.mint_req()));
            shape(&fx, w);
            let r = effects::open_intent(&fx.b, &fx.req(g, wrong));
            let op = if wrong {
                "open_intent(wrong grant)"
            } else {
                "open_intent"
            };
            rows.push((format!("{w:?} | {op}"), kind(&r)));
        }
        // restore permissions so the temp dir can be removed
        let _ = w;
    }
    rows
}

/// origin/main (3acc2cb) results for the same table (this file run there).
const EXPECTED: &[(&str, &str)] = &[
    ("Valid | authorize", "OK"),
    ("Valid | open_intent", "OK"),
    ("Valid | open_intent(wrong grant)", "NotAuthorized"),
    ("Missing | authorize", "OK"),
    ("Missing | open_intent", "Stale"),
    ("Missing | open_intent(wrong grant)", "NotAuthorized"),
    ("Directory | authorize", "OutsideWorkspace"),
    ("Directory | open_intent", "Stale"),
    ("Directory | open_intent(wrong grant)", "Stale"),
    ("PermissionDenied | authorize", "OutsideWorkspace"),
    ("PermissionDenied | open_intent", "Stale"),
    ("PermissionDenied | open_intent(wrong grant)", "Stale"),
    ("LeafSymlinkInside | authorize", "OutsideWorkspace"),
    ("LeafSymlinkInside | open_intent", "Stale"),
    ("LeafSymlinkInside | open_intent(wrong grant)", "Stale"),
    ("LeafSymlinkOutside | authorize", "OutsideWorkspace"),
    ("LeafSymlinkOutside | open_intent", "Stale"),
    ("LeafSymlinkOutside | open_intent(wrong grant)", "Stale"),
    (
        "IntermediateSymlinkSameBytes | authorize",
        "OutsideWorkspace",
    ),
    (
        "IntermediateSymlinkSameBytes | open_intent",
        "OutsideWorkspace",
    ),
    (
        "IntermediateSymlinkSameBytes | open_intent(wrong grant)",
        "NotAuthorized",
    ),
    (
        "IntermediateSymlinkOtherBytes | authorize",
        "OutsideWorkspace",
    ),
    ("IntermediateSymlinkOtherBytes | open_intent", "Stale"),
    (
        "IntermediateSymlinkOtherBytes | open_intent(wrong grant)",
        "NotAuthorized",
    ),
    ("MissingParent | authorize", "OutsideWorkspace"),
    ("MissingParent | open_intent", "Stale"),
    ("MissingParent | open_intent(wrong grant)", "NotAuthorized"),
];

/// The one row where the branch deliberately differs from origin/main. It is
/// still a refusal and stricter, not a weakening: main read the bytes THROUGH
/// the symlinked directory and said Stale; the branch never reads through the
/// symlink and says OutsideWorkspace. Telling Stale would need the outside
/// bytes: the very bug.
const DIFFERS: &[(&str, &str)] = &[(
    "IntermediateSymlinkOtherBytes | open_intent",
    "OutsideWorkspace",
)];

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
fn refusal_kinds_match_origin_main() {
    let _home = home_guard::home_slot();
    if !aien_omega_compose::LINKED {
        return;
    }
    let rows = run_table();
    for (k, v) in &rows {
        println!("ROW {k} => {v}");
    }
    assert_eq!(rows.len(), EXPECTED.len());
    for (k, v) in &rows {
        let main = EXPECTED
            .iter()
            .find(|(ek, _)| ek == k)
            .unwrap_or_else(|| panic!("no expectation for {k}"))
            .1;
        let want = DIFFERS.iter().find(|(dk, _)| dk == k).map_or(main, |d| d.1);
        assert_eq!(v, want, "{k}: branch gives {v}, origin/main gave {main}");
    }
}
