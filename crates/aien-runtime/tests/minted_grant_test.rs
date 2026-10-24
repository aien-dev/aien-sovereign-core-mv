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

//! sovereign-core #261: the daemon honours only grants it minted itself.
//!
//! The ordinary operator flow (propose -> authorize -> execute) is driven at
//! the bridge, exactly as the daemon runs it (no socket: the mock backend of
//! the socket tests cannot commit a proposal). Every test needs the linked
//! composition archive and is IGNORED in a stub build, never passed. The
//! socket-level refusal of a raw `authorization` note is
//! `c18_a1_raw_forged_authorization` in approved_attacks_test.rs.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ComposeNoteReport, ComposeTaskReport, ControlResponse};
use aien_runtime::effects::{self, IntentRequest, MintRequest};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PROPOSAL: &str = "filename: NOTES.md\nkeep every change inside the workspace\n";
const CONTENT: &str = "keep every change inside the workspace\n";

fn proposer() -> ComposeProposer {
    Arc::new(|_prompt: &str, _limit: std::time::Duration| {
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

fn noted(r: ControlResponse) -> ComposeNoteReport {
    match r {
        ControlResponse::ComposeNoted(n) => n,
        other => panic!("expected ComposeNoted, got {other:?}"),
    }
}

/// A stop, resume or revoke answer.
fn controlled(r: ControlResponse) -> aien_runtime::control::ComposeControlReport {
    match r {
        ControlResponse::ComposeControlled(c) => *c,
        other => panic!("expected ComposeControlled, got {other:?}"),
    }
}

/// The refusal text of `r`; panics when the attack succeeded.
fn refusal(r: ControlResponse, attack: &str) -> String {
    match r {
        ControlResponse::Error(e) => e,
        other => panic!("ATTACK SUCCEEDED ({attack}): {other:?}"),
    }
}

/// The composition engine holds one home at a time per process.
static ONE_HOME: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Fx {
    _turn: std::sync::MutexGuard<'static, ()>,
    _tmp: tempfile::TempDir,
    home: PathBuf,
    ws: PathBuf,
    b: ComposeBridge,
    report: ComposeTaskReport,
}

impl Fx {
    /// A home with one committed ordinary proposal; None in a stub build.
    fn new() -> Option<Fx> {
        let turn = ONE_HOME.lock().unwrap_or_else(|e| e.into_inner());
        let tmp = tempfile::tempdir().unwrap();
        let home = tmp.path().join("compose");
        let ws = std::fs::canonicalize({
            let w = tmp.path().join("ws");
            std::fs::create_dir_all(&w).unwrap();
            w
        })
        .unwrap();
        let b = ComposeBridge::new(home.clone(), proposer(), "test:fixed-proposer")
            .with_authorize_requires_desk(false);
        let run = b.run_task("write the note", ws.to_str().unwrap());
        if !aien_omega_compose::LINKED {
            assert!(matches!(run, ControlResponse::Error(_)));
            return None;
        }
        let ControlResponse::ComposeTaskResult(report) = run else {
            panic!("run: {run:?}")
        };
        assert!(report.committed, "{report:?}");
        Some(Fx {
            _turn: turn,
            _tmp: tmp,
            home,
            ws,
            b,
            report: *report,
        })
    }

    fn target(&self) -> String {
        self.ws.join("NOTES.md").display().to_string()
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

    fn mint(&self) -> ControlResponse {
        effects::mint_grant(&self.b, &self.mint_req())
    }

    fn req(&self, a: u64, alive: bool) -> IntentRequest {
        let (pid, start) = if alive {
            effects::self_executor()
        } else {
            (0, 0)
        };
        IntentRequest {
            authorization: a,
            proposal_sha256: sha(PROPOSAL.as_bytes()),
            path: "NOTES.md".into(),
            target: self.target(),
            content_sha256: sha(CONTENT.as_bytes()),
            executor_pid: pid,
            executor_start: start,
        }
    }

    fn open(&self, a: u64) -> ControlResponse {
        effects::open_intent(&self.b, &self.req(a, true))
    }

    /// The text of record `id`.
    fn text(&self, id: u64) -> Value {
        match self.b.recall(&[id], None) {
            ControlResponse::ComposeRecalled(r) => {
                serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap()
            }
            other => panic!("recall: {other:?}"),
        }
    }

    fn ack_state(&self, intent: u64) -> String {
        let n = noted(effects::ack(&self.b, intent, &json!({"executor": "test"})));
        self.text(n.id)["state"].as_str().unwrap().to_string()
    }
}

macro_rules! linked_test {
    ($name:ident, |$fx:ident| $body:block) => {
        #[test]
        #[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
        fn $name() {
            let _home = home_guard::home_slot();
            let Some($fx) = Fx::new() else { return };
            $body
        }
    };
}

linked_test!(ordinary_flow_end_to_end_with_a_daemon_minted_grant, |fx| {
    // The run left the daemon's own commit record, and the report names it.
    let commit = fx.report.compose_commit.expect("commit record id");
    let c = fx.text(commit);
    assert_eq!(c["compose_commit"], 1);
    assert_eq!(c["proposal_sha256"], sha(PROPOSAL.as_bytes()));
    assert_eq!(c["workspace"], fx.ws.display().to_string());
    // Authorize, intent, write, ack: DONE.
    let g = noted(fx.mint());
    let gt = fx.text(g.id);
    assert_eq!(gt["minted_grant"], 1);
    assert_eq!(gt["compose_commit"], commit);
    assert_eq!(gt["target"], fx.target());
    assert!(g.links.contains(&commit), "{:?}", g.links);
    let i = noted(fx.open(g.id)).id;
    std::fs::write(fx.target(), CONTENT).unwrap();
    assert_eq!(fx.ack_state(i), "DONE");
    assert_eq!(std::fs::read_to_string(fx.target()).unwrap(), CONTENT);
});

linked_test!(a_replayed_grant_is_refused, |fx| {
    let g = noted(fx.mint()).id;
    let i = noted(fx.open(g)).id;
    // Second intent while the first is open, then after it settled.
    assert!(refusal(fx.open(g), "intent while open").contains("ReconciliationRequired"));
    std::fs::write(fx.target(), CONTENT).unwrap();
    assert_eq!(fx.ack_state(i), "DONE");
    assert!(refusal(fx.open(g), "replay after DONE").contains("AlreadySpent"));
});

linked_test!(a_grant_names_only_what_the_daemon_committed, |fx| {
    let g = noted(fx.mint()).id;
    for (what, f) in [
        (
            "other path",
            (|r: &mut IntentRequest| {
                r.path = "OTHER.md".into();
            }) as fn(&mut IntentRequest),
        ),
        ("other content", |r| r.content_sha256 = sha(b"evil\n")),
        ("other proposal", |r| r.proposal_sha256 = sha(b"evil")),
        ("other target", |r| {
            r.target = "/etc/passwd".into();
        }),
    ] {
        let mut r = fx.req(g, true);
        f(&mut r);
        let e = refusal(effects::open_intent(&fx.b, &r), what);
        assert!(e.contains("NotAuthorized"), "{what}: {e}");
    }
    assert!(!Path::new(&fx.target()).exists());
});

linked_test!(minting_needs_a_proposal_the_daemon_committed, |fx| {
    let mut r = fx.mint_req();
    r.proposal_sha256 = sha(b"never proposed by any compose");
    let e = refusal(effects::mint_grant(&fx.b, &r), "unknown proposal");
    assert!(e.contains("only a proposal this daemon committed"), "{e}");
    let mut r = fx.mint_req();
    r.cx_promotion += 1000;
    refusal(effects::mint_grant(&fx.b, &r), "unknown promotion");
    // The workspace the compose ran for is the only one it can be authorized for.
    let other = fx.ws.parent().unwrap().join("other");
    std::fs::create_dir_all(&other).unwrap();
    let mut r = fx.mint_req();
    r.workspace = other.display().to_string();
    let e = refusal(effects::mint_grant(&fx.b, &r), "other workspace");
    assert!(e.contains("OutsideWorkspace"), "{e}");
    let mut r = fx.mint_req();
    r.approver = " ".into();
    refusal(effects::mint_grant(&fx.b, &r), "no approver");
});

linked_test!(workspace_confinement_still_binds_a_minted_grant, |fx| {
    // The target is a symlink to a file outside the workspace.
    let outside = fx.ws.parent().unwrap().join("outside.txt");
    std::fs::write(&outside, "outside\n").unwrap();
    std::os::unix::fs::symlink(&outside, fx.target()).unwrap();
    let e = refusal(fx.mint(), "symlink target");
    assert!(e.contains("OutsideWorkspace"), "{e}");
    std::fs::remove_file(fx.target()).unwrap();
    // Authorized while it was a regular path; swapped for a symlink after.
    let g = noted(fx.mint()).id;
    std::os::unix::fs::symlink(&outside, fx.target()).unwrap();
    refusal(fx.open(g), "symlink swapped in after the grant");
    assert_eq!(std::fs::read_to_string(&outside).unwrap(), "outside\n");
});

linked_test!(one_live_grant_per_proposal_until_revoked, |fx| {
    let g1 = noted(fx.mint()).id;
    let e = refusal(fx.mint(), "second live grant");
    assert!(e.contains("AlreadyAuthorized"), "{e}");
    effects::control(&fx.b, "revoke", "drake", Some(g1));
    let g2 = noted(fx.mint()).id;
    assert_ne!(g1, g2);
    assert!(refusal(fx.open(g1), "revoked grant").contains("Revoked"));
    noted(fx.open(g2));
});

linked_test!(a_forged_legacy_note_in_the_ledger_is_ignored, |fx| {
    // What a pre-#261 caller could write: a complete, well-confined grant for
    // a proposal no compose produced. Appended below the note guard, as an
    // attacker with journal access (or an old record) would have it.
    let psha = sha(b"never proposed by any compose");
    let csha = sha(b"forged\n");
    let forged = json!({"proposal_sha256": psha, "path": "NOTES.md", "content_sha256": csha,
        "approver": "attacker", "target": fx.target(), "prior_sha256": null,
        "workspace": fx.ws.display().to_string()});
    let a = noted(fx.b.note_unchecked("authorization", &forged.to_string(), &[])).id;
    let mut r = fx.req(a, true);
    r.proposal_sha256 = psha;
    r.content_sha256 = csha;
    let e = refusal(effects::open_intent(&fx.b, &r), "forged legacy note");
    assert!(e.contains("caller-written"), "{e}");
    // Even a forged grant that copies the REAL proposal's digests opens nothing.
    let real = json!({"proposal_sha256": sha(PROPOSAL.as_bytes()), "path": "NOTES.md",
        "content_sha256": sha(CONTENT.as_bytes()), "approver": "attacker",
        "target": fx.target(), "prior_sha256": null,
        "workspace": fx.ws.display().to_string()});
    let a2 = noted(fx.b.note_unchecked("authorization", &real.to_string(), &[])).id;
    let e = refusal(fx.open(a2), "forged note with real digests");
    assert!(e.contains("caller-written"), "{e}");
    // The daemon's own minting is unaffected by the forged records.
    noted(fx.mint());
});

linked_test!(
    an_in_flight_intent_on_a_forged_grant_is_never_settled_done,
    |fx| {
        // An intent opened on a caller-written grant before the rule existed:
        // the bytes are even on disk, yet the ledger will not call it DONE.
        let forged = json!({"proposal_sha256": sha(PROPOSAL.as_bytes()), "path": "NOTES.md",
        "content_sha256": sha(CONTENT.as_bytes()), "approver": "attacker",
        "target": fx.target(), "prior_sha256": null,
        "workspace": fx.ws.display().to_string()});
        let a = noted(fx.b.note_unchecked("authorization", &forged.to_string(), &[])).id;
        let intent = json!({"phase": "intent", "tool": "write_file", "authorization": a,
        "proposal_sha256": sha(PROPOSAL.as_bytes()), "path": "NOTES.md",
        "target": fx.target(), "content_sha256": sha(CONTENT.as_bytes()),
        "prior_sha256": null, "executor": {"pid": 0, "start": 0}});
        let i = noted(fx.b.note_unchecked("effect", &intent.to_string(), &[a])).id;
        std::fs::write(fx.target(), CONTENT).unwrap();
        assert_eq!(fx.ack_state(i), "UNRESOLVED");
        let d = aien_runtime::control::ReconcileDeclare {
            intent: i,
            state: "done".into(),
            approver: "drake".into(),
        };
        let e = refusal(effects::reconcile(&fx.b, Some(&d), "test"), "declare DONE");
        assert!(e.contains("NotDaemonMinted"), "{e}");
    }
);

linked_test!(
    a_minted_grant_survives_a_restart_and_recovery_still_settles,
    |fx| {
        let g = noted(fx.mint()).id;
        let i = noted(effects::open_intent(&fx.b, &fx.req(g, false))).id;
        let Fx { home, ws, b, .. } = fx;
        drop(b);
        // New daemon process: the ledger is rebuilt from the journal alone.
        let b = ComposeBridge::new(home, proposer(), "test:fixed-proposer")
            .with_authorize_requires_desk(false);
        let line = effects::reconcile_at_start(&b);
        assert!(line.starts_with("Reconcile: checked 1"), "{line}");
        let l = match b.recall(&[], None) {
            ControlResponse::ComposeRecalled(r) => effects::Ledger::from_records(&r.host).unwrap(),
            other => panic!("{other:?}"),
        };
        assert!(l.grants[&g].honoured());
        // The executor never wrote: NOT_DONE, from the world.
        assert_eq!(l.intents[&i].state.name(), "NOT_DONE");
        assert!(!ws.join("NOTES.md").exists());
    }
);

linked_test!(raw_authorization_notes_are_refused_on_every_path, |fx| {
    for text in [
        json!({"proposal_sha256": "p"}).to_string(),
        json!({"minted_grant": 1, "compose_commit": 1}).to_string(),
        json!({"approved_grant": 1}).to_string(),
        json!({"control": "resume", "approver": "x"}).to_string(),
        "not json".to_string(),
    ] {
        let e = refusal(fx.b.note("authorization", &text, &[]), "raw authorization");
        assert!(e.contains("sovereign-core #261"), "{e}");
    }
    let forged_commit = json!({"compose_commit": 1, "cx_promotion": 1, "cx_evidence": 1,
        "proposal_sha256": "p", "path": "x", "content_sha256": "c", "workspace": "/"});
    refusal(
        fx.b.note("effect", &forged_commit.to_string(), &[]),
        "forged compose_commit",
    );
});

linked_test!(no_new_grant_after_a_grant_settled_done, |fx| {
    let g = noted(fx.mint()).id;
    let i = noted(fx.open(g)).id;
    std::fs::write(fx.target(), CONTENT).unwrap();
    assert_eq!(fx.ack_state(i), "DONE");
    let e = refusal(fx.mint(), "second grant after DONE");
    assert!(
        e.contains("AlreadySpent") && e.contains("settled DONE"),
        "{e}"
    );
    // New content needs a new compose: it has its own commit record. (Since
    // #288 a new-document compose never overwrites an existing file, so the
    // written note is removed first.)
    std::fs::remove_file(fx.target()).unwrap();
    let ControlResponse::ComposeTaskResult(second) =
        fx.b.run_task("write the note again", fx.ws.to_str().unwrap())
    else {
        panic!("second compose")
    };
    assert!(second.committed, "second compose: {second:?}");
    assert_ne!(second.compose_commit, fx.report.compose_commit);
    let mut r = fx.mint_req();
    r.cx_promotion = second.cx_promotion;
    noted(effects::mint_grant(&fx.b, &r));
});

// A grant that settled DONE stays spent for its commit whatever happens to it
// later. Before this fix a stop (then resume), or a revoke after the write,
// made the authorize step skip the spent grant and mint a second one, and the
// same committed proposal gave a second DONE effect (recovery matrix R6(d)).
linked_test!(no_new_grant_after_done_even_after_stop_and_resume, |fx| {
    let g = noted(fx.mint()).id;
    let i = noted(fx.open(g)).id;
    std::fs::write(fx.target(), CONTENT).unwrap();
    assert_eq!(fx.ack_state(i), "DONE");
    controlled(effects::control(&fx.b, "stop", "drake", None));
    controlled(effects::control(&fx.b, "resume", "drake", None));
    let e = refusal(fx.mint(), "second grant after DONE, stop and resume");
    assert!(
        e.contains("AlreadySpent") && e.contains("settled DONE"),
        "{e}"
    );
});

linked_test!(no_new_grant_after_done_even_after_revoke, |fx| {
    let g = noted(fx.mint()).id;
    let i = noted(fx.open(g)).id;
    std::fs::write(fx.target(), CONTENT).unwrap();
    assert_eq!(fx.ack_state(i), "DONE");
    controlled(effects::control(&fx.b, "revoke", "drake", Some(g)));
    let e = refusal(fx.mint(), "second grant after DONE and revoke");
    assert!(
        e.contains("AlreadySpent") && e.contains("settled DONE"),
        "{e}"
    );
});

// An unsettled intent blocks a new grant even when its own grant was later
// made stale by a stop or revoked: the first write may still land.
linked_test!(
    no_new_grant_while_a_stale_or_revoked_grants_intent_is_unsettled,
    |fx| {
        let g = noted(fx.mint()).id;
        noted(fx.open(g));
        controlled(effects::control(&fx.b, "stop", "drake", None));
        controlled(effects::control(&fx.b, "resume", "drake", None));
        let e = refusal(
            fx.mint(),
            "second grant over an open intent, stop and resume",
        );
        assert!(e.contains("ReconciliationRequired"), "{e}");
        controlled(effects::control(&fx.b, "revoke", "drake", Some(g)));
        let e = refusal(fx.mint(), "second grant over an open intent, revoked");
        assert!(e.contains("ReconciliationRequired"), "{e}");
    }
);

// The one path that must stay open (sc#321 review): a grant whose intent
// settled NOT_DONE (nothing was written) does not block a new grant.
linked_test!(
    a_grant_whose_intent_settled_not_done_can_be_authorized_again,
    |fx| {
        let g = noted(fx.mint()).id;
        let i = noted(effects::open_intent(&fx.b, &fx.req(g, false))).id;
        let line = effects::reconcile_at_start(&fx.b);
        assert!(line.starts_with("Reconcile: checked 1"), "{line}");
        let l = match fx.b.recall(&[], None) {
            ControlResponse::ComposeRecalled(r) => effects::Ledger::from_records(&r.host).unwrap(),
            other => panic!("{other:?}"),
        };
        assert_eq!(l.intents[&i].state.name(), "NOT_DONE");
        let g2 = noted(fx.mint()).id;
        assert_ne!(g2, g, "a new, distinct grant");
        assert!(!fx.ws.join("NOTES.md").exists());
    }
);

// sovereign-core #323: the NEXT-PHASE-1 effect note (`write_file`, an
// authorization id, no `phase`) spends its authorization in the ledger. Only
// old journals carry it; the daemon writes no such note since NEXT-PHASE-2.
fn legacy_spend(a: u64) -> String {
    json!({"tool": "write_file", "authorization": a, "success": true}).to_string()
}

// Before the fix a caller of ComposeNote could write one naming a live minted
// grant: that grant then refused to execute (AlreadySpent) and a revoke of it
// recorded nothing. A denial, never a second effect, but a forged record all
// the same.
linked_test!(a_caller_cannot_write_a_legacy_spend_note, |fx| {
    let g = noted(fx.mint()).id;
    let e = refusal(
        fx.b.note("effect", &legacy_spend(g), &[]),
        "legacy spend note",
    );
    assert!(e.contains("write_file"), "{e}");
    // Not even for an id that is no grant (yet).
    refusal(
        fx.b.note("effect", &legacy_spend(g + 1000), &[]),
        "legacy note, future id",
    );
    // The grant is untouched: it still executes once.
    let i = noted(fx.open(g)).id;
    std::fs::write(fx.target(), CONTENT).unwrap();
    assert_eq!(fx.ack_state(i), "DONE");
    // Other caller effect notes are still accepted.
    noted(fx.b.note("effect", r#"{"tool":"inspect"}"#, &[]));
});

// An old journal that already holds such a note is still read the same way.
linked_test!(an_old_legacy_spend_note_still_spends_its_grant, |fx| {
    let g = noted(fx.mint()).id;
    noted(fx.b.note_unchecked("effect", &legacy_spend(g), &[]));
    let e = refusal(fx.open(g), "grant spent by an old note");
    assert!(
        e.contains("AlreadySpent") && e.contains("NEXT-PHASE-1"),
        "{e}"
    );
    let c = controlled(effects::control(&fx.b, "revoke", "drake", Some(g)));
    assert_eq!(c.revoked, Some(false));
    assert!(!Path::new(&fx.target()).exists());
});

// Pin: a legacy note can never hide a DONE effect from the authorize step.
// Every effect has an intent, and the intent is judged first; a legacy mark,
// a stop, a resume or a revoke afterwards changes nothing.
linked_test!(
    a_legacy_note_never_lets_a_done_commit_be_authorized_again,
    |fx| {
        let g = noted(fx.mint()).id;
        let i = noted(fx.open(g)).id;
        std::fs::write(fx.target(), CONTENT).unwrap();
        assert_eq!(fx.ack_state(i), "DONE");
        noted(fx.b.note_unchecked("effect", &legacy_spend(g), &[]));
        let done = |what: &str| {
            let e = refusal(fx.mint(), what);
            assert!(
                e.contains("AlreadySpent") && e.contains("settled DONE"),
                "{what}: {e}"
            );
        };
        done("after a legacy mark");
        controlled(effects::control(&fx.b, "stop", "drake", None));
        controlled(effects::control(&fx.b, "resume", "drake", None));
        done("after a legacy mark, stop and resume");
        controlled(effects::control(&fx.b, "revoke", "drake", Some(g)));
        done("after a legacy mark and revoke");
        assert!(refusal(fx.open(g), "replay").contains("AlreadySpent"));
    }
);

// A grant marked spent by an old note that never executed, then stopped: one
// new grant may mint (nothing was written), and it gives the one effect.
linked_test!(
    a_legacy_marked_unexecuted_grant_still_allows_exactly_one_effect,
    |fx| {
        let g = noted(fx.mint()).id;
        noted(fx.b.note_unchecked("effect", &legacy_spend(g), &[]));
        controlled(effects::control(&fx.b, "stop", "drake", None));
        controlled(effects::control(&fx.b, "resume", "drake", None));
        let g2 = noted(fx.mint()).id;
        assert!(refusal(fx.open(g), "the legacy-marked grant").contains("AlreadySpent"));
        let i = noted(fx.open(g2)).id;
        std::fs::write(fx.target(), CONTENT).unwrap();
        assert_eq!(fx.ack_state(i), "DONE");
        let e = refusal(fx.mint(), "a third grant");
        assert!(e.contains("AlreadySpent"), "{e}");
    }
);
