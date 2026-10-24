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

//! sovereign-core #297: optional approval-desk MAC on `ComposeAuthorize`.
//!
//! Driven at the bridge like minted_grant_test.rs (the socket tests' mock
//! backend cannot commit a proposal). Each test builds its OWN compose home
//! (and so its own desk key) in a fresh temp dir; nothing is shared. Every
//! test needs the linked composition archive and is IGNORED in a stub build,
//! never passed.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::approved_auth::{desk_key_path, AuthorizeBinding, DeskKey};
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ComposeNoteReport, ComposeTaskReport, ControlResponse, DeskProof};
use aien_runtime::effects::{self, IntentRequest, MintRequest};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

const PROPOSALS: [&str; 2] = [
    "filename: NOTES.md\nkeep every change inside the workspace\n",
    "filename: OTHER.md\nsecond note for a different proposal\n",
];
const CONTENTS: [&str; 2] = [
    "keep every change inside the workspace\n",
    "second note for a different proposal\n",
];
const PATHS: [&str; 2] = ["NOTES.md", "OTHER.md"];

fn proposer() -> ComposeProposer {
    let n = Arc::new(AtomicUsize::new(0));
    Arc::new(move |_prompt: &str, _limit: std::time::Duration| {
        let i = n.fetch_add(1, Ordering::SeqCst).min(1);
        Ok(Generation {
            text: PROPOSALS[i].to_string(),
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

fn refusal(r: ControlResponse, attack: &str) -> String {
    match r {
        ControlResponse::Error(e) => e,
        other => panic!("ATTACK SUCCEEDED ({attack}): {other:?}"),
    }
}

static ONE_HOME: std::sync::Mutex<()> = std::sync::Mutex::new(());

struct Fx {
    _turn: std::sync::MutexGuard<'static, ()>,
    _tmp: tempfile::TempDir,
    home: PathBuf,
    ws: PathBuf,
    b: ComposeBridge,
    /// Two committed proposals (A = NOTES.md, B = OTHER.md).
    reports: [ComposeTaskReport; 2],
    key: DeskKey,
}

impl Fx {
    /// `on`: the desk switch. `with_key`: create the desk key file.
    fn new(on: bool, with_key: bool) -> Option<Fx> {
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
            .with_authorize_requires_desk(on);
        let mut reports = vec![];
        for _ in 0..2 {
            let run = b.run_task("write the note", ws.to_str().unwrap());
            if !aien_omega_compose::LINKED {
                assert!(matches!(run, ControlResponse::Error(_)));
                return None;
            }
            let ControlResponse::ComposeTaskResult(r) = run else {
                panic!("run: {run:?}")
            };
            assert!(r.committed, "{r:?}");
            reports.push(*r);
        }
        // The desk key lives in this fixture's own compose home.
        let kp = desk_key_path(&home);
        let key = if with_key {
            DeskKey::create(&kp).unwrap()
        } else {
            // A key that exists only in memory: signs, but the daemon has none.
            DeskKey::create(&tmp.path().join("elsewhere.key")).unwrap()
        };
        let [a, bb] = <[ComposeTaskReport; 2]>::try_from(reports).unwrap();
        Some(Fx {
            _turn: turn,
            _tmp: tmp,
            home,
            ws,
            b,
            reports: [a, bb],
            key,
        })
    }

    fn target(&self, i: usize) -> String {
        self.ws.join(PATHS[i]).display().to_string()
    }

    fn req(&self, i: usize) -> MintRequest {
        MintRequest {
            cx_promotion: self.reports[i].cx_promotion,
            proposal_sha256: sha(PROPOSALS[i].as_bytes()),
            workspace: self.ws.display().to_string(),
            approver: "drake".into(),
            constraints: vec![],
        }
    }

    fn binding(&self, i: usize, nonce: &str) -> AuthorizeBinding {
        let r = self.req(i);
        AuthorizeBinding {
            cx_promotion: r.cx_promotion,
            proposal_sha256: r.proposal_sha256,
            path: PATHS[i].into(),
            content_sha256: sha(CONTENTS[i].as_bytes()),
            workspace: self.ws.display().to_string(),
            approver: r.approver,
            constraints: vec![],
            nonce: nonce.into(),
            desk_key_id: String::new(),
        }
    }

    fn proof(&self, key: &DeskKey, b: &AuthorizeBinding) -> DeskProof {
        DeskProof {
            nonce: b.nonce.clone(),
            mac: key.sign_authorize(b),
        }
    }

    fn good(&self, i: usize, nonce: &str) -> DeskProof {
        self.proof(&self.key, &self.binding(i, nonce))
    }

    fn auth(&self, i: usize, p: Option<&DeskProof>) -> ControlResponse {
        effects::authorize(&self.b, &self.req(i), p)
    }

    /// Id the next record gets: a refusal must not have consumed one.
    fn next_id(&self) -> u64 {
        noted(self.b.note("constraint", "probe", &[])).id
    }

    fn text(&self, id: u64) -> Value {
        match self.b.recall(&[id], None) {
            ControlResponse::ComposeRecalled(r) => {
                serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap()
            }
            other => panic!("recall: {other:?}"),
        }
    }

    fn intent(&self, a: u64, i: usize) -> IntentRequest {
        let (pid, start) = effects::self_executor();
        IntentRequest {
            authorization: a,
            proposal_sha256: sha(PROPOSALS[i].as_bytes()),
            path: PATHS[i].into(),
            target: self.target(i),
            content_sha256: sha(CONTENTS[i].as_bytes()),
            executor_pid: pid,
            executor_start: start,
        }
    }
}

macro_rules! linked_test {
    ($name:ident, $on:expr, $key:expr, |$fx:ident| $body:block) => {
        #[test]
        #[cfg_attr(not(compose_linked), ignore = "needs the linked librx_compose.a")]
        fn $name() {
            let _home = home_guard::home_slot();
            let Some($fx) = Fx::new($on, $key) else {
                return;
            };
            $body
        }
    };
}

/// Asserts the refusal names `name` and that no record was written by it.
fn refused_clean(fx: &Fx, r: ControlResponse, name: &str, what: &str) {
    let before = fx.next_id();
    let e = refusal(r, what);
    assert!(e.contains(name), "{what}: {e}");
    assert_eq!(fx.next_id(), before + 1, "{what}: a refusal wrote a record");
}

linked_test!(
    switch_on_no_mac_is_refused_and_writes_nothing,
    true,
    true,
    |fx| {
        refused_clean(&fx, fx.auth(0, None), "DeskMacRequired", "no MAC");
        // The legacy entry point is the same gate.
        refused_clean(
            &fx,
            effects::mint_grant(&fx.b, &fx.req(0)),
            "DeskMacRequired",
            "mint_grant without MAC",
        );
    }
);

linked_test!(switch_on_wrong_key_is_refused, true, true, |fx| {
    let other = DeskKey::create(&fx.home.parent().unwrap().join("other.key")).unwrap();
    let p = fx.proof(&other, &fx.binding(0, "n1"));
    refused_clean(&fx, fx.auth(0, Some(&p)), "DeskMacInvalid", "wrong key");
    let junk = DeskProof {
        nonce: "n1".into(),
        mac: "zz".into(),
    };
    refused_clean(&fx, fx.auth(0, Some(&junk)), "DeskMacInvalid", "junk MAC");
});

linked_test!(
    switch_on_mac_for_proposal_a_fails_on_proposal_b,
    true,
    true,
    |fx| {
        let for_a = fx.good(0, "n-a");
        refused_clean(
            &fx,
            fx.auth(1, Some(&for_a)),
            "DeskMacInvalid",
            "A's MAC on B",
        );
    }
);

linked_test!(switch_on_every_bound_field_is_checked, true, true, |fx| {
    type Mutate = fn(&mut AuthorizeBinding);
    let cases: [(&str, Mutate); 7] = [
        ("proposal", |b| b.proposal_sha256 = sha(b"other")),
        ("path", |b| b.path = "ELSE.md".into()),
        ("content", |b| b.content_sha256 = sha(b"evil\n")),
        ("workspace", |b| b.workspace = "/tmp".into()),
        ("approver", |b| b.approver = "mallory".into()),
        ("constraints", |b| b.constraints = vec![1]),
        ("promotion", |b| b.cx_promotion += 1),
    ];
    for (what, mutate) in cases {
        let mut b = fx.binding(0, "n-f");
        mutate(&mut b);
        let p = fx.proof(&fx.key, &b);
        refused_clean(&fx, fx.auth(0, Some(&p)), "DeskMacInvalid", what);
    }
    // A MAC made over one nonce does not carry another.
    let mut p = fx.good(0, "n-1");
    p.nonce = "n-2".into();
    refused_clean(&fx, fx.auth(0, Some(&p)), "DeskMacInvalid", "swapped nonce");
});

linked_test!(
    switch_on_correct_mac_mints_and_the_effect_runs,
    true,
    true,
    |fx| {
        let p = fx.good(0, "n-ok");
        let g = noted(fx.auth(0, Some(&p)));
        let t = fx.text(g.id);
        assert_eq!(t["minted_grant"], 1);
        assert_eq!(t["desk_nonce"], "n-ok");
        assert_eq!(t["desk_key_id"], fx.key.id());
        let i = noted(effects::open_intent(&fx.b, &fx.intent(g.id, 0))).id;
        std::fs::write(fx.target(0), CONTENTS[0]).unwrap();
        let a = noted(effects::ack(&fx.b, i, &json!({"executor": "test"})));
        assert_eq!(fx.text(a.id)["state"], "DONE");
        assert_eq!(std::fs::read_to_string(fx.target(0)).unwrap(), CONTENTS[0]);
    }
);

linked_test!(switch_on_replay_mints_no_second_grant, true, true, |fx| {
    let p = fx.good(0, "n-once");
    let g1 = noted(fx.auth(0, Some(&p))).id;
    // Same request again while the grant is live.
    refused_clean(&fx, fx.auth(0, Some(&p)), "Replayed", "replay while live");
    // After a revoke the proposal could be authorized again, but not by the
    // same MAC: the nonce is spent.
    effects::control(&fx.b, "revoke", "drake", Some(g1));
    refused_clean(&fx, fx.auth(0, Some(&p)), "Replayed", "replay after revoke");
    // A fresh authorize (new nonce, new MAC) does mint.
    let g2 = noted(fx.auth(0, Some(&fx.good(0, "n-twice")))).id;
    assert_ne!(g1, g2);
});

linked_test!(
    switch_on_missing_key_file_refuses_no_silent_fallback,
    true,
    false,
    |fx| {
        assert!(!desk_key_path(&fx.home).exists());
        // Even a MAC made with some other key, and no MAC at all: both refused.
        let p = fx.good(0, "n-nokey");
        refused_clean(&fx, fx.auth(0, Some(&p)), "NoDesk", "key file missing");
        refused_clean(&fx, fx.auth(0, None), "DeskMacRequired", "no MAC, no key");
    }
);

linked_test!(
    switch_on_rotated_key_refuses_the_old_mac,
    true,
    true,
    |fx| {
        let old = fx.good(0, "n-old");
        std::fs::remove_file(desk_key_path(&fx.home)).unwrap();
        let new = DeskKey::create(&desk_key_path(&fx.home)).unwrap();
        assert_ne!(new.id(), fx.key.id());
        refused_clean(
            &fx,
            fx.auth(0, Some(&old)),
            "DeskMacInvalid",
            "old MAC after rotation",
        );
        let p = fx.proof(&new, &fx.binding(0, "n-new"));
        noted(fx.auth(0, Some(&p)));
    }
);

linked_test!(switch_on_rejects_a_bad_nonce, true, true, |fx| {
    for n in ["", "has space", &"x".repeat(129)] {
        let p = fx.good(0, n);
        refused_clean(&fx, fx.auth(0, Some(&p)), "DeskMacInvalid", "bad nonce");
    }
});

// Legacy behaviour, NOT a security pass: with the switch off the daemon
// authenticates the operator by OS user only, exactly as before #297. A proof,
// good or bad, is ignored and not recorded.
linked_test!(switch_off_legacy_authorize_still_works, false, true, |fx| {
    let g = noted(fx.auth(0, None));
    assert!(fx.text(g.id).get("desk_nonce").is_none());
    let bad = DeskProof {
        nonce: "n".into(),
        mac: "00".into(),
    };
    let g2 = noted(fx.auth(1, Some(&bad)));
    assert!(fx.text(g2.id).get("desk_nonce").is_none());
    let i = noted(effects::open_intent(&fx.b, &fx.intent(g.id, 0))).id;
    std::fs::write(fx.target(0), CONTENTS[0]).unwrap();
    let a = noted(effects::ack(&fx.b, i, &json!({"executor": "test"})));
    assert_eq!(fx.text(a.id)["state"], "DONE");
});

#[test]
fn the_wire_field_is_optional_and_old_clients_parse() {
    use aien_runtime::control::ControlCommand;
    // An old client's line has no desk_proof: it parses, with None.
    let old = r#"{"ComposeAuthorize":{"cx_promotion":3,"proposal_sha256":"p","workspace":"/w","approver":"a","constraints":[]}}"#;
    match serde_json::from_str::<ControlCommand>(old).unwrap() {
        ControlCommand::ComposeAuthorize { desk_proof, .. } => assert!(desk_proof.is_none()),
        other => panic!("{other:?}"),
    }
    // None is not written, so an old daemon sees the old shape.
    let none = ControlCommand::ComposeAuthorize {
        cx_promotion: 3,
        proposal_sha256: "p".into(),
        workspace: "/w".into(),
        approver: "a".into(),
        constraints: vec![],
        desk_proof: None,
        objective_id: None,
    };
    assert_eq!(serde_json::to_string(&none).unwrap(), old);
    // A new line with the proof round-trips.
    let with = ControlCommand::ComposeAuthorize {
        cx_promotion: 3,
        proposal_sha256: "p".into(),
        workspace: "/w".into(),
        approver: "a".into(),
        constraints: vec![],
        desk_proof: Some(DeskProof {
            nonce: "n".into(),
            mac: "m".into(),
        }),
        objective_id: None,
    };
    let s = serde_json::to_string(&with).unwrap();
    assert!(s.contains("desk_proof"));
    assert!(matches!(
        serde_json::from_str::<ControlCommand>(&s).unwrap(),
        ControlCommand::ComposeAuthorize {
            desk_proof: Some(_),
            ..
        }
    ));
}

#[test]
fn the_desk_is_required_unless_a_dev_run_opts_out() {
    // sc#328 (Drake decision b, 2026-10-08): REQUIRED by default. "0" is the
    // dev-only opt-out; a strict (release qualification) run refuses it.
    use aien_runtime::compose::authorize_desk_setting;
    for dev in [false, true] {
        assert_eq!(
            authorize_desk_setting(None, dev),
            Ok(true),
            "unset, dev={dev}"
        );
        assert_eq!(
            authorize_desk_setting(Some("1"), dev),
            Ok(true),
            "1, dev={dev}"
        );
        assert!(
            authorize_desk_setting(Some("yes"), dev).is_err(),
            "yes, dev={dev}"
        );
        assert!(
            authorize_desk_setting(Some(""), dev).is_err(),
            "empty, dev={dev}"
        );
    }
    assert_eq!(authorize_desk_setting(Some("0"), true), Ok(false));
    let e = authorize_desk_setting(Some("0"), false).unwrap_err();
    assert!(
        e.contains("AIEN_DEV_FALLBACK") && e.contains("release qualification"),
        "{e}"
    );
}

#[test]
fn the_env_switch_is_strict() {
    // The daemon's reader: same rules, this process's dev mode. Run in one
    // test so the process-wide env is not raced by another test.
    use aien_inference_abi::strict::dev_fallback_active;
    use aien_runtime::compose::{authorize_requires_desk_from_env, AUTHORIZE_DESK_ENV};
    let _g = ONE_HOME.lock().unwrap_or_else(|e| e.into_inner());
    std::env::remove_var(AUTHORIZE_DESK_ENV);
    assert_eq!(authorize_requires_desk_from_env(), Ok(true));
    std::env::set_var(AUTHORIZE_DESK_ENV, "1");
    assert_eq!(authorize_requires_desk_from_env(), Ok(true));
    std::env::set_var(AUTHORIZE_DESK_ENV, "0");
    if dev_fallback_active() {
        assert_eq!(authorize_requires_desk_from_env(), Ok(false));
    } else {
        assert!(authorize_requires_desk_from_env().is_err());
    }
    std::env::set_var(AUTHORIZE_DESK_ENV, "yes");
    assert!(authorize_requires_desk_from_env().is_err());
    std::env::remove_var(AUTHORIZE_DESK_ENV);
}

#[test]
fn a_new_bridge_requires_the_desk() {
    // Secure by default for every library caller, not only the daemon.
    let tmp = tempfile::tempdir().unwrap();
    let b = ComposeBridge::new(
        tmp.path().join("compose"),
        proposer(),
        "test:fixed-proposer",
    );
    assert!(b.authorize_requires_desk());
    assert!(!b
        .with_authorize_requires_desk(false)
        .authorize_requires_desk());
}
