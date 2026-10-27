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

//! sovereign-core #408 (RECOVERY-AUDIT-2026-10-11 row 4): the objective id is
//! carried by the grant, the intent, the ack and the reconcile records, and
//! the objective record is re-read after a restart.
//!
//! The real compose home through the bridge, as the daemon drives it (no
//! socket). A restart is a new bridge on the same home. The objective record
//! is written here exactly as `aien-cli objective record` writes it (sc#388).
//! In a stub build only the refusal of the compose run is checked.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{
    ComposeNoteReport, ComposeReconcileReport, ControlResponse, ReconcileOutcome,
};
use aien_runtime::effects::{self, IntentRequest, Ledger, MintRequest};
use aien_runtime::objective;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

const PROPOSAL: &str = "filename: NOTES.md\nkeep every change inside the workspace\n";
const CONTENT: &str = "keep every change inside the workspace\n";
const GOAL: &str = "write the note";

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

fn refused(r: ControlResponse, name: &str) {
    match r {
        ControlResponse::Error(e) => assert!(
            e.starts_with(&format!("EFFECT_REFUSED {name}:")),
            "want {name}, got {e}"
        ),
        other => panic!("want EFFECT_REFUSED {name}, got {other:?}"),
    }
}

fn reconciled(r: ControlResponse) -> ComposeReconcileReport {
    match r {
        ControlResponse::ComposeReconciled(r) => *r,
        other => panic!("expected ComposeReconciled, got {other:?}"),
    }
}

fn ledger(b: &ComposeBridge) -> Ledger {
    match b.recall(&[], None) {
        ControlResponse::ComposeRecalled(r) => Ledger::from_records(&r.host).unwrap(),
        other => panic!("recall: {other:?}"),
    }
}

fn text_of(b: &ComposeBridge, record: u64) -> Value {
    match b.recall(&[record], None) {
        ControlResponse::ComposeRecalled(r) => {
            serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap()
        }
        other => panic!("recall: {other:?}"),
    }
}

/// The record `aien-cli objective record --text` writes (sc#388).
fn write_record(prov: &Path, text: &str) -> (String, PathBuf) {
    let id = objective::objective_id(text);
    let d = prov.join("objectives");
    std::fs::create_dir_all(&d).unwrap();
    let p = d.join(format!("{id}.json"));
    let rec = json!({"version": 1, "objective_id": id, "objective_text": text,
        "objective_text_sha256": sha(text.as_bytes()), "effect_class": "WORKSPACE_WRITE",
        "recorded_before_work": true, "operator_surface": "aien-cli",
        "utc": "2026-10-11T00:00:00Z"});
    std::fs::write(&p, serde_json::to_vec_pretty(&rec).unwrap()).unwrap();
    (id, p)
}

/// Rewrite the record in place with other text and a consistent digest: what
/// an edit of the file by hand would leave.
fn alter_record(p: &Path) {
    let mut v: Value = serde_json::from_slice(&std::fs::read(p).unwrap()).unwrap();
    let t = "write the note, then delete the inbox";
    v["objective_text"] = json!(t);
    v["objective_text_sha256"] = json!(sha(t.as_bytes()));
    std::fs::write(p, serde_json::to_vec_pretty(&v).unwrap()).unwrap();
}

struct Fx {
    ws: PathBuf,
    target: String,
    promotion: u64,
}

impl Fx {
    fn compose(b: &ComposeBridge, ws: &Path) -> u64 {
        match b.run_task(GOAL, ws.to_str().unwrap()) {
            ControlResponse::ComposeTaskResult(r) => {
                assert!(r.committed, "{r:?}");
                r.cx_promotion
            }
            other => panic!("compose: {other:?}"),
        }
    }
    fn mint(&self, b: &ComposeBridge, objective_id: Option<&str>) -> ControlResponse {
        effects::authorize_with_objective(
            b,
            &MintRequest {
                cx_promotion: self.promotion,
                proposal_sha256: sha(PROPOSAL.as_bytes()),
                workspace: self.ws.display().to_string(),
                approver: "drake".into(),
                constraints: vec![],
            },
            None,
            objective_id,
        )
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
            target: self.target.clone(),
            content_sha256: sha(CONTENT.as_bytes()),
            executor_pid: pid,
            executor_start: start,
        }
    }
}

fn bridge(home: &Path) -> ComposeBridge {
    ComposeBridge::new(home.to_path_buf(), proposer(), "test:fixed-proposer")
        .with_authorize_requires_desk(false)
}

fn outcome_for(line: &str, intent: u64) -> ReconcileOutcome {
    let json = &line[line.find("; [").expect("outcomes in the line") + 2..];
    let all: Vec<ReconcileOutcome> = serde_json::from_str(json).unwrap();
    all.into_iter().find(|o| o.intent == intent).unwrap()
}

#[test]
fn objective_identity_survives_a_restart_and_is_re_read() {
    let _home = home_guard::home_slot();
    let tmp = tempfile::tempdir().unwrap();
    let prov = tmp.path().join("prov");
    // The daemon and the CLI resolve the record directory the same way.
    std::env::set_var("AIEN_PROVENANCE_DIR", &prov);
    let home = tmp.path().join("compose");
    let ws = std::fs::canonicalize({
        let w = tmp.path().join("ws");
        std::fs::create_dir_all(&w).unwrap();
        w
    })
    .unwrap();
    let b = bridge(&home);
    if !aien_omega_compose::LINKED {
        assert!(matches!(
            b.run_task(GOAL, ws.to_str().unwrap()),
            ControlResponse::Error(_)
        ));
        return;
    }
    let mut f = Fx {
        target: ws.join("NOTES.md").display().to_string(),
        promotion: Fx::compose(&b, &ws),
        ws: ws.clone(),
    };
    // The daemon's own commit record names the goal it composed for.
    let l = ledger(&b);
    let commit = l.commits.values().last().unwrap();
    assert_eq!(
        commit.goal_sha256.as_deref(),
        Some(sha(GOAL.as_bytes()).as_str())
    );

    // 1. Refusals by name; nothing is minted.
    let missing = objective::objective_id("never recorded");
    refused(f.mint(&b, Some(&missing)), "ObjectiveMissing");
    let (other, _) = write_record(&prov, "some other objective");
    refused(f.mint(&b, Some(&other)), "ObjectiveMismatch");
    refused(f.mint(&b, Some("not-an-id")), "ObjectiveMissing");
    assert!(
        ledger(&b).grants.is_empty(),
        "a refused authorize mints nothing"
    );

    // 2. Bound grant, intent from an executor that dies: the id rides along.
    let (oid, rec_path) = write_record(&prov, GOAL);
    let rec_sha = sha(&std::fs::read(&rec_path).unwrap());
    let g = noted(f.mint(&b, Some(&oid))).id;
    let gt = text_of(&b, g);
    assert_eq!(gt["objective_id"], json!(oid));
    assert_eq!(gt["objective_text_sha256"], json!(sha(GOAL.as_bytes())));
    assert_eq!(gt["objective_record_sha256"], json!(rec_sha));
    let i = noted(effects::open_intent(&b, &f.req(g, false))).id;
    assert_eq!(text_of(&b, i)["objective_id"], json!(oid));
    assert_eq!(ledger(&b).view()["intents"][0]["objective_id"], json!(oid));

    // 3. Restart with the record intact: reconcile binds the intent to it.
    drop(b);
    let b = bridge(&home);
    let line = effects::reconcile_at_start(&b);
    let o = outcome_for(&line, i);
    assert_eq!(o.state, "NOT_DONE", "{line}");
    let ob = o.objective.expect("binding reported");
    assert_eq!(
        (ob.objective_id.as_str(), ob.record.as_str()),
        (oid.as_str(), "bound")
    );
    assert_eq!(ob.record_unchanged, Some(true));
    assert_eq!(text_of(&b, o.record.unwrap())["objective_id"], json!(oid));
    refused(effects::open_intent(&b, &f.req(g, true)), "AlreadySpent");

    // 4. Record removed: a new approval is refused by name; restored: minted.
    std::fs::rename(&rec_path, tmp.path().join("moved.json")).unwrap();
    refused(f.mint(&b, Some(&oid)), "ObjectiveMissing");
    std::fs::rename(tmp.path().join("moved.json"), &rec_path).unwrap();
    let g2 = noted(f.mint(&b, Some(&oid))).id;

    // 5. Record altered after the grant: the intent is refused, nothing written.
    let saved = std::fs::read(&rec_path).unwrap();
    alter_record(&rec_path);
    refused(
        effects::open_intent(&b, &f.req(g2, true)),
        "ObjectiveMismatch",
    );
    assert!(
        !ledger(&b).spent.contains_key(&g2),
        "no intent on a refusal"
    );
    std::fs::write(&rec_path, &saved).unwrap();

    // 6. Restored: the effect runs once and the ack names the objective.
    let i2 = noted(effects::open_intent(&b, &f.req(g2, true))).id;
    std::fs::write(&f.target, CONTENT).unwrap();
    let k = noted(effects::ack(&b, i2, &json!({})));
    let kt = text_of(&b, k.id);
    assert_eq!(
        (kt["state"].clone(), kt["objective_id"].clone()),
        (json!("DONE"), json!(oid))
    );

    // 7. Restart with an open bound intent whose record is gone: the intent is
    // still settled by the world, and the missing record is named, not silent.
    std::fs::remove_file(&f.target).unwrap();
    f.promotion = Fx::compose(&b, &ws);
    let g3 = noted(f.mint(&b, Some(&oid))).id;
    let i3 = noted(effects::open_intent(&b, &f.req(g3, false))).id;
    drop(b);
    std::fs::remove_file(&rec_path).unwrap();
    let b = bridge(&home);
    let line = effects::reconcile_at_start(&b);
    let o = outcome_for(&line, i3);
    assert_eq!(o.state, "NOT_DONE", "{line}");
    let ob = o.objective.expect("binding reported");
    assert_eq!(ob.record, "ObjectiveMissing", "{line}");
    assert!(ob.detail.is_some());
    refused(f.mint(&b, Some(&oid)), "ObjectiveMissing");

    // 8. Legacy path: a grant without an objective is minted and reconciled
    // exactly as before; nothing about objectives appears.
    f.promotion = Fx::compose(&b, &ws);
    let g4 = noted(f.mint(&b, None)).id;
    assert!(text_of(&b, g4).get("objective_id").is_none());
    let i4 = noted(effects::open_intent(&b, &f.req(g4, false))).id;
    assert!(text_of(&b, i4).get("objective_id").is_none());
    let r = reconciled(effects::reconcile(&b, None, "test"));
    let o = r.outcomes.iter().find(|o| o.intent == i4).unwrap();
    assert!(o.objective.is_none());
    assert!(!serde_json::to_string(o).unwrap().contains("objective"));
    std::env::remove_var("AIEN_PROVENANCE_DIR");
}
