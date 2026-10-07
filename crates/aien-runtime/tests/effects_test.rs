//! NEXT-PHASE-2: durable effect intents, ack, reconcile, stop/resume and
//! revoke through the real compose home (ACCEPTANCE-v2 section 2).
//!
//! With librx_compose.a linked this drives the bridge exactly as the daemon
//! does (no socket); in a stub build it only checks the refusal.
use aien_omega_compose::hex;
use aien_runtime::control::{
    ComposeControlReport, ComposeNoteReport, ComposeReconcileReport, ControlResponse,
    ReconcileDeclare,
};
use aien_runtime::effects::{self, IntentRequest, Ledger};
use aien_runtime::spine::{ComposeBridge, ComposeProposer, Generation};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
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

fn controlled(r: ControlResponse) -> ComposeControlReport {
    match r {
        ControlResponse::ComposeControlled(r) => *r,
        other => panic!("expected ComposeControlled, got {other:?}"),
    }
}

fn ledger(b: &ComposeBridge) -> Ledger {
    match b.recall(&[], None) {
        ControlResponse::ComposeRecalled(r) => Ledger::from_records(&r.host).unwrap(),
        other => panic!("recall: {other:?}"),
    }
}

fn state(b: &ComposeBridge, record: u64) -> String {
    match b.recall(&[record], None) {
        ControlResponse::ComposeRecalled(r) => {
            let t = r.cited[0].text.clone().unwrap();
            serde_json::from_str::<Value>(&t).unwrap()["state"]
                .as_str()
                .unwrap()
                .to_string()
        }
        other => panic!("recall: {other:?}"),
    }
}

struct Fixture {
    target: String,
    psha: String,
    csha: String,
}

impl Fixture {
    fn grant(&self, b: &ComposeBridge) -> u64 {
        let prior = effects::file_sha256(Path::new(&self.target)).unwrap();
        let text = json!({"proposal_sha256": self.psha, "path": "NOTES.md",
            "content_sha256": self.csha, "approver": "drake", "target": self.target,
            "prior_sha256": prior,
            "workspace": Path::new(&self.target).parent().unwrap().display().to_string()});
        noted(b.note("authorization", &text.to_string(), &[])).id
    }

    /// `alive`: this test process is the executor; otherwise a dead one.
    fn req(&self, a: u64, alive: bool) -> IntentRequest {
        let (pid, start) = if alive {
            effects::self_executor()
        } else {
            (0, 0)
        };
        IntentRequest {
            authorization: a,
            proposal_sha256: self.psha.clone(),
            path: "NOTES.md".into(),
            target: self.target.clone(),
            content_sha256: self.csha.clone(),
            executor_pid: pid,
            executor_start: start,
        }
    }
}

#[test]
fn effect_intents_are_at_most_once_or_unresolved() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("compose");
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let ws = std::fs::canonicalize(&ws).unwrap();
    let b = ComposeBridge::new(home.clone(), proposer(), "test:fixed-proposer");
    let run = b.run_task("write the note", ws.to_str().unwrap());
    if !aien_omega_compose::LINKED {
        assert!(matches!(run, ControlResponse::Error(_)));
        return;
    }
    let report = match run {
        ControlResponse::ComposeTaskResult(r) => *r,
        other => panic!("run: {other:?}"),
    };
    assert!(report.committed, "{report:?}");
    let f = Fixture {
        target: ws.join("NOTES.md").display().to_string(),
        psha: sha(PROPOSAL.as_bytes()),
        csha: sha(CONTENT.as_bytes()),
    };

    // 1. Grant, intent, write, ack: DONE once; the grant is then spent.
    let a1 = f.grant(&b);
    let i1 = noted(effects::open_intent(&b, &f.req(a1, true))).id;
    refused(
        effects::open_intent(&b, &f.req(a1, true)),
        "ReconciliationRequired",
    );
    let r = reconciled(effects::reconcile(&b, None, "test"));
    assert_eq!(r.outcomes.len(), 1);
    assert!(
        r.outcomes[0].record.is_none(),
        "live executor is never decided"
    );
    std::fs::write(&f.target, CONTENT).unwrap();
    let k1 = noted(effects::ack(&b, i1, &json!({"bytes": CONTENT.len()})));
    assert_eq!(state(&b, k1.id), "DONE");
    refused(effects::open_intent(&b, &f.req(a1, true)), "AlreadySpent");
    assert!(matches!(
        effects::ack(&b, i1, &json!({})),
        ControlResponse::Error(_)
    ));
    let c = controlled(effects::control(&b, "revoke", "drake", Some(a1)));
    assert_eq!((c.revoked, c.recorded), (Some(false), None));

    // 2. Executor died after the intent, before the write: NOT_DONE.
    // (Target removed first: a grant against the content itself would read
    // DONE, ACCEPTANCE-v2 2.2 checks the content digest first.)
    std::fs::remove_file(&f.target).unwrap();
    let a2 = f.grant(&b);
    let i2 = noted(effects::open_intent(&b, &f.req(a2, false))).id;
    let r = reconciled(effects::reconcile(&b, None, "test"));
    let o = r.outcomes.iter().find(|o| o.intent == i2).unwrap();
    assert_eq!(o.state, "NOT_DONE");
    assert!(o.record.is_some());
    refused(effects::open_intent(&b, &f.req(a2, true)), "AlreadySpent");

    // 3. Ambiguous world: UNRESOLVED once per digest, then an operator decides.
    let a3 = f.grant(&b);
    let i3 = noted(effects::open_intent(&b, &f.req(a3, false))).id;
    std::fs::write(&f.target, b"someone else\n").unwrap();
    let r = reconciled(effects::reconcile(&b, None, "test"));
    assert_eq!(r.outcomes[0].state, "UNRESOLVED");
    assert!(r.outcomes[0].record.is_some());
    let r = reconciled(effects::reconcile(&b, None, "test"));
    assert!(
        r.outcomes[0].record.is_none(),
        "same digest is not recorded twice"
    );
    refused(
        effects::open_intent(&b, &f.req(a3, true)),
        "ReconciliationRequired",
    );
    let d = ReconcileDeclare {
        intent: i3,
        state: "not_done".into(),
        approver: "drake".into(),
    };
    let r = reconciled(effects::reconcile(&b, Some(&d), "test"));
    assert_eq!(
        (r.outcomes[0].state.as_str(), r.outcomes[0].note.as_str()),
        ("NOT_DONE", "operator:drake")
    );
    assert!(matches!(
        effects::reconcile(&b, Some(&d), "test"),
        ControlResponse::Error(_)
    ));
    assert_eq!(std::fs::read(&f.target).unwrap(), b"someone else\n");

    // 4. Stale world, revoke, stop and resume.
    let a4 = f.grant(&b);
    std::fs::write(&f.target, b"changed after the grant\n").unwrap();
    refused(effects::open_intent(&b, &f.req(a4, true)), "Stale");
    let a5 = f.grant(&b);
    let c = controlled(effects::control(&b, "revoke", "drake", Some(a5)));
    assert_eq!(c.revoked, Some(true));
    refused(effects::open_intent(&b, &f.req(a5, true)), "Revoked");
    let a6 = f.grant(&b);
    controlled(effects::control(&b, "stop", "drake", None));
    refused(effects::open_intent(&b, &f.req(a6, true)), "Stopped");
    assert!(matches!(
        effects::control(&b, "resume", " ", None),
        ControlResponse::Error(_)
    ));
    controlled(effects::control(&b, "resume", "drake", None));
    refused(effects::open_intent(&b, &f.req(a6, true)), "Stale");
    let a7 = f.grant(&b);
    let i7 = noted(effects::open_intent(&b, &f.req(a7, true))).id;
    std::fs::write(&f.target, CONTENT).unwrap();
    assert_eq!(
        state(&b, noted(effects::ack(&b, i7, &json!({}))).id),
        "DONE"
    );

    // 5. Restart: the ledger is rebuilt from the journal, nothing is open,
    // and the forged-record guard holds.
    let before = ledger(&b).view();
    drop(b);
    let b = ComposeBridge::new(home, proposer(), "test:fixed-proposer");
    let line = effects::reconcile_at_start(&b);
    assert!(line.starts_with("Reconcile: checked 0"), "{line}");
    assert_eq!(ledger(&b).view(), before);
    assert!(effects::check_reserved_note("effect", r#"{"phase":"ack","intent":1}"#).is_err());
}
