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

//! Proposer hook (crate::approved): an approved proposal goes through the
//! production compose run (J-Space branch, AEGIS verify, World commit) and
//! then the durable effect ledger; unverified, unauthenticated (#249 A) and
//! replayed (#249 B) proposals are refused with no compose run. CPU only, no
//! model, no socket: the bridge is driven exactly as the daemon drives it. In
//! a stub build only the refusal paths run.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_omega_compose::hex;
use aien_runtime::approved::{
    self, ApprovedComposeReport, ApprovedProposal, ApprovedRefusal, ProposerHook,
};
use aien_runtime::approved_auth::{
    approval_key, canonical_workspace, desk_key_path, ApprovalIdentity, DeskKey,
};
use aien_runtime::control::{ComposeRecallReport, ControlResponse};
use aien_runtime::effects::{self, IntentRequest, Ledger};
use aien_runtime::{ComposeBridge, ComposeProposer};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

/// The proposer label of a run whose proposal came from the approved-proposal hook.
const APPROVED_PROPOSER_LABEL: &str = "hook:approved-proposal";

const PATH: &str = "NOTES.md";
const CONTENT: &str = "approved by the INTERPLANE desk\n";

/// The model proposer must never be called on the hook path.
fn model_never_called() -> ComposeProposer {
    Arc::new(|_: &str, _: std::time::Duration| panic!("the model proposer ran on the hook path"))
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

/// Independent of approved::approved_proposal_sha256: the interplane#77
/// canonical form written out by hand.
fn interplane_sha(path: &str, content: &str) -> String {
    let s = format!(
        "{{\"content\":{},\"path\":{}}}",
        serde_json::to_string(content).unwrap(),
        serde_json::to_string(path).unwrap()
    );
    sha(s.as_bytes())
}

/// Unsigned: tests sign with [`sign`] after any edit.
fn unsigned(request: &str, approval: &str) -> ApprovedProposal {
    ApprovedProposal {
        request_id: request.into(),
        trace_id: format!("trace-{request}"),
        approval_id: approval.into(),
        approver: "interplane-host".into(),
        path: PATH.into(),
        content: CONTENT.into(),
        approved_proposal_sha256: interplane_sha(PATH, CONTENT),
        content_sha256: sha(CONTENT.as_bytes()),
        approval_mac: String::new(),
        requirements: Some(String::new()),
        requirements_mac: String::new(),
        requirements_base: None,
    }
}

/// The desk key of a compose home (the test is the desk).
fn desk(b: &ComposeBridge) -> DeskKey {
    DeskKey::load(&desk_key_path(b.dir())).unwrap()
}

/// The workspace every test home pairs with its compose home (`setup`:
/// `<tmp>/compose` and `<tmp>/ws`); the approval binds it.
fn ws_of(b: &ComposeBridge) -> std::path::PathBuf {
    b.dir().parent().unwrap().join("ws")
}

fn sign(b: &ComposeBridge, mut p: ApprovedProposal) -> ApprovedProposal {
    desk(b).seal(&mut p, &ws_of(b));
    p
}

/// Re-sign after editing fields (an honest desk approving the edited change).
fn resign(b: &ComposeBridge, p: &ApprovedProposal) -> ApprovedProposal {
    sign(b, p.clone())
}

fn recalled(b: &ComposeBridge, ids: &[u64]) -> ComposeRecallReport {
    match b.recall(ids, None) {
        ControlResponse::ComposeRecalled(r) => *r,
        other => panic!("recall: {other:?}"),
    }
}

fn refused(
    r: Result<ApprovedComposeReport, Box<ApprovedRefusal>>,
    name: &str,
) -> Box<ApprovedRefusal> {
    match r {
        Err(e) => {
            assert_eq!(e.name, name, "want {name}, got {e}");
            e
        }
        Ok(v) => panic!("want refusal {name}, got {v:?}"),
    }
}

/// Stub build: nothing opens the compose home, so the claim is refused.
fn stub_refused(r: Result<ApprovedComposeReport, Box<ApprovedRefusal>>) {
    let e = r.expect_err("stub build must refuse");
    assert!(
        e.refused_by == "REPLAY_REFUSED" || e.name == "ComposeError",
        "{e}"
    );
}

fn committed(r: Result<ApprovedComposeReport, Box<ApprovedRefusal>>) -> ApprovedComposeReport {
    let r = r.unwrap_or_else(|e| panic!("want COMMITTED, got {e}"));
    assert_eq!(r.state, "COMMITTED");
    r
}

/// omega caps the compose homes one process may hold open (E_FULL): the
/// tests of this binary take turns.
static HOMES: std::sync::Mutex<()> = std::sync::Mutex::new(());

fn turn() -> std::sync::MutexGuard<'static, ()> {
    HOMES.lock().unwrap_or_else(|e| e.into_inner())
}

fn setup() -> (tempfile::TempDir, Arc<ComposeBridge>, ProposerHook, String) {
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    std::fs::create_dir_all(&ws).unwrap();
    let b = Arc::new(ComposeBridge::new(
        tmp.path().join("compose"),
        model_never_called(),
        "test:model-never-called",
    ));
    DeskKey::create(&desk_key_path(b.dir())).unwrap();
    let hook = ProposerHook::new(b.clone());
    let ws = ws.display().to_string();
    (tmp, b, hook, ws)
}

#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn approved_proposal_commits_then_effect_is_done() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = sign(&b, unsigned("req-1", "appr-1"));
    let out = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    let r = committed(out);
    // Correlation unchanged, authentication and replay identity reported.
    assert_eq!(
        (
            r.request_id.as_str(),
            r.trace_id.as_str(),
            r.approval_id.as_str()
        ),
        ("req-1", "trace-req-1", "appr-1")
    );
    let d = desk(&b);
    assert_eq!(r.desk_key_id, d.id());
    assert_eq!(
        r.approval_key,
        approval_key(&ApprovalIdentity::of(
            &p,
            &canonical_workspace(&ws_of(&b)).unwrap(),
            d.id()
        ))
    );
    assert!(r.replay_claim != 0);
    assert_eq!(r.approved_proposal_sha256, interplane_sha(PATH, CONTENT));
    let text = format!("filename: {PATH}\n{CONTENT}");
    assert_eq!(r.compose_proposal_sha256, sha(text.as_bytes()));
    assert_ne!(r.approved_proposal_sha256, r.compose_proposal_sha256);
    // The compose run: one J-Space branch, AEGIS pass, World commit.
    let t = r.task.as_ref().unwrap();
    assert!(t.committed, "{t:?}");
    assert_eq!((t.outcome, t.branch_count, t.winner), (1, 1, Some(0)));
    assert_eq!(t.aegis_pass_mask & 1, 1);
    assert_eq!(t.proposer, APPROVED_PROPOSER_LABEL);
    assert_eq!(t.proposal_attempts.len(), 1);
    assert_eq!(t.proposal_attempts[0].aegis.as_deref(), Some("pass"));
    assert!(t.cx_goal != 0 && t.cx_evidence != 0 && t.cx_promotion != 0);
    assert_eq!(r.grant_links, vec![t.cx_promotion, t.cx_evidence]);
    // The World commit record is in the Cortex journal and verifies.
    let c = recalled(&b, &[t.cx_promotion]);
    assert!(c.missing.is_empty(), "{c:?}");
    assert!(c.cited[0].verified, "{:?}", c.cited[0]);
    assert_eq!(c.cited[0].id, t.cx_promotion);
    // Nothing touched the workspace yet: the hook grants nothing.
    let target = Path::new(&ws).join(PATH);
    assert!(!target.exists());

    // Effect: the daemon's own grant (reserved, keyed on the compose hash,
    // linked to promotion, evidence and the replay claim), intent, write, ack.
    let a = r.approved_grant.expect("the daemon wrote the grant");
    let tgt = r.target.clone().expect("grant target");
    assert_eq!(
        Path::new(&tgt),
        std::fs::canonicalize(&ws).unwrap().join(PATH)
    );
    let g = recalled(&b, &[a]);
    assert!(g.cited[0].verified);
    assert_eq!(g.cited[0].note.as_deref(), Some("authorization"));
    for x in [t.cx_promotion, t.cx_evidence, r.replay_claim] {
        assert!(g.cited[0].links.contains(&x), "{:?}", g.cited[0]);
    }
    let gv: Value = serde_json::from_str(g.cited[0].text.as_deref().unwrap()).unwrap();
    assert_eq!(gv["approved_grant"], 1);
    assert_eq!(gv["proposal_sha256"], json!(r.compose_proposal_sha256));
    assert_eq!(
        gv["approved_proposal_sha256"],
        json!(r.approved_proposal_sha256)
    );
    assert_eq!(gv["approval_key"], json!(r.approval_key));
    assert_eq!(
        (gv["request_id"].as_str(), gv["trace_id"].as_str()),
        (Some(r.request_id.as_str()), Some(r.trace_id.as_str()))
    );
    assert_eq!(gv["prior_sha256"], Value::Null);
    let target = Path::new(&tgt).to_path_buf();
    let (pid, start) = effects::self_executor();
    let req = IntentRequest {
        authorization: a,
        proposal_sha256: r.compose_proposal_sha256.clone(),
        path: r.path.clone(),
        target: tgt.clone(),
        content_sha256: r.content_sha256.clone(),
        executor_pid: pid,
        executor_start: start,
    };
    let i = match effects::open_intent(&b, &req) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("intent: {other:?}"),
    };
    std::fs::write(&target, CONTENT).unwrap();
    let k = match effects::ack(&b, i, &json!({"written_sha256": r.content_sha256})) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("ack: {other:?}"),
    };
    let rec = recalled(&b, &[k]);
    let v: Value = serde_json::from_str(rec.cited[0].text.as_deref().unwrap()).unwrap();
    assert_eq!(v["state"], "DONE", "{v}");
    assert_eq!(std::fs::read(&target).unwrap(), CONTENT.as_bytes());
}

#[test]
fn unverified_proposal_is_refused_with_zero_effects() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    println!(
        "approved_proposal_test: compose LINKED={} (zero-journal assertions {})",
        aien_omega_compose::LINKED,
        if aien_omega_compose::LINKED {
            "run"
        } else {
            "skipped: stub build"
        }
    );
    let before = aien_omega_compose::LINKED.then(|| recalled(&b, &[]).records_total);
    let mut cases: Vec<(&str, ApprovedProposal)> = Vec::new();
    let proposal = |r: &str, a: &str| unsigned(r, a);
    let mut p = proposal("u1", "a1");
    p.content_sha256 = sha(b"other bytes\n");
    cases.push(("content sha mismatch", p));
    let mut p = proposal("u2", "a2");
    // The compose text sha in the approved slot: the other hash semantics.
    p.approved_proposal_sha256 = sha(format!("filename: {PATH}\n{CONTENT}").as_bytes());
    cases.push(("text sha in the approved slot", p));
    let mut p = proposal("u3", "a3");
    p.path = "../escape.md".into();
    p.approved_proposal_sha256 = interplane_sha(&p.path, CONTENT);
    cases.push(("path outside the workspace", p));
    let mut p = proposal("u4", "a4");
    p.content = "no trailing newline".into();
    p.content_sha256 = sha(p.content.as_bytes());
    p.approved_proposal_sha256 = interplane_sha(PATH, &p.content);
    cases.push(("content the template would change", p));
    let mut p = proposal("u5", "");
    p.approval_id = " ".into();
    cases.push(("no approval id", p));
    let mut p = proposal("", "a6");
    p.request_id = String::new();
    cases.push(("no request id", p));
    for (what, p) in &cases {
        // Signed by the desk: verification still refuses.
        let r = hook.submit(&resign(&b, p), &ws);
        assert!(r.is_err(), "{what}: {r:?}");
        refused(r, "Unverified");
    }
    assert!(!Path::new(&ws).join(PATH).exists());
    assert!(!Path::new(&ws).join("../escape.md").exists());
    if let Some(n) = before {
        let after = recalled(&b, &[]);
        assert_eq!(
            after.records_total, n,
            "a refused proposal appended records"
        );
        let l = Ledger::from_records(&after.host).unwrap();
        assert!(l.grants.is_empty() && l.intents.is_empty());
    }
}

/// Replay in the same process: the same approval again returns the original
/// result without a second run (ALREADY_COMMITTED); the request id or the
/// approval id under another approval is refused; nothing is appended.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn duplicate_proposal_is_refused() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = sign(&b, unsigned("req-d", "appr-d"));
    let first = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(first);
        stub_refused(hook.submit(&p, &ws));
        return;
    }
    let first = committed(first);
    let n = recalled(&b, &[]).records_total;
    let again = hook.submit(&p, &ws).unwrap();
    assert_eq!(again.state, "ALREADY_COMMITTED");
    assert!(again.task.is_none());
    assert_eq!(again.compose_proposal_sha256, first.compose_proposal_sha256);
    assert_eq!(again.grant_links, first.grant_links);
    assert_eq!(again.replay_claim, first.replay_claim);
    // Same approval id, new request (desk-signed): refused, not "already committed".
    let e = refused(
        hook.submit(&sign(&b, unsigned("req-other", "appr-d")), &ws),
        "AlreadyCommitted",
    );
    assert_eq!(e.refused_by, "REPLAY_REFUSED");
    assert_eq!(e.replay_claim, Some(first.replay_claim));
    refused(
        hook.submit(&sign(&b, unsigned("req-d", "appr-other")), &ws),
        "AlreadyCommitted",
    );
    // Same ids, another trace: refused too.
    let mut t = unsigned("req-d", "appr-d");
    t.trace_id = "trace-other".into();
    refused(hook.submit(&sign(&b, t), &ws), "AlreadyCommitted");
    assert_eq!(
        recalled(&b, &[]).records_total,
        n,
        "a replay appended records"
    );
    // The same change under a fresh request and approval is a new proposal.
    committed(hook.submit(&sign(&b, unsigned("req-e", "appr-e")), &ws));
}

/// #249 A: a caller without the desk key cannot make the hook trust any
/// field. Every variant is refused Unauthenticated before anything runs:
/// zero records, nothing on disk.
#[test]
fn unauthenticated_approval_is_refused_with_no_compose() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (tmp, b, hook, ws) = setup();
    let good = sign(&b, unsigned("req-a", "appr-a"));
    let before = aien_omega_compose::LINKED.then(|| recalled(&b, &[]).records_total);
    let mut cases: Vec<(&str, ApprovedProposal)> = Vec::new();
    let mut p = good.clone();
    p.approver = "drake".into();
    cases.push(("forged approver, correct bytes and hashes", p));
    let mut p = good.clone();
    p.request_id = "req-a2".into();
    cases.push(("changed request_id", p));
    let mut p = good.clone();
    p.trace_id = "trace-x".into();
    cases.push(("changed trace_id", p));
    let mut p = good.clone();
    p.approval_id = "appr-a2".into();
    cases.push(("changed approval_id", p));
    let mut p = good.clone();
    p.path = "OTHER.md".into();
    p.approved_proposal_sha256 = interplane_sha(&p.path, &p.content);
    cases.push(("changed path, hashes recomputed", p));
    let mut p = good.clone();
    p.content = "other approved bytes\n".into();
    p.content_sha256 = sha(p.content.as_bytes());
    p.approved_proposal_sha256 = interplane_sha(PATH, &p.content);
    cases.push(("changed content, hashes recomputed", p));
    let mut p = good.clone();
    p.approval_mac = String::new();
    cases.push(("no MAC", p));
    let mut p = good.clone();
    p.approval_mac = sha(b"caller text is not approval");
    cases.push(("MAC made of caller text", p));
    let other = DeskKey::create(&tmp.path().join("other.key")).unwrap();
    let mut p = good.clone();
    p.approval_mac = other.sign(&p, &ws_of(&b));
    cases.push(("MAC under another key", p));
    let mut p = good.clone();
    p.approval_mac = p.approval_mac.to_uppercase();
    cases.push(("MAC in another hex case", p));
    for (what, p) in &cases {
        let e = refused(hook.submit(p, &ws), "Unauthenticated");
        assert_eq!(e.refused_by, "PROPOSAL_REFUSED", "{what}");
        assert_eq!(e.request_id, p.request_id, "{what}: correlation kept");
    }
    assert!(!Path::new(&ws).join(PATH).exists());
    if let Some(n) = before {
        assert_eq!(
            recalled(&b, &[]).records_total,
            n,
            "a refusal appended records"
        );
        // The honest approval was not consumed by the forgeries.
        committed(hook.submit(&good, &ws));
    }
}

/// #249 A key rules: absent, group/other-readable, symlinked or malformed
/// keys are refused NoDesk; refusals and reports never carry the key.
#[test]
fn desk_key_file_rules() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    use std::os::unix::fs::PermissionsExt;
    let (tmp, b, hook, ws) = setup();
    let path = desk_key_path(b.dir());
    let key_text = std::fs::read_to_string(&path).unwrap();
    let p = sign(&b, unsigned("req-k", "appr-k"));
    for mode in [0o640, 0o604, 0o644] {
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(mode)).unwrap();
        let e = refused(hook.submit(&p, &ws), "NoDesk");
        assert!(e.detail.contains("group or other"), "{e}");
        assert!(!e.to_string().contains(key_text.trim()));
    }
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    // A symlink to a well-formed 0600 key is refused.
    let real = tmp.path().join("real.key");
    std::fs::rename(&path, &real).unwrap();
    std::os::unix::fs::symlink(&real, &path).unwrap();
    let e = refused(hook.submit(&p, &ws), "NoDesk");
    assert!(e.detail.contains("symlink"), "{e}");
    std::fs::remove_file(&path).unwrap();
    // Absent.
    refused(hook.submit(&p, &ws), "NoDesk");
    // Malformed.
    std::fs::write(&path, "not a key\n").unwrap();
    std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o600)).unwrap();
    refused(hook.submit(&p, &ws), "NoDesk");
    // create never overwrites.
    assert!(DeskKey::create(&path).is_err());
    std::fs::remove_file(&path).unwrap();
    // Restored key: the MAC verifies again. Rotation: a new key refuses the old MAC.
    std::fs::rename(&real, &path).unwrap();
    let k = desk(&b);
    assert!(!format!("{k:?}").contains(key_text.trim()));
    std::fs::remove_file(&path).unwrap();
    DeskKey::create(&path).unwrap();
    let e = refused(hook.submit(&p, &ws), "Unauthenticated");
    assert!(!e.to_string().contains(key_text.trim()));
    if aien_omega_compose::LINKED {
        let r = committed(hook.submit(&resign(&b, &p), &ws));
        assert!(!serde_json::to_string(&r).unwrap().contains(key_text.trim()));
        assert_ne!(r.desk_key_id, k.id(), "rotation changes the key id");
    }
}

/// #249 A confinement: a compose home (desk key, journal) inside the
/// workspace, or a workspace inside the compose home, refuses the desk.
#[test]
fn compose_home_overlapping_the_workspace_is_refused() {
    let _turn = turn();
    let tmp = tempfile::tempdir().unwrap();
    let ws = tmp.path().join("ws");
    let inner = ws.join(".aien-compose");
    std::fs::create_dir_all(&inner).unwrap();
    let b = Arc::new(ComposeBridge::new(inner.clone(), model_never_called(), "t"));
    DeskKey::create(&desk_key_path(&inner)).unwrap();
    let hook = ProposerHook::new(b.clone());
    let p = sign(&b, unsigned("req-c", "appr-c"));
    refused(hook.submit(&p, ws.to_str().unwrap()), "Confinement");
    // The other way round: the workspace inside the compose home.
    let home = tmp.path().join("home");
    let ws2 = home.join("ws");
    std::fs::create_dir_all(&ws2).unwrap();
    let b2 = Arc::new(ComposeBridge::new(home.clone(), model_never_called(), "t"));
    DeskKey::create(&desk_key_path(&home)).unwrap();
    let p2 = sign(&b2, unsigned("req-c", "appr-c"));
    refused(
        ProposerHook::new(b2).submit(&p2, ws2.to_str().unwrap()),
        "Confinement",
    );
    // Nothing was ever opened: no journal in either home.
    assert!(!inner.join("cortex.cx").exists() && !home.join("cortex.cx").exists());
}

/// #249 B, CENTRAL: a fresh bridge over the same compose home (what a
/// daemon restart is) refuses every replay; the retry of the same approval
/// gets the original result back and nothing runs again.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn replay_is_refused_after_restart() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (tmp, b, hook, ws) = setup();
    let p = sign(&b, unsigned("req-r", "appr-r"));
    let first = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(first);
        return;
    }
    let first = committed(first);
    drop(hook);
    let dir = b.dir().to_path_buf();
    drop(b);
    let b2 = Arc::new(ComposeBridge::new(dir, model_never_called(), "t"));
    let hook2 = ProposerHook::new(b2.clone());
    let n = recalled(&b2, &[]).records_total;
    let again = hook2.submit(&p, &ws).unwrap();
    assert_eq!(again.state, "ALREADY_COMMITTED");
    assert_eq!(again.compose_proposal_sha256, first.compose_proposal_sha256);
    refused(
        hook2.submit(&sign(&b2, unsigned("req-r2", "appr-r")), &ws),
        "AlreadyCommitted",
    );
    refused(
        hook2.submit(&sign(&b2, unsigned("req-r", "appr-r2")), &ws),
        "AlreadyCommitted",
    );
    assert_eq!(
        recalled(&b2, &[]).records_total,
        n,
        "a replay after restart appended records"
    );
    let _ = tmp;
}

/// #249 B: concurrent duplicates. Eight threads submit the same approval and
/// eight more submit new request ids under one approval id: exactly one
/// compose run per approval id.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn concurrent_duplicates_run_once() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    if !aien_omega_compose::LINKED {
        return;
    }
    let hook = Arc::new(hook);
    let same = sign(&b, unsigned("req-cc", "appr-cc"));
    let mut handles = Vec::new();
    for i in 0..16 {
        let (h, ws) = (hook.clone(), ws.clone());
        let p = if i < 8 {
            same.clone()
        } else {
            sign(&b, unsigned(&format!("req-cc-{i}"), "appr-cc"))
        };
        handles.push(std::thread::spawn(move || h.submit(&p, &ws)));
    }
    let outs: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    let ran: Vec<_> = outs
        .iter()
        .filter_map(|o| o.as_ref().ok())
        .filter(|r| r.task.is_some())
        .collect();
    assert_eq!(ran.len(), 1, "exactly one submission runs: {outs:?}");
    for o in &outs {
        match o {
            Ok(r) => assert!(r.state == "COMMITTED" || r.state == "ALREADY_COMMITTED"),
            Err(e) => assert!(
                ["InFlight", "AlreadyCommitted"].contains(&e.name.as_str()),
                "{e}"
            ),
        }
    }
    // One promotion in the journal for this approval: one World commit.
    let host = recalled(&b, &[]).host;
    let claims = host
        .iter()
        .filter(|r| {
            r.text
                .as_deref()
                .is_some_and(|t| t.contains("\"approved_submission\":\"accepted\""))
        })
        .count();
    assert_eq!(claims, 1);
}

#[test]
fn approved_hash_is_the_interplane_form() {
    let _turn = turn();
    assert_eq!(
        approved::approved_proposal_sha256("a/b.md", "x \"q\"\n\u{1}é\n"),
        interplane_sha("a/b.md", "x \"q\"\n\u{1}é\n")
    );
}

/// The path already exists: RunComposeTask would treat it as an edit target
/// and merge the reply into the seed (v7 T5). An approved whole-file
/// proposal must commit byte-exact, with no seed line merged in.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn approved_proposal_over_existing_file_commits_byte_exact() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, _b, hook, ws) = setup();
    let seed = "# Notes\nseed line one\nseed line two\n";
    std::fs::write(Path::new(&ws).join(PATH), seed).unwrap();
    // Shares "# Notes" with the seed, so an edit merge would keep both seed lines.
    let content = "# Notes\napproved replacement line\n";
    let mut p = unsigned("req-x", "appr-x");
    p.content = content.into();
    p.content_sha256 = sha(content.as_bytes());
    p.approved_proposal_sha256 = interplane_sha(PATH, content);
    let p = sign(&_b, p);
    let out = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    let r = committed(out);
    let t = r.task.as_ref().unwrap();
    assert!(t.committed, "{t:?}");
    let text = format!("filename: {PATH}\n{content}");
    assert_eq!(t.proposal.as_deref(), Some(text.as_str()));
    assert!(!t.proposal.as_deref().unwrap().contains("seed line"));
    assert_eq!(
        t.proposal_content_sha256.as_deref(),
        Some(p.content_sha256.as_str())
    );
    assert_eq!(r.compose_proposal_sha256, sha(text.as_bytes()));
    // The hook wrote nothing: the seed is untouched.
    assert_eq!(
        std::fs::read_to_string(Path::new(&ws).join(PATH)).unwrap(),
        seed
    );
}

/// 476ca4 c17: a valid approval presented with another workspace is refused
/// as Unauthenticated (the workspace is in the MAC), nothing is consumed, and
/// the same approval still commits for the workspace it was made for; the
/// daemon's grant targets the bound workspace.
#[test]
fn approval_redirected_to_another_workspace_is_refused() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (tmp, b, hook, ws) = setup();
    let other = tmp.path().join("elsewhere");
    std::fs::create_dir_all(&other).unwrap();
    // A symlink to the right workspace resolves to it: same canonical path.
    let alias = tmp.path().join("ws-alias");
    std::os::unix::fs::symlink(&ws, &alias).unwrap();
    let p = sign(&b, unsigned("req-w", "appr-w"));
    let before = aien_omega_compose::LINKED.then(|| recalled(&b, &[]).records_total);
    for w in [other.to_str().unwrap(), "/", tmp.path().to_str().unwrap()] {
        let e = hook.submit(&p, w).unwrap_err();
        assert!(
            e.name == "Unauthenticated" || e.name == "Confinement",
            "{w}: {e}"
        );
    }
    if let Some(n) = before {
        assert_eq!(recalled(&b, &[]).records_total, n, "nothing consumed");
    }
    assert!(!other.join(PATH).exists());
    let out = hook.submit(&p, alias.to_str().unwrap());
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    let r = committed(out);
    assert_eq!(
        r.target.as_deref().map(Path::new),
        Some(
            canonical_workspace(Path::new(&ws))
                .map(|w| Path::new(&w).join(PATH))
                .unwrap()
                .as_path()
        )
    );
}

/// 476ca4 c25: a journal-level byte copy of the daemon's grant (written past
/// ComposeNote's reserved check, as only direct journal access could) gives
/// one approval two grants: neither opens an intent, nothing is written.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn copied_approved_grant_opens_nothing() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let r = committed(hook.submit(&sign(&b, unsigned("req-g", "appr-g")), &ws));
    let a = r.approved_grant.unwrap();
    let g = recalled(&b, &[a]);
    let text = g.cited[0].text.clone().unwrap();
    let copy = match b.note_unchecked("authorization", &text, &g.cited[0].links) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("bridge-level copy: {other:?}"),
    };
    let (pid, start) = effects::self_executor();
    for id in [a, copy] {
        let req = IntentRequest {
            authorization: id,
            proposal_sha256: r.compose_proposal_sha256.clone(),
            path: r.path.clone(),
            target: r.target.clone().unwrap(),
            content_sha256: r.content_sha256.clone(),
            executor_pid: pid,
            executor_start: start,
        };
        match effects::open_intent(&b, &req) {
            ControlResponse::Error(e) => {
                assert!(e.contains("more than one approved grant"), "#{id}: {e}")
            }
            other => panic!("#{id} opened: {other:?}"),
        }
    }
    let l = Ledger::from_records(&recalled(&b, &[]).host).unwrap();
    assert!(l.intents.is_empty());
    assert!(!Path::new(&r.target.unwrap()).exists());
}

/// A document with several fenced examples and prose after the last one.
const FENCED_DOC: &str = "# FAQ\n\nInstall:\n```bash\ncurl -fsSL https://example.com/i.sh | sh\n```\nThen verify:\n```bash\ntool --version\n```\nLast words after the final example.\n";

/// One approved document with embedded fences, through the real approval and
/// effect path: the approval binds sha256(content); the proposal text reads
/// back byte for byte; the intent only opens for that digest; and the effect
/// is DONE only when the bytes on disk hash to the approved digest. A write cut
/// at the first inner fence (the D5 truncation) is UNRESOLVED, never DONE.
#[test]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
fn approved_document_with_fences_is_saved_byte_exact_or_not_done() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    // The compose template keeps every byte (check_file_proposal reads it back).
    let text = format!("filename: {PATH}\n{FENCED_DOC}");
    let back = aien_runtime::check_file_proposal(&text).unwrap();
    assert_eq!(back.content, FENCED_DOC);
    let mut p = unsigned("req-f1", "appr-f1");
    p.content = FENCED_DOC.into();
    p.content_sha256 = sha(FENCED_DOC.as_bytes());
    p.approved_proposal_sha256 = interplane_sha(PATH, FENCED_DOC);
    let p = sign(&b, p);
    let out = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    let r = committed(out);
    assert_eq!(r.content_sha256, sha(FENCED_DOC.as_bytes()));
    assert_eq!(r.compose_proposal_sha256, sha(text.as_bytes()));
    let a = r.approved_grant.expect("grant");
    let tgt = r.target.clone().unwrap();
    let (pid, start) = effects::self_executor();
    let req = |content_sha: &str| IntentRequest {
        authorization: a,
        proposal_sha256: r.compose_proposal_sha256.clone(),
        path: r.path.clone(),
        target: tgt.clone(),
        content_sha256: content_sha.to_string(),
        executor_pid: pid,
        executor_start: start,
    };
    // An intent for any other bytes (for example the truncated text) is refused.
    let cut = FENCED_DOC.split("```bash").next().unwrap();
    match effects::open_intent(&b, &req(&sha(cut.as_bytes()))) {
        ControlResponse::Error(e) => assert!(e.contains("NotAuthorized"), "{e}"),
        other => panic!("want refusal, got {other:?}"),
    }
    let i = match effects::open_intent(&b, &req(&r.content_sha256)) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("intent: {other:?}"),
    };
    let state_of = |k: u64| -> Value {
        let rec = recalled(&b, &[k]);
        serde_json::from_str(rec.cited[0].text.as_deref().unwrap()).unwrap()
    };
    // Truncated bytes on disk: the ack records UNRESOLVED, not DONE.
    let target = Path::new(&tgt).to_path_buf();
    std::fs::write(&target, cut).unwrap();
    let k = match effects::ack(&b, i, &json!({})) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("ack: {other:?}"),
    };
    let v = state_of(k);
    assert_eq!(v["state"], "UNRESOLVED", "{v}");
    assert_ne!(sha(&std::fs::read(&target).unwrap()), p.content_sha256);

    // Second approval, same document at another path: exact bytes are DONE.
    let mut p2 = unsigned("req-f2", "appr-f2");
    p2.path = "FAQ.md".into();
    p2.content = FENCED_DOC.into();
    p2.content_sha256 = sha(FENCED_DOC.as_bytes());
    p2.approved_proposal_sha256 = interplane_sha("FAQ.md", FENCED_DOC);
    let r2 = committed(hook.submit(&sign(&b, p2.clone()), &ws));
    let a2 = r2.approved_grant.expect("grant");
    let tgt2 = r2.target.clone().unwrap();
    let i2 = match effects::open_intent(
        &b,
        &IntentRequest {
            authorization: a2,
            proposal_sha256: r2.compose_proposal_sha256.clone(),
            path: r2.path.clone(),
            target: tgt2.clone(),
            content_sha256: r2.content_sha256.clone(),
            executor_pid: pid,
            executor_start: start,
        },
    ) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("intent: {other:?}"),
    };
    std::fs::write(&tgt2, FENCED_DOC).unwrap();
    let k2 = match effects::ack(&b, i2, &json!({})) {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("ack: {other:?}"),
    };
    assert_eq!(state_of(k2)["state"], "DONE");
    let disk = std::fs::read(&tgt2).unwrap();
    assert_eq!(disk, FENCED_DOC.as_bytes());
    assert_eq!(sha(&disk), p2.content_sha256);
}

// ---- goal requirements bound into the desk-signed approval ----

/// `unsigned` with a requirement goal, sealed by the desk.
fn with_goal(
    b: &ComposeBridge,
    request: &str,
    approval: &str,
    goal: Option<&str>,
) -> ApprovedProposal {
    let mut p = unsigned(request, approval);
    p.requirements = goal.map(str::to_string);
    sign(b, p)
}

#[test]
fn bound_requirements_the_bytes_miss_are_refused_and_nothing_is_consumed() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = with_goal(
        &b,
        "req-rq1",
        "appr-rq1",
        Some("write NOTES.md in at least 5 lines"),
    );
    let e = refused(hook.submit(&p, &ws), "RequirementsUnmet");
    assert!(
        e.detail.contains("at least 5 non-empty lines, found 1"),
        "{e}"
    );
    // Refused before the claim: no replay claim was consumed, nothing was
    // written, and the same ids still work for bytes that meet the goal.
    assert!(!Path::new(&ws).join(PATH).exists());
    if !aien_omega_compose::LINKED {
        return;
    }
    let ok = with_goal(
        &b,
        "req-rq1",
        "appr-rq1",
        Some("write NOTES.md in at least 1 line."),
    );
    let r = committed(hook.submit(&ok, &ws));
    assert_eq!(r.content_sha256, sha(CONTENT.as_bytes()));
    assert_eq!(
        r.task.as_ref().unwrap().proposal_content_sha256.as_deref(),
        Some(r.content_sha256.as_str())
    );
}

#[test]
fn bytes_meeting_the_bound_requirements_commit_with_equal_sha256() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = with_goal(
        &b,
        "req-rq2",
        "appr-rq2",
        Some("write it in at least 1 line"),
    );
    let out = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    let r = committed(out);
    let t = r.task.as_ref().unwrap();
    assert!(t.committed);
    assert_eq!(r.content_sha256, sha(CONTENT.as_bytes()));
    assert_eq!(
        t.proposal_content_sha256.as_deref(),
        Some(r.content_sha256.as_str())
    );
}

#[test]
fn explicitly_signed_empty_requirements_are_allowed() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = with_goal(&b, "req-rq3", "appr-rq3", Some(""));
    let out = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    committed(out);
}

#[test]
fn missing_requirement_binding_is_refused_even_when_signed() {
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = with_goal(&b, "req-rq4", "appr-rq4", None);
    // Refused in the checks that need no compose home, stub build included.
    refused(hook.submit(&p, &ws), "RequirementsUnbound");
}

#[test]
fn tampered_requirement_set_fails_the_mac() {
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    // The desk signed a demanding goal; the caller swaps in an empty one.
    let mut p = with_goal(
        &b,
        "req-rq5",
        "appr-rq5",
        Some("write NOTES.md in at least 5 lines"),
    );
    p.requirements = Some(String::new());
    refused(hook.submit(&p, &ws), "Unauthenticated");
    // And a bound goal cannot be dropped to None either.
    let mut q = with_goal(&b, "req-rq6", "appr-rq6", Some(""));
    q.requirements = None;
    refused(hook.submit(&q, &ws), "RequirementsUnbound");
}

#[test]
fn uncertain_bound_requirements_are_refused() {
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let p = with_goal(
        &b,
        "req-rq7",
        "appr-rq7",
        Some("at least 3 lines of context"),
    );
    refused(hook.submit(&p, &ws), "RequirementsUncertain");
}

#[test]
fn bound_added_line_counts_are_measured_against_the_existing_file() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    std::fs::write(Path::new(&ws).join(PATH), "one\ntwo\n").unwrap();
    // The approved bytes add one line to the two that exist.
    let mut p = unsigned("req-al1", "appr-al1");
    p.content = "one\ntwo\nthree\n".into();
    p.approved_proposal_sha256 = interplane_sha(PATH, &p.content);
    p.content_sha256 = sha(p.content.as_bytes());
    p.requirements = Some("Add 2 lines to NOTES.md".into());
    p.requirements_base = Some(sha(b"one\ntwo\n"));
    let two = sign(&b, p.clone());
    let e = refused(hook.submit(&two, &ws), "RequirementsUnmet");
    assert!(
        e.detail
            .contains("exactly 2 added non-empty lines, found 1"),
        "{e}"
    );
    // One added line meets "Add one line": never a requirement refusal.
    p.request_id = "req-al2".into();
    p.trace_id = "trace-req-al2".into();
    p.approval_id = "appr-al2".into();
    p.requirements = Some("Add one line to NOTES.md".into());
    let one = sign(&b, p);
    if let Err(e) = hook.submit(&one, &ws) {
        assert!(!e.name.starts_with("Requirements"), "{e}");
    }
}

#[test]
fn a_path_that_reads_like_a_requirement_is_not_one() {
    let _home = home_guard::home_slot();
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    let path = "docs/add-2-lines-to-10-lines.md";
    std::fs::create_dir_all(Path::new(&ws).join("docs")).unwrap();
    let mut p = unsigned("req-mg1", "appr-mg1");
    p.path = path.into();
    p.approved_proposal_sha256 = interplane_sha(path, CONTENT);
    let p = sign(&b, p);
    let out = hook.submit(&p, &ws);
    if !aien_omega_compose::LINKED {
        stub_refused(out);
        return;
    }
    committed(out);
}

#[test]
fn added_line_goal_without_a_bound_base_or_with_a_stale_one_is_refused() {
    let _turn = turn();
    let (_tmp, b, hook, ws) = setup();
    std::fs::write(Path::new(&ws).join(PATH), "one\ntwo\n").unwrap();
    let mut p = unsigned("req-bs1", "appr-bs1");
    p.content = "one\ntwo\nthree\n".into();
    p.approved_proposal_sha256 = interplane_sha(PATH, &p.content);
    p.content_sha256 = sha(p.content.as_bytes());
    p.requirements = Some("Add one line to NOTES.md".into());
    // No base bound: the check could not be tied to a file state.
    let none = sign(&b, p.clone());
    refused(hook.submit(&none, &ws), "RequirementsUnbound");
    // A base that is not the file now (it changed after the desk looked).
    p.request_id = "req-bs2".into();
    p.trace_id = "trace-req-bs2".into();
    p.approval_id = "appr-bs2".into();
    p.requirements_base = Some(sha(b"something else\n"));
    let stale = sign(&b, p.clone());
    let e = refused(hook.submit(&stale, &ws), "BaseChanged");
    assert!(e.detail.contains("changed since the desk bound it"), "{e}");
    // The base is covered by the MAC: swapping it after signing fails authentication.
    p.request_id = "req-bs3".into();
    p.trace_id = "trace-req-bs3".into();
    p.approval_id = "appr-bs3".into();
    p.requirements_base = Some(sha(b"one\ntwo\n"));
    let mut signed = sign(&b, p);
    signed.requirements_base = Some(sha(b"something else\n"));
    refused(hook.submit(&signed, &ws), "Unauthenticated");
}
