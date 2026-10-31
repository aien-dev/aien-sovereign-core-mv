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

//! Headstart M1 (#533): completion gates are enforced, fail closed, and
//! bound to a harness-minted aien-proof receipt.

use aien_proof::evidence::{self, Verdict};
use serde_json::{json, Value};
use spark_harness::completion::{
    artifact_digest, verify_completion, CompletionError, CompletionSubmission,
};
use spark_harness::{HarnessEngine, WorkflowState, DONE_STATE};
use std::fs;
use std::path::{Path, PathBuf};

const COMMIT: &str = "1111111111111111111111111111111111111111";
const OTHER_COMMIT: &str = "2222222222222222222222222222222222222222";

/// Temp copy of the shipped definitions, removed on drop.
struct Fixture {
    dir: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let dir =
            std::env::temp_dir().join(format!("spark-harness-gates-{}", uuid::Uuid::new_v4()));
        let src = Path::new(env!("CARGO_MANIFEST_DIR"));
        fs::create_dir_all(dir.join("schemas")).unwrap();
        for f in ["workflows.json", "evals.json"] {
            fs::copy(src.join(f), dir.join(f)).unwrap();
        }
        for entry in fs::read_dir(src.join("schemas")).unwrap() {
            let entry = entry.unwrap();
            fs::copy(entry.path(), dir.join("schemas").join(entry.file_name())).unwrap();
        }
        Fixture { dir }
    }

    fn engine(&self) -> HarnessEngine {
        HarnessEngine::load(&self.dir).expect("shipped definitions load")
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.dir);
    }
}

fn wf_at(workflow_type: &str, state: &str) -> WorkflowState {
    WorkflowState {
        id: format!("wf-{workflow_type}"),
        workflow_type: workflow_type.to_string(),
        current_state: state.to_string(),
        history: Vec::new(),
        created_at: "2026-10-15T00:00:00Z".to_string(),
    }
}

fn sub(task: &str, generation: u64, artifact: Value) -> CompletionSubmission {
    CompletionSubmission {
        task_id: task.to_string(),
        generation,
        source_commit: COMMIT.to_string(),
        artifact,
    }
}

fn engineering_fix_artifact() -> Value {
    json!({
        "bug": "Parser rejects valid token sequence",
        "reproduction_steps": ["Run parser test"],
        "failure_evidence": "thread panicked at line 42",
        "regression_test": "tests::test_parser",
        "diff_summary": "Fixed token parser logic",
        "retest_evidence": "All tests passing cleanly"
    })
}

fn research_handoff_artifact() -> Value {
    json!({
        "question": "Which serde_json feature changes map ordering?",
        "claims": [{
            "claim": "preserve_order switches Map to an insertion ordered map",
            "source_url": "https://docs.rs/serde_json",
            "excerpt": "preserve_order: use IndexMap instead of BTreeMap"
        }],
        "sources": ["https://docs.rs/serde_json"],
        "open_questions": [],
        "evidence": ["Cargo.lock serde_json entry lists no indexmap"]
    })
}

fn harness_change_artifact() -> Value {
    json!({
        "change": "Tighten the slop gate phrase list",
        "why": "Replay showed two missed tropes",
        "replay_report": "12 of 12 replayed failures now caught",
        "eval_results": ["slop_gate: pass"],
        "rollback": "Revert the evals.json commit",
        "evidence": ["replay log 2026-10-15"]
    })
}

fn python_result_artifact() -> Value {
    json!({
        "goal": "Sum a column",
        "code": "print(sum([1, 2, 3]))",
        "stdout": "6\n",
        "ok": "yes",
        "evidence": ["stdout matched expected 6"]
    })
}

fn receipt_count(engine: &HarnessEngine) -> usize {
    evidence::list_receipts(&engine.completion_store).len()
}

fn gate_failures(err: CompletionError) -> (String, Vec<String>) {
    match err {
        CompletionError::GatesFailed {
            receipt_id,
            failures,
        } => (receipt_id, failures),
        other => panic!("expected GatesFailed, got {other:?}"),
    }
}

// ---------- success for all four shipped workflows ----------

fn assert_completes(workflow_type: &str, pre_done: &str, artifact: Value) {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at(workflow_type, pre_done);
    let s = sub(&format!("task-{workflow_type}"), 1, artifact.clone());
    let rec = engine.complete_verified_workflow(&mut wf, &s).unwrap();
    assert_eq!(wf.current_state, DONE_STATE);
    assert_eq!(wf.history, vec![pre_done.to_string()]);
    assert_eq!(rec.artifact_digest, artifact_digest(&artifact));
    assert_eq!(receipt_count(&engine), 1);

    let receipt = evidence::load_receipt(&engine.completion_store, &rec.receipt_id).unwrap();
    assert_eq!(receipt.result, Verdict::Pass);
    assert_eq!(receipt.commit, COMMIT);
    assert_eq!(
        receipt.output_artifacts,
        vec![format!("blake3:{}", rec.artifact_digest)]
    );
    let ids: Vec<&str> = receipt.assertions.iter().map(|a| a.id.as_str()).collect();
    let def = &engine.workflows[workflow_type];
    assert!(ids.contains(&format!("schema:{}", def["required_schema"].as_str().unwrap()).as_str()));
    for e in def["required_evals_on_done"].as_array().unwrap() {
        assert!(ids.contains(&format!("eval:{}", e.as_str().unwrap()).as_str()));
    }

    let v = verify_completion(
        &engine.completion_store,
        &rec.receipt_id,
        &s.task_id,
        1,
        COMMIT,
        &rec.artifact_digest,
    )
    .unwrap();
    assert_eq!(v.workflow_id, wf.id);
    assert_eq!(v.generation, 1);
}

#[test]
fn engineering_fix_completes() {
    assert_completes("engineering_fix", "review", engineering_fix_artifact());
}

#[test]
fn research_handoff_completes() {
    assert_completes("research_handoff", "deliver", research_handoff_artifact());
}

#[test]
fn harness_change_completes() {
    assert_completes("harness_change", "verify", harness_change_artifact());
}

#[test]
fn python_pipeline_completes() {
    assert_completes("python_pipeline", "deliver", python_result_artifact());
}

// ---------- gate failures ----------

fn assert_fails_gate(
    engine: &HarnessEngine,
    workflow_type: &str,
    pre_done: &str,
    artifact: Value,
    failing: &str,
) {
    let mut wf = wf_at(workflow_type, pre_done);
    let before = receipt_count(engine);
    let err = engine
        .complete_verified_workflow(&mut wf, &sub("task-fail", 1, artifact.clone()))
        .unwrap_err();
    let (receipt_id, failures) = gate_failures(err);
    assert!(
        failures.iter().any(|f| f.starts_with(failing)),
        "expected {failing} in {failures:?}"
    );
    assert_eq!(wf.current_state, pre_done, "state must not move on failure");
    assert!(wf.history.is_empty());
    // The failure is recorded as a FAIL receipt, which never verifies.
    assert_eq!(receipt_count(engine), before + 1);
    let r = evidence::load_receipt(&engine.completion_store, &receipt_id).unwrap();
    assert_eq!(r.result, Verdict::Fail);
    let v = verify_completion(
        &engine.completion_store,
        &receipt_id,
        "task-fail",
        1,
        COMMIT,
        &artifact_digest(&artifact),
    );
    assert!(matches!(v, Err(CompletionError::NotPass(_))), "{v:?}");
    // A failure does not consume the generation: a fixed artifact may follow.
}

#[test]
fn schema_failure_blocks_done() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut art = engineering_fix_artifact();
    art.as_object_mut().unwrap().remove("retest_evidence");
    assert_fails_gate(
        &engine,
        "engineering_fix",
        "review",
        art,
        "schema:engineering_fix",
    );
}

#[test]
fn reproduction_present_failure_blocks_done() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut art = engineering_fix_artifact();
    // Passes the schema (minLength 8) but is blank to the eval.
    art["failure_evidence"] = json!("        ");
    assert!(engine.validate_schema("engineering_fix", &art).is_ok());
    assert_fails_gate(
        &engine,
        "engineering_fix",
        "review",
        art,
        "eval:reproduction_present",
    );
}

#[test]
fn required_evidence_failure_blocks_done() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut art = harness_change_artifact();
    art.as_object_mut().unwrap().remove("evidence");
    assert!(engine.validate_schema("harness_change", &art).is_ok());
    assert_fails_gate(
        &engine,
        "harness_change",
        "verify",
        art,
        "eval:required_evidence",
    );
}

#[test]
fn slop_gate_failure_blocks_done() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut art = research_handoff_artifact();
    art["claims"][0]["claim"] = json!("This finding is a game-changer for ordering");
    assert!(engine.validate_schema("research_handoff", &art).is_ok());
    assert_fails_gate(
        &engine,
        "research_handoff",
        "deliver",
        art,
        "eval:slop_gate",
    );
}

#[test]
fn unknown_eval_rule_fails_closed() {
    let fx = Fixture::new();
    let mut engine = fx.engine();
    engine.workflows.get_mut("python_pipeline").unwrap()["required_evals_on_done"] =
        json!(["required_evidence", "no_such_eval"]);
    assert_fails_gate(
        &engine,
        "python_pipeline",
        "deliver",
        python_result_artifact(),
        "eval:no_such_eval",
    );
}

#[test]
fn judgment_eval_cannot_gate_completion() {
    let fx = Fixture::new();
    let mut engine = fx.engine();
    engine.workflows.get_mut("python_pipeline").unwrap()["required_evals_on_done"] =
        json!(["source_supports_claim"]);
    assert_fails_gate(
        &engine,
        "python_pipeline",
        "deliver",
        python_result_artifact(),
        "eval:source_supports_claim",
    );
}

#[test]
fn eval_without_definition_fails_closed() {
    let fx = Fixture::new();
    let mut engine = fx.engine();
    engine.evals.remove("required_evidence");
    assert_fails_gate(
        &engine,
        "python_pipeline",
        "deliver",
        python_result_artifact(),
        "eval:required_evidence",
    );
}

#[test]
fn missing_schema_file_fails_closed() {
    let fx = Fixture::new();
    fs::remove_file(fx.dir.join("schemas/engineering_fix.json")).unwrap();
    let engine = fx.engine();
    assert_fails_gate(
        &engine,
        "engineering_fix",
        "review",
        engineering_fix_artifact(),
        "schema:engineering_fix",
    );
}

#[test]
fn invalid_schema_file_fails_closed() {
    let fx = Fixture::new();
    fs::write(fx.dir.join("schemas/python_result.json"), "{ not json").unwrap();
    let engine = fx.engine();
    assert_fails_gate(
        &engine,
        "python_pipeline",
        "deliver",
        python_result_artifact(),
        "schema:python_result",
    );
    fs::write(fx.dir.join("schemas/python_result.json"), r#"{"type": 12}"#).unwrap();
    assert_fails_gate(
        &engine,
        "python_pipeline",
        "deliver",
        python_result_artifact(),
        "schema:python_result",
    );
}

// ---------- fail-closed loading ----------

#[test]
fn malformed_workflows_json_blocks_completion() {
    let fx = Fixture::new();
    fs::write(fx.dir.join("workflows.json"), "{ \"engineering_fix\": ").unwrap();
    assert!(HarnessEngine::load(&fx.dir).is_err());
    let engine = HarnessEngine::new(&fx.dir);
    assert!(!engine.load_errors.is_empty());
    let mut wf = wf_at("engineering_fix", "review");
    let err = engine
        .complete_verified_workflow(&mut wf, &sub("t", 1, engineering_fix_artifact()))
        .unwrap_err();
    assert!(
        matches!(err, CompletionError::DefinitionsNotLoaded(_)),
        "{err:?}"
    );
    assert_eq!(wf.current_state, "review");
    assert_eq!(receipt_count(&engine), 0);
}

#[test]
fn structurally_bad_workflow_rejected_at_load() {
    let fx = Fixture::new();
    let mut defs: Value =
        serde_json::from_str(&fs::read_to_string(fx.dir.join("workflows.json")).unwrap()).unwrap();
    defs["engineering_fix"]
        .as_object_mut()
        .unwrap()
        .remove("required_schema");
    fs::write(fx.dir.join("workflows.json"), defs.to_string()).unwrap();
    let err = HarnessEngine::load(&fx.dir).unwrap_err();
    assert!(err.contains("required_schema"), "{err}");
}

#[test]
fn missing_evals_json_blocks_completion() {
    let fx = Fixture::new();
    fs::remove_file(fx.dir.join("evals.json")).unwrap();
    assert!(HarnessEngine::load(&fx.dir).is_err());
    let engine = HarnessEngine::new(&fx.dir);
    let mut wf = wf_at("python_pipeline", "deliver");
    let err = engine
        .complete_verified_workflow(&mut wf, &sub("t", 1, python_result_artifact()))
        .unwrap_err();
    assert!(
        matches!(err, CompletionError::DefinitionsNotLoaded(_)),
        "{err:?}"
    );
}

// ---------- forged success flags ----------

#[test]
fn forged_success_flag_in_artifact_is_not_proof() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut art = python_result_artifact();
    art["evidence"] = json!([]);
    art["passed"] = json!(true);
    art["verified"] = json!(true);
    art["receipt"] = json!({"result": "PASS"});
    assert_fails_gate(
        &engine,
        "python_pipeline",
        "deliver",
        art,
        "schema:python_result",
    );
}

#[test]
fn forged_fields_in_submission_are_rejected() {
    for extra in ["passed", "receipt", "verdict"] {
        let mut raw = json!({
            "task_id": "t",
            "generation": 1,
            "source_commit": COMMIT,
            "artifact": python_result_artifact()
        });
        raw[extra] = json!(true);
        assert!(
            serde_json::from_value::<CompletionSubmission>(raw).is_err(),
            "{extra}"
        );
    }
    // Same through the MCP adapter, which must answer with an explicit error.
    let fx = Fixture::new();
    let engine = fx.engine();
    let args = json!({
        "workflow": wf_at("python_pipeline", "deliver"),
        "submission": {
            "task_id": "t", "generation": 1, "source_commit": COMMIT,
            "artifact": python_result_artifact(), "passed": true
        }
    });
    let err = engine.handle_complete_request(&args).unwrap_err();
    assert!(matches!(err, CompletionError::InvalidSubmission(_)));
    assert_eq!(err.rpc_error()["code"], json!(-32602));
    assert_eq!(receipt_count(&engine), 0);
}

#[test]
fn bad_commit_is_rejected_before_evaluation() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("python_pipeline", "deliver");
    let mut s = sub("t", 1, python_result_artifact());
    s.source_commit = "HEAD".to_string();
    let err = engine.complete_verified_workflow(&mut wf, &s).unwrap_err();
    assert!(
        matches!(err, CompletionError::InvalidSubmission(_)),
        "{err:?}"
    );
    assert_eq!(receipt_count(&engine), 0);
}

// ---------- state progression ----------

#[test]
fn advance_workflow_to_done_is_refused() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("engineering_fix", "review");
    let err = engine.advance_workflow(&mut wf, DONE_STATE).unwrap_err();
    assert!(err.contains("complete_verified_workflow"), "{err}");
    assert_eq!(wf.current_state, "review");
    assert!(wf.history.is_empty());
    // Other legal transitions still work.
    engine.advance_workflow(&mut wf, "implement").unwrap();
    assert_eq!(wf.current_state, "implement");
}

#[test]
fn completion_from_state_that_cannot_reach_done() {
    let fx = Fixture::new();
    let engine = fx.engine();
    for state in ["implement", "nonexistent_state"] {
        let mut wf = wf_at("engineering_fix", state);
        let err = engine
            .complete_verified_workflow(&mut wf, &sub("t", 1, engineering_fix_artifact()))
            .unwrap_err();
        assert!(
            matches!(err, CompletionError::IllegalTransition { .. }),
            "{err:?}"
        );
        assert_eq!(wf.current_state, state);
    }
    assert_eq!(receipt_count(&engine), 0);
}

#[test]
fn unknown_workflow_type_is_refused() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("no_such_flow", "review");
    let err = engine
        .complete_verified_workflow(&mut wf, &sub("t", 1, engineering_fix_artifact()))
        .unwrap_err();
    assert!(
        matches!(err, CompletionError::UnknownWorkflow(_)),
        "{err:?}"
    );
}

// ---------- duplicates, replay, generations ----------

#[test]
fn duplicate_completion_on_done_workflow_is_refused() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("engineering_fix", "review");
    let s = sub("dup", 1, engineering_fix_artifact());
    engine.complete_verified_workflow(&mut wf, &s).unwrap();
    let err = engine.complete_verified_workflow(&mut wf, &s).unwrap_err();
    assert!(
        matches!(err, CompletionError::IllegalTransition { .. }),
        "{err:?}"
    );
    assert_eq!(receipt_count(&engine), 1);
}

#[test]
fn replayed_submission_mints_no_second_receipt() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let snapshot = wf_at("engineering_fix", "review");
    let s = sub("replay", 1, engineering_fix_artifact());
    let mut first = snapshot.clone();
    let rec = engine.complete_verified_workflow(&mut first, &s).unwrap();

    // Replay with a stale copy of the workflow that still says "review".
    let mut replay = snapshot.clone();
    let err = engine
        .complete_verified_workflow(&mut replay, &s)
        .unwrap_err();
    match err {
        CompletionError::DuplicateCompletion {
            receipt_id,
            generation,
            ..
        } => {
            assert_eq!(receipt_id, rec.receipt_id);
            assert_eq!(generation, 1);
        }
        other => panic!("expected DuplicateCompletion, got {other:?}"),
    }
    assert_eq!(replay.current_state, "review");
    assert_eq!(
        evidence::list_receipts(&engine.completion_store),
        vec![rec.receipt_id.clone()]
    );

    // A fresh engine on the same store sees the durable record too.
    let engine2 = fx.engine();
    let mut replay2 = snapshot.clone();
    let err = engine2
        .complete_verified_workflow(&mut replay2, &s)
        .unwrap_err();
    assert!(
        matches!(err, CompletionError::DuplicateCompletion { .. }),
        "{err:?}"
    );
    assert_eq!(receipt_count(&engine2), 1);

    // Through MCP the replay is an explicit JSON-RPC error.
    let args = json!({ "workflow": snapshot, "submission": s });
    let rpc = engine2
        .handle_complete_request(&args)
        .unwrap_err()
        .rpc_error();
    assert_eq!(rpc["code"], json!(-32005));
    assert_eq!(rpc["data"]["kind"], json!("duplicate_completion"));
}

#[test]
fn stale_and_newer_generations() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("engineering_fix", "review");
    engine
        .complete_verified_workflow(&mut wf, &sub("gen", 5, engineering_fix_artifact()))
        .unwrap();

    let mut lower = wf_at("engineering_fix", "review");
    let err = engine
        .complete_verified_workflow(&mut lower, &sub("gen", 3, engineering_fix_artifact()))
        .unwrap_err();
    assert_eq!(
        err,
        CompletionError::StaleGeneration {
            task_id: "gen".to_string(),
            submitted: 3,
            recorded: 5
        }
    );
    assert_eq!(receipt_count(&engine), 1);

    // A newer generation (reworked task) is a new completion.
    let mut newer = wf_at("engineering_fix", "review");
    let rec = engine
        .complete_verified_workflow(&mut newer, &sub("gen", 6, engineering_fix_artifact()))
        .unwrap();
    assert_eq!(receipt_count(&engine), 2);
    let mut back = wf_at("engineering_fix", "review");
    assert!(matches!(
        engine.complete_verified_workflow(&mut back, &sub("gen", 5, engineering_fix_artifact())),
        Err(CompletionError::StaleGeneration { recorded: 6, .. })
    ));
    // The receipt for generation 6 does not verify as generation 5.
    assert!(matches!(
        verify_completion(
            &engine.completion_store,
            &rec.receipt_id,
            "gen",
            5,
            COMMIT,
            &rec.artifact_digest
        ),
        Err(CompletionError::BindingMismatch {
            field: "generation",
            ..
        })
    ));
}

#[test]
fn corrupt_consumed_record_fails_closed() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("engineering_fix", "review");
    engine
        .complete_verified_workflow(&mut wf, &sub("corrupt", 1, engineering_fix_artifact()))
        .unwrap();
    let dir = engine.completion_store.join("completions");
    for entry in fs::read_dir(&dir).unwrap() {
        let p = entry.unwrap().path();
        if p.extension().is_some_and(|e| e == "json") {
            fs::write(&p, "garbage").unwrap();
        }
    }
    let mut again = wf_at("engineering_fix", "review");
    let err = engine
        .complete_verified_workflow(&mut again, &sub("corrupt", 2, engineering_fix_artifact()))
        .unwrap_err();
    assert!(matches!(err, CompletionError::Store(_)), "{err:?}");
    assert_eq!(again.current_state, "review");
}

// ---------- verify_completion bindings ----------

fn completed() -> (Fixture, HarnessEngine, String, String) {
    let fx = Fixture::new();
    let engine = fx.engine();
    let mut wf = wf_at("python_pipeline", "deliver");
    let rec = engine
        .complete_verified_workflow(&mut wf, &sub("verify-me", 7, python_result_artifact()))
        .unwrap();
    (fx, engine, rec.receipt_id, rec.artifact_digest)
}

#[test]
fn verify_completion_accepts_exact_bindings() {
    let (_fx, engine, id, digest) = completed();
    let v = verify_completion(
        &engine.completion_store,
        &id,
        "verify-me",
        7,
        COMMIT,
        &digest,
    )
    .unwrap();
    assert_eq!(v.receipt_id, id);
    let prefixed = format!("blake3:{digest}");
    assert!(verify_completion(
        &engine.completion_store,
        &id,
        "verify-me",
        7,
        COMMIT,
        &prefixed
    )
    .is_ok());
}

#[test]
fn verify_completion_wrong_digest() {
    let (_fx, engine, id, _) = completed();
    let mut other = python_result_artifact();
    other["stdout"] = json!("7\n");
    let err = verify_completion(
        &engine.completion_store,
        &id,
        "verify-me",
        7,
        COMMIT,
        &artifact_digest(&other),
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            CompletionError::BindingMismatch {
                field: "artifact_digest",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn verify_completion_wrong_commit() {
    let (_fx, engine, id, digest) = completed();
    let err = verify_completion(
        &engine.completion_store,
        &id,
        "verify-me",
        7,
        OTHER_COMMIT,
        &digest,
    )
    .unwrap_err();
    assert!(
        matches!(
            err,
            CompletionError::BindingMismatch {
                field: "commit",
                ..
            }
        ),
        "{err:?}"
    );
}

#[test]
fn verify_completion_wrong_task() {
    let (_fx, engine, id, digest) = completed();
    let err = verify_completion(
        &engine.completion_store,
        &id,
        "someone-else",
        7,
        COMMIT,
        &digest,
    )
    .unwrap_err();
    assert!(
        matches!(err, CompletionError::BindingMismatch { field: "task", .. }),
        "{err:?}"
    );
}

#[test]
fn verify_completion_missing_receipt() {
    let (_fx, engine, _, digest) = completed();
    let err = verify_completion(
        &engine.completion_store,
        &"a".repeat(64),
        "verify-me",
        7,
        COMMIT,
        &digest,
    )
    .unwrap_err();
    assert!(matches!(err, CompletionError::ReceiptInvalid(_)), "{err:?}");
}

#[test]
fn verify_completion_tampered_receipt_file() {
    let (_fx, engine, id, digest) = completed();
    let path = engine
        .completion_store
        .join(evidence::RECEIPT_DIR)
        .join(format!("{id}.json"));
    let mut r: Value = serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
    // Rebind the receipt to another task without re-minting it.
    r["external_refs"] = json!([
        "task:someone-else",
        "workflow:wf-python_pipeline",
        "generation:7"
    ]);
    fs::write(&path, serde_json::to_string_pretty(&r).unwrap()).unwrap();
    let err = verify_completion(
        &engine.completion_store,
        &id,
        "someone-else",
        7,
        COMMIT,
        &digest,
    )
    .unwrap_err();
    assert!(matches!(err, CompletionError::ReceiptInvalid(_)), "{err:?}");
}

#[test]
fn mcp_success_returns_record_and_done_state() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let args = json!({
        "workflow": wf_at("harness_change", "verify"),
        "submission": sub("mcp-task", 1, harness_change_artifact())
    });
    let out = engine.handle_complete_request(&args).unwrap();
    assert_eq!(out["workflow"]["current_state"], json!(DONE_STATE));
    let id = out["record"]["receipt_id"].as_str().unwrap();
    let digest = out["record"]["artifact_digest"].as_str().unwrap();
    assert!(verify_completion(&engine.completion_store, id, "mcp-task", 1, COMMIT, digest).is_ok());
}

// ---------- consumed record is the authority (inspector finding 1) ----------

fn consumed_records(engine: &HarnessEngine) -> Vec<PathBuf> {
    let dir = engine.completion_store.join("completions");
    fs::read_dir(&dir)
        .map(|it| {
            it.flatten()
                .map(|e| e.path())
                .filter(|p| p.extension().is_some_and(|e| e == "json"))
                .collect()
        })
        .unwrap_or_default()
}

fn assert_not_current(r: Result<spark_harness::completion::VerifiedCompletion, CompletionError>) {
    match r {
        Err(CompletionError::NotCurrentCompletion(_)) => {}
        other => panic!("expected NotCurrentCompletion, got {other:?}"),
    }
}

/// Real write failure: the consumed-record directory is read-only, so
/// write_consumed fails after store_receipt succeeded.
#[test]
fn orphan_receipt_from_failed_record_write_never_verifies() {
    use std::os::unix::fs::PermissionsExt;
    let fx = Fixture::new();
    let mut engine = fx.engine();
    engine.completion_clock = || 1_000;
    let dir = engine.completion_store.join("completions");
    fs::create_dir_all(&dir).unwrap();
    fs::File::create(dir.join(".lock")).unwrap();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o555)).unwrap();
    let probe = dir.join("probe");
    if fs::File::create(&probe).is_ok() {
        // Running with permission override (root): the injection cannot work.
        let _ = fs::remove_file(&probe);
        fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
        eprintln!("skipping permission injection: directory stayed writable");
        return;
    }

    let art_a = python_result_artifact();
    let mut wf = wf_at("python_pipeline", "deliver");
    let err = engine
        .complete_verified_workflow(&mut wf, &sub("orphan", 1, art_a.clone()))
        .unwrap_err();
    fs::set_permissions(&dir, fs::Permissions::from_mode(0o755)).unwrap();
    assert!(matches!(err, CompletionError::Store(_)), "{err:?}");
    assert_eq!(
        wf.current_state, "deliver",
        "no done without a consumed record"
    );
    let ids = evidence::list_receipts(&engine.completion_store);
    assert_eq!(ids.len(), 1);
    let orphan = ids[0].clone();
    let r = evidence::load_receipt(&engine.completion_store, &orphan).unwrap();
    assert_eq!(
        r.result,
        Verdict::Pass,
        "the orphan is a well-formed PASS receipt"
    );
    assert_not_current(verify_completion(
        &engine.completion_store,
        &orphan,
        "orphan",
        1,
        COMMIT,
        &artifact_digest(&art_a),
    ));

    // Retry: the SAME task, generation, commit and artifact, one second
    // later. Every binding the verifier checks is identical for both
    // receipts; only the timestamp (and therefore the ID) differs.
    engine.completion_clock = || 1_001;
    let mut wf2 = wf_at("python_pipeline", "deliver");
    let rec = engine
        .complete_verified_workflow(&mut wf2, &sub("orphan", 1, art_a.clone()))
        .unwrap();
    assert_eq!(wf2.current_state, "done");
    assert_ne!(rec.receipt_id, orphan, "retry must mint a distinct receipt");
    assert_eq!(receipt_count(&engine), 2);
    // Exactly one of the two PASS receipts verifies: the receipt-id match
    // against the consumed record is the only thing that separates them.
    assert_not_current(verify_completion(
        &engine.completion_store,
        &orphan,
        "orphan",
        1,
        COMMIT,
        &artifact_digest(&art_a),
    ));
    assert!(verify_completion(
        &engine.completion_store,
        &rec.receipt_id,
        "orphan",
        1,
        COMMIT,
        &artifact_digest(&art_a)
    )
    .is_ok());
}

/// Simulated write failure: the consumed record vanishes after a completion.
#[test]
fn receipt_without_consumed_record_never_verifies() {
    let (_fx, engine, id, digest) = completed();
    for p in consumed_records(&engine) {
        fs::remove_file(p).unwrap();
    }
    assert_not_current(verify_completion(
        &engine.completion_store,
        &id,
        "verify-me",
        7,
        COMMIT,
        &digest,
    ));
}

#[test]
fn corrupt_consumed_record_blocks_verification() {
    let (_fx, engine, id, digest) = completed();
    for p in consumed_records(&engine) {
        fs::write(p, "garbage").unwrap();
    }
    let err = verify_completion(
        &engine.completion_store,
        &id,
        "verify-me",
        7,
        COMMIT,
        &digest,
    )
    .unwrap_err();
    assert!(matches!(err, CompletionError::Store(_)), "{err:?}");
}

#[test]
fn superseded_generation_stops_verifying() {
    let fx = Fixture::new();
    let engine = fx.engine();
    let art = engineering_fix_artifact();
    let mut wf1 = wf_at("engineering_fix", "review");
    let r1 = engine
        .complete_verified_workflow(&mut wf1, &sub("super", 1, art.clone()))
        .unwrap();
    assert!(verify_completion(
        &engine.completion_store,
        &r1.receipt_id,
        "super",
        1,
        COMMIT,
        &r1.artifact_digest
    )
    .is_ok());
    let mut wf2 = wf_at("engineering_fix", "review");
    let r2 = engine
        .complete_verified_workflow(&mut wf2, &sub("super", 2, art))
        .unwrap();
    assert_not_current(verify_completion(
        &engine.completion_store,
        &r1.receipt_id,
        "super",
        1,
        COMMIT,
        &r1.artifact_digest,
    ));
    assert!(verify_completion(
        &engine.completion_store,
        &r2.receipt_id,
        "super",
        2,
        COMMIT,
        &r2.artifact_digest
    )
    .is_ok());
}

// ---------- deterministic-kind guard (inspector finding 3) ----------

#[test]
fn executable_eval_marked_judgment_is_refused() {
    let fx = Fixture::new();
    let mut engine = fx.engine();
    // run_eval can execute required_evidence and would pass this artifact;
    // only the kind guard can refuse it.
    let art = python_result_artifact();
    assert!(engine.run_eval("required_evidence", &art).passed);
    engine.evals.get_mut("required_evidence").unwrap()["kind"] = json!("judgment");
    let mut wf = wf_at("python_pipeline", "deliver");
    let err = engine
        .complete_verified_workflow(&mut wf, &sub("kind", 1, art))
        .unwrap_err();
    let (_, failures) = gate_failures(err);
    assert!(
        failures
            .iter()
            .any(|f| f.starts_with("eval:required_evidence") && f.contains("judgment")),
        "{failures:?}"
    );
    assert_eq!(wf.current_state, "deliver");
}
