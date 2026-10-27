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

//! Verified workflow completion (Headstart M1, issue #533).
//!
//! [`HarnessEngine::complete_verified_workflow`] is the only way a workflow
//! reaches [`DONE_STATE`]. The harness itself:
//!
//! 1. refuses to run when workflows.json or evals.json did not load cleanly;
//! 2. checks that the workflow's current state may legally move to `done`;
//! 3. validates the workflow's `required_schema` against the submitted
//!    artifact and runs every `required_evals_on_done` entry on it (unknown
//!    eval, missing definition, non-deterministic eval kind, missing or
//!    invalid schema all count as failures);
//! 4. mints an aien-proof `EvidenceReceiptV1` from its own results (one
//!    assertion per gate, artifact digest, source commit, task, workflow and
//!    generation bindings) and stores it in its own store;
//! 5. records the consumed (task, generation) pair atomically, and only then
//!    moves the workflow to `done`.
//!
//! The caller never supplies a receipt, a verdict or a pass flag:
//! [`CompletionSubmission`] rejects unknown fields, and any extra field in
//! the artifact is judged by the workflow schema like any other content.
//!
//! What `done` means: the completion gates passed on this artifact and a PASS
//! receipt is in the harness store. It does not mean committed. World commit
//! and authorization stay in aien-runtime and are not touched here.
//!
//! Limitations, stated plainly:
//! - EvidenceReceiptV1 is unsigned (aien-proof `evidence.rs`, `Receipt` has no
//!   signature field; its ID is a BLAKE3 hash of the canonical encoding).
//!   Anyone who can write to the completion store can write a receipt with a
//!   matching ID. Authenticity therefore rests on store ownership (filesystem
//!   trust) plus recomputation: [`verify_completion`] recomputes the receipt
//!   ID and checks every binding, and a verifier holding the artifact can
//!   recompute [`artifact_digest`] and rerun the gates.
//! - `source_commit` is the caller-declared revision. The harness checks its
//!   format (aien-proof `check_commit`) but does not inspect a source tree, so
//!   `dirty` is written as `false` meaning "no tree inspection was made", not
//!   "the tree was verified clean".
//! - The tier is `HOST_TEST`: deterministic checks executed on the host, no
//!   hardware and no physical qualification.
//! - If writing the consumed record fails (or the process dies) after the
//!   receipt is stored, an orphan PASS receipt stays in the store and a
//!   resubmission can mint another one for the same generation. The workflow
//!   never moves to `done` in that case, and [`verify_completion`] accepts
//!   only the receipt the consumed record names, so at most one receipt per
//!   task verifies at any time.
//! - One exception to "error means nothing verifies": if the record rename
//!   succeeds but the following directory fsync fails, completion returns a
//!   `Store` error and the workflow is not moved to `done`, yet the record is
//!   already in place and the receipt it names verifies. The rename is not
//!   guaranteed durable across a power loss in that case.
//! - The consumed record is keyed by task and holds only the highest
//!   completed generation. Completing a newer generation supersedes the older
//!   one: its receipt stops verifying. Verification answers "is this the
//!   task's current completion", which is what M2 needs.

use crate::{HarnessEngine, WorkflowState, DONE_STATE};
use aien_proof::evidence::{self, Assertion, Mutation, Receipt, Tier, Verdict};
use aien_proof::lock::FileLock;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::fmt;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};

/// Receipt `kind` for harness completion receipts.
pub const COMPLETION_KIND: &str = "spark-harness-completion";
/// Honest tier for harness completions: host-executed deterministic checks.
pub const COMPLETION_TIER: Tier = Tier::HostTest;
/// Subdirectory of the completion store holding consumed completion records.
pub const CONSUMED_DIR: &str = "completions";
const MAX_TASK_ID_LEN: usize = 256;
const ASSERTION_SOURCE: &str = "spark-harness";

/// What a caller submits for completion. There is deliberately no field for
/// a receipt, verdict or pass flag; unknown fields are rejected.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CompletionSubmission {
    pub task_id: String,
    /// Monotonic generation of the work for this task. A completed
    /// generation can never be completed again, and a lower one is stale.
    pub generation: u64,
    /// Source revision the artifact was produced from: 40 or 64 lowercase hex.
    pub source_commit: String,
    pub artifact: Value,
}

/// Result of a successful verified completion.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CompletionRecord {
    pub receipt_id: String,
    pub task_id: String,
    pub workflow_id: String,
    pub workflow_type: String,
    pub generation: u64,
    pub source_commit: String,
    /// BLAKE3 hex of [`canonical_json`] of the artifact.
    pub artifact_digest: String,
}

/// Durable record of the highest completed generation for one task.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ConsumedRecord {
    task_id: String,
    generation: u64,
    receipt_id: String,
    workflow_id: String,
    artifact_digest: String,
}

/// A completion receipt that passed [`verify_completion`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifiedCompletion {
    pub receipt_id: String,
    pub task_id: String,
    pub workflow_id: String,
    pub generation: u64,
    pub source_commit: String,
    pub artifact_digest: String,
}

/// Every way completion or verification can refuse.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CompletionError {
    /// workflows.json or evals.json failed to load; nothing can complete.
    DefinitionsNotLoaded(Vec<String>),
    /// Submission or request arguments are malformed.
    InvalidSubmission(String),
    UnknownWorkflow(String),
    MalformedWorkflow(String),
    /// The current state cannot legally move to `done`.
    IllegalTransition {
        from: String,
        allowed: Vec<String>,
    },
    /// This task and generation already completed; no new receipt minted.
    DuplicateCompletion {
        task_id: String,
        generation: u64,
        receipt_id: String,
    },
    /// The submitted generation is lower than the recorded one.
    StaleGeneration {
        task_id: String,
        submitted: u64,
        recorded: u64,
    },
    /// At least one gate failed. A FAIL receipt was stored; state unchanged.
    GatesFailed {
        receipt_id: String,
        failures: Vec<String>,
    },
    /// The completion store could not be read or written.
    Store(String),
    /// The receipt is missing, malformed or its identity does not recompute.
    ReceiptInvalid(String),
    /// The receipt does not carry a PASS verdict.
    NotPass(String),
    /// The receipt is valid but bound to something other than expected.
    BindingMismatch {
        field: &'static str,
        expected: String,
        found: String,
    },
    /// The receipt is not the task's current completion per the consumed
    /// record (orphan receipt, superseded generation, or no record).
    NotCurrentCompletion(String),
}

impl CompletionError {
    /// Stable machine-readable name of the refusal.
    pub fn kind(&self) -> &'static str {
        match self {
            CompletionError::DefinitionsNotLoaded(_) => "definitions_not_loaded",
            CompletionError::InvalidSubmission(_) => "invalid_submission",
            CompletionError::UnknownWorkflow(_) => "unknown_workflow",
            CompletionError::MalformedWorkflow(_) => "malformed_workflow",
            CompletionError::IllegalTransition { .. } => "illegal_transition",
            CompletionError::DuplicateCompletion { .. } => "duplicate_completion",
            CompletionError::StaleGeneration { .. } => "stale_generation",
            CompletionError::GatesFailed { .. } => "gates_failed",
            CompletionError::Store(_) => "store_error",
            CompletionError::ReceiptInvalid(_) => "receipt_invalid",
            CompletionError::NotPass(_) => "not_pass",
            CompletionError::BindingMismatch { .. } => "binding_mismatch",
            CompletionError::NotCurrentCompletion(_) => "not_current_completion",
        }
    }

    /// JSON-RPC error code. Invalid arguments use the standard -32602; every
    /// other refusal uses the server-defined range.
    pub fn rpc_code(&self) -> i64 {
        match self {
            CompletionError::InvalidSubmission(_) => -32602,
            CompletionError::DefinitionsNotLoaded(_) => -32001,
            CompletionError::UnknownWorkflow(_) => -32002,
            CompletionError::MalformedWorkflow(_) => -32003,
            CompletionError::IllegalTransition { .. } => -32004,
            CompletionError::DuplicateCompletion { .. } => -32005,
            CompletionError::StaleGeneration { .. } => -32006,
            CompletionError::GatesFailed { .. } => -32007,
            CompletionError::Store(_) => -32008,
            CompletionError::ReceiptInvalid(_) => -32009,
            CompletionError::NotPass(_) => -32010,
            CompletionError::BindingMismatch { .. } => -32011,
            CompletionError::NotCurrentCompletion(_) => -32012,
        }
    }

    /// JSON-RPC `error` object for an MCP refusal.
    pub fn rpc_error(&self) -> Value {
        let mut data = json!({ "kind": self.kind() });
        match self {
            CompletionError::GatesFailed {
                receipt_id,
                failures,
            } => {
                data["receipt_id"] = json!(receipt_id);
                data["failures"] = json!(failures);
            }
            CompletionError::DuplicateCompletion { receipt_id, .. } => {
                data["receipt_id"] = json!(receipt_id);
            }
            _ => {}
        }
        json!({ "code": self.rpc_code(), "message": self.to_string(), "data": data })
    }
}

impl fmt::Display for CompletionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompletionError::DefinitionsNotLoaded(errs) => write!(
                f,
                "workflow/eval definitions did not load, completion refused: {}",
                errs.join("; ")
            ),
            CompletionError::InvalidSubmission(m) => write!(f, "invalid submission: {m}"),
            CompletionError::UnknownWorkflow(w) => write!(f, "unknown workflow type '{w}'"),
            CompletionError::MalformedWorkflow(m) => write!(f, "malformed workflow: {m}"),
            CompletionError::IllegalTransition { from, allowed } => write!(
                f,
                "state '{from}' cannot move to '{DONE_STATE}'; allowed next states: {allowed:?}"
            ),
            CompletionError::DuplicateCompletion {
                task_id,
                generation,
                receipt_id,
            } => write!(
                f,
                "task '{task_id}' generation {generation} already completed (receipt {receipt_id})"
            ),
            CompletionError::StaleGeneration {
                task_id,
                submitted,
                recorded,
            } => write!(
                f,
                "stale generation for task '{task_id}': submitted {submitted}, completed {recorded}"
            ),
            CompletionError::GatesFailed {
                receipt_id,
                failures,
            } => write!(
                f,
                "completion gates failed (FAIL receipt {receipt_id}): {}",
                failures.join("; ")
            ),
            CompletionError::Store(m) => write!(f, "completion store error: {m}"),
            CompletionError::ReceiptInvalid(m) => write!(f, "receipt invalid: {m}"),
            CompletionError::NotPass(m) => write!(f, "receipt is not a PASS: {m}"),
            CompletionError::NotCurrentCompletion(m) => {
                write!(f, "receipt is not the current completion: {m}")
            }
            CompletionError::BindingMismatch {
                field,
                expected,
                found,
            } => write!(
                f,
                "receipt binding '{field}' mismatch: expected {expected:?}, found {found:?}"
            ),
        }
    }
}

impl std::error::Error for CompletionError {}

/// Canonical JSON encoding used for the artifact digest: object keys sorted
/// by byte order at every level, no whitespace, arrays in order, scalars as
/// serde_json prints them. Keys are sorted here explicitly, so the digest
/// does not depend on whether serde_json's `preserve_order` feature is on.
pub fn canonical_json(v: &Value) -> Vec<u8> {
    let mut out = Vec::new();
    write_canonical(v, &mut out);
    out
}

fn write_canonical(v: &Value, out: &mut Vec<u8>) {
    match v {
        Value::Array(items) => {
            out.push(b'[');
            for (i, item) in items.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                write_canonical(item, out);
            }
            out.push(b']');
        }
        Value::Object(map) => {
            let mut keys: Vec<&String> = map.keys().collect();
            keys.sort();
            out.push(b'{');
            for (i, k) in keys.iter().enumerate() {
                if i > 0 {
                    out.push(b',');
                }
                // Serializing a string or scalar Value cannot fail.
                out.extend_from_slice(serde_json::to_string(k).unwrap_or_default().as_bytes());
                out.push(b':');
                write_canonical(&map[k.as_str()], out);
            }
            out.push(b'}');
        }
        scalar => {
            out.extend_from_slice(serde_json::to_string(scalar).unwrap_or_default().as_bytes())
        }
    }
}

/// BLAKE3 hex digest of [`canonical_json`] of the artifact.
pub fn artifact_digest(artifact: &Value) -> String {
    blake3::hash(&canonical_json(artifact)).to_hex().to_string()
}

/// File name key for a task's consumed record (task IDs are free text).
pub(crate) fn task_key(task_id: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(b"spark-harness-task\0");
    h.update(task_id.as_bytes());
    h.finalize().to_hex().to_string()
}

pub(crate) fn check_identifier(what: &str, value: &str) -> Result<(), CompletionError> {
    if value.is_empty() || value.trim() != value {
        return Err(CompletionError::InvalidSubmission(format!(
            "{what} must be non-empty with no leading or trailing whitespace"
        )));
    }
    if value.len() > MAX_TASK_ID_LEN || value.chars().any(char::is_control) {
        return Err(CompletionError::InvalidSubmission(format!(
            "{what} must be at most {MAX_TASK_ID_LEN} bytes with no control characters"
        )));
    }
    Ok(())
}

fn collect_strings<'a>(v: &'a Value, path: String, out: &mut Vec<(String, &'a str)>) {
    match v {
        Value::String(s) => out.push((path, s)),
        Value::Array(items) => {
            for (i, item) in items.iter().enumerate() {
                collect_strings(item, format!("{path}/{i}"), out);
            }
        }
        Value::Object(map) => {
            for (k, item) in map {
                collect_strings(item, format!("{path}/{k}"), out);
            }
        }
        _ => {}
    }
}

pub(crate) fn host_identity() -> String {
    let from_proc = fs::read_to_string("/proc/sys/kernel/hostname")
        .map(|s| s.trim().to_string())
        .unwrap_or_default();
    if !from_proc.is_empty() {
        return from_proc;
    }
    std::env::var("HOSTNAME")
        .ok()
        .filter(|s| !s.trim().is_empty())
        .unwrap_or_else(|| "unknown-host".to_string())
}

fn read_consumed(path: &Path, task_id: &str) -> Result<Option<ConsumedRecord>, CompletionError> {
    let bytes = match fs::read(path) {
        Ok(b) => b,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(None),
        Err(e) => {
            return Err(CompletionError::Store(format!(
                "cannot read consumed record {}: {e}",
                path.display()
            )))
        }
    };
    let rec: ConsumedRecord = serde_json::from_slice(&bytes).map_err(|e| {
        CompletionError::Store(format!(
            "consumed record {} is malformed, refusing: {e}",
            path.display()
        ))
    })?;
    if rec.task_id != task_id {
        return Err(CompletionError::Store(format!(
            "consumed record {} belongs to task '{}', not '{task_id}'",
            path.display(),
            rec.task_id
        )));
    }
    Ok(Some(rec))
}

/// Write temp file, fsync, rename over the target, fsync the directory.
///
/// Known edge: if the rename succeeds but the directory fsync fails, this
/// returns `Store` and the workflow stays un-done, yet the new record is
/// already in place, so the just-minted receipt verifies (the rename may
/// still be lost on power failure). A resubmission of the same generation
/// is then refused as a duplicate.
fn write_consumed(path: &Path, rec: &ConsumedRecord) -> Result<(), CompletionError> {
    let text = serde_json::to_vec_pretty(rec).map_err(|e| CompletionError::Store(e.to_string()))?;
    write_atomic(path, &text).map_err(|e| {
        CompletionError::Store(format!(
            "cannot write consumed record {}: {e}",
            path.display()
        ))
    })
}

/// Atomic replace used by the consumed and interface records: write a temp
/// file in the same directory, append a newline, fsync it, rename it over
/// `path`, fsync the directory. The temp file is removed on failure.
pub(crate) fn write_atomic(path: &Path, bytes: &[u8]) -> std::io::Result<()> {
    let dir = path.parent().ok_or_else(|| {
        std::io::Error::new(std::io::ErrorKind::InvalidInput, "record has no parent dir")
    })?;
    let name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    let tmp = dir.join(format!(
        ".{name}.tmp-{}-{}",
        std::process::id(),
        uuid::Uuid::new_v4()
    ));
    let result = (|| -> std::io::Result<()> {
        let mut f = fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.write_all(b"\n")?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        fs::File::open(dir)?.sync_all()
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

fn failed_assertion(id: String, expected: &str, observed: String) -> Assertion {
    Assertion {
        id,
        expected: expected.to_string(),
        observed,
        pass: false,
        source: ASSERTION_SOURCE.to_string(),
        note: String::new(),
    }
}

impl HarnessEngine {
    fn consumed_dir(&self) -> PathBuf {
        self.completion_store.join(CONSUMED_DIR)
    }

    /// Run the schema gate and every `required_evals_on_done` entry on the
    /// artifact. Returns one assertion per gate; never short-circuits.
    fn evaluate_gates(&self, wf_def: &Value, artifact: &Value) -> Vec<Assertion> {
        let mut out = Vec::new();
        // validate_workflow_def already guaranteed these shapes.
        let schema = wf_def["required_schema"].as_str().unwrap_or_default();
        let schema_id = format!("schema:{schema}");
        let expected = "artifact valid against the workflow schema";
        out.push(match self.validate_schema(schema, artifact) {
            Ok(()) => Assertion {
                id: schema_id,
                expected: expected.to_string(),
                observed: "valid".to_string(),
                pass: true,
                source: ASSERTION_SOURCE.to_string(),
                note: String::new(),
            },
            Err(e) => failed_assertion(schema_id, expected, e),
        });
        let evals = wf_def["required_evals_on_done"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        for name in evals.iter().filter_map(Value::as_str) {
            out.push(self.completion_eval(name, artifact));
        }
        out
    }

    /// One eval as a completion gate. Fail closed on a missing definition,
    /// a non-deterministic kind, or a name the harness cannot execute.
    fn completion_eval(&self, name: &str, artifact: &Value) -> Assertion {
        let id = format!("eval:{name}");
        let expected = "eval passes";
        let Some(def) = self.evals.get(name) else {
            return failed_assertion(id, expected, "no definition in evals.json".to_string());
        };
        let kind = def.get("kind").and_then(Value::as_str).unwrap_or("");
        if kind != "deterministic" {
            return failed_assertion(
                id,
                expected,
                format!("eval kind '{kind}' cannot gate completion; only deterministic evals run in the harness"),
            );
        }
        let (passed, reason) = if name == "slop_gate" {
            // `run_eval("slop_gate")` reads only a top-level `text` field,
            // which no completion schema declares, so on its own it would
            // pass every artifact. As a completion gate it scans every string
            // value in the artifact instead.
            let mut strings = Vec::new();
            collect_strings(artifact, String::new(), &mut strings);
            match strings
                .iter()
                .find_map(|(p, s)| crate::slop_violation(s).map(|r| format!("at '{p}': {r}")))
            {
                Some(r) => (false, r),
                None => (true, format!("{} string value(s) clean", strings.len())),
            }
        } else {
            let r = self.run_eval(name, artifact);
            (r.passed, r.reason)
        };
        Assertion {
            id,
            expected: expected.to_string(),
            observed: reason,
            pass: passed,
            source: ASSERTION_SOURCE.to_string(),
            note: String::new(),
        }
    }

    fn mint_receipt(
        &self,
        wf: &WorkflowState,
        wf_def: &Value,
        sub: &CompletionSubmission,
        digest: &str,
        assertions: Vec<Assertion>,
    ) -> Result<Receipt, CompletionError> {
        let result = if assertions.iter().all(|a| a.pass) {
            Verdict::Pass
        } else {
            Verdict::Fail
        };
        let assertion_bytes =
            serde_json::to_vec(&assertions).map_err(|e| CompletionError::Store(e.to_string()))?;
        let repo = if self.completion_repo.trim().is_empty() {
            crate::UNSPECIFIED_REPO.to_string()
        } else {
            self.completion_repo.clone()
        };
        let mut r = Receipt {
            schema: evidence::SCHEMA.to_string(),
            version: evidence::SCHEMA_VERSION,
            id: String::new(),
            kind: COMPLETION_KIND.to_string(),
            tier: COMPLETION_TIER,
            result,
            timestamp: (self.completion_clock)(),
            repo,
            commit: sub.source_commit.clone(),
            dirty: false,
            toolchain: format!("spark-harness {}", env!("CARGO_PKG_VERSION")),
            procedure: format!(
                "spark-harness complete_verified_workflow workflow_type={}",
                wf.workflow_type
            ),
            machine: host_identity(),
            env_class: COMPLETION_TIER.as_str().to_string(),
            // The gate definition that was applied.
            input_artifacts: vec![format!("blake3:{}", artifact_digest(wf_def))],
            output_artifacts: vec![format!("blake3:{digest}")],
            assertions,
            dependencies: Vec::new(),
            declared_mutation: Mutation::None,
            observed_mutation: Mutation::None,
            authority: String::new(),
            output_digest: blake3::hash(&assertion_bytes).to_hex().to_string(),
            external_refs: vec![
                format!("task:{}", sub.task_id),
                format!("workflow:{}", wf.id),
                format!("generation:{}", sub.generation),
            ],
            ledger: None,
            lease: None,
            required_features: Vec::new(),
            reserved: String::new(),
        };
        r.id = evidence::receipt_id(&r).map_err(CompletionError::Store)?;
        Ok(r)
    }

    /// The only path into `done`. See the module docs for the full contract.
    pub fn complete_verified_workflow(
        &self,
        wf: &mut WorkflowState,
        sub: &CompletionSubmission,
    ) -> Result<CompletionRecord, CompletionError> {
        if !self.load_errors.is_empty() {
            return Err(CompletionError::DefinitionsNotLoaded(
                self.load_errors.clone(),
            ));
        }
        check_identifier("task_id", &sub.task_id)?;
        check_identifier("workflow id", &wf.id)?;
        evidence::check_commit(&sub.source_commit).map_err(CompletionError::InvalidSubmission)?;

        let wf_def = self
            .workflows
            .get(&wf.workflow_type)
            .ok_or_else(|| CompletionError::UnknownWorkflow(wf.workflow_type.clone()))?;
        crate::validate_workflow_def(&wf.workflow_type, wf_def)
            .map_err(CompletionError::MalformedWorkflow)?;
        let allowed: Vec<String> = wf_def["states"]
            .get(&wf.current_state)
            .and_then(|s| s.get("next"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_str)
                    .map(str::to_string)
                    .collect()
            })
            .unwrap_or_default();
        if !allowed.iter().any(|s| s == DONE_STATE) {
            return Err(CompletionError::IllegalTransition {
                from: wf.current_state.clone(),
                allowed,
            });
        }

        let consumed_dir = self.consumed_dir();
        fs::create_dir_all(&consumed_dir).map_err(|e| {
            CompletionError::Store(format!("cannot create {}: {e}", consumed_dir.display()))
        })?;
        // Serializes completions across threads and processes sharing this
        // store. flock is released by the kernel if the holder dies.
        let _lock = FileLock::acquire(&consumed_dir.join(".lock"))
            .map_err(|e| CompletionError::Store(format!("cannot lock completion store: {e}")))?;
        let record_path = consumed_dir.join(format!("{}.json", task_key(&sub.task_id)));
        if let Some(prev) = read_consumed(&record_path, &sub.task_id)? {
            if sub.generation == prev.generation {
                return Err(CompletionError::DuplicateCompletion {
                    task_id: sub.task_id.clone(),
                    generation: sub.generation,
                    receipt_id: prev.receipt_id,
                });
            }
            if sub.generation < prev.generation {
                return Err(CompletionError::StaleGeneration {
                    task_id: sub.task_id.clone(),
                    submitted: sub.generation,
                    recorded: prev.generation,
                });
            }
        }

        let assertions = self.evaluate_gates(wf_def, &sub.artifact);
        let failures: Vec<String> = assertions
            .iter()
            .filter(|a| !a.pass)
            .map(|a| format!("{}: {}", a.id, a.observed))
            .collect();
        let digest = artifact_digest(&sub.artifact);
        let receipt = self.mint_receipt(wf, wf_def, sub, &digest, assertions)?;
        let receipt_id = evidence::store_receipt(&self.completion_store, &receipt)
            .map_err(CompletionError::Store)?;
        if receipt.result != Verdict::Pass {
            return Err(CompletionError::GatesFailed {
                receipt_id,
                failures,
            });
        }

        write_consumed(
            &record_path,
            &ConsumedRecord {
                task_id: sub.task_id.clone(),
                generation: sub.generation,
                receipt_id: receipt_id.clone(),
                workflow_id: wf.id.clone(),
                artifact_digest: digest.clone(),
            },
        )?;
        wf.history.push(wf.current_state.clone());
        wf.current_state = DONE_STATE.to_string();
        Ok(CompletionRecord {
            receipt_id,
            task_id: sub.task_id.clone(),
            workflow_id: wf.id.clone(),
            workflow_type: wf.workflow_type.clone(),
            generation: sub.generation,
            source_commit: sub.source_commit.clone(),
            artifact_digest: digest,
        })
    }

    /// MCP `harness_complete` adapter. Arguments:
    /// `{ "workflow": WorkflowState, "submission": CompletionSubmission }`.
    /// The MCP server holds no workflow state, so `workflow.current_state` is
    /// caller-declared there; the binding proof is the stored receipt and the
    /// consumed record, both written only by the harness.
    pub fn handle_complete_request(&self, args: &Value) -> Result<Value, CompletionError> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct CompleteArgs {
            workflow: WorkflowState,
            submission: CompletionSubmission,
        }
        let parsed: CompleteArgs = serde_json::from_value(args.clone()).map_err(|e| {
            CompletionError::InvalidSubmission(format!("harness_complete arguments: {e}"))
        })?;
        let mut wf = parsed.workflow;
        let record = self.complete_verified_workflow(&mut wf, &parsed.submission)?;
        Ok(json!({ "record": record, "workflow": wf }))
    }
}

pub(crate) fn single_ref<'a>(refs: &'a [String], prefix: &str) -> Result<&'a str, CompletionError> {
    let found: Vec<&str> = refs.iter().filter_map(|r| r.strip_prefix(prefix)).collect();
    match found.as_slice() {
        [one] => Ok(one),
        _ => Err(CompletionError::ReceiptInvalid(format!(
            "expected exactly one '{prefix}' external ref, found {}",
            found.len()
        ))),
    }
}

pub(crate) fn require(
    field: &'static str,
    expected: &str,
    found: &str,
) -> Result<(), CompletionError> {
    if expected == found {
        Ok(())
    } else {
        Err(CompletionError::BindingMismatch {
            field,
            expected: expected.to_string(),
            found: found.to_string(),
        })
    }
}

/// Check that `receipt_id` in `store` is a PASS harness completion receipt
/// bound to exactly this task, generation, source commit and artifact digest
/// (bare BLAKE3 hex or `blake3:<hex>`), and that it is the task's current
/// completion: the consumed record must name this receipt, generation and
/// digest. Superseded generations and orphan receipts fail. Identity is
/// recomputed by aien-proof; see the module docs for what that does and does
/// not prove.
pub fn verify_completion(
    store: &Path,
    receipt_id: &str,
    task_id: &str,
    generation: u64,
    commit: &str,
    artifact_digest: &str,
) -> Result<VerifiedCompletion, CompletionError> {
    let receipt =
        evidence::load_receipt(store, receipt_id).map_err(CompletionError::ReceiptInvalid)?;
    // load_receipt checks identity but discards the consistency report, so
    // re-read the same file and take the report from verify_bytes.
    let path = store
        .join(evidence::RECEIPT_DIR)
        .join(format!("{}.json", receipt.id));
    let bytes = fs::read(&path)
        .map_err(|e| CompletionError::ReceiptInvalid(format!("{}: {e}", path.display())))?;
    let (again, report) =
        evidence::verify_bytes(&bytes).map_err(CompletionError::ReceiptInvalid)?;
    if again != receipt {
        return Err(CompletionError::ReceiptInvalid(
            "receipt file changed while being verified".to_string(),
        ));
    }
    if receipt.result != Verdict::Pass || !report.ok() {
        return Err(CompletionError::NotPass(format!(
            "{} ({})",
            receipt.result.as_str(),
            report.reasons.join("; ")
        )));
    }
    require("kind", COMPLETION_KIND, &receipt.kind)?;
    require("tier", COMPLETION_TIER.as_str(), receipt.tier.as_str())?;
    if !receipt
        .assertions
        .iter()
        .any(|a| a.id.starts_with("schema:"))
    {
        return Err(CompletionError::ReceiptInvalid(
            "completion receipt carries no schema assertion".to_string(),
        ));
    }
    require("commit", commit, &receipt.commit)?;
    let want_digest = artifact_digest
        .strip_prefix("blake3:")
        .unwrap_or(artifact_digest);
    let want = format!("blake3:{}", want_digest.to_ascii_lowercase());
    let found = receipt.output_artifacts.join(",");
    require("artifact_digest", &want, &found)?;
    let found_task = single_ref(&receipt.external_refs, "task:")?;
    require("task", task_id, found_task)?;
    let found_gen = single_ref(&receipt.external_refs, "generation:")?;
    require("generation", &generation.to_string(), found_gen)?;
    let workflow_id = single_ref(&receipt.external_refs, "workflow:")?;
    // The consumed record is the authority on which receipt is the accepted
    // completion. A PASS receipt it does not name (an orphan left by a failed
    // record write, or a superseded generation) never verifies.
    let digest_hex = want_digest.to_ascii_lowercase();
    let record_path = store
        .join(CONSUMED_DIR)
        .join(format!("{}.json", task_key(task_id)));
    let current = read_consumed(&record_path, task_id)?.ok_or_else(|| {
        CompletionError::NotCurrentCompletion(format!(
            "no consumed completion record for task '{task_id}'; receipt {} is an orphan",
            receipt.id
        ))
    })?;
    if current.receipt_id != receipt.id
        || current.generation != generation
        || current.artifact_digest != digest_hex
    {
        return Err(CompletionError::NotCurrentCompletion(format!(
            "task '{task_id}' current completion is generation {} receipt {} digest {}; \
             receipt {} generation {generation} digest {digest_hex} is not it",
            current.generation, current.receipt_id, current.artifact_digest, receipt.id
        )));
    }
    Ok(VerifiedCompletion {
        receipt_id: receipt.id.clone(),
        task_id: task_id.to_string(),
        workflow_id: workflow_id.to_string(),
        generation,
        source_commit: receipt.commit.clone(),
        artifact_digest: digest_hex,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn canonical_json_sorts_keys_at_every_level() {
        let a = json!({"b": 1, "a": {"y": [1, {"d": true, "c": null}], "x": "s"}});
        assert_eq!(
            canonical_json(&a),
            br#"{"a":{"x":"s","y":[1,{"c":null,"d":true}]},"b":1}"#.to_vec()
        );
        let b: Value =
            serde_json::from_str(r#"{"a":{"y":[1,{"c":null,"d":true}],"x":"s"},"b":1}"#).unwrap();
        assert_eq!(artifact_digest(&a), artifact_digest(&b));
        assert_ne!(artifact_digest(&a), artifact_digest(&json!({"b": 2})));
    }

    #[test]
    fn task_key_is_stable_and_distinct() {
        assert_eq!(task_key("t1"), task_key("t1"));
        assert_ne!(task_key("t1"), task_key("t2"));
        assert_eq!(task_key("../etc").len(), 64);
    }

    #[test]
    fn identifiers_are_checked() {
        assert!(check_identifier("task_id", "task-1").is_ok());
        assert!(check_identifier("task_id", "").is_err());
        assert!(check_identifier("task_id", " task").is_err());
        assert!(check_identifier("task_id", "a\nb").is_err());
        assert!(check_identifier("task_id", &"x".repeat(MAX_TASK_ID_LEN + 1)).is_err());
    }

    #[test]
    fn single_ref_requires_exactly_one() {
        let refs = vec!["task:a".to_string(), "task:b".to_string()];
        assert!(single_ref(&refs, "task:").is_err());
        assert!(single_ref(&refs, "generation:").is_err());
        assert_eq!(single_ref(&refs[..1], "task:").unwrap(), "a");
    }
}
