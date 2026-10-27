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

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub mod completion;
pub mod dependency;
pub mod execution;
pub mod journal;
pub mod readiness;
pub mod scheduler;

/// The terminal workflow state. Since Headstart M1 (#533) it is reachable
/// only through [`HarnessEngine::complete_verified_workflow`], never through
/// [`HarnessEngine::advance_workflow`]. `done` means "the harness ran the
/// workflow's completion gates on the submitted artifact and stored a PASS
/// receipt". It does not mean committed: World commit and authorization stay
/// in aien-runtime and are not touched by this crate.
pub const DONE_STATE: &str = "done";

/// Default subdirectory of `base_path` holding the harness-owned completion
/// store (receipts plus consumed completion records).
pub const DEFAULT_COMPLETION_STORE: &str = "completion-store";

/// Repo label written into completion receipts when the deployer has not set
/// one. The harness does not know which repository a submitted commit lives
/// in, so it says so instead of guessing.
pub const UNSPECIFIED_REPO: &str = "unspecified";

/// Slop scan shared by `run_eval("slop_gate")` and the completion gate.
/// Returns the failure reason, or `None` when the text is clean.
pub(crate) fn slop_violation(text: &str) -> Option<String> {
    let lower = text.to_lowercase();

    // Disallowed phrases
    let forbidden = [
        "this is the part where",
        "the part everyone's missing",
        "the part everyone is missing",
        "in today's fast-paced",
        "game-changer",
        "delve into",
        "unlock the power",
        "at the end of the day",
        "it is important to note",
        "in conclusion,",
    ];

    for p in forbidden {
        if lower.contains(p) {
            return Some(format!("Forbidden AI trope detected: '{}'", p));
        }
    }

    // Check antithesis pattern: "not X, it's Y"
    let antithesis_re = Regex::new(r"(?i)\bnot\s+([a-z0-9_\-\s]+),\s*(it's|it\s+is)\s+").unwrap();
    if antithesis_re.is_match(&lower) {
        return Some("Formulaic antithesis trope detected ('not X, it's Y')".to_string());
    }
    None
}

fn read_definition_file(path: &Path) -> Result<HashMap<String, Value>, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path.display(), e))?;
    serde_json::from_str::<HashMap<String, Value>>(&content)
        .map_err(|e| format!("invalid JSON object in {}: {}", path.display(), e))
}

/// Structural check of one workflow definition. Used at load time and again
/// at completion time, because `workflows` is a public map callers can edit.
pub fn validate_workflow_def(name: &str, def: &Value) -> Result<(), String> {
    let states = def
        .get("states")
        .and_then(|v| v.as_object())
        .ok_or_else(|| format!("workflow '{}': 'states' must be an object", name))?;
    for (state, sdef) in states {
        let next = sdef.get("next").and_then(|v| v.as_array()).ok_or_else(|| {
            format!(
                "workflow '{}': state '{}' needs a 'next' array",
                name, state
            )
        })?;
        for n in next {
            let n = n.as_str().ok_or_else(|| {
                format!(
                    "workflow '{}': state '{}' has a non-string next entry",
                    name, state
                )
            })?;
            if !states.contains_key(n) {
                return Err(format!(
                    "workflow '{}': state '{}' points to undeclared state '{}'",
                    name, state, n
                ));
            }
        }
    }
    if let Some(initial) = def.get("initial") {
        let initial = initial
            .as_str()
            .ok_or_else(|| format!("workflow '{}': 'initial' must be a string", name))?;
        if !states.contains_key(initial) {
            return Err(format!(
                "workflow '{}': initial state '{}' is not declared",
                name, initial
            ));
        }
    }
    def.get("required_schema")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .ok_or_else(|| {
            format!(
                "workflow '{}': 'required_schema' must be a non-empty string",
                name
            )
        })?;
    let evals = def
        .get("required_evals_on_done")
        .and_then(|v| v.as_array())
        .ok_or_else(|| {
            format!(
                "workflow '{}': 'required_evals_on_done' must be an array",
                name
            )
        })?;
    let mut seen = std::collections::BTreeSet::new();
    for e in evals {
        let e = e
            .as_str()
            .filter(|s| !s.is_empty())
            .ok_or_else(|| format!("workflow '{}': eval names must be non-empty strings", name))?;
        if !seen.insert(e) {
            return Err(format!("workflow '{}': eval '{}' listed twice", name, e));
        }
    }
    Ok(())
}

fn validate_eval_def(name: &str, def: &Value) -> Result<(), String> {
    def.get("kind")
        .and_then(|v| v.as_str())
        .filter(|s| !s.is_empty())
        .map(|_| ())
        .ok_or_else(|| format!("eval '{}': 'kind' must be a non-empty string", name))
}

fn load_evals(base: &Path) -> Result<HashMap<String, Value>, String> {
    let m = read_definition_file(&base.join("evals.json"))?;
    for (k, v) in &m {
        validate_eval_def(k, v)?;
    }
    Ok(m)
}

fn load_workflows(base: &Path) -> Result<HashMap<String, Value>, String> {
    let m = read_definition_file(&base.join("workflows.json"))?;
    for (k, v) in &m {
        validate_workflow_def(k, v)?;
    }
    Ok(m)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvalResult {
    pub name: String,
    pub passed: bool,
    pub reason: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkflowState {
    pub id: String,
    pub workflow_type: String,
    pub current_state: String,
    pub history: Vec<String>,
    pub created_at: String,
}

#[derive(Debug, Clone)]
pub struct HarnessEngine {
    pub base_path: PathBuf,
    pub evals: HashMap<String, Value>,
    pub workflows: HashMap<String, Value>,
    /// Errors from the last definition load. The verified completion API
    /// refuses to run while this is non-empty, so a missing or broken
    /// workflows.json or evals.json can never let a completion through.
    pub load_errors: Vec<String>,
    /// Harness-owned evidence store for completion receipts and consumed
    /// completion records. Defaults to `<base_path>/completion-store`.
    pub completion_store: PathBuf,
    /// Repo identity recorded in completion receipts.
    pub completion_repo: String,
    /// Unix-seconds clock for receipt timestamps. Defaults to the system
    /// clock; tests swap it to make receipt IDs deterministic.
    pub completion_clock: fn() -> u64,
}

/// System clock in Unix seconds (0 if the clock is before the epoch).
pub fn system_clock_secs() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

impl HarnessEngine {
    /// Compatibility constructor. Loads what it can, as before, but records
    /// every load error in `load_errors` so the verified completion API
    /// refuses to run on missing or broken definitions.
    pub fn new<P: AsRef<Path>>(base_path: P) -> Self {
        let base = base_path.as_ref().to_path_buf();
        let mut engine = Self {
            completion_store: base.join(DEFAULT_COMPLETION_STORE),
            base_path: base,
            evals: HashMap::new(),
            workflows: HashMap::new(),
            load_errors: Vec::new(),
            completion_repo: UNSPECIFIED_REPO.to_string(),
            completion_clock: system_clock_secs,
        };
        engine.reload();
        engine
    }

    /// Fail-closed constructor: errors when workflows.json or evals.json is
    /// missing, is not a JSON object, or holds a malformed definition.
    pub fn load<P: AsRef<Path>>(base_path: P) -> Result<Self, String> {
        let engine = Self::new(base_path);
        if engine.load_errors.is_empty() {
            Ok(engine)
        } else {
            Err(engine.load_errors.join("; "))
        }
    }

    /// Re-read both definition files. A file that fails to load keeps the
    /// previous in-memory map (the old behavior) and adds an entry to
    /// `load_errors`, which blocks verified completion until a clean reload.
    pub fn reload(&mut self) {
        self.load_errors.clear();
        match load_evals(&self.base_path) {
            Ok(m) => self.evals = m,
            Err(e) => self.load_errors.push(e),
        }
        match load_workflows(&self.base_path) {
            Ok(m) => self.workflows = m,
            Err(e) => self.load_errors.push(e),
        }
    }

    /// Layer 1: Schema Validation
    pub fn validate_schema(&self, schema_name: &str, data: &Value) -> Result<(), String> {
        let schema_path = self
            .base_path
            .join("schemas")
            .join(format!("{}.json", schema_name));
        if !schema_path.exists() {
            return Err(format!(
                "Schema '{}' not found at {:?}",
                schema_name, schema_path
            ));
        }

        let content = std::fs::read_to_string(&schema_path)
            .map_err(|e| format!("Failed to read schema {}: {}", schema_name, e))?;
        let schema_json: Value = serde_json::from_str(&content)
            .map_err(|e| format!("Invalid JSON in schema {}: {}", schema_name, e))?;

        let compiled = jsonschema::JSONSchema::compile(&schema_json)
            .map_err(|e| format!("Schema compilation failed: {}", e))?;

        if let Err(errors) = compiled.validate(data) {
            let err_msgs: Vec<String> = errors.map(|e| e.to_string()).collect();
            return Err(format!("Schema validation failed: {}", err_msgs.join("; ")));
        }

        Ok(())
    }

    /// Layer 2: Deterministic Evals & Guardrails
    pub fn run_eval(&self, eval_name: &str, payload: &Value) -> EvalResult {
        match eval_name {
            "slop_gate" => {
                let text = payload.get("text").and_then(|v| v.as_str()).unwrap_or("");
                match slop_violation(text) {
                    Some(reason) => EvalResult {
                        name: eval_name.to_string(),
                        passed: false,
                        reason,
                    },
                    None => EvalResult {
                        name: eval_name.to_string(),
                        passed: true,
                        reason: "Slop gate passed clean.".to_string(),
                    },
                }
            }

            "required_evidence" => {
                let evidence = payload.get("evidence").and_then(|v| v.as_array());
                match evidence {
                    Some(arr) if !arr.is_empty() => EvalResult {
                        name: eval_name.to_string(),
                        passed: true,
                        reason: format!("Evidence present ({} item(s)).", arr.len()),
                    },
                    _ => EvalResult {
                        name: eval_name.to_string(),
                        passed: false,
                        reason: "Required evidence array is missing or empty.".to_string(),
                    },
                }
            }

            "reproduction_present" => {
                let repro = payload.get("reproduction_steps").and_then(|v| v.as_array());
                let failure = payload.get("failure_evidence").and_then(|v| v.as_str());
                if repro.map(|a| !a.is_empty()).unwrap_or(false)
                    && failure.map(|s| !s.trim().is_empty()).unwrap_or(false)
                {
                    EvalResult {
                        name: eval_name.to_string(),
                        passed: true,
                        reason: "Reproduction steps and failure evidence verified.".to_string(),
                    }
                } else {
                    EvalResult {
                        name: eval_name.to_string(),
                        passed: false,
                        reason: "Reproduction steps and failure evidence must be non-empty."
                            .to_string(),
                    }
                }
            }

            _ => EvalResult {
                name: eval_name.to_string(),
                passed: false,
                reason: format!("Unknown eval rule: '{}'", eval_name),
            },
        }
    }

    /// Layer 4: State Machine Workflow Engine
    ///
    /// Checks transition legality only. A transition into [`DONE_STATE`] is
    /// always refused here: a legal transition by itself is not proof of
    /// completion. Use [`HarnessEngine::complete_verified_workflow`], which
    /// runs the workflow's completion gates on the real artifact and stores a
    /// harness-minted receipt before moving to `done`.
    pub fn advance_workflow(&self, wf: &mut WorkflowState, next_state: &str) -> Result<(), String> {
        if next_state == DONE_STATE {
            return Err(format!(
                "Refusing transition from '{}' to '{}': '{}' is reachable only through \
                 verified completion (complete_verified_workflow / MCP harness_complete).",
                wf.current_state, DONE_STATE, DONE_STATE
            ));
        }
        let wf_def = self
            .workflows
            .get(&wf.workflow_type)
            .ok_or_else(|| format!("Unknown workflow type: '{}'", wf.workflow_type))?;

        let states = wf_def
            .get("states")
            .and_then(|v| v.as_object())
            .ok_or_else(|| "Malformed workflow states definition".to_string())?;

        let current_def = states
            .get(&wf.current_state)
            .ok_or_else(|| format!("Current state '{}' not found in workflow", wf.current_state))?;

        let allowed_next: Vec<&str> = current_def
            .get("next")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().filter_map(|s| s.as_str()).collect())
            .unwrap_or_default();

        if !allowed_next.contains(&next_state) {
            return Err(format!(
                "Invalid transition from '{}' to '{}'. Allowed: {:?}",
                wf.current_state, next_state, allowed_next
            ));
        }

        wf.history.push(wf.current_state.clone());
        wf.current_state = next_state.to_string();
        Ok(())
    }

    /// Layer 5: Failure Replay Verification
    pub fn record_failure(
        &self,
        task_id: &str,
        failure_type: &str,
        details: &Value,
    ) -> Result<PathBuf, String> {
        let failures_dir = self.base_path.join("failures");
        std::fs::create_dir_all(&failures_dir)
            .map_err(|e| format!("Failed to create failures dir: {}", e))?;

        let file_name = format!(
            "failure_{}_{}.json",
            task_id,
            chrono::Utc::now().timestamp()
        );
        let file_path = failures_dir.join(file_name);

        let record = json!({
            "task_id": task_id,
            "failure_type": failure_type,
            "timestamp": chrono::Utc::now().to_rfc3339(),
            "details": details,
        });

        std::fs::write(&file_path, serde_json::to_string_pretty(&record).unwrap())
            .map_err(|e| format!("Failed to write failure log: {}", e))?;

        Ok(file_path)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_slop_gate() {
        let engine = HarnessEngine::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        let clean = json!({ "text": "Direct execution of native Rust binary on DGX Spark." });
        let slopped =
            json!({ "text": "In today's fast-paced world, this is a game-changer to delve into." });
        let antithesis = json!({ "text": "It is not speed, it's quality." });

        assert!(engine.run_eval("slop_gate", &clean).passed);
        assert!(!engine.run_eval("slop_gate", &slopped).passed);
        assert!(!engine.run_eval("slop_gate", &antithesis).passed);
    }

    #[test]
    fn test_required_evidence() {
        let engine = HarnessEngine::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        let no_evidence = json!({ "text": "Sample" });
        let has_evidence = json!({ "text": "Sample", "evidence": ["commit c9db3e2"] });

        assert!(!engine.run_eval("required_evidence", &no_evidence).passed);
        assert!(engine.run_eval("required_evidence", &has_evidence).passed);
    }

    #[test]
    fn test_reproduction_present() {
        let engine = HarnessEngine::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        let invalid = json!({ "reproduction_steps": [] });
        let valid = json!({
            "reproduction_steps": ["step 1"],
            "failure_evidence": "panic log"
        });

        assert!(!engine.run_eval("reproduction_present", &invalid).passed);
        assert!(engine.run_eval("reproduction_present", &valid).passed);
    }

    #[test]
    fn test_workflow_transitions() {
        let mut engine = HarnessEngine::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        engine.workflows.insert(
            "test_flow".to_string(),
            json!({
                "states": {
                    "start": { "next": ["in_progress"] },
                    "in_progress": { "next": ["done"] },
                    "done": { "next": [] }
                }
            }),
        );

        let mut wf = WorkflowState {
            id: "wf-1".to_string(),
            workflow_type: "test_flow".to_string(),
            current_state: "start".to_string(),
            history: Vec::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
        };

        assert!(engine.advance_workflow(&mut wf, "in_progress").is_ok());
        assert_eq!(wf.current_state, "in_progress");
        // `done` is legal per the state table but only reachable through
        // verified completion, so advance_workflow refuses it and leaves the
        // state and history untouched.
        let err = engine.advance_workflow(&mut wf, "done").unwrap_err();
        assert!(err.contains("complete_verified_workflow"), "{err}");
        assert_eq!(wf.current_state, "in_progress");
        assert_eq!(wf.history, vec!["start".to_string()]);
        assert!(engine.advance_workflow(&mut wf, "start").is_err());
    }

    #[test]
    fn test_shipped_definitions_load_clean() {
        let engine = HarnessEngine::load(env!("CARGO_MANIFEST_DIR")).unwrap();
        assert!(engine.load_errors.is_empty());
        for wf in [
            "engineering_fix",
            "research_handoff",
            "harness_change",
            "python_pipeline",
        ] {
            assert!(engine.workflows.contains_key(wf), "{wf}");
        }
    }

    #[test]
    fn test_schema_validation() {
        let engine = HarnessEngine::new(std::path::Path::new(env!("CARGO_MANIFEST_DIR")));
        let valid_payload = json!({
            "bug": "Parser rejects valid token sequence",
            "reproduction_steps": ["Run parser test"],
            "failure_evidence": "thread panicked at line 42",
            "regression_test": "tests::test_parser",
            "diff_summary": "Fixed token parser logic",
            "retest_evidence": "All tests passing cleanly"
        });

        assert!(engine
            .validate_schema("engineering_fix", &valid_payload)
            .is_ok());
    }
}
