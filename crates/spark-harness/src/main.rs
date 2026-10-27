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

fn default_harness_path() -> PathBuf {
    if std::path::Path::new("schemas").is_dir() && std::path::Path::new("evals.json").is_file() {
        return PathBuf::from(".");
    }
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join("workspace/spark-harness-data")
}

use clap::{Parser, Subcommand};
use serde_json::{json, Value};
use spark_harness::completion::CompletionSubmission;
use spark_harness::{HarnessEngine, WorkflowState};
use std::io::{self, BufRead, Write};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "spark-harness",
    about = "Native Rust 5-Layer Change-Management Harness & MCP Server"
)]
struct Cli {
    #[arg(long, default_value_os_t = default_harness_path())]
    harness_path: PathBuf,

    #[command(subcommand)]
    command: Option<Commands>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run self-test across all 5 layers
    Test,
    /// Run as Stdio JSON-RPC MCP server
    Mcp,
    /// Validate JSON against a schema
    Validate {
        #[arg(long)]
        schema: String,
        #[arg(long)]
        data: String,
    },
    /// Evaluate content through a guardrail gate
    Eval {
        #[arg(long)]
        gate: String,
        #[arg(long)]
        payload: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();
    let engine = HarnessEngine::new(&cli.harness_path);

    match cli.command.unwrap_or(Commands::Test) {
        Commands::Test => {
            println!("=== Running spark-harness Native 5-Layer Verification Suite ===");

            // 1. Slop Gate Eval Test
            let clean_payload =
                json!({ "text": "Direct execution of native Rust binary on DGX Spark." });
            let slopped_payload = json!({ "text": "In today's fast-paced world, this is a game-changer to delve into." });

            let res_clean = engine.run_eval("slop_gate", &clean_payload);
            let res_slop = engine.run_eval("slop_gate", &slopped_payload);

            assert!(res_clean.passed, "Clean text should pass slop gate");
            assert!(
                !res_slop.passed,
                "Sloppy text must be rejected by slop gate"
            );
            println!("Layer 2 (Slop Gate Eval): PASSED");

            // 2. Evidence Requirement Eval Test
            let no_evidence = json!({ "text": "Sample" });
            let has_evidence = json!({ "text": "Sample", "evidence": ["commit c9db3e2"] });
            assert!(!engine.run_eval("required_evidence", &no_evidence).passed);
            assert!(engine.run_eval("required_evidence", &has_evidence).passed);
            println!("Layer 2 (Required Evidence Eval): PASSED");

            // 3. Workflow State Transition Test
            let mut wf = WorkflowState {
                id: "test-001".to_string(),
                workflow_type: "engineering_fix".to_string(),
                current_state: "reproduce".to_string(),
                history: Vec::new(),
                created_at: chrono::Utc::now().to_rfc3339(),
            };

            assert!(engine.advance_workflow(&mut wf, "regression_fail").is_ok());
            assert_eq!(wf.current_state, "regression_fail");
            assert!(engine.advance_workflow(&mut wf, "implement").is_ok());
            assert_eq!(wf.current_state, "implement");
            assert!(
                engine.advance_workflow(&mut wf, "done").is_err(),
                "Cannot jump directly to done"
            );
            assert!(engine.advance_workflow(&mut wf, "retest").is_ok());
            assert!(engine.advance_workflow(&mut wf, "review").is_ok());
            assert!(
                engine.advance_workflow(&mut wf, "done").is_err(),
                "done is reachable only through verified completion"
            );
            assert_eq!(wf.current_state, "review");
            println!("Layer 4 (Deterministic Workflows): PASSED");

            // 3b. Verified completion into a throwaway store.
            let mut gated = engine.clone();
            gated.completion_store = std::env::temp_dir()
                .join(format!("spark-harness-selftest-{}", uuid::Uuid::new_v4()));
            let sub = CompletionSubmission {
                task_id: "selftest-task".to_string(),
                generation: 1,
                source_commit: "0".repeat(40),
                artifact: json!({
                    "bug": "Parser rejects valid token sequence",
                    "reproduction_steps": ["Run parser test"],
                    "failure_evidence": "thread panicked at line 42",
                    "regression_test": "tests::test_parser",
                    "diff_summary": "Fixed token parser logic",
                    "retest_evidence": "All tests passing cleanly"
                }),
            };
            let outcome = gated.complete_verified_workflow(&mut wf, &sub);
            let _ = std::fs::remove_dir_all(&gated.completion_store);
            let record = outcome?;
            assert_eq!(wf.current_state, "done");
            assert_eq!(record.receipt_id.len(), 64);
            println!("Layer 4b (Verified Completion Gate): PASSED");

            // 4. Failure Replay Record Test
            let log_path = engine.record_failure(
                "test-task-1",
                "RegressionTestFailed",
                &json!({"test": "assert_eq"}),
            )?;
            assert!(log_path.exists(), "Failure log should be created on disk");
            println!("Layer 5 (Failure Replay Recording): PASSED");

            println!(
                "\nAll 5-layer change management harness invariants verified in pure native Rust."
            );
        }

        Commands::Validate { schema, data } => {
            let parsed_data: Value = serde_json::from_str(&data)?;
            match engine.validate_schema(&schema, &parsed_data) {
                Ok(_) => println!("VALIDATION PASSED: Schema '{}'", schema),
                Err(err) => {
                    eprintln!("VALIDATION FAILED: {}", err);
                    std::process::exit(1);
                }
            }
        }

        Commands::Eval { gate, payload } => {
            let parsed: Value = serde_json::from_str(&payload)?;
            let result = engine.run_eval(&gate, &parsed);
            println!("{}", serde_json::to_string_pretty(&result)?);
            if !result.passed {
                std::process::exit(1);
            }
        }

        Commands::Mcp => {
            run_mcp_stdio_loop(engine).await?;
        }
    }

    Ok(())
}

async fn run_mcp_stdio_loop(engine: HarnessEngine) -> io::Result<()> {
    let stdin = io::stdin();
    let mut stdout = io::stdout();

    for line in stdin.lock().lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }

        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let req_id = req.get("id").cloned().unwrap_or(Value::Null);
        let method = req.get("method").and_then(|m| m.as_str()).unwrap_or("");

        let resp = match method {
            "initialize" => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {
                        "name": "spark-harness-rs",
                        "version": "0.1.0"
                    },
                    "capabilities": {
                        "tools": {}
                    }
                }
            }),

            "tools/list" => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "tools": [
                        {
                            "name": "harness_eval",
                            "description": "Run an artifact or text through change-management guardrail evals (slop_gate, required_evidence)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "gate": { "type": "string" },
                                    "payload": { "type": "object" }
                                },
                                "required": ["gate", "payload"]
                            }
                        },
                        {
                            "name": "harness_validate",
                            "description": "Validate JSON object against registered harness schemas (engineering_fix, research_handoff, harness_change)",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "schema": { "type": "string" },
                                    "data": { "type": "object" }
                                },
                                "required": ["schema", "data"]
                            }
                        },
                        {
                            "name": "harness_complete",
                            "description": "Verified completion: runs the workflow's required schema and completion evals on the artifact, stores a harness-minted EvidenceReceiptV1, and only then moves the workflow to done. Refusals are JSON-RPC errors. done does not mean committed.",
                            "inputSchema": {
                                "type": "object",
                                "properties": {
                                    "workflow": {
                                        "type": "object",
                                        "properties": {
                                            "id": { "type": "string" },
                                            "workflow_type": { "type": "string" },
                                            "current_state": { "type": "string" },
                                            "history": { "type": "array", "items": { "type": "string" } },
                                            "created_at": { "type": "string" }
                                        },
                                        "required": ["id", "workflow_type", "current_state", "history", "created_at"]
                                    },
                                    "submission": {
                                        "type": "object",
                                        "properties": {
                                            "task_id": { "type": "string" },
                                            "generation": { "type": "integer", "minimum": 0 },
                                            "source_commit": { "type": "string" },
                                            "artifact": { "type": "object" }
                                        },
                                        "required": ["task_id", "generation", "source_commit", "artifact"],
                                        "additionalProperties": false
                                    }
                                },
                                "required": ["workflow", "submission"],
                                "additionalProperties": false
                            }
                        }
                    ]
                }
            }),

            "tools/call" => {
                let params = req.get("params").cloned().unwrap_or(json!({}));
                let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
                let args = params.get("arguments").cloned().unwrap_or(json!({}));

                match name {
                    "harness_eval" => {
                        let gate = args.get("gate").and_then(|g| g.as_str()).unwrap_or("");
                        let payload = args.get("payload").cloned().unwrap_or(json!({}));
                        let eval_res = engine.run_eval(gate, &payload);
                        json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [
                                    { "type": "text", "text": serde_json::to_string(&eval_res).unwrap() }
                                ]
                            }
                        })
                    }

                    "harness_validate" => {
                        let schema = args.get("schema").and_then(|s| s.as_str()).unwrap_or("");
                        let data = args.get("data").cloned().unwrap_or(json!({}));
                        match engine.validate_schema(schema, &data) {
                            Ok(_) => json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "content": [{ "type": "text", "text": "Validation passed." }]
                                }
                            }),
                            Err(e) => json!({
                                "jsonrpc": "2.0",
                                "id": req_id,
                                "result": {
                                    "content": [{ "type": "text", "text": format!("Validation failed: {}", e) }],
                                    "isError": true
                                }
                            }),
                        }
                    }

                    "harness_complete" => match engine.handle_complete_request(&args) {
                        Ok(done) => json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "result": {
                                "content": [
                                    { "type": "text", "text": done.to_string() }
                                ]
                            }
                        }),
                        Err(e) => json!({
                            "jsonrpc": "2.0",
                            "id": req_id,
                            "error": e.rpc_error()
                        }),
                    },

                    _ => json!({
                        "jsonrpc": "2.0",
                        "id": req_id,
                        "error": { "code": -32601, "message": "Method or tool not found" }
                    }),
                }
            }

            _ => json!({
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {}
            }),
        };

        let out = serde_json::to_string(&resp).unwrap();
        writeln!(stdout, "{}", out)?;
        stdout.flush()?;
    }

    Ok(())
}
