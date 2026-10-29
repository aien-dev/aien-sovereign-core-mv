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

//! Test harness utilities for AIEN Sovereign Core E2E testing.
//! Provides safetensors serialization, CLI execution, mathematical telemetry, and invariant auditing.
//! Uses pure standard library and serde_json to minimize disk space.

use std::collections::HashMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use serde_json::{json, Value};

/// Result of executing a CLI binary in opaque-box testing.
#[derive(Debug, Clone)]
pub struct CliResult {
    pub exit_code: i32,
    pub stdout: String,
    pub stderr: String,
}

impl CliResult {
    pub fn success(&self) -> bool {
        self.exit_code == 0
    }

    pub fn assert_no_panics(&self) {
        assert!(
            !self.stderr.contains("panicked at"),
            "CLI panicked in stderr: {}",
            self.stderr
        );
        assert!(
            !self.stdout.contains("panicked at"),
            "CLI panicked in stdout: {}",
            self.stdout
        );
        assert!(
            !self.stderr.contains("SIGSEGV"),
            "CLI crashed with segmentation fault: {}",
            self.stderr
        );
    }
}

/// Constructs a raw safetensors binary buffer according to the Hugging Face format specification.
/// Header: 8-byte little-endian length prefix, JSON metadata, then contiguous raw tensor bytes.
pub fn create_safetensors_bytes(
    tensors: &[(&str, &[usize], &str, &[u8])],
    extra_metadata: Option<HashMap<String, String>>,
) -> Vec<u8> {
    let mut header_map = serde_json::Map::new();

    if let Some(meta) = extra_metadata {
        let mut meta_map = serde_json::Map::new();
        for (k, v) in meta {
            meta_map.insert(k, Value::String(v));
        }
        header_map.insert("__metadata__".to_string(), Value::Object(meta_map));
    }

    let mut current_offset: usize = 0;
    let mut tensor_data = Vec::new();

    for &(name, shape, dtype, data) in tensors {
        let start = current_offset;
        let end = current_offset + data.len();
        current_offset = end;
        tensor_data.extend_from_slice(data);

        let shape_vals: Vec<Value> = shape.iter().map(|&s| json!(s)).collect();
        let tensor_entry = json!({
            "dtype": dtype,
            "shape": shape_vals,
            "data_offsets": [start, end]
        });

        header_map.insert(name.to_string(), tensor_entry);
    }

    let header_json = serde_json::to_string(&Value::Object(header_map)).expect("JSON serialization");
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + tensor_data.len());
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    out.extend_from_slice(&tensor_data);

    out
}

/// Executes the compiled aien-cli binary with specified arguments.
pub fn run_aien_cli(args: &[&str]) -> CliResult {
    let manifest_dir = env!("CARGO_MANIFEST_DIR");
    let project_root = Path::new(manifest_dir).parent().unwrap_or(Path::new(manifest_dir));
    let binary_path = project_root.join("target/debug/aien-cli");

    let mut cmd = if binary_path.exists() {
        Command::new(&binary_path)
    } else {
        let mut c = Command::new("cargo");
        c.arg("run").arg("-p").arg("aien-cli").arg("--bin").arg("aien-cli").arg("--");
        c
    };

    cmd.current_dir(project_root);
    for arg in args {
        cmd.arg(arg);
    }

    let output = cmd.output().expect("Failed to execute aien-cli process");

    CliResult {
        exit_code: output.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&output.stdout).to_string(),
        stderr: String::from_utf8_lossy(&output.stderr).to_string(),
    }
}

/// Computes cosine similarity between two float vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match for cosine similarity");
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot / (norm_a.sqrt() * norm_b.sqrt())
}

/// Computes maximum absolute error between two float vectors.
pub fn max_absolute_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match for absolute error");
    let mut max_err = 0.0f32;
    for i in 0..a.len() {
        let diff = (a[i] - b[i]).abs();
        if diff > max_err {
            max_err = diff;
        }
    }
    max_err
}

/// Computes maximum relative error between two float vectors: |a - b| / max(|a|, |b|, 1e-12).
pub fn max_relative_error(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "Vector lengths must match for relative error");
    let mut max_rel = 0.0f32;
    for i in 0..a.len() {
        let denom = a[i].abs().max(b[i].abs()).max(1e-12);
        let rel = (a[i] - b[i]).abs() / denom;
        if rel > max_rel {
            max_rel = rel;
        }
    }
    max_rel
}

/// Recursive directory scanner using only std::fs.
fn collect_files_recursively(dir: &Path, files: &mut Vec<PathBuf>) {
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            let file_name = entry.file_name().to_string_lossy().to_string();
            if file_name == ".git" || file_name == "target" || file_name == ".gemini" {
                continue;
            }
            if path.is_dir() {
                collect_files_recursively(&path, files);
            } else if path.is_file() {
                files.push(path);
            }
        }
    }
}

/// Scans workspace directory to verify zero plaintext .env or credential secret files exist.
pub fn audit_zero_disk_secrets(root: &Path) -> Result<(), Vec<PathBuf>> {
    let mut all_files = Vec::new();
    collect_files_recursively(root, &mut all_files);

    let mut violations = Vec::new();
    for path in all_files {
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if name == ".env" || name.starts_with(".env.") {
                violations.push(path);
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}

/// Audits documentation and code files for strict Sovereign Voice compliance:
/// Zero em dashes (\u2014) and zero en dashes (\u2013).
pub fn audit_sovereign_voice_dashes(root: &Path) -> Result<(), Vec<(PathBuf, usize, String)>> {
    let mut all_files = Vec::new();
    collect_files_recursively(root, &mut all_files);

    let mut violations = Vec::new();
    for path in all_files {
        let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("");
        if ext == "rs" || ext == "md" || ext == "toml" || ext == "sh" {
            if let Ok(content) = fs::read_to_string(&path) {
                for (line_idx, line) in content.lines().enumerate() {
                    if line.contains('\u{2014}') || line.contains('\u{2013}') {
                        violations.push((path.clone(), line_idx + 1, line.to_string()));
                    }
                }
            }
        }
    }

    if violations.is_empty() {
        Ok(())
    } else {
        Err(violations)
    }
}
