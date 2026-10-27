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

//! Bundle integrity and binding to the frozen spec (SPEC.md Section 10).

use serde::Deserialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::fs;
use std::path::Path;

pub const SUMS_FILE: &str = "sha256sums.txt";
pub const CAMPAIGN_ID: &str = "c1";

/// Subset of `manifest.json` the verifier binds against. Unknown fields are ignored.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub campaign_id: String,
    pub acceptance_sha256: String,
    pub placement_sha256: String,
    pub kv_contract_sha256: String,
    pub code_under_test_commit: String,
    pub dirty_tree: bool,
    pub kv_profile: String,
}

pub fn sha256_file(path: &Path) -> Result<String, String> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect())
}

/// Every file in the bundle is listed in `sha256sums.txt` and matches its digest.
pub fn check_sha256sums(bundle: &Path) -> Vec<String> {
    let sums = match fs::read_to_string(bundle.join(SUMS_FILE)) {
        Ok(s) => s,
        Err(e) => return vec![format!("{SUMS_FILE}: {e}")],
    };

    let mut v = Vec::new();
    let mut listed = BTreeSet::new();
    for (n, line) in sums.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let Some((digest, rel)) = line.split_once(char::is_whitespace) else {
            v.push(format!("{SUMS_FILE}:{}: malformed line", n + 1));
            continue;
        };
        let rel = rel.trim_start().trim_start_matches('*');
        if rel.starts_with('/') || rel.split('/').any(|c| c == "..") {
            v.push(format!("{SUMS_FILE}:{}: path escapes bundle: {rel}", n + 1));
            continue;
        }
        listed.insert(rel.to_string());
        match sha256_file(&bundle.join(rel)) {
            Ok(actual) if actual == digest.to_ascii_lowercase() => {}
            Ok(_) => v.push(format!("{rel}: digest mismatch")),
            Err(e) => v.push(e),
        }
    }

    match list_files(bundle) {
        Ok(files) => {
            for f in files {
                if f != SUMS_FILE && !listed.contains(&f) {
                    v.push(format!("{f}: not covered by {SUMS_FILE}"));
                }
            }
        }
        Err(e) => v.push(e),
    }
    v
}

/// The manifest names this campaign, a clean tree, and the exact spec files the verifier holds.
pub fn check_manifest_against_spec(m: &Manifest, spec_dir: &Path) -> Vec<String> {
    let mut v = Vec::new();
    if m.campaign_id != CAMPAIGN_ID {
        v.push(format!(
            "campaign_id {} is not {CAMPAIGN_ID}",
            m.campaign_id
        ));
    }
    if m.dirty_tree {
        v.push("code under test was built from a dirty tree".into());
    }
    let commit = &m.code_under_test_commit;
    if commit.len() != 40 || !commit.chars().all(|c| c.is_ascii_hexdigit()) {
        v.push(format!(
            "code_under_test_commit {commit:?} is not a full SHA-1"
        ));
    }
    let bound = [
        ("acceptance.toml", &m.acceptance_sha256),
        ("placement.toml", &m.placement_sha256),
        ("kv_contract.toml", &m.kv_contract_sha256),
    ];
    for (file, recorded) in bound {
        match sha256_file(&spec_dir.join(file)) {
            Ok(actual) if actual == recorded.to_ascii_lowercase() => {}
            Ok(actual) => v.push(format!(
                "{file}: bundle recorded {recorded}, frozen spec is {actual}"
            )),
            Err(e) => v.push(e),
        }
    }
    v
}

fn list_files(root: &Path) -> Result<Vec<String>, String> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let entries = fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for entry in entries {
            let path = entry.map_err(|e| format!("{}: {e}", dir.display()))?.path();
            if path.is_dir() {
                stack.push(path);
            } else if let Ok(rel) = path.strip_prefix(root) {
                let rel: Vec<String> = rel
                    .components()
                    .map(|c| c.as_os_str().to_string_lossy().into_owned())
                    .collect();
                out.push(rel.join("/"));
            }
        }
    }
    out.sort();
    Ok(out)
}
