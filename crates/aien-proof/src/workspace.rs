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

//! Split a cargo workspace into one job per member crate. Each job's inputs are
//! the crate, every local crate it depends on (transitively, dev-dependencies
//! included), and the root manifest and lockfile, so a stamp goes stale exactly
//! when something the crate's tests can see changes.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::path::{Path, PathBuf};
use std::process::Command;

pub struct Member {
    pub name: String,
    pub inputs: Vec<PathBuf>,
    pub gpu: bool,
}

fn rel(root: &Path, p: &Path) -> PathBuf {
    p.strip_prefix(root)
        .map(Path::to_path_buf)
        .unwrap_or_else(|_| p.to_path_buf())
}

/// Returns the workspace root and its members, sorted by name.
pub fn members(start: &Path) -> io::Result<(PathBuf, Vec<Member>)> {
    let out = Command::new("cargo")
        .args(["metadata", "--format-version", "1", "--no-deps"])
        .current_dir(start)
        .output()?;
    if !out.status.success() {
        return Err(io::Error::other(
            String::from_utf8_lossy(&out.stderr).into_owned(),
        ));
    }
    let meta: Value = serde_json::from_slice(&out.stdout)?;
    parse(&meta)
}

pub fn parse(meta: &Value) -> io::Result<(PathBuf, Vec<Member>)> {
    let root = PathBuf::from(
        meta["workspace_root"]
            .as_str()
            .ok_or_else(|| io::Error::other("no workspace_root"))?,
    );
    let ids: BTreeSet<&str> = meta["workspace_members"]
        .as_array()
        .map(|a| a.iter().filter_map(Value::as_str).collect())
        .unwrap_or_default();

    // crate dir (relative) -> (name, local dependency dirs, gpu flag)
    let mut crates: BTreeMap<PathBuf, (String, Vec<PathBuf>, bool)> = BTreeMap::new();
    for pkg in meta["packages"].as_array().into_iter().flatten() {
        if !pkg["id"].as_str().is_some_and(|id| ids.contains(id)) {
            continue;
        }
        let manifest = PathBuf::from(pkg["manifest_path"].as_str().unwrap_or_default());
        let dir = rel(&root, manifest.parent().unwrap_or(&root));
        let deps = pkg["dependencies"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|d| d["path"].as_str())
            .map(|p| rel(&root, Path::new(p)))
            .collect();
        let gpu = pkg["metadata"]["aien-proof"]["gpu"]
            .as_bool()
            .unwrap_or(false);
        crates.insert(
            dir,
            (
                pkg["name"].as_str().unwrap_or_default().to_string(),
                deps,
                gpu,
            ),
        );
    }

    let mut shared = vec![PathBuf::from("Cargo.toml"), PathBuf::from("Cargo.lock")];
    if root.join(".cargo").is_dir() {
        shared.push(PathBuf::from(".cargo"));
    }
    shared.retain(|p| root.join(p).exists());

    let mut members = Vec::new();
    for (dir, (name, _, gpu)) in &crates {
        let mut seen = BTreeSet::new();
        let mut stack = vec![dir.clone()];
        while let Some(d) = stack.pop() {
            if !seen.insert(d.clone()) {
                continue;
            }
            if let Some((_, deps, _)) = crates.get(&d) {
                stack.extend(deps.iter().cloned());
            }
        }
        let mut inputs: Vec<PathBuf> = seen.into_iter().collect();
        inputs.extend(shared.iter().cloned());
        members.push(Member {
            name: name.clone(),
            inputs,
            gpu: *gpu,
        });
    }
    members.sort_by(|a, b| a.name.cmp(&b.name));
    Ok((root, members))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn inputs_follow_local_dependencies_transitively() {
        let meta = json!({
            "workspace_root": "/ws",
            "workspace_members": ["a", "b", "c"],
            "packages": [
                {"id": "a", "name": "a", "manifest_path": "/ws/crates/a/Cargo.toml",
                 "dependencies": [{"name": "b", "path": "/ws/crates/b"}, {"name": "serde"}], "metadata": null},
                {"id": "b", "name": "b", "manifest_path": "/ws/crates/b/Cargo.toml",
                 "dependencies": [{"name": "c", "path": "/ws/crates/c"}], "metadata": null},
                {"id": "c", "name": "c", "manifest_path": "/ws/crates/c/Cargo.toml",
                 "dependencies": [], "metadata": {"aien-proof": {"gpu": true}}}
            ]
        });
        let (_, members) = parse(&meta).unwrap();
        let a = members.iter().find(|m| m.name == "a").unwrap();
        let dirs: Vec<_> = a
            .inputs
            .iter()
            .map(|p| p.to_string_lossy().into_owned())
            .collect();
        assert!(dirs.contains(&"crates/a".into()));
        assert!(dirs.contains(&"crates/b".into()));
        assert!(dirs.contains(&"crates/c".into()));
        let c = members.iter().find(|m| m.name == "c").unwrap();
        assert!(c.gpu);
        assert_eq!(
            c.inputs.iter().filter(|p| p.starts_with("crates")).count(),
            1
        );
    }
}
