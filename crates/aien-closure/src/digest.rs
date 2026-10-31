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

//! Source digest of a component directory.
//!
//! BLAKE3 over every regular file under the directory, in byte-sorted
//! relative-path order, skipping `target`, `.git`, `closure.toml` and
//! `closure.lock` (the manifest records the digest, so it cannot be part of
//! it). Each file contributes `u64be(len(path)) path u64be(len(bytes)) bytes`
//! under the domain `AIEN_CLOSURE_SOURCE_V1`. Symlinks and other non-regular
//! files are an error, never silently skipped.

use std::fs;
use std::path::{Path, PathBuf};

pub const DOMAIN: &[u8] = b"AIEN_CLOSURE_SOURCE_V1";
const SKIP_DIRS: &[&str] = &["target", ".git"];
const SKIP_FILES: &[&str] = &["closure.toml", "closure.lock"];

fn collect(base: &Path, dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    let entries = fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|e| e.to_string())?;
        let path = entry.path();
        let name = entry.file_name().to_string_lossy().into_owned();
        let ty = entry.file_type().map_err(|e| e.to_string())?;
        if ty.is_dir() {
            if SKIP_DIRS.contains(&name.as_str()) {
                continue;
            }
            collect(base, &path, out)?;
        } else if ty.is_file() {
            if dir == base && SKIP_FILES.contains(&name.as_str()) {
                continue;
            }
            out.push(path);
        } else {
            return Err(format!(
                "non-regular file in component source: {}",
                path.display()
            ));
        }
    }
    Ok(())
}

/// Returns the lowercase hex digest (64 digits).
pub fn source_digest(dir: &Path) -> Result<String, String> {
    let mut files = Vec::new();
    collect(dir, dir, &mut files)?;
    let mut keyed: Vec<(String, PathBuf)> = files
        .into_iter()
        .map(|p| {
            let rel = p
                .strip_prefix(dir)
                .unwrap_or(&p)
                .to_string_lossy()
                .replace('\\', "/");
            (rel, p)
        })
        .collect();
    keyed.sort_by(|a, b| a.0.as_bytes().cmp(b.0.as_bytes()));
    let mut h = blake3::Hasher::new();
    h.update(DOMAIN);
    for (rel, path) in keyed {
        let bytes = fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        h.update(&(rel.len() as u64).to_be_bytes());
        h.update(rel.as_bytes());
        h.update(&(bytes.len() as u64).to_be_bytes());
        h.update(&bytes);
    }
    Ok(h.finalize().to_hex().to_string())
}
