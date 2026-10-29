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

//! Job fingerprints. Paths are hashed relative to the job's base directory, so
//! the same code in two different worktrees produces the same key and shares
//! one stamp. Inside a git checkout, files git ignores (build output such as a
//! compiled `.so` dropped into a source dir) are left out, exactly as git would.

use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::ffi::OsStrExt;
use std::path::{Component, Path, PathBuf};
use std::process::Command;

/// Directory names never hashed: build output (Cargo `target`, oracle `.lake`), git
/// internals, live crumb scents.
const SKIP: &[&str] = &["target", ".git", ".crumb.local", ".lake"];

fn field(h: &mut blake3::Hasher, bytes: &[u8]) {
    h.update(&(bytes.len() as u64).to_be_bytes());
    h.update(bytes);
}

fn collect(path: &Path, out: &mut Vec<PathBuf>) -> io::Result<()> {
    let meta = fs::metadata(path)?;
    if meta.is_dir() {
        for entry in fs::read_dir(path)? {
            let entry = entry?;
            let name = entry.file_name();
            if SKIP.iter().any(|s| name == *s) {
                continue;
            }
            collect(&entry.path(), out)?;
        }
    } else if meta.is_file() {
        out.push(path.to_path_buf());
    }
    Ok(())
}

/// Tracked plus untracked-but-not-ignored files under `inputs`, or `None` when
/// `base` is not a git checkout.
fn git_files(base: &Path, inputs: &[&PathBuf]) -> Option<Vec<PathBuf>> {
    if inputs.is_empty() {
        return Some(Vec::new());
    }
    let out = Command::new("git")
        .arg("-C")
        .arg(base)
        .args([
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
            "--",
        ])
        .args(inputs)
        .output()
        .ok()?;
    if !out.status.success() {
        return None;
    }
    let files = out
        .stdout
        .split(|b| *b == 0)
        .filter(|s| !s.is_empty())
        .map(|s| base.join(OsStr::from_bytes(s)))
        .filter(|p| {
            !p.components()
                .any(|c| SKIP.iter().any(|s| c.as_os_str() == *s))
        })
        .collect();
    Some(files)
}

fn inside(input: &Path) -> bool {
    input.is_relative() && !input.components().any(|c| c == Component::ParentDir)
}

/// Feed the sorted file set under `inputs` (paths and content hashes) into `h`.
fn hash_inputs(h: &mut blake3::Hasher, base: &Path, inputs: &[PathBuf]) -> io::Result<()> {
    // Inputs inside the checkout go through git's view; anything outside it
    // (sibling repos reached by `../` path dependencies) is walked directly.
    let (local, outside): (Vec<&PathBuf>, Vec<&PathBuf>) = inputs.iter().partition(|p| inside(p));
    let mut files = Vec::new();
    match git_files(base, &local) {
        Some(tracked) => files.extend(tracked),
        None => {
            for input in &local {
                collect(&base.join(input), &mut files)?;
            }
        }
    }
    for input in &outside {
        collect(&base.join(input), &mut files)?;
    }
    // Git lists a symlinked directory (e.g. a crate linked in from a sibling
    // repo) as one entry; hash what it points at so edits there void the stamp.
    let mut expanded = Vec::with_capacity(files.len());
    for file in files {
        if fs::metadata(&file).is_ok_and(|m| m.is_dir()) {
            collect(&file, &mut expanded)?;
        } else {
            expanded.push(file);
        }
    }
    let mut files = expanded;
    files.sort();
    files.dedup();
    h.update(&(files.len() as u64).to_be_bytes());
    for file in &files {
        let rel = file.strip_prefix(base).unwrap_or(file);
        field(h, rel.to_string_lossy().as_bytes());
        match fs::read(file) {
            Ok(data) => field(h, blake3::hash(&data).as_bytes()),
            // Tracked in git but deleted in this worktree.
            Err(e) if e.kind() == io::ErrorKind::NotFound => field(h, b"<deleted>"),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

pub fn job_key(
    base: &Path,
    job: &str,
    cmd: &[String],
    toolchain: &str,
    inputs: &[PathBuf],
) -> io::Result<String> {
    let mut h = blake3::Hasher::new();
    h.update(b"AIEN_PROOF_JOB_V2");
    field(&mut h, job.as_bytes());
    h.update(&(cmd.len() as u64).to_be_bytes());
    for arg in cmd {
        field(&mut h, arg.as_bytes());
    }
    field(&mut h, toolchain.as_bytes());
    hash_inputs(&mut h, base, inputs)?;
    Ok(h.finalize().to_hex().to_string())
}

/// Digest of the file set under `inputs` alone (no command, no toolchain).
/// Receipts use it to name one slice of a job's inputs, such as a formal
/// model or the implementation file a proof covers, with the same file rules
/// the job fingerprint uses.
pub fn inputs_digest(base: &Path, inputs: &[PathBuf]) -> io::Result<String> {
    let mut h = blake3::Hasher::new();
    h.update(b"AIEN_PROOF_INPUTS_V1");
    hash_inputs(&mut h, base, inputs)?;
    Ok(h.finalize().to_hex().to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::temp_dir;

    fn tree(root: &Path) {
        fs::create_dir_all(root.join("src")).unwrap();
        fs::create_dir_all(root.join("src/target")).unwrap();
        fs::write(root.join("src/lib.rs"), "pub fn a() {}").unwrap();
        fs::write(root.join("src/target/junk.o"), "junk").unwrap();
    }

    fn key(root: &Path) -> String {
        job_key(
            root,
            "t",
            &["cargo".into(), "test".into()],
            "rustc 1",
            &[PathBuf::from("src")],
        )
        .unwrap()
    }

    #[test]
    fn same_code_in_two_worktrees_shares_a_key() {
        let (a, b) = (temp_dir("fp-a"), temp_dir("fp-b"));
        tree(&a);
        tree(&b);
        assert_eq!(key(&a), key(&b));
    }

    #[test]
    fn content_change_changes_key_and_build_output_does_not() {
        let root = temp_dir("fp-c");
        tree(&root);
        let before = key(&root);
        fs::write(root.join("src/target/junk.o"), "different junk").unwrap();
        assert_eq!(before, key(&root));
        fs::write(root.join("src/lib.rs"), "pub fn b() {}").unwrap();
        assert_ne!(before, key(&root));
    }

    #[test]
    fn git_ignored_files_do_not_change_the_key() {
        let root = temp_dir("fp-git");
        tree(&root);
        fs::write(root.join(".gitignore"), "*.so\n").unwrap();
        let git = |args: &[&str]| {
            assert!(Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(args)
                .output()
                .unwrap()
                .status
                .success());
        };
        git(&["init", "-q"]);
        git(&["add", "."]);
        fs::write(root.join("src/libkernel.so"), "build 1").unwrap();
        let before = key(&root);
        fs::write(root.join("src/libkernel.so"), "build 2").unwrap();
        assert_eq!(before, key(&root));
        // Untracked source that git does not ignore still counts.
        fs::write(root.join("src/new.rs"), "pub fn n() {}").unwrap();
        assert_ne!(before, key(&root));
    }

    #[test]
    fn edits_behind_a_tracked_directory_symlink_change_the_key() {
        let (root, linked) = (temp_dir("fp-link"), temp_dir("fp-link-target"));
        fs::create_dir_all(&root).unwrap();
        fs::write(linked.join("lib.rs"), "pub fn a() {}").unwrap();
        std::os::unix::fs::symlink(&linked, root.join("src")).unwrap();
        for args in [&["init", "-q"][..], &["add", "."][..]] {
            assert!(Command::new("git")
                .arg("-C")
                .arg(&root)
                .args(args)
                .output()
                .unwrap()
                .status
                .success());
        }
        let before = key(&root);
        fs::write(linked.join("lib.rs"), "pub fn b() {}").unwrap();
        assert_ne!(before, key(&root));
    }

    #[test]
    fn command_and_toolchain_are_part_of_the_key() {
        let root = temp_dir("fp-d");
        tree(&root);
        let inputs = [PathBuf::from("src")];
        let base = job_key(&root, "t", &["x".into()], "rustc 1", &inputs).unwrap();
        assert_ne!(
            base,
            job_key(&root, "t", &["y".into()], "rustc 1", &inputs).unwrap()
        );
        assert_ne!(
            base,
            job_key(&root, "t", &["x".into()], "rustc 2", &inputs).unwrap()
        );
    }
}
