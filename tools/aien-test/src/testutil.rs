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

//! Test helpers: unique temp dirs and throwaway git repositories.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicU64, Ordering};

static COUNTER: AtomicU64 = AtomicU64::new(0);

pub struct TempDir {
    path: PathBuf,
}

impl TempDir {
    pub fn new(tag: &str) -> TempDir {
        let n = COUNTER.fetch_add(1, Ordering::SeqCst);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.subsec_nanos())
            .unwrap_or(0);
        let path = std::env::temp_dir().join(format!(
            "aien-test-{tag}-{}-{n}-{nanos}",
            std::process::id()
        ));
        fs::create_dir_all(&path).expect("create temp dir");
        let path = fs::canonicalize(&path).expect("canonicalize temp dir");
        TempDir { path }
    }

    pub fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

pub fn git(dir: &Path, args: &[&str]) {
    let mut cmd = Command::new("git");
    cmd.arg("-C")
        .arg(dir)
        .args([
            "-c",
            "user.name=test",
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "commit.gpgsign=false",
        ])
        .args(args);
    // Gated like every spawn: see `process::SPAWN_GATE`.
    let out = crate::process::output_gated(&mut cmd).expect("run git");
    assert!(
        out.status.success(),
        "git {:?} failed: {}",
        args,
        String::from_utf8_lossy(&out.stderr)
    );
}

pub fn write_file(dir: &Path, rel: &str, content: &str, executable: bool) {
    let p = dir.join(rel);
    if let Some(parent) = p.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(&p, content).expect("write file");
    if executable {
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).expect("chmod");
    }
}

/// A fresh git repository with one commit containing whatever `files` lists
/// as (relative path, content, executable).
pub fn init_repo(tag: &str, files: &[(&str, &str, bool)]) -> TempDir {
    let t = TempDir::new(tag);
    git(t.path(), &["init", "-q"]);
    for (rel, content, exec) in files {
        write_file(t.path(), rel, content, *exec);
    }
    git(t.path(), &["add", "-A"]);
    git(t.path(), &["commit", "-q", "-m", "init"]);
    t
}

/// Whether `/proc/locks` shows a blocked `flock` waiter on `path` (a line
/// `N: -> FLOCK ... maj:min:inode ...`), so a test holder can let go only
/// once the code under test is really queued behind it.
pub fn queued_on(path: &Path) -> bool {
    use std::os::unix::fs::MetadataExt;
    let ino = match fs::metadata(path) {
        Ok(m) => m.ino(),
        Err(_) => return false,
    };
    let suffix = format!(":{ino}");
    fs::read_to_string("/proc/locks")
        .unwrap_or_default()
        .lines()
        .filter(|l| l.contains("-> FLOCK"))
        .any(|l| l.split_whitespace().any(|w| w.ends_with(&suffix)))
}
