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

//! `aien-test status`: what the local pipeline is doing right now (ADR 0033
//! Decision 10). Read-only: it never takes a lock or runs a gate.

use crate::evidence::{utc_secs, utc_string};
use crate::process;
use crate::resources::{GpuConfig, Hardware, PoolSizes};
use crate::worker::{self, StateDir};
use serde_json::Value;
use std::fs;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{SystemTime, UNIX_EPOCH};

/// A lock file's state, read from `/proc/locks` by inode (never by probing
/// the lock, which could disturb a running gate).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LockState {
    Absent,
    Free,
    Held(Vec<u32>),
}

impl LockState {
    pub fn text(&self) -> String {
        match self {
            LockState::Absent => "absent".to_string(),
            LockState::Free => "free".to_string(),
            LockState::Held(pids) => format!(
                "held (pid {})",
                pids.iter()
                    .map(|p| p.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            ),
        }
    }
}

/// Pids holding a FLOCK on `inode` according to `/proc/locks` text.
pub fn flock_holders(locks: &str, inode: u64) -> Vec<u32> {
    let mut pids = Vec::new();
    for line in locks.lines() {
        let f: Vec<&str> = line.split_whitespace().collect();
        // "N: FLOCK ADVISORY WRITE pid maj:min:inode start end"
        if f.len() < 6 || f[1] != "FLOCK" {
            continue;
        }
        let ino = f[5].rsplit(':').next().and_then(|s| s.parse::<u64>().ok());
        if ino == Some(inode) {
            if let Ok(pid) = f[4].parse::<u32>() {
                if !pids.contains(&pid) {
                    pids.push(pid);
                }
            }
        }
    }
    pids
}

pub fn lock_state(path: &Path, locks: &str) -> LockState {
    match fs::metadata(path) {
        Err(_) => LockState::Absent,
        Ok(m) => {
            let pids = flock_holders(locks, m.ino());
            if pids.is_empty() {
                LockState::Free
            } else {
                LockState::Held(pids)
            }
        }
    }
}

fn proc_locks() -> String {
    fs::read_to_string("/proc/locks").unwrap_or_default()
}

/// The quiet flag: `quietlock check` exit code when the tool exists, else the
/// flag file.
pub fn quiet_flag(gpu: &GpuConfig) -> String {
    let mut cmd = Command::new("quietlock");
    cmd.arg("check");
    match process::output_gated(&mut cmd) {
        Ok(o) if o.status.success() => "clear (quietlock check rc 0)".to_string(),
        Ok(o) => format!(
            "RAISED (quietlock check rc {}): {}",
            o.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&o.stdout)
                .lines()
                .next()
                .unwrap_or("")
                .trim()
        ),
        Err(_) => match &gpu.quiet_flag {
            Some(p) if p.exists() => format!("RAISED (file {})", p.display()),
            Some(p) => format!("clear (no file {})", p.display()),
            None => "unknown (no quietlock, no HOME)".to_string(),
        },
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// The worker heartbeat line.
pub fn heartbeat_line(state: &StateDir, now: u64) -> String {
    let p = state.heartbeat();
    let Ok(text) = fs::read_to_string(&p) else {
        return "Worker: no heartbeat".to_string();
    };
    let Ok(v) = serde_json::from_str::<Value>(&text) else {
        return "Worker: heartbeat unreadable".to_string();
    };
    let age = v["utc"]
        .as_str()
        .and_then(utc_secs)
        .map(|t| now.saturating_sub(t));
    let alive = v["pid"]
        .as_u64()
        .map(|pid| Path::new(&format!("/proc/{pid}")).exists());
    format!(
        "Worker: pid {} {} phase {} repo {} sha {} heartbeat {} ago",
        v["pid"].as_u64().unwrap_or(0),
        match alive {
            Some(true) => "alive",
            Some(false) => "GONE",
            None => "?",
        },
        v["phase"].as_str().unwrap_or("?"),
        v["repo"].as_str().unwrap_or("-"),
        v["sha"]
            .as_str()
            .map(|s| &s[..12.min(s.len())])
            .unwrap_or("-"),
        match age {
            Some(a) => format!("{a} s"),
            None => "? s".to_string(),
        }
    )
}

/// Newest bundle per repository: (repo, sha, verdict, finished_utc).
pub fn newest_bundles(state: &StateDir) -> Vec<(String, String, String, String)> {
    let mut out = Vec::new();
    let Ok(repos) = fs::read_dir(state.root.join("bundles")) else {
        return out;
    };
    let mut repos: Vec<PathBuf> = repos.flatten().map(|e| e.path()).collect();
    repos.sort();
    for r in repos {
        let Ok(shas) = fs::read_dir(&r) else {
            continue;
        };
        let mut best: Option<(String, String, String, String)> = None;
        for s in shas.flatten() {
            let p = s.path().join("bundle.json");
            let Ok(text) = fs::read(&p) else {
                continue;
            };
            let Ok(v) = serde_json::from_slice::<Value>(&text) else {
                continue;
            };
            let fin = v["finished_utc"].as_str().unwrap_or("").to_string();
            if best.as_ref().is_none_or(|b| fin > b.3) {
                best = Some((
                    r.file_name()
                        .map(|n| n.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    v["commit"].as_str().unwrap_or("?").to_string(),
                    match v["plan"]["verdict"].as_str() {
                        Some(p) if p != v["verdict"].as_str().unwrap_or("") => {
                            format!("{} (plan {p})", v["verdict"].as_str().unwrap_or("?"))
                        }
                        _ => v["verdict"].as_str().unwrap_or("?").to_string(),
                    },
                    fin,
                ));
            }
        }
        if let Some(b) = best {
            out.push(b);
        }
    }
    out
}

/// The whole status block.
pub fn lines(state: &StateDir) -> Vec<String> {
    let now = now_secs();
    let mut out = vec![
        "AIEN STATUS".to_string(),
        format!("Time: {}", utc_string(now)),
    ];
    let pools = PoolSizes::detect();
    let hw = Hardware::detect();
    out.push(format!(
        "Pools: host={} qemu={} gb10=1",
        pools.host, pools.qemu
    ));
    out.push(format!(
        "Hardware: gb10 {} qemu {}",
        if hw.gb10_present { "present" } else { "absent" },
        if hw.qemu_available {
            "available"
        } else {
            "absent"
        }
    ));
    let gpu = GpuConfig::detect();
    let locks = proc_locks();
    out.push(format!(
        "GPU lock {}: {}",
        gpu.lock_path.display(),
        lock_state(&gpu.lock_path, &locks).text()
    ));
    match &gpu.proof_lock_path {
        Some(p) => {
            let slots = p.parent().map(Path::to_path_buf).unwrap_or_default();
            let mut names: Vec<PathBuf> = fs::read_dir(&slots)
                .map(|rd| rd.flatten().map(|e| e.path()).collect())
                .unwrap_or_default();
            names.sort();
            names.retain(|n| n.extension().is_some_and(|e| e == "lock"));
            if names.is_empty() {
                out.push(format!("aien-proof slots {}: none", slots.display()));
            }
            for n in names {
                out.push(format!(
                    "aien-proof slot {}: {}",
                    n.file_stem()
                        .map(|s| s.to_string_lossy().into_owned())
                        .unwrap_or_default(),
                    lock_state(&n, &locks).text()
                ));
            }
        }
        None => out.push("aien-proof slots: no board".to_string()),
    }
    out.push(format!("Quiet flag: {}", quiet_flag(&gpu)));
    out.push(format!("State dir: {}", state.root.display()));
    out.push(heartbeat_line(state, now));
    let triggers = worker::triggers(state);
    out.push(format!(
        "Triggers pending: {}{}",
        triggers.len(),
        if triggers.is_empty() {
            String::new()
        } else {
            format!(
                " ({})",
                triggers
                    .iter()
                    .map(|(r, s, _)| format!("{r}@{}", &s[..12]))
                    .collect::<Vec<_>>()
                    .join(" ")
            )
        }
    ));
    let outbox = worker::outbox_files(state).len();
    out.push(format!("Outbox pending: {outbox}"));
    let bundles = newest_bundles(state);
    if bundles.is_empty() {
        out.push("Newest bundles: none".to_string());
    }
    for (repo, sha, verdict, fin) in bundles {
        out.push(format!(
            "Newest bundle {repo}: {} {verdict} {fin}",
            &sha[..12.min(sha.len())]
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::TempDir;

    #[test]
    fn proc_locks_parsing() {
        let text = "1: POSIX  ADVISORY  READ 4076 103:02:10631027 1073741826 1073742335\n\
                    2: FLOCK  ADVISORY  WRITE 2379546 00:a9:8128022 0 EOF\n\
                    3: FLOCK  ADVISORY  WRITE 2502 103:02:8128022 0 EOF\n\
                    4: -> FLOCK  ADVISORY  WRITE 99 00:a9:8128022 0 EOF\n";
        assert_eq!(flock_holders(text, 8128022), vec![2379546, 2502]);
        assert!(flock_holders(text, 10631027).is_empty());
        let t = TempDir::new("status-lock");
        let f = t.path().join("x.lock");
        assert_eq!(lock_state(&f, text), LockState::Absent);
        fs::write(&f, "").unwrap();
        assert_eq!(lock_state(&f, text), LockState::Free);
        let ino = fs::metadata(&f).unwrap().ino();
        let held = format!("5: FLOCK  ADVISORY  WRITE 777 00:01:{ino} 0 EOF\n");
        assert_eq!(lock_state(&f, &held), LockState::Held(vec![777]));
        assert_eq!(LockState::Held(vec![1, 2]).text(), "held (pid 1,2)");
    }

    #[test]
    fn heartbeat_and_bundles() {
        let t = TempDir::new("status-state");
        let s = StateDir::new(t.path());
        assert_eq!(heartbeat_line(&s, 10), "Worker: no heartbeat");
        s.beat("r", &"c".repeat(40), "run");
        let l = heartbeat_line(&s, now_secs() + 5);
        assert!(l.contains("alive") && l.contains("phase run"), "{l}");
        assert!(newest_bundles(&s).is_empty());
        for (sha, fin, v) in [
            ("a", "2026-10-11T01:00:00Z", "PASS"),
            ("b", "2026-10-11T02:00:00Z", "FAIL"),
        ] {
            let d = s.root.join("bundles/r").join(sha.repeat(40));
            fs::create_dir_all(&d).unwrap();
            fs::write(
                d.join("bundle.json"),
                format!(
                    "{{\"commit\":\"{}\",\"verdict\":\"{v}\",\"finished_utc\":\"{fin}\"}}",
                    sha.repeat(40)
                ),
            )
            .unwrap();
        }
        let n = newest_bundles(&s);
        assert_eq!(n.len(), 1);
        assert_eq!(n[0].2, "FAIL");
        let all = lines(&s);
        assert_eq!(all[0], "AIEN STATUS");
        assert!(all
            .iter()
            .any(|l| l.starts_with("Newest bundle r: bbbbbbbbbbbb FAIL")));
    }
}
