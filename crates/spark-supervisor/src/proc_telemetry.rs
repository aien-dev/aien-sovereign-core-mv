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

use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
pub struct ProcessStats {
    pub pid: u32,
    pub is_alive: bool,
    pub rss_bytes: u64,
    pub cmdline: String,
}

/// Reads the state letter, field (3) of `/proc/<pid>/stat`. Field (2) `comm` is in
/// parentheses and may itself contain spaces or `)`, so the state is the first token
/// after the last `)` (proc_pid_stat(5)).
pub fn parse_stat_state(stat: &str) -> Option<char> {
    let rest = &stat[stat.rfind(')')? + 1..];
    rest.split_whitespace().next()?.chars().next()
}

/// A pid counts as alive unless its state is Z (zombie: exited, not yet reaped) or
/// X/x (dead), per proc_pid_stat(5). A zombie still has a `/proc/<pid>` directory, so
/// directory existence alone would report an exited child as alive.
pub fn stat_state_is_alive(state: char) -> bool {
    !matches!(state, 'Z' | 'X' | 'x')
}

pub fn get_process_stats(pid: u32) -> ProcessStats {
    let proc_dir = format!("/proc/{}", pid);
    let path = Path::new(&proc_dir);
    // Unreadable or unparsable stat (the process vanished mid-read) counts as not alive.
    let alive = fs::read_to_string(path.join("stat"))
        .ok()
        .and_then(|s| parse_stat_state(&s))
        .is_some_and(stat_state_is_alive);
    if !alive {
        return ProcessStats {
            pid,
            is_alive: false,
            rss_bytes: 0,
            cmdline: String::new(),
        };
    }

    let cmdline = fs::read_to_string(path.join("cmdline"))
        .map(|s| s.replace('\0', " ").trim().to_string())
        .unwrap_or_default();

    let mut rss_bytes = 0;
    if let Ok(statm) = fs::read_to_string(path.join("statm")) {
        let parts: Vec<&str> = statm.split_whitespace().collect();
        if parts.len() >= 2 {
            if let Ok(resident_pages) = parts[1].parse::<u64>() {
                // Typical page size is 4096 bytes on Linux arm64/x86_64
                rss_bytes = resident_pages * 4096;
            }
        }
    }

    ProcessStats {
        pid,
        is_alive: true,
        rss_bytes,
        cmdline,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{Duration, Instant};

    #[test]
    fn parses_state_after_last_paren() {
        assert_eq!(parse_stat_state("123 (sleep) S 1 123 123"), Some('S'));
        assert_eq!(parse_stat_state("123 (a) b) Z 1 123"), Some('Z'));
        assert_eq!(parse_stat_state("123 (sleep"), None);
        assert_eq!(parse_stat_state(""), None);
    }

    #[test]
    fn zombie_and_dead_are_not_alive() {
        for s in ['Z', 'X', 'x'] {
            assert!(!stat_state_is_alive(s), "{s}");
        }
        for s in ['R', 'S', 'D', 'T', 't'] {
            assert!(stat_state_is_alive(s), "{s}");
        }
    }

    #[test]
    fn own_process_is_alive_missing_pid_is_not() {
        assert!(get_process_stats(std::process::id()).is_alive);
        assert!(!get_process_stats(u32::MAX).is_alive);
    }

    #[test]
    fn unreaped_child_zombie_is_not_alive() {
        let mut child = std::process::Command::new("true").spawn().unwrap();
        let pid = child.id();
        let stat = format!("/proc/{pid}/stat");
        let deadline = Instant::now() + Duration::from_secs(5);
        // Not reaped yet: wait for the kernel to show it as a zombie.
        loop {
            let s = fs::read_to_string(&stat).expect("zombie keeps its /proc entry");
            if parse_stat_state(&s) == Some('Z') {
                break;
            }
            assert!(Instant::now() < deadline, "never became a zombie: {s}");
            std::thread::sleep(Duration::from_millis(1));
        }
        assert!(Path::new(&format!("/proc/{pid}")).exists());
        assert!(!get_process_stats(pid).is_alive);
        child.wait().unwrap();
    }
}
