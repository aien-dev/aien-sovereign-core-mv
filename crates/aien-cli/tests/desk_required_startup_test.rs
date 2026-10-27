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

//! sc#328 (Drake decision b, 2026-10-08): the approval desk is REQUIRED by
//! default. A real `aien daemon` process, started four ways:
//!   D1  no desk key, switch unset: startup is refused before the socket
//!       exists, the message names the key path and how to create it.
//!   D2  the dev opt-out (`AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK=0`) in a
//!       strict run (no AIEN_DEV_FALLBACK, as release qualification runs):
//!       startup is refused.
//!   D3  the dev opt-out in a dev run: the daemon serves and says so loudly
//!       at startup ("Authorize MAC: OFF, DEV OPT-OUT").
//!   D4  a desk key created by `aien compose desk-key --create`: the daemon
//!       serves with "Authorize MAC: on".
//! A helper whose check fails still stops its daemon (a `Daemon` guard
//! kills and reaps it on drop), so a failing run leaves nothing running.
//! No model and no composition library are needed (host CPU only); these run
//! in every build.
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::time::Duration;

const SWITCH: &str = "AIEN_COMPOSE_AUTHORIZE_REQUIRES_DESK";

struct Rig {
    _tmp: tempfile::TempDir,
    root: PathBuf,
}

impl Rig {
    fn new() -> Rig {
        // Short path: unix socket paths are limited to 108 bytes.
        let tmp = tempfile::Builder::new()
            .prefix("dk-")
            .tempdir_in("/tmp")
            .unwrap();
        let root = std::fs::canonicalize(tmp.path()).unwrap();
        for d in ["state", "home", "prov"] {
            std::fs::create_dir_all(root.join(d)).unwrap();
        }
        Rig { _tmp: tmp, root }
    }
    fn sock(&self) -> PathBuf {
        self.root.join("s")
    }
    fn compose(&self) -> PathBuf {
        self.root.join("compose")
    }
    fn log(&self) -> PathBuf {
        self.root.join("daemon.log")
    }
    fn cmd(&self, dev: bool, switch: Option<&str>) -> Command {
        let mut c = Command::new(env!("CARGO_BIN_EXE_aien-cli"));
        c.current_dir(&self.root)
            .env("HOME", self.root.join("home"))
            .env("NO_COLOR", "1")
            .env("AIEN_RUNTIME_SOCK", self.sock())
            .env("AIEN_RUNTIME_STATE_DIR", self.root.join("state"))
            .env("AIEN_COMPOSE_DIR", self.compose())
            .env("AIEN_PROVENANCE_DIR", self.root.join("prov"))
            .env("AIEN_REQUIRE_CHECKPOINT", "0");
        for k in [
            "AIEN_OMEGA_DIR",
            "AIEN_OMEGA_GPU_LIB",
            "AIEN_REQUIRE_BLACKWELL",
            "AIEN_GPU_BACKEND",
            "AIEN_MODEL_PATH",
            "AIEN_TOKENIZER_PATH",
            "AIEN_MODEL_DIR",
            "AIEN_DEV_FALLBACK",
            SWITCH,
        ] {
            c.env_remove(k);
        }
        if dev {
            c.env("AIEN_DEV_FALLBACK", "1");
        }
        if let Some(v) = switch {
            c.env(SWITCH, v);
        }
        c
    }
    fn spawn(&self, dev: bool, switch: Option<&str>) -> Daemon {
        let f = std::fs::File::create(self.log()).unwrap();
        let mut c = self.cmd(dev, switch);
        c.arg("daemon")
            .process_group(0)
            .stdin(Stdio::null())
            .stdout(f.try_clone().unwrap())
            .stderr(f);
        Daemon(c.spawn().expect("spawn daemon"))
    }
    fn text(&self) -> String {
        std::fs::read_to_string(self.log()).unwrap_or_default()
    }

    /// The daemon must exit on its own, refused, without ever binding the socket.
    fn refused(&self, dev: bool, switch: Option<&str>) -> String {
        let mut d = self.spawn(dev, switch);
        for _ in 0..1200 {
            if let Some(st) = d.0.try_wait().unwrap() {
                let text = self.text();
                assert!(
                    !st.success(),
                    "a refused start must exit non-zero ({st:?}): {text}"
                );
                assert!(
                    !self.sock().exists(),
                    "socket bound before the refusal: {text}"
                );
                return text;
            }
            assert!(
                !self.sock().exists(),
                "the daemon bound its socket: {}",
                self.text()
            );
            std::thread::sleep(Duration::from_millis(50));
        }
        panic!("the daemon did not refuse within 60 s: {}", self.text());
    }

    /// The daemon must serve; returns its startup log, then stops it.
    fn serves(&self, dev: bool, switch: Option<&str>) -> String {
        let mut d = self.spawn(dev, switch);
        for _ in 0..1200 {
            std::thread::sleep(Duration::from_millis(50));
            if let Some(st) = d.0.try_wait().unwrap() {
                panic!("daemon exited early ({st:?}): {}", self.text());
            }
            let text = self.text();
            if self.sock().exists() && text.contains("Authorize MAC:") {
                return text;
            }
        }
        panic!("the daemon did not come up in 60 s: {}", self.text());
    }
}

/// A spawned daemon, killed and reaped when dropped: on return and on a
/// failed assert alike, so a failing check never leaves it running.
struct Daemon(Child);

impl Drop for Daemon {
    fn drop(&mut self) {
        let _ = self.0.kill();
        let _ = self.0.wait();
    }
}

fn key_path(compose: &Path) -> String {
    compose.join("approval-desk.key").display().to_string()
}

#[test]
fn d1_no_desk_key_refuses_startup() {
    let r = Rig::new();
    for dev in [false, true] {
        let text = r.refused(dev, None);
        assert!(text.contains(&key_path(&r.compose())), "dev={dev}: {text}");
        assert!(text.contains("desk-key --create"), "dev={dev}: {text}");
    }
    // "1" is the same requirement, spelled out.
    let text = r.refused(false, Some("1"));
    assert!(text.contains(&key_path(&r.compose())), "{text}");
    assert!(
        !Path::new(&key_path(&r.compose())).exists(),
        "startup must never mint a key"
    );
}

#[test]
fn d2_dev_opt_out_is_refused_in_a_strict_run() {
    let r = Rig::new();
    if aien_inference_abi::strict::DEV_FALLBACK {
        // A --features dev-fallback build is never strict: there the opt-out serves.
        let text = r.serves(false, Some("0"));
        assert!(text.contains("Authorize MAC: OFF, DEV OPT-OUT"), "{text}");
        return;
    }
    let text = r.refused(false, Some("0"));
    assert!(text.contains(SWITCH), "{text}");
    assert!(text.contains("release qualification"), "{text}");
    assert!(text.contains("AIEN_DEV_FALLBACK"), "{text}");
}

#[test]
fn d3_dev_opt_out_starts_and_says_so() {
    let r = Rig::new();
    let text = r.serves(true, Some("0"));
    let line = text
        .lines()
        .find(|l| l.contains("Authorize MAC:"))
        .unwrap_or_default();
    assert!(line.contains("Authorize MAC: OFF, DEV OPT-OUT"), "{line}");
    assert!(
        line.contains("not valid for release qualification"),
        "{line}"
    );
    assert!(
        !Path::new(&key_path(&r.compose())).exists(),
        "startup must never mint a key"
    );
}

#[test]
fn d4_with_the_desk_key_the_daemon_serves_with_the_mac_on() {
    let r = Rig::new();
    let o = r
        .cmd(false, None)
        .args(["compose", "desk-key", "--create", "1"])
        .output()
        .unwrap();
    assert!(o.status.success(), "{}", String::from_utf8_lossy(&o.stdout));
    assert!(Path::new(&key_path(&r.compose())).exists());
    let text = r.serves(false, None);
    let line = text
        .lines()
        .find(|l| l.contains("Authorize MAC:"))
        .unwrap_or_default();
    assert!(line.starts_with("Authorize MAC: on"), "{line}");
}

/// Daemons still running on this socket (their environment names it).
fn daemons_on(sock: &Path) -> Vec<String> {
    let want = format!("AIEN_RUNTIME_SOCK={}", sock.display());
    let mut pids = Vec::new();
    for e in std::fs::read_dir("/proc").unwrap().flatten() {
        let pid = e.file_name().to_string_lossy().to_string();
        if !pid.chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if let Ok(env) = std::fs::read(e.path().join("environ")) {
            if env.split(|b| *b == 0).any(|v| v == want.as_bytes()) {
                pids.push(pid);
            }
        }
    }
    pids
}

/// A helper that fails must not leave its daemon running: here `refused`
/// meets a daemon that serves (the dev opt-out), so it panics.
#[test]
fn a_failed_check_stops_its_daemon() {
    let r = Rig::new();
    let failed = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        r.refused(true, Some("0"));
    }));
    assert!(failed.is_err(), "the daemon was expected to serve");
    let left = daemons_on(&r.sock());
    for pid in &left {
        let _ = Command::new("kill").args(["-KILL", pid]).status();
    }
    assert!(
        left.is_empty(),
        "the failed check left daemon(s) {left:?} running"
    );
}

/// sc#414 defect 1: `aien start` itself (not just the daemon it spawns) refuses
/// without a desk key, in plain words, naming the exact command, and never
/// leaves a daemon or a socket behind.
#[test]
fn s1_aien_start_without_desk_key_refuses_with_the_command() {
    let rig = Rig::new();
    let out = rig
        .cmd(false, None)
        .arg("start")
        .stdin(Stdio::null())
        .output()
        .expect("run aien start");
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "start must refuse: {all}");
    assert!(all.contains("AIEN will not start yet"), "{all}");
    assert!(all.contains("aien compose desk-key --create 1"), "{all}");
    assert!(!rig.sock().exists(), "no daemon may be left: {all}");
}

/// sc#414 defect 2: `aien harness` refuses with a non-zero exit; it never
/// reports a pass, and never calls the non-existent `spark-harness check`.
#[test]
fn s2_aien_harness_refuses_not_shipped() {
    let rig = Rig::new();
    let out = rig.cmd(false, None).arg("harness").output().unwrap();
    let all = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(!out.status.success(), "{all}");
    assert!(all.contains("not shipped in this release"), "{all}");
}
