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

//! ALLEN identity gate wired into the compose-home open (compose/bridge.rs `allen_gate`).
//!
//! Refusals exit the process (merge-control condition b), so every case runs
//! the daemon open path in a CHILD process: this test binary re-executes
//! itself on `child_open` with the environment under test, and the parent
//! checks exit code and output. With the stub omega library (no
//! librx_compose.a) the compose home cannot open at all, so every case
//! reports NOT_RUN and returns; with `AIEN_OMEGA_COMPOSE_DIR` set they run.
#[path = "support/home_guard.rs"]
mod home_guard;

#[path = "../../aien-allen/tests/support/mod.rs"]
mod support;

use aien_allen::binding::{self, pin_path};
use aien_allen::{hex, ENV_ADOPT, ENV_SUBJECT, EXIT_REFUSED, NOT_ENGAGED_LINE};
use aien_omega_compose::{Compose, RootKind, LINKED};
use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

fn proposer() -> ComposeProposer {
    Arc::new(|_p: &str, _l: std::time::Duration| {
        Ok(Generation {
            text: "filename: NOTES.md\nnote\n".into(),
            tokens: 4,
            finish_reason: Some("eos".into()),
            ..Default::default()
        })
    })
}

/// The child: open the home twice (as two daemon requests would) and note once.
#[test]
fn child_open() {
    let Ok(home) = std::env::var("ALLEN_SPINE_CHILD_HOME") else {
        return; // not the child
    };
    let b = ComposeBridge::new(PathBuf::from(home), proposer(), "test");
    let r = b.note("constraint", "keep changes inside the workspace", &[]);
    println!(
        "CHILD_NOTE_OK={}",
        matches!(r, aien_runtime::control::ControlResponse::ComposeNoted(_))
    );
    if !matches!(r, aien_runtime::control::ControlResponse::ComposeNoted(_)) {
        println!("CHILD_NOTE_RESPONSE={r:?}");
    }
    let r2 = b.recall(&[], None);
    println!(
        "CHILD_RECALL_OK={}",
        matches!(
            r2,
            aien_runtime::control::ControlResponse::ComposeRecalled(_)
        )
    );
}

struct Out {
    code: i32,
    stdout: String,
    stderr: String,
}

fn run_child(home: &Path, env: &[(&str, &str)]) -> Out {
    let mut c = Command::new(std::env::current_exe().unwrap());
    c.args(["--exact", "child_open", "--nocapture", "--test-threads=1"])
        .env("ALLEN_SPINE_CHILD_HOME", home)
        .env_remove(ENV_SUBJECT)
        .env_remove(ENV_ADOPT);
    for (k, v) in env {
        c.env(k, v);
    }
    let o = c.output().unwrap();
    Out {
        code: o.status.code().unwrap_or(-1),
        stdout: String::from_utf8_lossy(&o.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&o.stderr).into_owned(),
    }
}

fn skip() -> bool {
    if !LINKED {
        println!("NOT_RUN: stub omega build (no librx_compose.a); set AIEN_OMEGA_COMPOSE_DIR");
        return true;
    }
    false
}

fn names(dir: &Path) -> Vec<String> {
    let mut v: Vec<String> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    v.sort();
    v
}

/// (machine id, digest of Cortex record 1) of a home the daemon already opened.
fn home_facts(home: &Path) -> ([u8; 32], [u8; 32]) {
    let mut name = home.as_os_str().to_owned();
    name.push(".machine-root");
    let root = std::fs::read(PathBuf::from(name)).unwrap();
    let (mut c, info) = Compose::open(home, RootKind::Provisioned, &root, 0xA1E4_0001).unwrap();
    let d = c.record(1).unwrap().digest;
    assert_ne!(d, [0u8; 32], "record 1 has a digest");
    (info.machine_id, d)
}

fn subject_for(dir: &Path, lineage: [u8; 32]) -> (PathBuf, support::Fx) {
    let mut f = support::fx("spine");
    f.cortex = lineage;
    let c = support::chain(&f, 2);
    (
        support::write_head(&dir.join("subject.bin"), c.last().unwrap()),
        f,
    )
}

#[test]
fn not_engaged_path_is_unchanged() {
    if skip() {
        return;
    }
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("compose");
    let o = run_child(&home, &[]);
    assert_eq!(o.code, 0, "{}{}", o.stdout, o.stderr);
    assert!(o.stdout.contains("CHILD_NOTE_OK=true") && o.stdout.contains("CHILD_RECALL_OK=true"));
    // Exactly one not-engaged line even though the home opened more than once
    // per process, and the line is the agreed text.
    assert_eq!(
        o.stdout.matches(NOT_ENGAGED_LINE).count(),
        1,
        "{}",
        o.stdout
    );
    assert!(!o.stderr.contains("ALLEN"), "{}", o.stderr);
    // No ALLEN artifact beside the home: the gate wrote and read nothing.
    let beside = names(t.path());
    assert!(beside.iter().all(|n| !n.contains("allen")), "{beside:?}");
    // Reopening gives the same home and the same record count (second run adds
    // one more note; nothing else).
    let before = names(&home);
    let o2 = run_child(&home, &[]);
    assert_eq!(o2.code, 0);
    assert_eq!(before, names(&home));
}

#[test]
fn engaged_without_pin_or_adopt_is_fatal_and_creates_nothing() {
    let _home = home_guard::home_slot();
    if skip() {
        return;
    }
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("compose");
    assert_eq!(run_child(&home, &[]).code, 0); // creates a journal
    let (_, lin) = home_facts(&home);
    let (subj, _) = subject_for(t.path(), lin);
    let o = run_child(&home, &[(ENV_SUBJECT, subj.to_str().unwrap())]);
    assert_eq!(o.code, EXIT_REFUSED, "{}{}", o.stdout, o.stderr);
    assert!(
        o.stderr.contains("FATAL ALLEN refused: PinAbsent"),
        "{}",
        o.stderr
    );
    assert!(!pin_path(&home).exists(), "no silent adoption");
}

#[test]
fn engaged_missing_subject_is_fatal_never_minted() {
    if skip() {
        return;
    }
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("compose");
    assert_eq!(run_child(&home, &[]).code, 0);
    let missing = t.path().join("none.bin");
    let o = run_child(&home, &[(ENV_SUBJECT, missing.to_str().unwrap())]);
    assert_eq!(o.code, EXIT_REFUSED);
    assert!(
        o.stderr.contains("FATAL ALLEN refused: Absent"),
        "{}",
        o.stderr
    );
    assert!(!missing.exists() && !pin_path(&home).exists());
}

#[test]
fn adopt_restart_machine_change_and_foreign_journal() {
    let _home = home_guard::home_slot();
    if skip() {
        return;
    }
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("compose");
    assert_eq!(run_child(&home, &[]).code, 0);
    let (machine, lin) = home_facts(&home);
    let (subj, f) = subject_for(t.path(), lin);
    let s = subj.to_str().unwrap();
    // Wrong adopt value: fatal, no pin.
    let wrong = hex(&support::h32("someone else"));
    let o = run_child(&home, &[(ENV_SUBJECT, s), (ENV_ADOPT, &wrong)]);
    assert_eq!(o.code, EXIT_REFUSED);
    assert!(o.stderr.contains("AdoptMismatch"), "{}", o.stderr);
    assert!(!pin_path(&home).exists());
    // Operator adoption with the right agent id: engaged, pin written.
    let agent = hex(&f.agent);
    let o = run_child(&home, &[(ENV_SUBJECT, s), (ENV_ADOPT, &agent)]);
    assert_eq!(o.code, 0, "{}{}", o.stdout, o.stderr);
    assert!(o.stdout.contains("ADOPTED"), "{}", o.stdout);
    let pin = binding::read(&pin_path(&home)).unwrap().unwrap();
    assert_eq!(pin.agent, f.agent);
    assert_eq!(pin.machine_id, machine);
    // Restart without the adopt variable: same identity, pin unchanged.
    let pin_bytes = std::fs::read(pin_path(&home)).unwrap();
    let o = run_child(&home, &[(ENV_SUBJECT, s)]);
    assert_eq!(o.code, 0, "{}{}", o.stdout, o.stderr);
    assert!(o.stdout.contains(&format!("agent={agent}")));
    assert_eq!(pin_bytes, std::fs::read(pin_path(&home)).unwrap());
    // The adopt variable is refused once a pin exists.
    let o = run_child(&home, &[(ENV_SUBJECT, s), (ENV_ADOPT, &agent)]);
    assert_eq!(o.code, EXIT_REFUSED);
    assert!(o.stderr.contains("AdoptRefusedPinExists"), "{}", o.stderr);
    // Corrupt the pin: fatal.
    let mut bad = pin_bytes.clone();
    bad[20] ^= 1;
    std::fs::write(pin_path(&home), &bad).unwrap();
    let o = run_child(&home, &[(ENV_SUBJECT, s)]);
    assert_eq!(o.code, EXIT_REFUSED);
    assert!(o.stderr.contains("PinDamaged"), "{}", o.stderr);
    // Machine id differs from the pin: fatal, never rebound.
    let mut other = pin.clone();
    other.machine_id = [0xEE; 32];
    binding::write_atomic(&pin_path(&home), &other).unwrap();
    let o = run_child(&home, &[(ENV_SUBJECT, s)]);
    assert_eq!(o.code, EXIT_REFUSED);
    assert!(o.stderr.contains("MachineChanged"), "{}", o.stderr);
    assert_eq!(
        binding::read(&pin_path(&home)).unwrap().unwrap(),
        other,
        "not rebound"
    );
    // Foreign journal: restore the pin, point at another home's journal.
    binding::write_atomic(&pin_path(&home), &pin).unwrap();
    let home2 = t.path().join("compose2");
    assert_eq!(run_child(&home2, &[]).code, 0);
    let (_, lin2) = home_facts(&home2);
    // Two fresh homes may share record-1 content; only test when they differ.
    if lin2 != lin {
        let o = run_child(&home2, &[(ENV_SUBJECT, s), (ENV_ADOPT, &agent)]);
        assert_eq!(o.code, EXIT_REFUSED);
        assert!(o.stderr.contains("LineageMismatch"), "{}", o.stderr);
        assert!(!pin_path(&home2).exists());
    } else {
        println!("NOT_RUN: foreign-journal row (two fresh journals have the same record 1)");
    }
}
