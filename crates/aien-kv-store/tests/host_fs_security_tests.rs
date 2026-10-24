#![cfg(feature = "std")]
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

//! At-rest security of the host filesystem adapter (ADR 0003 section 19).
//! The objects hold K/V derived from user prompts, so the store directory is
//! private to the runtime identity: directories 0700, files 0600, an existing
//! directory that others can reach or that is a symlink is refused, and no
//! path inside the store is followed through a symlink.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::HostFsStore;
use aien_kv_store::{Incarnation, JobStage, KvStore, StoreState, Terminal};
use aien_platform::PlatformError;
use rig::*;
use std::os::unix::fs::{MetadataExt, PermissionsExt};
use std::path::Path;

fn mode_of(path: &Path) -> u32 {
    std::fs::symlink_metadata(path).unwrap().mode() & 0o777
}

fn chmod(path: &Path, mode: u32) {
    std::fs::set_permissions(path, std::fs::Permissions::from_mode(mode)).unwrap();
}

fn assert_private(path: &Path) {
    let m = mode_of(path);
    assert_eq!(
        m & 0o077,
        0,
        "{} is mode {m:o}; group and others must have no access",
        path.display()
    );
}

/// Every directory and file under `root`, root included.
fn walk(root: &Path, out: &mut Vec<std::path::PathBuf>) {
    out.push(root.to_path_buf());
    if std::fs::symlink_metadata(root).unwrap().is_dir() {
        for e in std::fs::read_dir(root).unwrap() {
            walk(&e.unwrap().path(), out);
        }
    }
}

fn refused(r: Result<HostFsStore, PlatformError>) -> bool {
    matches!(r, Err(PlatformError::AccessDenied))
}

#[test]
fn a_store_created_from_nothing_is_private_whatever_the_umask() {
    let _g = no_spawn();
    let fx = Fx::new();
    // After the fixture: the umask applies to what the store creates.
    if let Ok(m) = std::env::var("AIEN_KV_UMASK") {
        let m = u32::from_str_radix(&m, 8).unwrap();
        // SAFETY: umask only swaps the process file-creation mask.
        unsafe { libc::umask(m as libc::mode_t) };
    }
    let root = fx.root().join("fresh");
    let mut cfg = fx.config();
    cfg.root = root.clone();
    let s = HostFsStore::open(cfg).expect("open");
    recover_all(&s);
    // Publish one object so a real object file exists.
    let p = fx.put(&s, 1, 8);
    assert_eq!(
        terminal_of(&p.completions, p.job.unwrap()),
        Terminal::Done(JobStage::Durable)
    );

    let mut all = Vec::new();
    walk(&root, &mut all);
    assert!(all.iter().any(|p| p.ends_with("STORE_ID")));
    assert!(all
        .iter()
        .any(|p| p.extension().is_some_and(|e| e == "kvc")));
    for p in &all {
        assert_private(p);
    }
}

#[test]
fn an_existing_root_that_others_can_read_is_refused_and_left_alone() {
    let _g = no_spawn();
    let fx = Fx::new();
    chmod(fx.root(), 0o755);
    assert!(refused(HostFsStore::open(fx.config())));
    assert_eq!(mode_of(fx.root()), 0o755, "never chmod silently");
    assert!(
        std::fs::read_dir(fx.root()).unwrap().next().is_none(),
        "nothing is created under a refused root"
    );
}

#[test]
fn a_group_writable_root_is_refused() {
    let _g = no_spawn();
    let fx = Fx::new();
    chmod(fx.root(), 0o720);
    assert!(refused(HostFsStore::open(fx.config())));
}

#[test]
fn a_shared_subdirectory_of_a_private_root_is_refused() {
    let _g = no_spawn();
    for sub in ["objects", "tmp", "objects/7f"] {
        let fx = Fx::new();
        fx.prime();
        chmod(&fx.root().join(sub), 0o755);
        assert!(refused(HostFsStore::open(fx.config())), "{sub}");
    }
}

#[test]
fn a_root_that_is_a_symlink_is_refused() {
    let _g = no_spawn();
    let fx = Fx::new();
    let real = fx.root().join("real");
    std::fs::create_dir(&real).unwrap();
    chmod(&real, 0o700);
    let link = fx.root().join("link");
    std::os::unix::fs::symlink(&real, &link).unwrap();
    let mut cfg = fx.config();
    cfg.root = link;
    assert!(refused(HostFsStore::open(cfg)));
    assert!(std::fs::read_dir(&real).unwrap().next().is_none());
}

#[test]
fn a_subdirectory_replaced_by_a_symlink_is_refused() {
    let _g = no_spawn();
    let fx = Fx::new();
    fx.prime();
    let elsewhere = fx.root().join("elsewhere");
    std::fs::create_dir(&elsewhere).unwrap();
    chmod(&elsewhere, 0o700);
    let tmp = fx.root().join("tmp");
    std::fs::remove_dir(&tmp).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &tmp).unwrap();
    assert!(refused(HostFsStore::open(fx.config())));
}

#[test]
fn a_symlink_planted_as_an_object_is_never_read_or_served() {
    let _g = no_spawn();
    let fx = Fx::new();
    fx.prime();
    // A valid object of ours, moved outside the store and linked back in.
    let (key, header) = fx.chunk(1, 8);
    let payload = expected_payload(1, 8);
    fx.plant(&key, &header, &payload, BS);
    let inside = fx.object_file(&key);
    let outside = fx.root().join("outside.kvc");
    std::fs::rename(&inside, &outside).unwrap();
    std::os::unix::fs::symlink(&outside, &inside).unwrap();

    let s = fx.open();
    assert_eq!(s.capacity().state, StoreState::Open);
    assert_eq!(
        s.lookup(&[key]),
        vec![None],
        "a symlink is not one of our objects"
    );
    let (t, _mem) = fx.load(&s, key, &header);
    assert!(
        !matches!(t, Ok(Terminal::Done(_))),
        "a symlink is never served"
    );
    assert!(outside.exists(), "the link target is never touched");
}

#[test]
fn a_symlink_planted_at_a_temp_path_is_not_written_through() {
    let _g = no_spawn();
    let fx = Fx::new();
    let s = fx.open();
    let (key, header) = fx.chunk(1, 8);
    let victim = fx.root().join("victim");
    std::fs::write(&victim, b"keep").unwrap();
    // Temp names are <key hex>.<job>.<seq>.<epoch>.part; the first job is 1.
    let mem = Mem::new(blocks_for(8), 3);
    let (src, _h) = mem.source(blocks_for(8), tail_valid(8));
    let inc = Incarnation::seq(1);
    for job in 0..4u64 {
        let name = format!("{}.{job}.{}.{}.part", hex(&key.0 .0), inc.seq_id, inc.epoch);
        std::os::unix::fs::symlink(&victim, fx.root().join("tmp").join(name)).unwrap();
    }
    let _ = s.submit_store(header, src, inc);
    s.run_until_idle();
    drain(&s);
    assert!(
        std::fs::read(&victim).unwrap() == b"keep",
        "the store wrote through a planted temp-path symlink"
    );
}

#[test]
fn a_refused_open_creates_nothing_even_when_the_bad_directory_comes_late() {
    let _g = no_spawn();
    // Every existing directory is checked before anything is created, so a
    // shared last shard still leaves the store exactly as it was.
    let fx = Fx::new();
    let objects = fx.root().join("objects");
    std::fs::create_dir(&objects).unwrap();
    chmod(&objects, 0o700);
    let last = objects.join("ff");
    std::fs::create_dir(&last).unwrap();
    chmod(&last, 0o755);
    assert!(refused(HostFsStore::open(fx.config())));
    let mut all = Vec::new();
    walk(fx.root(), &mut all);
    all.sort();
    assert_eq!(all, vec![fx.root().to_path_buf(), objects, last]);
}

#[test]
fn a_root_whose_parent_others_can_write_is_refused_unless_sticky() {
    let _g = no_spawn();
    let fx = Fx::new();
    let mut cfg = fx.config();
    cfg.root = fx.root().join("store");
    for shared in [0o777, 0o770, 0o722] {
        chmod(fx.root(), shared);
        assert!(refused(HostFsStore::open(cfg.clone())), "parent {shared:o}");
        assert!(
            !cfg.root.exists(),
            "nothing created under parent {shared:o}"
        );
    }
    // A sticky shared directory (like /tmp) is trusted: others can create
    // their own entries but cannot rename or remove ours.
    chmod(fx.root(), 0o1777);
    let s = HostFsStore::open(cfg.clone()).expect("sticky parent is trusted");
    drop(s);
    assert_private(&cfg.root);
    chmod(fx.root(), 0o700);
}

#[test]
fn a_legacy_object_others_can_read_is_tightened_on_recovery() {
    let _g = no_spawn();
    let fx = Fx::new();
    fx.prime();
    let (key, header) = fx.chunk(1, 8);
    fx.plant(&key, &header, &expected_payload(1, 8), BS);
    let path = fx.object_file(&key);
    chmod(&path, 0o644);
    let s = fx.open();
    assert_eq!(
        mode_of(&path),
        0o600,
        "recovery tightens our own old objects"
    );
    assert!(s.lookup(&[key])[0].is_some(), "and still serves them");
    let (t, _mem) = fx.load(&s, key, &header);
    assert!(matches!(t, Ok(Terminal::Done(_))));
}

#[test]
fn an_object_owned_by_someone_else_is_never_served() {
    let _g = no_spawn();
    // Only root can give a file away; as a normal user this case cannot be
    // staged (and the private directories already keep others out).
    let fx = Fx::new();
    if std::fs::metadata(fx.root()).unwrap().uid() != 0 {
        return;
    }
    fx.prime();
    let (key, header) = fx.chunk(1, 8);
    fx.plant(&key, &header, &expected_payload(1, 8), BS);
    let path = fx.object_file(&key);
    std::os::unix::fs::chown(&path, Some(65534), Some(65534)).unwrap();
    let s = fx.open();
    assert_eq!(s.lookup(&[key]), vec![None]);
    assert!(path.exists(), "a file we did not write is left alone");
}

/// The umask is process-wide, so each umask case runs in a child process: the
/// test binary re-invokes itself for one test with `AIEN_KV_UMASK` set.
fn run_under_umask(umask: u32) {
    let _g = SPAWN.write().unwrap_or_else(|e| e.into_inner());
    let exe = std::env::current_exe().unwrap();
    let out = std::process::Command::new(exe)
        .args([
            "--exact",
            "a_store_created_from_nothing_is_private_whatever_the_umask",
            "--test-threads=1",
        ])
        .env("AIEN_KV_UMASK", format!("{umask:o}"))
        .output()
        .unwrap();
    assert!(
        out.status.success(),
        "umask {umask:o}: {}",
        String::from_utf8_lossy(&out.stdout)
    );
    assert!(String::from_utf8_lossy(&out.stdout).contains("1 passed"));
}

#[test]
fn a_permissive_umask_does_not_widen_the_store() {
    run_under_umask(0o000);
}

#[test]
fn a_narrow_umask_does_not_lock_the_store_out_of_itself() {
    run_under_umask(0o277);
}
