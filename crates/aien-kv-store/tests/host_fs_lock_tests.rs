#![cfg(feature = "std")]
//! One open store per store directory (#521). ADR 0003 supports one runtime
//! per store directory; the host adapter enforces it with an exclusive lock
//! on `LOCK` in the store root, taken at open and held until the store drops.
//! Without it, a second opener's recovery deletes the first opener's
//! in-flight temp files and the two in-memory indexes drift apart.

mod common;
#[path = "common/host_fs_rig.rs"]
mod rig;

use aien_kv_store::host_fs::{HostFsConfig, HostFsStore};
use aien_platform::PlatformError;
use rig::Fx;

/// The child process of the cross-process test reads its store root here.
const CHILD_ROOT_ENV: &str = "AIEN_KV_LOCK_TEST_CHILD_ROOT";

fn in_use(r: Result<HostFsStore, PlatformError>) -> bool {
    matches!(r, Err(PlatformError::TaskRejected(ref m)) if m.contains("in use"))
}

#[test]
fn a_second_open_of_a_held_store_directory_is_refused() {
    let fx = Fx::new();
    let first = fx.open();
    assert!(fx.root().join("LOCK").exists());
    assert!(in_use(HostFsStore::open(fx.config())));
    drop(first);
}

#[test]
fn a_refused_second_open_leaves_the_first_stores_temp_files_alone() {
    let fx = Fx::new();
    let first = fx.open();
    // Stands for a park the first store has in flight.
    let part = fx.root().join("tmp").join("deadbeef.123.4.part");
    std::fs::write(&part, b"in flight").unwrap();

    assert!(in_use(HostFsStore::open(fx.config())));
    assert!(part.exists(), "the refused opener must not touch tmp/");

    // Once the first store is gone the folder opens, and recovery then
    // removes the leftover temp as a crash leftover (so it was the lock, not
    // luck, that kept it).
    drop(first);
    let second = fx.open();
    assert!(!part.exists());
    drop(second);
}

#[test]
fn the_lock_is_released_when_the_store_drops() {
    let fx = Fx::new();
    for _ in 0..3 {
        drop(fx.open());
    }
}

#[test]
fn a_store_directory_held_by_another_process_is_refused() {
    let fx = Fx::new();
    let first = fx.open();
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            "child_opens_the_store_directory_and_expects_refusal",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(CHILD_ROOT_ENV, fx.root())
        .output()
        .expect("spawn the child test process");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("1 passed"),
        "child process: {}\n{stdout}\n{}",
        out.status,
        String::from_utf8_lossy(&out.stderr)
    );
    drop(first);
}

/// Runs only as the child of the cross-process test (a no-op otherwise).
#[test]
fn child_opens_the_store_directory_and_expects_refusal() {
    let Some(root) = std::env::var_os(CHILD_ROOT_ENV) else {
        return;
    };
    let mut c = HostFsConfig::from_fields(root.into(), &rig::geometry()).unwrap();
    c.seal_key = Some(rig::test_key());
    assert!(in_use(HostFsStore::open(c)));
}
