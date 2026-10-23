//! Host-bridge protocol refusals; shell peers are test doubles, never policy fallbacks.
use aien_heartbeat::weight_policy::NativePolicy;
use std::os::unix::fs::PermissionsExt;
use std::{
    path::PathBuf,
    sync::atomic::{AtomicUsize, Ordering},
    time::{Duration, Instant},
};
static NEXT: AtomicUsize = AtomicUsize::new(0);
struct Peer(PathBuf);
impl Peer {
    fn new(body: &str) -> Self {
        let dir = std::env::temp_dir().join(format!(
            "weight-policy-{}-{}",
            std::process::id(),
            NEXT.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o700)).unwrap();
        let path = dir.join("peer");
        std::fs::write(&path, format!("#!/bin/sh\n{body}\n")).unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(path)
    }
}
impl Drop for Peer {
    fn drop(&mut self) {
        std::fs::remove_dir_all(self.0.parent().unwrap()).unwrap();
    }
}
const FRAME: [u64; 10] = [1, 0, 1, 1, 0, 0, 0, 0, 0, 0];
const GOOD: &str = r"printf '\001\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'";
#[test]
fn bridge_refuses_missing_dead_short_and_out_of_bounds_peers() {
    assert!(NativePolicy::open(std::path::Path::new("/nonexistent-weight-flow-policy")).is_err());
    for body in [
        "exit 0",
        r"printf '\001'",
        r"printf '\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000\000'",
    ] {
        let p = Peer::new(body);
        assert!(NativePolicy::open(&p.0).is_err());
    }
}
#[test]
fn bridge_deadline_is_bounded() {
    let p = Peer::new("exec sleep 30");
    let t = Instant::now();
    assert!(NativePolicy::open(&p.0).unwrap_err().contains("deadline"));
    assert!(t.elapsed() < Duration::from_secs(5));
}
#[test]
fn bridge_child_death_latches_failure_without_fallback() {
    let p = Peer::new(&format!(
        "dd bs=80 count=1 of=/dev/null 2>/dev/null; {GOOD}"
    ));
    let mut policy = NativePolicy::open(&p.0).unwrap();
    assert!(policy.decide(FRAME).is_err());
    assert!(policy.decide(FRAME).unwrap_err().contains("is failed"));
}
#[test]
fn bridge_reassembles_fragmented_replies_and_repeated_frames() {
    let p = Peer::new(
        r"while dd bs=80 count=1 of=/dev/null 2>/dev/null; do printf '\001\000\000\000\000\000\000\000'; sleep 0.01; printf '\000\000\000\000\000\000\000\000'; done",
    );
    let mut policy = NativePolicy::open(&p.0).unwrap();
    for _ in 0..3 {
        assert_eq!(policy.decide(FRAME).unwrap(), (1, 0));
    }
}
