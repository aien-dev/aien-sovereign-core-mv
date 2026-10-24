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

//! Callback safety across the rxc_host boundary, against the real librx_compose.a:
//! a failing or panicking Skill or verifier is contained and reported as a failed alternative
//! (never unwinds into C), repeated open/close neither leaks handle slots nor keeps callback
//! state alive, and several homes driven from several threads at once keep their own
//! callbacks apart (the C side routes ctx-less trampolines through a shared slot table).
//!
//! Ignored unless the library is linked. Note: the workspace release profile sets
//! `panic = "abort"`, so in a release binary a panicking callback aborts the process instead
//! (also never an unwind across C); these tests run in the unwinding test profile, where the
//! `catch_unwind` in the trampolines is what keeps the panic out of C.

use aien_omega_compose::*;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;

const ROOT: &[u8] = b"abi-qual-callback-test-machine";

fn propose(t: u64) -> u64 {
    t.wrapping_mul(7).wrapping_add(3)
}

#[test]
#[cfg_attr(
    not(has_omega_compose),
    ignore = "needs librx_compose.a (AIEN_OMEGA_COMPOSE_DIR)"
)]
fn panicking_skill_is_contained_and_reported_as_failure() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _) = Compose::open(&tmp.path().join("h"), RootKind::Provisioned, ROOT, 1).unwrap();
    c.register_skill("abi.panics", None, 1, |_| panic!("skill panic"))
        .unwrap();
    let r = c.run(0xA1, 1_000_000).unwrap();
    assert_eq!(r.committed, 0, "a panicking Skill must not commit: {r:?}");
    assert_ne!(r.outcome, OUT_COMMITTED);
    assert_eq!(r.result, 0);
    // The handle is still usable after the contained panic.
    let r2 = c.run(0xA2, 2_000_000).unwrap();
    assert_eq!(r2.committed, 0);
}

#[test]
#[cfg_attr(
    not(has_omega_compose),
    ignore = "needs librx_compose.a (AIEN_OMEGA_COMPOSE_DIR)"
)]
fn failing_and_zero_results_do_not_commit() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _) = Compose::open(&tmp.path().join("h"), RootKind::Provisioned, ROOT, 2).unwrap();
    c.register_skill("abi.none", None, 1, |_| None).unwrap();
    c.register_skill("abi.zero", None, 2, |_| Some(0)).unwrap();
    let r = c.run(0xB1, 1_000_000).unwrap();
    assert_eq!(r.committed, 0, "{r:?}");
    assert_eq!(r.aegis_pass_mask, 0, "{r:?}");
}

#[test]
#[cfg_attr(
    not(has_omega_compose),
    ignore = "needs librx_compose.a (AIEN_OMEGA_COMPOSE_DIR)"
)]
fn panicking_verifier_fails_the_contract() {
    let tmp = tempfile::tempdir().unwrap();
    let (mut c, _) = Compose::open(&tmp.path().join("h"), RootKind::Provisioned, ROOT, 3).unwrap();
    c.register_skill("abi.ok", None, 1, |t| Some(propose(t)))
        .unwrap();
    c.set_verify(|_, _| panic!("verifier panic")).unwrap();
    let r = c.run(0xC1, 1_000_000).unwrap();
    assert_eq!(r.committed, 0, "{r:?}");
    assert_eq!(r.aegis_pass_mask, 0, "{r:?}");
}

#[test]
#[cfg_attr(
    not(has_omega_compose),
    ignore = "needs librx_compose.a (AIEN_OMEGA_COMPOSE_DIR)"
)]
fn repeated_open_close_releases_slots_and_callback_state() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("h");
    let state = Arc::new(AtomicU64::new(0));
    // More cycles than RXC_HOST_MAX_HANDLES (4): a leaked slot would refuse with E_FULL.
    for i in 0..12u64 {
        let s = state.clone();
        let (mut c, info) = Compose::open(&home, RootKind::Provisioned, ROOT, 10 + i).unwrap();
        assert_eq!(info.abi_version, ffi::RXC_HOST_ABI_VERSION);
        c.register_skill("abi.count", None, 1, move |t| {
            s.fetch_add(1, Ordering::SeqCst);
            Some(propose(t))
        })
        .unwrap();
        let r = c.run(0xD000 + i, 1_000_000 * (i + 1)).unwrap();
        assert_eq!(r.committed, 1, "cycle {i}: {r:?}");
        assert_eq!(r.result, propose(0xD000 + i));
        drop(c);
        // The closure (and its clone of `state`) is freed with the handle.
        assert_eq!(
            Arc::strong_count(&state),
            1,
            "cycle {i}: callback state outlived close"
        );
    }
    assert_eq!(state.load(Ordering::SeqCst), 12);
}

#[test]
#[cfg_attr(
    not(has_omega_compose),
    ignore = "needs librx_compose.a (AIEN_OMEGA_COMPOSE_DIR)"
)]
fn concurrent_homes_keep_their_callbacks_apart() {
    let tmp = Arc::new(tempfile::tempdir().unwrap());
    let calls = Arc::new(AtomicUsize::new(0));
    let threads: Vec<_> = (0..3u64)
        .map(|k| {
            let tmp = tmp.clone();
            let calls = calls.clone();
            std::thread::spawn(move || {
                let home = tmp.path().join(format!("h{k}"));
                let (mut c, _) =
                    Compose::open(&home, RootKind::Provisioned, ROOT, 100 + k).unwrap();
                let tag = k << 56;
                let seen = Arc::new(AtomicU64::new(0));
                let s2 = seen.clone();
                let calls2 = calls.clone();
                c.register_skill("abi.tagged", None, 1, move |t| {
                    calls2.fetch_add(1, Ordering::SeqCst);
                    s2.store(t, Ordering::SeqCst);
                    Some(propose(t) | 1)
                })
                .unwrap();
                // A verifier that only accepts this home's own tag: a callback routed to the
                // wrong handle would fail the contract.
                c.set_verify(move |t, r| t >> 56 == k && r == (propose(t) | 1))
                    .unwrap();
                for i in 1..=5u64 {
                    let task = tag | i;
                    let r = c.run(task, i * 1_000_000).unwrap();
                    assert_eq!(r.committed, 1, "home {k} goal {i}: {r:?}");
                    assert_eq!(r.result, propose(task) | 1);
                    assert_eq!(seen.load(Ordering::SeqCst), task);
                }
            })
        })
        .collect();
    for t in threads {
        t.join().expect("worker thread");
    }
    assert_eq!(calls.load(Ordering::SeqCst), 15);
}
