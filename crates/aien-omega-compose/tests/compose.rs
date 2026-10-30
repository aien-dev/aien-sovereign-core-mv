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

//! The omega C test (tests/runtime/rxc_host_abi_test.c) repeated from Rust:
//! run one goal, restart + recall, identity refusal, torn tail refused then
//! recovered, host notes recalled after restart. Ignored
//! unless librx_compose.a is linked (AIEN_OMEGA_COMPOSE_DIR or
//! AIEN_OMEGA_COMPOSE_LIB at build time; see build.rs / README.md).
use aien_omega_compose::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

const ROOT: &[u8] = b"np1-host-abi-test-machine";
fn propose(t: u64) -> u64 {
    t.wrapping_mul(3).wrapping_add(1)
}

fn open(dir: &std::path::Path, seen: Arc<AtomicU64>) -> Result<(Compose, Info), ComposeError> {
    let (mut c, info) = Compose::open(dir, RootKind::Provisioned, ROOT, 0x5E55)?;
    c.register_skill("np1.echo-proposal", None, 10, move |task| {
        seen.store(task, Ordering::SeqCst);
        Some(propose(task))
    })?;
    c.set_verify(|task, result| result == propose(task))?;
    Ok((c, info))
}

#[test]
#[cfg_attr(
    not(has_omega_compose),
    ignore = "needs librx_compose.a (AIEN_OMEGA_COMPOSE_DIR)"
)]
fn run_restart_recall_torn_tail() {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("home");
    let seen = Arc::new(AtomicU64::new(0));

    // T1: one goal
    let (mut c, i1) = open(&home, seen.clone()).unwrap();
    assert_eq!(i1.abi_version, 2);
    assert_eq!(i1.machine_id_was_stored, 0);
    let task = 0x4E50_3100_0000_0001u64;
    let r = c.run(task, 1_000_000).unwrap();
    assert_eq!(r.outcome, OUT_COMMITTED);
    assert_eq!(r.committed, 1);
    assert_eq!(seen.load(Ordering::SeqCst), task);
    assert_eq!(r.result, propose(task));
    assert_eq!((r.n_branches, r.winner, r.winner_skill), (1, 0, 0));
    assert_eq!(r.aegis_pass_mask & 1, 1);
    let cited = [r.cx_goal, r.cx_candidate[0], r.cx_evidence, r.cx_promotion];
    assert!(cited.iter().all(|&id| id != 0), "{cited:?}");
    let cited_digests: Vec<[u8; 32]> = cited
        .iter()
        .map(|&id| c.record(id).unwrap().digest)
        .collect();
    let (before, total) = c.recall(SUBJECT_STATE, 64).unwrap();
    assert!(!before.is_empty() && before.len() as u64 == total);
    let mid = c.info().unwrap().machine_id;
    drop(c);

    // T2: restart + recall (pre-restart records are an unchanged prefix)
    let (mut c, i2) = open(&home, seen.clone()).unwrap();
    assert_eq!(i2.machine_id_was_stored, 1);
    assert_eq!(i2.machine_id, mid);
    assert_eq!(i2.tail_torn, 0);
    let (after, _) = c.recall(SUBJECT_STATE, 64).unwrap();
    assert!(after.len() >= before.len());
    for (a, b) in after.iter().zip(before.iter()) {
        assert_eq!((a.id, a.digest, a.verified), (b.id, b.digest, 1));
    }
    for (id, d) in cited.iter().zip(cited_digests.iter()) {
        assert_eq!(&c.record(*id).unwrap().digest, d);
    }
    drop(c);

    // T3: another machine is refused on this home
    let other = Compose::open(&home, RootKind::Provisioned, b"np1-some-other-machine", 1);
    assert!(matches!(
        other.err(),
        Some(ComposeError::Code { code: -2, .. })
    ));

    // T5: a run's own tail torn: open refuses (E_TORN) without writing, stably;
    // recover cuts the torn record, records the cut, re-anchors; open succeeds
    // with the old records an unchanged prefix; a goal commits.
    let (mut c, _) = open(&home, seen.clone()).unwrap();
    let r = c.run(0x4E50_3100_0000_0003, 3_000_000).unwrap();
    assert_eq!(r.committed, 1);
    let n0 = c.info().unwrap().records;
    let all: Vec<Record> = (1..=n0).map(|id| c.record(id).unwrap()).collect();
    drop(c);
    let cx = home.join("cortex.cx");
    let len = std::fs::metadata(&cx).unwrap().len();
    let f = std::fs::OpenOptions::new().write(true).open(&cx).unwrap();
    f.set_len(len - 5).unwrap();
    drop(f);
    for _ in 0..2 {
        let refused = open(&home, seen.clone());
        assert!(matches!(
            refused.err(),
            Some(ComposeError::Code { code: -3, .. })
        ));
        assert_eq!(
            std::fs::metadata(&cx).unwrap().len(),
            len - 5,
            "refusal wrote"
        );
    }
    let rep = Compose::recover(&home, RootKind::Provisioned, ROOT).unwrap();
    assert_eq!(
        (rep.repaired, rep.tail_torn, rep.cause, rep.opens),
        (1, 1, -8, 1)
    );
    assert_eq!(rep.cut_hi, len - 5);
    assert_eq!(rep.cut_bytes_kept, rep.cut_hi - rep.cut_lo);
    assert!(rep.dropped_records >= 1);
    assert_eq!(rep.event_id, rep.records_kept + 1);
    let again = Compose::recover(&home, RootKind::Provisioned, ROOT).unwrap();
    assert_eq!(again.repaired, 0, "second recover is a no-op");
    let (mut c, ir) = open(&home, seen.clone()).unwrap();
    assert_eq!(ir.tail_torn, 0);
    c.info().unwrap();
    for rec in &all[..rep.records_kept as usize] {
        assert_eq!(c.record(rec.id).unwrap().digest, rec.digest);
    }
    let ev = c.record(rep.event_id).unwrap();
    assert_eq!(
        (ev.subject, ev.tag),
        (SUBJECT_HOST, ffi::RXC_HOST_TAG_REPAIR_TAIL)
    );
    let p = c.payload(rep.event_id).unwrap();
    assert_eq!(p[ffi::RXC_HOST_RP_CUT_LO], rep.cut_lo);
    assert_eq!(p[ffi::RXC_HOST_RP_DROPPED], rep.dropped_records);
    let r = c.run(0x4E50_3100_0000_0004, 4_000_000).unwrap();
    assert_eq!(r.committed, 1);

    // T7: host notes through the composition writer, recalled after restart
    let text = b"constraint: only touch files under the workspace";
    let cid = c.note(NoteKind::Constraint, [0; 4], text).unwrap();
    let eid = c
        .note(
            NoteKind::Effect,
            [r.cx_promotion, cid, 0, 0],
            b"effect: wrote a.txt",
        )
        .unwrap();
    assert!(eid > cid);
    assert!(c.note(NoteKind::Effect, [1 << 40, 0, 0, 0], b"x").is_err());
    let (before, _) = c.recall(SUBJECT_HOST, 64).unwrap();
    drop(c);
    let (mut c, i7) = open(&home, seen.clone()).unwrap();
    assert_eq!(i7.machine_id, mid);
    let (after, _) = c.recall(SUBJECT_HOST, 64).unwrap();
    assert_eq!(after.len(), before.len());
    for (a, b) in after.iter().zip(before.iter()) {
        assert_eq!((a.id, a.digest, a.verified), (b.id, b.digest, 1));
    }
    let got = note_bytes(&c.payload(cid).unwrap()).unwrap();
    assert_eq!(got, text);
    assert_eq!(c.record(eid).unwrap().links[..2], [r.cx_promotion, cid]);
}
