//! Coordinator and worker over a socket pair: the happy path, every refusal
//! the issue names (wrong peer, duplicate id, out of budget, stale result,
//! timeout, cancel, disconnect, denied task) and receipt tampering.

use aien_hetjob::coordinator::{dispatch, Job, OracleCheck, Outcome, Policy, Stats};
use aien_hetjob::receipt::{Platform, Receipt};
use aien_hetjob::store::Store;
use aien_hetjob::task::{self, Limits, TaskSpec};
use aien_hetjob::wire::{self, Ids, Inbound};
use aien_hetjob::worker::{Summary, Worker};
use aien_pi_spark_task as pst;
use ed25519_dalek::{SigningKey, VerifyingKey};
use sha2::{Digest, Sha256};
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::time::Duration;

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aien-hetjob-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32])
}

fn platform() -> Platform {
    Platform {
        os: "testos".into(),
        arch: "testarch".into(),
        cpus: 2,
        memory_bytes: 1 << 30,
        program_digest: [7; 32],
        version: "0.1.0".into(),
    }
}

fn worker(tag: &str, limits: Limits) -> Worker {
    Worker {
        key: key(1),
        limits,
        store: Store::open(&tmpdir(tag)).unwrap(),
        platform: platform(),
    }
}

fn ids(task: u8, idem: u8) -> Ids {
    Ids {
        cause_id: [0xC1; 32],
        origin_run_id: [0; 32],
        objective_id: [0x0B; 32],
        subject_id: [0; 32],
        task_id: [task; 16],
        idempotence_key: [idem; 16],
    }
}

fn policy(spec: &TaskSpec) -> Policy {
    Policy {
        deadline: Duration::from_secs(20),
        cancel_grace: Duration::from_secs(5),
        oracle: Some(task::oracle(spec)),
    }
}

const SMALL: TaskSpec = TaskSpec::MatmulU32 {
    m: 9,
    k: 13,
    n: 5,
    seed: 42,
};

/// Run one job against a real worker on a fresh connection.
fn run_job(
    w: &Worker,
    job: &Job,
    roster: &[VerifyingKey],
    p: &Policy,
) -> (Outcome, Stats, Summary) {
    let (c, s) = UnixStream::pair().unwrap();
    std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            let s2 = s.try_clone().unwrap();
            w.serve(s2, s).unwrap()
        });
        let r = c.try_clone().unwrap();
        let mut wr = c;
        let (o, st) = dispatch(r, &mut wr, job, roster, p).unwrap();
        wr.shutdown(std::net::Shutdown::Both).unwrap();
        (o, st, h.join().unwrap())
    })
}

/// A scripted worker: reads the request, then writes whatever `script` returns.
fn scripted(
    job: &Job,
    roster: &[VerifyingKey],
    p: &Policy,
    script: impl FnOnce(&Ids) -> Vec<Vec<u8>> + Send,
) -> (Outcome, Stats) {
    let (c, mut s) = UnixStream::pair().unwrap();
    std::thread::scope(|sc| {
        sc.spawn(move || {
            let req = wire::read_frame(&mut s).unwrap().unwrap();
            let m = wire::decode(&req).unwrap();
            for f in script(&Ids::of(&m.header)) {
                if wire::write_frame(&mut s, &f).is_err() {
                    return;
                }
            }
            // Keep reading (a cancel may arrive) until the coordinator closes.
            while let Ok(Some(_)) = wire::read_frame(&mut s) {}
        });
        let r = c.try_clone().unwrap();
        let mut wr = c;
        let out = dispatch(r, &mut wr, job, roster, p).unwrap();
        wr.shutdown(std::net::Shutdown::Both).unwrap();
        out
    })
}

/// The frames an honest worker sends for `spec`, signed by `k`.
fn honest_frames(ids: &Ids, spec: &TaskSpec, k: &SigningKey) -> (Vec<Vec<u8>>, Vec<u8>, [u8; 32]) {
    let out = task::oracle(spec);
    let receipt = Receipt {
        task_id: ids.task_id,
        idempotence_key: ids.idempotence_key,
        cause_id: ids.cause_id,
        kind: spec.kind_name().into(),
        input_digest: Sha256::digest(spec.encode()).into(),
        output_digest: out,
        started_unix_ms: 1,
        compute_ns: 2,
        threads: 1,
        platform: platform(),
    }
    .sign(k);
    (frames_for(ids, &receipt, &out), receipt, out)
}

fn frames_for(ids: &Ids, receipt: &[u8], out: &[u8; 32]) -> Vec<Vec<u8>> {
    let mut f = vec![
        wire::build(ids, wire::T_ACCEPTED, 0, &[]),
        wire::build(ids, wire::T_RUNNING, 1, &[]),
    ];
    let mut ord = 2;
    for c in receipt.chunks(pst::MAX_PAYLOAD) {
        f.push(wire::build(ids, wire::T_TEXT_CHUNK, ord, c));
        ord += 1;
    }
    let r: [u8; 32] = Sha256::digest(receipt).into();
    f.push(wire::build(ids, wire::T_RECEIPT_REFERENCE, ord, &r));
    f.push(wire::build(ids, wire::T_COMPLETED, ord + 1, out));
    f
}

fn rejected(o: &Outcome) -> String {
    match o {
        Outcome::Rejected(s) => s.clone(),
        other => panic!("expected Rejected, got {other:?}"),
    }
}

fn failed(o: &Outcome) -> String {
    match o {
        Outcome::Failed(s) => s.clone(),
        other => panic!("expected Failed, got {other:?}"),
    }
}

#[test]
fn verified_matmul_and_chain_match_the_oracle() {
    let w = worker("happy", Limits::default());
    let roster = [key(1).verifying_key()];
    for spec in [
        SMALL,
        TaskSpec::DigestChain {
            rounds: 5000,
            seed: [3; 32],
        },
    ] {
        let job = Job {
            spec,
            ids: ids(1, 1 + spec.encode()[0]),
        };
        let (o, st, sum) = run_job(&w, &job, &roster, &policy(&spec));
        match o {
            Outcome::Verified(v) => {
                assert_eq!(v.oracle, OracleCheck::Matched);
                assert_eq!(v.output, task::oracle(&spec));
                assert_eq!(v.receipt.platform, platform());
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(st.stale_ignored, 0);
        assert_eq!(sum.completed, 1);
    }
}

/// Frozen answers: every machine, every architecture, must give exactly
/// these digests. A mismatch on any host is a FAIL no matter what else agrees.
#[test]
fn golden_digests_are_frozen() {
    let hex = |d: [u8; 32]| d.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let cases = [
        (
            TaskSpec::MatmulU32 {
                m: 1,
                k: 1,
                n: 1,
                seed: 0,
            },
            GOLD_1X1X1,
        ),
        (
            TaskSpec::MatmulU32 {
                m: 64,
                k: 48,
                n: 32,
                seed: 7,
            },
            GOLD_64X48X32,
        ),
        (
            TaskSpec::MatmulU32 {
                m: 257,
                k: 129,
                n: 63,
                seed: 637,
            },
            GOLD_257X129X63,
        ),
        (
            TaskSpec::DigestChain {
                rounds: 100_000,
                seed: [0xAB; 32],
            },
            GOLD_CHAIN_100K,
        ),
    ];
    for (spec, want) in cases {
        assert_eq!(hex(task::oracle(&spec)), want, "{spec:?}");
    }
}

const GOLD_1X1X1: &str = "ba9dd3b4a9ec4978d83ac9e8fb8027811cd1159a2213ff8275979ced89ada395";
const GOLD_64X48X32: &str = "45a719e726d84e362c0d334773d0952540e4538d690456bb7f00e2ca9e851497";
const GOLD_257X129X63: &str = "156a39687d2d61627f1df918c4187ae5d923bd6c12b45d390397384880a0e25c";
const GOLD_CHAIN_100K: &str = "10430bf8db9967c61125c551188b695087b206817032692bb40e09d7c07f50b7";

#[test]
fn worker_algorithm_equals_oracle_for_awkward_shapes_and_threads() {
    let never = std::sync::atomic::AtomicBool::new(false);
    for (m, k, n) in [(1, 1, 1), (3, 1, 7), (17, 33, 1), (64, 64, 64), (5, 200, 9)] {
        for threads in [1, 2, 3, 8, 64] {
            let spec = TaskSpec::MatmulU32 {
                m,
                k,
                n,
                seed: u64::from(m) * 1000 + u64::from(n),
            };
            let ctl = task::Control {
                cancel: &never,
                deadline: std::time::Instant::now() + Duration::from_secs(60),
                threads,
            };
            assert_eq!(
                task::run(&spec, &ctl).unwrap(),
                task::oracle(&spec),
                "{spec:?} threads={threads}"
            );
        }
    }
}

#[test]
fn wrong_peer_is_rejected() {
    let w = worker("wrongpeer", Limits::default());
    let job = Job {
        spec: SMALL,
        ids: ids(2, 2),
    };
    let (o, _, _) = run_job(&w, &job, &[key(9).verifying_key()], &policy(&SMALL));
    assert!(rejected(&o).starts_with("WRONG_PEER:"), "{o:?}");
}

#[test]
fn duplicate_id_is_refused_and_same_request_replays() {
    let w = worker("dup", Limits::default());
    let roster = [key(1).verifying_key()];
    let first = Job {
        spec: SMALL,
        ids: ids(3, 3),
    };
    assert!(matches!(
        run_job(&w, &first, &roster, &policy(&SMALL)).0,
        Outcome::Verified(_)
    ));
    // Same idempotence key, different task id: refused, no second task.
    let other_task = Job {
        spec: SMALL,
        ids: ids(4, 3),
    };
    let (o, _, sum) = run_job(&w, &other_task, &roster, &policy(&SMALL));
    assert_eq!(failed(&o), "DUPLICATE_ID");
    assert_eq!(sum.completed, 0);
    // Same key, same task, different job body: also refused.
    let other_body = Job {
        spec: TaskSpec::MatmulU32 {
            m: 9,
            k: 13,
            n: 5,
            seed: 43,
        },
        ids: ids(3, 3),
    };
    assert_eq!(
        failed(&run_job(&w, &other_body, &roster, &policy(&other_body.spec)).0),
        "DUPLICATE_ID"
    );
    // Same request again: the stored receipt is replayed, nothing recomputed.
    let (o, _, sum) = run_job(&w, &first, &roster, &policy(&SMALL));
    assert!(matches!(o, Outcome::Verified(_)), "{o:?}");
    assert_eq!((sum.replayed, sum.completed), (1, 0));
}

#[test]
fn out_of_budget_is_refused_before_allocation() {
    let lim = Limits {
        max_memory_bytes: 1 << 20,
        max_work_units: 1 << 20,
        ..Limits::default()
    };
    let w = worker("budget", lim);
    let roster = [key(1).verifying_key()];
    let big_mem = TaskSpec::MatmulU32 {
        m: 1024,
        k: 1024,
        n: 1,
        seed: 1,
    };
    let o = run_job(
        &w,
        &Job {
            spec: big_mem,
            ids: ids(5, 5),
        },
        &roster,
        &policy(&SMALL),
    )
    .0;
    assert!(failed(&o).starts_with("OVER_BUDGET:MEMORY"), "{o:?}");
    let big_work = TaskSpec::DigestChain {
        rounds: (1 << 20) + 1,
        seed: [1; 32],
    };
    let o = run_job(
        &w,
        &Job {
            spec: big_work,
            ids: ids(6, 6),
        },
        &roster,
        &policy(&SMALL),
    )
    .0;
    assert!(failed(&o).starts_with("OVER_BUDGET:WORK"), "{o:?}");
    let empty = TaskSpec::MatmulU32 {
        m: 0,
        k: 4,
        n: 4,
        seed: 1,
    };
    let o = run_job(
        &w,
        &Job {
            spec: empty,
            ids: ids(7, 7),
        },
        &roster,
        &policy(&SMALL),
    )
    .0;
    assert_eq!(failed(&o), "OVER_BUDGET:EMPTY");
}

/// Send raw frames to a real worker, read `replies` answers, then close.
fn raw_exchange(w: &Worker, frames: &[Vec<u8>], replies: usize) -> (Vec<wire::Msg>, Summary) {
    let (mut c, s) = UnixStream::pair().unwrap();
    std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            let s2 = s.try_clone().unwrap();
            w.serve(s2, s).unwrap()
        });
        for f in frames {
            wire::write_frame(&mut c, f).unwrap();
        }
        let mut got = Vec::new();
        while got.len() < replies {
            let f = wire::read_frame(&mut c)
                .unwrap()
                .expect("worker closed early");
            got.push(wire::decode(&f).unwrap());
        }
        c.shutdown(std::net::Shutdown::Write).unwrap();
        while let Ok(Some(f)) = wire::read_frame(&mut c) {
            got.push(wire::decode(&f).unwrap());
        }
        (got, h.join().unwrap())
    })
}

#[test]
fn denied_task_wrong_side_and_wrong_effect_create_no_task() {
    let w = worker("denied", Limits::default());
    let i = ids(8, 8);
    let unknown_kind = wire::build(&i, wire::T_REQUEST, 0, &[9, 1, 2, 3]);
    let wrong_side = wire::build(&ids(9, 9), wire::T_COMPLETED, 0, &[0; 32]);
    let mut h = wire::decode(&wire::build(
        &ids(10, 10),
        wire::T_REQUEST,
        0,
        &SMALL.encode(),
    ))
    .unwrap()
    .header;
    h.effect_class = 2;
    let payload = SMALL.encode();
    let mut effect = vec![0u8; pst::HEADER_LEN + payload.len()];
    pst::encode(&h, &payload, &mut effect).unwrap();
    let (got, sum) = raw_exchange(&w, &[unknown_kind, wrong_side, effect], 3);
    let reasons: Vec<String> = got
        .iter()
        .map(|m| String::from_utf8_lossy(&m.payload).into_owned())
        .collect();
    assert!(
        got.iter()
            .all(|m| m.header.msg_type == wire::T_FAILED && m.header.ordinal == 0),
        "{reasons:?}"
    );
    assert_eq!(
        reasons,
        [
            "DENIED_TASK:UnknownKind(9)",
            "REFUSED:PST_WRONG_SIDE",
            "DENIED_EFFECT"
        ]
    );
    assert_eq!((sum.denied, sum.completed), (3, 0));
}

#[test]
fn corrupt_frame_is_dropped_and_the_next_request_still_runs() {
    let w = worker("corrupt", Limits::default());
    let mut bad = wire::build(&ids(11, 11), wire::T_REQUEST, 0, &SMALL.encode());
    let last = bad.len() - 1;
    bad[last] ^= 1; // payload no longer matches its digest
    let good = wire::build(&ids(12, 12), wire::T_REQUEST, 0, &SMALL.encode());
    let (got, sum) = raw_exchange(&w, &[bad, good], 6);
    assert_eq!(sum.refused_frames, 1);
    assert!(got.iter().all(|m| m.header.task_id == [12; 16]));
    assert_eq!(got.last().unwrap().header.msg_type, wire::T_COMPLETED);
}

/// A writer whose peer has hung up.
struct HungUp;

impl std::io::Write for HungUp {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::BrokenPipe.into())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

#[test]
fn peer_hanging_up_mid_reply_ends_the_session_cleanly() {
    // The input stays open, so the hang-up is seen only by the failed write
    // (the order the wrong-peer test can hit when the result is fast).
    let w = worker("hungup", Limits::default());
    let (mut c, s) = UnixStream::pair().unwrap();
    let req = wire::build(&ids(14, 14), wire::T_REQUEST, 0, &SMALL.encode());
    wire::write_frame(&mut c, &req).unwrap();
    let sum = w.serve(s, HungUp).expect("a hang-up is not a worker error");
    assert_eq!(sum.completed + sum.failed, 0);
}

#[test]
fn stale_frames_for_another_task_are_ignored() {
    let spec = SMALL;
    let job = Job {
        spec,
        ids: ids(13, 13),
    };
    let k = key(1);
    let (o, st) = scripted(&job, &[k.verifying_key()], &policy(&spec), |i| {
        let (mut f, _, _) = honest_frames(i, &spec, &k);
        let old = ids(99, 99);
        let (stale, _, _) = honest_frames(&old, &spec, &k);
        f.insert(0, stale.last().unwrap().clone()); // an old task's `completed`
        f
    });
    assert!(matches!(o, Outcome::Verified(_)), "{o:?}");
    assert_eq!(st.stale_ignored, 1);
}

#[test]
fn out_of_order_and_missing_receipt_are_rejected() {
    let k = key(1);
    let roster = [k.verifying_key()];
    let job = Job {
        spec: SMALL,
        ids: ids(14, 14),
    };
    let (o, _) = scripted(&job, &roster, &policy(&SMALL), |i| {
        vec![wire::build(i, wire::T_RUNNING, 1, &[])]
    });
    assert_eq!(rejected(&o), "OUT_OF_ORDER:expected 0 got 1");
    let (o, _) = scripted(&job, &roster, &policy(&SMALL), |i| {
        vec![
            wire::build(i, wire::T_ACCEPTED, 0, &[]),
            wire::build(i, wire::T_COMPLETED, 1, &task::oracle(&SMALL)),
        ]
    });
    assert_eq!(rejected(&o), "COMPLETED_WITHOUT_RECEIPT");
}

#[test]
fn tampered_receipts_and_wrong_outputs_are_rejected() {
    let k = key(1);
    let roster = [k.verifying_key()];
    let job = Job {
        spec: SMALL,
        ids: ids(15, 15),
    };
    let p = policy(&SMALL);
    // completed carries a different digest than the signed receipt.
    let (o, _) = scripted(&job, &roster, &p, |i| {
        let (_, receipt, mut out) = honest_frames(i, &SMALL, &k);
        out[0] ^= 1;
        frames_for(i, &receipt, &out)
    });
    assert_eq!(rejected(&o), "RECEIPT_MISMATCH:output_digest");
    // A receipt byte changed after the reference digest was taken.
    let (o, _) = scripted(&job, &roster, &p, |i| {
        let (mut f, _, _) = honest_frames(i, &SMALL, &k);
        let chunk = wire::decode(&f[2]).unwrap();
        let mut body = chunk.payload.clone();
        body[10] ^= 1;
        f[2] = wire::build(i, wire::T_TEXT_CHUNK, 2, &body);
        f
    });
    assert_eq!(rejected(&o), "RECEIPT_REFERENCE_MISMATCH");
    // A field edited under an honest reference digest: the signature fails.
    let (o, _) = scripted(&job, &roster, &p, |i| {
        let (_, receipt, out) = honest_frames(i, &SMALL, &k);
        let edited = String::from_utf8(receipt)
            .unwrap()
            .replace("\"compute_ns\":2", "\"compute_ns\":3");
        frames_for(i, edited.as_bytes(), &out)
    });
    assert_eq!(rejected(&o), "BAD_SIGNATURE");
    // A receipt honestly signed for another job body.
    let (o, _) = scripted(&job, &roster, &p, |i| {
        let other = TaskSpec::MatmulU32 {
            m: 9,
            k: 13,
            n: 5,
            seed: 1,
        };
        let (f, _, _) = honest_frames(i, &other, &k);
        f
    });
    assert_eq!(rejected(&o), "RECEIPT_MISMATCH:input_digest");
    // A worker that signs a wrong answer honestly: only the oracle catches it.
    let wrong = Policy {
        oracle: Some([0x55; 32]),
        ..p
    };
    let (o, _) = scripted(&job, &roster, &wrong, |i| honest_frames(i, &SMALL, &k).0);
    assert_eq!(rejected(&o), "ORACLE_MISMATCH");
}

#[test]
fn coordinator_deadline_cancels_the_running_job() {
    let w = worker("cancel", Limits::default());
    let long = TaskSpec::DigestChain {
        rounds: u32::MAX,
        seed: [5; 32],
    };
    let p = Policy {
        deadline: Duration::from_millis(200),
        cancel_grace: Duration::from_secs(10),
        oracle: None,
    };
    let (o, _, sum) = run_job(
        &w,
        &Job {
            spec: long,
            ids: ids(16, 16),
        },
        &[key(1).verifying_key()],
        &p,
    );
    assert!(matches!(o, Outcome::Cancelled), "{o:?}");
    assert_eq!(sum.cancelled, 1);
}

#[test]
fn worker_wall_limit_reports_timeout() {
    let lim = Limits {
        max_wall: Duration::from_millis(100),
        ..Limits::default()
    };
    let w = worker("timeout", lim);
    let long = TaskSpec::DigestChain {
        rounds: u32::MAX,
        seed: [6; 32],
    };
    let p = Policy {
        deadline: Duration::from_secs(30),
        cancel_grace: Duration::from_secs(5),
        oracle: None,
    };
    let (o, _, _) = run_job(
        &w,
        &Job {
            spec: long,
            ids: ids(17, 17),
        },
        &[key(1).verifying_key()],
        &p,
    );
    assert_eq!(failed(&o), "TIMEOUT");
}

#[test]
fn disconnect_and_silence_are_uncertain_never_failed() {
    let roster = [key(1).verifying_key()];
    let job = Job {
        spec: SMALL,
        ids: ids(18, 18),
    };
    // The worker accepts, then the link drops.
    let (c, mut s) = UnixStream::pair().unwrap();
    let h = std::thread::spawn(move || {
        let req = wire::read_frame(&mut s).unwrap().unwrap();
        let i = Ids::of(&wire::decode(&req).unwrap().header);
        wire::write_frame(&mut s, &wire::build(&i, wire::T_ACCEPTED, 0, &[])).unwrap();
    });
    let r = c.try_clone().unwrap();
    let mut wr = c;
    let (o, _) = dispatch(r, &mut wr, &job, &roster, &policy(&SMALL)).unwrap();
    h.join().unwrap();
    assert!(
        matches!(&o, Outcome::Uncertain(s) if s.starts_with("DISCONNECTED")),
        "{o:?}"
    );
    // A worker that never answers, even to the cancel.
    let p = Policy {
        deadline: Duration::from_millis(100),
        cancel_grace: Duration::from_millis(100),
        oracle: None,
    };
    let (o, _) = scripted(&job, &roster, &p, |_| Vec::new());
    assert!(
        matches!(&o, Outcome::Uncertain(s) if s.starts_with("DEADLINE")),
        "{o:?}"
    );
}

#[test]
fn a_second_worker_cannot_open_the_same_state_folder() {
    let d = tmpdir("lock");
    let _first = Store::open(&d).unwrap();
    let second = Store::open(&d).err().expect("second open must be refused");
    assert_eq!(second.kind(), std::io::ErrorKind::WouldBlock);
}

#[test]
fn frames_above_1024_bytes_are_refused_before_reading() {
    let mut src: &[u8] = &[0x04, 0x01, 0, 0];
    let e = wire::read_frame(&mut src).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidData);
    let (tx, rx) = std::sync::mpsc::channel();
    wire::pump(&[0x00, 0x03, 1, 2, 3][..], tx, |i| i);
    assert!(matches!(
        rx.recv().unwrap(),
        Inbound::Refused(pst::Refusal::BadLength)
    ));
}
