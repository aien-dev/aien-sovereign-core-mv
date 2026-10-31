//! The bounded read adapter over the deterministic fake source: verification
//! before ready, scrubbing on failure, every bound, deadlines, cancellation
//! and concurrent readers.

mod common;

use aien_expert_store::fake::{FakeSource, Fault};
use aien_expert_store::{
    CancelOutcome, IndexEntry, Outcome, Projection, ReadAdapter, ReadCompletion, ReadError,
    ReadLimits, StagingBuffer,
};
use common::*;
use std::io::ErrorKind;
use std::sync::Arc;
use std::time::{Duration, Instant};

const WAIT: Duration = Duration::from_secs(10);

fn limits() -> ReadLimits {
    ReadLimits {
        max_in_flight: 2,
        max_inflight_bytes: 1 << 20,
        max_object_bytes: 512 << 10,
        chunk_bytes: 4096,
    }
}

fn setup(lim: ReadLimits) -> (FakeSource, ReadAdapter, IndexEntry, Vec<u8>, Vec<u8>) {
    let fake = FakeSource::new();
    let (e, bytes, w, s) = artifact(key(1, 2, Projection::Up), 192, 256, 3);
    fake.insert(e.data_digest, bytes);
    let a = ReadAdapter::new(Arc::new(fake.clone()), lim).unwrap();
    (fake, a, e, w, s)
}

fn far() -> Instant {
    Instant::now() + Duration::from_secs(30)
}

fn buf(e: &IndexEntry) -> StagingBuffer {
    StagingBuffer::new(e.object_len as usize).unwrap()
}

fn one(a: &ReadAdapter, e: IndexEntry) -> ReadCompletion {
    let t = a.submit(e, buf(&e), far()).unwrap();
    let c = a.wait(WAIT).expect("terminal");
    assert_eq!(c.ticket, t);
    c
}

fn expect_failed(c: ReadCompletion) -> ReadError {
    match c.outcome {
        Outcome::Failed { error, buf } => {
            assert!(
                buf.as_slice().iter().all(|&b| b == 0),
                "buffer not scrubbed"
            );
            error
        }
        o => panic!("expected Failed, got {o:?}"),
    }
}

#[test]
fn ready_only_after_verification_and_bytes_round_trip() {
    let (_f, a, e, w, s) = setup(limits());
    match one(&a, e).outcome {
        Outcome::Ready(r) => {
            assert_eq!(r.weights(), &w[..]);
            assert_eq!(r.scales(), &s[..]);
            assert_eq!(r.header.key, e.key);
            assert_eq!(r.artifact().len() as u64, e.object_len);
        }
        o => panic!("{o:?}"),
    }
    assert_eq!(a.in_flight(), (0, 0));
}

#[test]
fn corruption_is_a_digest_mismatch_and_scrubs() {
    let (f, a, e, _, _) = setup(limits());
    for at in [0u64, 5000, e.object_len - 1] {
        f.set_fault(e.data_digest, Fault::CorruptByte(at));
        assert_eq!(
            expect_failed(one(&a, e)),
            ReadError::DigestMismatch,
            "at {at}"
        );
    }
}

#[test]
fn truncation_eof_and_io_errors_are_explicit() {
    let (f, a, e, _, _) = setup(limits());
    f.set_fault(e.data_digest, Fault::Truncate(e.object_len - 4096));
    assert_eq!(
        expect_failed(one(&a, e)),
        ReadError::LengthMismatch {
            expected: e.object_len,
            found: e.object_len - 4096
        }
    );
    f.set_fault(e.data_digest, Fault::EofAt(8192 + 100));
    assert_eq!(
        expect_failed(one(&a, e)),
        ReadError::UnexpectedEof {
            read: 8192 + 100,
            expected: e.object_len
        }
    );
    f.set_fault(e.data_digest, Fault::IoError(ErrorKind::Other));
    assert_eq!(expect_failed(one(&a, e)), ReadError::Io(ErrorKind::Other));
}

#[test]
fn short_reads_are_resumed_not_trusted() {
    let (f, a, e, w, _) = setup(limits());
    f.set_fault(e.data_digest, Fault::ShortReads(1000));
    match one(&a, e).outcome {
        Outcome::Ready(r) => assert_eq!(r.weights(), &w[..]),
        o => panic!("{o:?}"),
    }
    assert!(f.reads() as u64 >= e.object_len / 1000);
}

#[test]
fn wrong_hash_wrong_object_and_wrong_key_are_refused() {
    let (f, a, e, _, _) = setup(limits());
    // Index digest names no object.
    let mut missing = e;
    missing.data_digest[0] ^= 1;
    assert_eq!(expect_failed(one(&a, missing)), ReadError::NotFound);
    // The object stored under the digest is a different (valid) artifact.
    let (other, other_bytes, _, _) = artifact(key(1, 2, Projection::Up), 192, 256, 77);
    assert_ne!(other.data_digest, e.data_digest);
    f.insert(e.data_digest, other_bytes.clone());
    assert_eq!(expect_failed(one(&a, e)), ReadError::DigestMismatch);
    // The digest matches, but the index entry claims another key.
    f.insert(other.data_digest, other_bytes);
    let mut lying = other;
    lying.key = key(9, 9, Projection::Down);
    assert_eq!(expect_failed(one(&a, lying)), ReadError::HeaderMismatch);
}

#[test]
fn submit_bounds_refuse_and_return_the_buffer() {
    let (f, a, e, _, _) = setup(limits());
    // Insufficient capacity.
    let small = StagingBuffer::new(4096).unwrap();
    let r = a.submit(e, small, far()).unwrap_err();
    assert_eq!(
        r.error,
        ReadError::InsufficientCapacity {
            need: e.object_len,
            have: 4096
        }
    );
    assert_eq!(r.buf.capacity(), 4096);
    // Object larger than max_object_bytes.
    let mut huge = e;
    huge.object_len = 1 << 20;
    let r = a
        .submit(huge, StagingBuffer::new(1 << 20).unwrap(), far())
        .unwrap_err();
    assert_eq!(r.error, ReadError::ObjectTooLarge);
    // Malformed entry length.
    let mut odd = e;
    odd.object_len = 4097;
    assert_eq!(
        a.submit(odd, buf(&e), far()).unwrap_err().error,
        ReadError::InvalidEntry
    );
    // Deadline already passed.
    let r = a.submit(e, buf(&e), Instant::now()).unwrap_err();
    assert_eq!(r.error, ReadError::DeadlineExceeded);
    // Queue depth: hold the device, fill both slots, the third is refused.
    f.hold();
    let t1 = a.submit(e, buf(&e), far()).unwrap();
    let t2 = a.submit(e, buf(&e), far()).unwrap();
    assert_eq!(
        a.submit(e, buf(&e), far()).unwrap_err().error,
        ReadError::QueueFull
    );
    assert_eq!(a.in_flight(), (2, 2 * e.object_len));
    f.release();
    let mut got = vec![a.wait(WAIT).unwrap().ticket, a.wait(WAIT).unwrap().ticket];
    got.sort();
    assert_eq!(got, vec![t1, t2]);
    assert_eq!(a.in_flight(), (0, 0));
}

#[test]
fn inflight_byte_budget_is_enforced() {
    let fake = FakeSource::new();
    let (e, bytes, _, _) = artifact(key(0, 0, Projection::Gate), 192, 256, 1);
    fake.insert(e.data_digest, bytes);
    let lim = ReadLimits {
        max_in_flight: 8,
        max_inflight_bytes: e.object_len * 2 + 1,
        ..limits()
    };
    let a = ReadAdapter::new(Arc::new(fake.clone()), lim).unwrap();
    fake.hold();
    a.submit(e, buf(&e), far()).unwrap();
    a.submit(e, buf(&e), far()).unwrap();
    assert_eq!(
        a.submit(e, buf(&e), far()).unwrap_err().error,
        ReadError::InflightBytesExceeded
    );
    fake.release();
    for _ in 0..2 {
        assert!(matches!(a.wait(WAIT).unwrap().outcome, Outcome::Ready(_)));
    }
}

#[test]
fn invalid_limits_are_refused() {
    let src = Arc::new(FakeSource::new());
    for lim in [
        ReadLimits {
            max_in_flight: 0,
            ..limits()
        },
        ReadLimits {
            max_inflight_bytes: 0,
            ..limits()
        },
        ReadLimits {
            max_object_bytes: 0,
            ..limits()
        },
        ReadLimits {
            chunk_bytes: 1000,
            ..limits()
        },
    ] {
        assert_eq!(
            ReadAdapter::new(src.clone(), lim).err(),
            Some(ReadError::InvalidLimits)
        );
    }
}

#[test]
fn a_stalled_read_hits_its_deadline_with_no_ready_state() {
    let (f, a, e, _, _) = setup(limits());
    f.set_fault(e.data_digest, Fault::Delay(Duration::from_millis(20)));
    let t0 = Instant::now();
    a.submit(e, buf(&e), Instant::now() + Duration::from_millis(60))
        .unwrap();
    let c = a.wait(WAIT).unwrap();
    assert_eq!(expect_failed(c), ReadError::DeadlineExceeded);
    // Detected within a few chunk delays, not after the whole object.
    assert!(t0.elapsed() < Duration::from_secs(2));
    assert_eq!(a.in_flight(), (0, 0));
}

#[test]
fn cancellation_yields_one_scrubbed_terminal() {
    let (f, a, e, _, _) = setup(limits());
    f.hold();
    let t = a.submit(e, buf(&e), far()).unwrap();
    assert_eq!(a.cancel(t), CancelOutcome::Requested);
    f.release();
    let c = a.wait(WAIT).unwrap();
    assert_eq!(c.ticket, t);
    match c.outcome {
        Outcome::Cancelled { buf } => assert!(buf.as_slice().iter().all(|&b| b == 0)),
        o => panic!("{o:?}"),
    }
    assert!(
        a.wait(Duration::from_millis(50)).is_none(),
        "exactly one terminal"
    );
    assert_eq!(a.cancel(t), CancelOutcome::Unknown);
}

#[test]
fn concurrent_readers_respect_queue_depth_and_all_verify() {
    let fake = FakeSource::new();
    let mut entries = Vec::new();
    for ex in 0..8u32 {
        let (e, bytes, w, _) = artifact(key(0, ex, Projection::Gate), 192, 256, ex);
        fake.insert(e.data_digest, bytes);
        entries.push((e, w));
    }
    fake.set_fault(entries[3].0.data_digest, Fault::CorruptByte(9000));
    let lim = ReadLimits {
        max_in_flight: 4,
        max_inflight_bytes: 64 << 20,
        ..limits()
    };
    let a = Arc::new(ReadAdapter::new(Arc::new(fake.clone()), lim).unwrap());
    let readers: Vec<_> = (0..6)
        .map(|t| {
            let a = Arc::clone(&a);
            let entries = entries.clone();
            std::thread::spawn(move || {
                let mut submitted = 0;
                for i in 0..16 {
                    let (e, _) = entries[(t + i) % entries.len()];
                    let mut b = buf(&e);
                    loop {
                        match a.submit(e, b, far()) {
                            Ok(_) => break,
                            Err(r) => {
                                assert_eq!(r.error, ReadError::QueueFull);
                                b = r.buf;
                                std::thread::yield_now();
                            }
                        }
                    }
                    submitted += 1;
                }
                submitted
            })
        })
        .collect();
    let total: usize = readers.into_iter().map(|h| h.join().unwrap()).sum();
    let (mut ok, mut bad) = (0, 0);
    for _ in 0..total {
        match a.wait(WAIT).expect("terminal").outcome {
            Outcome::Ready(r) => {
                let (_, w) = entries.iter().find(|(e, _)| e.key == r.header.key).unwrap();
                assert_eq!(r.weights(), &w[..]);
                ok += 1;
            }
            Outcome::Failed { error, buf } => {
                assert_eq!(error, ReadError::DigestMismatch);
                assert!(buf.as_slice().iter().all(|&b| b == 0));
                bad += 1;
            }
            o => panic!("{o:?}"),
        }
    }
    assert_eq!(ok + bad, 96);
    assert_eq!(
        bad, 12,
        "every read of the corrupt object fails, no other does"
    );
    assert!(fake.peak() <= 4, "peak {} exceeds queue depth", fake.peak());
    assert_eq!(a.in_flight(), (0, 0));
}

#[test]
fn drop_cancels_queued_reads_and_joins() {
    let (f, a, e, _, _) = setup(limits());
    f.hold();
    a.submit(e, buf(&e), far()).unwrap();
    let releaser = {
        let f = f.clone();
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            f.release();
        })
    };
    drop(a);
    releaser.join().unwrap();
}
