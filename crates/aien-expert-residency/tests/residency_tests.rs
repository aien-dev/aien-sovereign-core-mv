mod common;

use aien_expert_residency::{
    Acquired, ExpertIdentity, ExpertPart, ExpertState, FailureCause, Phase, Pin, PinStatus,
    Refusal, ResidencyManager, Stats,
};
use common::*;

fn fetch(a: Acquired) -> (Pin, aien_expert_residency::FetchTicket) {
    match a {
        Acquired::Fetch(p, t) => (p, t),
        other => panic!("expected a miss, got {other:?}"),
    }
}

#[test]
fn full_lifecycle_and_states() {
    let mut m = manager(2);
    let k = id(0, 5);
    assert_eq!(m.state(&k), ExpertState::Absent);
    let (pin, t) = fetch(m.acquire(k, SLOT, 100).unwrap());
    assert_eq!(m.state(&k), ExpertState::Fetching);
    assert_eq!(m.poll(&pin), Ok(PinStatus::Waiting(Phase::Fetching)));
    m.mark_verified(&t).unwrap();
    assert_eq!(m.state(&k), ExpertState::Verified);
    m.mark_copying(&t).unwrap();
    assert_eq!(m.state(&k), ExpertState::Copying);
    m.publish(&t, 350).unwrap();
    assert_eq!(m.state(&k), ExpertState::InUse);
    assert_eq!(m.poll(&pin), Ok(PinStatus::Ready));
    m.release(pin).unwrap();
    assert_eq!(m.state(&k), ExpertState::Evictable);
    let s = m.stats();
    assert_eq!((s.misses, s.published, s.hits), (1, 1, 0));
    assert_eq!((s.fetch_wait_ns_total, s.fetch_wait_ns_max), (250, 250));
    assert_eq!(s.current_resident_bytes, SLOT);
    assert_eq!(s.current_pins, 0);
    m.check_invariants().unwrap();
}

#[test]
fn published_without_pins_is_ready_not_evictable() {
    let mut m = manager(2);
    let k = id(1, 1);
    let (pin, t) = fetch(m.acquire(k, SLOT, 0).unwrap());
    m.release(pin).unwrap();
    complete(&mut m, &t, 0);
    assert_eq!(m.state(&k), ExpertState::Ready);
}

#[test]
fn phase_steps_cannot_be_skipped_or_repeated() {
    let mut m = manager(2);
    let (pin, t) = fetch(m.acquire(id(0, 0), SLOT, 0).unwrap());
    let slot = t.slot();
    assert_eq!(
        m.publish(&t, 0),
        Err(Refusal::WrongPhase {
            slot,
            expected: Phase::Copying,
            found: Phase::Fetching
        })
    );
    assert!(matches!(
        m.mark_copying(&t),
        Err(Refusal::WrongPhase { .. })
    ));
    m.mark_verified(&t).unwrap();
    assert!(matches!(
        m.mark_verified(&t),
        Err(Refusal::WrongPhase { .. })
    ));
    m.mark_copying(&t).unwrap();
    m.publish(&t, 0).unwrap();
    assert!(matches!(m.publish(&t, 0), Err(Refusal::WrongPhase { .. })));
    // A published entry cannot be failed.
    assert_eq!(
        m.fail(&t, FailureCause::Read),
        Err(Refusal::WrongPhase {
            slot,
            expected: Phase::Fetching,
            found: Phase::Ready
        })
    );
    m.release(pin).unwrap();
    m.check_invariants().unwrap();
}

#[test]
fn hit_and_join_counters() {
    let mut m = manager(2);
    let k = id(3, 9);
    let (p1, t) = fetch(m.acquire(k, SLOT, 0).unwrap());
    let p2 = match m.acquire(k, SLOT, 0).unwrap() {
        Acquired::Pending(p) => p,
        other => panic!("{other:?}"),
    };
    complete(&mut m, &t, 0);
    let p3 = match m.acquire(k, SLOT, 0).unwrap() {
        Acquired::Ready(p) => p,
        other => panic!("{other:?}"),
    };
    let s = m.stats();
    assert_eq!((s.misses, s.joined, s.hits, s.current_pins), (1, 1, 1, 3));
    for p in [p1, p2, p3] {
        m.release(p).unwrap();
    }
    m.check_invariants().unwrap();
}

#[test]
fn eviction_is_lru_and_deterministic() {
    let run = || {
        let mut m = manager(3);
        for e in 0..3 {
            touch(&mut m, id(0, e));
        }
        // Use order now 0,1,2. Touch 0 so 1 is the oldest.
        touch(&mut m, id(0, 0));
        let mut order = Vec::new();
        for e in 10..14 {
            let victim = m.next_victim().unwrap();
            let before: Vec<_> = (0..20)
                .filter(|&x| m.state(&id(0, x)) != ExpertState::Absent)
                .collect();
            touch(&mut m, id(0, e));
            let after: Vec<_> = (0..20)
                .filter(|&x| m.state(&id(0, x)) != ExpertState::Absent)
                .collect();
            let gone: Vec<_> = before
                .iter()
                .filter(|x| !after.contains(x))
                .copied()
                .collect();
            order.push((victim, gone));
        }
        m.check_invariants().unwrap();
        (order, m.stats())
    };
    let (order, stats) = run();
    let gone: Vec<u32> = order.iter().flat_map(|(_, g)| g.clone()).collect();
    assert_eq!(gone, vec![1, 2, 0, 10]);
    assert_eq!(stats.evictions, 4);
    assert_eq!(run(), (order, stats), "same calls, same order");
}

#[test]
fn pinned_entry_is_never_evicted() {
    let mut m = manager(2);
    let held = load(&mut m, id(0, 0)); // oldest, but pinned
    touch(&mut m, id(0, 1));
    for e in 2..50 {
        touch(&mut m, id(1, e));
        assert_eq!(m.state(&id(0, 0)), ExpertState::InUse);
        assert_eq!(m.poll(&held), Ok(PinStatus::Ready));
    }
    m.release(held).unwrap();
    touch(&mut m, id(2, 0));
    touch(&mut m, id(2, 1));
    assert_eq!(m.state(&id(0, 0)), ExpertState::Absent);
    m.check_invariants().unwrap();
}

#[test]
fn pinned_exhaustion_is_refused_without_side_effects() {
    let mut m = manager(2);
    let a = load(&mut m, id(0, 0));
    let (b, tb) = fetch(m.acquire(id(0, 1), SLOT, 0).unwrap()); // in flight
    let before = m.stats();
    let r = m.acquire(id(0, 2), SLOT, 0);
    assert_eq!(
        r,
        Err(Refusal::PinnedExhaustion {
            slots: 2,
            pinned: 2,
            in_flight: 1
        })
    );
    let after = m.stats();
    assert_eq!(after.refusals, before.refusals + 1);
    assert_eq!(after.evictions, before.evictions);
    assert_eq!(m.state(&id(0, 2)), ExpertState::Absent);
    // An in-flight entry with no pins is not evictable either.
    m.release(b).unwrap();
    assert!(matches!(
        m.acquire(id(0, 2), SLOT, 0),
        Err(Refusal::PinnedExhaustion {
            pinned: 1,
            in_flight: 1,
            ..
        })
    ));
    complete(&mut m, &tb, 0);
    // Now id(0,1) is Ready and unpinned: it is the victim.
    let (p, t) = fetch(m.acquire(id(0, 2), SLOT, 0).unwrap());
    assert_eq!(m.state(&id(0, 1)), ExpertState::Absent);
    complete(&mut m, &t, 0);
    m.release(p).unwrap();
    m.release(a).unwrap();
    m.check_invariants().unwrap();
}

#[test]
fn refused_miss_does_not_evict() {
    let mut m = manager(1);
    touch(&mut m, id(0, 0));
    assert_eq!(
        m.acquire(id(0, 1), SLOT + 1, 0),
        Err(Refusal::ObjectTooLarge {
            bytes: SLOT + 1,
            slot_bytes: SLOT
        })
    );
    assert_eq!(m.acquire(id(0, 1), 0, 0), Err(Refusal::ZeroSizedObject));
    assert_eq!(m.state(&id(0, 0)), ExpertState::Evictable);
    assert_eq!(m.stats().evictions, 0);
}

#[test]
fn size_mismatch_is_refused() {
    let mut m = manager(2);
    touch(&mut m, id(0, 0));
    assert_eq!(
        m.acquire(id(0, 0), SLOT - 1, 0),
        Err(Refusal::SizeMismatch {
            admitted: SLOT,
            requested: SLOT - 1
        })
    );
    assert_eq!(m.stats().current_pins, 0);
}

#[test]
fn backpressure_by_count_and_bytes() {
    let mut m = ResidencyManager::new(config(8, 2, 3 * SLOT)).unwrap();
    let (p1, t1) = fetch(m.acquire(id(0, 0), SLOT, 0).unwrap());
    let (p2, t2) = fetch(m.acquire(id(0, 1), SLOT, 0).unwrap());
    assert_eq!(
        m.acquire(id(0, 2), SLOT, 0),
        Err(Refusal::InFlightLimit { limit: 2 })
    );
    // Joining an in-flight fetch is not a new fetch: allowed.
    let j = m.acquire(id(0, 0), SLOT, 0).unwrap();
    complete(&mut m, &t1, 0);
    assert_eq!(m.stats().current_in_flight, 1);

    let mut m2 = ResidencyManager::new(config(8, 4, 2 * SLOT)).unwrap();
    let (q1, s1) = fetch(m2.acquire(id(0, 0), SLOT, 0).unwrap());
    let (q2, s2) = fetch(m2.acquire(id(0, 1), SLOT / 2, 0).unwrap());
    assert_eq!(
        m2.acquire(id(0, 2), SLOT, 0),
        Err(Refusal::InFlightBytesLimit {
            in_flight_bytes: SLOT + SLOT / 2,
            requested: SLOT,
            limit: 2 * SLOT
        })
    );
    m2.fail(&s2, FailureCause::Cancelled).unwrap();
    let (q3, s3) = fetch(m2.acquire(id(0, 2), SLOT, 0).unwrap());
    for t in [&s1, &s3] {
        complete(&mut m2, t, 0);
    }
    assert_eq!(m2.stats().peak_in_flight_bytes, 2 * SLOT);
    for p in [q1, q2, q3] {
        m2.release(p).unwrap();
    }
    complete(&mut m, &t2, 0);
    for p in [p1, p2, j.pin_owned()] {
        m.release(p).unwrap();
    }
    m.check_invariants().unwrap();
    m2.check_invariants().unwrap();
}

trait PinOwned {
    fn pin_owned(self) -> Pin;
}
impl PinOwned for Acquired {
    fn pin_owned(self) -> Pin {
        match self {
            Acquired::Ready(p) | Acquired::Pending(p) | Acquired::Fetch(p, _) => p,
        }
    }
}

#[test]
fn simultaneous_readers_hold_until_last_release() {
    let mut m = manager(1);
    let k = id(4, 4);
    let readers: Vec<Pin> = (0..5).map(|_| load(&mut m, k)).collect();
    assert_eq!(m.stats().current_pins, 5);
    let mut readers = readers.into_iter();
    for p in readers.by_ref().take(4) {
        m.release(p).unwrap();
        assert!(matches!(
            m.acquire(id(5, 5), SLOT, 0),
            Err(Refusal::PinnedExhaustion { .. })
        ));
    }
    m.release(readers.next().unwrap()).unwrap();
    touch(&mut m, id(5, 5));
    assert_eq!(m.state(&k), ExpertState::Absent);
    m.check_invariants().unwrap();
}

#[test]
fn layer_crossing_and_part_identities_are_distinct() {
    let mut m = manager(8);
    let a = id(0, 7);
    let b = id(1, 7);
    let mut c = id(0, 7);
    c.part = ExpertPart::GateUp;
    let mut d = id(0, 7);
    d.checkpoint_digest = [8; 32];
    for k in [a, b, c, d] {
        touch(&mut m, k);
    }
    assert_eq!(m.stats().misses, 4);
    let slots: std::collections::BTreeSet<_> = [a, b, c, d]
        .iter()
        .map(|k| m.slot_of(k).unwrap().0)
        .collect();
    assert_eq!(slots.len(), 4);
    touch(&mut m, a);
    assert_eq!(m.stats().hits, 1);
}

#[test]
fn aba_slot_reuse_refuses_stale_pin_and_ticket() {
    let mut m = manager(1);
    let (old_pin, old_t) = fetch(m.acquire(id(0, 0), SLOT, 0).unwrap());
    let (slot, g1) = (old_t.slot(), old_t.generation());
    complete(&mut m, &old_t, 0);
    m.release(old_pin).unwrap();
    // Reuse the same slot for another key.
    let (new_pin, new_t) = fetch(m.acquire(id(9, 9), SLOT, 0).unwrap());
    assert_eq!(new_t.slot(), slot);
    assert!(new_t.generation() > g1);
    assert_eq!(
        m.mark_verified(&old_t),
        Err(Refusal::StaleTicket {
            slot,
            generation: g1
        })
    );
    assert_eq!(
        m.fail(&old_t, FailureCause::Read),
        Err(Refusal::StaleTicket {
            slot,
            generation: g1
        })
    );
    // Same slot, re-admitted under the original key: still a new generation.
    complete(&mut m, &new_t, 0);
    m.release(new_pin).unwrap();
    let (p3, t3) = fetch(m.acquire(id(0, 0), SLOT, 0).unwrap());
    assert_eq!(t3.slot(), slot);
    assert!(t3.generation() > new_t.generation());
    assert!(matches!(
        m.publish(&new_t, 0),
        Err(Refusal::StaleTicket { .. })
    ));
    m.fail(&t3, FailureCause::Cancelled).unwrap();
    assert!(matches!(m.poll(&p3), Err(Refusal::FetchFailed { .. })));
    m.release(p3).unwrap();
    m.check_invariants().unwrap();
}

#[test]
fn failure_with_waiters_is_explicit_and_recoverable() {
    let mut m = manager(2);
    let k = id(2, 3);
    let (p1, t) = fetch(m.acquire(k, SLOT, 0).unwrap());
    let p2 = m.acquire(k, SLOT, 0).unwrap().pin_owned();
    m.mark_verified(&t).unwrap();
    m.fail(&t, FailureCause::DigestMismatch).unwrap();
    let failed = Err(Refusal::FetchFailed {
        slot: t.slot(),
        generation: t.generation(),
        cause: FailureCause::DigestMismatch,
    });
    assert_eq!(m.poll(&p1), failed);
    assert_eq!(m.poll(&p2), failed);
    // The key is unmapped at once: a retry is a clean miss in another slot.
    assert_eq!(m.state(&k), ExpertState::Absent);
    let (p3, t3) = fetch(m.acquire(k, SLOT, 0).unwrap());
    assert_ne!(t3.slot(), t.slot());
    // Failed slot is held for its waiters, so a third key has no room.
    assert!(matches!(
        m.acquire(id(7, 7), SLOT, 0),
        Err(Refusal::PinnedExhaustion { .. })
    ));
    m.release(p1).unwrap();
    m.release(p2).unwrap();
    complete(&mut m, &t3, 0);
    m.release(p3).unwrap();
    touch(&mut m, id(7, 7));
    let s = m.stats();
    assert_eq!((s.failures, s.published), (1, 2));
    assert_eq!(s.current_in_flight, 0);
    m.check_invariants().unwrap();
}

#[test]
fn failure_without_pins_frees_the_slot_at_once() {
    let mut m = manager(1);
    let (p, t) = fetch(m.acquire(id(0, 0), SLOT, 0).unwrap());
    m.release(p).unwrap();
    m.mark_verified(&t).unwrap();
    m.mark_copying(&t).unwrap();
    m.fail(&t, FailureCause::GpuFence).unwrap();
    touch(&mut m, id(0, 1));
    assert_eq!(m.stats().evictions, 0);
    m.check_invariants().unwrap();
}

#[test]
fn acquire_all_is_all_or_nothing() {
    let mut m = manager(3);
    touch(&mut m, id(0, 0));
    let held = load(&mut m, id(0, 1));
    // Request needs 3 distinct new keys plus a repeat; only 2 slots usable.
    let req = [
        (id(0, 1), SLOT),
        (id(1, 0), SLOT),
        (id(1, 0), SLOT),
        (id(1, 1), SLOT),
        (id(1, 2), SLOT),
    ];
    let before_pins = m.stats().current_pins;
    let r = m.acquire_all(&req, 0);
    assert!(matches!(r, Err(Refusal::PinnedExhaustion { .. })), "{r:?}");
    let s = m.stats();
    assert_eq!(s.current_pins, before_pins);
    assert_eq!(s.current_in_flight, 0);
    assert_eq!(s.rolled_back, 2);
    assert_eq!(m.state(&id(1, 0)), ExpertState::Absent);
    assert_eq!(m.state(&id(1, 1)), ExpertState::Absent);
    assert_eq!(m.state(&id(0, 1)), ExpertState::InUse);
    m.check_invariants().unwrap();

    // A request that fits: all pinned, duplicates share one fetch.
    let ok = m
        .acquire_all(&[(id(0, 1), SLOT), (id(2, 0), SLOT), (id(2, 0), SLOT)], 0)
        .unwrap();
    assert!(matches!(ok[0], Acquired::Ready(_)));
    assert!(matches!(ok[1], Acquired::Fetch(..)));
    assert!(matches!(ok[2], Acquired::Pending(_)));
    let mut pins = Vec::new();
    for a in ok {
        match a {
            Acquired::Fetch(p, t) => {
                complete(&mut m, &t, 0);
                pins.push(p);
            }
            other => pins.push(other.pin_owned()),
        }
    }
    for p in pins {
        m.release(p).unwrap();
    }
    m.release(held).unwrap();
    m.check_invariants().unwrap();
}

/// Random operation sequences against small caches. After every operation:
/// internal invariants hold, every live pin still resolves to its own slot
/// generation (so a pinned entry was never evicted or reused), and refusals
/// leave pins untouched. The whole trace is reproduced exactly by a second
/// run with the same seed.
fn random_run(seed: u64, slots: u32, steps: usize) -> (Vec<String>, Stats) {
    let mut m = ResidencyManager::<ExpertIdentity>::new(config(slots, 3, 3 * SLOT)).unwrap();
    let mut rng = Rng(seed);
    let mut pins: Vec<(ExpertIdentity, Pin)> = Vec::new();
    let mut tickets: Vec<aien_expert_residency::FetchTicket> = Vec::new();
    let mut trace = Vec::with_capacity(steps);
    for step in 0..steps {
        let op = rng.below(10);
        let line = match op {
            0..=3 => {
                let k = id(rng.below(3) as u32, rng.below(6) as u32);
                let predicted = m.next_victim();
                let evictions = m.stats().evictions;
                match m.acquire(k, SLOT, step as u64) {
                    Ok(Acquired::Fetch(p, t)) => {
                        if m.stats().evictions > evictions {
                            assert_eq!(
                                Some(t.slot()),
                                predicted,
                                "seed {seed} step {step}: not the LRU victim"
                            );
                        }
                        let s = format!("miss {k:?} -> {}", t.slot());
                        pins.push((k, p));
                        tickets.push(t);
                        s
                    }
                    Ok(a) => {
                        let s = format!("acq {a:?}");
                        pins.push((k, a.pin_owned()));
                        s
                    }
                    Err(r) => format!("refused {r:?}"),
                }
            }
            4..=6 if !tickets.is_empty() => {
                let i = rng.below(tickets.len() as u64) as usize;
                let t = &tickets[i];
                let r = match rng.below(4) {
                    0 => m.mark_verified(t),
                    1 => m.mark_copying(t),
                    2 => m.publish(t, step as u64),
                    _ => {
                        if rng.below(4) == 0 {
                            m.fail(t, FailureCause::Read)
                        } else {
                            m.publish(t, step as u64)
                        }
                    }
                };
                let s = format!("step {} {r:?}", t.slot());
                let done = !matches!(
                    m.ticket_phase(t),
                    Ok(Phase::Fetching | Phase::Verified | Phase::Copying)
                );
                if done {
                    let _finished = tickets.swap_remove(i);
                }
                s
            }
            _ if !pins.is_empty() => {
                let i = rng.below(pins.len() as u64) as usize;
                let (k, p) = pins.swap_remove(i);
                format!("rel {k:?} {:?}", m.release(p))
            }
            _ => "noop".into(),
        };
        m.check_invariants()
            .unwrap_or_else(|e| panic!("seed {seed} step {step} {line}: {e}"));
        for (k, p) in &pins {
            match m.poll(p) {
                Ok(_) => {
                    let (slot, g) = m.slot_of(k).expect("pinned key must stay admitted");
                    assert_eq!(
                        (slot, g),
                        (p.slot(), p.generation()),
                        "seed {seed} step {step}"
                    );
                }
                Err(Refusal::FetchFailed { .. }) => {}
                Err(e) => panic!("seed {seed} step {step}: live pin refused {e:?}"),
            }
        }
        trace.push(line);
    }
    let stats = m.stats();
    // Drain: every remaining ticket is still in flight; fail it, then
    // release every pin. The cache must be fully evictable again.
    for t in tickets.drain(..) {
        m.fail(&t, FailureCause::Cancelled).unwrap();
    }
    for (_, p) in pins.drain(..) {
        m.release(p).unwrap();
    }
    m.check_invariants().unwrap();
    assert_eq!(m.stats().current_pins, 0);
    assert_eq!(m.stats().current_in_flight, 0);
    (trace, stats)
}

#[test]
fn property_random_sequences_hold_invariants() {
    let mut total = Stats::default();
    for seed in 1..=40u64 {
        for slots in [1u32, 2, 3, 5] {
            let (_, s) = random_run(seed * 0x1000_0001 + u64::from(slots), slots, 400);
            total.hits += s.hits;
            total.joined += s.joined;
            total.evictions += s.evictions;
            total.failures += s.failures;
            total.refusals += s.refusals;
            total.published += s.published;
        }
    }
    // The generator must actually reach every interesting path.
    eprintln!("property totals: {total:?}");
    assert!(
        total.hits > 0
            && total.joined > 0
            && total.evictions > 0
            && total.failures > 0
            && total.refusals > 0
            && total.published > 0,
        "{total:?}"
    );
}

#[test]
fn property_traces_are_deterministic() {
    for seed in [3u64, 99, 12345] {
        assert_eq!(random_run(seed, 3, 600), random_run(seed, 3, 600));
    }
}
