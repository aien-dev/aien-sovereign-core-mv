//! Fixed versioned slots with publish/lease ordering.
//!
//! Slot life cycle: `Empty -> Filling -> Published -> Filling -> ...`.
//! * `begin_fill` waits until no reader holds a lease, then hides the slot
//!   (readers cannot lease a `Filling` slot) and hands out exclusive write
//!   access.
//! * `FillGuard::publish` hashes the payload and compares against the
//!   expected digest; only on a match does the slot become `Published` with a
//!   fresh ring-wide version.
//! * Any failure marks the slot `Poisoned` and the whole ring failed: every
//!   later `begin_fill` and `lease` returns [`HeartbeatError::RingFailed`].

use std::cell::UnsafeCell;
#[cfg(test)]
use std::sync::atomic::AtomicU32;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Condvar, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use crate::aligned::AlignedBuf;
use crate::error::HeartbeatError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SlotTag {
    pub layer: u16,
    pub expert: u16,
    pub version: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Empty,
    Filling,
    Published,
    Poisoned,
}

#[derive(Debug)]
struct SlotState {
    phase: Phase,
    tag: Option<SlotTag>,
    leases: u32,
    /// Payload ranges (offset, len) in the slot buffer, in payload order.
    ranges: Vec<(usize, usize)>,
}

struct Slot {
    buf: UnsafeCell<AlignedBuf>,
    state: Mutex<SlotState>,
    released: Condvar,
    /// Test-only: writers currently blocked in `begin_fill` on this slot.
    #[cfg(test)]
    writers_waiting: AtomicU32,
}

// SAFETY: `buf` is written only by the holder of the unique FillGuard while
// phase == Filling (no leases can exist then), and read only through a Lease
// while phase == Published (no FillGuard can exist then). Both transitions are
// made under `state`'s mutex.
unsafe impl Sync for Slot {}

pub struct Ring {
    slots: Vec<Slot>,
    failed: AtomicBool,
    next_version: AtomicU64,
}

fn lock(m: &Mutex<SlotState>) -> MutexGuard<'_, SlotState> {
    // A panicking holder poisons the mutex; treat state as still usable since
    // the ring's failure flag is the authority.
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl Ring {
    pub fn new(num_slots: usize, slot_bytes: usize) -> Self {
        let slots = (0..num_slots)
            .map(|_| Slot {
                buf: UnsafeCell::new(AlignedBuf::new(slot_bytes)),
                state: Mutex::new(SlotState {
                    phase: Phase::Empty,
                    tag: None,
                    leases: 0,
                    ranges: Vec::new(),
                }),
                released: Condvar::new(),
                #[cfg(test)]
                writers_waiting: AtomicU32::new(0),
            })
            .collect();
        Self {
            slots,
            failed: AtomicBool::new(false),
            next_version: AtomicU64::new(1),
        }
    }

    pub fn num_slots(&self) -> usize {
        self.slots.len()
    }

    pub fn slot_capacity(&self) -> usize {
        // SAFETY: capacity is immutable after construction.
        unsafe { (*self.slots[0].buf.get()).len() }
    }

    pub fn is_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Mark the ring failed (fail closed) and wake every waiter.
    pub fn fail(&self) {
        self.failed.store(true, Ordering::Release);
        for s in &self.slots {
            let _g = lock(&s.state);
            s.released.notify_all();
        }
    }

    /// Current tag of a slot if it is published.
    pub fn tag(&self, slot: usize) -> Option<SlotTag> {
        let st = lock(&self.slots[slot].state);
        (st.phase == Phase::Published).then_some(st.tag).flatten()
    }

    /// Lease the published slot holding `(layer, expert)`, if any.
    pub fn lease(&self, layer: u16, expert: u16) -> Result<Option<Lease<'_>>, HeartbeatError> {
        if self.is_failed() {
            return Err(HeartbeatError::RingFailed);
        }
        for (i, s) in self.slots.iter().enumerate() {
            let mut st = lock(&s.state);
            if st.phase == Phase::Published {
                if let Some(tag) = st.tag {
                    if tag.layer == layer && tag.expert == expert {
                        st.leases += 1;
                        return Ok(Some(Lease {
                            ring: self,
                            slot: i,
                            tag,
                            ranges: st.ranges.clone(),
                        }));
                    }
                }
            }
        }
        Ok(None)
    }

    /// Take exclusive write access to `slot`, waiting (up to `wait`) for
    /// readers to drop their leases. Returns the time spent waiting.
    pub fn begin_fill(
        &self,
        slot: usize,
        wait: Duration,
    ) -> Result<(FillGuard<'_>, Duration), HeartbeatError> {
        let s = &self.slots[slot];
        let t0 = Instant::now();
        let mut st = lock(&s.state);
        loop {
            if self.is_failed() || st.phase == Phase::Poisoned {
                return Err(HeartbeatError::RingFailed);
            }
            if st.phase == Phase::Filling {
                return Err(HeartbeatError::SlotBusy(slot));
            }
            if st.leases == 0 {
                break;
            }
            let left = wait.saturating_sub(t0.elapsed());
            if left.is_zero() {
                return Err(HeartbeatError::SlotBusy(slot));
            }
            #[cfg(test)]
            s.writers_waiting.fetch_add(1, Ordering::SeqCst);
            st = s
                .released
                .wait_timeout(st, left)
                .unwrap_or_else(|p| p.into_inner())
                .0;
            #[cfg(test)]
            s.writers_waiting.fetch_sub(1, Ordering::SeqCst);
        }
        st.phase = Phase::Filling;
        st.tag = None;
        drop(st);
        Ok((
            FillGuard {
                ring: self,
                slot,
                done: false,
            },
            t0.elapsed(),
        ))
    }
}

/// Exclusive write access to one slot. Dropping it without publishing returns
/// the slot to `Empty`; [`FillGuard::fail`] poisons the ring.
pub struct FillGuard<'r> {
    ring: &'r Ring,
    slot: usize,
    done: bool,
}

impl FillGuard<'_> {
    pub fn slot(&self) -> usize {
        self.slot
    }

    pub fn buf_mut(&mut self) -> &mut [u8] {
        // SAFETY: phase == Filling and this guard is the unique writer.
        unsafe { (*self.ring.slots[self.slot].buf.get()).as_mut_slice() }
    }

    /// Verify BLAKE3 over `ranges` (in order) == expected, then publish. On
    /// mismatch or a bad range the ring fails closed. Returns the digest time.
    pub fn publish(
        mut self,
        layer: u16,
        expert: u16,
        ranges: &[(usize, usize)],
        expected: &[u8; 32],
    ) -> Result<(SlotTag, Duration), HeartbeatError> {
        let cap = self.buf_mut().len();
        for &(off, len) in ranges {
            if off.checked_add(len).is_none_or(|end| end > cap) {
                self.fail_inner();
                return Err(HeartbeatError::PayloadTooLarge {
                    slot: self.slot,
                    need: off.saturating_add(len),
                    cap,
                });
            }
        }
        let t0 = Instant::now();
        let ok = {
            let buf = &self.buf_mut()[..];
            let mut h = blake3::Hasher::new();
            for &(off, len) in ranges {
                h.update(&buf[off..off + len]);
            }
            h.finalize().as_bytes() == expected
        };
        let dt = t0.elapsed();
        if !ok || self.ring.is_failed() {
            self.fail_inner();
            return Err(if ok {
                HeartbeatError::RingFailed
            } else {
                HeartbeatError::DigestMismatch {
                    expert: (layer, expert),
                }
            });
        }
        let version = self.ring.next_version.fetch_add(1, Ordering::AcqRel);
        let tag = SlotTag {
            layer,
            expert,
            version,
        };
        let s = &self.ring.slots[self.slot];
        let mut st = lock(&s.state);
        st.phase = Phase::Published;
        st.tag = Some(tag);
        st.ranges = ranges.to_vec();
        drop(st);
        self.done = true;
        Ok((tag, dt))
    }

    /// Release without publishing (slot becomes Empty, not visible).
    pub fn discard(mut self) {
        let mut st = lock(&self.ring.slots[self.slot].state);
        st.phase = Phase::Empty;
        drop(st);
        self.done = true;
    }

    /// Poison this slot and fail the ring closed.
    pub fn fail(mut self) {
        self.fail_inner();
    }

    fn fail_inner(&mut self) {
        {
            let mut st = lock(&self.ring.slots[self.slot].state);
            st.phase = Phase::Poisoned;
            st.tag = None;
        }
        self.done = true;
        self.ring.fail();
    }
}

impl Drop for FillGuard<'_> {
    fn drop(&mut self) {
        if !self.done {
            // Unwinding mid-fill: never publish, fail closed.
            self.fail_inner();
        }
    }
}

/// Read access to a published slot; the slot cannot be refilled while any
/// lease is alive.
pub struct Lease<'r> {
    ring: &'r Ring,
    slot: usize,
    tag: SlotTag,
    ranges: Vec<(usize, usize)>,
}

impl Lease<'_> {
    pub fn tag(&self) -> SlotTag {
        self.tag
    }

    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Payload pieces in order (one per manifest segment).
    pub fn segments(&self) -> impl Iterator<Item = &[u8]> + '_ {
        // SAFETY: leases > 0 keeps phase == Published, so no writer exists.
        let buf = unsafe { &*self.ring.slots[self.slot].buf.get() }.as_slice();
        self.ranges.iter().map(move |&(o, l)| &buf[o..o + l])
    }

    /// Payload copied into one contiguous vector.
    pub fn to_vec(&self) -> Vec<u8> {
        self.segments().flatten().copied().collect()
    }
}

impl Drop for Lease<'_> {
    fn drop(&mut self) {
        let s = &self.ring.slots[self.slot];
        let mut st = lock(&s.state);
        st.leases -= 1;
        if st.leases == 0 {
            s.released.notify_all();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn fill(ring: &Ring, slot: usize, layer: u16, expert: u16, byte: u8) -> SlotTag {
        let (mut g, _) = ring.begin_fill(slot, Duration::from_secs(1)).unwrap();
        g.buf_mut()[..16].fill(byte);
        let d = *blake3::hash(&[byte; 16]).as_bytes();
        g.publish(layer, expert, &[(0, 16)], &d).unwrap().0
    }

    #[test]
    fn not_visible_until_published() {
        let ring = Ring::new(2, 4096);
        let (mut g, _) = ring.begin_fill(0, Duration::ZERO).unwrap();
        g.buf_mut()[..16].fill(7);
        assert!(ring.lease(0, 3).unwrap().is_none());
        let d = *blake3::hash(&[7u8; 16]).as_bytes();
        let (tag, _) = g.publish(0, 3, &[(0, 16)], &d).unwrap();
        let l = ring.lease(0, 3).unwrap().unwrap();
        assert_eq!(l.tag(), tag);
        assert_eq!(l.to_vec(), vec![7u8; 16]);
    }

    #[test]
    fn versions_increase_monotonically() {
        let ring = Ring::new(1, 4096);
        let a = fill(&ring, 0, 0, 1, 1);
        let b = fill(&ring, 0, 0, 2, 2);
        assert!(b.version > a.version);
        assert!(
            ring.lease(0, 1).unwrap().is_none(),
            "old tag retired on refill"
        );
    }

    #[test]
    fn leased_slot_is_never_overwritten() {
        let ring = Ring::new(1, 4096);
        fill(&ring, 0, 1, 1, 9);
        let l = ring.lease(1, 1).unwrap().unwrap();
        assert!(matches!(
            ring.begin_fill(0, Duration::from_millis(20)),
            Err(HeartbeatError::SlotBusy(0))
        ));
        assert_eq!(l.to_vec(), vec![9u8; 16]);
        drop(l);
        assert!(ring.begin_fill(0, Duration::ZERO).is_ok());
    }

    #[test]
    fn writer_waits_for_lease_release() {
        let ring = Arc::new(Ring::new(1, 4096));
        fill(&ring, 0, 1, 1, 9);
        // Ordering comes from events, never from wall time: the reader signals
        // once it holds the lease, the main thread signals once the writer is
        // observed blocked inside `begin_fill`, and only then does the reader
        // read and release.
        let (leased_tx, leased_rx) = std::sync::mpsc::channel::<()>();
        let (release_tx, release_rx) = std::sync::mpsc::channel::<()>();
        let r2 = ring.clone();
        let reader = std::thread::spawn(move || {
            let l = r2.lease(1, 1).unwrap().unwrap();
            leased_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            let v = l.to_vec();
            drop(l);
            v
        });
        leased_rx.recv().unwrap();
        let r3 = ring.clone();
        let writer = std::thread::spawn(move || {
            let (g, _) = r3.begin_fill(0, Duration::from_secs(60)).unwrap();
            g.discard();
        });
        // The counter is raised under the slot mutex, so once it reads 1 the
        // writer is parked on the condvar and the lease is still held. The
        // deadline only stops a broken build from hanging forever.
        let hang_guard = Instant::now();
        while ring.slots[0].writers_waiting.load(Ordering::SeqCst) == 0 {
            assert!(
                hang_guard.elapsed() < Duration::from_secs(60),
                "writer never reached the lease wait"
            );
            assert!(!writer.is_finished(), "writer passed a held lease");
            std::thread::yield_now();
        }
        assert!(!writer.is_finished(), "writer passed a held lease");
        release_tx.send(()).unwrap();
        writer.join().unwrap();
        assert_eq!(reader.join().unwrap(), vec![9u8; 16]);
    }

    #[test]
    fn filling_slot_cannot_be_leased_or_double_filled() {
        let ring = Ring::new(1, 4096);
        fill(&ring, 0, 0, 0, 1);
        let (g, _) = ring.begin_fill(0, Duration::ZERO).unwrap();
        assert!(ring.lease(0, 0).unwrap().is_none());
        assert!(matches!(
            ring.begin_fill(0, Duration::ZERO),
            Err(HeartbeatError::SlotBusy(0))
        ));
        g.discard();
    }

    #[test]
    fn digest_mismatch_fails_closed() {
        let ring = Ring::new(2, 4096);
        fill(&ring, 1, 0, 5, 3);
        let (mut g, _) = ring.begin_fill(0, Duration::ZERO).unwrap();
        g.buf_mut()[..16].fill(1);
        let wrong = *blake3::hash(&[2u8; 16]).as_bytes();
        assert!(matches!(
            g.publish(0, 0, &[(0, 16)], &wrong),
            Err(HeartbeatError::DigestMismatch { .. })
        ));
        assert!(ring.is_failed());
        assert!(matches!(ring.lease(0, 5), Err(HeartbeatError::RingFailed)));
        assert!(matches!(
            ring.begin_fill(1, Duration::ZERO),
            Err(HeartbeatError::RingFailed)
        ));
        assert!(ring.tag(0).is_none());
    }

    #[test]
    fn oversized_payload_fails_closed() {
        let ring = Ring::new(1, 4096);
        let (g, _) = ring.begin_fill(0, Duration::ZERO).unwrap();
        assert!(matches!(
            g.publish(0, 0, &[(4000, 200)], &[0; 32]),
            Err(HeartbeatError::PayloadTooLarge { .. })
        ));
        assert!(ring.is_failed());
    }

    #[test]
    fn dropped_guard_fails_closed() {
        let ring = Ring::new(1, 4096);
        {
            let _g = ring.begin_fill(0, Duration::ZERO).unwrap();
            // dropped without publish/discard (e.g. unwinding)
        }
        assert!(ring.is_failed());
    }
}
