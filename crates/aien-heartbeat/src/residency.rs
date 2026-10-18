//! Bounded expert residency (Expert Ribbon P2, sc#630): which (layer, expert) lives in which GPU slot,
//! who is using it, and which slot to reuse next. Host-only; the GPU side is behind
//! [`crate::pager::SlotBackend`].
//!
//! Slot life cycle (one slot = one gate|up slot + one down slot on the device):
//! ```text
//!   FREE --acquire(miss)--> FETCHING(id) --fetched(ver)--> READY(id, ver) pinned by the requester
//!                           FETCHING(id) --fetch_failed--> FREE (nothing published)
//!   READY, pins > 0  = IN USE      (never chosen as a victim)
//!   READY, pins == 0 = EVICTABLE   (least recently used one is reused on a miss)
//! ```
//! Exactly the router's experts run: a miss waits for that expert or fails; nothing is substituted.
//! When every slot is pinned or fetching the request fails with [`Acquire::Exhausted`] (backpressure, explicit).

use std::collections::HashMap;

/// `(layer, expert)`.
pub type ExpertId = (u16, u16);

/// Device versions of one slot pair (omega_gpu_fp8_slots versions of the gate|up and the down slot).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct SlotVersion {
    pub gate_up: u64,
    pub down: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Free,
    Fetching(ExpertId),
    Ready(ExpertId, SlotVersion),
}

#[derive(Debug, Clone, Copy)]
struct Slot {
    state: State,
    pins: u32,
    last_use: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Acquire {
    /// Resident: pinned for the caller.
    Hit { slot: u32, ver: SlotVersion },
    /// Not resident: `slot` is reserved (FETCHING) for the caller, who must call `fetched` or `fetch_failed`.
    Fetch {
        slot: u32,
        evicted: Option<ExpertId>,
    },
    /// Another request is fetching this expert right now.
    Pending,
    /// Every slot is pinned or fetching.
    Exhausted,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub fetch_failures: u64,
    pub exhausted: u64,
    pub pinned_high: u32,
}

#[derive(Debug)]
pub struct Residency {
    slots: Vec<Slot>,
    map: HashMap<ExpertId, u32>,
    tick: u64,
    pinned: u32,
    stats: ResidencyStats,
}

impl Residency {
    pub fn new(slots: usize) -> Self {
        assert!(slots > 0 && slots <= u32::MAX as usize);
        Self {
            slots: vec![
                Slot {
                    state: State::Free,
                    pins: 0,
                    last_use: 0,
                };
                slots
            ],
            map: HashMap::new(),
            tick: 0,
            pinned: 0,
            stats: ResidencyStats::default(),
        }
    }

    pub fn capacity(&self) -> usize {
        self.slots.len()
    }

    pub fn stats(&self) -> ResidencyStats {
        self.stats.clone()
    }

    /// Experts currently READY (resident).
    pub fn resident(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| matches!(s.state, State::Ready(..)))
            .count()
    }

    /// Slot and version of `id` if READY (no pin taken).
    pub fn lookup(&self, id: ExpertId) -> Option<(u32, SlotVersion)> {
        let s = *self.map.get(&id)?;
        match self.slots[s as usize].state {
            State::Ready(_, v) => Some((s, v)),
            _ => None,
        }
    }

    pub fn pins(&self, slot: u32) -> u32 {
        self.slots[slot as usize].pins
    }

    fn pin(&mut self, slot: u32) {
        self.tick += 1;
        let s = &mut self.slots[slot as usize];
        s.pins += 1;
        s.last_use = self.tick;
        self.pinned += 1;
        self.stats.pinned_high = self.stats.pinned_high.max(self.pinned);
    }

    pub fn acquire(&mut self, id: ExpertId) -> Acquire {
        if let Some(&slot) = self.map.get(&id) {
            return match self.slots[slot as usize].state {
                State::Ready(_, ver) => {
                    self.pin(slot);
                    self.stats.hits += 1;
                    Acquire::Hit { slot, ver }
                }
                State::Fetching(_) => Acquire::Pending,
                State::Free => unreachable!("map points at a free slot"),
            };
        }
        // Victim: a free slot first, else the least recently used unpinned READY slot.
        let victim = self
            .slots
            .iter()
            .position(|s| s.state == State::Free)
            .or_else(|| {
                self.slots
                    .iter()
                    .enumerate()
                    .filter(|(_, s)| s.pins == 0 && matches!(s.state, State::Ready(..)))
                    .min_by_key(|(_, s)| s.last_use)
                    .map(|(i, _)| i)
            });
        let Some(v) = victim else {
            self.stats.exhausted += 1;
            return Acquire::Exhausted;
        };
        let evicted = match self.slots[v].state {
            State::Ready(old, _) => {
                self.map.remove(&old);
                self.stats.evictions += 1;
                Some(old)
            }
            _ => None,
        };
        self.slots[v].state = State::Fetching(id);
        self.map.insert(id, v as u32);
        self.stats.misses += 1;
        Acquire::Fetch {
            slot: v as u32,
            evicted,
        }
    }

    /// The fetch into `slot` completed and was published at `ver`: READY, pinned for the requester.
    pub fn fetched(&mut self, slot: u32, ver: SlotVersion) {
        let State::Fetching(id) = self.slots[slot as usize].state else {
            panic!("fetched on a slot that is not fetching");
        };
        self.slots[slot as usize].state = State::Ready(id, ver);
        self.pin(slot);
    }

    /// The fetch into `slot` failed: the slot is FREE, the expert is not resident.
    pub fn fetch_failed(&mut self, slot: u32) {
        let State::Fetching(id) = self.slots[slot as usize].state else {
            panic!("fetch_failed on a slot that is not fetching");
        };
        self.map.remove(&id);
        self.slots[slot as usize].state = State::Free;
        self.stats.fetch_failures += 1;
    }

    pub fn release(&mut self, slot: u32) {
        let s = &mut self.slots[slot as usize];
        assert!(s.pins > 0, "release without a pin");
        s.pins -= 1;
        self.pinned -= 1;
    }

    /// Total pins outstanding (0 between requests).
    pub fn pinned(&self) -> u32 {
        self.pinned
    }
}

/// Memory ledger for admitting expert slots (sc#630): everything else is budgeted first, slots get what is
/// left, and a request for more is refused rather than assumed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// MemAvailable at load time (bytes).
    pub available: u64,
    /// Kept free for the OS, the driver and other processes.
    pub headroom: u64,
    /// Resident non-expert weights (attention, router, lm_head, host embedding).
    pub non_expert: u64,
    /// KV cache at the configured max context.
    pub kv: u64,
    /// Launch scratch, staging buffers and other transients.
    pub scratch: u64,
}

impl MemoryBudget {
    pub fn for_slots(&self) -> u64 {
        self.available
            .saturating_sub(self.headroom)
            .saturating_sub(self.non_expert)
            .saturating_sub(self.kv)
            .saturating_sub(self.scratch)
    }

    /// Admits `want` slots of `slot_bytes` each, or refuses with the ledger.
    pub fn admit(&self, want: usize, slot_bytes: u64) -> Result<(), String> {
        let need = want as u64 * slot_bytes;
        if need > self.for_slots() {
            return Err(format!(
                "expert slots need {need} bytes ({want} x {slot_bytes}) but only {} are left: available {} - headroom {} - non-expert {} - kv {} - scratch {}",
                self.for_slots(),
                self.available,
                self.headroom,
                self.non_expert,
                self.kv,
                self.scratch
            ));
        }
        Ok(())
    }
}

/// `MemAvailable` from /proc/meminfo, bytes.
pub fn mem_available() -> Option<u64> {
    let s = std::fs::read_to_string("/proc/meminfo").ok()?;
    let l = s.lines().find(|l| l.starts_with("MemAvailable:"))?;
    let kb: u64 = l.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(n: u64) -> SlotVersion {
        SlotVersion {
            gate_up: n,
            down: n + 1000,
        }
    }

    fn load(r: &mut Residency, id: ExpertId, n: u64) -> u32 {
        match r.acquire(id) {
            Acquire::Fetch { slot, .. } => {
                r.fetched(slot, v(n));
                slot
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn miss_then_hit_and_release() {
        let mut r = Residency::new(2);
        let s = load(&mut r, (0, 5), 1);
        assert_eq!(r.pins(s), 1);
        r.release(s);
        assert_eq!(r.acquire((0, 5)), Acquire::Hit { slot: s, ver: v(1) });
        r.release(s);
        assert_eq!(r.stats().hits, 1);
        assert_eq!(r.stats().misses, 1);
        assert_eq!(r.pinned(), 0);
    }

    #[test]
    fn lru_eviction_order_and_layer_crossing_identities() {
        let mut r = Residency::new(3);
        for (i, id) in [(0, 1), (1, 1), (2, 1)].into_iter().enumerate() {
            let s = load(&mut r, id, i as u64 + 1);
            r.release(s);
        }
        // touch (0,1) so (1,1) is the least recent
        let Acquire::Hit { slot, .. } = r.acquire((0, 1)) else {
            panic!()
        };
        r.release(slot);
        match r.acquire((3, 1)) {
            Acquire::Fetch { evicted, slot } => {
                assert_eq!(
                    evicted,
                    Some((1, 1)),
                    "same expert number, other layer: distinct ids"
                );
                r.fetched(slot, v(9));
                r.release(slot);
            }
            o => panic!("{o:?}"),
        }
        assert!(r.lookup((1, 1)).is_none());
        assert!(r.lookup((0, 1)).is_some() && r.lookup((2, 1)).is_some());
        assert_eq!(r.stats().evictions, 1);
    }

    #[test]
    fn pinned_slots_are_never_evicted_and_exhaustion_is_explicit() {
        let mut r = Residency::new(2);
        let a = load(&mut r, (0, 0), 1);
        let b = load(&mut r, (0, 1), 2);
        assert_eq!(r.acquire((0, 2)), Acquire::Exhausted);
        assert_eq!(r.stats().exhausted, 1);
        r.release(b);
        match r.acquire((0, 2)) {
            Acquire::Fetch { slot, evicted } => {
                assert_eq!((slot, evicted), (b, Some((0, 1))));
                r.fetched(slot, v(3));
                r.release(slot);
            }
            o => panic!("{o:?}"),
        }
        assert_eq!(r.lookup((0, 0)).map(|x| x.0), Some(a), "pinned slot kept");
        r.release(a);
    }

    #[test]
    fn simultaneous_readers_share_a_slot() {
        let mut r = Residency::new(1);
        let s = load(&mut r, (4, 4), 1);
        assert!(matches!(r.acquire((4, 4)), Acquire::Hit { .. }));
        assert_eq!(r.pins(s), 2);
        assert_eq!(r.acquire((4, 5)), Acquire::Exhausted);
        r.release(s);
        assert_eq!(
            r.acquire((4, 5)),
            Acquire::Exhausted,
            "one reader still holds it"
        );
        r.release(s);
        assert!(matches!(r.acquire((4, 5)), Acquire::Fetch { .. }));
    }

    #[test]
    fn fetching_is_pending_for_others_and_failure_recovers() {
        let mut r = Residency::new(2);
        let Acquire::Fetch { slot, .. } = r.acquire((1, 2)) else {
            panic!()
        };
        assert_eq!(r.acquire((1, 2)), Acquire::Pending);
        r.fetch_failed(slot);
        assert!(r.lookup((1, 2)).is_none());
        assert_eq!(r.stats().fetch_failures, 1);
        let s2 = load(&mut r, (1, 2), 7);
        assert_eq!(r.lookup((1, 2)), Some((s2, v(7))));
        r.release(s2);
    }

    #[test]
    fn aba_reuse_gets_new_versions() {
        // The same slot holds (0,0), then (0,1), then (0,0) again: the backend version differs each time,
        // so a stale (slot, version) handle from the first load can never match the third.
        let mut r = Residency::new(1);
        let s = load(&mut r, (0, 0), 1);
        r.release(s);
        let s1 = load(&mut r, (0, 1), 2);
        r.release(s1);
        let s2 = load(&mut r, (0, 0), 3);
        assert_eq!((s, s1, s2), (0, 0, 0));
        assert_eq!(r.lookup((0, 0)), Some((0, v(3))));
        r.release(s2);
    }

    #[test]
    fn budget_admits_and_refuses() {
        let b = MemoryBudget {
            available: 100,
            headroom: 10,
            non_expert: 20,
            kv: 30,
            scratch: 5,
        };
        assert_eq!(b.for_slots(), 35);
        assert!(b.admit(7, 5).is_ok());
        let e = b.admit(8, 5).unwrap_err();
        assert!(e.contains("need 40"), "{e}");
    }
}
