//! The residency state machine.
//!
//! ## Lifecycle
//!
//! ```text
//! Absent --acquire (miss)--> Fetching --mark_verified--> Verified
//!        --mark_copying--> Copying --publish--> Ready
//! Ready --pin--> InUse --last release--> Evictable --evict--> Absent
//! Fetching | Verified | Copying --fail--> Failed (held until the last pin
//!                                          is released) --> Absent
//! ```
//!
//! `publish` must be called only after the transfer completed, the digest
//! matched and the GPU visibility fence signalled (aienix#4 invariant 3).
//!
//! ## Pins
//!
//! [`ResidencyManager::acquire`] always returns a [`Pin`], also on a miss: a
//! request pins its experts from the moment the router selects them until
//! its GPU read fence has passed, then releases each pin. An entry with any
//! pin, or with a fetch in flight, is never evicted. A [`Pin`] is not
//! `Clone`, so it can be released only once.
//!
//! ## Generations (ABA)
//!
//! Every admission gets a fresh generation from one counter that never
//! repeats. Pins and tickets carry `(slot, generation)`; after a slot is
//! reused, an old pin or ticket is refused as stale instead of acting on the
//! new occupant.
//!
//! ## Eviction order
//!
//! Least recently used, by a logical clock that advances on every acquire
//! and release. Ticks are unique, so the order is total and deterministic for
//! a given sequence of calls. Free slots are reused lowest index first.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Debug;

use crate::budget::{Ledger, MemoryBudget};
use crate::error::{BudgetError, FailureCause, Refusal};

/// Index of a fixed logical expert slot.
pub type SlotId = u32;

/// Which part of an expert's MLP an identity names. `GateUp` is for
/// checkpoints that store gate and up fused.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ExpertPart {
    Gate,
    Up,
    GateUp,
    Down,
}

/// A stable expert identity: checkpoint digest, layer, expert, part. The
/// manager is generic over the key, so the sc#629 store key (which also
/// carries the quantization format) can be used directly instead.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ExpertIdentity {
    pub checkpoint_digest: [u8; 32],
    pub layer: u32,
    pub expert: u32,
    pub part: ExpertPart,
}

/// Internal phase of an admitted entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Phase {
    Fetching,
    Verified,
    Copying,
    Ready,
    /// The fetch failed; the slot is held only until its pins are released.
    Failed(FailureCause),
}

/// Externally visible state of an expert identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExpertState {
    Absent,
    Fetching,
    Verified,
    Copying,
    /// Published and not pinned, not yet used since publish.
    Ready,
    /// Published and pinned by at least one request.
    InUse,
    /// Published, not pinned, used at least once.
    Evictable,
}

/// What a pin holder sees when it polls.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PinStatus {
    /// The expert is resident and published in this slot generation.
    Ready,
    /// Still in flight; the holder must wait (never substitute).
    Waiting(Phase),
}

/// A request's hold on one expert. Not `Clone`: release it exactly once.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a pin must be released, or the slot stays pinned"]
pub struct Pin {
    slot: SlotId,
    generation: u64,
}

impl Pin {
    pub fn slot(&self) -> SlotId {
        self.slot
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
}

/// The right to drive one fetch through its steps. Handed to exactly one
/// caller, the one whose acquire missed.
#[derive(Debug, PartialEq, Eq)]
#[must_use = "a fetch ticket must be published or failed"]
pub struct FetchTicket {
    slot: SlotId,
    generation: u64,
    bytes: u64,
}

impl FetchTicket {
    pub fn slot(&self) -> SlotId {
        self.slot
    }
    pub fn generation(&self) -> u64 {
        self.generation
    }
    pub fn bytes(&self) -> u64 {
        self.bytes
    }
}

/// Result of a successful acquire.
#[derive(Debug, PartialEq, Eq)]
pub enum Acquired {
    /// Hit: resident and published.
    Ready(Pin),
    /// Miss: a slot was assigned; the caller must fetch into it.
    Fetch(Pin, FetchTicket),
    /// Another caller's fetch for the same key is in flight; wait for it.
    Pending(Pin),
}

impl Acquired {
    pub fn pin(&self) -> &Pin {
        match self {
            Self::Ready(p) | Self::Fetch(p, _) | Self::Pending(p) => p,
        }
    }
}

/// Configuration of one manager.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResidencyConfig {
    pub budget: MemoryBudget,
    /// Size of one fixed expert slot. Every admitted object must fit.
    pub slot_bytes: u64,
    /// Optional cap on the number of slots below what the budget allows.
    pub max_slots: Option<u32>,
    /// Maximum fetches in flight (queue depth).
    pub max_in_flight: u32,
    /// Maximum bytes in flight. Must fit the staging reservation and hold at
    /// least one full slot.
    pub max_in_flight_bytes: u64,
}

/// Counters. All monotonic except the `current_*` gauges.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Stats {
    pub hits: u64,
    /// Acquires that joined a fetch already in flight.
    pub joined: u64,
    pub misses: u64,
    pub published: u64,
    pub evictions: u64,
    pub failures: u64,
    /// Fresh fetches rolled back by [`ResidencyManager::acquire_all`].
    pub rolled_back: u64,
    pub refusals: u64,
    pub current_in_flight: u32,
    pub current_in_flight_bytes: u64,
    pub peak_in_flight_bytes: u64,
    pub current_resident_bytes: u64,
    pub peak_resident_bytes: u64,
    pub current_pins: u64,
    /// Sum and maximum of admit-to-publish time, from caller timestamps.
    pub fetch_wait_ns_total: u64,
    pub fetch_wait_ns_max: u64,
}

#[derive(Debug, Clone)]
struct Entry<K> {
    key: K,
    generation: u64,
    phase: Phase,
    pins: u32,
    bytes: u64,
    last_use: u64,
    used: bool,
    admitted_ns: u64,
}

impl<K> Entry<K> {
    fn in_flight(&self) -> bool {
        matches!(
            self.phase,
            Phase::Fetching | Phase::Verified | Phase::Copying
        )
    }
    fn evictable(&self) -> bool {
        self.phase == Phase::Ready && self.pins == 0
    }
}

/// The bounded residency manager.
#[derive(Debug)]
pub struct ResidencyManager<K> {
    ledger: Ledger,
    max_in_flight: u32,
    max_in_flight_bytes: u64,
    slots: Vec<Option<Entry<K>>>,
    index: BTreeMap<K, SlotId>,
    free: BTreeSet<SlotId>,
    clock: u64,
    next_generation: u64,
    stats: Stats,
}

impl<K: Ord + Clone + Debug> ResidencyManager<K> {
    /// Plan the ledger and create an empty manager. Fails closed on any
    /// oversubscription or inconsistent limit.
    pub fn new(config: ResidencyConfig) -> Result<Self, BudgetError> {
        let ledger = Ledger::plan(&config.budget, config.slot_bytes, config.max_slots)?;
        if config.max_in_flight == 0 {
            return Err(BudgetError::ZeroInFlight);
        }
        if config.max_in_flight_bytes > ledger.staging_bytes {
            return Err(BudgetError::InFlightBytesExceedStaging {
                max_in_flight_bytes: config.max_in_flight_bytes,
                staging_bytes: ledger.staging_bytes,
            });
        }
        if config.max_in_flight_bytes < config.slot_bytes {
            return Err(BudgetError::InFlightBytesBelowSlot {
                max_in_flight_bytes: config.max_in_flight_bytes,
                slot_bytes: config.slot_bytes,
            });
        }
        let n = ledger.slot_count;
        Ok(Self {
            ledger,
            max_in_flight: config.max_in_flight,
            max_in_flight_bytes: config.max_in_flight_bytes,
            slots: (0..n).map(|_| None).collect(),
            index: BTreeMap::new(),
            free: (0..n).collect(),
            clock: 0,
            next_generation: 1,
            stats: Stats::default(),
        })
    }

    pub fn ledger(&self) -> &Ledger {
        &self.ledger
    }

    pub fn stats(&self) -> Stats {
        self.stats
    }

    pub fn slot_count(&self) -> u32 {
        self.ledger.slot_count
    }

    /// The state of one identity.
    pub fn state(&self, key: &K) -> ExpertState {
        let Some(&slot) = self.index.get(key) else {
            return ExpertState::Absent;
        };
        let e = self.entry(slot);
        match e.phase {
            Phase::Fetching => ExpertState::Fetching,
            Phase::Verified => ExpertState::Verified,
            Phase::Copying => ExpertState::Copying,
            Phase::Ready if e.pins > 0 => ExpertState::InUse,
            Phase::Ready if e.used => ExpertState::Evictable,
            Phase::Ready => ExpertState::Ready,
            // Failed entries are removed from the index.
            Phase::Failed(_) => ExpertState::Absent,
        }
    }

    /// The slot and generation currently holding `key`, if admitted.
    pub fn slot_of(&self, key: &K) -> Option<(SlotId, u64)> {
        self.index.get(key).map(|&s| (s, self.entry(s).generation))
    }

    /// Pin `key` of `bytes` bytes. On a hit the pin is ready; on a miss a
    /// slot is assigned (evicting the least recently used unpinned resident
    /// entry if no slot is free) and the caller gets the fetch ticket.
    /// `now_ns` is the caller's monotonic clock, used only for wait counters.
    pub fn acquire(&mut self, key: K, bytes: u64, now_ns: u64) -> Result<Acquired, Refusal> {
        let r = self.acquire_inner(key, bytes, now_ns);
        if r.is_err() {
            self.stats.refusals += 1;
        }
        r
    }

    fn acquire_inner(&mut self, key: K, bytes: u64, now_ns: u64) -> Result<Acquired, Refusal> {
        if let Some(&slot) = self.index.get(&key) {
            let e = self.entry(slot);
            if e.bytes != bytes {
                return Err(Refusal::SizeMismatch {
                    admitted: e.bytes,
                    requested: bytes,
                });
            }
            let pins = e.pins.checked_add(1).ok_or(Refusal::PinOverflow { slot })?;
            let tick = self.tick();
            let e = self.slots[slot as usize].as_mut().expect("indexed slot");
            e.pins = pins;
            e.last_use = tick;
            let pin = Pin {
                slot,
                generation: e.generation,
            };
            let ready = e.phase == Phase::Ready;
            self.stats.current_pins += 1;
            return Ok(if ready {
                self.stats.hits += 1;
                Acquired::Ready(pin)
            } else {
                self.stats.joined += 1;
                Acquired::Pending(pin)
            });
        }

        // Miss: validate everything before evicting anything.
        if bytes == 0 {
            return Err(Refusal::ZeroSizedObject);
        }
        if bytes > self.ledger.slot_bytes {
            return Err(Refusal::ObjectTooLarge {
                bytes,
                slot_bytes: self.ledger.slot_bytes,
            });
        }
        if self.stats.current_in_flight >= self.max_in_flight {
            return Err(Refusal::InFlightLimit {
                limit: self.max_in_flight,
            });
        }
        let in_flight_bytes = self.stats.current_in_flight_bytes;
        if in_flight_bytes + bytes > self.max_in_flight_bytes {
            return Err(Refusal::InFlightBytesLimit {
                in_flight_bytes,
                requested: bytes,
                limit: self.max_in_flight_bytes,
            });
        }
        let slot = match self.free.first().copied() {
            Some(s) => {
                self.free.remove(&s);
                s
            }
            None => {
                let victim = self.lru_victim().ok_or_else(|| self.exhaustion())?;
                self.evict(victim);
                victim
            }
        };

        let tick = self.tick();
        let generation = self.next_generation;
        self.next_generation += 1;
        self.slots[slot as usize] = Some(Entry {
            key: key.clone(),
            generation,
            phase: Phase::Fetching,
            pins: 1,
            bytes,
            last_use: tick,
            used: false,
            admitted_ns: now_ns,
        });
        self.index.insert(key, slot);
        self.stats.misses += 1;
        self.stats.current_pins += 1;
        self.stats.current_in_flight += 1;
        self.stats.current_in_flight_bytes += bytes;
        self.stats.peak_in_flight_bytes = self
            .stats
            .peak_in_flight_bytes
            .max(self.stats.current_in_flight_bytes);
        Ok(Acquired::Fetch(
            Pin { slot, generation },
            FetchTicket {
                slot,
                generation,
                bytes,
            },
        ))
    }

    /// Acquire every key of one request, or none. Keys may repeat (each
    /// occurrence takes its own pin). On a refusal every pin taken by this
    /// call is released and every fetch it started is rolled back, then the
    /// refusal is returned. Entries evicted to make room stay evicted.
    pub fn acquire_all(
        &mut self,
        keys: &[(K, u64)],
        now_ns: u64,
    ) -> Result<Vec<Acquired>, Refusal> {
        let mut taken: Vec<Acquired> = Vec::with_capacity(keys.len());
        for (key, bytes) in keys {
            match self.acquire(key.clone(), *bytes, now_ns) {
                Ok(a) => taken.push(a),
                Err(refusal) => {
                    // Release in reverse so fresh fetches lose their last
                    // pin before being rolled back.
                    while let Some(a) = taken.pop() {
                        self.rollback(a);
                    }
                    return Err(refusal);
                }
            }
        }
        Ok(taken)
    }

    fn rollback(&mut self, a: Acquired) {
        match a {
            Acquired::Ready(pin) | Acquired::Pending(pin) => {
                self.release(pin).expect("pin taken by this call");
            }
            Acquired::Fetch(pin, ticket) => {
                self.release(pin).expect("pin taken by this call");
                let slot = ticket.slot;
                let e = self.entry(slot);
                debug_assert!(e.generation == ticket.generation && e.phase == Phase::Fetching);
                // Pins are released newest first, so every later duplicate of
                // this key (which joined this fetch) is already released.
                assert_eq!(e.pins, 0, "fresh fetch still pinned during rollback");
                self.finish_in_flight(slot);
                let e = self.slots[slot as usize].take().expect("slot");
                self.index.remove(&e.key);
                self.free.insert(slot);
                self.stats.rolled_back += 1;
            }
        }
    }

    /// Fetching -> Verified: the bytes are in staging and the digest matched.
    pub fn mark_verified(&mut self, ticket: &FetchTicket) -> Result<(), Refusal> {
        self.step(ticket, Phase::Fetching, Phase::Verified)
    }

    /// Verified -> Copying: the staging-to-slot copy has started.
    pub fn mark_copying(&mut self, ticket: &FetchTicket) -> Result<(), Refusal> {
        self.step(ticket, Phase::Verified, Phase::Copying)
    }

    /// Copying -> Ready. Call only after the copy completed and the GPU
    /// visibility fence signalled.
    pub fn publish(&mut self, ticket: &FetchTicket, now_ns: u64) -> Result<(), Refusal> {
        self.step(ticket, Phase::Copying, Phase::Ready)?;
        let slot = ticket.slot;
        let (bytes, admitted_ns) = {
            let e = self.entry(slot);
            (e.bytes, e.admitted_ns)
        };
        self.finish_in_flight(slot);
        let wait = now_ns.saturating_sub(admitted_ns);
        let s = &mut self.stats;
        s.published += 1;
        s.current_resident_bytes += bytes;
        s.peak_resident_bytes = s.peak_resident_bytes.max(s.current_resident_bytes);
        s.fetch_wait_ns_total = s.fetch_wait_ns_total.saturating_add(wait);
        s.fetch_wait_ns_max = s.fetch_wait_ns_max.max(wait);
        Ok(())
    }

    /// Any in-flight phase -> Failed. The key is unmapped at once, so the
    /// next acquire starts a clean fetch; pin holders see
    /// [`Refusal::FetchFailed`] when they poll and must release.
    pub fn fail(&mut self, ticket: &FetchTicket, cause: FailureCause) -> Result<(), Refusal> {
        let slot = self.check_ticket(ticket)?;
        let e = self.entry(slot);
        if !e.in_flight() {
            return Err(Refusal::WrongPhase {
                slot,
                expected: Phase::Fetching,
                found: e.phase,
            });
        }
        self.finish_in_flight(slot);
        self.stats.failures += 1;
        let e = self.slots[slot as usize].as_mut().expect("slot");
        e.phase = Phase::Failed(cause);
        let key = e.key.clone();
        let pins = e.pins;
        self.index.remove(&key);
        if pins == 0 {
            self.slots[slot as usize] = None;
            self.free.insert(slot);
        }
        Ok(())
    }

    /// The current phase of the fetch behind `ticket`. A ticket whose slot
    /// generation is gone (failed and freed, or evicted after publish) is
    /// refused as stale.
    pub fn ticket_phase(&self, ticket: &FetchTicket) -> Result<Phase, Refusal> {
        let slot = self.check_ticket(ticket)?;
        Ok(self.entry(slot).phase)
    }

    /// Where the pinned expert stands.
    pub fn poll(&self, pin: &Pin) -> Result<PinStatus, Refusal> {
        let e = self.pinned_entry(pin)?;
        match e.phase {
            Phase::Ready => Ok(PinStatus::Ready),
            Phase::Failed(cause) => Err(Refusal::FetchFailed {
                slot: pin.slot,
                generation: pin.generation,
                cause,
            }),
            p => Ok(PinStatus::Waiting(p)),
        }
    }

    /// Drop one pin. Call after the GPU read fence of the request has
    /// passed. The slot becomes evictable when its last pin goes.
    pub fn release(&mut self, pin: Pin) -> Result<(), Refusal> {
        self.pinned_entry(&pin)?;
        let tick = self.tick();
        let slot = pin.slot;
        let e = self.slots[slot as usize].as_mut().expect("slot");
        e.pins -= 1;
        e.last_use = tick;
        if e.phase == Phase::Ready {
            e.used = true;
        }
        let free_failed = matches!(e.phase, Phase::Failed(_)) && e.pins == 0;
        self.stats.current_pins -= 1;
        if free_failed {
            self.slots[slot as usize] = None;
            self.free.insert(slot);
        }
        Ok(())
    }

    /// The slot that the next miss would evict, if no slot is free.
    pub fn next_victim(&self) -> Option<SlotId> {
        if self.free.is_empty() {
            self.lru_victim()
        } else {
            None
        }
    }

    /// Check every internal invariant. For tests and debug assertions.
    pub fn check_invariants(&self) -> Result<(), String> {
        let mut pins = 0u64;
        let mut in_flight = 0u32;
        let mut in_flight_bytes = 0u64;
        let mut resident = 0u64;
        let mut generations = BTreeSet::new();
        for (i, s) in self.slots.iter().enumerate() {
            let slot = i as SlotId;
            match s {
                None => {
                    if !self.free.contains(&slot) {
                        return Err(format!("empty slot {slot} not on free list"));
                    }
                }
                Some(e) => {
                    if self.free.contains(&slot) {
                        return Err(format!("occupied slot {slot} on free list"));
                    }
                    if !generations.insert(e.generation) || e.generation >= self.next_generation {
                        return Err(format!("slot {slot} bad generation {}", e.generation));
                    }
                    if e.bytes == 0 || e.bytes > self.ledger.slot_bytes {
                        return Err(format!("slot {slot} bad size {}", e.bytes));
                    }
                    pins += u64::from(e.pins);
                    match e.phase {
                        Phase::Failed(_) => {
                            if e.pins == 0 {
                                return Err(format!("failed slot {slot} held with no pins"));
                            }
                            if self.index.get(&e.key) == Some(&slot) {
                                return Err(format!("failed slot {slot} still indexed"));
                            }
                        }
                        _ => {
                            if self.index.get(&e.key) != Some(&slot) {
                                return Err(format!("slot {slot} not indexed by its key"));
                            }
                        }
                    }
                    if e.in_flight() {
                        in_flight += 1;
                        in_flight_bytes += e.bytes;
                    }
                    if e.phase == Phase::Ready {
                        resident += e.bytes;
                    }
                }
            }
        }
        for (k, &slot) in &self.index {
            match self.slots.get(slot as usize).and_then(Option::as_ref) {
                Some(e) if &e.key == k => {}
                _ => return Err(format!("index entry {k:?} -> {slot} dangling")),
            }
        }
        let s = &self.stats;
        if pins != s.current_pins {
            return Err(format!("pins {pins} != counter {}", s.current_pins));
        }
        if in_flight != s.current_in_flight || in_flight_bytes != s.current_in_flight_bytes {
            return Err("in-flight counters drifted".into());
        }
        if in_flight > self.max_in_flight || in_flight_bytes > self.max_in_flight_bytes {
            return Err("in-flight limits exceeded".into());
        }
        if resident != s.current_resident_bytes || resident > self.ledger.expert_pool_bytes {
            return Err(format!("resident bytes {resident} inconsistent"));
        }
        if s.peak_in_flight_bytes > self.ledger.staging_bytes
            || s.peak_resident_bytes > self.ledger.expert_pool_bytes
        {
            return Err("high-water mark above reservation".into());
        }
        Ok(())
    }

    // ---- internals ----

    fn tick(&mut self) -> u64 {
        self.clock += 1;
        self.clock
    }

    fn entry(&self, slot: SlotId) -> &Entry<K> {
        self.slots[slot as usize].as_ref().expect("occupied slot")
    }

    fn lru_victim(&self) -> Option<SlotId> {
        self.slots
            .iter()
            .enumerate()
            .filter_map(|(i, s)| {
                s.as_ref()
                    .filter(|e| e.evictable())
                    .map(|e| (e.last_use, i))
            })
            .min()
            .map(|(_, i)| i as SlotId)
    }

    fn exhaustion(&self) -> Refusal {
        let pinned = self.slots.iter().flatten().filter(|e| e.pins > 0).count() as u32;
        Refusal::PinnedExhaustion {
            slots: self.ledger.slot_count,
            pinned,
            in_flight: self.stats.current_in_flight,
        }
    }

    fn evict(&mut self, slot: SlotId) {
        let e = self.slots[slot as usize].take().expect("victim");
        debug_assert!(e.evictable());
        self.index.remove(&e.key);
        self.stats.current_resident_bytes -= e.bytes;
        self.stats.evictions += 1;
    }

    fn finish_in_flight(&mut self, slot: SlotId) {
        let bytes = self.entry(slot).bytes;
        self.stats.current_in_flight -= 1;
        self.stats.current_in_flight_bytes -= bytes;
    }

    fn check_ticket(&self, t: &FetchTicket) -> Result<SlotId, Refusal> {
        match self.slots.get(t.slot as usize).and_then(Option::as_ref) {
            Some(e) if e.generation == t.generation => Ok(t.slot),
            _ => Err(Refusal::StaleTicket {
                slot: t.slot,
                generation: t.generation,
            }),
        }
    }

    fn pinned_entry(&self, pin: &Pin) -> Result<&Entry<K>, Refusal> {
        match self.slots.get(pin.slot as usize).and_then(Option::as_ref) {
            Some(e) if e.generation == pin.generation && e.pins > 0 => Ok(e),
            _ => Err(Refusal::StalePin {
                slot: pin.slot,
                generation: pin.generation,
            }),
        }
    }

    fn step(&mut self, ticket: &FetchTicket, from: Phase, to: Phase) -> Result<(), Refusal> {
        let slot = self.check_ticket(ticket)?;
        let e = self.slots[slot as usize].as_mut().expect("slot");
        if e.phase != from {
            return Err(Refusal::WrongPhase {
                slot,
                expected: from,
                found: e.phase,
            });
        }
        e.phase = to;
        Ok(())
    }
}
