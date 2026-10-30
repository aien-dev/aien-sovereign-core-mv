#![allow(dead_code)]

use aien_expert_residency::{
    Acquired, ExpertIdentity, ExpertPart, FetchTicket, KvShape, MemoryBudget, Pin, ResidencyConfig,
    ResidencyManager,
};

pub const SLOT: u64 = 4096;
pub const MIB: u64 = 1 << 20;

pub fn kv_none() -> KvShape {
    KvShape {
        layers: 0,
        kv_heads: 0,
        head_dim: 0,
        bytes_per_element: 0,
        max_context_tokens: 0,
        max_sequences: 0,
    }
}

/// A config with exactly `slots` slots of `SLOT` bytes and the given
/// in-flight limits. Staging is sized to `max_in_flight_bytes`.
pub fn config(slots: u32, max_in_flight: u32, max_in_flight_bytes: u64) -> ResidencyConfig {
    let staging = max_in_flight_bytes;
    ResidencyConfig {
        budget: MemoryBudget {
            total_bytes: 1000 + staging + u64::from(slots) * SLOT + 100,
            safety_headroom_bytes: 1000,
            non_expert_weight_bytes: 0,
            kv: kv_none(),
            scratch_bytes: 0,
            staging_bytes: staging,
        },
        slot_bytes: SLOT,
        max_slots: Some(slots),
        max_in_flight,
        max_in_flight_bytes,
    }
}

pub fn manager(slots: u32) -> ResidencyManager<ExpertIdentity> {
    ResidencyManager::new(config(slots, slots.max(1), u64::from(slots.max(1)) * SLOT)).unwrap()
}

pub fn id(layer: u32, expert: u32) -> ExpertIdentity {
    ExpertIdentity {
        checkpoint_digest: [7; 32],
        layer,
        expert,
        part: ExpertPart::Down,
    }
}

/// Drive a fetch ticket all the way to Ready.
pub fn complete(m: &mut ResidencyManager<ExpertIdentity>, t: &FetchTicket, now: u64) {
    m.mark_verified(t).unwrap();
    m.mark_copying(t).unwrap();
    m.publish(t, now).unwrap();
}

/// Acquire `key`, completing a fetch if it misses. Returns the pin.
pub fn load(m: &mut ResidencyManager<ExpertIdentity>, key: ExpertIdentity) -> Pin {
    match m.acquire(key, SLOT, 0).unwrap() {
        Acquired::Ready(p) | Acquired::Pending(p) => p,
        Acquired::Fetch(p, t) => {
            complete(m, &t, 0);
            p
        }
    }
}

/// Load and immediately release.
pub fn touch(m: &mut ResidencyManager<ExpertIdentity>, key: ExpertIdentity) {
    let p = load(m, key);
    m.release(p).unwrap();
}

/// xorshift64*: deterministic, dependency-free generator for property tests.
pub struct Rng(pub u64);

impl Rng {
    pub fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    pub fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}
