mod common;

use aien_expert_residency::{
    AllocationReport, BudgetError, Category, KvShape, Ledger, MemoryBudget, ResidencyConfig,
    ResidencyManager,
};
use common::*;

const GIB: u64 = 1 << 30;

/// Qwen3-30B-A3B-like attention shape: 48 layers, 4 KV heads, head dim 128,
/// BF16, 32K context, 1 sequence.
fn qwen_kv(ctx: u32, seqs: u32) -> KvShape {
    KvShape {
        layers: 48,
        kv_heads: 4,
        head_dim: 128,
        bytes_per_element: 2,
        max_context_tokens: ctx,
        max_sequences: seqs,
    }
}

fn spark_budget(ctx: u32) -> MemoryBudget {
    MemoryBudget {
        total_bytes: 128 * GIB,
        safety_headroom_bytes: 8 * GIB,
        non_expert_weight_bytes: 2 * GIB,
        kv: qwen_kv(ctx, 1),
        scratch_bytes: 2 * GIB,
        staging_bytes: 256 * MIB,
    }
}

#[test]
fn kv_bytes_match_hand_computation() {
    // 2 * 48 * 4 * 128 * 2 * 32768 * 1 = 3 GiB.
    assert_eq!(qwen_kv(32768, 1).bytes().unwrap(), 3 * GIB);
    assert_eq!(qwen_kv(32768, 4).bytes().unwrap(), 12 * GIB);
    assert_eq!(kv_none().bytes().unwrap(), 0);
}

#[test]
fn kv_overflow_is_refused() {
    let kv = KvShape {
        layers: u32::MAX,
        kv_heads: u32::MAX,
        head_dim: u32::MAX,
        bytes_per_element: u32::MAX,
        max_context_tokens: u32::MAX,
        max_sequences: u32::MAX,
    };
    assert_eq!(kv.bytes(), Err(BudgetError::Overflow { what: "kv cache" }));
}

#[test]
fn ledger_partitions_total_exactly() {
    let slot = 4_718_592; // one Qwen3-30B expert, 3 x 2048 x 768 FP8
    for ctx in [4096, 32768, 65536] {
        let b = spark_budget(ctx);
        let l = Ledger::plan(&b, slot, None).unwrap();
        let sum = l.safety_headroom_bytes
            + l.non_expert_weight_bytes
            + l.kv_cache_bytes
            + l.scratch_bytes
            + l.staging_bytes
            + l.expert_pool_bytes
            + l.unassigned_bytes;
        assert_eq!(sum, b.total_bytes, "ctx {ctx}");
        assert!(l.unassigned_bytes < slot);
        assert_eq!(l.expert_pool_bytes, u64::from(l.slot_count) * slot);
        assert_eq!(l.committed_bytes() + l.unassigned_bytes, b.total_bytes);
        // Never assume all memory can store experts.
        assert!(l.expert_pool_bytes <= b.total_bytes - b.safety_headroom_bytes);
    }
}

#[test]
fn longer_context_leaves_fewer_slots() {
    let slot = 4_718_592;
    let counts: Vec<u32> = [4096, 32768, 65536]
        .iter()
        .map(|&c| {
            Ledger::plan(&spark_budget(c), slot, None)
                .unwrap()
                .slot_count
        })
        .collect();
    assert!(counts[0] > counts[1] && counts[1] > counts[2], "{counts:?}");
}

#[test]
fn oversubscription_is_refused() {
    let mut b = spark_budget(32768);
    b.non_expert_weight_bytes = 120 * GIB;
    match Ledger::plan(&b, 4096, None) {
        Err(BudgetError::Oversubscribed { required, total }) => {
            assert!(required > total);
            assert_eq!(total, 128 * GIB);
        }
        other => panic!("{other:?}"),
    }
}

#[test]
fn exactly_full_reservations_leave_no_slot() {
    let b = MemoryBudget {
        total_bytes: 10_000,
        safety_headroom_bytes: 10_000,
        non_expert_weight_bytes: 0,
        kv: kv_none(),
        scratch_bytes: 0,
        staging_bytes: 0,
    };
    assert_eq!(
        Ledger::plan(&b, 1, None),
        Err(BudgetError::NoExpertSlots {
            available_bytes: 0,
            slot_bytes: 1
        })
    );
}

#[test]
fn less_than_one_slot_is_refused() {
    let b = MemoryBudget {
        total_bytes: 10_000,
        safety_headroom_bytes: 6_000,
        non_expert_weight_bytes: 0,
        kv: kv_none(),
        scratch_bytes: 0,
        staging_bytes: 0,
    };
    assert_eq!(
        Ledger::plan(&b, 4096, None),
        Err(BudgetError::NoExpertSlots {
            available_bytes: 4_000,
            slot_bytes: 4096
        })
    );
    assert_eq!(Ledger::plan(&b, 0, None), Err(BudgetError::ZeroSlotBytes));
    assert_eq!(
        Ledger::plan(&b, 4000, Some(0)).map(|l| l.slot_count),
        Err(BudgetError::NoExpertSlots {
            available_bytes: 4_000,
            slot_bytes: 4000
        })
    );
}

#[test]
fn reservation_sum_overflow_is_refused() {
    let mut b = spark_budget(4096);
    b.safety_headroom_bytes = u64::MAX;
    assert_eq!(
        Ledger::plan(&b, 4096, None),
        Err(BudgetError::Overflow {
            what: "fixed reservations"
        })
    );
}

#[test]
fn max_slots_caps_pool_and_rest_is_unassigned() {
    let b = spark_budget(4096);
    let l = Ledger::plan(&b, MIB, Some(10)).unwrap();
    assert_eq!(l.slot_count, 10);
    assert_eq!(l.expert_pool_bytes, 10 * MIB);
    assert!(l.unassigned_bytes > 100 * GIB);
}

#[test]
fn allocation_report_checked_per_category() {
    let l = Ledger::plan(&spark_budget(32768), MIB, Some(100)).unwrap();
    let ok = AllocationReport {
        non_expert_weight_bytes: 2 * GIB,
        kv_cache_bytes: 3 * GIB,
        scratch_bytes: GIB,
        staging_bytes: 256 * MIB,
        expert_pool_bytes: 100 * MIB,
    };
    assert_eq!(l.check_allocations(&ok), Ok(()));
    let over = AllocationReport {
        kv_cache_bytes: 3 * GIB + 1,
        ..ok
    };
    assert_eq!(
        l.check_allocations(&over),
        Err(BudgetError::AllocationExceedsReservation {
            category: Category::KvCache,
            reserved: 3 * GIB,
            actual: 3 * GIB + 1
        })
    );
    let pool_over = AllocationReport {
        expert_pool_bytes: 100 * MIB + 1,
        ..ok
    };
    assert!(matches!(
        l.check_allocations(&pool_over),
        Err(BudgetError::AllocationExceedsReservation {
            category: Category::ExpertPool,
            ..
        })
    ));
}

#[test]
fn manager_config_limits_are_validated() {
    let mut c = config(4, 2, 2 * SLOT);
    c.max_in_flight = 0;
    assert_eq!(
        ResidencyManager::<u32>::new(c).err(),
        Some(BudgetError::ZeroInFlight)
    );
    let mut c = config(4, 2, 2 * SLOT);
    c.max_in_flight_bytes = 3 * SLOT;
    assert_eq!(
        ResidencyManager::<u32>::new(c).err(),
        Some(BudgetError::InFlightBytesExceedStaging {
            max_in_flight_bytes: 3 * SLOT,
            staging_bytes: 2 * SLOT
        })
    );
    let mut c = config(4, 2, 2 * SLOT);
    c.max_in_flight_bytes = SLOT - 1;
    assert_eq!(
        ResidencyManager::<u32>::new(c).err(),
        Some(BudgetError::InFlightBytesBelowSlot {
            max_in_flight_bytes: SLOT - 1,
            slot_bytes: SLOT
        })
    );
}

/// High-water marks stay within their reservations and are stable: running
/// the same workload twice more does not raise them past the first plateau.
#[test]
fn high_water_is_bounded_and_stable_at_varied_cache_sizes() {
    for slots in [1u32, 2, 3, 8, 33] {
        let c: ResidencyConfig = config(slots, 2, 2 * SLOT);
        let mut m = ResidencyManager::new(c).unwrap();
        let mut rng = Rng(0x9E37_79B9 ^ u64::from(slots));
        let mut run = |m: &mut ResidencyManager<_>| {
            for _ in 0..500 {
                let k = id(rng.below(4) as u32, rng.below(16) as u32);
                touch(m, k);
            }
        };
        run(&mut m);
        let first = m.stats();
        run(&mut m);
        run(&mut m);
        let s = m.stats();
        assert_eq!(
            s.peak_resident_bytes, first.peak_resident_bytes,
            "slots {slots}"
        );
        assert_eq!(s.peak_in_flight_bytes, first.peak_in_flight_bytes);
        assert!(s.peak_resident_bytes <= m.ledger().expert_pool_bytes);
        assert!(s.peak_in_flight_bytes <= m.ledger().staging_bytes);
        assert_eq!(s.peak_resident_bytes, u64::from(slots.min(64)) * SLOT);
        m.check_invariants().unwrap();
    }
}
