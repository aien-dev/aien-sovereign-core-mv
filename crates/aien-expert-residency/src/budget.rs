//! The memory ledger: fixed reservations first, expert slots from what is
//! left.
//!
//! The plan is computed once and does not change while the manager runs, so
//! the committed total is also the high-water mark of the plan. Live usage
//! (resident and in-flight bytes) is tracked by the manager and is always at
//! or below the corresponding reservation.

use crate::error::BudgetError;

/// Shape of the model attention KV cache at the configured maximum context.
/// This is the persisted-attention KV, a different object from expert
/// weights; it is budgeted here only so expert slots cannot eat into it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KvShape {
    pub layers: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub bytes_per_element: u32,
    pub max_context_tokens: u32,
    pub max_sequences: u32,
}

impl KvShape {
    /// Bytes for K and V of every layer, KV head, position and sequence.
    pub fn bytes(&self) -> Result<u64, BudgetError> {
        [
            2,
            u64::from(self.layers),
            u64::from(self.kv_heads),
            u64::from(self.head_dim),
            u64::from(self.bytes_per_element),
            u64::from(self.max_context_tokens),
            u64::from(self.max_sequences),
        ]
        .iter()
        .try_fold(1u64, |acc, &f| acc.checked_mul(f))
        .ok_or(BudgetError::Overflow { what: "kv cache" })
    }
}

/// What the machine has and what must be set aside before any expert slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MemoryBudget {
    /// Total unified memory the plan may describe.
    pub total_bytes: u64,
    /// OS, driver and resource-manager safety headroom. Never allocated.
    pub safety_headroom_bytes: u64,
    /// Resident non-expert weights (embeddings, attention, norms, router,
    /// any experts kept permanently resident outside the cache).
    pub non_expert_weight_bytes: u64,
    /// Attention KV cache shape.
    pub kv: KvShape,
    /// Activations and kernel workspace.
    pub scratch_bytes: u64,
    /// NVMe read staging buffers. Bytes in flight are charged here.
    pub staging_bytes: u64,
}

/// A ledger category.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Category {
    SafetyHeadroom,
    NonExpertWeights,
    KvCache,
    Scratch,
    Staging,
    ExpertPool,
}

/// The computed plan. Every byte of `total_bytes` is in exactly one field:
/// the five reservations, the expert pool or `unassigned_bytes`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ledger {
    pub total_bytes: u64,
    pub safety_headroom_bytes: u64,
    pub non_expert_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub staging_bytes: u64,
    pub slot_bytes: u64,
    pub slot_count: u32,
    /// `slot_count * slot_bytes`.
    pub expert_pool_bytes: u64,
    /// Left over: less than one slot, or above `max_slots`. Not allocated.
    pub unassigned_bytes: u64,
}

/// What the runtime actually allocated, per category (from the allocator or
/// the driver's report).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct AllocationReport {
    pub non_expert_weight_bytes: u64,
    pub kv_cache_bytes: u64,
    pub scratch_bytes: u64,
    pub staging_bytes: u64,
    pub expert_pool_bytes: u64,
}

impl Ledger {
    /// Reserve the fixed categories, then cut the remainder into whole slots
    /// of `slot_bytes`, at most `max_slots` of them.
    pub fn plan(
        budget: &MemoryBudget,
        slot_bytes: u64,
        max_slots: Option<u32>,
    ) -> Result<Self, BudgetError> {
        if slot_bytes == 0 {
            return Err(BudgetError::ZeroSlotBytes);
        }
        let kv_cache_bytes = budget.kv.bytes()?;
        let required = [
            budget.safety_headroom_bytes,
            budget.non_expert_weight_bytes,
            kv_cache_bytes,
            budget.scratch_bytes,
            budget.staging_bytes,
        ]
        .iter()
        .try_fold(0u64, |acc, &b| acc.checked_add(b))
        .ok_or(BudgetError::Overflow {
            what: "fixed reservations",
        })?;
        if required > budget.total_bytes {
            return Err(BudgetError::Oversubscribed {
                required,
                total: budget.total_bytes,
            });
        }
        let available = budget.total_bytes - required;
        let fit = available / slot_bytes;
        let mut slot_count = u32::try_from(fit).unwrap_or(u32::MAX);
        if let Some(cap) = max_slots {
            slot_count = slot_count.min(cap);
        }
        if slot_count == 0 {
            return Err(BudgetError::NoExpertSlots {
                available_bytes: available,
                slot_bytes,
            });
        }
        // Cannot overflow: slot_count * slot_bytes <= available.
        let expert_pool_bytes = u64::from(slot_count) * slot_bytes;
        Ok(Self {
            total_bytes: budget.total_bytes,
            safety_headroom_bytes: budget.safety_headroom_bytes,
            non_expert_weight_bytes: budget.non_expert_weight_bytes,
            kv_cache_bytes,
            scratch_bytes: budget.scratch_bytes,
            staging_bytes: budget.staging_bytes,
            slot_bytes,
            slot_count,
            expert_pool_bytes,
            unassigned_bytes: available - expert_pool_bytes,
        })
    }

    /// Everything reserved, including the expert pool. Excludes only
    /// `unassigned_bytes`. Fixed for the life of the plan.
    pub fn committed_bytes(&self) -> u64 {
        self.total_bytes - self.unassigned_bytes
    }

    /// The reservation for one category.
    pub fn reserved(&self, category: Category) -> u64 {
        match category {
            Category::SafetyHeadroom => self.safety_headroom_bytes,
            Category::NonExpertWeights => self.non_expert_weight_bytes,
            Category::KvCache => self.kv_cache_bytes,
            Category::Scratch => self.scratch_bytes,
            Category::Staging => self.staging_bytes,
            Category::ExpertPool => self.expert_pool_bytes,
        }
    }

    /// Check a real allocation report against the plan. Every category must
    /// be at or below its reservation; the first one over is returned.
    pub fn check_allocations(&self, report: &AllocationReport) -> Result<(), BudgetError> {
        let rows = [
            (Category::NonExpertWeights, report.non_expert_weight_bytes),
            (Category::KvCache, report.kv_cache_bytes),
            (Category::Scratch, report.scratch_bytes),
            (Category::Staging, report.staging_bytes),
            (Category::ExpertPool, report.expert_pool_bytes),
        ];
        for (category, actual) in rows {
            let reserved = self.reserved(category);
            if actual > reserved {
                return Err(BudgetError::AllocationExceedsReservation {
                    category,
                    reserved,
                    actual,
                });
            }
        }
        Ok(())
    }
}
