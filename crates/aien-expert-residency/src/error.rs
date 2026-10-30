//! Typed errors. Configuration and ledger problems are [`BudgetError`];
//! everything the manager refuses at run time is a [`Refusal`].

use core::fmt;

use crate::budget::Category;
use crate::manager::{Phase, SlotId};

/// The memory plan or an allocation report does not fit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetError {
    /// A size computation overflowed `u64`.
    Overflow { what: &'static str },
    /// The fixed reservations alone exceed the total memory.
    Oversubscribed { required: u64, total: u64 },
    /// `slot_bytes` is zero.
    ZeroSlotBytes,
    /// After the fixed reservations no whole expert slot fits.
    NoExpertSlots {
        available_bytes: u64,
        slot_bytes: u64,
    },
    /// `max_in_flight` is zero, so no fetch could ever start.
    ZeroInFlight,
    /// The in-flight byte limit is larger than the staging reservation.
    InFlightBytesExceedStaging {
        max_in_flight_bytes: u64,
        staging_bytes: u64,
    },
    /// The in-flight byte limit is smaller than one slot, so a full-size
    /// expert could never be fetched.
    InFlightBytesBelowSlot {
        max_in_flight_bytes: u64,
        slot_bytes: u64,
    },
    /// A reported allocation is larger than the ledger reserved for it.
    AllocationExceedsReservation {
        category: Category,
        reserved: u64,
        actual: u64,
    },
}

impl fmt::Display for BudgetError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Overflow { what } => write!(f, "size overflow computing {what}"),
            Self::Oversubscribed { required, total } => write!(
                f,
                "fixed reservations need {required} bytes but only {total} exist"
            ),
            Self::ZeroSlotBytes => write!(f, "slot size is zero"),
            Self::NoExpertSlots {
                available_bytes,
                slot_bytes,
            } => write!(
                f,
                "{available_bytes} bytes left after reservations, less than one {slot_bytes}-byte slot"
            ),
            Self::ZeroInFlight => write!(f, "max_in_flight is zero"),
            Self::InFlightBytesExceedStaging {
                max_in_flight_bytes,
                staging_bytes,
            } => write!(
                f,
                "in-flight limit {max_in_flight_bytes} exceeds staging reservation {staging_bytes}"
            ),
            Self::InFlightBytesBelowSlot {
                max_in_flight_bytes,
                slot_bytes,
            } => write!(
                f,
                "in-flight limit {max_in_flight_bytes} is below one slot of {slot_bytes} bytes"
            ),
            Self::AllocationExceedsReservation {
                category,
                reserved,
                actual,
            } => write!(
                f,
                "{category:?} allocation {actual} exceeds its reservation {reserved}"
            ),
        }
    }
}

impl std::error::Error for BudgetError {}

/// Why a fetch did not reach `Ready`. Reported by the caller.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FailureCause {
    /// The NVMe read failed, was short or timed out.
    Read,
    /// The bytes did not match the content address.
    DigestMismatch,
    /// The staging-to-slot copy failed.
    Copy,
    /// The GPU visibility fence failed or timed out.
    GpuFence,
    /// The fetch was cancelled.
    Cancelled,
}

/// A run-time refusal. Nothing is changed when one is returned, except the
/// refusal counter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Refusal {
    /// Every slot is pinned or in flight; nothing can be evicted.
    PinnedExhaustion {
        slots: u32,
        pinned: u32,
        in_flight: u32,
    },
    /// The number of fetches in flight is at its limit (backpressure).
    InFlightLimit { limit: u32 },
    /// Starting this fetch would exceed the in-flight byte limit.
    InFlightBytesLimit {
        in_flight_bytes: u64,
        requested: u64,
        limit: u64,
    },
    /// The object does not fit one slot.
    ObjectTooLarge { bytes: u64, slot_bytes: u64 },
    /// A zero-byte object was requested.
    ZeroSizedObject,
    /// The key is already admitted with a different size.
    SizeMismatch { admitted: u64, requested: u64 },
    /// The ticket refers to a slot generation that no longer exists.
    StaleTicket { slot: SlotId, generation: u64 },
    /// The pin refers to a slot generation that no longer exists.
    StalePin { slot: SlotId, generation: u64 },
    /// The step is not allowed from the entry's current phase.
    WrongPhase {
        slot: SlotId,
        expected: Phase,
        found: Phase,
    },
    /// The pin count of one entry would overflow `u32`.
    PinOverflow { slot: SlotId },
    /// The fetch behind this pin failed; the expert is not resident.
    FetchFailed {
        slot: SlotId,
        generation: u64,
        cause: FailureCause,
    },
}

impl fmt::Display for Refusal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::PinnedExhaustion {
                slots,
                pinned,
                in_flight,
            } => write!(
                f,
                "all {slots} slots busy ({pinned} pinned, {in_flight} in flight)"
            ),
            Self::InFlightLimit { limit } => write!(f, "{limit} fetches already in flight"),
            Self::InFlightBytesLimit {
                in_flight_bytes,
                requested,
                limit,
            } => write!(
                f,
                "{in_flight_bytes} bytes in flight + {requested} would exceed {limit}"
            ),
            Self::ObjectTooLarge { bytes, slot_bytes } => {
                write!(f, "object of {bytes} bytes exceeds slot size {slot_bytes}")
            }
            Self::ZeroSizedObject => write!(f, "zero-sized object"),
            Self::SizeMismatch {
                admitted,
                requested,
            } => write!(
                f,
                "key admitted with {admitted} bytes, requested with {requested}"
            ),
            Self::StaleTicket { slot, generation } => {
                write!(f, "stale ticket for slot {slot} generation {generation}")
            }
            Self::StalePin { slot, generation } => {
                write!(f, "stale pin for slot {slot} generation {generation}")
            }
            Self::WrongPhase {
                slot,
                expected,
                found,
            } => write!(f, "slot {slot}: expected {expected:?}, found {found:?}"),
            Self::PinOverflow { slot } => write!(f, "slot {slot}: pin count overflow"),
            Self::FetchFailed {
                slot,
                generation,
                cause,
            } => write!(
                f,
                "fetch for slot {slot} generation {generation} failed: {cause:?}"
            ),
        }
    }
}

impl std::error::Error for Refusal {}
