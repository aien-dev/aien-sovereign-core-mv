//! Omega Expert Ribbon P2 (sc#630): the bounded host-side expert residency
//! manager.
//!
//! This crate is the cache policy and correctness state machine only. It
//! hands out slot numbers and generations; it never touches GPU pointers,
//! NVMe devices or decode. The caller (the read adapter of sc#629, the
//! versioned GPU slots of omega#533, wiring in sc#632) drives the transfers
//! and reports each step back.
//!
//! * [`budget`]: the memory ledger. OS / resource-manager headroom,
//!   non-expert weights, the model KV cache at the configured maximum context,
//!   scratch and staging are reserved first; only what is left becomes fixed
//!   expert slots. Oversubscription is refused, never assumed away.
//! * [`manager`]: the residency state machine. One expert identity maps to
//!   one slot and one generation. States run
//!   `Absent -> Fetching -> Verified -> Copying -> Ready -> InUse -> Evictable`,
//!   and a failure is explicit. A pinned slot is never evicted. Eviction is
//!   least recently used with a deterministic logical clock. Every refusal is
//!   a typed [`Refusal`].
//!
//! Correctness rule (aienix#4 invariant 1): a miss either waits for the exact
//! expert the router selected or fails explicitly. This crate has no notion of
//! a substitute expert, a dropped expert or a lower-precision fallback.

pub mod budget;
pub mod error;
pub mod manager;

pub use budget::{AllocationReport, Category, KvShape, Ledger, MemoryBudget};
pub use error::{BudgetError, FailureCause, Refusal};
pub use manager::{
    Acquired, ExpertIdentity, ExpertPart, ExpertState, FetchTicket, Phase, Pin, PinStatus,
    ResidencyConfig, ResidencyManager, SlotId, Stats,
};
