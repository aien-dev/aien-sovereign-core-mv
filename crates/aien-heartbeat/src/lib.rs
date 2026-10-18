//! Expert Heartbeat, milestone H1 (aienix#4).
//!
//! A fixed set of aligned memory slots is refilled from the NVMe in layer
//! order at a steady rhythm. A slot's `(layer, expert, version)` becomes
//! visible to readers only after the whole expert has been read and its
//! BLAKE3 digest matches the manifest. A slot with a live lease is never
//! overwritten. Any I/O error or digest mismatch fails the whole ring closed.
//!
//! Out of scope for H1: GPU visibility fence (H2 work; unified memory on the
//! GB10 still needs an explicit fence before kernels may read a slot), model
//! math, station queues.

pub mod aligned;
pub mod error;
pub mod layout;
pub mod manifest;
pub mod pager;
pub mod reader;
pub mod residency;
pub mod ring;
pub mod stream;

pub use error::HeartbeatError;
pub use layout::{ExpertLayout, ExpertParts, ExpertShape, Payload};
pub use manifest::{ExpertEntry, Manifest};
pub use pager::{ExpertPager, PagerError, PagerStats, Pinned, SlotBackend};
pub use reader::{
    FakeShards, Fault, FileShards, ReadAdapter, ReadConfig, ReadError, ShardDevice, VerifiedExpert,
};
pub use residency::{Acquire, ExpertId, MemoryBudget, Residency, ResidencyStats, SlotVersion};
pub use ring::{Lease, Ring, SlotTag};
pub use stream::{RotationStats, Schedule, StreamConfig, Streamer};

/// O_DIRECT alignment used for buffers, offsets and lengths.
pub const IO_ALIGN: usize = 4096;
