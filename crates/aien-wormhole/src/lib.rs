//! Wormhole (ledger L080, design docs/WORMHOLE.md): two models with one vocabulary work as draft and block
//! verify. The verifier's greedy output is the output, word for word; the drafter only saves verifier passes.
//! Flash-Next's routed experts (68 GB at NVFP4) live on the NVMe and are served through the Expert Heartbeat
//! ring (aien-heartbeat): fixed versioned slots, readable only after a full read and a BLAKE3 match.
//!
//! Host-only in this crate: the cache, the tensor source the Flash-Next reference reads through, and the
//! round. The chip forward of Flash-Next is the flashnext/port work (ledger L020).

pub mod align;
pub mod cache;
pub mod moe30;
pub mod source;
pub mod spec;
