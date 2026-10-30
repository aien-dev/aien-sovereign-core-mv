//! Omega Expert Ribbon P1 (sc#629): a content-addressed store for FP8 expert
//! tensors and a bounded, verified read adapter.
//!
//! * [`format`]: the canonical artifact (one projection of one expert: a
//!   4096-byte header, the FP8 weight bytes and the block-128 scale bytes,
//!   each 4096-aligned) and the versioned index keyed by
//!   (model digest, layer, expert, projection, quantization) with the data
//!   digest. The content address is SHA-256 of the canonical artifact bytes.
//! * [`store`]: the on-disk store. Objects and the index are written to a
//!   private temporary file, synced, renamed into place and the directory
//!   synced, so a crash leaves either the old or the new state.
//! * [`indexer`]: a deterministic, read-only checkpoint-to-artifact indexer
//!   for Qwen3-MoE FP8 safetensors checkpoints. The checkpoint is never
//!   opened for writing; the artifact keeps the original bytes unchanged.
//! * [`adapter`]: the bounded asynchronous read adapter (queue depth, in-flight
//!   bytes, object size, deadline, cancellation) into caller-provided aligned
//!   staging buffers. Every object is hashed and compared in constant time
//!   before it is handed back as ready; on any failure the staging buffer is
//!   scrubbed before the caller gets it back.
//! * [`source`]: the byte sources behind the adapter (buffered and direct file
//!   reads behind one interface) and [`fake`], a deterministic in-memory source
//!   with injectable faults for tests.
//!
//! Not in this crate: GPU slots, residency policy and decode wiring (sc#630 to
//! sc#632, omega#533).

pub mod adapter;
pub mod error;
pub mod fake;
pub mod format;
pub mod indexer;
pub mod source;
pub mod staging;
pub mod store;

pub use adapter::{
    CancelOutcome, Outcome, ReadAdapter, ReadCompletion, ReadLimits, ReadyExpert, Refused, Ticket,
};
pub use error::{ReadError, StoreError};
pub use format::{
    ArtifactHeader, ExpertKey, IndexEntry, Projection, QuantFormat, ScaleDtype, ALIGN, FP8_BLOCK,
    HEADER_REGION,
};
pub use source::{BlockReader, BlockSource, FileSource, IoMode};
pub use staging::StagingBuffer;
pub use store::{ExpertStore, StoreLimits};
