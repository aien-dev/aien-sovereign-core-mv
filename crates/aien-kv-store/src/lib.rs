// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Persistent KV cache: versioned chunk identity, the 256-byte record header in
//! its 4096-byte region, validation, and the `KvStore` contract with a
//! deterministic fake adapter. Binding design: `docs/adr/0003-persistent-kv-identity-storage-contract-and-ownership.md`.
//!
//! `no_std` + `alloc` for identity, codec, validation, pinned views and the
//! trait. The `std` feature (default) adds the in-memory fake adapter.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod coalesce;
pub mod error;
#[cfg(feature = "std")]
pub mod fake;
pub mod holder;
#[cfg(all(feature = "std", unix))]
pub mod host_fs;
pub mod identity;
pub mod index;
pub mod pin;
pub mod record;
#[cfg(feature = "std")]
pub mod seal;
#[cfg(all(test, feature = "std", unix))]
mod seal_budget_tests;
#[cfg(all(test, feature = "std", unix))]
mod seal_nonce_tests;
pub mod store;
pub mod validate;
#[cfg(feature = "std")]
pub mod worker;

pub use error::KvError;
pub use identity::{
    chunk_chain, chunk_key, BackendFormat, ChunkKey, ComputeConfig, EngineConfig, EngineIdentity,
    HiddenAct, KvDtype, LayoutStrides, NamespaceFields, NamespaceId, RopeScaling, WeightFormat,
};
pub use pin::{HolderGuard, HolderHandle, PinnedDest, PinnedSource};
pub use record::{
    payload_root, segment_count, ChunkHeader, RecordHeader, HEADER_LEN, HEADER_REGION_LEN, MAGIC,
    MAX_SEGMENTS, PAYLOAD_OFFSET, SEGMENT_TABLE_OFFSET,
};
pub use store::{
    CancelOutcome, Capacity, Completion, DrainReport, Incarnation, JobId, JobStage, KvStore,
    LandingHolder, PinToken, PublishOutcome, RecoveryProgress, StoreState, Terminal, UnfinishedJob,
};
pub use validate::validate_record;
