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

//! The Omegese guardrail: a [`Constraint`](aien_inference_abi::generation::Constraint) for the
//! generation controller (ADR 0004 section 2.3) that keeps a model's reply a prefix of a program
//! `omegese emit` accepts.
//!
//! A separately tested feature (Drake's guardrail decision, ADR 0004): nothing depends on this
//! crate by default, the controller with no constraint is unchanged, and the sparse output head
//! stays the default gather. The legal-token source is any [`ItemOracle`] (the contract is omega
//! `docs/one_layer/OMEGESE.md` section 6.2); [`SubprocessOracle`] runs `omegese next --machine`.
//! [`TokenTable`] holds the model vocabulary as bytes (Qwen's `tokenizer.json`, or a synthetic
//! table in tests). [`OmegeseConstraint`] forces the opening fence and the entry header, masks the
//! body, and maps `eof` to the closing fence plus the stop token; each part is switchable
//! ([`Framing`]).

pub mod constraint;
pub mod oracle;
pub mod signature;
pub mod table;
pub mod word;

pub use constraint::{Framing, OmegeseConstraint, Stats};
pub use oracle::{parse_machine, Answer, Item, ItemOracle, SubprocessOracle, DEFAULT_TIMEOUT};
pub use signature::Signature;
pub use table::{Shape, TokenTable};
