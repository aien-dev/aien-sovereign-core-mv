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

//! `aien-serve-http`: the OpenAI-compatible HTTP door for AIEN's own-kernel model
//! serving (OWN-30B cut 4). Loopback only, one generation at a time unless the engine has more slots (--batch-slots), no CUDA, no
//! Python. The engine behind the door is a trait ([`engine::Engine`]); this crate
//! ships a stub engine for tests and the CPU reference forward as a real-but-slow
//! engine. The Omega GPU engine plugs into the same trait.

pub mod chip_reservation;
pub mod codec;
pub mod cpu_engine;
pub mod decode_timeline;
pub mod engine;
pub mod kv_persist;
pub mod omega_engine;
pub mod openai;
pub mod qwen36_batch;
pub mod qwen38_engine;
pub mod readers;
pub mod sampling;
pub mod server;
pub mod text;
pub mod tools;

pub use codec::{ByteCodec, Codec, HfCodec};
pub use engine::{Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams, StubEngine};
pub use server::{serve, ServerConfig};
