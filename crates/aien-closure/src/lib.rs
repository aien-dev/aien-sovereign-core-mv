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

//! aien-closure: the `verify-closure` bridge (Verified Crumb stage 5).
//!
//! Rebuilds the internal dependency graph of a workspace and checks that every
//! internal edge is declared in a `closure.toml` manifest and backed by a
//! verified aien-proof receipt. Fails closed: any finding is a non-zero exit.

pub mod declare;
pub mod digest;
pub mod error;
pub mod graph;
pub mod manifest;
pub mod verify;

pub use error::{Code, Finding};
pub use verify::{verify, Options, Outcome};
