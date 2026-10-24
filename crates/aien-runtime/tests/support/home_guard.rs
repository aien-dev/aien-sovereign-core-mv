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

//! Test-only guard for tests that open a compose home (an `rxc_host` handle).
//!
//! omega's `rxc_host` allows `RXC_HOST_MAX_HANDLES` (4) open handles per
//! process (omega `src/runtime/rxc_host_abi.h`); a fifth open is refused with
//! `E_FULL (-5)`. Rust runs the tests of one test binary on parallel threads in
//! one process, so five tests that each hold a home at the same moment fail
//! with a refusal that has nothing to do with what they check.
//!
//! Every test that opens a home, directly or through a `ComposeBridge`, takes
//! the slot first and keeps it until it ends. One slot for the whole process:
//! those tests run one after another, and a single test may hold up to
//! `RXC_HOST_MAX_HANDLES` homes. More than 4 concurrent homes in one process is
//! not qualified here; that limit belongs to omega.
//!
//! Include with `#[path = "support/home_guard.rs"] mod home_guard;`. Each test
//! binary gets its own copy, so the slot is per process, the same scope as the
//! handle table.
#![allow(dead_code)]

use std::sync::{Mutex, MutexGuard};

static SLOT: Mutex<()> = Mutex::new(());

/// Holds the process-wide compose-home slot until dropped. A wrapper, not the
/// bare `MutexGuard`, so an async test may keep it across an `.await`
/// (clippy::await_holding_lock only looks at the bare guard).
pub struct HomeSlot(MutexGuard<'static, ()>);

/// Wait for the slot. A test that panicked while holding it does not wedge the
/// others: the poison flag is cleared, the slot guards no data.
pub fn home_slot() -> HomeSlot {
    HomeSlot(SLOT.lock().unwrap_or_else(|e| e.into_inner()))
}
