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

//! aien-proof: the shared test board for the hive.
//!
//! Every test job is fingerprinted from its command, toolchain, and the exact
//! contents of its inputs. A passing run leaves a stamp; any later request with
//! the same fingerprint, from any agent or worktree, replays the stamp instead of
//! running. An identical job already in flight is joined, never duplicated.
//! Real runs pass a CPU slot gate, a free-memory gate, and (for GPU jobs) an
//! exclusive GPU key. Every real run, pass or fail, is sealed into a BLAKE3
//! hash-chained ledger using the Crumb ledger event format (RFC-0001 section 8),
//! with the run's full output as the payload.

pub mod board;
pub mod chain;
pub mod evidence;
pub mod fingerprint;
pub mod formal;
pub mod gate;
pub mod import;
pub mod lease;
pub mod ledger;
pub mod ledger_bind;
pub mod lock;
pub mod store_assert;
pub mod workspace;

#[cfg(test)]
pub(crate) mod testutil {
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static N: AtomicUsize = AtomicUsize::new(0);

    pub fn temp_dir(tag: &str) -> PathBuf {
        loop {
            let n = N.fetch_add(1, Ordering::SeqCst);
            let dir =
                std::env::temp_dir().join(format!("aien-proof-{tag}-{}-{n}", std::process::id()));
            match std::fs::create_dir(&dir) {
                Ok(()) => return dir,
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("cannot create test directory: {error}"),
            }
        }
    }
}
