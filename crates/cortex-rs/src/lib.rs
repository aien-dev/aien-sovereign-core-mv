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

//! Sovereign Native Memory Engine in Rust (Cortex)
//! Exposes in-process database and memory retrieval interfaces for the AIEN runtime.

pub mod auth;
pub mod bench;
pub mod compiler;
pub mod db;
pub mod embeddings;
pub mod evidence;
pub mod handlers;
pub mod models;
pub mod promotion;
pub mod security;
pub mod workers;

pub use db::Database;
pub use models::*;
use std::sync::Arc;

/// In-process Cortex memory runtime avoiding localhost HTTP serialization overhead.
pub struct CortexRuntime {
    db: Arc<Database>,
}

impl CortexRuntime {
    pub fn new(db: Arc<Database>) -> Self {
        Self { db }
    }

    pub fn in_memory() -> Result<Self, String> {
        let db = Database::open_in_memory().map_err(|e| e.to_string())?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn open<P: AsRef<std::path::Path>>(path: P) -> Result<Self, String> {
        let db = Database::open(path).map_err(|e| e.to_string())?;
        Ok(Self { db: Arc::new(db) })
    }

    pub fn database(&self) -> &Arc<Database> {
        &self.db
    }
}
