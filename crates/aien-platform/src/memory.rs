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

use crate::PlatformError;
use alloc::string::String;
use alloc::vec::Vec;

pub struct RecallQuery<'a> {
    pub text: &'a str,
    pub space: &'a str,
    pub limit: usize,
}

pub struct MemoryEntity {
    pub id: [u8; 16],
    pub canonical_name: String,
    pub content: String,
    pub score: f32,
}

pub struct RecallResult {
    pub entities: Vec<MemoryEntity>,
}

pub trait MemoryService {
    fn recall(&self, query: RecallQuery<'_>) -> Result<RecallResult, PlatformError>;
}
