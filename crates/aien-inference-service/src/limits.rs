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

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InferenceLimits {
    pub max_tokens: usize,
    pub temperature: f32,
    pub top_p: Option<f32>,
    pub stop_sequences: Vec<String>,
}

impl Default for InferenceLimits {
    fn default() -> Self {
        Self {
            max_tokens: 1024,
            temperature: 0.7,
            top_p: None,
            stop_sequences: Vec::new(),
        }
    }
}
