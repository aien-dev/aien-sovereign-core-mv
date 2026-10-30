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

//! In-Process Context Composition and Revision Management
//! Manages immutable ContextRevisions and computes missing prefill spans for the scheduler.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenSegment {
    pub id: u64,
    pub tokens: Arc<Vec<u32>>,
}

impl TokenSegment {
    pub fn new(id: u64, tokens: Vec<u32>) -> Self {
        Self {
            id,
            tokens: Arc::new(tokens),
        }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

#[derive(Debug, Clone)]
pub enum ContextSegment {
    System(TokenSegment),
    User(TokenSegment),
    Assistant(TokenSegment),
    Memory {
        memory_id: u64,
        tokens: TokenSegment,
    },
    ToolResult(TokenSegment),
}

impl ContextSegment {
    pub fn tokens(&self) -> &TokenSegment {
        match self {
            Self::System(t) => t,
            Self::User(t) => t,
            Self::Assistant(t) => t,
            Self::Memory { tokens, .. } => tokens,
            Self::ToolResult(t) => t,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ContextRevision {
    pub id: u64,
    pub parent: Option<u64>,
    pub segments: Arc<Vec<ContextSegment>>,
    pub total_tokens: usize,
}

impl ContextRevision {
    pub fn flatten_tokens(&self) -> Vec<u32> {
        let mut out = Vec::with_capacity(self.total_tokens);
        for seg in self.segments.iter() {
            out.extend_from_slice(&seg.tokens().tokens);
        }
        out
    }

    pub fn slice_tokens(&self, start: usize, len: usize) -> Vec<u32> {
        let all = self.flatten_tokens();
        if start >= all.len() {
            return Vec::new();
        }
        let end = (start + len).min(all.len());
        all[start..end].to_vec()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextPolicy {
    pub max_memory_tokens: usize,
    pub max_items: usize,
    pub min_relevance: f32,
}

impl Default for ContextPolicy {
    fn default() -> Self {
        Self {
            max_memory_tokens: 4096,
            max_items: 8,
            min_relevance: 0.65,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PrefillSpan {
    pub start: usize,
    pub len: usize,
    pub tokens: Vec<u32>,
}

/// Thread-safe ContextComposer caching tokenized segments and revision trees.
pub struct ContextComposer {
    revisions: HashMap<u64, ContextRevision>,
    #[allow(dead_code)]
    segment_cache: HashMap<String, TokenSegment>,
    next_revision_id: u64,
    next_segment_id: u64,
}

impl Default for ContextComposer {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextComposer {
    pub fn new() -> Self {
        Self {
            revisions: HashMap::new(),
            segment_cache: HashMap::new(),
            next_revision_id: 1,
            next_segment_id: 1,
        }
    }

    pub fn create_root_revision(
        &mut self,
        system_prompt_tokens: Vec<u32>,
        user_prompt_tokens: Vec<u32>,
    ) -> u64 {
        let seg_id1 = self.next_segment_id;
        self.next_segment_id += 1;
        let sys_seg = ContextSegment::System(TokenSegment::new(seg_id1, system_prompt_tokens));

        let seg_id2 = self.next_segment_id;
        self.next_segment_id += 1;
        let user_seg = ContextSegment::User(TokenSegment::new(seg_id2, user_prompt_tokens));

        let total_tokens = sys_seg.tokens().len() + user_seg.tokens().len();
        let rev_id = self.next_revision_id;
        self.next_revision_id += 1;

        let revision = ContextRevision {
            id: rev_id,
            parent: None,
            segments: Arc::new(vec![sys_seg, user_seg]),
            total_tokens,
        };

        self.revisions.insert(rev_id, revision);
        rev_id
    }

    pub fn append_memory(
        &mut self,
        parent_rev_id: u64,
        memory_id: u64,
        memory_tokens: Vec<u32>,
    ) -> Result<u64, String> {
        let parent = self
            .revisions
            .get(&parent_rev_id)
            .ok_or_else(|| format!("Parent revision {} not found", parent_rev_id))?;

        let seg_id = self.next_segment_id;
        self.next_segment_id += 1;
        let mem_seg = ContextSegment::Memory {
            memory_id,
            tokens: TokenSegment::new(seg_id, memory_tokens),
        };

        let mut new_segments = (*parent.segments).clone();
        let added_tokens = mem_seg.tokens().len();
        new_segments.push(mem_seg);

        let rev_id = self.next_revision_id;
        self.next_revision_id += 1;

        let revision = ContextRevision {
            id: rev_id,
            parent: Some(parent_rev_id),
            segments: Arc::new(new_segments),
            total_tokens: parent.total_tokens + added_tokens,
        };

        self.revisions.insert(rev_id, revision);
        Ok(rev_id)
    }

    pub fn get_revision(&self, rev_id: u64) -> Option<&ContextRevision> {
        self.revisions.get(&rev_id)
    }

    /// Computes missing prefill tokens for a sequence whose current prefill cursor lags the revision.
    pub fn compute_missing_prefill(
        &self,
        rev_id: u64,
        prefill_cursor: usize,
    ) -> Option<PrefillSpan> {
        let rev = self.get_revision(rev_id)?;
        if prefill_cursor >= rev.total_tokens {
            return None;
        }

        let missing_len = rev.total_tokens - prefill_cursor;
        let tokens = rev.slice_tokens(prefill_cursor, missing_len);

        Some(PrefillSpan {
            start: prefill_cursor,
            len: missing_len,
            tokens,
        })
    }
}
