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

//! The compose bridge (NEXT-PHASE-1 cut 1b): omega COMPOSITION-2 through
//! aien-omega-compose. A goal becomes one proposed file change, checked,
//! recorded in the Cortex journal and committed (or refused) by AEGIS.
//! One composition home per process, opened on first use and kept open: the
//! Cortex journal takes an exclusive writer lock.
//!
//! Lifted out of `spine.rs`, which keeps the runtime daemon's step loop.

mod bridge;

// The bridge: a goal in, one proposed, checked and recorded file change (or
// a refusal) out, with the operator commands on the same compose home.
pub use bridge::{ComposeBridge, ComposeProposer, Generation};

// Reading a model reply as a file proposal (the approval path re-checks one).
pub use bridge::{check_file_proposal, parse_file_proposal, FileProposal};

// Settings read from the environment: at daemon start (aien-cli, server.rs)
// and, for the budgets, again when a compose home opens. Public so the env
// tests check every accepted and refused value through these functions.
pub use bridge::{
    authorize_desk_at_start_from_env, authorize_desk_setting, authorize_requires_desk_from_env,
    compose_budgets_from_env, compose_dir_from_env, compose_doc_max_tokens_from_env,
    ComposeBudgets, AUTHORIZE_DESK_ENV,
};

// The digest a generation record carries for a token id list.
pub use bridge::token_ids_sha256;

// For the daemon (server.rs) and the approval and effect paths only.
pub(crate) use bridge::{
    authorize_mac_line, check_desk_key_at_start, classify_target, compose_assistant_prefix,
    record_view, wall_clock_reason, ComposeHome, TargetClass, COMPOSE_ASSISTANT_PREFIX,
};

/// The exact bytes the bridge hands the model and the proposal it makes of a
/// reply, for offline re-derivation tools that reproduce a recorded run
/// without a compose home (examples/np1_reference.rs, np1_edit_merge.rs,
/// qwen3_8b_specverify.rs) and for the prompt-size measurements. They
/// reproduce; they do not run a task: callers and tests run tasks through
/// [`ComposeBridge`](crate::compose::ComposeBridge).
pub mod rederive {
    pub use super::bridge::{
        classify_destination, compose_assistant_prefix, corrective_retry_prompt, edit_proposal,
        new_document_block, proposal_prompt, retry_prompt, task_decision, task_prompt,
        ProposalKind, TargetClass, TaskPrompt, COMPOSE_ASSISTANT_PREFIX,
    };
}
