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

//! ALLEN persona profile glue (arch#159): turns the engaged identity plus the
//! profile store into (a) the prompt prefix and report of a compose task and
//! (b) the answers to the `Allen*` control commands.
//!
//! The profile store is written ONLY by `aien-allen-profile`, and only from
//! the operator commands handled here. Nothing in the compose task path
//! (model text, proposals, workspace files) calls a writer.
use crate::control::{
    AllenHistoryReport, AllenProfileReport, AllenRefusalReport, AllenStatusReport, ControlCommand,
    ControlResponse, DeploymentSummary, PersonaReport,
};
use aien_allen::Resolved;
use aien_allen_profile::{Identity, Persona, PersonaContext, ProfileRefusal, Store};
use std::path::Path;

/// Facilities in the plan that v1 does not have. Reported, never given a control.
pub const UNSUPPORTED: [&str; 2] = ["avatar", "voice"];

/// `None` = ALLEN is not engaged: no persona text, no change in behaviour.
pub fn context_for(dir: &Path, id: Option<&Resolved>) -> Option<PersonaContext> {
    let r = id?;
    let store = Store::new(dir, Identity::from(r));
    let ctx = PersonaContext::from_store(&store);
    if let Some(why) = &ctx.reason {
        tracing::warn!("ALLEN persona profile not used (defaults apply): {why}");
    }
    Some(ctx)
}

pub fn report_for(ctx: Option<&PersonaContext>) -> PersonaReport {
    match ctx {
        None => PersonaReport {
            state: "not_engaged".into(),
            display_name: String::new(),
            revision: 0,
            reason: None,
        },
        Some(c) => PersonaReport {
            state: c.state.as_str().into(),
            display_name: c.persona.display_name.clone(),
            revision: c.revision,
            reason: c.reason.clone(),
        },
    }
}

/// The persona block in front of the task prompt; the prompt itself is
/// unchanged when there is no context.
pub fn prefix_prompt(prompt: &str, ctx: Option<&PersonaContext>) -> String {
    match ctx {
        None => prompt.to_string(),
        Some(c) => format!("{}\n{prompt}", c.render()),
    }
}

fn refused(e: &ProfileRefusal) -> ControlResponse {
    let current_revision = match e {
        ProfileRefusal::StaleUpdate { current, .. } => Some(*current),
        _ => None,
    };
    ControlResponse::AllenRefused(Box::new(AllenRefusalReport {
        code: e.code().into(),
        message: e.to_string(),
        current_revision,
    }))
}

fn profile_report(store: &Store) -> Result<ControlResponse, ProfileRefusal> {
    let head = store.head()?;
    Ok(ControlResponse::AllenProfile(Box::new(match head {
        Some(p) => AllenProfileReport {
            revision: p.revision,
            state: "applied".into(),
            profile: Some(p),
            defaults: None,
        },
        None => AllenProfileReport {
            revision: 0,
            state: "default".into(),
            profile: None,
            defaults: Some(Persona::default()),
        },
    })))
}

fn written(p: aien_allen_profile::Profile) -> ControlResponse {
    ControlResponse::AllenProfile(Box::new(AllenProfileReport {
        revision: p.revision,
        state: "applied".into(),
        profile: Some(p),
        defaults: None,
    }))
}

/// Answer one `Allen*` command. `id` is the identity this daemon resolved
/// (`None` = not engaged); `label` is the proposer label the daemon runs.
pub fn handle_allen_command(
    dir: &Path,
    id: Option<&Resolved>,
    label: &str,
    cmd: &ControlCommand,
) -> ControlResponse {
    if let ControlCommand::AllenStatus = cmd {
        return status(dir, id, label);
    }
    let Some(r) = id else {
        return refused(&ProfileRefusal::NotEngaged);
    };
    let store = Store::new(dir, Identity::from(r));
    let out = match cmd {
        ControlCommand::AllenProfileShow => profile_report(&store),
        ControlCommand::AllenProfileSet {
            expected_revision,
            changes,
        } => store.set(*expected_revision, changes).map(written),
        ControlCommand::AllenProfileHistory => store
            .history()
            .map(|entries| ControlResponse::AllenHistory(Box::new(AllenHistoryReport { entries }))),
        ControlCommand::AllenProfileRevert {
            expected_revision,
            to,
        } => store.revert(*expected_revision, *to).map(written),
        ControlCommand::AllenProfileReset { expected_revision } => {
            store.reset(*expected_revision).map(written)
        }
        _ => {
            return ControlResponse::Error("not an ALLEN profile command".into());
        }
    };
    out.unwrap_or_else(|e| refused(&e))
}

fn status(dir: &Path, id: Option<&Resolved>, label: &str) -> ControlResponse {
    let ctx = context_for(dir, id);
    let deployment = id.and_then(|r| {
        aien_allen::deployment::verify(&aien_allen::deployment::deployments_path(dir), &r.agent)
            .ok()
            .and_then(|v| v.last().cloned())
            .map(|e| DeploymentSummary {
                seq: e.seq,
                candidate_id: e.candidate_id,
                placeholder: e.placeholder,
            })
    });
    ControlResponse::AllenStatusReport(Box::new(AllenStatusReport {
        identity: if id.is_some() {
            "engaged"
        } else {
            "not_engaged"
        }
        .into(),
        fingerprint: id.map(|r| Identity::from(r).fingerprint()),
        head_sequence: id.map(|r| r.head_seq),
        chain_verified: id.map(|r| r.chain_verified),
        persona: report_for(ctx.as_ref()),
        model: label.to_string(),
        execution_mode: if label.starts_with("model:") {
            "local_model"
        } else {
            "stub"
        }
        .into(),
        deployment,
        unsupported: UNSUPPORTED.iter().map(|s| s.to_string()).collect(),
    }))
}
