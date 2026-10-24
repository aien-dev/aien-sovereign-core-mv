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

//! Proposer hook: an approved proposal enters the production compose path.
//!
//! The approval desk upstream (INTERPLANE `ComposeLedgerAuthority`,
//! interplane#77) hands over one file change it approved. `ProposerHook`
//! checks it, then drives it through the same compose run `RunComposeTask`
//! uses: the model Skill returns the approved text instead of calling the
//! model, AEGIS checks it with the bridge's verify callback, the winner
//! commits through the World on a J-Space branch, Cortex records it.
//!
//! The hook grants nothing. It never writes the workspace and never mints an
//! authorization: the effect still needs an `authorization` note, a
//! `ComposeEffectIntent` and an ack (crate::effects). Anything it cannot
//! verify, any duplicate and any run that does not commit is refused with a
//! named error (`PROPOSAL_REFUSED <Name>: ...`), never dropped.
//!
//! Hash semantics (both are reported, never conflated):
//! - `approved_proposal_sha256`: sha256 of the compact JSON object
//!   `{"content":..,"path":..}` with sorted keys, as interplane#77 computes it.
//! - `compose_proposal_sha256`: sha256 of the committed proposal text
//!   (`filename: <path>\n<content>`), the value `aien compose authorize` puts
//!   in the authorization grant and `ComposeEffectIntent` must repeat.
//!
//! Authentication (#249 A, crate::approved_auth): `approval_mac` must be the
//! approval desk key's HMAC over every approved field, else `Unauthenticated`.
//! Replay (#249 B, crate::approved_replay): the approval is claimed durably
//! in the Cortex journal before the run and settled after it (committed,
//! failed, uncertain); a claimed request id, approval id or approval key is
//! refused for ever, across restarts.
//!
//! Design note: docs/COMPOSE_PROPOSER_HOOK.md.
use crate::approved_auth::{
    approval_key, check_confinement, desk_key_path, ApprovalIdentity, DeskKey,
};
use crate::approved_replay::{self, ClaimKeys, CommitEvidence};
use crate::compose::{check_file_proposal, ComposeBridge};
use crate::control::ComposeTaskReport;
use crate::effects::{write_approved_grant, ApprovedLink};
use aien_omega_compose::hex;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;

/// Prefix of every hook refusal.
pub const REFUSED: &str = "PROPOSAL_REFUSED";

/// One file change an upstream approval desk approved (the interplane#77
/// ledger fields plus the correlation ids).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovedProposal {
    /// INTERPLANE request id and trace id: carried into the report.
    pub request_id: String,
    pub trace_id: String,
    /// The approval the desk consumed (single use).
    pub approval_id: String,
    pub approver: String,
    /// Workspace-relative path and the complete new content.
    pub path: String,
    pub content: String,
    /// `approved_proposal_sha256(path, content)` as the desk computed it.
    pub approved_proposal_sha256: String,
    /// sha256 of `content`.
    pub content_sha256: String,
    /// HMAC-SHA256 (hex) of the approval binding under the approval desk key
    /// (crate::approved_auth). Never caller text: only the desk key holder
    /// can make it.
    pub approval_mac: String,
    /// The goal text whose measurable requirements the approved bytes must
    /// meet (crate::requirements), bound by `requirements_mac`. `Some("")` is
    /// an explicit, signed "no requirements"; `None` (field missing) is
    /// refused: an absent binding is never read as an empty requirement set.
    #[serde(default)]
    pub requirements: Option<String>,
    /// HMAC-SHA256 (hex) under the desk key over the approval binding and
    /// `requirements` (crate::approved_auth::DeskKey::sign_requirements).
    #[serde(default)]
    pub requirements_mac: String,
    /// Only when a requirement depends on the file the bytes replace (an added-line
    /// count): the sha256 (hex) of that file as the desk saw it, or "absent". Bound by
    /// `requirements_mac` (tag v2). The daemon refuses unless the file still has this
    /// sha256 when the grant is written (same compose-home lock).
    #[serde(default)]
    pub requirements_base: Option<String>,
}

/// What a committed approved proposal hands to the effect ledger.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovedComposeReport {
    /// "COMMITTED" (this submission ran and the World committed it) or
    /// "ALREADY_COMMITTED" (a retry of the same authenticated approval whose
    /// first response was lost: nothing ran again, `task` is None and the
    /// evidence is the original run's, read from the replay ledger).
    pub state: String,
    pub request_id: String,
    pub trace_id: String,
    pub approval_id: String,
    pub approver: String,
    pub path: String,
    pub content_sha256: String,
    pub approved_proposal_sha256: String,
    /// sha256 of the committed proposal text: the authorization grant and the
    /// effect intent must name this value.
    pub compose_proposal_sha256: String,
    /// Links the authorization grant carries, as `aien compose authorize`
    /// does: [cx_promotion, cx_evidence].
    pub grant_links: Vec<u64>,
    /// The desk key that authenticated the approval (its id, never the key).
    pub desk_key_id: String,
    /// The durable replay key (crate::approved_auth::approval_key).
    pub approval_key: String,
    /// The replay claim (its `accepted` record id in the Cortex journal).
    pub replay_claim: u64,
    /// The compose run (J-Space branch, AEGIS verdict, World commit records);
    /// None for ALREADY_COMMITTED.
    pub task: Option<ComposeTaskReport>,
    /// The authorization grant the daemon wrote for this approval (reserved
    /// `approved_grant` record, keyed on `compose_proposal_sha256`, linked to
    /// promotion, evidence and the replay claim). The effect intent names it;
    /// a caller never writes its own. None for ALREADY_COMMITTED.
    #[serde(default)]
    pub approved_grant: Option<u64>,
    /// The grant's absolute target (`<canonical workspace>/<path>`).
    #[serde(default)]
    pub target: Option<String>,
}

/// A refusal of the approved-proposal command, with the correlation ids.
/// `refused_by` is "PROPOSAL_REFUSED" (checks before or around the run) or
/// "REPLAY_REFUSED" (crate::approved_replay); `name` is the refusal name.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ApprovedRefusal {
    pub refused_by: String,
    pub name: String,
    pub detail: String,
    pub request_id: String,
    pub trace_id: String,
    pub approval_id: String,
    /// The replay claim the refusal concerns, when there is one.
    pub replay_claim: Option<u64>,
}

impl std::fmt::Display for ApprovedRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} {}: {}", self.refused_by, self.name, self.detail)
    }
}

impl ApprovedRefusal {
    /// From a "<PREFIX> <Name>: detail" text.
    fn from_text(p: &ApprovedProposal, text: &str, claim: Option<u64>) -> Box<Self> {
        let (head, detail) = text.split_once(": ").unwrap_or((text, ""));
        let (by, name) = head.split_once(' ').unwrap_or((REFUSED, head));
        Box::new(Self {
            refused_by: by.to_string(),
            name: name.to_string(),
            detail: detail.to_string(),
            request_id: p.request_id.clone(),
            trace_id: p.trace_id.clone(),
            approval_id: p.approval_id.clone(),
            replay_claim: claim,
        })
    }
}

fn sha256_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

/// Start of the machine goal of an approved run. It carries the desk-supplied path, not
/// a requirement statement: the run does not read requirements out of it (the
/// goal bound into the approval was checked on the same bytes before the claim).
pub(crate) const APPROVED_GOAL_PREFIX: &str = "apply approved proposal for ";

fn refuse(name: &str, why: impl std::fmt::Display) -> String {
    format!("{REFUSED} {name}: {why}")
}

/// sha256 of the compact JSON `{"content":..,"path":..}`, keys sorted
/// (inserted in sorted order, so serde_json's map ordering does not matter).
pub fn approved_proposal_sha256(path: &str, content: &str) -> String {
    let mut m = Map::new();
    m.insert("content".into(), Value::String(content.to_string()));
    m.insert("path".into(), Value::String(path.to_string()));
    sha256_hex(Value::Object(m).to_string().as_bytes())
}

/// The proposal text in the compose template. Refused unless the template
/// parser reads back exactly this path and content (a content the parser
/// would normalise is refused, not silently changed).
pub fn proposal_text(path: &str, content: &str) -> Result<String, String> {
    let text = format!("filename: {path}\n{content}");
    let p = check_file_proposal(&text).map_err(|e| refuse("Unverified", e))?;
    if p.path != path || p.content != content {
        return Err(refuse(
            "Unverified",
            "content does not read back unchanged through the compose template \
             (it must end with one newline, not start with a blank line or a code fence)",
        ));
    }
    Ok(text)
}

/// Every check that needs no compose home. Ok = the proposal text.
pub fn verify(p: &ApprovedProposal) -> Result<String, String> {
    for (k, v) in [
        ("request_id", &p.request_id),
        ("trace_id", &p.trace_id),
        ("approval_id", &p.approval_id),
        ("approver", &p.approver),
    ] {
        if v.trim().is_empty() {
            return Err(refuse("Unverified", format!("empty {k}")));
        }
    }
    if sha256_hex(p.content.as_bytes()) != p.content_sha256 {
        return Err(refuse(
            "Unverified",
            "content_sha256 does not match content",
        ));
    }
    if approved_proposal_sha256(&p.path, &p.content) != p.approved_proposal_sha256 {
        return Err(refuse(
            "Unverified",
            "approved_proposal_sha256 does not match {content, path}",
        ));
    }
    if p.requirements.is_none() {
        return Err(refuse(
            "RequirementsUnbound",
            "the approval carries no requirement binding (requirements); an absent binding is refused, sign an empty goal for none",
        ));
    }
    proposal_text(&p.path, &p.content)
}

/// The requirements the desk bound to this approval, checked on the exact
/// approved content (the bytes that would be saved). Uncertain wording refuses.
fn check_bound_requirements(p: &ApprovedProposal, workspace: &Path) -> Result<(), String> {
    let goal = p.requirements.as_deref().ok_or_else(|| {
        refuse(
            "RequirementsUnbound",
            "the approval carries no requirement binding",
        )
    })?;
    let mut ex = crate::requirements::analyze(goal);
    if let Some(why) = ex.refusal() {
        return Err(refuse("RequirementsUncertain", why));
    }
    if ex.needs_prior() {
        // An added-line count is judged against the file these bytes replace.
        let class = if p.path.contains(char::is_whitespace) {
            crate::compose::TargetClass::Refused(String::new())
        } else {
            crate::compose::classify_target(&p.path, workspace)
        };
        let (prior, exists) =
            match class {
                crate::compose::TargetClass::Edit(_, c) => (c, true),
                crate::compose::TargetClass::New => (String::new(), false),
                _ => return Err(refuse(
                    "RequirementsUnmet",
                    "the existing file cannot be read, so the added-line count cannot be measured",
                )),
            };
        let base = p.requirements_base.as_deref().ok_or_else(|| {
            refuse(
                "RequirementsUnbound",
                "a requirement depends on the file these bytes replace, so the approval must bind requirements_base (its sha256, or \"absent\")",
            )
        })?;
        let now = if exists {
            sha256_hex(prior.as_bytes())
        } else {
            "absent".to_string()
        };
        if now != base {
            return Err(refuse(
                "BaseChanged",
                format!("the file changed since the desk bound it (bound {base}, now {now})"),
            ));
        }
        ex = ex.resolved(&prior);
    }
    match crate::requirements::refusal_reason(&ex.requirements, &p.content) {
        Some(why) => Err(refuse("RequirementsUnmet", why)),
        None => Ok(()),
    }
}

/// The hook over one compose bridge (the daemon's, or a test's). Replay
/// state lives only in the bridge's Cortex journal (crate::approved_replay):
/// two hooks over one compose home, or a restarted daemon, see the same claims.
pub struct ProposerHook {
    bridge: Arc<ComposeBridge>,
}

impl ProposerHook {
    pub fn new(bridge: Arc<ComposeBridge>) -> Self {
        Self { bridge }
    }

    /// Blocking (runs the composition). Order: verify (hashes, template) ->
    /// start-up reconcile gate -> desk key (file rules) -> confinement ->
    /// authenticate (MAC) -> durable claim -> in_flight -> one compose run ->
    /// committed | failed | uncertain. Nothing runs unless every step before
    /// it passed; nothing before the claim writes a record.
    pub fn submit(
        &self,
        p: &ApprovedProposal,
        workspace: &str,
    ) -> Result<ApprovedComposeReport, Box<ApprovedRefusal>> {
        let refusal = |t: String, c: Option<u64>| ApprovedRefusal::from_text(p, &t, c);
        let text = verify(p).map_err(|e| refusal(e, None))?;
        if let Some(why) = self.bridge.reconcile_failed() {
            return Err(refusal(
                refuse(
                    "ReconcileFailed",
                    format!("the start-up reconcile did not complete ({why}); run aien compose reconcile"),
                ),
                None,
            ));
        }
        let identity = (|| {
            let desk = DeskKey::load(&desk_key_path(self.bridge.dir()))?;
            check_confinement(self.bridge.dir(), Path::new(workspace))?;
            desk.authenticate(p, Path::new(workspace))
        })()
        .map_err(|e| refusal(e, None))?;
        // The goal requirements bound by the desk, checked on the exact approved
        // bytes BEFORE any claim, run or commit.
        check_bound_requirements(p, Path::new(workspace)).map_err(|e| refusal(e, None))?;
        let keys = ClaimKeys {
            approval_key: approval_key(&identity),
            request_id: p.request_id.clone(),
            approval_id: p.approval_id.clone(),
            trace_id: p.trace_id.clone(),
        };
        let claim = match approved_replay::claim(&self.bridge, &keys) {
            Ok(c) => c,
            Err(r) => return self.replayed(p, &identity, &keys, r),
        };
        crash_point("approved_after_claim");
        let rr = |r: approved_replay::Refusal| refusal(r.to_string(), Some(claim.id));
        approved_replay::mark_in_flight(&self.bridge, &claim).map_err(|e| {
            // Nothing ran; end the claim as not_executed if the journal lets us.
            let _ = approved_replay::not_executed(&self.bridge, &claim, &e.to_string());
            rr(e)
        })?;
        // Exactly one compose run per claim.
        let goal = format!(
            "{APPROVED_GOAL_PREFIX}{} (request {})",
            p.path, p.request_id
        );
        let r = match self
            .bridge
            .run_approved_task(&goal, &identity.workspace, &text)
        {
            Ok(r) => r,
            Err(e) => {
                // The run may have reached the World: never guess.
                let why = refuse("ComposeError", &e);
                approved_replay::uncertain(&self.bridge, &claim, &why).map_err(rr)?;
                return Err(refusal(why, Some(claim.id)));
            }
        };
        if !r.committed {
            let why = refuse(
                "NotCommitted",
                format!(
                    "outcome {} (AEGIS pass mask {:#x}); {}",
                    r.outcome,
                    r.aegis_pass_mask,
                    r.proposer_error.as_deref().unwrap_or("no proposer error")
                ),
            );
            approved_replay::fail(&self.bridge, &claim, &why).map_err(rr)?;
            return Err(refusal(why, Some(claim.id)));
        }
        let compose_sha = sha256_hex(text.as_bytes());
        // The prior file the bound requirements were checked against (None when none depends on it).
        let base = crate::requirements::analyze(p.requirements.as_deref().unwrap_or(""))
            .needs_prior()
            .then(|| p.requirements_base.clone())
            .flatten();
        if r.proposal.as_deref() != Some(text.as_str())
            || r.proposal_sha256.as_deref() != Some(compose_sha.as_str())
            || r.proposal_path.as_deref() != Some(p.path.as_str())
            || r.proposal_content_sha256.as_deref() != Some(p.content_sha256.as_str())
        {
            let why = refuse(
                "Mismatch",
                format!(
                    "the World committed task {} (promotion #{}) but not the approved proposal",
                    r.task, r.cx_promotion
                ),
            );
            approved_replay::uncertain(&self.bridge, &claim, &why).map_err(rr)?;
            return Err(refusal(why, Some(claim.id)));
        }
        let evidence = CommitEvidence {
            compose_proposal_sha256: compose_sha.clone(),
            cx_promotion: r.cx_promotion,
            cx_evidence: r.cx_evidence,
            task: r.task,
        };
        crash_point("approved_after_compose");
        // If this append fails the claim stays in_flight: refused forever,
        // UNCERTAIN after a restart (fail closed).
        approved_replay::commit(&self.bridge, &claim, &evidence).map_err(rr)?;
        crash_point("approved_after_commit");
        // The one grant for this approval, written by the daemon (never by the
        // caller). If it cannot be written the approval stays spent and no
        // effect is possible: a new approval is needed (fail closed).
        // The grant's target comes from the bound (MAC-covered) workspace only.
        let link = ApprovedLink {
            approval_key: keys.approval_key.clone(),
            replay_claim: claim.id,
            cx_promotion: evidence.cx_promotion,
            cx_evidence: evidence.cx_evidence,
        };
        let ids = serde_json::json!({"request_id": p.request_id, "trace_id": p.trace_id,
            "approval_id": p.approval_id, "desk_key_id": identity.desk_key_id,
            "approved_proposal_sha256": p.approved_proposal_sha256});
        #[cfg(test)]
        if let Some(f) = BEFORE_GRANT.lock().unwrap().as_ref() {
            f();
        }
        let (grant, target) = write_approved_grant(
            &self.bridge,
            &identity.workspace,
            &p.path,
            &p.content_sha256,
            &p.approver,
            &ids,
            &link,
            &compose_sha,
            base.as_deref(),
        )
        .map_err(|e| {
            refusal(
                refuse(
                    "GrantNotWritten",
                    format!("the approval is committed and spent; no effect is possible: {e}"),
                ),
                Some(claim.id),
            )
        })?;
        let mut out = self.report(
            p,
            &identity,
            &keys,
            claim.id,
            "COMMITTED",
            &evidence,
            Some(r),
        );
        out.approved_grant = Some(grant);
        out.target = Some(target);
        Ok(out)
    }

    /// A claim refusal. A retry of the very same authenticated approval (all
    /// keys equal) whose claim committed gets the original result back as
    /// ALREADY_COMMITTED; everything else is refused with the claim's state.
    fn replayed(
        &self,
        p: &ApprovedProposal,
        identity: &ApprovalIdentity,
        keys: &ClaimKeys,
        r: approved_replay::Refusal,
    ) -> Result<ApprovedComposeReport, Box<ApprovedRefusal>> {
        if r.name == "AlreadyCommitted" {
            if let (Some(id), Some(ev)) = (r.claim, r.evidence.as_ref()) {
                if self.claim_keys(id).as_ref() == Ok(keys) {
                    return Ok(self.report(p, identity, keys, id, "ALREADY_COMMITTED", ev, None));
                }
            }
        }
        Err(ApprovedRefusal::from_text(p, &r.to_string(), r.claim))
    }

    /// The keys a claim recorded, read back from the journal.
    fn claim_keys(&self, id: u64) -> Result<ClaimKeys, String> {
        self.bridge.with_home(|home| {
            let (recs, _) = home
                .compose
                .recall(aien_omega_compose::SUBJECT_HOST, 4096)
                .map_err(|e| format!("host records: {e}"))?;
            let views: Vec<_> = recs
                .iter()
                .map(|r| crate::compose::record_view(&mut home.compose, r))
                .collect();
            let l =
                approved_replay::ReplayLedger::from_records(&views).map_err(|e| e.to_string())?;
            l.claims
                .get(&id)
                .map(|row| row.keys.clone())
                .ok_or_else(|| format!("#{id} is not a claim"))
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn report(
        &self,
        p: &ApprovedProposal,
        identity: &ApprovalIdentity,
        keys: &ClaimKeys,
        claim: u64,
        state: &str,
        ev: &CommitEvidence,
        task: Option<ComposeTaskReport>,
    ) -> ApprovedComposeReport {
        ApprovedComposeReport {
            state: state.into(),
            request_id: p.request_id.clone(),
            trace_id: p.trace_id.clone(),
            approval_id: p.approval_id.clone(),
            approver: p.approver.clone(),
            path: p.path.clone(),
            content_sha256: p.content_sha256.clone(),
            approved_proposal_sha256: p.approved_proposal_sha256.clone(),
            compose_proposal_sha256: ev.compose_proposal_sha256.clone(),
            grant_links: vec![ev.cx_promotion, ev.cx_evidence],
            desk_key_id: identity.desk_key_id.clone(),
            approval_key: keys.approval_key.clone(),
            replay_claim: claim,
            task,
            approved_grant: None,
            target: None,
        }
    }
}

/// Test builds only (cargo feature `fault-hold`): abort the process at the
/// named point when `AIEN_FAULT_HOLD` names it, to prove the crash
/// boundaries of #249 B. The default build compiles this to nothing.
#[cfg(feature = "fault-hold")]
fn crash_point(name: &str) {
    if std::env::var("AIEN_FAULT_HOLD").as_deref() == Ok(name) {
        eprintln!("fault hold {name}: aborting (test build)");
        std::process::abort();
    }
}

#[cfg(not(feature = "fault-hold"))]
fn crash_point(_: &str) {}

/// Test seam: runs between the compose commit and the grant write, where a
/// concurrent edit of the target would land.
#[cfg(test)]
static BEFORE_GRANT: std::sync::Mutex<Option<Box<dyn Fn() + Send>>> = std::sync::Mutex::new(None);

#[cfg(test)]
mod toctou_tests {
    use super::*;
    use crate::approved_auth::{desk_key_path, DeskKey};

    fn proposal(b: &ComposeBridge, ws: &Path, id: &str, base: &str) -> ApprovedProposal {
        let content = "one\ntwo\nthree\n".to_string();
        let mut p = ApprovedProposal {
            request_id: id.into(),
            trace_id: format!("trace-{id}"),
            approval_id: format!("appr-{id}"),
            approver: "interplane-host".into(),
            path: "NOTES.md".into(),
            approved_proposal_sha256: approved_proposal_sha256("NOTES.md", &content),
            content_sha256: sha256_hex(content.as_bytes()),
            content,
            approval_mac: String::new(),
            requirements: Some("Add one line to NOTES.md".into()),
            requirements_mac: String::new(),
            requirements_base: Some(base.into()),
        };
        DeskKey::load(&desk_key_path(b.dir()))
            .unwrap()
            .seal(&mut p, ws);
        p
    }

    /// The target changes after the requirements were checked and before the
    /// grant is written: refused with BaseChanged, nothing authorizes a write.
    /// Unchanged, the same approval shape passes. (Linked build: the stub
    /// refuses at the claim, before the seam.)
    #[test]
    fn prior_changed_between_check_and_commit_is_refused() {
        let _home = crate::home_guard::home_slot();
        if !aien_omega_compose::LINKED {
            return;
        }
        let tmp = tempfile::tempdir().unwrap();
        let ws = tmp.path().join("ws");
        std::fs::create_dir_all(&ws).unwrap();
        let proposer: crate::compose::ComposeProposer =
            Arc::new(|_: &str, _: std::time::Duration| panic!("model ran"));
        let b = Arc::new(ComposeBridge::new(
            tmp.path().join("compose"),
            proposer,
            "t",
        ));
        DeskKey::create(&desk_key_path(b.dir())).unwrap();
        let hook = ProposerHook::new(b.clone());
        let wss = ws.display().to_string();
        let target = ws.join("NOTES.md");
        std::fs::write(&target, "one\ntwo\n").unwrap();
        let base = sha256_hex(b"one\ntwo\n");

        let t2 = target.clone();
        let fired = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let f2 = fired.clone();
        *BEFORE_GRANT.lock().unwrap() = Some(Box::new(move || {
            f2.store(true, std::sync::atomic::Ordering::SeqCst);
            std::fs::write(&t2, "one\ntwo\nEXTRA\n").unwrap()
        }));
        let e = hook
            .submit(&proposal(&b, &ws, "toctou1", &base), &wss)
            .expect_err("a changed prior must be refused");
        *BEFORE_GRANT.lock().unwrap() = None;
        // The seam sits after the compose commit, so firing proves the run reached it.
        assert!(
            fired.load(std::sync::atomic::Ordering::SeqCst),
            "seam did not fire; refusal was {}: {}",
            e.name,
            e.detail
        );
        eprintln!(
            "TOCTOU seam fired after compose commit; refusal = {}: {}",
            e.name, e.detail
        );
        assert_eq!(e.name, "GrantNotWritten", "{e}");
        assert!(e.detail.contains("BaseChanged"), "{e}");

        // Unchanged prior: the grant is written.
        std::fs::write(&target, "one\ntwo\n").unwrap();
        let ok = hook
            .submit(&proposal(&b, &ws, "toctou2", &base), &wss)
            .unwrap_or_else(|e| panic!("unchanged prior must pass: {e}"));
        assert_eq!(ok.state, "COMMITTED");
    }
}
