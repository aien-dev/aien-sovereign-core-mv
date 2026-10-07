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
use crate::control::ComposeTaskReport;
use crate::spine::{check_file_proposal, ComposeBridge};
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
    proposal_text(&p.path, &p.content)
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
            desk.authenticate(p)
        })()
        .map_err(|e| refusal(e, None))?;
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
            "apply approved proposal for {} (request {})",
            p.path, p.request_id
        );
        let r = match self.bridge.run_approved_task(&goal, workspace, &text) {
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
        Ok(self.report(
            p,
            &identity,
            &keys,
            claim.id,
            "COMMITTED",
            &evidence,
            Some(r),
        ))
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
                .map(|r| crate::spine::record_view(&mut home.compose, r))
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
