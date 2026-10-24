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

//! The daemon's generation record: evidence that THIS daemon generated THIS
//! text from the model file it loaded (INTERPLANE provenance, interplane#76).
//!
//! Written by the daemon when a `StreamTurn` finishes, into the compose
//! ledger, as an `effect`-class host note carrying the marker field
//! [`GENERATION`]. EVIDENCE ONLY: no grant, intent, replay, commit or effect
//! decision reads it (`effects::Ledger` skips it: it has neither a `phase` nor a
//! `tool` field). `ComposeNote` refuses to write one (`check_reserved_note`),
//! so only the daemon can. It claims nothing about capability or authority.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};

/// Marker field (and refused `ComposeNote` kind) of the generation record.
pub const GENERATION: &str = "generation";

/// The model files the daemon loaded, hashed once at load time (streamed,
/// 1 MiB chunks). `model_sha256` is the digest of what was loaded, in the form
/// `model_digest_kind` names: `file` (the single safetensors file's sha256) or
/// `index+shards` (sc#338: the manifest digest over the shard index and every
/// shard it names; docs/DAEMON_GENERATION_RECORD.md). `tokenizer_sha256` is the
/// tokenizer file's sha256.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelIdentity {
    pub model_sha256: String,
    pub model_digest_kind: String,
    pub model_path: String,
    pub tokenizer_sha256: String,
    pub tokenizer_path: String,
}

/// What one finished turn contributes to its record.
pub struct TurnEvidence<'a> {
    pub prompt_ids: &'a [u32],
    pub output_ids: &'a [u32],
    /// The exact text returned in `TurnFinished`.
    pub text: &'a str,
    pub total_tokens: usize,
    pub finish_reason: &'a str,
    /// Caller-supplied envelope ids (self-asserted by the client, no authority).
    pub request_id: u64,
    pub operation_id: u128,
    /// How the backend chose the output tokens (sc#294); None: no claim.
    pub decoding: Option<&'a aien_abi_core::DecodeObservation>,
    /// The tensor backend and its op counters when the call finished (sc#337); None: no claim.
    pub ops: Option<&'a aien_abi_core::OpEvidence>,
}

/// When this daemon started: unix milliseconds, captured once.
#[derive(Debug, Clone, Copy)]
pub struct DaemonStart(pub u128);

impl DaemonStart {
    pub fn now() -> Self {
        Self(
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis())
                .unwrap_or(0),
        )
    }
}

fn hex_sha256(bytes: &[u8]) -> String {
    aien_omega_compose::hex(&Sha256::digest(bytes))
}

/// An f32 setting as the JSON number it was written as (0.7, not 0.699999988).
fn f32_number(v: f32) -> Value {
    v.to_string()
        .parse::<f64>()
        .ok()
        .and_then(serde_json::Number::from_f64)
        .map_or(Value::Null, Value::Number)
}

/// Adds `decoding` (sc#294): how the backend chose the output tokens, counted
/// in the sampling branch it took, not read from the request's settings. The
/// field is left out when nothing observed it (a stub or a backend without the
/// observation): no field means no claim, never "greedy".
fn with_decoding(record: &mut Value, d: Option<&aien_abi_core::DecodeObservation>) {
    let Some(d) = d else { return };
    let mut o = json!({
        "mode": d.mode(),
        "greedy_tokens": d.greedy_tokens,
        "sampled_tokens": d.sampled_tokens,
    });
    if let Some(t) = d.temperature {
        o["temperature"] = f32_number(t);
    }
    if let Some(p) = d.top_p {
        o["top_p"] = f32_number(p);
    }
    if let Some(s) = d.seed_request_id {
        o["seed_request_id"] = json!(s);
    }
    record["decoding"] = o;
}

/// Adds `ops` (sc#337): the tensor backend that ran the call and its op
/// counters when the call finished, as process totals since the daemon built
/// the backend (`native_fallbacks`: claimed-native ops that ran on the
/// reference CPU path, always 0 in a production build, which panics on the
/// first; `reference_runs`: ops on the reference path by design). Left out
/// when the backend does not account its ops: no field means no claim.
fn with_ops(record: &mut Value, o: Option<&aien_abi_core::OpEvidence>) {
    let Some(o) = o else { return };
    record["ops"] = json!({
        "backend": o.backend,
        "native_fallbacks": o.native_fallbacks,
        "reference_runs": o.reference_runs,
        "scope": "process",
        "report": o.report,
    });
    if let Some(n) = o.gpu_executions {
        record["ops"]["gpu_executions"] = json!(n); // sc#402, process total like the rest
    }
}

/// The line the daemon logs at the end of every model call (sc#337): the
/// backend's `OP_REPORT ...` line with ` backend=<name>` appended and, after it,
/// ` gpu_executions=<n>` when the backend counts chip calls (sc#402). None when
/// the backend does not account its ops.
///
/// Layout contract (sc#384, 2026-10-11): the frozen OPEN-MODEL-QWEN3 v5 row
/// `-OPR` parses `OP_REPORT native=[..] reference=[..] native_fallbacks=[..]
/// reference_runs=[..] backend=<rest>` to the end of the line; any new field
/// goes AFTER `backend=`, never between the bracketed fields and `backend=`.
/// Putting `gpu_executions` before `backend=` (sc#405) made every v7 dry
/// launch fail that row.
pub fn op_report_line(o: Option<&aien_abi_core::OpEvidence>) -> Option<String> {
    o.map(|o| match o.gpu_executions {
        Some(n) => format!("{} backend={} gpu_executions={n}", o.report, o.backend),
        None => format!("{} backend={}", o.report, o.backend),
    })
}

/// The record's JSON body (field list is documented in docs/DAEMON_GENERATION_RECORD.md).
pub fn build_record(id: &ModelIdentity, t: &TurnEvidence, start: DaemonStart) -> Value {
    let (pid, start_ticks) = crate::effects::self_executor();
    let mut record = json!({
        GENERATION: 1,
        "v": 1,
        "model_sha256": id.model_sha256,
        "model_digest_kind": id.model_digest_kind,
        "model_path": id.model_path,
        "tokenizer_sha256": id.tokenizer_sha256,
        "tokenizer_path": id.tokenizer_path,
        "prompt_ids_sha256": crate::compose::token_ids_sha256(t.prompt_ids),
        "output_token_ids_sha256": crate::compose::token_ids_sha256(t.output_ids),
        "output_text_sha256": hex_sha256(t.text.as_bytes()),
        "total_tokens": t.total_tokens,
        "output_tokens": t.output_ids.len(),
        "finish_reason": t.finish_reason,
        "daemon": {"pid": pid, "start_ticks": start_ticks, "started_unix_ms": start.0.to_string()},
        // Caller-asserted: the client chooses the envelope ids; recorded, not trusted.
        "request_id": t.request_id,
        "operation_id": t.operation_id.to_string(),
    });
    with_decoding(&mut record, t.decoding);
    with_ops(&mut record, t.ops);
    record
}

/// Why an `effect`-class note carrying a `generation` marker, and not a new
/// note kind: omega-compose's `NoteKind` enum (constraint, authorization,
/// effect) is a pinned external dependency with fixed kinds, so this crate
/// cannot add one. The marker field plays the role of a kind: `ComposeNote`
/// refuses it and no ledger reader acts on it.
/// Append `record` to the ledger; the id is returned only once it is written.
pub(crate) fn write(home: &mut crate::compose::ComposeHome, record: &Value) -> Result<u64, String> {
    crate::effects::append(home, aien_omega_compose::NoteKind::Effect, &[], record).map(|n| n.id)
}

// ---------------------------------------------------------------------------
// Provenance link: which generation produced a committed proposal, and which
// ALLEN asked for it. EVIDENCE ONLY, like the generation record itself.
// ---------------------------------------------------------------------------

use crate::control::ComposeRecordView;

/// Top-level field of the evidence the daemon copies along the chain
/// compose-commit -> minted grant -> intent -> ack (and onto an approved
/// grant). Reserved: `ComposeNote` refuses any note that carries it, and no
/// ledger reader (`effects::Ledger`) parses it. Only [`Provenance::from_text`]
/// does, and only `effects::stamp` calls that.
pub const PROVENANCE: &str = "provenance";

/// The `allen_agent` value when no ALLEN identity is attached to the daemon.
pub const NO_AGENT: &str = "none";

/// The evidence fields of the link.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Ledger id of the generation record of the attempt the committed
    /// proposal came from. `None` when no record was written (no ledger, no
    /// model identity, a proposer that does not expose token ids, an approved
    /// proposal): an absent claim, not a pass.
    pub generation_record: Option<u64>,
    /// The ALLEN LogicalAgentId (64 hex, `aien_allen::Resolved::agent`) the
    /// task ran under, or [`NO_AGENT`].
    pub allen_agent: String,
    /// Set when a copy found the source's generation id not naming a verified
    /// generation record: the old id, kept visible. `generation_record` is then
    /// null, and this marker tells that apart from "no model ran".
    pub stale_generation: Option<u64>,
    /// Why the evidence is degraded (`generation_record_stale`,
    /// `allen_agent_malformed`, `provenance_malformed`), `;`-joined.
    /// Absent on ordinary records.
    pub note: Option<String>,
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            generation_record: None,
            allen_agent: NO_AGENT.to_string(),
            stale_generation: None,
            note: None,
        }
    }
}

impl Provenance {
    pub fn new(generation_record: Option<u64>, allen: Option<&aien_allen::Resolved>) -> Self {
        Self {
            generation_record,
            allen_agent: allen.map_or_else(|| NO_AGENT.to_string(), |r| aien_allen::hex(&r.agent)),
            ..Self::default()
        }
    }

    fn add_note(&mut self, n: &str) {
        match &mut self.note {
            Some(x) if x.split("; ").any(|y| y == n) => {}
            Some(x) => {
                x.push_str("; ");
                x.push_str(n);
            }
            None => self.note = Some(n.to_string()),
        }
    }

    pub fn to_json(&self) -> Value {
        let mut v =
            json!({"generation_record": self.generation_record, "allen_agent": self.allen_agent});
        if let Some(s) = self.stale_generation {
            v["generation_record_stale"] = json!(s);
        }
        if let Some(n) = &self.note {
            v["provenance_note"] = json!(n);
        }
        v
    }

    /// Lenient read of a record's text, never an error. A record with no
    /// provenance field at all (an old record) reads as the plain default.
    /// A field that is present but damaged reads as the default values PLUS a
    /// visible note, so it is never mistaken for "no model ran" / "no ALLEN".
    pub fn from_text(text: &str) -> Self {
        let Ok(v) = serde_json::from_str::<Value>(text) else {
            return Self::default();
        };
        let Some(p) = v.get(PROVENANCE) else {
            return Self::default();
        };
        let mut out = Self::default();
        if !p.is_object() {
            tracing::warn!("provenance field is not an object; copied as default with a note");
            out.add_note("provenance_malformed");
            return out;
        }
        match p.get("allen_agent").and_then(Value::as_str) {
            Some(a)
                if a == NO_AGENT
                    || (a.len() == 64
                        && a.bytes().all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f'))) =>
            {
                out.allen_agent = a.to_string()
            }
            _ => {
                tracing::warn!(
                    "provenance allen_agent is missing or malformed; copied as none with a note"
                );
                out.add_note("allen_agent_malformed");
            }
        }
        out.generation_record = p.get("generation_record").and_then(Value::as_u64);
        out.stale_generation = p.get("generation_record_stale").and_then(Value::as_u64);
        // Carry an earlier note along the chain (bounded, printable ASCII only).
        if let Some(n) = p.get("provenance_note").and_then(Value::as_str) {
            let n: String = n
                .chars()
                .filter(|c| c.is_ascii_graphic() || *c == ' ')
                .take(200)
                .collect();
            for part in n.split("; ").filter(|s| !s.is_empty()) {
                out.add_note(part);
            }
        }
        out
    }
}

/// True when `id` names a verified `effect` note in `views` that is a
/// generation record (marker 1, no links).
pub fn generation_exists(views: &[ComposeRecordView], id: u64) -> bool {
    views.iter().any(|r| {
        r.id == id
            && r.verified
            && r.note.as_deref() == Some("effect")
            && r.links.iter().all(|&l| l == 0)
            && r.text
                .as_deref()
                .and_then(|t| serde_json::from_str::<Value>(t).ok())
                .is_some_and(|v| v.get(GENERATION).and_then(Value::as_u64) == Some(1))
    })
}

/// The provenance to copy from record `source` onto the next record of the
/// chain. A generation id that does not name a real generation record is
/// dropped (not refused): copying must not change any decision.
pub fn inherit(views: &[ComposeRecordView], source: u64) -> Provenance {
    let mut p = views
        .iter()
        .find(|r| r.id == source)
        .and_then(|r| r.text.as_deref())
        .map(Provenance::from_text)
        .unwrap_or_default();
    if let Some(g) = p.generation_record {
        if !generation_exists(views, g) {
            tracing::warn!(
                "provenance names generation record #{g}, which is not a verified generation record; copied as null with a stale marker"
            );
            p.generation_record = None;
            p.stale_generation = Some(g);
            p.add_note("generation_record_stale");
        }
    }
    p
}

/// What one compose proposal attempt contributes to its generation record.
pub struct ComposeEvidence<'a> {
    /// `token_ids_sha256` of the submitted prompt ids.
    pub prompt_ids_sha256: &'a str,
    pub prompt_tokens: usize,
    pub output_ids: &'a [u32],
    /// The reply text handed to the template parser (assistant prefix plus the
    /// decoded generated tokens).
    pub text: &'a str,
    pub finish_reason: &'a str,
    /// The compose task and attempt (1-based) that made the generation. Daemon-assigned.
    pub task: u64,
    pub attempt: u32,
    /// How the backend chose the output tokens (sc#294); None: no claim.
    pub decoding: Option<&'a aien_abi_core::DecodeObservation>,
    /// The tensor backend and its op counters when the call finished (sc#337); None: no claim.
    pub ops: Option<&'a aien_abi_core::OpEvidence>,
}

/// The generation record of one compose proposal attempt: the same fields as
/// [`build_record`] plus `origin`, `task` and `attempt`. `request_id` and
/// `operation_id` are 0: there is no client envelope for a model call the
/// compose task makes itself.
pub fn build_compose_record(id: &ModelIdentity, e: &ComposeEvidence, start: DaemonStart) -> Value {
    let (pid, start_ticks) = crate::effects::self_executor();
    let mut record = json!({
        GENERATION: 1,
        "v": 1,
        "origin": "compose_proposal",
        "task": e.task,
        "attempt": e.attempt,
        "model_sha256": id.model_sha256,
        "model_digest_kind": id.model_digest_kind,
        "model_path": id.model_path,
        "tokenizer_sha256": id.tokenizer_sha256,
        "tokenizer_path": id.tokenizer_path,
        "prompt_ids_sha256": e.prompt_ids_sha256,
        "output_token_ids_sha256": crate::compose::token_ids_sha256(e.output_ids),
        "output_text_sha256": hex_sha256(e.text.as_bytes()),
        "total_tokens": e.prompt_tokens + e.output_ids.len(),
        "output_tokens": e.output_ids.len(),
        "finish_reason": e.finish_reason,
        "daemon": {"pid": pid, "start_ticks": start_ticks, "started_unix_ms": start.0.to_string()},
        "request_id": 0,
        "operation_id": "0",
    });
    with_decoding(&mut record, e.decoding);
    with_ops(&mut record, e.ops);
    record
}
