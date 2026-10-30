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

//! Correlation-only execution trace for AIEN (aien-sovereign-core#398).
//!
//! A trace event says "this boundary was crossed, with these ids". It is
//! OBSERVATIONAL. It is not evidence and it is not authorization: the
//! `EffectReceipt` ledger, the compose ledger, the generation record and the
//! independent receipt verifier keep governing. An event only points at those
//! artifacts by digest or relative path.
//!
//! Privacy is enforced by type. This crate never carries prompts, tool
//! arguments, model output or memory items. The only free text is
//! [`BoundedNote`] (64 bytes, sanitized). Every other string is a
//! [`BoundedId`] (64 chars from a small alphabet), a [`Digest`] (hex) or a
//! [`RelPath`] (relative, no `..`). Deserialization re-applies the same rules,
//! so a tampered file cannot smuggle a longer or wider field through.
//!
//! The default sink is [`NullSink`]: tracing is off unless a caller installs
//! another sink.

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Longest accepted [`BoundedId`], in characters.
pub const MAX_ID_CHARS: usize = 64;
/// Longest accepted [`BoundedNote`], in bytes.
pub const MAX_NOTE_BYTES: usize = 64;
/// Default queue capacity of [`BoundedJsonlSink`].
pub const DEFAULT_CAPACITY: usize = 4096;
const MAX_DIGEST_CHARS: usize = 128;
const MAX_PATH_BYTES: usize = 256;

/// Errors of this crate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TraceError {
    /// Not 32 lowercase-or-uppercase hex characters.
    BadTraceId,
    /// A digest was empty, too long or not hex.
    BadDigest,
    /// A path was absolute, contained `..`, or was empty or too long.
    BadPath,
    /// An event names a parent that is not in the set.
    MissingParent { event_id: u64, parent: u64 },
    /// The events do not all share one trace id.
    MixedTraceIds,
    /// Two events share an event id.
    DuplicateEvent(u64),
    /// No events given.
    Empty,
}

impl fmt::Display for TraceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadTraceId => write!(f, "trace id must be 32 hex characters"),
            Self::BadDigest => write!(f, "digest must be 1..=128 hex characters"),
            Self::BadPath => write!(f, "path must be relative, without `..`"),
            Self::MissingParent { event_id, parent } => {
                write!(f, "event {event_id} names missing parent {parent}")
            }
            Self::MixedTraceIds => write!(f, "events belong to more than one trace"),
            Self::DuplicateEvent(id) => write!(f, "duplicate event id {id}"),
            Self::Empty => write!(f, "no events"),
        }
    }
}

impl std::error::Error for TraceError {}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Wall clock in unix milliseconds (0 if the clock is before the epoch).
pub fn now_unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

fn mix64(mut x: u64) -> u64 {
    // splitmix64 finalizer.
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

/// 16-byte run identifier.
///
/// A correlation key only. It is NEVER a secret and NEVER a proof: anyone can
/// compute or copy one, so nothing may be authorized or believed because a
/// trace id matches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TraceId(pub [u8; 16]);

impl TraceId {
    /// Mixes the inputs with splitmix64 applied twice per half. Deterministic:
    /// the same inputs give the same id. Not a cryptographic hash.
    pub fn derive(seed_a: u64, seed_b: u64, counter: u64) -> Self {
        let hi = mix64(mix64(seed_a) ^ counter);
        let lo = mix64(mix64(seed_b ^ hi) ^ counter.rotate_left(32));
        let mut out = [0u8; 16];
        out[..8].copy_from_slice(&hi.to_be_bytes());
        out[8..].copy_from_slice(&lo.to_be_bytes());
        Self(out)
    }

    /// 32 lowercase hex characters.
    pub fn to_hex(&self) -> String {
        const H: &[u8; 16] = b"0123456789abcdef";
        let mut s = String::with_capacity(32);
        for b in self.0 {
            s.push(H[(b >> 4) as usize] as char);
            s.push(H[(b & 15) as usize] as char);
        }
        s
    }

    /// Parses 32 hex characters (either case).
    pub fn parse_hex(s: &str) -> Result<Self, TraceError> {
        let b = s.as_bytes();
        if b.len() != 32 {
            return Err(TraceError::BadTraceId);
        }
        let nib = |c: u8| match c {
            b'0'..=b'9' => Ok(c - b'0'),
            b'a'..=b'f' => Ok(c - b'a' + 10),
            b'A'..=b'F' => Ok(c - b'A' + 10),
            _ => Err(TraceError::BadTraceId),
        };
        let mut out = [0u8; 16];
        for i in 0..16 {
            out[i] = (nib(b[2 * i])? << 4) | nib(b[2 * i + 1])?;
        }
        Ok(Self(out))
    }
}

impl fmt::Display for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

impl fmt::Debug for TraceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "TraceId({})", self.to_hex())
    }
}

impl Serialize for TraceId {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_hex())
    }
}

impl<'de> Deserialize<'de> for TraceId {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        Self::parse_hex(&s).map_err(serde::de::Error::custom)
    }
}

/// What boundary was crossed.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EventKind {
    SwarmLaunched,
    BranchForked,
    SequenceAdmitted,
    SequencePreempted,
    SequenceFinished,
    SwarmCancelled,
    ModelTurn,
    ToolRequested,
    AuthorityDecided,
    EffectExecuted,
    ResultReceived,
    Failure,
    Cancelled,
    TimedOut,
    InterplaneRequest,
    TraceDropped,
}

/// Terminal or current state of the boundary.
///
/// One to one with the AEGIS `FailureClass` vocabulary (Rejected,
/// Unavailable, Timeout, Cancelled, Uncertain, Failed, Malformed,
/// ApprovalPending) plus `ok` for success. `rejected` also covers Interplane
/// decisions `denied`, `not_found` and `invalid`. `approval_pending` is not a
/// failure and is terminal for the attempt. `uncertain` maps to aien-mcp
/// `CallOutcome::Uncertain`: the request may have reached the server, and it
/// is never retried automatically.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EventStatus {
    Ok,
    Rejected,
    Unavailable,
    TimedOut,
    Cancelled,
    Uncertain,
    Failed,
    Malformed,
    ApprovalPending,
}

/// Whether an external effect happened, as far as the recorder knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum EffectCertainty {
    NoEffect,
    Uncertain,
    EffectOccurred,
}

/// Short identifier: at most 64 characters from `[A-Za-z0-9._:-]`; anything
/// else is dropped, extra length is cut.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub struct BoundedId(String);

impl BoundedId {
    pub fn new(s: &str) -> Self {
        Self(
            s.chars()
                .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | ':' | '-'))
                .take(MAX_ID_CHARS)
                .collect(),
        )
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl From<String> for BoundedId {
    fn from(s: String) -> Self {
        Self::new(&s)
    }
}
impl From<BoundedId> for String {
    fn from(b: BoundedId) -> Self {
        b.0
    }
}

/// The ONLY free-text field of an event. Control characters, `<` and `>` are
/// removed, then the text is cut to 64 bytes on a character boundary.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(from = "String", into = "String")]
pub struct BoundedNote(String);

impl BoundedNote {
    pub fn new(s: &str) -> Self {
        let mut out = String::new();
        for c in s
            .chars()
            .filter(|c| !c.is_control() && *c != '<' && *c != '>')
        {
            if out.len() + c.len_utf8() > MAX_NOTE_BYTES {
                break;
            }
            out.push(c);
        }
        Self(out)
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl From<String> for BoundedNote {
    fn from(s: String) -> Self {
        Self::new(&s)
    }
}
impl From<BoundedNote> for String {
    fn from(b: BoundedNote) -> Self {
        b.0
    }
}

/// Hex digest, 1..=128 characters.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct Digest(String);

impl Digest {
    pub fn new(s: &str) -> Result<Self, TraceError> {
        if s.is_empty() || s.len() > MAX_DIGEST_CHARS || !s.bytes().all(|b| b.is_ascii_hexdigit()) {
            return Err(TraceError::BadDigest);
        }
        Ok(Self(s.to_ascii_lowercase()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for Digest {
    type Error = TraceError;
    fn try_from(s: String) -> Result<Self, TraceError> {
        Self::new(&s)
    }
}
impl From<Digest> for String {
    fn from(d: Digest) -> Self {
        d.0
    }
}

/// Relative path: not absolute, no `..` component, no NUL, at most 256 bytes.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub struct RelPath(String);

impl RelPath {
    pub fn new(s: &str) -> Result<Self, TraceError> {
        let bad = s.is_empty()
            || s.len() > MAX_PATH_BYTES
            || s.contains('\0')
            || s.starts_with('/')
            || s.starts_with('\\')
            || s.as_bytes().get(1) == Some(&b':')
            || s.split(['/', '\\']).any(|p| p == "..");
        if bad {
            return Err(TraceError::BadPath);
        }
        Ok(Self(s.to_string()))
    }
    pub fn as_str(&self) -> &str {
        &self.0
    }
}
impl TryFrom<String> for RelPath {
    type Error = TraceError;
    fn try_from(s: String) -> Result<Self, TraceError> {
        Self::new(&s)
    }
}
impl From<RelPath> for String {
    fn from(p: RelPath) -> Self {
        p.0
    }
}

/// Ids a trace can join on. All optional.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct CorrelationIds {
    pub swarm_id: Option<u64>,
    pub world_id: Option<u64>,
    pub root_sequence_id: Option<u64>,
    pub sequence_id: Option<u64>,
    pub step_id: Option<u64>,
    pub model_turn: Option<u64>,
    pub tool_request_id: Option<BoundedId>,
    pub decision_id: Option<BoundedId>,
    pub grant_id: Option<BoundedId>,
    pub generation_record: Option<u64>,
    pub interplane_trace_id: Option<TraceId>,
    pub interplane_message_id: Option<BoundedId>,
    pub interplane_request_id: Option<BoundedId>,
}

/// Pointers to independent proof artifacts. A pointer is not proof.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct EvidenceRefs {
    pub effect_receipt_digest: Option<Digest>,
    pub result_digest: Option<Digest>,
    pub provenance_record: Option<RelPath>,
    pub intent_digest: Option<Digest>,
}

impl EvidenceRefs {
    pub fn with_effect_receipt_digest(mut self, hex: &str) -> Result<Self, TraceError> {
        self.effect_receipt_digest = Some(Digest::new(hex)?);
        Ok(self)
    }
    pub fn with_result_digest(mut self, hex: &str) -> Result<Self, TraceError> {
        self.result_digest = Some(Digest::new(hex)?);
        Ok(self)
    }
    pub fn with_intent_digest(mut self, hex: &str) -> Result<Self, TraceError> {
        self.intent_digest = Some(Digest::new(hex)?);
        Ok(self)
    }
    /// Refuses absolute paths and `..`.
    pub fn with_provenance_record(mut self, path: &str) -> Result<Self, TraceError> {
        self.provenance_record = Some(RelPath::new(path)?);
        Ok(self)
    }
}

/// One boundary crossing.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TraceEvent {
    pub trace_id: TraceId,
    /// Monotonic per sink.
    pub event_id: u64,
    pub parent_event_id: Option<u64>,
    pub at_unix_ms: u64,
    pub kind: EventKind,
    pub status: EventStatus,
    #[serde(default)]
    pub effect_certainty: Option<EffectCertainty>,
    #[serde(default)]
    pub ids: CorrelationIds,
    #[serde(default)]
    pub refs: EvidenceRefs,
    #[serde(default)]
    pub note: Option<BoundedNote>,
}

/// Destination of events. Implementations must never block the caller for I/O.
pub trait TraceSink: Send + Sync {
    /// False for sinks that discard everything, so callers can skip building events.
    fn enabled(&self) -> bool {
        true
    }
    fn emit(&self, event: TraceEvent);
    fn next_event_id(&self) -> u64;
}

/// Discards everything. The default everywhere.
#[derive(Debug, Default)]
pub struct NullSink;

impl TraceSink for NullSink {
    fn enabled(&self) -> bool {
        false
    }
    fn emit(&self, _event: TraceEvent) {}
    fn next_event_id(&self) -> u64 {
        0
    }
}

/// Keeps events in memory. For tests.
#[derive(Debug, Default)]
pub struct MemorySink {
    events: Mutex<Vec<TraceEvent>>,
    next: AtomicU64,
}

impl MemorySink {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn drain(&self) -> Vec<TraceEvent> {
        std::mem::take(&mut *lock(&self.events))
    }
}

impl TraceSink for MemorySink {
    fn emit(&self, event: TraceEvent) {
        lock(&self.events).push(event);
    }
    fn next_event_id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Bounded in-memory queue flushed to an append-only JSONL file.
///
/// `emit` takes a mutex only for one queue push and never does I/O. When the
/// queue is full the event is dropped and counted; the next `flush` appends one
/// `trace_dropped` event saying how many. No thread runs unless
/// [`BoundedJsonlSink::spawn_flusher`] is called.
#[derive(Debug)]
pub struct BoundedJsonlSink {
    path: PathBuf,
    capacity: usize,
    queue: Mutex<VecDeque<TraceEvent>>,
    next: AtomicU64,
    dropped: AtomicU64,
    dropped_total: AtomicU64,
    last_trace: Mutex<TraceId>,
}

/// Refuses an export path that is a symbolic link, so nothing is written
/// through a link planted at the configured path. An absent path is fine (the
/// file is created). Metadata errors other than "absent" are returned.
fn refuse_symlink(path: &Path) -> std::io::Result<()> {
    match std::fs::symlink_metadata(path) {
        Ok(m) if m.file_type().is_symlink() => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "trace export path is a symbolic link",
        )),
        Ok(_) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

impl BoundedJsonlSink {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self::with_capacity(path, DEFAULT_CAPACITY)
    }

    pub fn with_capacity(path: impl Into<PathBuf>, capacity: usize) -> Self {
        let capacity = capacity.max(1);
        Self {
            path: path.into(),
            capacity,
            queue: Mutex::new(VecDeque::with_capacity(capacity)),
            next: AtomicU64::new(0),
            dropped: AtomicU64::new(0),
            dropped_total: AtomicU64::new(0),
            last_trace: Mutex::new(TraceId([0; 16])),
        }
    }

    /// Lifetime number of dropped events.
    pub fn dropped_total(&self) -> u64 {
        self.dropped_total.load(Ordering::Relaxed)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends queued events, one JSON object per line. The path must not be a
    /// symbolic link (a link is refused and the batch counted as dropped); the
    /// file is created with mode 0o600 on unix and set back to 0o600 on every
    /// flush. After drops, one `trace_dropped` event follows.
    pub fn flush(&self) -> std::io::Result<()> {
        let batch: Vec<TraceEvent> = lock(&self.queue).drain(..).collect();
        let pending = self.dropped.swap(0, Ordering::Relaxed);
        if batch.is_empty() && pending == 0 {
            return Ok(());
        }
        let mut lines = Vec::new();
        for e in &batch {
            serde_json::to_writer(&mut lines, e).map_err(std::io::Error::other)?;
            lines.push(b'\n');
        }
        if pending > 0 {
            let trace_id = batch
                .last()
                .map(|e| e.trace_id)
                .unwrap_or(*lock(&self.last_trace));
            let marker = TraceEvent {
                trace_id,
                event_id: self.next_event_id(),
                parent_event_id: None,
                at_unix_ms: now_unix_ms(),
                kind: EventKind::TraceDropped,
                status: EventStatus::Ok,
                effect_certainty: None,
                ids: CorrelationIds::default(),
                refs: EvidenceRefs::default(),
                note: Some(BoundedNote::new(&format!("dropped {pending} events"))),
            };
            serde_json::to_writer(&mut lines, &marker).map_err(std::io::Error::other)?;
            lines.push(b'\n');
        }
        let mut opts = OpenOptions::new();
        opts.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            opts.mode(0o600);
        }
        let result = refuse_symlink(&self.path)
            .and_then(|()| opts.open(&self.path))
            .and_then(|mut f| {
                #[cfg(unix)]
                {
                    use std::os::unix::fs::PermissionsExt;
                    f.set_permissions(std::fs::Permissions::from_mode(0o600))?;
                }
                f.write_all(&lines)
            });
        if result.is_err() {
            // Not written: report the loss instead of hiding it.
            let lost = batch.len() as u64 + pending;
            self.dropped.fetch_add(lost, Ordering::Relaxed);
            self.dropped_total
                .fetch_add(batch.len() as u64, Ordering::Relaxed);
        }
        result
    }

    /// Starts a thread that flushes every `interval` and once more on stop.
    pub fn spawn_flusher(self: &Arc<Self>, interval: Duration) -> FlusherHandle {
        let sink = Arc::clone(self);
        let (tx, rx) = mpsc::channel::<()>();
        let join = std::thread::spawn(move || loop {
            match rx.recv_timeout(interval) {
                Err(RecvTimeoutError::Timeout) => {
                    let _ = sink.flush();
                }
                _ => {
                    let _ = sink.flush();
                    break;
                }
            }
        });
        FlusherHandle {
            stop: Some(tx),
            join: Some(join),
        }
    }
}

impl TraceSink for BoundedJsonlSink {
    fn emit(&self, event: TraceEvent) {
        let mut q = lock(&self.queue);
        if q.len() >= self.capacity {
            drop(q);
            self.dropped.fetch_add(1, Ordering::Relaxed);
            self.dropped_total.fetch_add(1, Ordering::Relaxed);
            return;
        }
        q.push_back(event);
    }
    fn next_event_id(&self) -> u64 {
        self.next.fetch_add(1, Ordering::Relaxed) + 1
    }
}

/// Handle of the flusher thread. Dropping it stops the thread.
#[derive(Debug)]
pub struct FlusherHandle {
    stop: Option<mpsc::Sender<()>>,
    join: Option<JoinHandle<()>>,
}

impl FlusherHandle {
    /// Stops the thread after one last flush.
    pub fn stop(mut self) {
        self.shutdown();
    }
    fn shutdown(&mut self) {
        self.stop.take();
        if let Some(j) = self.join.take() {
            let _ = j.join();
        }
    }
}

impl Drop for FlusherHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// Emits events of one trace with correct parent links.
#[derive(Clone)]
pub struct TraceContext {
    trace_id: TraceId,
    sink: Arc<dyn TraceSink>,
    last: Arc<AtomicU64>,
}

const NO_EVENT: u64 = u64::MAX;

impl TraceContext {
    pub fn new(trace_id: TraceId, sink: Arc<dyn TraceSink>) -> Self {
        Self {
            trace_id,
            sink,
            last: Arc::new(AtomicU64::new(NO_EVENT)),
        }
    }

    pub fn trace_id(&self) -> TraceId {
        self.trace_id
    }

    /// Id of the most recent event emitted through this context, if any.
    pub fn last_event_id(&self) -> Option<u64> {
        match self.last.load(Ordering::Relaxed) {
            NO_EVENT => None,
            v => Some(v),
        }
    }

    fn put(
        &self,
        parent: Option<u64>,
        kind: EventKind,
        status: EventStatus,
        ids: CorrelationIds,
        refs: EvidenceRefs,
        note: Option<BoundedNote>,
    ) -> u64 {
        if !self.sink.enabled() {
            return 0;
        }
        let id = self.sink.next_event_id();
        self.sink.emit(TraceEvent {
            trace_id: self.trace_id,
            event_id: id,
            parent_event_id: parent,
            at_unix_ms: now_unix_ms(),
            kind,
            status,
            effect_certainty: None,
            ids,
            refs,
            note,
        });
        self.last.store(id, Ordering::Relaxed);
        id
    }

    /// First event: no parent.
    pub fn root(
        &self,
        kind: EventKind,
        status: EventStatus,
        ids: CorrelationIds,
        refs: EvidenceRefs,
        note: Option<BoundedNote>,
    ) -> u64 {
        self.put(None, kind, status, ids, refs, note)
    }

    /// Child of the most recent event of this context.
    pub fn child(
        &self,
        kind: EventKind,
        status: EventStatus,
        ids: CorrelationIds,
        refs: EvidenceRefs,
        note: Option<BoundedNote>,
    ) -> u64 {
        self.put(self.last_event_id(), kind, status, ids, refs, note)
    }

    /// Child of a named event, for fan-out and across task boundaries.
    pub fn child_of(
        &self,
        parent: u64,
        kind: EventKind,
        status: EventStatus,
        ids: CorrelationIds,
        refs: EvidenceRefs,
        note: Option<BoundedNote>,
    ) -> u64 {
        self.put(Some(parent), kind, status, ids, refs, note)
    }
}

/// Parent/child tree of one trace.
#[derive(Debug)]
pub struct Tree {
    events: Vec<TraceEvent>,
    children: HashMap<u64, Vec<u64>>,
    roots: Vec<u64>,
}

impl Tree {
    /// Events sorted by `(at_unix_ms, event_id)`.
    pub fn chronological(&self) -> Vec<&TraceEvent> {
        let mut v: Vec<&TraceEvent> = self.events.iter().collect();
        v.sort_by_key(|e| (e.at_unix_ms, e.event_id));
        v
    }
    /// Event ids whose parent is `id`, in event id order.
    pub fn children(&self, id: u64) -> &[u64] {
        self.children.get(&id).map(Vec::as_slice).unwrap_or(&[])
    }
    /// Events without a parent.
    pub fn roots(&self) -> &[u64] {
        &self.roots
    }
    pub fn get(&self, id: u64) -> Option<&TraceEvent> {
        self.events.iter().find(|e| e.event_id == id)
    }
    pub fn len(&self) -> usize {
        self.events.len()
    }
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }
}

/// Builds the tree of one trace. Fails on mixed trace ids, duplicate event
/// ids, or a parent that is not in the set.
pub fn reconstruct(events: &[TraceEvent]) -> Result<Tree, TraceError> {
    let first = events.first().ok_or(TraceError::Empty)?;
    let mut ids = HashMap::new();
    for e in events {
        if e.trace_id != first.trace_id {
            return Err(TraceError::MixedTraceIds);
        }
        if ids.insert(e.event_id, ()).is_some() {
            return Err(TraceError::DuplicateEvent(e.event_id));
        }
    }
    let mut children: HashMap<u64, Vec<u64>> = HashMap::new();
    let mut roots = Vec::new();
    for e in events {
        match e.parent_event_id {
            None => roots.push(e.event_id),
            Some(p) if ids.contains_key(&p) => children.entry(p).or_default().push(e.event_id),
            Some(p) => {
                return Err(TraceError::MissingParent {
                    event_id: e.event_id,
                    parent: p,
                })
            }
        }
    }
    roots.sort_unstable();
    for v in children.values_mut() {
        v.sort_unstable();
    }
    Ok(Tree {
        events: events.to_vec(),
        children,
        roots,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    const ALL_STATUS: [EventStatus; 9] = [
        EventStatus::Ok,
        EventStatus::Rejected,
        EventStatus::Unavailable,
        EventStatus::TimedOut,
        EventStatus::Cancelled,
        EventStatus::Uncertain,
        EventStatus::Failed,
        EventStatus::Malformed,
        EventStatus::ApprovalPending,
    ];

    fn tmp(name: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "aien-trace-{name}-{}-{}-{}.jsonl",
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed),
            now_unix_ms()
        ))
    }

    fn ev(trace: TraceId, id: u64, parent: Option<u64>) -> TraceEvent {
        TraceEvent {
            trace_id: trace,
            event_id: id,
            parent_event_id: parent,
            at_unix_ms: id,
            kind: EventKind::ModelTurn,
            status: EventStatus::Ok,
            effect_certainty: None,
            ids: CorrelationIds::default(),
            refs: EvidenceRefs::default(),
            note: None,
        }
    }

    #[test]
    fn hex_round_trip_and_derive_is_deterministic() {
        let a = TraceId::derive(1, 2, 3);
        assert_eq!(a, TraceId::derive(1, 2, 3));
        assert_ne!(a, TraceId::derive(1, 2, 4));
        assert_ne!(a, TraceId::derive(2, 1, 3));
        let h = a.to_hex();
        assert_eq!(h.len(), 32);
        assert!(h
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)));
        assert_eq!(TraceId::parse_hex(&h).unwrap(), a);
        assert_eq!(format!("{a}"), h);
        assert!(TraceId::parse_hex("zz").is_err());
        assert!(TraceId::parse_hex(&"g".repeat(32)).is_err());
    }

    #[test]
    fn status_strings_are_distinct_and_exact() {
        let mut got: Vec<String> = ALL_STATUS
            .iter()
            .map(|s| {
                serde_json::to_value(s)
                    .unwrap()
                    .as_str()
                    .unwrap()
                    .to_string()
            })
            .collect();
        got.sort();
        let mut want: Vec<String> = [
            "ok",
            "rejected",
            "unavailable",
            "timed_out",
            "cancelled",
            "uncertain",
            "failed",
            "malformed",
            "approval_pending",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        want.sort();
        assert_eq!(got, want);
        let mut dedup = got.clone();
        dedup.dedup();
        assert_eq!(dedup.len(), 9);
        let c: Vec<String> = [
            EffectCertainty::NoEffect,
            EffectCertainty::Uncertain,
            EffectCertainty::EffectOccurred,
        ]
        .iter()
        .map(|s| {
            serde_json::to_value(s)
                .unwrap()
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
        assert_eq!(c, ["no_effect", "uncertain", "effect_occurred"]);
    }

    #[test]
    fn fan_out_reconstructs_with_valid_parents() {
        let sink = Arc::new(MemorySink::new());
        let ctx = TraceContext::new(TraceId::derive(7, 8, 1), sink.clone());
        let root = ctx.root(
            EventKind::SwarmLaunched,
            EventStatus::Ok,
            CorrelationIds {
                swarm_id: Some(1),
                ..Default::default()
            },
            EvidenceRefs::default(),
            None,
        );
        let mut tool_events = Vec::new();
        for b in 0..3u64 {
            let branch = ctx.child_of(
                root,
                EventKind::BranchForked,
                EventStatus::Ok,
                CorrelationIds {
                    sequence_id: Some(b),
                    ..Default::default()
                },
                EvidenceRefs::default(),
                None,
            );
            let t = ctx.child(
                EventKind::ToolRequested,
                EventStatus::Ok,
                CorrelationIds::default(),
                EvidenceRefs::default(),
                None,
            );
            let d = ctx.child(
                EventKind::AuthorityDecided,
                EventStatus::ApprovalPending,
                CorrelationIds::default(),
                EvidenceRefs::default(),
                None,
            );
            tool_events.push((branch, t, d));
        }
        let events = sink.drain();
        assert_eq!(events.len(), 10);
        let tree = reconstruct(&events).unwrap();
        assert_eq!(tree.roots(), &[root]);
        assert_eq!(tree.children(root).len(), 3);
        for (branch, t, d) in tool_events {
            assert_eq!(tree.get(t).unwrap().parent_event_id, Some(branch));
            assert_eq!(tree.get(d).unwrap().parent_event_id, Some(t));
            assert_eq!(tree.children(branch), &[t]);
        }
        let order: Vec<u64> = tree.chronological().iter().map(|e| e.event_id).collect();
        let mut sorted = order.clone();
        sorted.sort();
        assert_eq!(order, sorted);
    }

    #[test]
    fn reconstruct_rejects_missing_parent_and_mixed_traces() {
        let t = TraceId::derive(1, 1, 1);
        let evs = vec![ev(t, 1, None), ev(t, 2, Some(9))];
        assert_eq!(
            reconstruct(&evs).unwrap_err(),
            TraceError::MissingParent {
                event_id: 2,
                parent: 9
            }
        );
        let evs = vec![ev(t, 1, None), ev(TraceId::derive(2, 2, 2), 2, Some(1))];
        assert_eq!(reconstruct(&evs).unwrap_err(), TraceError::MixedTraceIds);
        assert_eq!(reconstruct(&[]).unwrap_err(), TraceError::Empty);
    }

    #[test]
    fn jsonl_round_trip() {
        let path = tmp("rt");
        let sink = Arc::new(BoundedJsonlSink::new(&path));
        let ctx = TraceContext::new(TraceId::derive(3, 3, 3), sink.clone());
        let r = ctx.root(
            EventKind::SwarmLaunched,
            EventStatus::Ok,
            CorrelationIds::default(),
            EvidenceRefs::default(),
            None,
        );
        ctx.child(
            EventKind::EffectExecuted,
            EventStatus::Uncertain,
            CorrelationIds {
                tool_request_id: Some(BoundedId::new("req-1")),
                ..Default::default()
            },
            EvidenceRefs::default()
                .with_intent_digest("ABCDEF01")
                .unwrap(),
            Some(BoundedNote::new("may have reached server")),
        );
        sink.flush().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let events: Vec<TraceEvent> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        assert_eq!(events.len(), 2);
        assert_eq!(events[0].event_id, r);
        assert_eq!(
            events[1].refs.intent_digest.as_ref().unwrap().as_str(),
            "abcdef01"
        );
        assert_eq!(events[1].status, EventStatus::Uncertain);
        assert!(reconstruct(&events).is_ok());
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = std::fs::remove_file(&path);
    }

    #[cfg(unix)]
    #[test]
    fn symlink_path_is_refused_and_mode_is_reset_on_every_flush() {
        use std::os::unix::fs::PermissionsExt;
        // A link planted at the export path: nothing is written through it,
        // the batch is counted as dropped, the target stays empty.
        let target = tmp("symlink-target");
        std::fs::write(&target, b"").unwrap();
        let link = tmp("symlink");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let sink = BoundedJsonlSink::new(&link);
        let t = TraceId::derive(1, 2, 3);
        sink.emit(TraceEvent {
            trace_id: t,
            event_id: sink.next_event_id(),
            parent_event_id: None,
            at_unix_ms: 1,
            kind: EventKind::SwarmLaunched,
            status: EventStatus::Ok,
            effect_certainty: None,
            ids: CorrelationIds::default(),
            refs: EvidenceRefs::default(),
            note: None,
        });
        assert!(sink.flush().is_err());
        assert_eq!(std::fs::read(&target).unwrap().len(), 0);
        assert_eq!(sink.dropped_total(), 1);
        let _ = std::fs::remove_file(&link);
        let _ = std::fs::remove_file(&target);

        // A pre-existing file with a wide mode is narrowed to 0o600 on flush.
        let path = tmp("mode");
        std::fs::write(&path, b"").unwrap();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o644)).unwrap();
        let sink = BoundedJsonlSink::new(&path);
        sink.emit(TraceEvent {
            trace_id: t,
            event_id: sink.next_event_id(),
            parent_event_id: None,
            at_unix_ms: 1,
            kind: EventKind::SwarmLaunched,
            status: EventStatus::Ok,
            effect_certainty: None,
            ids: CorrelationIds::default(),
            refs: EvidenceRefs::default(),
            note: None,
        });
        sink.flush().unwrap();
        assert_eq!(
            std::fs::metadata(&path).unwrap().permissions().mode() & 0o777,
            0o600
        );
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn overflow_counts_drops_and_writes_one_marker() {
        let path = tmp("drop");
        let sink = BoundedJsonlSink::with_capacity(&path, 4);
        let t = TraceId::derive(4, 4, 4);
        for i in 0..10u64 {
            sink.emit(ev(
                t,
                sink.next_event_id(),
                if i == 0 { None } else { Some(1) },
            ));
        }
        assert_eq!(sink.dropped_total(), 6);
        sink.flush().unwrap();
        let text = std::fs::read_to_string(&path).unwrap();
        let events: Vec<TraceEvent> = text
            .lines()
            .map(|l| serde_json::from_str(l).unwrap())
            .collect();
        let drops: Vec<_> = events
            .iter()
            .filter(|e| e.kind == EventKind::TraceDropped)
            .collect();
        assert_eq!(drops.len(), 1);
        assert_eq!(drops[0].note.as_ref().unwrap().as_str(), "dropped 6 events");
        assert_eq!(events.len(), 5);
        // Counter reset: a second flush adds no marker.
        sink.flush().unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 5);
        assert_eq!(sink.dropped_total(), 6);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn flusher_writes_on_stop() {
        let path = tmp("flusher");
        let sink = Arc::new(BoundedJsonlSink::new(&path));
        let h = sink.spawn_flusher(Duration::from_secs(3600));
        sink.emit(ev(TraceId::derive(5, 5, 5), sink.next_event_id(), None));
        h.stop();
        assert_eq!(std::fs::read_to_string(&path).unwrap().lines().count(), 1);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn note_strips_and_truncates_on_char_boundary() {
        let n = BoundedNote::new("a<b>c\u{0}d\ne\u{7f}f");
        assert_eq!(n.as_str(), "abcdef");
        // 63 ASCII bytes then a 2-byte char: the cut must fall before it.
        let s = format!("{}\u{e9}z", "x".repeat(63));
        let n = BoundedNote::new(&s);
        assert_eq!(n.as_str().len(), 63);
        assert!(n.as_str().chars().all(|c| c == 'x'));
        let s = format!("{}\u{e9}", "x".repeat(62));
        assert_eq!(BoundedNote::new(&s).as_str().len(), 64);
        // Deserialization re-applies the rule.
        let back: BoundedNote = serde_json::from_str("\"<script>\"").unwrap();
        assert_eq!(back.as_str(), "script");
    }

    #[test]
    fn refs_refuse_absolute_and_parent_paths() {
        let r = EvidenceRefs::default();
        assert!(r.clone().with_provenance_record("/etc/passwd").is_err());
        assert!(r.clone().with_provenance_record("a/../b").is_err());
        assert!(r.clone().with_provenance_record("..").is_err());
        assert!(r.clone().with_provenance_record("C:/x").is_err());
        assert!(r
            .clone()
            .with_provenance_record("ledger/rec-4.json")
            .is_ok());
        assert!(r.clone().with_result_digest("not hex").is_err());
        let bad = r#"{"provenance_record":"/abs"}"#;
        assert!(serde_json::from_str::<EvidenceRefs>(bad).is_err());
    }

    #[test]
    fn ids_are_bounded() {
        assert_eq!(BoundedId::new(&"a".repeat(100)).as_str().len(), 64);
        assert_eq!(BoundedId::new("a b<c>/d").as_str(), "abcd");
        let ids: CorrelationIds =
            serde_json::from_str(&format!("{{\"grant_id\":\"{}\"}}", "9".repeat(200))).unwrap();
        assert_eq!(ids.grant_id.unwrap().as_str().len(), 64);
    }

    fn keys(v: &Value, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, v) in m {
                    out.push(k.clone());
                    keys(v, out);
                }
            }
            Value::Array(a) => a.iter().for_each(|v| keys(v, out)),
            _ => {}
        }
    }

    #[test]
    fn serialized_event_has_no_content_field_names() {
        let e = TraceEvent {
            ids: CorrelationIds {
                swarm_id: Some(1),
                tool_request_id: Some(BoundedId::new("t")),
                interplane_trace_id: Some(TraceId::derive(1, 2, 3)),
                interplane_message_id: Some(BoundedId::new("m")),
                interplane_request_id: Some(BoundedId::new("r")),
                ..Default::default()
            },
            refs: EvidenceRefs::default()
                .with_provenance_record("p/q")
                .unwrap(),
            note: Some(BoundedNote::new("n")),
            effect_certainty: Some(EffectCertainty::NoEffect),
            ..ev(TraceId::derive(9, 9, 9), 1, None)
        };
        let v = serde_json::to_value(&e).unwrap();
        let mut ks = Vec::new();
        keys(&v, &mut ks);
        assert!(ks.len() > 10);
        for k in ks {
            for bad in ["prompt", "arguments", "content"] {
                assert!(!k.contains(bad), "field name {k} contains {bad}");
            }
        }
        let back: TraceEvent = serde_json::from_value(v).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn null_sink_is_default_and_silent() {
        let s = NullSink;
        s.emit(ev(TraceId::derive(0, 0, 0), 1, None));
        assert_eq!(s.next_event_id(), 0);
    }
}
