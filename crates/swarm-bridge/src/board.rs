//! The board as the bridge sees it: a small trait over the synapse#83 swarm
//! board, so the delivery logic can be tested against an in-memory fake and
//! wired to `synapse-coord`'s real board without changing.
//!
//! Every value here is untrusted coordination data. Nothing in this module
//! grants approval, file ownership, evidence status or authority.

use std::fmt;
use std::time::Duration;

/// Event kinds of protocol `aien-board/1` (PLAN D4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    Finding,
    Question,
    Answer,
    Blocked,
    Ready,
    Handoff,
    ReceiptRef,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Finding => "finding",
            Kind::Question => "question",
            Kind::Answer => "answer",
            Kind::Blocked => "blocked",
            Kind::Ready => "ready",
            Kind::Handoff => "handoff",
            Kind::ReceiptRef => "receipt-ref",
        }
    }

    pub fn parse(s: &str) -> Option<Kind> {
        Some(match s {
            "finding" => Kind::Finding,
            "question" => Kind::Question,
            "answer" => Kind::Answer,
            "blocked" => Kind::Blocked,
            "ready" => Kind::Ready,
            "handoff" => Kind::Handoff,
            "receipt-ref" => Kind::ReceiptRef,
            _ => return None,
        })
    }

    /// Kinds that are alerts: they always lead the subscribed section.
    pub fn is_alert(self) -> bool {
        matches!(self, Kind::Blocked | Kind::Handoff)
    }
}

/// Reference kinds (PLAN D4). A reference is a pointer, never copied content.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RefKind {
    Issue,
    Commit,
    Receipt,
    Pr,
    Path,
}

impl RefKind {
    pub fn as_str(self) -> &'static str {
        match self {
            RefKind::Issue => "issue",
            RefKind::Commit => "commit",
            RefKind::Receipt => "receipt",
            RefKind::Pr => "pr",
            RefKind::Path => "path",
        }
    }

    pub fn parse(s: &str) -> Option<RefKind> {
        Some(match s {
            "issue" => RefKind::Issue,
            "commit" => RefKind::Commit,
            "receipt" => RefKind::Receipt,
            "pr" => RefKind::Pr,
            "path" => RefKind::Path,
            _ => return None,
        })
    }

    /// References that claim a result (a commit, receipt or PR). The bridge
    /// marks these unverified; only the harness checking the receipt can
    /// promote them.
    pub fn claims_result(self) -> bool {
        matches!(self, RefKind::Commit | RefKind::Receipt | RefKind::Pr)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Ref {
    pub repo: String,
    pub kind: RefKind,
    pub value: String,
}

/// One committed board event.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Event {
    pub id: u64,
    /// Sender's addressing tag (first identity field, not authentication).
    pub from_tag: String,
    /// Sender's session key.
    pub from: String,
    pub thread: String,
    pub topic: String,
    /// Addressee tags; empty means not addressed to anyone in particular.
    pub to: Vec<String>,
    pub kind: Kind,
    pub parent: Option<u64>,
    pub ts: u64,
    pub text: String,
    pub refs: Vec<Ref>,
}

/// A post request. One call is one durable write on the board.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Post {
    pub from_tag: String,
    pub from: String,
    pub thread: String,
    pub topic: String,
    pub to: Vec<String>,
    pub kind: Kind,
    pub parent: Option<u64>,
    pub text: String,
    pub refs: Vec<Ref>,
    pub idem: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Sub {
    Thread(String),
    Topic(String),
}

impl Sub {
    pub fn describe(&self) -> String {
        match self {
            Sub::Thread(t) => format!("thread:{t}"),
            Sub::Topic(t) => format!("topic:{t}"),
        }
    }

    /// Parse `thread:<name>` or `topic:<name>`.
    pub fn parse(s: &str) -> Option<Sub> {
        let (k, v) = s.split_once(':')?;
        let v = v.trim();
        if v.is_empty() {
            return None;
        }
        match k.trim() {
            "thread" => Some(Sub::Thread(v.to_string())),
            "topic" => Some(Sub::Topic(v.to_string())),
            _ => None,
        }
    }
}

/// A bounded page from `read_since`: events addressed to the reader first,
/// then subscribed events, each part in id order. Every matching id at or
/// below `through` is in this page (or was at or below `after`); `more` is
/// true when matching events may exist past `through`.
#[derive(Debug, Clone, Default)]
pub struct Page {
    pub events: Vec<Event>,
    pub through: u64,
    pub more: bool,
}

/// Named failures. The hook never turns any of these into an agent-visible
/// failure; the CLI prints them and exits non-zero.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BridgeError {
    /// No board root configured (`SWARM_BRIDGE_BOARD_ROOT` / `SYNAPSE_BOARD_ROOT`).
    NotConfigured,
    /// The hook stdin was not the JSON Claude Code sends.
    BadHookInput(String),
    /// No session identity could be derived.
    NoSession,
    /// Another process of the same session holds its cursor lock.
    CursorBusy,
    /// The tag is bound to another session.
    TagTaken(String),
    /// Tag, thread, topic or session string outside the allowed alphabet.
    BadName(String),
    BadArgs(String),
    /// `answer` without `--parent`.
    AnswerNeedsParent,
    /// The local bridge state file is unreadable or not ours.
    StateCorrupt(String),
    Io(String),
    /// A refusal from the board itself (named by the board).
    Board(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::NotConfigured => write!(
                f,
                "NotConfigured: set SWARM_BRIDGE_BOARD_ROOT (or SYNAPSE_BOARD_ROOT) or pass --board-root"
            ),
            BridgeError::BadHookInput(m) => write!(f, "BadHookInput: {m}"),
            BridgeError::NoSession => write!(
                f,
                "NoSession: no session id (hook stdin session_id, --as, SWARM_BRIDGE_SESSION or CLAUDE_CODE_SESSION_ID)"
            ),
            BridgeError::CursorBusy => {
                write!(f, "CursorBusy: another process of this session holds the cursor lock")
            }
            BridgeError::TagTaken(t) => write!(f, "TagTaken: {t} is bound to another session"),
            BridgeError::BadName(m) => write!(f, "BadName: {m}"),
            BridgeError::BadArgs(m) => write!(f, "BadArgs: {m}"),
            BridgeError::AnswerNeedsParent => {
                write!(f, "AnswerNeedsParent: answer requires --parent <event id>")
            }
            BridgeError::StateCorrupt(m) => write!(f, "StateCorrupt: {m}"),
            BridgeError::Io(m) => write!(f, "Io: {m}"),
            BridgeError::Board(m) => write!(f, "Board: {m}"),
        }
    }
}

impl std::error::Error for BridgeError {}

/// The swarm board operations the bridge needs (synapse#83 D3-D6).
pub trait Board {
    /// Board identity (hex of `BOARD_ID`).
    fn mission(&self) -> String;
    /// Bind `tag` to `session`. Idempotent for the same pair; another
    /// session's tag is refused with [`BridgeError::TagTaken`].
    fn register(&self, session: &str, tag: &str) -> Result<(), BridgeError>;
    /// The live tag `session` already holds, if any.
    fn tag_of(&self, session: &str) -> Result<Option<String>, BridgeError>;
    /// Append one event; returns its id. Same idempotency key and same body
    /// returns the existing id.
    fn post(&self, post: &Post) -> Result<u64, BridgeError>;
    fn subscribe(&self, session: &str, sub: &Sub) -> Result<(), BridgeError>;
    fn unsubscribe(&self, session: &str, sub: &Sub) -> Result<(), BridgeError>;
    fn subscriptions(&self, session: &str) -> Result<Vec<Sub>, BridgeError>;
    /// Events after `after` that match the session's subscriptions or are
    /// addressed to its tag: addressed first, then subscribed, at most `limit`.
    fn read_since(&self, session: &str, after: u64, limit: usize) -> Result<Page, BridgeError>;
    /// One event by id, if it is visible to `session`.
    fn get(&self, session: &str, id: u64) -> Result<Option<Event>, BridgeError>;
    fn acked(&self, session: &str) -> Result<u64, BridgeError>;
    /// Move the acknowledged cursor forward. Ids at or below the current
    /// cursor are a no-op (idempotent); returns the cursor after the call.
    fn ack(&self, session: &str, id: u64) -> Result<u64, BridgeError>;
    /// Block until something new for `session` may exist or `timeout`
    /// passes. Returns true at once when an event after `after` that is not
    /// in `seen` is already visible (race-free: the board notifier is armed
    /// before that check), true when notified, false on timeout. A lost
    /// notification costs only the timeout.
    fn wait(
        &self,
        session: &str,
        after: u64,
        seen: &[u64],
        timeout: Duration,
    ) -> Result<bool, BridgeError>;
}
