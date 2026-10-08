//! Schema `aien.allen.memory/1`. Every struct read from disk denies unknown
//! fields. The log holds ciphertext only; plaintext never reaches `log/`.
use crate::refusal::MemoryRefusal as R;
use serde::{Deserialize, Serialize};
use std::fmt;

pub const MEMORY_SCHEMA: &str = "aien.allen.memory/1";
/// Printed with every standing goal. A host goal is not a SubjectState intent.
pub const GOAL_LABEL: &str = "host goal record, not a subject intent";
pub const MAX_TEXT_BYTES: usize = 2048;
pub const MAX_RECORD_BYTES: u64 = 8 * 1024;
pub const MAX_RECORDS: u64 = 100_000;
pub const GENESIS_PREV: &str = "0000000000000000000000000000000000000000000000000000000000000000";
pub const TAG_LEN: usize = 16;
pub const NONCE_LEN: usize = 12;

/// `personal`, `work`, or `project:<name>` with name `^[a-z0-9_.-]{1,48}$`.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(try_from = "String", into = "String")]
pub enum Scope {
    Personal,
    Work,
    Project(String),
}

impl Scope {
    pub fn parse(s: &str) -> Result<Scope, R> {
        match s {
            "personal" => Ok(Scope::Personal),
            "work" => Ok(Scope::Work),
            _ => match s.strip_prefix("project:") {
                Some(n)
                    if (1..=48).contains(&n.len())
                        && n.bytes().all(|b| {
                            b.is_ascii_lowercase()
                                || b.is_ascii_digit()
                                || b == b'_'
                                || b == b'.'
                                || b == b'-'
                        }) =>
                {
                    Ok(Scope::Project(n.to_string()))
                }
                _ => Err(R::Invalid(format!(
                    "scope \"{s}\" (use personal, work or project:<name> with name of 1 to 48 of a-z 0-9 _ . -)"
                ))),
            },
        }
    }
}

impl fmt::Display for Scope {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Scope::Personal => f.write_str("personal"),
            Scope::Work => f.write_str("work"),
            Scope::Project(n) => write!(f, "project:{n}"),
        }
    }
}

impl TryFrom<String> for Scope {
    type Error = String;
    fn try_from(s: String) -> Result<Scope, String> {
        Scope::parse(&s).map_err(|e| e.to_string())
    }
}

impl From<Scope> for String {
    fn from(s: Scope) -> String {
        s.to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    Fact,
    Preference,
    Goal,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Fact => "fact",
            Kind::Preference => "preference",
            Kind::Goal => "goal",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Record {
    pub schema: String,
    pub agent: String,
    pub root: String,
    pub seq: u64,
    pub prev_sha256: String,
    pub at: String,
    pub op: Op,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "snake_case", deny_unknown_fields)]
pub enum Op {
    Put {
        item: String,
        scope: Scope,
        kind: Kind,
        version: u32,
        ciphertext: String,
        nonce: String,
        key_id: String,
    },
    Correct {
        item: String,
        supersedes_version: u32,
        version: u32,
        ciphertext: String,
        nonce: String,
        key_id: String,
    },
    /// Metadata only: item id and scope. The content key is already destroyed.
    ForgetItem {
        item: String,
        scope: Scope,
    },
    ForgetScope {
        scope: Scope,
    },
    GoalClose {
        item: String,
    },
}

pub fn is_hex(s: &str, len: Option<usize>) -> bool {
    len.is_none_or(|l| s.len() == l)
        && !s.is_empty()
        && s.bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

pub fn unhex(s: &str) -> Option<Vec<u8>> {
    if !s.len().is_multiple_of(2) || !s.is_ascii() {
        return None;
    }
    (0..s.len() / 2)
        .map(|i| u8::from_str_radix(&s[2 * i..2 * i + 2], 16).ok())
        .collect()
}

pub fn key_id(item: &str, version: u32) -> String {
    format!("{item}-v{version}")
}

fn check_blob(item: &str, version: u32, ct: &str, nonce: &str, kid: &str) -> Result<(), String> {
    if !is_hex(item, Some(32)) {
        return Err("item id is not 32 hex characters".into());
    }
    if version == 0 {
        return Err("version 0".into());
    }
    if !is_hex(nonce, Some(NONCE_LEN * 2)) {
        return Err("bad nonce".into());
    }
    if !is_hex(ct, None)
        || !ct.len().is_multiple_of(2)
        || ct.len() < TAG_LEN * 2
        || ct.len() > (MAX_TEXT_BYTES + TAG_LEN) * 2
    {
        return Err("bad ciphertext size".into());
    }
    if kid != key_id(item, version) {
        return Err("key_id does not match item and version".into());
    }
    Ok(())
}

impl Record {
    pub fn validate(&self) -> Result<(), String> {
        if self.schema != MEMORY_SCHEMA {
            return Err(format!("unknown schema \"{}\"", self.schema));
        }
        if !is_hex(&self.agent, Some(64)) || !is_hex(&self.root, Some(64)) {
            return Err("agent or root is not 64 hex characters".into());
        }
        if !is_hex(&self.prev_sha256, Some(64)) {
            return Err("prev_sha256 is not 64 hex characters".into());
        }
        if self.at.len() != 20 || !self.at.is_ascii() || self.seq == 0 {
            return Err("bad timestamp or sequence".into());
        }
        match &self.op {
            Op::Put {
                item,
                version,
                ciphertext,
                nonce,
                key_id,
                ..
            } => {
                if *version != 1 {
                    return Err("put must be version 1".into());
                }
                check_blob(item, *version, ciphertext, nonce, key_id)
            }
            Op::Correct {
                item,
                supersedes_version,
                version,
                ciphertext,
                nonce,
                key_id,
            } => {
                if *supersedes_version == 0 || *version != supersedes_version.saturating_add(1) {
                    return Err("correct must supersede the previous version by one".into());
                }
                check_blob(item, *version, ciphertext, nonce, key_id)
            }
            Op::ForgetItem { item, .. } | Op::GoalClose { item } => {
                if is_hex(item, Some(32)) {
                    Ok(())
                } else {
                    Err("item id is not 32 hex characters".into())
                }
            }
            Op::ForgetScope { .. } => Ok(()),
        }
    }
}

/// The intent written before keys are destroyed (see RETENTION.md).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, rename_all = "snake_case")]
pub enum Marker {
    Item(String),
    Scope(Scope),
}

/// UTC `YYYY-MM-DDTHH:MM:SSZ` from the system clock (informational only).
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86400), secs.rem_euclid(86400));
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z.rem_euclid(146097);
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + i64::from(m <= 2);
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}
