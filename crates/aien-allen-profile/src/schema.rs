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

//! Schema `aien.allen.profile/1` and its validation. Every struct that is
//! read from disk or from a request denies unknown fields.
use crate::refusal::ProfileRefusal;
use serde::{Deserialize, Serialize};

pub const PROFILE_SCHEMA: &str = "aien.allen.profile/1";
pub const DEFAULT_NAME: &str = "ALLEN";
/// Largest revision file, in bytes.
pub const MAX_FILE_BYTES: u64 = 16 * 1024;
pub const MAX_NAME_CHARS: usize = 64;
pub const MAX_NOTE_CHARS: usize = 200;
pub const MAX_VALUE_CHARS: usize = 200;
pub const MAX_PREFS: usize = 32;
pub const GENESIS_PREV: &str = "0000000000000000000000000000000000000000000000000000000000000000";
const PROVENANCE: &str = "user_explicit";
/// A key containing any of these is refused (case-insensitive).
const PERMISSION_WORDS: [&str; 11] = [
    "grant",
    "perm",
    "approv",
    "capab",
    "allow",
    "authori",
    "bypass",
    "sudo",
    "admin",
    "override",
    "unrestrict",
];

/// The engaged identity a profile is bound to (the resolved agent and root).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Identity {
    pub agent: [u8; 32],
    pub root: [u8; 32],
}

impl From<&aien_allen::Resolved> for Identity {
    fn from(r: &aien_allen::Resolved) -> Self {
        Identity {
            agent: r.agent,
            root: r.root,
        }
    }
}

impl Identity {
    pub fn agent_hex(&self) -> String {
        aien_allen::hex(&self.agent)
    }
    pub fn root_hex(&self) -> String {
        aien_allen::hex(&self.root)
    }
    /// First 8 hex characters of the agent id, for display.
    pub fn fingerprint(&self) -> String {
        self.agent_hex()[..8].to_string()
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Tone {
    Neutral,
    Warm,
    Direct,
    Formal,
}

impl Tone {
    pub fn as_str(self) -> &'static str {
        match self {
            Tone::Neutral => "neutral",
            Tone::Warm => "warm",
            Tone::Direct => "direct",
            Tone::Formal => "formal",
        }
    }
    pub fn parse(s: &str) -> Result<Tone, ProfileRefusal> {
        match s {
            "neutral" => Ok(Tone::Neutral),
            "warm" => Ok(Tone::Warm),
            "direct" => Ok(Tone::Direct),
            "formal" => Ok(Tone::Formal),
            o => Err(ProfileRefusal::Invalid(format!(
                "tone \"{o}\" (use neutral, warm, direct or formal)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Verbosity {
    Brief,
    Normal,
    Detailed,
}

impl Verbosity {
    pub fn as_str(self) -> &'static str {
        match self {
            Verbosity::Brief => "brief",
            Verbosity::Normal => "normal",
            Verbosity::Detailed => "detailed",
        }
    }
    pub fn parse(s: &str) -> Result<Verbosity, ProfileRefusal> {
        match s {
            "brief" => Ok(Verbosity::Brief),
            "normal" => Ok(Verbosity::Normal),
            "detailed" => Ok(Verbosity::Detailed),
            o => Err(ProfileRefusal::Invalid(format!(
                "verbosity \"{o}\" (use brief, normal or detailed)"
            ))),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Scope {
    All,
    Personal,
    Work,
    Project,
}

impl Scope {
    pub fn as_str(self) -> &'static str {
        match self {
            Scope::All => "all",
            Scope::Personal => "personal",
            Scope::Work => "work",
            Scope::Project => "project",
        }
    }
    pub fn parse(s: &str) -> Result<Scope, ProfileRefusal> {
        match s {
            "all" => Ok(Scope::All),
            "personal" => Ok(Scope::Personal),
            "work" => Ok(Scope::Work),
            "project" => Ok(Scope::Project),
            o => Err(ProfileRefusal::Invalid(format!(
                "scope \"{o}\" (use all, personal, work or project)"
            ))),
        }
    }
}

/// Who wrote a revision: the user, a reset, or a revert to revision N.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Author {
    User,
    Reset,
    Revert(u64),
}

impl Author {
    fn to_text(self) -> String {
        match self {
            Author::User => "user".into(),
            Author::Reset => "reset".into(),
            Author::Revert(n) => format!("revert:{n}"),
        }
    }
    fn from_text(s: &str) -> Option<Author> {
        match s {
            "user" => Some(Author::User),
            "reset" => Some(Author::Reset),
            _ => s
                .strip_prefix("revert:")
                .and_then(|n| n.parse().ok())
                .filter(|n: &u64| *n >= 1)
                .map(Author::Revert),
        }
    }
}

impl Serialize for Author {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_text())
    }
}

impl<'de> Deserialize<'de> for Author {
    fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let t = String::deserialize(d)?;
        Author::from_text(&t).ok_or_else(|| serde::de::Error::custom(format!("author {t:?}")))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Persona {
    pub display_name: String,
    pub tone: Tone,
    pub verbosity: Verbosity,
    pub plain_language: bool,
}

impl Default for Persona {
    fn default() -> Self {
        Persona {
            display_name: DEFAULT_NAME.into(),
            tone: Tone::Neutral,
            verbosity: Verbosity::Normal,
            plain_language: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct WorkingPref {
    pub key: String,
    pub value: String,
    pub scope: Scope,
    pub provenance: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Profile {
    pub schema: String,
    pub agent: String,
    pub root: String,
    pub revision: u64,
    pub prev_sha256: String,
    /// UTC, informational only. Nothing orders or trusts it.
    pub written_at: String,
    pub author: Author,
    pub note: String,
    pub persona: Persona,
    pub working_preferences: Vec<WorkingPref>,
}

/// One requested preference change.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PrefChange {
    pub key: String,
    pub value: String,
    pub scope: Scope,
}

/// What `set` may change. Anything left `None`/empty is kept as it is.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Changes {
    pub name: Option<String>,
    pub tone: Option<Tone>,
    pub verbosity: Option<Verbosity>,
    pub plain_language: Option<bool>,
    pub set_prefs: Vec<PrefChange>,
    pub unset_prefs: Vec<String>,
    pub note: Option<String>,
}

/// One line of `history`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HistoryEntry {
    pub revision: u64,
    pub written_at: String,
    pub author: Author,
    pub note: String,
    pub display_name: String,
}

fn has_control(s: &str) -> bool {
    s.chars().any(|c| {
        c.is_control()
            || matches!(c, '\u{2028}' | '\u{2029}' | '\u{202A}'..='\u{202E}' | '\u{2066}'..='\u{2069}')
    })
}

pub fn is_hex64(s: &str) -> bool {
    s.len() == 64 && s.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

pub fn clean_name(raw: &str) -> Result<String, ProfileRefusal> {
    let n = raw.trim();
    let chars = n.chars().count();
    if chars == 0 || chars > MAX_NAME_CHARS {
        return Err(ProfileRefusal::Invalid(format!(
            "the name must be 1 to {MAX_NAME_CHARS} characters"
        )));
    }
    if has_control(n) {
        return Err(ProfileRefusal::Invalid(
            "the name may not contain control characters".into(),
        ));
    }
    Ok(n.to_string())
}

pub fn clean_note(raw: &str) -> Result<String, ProfileRefusal> {
    let n = raw.trim();
    if n.chars().count() > MAX_NOTE_CHARS || has_control(n) {
        return Err(ProfileRefusal::Invalid(format!(
            "a note is at most {MAX_NOTE_CHARS} characters, no control characters"
        )));
    }
    Ok(n.to_string())
}

pub fn check_key(key: &str) -> Result<(), ProfileRefusal> {
    let ok = !key.is_empty()
        && key.len() <= 48
        && key
            .bytes()
            .all(|b| matches!(b, b'a'..=b'z' | b'0'..=b'9' | b'_' | b'.' | b'-'));
    if !ok {
        return Err(ProfileRefusal::Invalid(format!(
            "preference key \"{key}\" (1 to 48 characters from a-z, 0-9, _ . -)"
        )));
    }
    let norm = key.to_ascii_lowercase().replace(['-', '.'], "_");
    if PERMISSION_WORDS.iter().any(|w| norm.contains(w)) {
        return Err(ProfileRefusal::PermissionKey(key.to_string()));
    }
    Ok(())
}

pub fn check_value(key: &str, value: &str) -> Result<(), ProfileRefusal> {
    let n = value.chars().count();
    if n == 0 || n > MAX_VALUE_CHARS || has_control(value) {
        return Err(ProfileRefusal::Invalid(format!(
            "the value for \"{key}\" must be 1 to {MAX_VALUE_CHARS} characters, one line"
        )));
    }
    Ok(())
}

impl Profile {
    /// Structural validation of a profile (as read from disk or about to be written).
    pub fn validate(&self) -> Result<(), ProfileRefusal> {
        let bad = |w: String| Err(ProfileRefusal::Damaged(w));
        if self.schema != PROFILE_SCHEMA {
            return bad(format!("unknown schema \"{}\"", self.schema));
        }
        if !is_hex64(&self.agent) || !is_hex64(&self.root) || !is_hex64(&self.prev_sha256) {
            return bad("agent, root and prev_sha256 must be 64 hex characters".into());
        }
        if self.revision == 0 {
            return bad("revision 0".into());
        }
        if self.written_at.len() > 40 || has_control(&self.written_at) {
            return bad("written_at".into());
        }
        clean_note(&self.note).map_err(|e| ProfileRefusal::Damaged(e.to_string()))?;
        clean_name(&self.persona.display_name)
            .map_err(|e| ProfileRefusal::Damaged(e.to_string()))?;
        if self.persona.display_name != self.persona.display_name.trim() {
            return bad("display_name is not trimmed".into());
        }
        if self.working_preferences.len() > MAX_PREFS {
            return bad(format!("more than {MAX_PREFS} preferences"));
        }
        let mut last: Option<&str> = None;
        for p in &self.working_preferences {
            check_key(&p.key).map_err(|e| ProfileRefusal::Damaged(e.to_string()))?;
            check_value(&p.key, &p.value).map_err(|e| ProfileRefusal::Damaged(e.to_string()))?;
            if p.provenance != PROVENANCE {
                return bad(format!("preference \"{}\" provenance", p.key));
            }
            if last.is_some_and(|l| l >= p.key.as_str()) {
                return bad("preferences are not sorted by unique key".into());
            }
            last = Some(&p.key);
        }
        Ok(())
    }

    pub fn is_bound_to(&self, id: &Identity) -> bool {
        self.agent == id.agent_hex() && self.root == id.root_hex()
    }

    /// The editable part, to compare "would anything change".
    pub fn content(&self) -> (&Persona, &[WorkingPref]) {
        (&self.persona, &self.working_preferences)
    }
}

/// Apply `ch` to a persona and preference list, validating every part.
pub fn apply_changes(
    persona: &Persona,
    prefs: &[WorkingPref],
    ch: &Changes,
) -> Result<(Persona, Vec<WorkingPref>), ProfileRefusal> {
    let mut p = persona.clone();
    if let Some(n) = &ch.name {
        p.display_name = clean_name(n)?;
    }
    if let Some(t) = ch.tone {
        p.tone = t;
    }
    if let Some(v) = ch.verbosity {
        p.verbosity = v;
    }
    if let Some(b) = ch.plain_language {
        p.plain_language = b;
    }
    let mut list: Vec<WorkingPref> = prefs.to_vec();
    for key in &ch.unset_prefs {
        let before = list.len();
        list.retain(|w| &w.key != key);
        if list.len() == before {
            return Err(ProfileRefusal::Invalid(format!(
                "no saved preference \"{key}\" to remove"
            )));
        }
    }
    for c in &ch.set_prefs {
        check_key(&c.key)?;
        check_value(&c.key, &c.value)?;
        let new = WorkingPref {
            key: c.key.clone(),
            value: c.value.clone(),
            scope: c.scope,
            provenance: PROVENANCE.into(),
        };
        match list.iter_mut().find(|w| w.key == c.key) {
            Some(slot) => *slot = new,
            None => list.push(new),
        }
    }
    if list.len() > MAX_PREFS {
        return Err(ProfileRefusal::Invalid(format!(
            "at most {MAX_PREFS} preferences"
        )));
    }
    list.sort_by(|a, b| a.key.cmp(&b.key));
    Ok((p, list))
}

pub fn preference_provenance() -> &'static str {
    PROVENANCE
}

/// UTC "YYYY-MM-DDTHH:MM:SSZ" from the system clock.
pub fn now_utc() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let (days, rem) = ((secs / 86_400) as i64, secs % 86_400);
    // Civil-from-days (Howard Hinnant's algorithm).
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        (rem % 3600) / 60,
        rem % 60
    )
}
