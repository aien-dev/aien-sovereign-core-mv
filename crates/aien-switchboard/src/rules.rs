//! The rule table: one plain TOML file the operator edits.
//!
//! ```toml
//! [settings]
//! ceiling_percent = 90        # an account at or over this is treated as full
//! default_backoff_secs = 1800 # cooldown when a limit hit gives no reset time
//!
//! [[accounts]]
//! id = "anthropic-1"
//! provider = "anthropic"      # anthropic openai google xai openrouter local
//! program = "claude"
//! settings_dir = "/home/me/.switchboard/anthropic-1"
//!
//! [[accounts]]
//! id = "openrouter-1"
//! provider = "openrouter"
//! program = "http"
//! key_ref = "openrouter-key"  # a NAME in the vault, never the key itself
//!
//! [[accounts]]
//! id = "local"
//! provider = "local"
//! program = "aien"
//!
//! [routes]
//! code = ["anthropic-1", "openrouter-1"]
//! default = ["anthropic-1"]
//! ```
//!
//! `local` is always the last fallback of every route so AIEN works offline.
//! If the file lists it earlier it is moved to the end; if the file omits it
//! it is added.

use crate::account::{Account, AccountId, Credential, Provider};
use crate::job::JobKind;
use serde::Deserialize;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RulesError {
    Read(String),
    /// Syntax or shape error, with the parser's own line and column text.
    Parse(String),
    /// The file parsed but breaks a rule; the text says which and how to fix.
    Invalid(String),
}

impl fmt::Display for RulesError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            RulesError::Read(m) => write!(f, "cannot read rule file: {m}"),
            RulesError::Parse(m) => write!(f, "rule file is not valid: {m}"),
            RulesError::Invalid(m) => write!(f, "rule file has a mistake: {m}"),
        }
    }
}

impl std::error::Error for RulesError {}

fn bad<T>(m: impl Into<String>) -> Result<T, RulesError> {
    Err(RulesError::Invalid(m.into()))
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawFile {
    #[serde(default)]
    settings: RawSettings,
    #[serde(default)]
    accounts: Vec<RawAccount>,
    routes: RawRoutes,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawSettings {
    #[serde(default = "default_ceiling")]
    ceiling_percent: f64,
    #[serde(default = "default_backoff")]
    default_backoff_secs: u64,
}

fn default_ceiling() -> f64 {
    90.0
}
fn default_backoff() -> u64 {
    1800
}

impl Default for RawSettings {
    fn default() -> Self {
        RawSettings {
            ceiling_percent: default_ceiling(),
            default_backoff_secs: default_backoff(),
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAccount {
    id: String,
    provider: Provider,
    program: String,
    settings_dir: Option<PathBuf>,
    key_ref: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RawRoutes {
    code: Option<Vec<String>>,
    review: Option<Vec<String>>,
    research: Option<Vec<String>>,
    default: Vec<String>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct RuleTable {
    accounts: Vec<Account>,
    ceiling_percent: f64,
    default_backoff_secs: u64,
    routes: BTreeMap<JobKind, Vec<AccountId>>,
    default_route: Vec<AccountId>,
    local: AccountId,
}

fn valid_name(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 64
        && s.chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.')
}

impl RuleTable {
    pub fn from_path(path: &Path) -> Result<Self, RulesError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| RulesError::Read(format!("{}: {e}", path.display())))?;
        Self::from_toml_str(&text)
    }

    pub fn from_toml_str(text: &str) -> Result<Self, RulesError> {
        let raw: RawFile = toml::from_str(text).map_err(|e| RulesError::Parse(e.to_string()))?;

        let c = raw.settings.ceiling_percent;
        if !(c > 0.0 && c <= 100.0) {
            return bad(format!(
                "ceiling_percent must be above 0 and at most 100, got {c}"
            ));
        }

        let mut accounts = Vec::new();
        let mut seen = BTreeSet::new();
        let mut local: Option<AccountId> = None;
        for a in raw.accounts {
            if !valid_name(&a.id) {
                return bad(format!(
                    "account id {:?} must be 1 to 64 letters, digits, '-', '_' or '.'",
                    a.id
                ));
            }
            if !seen.insert(a.id.clone()) {
                return bad(format!("account id {:?} is listed twice", a.id));
            }
            if a.program.trim().is_empty() {
                return bad(format!("account {:?} has an empty program name", a.id));
            }
            let credential = match (a.settings_dir, a.key_ref) {
                (Some(_), Some(_)) => {
                    return bad(format!(
                        "account {:?} sets both settings_dir and key_ref; pick one",
                        a.id
                    ))
                }
                (Some(p), None) => Credential::SettingsFolder(p),
                (None, Some(k)) => {
                    if !valid_name(&k) {
                        return bad(format!(
                            "account {:?}: key_ref must be a key NAME (letters, digits, '-', '_', '.'), never the key itself",
                            a.id
                        ));
                    }
                    Credential::KeyRef(k)
                }
                (None, None) => Credential::None,
            };
            if a.provider == Provider::Local {
                if credential != Credential::None {
                    return bad(format!(
                        "account {:?} is local and must not set settings_dir or key_ref",
                        a.id
                    ));
                }
                if local.is_some() {
                    return bad("more than one local account; exactly one is allowed");
                }
                local = Some(AccountId::new(a.id.clone()));
            }
            accounts.push(Account {
                id: AccountId::new(a.id),
                provider: a.provider,
                program: a.program,
                credential,
            });
        }
        let local = match local {
            Some(l) => l,
            None => {
                return bad(
                    "no account with provider = \"local\"; AIEN must always have the local model as the last fallback",
                )
            }
        };

        let resolve = |label: &str, ids: Vec<String>| -> Result<Vec<AccountId>, RulesError> {
            if ids.is_empty() {
                return bad(format!("route {label:?} is an empty list"));
            }
            let mut out: Vec<AccountId> = Vec::new();
            for id in ids {
                if !seen.contains(&id) {
                    return bad(format!(
                        "route {label:?} names account {id:?}, which is not in [[accounts]]"
                    ));
                }
                let id = AccountId::new(id);
                if out.contains(&id) {
                    return bad(format!("route {label:?} lists {id} twice"));
                }
                out.push(id);
            }
            // Local is always last, exactly once.
            out.retain(|i| *i != local);
            out.push(local.clone());
            Ok(out)
        };

        let default_route = resolve("default", raw.routes.default)?;
        let mut routes = BTreeMap::new();
        for (kind, list) in [
            (JobKind::Code, raw.routes.code),
            (JobKind::Review, raw.routes.review),
            (JobKind::Research, raw.routes.research),
        ] {
            if let Some(ids) = list {
                routes.insert(kind, resolve(kind.as_str(), ids)?);
            }
        }

        Ok(RuleTable {
            accounts,
            ceiling_percent: c,
            default_backoff_secs: raw.settings.default_backoff_secs,
            routes,
            default_route,
            local,
        })
    }

    pub fn ceiling_percent(&self) -> f64 {
        self.ceiling_percent
    }

    pub fn default_backoff_secs(&self) -> u64 {
        self.default_backoff_secs
    }

    /// STUB (red step).
    pub fn pass_env(&self) -> &[String] {
        &[]
    }

    pub fn local(&self) -> &AccountId {
        &self.local
    }

    pub fn accounts(&self) -> &[Account] {
        &self.accounts
    }

    pub fn account(&self, id: &AccountId) -> Option<&Account> {
        self.accounts.iter().find(|a| a.id == *id)
    }

    /// Ordered account ids for this kind of job; the last one is always local.
    pub fn route_for(&self, kind: JobKind) -> &[AccountId] {
        self.routes.get(&kind).unwrap_or(&self.default_route)
    }
}
