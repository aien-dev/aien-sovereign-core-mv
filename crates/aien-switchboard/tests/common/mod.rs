//! Shared fixtures: every test works inside its own temp folder that stands
//! in for HOME. No test touches the real home folder, the real `claude`, or
//! any real login.
#![allow(dead_code)]

use aien_switchboard::{Env, RuleTable};
use std::ffi::OsString;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

pub struct Fixture {
    pub tmp: tempfile::TempDir,
}

impl Fixture {
    pub fn new() -> Self {
        Fixture {
            tmp: tempfile::tempdir().unwrap(),
        }
    }
    pub fn home(&self) -> PathBuf {
        self.tmp.path().join("home")
    }
    pub fn bin(&self) -> PathBuf {
        self.tmp.path().join("bin")
    }
    /// Profile folder for an alias, inside the temp HOME.
    pub fn profile_dir(&self, alias: &str) -> PathBuf {
        self.home().join(".switchboard").join(alias)
    }
    /// A stand-in `claude` executable. Tests never run it; the fake spawner
    /// and fake probe only record what would have been started.
    pub fn install_fake_claude(&self) -> PathBuf {
        fs::create_dir_all(self.bin()).unwrap();
        let p = self.bin().join("claude");
        fs::write(&p, "#!/bin/sh\nexit 0\n").unwrap();
        fs::set_permissions(&p, fs::Permissions::from_mode(0o755)).unwrap();
        p
    }
    pub fn make_private_dir(&self, p: &Path) {
        fs::create_dir_all(p).unwrap();
        fs::set_permissions(p, fs::Permissions::from_mode(0o700)).unwrap();
    }
    /// Rule file with two Claude profiles of different providers' routes kept
    /// apart, an OpenRouter key name and the local model. `extra` is appended
    /// verbatim (for a `[launch]` table).
    pub fn rules_toml(&self, extra: &str) -> String {
        format!(
            r#"{extra}
[[accounts]]
id = "anthropic-1"
provider = "anthropic"
program = "claude"
settings_dir = "{a1}"

[[accounts]]
id = "anthropic-2"
provider = "anthropic"
program = "claude"
settings_dir = "{a2}"

[[accounts]]
id = "openrouter-1"
provider = "openrouter"
program = "http"
key_ref = "openrouter-key"

[[accounts]]
id = "local"
provider = "local"
program = "aien"

[routes]
code = ["anthropic-1", "openrouter-1"]
review = ["anthropic-2", "openrouter-1"]
default = ["local"]
"#,
            a1 = self.profile_dir("anthropic-1").display(),
            a2 = self.profile_dir("anthropic-2").display(),
        )
    }
    pub fn rules(&self, extra: &str) -> RuleTable {
        RuleTable::from_toml_str(&self.rules_toml(extra)).unwrap()
    }
    /// A minimal clean environment: PATH points only at the fake bin folder
    /// and HOME at the temp home.
    pub fn base_env(&self) -> Env {
        env(&[
            ("PATH", self.bin().to_str().unwrap()),
            ("HOME", self.home().to_str().unwrap()),
        ])
    }
}

pub fn env(pairs: &[(&str, &str)]) -> Env {
    pairs
        .iter()
        .map(|(k, v)| (OsString::from(k), OsString::from(v)))
        .collect()
}

pub fn has(env: &Env, name: &str) -> bool {
    env.contains_key(&OsString::from(name))
}
