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

//! The revision store `<home>.allen-profile/`: immutable files
//! `r00000000000000000001.json`, `r...2.json`, ... never edited in place.
//!
//! Write order for revision N: create a private temp file, write, fsync,
//! `hard_link(temp, rN)` (fails if rN already exists: that is a lost race,
//! reported as a stale update), fsync the directory, remove the temp file.
//! A crash at any step leaves the old head or the new head, never a mix; a
//! leftover temp file (`.tmp-*`) is ignored. The head is the highest N whose
//! chain 1..N is intact (each `prev_sha256` is the sha256 of the previous
//! file's bytes). Anything else in the directory (a gap, a damaged or
//! oversize file, an unknown schema or field, a foreign file, another
//! identity's revision) is a refusal: the store never silently replaces or
//! skips it.
use crate::refusal::ProfileRefusal as R;
use crate::schema::{
    apply_changes, clean_note, now_utc, Author, Changes, HistoryEntry, Identity, Persona, Profile,
    WorkingPref, GENESIS_PREV, MAX_FILE_BYTES, PROFILE_SCHEMA,
};
use sha2::{Digest, Sha256};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// `<home>.allen-profile`, beside `<home>.allen-binding`.
pub fn profile_dir(home: &Path) -> PathBuf {
    let mut n = home.as_os_str().to_owned();
    n.push(".allen-profile");
    PathBuf::from(n)
}

/// The steps of one revision write, in order. A step is named after it completes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WriteStep {
    TempCreated,
    TempWritten,
    TempSynced,
    Linked,
    DirSynced,
    TempRemoved,
}

impl WriteStep {
    pub const ALL: [WriteStep; 6] = [
        WriteStep::TempCreated,
        WriteStep::TempWritten,
        WriteStep::TempSynced,
        WriteStep::Linked,
        WriteStep::DirSynced,
        WriteStep::TempRemoved,
    ];
}

#[cfg(feature = "fault-injection")]
type Hook = std::sync::Arc<dyn Fn(WriteStep) -> bool + Send + Sync>;

pub struct Store {
    dir: PathBuf,
    id: Identity,
    #[cfg(feature = "fault-injection")]
    hook: Option<Hook>,
}

struct Rev {
    profile: Profile,
    sha: String,
}

fn sha_hex(b: &[u8]) -> String {
    aien_allen::hex(&Sha256::digest(b))
}

fn file_name(n: u64) -> String {
    format!("r{n:020}.json")
}

fn parse_name(s: &str) -> Option<u64> {
    let d = s.strip_prefix('r')?.strip_suffix(".json")?;
    if d.len() != 20 || !d.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    d.parse().ok().filter(|n| *n >= 1)
}

fn io(what: &str, e: std::io::Error) -> R {
    R::Io(format!("{what}: {e}"))
}

impl Store {
    /// The store beside `home` for the engaged identity `id`.
    pub fn new(home: &Path, id: Identity) -> Store {
        Store::at(profile_dir(home), id)
    }

    pub fn at(dir: PathBuf, id: Identity) -> Store {
        Store {
            dir,
            id,
            #[cfg(feature = "fault-injection")]
            hook: None,
        }
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    /// Test seam: `hook(step)` runs after each write step; returning true
    /// stops the write there, as a crash would (no cleanup).
    #[cfg(feature = "fault-injection")]
    pub fn with_fault_hook(mut self, hook: Hook) -> Store {
        self.hook = Some(hook);
        self
    }

    fn after(&self, step: WriteStep) -> Result<(), R> {
        #[cfg(feature = "fault-injection")]
        if let Some(h) = &self.hook {
            if h(step) {
                return Err(R::Crashed(format!("{step:?}")));
            }
        }
        let _ = step;
        Ok(())
    }

    /// Every revision 1..=head, fully verified. Empty when no store exists.
    fn chain(&self) -> Result<Vec<Rev>, R> {
        let rd = match std::fs::read_dir(&self.dir) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io("read profile folder", e)),
        };
        let mut nums: Vec<u64> = Vec::new();
        for ent in rd {
            let ent = ent.map_err(|e| io("read profile folder", e))?;
            let name = ent.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                continue; // a crashed writer's leftover
            }
            match parse_name(&name) {
                Some(n) => nums.push(n),
                None => return Err(R::Damaged(format!("unexpected file \"{name}\""))),
            }
        }
        nums.sort_unstable();
        for (i, n) in nums.iter().enumerate() {
            if *n != i as u64 + 1 {
                return Err(R::Damaged(format!(
                    "revision files are not contiguous (found {n} where {} was expected)",
                    i + 1
                )));
            }
        }
        let mut out: Vec<Rev> = Vec::with_capacity(nums.len());
        let mut prev = GENESIS_PREV.to_string();
        for n in nums {
            let path = self.dir.join(file_name(n));
            let meta = std::fs::metadata(&path).map_err(|e| io("stat revision", e))?;
            if meta.len() > MAX_FILE_BYTES {
                return Err(R::Damaged(format!(
                    "revision {n} is {} bytes (limit {MAX_FILE_BYTES})",
                    meta.len()
                )));
            }
            let bytes = std::fs::read(&path).map_err(|e| io("read revision", e))?;
            let p: Profile = serde_json::from_slice(&bytes)
                .map_err(|e| R::Damaged(format!("revision {n}: {e}")))?;
            p.validate()
                .map_err(|e| R::Damaged(format!("revision {n}: {e}")))?;
            if !p.is_bound_to(&self.id) {
                return Err(R::ForeignProfile(format!(
                    "revision {n} names agent {}",
                    &p.agent[..8]
                )));
            }
            if p.revision != n {
                return Err(R::Damaged(format!(
                    "revision {n} says it is revision {}",
                    p.revision
                )));
            }
            if p.prev_sha256 != prev {
                return Err(R::Damaged(format!("revision {n} breaks the hash chain")));
            }
            prev = sha_hex(&bytes);
            out.push(Rev {
                profile: p,
                sha: prev.clone(),
            });
        }
        Ok(out)
    }

    /// The current head, or `None` when no profile has been saved.
    pub fn head(&self) -> Result<Option<Profile>, R> {
        Ok(self.chain()?.pop().map(|r| r.profile))
    }

    /// Current revision number (0 = none).
    pub fn revision(&self) -> Result<u64, R> {
        Ok(self.head()?.map_or(0, |p| p.revision))
    }

    pub fn get(&self, revision: u64) -> Result<Profile, R> {
        self.chain()?
            .into_iter()
            .find(|r| r.profile.revision == revision)
            .map(|r| r.profile)
            .ok_or(R::UnknownRevision(revision))
    }

    pub fn history(&self) -> Result<Vec<HistoryEntry>, R> {
        Ok(self
            .chain()?
            .into_iter()
            .map(|r| HistoryEntry {
                revision: r.profile.revision,
                written_at: r.profile.written_at,
                author: r.profile.author,
                note: r.profile.note,
                display_name: r.profile.persona.display_name,
            })
            .collect())
    }

    /// Apply `changes` on top of revision `expected`.
    pub fn set(&self, expected: u64, changes: &Changes) -> Result<Profile, R> {
        let chain = self.chain()?;
        check_expected(expected, &chain)?;
        let (base_p, base_w) = base(&chain);
        let note = match &changes.note {
            Some(n) => clean_note(n)?,
            None => String::new(),
        };
        let (p, w) = apply_changes(&base_p, &base_w, changes)?;
        self.commit(chain, Author::User, note, p, w)
    }

    /// Write a NEW revision whose content is a copy of revision `to`.
    pub fn revert(&self, expected: u64, to: u64) -> Result<Profile, R> {
        let chain = self.chain()?;
        check_expected(expected, &chain)?;
        if chain.is_empty() {
            return Err(R::NoProfile);
        }
        let src = chain
            .iter()
            .find(|r| r.profile.revision == to)
            .ok_or(R::UnknownRevision(to))?;
        let (p, w) = (
            src.profile.persona.clone(),
            src.profile.working_preferences.clone(),
        );
        self.commit(
            chain,
            Author::Revert(to),
            format!("reverted to revision {to}"),
            p,
            w,
        )
    }

    /// Write a NEW revision with the defaults. The identity is untouched.
    pub fn reset(&self, expected: u64) -> Result<Profile, R> {
        let chain = self.chain()?;
        check_expected(expected, &chain)?;
        if chain.is_empty() {
            return Err(R::NoProfile);
        }
        self.commit(
            chain,
            Author::Reset,
            "reset to defaults".into(),
            Persona::default(),
            Vec::new(),
        )
    }

    fn commit(
        &self,
        chain: Vec<Rev>,
        author: Author,
        note: String,
        persona: Persona,
        prefs: Vec<WorkingPref>,
    ) -> Result<Profile, R> {
        let (cur_p, cur_w) = base(&chain);
        if persona == cur_p && prefs == cur_w {
            return Err(R::NothingToChange);
        }
        let revision = chain.len() as u64 + 1;
        let prev_sha256 = chain
            .last()
            .map_or_else(|| GENESIS_PREV.to_string(), |r| r.sha.clone());
        let profile = Profile {
            schema: PROFILE_SCHEMA.into(),
            agent: self.id.agent_hex(),
            root: self.id.root_hex(),
            revision,
            prev_sha256,
            written_at: now_utc(),
            author,
            note,
            persona,
            working_preferences: prefs,
        };
        profile.validate().map_err(|e| match e {
            R::Damaged(w) => R::Invalid(w),
            o => o,
        })?;
        let bytes = serde_json::to_vec(&profile).map_err(|e| R::Io(format!("serialize: {e}")))?;
        if bytes.len() as u64 > MAX_FILE_BYTES {
            return Err(R::Invalid(format!(
                "the profile would be {} bytes (limit {MAX_FILE_BYTES})",
                bytes.len()
            )));
        }
        self.write_revision(revision, &bytes)?;
        Ok(profile)
    }

    fn write_revision(&self, n: u64, bytes: &[u8]) -> Result<(), R> {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let fresh = !self.dir.exists();
        std::fs::create_dir_all(&self.dir).map_err(|e| io("create profile folder", e))?;
        if fresh {
            if let Some(parent) = self.dir.parent() {
                sync_dir(parent)?;
            }
        }
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = self.dir.join(format!(
            ".tmp-{}-{}-{nanos}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let r = self.write_steps(&tmp, &self.dir.join(file_name(n)), bytes);
        match &r {
            // A simulated crash leaves things exactly as they are.
            Err(R::Crashed(_)) => {}
            // Any other failure: the temp file must not linger.
            Err(_) => {
                let _ = std::fs::remove_file(&tmp);
            }
            Ok(()) => {}
        }
        match r {
            Err(R::StaleUpdate { .. }) => {
                let current = self.revision().unwrap_or(n);
                Err(R::StaleUpdate {
                    expected: n - 1,
                    current,
                })
            }
            o => o,
        }
    }

    fn write_steps(&self, tmp: &Path, dst: &Path, bytes: &[u8]) -> Result<(), R> {
        let mut oo = std::fs::OpenOptions::new();
        oo.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            oo.mode(0o600);
        }
        let mut f = oo.open(tmp).map_err(|e| io("create temp file", e))?;
        self.after(WriteStep::TempCreated)?;
        f.write_all(bytes).map_err(|e| io("write", e))?;
        self.after(WriteStep::TempWritten)?;
        f.sync_all().map_err(|e| io("fsync", e))?;
        drop(f);
        self.after(WriteStep::TempSynced)?;
        match std::fs::hard_link(tmp, dst) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = std::fs::remove_file(tmp);
                return Err(R::StaleUpdate {
                    expected: 0,
                    current: 0,
                });
            }
            Err(e) => return Err(io("link revision", e)),
        }
        self.after(WriteStep::Linked)?;
        sync_dir(&self.dir)?;
        self.after(WriteStep::DirSynced)?;
        std::fs::remove_file(tmp).map_err(|e| io("remove temp file", e))?;
        self.after(WriteStep::TempRemoved)?;
        Ok(())
    }
}

fn sync_dir(p: &Path) -> Result<(), R> {
    std::fs::File::open(p)
        .and_then(|d| d.sync_all())
        .map_err(|e| io("fsync folder", e))
}

fn check_expected(expected: u64, chain: &[Rev]) -> Result<(), R> {
    let current = chain.len() as u64;
    if expected != current {
        return Err(R::StaleUpdate { expected, current });
    }
    Ok(())
}

fn base(chain: &[Rev]) -> (Persona, Vec<WorkingPref>) {
    match chain.last() {
        Some(r) => (
            r.profile.persona.clone(),
            r.profile.working_preferences.clone(),
        ),
        None => (Persona::default(), Vec::new()),
    }
}
