//! The store `<home>.allen-memory/`:
//!   log/      `l<20 digits>.json` immutable hash-chained records (ciphertext only)
//!   keys/     `<item>-v<version>.key`, 32 random bytes, mode 0600 (dir 0700)
//!   pending/  forget intent markers (item id or scope only)
//! Append order: temp file, fsync, `hard_link` (fails if the number exists:
//! a lost race), fsync dir, remove temp. A crash leaves the old head or the
//! new head. Anything unexpected in `log/` is a refusal, never skipped.
use crate::refusal::MemoryRefusal as R;
use crate::schema::*;
use aien_allen::{hex, Resolved};
use ring::aead::{Aad, LessSafeKey, Nonce, UnboundKey, CHACHA20_POLY1305};
use ring::rand::{SecureRandom, SystemRandom};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

/// `<home>.allen-memory`, beside `<home>.allen-binding`.
pub fn memory_dir(home: &Path) -> PathBuf {
    let mut n = home.as_os_str().to_owned();
    n.push(".allen-memory");
    PathBuf::from(n)
}

/// Named points in a write, for crash injection (feature `fault-injection`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Step {
    KeyWritten,
    TempCreated,
    TempWritten,
    TempSynced,
    Linked,
    DirSynced,
    TempRemoved,
    MarkerWritten,
    KeyZeroed,
    KeyUnlinked,
    KeysDestroyed,
    ForgetRecorded,
    MarkerRemoved,
    SupersededKeyDestroyed,
}

#[cfg(feature = "fault-injection")]
type Hook = std::sync::Arc<dyn Fn(Step) -> bool + Send + Sync>;

/// The right to read ONE scope. Built by the daemon from the request's
/// authorised context. There is no constructor from text found in a query,
/// a model reply or a task, and no "all" value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScopeGrant {
    scope: Scope,
}

impl ScopeGrant {
    pub fn new(scope: Scope) -> ScopeGrant {
        ScopeGrant { scope }
    }
    pub fn scope(&self) -> &Scope {
        &self.scope
    }
}

/// Owner-facing inspect across all scopes. Only `inspect_all` and
/// `goals_all` take it; recall and export never do. (Type-level discipline:
/// the crate cannot stop a caller from constructing it. UNVERIFIED beyond that.)
#[derive(Debug, Clone, Copy)]
pub struct InspectAll(());

/// Who is asking to change an item: a grant for one scope, or the owner.
#[derive(Clone, Copy)]
pub enum Authority<'a> {
    Scope(&'a ScopeGrant),
    Owner(&'a InspectAll),
}

impl<'a> From<&'a ScopeGrant> for Authority<'a> {
    fn from(g: &'a ScopeGrant) -> Self {
        Authority::Scope(g)
    }
}

impl<'a> From<&'a InspectAll> for Authority<'a> {
    fn from(g: &'a InspectAll) -> Self {
        Authority::Owner(g)
    }
}

impl Authority<'_> {
    fn check(&self, item: &str, scope: &Scope) -> Result<(), R> {
        match self {
            Authority::Owner(_) => Ok(()),
            Authority::Scope(g) if g.scope == *scope => Ok(()),
            Authority::Scope(_) => Err(R::ScopeMismatch(item.to_string())),
        }
    }
}

impl InspectAll {
    pub fn owner() -> InspectAll {
        InspectAll(())
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RecallLimits {
    pub max_items: usize,
    pub max_bytes: usize,
}

impl Default for RecallLimits {
    fn default() -> Self {
        RecallLimits {
            max_items: 16,
            max_bytes: 4096,
        }
    }
}

pub enum ForgetTarget {
    Item(String),
    Scope(Scope),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ItemStatus {
    Live,
    Corrected,
    Closed,
    Forgotten,
    /// Intent recorded, record not yet appended. `key_missing` says whether the key is already gone.
    ForgottenPending {
        key_missing: bool,
    },
    Unresolved(String),
}

impl ItemStatus {
    pub fn text(&self) -> String {
        match self {
            ItemStatus::Live => "live".into(),
            ItemStatus::Corrected => "corrected".into(),
            ItemStatus::Closed => "closed".into(),
            ItemStatus::Forgotten => "forgotten".into(),
            ItemStatus::ForgottenPending { key_missing: true } => {
                "forgotten (key missing, record pending)".into()
            }
            ItemStatus::ForgottenPending { key_missing: false } => {
                "forgotten (key removal pending)".into()
            }
            ItemStatus::Unresolved(w) => format!("Unresolved: {w}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ItemView {
    pub item: String,
    pub scope: String,
    pub kind: Kind,
    pub version: u32,
    pub created: String,
    pub state: String,
    /// Present only for live and closed items. Forgotten items never show content.
    pub text: Option<String>,
    /// Set on goals: "host goal record, not a subject intent".
    pub label: Option<&'static str>,
    #[serde(skip)]
    pub status: Option<ItemStatus>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Recall {
    pub scope: Scope,
    pub items: Vec<ItemView>,
    /// Items left out because of the count or byte limit.
    pub omitted: usize,
    /// Items in this scope whose key is missing (never treated as live).
    pub unresolved: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GoalView {
    pub item: String,
    pub scope: String,
    pub text: String,
    pub created: String,
    /// `open` or `closed`.
    pub state: &'static str,
    pub label: &'static str,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct GoalList {
    pub label: &'static str,
    pub goals: Vec<GoalView>,
}

struct Ver {
    version: u32,
    at: String,
    ciphertext: Vec<u8>,
    nonce: [u8; NONCE_LEN],
}

struct Item {
    scope: Scope,
    kind: Kind,
    created_seq: u64,
    versions: Vec<Ver>,
    forgotten: bool,
    closed: bool,
}

impl Item {
    fn current(&self) -> &Ver {
        self.versions.last().expect("an item has a version")
    }
}

struct Replay {
    items: BTreeMap<String, Item>,
    head_seq: u64,
    head_sha: String,
}

pub struct Memory {
    dir: PathBuf,
    agent: [u8; 32],
    root: [u8; 32],
    #[cfg(feature = "fault-injection")]
    hook: Option<Hook>,
}

fn io(what: &str, e: std::io::Error) -> R {
    R::Io(format!("{what}: {e}"))
}

fn sha_hex(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

fn log_name(n: u64) -> String {
    format!("l{n:020}.json")
}

fn parse_log_name(s: &str) -> Option<u64> {
    let d = s.strip_prefix('l')?.strip_suffix(".json")?;
    if d.len() != 20 || !d.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    d.parse().ok().filter(|n| *n >= 1)
}

fn sync_dir(p: &Path) -> Result<(), R> {
    std::fs::File::open(p)
        .and_then(|d| d.sync_all())
        .map_err(|e| io("sync dir", e))
}

fn rand_hex(n: usize) -> Result<String, R> {
    let mut b = vec![0u8; n];
    SystemRandom::new()
        .fill(&mut b)
        .map_err(|_| R::Io("random source failed".into()))?;
    Ok(hex(&b))
}

impl Memory {
    /// Open the store for the engaged identity. `None` (not engaged) is refused.
    /// Verifies the whole log, finishes any pending forget, and re-destroys keys
    /// of forgotten or superseded versions (see [`Memory::reapply_forgets`]).
    pub fn open(home: &Path, engaged: Option<&Resolved>) -> Result<Memory, R> {
        Memory::at(memory_dir(home), engaged)
    }

    pub fn at(dir: PathBuf, engaged: Option<&Resolved>) -> Result<Memory, R> {
        let r = engaged.ok_or(R::NotEngaged)?;
        let m = Memory {
            dir,
            agent: r.agent,
            root: r.root,
            #[cfg(feature = "fault-injection")]
            hook: None,
        };
        m.recover()?;
        Ok(m)
    }

    /// Reopen with a crash hook installed AFTER recovery (test seam).
    #[cfg(feature = "fault-injection")]
    pub fn with_fault_hook(mut self, hook: Hook) -> Memory {
        self.hook = Some(hook);
        self
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn after(&self, step: Step) -> Result<(), R> {
        #[cfg(feature = "fault-injection")]
        if let Some(h) = &self.hook {
            if h(step) {
                return Err(R::Crashed(format!("{step:?}")));
            }
        }
        let _ = step;
        Ok(())
    }

    fn log_dir(&self) -> PathBuf {
        self.dir.join("log")
    }
    fn keys_dir(&self) -> PathBuf {
        self.dir.join("keys")
    }
    fn pending_dir(&self) -> PathBuf {
        self.dir.join("pending")
    }
    fn key_path(&self, kid: &str) -> PathBuf {
        self.keys_dir().join(format!("{kid}.key"))
    }

    // ---------- reading the log ----------

    fn replay(&self) -> Result<Replay, R> {
        let mut rp = Replay {
            items: BTreeMap::new(),
            head_seq: 0,
            head_sha: GENESIS_PREV.to_string(),
        };
        let rd = match std::fs::read_dir(self.log_dir()) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(rp),
            Err(e) => return Err(io("read memory log", e)),
        };
        let mut nums = Vec::new();
        for ent in rd {
            let ent = ent.map_err(|e| io("read memory log", e))?;
            let name = ent.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                continue; // a crashed writer's leftover
            }
            match parse_log_name(&name) {
                Some(n) => nums.push(n),
                None => return Err(R::Damaged(format!("unexpected file \"{name}\""))),
            }
        }
        nums.sort_unstable();
        if nums.len() as u64 > MAX_RECORDS {
            return Err(R::Damaged("too many records".into()));
        }
        for (i, n) in nums.iter().enumerate() {
            if *n != i as u64 + 1 {
                return Err(R::Damaged(format!(
                    "records are not contiguous (found {n} where {} was expected)",
                    i + 1
                )));
            }
        }
        let (agent, root) = (hex(&self.agent), hex(&self.root));
        for n in nums {
            let path = self.log_dir().join(log_name(n));
            let meta = std::fs::metadata(&path).map_err(|e| io("stat record", e))?;
            if meta.len() > MAX_RECORD_BYTES {
                return Err(R::Damaged(format!("record {n} is {} bytes", meta.len())));
            }
            let bytes = std::fs::read(&path).map_err(|e| io("read record", e))?;
            let rec: Record = serde_json::from_slice(&bytes)
                .map_err(|e| R::Damaged(format!("record {n}: {e}")))?;
            rec.validate()
                .map_err(|e| R::Damaged(format!("record {n}: {e}")))?;
            if rec.agent != agent || rec.root != root {
                return Err(R::ForeignMemory(format!(
                    "record {n} names agent {}",
                    &rec.agent[..8]
                )));
            }
            if rec.seq != n {
                return Err(R::Damaged(format!("record {n} says it is {}", rec.seq)));
            }
            if rec.prev_sha256 != rp.head_sha {
                return Err(R::Damaged(format!("record {n} breaks the hash chain")));
            }
            apply(&mut rp, &rec).map_err(|e| R::Damaged(format!("record {n}: {e}")))?;
            rp.head_seq = n;
            rp.head_sha = sha_hex(&bytes);
        }
        Ok(rp)
    }

    fn markers(&self) -> Result<Vec<(PathBuf, Marker)>, R> {
        let rd = match std::fs::read_dir(self.pending_dir()) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
            Err(e) => return Err(io("read pending", e)),
        };
        let mut out = Vec::new();
        for ent in rd {
            let ent = ent.map_err(|e| io("read pending", e))?;
            let name = ent.file_name().to_string_lossy().into_owned();
            if name.starts_with(".tmp-") {
                let _ = std::fs::remove_file(ent.path());
                continue;
            }
            let bytes = std::fs::read(ent.path()).map_err(|e| io("read marker", e))?;
            if bytes.len() > 512 {
                return Err(R::Damaged(format!("marker {name} is oversize")));
            }
            let m: Marker = serde_json::from_slice(&bytes)
                .map_err(|e| R::Damaged(format!("marker {name}: {e}")))?;
            out.push((ent.path(), m));
        }
        out.sort_by(|a, b| a.0.cmp(&b.0));
        Ok(out)
    }

    fn marker_covers(markers: &[(PathBuf, Marker)], id: &str, it: &Item) -> bool {
        markers.iter().any(|(_, m)| match m {
            Marker::Item(i) => i == id,
            Marker::Scope(s) => *s == it.scope,
        })
    }

    // ---------- crypto and keys ----------

    fn aad(&self, item: &str, version: u32, scope: &Scope, kind: Kind) -> Vec<u8> {
        format!(
            "{MEMORY_SCHEMA}|{}|{}|{item}|{version}|{scope}|{}",
            hex(&self.agent),
            hex(&self.root),
            kind.as_str()
        )
        .into_bytes()
    }

    fn make_key(&self, kid: &str) -> Result<[u8; 32], R> {
        let mut key = [0u8; 32];
        SystemRandom::new()
            .fill(&mut key)
            .map_err(|_| R::Io("random source failed".into()))?;
        self.ensure_dirs()?;
        // A key file for a version that is not in the log yet can only be an
        // orphan of a crashed write (single writer): destroy it, then create.
        self.destroy_key(kid)?;
        let mut oo = std::fs::OpenOptions::new();
        oo.write(true).create_new(true);
        {
            use std::os::unix::fs::OpenOptionsExt;
            oo.mode(0o600);
        }
        let mut f = oo
            .open(self.key_path(kid))
            .map_err(|e| io("create key", e))?;
        f.write_all(&key).map_err(|e| io("write key", e))?;
        f.sync_all().map_err(|e| io("fsync key", e))?;
        drop(f);
        sync_dir(&self.keys_dir())?;
        self.after(Step::KeyWritten)?;
        Ok(key)
    }

    fn seal(&self, key: &[u8; 32], aad: &[u8], text: &str) -> Result<(String, String), R> {
        let mut nonce = [0u8; NONCE_LEN];
        SystemRandom::new()
            .fill(&mut nonce)
            .map_err(|_| R::Io("random source failed".into()))?;
        let k = LessSafeKey::new(
            UnboundKey::new(&CHACHA20_POLY1305, key).map_err(|_| R::Io("bad key".into()))?,
        );
        let mut buf = text.as_bytes().to_vec();
        k.seal_in_place_append_tag(
            Nonce::assume_unique_for_key(nonce),
            Aad::from(aad),
            &mut buf,
        )
        .map_err(|_| R::Io("seal failed".into()))?;
        Ok((hex(&buf), hex(&nonce)))
    }

    /// Err(Ok(status)) shapes are avoided: returns the plaintext or an `ItemStatus::Unresolved` reason.
    fn read_text(&self, id: &str, it: &Item, v: &Ver) -> Result<String, ItemStatus> {
        let kid = key_id(id, v.version);
        let key = match read_key(&self.key_path(&kid)) {
            Ok(k) if k.len() == 32 => k,
            Ok(_) => return Err(ItemStatus::Unresolved("key damaged".into())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Err(ItemStatus::Unresolved("key missing".into()))
            }
            Err(_) => return Err(ItemStatus::Unresolved("key unreadable".into())),
        };
        let k = LessSafeKey::new(
            UnboundKey::new(&CHACHA20_POLY1305, &key)
                .map_err(|_| ItemStatus::Unresolved("key damaged".into()))?,
        );
        let aad = self.aad(id, v.version, &it.scope, it.kind);
        let mut buf = v.ciphertext.clone();
        match k.open_in_place(
            Nonce::assume_unique_for_key(v.nonce),
            Aad::from(&aad[..]),
            &mut buf,
        ) {
            Ok(p) => String::from_utf8(p.to_vec())
                .map_err(|_| ItemStatus::Unresolved("cannot decrypt".into())),
            Err(_) => Err(ItemStatus::Unresolved("cannot decrypt".into())),
        }
    }

    /// Overwrite with zeros, fsync, unlink. Returns whether a file existed.
    fn destroy_key(&self, kid: &str) -> Result<bool, R> {
        let p = self.key_path(kid);
        let (mut f, len) = match open_regular(&p, true) {
            Ok(f) => {
                let len = f.metadata().map_err(|e| io("stat key", e))?.len() as usize;
                (f, len)
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(false),
            Err(e) => {
                return Err(R::Damaged(format!(
                    "key file {kid} is not a plain file ({e}); refusing to touch it"
                )))
            }
        };
        {
            f.write_all(&vec![0u8; len])
                .map_err(|e| io("zero key", e))?;
            f.sync_all().map_err(|e| io("fsync key", e))?;
        }
        self.after(Step::KeyZeroed)?;
        std::fs::remove_file(&p).map_err(|e| io("unlink key", e))?;
        self.after(Step::KeyUnlinked)?;
        Ok(true)
    }

    fn destroy_keys(&self, kids: &[String]) -> Result<usize, R> {
        let mut n = 0;
        for k in kids {
            if self.destroy_key(k)? {
                n += 1;
            }
        }
        if self.keys_dir().exists() {
            sync_dir(&self.keys_dir())?;
        }
        self.after(Step::KeysDestroyed)?;
        Ok(n)
    }

    fn all_kids(id: &str, it: &Item) -> Vec<String> {
        it.versions.iter().map(|v| key_id(id, v.version)).collect()
    }

    // ---------- writing ----------

    fn ensure_dirs(&self) -> Result<(), R> {
        for (sub, mode) in [("log", 0o755), ("keys", 0o700), ("pending", 0o755)] {
            let p = self.dir.join(sub);
            if p.exists() {
                continue;
            }
            std::fs::create_dir_all(&p).map_err(|e| io("create folder", e))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt;
                std::fs::set_permissions(&p, std::fs::Permissions::from_mode(mode))
                    .map_err(|e| io("set folder mode", e))?;
            }
            let _ = mode;
            sync_dir(&self.dir)?;
            if let Some(parent) = self.dir.parent() {
                sync_dir(parent)?;
            }
        }
        Ok(())
    }

    fn append(&self, rp: &Replay, op: Op) -> Result<(), R> {
        if rp.head_seq >= MAX_RECORDS {
            return Err(R::Invalid("the memory log is full".into()));
        }
        self.ensure_dirs()?;
        let rec = Record {
            schema: MEMORY_SCHEMA.into(),
            agent: hex(&self.agent),
            root: hex(&self.root),
            seq: rp.head_seq + 1,
            prev_sha256: rp.head_sha.clone(),
            at: now_utc(),
            op,
        };
        let bytes = serde_json::to_vec(&rec).map_err(|e| R::Io(format!("serialize: {e}")))?;
        if bytes.len() as u64 > MAX_RECORD_BYTES {
            return Err(R::Invalid("record too large".into()));
        }
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_nanos())
            .unwrap_or(0);
        let tmp = self.log_dir().join(format!(
            ".tmp-{}-{}-{nanos}",
            std::process::id(),
            COUNTER.fetch_add(1, Ordering::SeqCst)
        ));
        let r = self.append_steps(&tmp, &self.log_dir().join(log_name(rec.seq)), &bytes);
        match &r {
            Err(R::Crashed(_)) | Ok(()) => {}
            Err(_) => {
                let _ = std::fs::remove_file(&tmp);
            }
        }
        r
    }

    fn append_steps(&self, tmp: &Path, dst: &Path, bytes: &[u8]) -> Result<(), R> {
        let mut oo = std::fs::OpenOptions::new();
        oo.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            oo.mode(0o600);
        }
        let mut f = oo.open(tmp).map_err(|e| io("create temp", e))?;
        self.after(Step::TempCreated)?;
        f.write_all(bytes).map_err(|e| io("write", e))?;
        self.after(Step::TempWritten)?;
        f.sync_all().map_err(|e| io("fsync", e))?;
        drop(f);
        self.after(Step::TempSynced)?;
        match std::fs::hard_link(tmp, dst) {
            Ok(()) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                let _ = std::fs::remove_file(tmp);
                return Err(R::Conflict);
            }
            Err(e) => return Err(io("link record", e)),
        }
        self.after(Step::Linked)?;
        sync_dir(&self.log_dir())?;
        self.after(Step::DirSynced)?;
        std::fs::remove_file(tmp).map_err(|e| io("remove temp", e))?;
        self.after(Step::TempRemoved)?;
        Ok(())
    }

    fn write_marker(&self, m: &Marker) -> Result<PathBuf, R> {
        self.ensure_dirs()?;
        let name = format!("forget-{}.json", rand_hex(8)?);
        let (tmp, dst) = (
            self.pending_dir().join(format!(".tmp-{name}")),
            self.pending_dir().join(&name),
        );
        let mut oo = std::fs::OpenOptions::new();
        oo.write(true).create(true).truncate(true);
        {
            use std::os::unix::fs::OpenOptionsExt;
            oo.mode(0o600);
        }
        let mut f = oo.open(&tmp).map_err(|e| io("create marker", e))?;
        f.write_all(&serde_json::to_vec(m).map_err(|e| R::Io(e.to_string()))?)
            .map_err(|e| io("write marker", e))?;
        f.sync_all().map_err(|e| io("fsync marker", e))?;
        drop(f);
        std::fs::rename(&tmp, &dst).map_err(|e| io("rename marker", e))?;
        sync_dir(&self.pending_dir())?;
        self.after(Step::MarkerWritten)?;
        Ok(dst)
    }

    fn remove_marker(&self, p: &Path) -> Result<(), R> {
        std::fs::remove_file(p).map_err(|e| io("remove marker", e))?;
        sync_dir(&self.pending_dir())?;
        self.after(Step::MarkerRemoved)
    }

    // ---------- recovery ----------

    /// Destroy any key whose item has a Forget record, any key of a
    /// superseded version, and any orphan key no record refers to. Idempotent; returns how many key files it removed.
    /// Run on every open, so restoring an old copy of `keys/` cannot bring
    /// forgotten content back once the log is intact.
    pub fn reapply_forgets(&self) -> Result<usize, R> {
        let rp = self.replay()?;
        let mut keep = std::collections::BTreeSet::new();
        for (id, it) in &rp.items {
            if !it.forgotten {
                keep.insert(key_id(id, it.current().version));
            }
        }
        let rd = match std::fs::read_dir(self.keys_dir()) {
            Ok(rd) => rd,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(0),
            Err(e) => return Err(io("read keys folder", e)),
        };
        let mut kids = Vec::new();
        for ent in rd {
            let name = ent
                .map_err(|e| io("read keys folder", e))?
                .file_name()
                .to_string_lossy()
                .into_owned();
            if let Some(kid) = name.strip_suffix(".key") {
                if !keep.contains(kid) {
                    kids.push(kid.to_string());
                }
            }
        }
        kids.sort();
        self.destroy_keys(&kids)
    }

    fn check_layout(&self) -> Result<(), R> {
        check_dir(&self.dir, false)?;
        check_dir(&self.log_dir(), false)?;
        check_dir(&self.keys_dir(), true)?;
        check_dir(&self.pending_dir(), false)
    }

    fn recover(&self) -> Result<(), R> {
        self.check_layout()?;
        // Leftovers of a crashed append (single writer: none can be in flight now).
        if let Ok(rd) = std::fs::read_dir(self.log_dir()) {
            for ent in rd.flatten() {
                if ent.file_name().to_string_lossy().starts_with(".tmp-") {
                    let _ = std::fs::remove_file(ent.path());
                }
            }
        }
        // Finish interrupted forgets: destroy keys, append the record, drop the marker.
        for (path, m) in self.markers()? {
            let rp = self.replay()?;
            let mut kids = Vec::new();
            let mut need_record: Vec<(String, Scope)> = Vec::new();
            for (id, it) in &rp.items {
                let hit = match &m {
                    Marker::Item(i) => i == id,
                    Marker::Scope(s) => *s == it.scope,
                };
                if hit {
                    kids.extend(Memory::all_kids(id, it));
                    if !it.forgotten {
                        need_record.push((id.clone(), it.scope.clone()));
                    }
                }
            }
            if self.keys_dir().exists() {
                self.destroy_keys(&kids)?;
            }
            match &m {
                Marker::Scope(s) if !need_record.is_empty() => {
                    self.append(&rp, Op::ForgetScope { scope: s.clone() })?;
                }
                Marker::Item(i) if !need_record.is_empty() => {
                    self.append(
                        &rp,
                        Op::ForgetItem {
                            item: i.clone(),
                            scope: need_record[0].1.clone(),
                        },
                    )?;
                }
                _ => {}
            }
            self.remove_marker(&path)?;
        }
        self.replay()?; // verify the log even when nothing is pending
        self.reapply_forgets()?;
        Ok(())
    }

    // ---------- public operations ----------

    fn check_text(text: &str) -> Result<(), R> {
        if text.trim().is_empty() {
            return Err(R::Invalid("empty content".into()));
        }
        if text.len() > MAX_TEXT_BYTES {
            return Err(R::Invalid(format!(
                "content is {} bytes (limit {MAX_TEXT_BYTES})",
                text.len()
            )));
        }
        Ok(())
    }

    /// Store a new item; returns its id.
    pub fn put(&self, scope: Scope, kind: Kind, text: &str) -> Result<String, R> {
        Memory::check_text(text)?;
        let rp = self.replay()?;
        let item = rand_hex(16)?;
        let kid = key_id(&item, 1);
        let key = self.make_key(&kid)?;
        let (ciphertext, nonce) = self.seal(&key, &self.aad(&item, 1, &scope, kind), text)?;
        self.append(
            &rp,
            Op::Put {
                item: item.clone(),
                scope,
                kind,
                version: 1,
                ciphertext,
                nonce,
                key_id: kid,
            },
        )?;
        Ok(item)
    }

    /// Write a new version and destroy the superseded version's key.
    pub fn correct<'a>(
        &self,
        auth: impl Into<Authority<'a>>,
        item: &str,
        text: &str,
    ) -> Result<u32, R> {
        Memory::check_text(text)?;
        let rp = self.replay()?;
        let it = rp
            .items
            .get(item)
            .ok_or_else(|| R::UnknownItem(item.into()))?;
        auth.into().check(item, &it.scope)?;
        let markers = self.markers()?;
        if it.forgotten || Memory::marker_covers(&markers, item, it) {
            return Err(R::ItemForgotten(item.into()));
        }
        let old = it.current().version;
        let new = old + 1;
        let kid = key_id(item, new);
        let key = self.make_key(&kid)?;
        let (ciphertext, nonce) =
            self.seal(&key, &self.aad(item, new, &it.scope, it.kind), text)?;
        self.append(
            &rp,
            Op::Correct {
                item: item.into(),
                supersedes_version: old,
                version: new,
                ciphertext,
                nonce,
                key_id: kid,
            },
        )?;
        self.destroy_keys(&[key_id(item, old)])?;
        self.after(Step::SupersededKeyDestroyed)?;
        Ok(new)
    }

    /// Forget an item or a whole scope. Order: intent marker, destroy keys
    /// (zero, fsync, unlink, fsync dir), append Forget record, drop marker.
    /// Returns the ids forgotten.
    pub fn forget<'a>(
        &self,
        auth: impl Into<Authority<'a>>,
        target: ForgetTarget,
    ) -> Result<Vec<String>, R> {
        let rp = self.replay()?;
        let auth = auth.into();
        let (marker, ids, op) = match &target {
            ForgetTarget::Item(i) => {
                let it = rp.items.get(i).ok_or_else(|| R::UnknownItem(i.clone()))?;
                auth.check(i, &it.scope)?;
                if it.forgotten {
                    return Err(R::NothingToForget);
                }
                (
                    Marker::Item(i.clone()),
                    vec![i.clone()],
                    Op::ForgetItem {
                        item: i.clone(),
                        scope: it.scope.clone(),
                    },
                )
            }
            ForgetTarget::Scope(s) => {
                auth.check("(scope)", s)?;
                let ids: Vec<String> = rp
                    .items
                    .iter()
                    .filter(|(_, it)| it.scope == *s && !it.forgotten)
                    .map(|(i, _)| i.clone())
                    .collect();
                if ids.is_empty() {
                    return Err(R::NothingToForget);
                }
                (
                    Marker::Scope(s.clone()),
                    ids,
                    Op::ForgetScope { scope: s.clone() },
                )
            }
        };
        let kids: Vec<String> = ids
            .iter()
            .flat_map(|i| Memory::all_kids(i, &rp.items[i]))
            .collect();
        let mpath = self.write_marker(&marker)?;
        if self.keys_dir().exists() {
            self.destroy_keys(&kids)?;
        }
        self.append(&rp, op)?;
        self.after(Step::ForgetRecorded)?;
        self.remove_marker(&mpath)?;
        Ok(ids)
    }

    /// Mark a goal closed. Its content stays readable (the key is kept).
    pub fn close_goal<'a>(&self, auth: impl Into<Authority<'a>>, item: &str) -> Result<(), R> {
        let rp = self.replay()?;
        let it = rp
            .items
            .get(item)
            .ok_or_else(|| R::UnknownItem(item.into()))?;
        auth.into().check(item, &it.scope)?;
        if it.kind != Kind::Goal {
            return Err(R::NotAGoal(item.into()));
        }
        if it.forgotten {
            return Err(R::ItemForgotten(item.into()));
        }
        if it.closed {
            return Err(R::Invalid("goal is already closed".into()));
        }
        self.append(&rp, Op::GoalClose { item: item.into() })
    }

    fn views(&self, only: Option<&Scope>) -> Result<Vec<ItemView>, R> {
        let rp = self.replay()?;
        let markers = self.markers()?;
        let mut order: Vec<(&String, &Item)> = rp
            .items
            .iter()
            .filter(|(_, it)| only.is_none_or(|s| *s == it.scope))
            .collect();
        order.sort_by(|a, b| (a.1.created_seq, a.0).cmp(&(b.1.created_seq, b.0)));
        let mut out = Vec::new();
        for (id, it) in order {
            let label = (it.kind == Kind::Goal).then_some(GOAL_LABEL);
            let mk = |v: &Ver, st: ItemStatus, text: Option<String>| ItemView {
                item: id.clone(),
                scope: it.scope.to_string(),
                kind: it.kind,
                version: v.version,
                created: v.at.clone(),
                state: st.text(),
                text,
                label,
                status: Some(st),
            };
            let cur = it.current();
            if it.forgotten {
                out.push(mk(cur, ItemStatus::Forgotten, None));
                continue;
            }
            for v in &it.versions[..it.versions.len() - 1] {
                out.push(mk(v, ItemStatus::Corrected, None));
            }
            if Memory::marker_covers(&markers, id, it) {
                let gone = !self.key_path(&key_id(id, cur.version)).exists();
                out.push(mk(
                    cur,
                    ItemStatus::ForgottenPending { key_missing: gone },
                    None,
                ));
                continue;
            }
            match self.read_text(id, it, cur) {
                Ok(t) => {
                    let st = if it.closed {
                        ItemStatus::Closed
                    } else {
                        ItemStatus::Live
                    };
                    out.push(mk(cur, st, Some(t)));
                }
                Err(st) => out.push(mk(cur, st, None)),
            }
        }
        Ok(out)
    }

    /// Owner-facing: every item in one scope, with its state.
    pub fn inspect(&self, grant: &ScopeGrant) -> Result<Vec<ItemView>, R> {
        self.views(Some(&grant.scope))
    }

    /// Owner-facing: every item in every scope.
    pub fn inspect_all(&self, _all: &InspectAll) -> Result<Vec<ItemView>, R> {
        self.views(None)
    }

    /// Context assembly. Returns only items in the granted scope whose content
    /// is live (open goals included, closed goals excluded). `query`, if given,
    /// is a case-insensitive substring filter on the plaintext; it can never
    /// widen the scope. Deterministic order (creation order, then id), bounded
    /// by `limits`. A missing key is listed in `unresolved`; a failed integrity
    /// check is an error.
    pub fn recall(
        &self,
        grant: &ScopeGrant,
        query: Option<&str>,
        limits: &RecallLimits,
    ) -> Result<Recall, R> {
        let rp = self.replay()?;
        let markers = self.markers()?;
        let mut order: Vec<(&String, &Item)> = rp
            .items
            .iter()
            .filter(|(_, it)| it.scope == grant.scope && !it.forgotten && !it.closed)
            .collect();
        order.sort_by(|a, b| (a.1.created_seq, a.0).cmp(&(b.1.created_seq, b.0)));
        let q = query.map(|q| q.to_lowercase());
        let mut rc = Recall {
            scope: grant.scope.clone(),
            items: Vec::new(),
            omitted: 0,
            unresolved: Vec::new(),
        };
        let (mut bytes, mut full) = (0usize, false);
        for (id, it) in order {
            if Memory::marker_covers(&markers, id, it) {
                continue;
            }
            let cur = it.current();
            let text = match self.read_text(id, it, cur) {
                Ok(t) => t,
                Err(ItemStatus::Unresolved(w)) if w == "cannot decrypt" => {
                    return Err(R::Tampered(id.clone()))
                }
                Err(_) => {
                    rc.unresolved.push(id.clone());
                    continue;
                }
            };
            if q.as_ref().is_some_and(|q| !text.to_lowercase().contains(q)) {
                continue;
            }
            if full || rc.items.len() >= limits.max_items || bytes + text.len() > limits.max_bytes {
                full = true;
                rc.omitted += 1;
                continue;
            }
            bytes += text.len();
            rc.items.push(ItemView {
                item: id.clone(),
                scope: it.scope.to_string(),
                kind: it.kind,
                version: cur.version,
                created: cur.at.clone(),
                state: ItemStatus::Live.text(),
                text: Some(text),
                label: (it.kind == Kind::Goal).then_some(GOAL_LABEL),
                status: Some(ItemStatus::Live),
            });
        }
        Ok(rc)
    }

    /// JSON of the live (and closed) items' plaintext in one scope, for the owner.
    pub fn export(&self, grant: &ScopeGrant) -> Result<String, R> {
        let items: Vec<ItemView> = self
            .views(Some(&grant.scope))?
            .into_iter()
            .filter(|v| matches!(v.status, Some(ItemStatus::Live | ItemStatus::Closed)))
            .collect();
        serde_json::to_string_pretty(&serde_json::json!({
            "schema": "aien.allen.memory.export/1",
            "scope": grant.scope.to_string(),
            "items": items,
        }))
        .map_err(|e| R::Io(e.to_string()))
    }

    fn goal_list(&self, only: Option<&Scope>) -> Result<GoalList, R> {
        let goals = self
            .views(only)?
            .into_iter()
            .filter(|v| {
                v.kind == Kind::Goal
                    && matches!(v.status, Some(ItemStatus::Live | ItemStatus::Closed))
            })
            .map(|v| GoalView {
                item: v.item,
                scope: v.scope,
                text: v.text.unwrap_or_default(),
                created: v.created,
                state: if v.status == Some(ItemStatus::Closed) {
                    "closed"
                } else {
                    "open"
                },
                label: GOAL_LABEL,
            })
            .collect();
        Ok(GoalList {
            label: GOAL_LABEL,
            goals,
        })
    }

    /// Host standing goals in one scope. Not subject intents.
    pub fn goals(&self, grant: &ScopeGrant) -> Result<GoalList, R> {
        self.goal_list(Some(&grant.scope))
    }

    /// Host standing goals in every scope (owner-facing).
    pub fn goals_all(&self, _all: &InspectAll) -> Result<GoalList, R> {
        self.goal_list(None)
    }
}

fn apply(rp: &mut Replay, rec: &Record) -> Result<(), String> {
    let nonce_of = |s: &str| -> Result<[u8; NONCE_LEN], String> {
        unhex(s)
            .and_then(|v| v.try_into().ok())
            .ok_or_else(|| "bad nonce".to_string())
    };
    let ct_of = |s: &str| unhex(s).ok_or_else(|| "bad ciphertext".to_string());
    match &rec.op {
        Op::Put {
            item,
            scope,
            kind,
            version,
            ciphertext,
            nonce,
            ..
        } => {
            if rp.items.contains_key(item) {
                return Err("item put twice".into());
            }
            rp.items.insert(
                item.clone(),
                Item {
                    scope: scope.clone(),
                    kind: *kind,
                    created_seq: rec.seq,
                    versions: vec![Ver {
                        version: *version,
                        at: rec.at.clone(),
                        ciphertext: ct_of(ciphertext)?,
                        nonce: nonce_of(nonce)?,
                    }],
                    forgotten: false,
                    closed: false,
                },
            );
        }
        Op::Correct {
            item,
            supersedes_version,
            version,
            ciphertext,
            nonce,
            ..
        } => {
            let it = rp.items.get_mut(item).ok_or("correct of unknown item")?;
            if it.forgotten || it.current().version != *supersedes_version {
                return Err("correct does not follow the current version".into());
            }
            it.versions.push(Ver {
                version: *version,
                at: rec.at.clone(),
                ciphertext: ct_of(ciphertext)?,
                nonce: nonce_of(nonce)?,
            });
        }
        Op::ForgetItem { item, scope } => {
            let it = rp.items.get_mut(item).ok_or("forget of unknown item")?;
            if it.scope != *scope {
                return Err("forget names the wrong scope".into());
            }
            it.forgotten = true;
        }
        Op::ForgetScope { scope } => {
            for it in rp.items.values_mut().filter(|i| i.scope == *scope) {
                it.forgotten = true;
            }
        }
        Op::GoalClose { item } => {
            let it = rp.items.get_mut(item).ok_or("close of unknown item")?;
            if it.kind != Kind::Goal || it.forgotten {
                return Err("close of a non-goal or forgotten item".into());
            }
            it.closed = true;
        }
    }
    Ok(())
}

/// Open an existing regular file without following a symlink: lstat must say
/// "regular file" and the opened handle must be the same inode (no swap race).
fn open_regular(path: &Path, write: bool) -> std::io::Result<std::fs::File> {
    use std::os::unix::fs::MetadataExt;
    let bad = |w: &str| std::io::Error::new(std::io::ErrorKind::PermissionDenied, w.to_string());
    let l = std::fs::symlink_metadata(path)?;
    if !l.file_type().is_file() {
        return Err(bad("not a regular file"));
    }
    let f = std::fs::OpenOptions::new()
        .read(true)
        .write(write)
        .open(path)?;
    let m = f.metadata()?;
    if m.dev() != l.dev() || m.ino() != l.ino() {
        return Err(bad("file changed while opening"));
    }
    Ok(f)
}

fn read_key(path: &Path) -> std::io::Result<Vec<u8>> {
    use std::io::Read;
    let mut b = Vec::new();
    open_regular(path, false)?.take(65).read_to_end(&mut b)?;
    Ok(b)
}

/// A real directory (not a symlink), mode 0700 when `private`; looser modes
/// are tightened.
fn check_dir(p: &Path, private: bool) -> Result<(), R> {
    use std::os::unix::fs::PermissionsExt;
    let m = match std::fs::symlink_metadata(p) {
        Ok(m) => m,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io("stat folder", e)),
    };
    if !m.file_type().is_dir() {
        return Err(R::Damaged(format!(
            "{} is not a plain folder (symlink or file)",
            p.display()
        )));
    }
    if private && m.permissions().mode() & 0o077 != 0 {
        std::fs::set_permissions(p, std::fs::Permissions::from_mode(0o700))
            .map_err(|e| io("tighten folder mode", e))?;
    }
    Ok(())
}
