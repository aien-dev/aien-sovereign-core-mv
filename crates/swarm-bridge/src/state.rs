//! Per-session bridge state, kept outside the board: which events this
//! session has been shown and confirmed, the last delivered batch awaiting
//! confirmation, and the session's tag. One file per session under
//! `<state dir>/<mission>/<key>.json`, written durably, guarded by an
//! exclusive `flock` so two processes of the same session never deliver the
//! same batch twice.

use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::board::BridgeError;

/// Largest state file the bridge will read.
pub const MAX_STATE_BYTES: u64 = 256 * 1024;
/// Most out-of-order delivered ids remembered past the confirmed prefix.
pub const MAX_AHEAD: usize = 512;
/// How long a second process of the same session waits for the lock.
const LOCK_TRIES: u32 = 40;
const LOCK_PAUSE: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pending {
    /// Ids shown in the batch.
    pub ids: Vec<u64>,
    /// Confirmed prefix once this batch is confirmed.
    pub confirmed: u64,
    /// Out-of-order ids once this batch is confirmed.
    pub ahead: Vec<u64>,
    /// Marker printed with the batch; found in the transcript = delivered.
    pub batch_tag: String,
    /// True when the batch carried the "You are <tag>" banner.
    #[serde(default)]
    pub banner: bool,
    /// Transcript files to search and the byte offset each had before.
    pub transcripts: Vec<(String, u64)>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct State {
    pub v: u32,
    pub mission: String,
    pub key: String,
    pub tag: Option<String>,
    /// True once the "You are <tag>" banner has been delivered.
    pub bannered: bool,
    /// Every matching id at or below this was delivered (or was our own).
    pub confirmed: u64,
    /// Delivered ids above `confirmed` (addressed events arrive first).
    pub ahead: Vec<u64>,
    pub pending: Option<Pending>,
    /// Batches delivered so far (for evidence; informational).
    pub batches: u64,
}

/// Locked handle to one session's state file.
pub struct StateFile {
    path: PathBuf,
    _lock: File,
}

fn io(ctx: &str, e: std::io::Error) -> BridgeError {
    BridgeError::Io(format!("{ctx}: {e}"))
}

/// Default state directory: `SWARM_BRIDGE_STATE_DIR`, else
/// `$XDG_STATE_HOME/swarm-bridge`, else `~/.local/state/swarm-bridge`.
pub fn default_state_dir() -> Option<PathBuf> {
    if let Some(d) = std::env::var_os("SWARM_BRIDGE_STATE_DIR") {
        return Some(PathBuf::from(d));
    }
    if let Some(d) = std::env::var_os("XDG_STATE_HOME") {
        return Some(PathBuf::from(d).join("swarm-bridge"));
    }
    std::env::var_os("HOME").map(|h| PathBuf::from(h).join(".local/state/swarm-bridge"))
}

impl StateFile {
    /// Open and lock the state of `key` on board `mission`. Fails with
    /// [`BridgeError::CursorBusy`] when another process holds the lock past
    /// a short wait.
    pub fn open(dir: &Path, mission: &str, key: &str) -> Result<StateFile, BridgeError> {
        let d = dir.join(mission);
        fs::create_dir_all(&d).map_err(|e| io("create state dir", e))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .read(true)
            .write(true)
            .custom_flags(libc::O_NOFOLLOW)
            .open(d.join(format!("{key}.lock")))
            .map_err(|e| io("open lock", e))?;
        let mut tries = 0;
        loop {
            if lock.try_lock().is_ok() {
                break;
            }
            tries += 1;
            if tries >= LOCK_TRIES {
                return Err(BridgeError::CursorBusy);
            }
            thread::sleep(LOCK_PAUSE);
        }
        Ok(StateFile {
            path: d.join(format!("{key}.json")),
            _lock: lock,
        })
    }

    pub fn load(&self, mission: &str, key: &str) -> Result<State, BridgeError> {
        let f = match OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
            .open(&self.path)
        {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                return Ok(State {
                    v: 1,
                    mission: mission.to_string(),
                    key: key.to_string(),
                    ..State::default()
                })
            }
            Err(e) => return Err(io("open state", e)),
        };
        let md = f.metadata().map_err(|e| io("stat state", e))?;
        if !md.is_file() {
            return Err(BridgeError::StateCorrupt(
                "state is not a regular file".into(),
            ));
        }
        if md.len() > MAX_STATE_BYTES {
            return Err(BridgeError::StateCorrupt(format!(
                "state is {} bytes, cap {MAX_STATE_BYTES}",
                md.len()
            )));
        }
        let mut buf = Vec::new();
        f.take(MAX_STATE_BYTES)
            .read_to_end(&mut buf)
            .map_err(|e| io("read state", e))?;
        let st: State = serde_json::from_slice(&buf)
            .map_err(|e| BridgeError::StateCorrupt(format!("parse: {e}")))?;
        if st.mission != mission || st.key != key {
            return Err(BridgeError::StateCorrupt(
                "state belongs to another mission or session".into(),
            ));
        }
        Ok(st)
    }

    /// Write tmp, fsync, rename, fsync dir.
    pub fn save(&self, st: &State) -> Result<(), BridgeError> {
        let mut st = st.clone();
        if st.ahead.len() > MAX_AHEAD {
            let cut = st.ahead.len() - MAX_AHEAD;
            st.ahead.drain(..cut);
        }
        let body = serde_json::to_vec(&st).map_err(|e| BridgeError::Io(format!("encode: {e}")))?;
        let tmp = self
            .path
            .with_extension(format!("tmp.{}", std::process::id()));
        {
            let mut f = OpenOptions::new()
                .create(true)
                .truncate(true)
                .write(true)
                .custom_flags(libc::O_NOFOLLOW)
                .open(&tmp)
                .map_err(|e| io("create tmp state", e))?;
            f.write_all(&body).map_err(|e| io("write tmp state", e))?;
            f.sync_all().map_err(|e| io("fsync tmp state", e))?;
        }
        fs::rename(&tmp, &self.path).map_err(|e| io("rename state", e))?;
        if let Some(parent) = self.path.parent() {
            if let Ok(d) = File::open(parent) {
                d.sync_all().map_err(|e| io("fsync state dir", e))?;
            }
        }
        Ok(())
    }
}

/// Result of looking for a batch tag in a transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Scan {
    Found,
    NotFound,
    Unreadable,
}

/// Most transcript bytes scanned for one confirmation.
pub const MAX_SCAN_BYTES: u64 = 32 * 1024 * 1024;

/// Current length of a transcript, if it is a readable regular file.
pub fn transcript_len(path: &Path) -> Option<u64> {
    let md = fs::symlink_metadata(path).ok()?;
    md.is_file().then_some(md.len())
}

/// Look for `needle` in `path` from byte `offset` (bounded). Claude Code
/// records hook additional context in the session transcript before the
/// next model request, so finding the batch tag there is durable proof the
/// batch reached the session.
pub fn transcript_contains(path: &Path, offset: u64, needle: &str) -> Scan {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_NONBLOCK)
        .open(path);
    let mut f = match f {
        Ok(f) => f,
        Err(_) => return Scan::Unreadable,
    };
    let len = match f.metadata() {
        Ok(m) if m.is_file() => m.len(),
        _ => return Scan::Unreadable,
    };
    // A transcript shorter than the recorded offset was rewritten; search it all.
    let mut start = if offset > len { 0 } else { offset };
    if len - start > MAX_SCAN_BYTES {
        start = len - MAX_SCAN_BYTES;
    }
    if f.seek(SeekFrom::Start(start)).is_err() {
        return Scan::Unreadable;
    }
    let needle = needle.as_bytes();
    if needle.is_empty() {
        return Scan::Found;
    }
    let mut carry: Vec<u8> = Vec::new();
    let mut buf = vec![0u8; 1 << 20];
    loop {
        let n = match f.read(&mut buf) {
            Ok(0) => return Scan::NotFound,
            Ok(n) => n,
            Err(_) => return Scan::Unreadable,
        };
        carry.extend_from_slice(&buf[..n]);
        if carry.windows(needle.len()).any(|w| w == needle) {
            return Scan::Found;
        }
        let keep = needle.len().saturating_sub(1).min(carry.len());
        let cut = carry.len() - keep;
        carry.drain(..cut);
    }
}
