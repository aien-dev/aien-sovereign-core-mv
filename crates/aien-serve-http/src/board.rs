//! The team board inside the server (L376; Drake 2026-10-21: "stop making these agents blind to each
//! other ... have them share memory too. It's all a team effort").
//!
//! Two append-only lists every conversation on this server shares: the **board** (what happened:
//! facts, results, paths, numbers, blockers) and the **memory** (durable facts). Each conversation
//! keeps a cursor per list; at its next turn the server appends the entries it has not seen yet as a
//! short text block at the end of the newest user turn (`server.rs`), so a row never re-reads its
//! history: text appended at a row's next turn costs only its own tokens. The block appended at a
//! turn is remembered as a [`Patch`] and re-applied at the same turn on every later request of that
//! conversation (the OpenAI API is stateless: the client resends the whole history), so the rendered
//! prompt stays append-only and a shared prefix (L375) keeps matching.
//!
//! Storage: one JSON line per entry in `board.jsonl` and `memory.jsonl` (a torn last line, from a
//! crash mid-write, is cut off at the next start), each conversation's cursors and patches in
//! `conversations/<key>.json` (written whole to a temp file and renamed), all under [`DIR_ENV`]
//! (default `~/.local/state/aien-q36/board/`), reloaded at start so the board survives restarts
//! and swaps.

use std::collections::HashMap;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use sha2::{Digest, Sha256};

/// Where the board lives; unset: `$HOME/.local/state/aien-q36/board`.
pub const DIR_ENV: &str = "AIEN_BOARD_DIR";
/// `0` turns the board off entirely (no delivery, no tools, no routes).
pub const ON_ENV: &str = "AIEN_BOARD";
/// `0` keeps the board but stops offering the built-in tools to tool requests.
pub const TOOLS_ENV: &str = "AIEN_BOARD_TOOLS";
/// Entries of one conversation and text within the newest this many are posted once (a retry
/// of the same request, the same note from a tool round run twice).
pub const DEDUPE_WINDOW: usize = 50;

/// Longest text one entry keeps.
pub const MAX_TEXT_CHARS: usize = 4000;
/// Most entries one delivery block carries (older ones are counted, reachable with `board_read`).
pub const BLOCK_MAX_ENTRIES: usize = 24;
/// Most characters one delivery block carries.
pub const BLOCK_MAX_CHARS: usize = 6000;
/// Most conversations remembered (oldest `last_used` forgotten first).
pub const MAX_CONVERSATIONS: usize = 4096;
/// Most chars of blocks one conversation carries (every patched turn is re-applied each request
/// and written whole on admission); past it the conversation gets no new blocks, with a log line.
pub const PATCH_BUDGET: usize = 1 << 20;

pub const BOARD_HEAD: &str = "[team board, new since your last turn]";
pub const MEMORY_HEAD: &str = "[team memory, new since your last turn]";
pub const BLOCK_TAIL: &str = "[end of team board]";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum List {
    Board,
    Memory,
}

impl List {
    fn file(self) -> &'static str {
        match self {
            Self::Board => "board.jsonl",
            Self::Memory => "memory.jsonl",
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Self::Board => "board",
            Self::Memory => "memory",
        }
    }
}

/// One line on the board or in the memory.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Entry {
    pub seq: u64,
    /// Unix seconds.
    pub time: u64,
    pub author: String,
    pub text: String,
    /// The conversation that posted it (its key), so it is not delivered back to that conversation.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub conv: Option<String>,
}

/// Text appended to user turn `turn` of a conversation (whose original text hashed to `hash`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Patch {
    pub turn: usize,
    pub hash: String,
    pub text: String,
    /// The cursors after this patch: where a fork of the conversation (the client rewrote a
    /// later patched turn) resumes, so it gets what the dropped patches carried.
    #[serde(default)]
    pub board_seen: u64,
    #[serde(default)]
    pub memory_seen: u64,
}

/// What the server remembers about one conversation.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct Conversation {
    pub author: String,
    pub board_seen: u64,
    pub memory_seen: u64,
    #[serde(default)]
    pub patches: Vec<Patch>,
    #[serde(default)]
    pub last_used: u64,
    /// The key this record was forked from (the client rewrote a patched turn).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub forked_from: Option<String>,
}

impl Conversation {
    /// The record a fork starts from: the cursors as they were after the last patch kept
    /// (`keep` patches), so the fork receives what the dropped patches delivered.
    pub fn fork(&self, keep: usize, from: &str) -> Self {
        let kept = &self.patches[..keep.min(self.patches.len())];
        let (board_seen, memory_seen) = kept
            .last()
            .map(|p| (p.board_seen, p.memory_seen))
            .unwrap_or((0, 0));
        Self {
            author: self.author.clone(),
            board_seen,
            memory_seen,
            patches: kept.to_vec(),
            last_used: 0,
            forked_from: Some(from.to_string()),
        }
    }
}

/// The key without its fork suffix (`-f<16 hex>`): the conversation a fork and its forks belong to.
pub fn base_key(key: &str) -> &str {
    match key.rfind("-f") {
        Some(i)
            if key.len() - i == 18
                && key[i + 2..]
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)) =>
        {
            &key[..i]
        }
        _ => key,
    }
}

/// `<base key>-f<digest of (key, rewritten turn)>`: the key of a fork. One suffix only, so a
/// conversation rewritten many times (a client compacting its history) keeps a bounded key.
pub fn fork_key(key: &str, turn_text: &str) -> String {
    format!(
        "{}-f{}",
        base_key(key),
        digest(&format!("{key}\0{turn_text}"))
    )
}

/// Keys name files under `conversations/`: lowercase letters, digits, `-`, `_`, `.` (not leading).
pub fn file_safe(key: &str) -> bool {
    !key.is_empty()
        && key.len() <= 160
        && !key.starts_with('.')
        && key.bytes().all(|b| {
            b.is_ascii_lowercase() || b.is_ascii_digit() || b == b'-' || b == b'_' || b == b'.'
        })
}

struct Inner {
    board: Vec<Entry>,
    memory: Vec<Entry>,
    convs: HashMap<String, Conversation>,
    dir: Option<PathBuf>,
    /// The exclusive lock on `<dir>/lock`, held for the life of the board.
    _lock: Option<fs::File>,
}

/// Why a post was refused: the caller's text (a 400) or the board's disk (a 500).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PostError {
    Invalid(String),
    Io(String),
}

impl std::fmt::Display for PostError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(s) | Self::Io(s) => f.write_str(s),
        }
    }
}

/// The shared board, memory and conversation cursors. One per server.
pub struct Board {
    inner: Mutex<Inner>,
    tools: bool,
}

/// Startup settings read from the environment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Settings {
    pub dir: Option<PathBuf>,
    pub tools: bool,
}

impl Settings {
    /// `None`: the board is off (`AIEN_BOARD=0`).
    pub fn from_lookup(lookup: impl Fn(&str) -> Option<String>) -> Option<Self> {
        if lookup(ON_ENV).as_deref() == Some("0") {
            return None;
        }
        let dir = match lookup(DIR_ENV) {
            Some(d) if !d.is_empty() => PathBuf::from(d),
            _ => PathBuf::from(lookup("HOME").unwrap_or_else(|| ".".into()))
                .join(".local/state/aien-q36/board"),
        };
        Some(Self {
            dir: Some(dir),
            tools: lookup(TOOLS_ENV).as_deref() != Some("0"),
        })
    }
}

fn now_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// `MM-DD HH:MMZ` from unix seconds (civil date from days since 1970-01-01, no calendar crate).
pub fn stamp(secs: u64) -> String {
    let days = (secs / 86400) as i64;
    let rem = secs % 86400;
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    format!("{m:02}-{d:02} {:02}:{:02}Z", rem / 3600, (rem % 3600) / 60)
}

/// A short hex digest of a text (patch binding, conversation keys).
pub fn digest(text: &str) -> String {
    let h = Sha256::digest(text.as_bytes());
    h.iter().take(8).map(|b| format!("{b:02x}")).collect()
}

/// The conversation key when the request names none: the system text and the first user text.
pub fn fingerprint(system: &str, first_user: &str) -> String {
    let mut h = Sha256::new();
    h.update(system.as_bytes());
    h.update([0u8]);
    h.update(first_user.as_bytes());
    let h = h.finalize();
    format!(
        "fp-{}",
        h.iter()
            .take(8)
            .map(|b| format!("{b:02x}"))
            .collect::<String>()
    )
}

/// Reads a list file. A last line that does not parse (torn by a crash mid-write) is cut off the
/// file and reported on stderr; any other bad line, or a seq out of order, refuses the file.
/// Reads one list file. A torn last line (a crash mid-write) is cut off. Any other damage (a bad
/// line in the middle, seq out of order: two processes on one dir) sets the whole file aside as
/// `<file>.bad-<unix secs>` with one loud line, and the list starts empty: the notice board
/// never keeps the motor from serving.
fn read_list(path: &Path) -> Result<Vec<Entry>, String> {
    let text = match fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(format!("{}: {e}", path.display())),
    };
    let mut out: Vec<Entry> = Vec::new();
    let mut good_end = 0usize;
    let mut lines = text.split_inclusive('\n').enumerate().peekable();
    let damage: Option<String> = loop {
        let Some((n, raw)) = lines.next() else {
            break None;
        };
        let line = raw.trim_end_matches(['\n', '\r']);
        if line.trim().is_empty() {
            good_end += raw.len();
            continue;
        }
        let e = match serde_json::from_str::<Entry>(line) {
            Ok(e) => e,
            Err(err) if lines.peek().is_none() => {
                eprintln!(
                    "aien-serve-http: board: {}: line {} is torn ({err}); cutting it off",
                    path.display(),
                    n + 1
                );
                let cut = OpenOptions::new()
                    .write(true)
                    .open(path)
                    .and_then(|f| f.set_len(good_end as u64).and_then(|_| f.sync_data()));
                break match cut {
                    Ok(()) => None,
                    Err(e) => Some(format!(
                        "line {} is torn and could not be cut off: {e}",
                        n + 1
                    )),
                };
            }
            Err(err) => break Some(format!("line {}: {err}", n + 1)),
        };
        if out.last().is_some_and(|last| e.seq <= last.seq) {
            break Some(format!(
                "line {}: seq {} is not after {}",
                n + 1,
                e.seq,
                out.last().map(|l| l.seq).unwrap_or(0)
            ));
        }
        out.push(e);
        good_end += raw.len();
    };
    let Some(why) = damage else {
        return Ok(out);
    };
    let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("list");
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let aside = path.with_file_name(format!("{name}.bad-{nanos}"));
    fs::rename(path, &aside).map_err(|e| {
        format!(
            "{}: {why}; could not set it aside as {}: {e}",
            path.display(),
            aside.display()
        )
    })?;
    eprintln!(
        "aien-serve-http: board: {}: {why}; set aside as {} and starting empty",
        path.display(),
        aside.display()
    );
    Ok(Vec::new())
}

/// One server per board dir: an exclusive `flock` on `<dir>/lock`, held for the life of the
/// board, the holder's pid in the file. A second server on the same dir would interleave its own
/// seq numbers into the lists and leave them unreadable at the next restart.
fn take_lock(dir: &Path) -> Result<fs::File, String> {
    use std::os::unix::io::AsRawFd;
    let path = dir.join("lock");
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .truncate(false)
        .open(&path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    // SAFETY: flock on a valid open descriptor; non-blocking, so a held lock answers at once.
    let rc = unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    if rc != 0 {
        let err = std::io::Error::last_os_error();
        let holder = fs::read_to_string(&path).unwrap_or_default();
        let holder = holder.trim();
        return Err(if err.raw_os_error() == Some(libc::EWOULDBLOCK) {
            format!(
                "{} is held by another server (pid {})",
                dir.display(),
                if holder.is_empty() { "unknown" } else { holder }
            )
        } else {
            format!("{}: flock: {err}", path.display())
        });
    }
    let _ = f
        .set_len(0)
        .and_then(|_| f.write_all(format!("{}\n", std::process::id()).as_bytes()));
    Ok(f)
}

/// Loads every `conversations/<key>.json`; a file that does not parse is reported and skipped.
fn read_conversations(dir: &Path) -> Result<HashMap<String, Conversation>, String> {
    let cdir = dir.join("conversations");
    let mut out = HashMap::new();
    let rd = match fs::read_dir(&cdir) {
        Ok(rd) => rd,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(out),
        Err(e) => return Err(format!("{}: {e}", cdir.display())),
    };
    for ent in rd {
        let ent = ent.map_err(|e| format!("{}: {e}", cdir.display()))?;
        let path = ent.path();
        let Some(key) = path
            .file_name()
            .and_then(|n| n.to_str())
            .and_then(|n| n.strip_suffix(".json"))
        else {
            continue;
        };
        match fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| serde_json::from_str::<Conversation>(&t).map_err(|e| e.to_string()))
        {
            Ok(c) => {
                out.insert(key.to_string(), c);
            }
            Err(e) => eprintln!("aien-serve-http: board: {}: {e}; skipped", path.display()),
        }
    }
    Ok(out)
}

fn append_line(path: &Path, line: &str) -> Result<(), String> {
    let mut f = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .map_err(|e| format!("{}: {e}", path.display()))?;
    f.write_all(line.as_bytes())
        .and_then(|_| f.write_all(b"\n"))
        .and_then(|_| f.sync_data())
        .map_err(|e| format!("{}: {e}", path.display()))
}

impl Board {
    /// Opens (or creates) the board under `settings.dir`; `dir: None` keeps everything in memory (tests).
    pub fn open(settings: &Settings) -> Result<Self, String> {
        let (board, memory, convs, lock) = match &settings.dir {
            Some(dir) => {
                fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
                let lock = take_lock(dir)?;
                let board = read_list(&dir.join(List::Board.file()))?;
                let memory = read_list(&dir.join(List::Memory.file()))?;
                let convs = read_conversations(dir)?;
                (board, memory, convs, Some(lock))
            }
            None => (Vec::new(), Vec::new(), HashMap::new(), None),
        };
        Ok(Self {
            inner: Mutex::new(Inner {
                board,
                memory,
                convs,
                dir: settings.dir.clone(),
                _lock: lock,
            }),
            tools: settings.tools,
        })
    }

    /// An in-memory board (tests, the stub engine without a state dir).
    pub fn in_memory() -> Self {
        Self::open(&Settings {
            dir: None,
            tools: true,
        })
        .expect("in-memory board")
    }

    /// Whether tool requests get `board_post` / `board_read` / `memory_note`.
    pub fn offers_tools(&self) -> bool {
        self.tools
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Appends one entry. `conv`: the posting conversation's key (a tool call), `None` for HTTP posts.
    pub fn post(
        &self,
        list: List,
        author: &str,
        text: &str,
        conv: Option<&str>,
    ) -> Result<Entry, PostError> {
        let text = text.trim();
        if text.is_empty() {
            return Err(PostError::Invalid("the text is empty".into()));
        }
        let text: String = text.chars().take(MAX_TEXT_CHARS).collect();
        let author = author.trim();
        let author = if author.is_empty() {
            "anonymous"
        } else {
            author
        };
        let author: String = author.chars().take(80).collect();
        let mut g = self.lock();
        let entries = match list {
            List::Board => &g.board,
            List::Memory => &g.memory,
        };
        // The same poster saying the same thing again within the window (a retried request, a
        // tool round run twice) is the entry already there.
        let window = entries.len().saturating_sub(DEDUPE_WINDOW);
        if let Some(old) = entries[window..]
            .iter()
            .rev()
            .find(|e| e.conv.as_deref() == conv && e.author == author && e.text == text)
        {
            return Ok(old.clone());
        }
        let entry = Entry {
            seq: entries.last().map(|e| e.seq + 1).unwrap_or(1),
            time: now_secs(),
            author,
            text,
            conv: conv.map(str::to_string),
        };
        if let Some(dir) = g.dir.clone() {
            let line = serde_json::to_string(&entry).map_err(|e| PostError::Io(e.to_string()))?;
            append_line(&dir.join(list.file()), &line).map_err(PostError::Io)?;
        }
        match list {
            List::Board => g.board.push(entry.clone()),
            List::Memory => g.memory.push(entry.clone()),
        }
        Ok(entry)
    }

    /// Entries with `seq > since`, the last `limit` of them, and the newest seq of the list.
    pub fn read(&self, list: List, since: u64, limit: usize) -> (Vec<Entry>, u64) {
        let g = self.lock();
        let entries = match list {
            List::Board => &g.board,
            List::Memory => &g.memory,
        };
        let newest = entries.last().map(|e| e.seq).unwrap_or(0);
        let start = entries.partition_point(|e| e.seq <= since);
        let tail = &entries[start..];
        let keep = tail.len().saturating_sub(limit.max(1));
        (tail[keep..].to_vec(), newest)
    }

    /// `(board entries, memory entries, conversations)`.
    pub fn counts(&self) -> (usize, usize, usize) {
        let g = self.lock();
        (g.board.len(), g.memory.len(), g.convs.len())
    }

    pub fn dir(&self) -> Option<PathBuf> {
        self.lock().dir.clone()
    }

    /// The remembered state of a conversation (default when new).
    pub fn conversation(&self, key: &str) -> Conversation {
        self.lock().convs.get(key).cloned().unwrap_or_default()
    }

    /// Stores a conversation's cursors and patches (called when its request is admitted): its own
    /// file, written whole to a temp file and renamed. Returns the time the write took.
    pub fn commit(&self, key: &str, mut conv: Conversation) -> Result<Duration, String> {
        if !file_safe(key) {
            return Err(format!("conversation key {key:?} is not file-safe"));
        }
        conv.last_used = now_secs();
        let mut g = self.lock();
        // Nothing new (no block this turn, cursors unchanged): the file is already right.
        let same = g.convs.get(key).is_some_and(|old| {
            old.author == conv.author
                && old.board_seen == conv.board_seen
                && old.memory_seen == conv.memory_seen
                && old.patches == conv.patches
                && old.forked_from == conv.forked_from
        });
        g.convs.insert(key.to_string(), conv);
        if same {
            return Ok(Duration::ZERO);
        }
        let mut evicted = Vec::new();
        if g.convs.len() > MAX_CONVERSATIONS {
            let mut by_age: Vec<(u64, String)> = g
                .convs
                .iter()
                .map(|(k, c)| (c.last_used, k.clone()))
                .collect();
            by_age.sort();
            for (_, k) in by_age.iter().take(g.convs.len() - MAX_CONVERSATIONS) {
                g.convs.remove(k);
                evicted.push(k.clone());
            }
        }
        let t = Instant::now();
        if let Some(dir) = g.dir.clone() {
            let cdir = dir.join("conversations");
            fs::create_dir_all(&cdir).map_err(|e| format!("{}: {e}", cdir.display()))?;
            let path = cdir.join(format!("{key}.json"));
            let tmp = cdir.join(format!("{key}.json.tmp"));
            let text = serde_json::to_string(&g.convs[key]).map_err(|e| e.to_string())?;
            fs::write(&tmp, text)
                .and_then(|_| fs::rename(&tmp, &path))
                .map_err(|e| format!("{}: {e}", path.display()))?;
            for k in evicted {
                let _ = fs::remove_file(cdir.join(format!("{k}.json")));
            }
        }
        Ok(t.elapsed())
    }

    /// What a conversation has not seen yet: new board and memory entries (not its own posts),
    /// and how many older board entries fall outside the block cap.
    pub fn unseen(&self, key: &str, conv: &Conversation) -> Unseen {
        let g = self.lock();
        let base = base_key(key);
        let pick = |entries: &[Entry], since: u64| -> Vec<Entry> {
            let start = entries.partition_point(|e| e.seq <= since);
            entries[start..]
                .iter()
                .filter(|e| !e.conv.as_deref().is_some_and(|c| base_key(c) == base))
                .cloned()
                .collect()
        };
        let board = pick(&g.board, conv.board_seen);
        let memory = pick(&g.memory, conv.memory_seen);
        Unseen {
            board_newest: g.board.last().map(|e| e.seq).unwrap_or(0),
            memory_newest: g.memory.last().map(|e| e.seq).unwrap_or(0),
            board,
            memory,
        }
    }
}

/// New entries for one conversation, with the newest seq of each list (the cursors to commit).
#[derive(Debug, Clone, Default)]
pub struct Unseen {
    pub board: Vec<Entry>,
    pub memory: Vec<Entry>,
    pub board_newest: u64,
    pub memory_newest: u64,
}

/// A rendered block and the cursors a conversation holds after it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Rendered {
    pub text: String,
    pub board_seen: u64,
    pub memory_seen: u64,
}

impl Unseen {
    pub fn is_empty(&self) -> bool {
        self.board.is_empty() && self.memory.is_empty()
    }

    /// The text block appended to the newest user turn, with the cursors after it. Memory first
    /// (durable facts), OLDEST first within the caps and the rest carried to the next turn (the
    /// memory cursor stops at the last note shown, so no note is ever skipped). Then the board,
    /// NEWEST kept when the cap cuts (older ones counted; `board_read` reaches them) and the
    /// board cursor at the newest entry.
    pub fn render(&self) -> Option<Rendered> {
        if self.is_empty() {
            return None;
        }
        let mut out = String::new();
        let mut budget = BLOCK_MAX_CHARS;
        let mut memory_seen = self.memory_newest;
        if !self.memory.is_empty() {
            out.push_str(MEMORY_HEAD);
            out.push('\n');
            let (lines, shown) = render_oldest(&self.memory, BLOCK_MAX_ENTRIES, &mut budget);
            out.push_str(&lines);
            let left = self.memory.len() - shown;
            if left > 0 {
                memory_seen = self.memory[shown - 1].seq;
                out.push_str(&format!(
                    "(+{left} more team memory notes follow next turn)\n"
                ));
            }
        }
        if !self.board.is_empty() {
            if !out.is_empty() {
                out.push('\n');
            }
            out.push_str(BOARD_HEAD);
            out.push('\n');
            let (lines, hidden) = render_lines(&self.board, BLOCK_MAX_ENTRIES, &mut budget);
            out.push_str(&lines);
            if hidden > 0 {
                out.push_str(&format!(
                    "(+{hidden} older board entries not shown; board_read(since) reaches them)\n"
                ));
            }
        }
        out.push_str(BLOCK_TAIL);
        Some(Rendered {
            text: out,
            board_seen: self.board_newest,
            memory_seen,
        })
    }

    /// The block's text alone.
    pub fn block(&self) -> Option<String> {
        self.render().map(|r| r.text)
    }
}

/// `- MM-DD HH:MMZ author: text`.
fn line_of(e: &Entry) -> String {
    format!(
        "- {} {}: {}\n",
        stamp(e.time),
        e.author,
        e.text.replace('\n', " ")
    )
}

/// Lines for the NEWEST `max` entries within `budget` chars (at least one); the count left out.
fn render_lines(entries: &[Entry], max: usize, budget: &mut usize) -> (String, usize) {
    let mut picked: Vec<String> = Vec::new();
    let mut used = 0usize;
    for e in entries.iter().rev() {
        if picked.len() >= max {
            break;
        }
        let line = line_of(e);
        if used + line.len() > *budget && !picked.is_empty() {
            break;
        }
        used += line.len();
        picked.push(line);
    }
    *budget = budget.saturating_sub(used);
    let hidden = entries.len() - picked.len();
    picked.reverse();
    (picked.concat(), hidden)
}

/// Lines for the OLDEST `max` entries within `budget` chars (at least one); the count taken.
fn render_oldest(entries: &[Entry], max: usize, budget: &mut usize) -> (String, usize) {
    let mut out = String::new();
    let mut taken = 0usize;
    for e in entries {
        if taken >= max {
            break;
        }
        let line = line_of(e);
        if out.len() + line.len() > *budget && taken > 0 {
            break;
        }
        out.push_str(&line);
        taken += 1;
    }
    *budget = budget.saturating_sub(out.len());
    (out, taken)
}

/// Whether a user turn's text is shaped like the template's tool-response turn (the Qwen3.6
/// `last_query` rule: such a turn must keep that shape).
pub fn is_tool_response_turn(text: &str) -> bool {
    let c = text.trim();
    c.starts_with("<tool_response>") && c.ends_with("</tool_response>")
}

/// A user turn's text with the block appended: a paragraph, or one more `<tool_response>` entry
/// on a tool-response turn (so the template still sees a tool-response turn).
pub fn apply_block(user_text: &str, block: &str) -> String {
    if is_tool_response_turn(user_text) {
        format!(
            "{}\n<tool_response>\n{block}\n</tool_response>",
            user_text.trim_end()
        )
    } else {
        format!("{}\n\n{block}", user_text.trim_end())
    }
}

/// The built-in tools in OpenAI shape (rendered first in the tools block: stable-first order).
pub fn tool_defs() -> Vec<Value> {
    vec![
        serde_json::json!({"type": "function", "function": {
            "name": "board_post",
            "description": "Post one short line to the team board shared by every conversation on this server; the other agents see it at their next turn. Post facts, results, paths, measured numbers and blockers others can use. Never post secrets.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "The line to post: one plain fact, with the sha, path or number in it."}
            }, "required": ["text"]}}}),
        serde_json::json!({"type": "function", "function": {
        "name": "board_read",
        "description": "Read the team board: the entries after a sequence number (0 = from the start), newest last.",
        "parameters": {"type": "object", "properties": {
            "since": {"type": "integer", "description": "Return entries with a sequence number greater than this (default 0)."},
            "limit": {"type": "integer", "description": "At most this many of the newest matching entries (default 20, max 200)."}
        }}}}),
        serde_json::json!({"type": "function", "function": {
            "name": "memory_note",
            "description": "Keep one durable fact in the team's shared memory (facts that stay true: decisions, paths, measured numbers). Every conversation receives it at its next turn.",
            "parameters": {"type": "object", "properties": {
                "text": {"type": "string", "description": "The fact, one plain line."}
            }, "required": ["text"]}}}),
    ]
}

pub const TOOL_NAMES: [&str; 3] = ["board_post", "board_read", "memory_note"];

pub fn is_builtin(name: &str) -> bool {
    TOOL_NAMES.contains(&name)
}

/// Runs one built-in call for conversation `key` (author `author`); the text the model reads back.
pub fn call(board: &Board, key: &str, author: &str, name: &str, args: &Value) -> String {
    let text = args.get("text").and_then(Value::as_str).unwrap_or("");
    match name {
        "board_post" => match board.post(List::Board, author, text, Some(key)) {
            Ok(e) => format!("posted to the team board as entry {}", e.seq),
            Err(e) => format!("board_post refused: {e}"),
        },
        "memory_note" => match board.post(List::Memory, author, text, Some(key)) {
            Ok(e) => format!("kept in the team memory as entry {}", e.seq),
            Err(e) => format!("memory_note refused: {e}"),
        },
        "board_read" => {
            let num = |k: &str, d: u64| -> u64 {
                match args.get(k) {
                    Some(Value::Number(n)) => n.as_u64().unwrap_or(d),
                    Some(Value::String(s)) => s.trim().parse().unwrap_or(d),
                    _ => d,
                }
            };
            let since = num("since", 0);
            let limit = num("limit", 20).clamp(1, 200) as usize;
            let (entries, newest) = board.read(List::Board, since, limit);
            if entries.is_empty() {
                return format!("the team board has no entries after {since} (newest is {newest})");
            }
            let mut out = String::new();
            for e in &entries {
                out.push_str(&format!(
                    "{} {} {}: {}\n",
                    e.seq,
                    stamp(e.time),
                    e.author,
                    e.text.replace('\n', " ")
                ));
            }
            out.push_str(&format!("(newest entry is {newest})"));
            out
        }
        other => format!("unknown built-in tool {other}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stamp_renders_civil_date_and_time() {
        // 2026-10-21T12:34:56Z
        assert_eq!(stamp(1792586096), "10-21 12:34Z");
        assert_eq!(stamp(0), "01-01 00:00Z");
    }

    #[test]
    fn post_read_and_unseen_skip_own_posts() {
        let b = Board::in_memory();
        let a = b
            .post(List::Board, "alice", "fact one", Some("conv-a"))
            .unwrap();
        assert_eq!(a.seq, 1);
        let m = b
            .post(List::Memory, "alice", "durable", Some("conv-a"))
            .unwrap();
        assert_eq!(m.seq, 1);
        b.post(List::Board, "bob", "fact two", Some("conv-b"))
            .unwrap();
        let (got, newest) = b.read(List::Board, 0, 10);
        assert_eq!(got.len(), 2);
        assert_eq!(newest, 2);
        let (got, _) = b.read(List::Board, 1, 10);
        assert_eq!(got.len(), 1);
        assert_eq!(got[0].text, "fact two");
        let (got, _) = b.read(List::Board, 0, 1);
        assert_eq!(got[0].seq, 2);
        let u = b.unseen("conv-a", &Conversation::default());
        assert_eq!(u.board.iter().map(|e| e.seq).collect::<Vec<_>>(), vec![2]);
        assert!(u.memory.is_empty());
        assert_eq!((u.board_newest, u.memory_newest), (2, 1));
        let u = b.unseen("conv-c", &Conversation::default());
        assert_eq!(u.board.len(), 2);
        assert_eq!(u.memory.len(), 1);
        let block = u.block().unwrap();
        assert!(block.starts_with(MEMORY_HEAD));
        assert!(block.contains("alice: fact one\n"));
        assert!(block.ends_with(BLOCK_TAIL));
    }

    #[test]
    fn post_refuses_empty_and_caps_text() {
        let b = Board::in_memory();
        assert!(b.post(List::Board, "a", "   ", None).is_err());
        let long = "x".repeat(MAX_TEXT_CHARS + 10);
        let e = b.post(List::Board, "a", &long, None).unwrap();
        assert_eq!(e.text.chars().count(), MAX_TEXT_CHARS);
    }

    #[test]
    fn block_caps_entries_and_counts_the_rest() {
        let b = Board::in_memory();
        for i in 0..(BLOCK_MAX_ENTRIES + 5) {
            b.post(List::Board, "a", &format!("entry {i}"), None)
                .unwrap();
        }
        let u = b.unseen("k", &Conversation::default());
        let block = u.block().unwrap();
        assert!(block.contains("(+5 older board entries not shown"));
        assert!(!block.contains("entry 4\n"));
        assert!(block.contains("entry 5\n"));
        assert!(block.contains(&format!("entry {}\n", BLOCK_MAX_ENTRIES + 4)));
    }

    #[test]
    fn apply_block_keeps_tool_response_shape() {
        let t = "<tool_response>\nok\n</tool_response>";
        let out = apply_block(t, "B");
        assert!(is_tool_response_turn(&out));
        assert_eq!(
            out,
            "<tool_response>\nok\n</tool_response>\n<tool_response>\nB\n</tool_response>"
        );
        assert_eq!(apply_block("hello  \n", "B"), "hello\n\nB");
    }

    fn temp_dir(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("aien-board-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        dir
    }

    #[test]
    fn persists_entries_and_conversations_across_reopen() {
        let dir = temp_dir("persist");
        let s = Settings {
            dir: Some(dir.clone()),
            tools: true,
        };
        {
            let b = Board::open(&s).unwrap();
            b.post(List::Board, "a", "one", Some("k1")).unwrap();
            b.post(List::Memory, "a", "keep", None).unwrap();
            b.post(List::Board, "b", "two\nlines", None).unwrap();
            let took = b
                .commit(
                    "k1",
                    Conversation {
                        author: "a".into(),
                        board_seen: 2,
                        memory_seen: 1,
                        patches: vec![Patch {
                            turn: 1,
                            hash: digest("hi"),
                            text: "B".into(),
                            board_seen: 2,
                            memory_seen: 1,
                        }],
                        last_used: 0,
                        forked_from: None,
                    },
                )
                .unwrap();
            eprintln!("commit write took {} us", took.as_micros());
            assert!(b.commit("Bad Key", Conversation::default()).is_err());
            assert!(dir.join("conversations/k1.json").is_file());
        }
        let b = Board::open(&s).unwrap();
        assert_eq!(b.counts(), (2, 1, 1));
        let (got, newest) = b.read(List::Board, 0, 10);
        assert_eq!(newest, 2);
        assert_eq!(got[1].text, "two\nlines");
        assert_eq!(got[0].conv.as_deref(), Some("k1"));
        let c = b.conversation("k1");
        assert_eq!(c.board_seen, 2);
        assert_eq!(c.patches.len(), 1);
        assert_eq!(c.patches[0].board_seen, 2);
        assert!(c.last_used > 0);
        let e = b.post(List::Board, "c", "three", None).unwrap();
        assert_eq!(e.seq, 3);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_torn_last_line_is_cut_off_and_the_list_goes_on() {
        let dir = temp_dir("torn");
        fs::create_dir_all(&dir).unwrap();
        let good = "{\"seq\":1,\"time\":1,\"author\":\"a\",\"text\":\"x\"}\n";
        fs::write(
            dir.join("board.jsonl"),
            format!("{good}{{\"seq\":2,\"time\":1,\"au"),
        )
        .unwrap();
        let s = Settings {
            dir: Some(dir.clone()),
            tools: true,
        };
        let b = Board::open(&s).unwrap();
        assert_eq!(b.counts().0, 1);
        assert_eq!(fs::read_to_string(dir.join("board.jsonl")).unwrap(), good);
        let e = b.post(List::Board, "b", "y", None).unwrap();
        assert_eq!(e.seq, 2);
        drop(b);
        let b = Board::open(&s).unwrap();
        assert_eq!(b.read(List::Board, 0, 10).0.len(), 2);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_same_post_twice_is_one_entry() {
        let b = Board::in_memory();
        let a = b.post(List::Board, "a", "sha 1234", Some("k")).unwrap();
        let again = b.post(List::Board, "a", "sha 1234", Some("k")).unwrap();
        assert_eq!(again.seq, a.seq);
        let other = b.post(List::Board, "a", "sha 1234", Some("k2")).unwrap();
        assert_eq!(other.seq, 2);
        let http = b.post(List::Board, "a", "sha 1234", None).unwrap();
        assert_eq!(http.seq, 3);
        assert_eq!(b.post(List::Board, "a", "sha 1234", None).unwrap().seq, 3);
        assert_eq!(b.counts().0, 3);
    }

    #[test]
    fn a_fork_resumes_from_the_last_kept_patch() {
        let c = Conversation {
            author: "a".into(),
            board_seen: 9,
            memory_seen: 4,
            patches: vec![
                Patch {
                    turn: 1,
                    hash: digest("one"),
                    text: "B1".into(),
                    board_seen: 3,
                    memory_seen: 1,
                },
                Patch {
                    turn: 3,
                    hash: digest("three"),
                    text: "B2".into(),
                    board_seen: 9,
                    memory_seen: 4,
                },
            ],
            last_used: 5,
            forked_from: None,
        };
        let f = c.fork(1, "k");
        assert_eq!((f.board_seen, f.memory_seen), (3, 1));
        assert_eq!(f.patches.len(), 1);
        assert_eq!(f.forked_from.as_deref(), Some("k"));
        let f0 = c.fork(0, "k");
        assert_eq!((f0.board_seen, f0.memory_seen), (0, 0));
        assert!(f0.patches.is_empty());
        assert!(file_safe(&fork_key("fp-ab12", "three")));
        assert!(fork_key("fp-ab12", "three").starts_with("fp-ab12-f"));
        assert!(file_safe("u-1a2b-fp-3c4d") && file_safe("id-00ff"));
        assert!(!file_safe("") && !file_safe("../x") && !file_safe("A") && !file_safe("a b"));
    }

    fn set_aside_files(dir: &Path) -> Vec<String> {
        let mut v: Vec<String> = fs::read_dir(dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("board.jsonl.bad-"))
            .collect();
        v.sort();
        v
    }

    #[test]
    fn a_damaged_list_is_set_aside_and_the_board_opens() {
        let dir = temp_dir("bad");
        fs::create_dir_all(&dir).unwrap();
        let s = Settings {
            dir: Some(dir.clone()),
            tools: true,
        };
        // seq out of order: two processes appended to one file.
        fs::write(dir.join("board.jsonl"), "{\"seq\":2,\"time\":1,\"author\":\"a\",\"text\":\"x\"}\n{\"seq\":1,\"time\":1,\"author\":\"a\",\"text\":\"y\"}\n").unwrap();
        let b = Board::open(&s).expect("the motor still serves");
        assert_eq!(b.counts().0, 0);
        assert_eq!(set_aside_files(&dir).len(), 1);
        assert!(!dir.join("board.jsonl").exists());
        // The board goes on in a fresh file.
        assert_eq!(b.post(List::Board, "a", "fresh", None).unwrap().seq, 1);
        drop(b);
        // A bad line in the middle.
        fs::write(dir.join("board.jsonl"), "{\"seq\":1,\"time\":1,\"author\":\"a\",\"text\":\"x\"}\nnot json\n{\"seq\":2,\"time\":1,\"author\":\"a\",\"text\":\"y\"}\n").unwrap();
        let b = Board::open(&s).expect("opens");
        assert_eq!(b.counts().0, 0);
        let aside = set_aside_files(&dir);
        assert_eq!(aside.len(), 2, "{aside:?}");
        let kept = fs::read_to_string(dir.join(&aside[1])).unwrap();
        assert!(kept.contains("not json"), "the damaged file is kept whole");
        drop(b);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn one_server_per_board_dir() {
        let dir = temp_dir("lock");
        let s = Settings {
            dir: Some(dir.clone()),
            tools: true,
        };
        let first = Board::open(&s).expect("opens");
        let e = Board::open(&s).err().expect("a second opener is refused");
        let me = std::process::id().to_string();
        assert!(
            e.contains("held by another server") && e.contains(&me),
            "{e}"
        );
        drop(first);
        let again = Board::open(&s).expect("free once the holder is gone");
        drop(again);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn forks_keep_one_suffix_and_stay_file_safe() {
        let dir = temp_dir("forks");
        let s = Settings {
            dir: Some(dir.clone()),
            tools: true,
        };
        let b = Board::open(&s).unwrap();
        let base = "u-0123456789abcdef-fp-fedcba9876543210";
        let mut key = base.to_string();
        let mut seen = std::collections::HashSet::new();
        for i in 0..10 {
            key = fork_key(&key, &format!("rewrite {i}"));
            assert_eq!(key.len(), base.len() + 18, "{key}");
            assert!(file_safe(&key), "{key}");
            assert_eq!(base_key(&key), base);
            assert!(seen.insert(key.clone()), "fork keys repeat: {key}");
            b.commit(&key, Conversation::default()).expect("stored");
            assert!(dir
                .join("conversations")
                .join(format!("{key}.json"))
                .exists());
        }
        // The same rewrite from the same key is the same fork (a retried request finds its record).
        assert_eq!(fork_key("id-00ff", "x"), fork_key("id-00ff", "x"));
        assert_ne!(fork_key("id-00ff", "x"), fork_key("id-00ff", "y"));
        assert_eq!(base_key("id-00ff"), "id-00ff");
        assert_eq!(base_key("fp-f123456789abcdef"), "fp-f123456789abcdef");
        drop(b);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn memory_notes_arrive_oldest_first_and_none_is_skipped() {
        let b = Board::in_memory();
        for i in 1..=30 {
            b.post(List::Memory, "a", &format!("note {i}"), None)
                .unwrap();
        }
        let mut conv = Conversation::default();
        let has = |text: &str, i: usize| text.contains(&format!(": note {i}\n"));
        let r1 = b.unseen("k", &conv).render().expect("first turn");
        assert!(
            (1..=BLOCK_MAX_ENTRIES).all(|i| has(&r1.text, i)),
            "{}",
            r1.text
        );
        assert!((BLOCK_MAX_ENTRIES + 1..=30).all(|i| !has(&r1.text, i)));
        assert!(r1
            .text
            .contains("(+6 more team memory notes follow next turn)"));
        assert_eq!(r1.memory_seen, BLOCK_MAX_ENTRIES as u64);
        conv.board_seen = r1.board_seen;
        conv.memory_seen = r1.memory_seen;
        let r2 = b.unseen("k", &conv).render().expect("second turn");
        assert!(
            (BLOCK_MAX_ENTRIES + 1..=30).all(|i| has(&r2.text, i)),
            "{}",
            r2.text
        );
        assert!((1..=BLOCK_MAX_ENTRIES).all(|i| !has(&r2.text, i)));
        assert!(!r2.text.contains("follow next turn"));
        assert_eq!(r2.memory_seen, 30);
        conv.memory_seen = r2.memory_seen;
        assert!(b.unseen("k", &conv).render().is_none());
    }

    #[test]
    fn a_fork_does_not_get_its_own_posts_back() {
        let b = Board::in_memory();
        let fork = fork_key("id-00ff", "rewritten");
        b.post(List::Board, "a", "mine", Some(&fork)).unwrap();
        b.post(List::Board, "b", "theirs", Some("id-00aa")).unwrap();
        let texts = |u: Unseen| u.board.iter().map(|e| e.text.clone()).collect::<Vec<_>>();
        assert_eq!(
            texts(b.unseen("id-00ff", &Conversation::default())),
            ["theirs"]
        );
        let again = fork_key(&fork, "again");
        assert_eq!(
            texts(b.unseen(&again, &Conversation::default())),
            ["theirs"]
        );
        assert_eq!(
            texts(b.unseen("id-00aa", &Conversation::default())),
            ["mine"]
        );
    }

    #[test]
    fn a_big_record_commits_in_bounded_time_and_an_unchanged_one_skips_the_write() {
        let dir = temp_dir("big");
        let s = Settings {
            dir: Some(dir.clone()),
            tools: true,
        };
        let b = Board::open(&s).unwrap();
        let conv = Conversation {
            author: "a".into(),
            board_seen: 64,
            memory_seen: 0,
            patches: (0..64)
                .map(|i| Patch {
                    turn: 2 * i + 1,
                    hash: digest("x"),
                    text: "y".repeat(BLOCK_MAX_CHARS),
                    board_seen: i as u64,
                    memory_seen: 0,
                })
                .collect(),
            last_used: 0,
            forked_from: None,
        };
        let took = b.commit("id-00ff", conv.clone()).unwrap();
        eprintln!(
            "receipt: commit of 64 patches x {BLOCK_MAX_CHARS} chars took {} us",
            took.as_micros()
        );
        assert!(took < Duration::from_secs(1), "{took:?}");
        let again = b.commit("id-00ff", conv).unwrap();
        assert_eq!(
            again,
            Duration::ZERO,
            "an unchanged record is not rewritten"
        );
        drop(b);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn settings_from_env() {
        let s = Settings::from_lookup(|n| match n {
            "HOME" => Some("/h".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(
            s.dir.as_deref(),
            Some(Path::new("/h/.local/state/aien-q36/board"))
        );
        assert!(s.tools);
        assert!(Settings::from_lookup(|n| (n == ON_ENV).then(|| "0".into())).is_none());
        let s = Settings::from_lookup(|n| match n {
            DIR_ENV => Some("/d".into()),
            TOOLS_ENV => Some("0".into()),
            _ => None,
        })
        .unwrap();
        assert_eq!(s.dir.as_deref(), Some(Path::new("/d")));
        assert!(!s.tools);
    }

    #[test]
    fn builtin_calls_round_trip() {
        let b = Board::in_memory();
        let r = call(
            &b,
            "k",
            "alice",
            "board_post",
            &serde_json::json!({"text": "sha 1234"}),
        );
        assert_eq!(r, "posted to the team board as entry 1");
        let r = call(
            &b,
            "k",
            "alice",
            "memory_note",
            &serde_json::json!({"text": "path /x"}),
        );
        assert_eq!(r, "kept in the team memory as entry 1");
        let r = call(
            &b,
            "k",
            "alice",
            "board_read",
            &serde_json::json!({"since": "0"}),
        );
        assert!(r.starts_with("1 "));
        assert!(r.contains("alice: sha 1234\n(newest entry is 1)"));
        let r = call(
            &b,
            "k",
            "alice",
            "board_read",
            &serde_json::json!({"since": 1}),
        );
        assert!(r.contains("no entries after 1"));
        let r = call(&b, "k", "alice", "board_post", &serde_json::json!({}));
        assert!(r.starts_with("board_post refused"));
        assert!(is_builtin("board_post") && !is_builtin("read_file"));
        assert_eq!(tool_defs().len(), 3);
    }
}
