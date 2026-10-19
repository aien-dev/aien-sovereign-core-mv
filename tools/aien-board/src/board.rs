//! The change board: one append-only TSV on the Spark (`<board-dir>/board.tsv`), one writer at a time
//! (`flock` on `board.lock`), never rewritten. Producers post what changed; each consumer answers a post
//! once with its own verdict. The changed paths of a post sit next to it in `posts/<id>.paths`.
//!
//! Columns: `id utc kind post producer sha base consumer verdict note`
//! - `POST`: `post` is its own id, `consumer` and `verdict` are `-`, `note` says how many paths changed.
//! - `ANSWER`: `post` is the answered id; `verdict` is PASS, FAIL, SKIP (no watched path changed) or
//!   ERROR (the check could not run; never counts as PASS); `note` names the receipt or the reason.
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::os::fd::AsRawFd;
use std::path::{Path, PathBuf};

pub const HEADER: &str = "id\tutc\tkind\tpost\tproducer\tsha\tbase\tconsumer\tverdict\tnote";

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Row {
    pub id: String,
    pub utc: String,
    pub kind: String,
    pub post: String,
    pub producer: String,
    pub sha: String,
    pub base: String,
    pub consumer: String,
    pub verdict: String,
    pub note: String,
}

impl Row {
    fn line(&self) -> String {
        [
            &self.id, &self.utc, &self.kind, &self.post, &self.producer, &self.sha, &self.base,
            &self.consumer, &self.verdict, &self.note,
        ]
        .iter()
        .map(|f| f.replace(['\t', '\n'], " "))
        .collect::<Vec<_>>()
        .join("\t")
    }

    fn parse(line: &str) -> Option<Row> {
        let f: Vec<&str> = line.split('\t').collect();
        if f.len() != 10 || f[0] == "id" {
            return None;
        }
        Some(Row {
            id: f[0].into(),
            utc: f[1].into(),
            kind: f[2].into(),
            post: f[3].into(),
            producer: f[4].into(),
            sha: f[5].into(),
            base: f[6].into(),
            consumer: f[7].into(),
            verdict: f[8].into(),
            note: f[9].into(),
        })
    }
}

pub struct Board {
    pub dir: PathBuf,
}

struct Lock(File);

impl Drop for Lock {
    fn drop(&mut self) {
        // SAFETY: the descriptor belongs to the open file held by self.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

pub fn now_utc() -> String {
    let t = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (t.div_euclid(86400), t.rem_euclid(86400));
    // Civil from days (Howard Hinnant).
    let z = days + 719468;
    let era = z.div_euclid(146097);
    let doe = z - era * 146097;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!(
        "{y:04}-{m:02}-{d:02}T{:02}:{:02}:{:02}Z",
        rem / 3600,
        rem % 3600 / 60,
        rem % 60
    )
}

impl Board {
    pub fn open(dir: &Path) -> Result<Board, String> {
        std::fs::create_dir_all(dir.join("posts")).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
        Ok(Board { dir: dir.to_path_buf() })
    }

    fn lock(&self) -> Result<Lock, String> {
        let f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("board.lock"))
            .map_err(|e| format!("cannot open board.lock: {e}"))?;
        // SAFETY: plain flock on a descriptor we own; blocks until the other writer is done.
        if unsafe { libc::flock(f.as_raw_fd(), libc::LOCK_EX) } != 0 {
            return Err("cannot lock the board".into());
        }
        Ok(Lock(f))
    }

    pub fn rows(&self) -> Result<Vec<Row>, String> {
        match std::fs::read_to_string(self.dir.join("board.tsv")) {
            Ok(t) => Ok(t.lines().filter_map(Row::parse).collect()),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
            Err(e) => Err(format!("cannot read board.tsv: {e}")),
        }
    }

    /// Append one row under the lock; the id is the next number. `paths` (POST only) go to posts/<id>.paths.
    pub fn append(&self, mut row: Row, paths: Option<&[String]>) -> Result<Row, String> {
        let _l = self.lock()?;
        let n = self.rows()?.len() + 1;
        row.id = format!("B{n:06}");
        if row.kind == "POST" {
            row.post = row.id.clone();
        }
        row.utc = now_utc();
        if let Some(p) = paths {
            let body: String = p.iter().map(|s| format!("{s}\n")).collect();
            std::fs::write(self.dir.join("posts").join(format!("{}.paths", row.id)), body)
                .map_err(|e| format!("cannot write the paths of {}: {e}", row.id))?;
        }
        let path = self.dir.join("board.tsv");
        let fresh = !path.exists();
        let mut f = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| format!("cannot open board.tsv: {e}"))?;
        let mut text = String::new();
        if fresh {
            text.push_str(HEADER);
            text.push('\n');
        }
        text.push_str(&row.line());
        text.push('\n');
        f.write_all(text.as_bytes()).map_err(|e| format!("cannot append to board.tsv: {e}"))?;
        f.sync_data().map_err(|e| format!("cannot sync board.tsv: {e}"))?;
        Ok(row)
    }

    pub fn paths_of(&self, post: &str) -> Result<Vec<String>, String> {
        let t = std::fs::read_to_string(self.dir.join("posts").join(format!("{post}.paths")))
            .map_err(|e| format!("cannot read the paths of {post}: {e}"))?;
        Ok(t.lines().filter(|l| !l.is_empty()).map(String::from).collect())
    }

    pub fn posted(&self, producer: &str, sha: &str) -> Result<Option<Row>, String> {
        Ok(self
            .rows()?
            .into_iter()
            .find(|r| r.kind == "POST" && r.producer == producer && r.sha == sha))
    }

    pub fn answered(&self, post: &str, consumer: &str) -> Result<Option<Row>, String> {
        Ok(self
            .rows()?
            .into_iter()
            .find(|r| r.kind == "ANSWER" && r.post == post && r.consumer == consumer))
    }

    /// The newest POST of `producer`, the base for its next post.
    pub fn last_post(&self, producer: &str) -> Result<Option<Row>, String> {
        Ok(self
            .rows()?
            .into_iter()
            .filter(|r| r.kind == "POST" && r.producer == producer)
            .last())
    }
}

pub fn post_row(producer: &str, sha: &str, base: &str, note: &str) -> Row {
    Row {
        id: String::new(),
        utc: String::new(),
        kind: "POST".into(),
        post: String::new(),
        producer: producer.into(),
        sha: sha.into(),
        base: base.into(),
        consumer: "-".into(),
        verdict: "-".into(),
        note: note.into(),
    }
}

pub fn answer_row(post: &Row, consumer: &str, verdict: &str, note: &str) -> Row {
    Row {
        id: String::new(),
        utc: String::new(),
        kind: "ANSWER".into(),
        post: post.id.clone(),
        producer: post.producer.clone(),
        sha: post.sha.clone(),
        base: post.base.clone(),
        consumer: consumer.into(),
        verdict: verdict.into(),
        note: note.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("aien-board-{name}-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    #[test]
    fn posts_and_answers_append_in_order_and_read_back() {
        let d = tmp("order");
        let b = Board::open(&d).unwrap();
        let sha = "a".repeat(40);
        let base = "b".repeat(40);
        let paths = vec!["src/x.h".to_string(), "docs/y.md".to_string()];
        let p = b.append(post_row("omega", &sha, &base, "2 paths"), Some(&paths)).unwrap();
        assert_eq!((p.id.as_str(), p.post.as_str()), ("B000001", "B000001"));
        let a = b.append(answer_row(&p, "aien-sovereign-core", "SKIP", "tab\tand\nnewline"), None).unwrap();
        assert_eq!(a.id, "B000002");
        let rows = b.rows().unwrap();
        assert_eq!(rows.len(), 2);
        assert_eq!(rows[1].note, "tab and newline");
        assert_eq!(b.paths_of("B000001").unwrap(), paths);
        assert!(b.posted("omega", &sha).unwrap().is_some());
        assert!(b.answered("B000001", "aien-sovereign-core").unwrap().is_some());
        assert!(b.answered("B000001", "physics").unwrap().is_none());
        assert_eq!(b.last_post("omega").unwrap().unwrap().id, "B000001");
        let text = std::fs::read_to_string(d.join("board.tsv")).unwrap();
        assert!(text.starts_with(HEADER));
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn now_utc_has_the_judge_form() {
        let t = now_utc();
        assert!(crate::measure::epoch(&t).is_some(), "{t}");
    }
}
