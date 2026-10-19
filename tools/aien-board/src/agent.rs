//! One repo agent pass. A repo agent owns one repository's checks and never edits another repository:
//! - `post`: its own new main commit goes on the board with every path changed since its last post;
//! - `answer`: every other repository's post that one of its own contracts covers gets one answer.
//!   No watched path changed: SKIP, nothing runs. Otherwise its own check runs against the posted
//!   producer commit, in scratch checkouts the agent owns, and the verdict goes back on the board.
//!
//! Layout under the agent home (default `~/.local/state/aien-board`):
//! `clones/<repo>` (own clones, fetched from the judge's trusted clones, read only there),
//! `wt/<repo>/<sha>` (detached scratch checkouts), `target/<repo>` (one build cache per consumer),
//! `receipts/<post>-<consumer>/` (`receipt.json`, `check.log`, `watch.learned` when the check emits one),
//! `board.tsv`, `posts/`.
use crate::board::{answer_row, post_row, Board, Row};
use crate::contract::{self, Contract};
use serde_json::json;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::Instant;

pub const MAIN: &str = "refs/remotes/upstream/main";

pub struct Agent {
    pub repo: String,
    /// The judge's trusted clones (`~/workspace/pipeline/clones`); only fetched from.
    pub source: PathBuf,
    pub home: PathBuf,
    /// Read own contracts and run own checks from this commit of a local repository instead of own
    /// main (a contract still in review; the slice demo). `(git dir, rev)`.
    pub own: Option<(PathBuf, String)>,
}

fn git(dir: &Path, args: &[&str]) -> Result<String, String> {
    let out = Command::new("git")
        .arg("-C")
        .arg(dir)
        .args(args)
        .env("GIT_OPTIONAL_LOCKS", "0")
        .output()
        .map_err(|e| format!("cannot run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} in {}: {}",
            args.join(" "),
            dir.display(),
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).trim().to_string())
}

impl Agent {
    pub fn board(&self) -> Result<Board, String> {
        Board::open(&self.home)
    }

    /// The agent's own clone of `repo`, with `upstream/*` = the judge clone's view of GitHub.
    pub fn clone_of(&self, repo: &str) -> Result<PathBuf, String> {
        let src = self.source.join(repo);
        let dst = self.home.join("clones").join(repo);
        if !dst.join(".git").exists() {
            std::fs::create_dir_all(self.home.join("clones")).map_err(|e| e.to_string())?;
            git(&self.home, &["clone", "--quiet", "--no-checkout", &src.to_string_lossy(), &dst.to_string_lossy()])?;
        }
        git(
            &dst,
            &["fetch", "--quiet", "--prune", &src.to_string_lossy(), "+refs/remotes/origin/*:refs/remotes/upstream/*"],
        )?;
        Ok(dst)
    }

    /// A detached scratch checkout of `repo` at `sha`, created once and reused.
    pub fn checkout(&self, repo: &str, sha: &str) -> Result<PathBuf, String> {
        let clone = self.clone_of(repo)?;
        let wt = self.home.join("wt").join(repo).join(sha);
        if !wt.join(".git").exists() {
            std::fs::create_dir_all(wt.parent().unwrap_or(&self.home)).map_err(|e| e.to_string())?;
            git(&clone, &["worktree", "add", "--quiet", "--detach", &wt.to_string_lossy(), sha])?;
        }
        let head = git(&wt, &["rev-parse", "HEAD"])?;
        if head != sha {
            return Err(format!("scratch checkout {} is at {head}, not {sha}", wt.display()));
        }
        Ok(wt)
    }

    /// Post `sha` (default: own main) with every path changed since `base` (default: the last post,
    /// else the first parent). Posting the same commit twice returns the first post.
    pub fn post(&self, sha: Option<&str>, base: Option<&str>) -> Result<Row, String> {
        let clone = self.clone_of(&self.repo)?;
        let sha = git(&clone, &["rev-parse", "--verify", &format!("{}^{{commit}}", sha.unwrap_or(MAIN))])?;
        let board = self.board()?;
        if let Some(r) = board.posted(&self.repo, &sha)? {
            return Ok(r);
        }
        let base = match base {
            Some(b) => git(&clone, &["rev-parse", "--verify", &format!("{b}^{{commit}}")])?,
            None => match board.last_post(&self.repo)? {
                Some(r) => r.sha,
                None => git(&clone, &["rev-parse", &format!("{sha}^1")])?,
            },
        };
        let changed = aien_test::plan::changed_between(&clone, &base, &sha)?;
        let note = format!("{} paths changed", changed.len());
        board.append(post_row(&self.repo, &sha, &base, &note), Some(&changed))
    }

    pub fn contracts(&self) -> Result<(String, Vec<Contract>), String> {
        let clone = self.clone_of(&self.repo)?;
        let sha = match &self.own {
            Some((dir, rev)) => {
                git(&clone, &["fetch", "--quiet", &dir.to_string_lossy(), &format!("+{rev}:refs/board/own")])?;
                git(&clone, &["rev-parse", "refs/board/own"])?
            }
            None => git(&clone, &["rev-parse", MAIN])?,
        };
        Ok((sha.clone(), contract::load_all(&clone, &self.repo, &sha)?))
    }

    /// Answer every unanswered post of another repository that one of this repository's contracts
    /// names (`only`: just that post). `dry`: decide and print, run nothing, write nothing.
    pub fn answer(&self, only: Option<&str>, dry: bool) -> Result<Vec<String>, String> {
        let (own_sha, contracts) = self.contracts()?;
        let board = self.board()?;
        let mut out = Vec::new();
        for post in board.rows()?.into_iter().filter(|r| r.kind == "POST" && r.producer != self.repo) {
            if only.is_some_and(|id| id != post.id) {
                continue;
            }
            let Some(c) = contracts.iter().find(|c| c.producer == post.producer) else { continue };
            if board.answered(&post.id, &self.repo)?.is_some() {
                continue;
            }
            let paths = board.paths_of(&post.id)?;
            let touched = contract::touched(c, &paths);
            if touched.is_empty() {
                let note = format!("no watched path changed ({} changed, {} watched)", paths.len(), c.watch.len());
                out.push(format!("{} {} SKIP {note}", post.id, post.producer));
                if !dry {
                    board.append(answer_row(&post, &self.repo, "SKIP", &note), None)?;
                }
                continue;
            }
            if dry {
                out.push(format!("{} {} WOULD-RUN {} on {} touched: {}", post.id, post.producer, c.check, touched.len(), touched.join(" ")));
                continue;
            }
            let (verdict, note) = self.run_check(c, &post, &own_sha, &touched);
            out.push(format!("{} {} {verdict} {note}", post.id, post.producer));
            board.append(answer_row(&post, &self.repo, &verdict, &note), None)?;
        }
        Ok(out)
    }

    fn run_check(&self, c: &Contract, post: &Row, own_sha: &str, touched: &[String]) -> (String, String) {
        let dir = self.home.join("receipts").join(format!("{}-{}", post.id, self.repo));
        let started = crate::board::now_utc();
        let t0 = Instant::now();
        let result = (|| -> Result<i32, String> {
            std::fs::create_dir_all(&dir).map_err(|e| e.to_string())?;
            let consumer = self.checkout(&self.repo, own_sha)?;
            let producer = self.checkout(&post.producer, &post.sha)?;
            for repo in ["physics", "aienos", "aien-architecture", "aien-protocols", "omega"] {
                if self.source.join(repo).exists() {
                    self.clone_of(repo)?;
                }
            }
            let log = std::fs::File::create(dir.join("check.log")).map_err(|e| e.to_string())?;
            let err = log.try_clone().map_err(|e| e.to_string())?;
            let status = Command::new("timeout")
                .args(["--signal=TERM", "--kill-after=30", &c.timeout_secs.to_string(), "bash", &c.check])
                .arg(&producer)
                .arg(&post.sha)
                .arg(&dir)
                .current_dir(&consumer)
                .env("AIEN_BOARD_CLONES", self.home.join("clones"))
                .env("CARGO_TARGET_DIR", self.home.join("target").join(&self.repo))
                .stdout(log)
                .stderr(err)
                .status()
                .map_err(|e| format!("cannot start the check: {e}"))?;
            Ok(status.code().unwrap_or(137))
        })();
        let secs = t0.elapsed().as_secs();
        let (verdict, reason) = match &result {
            Ok(0) => ("PASS", String::new()),
            Ok(124) => ("ERROR", "timeout".to_string()),
            Ok(3) => ("ERROR", "check setup missing (exit 3)".to_string()),
            Ok(code) => ("FAIL", format!("exit {code}")),
            Err(e) => ("ERROR", e.clone()),
        };
        let receipt = json!({
            "schema": "aien-board/ConsumerCheckV1",
            "post": post.id, "producer": post.producer, "producer_sha": post.sha, "producer_base": post.base,
            "consumer": self.repo, "consumer_sha": own_sha, "check": c.check, "touched": touched,
            "started_utc": started, "finished_utc": crate::board::now_utc(), "duration_s": secs,
            "exit": result.as_ref().ok(), "verdict": verdict, "reason": reason,
            "sandbox": "none (slice; switch-over runs it through aien-run-exec, docs/REPO_AGENTS.md)",
        });
        let _ = std::fs::write(dir.join("receipt.json"), serde_json::to_string_pretty(&receipt).unwrap_or_default() + "\n");
        let note = format!("{secs}s {} touched; receipt {}{}", touched.len(), dir.display(),
            if reason.is_empty() { String::new() } else { format!("; {reason}") });
        (verdict.to_string(), note)
    }
}
