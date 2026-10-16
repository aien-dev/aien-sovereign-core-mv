//! Power saving and warm standby for the worker: the pure parts.
//!
//! Everything here is decided from plain data (texts, times, file states) so
//! each rule has a unit test; `worker.rs` wires it to git, GitHub and the
//! sandbox.
//!
//! Markers in the state directory (`~/.local/state/aien-local-ci/`), shared
//! with `cron-tick.sh` and the merge train:
//!
//! ```text
//! QUIET-SINCE   written by the worker (only with --github): one line
//!               "<epoch-secs> <utc>" = the time of the last activity it saw.
//!               Present = nothing has moved since then. Removed the moment
//!               anything moves. Absent or unreadable = not quiet.
//! WAKE          touched by anyone (operator, train, coordinator). Never
//!               removed by a reader: each reader compares its mtime with the
//!               last one it saw. A newer mtime = wake to full cadence now.
//! PAUSE         unchanged (cron-tick exits; `aien-local-ci stop`). The
//!               worker treats PAUSE present -> absent as a wake.
//! warm/<repo>.json   the warm-standby record of target/<repo>/pr (below).
//! ```

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, SystemTime};

pub const QUIET_SINCE: &str = "QUIET-SINCE";
pub const WAKE: &str = "WAKE";
pub const PAUSE: &str = "PAUSE";
/// The label the merge pipeline acts on.
pub const MERGE_LABEL: &str = "ready-to-merge";
/// Default: nothing moved for this long before the expensive work backs off.
pub const DEFAULT_QUIET_AFTER: Duration = Duration::from_secs(600);
/// Default ceiling of the backed-off retry interval (heads that failed to run).
pub const DEFAULT_BACKOFF_CAP: Duration = Duration::from_secs(900);
/// The sliced sleep checks the local wake sources this often.
pub const SLICE: Duration = Duration::from_secs(5);
/// Default per-repository target dir quota (KiB): 120 GiB, as gc.sh.
pub const DEFAULT_TARGET_QUOTA_KB: u64 = 120 << 20;
/// Default free space the filesystem keeps (KiB): 50 GiB.
pub const DEFAULT_DISK_FLOOR_KB: u64 = 50 << 20;

/// The interval of the expensive work (retrying heads that failed to run,
/// which means a fetch and a sibling sync): `base` while anything moved within
/// `quiet_after`, then doubling every idle round up to `cap`. Any activity
/// resets it to `base`. The cheap probe itself never backs off.
#[derive(Debug, Clone)]
pub struct Backoff {
    pub base: Duration,
    pub cap: Duration,
    pub quiet_after: Duration,
    current: Duration,
    last_activity: SystemTime,
}

impl Backoff {
    pub fn new(base: Duration, cap: Duration, quiet_after: Duration, now: SystemTime) -> Backoff {
        let base = base.max(Duration::from_secs(1));
        Backoff {
            base,
            cap: cap.max(base),
            quiet_after,
            current: base,
            last_activity: now,
        }
    }

    /// Something moved: full cadence from now on.
    pub fn activity(&mut self, now: SystemTime) {
        self.current = self.base;
        self.last_activity = now;
    }

    /// One round finished with nothing moving: returns the next interval.
    pub fn idle_round(&mut self, now: SystemTime) -> Duration {
        if self.quiet(now) {
            self.current = (self.current * 2).min(self.cap);
        } else {
            self.current = self.base;
        }
        self.current
    }

    pub fn interval(&self) -> Duration {
        self.current
    }

    pub fn last_activity(&self) -> SystemTime {
        self.last_activity
    }

    /// Nothing moved for at least `quiet_after`.
    pub fn quiet(&self, now: SystemTime) -> bool {
        now.duration_since(self.last_activity)
            .map(|d| d >= self.quiet_after)
            .unwrap_or(false)
    }
}

/// One open pull request head as polled: its sha and whether it carries the
/// merge label.
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrHead {
    pub sha: String,
    pub labeled: bool,
}

/// What one GitHub poll of one repository saw.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Polled {
    pub prs: Vec<PrHead>,
    pub main: Option<String>,
}

/// Parse the pulls query output: one `<sha> <true|false>` per line. Lines
/// that are not that are ignored (the sha check is the safety rule; a bad
/// label field reads as unlabeled).
pub fn parse_pulls(text: &str) -> Vec<PrHead> {
    let mut out: Vec<PrHead> = Vec::new();
    for l in text.lines() {
        let mut f = l.split_whitespace();
        let sha = f.next().unwrap_or("").to_ascii_lowercase();
        if !crate::worker::valid_sha(&sha) {
            continue;
        }
        let labeled = f.next() == Some("true");
        out.push(PrHead { sha, labeled });
    }
    out.sort();
    out.dedup();
    out
}

/// Did anything move between two polls of one repository: a new or changed
/// head, a label added or removed, or main moved. The first poll of a
/// repository (`prev` None) counts as movement.
pub fn moved(prev: Option<&Polled>, now: &Polled) -> bool {
    match prev {
        None => true,
        Some(p) => p != now,
    }
}

/// The heads to run this round in service order: labeled heads first (the
/// merge is waiting on them), then the rest, each group in sha order; heads
/// that already have a bundle are dropped by the caller.
pub fn service_order(mut heads: Vec<PrHead>) -> Vec<PrHead> {
    heads.sort_by(|a, b| b.labeled.cmp(&a.labeled).then(a.sha.cmp(&b.sha)));
    heads
}

// ---------------------------------------------------------------- cheap probe

/// A name GitHub accepts as an owner or repository and that is safe to place
/// inside a GraphQL string literal (no quote, backslash or space).
fn probe_name_ok(s: &str) -> bool {
    !s.is_empty()
        && s.len() <= 100
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_' || b == b'.')
}

/// The one GraphQL query that sees every repository at once: per repository
/// (alias `r<i>`) the main head and the open pull request heads with their
/// labels. `None` when a name is not safe to embed.
pub fn probe_query(repos: &[(String, String)]) -> Option<String> {
    let mut q = String::from("query{");
    for (i, (owner, name)) in repos.iter().enumerate() {
        if !probe_name_ok(owner) || !probe_name_ok(name) {
            return None;
        }
        q.push_str(&format!(
            "r{i}:repository(owner:\"{owner}\",name:\"{name}\"){{ref(qualifiedName:\"refs/heads/main\"){{target{{oid}}}} pullRequests(states:OPEN,first:50){{nodes{{headRefOid labels(first:30){{nodes{{name}}}}}}}}}}"
        ));
    }
    q.push('}');
    Some(q)
}

/// The jq filter applied to the probe answer: `r<i> main <oid>` and
/// `r<i> pr <sha> <true|false>` lines. A repository GitHub could not resolve
/// (null) prints nothing, so the caller falls back to the per-repository poll.
pub fn probe_jq() -> String {
    format!(
        ".data | to_entries[] | select(.value != null) | .key as $k | \
         ((.value.ref.target.oid // empty | \"\\($k) main \\(.)\"), \
          (.value.pullRequests.nodes[] | \"\\($k) pr \\(.headRefOid) \\([.labels.nodes[].name] | index(\"{MERGE_LABEL}\") != null)\"))"
    )
}

/// Parse the probe lines into one `Polled` per repository index. A repository
/// is present only when its `main` line was seen with a valid sha (a partial
/// answer never reads as "no pull requests"); bad lines are ignored.
pub fn parse_probe(text: &str, n: usize) -> Vec<Option<Polled>> {
    let mut out: Vec<Option<Polled>> = vec![None; n];
    let mut prs: Vec<Vec<PrHead>> = vec![Vec::new(); n];
    for l in text.lines() {
        let f: Vec<&str> = l.split_whitespace().collect();
        let Some(i) = f
            .first()
            .and_then(|k| k.strip_prefix('r'))
            .and_then(|k| k.parse::<usize>().ok())
            .filter(|i| *i < n)
        else {
            continue;
        };
        let sha = f.get(2).map(|s| s.to_ascii_lowercase()).unwrap_or_default();
        if !crate::worker::valid_sha(&sha) {
            continue;
        }
        match f.get(1) {
            Some(&"main") if f.len() == 3 => {
                out[i] = Some(Polled {
                    prs: Vec::new(),
                    main: Some(sha),
                })
            }
            Some(&"pr") if f.len() == 4 => prs[i].push(PrHead {
                sha,
                labeled: f[3] == "true",
            }),
            _ => {}
        }
    }
    for (i, p) in out.iter_mut().enumerate() {
        if let Some(p) = p {
            let mut v = std::mem::take(&mut prs[i]);
            v.sort();
            v.dedup();
            p.prs = v;
        }
    }
    out
}

// ---------------------------------------------------------------- target lock

/// The flock file that serialises everything building in `target_dir` (gates
/// and priming): `<lock_dir>/<sanitised>.lock`, where sanitising replaces
/// every byte of the path as given outside `[A-Za-z0-9._-]` with `_`. Same
/// rule as the runner's `resources::target_lock_path` (fix/target-dir-serialise),
/// so both sides lock the same file; the kernel serialises them.
pub fn target_lock_path(lock_dir: &Path, target_dir: &Path) -> PathBuf {
    let s: String = target_dir
        .display()
        .to_string()
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '.' || c == '_' || c == '-' {
                c
            } else {
                '_'
            }
        })
        .collect();
    lock_dir.join(format!("{s}.lock"))
}

// ---------------------------------------------------------------- disk bound

/// What the disk allows before priming `target/<repo>/pr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DiskPlan {
    /// Under the quota and above the floor: prime into the dir as it is.
    Prime,
    /// Over: empty the pr class dir first (the lowest class, a cache the next
    /// run rebuilds); that alone brings the repository under the quota.
    WipeThenPrime,
    /// Over even without the pr class dir (main alone, or the disk floor):
    /// do not prime and leave pr as it is (what PR runs put there is gc.sh's
    /// to bound). No warm claim; runs are cold-from-main until space is freed.
    NoPrime(String),
}

/// The rule (sizes in KiB): the repository's target total (main + pr) must be
/// at most `quota`, and the filesystem must keep `floor` free. The pr class
/// dir is the only thing priming may evict; the main class is never touched.
pub fn disk_plan(total: u64, pr: u64, quota: u64, free: u64, floor: u64) -> DiskPlan {
    if total <= quota && free >= floor {
        return DiskPlan::Prime;
    }
    let rest = total.saturating_sub(pr);
    if rest <= quota && free.saturating_add(pr) >= floor {
        return DiskPlan::WipeThenPrime;
    }
    DiskPlan::NoPrime(format!(
        "target total {} GiB without pr is {} GiB (quota {} GiB), {} GiB free with pr emptied (floor {} GiB)",
        total >> 20,
        rest >> 20,
        quota >> 20,
        free.saturating_add(pr) >> 20,
        floor >> 20
    ))
}

/// While priming: stop when the repository's total went over the quota or
/// the filesystem fell under the floor (sizes in KiB).
pub fn disk_over(total: u64, quota: u64, free: u64, floor: u64) -> Option<String> {
    if total > quota {
        return Some(format!(
            "disk: target total {} GiB over the {} GiB quota",
            total >> 20,
            quota >> 20
        ));
    }
    if free < floor {
        return Some(format!(
            "disk: {} GiB free, under the {} GiB floor",
            free >> 20,
            floor >> 20
        ));
    }
    None
}

// ---------------------------------------------------------------- fetch skip

/// One fetch refspec of `remote.origin.fetch`, glob or exact.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Refspec {
    src: String,
    dst: String,
    glob: bool,
}

fn parse_refspec(s: &str) -> Option<Refspec> {
    let s = s.trim().trim_start_matches('+');
    if s.starts_with('^') {
        return None; // negative refspec: not mapped, fetch as before
    }
    let (src, dst) = s.split_once(':')?;
    let (sg, dg) = (src.matches('*').count(), dst.matches('*').count());
    match (sg, dg) {
        (0, 0) if !src.is_empty() && !dst.is_empty() => Some(Refspec {
            src: src.into(),
            dst: dst.into(),
            glob: false,
        }),
        (1, 1) if src.ends_with('*') && dst.ends_with('*') => Some(Refspec {
            src: src.trim_end_matches('*').into(),
            dst: dst.trim_end_matches('*').into(),
            glob: true,
        }),
        _ => None,
    }
}

fn map_ref(specs: &[Refspec], name: &str) -> Option<String> {
    for s in specs {
        if s.glob {
            if let Some(rest) = name.strip_prefix(&s.src) {
                return Some(format!("{}{rest}", s.dst));
            }
        } else if name == s.src {
            return Some(s.dst.clone());
        }
    }
    None
}

fn sha_lines(text: &str) -> Option<BTreeMap<String, String>> {
    let mut m = BTreeMap::new();
    for l in text.lines() {
        if l.trim().is_empty() {
            continue;
        }
        let (sha, name) = l.split_once(char::is_whitespace)?;
        let (sha, name) = (sha.trim().to_ascii_lowercase(), name.trim());
        if !crate::worker::valid_sha(&sha) || name.is_empty() {
            return None;
        }
        if name.ends_with("^{}") {
            continue; // peeled tag line: the tag object line is compared
        }
        m.insert(name.to_string(), sha);
    }
    Some(m)
}

/// Would `git fetch` change anything? `ls_remote` is `git ls-remote origin`,
/// `refspecs` the values of `remote.origin.fetch`, `local` is
/// `git for-each-ref --format='%(objectname) %(refname)'`.
///
/// `Some(true)` only when every remote ref a refspec maps, and every remote
/// tag, is present locally at the same sha (local extras are ignored: fetch
/// does not prune). `None` when anything cannot be read or mapped: the caller
/// then fetches exactly as before. Remote refs no refspec maps
/// (`refs/pull/*` with the default refspec, `HEAD`) are not fetched by
/// `fetch --all` and are ignored.
pub fn refs_unchanged(ls_remote: &str, refspecs: &[String], local: &str) -> Option<bool> {
    if refspecs.is_empty() {
        return None;
    }
    let specs: Vec<Refspec> = refspecs
        .iter()
        .map(|s| parse_refspec(s))
        .collect::<Option<Vec<_>>>()?;
    let remote = sha_lines(ls_remote)?;
    let local = sha_lines(local)?;
    if remote.is_empty() {
        return None;
    }
    for (name, sha) in &remote {
        let dst = if name.starts_with("refs/tags/") {
            Some(name.clone())
        } else {
            map_ref(&specs, name)
        };
        if let Some(dst) = dst {
            if local.get(&dst) != Some(sha) {
                return Some(false);
            }
        }
    }
    Some(true)
}

// -------------------------------------------------------------- warm standby

/// One priming line of the operator's prime map
/// (`tools/aien-local-ci/prime`): `<repo> [K=V ...] cargo <args...>`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PrimeLine {
    pub env: Vec<(String, String)>,
    pub args: Vec<String>,
}

/// Names the exec helper refuses (or that would change what the sandbox is).
const ENV_REFUSED: [&str; 8] = [
    "PATH",
    "HOME",
    "CARGO_HOME",
    "RUSTUP_HOME",
    "CARGO_TARGET_DIR",
    "BASH_ENV",
    "ENV",
    "RUSTC_WRAPPER",
];

fn env_name_ok(k: &str) -> bool {
    let mut b = k.bytes();
    matches!(b.next(), Some(c) if c.is_ascii_uppercase() || c == b'_')
        && b.all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == b'_')
        && !k.starts_with("LD_")
        && !k.starts_with("AIEN_RUN")
        && !ENV_REFUSED.contains(&k)
}

/// Parse the prime map and return the lines of `repo`. Every line is checked
/// (a broken map refuses priming for every repository). The program is
/// always `cargo`; arguments are plain words without quoting, `=`-free in
/// the program position, never starting the line with a flag.
pub fn parse_prime(text: &str, repo: &str) -> Result<Vec<PrimeLine>, String> {
    let mut out = Vec::new();
    for (n, raw) in text.lines().enumerate() {
        let line = raw.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let bad = |why: &str| format!("prime map line {}: {why}: {raw:?}", n + 1);
        let f: Vec<&str> = line.split_whitespace().collect();
        if !crate::worker::valid_repo_key(f[0]) {
            return Err(bad("the repository must be a plain name"));
        }
        let mut env = Vec::new();
        let mut i = 1;
        while i < f.len() && f[i].contains('=') {
            let (k, v) = f[i].split_once('=').unwrap_or(("", ""));
            if !env_name_ok(k) {
                return Err(bad("environment name refused"));
            }
            env.push((k.to_string(), v.to_string()));
            i += 1;
        }
        if f.get(i) != Some(&"cargo") {
            return Err(bad("the program must be cargo"));
        }
        let args: Vec<String> = f[i + 1..].iter().map(|s| s.to_string()).collect();
        if args.is_empty() || args[0].starts_with('-') {
            return Err(bad("needs a cargo subcommand"));
        }
        if !["build", "check", "clippy", "test"].contains(&args[0].as_str()) {
            return Err(bad("cargo subcommand must be build, check, clippy or test"));
        }
        if args[0] == "test" && !args.iter().any(|a| a == "--no-run") {
            return Err(bad("cargo test primes only with --no-run"));
        }
        if f[0] == repo {
            out.push(PrimeLine { env, args });
        }
    }
    Ok(out)
}

/// What `warm/<repo>.json` says about `target/<repo>/pr`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WarmRecord {
    pub primed_from_main: String,
    pub utc: String,
}

pub fn warm_path(state_root: &Path, repo: &str) -> PathBuf {
    state_root.join("warm").join(format!("{repo}.json"))
}

/// The record, only when it is complete and well formed.
pub fn read_warm(state_root: &Path, repo: &str) -> Option<WarmRecord> {
    let t = fs::read_to_string(warm_path(state_root, repo)).ok()?;
    let v: serde_json::Value = serde_json::from_str(&t).ok()?;
    if v["complete"] != serde_json::Value::Bool(true) || v["class"] != "pr" {
        return None;
    }
    let sha = v["primed_from_main"].as_str()?.to_string();
    crate::worker::valid_sha(&sha).then(|| WarmRecord {
        primed_from_main: sha,
        utc: v["utc"].as_str().unwrap_or("").to_string(),
    })
}

pub fn write_warm(
    state_root: &Path,
    repo: &str,
    rec: &WarmRecord,
    lines: usize,
) -> Result<(), String> {
    let p = warm_path(state_root, repo);
    let dir = p.parent().unwrap_or(state_root);
    fs::create_dir_all(dir).map_err(|e| format!("cannot create {}: {e}", dir.display()))?;
    let v = serde_json::json!({
        "class": "pr",
        "complete": true,
        "primed_from_main": rec.primed_from_main,
        "utc": rec.utc,
        "lines": lines,
    });
    let tmp = dir.join(format!(".{repo}.json.tmp"));
    fs::write(&tmp, v.to_string()).map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, &p).map_err(|e| format!("cannot write {}: {e}", p.display()))
}

pub fn clear_warm(state_root: &Path, repo: &str) {
    let _ = fs::remove_file(warm_path(state_root, repo));
}

/// The bundle member `target`: which cargo target dir the gates used and
/// whether it was warm (primed from main) when the run started.
pub fn target_member(class: &str, warm: Option<&WarmRecord>, cold: bool) -> serde_json::Value {
    let warm = if cold { None } else { warm };
    serde_json::json!({
        "class": class,
        "warm": warm.is_some(),
        "primed_from_main": warm.map(|w| w.primed_from_main.clone()),
        "primed_utc": warm.map(|w| w.utc.clone()),
        "cold": cold,
    })
}

// ------------------------------------------------------------------- markers

/// Write QUIET-SINCE (atomic) with the time of the last activity.
pub fn write_quiet_since(state_root: &Path, last_activity: SystemTime) -> Result<(), String> {
    let secs = last_activity
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let p = state_root.join(QUIET_SINCE);
    let tmp = state_root.join(format!(".{QUIET_SINCE}.tmp"));
    fs::write(
        &tmp,
        format!("{secs} {}\n", crate::evidence::utc_string(secs)),
    )
    .map_err(|e| format!("cannot write {}: {e}", tmp.display()))?;
    fs::rename(&tmp, &p).map_err(|e| format!("cannot write {}: {e}", p.display()))
}

pub fn remove_quiet_since(state_root: &Path) {
    let _ = fs::remove_file(state_root.join(QUIET_SINCE));
}

/// The epoch in QUIET-SINCE, None when absent or not a number.
pub fn read_quiet_since(state_root: &Path) -> Option<u64> {
    fs::read_to_string(state_root.join(QUIET_SINCE))
        .ok()?
        .split_whitespace()
        .next()?
        .parse()
        .ok()
}

/// Watches the local wake sources without any network or process: WAKE
/// touched since last seen, PAUSE removed, a trigger file waiting.
#[derive(Debug, Clone)]
pub struct WakeWatch {
    root: PathBuf,
    wake_seen: Option<SystemTime>,
    paused: bool,
}

fn mtime(p: &Path) -> Option<SystemTime> {
    fs::metadata(p).and_then(|m| m.modified()).ok()
}

impl WakeWatch {
    pub fn new(state_root: &Path) -> WakeWatch {
        WakeWatch {
            root: state_root.to_path_buf(),
            wake_seen: mtime(&state_root.join(WAKE)),
            paused: state_root.join(PAUSE).exists(),
        }
    }

    /// Why to wake now, if anything; each WAKE touch and each PAUSE removal
    /// is reported once.
    pub fn check(&mut self) -> Option<&'static str> {
        let w = mtime(&self.root.join(WAKE));
        let woke = match (w, self.wake_seen) {
            (Some(a), Some(b)) => a > b,
            (Some(_), None) => true,
            _ => false,
        };
        if w.is_some() {
            self.wake_seen = w;
        }
        let paused = self.root.join(PAUSE).exists();
        let unpaused = self.paused && !paused;
        self.paused = paused;
        if woke {
            Some("WAKE touched")
        } else if unpaused {
            Some("PAUSE removed")
        } else if trigger_waiting(&self.root.join("trigger")) {
            Some("trigger file")
        } else {
            None
        }
    }

    pub fn paused(&self) -> bool {
        self.paused
    }
}

fn trigger_waiting(dir: &Path) -> bool {
    fs::read_dir(dir)
        .map(|rd| {
            rd.flatten().any(|e| {
                // Only a name the worker would serve (`<repo>.<sha40>`): a stray
                // file is never removed, so it must not wake the loop forever.
                let n = e.file_name().to_string_lossy().into_owned();
                n.rsplit_once('.').is_some_and(|(r, s)| {
                    crate::worker::valid_repo_key(r)
                        && crate::worker::valid_sha(&s.to_ascii_lowercase())
                })
            })
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    const A: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const B: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
    const C: &str = "cccccccccccccccccccccccccccccccccccccccc";

    fn t(s: u64) -> SystemTime {
        SystemTime::UNIX_EPOCH + Duration::from_secs(1_000_000 + s)
    }

    #[test]
    fn probe_query_covers_every_repo_and_refuses_unsafe_names() {
        let q = probe_query(&[
            ("aien-dev".into(), "aienos".into()),
            ("aien-spark".into(), "aien-sovereign-core".into()),
        ])
        .unwrap();
        assert!(q.starts_with("query{r0:repository(owner:\"aien-dev\",name:\"aienos\")"));
        assert!(q.contains("r1:repository(owner:\"aien-spark\",name:\"aien-sovereign-core\")"));
        assert!(q.contains("refs/heads/main") && q.contains("states:OPEN,first:50"));
        assert_eq!(q.matches('{').count(), q.matches('}').count());
        for bad in ["a\"b", "a b", "a\\b", "", "a{b"] {
            assert!(
                probe_query(&[("o".into(), bad.into())]).is_none(),
                "{bad:?}"
            );
        }
        assert!(probe_jq().contains("index(\"ready-to-merge\") != null"));
    }

    #[test]
    fn parse_probe_needs_main_and_ignores_bad_lines() {
        let text = format!(
            "r0 main {A}\nr0 pr {B} true\nr0 pr {C} false\nr0 pr {B} true\n\
             r1 pr {C} true\nr2 main {A}\nr9 main {A}\nx main {A}\nr2 pr nope true\nr2 main short\n"
        );
        let p = parse_probe(&text, 3);
        assert_eq!(
            p[0],
            Some(Polled {
                prs: vec![
                    PrHead {
                        sha: B.into(),
                        labeled: true
                    },
                    PrHead {
                        sha: C.into(),
                        labeled: false
                    }
                ],
                main: Some(A.into())
            })
        );
        assert_eq!(p[1], None, "no main line: fall back, never 'no PRs'");
        assert_eq!(
            p[2],
            Some(Polled {
                prs: vec![],
                main: Some(A.into())
            })
        );
        assert_eq!(parse_probe("", 2), vec![None, None]);
    }

    #[test]
    fn target_lock_path_matches_the_runner_rule() {
        assert_eq!(
            target_lock_path(
                Path::new("/s/locks/target"),
                Path::new("/var/lib/aien-run/target/aien-sovereign-core/pr")
            ),
            PathBuf::from("/s/locks/target/_var_lib_aien-run_target_aien-sovereign-core_pr.lock")
        );
        // Nothing can climb out of the lock dir.
        let p = target_lock_path(Path::new("/l"), Path::new("../../x y"));
        assert_eq!(p, PathBuf::from("/l/.._.._x_y.lock"));
        assert_eq!(p.parent(), Some(Path::new("/l")));
    }

    #[test]
    fn disk_plan_evicts_pr_first_and_fails_closed_to_cold() {
        let g = 1u64 << 20; // KiB per GiB
        let (quota, floor) = (120 * g, 50 * g);
        assert_eq!(
            disk_plan(100 * g, 40 * g, quota, 200 * g, floor),
            DiskPlan::Prime
        );
        assert_eq!(
            disk_plan(120 * g, 40 * g, quota, 200 * g, floor),
            DiskPlan::Prime
        );
        // Over the quota, main alone under it: empty pr, then prime.
        assert_eq!(
            disk_plan(150 * g, 60 * g, quota, 200 * g, floor),
            DiskPlan::WipeThenPrime
        );
        // Under the floor; emptying pr gives the room back.
        assert_eq!(
            disk_plan(100 * g, 40 * g, quota, 20 * g, floor),
            DiskPlan::WipeThenPrime
        );
        // Main alone over the quota (the live sc state on 2026-10-16: 122 + 148).
        assert!(matches!(
            disk_plan(270 * g, 148 * g, quota, 150 * g, floor),
            DiskPlan::NoPrime(w) if w.contains("122 GiB")
        ));
        // The disk floor even with pr emptied.
        assert!(matches!(
            disk_plan(60 * g, 10 * g, quota, 5 * g, floor),
            DiskPlan::NoPrime(_)
        ));
        assert_eq!(disk_over(120 * g, quota, 50 * g, floor), None);
        assert!(disk_over(120 * g + 1, quota, 60 * g, floor)
            .unwrap()
            .contains("quota"));
        assert!(disk_over(10 * g, quota, 49 * g, floor)
            .unwrap()
            .contains("floor"));
    }

    #[test]
    fn backoff_holds_base_until_quiet_then_doubles_to_the_cap() {
        let mut b = Backoff::new(
            Duration::from_secs(60),
            Duration::from_secs(900),
            Duration::from_secs(600),
            t(0),
        );
        assert_eq!(b.idle_round(t(60)), Duration::from_secs(60));
        assert_eq!(b.idle_round(t(599)), Duration::from_secs(60));
        assert!(!b.quiet(t(599)));
        assert_eq!(b.idle_round(t(600)), Duration::from_secs(120));
        assert_eq!(b.idle_round(t(720)), Duration::from_secs(240));
        assert_eq!(b.idle_round(t(960)), Duration::from_secs(480));
        assert_eq!(b.idle_round(t(1440)), Duration::from_secs(900));
        assert_eq!(b.idle_round(t(2340)), Duration::from_secs(900), "capped");
        b.activity(t(2400));
        assert_eq!(b.interval(), Duration::from_secs(60), "activity resets");
        assert_eq!(b.idle_round(t(2460)), Duration::from_secs(60));
    }

    #[test]
    fn backoff_never_goes_below_one_second_or_cap_below_base() {
        let mut b = Backoff::new(Duration::ZERO, Duration::ZERO, Duration::ZERO, t(0));
        assert_eq!(b.interval(), Duration::from_secs(1));
        assert_eq!(b.idle_round(t(10)), Duration::from_secs(1));
        // A clock that goes backwards is never "quiet".
        let b = Backoff::new(
            Duration::from_secs(60),
            Duration::from_secs(900),
            Duration::from_secs(1),
            t(100),
        );
        assert!(!b.quiet(t(0)));
    }

    #[test]
    fn pulls_parse_labels_and_refuse_bad_shas() {
        let txt = format!("{A} true\n{B} false\nnot-a-sha true\n{C}\n{A} true\nZZZ\n");
        let p = parse_pulls(&txt);
        assert_eq!(
            p,
            vec![
                PrHead {
                    sha: A.into(),
                    labeled: true
                },
                PrHead {
                    sha: B.into(),
                    labeled: false
                },
                PrHead {
                    sha: C.into(),
                    labeled: false
                },
            ]
        );
        assert!(parse_pulls("").is_empty());
        assert!(
            !parse_pulls(&format!("{A} TRUE"))[0].labeled,
            "only exactly true"
        );
    }

    #[test]
    fn movement_is_any_head_label_or_main_change() {
        let base = Polled {
            prs: vec![PrHead {
                sha: A.into(),
                labeled: false,
            }],
            main: Some(B.into()),
        };
        assert!(moved(None, &base), "first poll counts");
        assert!(!moved(Some(&base), &base.clone()), "same poll is quiet");
        let mut l = base.clone();
        l.prs[0].labeled = true;
        assert!(moved(Some(&base), &l), "label added");
        assert!(moved(Some(&l), &base), "label removed");
        let mut h = base.clone();
        h.prs[0].sha = C.into();
        assert!(moved(Some(&base), &h), "new head");
        let mut m = base.clone();
        m.main = Some(C.into());
        assert!(moved(Some(&base), &m), "main moved");
    }

    #[test]
    fn labeled_heads_are_served_first() {
        let o = service_order(vec![
            PrHead {
                sha: C.into(),
                labeled: false,
            },
            PrHead {
                sha: B.into(),
                labeled: true,
            },
            PrHead {
                sha: A.into(),
                labeled: false,
            },
        ]);
        let s: Vec<&str> = o.iter().map(|h| h.sha.as_str()).collect();
        assert_eq!(s, vec![B, A, C]);
    }

    fn default_spec() -> Vec<String> {
        vec!["+refs/heads/*:refs/remotes/origin/*".to_string()]
    }

    #[test]
    fn fetch_skip_only_when_every_mapped_ref_matches() {
        let remote = format!(
            "{A}\tHEAD\n{A}\trefs/heads/main\n{B}\trefs/heads/feat\n{C}\trefs/pull/7/head\n{B}\trefs/tags/v1\n{C}\trefs/tags/v1^{{}}\n"
        );
        let local = format!(
            "{A} refs/remotes/origin/main\n{B} refs/remotes/origin/feat\n{B} refs/tags/v1\n{C} refs/remotes/origin/gone\n{A} refs/heads/main\n"
        );
        assert_eq!(
            refs_unchanged(&remote, &default_spec(), &local),
            Some(true),
            "peeled and pull lines ignored, local extras ignored"
        );
        // branch moved
        let moved = remote.replace(
            &format!("{B}\trefs/heads/feat"),
            &format!("{C}\trefs/heads/feat"),
        );
        assert_eq!(refs_unchanged(&moved, &default_spec(), &local), Some(false));
        // branch added
        let added = format!("{remote}{C}\trefs/heads/new\n");
        assert_eq!(refs_unchanged(&added, &default_spec(), &local), Some(false));
        // tag moved / tag added
        let tag = remote.replace(
            &format!("{B}\trefs/tags/v1\n"),
            &format!("{A}\trefs/tags/v1\n"),
        );
        assert_eq!(refs_unchanged(&tag, &default_spec(), &local), Some(false));
        let tag2 = format!("{remote}{A}\trefs/tags/v2\n");
        assert_eq!(refs_unchanged(&tag2, &default_spec(), &local), Some(false));
    }

    #[test]
    fn fetch_skip_maps_a_configured_pull_refspec() {
        let specs = vec![
            "+refs/heads/*:refs/remotes/origin/*".to_string(),
            "+refs/pull/*/head:refs/remotes/origin/pr/*".to_string(),
        ];
        // A refspec with the glob in the middle cannot be mapped: fetch as before.
        assert_eq!(
            refs_unchanged(
                &format!("{A}\trefs/heads/main\n"),
                &specs,
                &format!("{A} refs/remotes/origin/main\n")
            ),
            None
        );
        let specs = vec![
            "+refs/heads/*:refs/remotes/origin/*".to_string(),
            "+refs/pull/*:refs/remotes/origin/pull/*".to_string(),
        ];
        let remote = format!("{A}\trefs/heads/main\n{C}\trefs/pull/7/head\n");
        let local = format!("{A} refs/remotes/origin/main\n");
        assert_eq!(
            refs_unchanged(&remote, &specs, &local),
            Some(false),
            "pull ref mapped and missing"
        );
        let local = format!("{local}{C} refs/remotes/origin/pull/7/head\n");
        assert_eq!(refs_unchanged(&remote, &specs, &local), Some(true));
    }

    #[test]
    fn fetch_skip_refuses_what_it_cannot_read() {
        let r = format!("{A}\trefs/heads/main\n");
        let l = format!("{A} refs/remotes/origin/main\n");
        assert_eq!(refs_unchanged(&r, &[], &l), None, "no refspec");
        assert_eq!(
            refs_unchanged(&r, &["^refs/heads/x".into()], &l),
            None,
            "negative refspec"
        );
        assert_eq!(
            refs_unchanged(&r, &["refs/heads/*:refs/x".into()], &l),
            None,
            "unbalanced glob"
        );
        assert_eq!(refs_unchanged("garbage line\n", &default_spec(), &l), None);
        assert_eq!(refs_unchanged(&r, &default_spec(), "zz refs/x\n"), None);
        assert_eq!(
            refs_unchanged("", &default_spec(), &l),
            None,
            "empty remote answer"
        );
        // an exact refspec maps
        assert_eq!(
            refs_unchanged(&r, &["refs/heads/main:refs/remotes/origin/main".into()], &l),
            Some(true)
        );
    }

    #[test]
    fn prime_map_selects_the_repo_and_refuses_bad_lines() {
        let m = "# comment\nsc CI=true X_1=a cargo test --workspace --no-run\nsc cargo check --workspace --all-targets\nomega cargo build\n";
        let p = parse_prime(m, "sc").unwrap();
        assert_eq!(p.len(), 2);
        assert_eq!(
            p[0].env,
            vec![("CI".into(), "true".into()), ("X_1".into(), "a".into())]
        );
        assert_eq!(p[0].args, vec!["test", "--workspace", "--no-run"]);
        assert!(parse_prime(m, "nobody").unwrap().is_empty());
        for bad in [
            "sc bash -c x",
            "sc cargo",
            "sc cargo --version",
            "sc cargo run",
            "sc cargo install x",
            "sc cargo test --workspace",
            "sc PATH=/x cargo build",
            "sc LD_PRELOAD=/x cargo build",
            "sc CARGO_TARGET_DIR=/x cargo build",
            "sc AIEN_RUN_NESTED=0 cargo build",
            "sc lower=1 cargo build",
            "../x cargo build",
            "sc =1 cargo build",
        ] {
            assert!(
                parse_prime(&format!("{m}{bad}\n"), "omega").is_err(),
                "{bad} must refuse"
            );
        }
    }

    fn scratch(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("aien-idle-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn warm_record_round_trip_and_incomplete_records_are_cold() {
        let d = scratch("warm");
        assert_eq!(read_warm(&d, "sc"), None);
        let rec = WarmRecord {
            primed_from_main: A.into(),
            utc: "2026-10-16T10:00:00Z".into(),
        };
        write_warm(&d, "sc", &rec, 3).unwrap();
        assert_eq!(read_warm(&d, "sc"), Some(rec.clone()));
        let m = target_member("pr", Some(&rec), false);
        assert_eq!(m["warm"], true);
        assert_eq!(m["primed_from_main"], A);
        let c = target_member("pr", Some(&rec), true);
        assert_eq!(c["warm"], false, "a cold run is never warm");
        assert_eq!(c["cold"], true);
        assert_eq!(c["primed_from_main"], serde_json::Value::Null);
        fs::write(
            warm_path(&d, "sc"),
            format!(r#"{{"class":"pr","complete":false,"primed_from_main":"{A}"}}"#),
        )
        .unwrap();
        assert_eq!(read_warm(&d, "sc"), None, "incomplete");
        fs::write(
            warm_path(&d, "sc"),
            format!(r#"{{"class":"main","complete":true,"primed_from_main":"{A}"}}"#),
        )
        .unwrap();
        assert_eq!(read_warm(&d, "sc"), None, "wrong class");
        fs::write(warm_path(&d, "sc"), "garbage").unwrap();
        assert_eq!(read_warm(&d, "sc"), None);
        clear_warm(&d, "sc");
        assert!(!warm_path(&d, "sc").exists());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn quiet_since_round_trip() {
        let d = scratch("quiet");
        assert_eq!(read_quiet_since(&d), None);
        write_quiet_since(&d, t(5)).unwrap();
        assert_eq!(read_quiet_since(&d), Some(1_000_005));
        fs::write(d.join(QUIET_SINCE), "nonsense\n").unwrap();
        assert_eq!(read_quiet_since(&d), None);
        remove_quiet_since(&d);
        assert!(!d.join(QUIET_SINCE).exists());
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn wake_watch_sees_wake_pause_removal_and_triggers_once() {
        let d = scratch("wake");
        fs::create_dir_all(d.join("trigger")).unwrap();
        let mut w = WakeWatch::new(&d);
        assert_eq!(w.check(), None);
        fs::write(d.join(WAKE), "").unwrap();
        assert_eq!(w.check(), Some("WAKE touched"));
        assert_eq!(w.check(), None, "reported once");
        fs::write(d.join(PAUSE), "").unwrap();
        assert_eq!(w.check(), None, "pausing is no wake");
        assert!(w.paused());
        fs::remove_file(d.join(PAUSE)).unwrap();
        assert_eq!(w.check(), Some("PAUSE removed"));
        assert_eq!(w.check(), None);
        fs::write(d.join("trigger/.tmp"), "").unwrap();
        assert_eq!(w.check(), None, "dot files are no trigger");
        fs::write(d.join("trigger/stray-file"), "").unwrap();
        fs::write(d.join("trigger/sc.not-a-sha"), "").unwrap();
        assert_eq!(
            w.check(),
            None,
            "names the worker would not serve are no trigger"
        );
        fs::write(d.join(format!("trigger/sc.{A}")), "").unwrap();
        assert_eq!(w.check(), Some("trigger file"));
        // WAKE touched again later (mtime moves forward)
        let later = SystemTime::now() + Duration::from_secs(5);
        let f = fs::File::options().write(true).open(d.join(WAKE)).unwrap();
        f.set_modified(later).unwrap();
        assert_eq!(w.check(), Some("WAKE touched"));
        let _ = fs::remove_dir_all(&d);
    }
}

#[cfg(test)]
mod shipped_map {
    /// The prime map shipped beside the sibling map parses.
    #[test]
    fn shipped_prime_map_parses() {
        let text = include_str!("../../aien-local-ci/prime");
        for repo in ["aien-sovereign-core", "aienos"] {
            let l = super::parse_prime(text, repo).unwrap();
            assert!(!l.is_empty(), "{repo} has prime lines");
        }
    }
}
