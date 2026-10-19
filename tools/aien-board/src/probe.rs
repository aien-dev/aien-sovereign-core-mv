//! `aien-board probe`: latency and reliability of the two board backends, measured the same way.
//! One round = post one row, then read the board until the row is visible.
//! - `file`: the Spark TSV board (append under flock, read the file);
//! - `github`: one issue comment on a probe issue, then list the issue's comments through `gh api`.
use crate::board::{post_row, Board};
use serde_json::{json, Value};
use std::path::Path;
use std::process::Command;
use std::time::Instant;

fn stats(name: &str, ms: &[f64], fails: usize) -> Value {
    let mut v = ms.to_vec();
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pick = |q: f64| v.get(((v.len() as f64 - 1.0) * q).round() as usize).copied().unwrap_or(f64::NAN);
    json!({"backend": name, "rounds": ms.len() + fails, "ok": ms.len(), "failed": fails,
           "p50_ms": pick(0.5), "p90_ms": pick(0.9), "max_ms": pick(1.0)})
}

pub fn file(dir: &Path, rounds: usize) -> Result<Value, String> {
    let b = Board::open(dir)?;
    let (mut ms, mut fails) = (Vec::new(), 0);
    for i in 0..rounds {
        let t = Instant::now();
        let sha = format!("{i:040x}");
        let ok = b.append(post_row("probe", &sha, &sha, "probe"), Some(&[]))
            .and_then(|_| b.posted("probe", &sha))
            .map(|r| r.is_some())
            .unwrap_or(false);
        if ok { ms.push(t.elapsed().as_secs_f64() * 1000.0) } else { fails += 1 }
    }
    Ok(stats("file", &ms, fails))
}

fn gh(args: &[&str]) -> Result<String, String> {
    let out = Command::new("gh").args(args).output().map_err(|e| format!("cannot run gh: {e}"))?;
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).lines().last().unwrap_or("").to_string());
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

pub fn github(repo: &str, issue: &str, rounds: usize) -> Result<Value, String> {
    let (mut ms, mut fails, mut errors) = (Vec::new(), 0, Vec::new());
    for i in 0..rounds {
        let tag = format!("aien-board probe {i} {}", crate::board::now_utc());
        let t = Instant::now();
        let r = gh(&["issue", "comment", issue, "-R", repo, "--body", &tag]).and_then(|_| {
            for _ in 0..20 {
                let body = gh(&["api", &format!("repos/{repo}/issues/{issue}/comments?per_page=100"), "--jq", ".[].body"])?;
                if body.lines().any(|l| l == tag) {
                    return Ok(());
                }
            }
            Err("comment not visible after 20 reads".into())
        });
        match r {
            Ok(()) => ms.push(t.elapsed().as_secs_f64() * 1000.0),
            Err(e) => { fails += 1; errors.push(e) }
        }
    }
    let mut s = stats("github", &ms, fails);
    s["errors"] = json!(errors);
    Ok(s)
}
