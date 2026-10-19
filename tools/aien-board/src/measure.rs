//! `aien-board measure`: what the Spark judge spent, per gate run, on code the change could reach.
//!
//! Inputs are the judge's own records under its state dir, read only:
//! - `journal.jsonl`: one line per judgement (`repo`, `sha`, `base`, `utc`, `verdict`);
//! - `evidence/<repo>/<sha>/*.json`: one EvidenceReceiptV1 per gate run (gate, duration, pool wait,
//!   cache reuse, derived verdict).
//!
//! For every judged head the plan is recomputed with the same planner the worker used
//! (`aien_test::plan::compute`, git objects of the trusted clone, never a checkout), so each receipt
//! gets the reason its gate was selected:
//! - `direct`: a changed path is under one of the gate's own narrow watched inputs (the change can reach it);
//! - `broad`: selected only through a whole-tree input (`src/`, `crates/`, `Makefile`, ...; see BROAD);
//! - `always`: the `.aien/always` set, run whatever changed;
//! - `dependency`: pulled in because a selected gate depends on it;
//! - `dependent`: its dependency changed, its own inputs did not;
//! - `unplanned`: a receipt for a gate the plan does not select (a hand or `--no-cache` run);
//! - `noplan`: the plan could not be recomputed (base or head missing from the clone).
//!
//! Limits, printed in the summary: the plan is recomputed with today's planner over the manifests at
//! that head; one base per head (the newest journal line) even when a head was judged on two bases.
use aien_test::plan::{self, Request};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// Seconds since the epoch for `YYYY-MM-DDTHH:MM:SSZ` (the only form the judge writes).
pub fn epoch(utc: &str) -> Option<i64> {
    let b = utc.as_bytes();
    if b.len() < 20 || b[4] != b'-' || b[10] != b'T' || b[19] != b'Z' {
        return None;
    }
    let n = |s: usize, e: usize| utc.get(s..e)?.parse::<i64>().ok();
    let (y, m, d) = (n(0, 4)?, n(5, 7)?, n(8, 10)?);
    let (hh, mm, ss) = (n(11, 13)?, n(14, 16)?, n(17, 19)?);
    // Days from civil (Howard Hinnant).
    let y2 = if m <= 2 { y - 1 } else { y };
    let era = y2.div_euclid(400);
    let yoe = y2 - era * 400;
    let mp = (m + 9) % 12;
    let doy = (153 * mp + 2) / 5 + d - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    let days = era * 146097 + doe - 719468;
    Some(days * 86400 + hh * 3600 + mm * 60 + ss)
}

#[derive(Debug, Clone)]
pub struct Receipt {
    pub repo: String,
    pub sha: String,
    pub gate: String,
    pub pool: String,
    pub duration_ms: i64,
    pub wait_ms: i64,
    pub reused: bool,
    pub verdict: String,
    pub reason: String,
    pub finished: String,
    pub class: String,
    pub because: String,
}

fn str_at<'a>(v: &'a Value, path: &[&str]) -> &'a str {
    let mut cur = v;
    for p in path {
        cur = &cur[*p];
    }
    cur.as_str().unwrap_or("")
}

fn int_at(v: &Value, path: &[&str]) -> i64 {
    let mut cur = v;
    for p in path {
        cur = &cur[*p];
    }
    cur.as_i64().unwrap_or(0)
}

/// Newest base per (repo, sha) from the journal, for heads judged at or after `since`.
pub fn journal_bases(state: &Path, since: i64) -> Result<BTreeMap<(String, String), String>, String> {
    let text = std::fs::read_to_string(state.join("journal.jsonl"))
        .map_err(|e| format!("cannot read journal.jsonl: {e}"))?;
    let mut out: BTreeMap<(String, String), (i64, String)> = BTreeMap::new();
    for line in text.lines() {
        let Ok(v) = serde_json::from_str::<Value>(line) else { continue };
        let (repo, sha, base) = (str_at(&v, &["repo"]), str_at(&v, &["sha"]), str_at(&v, &["base"]));
        let Some(t) = epoch(str_at(&v, &["utc"])) else { continue };
        if repo.is_empty() || sha.len() != 40 || base.len() != 40 || t < since {
            continue;
        }
        let k = (repo.to_string(), sha.to_string());
        if out.get(&k).is_none_or(|(old, _)| t >= *old) {
            out.insert(k, (t, base.to_string()));
        }
    }
    Ok(out.into_iter().map(|(k, (_, b))| (k, b)).collect())
}

/// Every receipt of one head, unclassified.
pub fn receipts_of(state: &Path, repo: &str, sha: &str) -> Vec<Receipt> {
    let dir = state.join("evidence").join(repo).join(sha);
    let Ok(rd) = std::fs::read_dir(&dir) else { return Vec::new() };
    let mut out = Vec::new();
    for e in rd.flatten() {
        let p = e.path();
        if p.extension().and_then(|s| s.to_str()) != Some("json") {
            continue;
        }
        let Ok(text) = std::fs::read_to_string(&p) else { continue };
        let Ok(v) = serde_json::from_str::<Value>(&text) else { continue };
        if str_at(&v, &["schema"]) != "aien-test/EvidenceReceiptV1" {
            continue;
        }
        out.push(Receipt {
            repo: repo.to_string(),
            sha: sha.to_string(),
            gate: str_at(&v, &["core", "gate"]).to_string(),
            pool: str_at(&v, &["volatile", "pool"]).to_string(),
            duration_ms: int_at(&v, &["volatile", "duration_ms"]),
            wait_ms: int_at(&v, &["volatile", "pool_wait_ms"]),
            reused: v["volatile"]["cache"]["reused"].as_bool().unwrap_or(false),
            verdict: str_at(&v, &["core", "derived_verdict"]).to_string(),
            reason: str_at(&v, &["core", "reason"]).to_string(),
            finished: str_at(&v, &["volatile", "finished_utc"]).to_string(),
            class: String::new(),
            because: String::new(),
        });
    }
    out.sort_by(|a, b| a.finished.cmp(&b.finished).then(a.gate.cmp(&b.gate)));
    out
}

/// Whole-tree inputs: a gate that watches one of these runs for nearly any change in the repository.
pub const BROAD: [&str; 11] = [
    "src/", "Makefile", "mk/", "tests/", "tools/", "tests/gates/", "crates/", "Cargo.toml", "Cargo.lock",
    "rust-toolchain.toml", "scripts/",
];

fn is_broad(input: &str) -> bool {
    let i = input.trim_end_matches('/');
    BROAD.iter().any(|b| b.trim_end_matches('/') == i)
}

/// The selection reason of every gate the planner picks for base..head, plus the changed paths.
/// A `direct` gate none of whose narrow (non-BROAD) watched inputs covers a changed path is
/// reported as `broad`: it ran only because it watches a whole-tree directory.
pub fn reasons(clone: &Path, base: &str, head: &str) -> Result<(BTreeMap<String, String>, Vec<String>), String> {
    let mut req = Request::new(base, head);
    req.operator = true;
    let p = plan::compute(clone, &req)?;
    let head_sha = plan::resolve(clone, head)?;
    let rows = plan::tree_rows(clone, &head_sha)?;
    let (graph, rel_of) = plan::graph_at(clone, &head_sha, &rows)?;
    let mut map = BTreeMap::new();
    for e in p.entries {
        let mut because = e.because.clone();
        if class_of(&because) == "direct" {
            if let Some(i) = graph.index_of(&e.gate) {
                let rel = rel_of.get(&e.gate).cloned().unwrap_or_default();
                let w = plan::watched(&graph.node(i).manifest, &rel, &rows);
                let narrow = p.changed.iter().any(|c| w.iter().any(|inp| !is_broad(inp) && plan::covers(inp, c)));
                if !narrow {
                    because = format!("broad:{because}");
                }
            }
        }
        map.insert(e.gate, because);
    }
    Ok((map, p.changed))
}

pub fn class_of(because: &str) -> &'static str {
    match because {
        "always" => "always",
        "dependency" => "dependency",
        "dependent" => "dependent",
        b if b.starts_with("broad:") => "broad",
        _ => "direct",
    }
}

pub const CLASSES: [&str; 7] = ["direct", "broad", "always", "dependency", "dependent", "unplanned", "noplan"];

/// Measure every head judged since `since`. Returns (receipts, per-head rows).
pub fn run(state: &Path, clones: &Path, since: i64) -> Result<(Vec<Receipt>, Vec<Value>), String> {
    let bases = journal_bases(state, since)?;
    let mut all = Vec::new();
    let mut heads = Vec::new();
    for ((repo, sha), base) in &bases {
        let mut rs = receipts_of(state, repo, sha);
        rs.retain(|r| epoch(&r.finished).is_some_and(|t| t >= since));
        if rs.is_empty() {
            continue;
        }
        let planned = reasons(&clones.join(repo), base, sha);
        let (changed_n, note) = match &planned {
            Ok((_, ch)) => (ch.len() as i64, String::new()),
            Err(e) => (-1, e.lines().next().unwrap_or("").to_string()),
        };
        for r in rs.iter_mut() {
            match &planned {
                Ok((map, _)) => match map.get(&r.gate) {
                    Some(b) => {
                        r.class = class_of(b).to_string();
                        r.because = b.clone();
                    }
                    None => r.class = "unplanned".to_string(),
                },
                Err(_) => r.class = "noplan".to_string(),
            }
        }
        let gate_ms: i64 = rs.iter().map(|r| r.duration_ms).sum();
        let unreach_ms: i64 = rs
            .iter()
            .filter(|r| r.class != "direct" && r.class != "noplan")
            .map(|r| r.duration_ms)
            .sum();
        heads.push(json!({
            "repo": repo, "sha": sha, "base": base, "changed_paths": changed_n,
            "receipts": rs.len(), "gate_ms": gate_ms, "unreachable_ms": unreach_ms,
            "plan_note": note,
        }));
        all.extend(rs);
    }
    Ok((all, heads))
}

/// Totals by repository and by class; FAILs by class (the catches a unit-only run would miss are the
/// FAILs outside `direct`).
pub fn summary(receipts: &[Receipt], heads: &[Value], since: &str) -> Value {
    let mut by_repo: BTreeMap<String, BTreeMap<String, (i64, i64, i64, i64)>> = BTreeMap::new();
    let mut misses = Vec::new();
    for r in receipts {
        let e = by_repo
            .entry(r.repo.clone())
            .or_default()
            .entry(r.class.clone())
            .or_insert((0, 0, 0, 0));
        e.0 += 1;
        e.1 += r.duration_ms;
        e.2 += r.wait_ms;
        if r.verdict == "FAIL" {
            e.3 += 1;
            if r.class != "direct" && r.class != "noplan" {
                misses.push(json!({"repo": r.repo, "sha": r.sha, "gate": r.gate, "class": r.class,
                    "reason": r.reason, "finished": r.finished}));
            }
        }
    }
    let mut repos = serde_json::Map::new();
    let mut total: BTreeMap<String, (i64, i64, i64, i64)> = BTreeMap::new();
    for (repo, m) in &by_repo {
        let mut o = serde_json::Map::new();
        for (c, (n, ms, w, f)) in m {
            o.insert(c.clone(), json!({"receipts": n, "gate_ms": ms, "wait_ms": w, "fails": f}));
            let t = total.entry(c.clone()).or_insert((0, 0, 0, 0));
            t.0 += n;
            t.1 += ms;
            t.2 += w;
            t.3 += f;
        }
        let heads_n = heads.iter().filter(|h| h["repo"] == json!(repo)).count();
        o.insert("heads".into(), json!(heads_n));
        repos.insert(repo.clone(), Value::Object(o));
    }
    let totals: serde_json::Map<String, Value> = total
        .iter()
        .map(|(c, (n, ms, w, f))| (c.clone(), json!({"receipts": n, "gate_ms": ms, "wait_ms": w, "fails": f})))
        .collect();
    json!({
        "schema": "aien-board/MeasureV1",
        "since": since,
        "heads": heads.len(),
        "receipts": receipts.len(),
        "by_repo": repos,
        "totals": totals,
        "fails_outside_direct": misses,
        "limits": [
            "plan recomputed with the current planner over the manifests at each head",
            "one base per head: the newest journal line for that head",
            "gate granularity: a gate whose inputs cover the change counts as direct even if most of its tests are unrelated",
            "duration_ms is gate run time; pool_wait_ms (GPU lock, slots) is reported separately"
        ]
    })
}

/// One TSV line per receipt.
pub fn tsv(receipts: &[Receipt]) -> String {
    let mut s = String::from(
        "repo\tsha\tgate\tclass\tbecause\tpool\tduration_ms\twait_ms\treused\tverdict\treason\tfinished\n",
    );
    for r in receipts {
        s.push_str(&format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n",
            r.repo, r.sha, r.gate, r.class, r.because, r.pool, r.duration_ms, r.wait_ms, r.reused,
            r.verdict, r.reason, r.finished
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn epoch_reads_the_judge_utc_form_and_refuses_others() {
        assert_eq!(epoch("1970-01-01T00:00:00Z"), Some(0));
        assert_eq!(epoch("2026-10-19T20:36:43Z"), Some(1792442203));
        assert_eq!(epoch("2026-10-19 20:36:43"), None);
        assert_eq!(epoch(""), None);
    }

    #[test]
    fn class_of_names_the_four_planner_reasons() {
        assert_eq!(class_of("always"), "always");
        assert_eq!(class_of("dependency"), "dependency");
        assert_eq!(class_of("dependent"), "dependent");
        assert_eq!(class_of("crates/aien-cli/src/main.rs"), "direct");
        assert_eq!(class_of("broad:src/x.c"), "broad");
        assert!(is_broad("src") && is_broad("mk/") && !is_broad("src/turing/"));
    }
}
