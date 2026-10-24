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

//! `aien-test why GATE`: the evidence chain behind one gate, read from the
//! receipts already in the evidence store (ADR 0028 Decision 8).
//!
//! The report is plain lines, one fact per line, starting at the gate and
//! walking down its dependencies. Each gate gets one state:
//!
//! - PASS: the newest receipt for this commit says PASS and everything the
//!   gate depends on is PASS too.
//! - FAILED: the newest receipt for this commit says FAIL.
//! - BLOCKED: hardware or an operator step is missing.
//! - MISSING IMPLEMENTATION: the gate has no `run.exec` to run.
//! - NEEDS RERUN: no receipt for this commit, a NOT_RUN receipt, or a PASS
//!   that rests on a dependency which is not PASS.
//! - STALE: there are receipts, but only for other commits.
//!
//! What this checks, and says so in its output: that a receipt exists, what
//! its verdict is, and whether it was made at the current commit. Since Slice
//! C it also says, per gate, what the digest cache would do now: reuse an
//! earlier receipt, or run again and which of the manifest, program, inputs,
//! machine or dependencies changed (`crate::cache::explain_chain`).

use crate::evidence::{Index, Stored};
use crate::graph::Graph;
use crate::manifest;
use crate::verdict::Verdict;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Pass,
    Failed,
    Blocked,
    MissingImplementation,
    NeedsRerun,
    Stale,
}

impl State {
    pub fn label(self) -> &'static str {
        match self {
            State::Pass => "PASS",
            State::Failed => "FAILED",
            State::Blocked => "BLOCKED",
            State::MissingImplementation => "MISSING IMPLEMENTATION",
            State::NeedsRerun => "NEEDS RERUN",
            State::Stale => "STALE",
        }
    }
}

/// What the store holds for one gate.
enum Evidence<'a> {
    /// Newest receipt made at the current commit.
    Current(&'a Stored),
    /// No receipt for this commit; the newest one for any other commit.
    Other(&'a Stored),
    Absent,
}

fn evidence<'a>(index: &'a Index, commit: &str, gate: &str) -> Evidence<'a> {
    if let Some(r) = index.latest_at(gate, commit) {
        return Evidence::Current(r);
    }
    match index.latest(gate) {
        Some(r) => Evidence::Other(r),
        None => Evidence::Absent,
    }
}

fn own_state(e: &Evidence<'_>) -> State {
    match e {
        Evidence::Current(r) => match r.verdict() {
            Some(Verdict::Pass) => State::Pass,
            Some(Verdict::Fail) => State::Failed,
            Some(Verdict::BlockedHardware) | Some(Verdict::BlockedOperator) => State::Blocked,
            Some(Verdict::MissingImplementation) => State::MissingImplementation,
            Some(Verdict::NotRun) | None => State::NeedsRerun,
        },
        Evidence::Other(_) => State::Stale,
        Evidence::Absent => State::NeedsRerun,
    }
}

/// The first 12 characters, enough to recognise a digest or commit.
fn short(s: &str) -> String {
    s.chars().take(12).collect()
}

fn last_observation(r: &Stored, name: &str) -> Option<String> {
    let all = r.value["core"]["observations"].as_array()?;
    all.iter()
        .rev()
        .find(|o| o["name"] == name)
        .map(|o| o["value"].to_string())
}

/// For a FAIL receipt: the first expectation rule that did not hold, with the
/// value the test reported for it.
fn failing_rule_fact(r: &Stored) -> Option<String> {
    let rules = r.value["core"]["rules"].as_array()?;
    let first = rules.iter().find(|x| x["held"] == false)?;
    let text = first["rule"].as_str()?;
    let seen = manifest::parse_rule(text)
        .ok()
        .and_then(|rule| last_observation(r, &rule.name));
    Some(match seen {
        Some(v) => format!("first rule that did not hold: {text} (observed {v})"),
        None => format!("first rule that did not hold: {text} (no such observation)"),
    })
}

fn receipt_facts(r: &Stored) -> Vec<String> {
    let verdict = r.verdict().map_or("UNKNOWN", |v| v.as_str());
    let mut facts = Vec::new();
    if r.reason().is_empty() {
        facts.push(format!(
            "receipt {} for this commit: {verdict}",
            short(&r.digest)
        ));
    } else {
        facts.push(format!(
            "receipt {} for this commit: {verdict}, reason {}",
            short(&r.digest),
            r.reason()
        ));
    }
    if r.verdict() == Some(Verdict::Fail) {
        match r.reason() {
            "rc_nonzero" => {
                if let Some(code) = r.value["core"]["exit_status"].as_i64() {
                    facts.push(format!("the program exited with status {code}"));
                }
            }
            "timeout" => facts.push("the run hit its time limit and was stopped".to_string()),
            _ => {}
        }
        if let Some(f) = failing_rule_fact(r) {
            facts.push(f);
        }
    }
    let exceeded = r.value["volatile"]["timeout_exceeded"] == true;
    if exceeded && r.value["volatile"]["pool"] == "gb10" && r.reason() != "timeout" {
        facts.push(
            "ran past its time limit; gb10 jobs are never killed, so it was left to finish"
                .to_string(),
        );
    }
    if r.value["core"]["tree_clean_before"] == false {
        facts.push("the working tree had uncommitted changes when it ran".to_string());
    }
    facts
}

struct Report<'a> {
    graph: &'a Graph,
    index: &'a Index,
    commit: &'a str,
    /// The state of every gate in the chain, dependencies included.
    states: BTreeMap<usize, State>,
    shown: BTreeSet<usize>,
    lines: Vec<String>,
    /// Plain lines about the digest cache, per gate.
    cache: &'a BTreeMap<usize, Vec<String>>,
}

impl Report<'_> {
    fn state(&self, i: usize) -> State {
        self.states.get(&i).copied().unwrap_or(State::NeedsRerun)
    }

    fn facts(&self, i: usize) -> Vec<String> {
        let node = self.graph.node(i);
        let ev = evidence(self.index, self.commit, &node.id);
        let mut facts = match &ev {
            Evidence::Current(r) => receipt_facts(r),
            Evidence::Other(r) => vec![format!(
                "newest receipt {} is for commit {}, not the current commit {}",
                short(&r.digest),
                short(r.commit()),
                short(self.commit)
            )],
            Evidence::Absent => vec!["no receipt in the evidence store".to_string()],
        };
        if let Some(c) = self
            .graph
            .index_of(&node.id)
            .and_then(|i| self.cache.get(&i))
        {
            facts.extend(c.iter().cloned());
        }
        if own_state(&ev) == State::Pass {
            for &d in &node.deps {
                if self.state(d) != State::Pass {
                    facts.push(format!(
                        "rests on {}, which is {}",
                        self.graph.node(d).id,
                        self.state(d).label()
                    ));
                }
            }
        }
        facts
    }

    fn render(&mut self, i: usize, depth: usize) {
        let pad = "  ".repeat(depth);
        let id = self.graph.node(i).id.clone();
        let label = self.state(i).label();
        if !self.shown.insert(i) {
            self.lines.push(format!("{pad}{id}: {label} (shown above)"));
            return;
        }
        self.lines.push(format!("{pad}{id}: {label}"));
        for fact in self.facts(i) {
            self.lines.push(format!("{pad}  - {fact}"));
        }
        let deps = self.graph.node(i).deps.clone();
        for d in deps {
            self.render(d, depth + 1);
        }
    }
}

/// The state of every gate `target` rests on, and of `target` itself: a gate
/// whose own receipt is PASS still counts as needing a rerun when something it
/// depends on is not PASS.
pub fn states(graph: &Graph, index: &Index, commit: &str, target: usize) -> BTreeMap<usize, State> {
    let chain = graph.with_dependencies(&[target]);
    let mut out: BTreeMap<usize, State> = BTreeMap::new();
    // Dependencies come before the gates that need them in this order.
    for &i in graph.order() {
        if !chain.contains(&i) {
            continue;
        }
        let node = graph.node(i);
        let own = own_state(&evidence(index, commit, &node.id));
        let deps_pass = node.deps.iter().all(|d| out.get(d) == Some(&State::Pass));
        let state = if own == State::Pass && !deps_pass {
            State::NeedsRerun
        } else {
            own
        };
        out.insert(i, state);
    }
    out
}

/// The full report for `target` at `commit`, as lines to print.
pub fn explain(
    graph: &Graph,
    index: &Index,
    commit: &str,
    target: usize,
    cache: &BTreeMap<usize, Vec<String>>,
) -> Vec<String> {
    let mut report = Report {
        graph,
        index,
        commit,
        states: states(graph, index, commit, target),
        shown: BTreeSet::new(),
        lines: Vec::new(),
        cache,
    };
    report.lines.push(format!(
        "why {} at commit {}",
        graph.node(target).id,
        short(commit)
    ));
    report.render(target, 0);
    report.lines.push(
        "checked: whether a receipt exists, its verdict, and whether it is for the current commit"
            .to_string(),
    );
    report.lines.push(
        "cache: a result is reused only when the manifest, program, declared inputs, machine and dependency receipts all have the same digests as an earlier fresh run; anything else, a dirty tree, or --no-cache runs the gate"
            .to_string(),
    );
    report.lines
}

/// Find the gate a user means by `name`: a gate id, or the path of a `.gate`
/// file (relative to `cwd`) whose gate id is in the graph.
pub fn resolve_target(graph: &Graph, cwd: &Path, name: &str) -> Result<usize, String> {
    if let Some(i) = graph.index_of(name) {
        return Ok(i);
    }
    let path = cwd.join(name);
    if path.is_file() {
        let bytes = fs::read(&path).map_err(|e| format!("cannot read {name}: {e}"))?;
        let m = manifest::parse(&bytes).map_err(|e| e.0)?;
        return graph.index_of(&m.gate).ok_or_else(|| {
            format!(
                "gate {} in {name} is not part of this repository's gate graph",
                m.gate
            )
        });
    }
    Err(format!("no gate named {name}"))
}

/// The state of `target` alone, for callers that only need the verdict word.
pub fn state_of(graph: &Graph, index: &Index, commit: &str, target: usize) -> State {
    states(graph, index, commit, target)
        .get(&target)
        .copied()
        .unwrap_or(State::NeedsRerun)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{build_receipt, Store};
    use crate::graph;
    use crate::testutil::{write_file, TempDir};
    use serde_json::json;
    use std::path::PathBuf;

    const X: &str = "1111111111111111111111111111111111111111";
    const Y: &str = "2222222222222222222222222222222222222222";

    fn gate_text(id: &str, deps: &[&str]) -> String {
        let dep_block = if deps.is_empty() {
            String::new()
        } else {
            format!(
                "depends_on:\n{}",
                deps.iter()
                    .map(|d| format!("  - {d}\n"))
                    .collect::<String>()
            )
        };
        format!(
            "gate: {id}\nmanifest_version: 1\nowner: o\nrequires:\n  - host\nrun:\n  exec: x.sh\nexpects:\n  exit: 0\n  observe: []\n  verdict: PASS\ntimeout: 5s\n{dep_block}"
        )
    }

    fn graph_of(spec: &[(&str, &[&str])]) -> Graph {
        let entries = spec
            .iter()
            .map(|(id, deps)| {
                (
                    PathBuf::from(format!("tests/{id}.gate")),
                    manifest::parse(gate_text(id, deps).as_bytes()).unwrap(),
                )
            })
            .collect();
        graph::build(entries).unwrap()
    }

    /// Write one receipt; `finished` orders receipts of the same gate.
    fn put(store: &Store, gate: &str, commit: &str, verdict: &str, reason: &str, finished: &str) {
        let r = build_receipt(
            json!({
                "gate": gate,
                "commit": commit,
                "derived_verdict": verdict,
                "reason": reason,
                "exit_status": null,
                "observations": [],
                "rules": [],
                "tree_clean_before": true,
                "dependencies": [],
            }),
            json!({ "finished_utc": finished, "pool": "host", "timeout_exceeded": false }),
        );
        store.put_receipt(&r).unwrap();
    }

    fn pass(store: &Store, gate: &str, commit: &str) {
        put(store, gate, commit, "PASS", "", "2026-10-02T10:00:00Z");
    }

    fn text(lines: &[String]) -> String {
        lines.join("\n")
    }

    fn id(g: &Graph, name: &str) -> usize {
        g.index_of(name).unwrap()
    }

    #[test]
    fn a_chain_shows_each_gate_with_its_state() {
        let g = graph_of(&[("G-A", &[]), ("G-B", &["G-A"]), ("G-C", &["G-B"])]);
        let ev = TempDir::new("why1");
        let store = Store::new(ev.path());
        pass(&store, "G-A", X);
        let r = build_receipt(
            json!({
                "gate": "G-B", "commit": X, "derived_verdict": "FAIL", "reason": "rc_nonzero",
                "exit_status": 2,
                "observations": [{ "name": "n", "value": 4 }],
                "rules": [{ "rule": "n == 3", "held": false }],
                "tree_clean_before": true, "dependencies": [],
            }),
            json!({ "finished_utc": "2026-10-02T10:00:01Z", "pool": "host", "timeout_exceeded": false }),
        );
        store.put_receipt(&r).unwrap();
        let index = Index::load(&store);
        let out = text(&explain(&g, &index, X, id(&g, "G-C"), &BTreeMap::new()));
        assert!(out.contains("G-C: NEEDS RERUN"), "{out}");
        assert!(out.contains("no receipt in the evidence store"), "{out}");
        assert!(out.contains("G-B: FAILED"), "{out}");
        assert!(out.contains("the program exited with status 2"), "{out}");
        assert!(
            out.contains("first rule that did not hold: n == 3 (observed 4)"),
            "{out}"
        );
        assert!(out.contains("G-A: PASS"), "{out}");
        // The tree reads top down: C, then B, then A.
        let (c, b, a) = (
            out.find("G-C:").unwrap(),
            out.find("G-B:").unwrap(),
            out.find("G-A:").unwrap(),
        );
        assert!(c < b && b < a, "{out}");
    }

    #[test]
    fn receipts_for_other_commits_are_stale() {
        let g = graph_of(&[("G-A", &[])]);
        let ev = TempDir::new("why2");
        let store = Store::new(ev.path());
        pass(&store, "G-A", Y);
        let index = Index::load(&store);
        assert_eq!(state_of(&g, &index, X, 0), State::Stale);
        let out = text(&explain(&g, &index, X, 0, &BTreeMap::new()));
        assert!(out.contains("G-A: STALE"), "{out}");
        assert!(out.contains("is for commit 222222222222"), "{out}");
        assert_eq!(state_of(&g, &index, Y, 0), State::Pass);
    }

    #[test]
    fn a_pass_resting_on_a_failure_needs_a_rerun() {
        let g = graph_of(&[("G-A", &[]), ("G-B", &["G-A"])]);
        let ev = TempDir::new("why3");
        let store = Store::new(ev.path());
        put(
            &store,
            "G-A",
            X,
            "FAIL",
            "rc_nonzero",
            "2026-10-02T10:00:00Z",
        );
        pass(&store, "G-B", X);
        let index = Index::load(&store);
        assert_eq!(state_of(&g, &index, X, id(&g, "G-B")), State::NeedsRerun);
        let out = text(&explain(&g, &index, X, id(&g, "G-B"), &BTreeMap::new()));
        assert!(out.contains("rests on G-A, which is FAILED"), "{out}");
    }

    #[test]
    fn blocked_and_missing_implementation_have_their_own_states() {
        let g = graph_of(&[("G-A", &[]), ("G-B", &[]), ("G-C", &[])]);
        let ev = TempDir::new("why4");
        let store = Store::new(ev.path());
        put(
            &store,
            "G-A",
            X,
            "BLOCKED_HARDWARE",
            "NO_GB10",
            "2026-10-02T10:00:00Z",
        );
        put(
            &store,
            "G-B",
            X,
            "MISSING_IMPLEMENTATION",
            "NO_RUN_EXEC",
            "2026-10-02T10:00:00Z",
        );
        put(
            &store,
            "G-C",
            X,
            "NOT_RUN",
            "dirty_tree",
            "2026-10-02T10:00:00Z",
        );
        let index = Index::load(&store);
        assert_eq!(state_of(&g, &index, X, id(&g, "G-A")), State::Blocked);
        assert_eq!(
            state_of(&g, &index, X, id(&g, "G-B")),
            State::MissingImplementation
        );
        assert_eq!(state_of(&g, &index, X, id(&g, "G-C")), State::NeedsRerun);
        let out = text(&explain(&g, &index, X, id(&g, "G-A"), &BTreeMap::new()));
        assert!(out.contains("BLOCKED_HARDWARE, reason NO_GB10"), "{out}");
    }

    #[test]
    fn the_newest_receipt_for_the_commit_wins() {
        let g = graph_of(&[("G-A", &[])]);
        let ev = TempDir::new("why5");
        let store = Store::new(ev.path());
        put(
            &store,
            "G-A",
            X,
            "FAIL",
            "rc_nonzero",
            "2026-10-02T10:00:00Z",
        );
        put(&store, "G-A", X, "PASS", "", "2026-10-02T11:00:00Z");
        let index = Index::load(&store);
        assert_eq!(state_of(&g, &index, X, 0), State::Pass);
        put(&store, "G-A", X, "FAIL", "timeout", "2026-10-02T12:00:00Z");
        let index = Index::load(&store);
        assert_eq!(state_of(&g, &index, X, 0), State::Failed);
    }

    #[test]
    fn an_exceeded_gb10_timeout_is_reported() {
        let g = graph_of(&[("G-A", &[])]);
        let ev = TempDir::new("why6");
        let store = Store::new(ev.path());
        let r = build_receipt(
            json!({
                "gate": "G-A", "commit": X, "derived_verdict": "PASS", "reason": "",
                "exit_status": 0, "observations": [], "rules": [],
                "tree_clean_before": true, "dependencies": [],
            }),
            json!({ "finished_utc": "2026-10-02T10:00:00Z", "pool": "gb10", "timeout_exceeded": true }),
        );
        store.put_receipt(&r).unwrap();
        let index = Index::load(&store);
        let out = text(&explain(&g, &index, X, 0, &BTreeMap::new()));
        assert!(out.contains("gb10 jobs are never killed"), "{out}");
        assert!(out.contains("G-A: PASS"), "{out}");
    }

    #[test]
    fn a_gate_reached_twice_is_expanded_once() {
        let g = graph_of(&[
            ("G-A", &[]),
            ("G-B", &["G-A"]),
            ("G-C", &["G-A"]),
            ("G-D", &["G-B", "G-C"]),
        ]);
        let ev = TempDir::new("why7");
        let store = Store::new(ev.path());
        for gate in ["G-A", "G-B", "G-C", "G-D"] {
            pass(&store, gate, X);
        }
        let index = Index::load(&store);
        let lines = explain(&g, &index, X, id(&g, "G-D"), &BTreeMap::new());
        let out = text(&lines);
        assert_eq!(out.matches("G-A: PASS").count(), 2, "{out}");
        assert_eq!(out.matches("(shown above)").count(), 1, "{out}");
        assert_eq!(out.matches("for this commit: PASS").count(), 4, "{out}");
    }

    #[test]
    fn the_report_says_what_it_checks_and_how_the_cache_decides() {
        let g = graph_of(&[("G-A", &[])]);
        let ev = TempDir::new("why8");
        let index = Index::load(&Store::new(ev.path()));
        let lines = explain(&g, &index, X, 0, &BTreeMap::new());
        let out = text(&lines);
        assert!(out.contains("checked: whether a receipt exists"), "{out}");
        assert!(out.contains("cache: a result is reused only when"), "{out}");
        assert!(!out.contains("Slice C"), "{out}");
        assert!(lines[0].starts_with("why G-A at commit 111111111111"));
    }

    #[test]
    fn a_gate_is_found_by_id_or_by_file() {
        let g = graph_of(&[("G-A", &[]), ("G-B", &["G-A"])]);
        let dir = TempDir::new("why10");
        write_file(
            dir.path(),
            "tests/b.gate",
            &gate_text("G-B", &["G-A"]),
            false,
        );
        assert_eq!(resolve_target(&g, dir.path(), "G-B"), Ok(1));
        assert_eq!(resolve_target(&g, dir.path(), "tests/b.gate"), Ok(1));
        assert!(resolve_target(&g, dir.path(), "G-NOPE").is_err());
        write_file(dir.path(), "tests/z.gate", &gate_text("G-Z", &[]), false);
        let e = resolve_target(&g, dir.path(), "tests/z.gate").unwrap_err();
        assert!(e.contains("not part of"), "{e}");
    }

    #[test]
    fn short_never_splits_a_character() {
        assert_eq!(short("abc"), "abc");
        assert_eq!(short("0123456789abcdef"), "0123456789ab");
        assert_eq!(short("ééééééééééééééé").chars().count(), 12);
    }

    #[test]
    fn a_receipt_that_ran_on_a_dirty_tree_says_so() {
        let g = graph_of(&[("G-A", &[])]);
        let ev = TempDir::new("why9");
        let store = Store::new(ev.path());
        let r = build_receipt(
            json!({
                "gate": "G-A", "commit": X, "derived_verdict": "PASS", "reason": "",
                "exit_status": 0, "observations": [], "rules": [],
                "tree_clean_before": false, "dependencies": [],
            }),
            json!({ "finished_utc": "2026-10-02T10:00:00Z", "pool": "host", "timeout_exceeded": false }),
        );
        store.put_receipt(&r).unwrap();
        let index = Index::load(&store);
        let out = text(&explain(&g, &index, X, 0, &BTreeMap::new()));
        assert!(out.contains("uncommitted changes"), "{out}");
    }
}
