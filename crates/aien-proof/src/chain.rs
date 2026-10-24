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

//! Receipt dependency graph verification (`aien-proof verify-chain`).
//!
//! Rules, all fail closed:
//! - dependency IDs are immutable hashes; bytes that do not hash to the
//!   claimed ID are rejected, so substitution and duplicates with different
//!   bytes cannot pass;
//! - cycles are rejected;
//! - a missing dependency yields INCOMPLETE, never PASS;
//! - validation traverses the whole reachable graph;
//! - a PASS root requires every reachable receipt to be PASS; a FAIL
//!   dependency fails the chain, BLOCKED or INCOMPLETE dependencies block it;
//! - TEST_ONLY_TRUST ancestry can never support a PRODUCTION receipt;
//! - a QEMU tier receipt can never satisfy a physical tier requirement
//!   (enforced per requirement in [`crate::gate`]).

use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

use crate::evidence::{load_receipt, verify_with_store, Receipt, Report, Tier, Verdict};

/// Full reachable graph report for one root receipt.
#[derive(Debug, Clone)]
pub struct ChainReport {
    pub root: String,
    pub status: Verdict,
    pub lines: Vec<ChainLine>,
    pub reasons: Vec<String>,
}

/// One visited receipt in traversal order.
#[derive(Debug, Clone)]
pub struct ChainLine {
    pub id: String,
    pub kind: String,
    pub tier: Tier,
    pub status: Verdict,
    pub depth: usize,
    pub note: String,
}

/// Dependency resolver: receipt ID to verified receipt plus its report.
pub type Resolver<'a> = &'a dyn Fn(&str) -> Result<(Receipt, Report), String>;

/// Verify the complete reachable receipt graph for the receipt file at
/// `path`. Dependencies resolve from `store`.
pub fn verify_chain(path: &Path, store: &Path) -> Result<ChainReport, String> {
    let text = std::fs::read(path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
    let (root, root_report) = verify_with_store(&text, store)?;
    let resolve = |id: &str| -> Result<(Receipt, Report), String> {
        let receipt = load_receipt(store, id)?;
        let bytes = std::fs::read(store.join("receipts").join(format!("{id}.json")))
            .map_err(|e| e.to_string())?;
        let (_, report) = verify_with_store(&bytes, store)?;
        Ok((receipt, report))
    };
    Ok(verify_graph(&root, &root_report, store, &resolve))
}

struct Walker<'a> {
    store: &'a Path,
    resolve: Resolver<'a>,
    graph: BTreeMap<String, Receipt>,
    visiting: BTreeSet<String>,
    done: BTreeSet<String>,
    lines: Vec<ChainLine>,
    missing: Vec<String>,
    failed: Vec<String>,
    blocked: Vec<String>,
    cycle: Option<String>,
}

impl<'a> Walker<'a> {
    fn visit(&mut self, id: &str, depth: usize) {
        if self.cycle.is_some() || self.done.contains(id) {
            return;
        }
        if self.visiting.contains(id) {
            self.cycle = Some(format!("dependency cycle through receipt {id}"));
            return;
        }
        self.visiting.insert(id.to_string());
        let (receipt, report) = match (self.resolve)(id) {
            Ok(v) => v,
            Err(e) => {
                self.missing.push(format!("{id}: {e}"));
                self.lines.push(ChainLine {
                    id: id.to_string(),
                    kind: "?".to_string(),
                    tier: Tier::TestOnlyTrust,
                    status: Verdict::Incomplete,
                    depth,
                    note: "MISSING".to_string(),
                });
                self.visiting.remove(id);
                self.done.insert(id.to_string());
                return;
            }
        };
        let lease_report = crate::lease::lease_status(&receipt, self.store);
        let lease_bad = lease_report
            .as_ref()
            .is_some_and(|r| r.status != Verdict::Pass);
        let mut note = report.reasons.join("; ");
        if let Some(lease) = &lease_report {
            if lease.status != Verdict::Pass {
                if !note.is_empty() {
                    note.push_str("; ");
                }
                note.push_str(&lease.reasons.join("; "));
                match lease.status {
                    Verdict::Fail => self.failed.push(format!("{id}: lease binding failed")),
                    Verdict::Incomplete => self.missing.push(format!("{id}: lease incomplete")),
                    _ => self
                        .blocked
                        .push(format!("{id}: lease {}", lease.status.as_str())),
                }
            }
        }
        let status = if lease_bad && report.status == Verdict::Pass {
            Verdict::Incomplete
        } else {
            report.status
        };
        match status {
            Verdict::Pass => {}
            Verdict::Fail => self.failed.push(format!("{id}: dependency records FAIL")),
            Verdict::Blocked => self.blocked.push(format!("{id}: dependency BLOCKED")),
            Verdict::Incomplete => self.missing.push(format!("{id}: dependency INCOMPLETE")),
            Verdict::Skipped => self.blocked.push(format!("{id}: dependency SKIPPED")),
        }
        if note.is_empty() {
            note = status.as_str().to_string();
        }
        self.lines.push(ChainLine {
            id: id.to_string(),
            kind: receipt.kind.clone(),
            tier: receipt.tier,
            status,
            depth,
            note,
        });
        self.graph.insert(id.to_string(), receipt.clone());
        let mut deps = receipt.dependencies.clone();
        deps.sort();
        for dep in deps {
            self.visit(&dep.to_ascii_lowercase(), depth + 1);
        }
        self.visiting.remove(id);
        self.done.insert(id.to_string());
    }
}

/// Graph traversal over an abstract resolver. The file backed [`verify_chain`]
/// supplies a store resolver; tests supply an in memory resolver so cycles,
/// which hash linked IDs make infeasible to construct on disk, are still
/// exercised directly.
pub fn verify_graph(
    root: &Receipt,
    root_report: &Report,
    store: &Path,
    resolve: Resolver<'_>,
) -> ChainReport {
    let root_id = root.id.to_ascii_lowercase();
    let mut walker = Walker {
        store,
        resolve,
        graph: BTreeMap::new(),
        visiting: BTreeSet::new(),
        done: BTreeSet::new(),
        lines: Vec::new(),
        missing: Vec::new(),
        failed: Vec::new(),
        blocked: Vec::new(),
        cycle: None,
    };
    let mut reasons: Vec<String> = Vec::new();

    if root_report.status == Verdict::Fail {
        walker.failed.push(format!("{root_id}: root receipt FAIL"));
    }
    let mut deps = root.dependencies.clone();
    deps.sort();
    for dep in deps {
        walker.visit(&dep.to_ascii_lowercase(), 1);
    }
    walker
        .lines
        .sort_by(|a, b| (a.depth, &a.id).cmp(&(b.depth, &b.id)));

    if let Some(c) = walker.cycle {
        return ChainReport {
            root: root_id,
            status: Verdict::Fail,
            lines: walker.lines,
            reasons: vec![c],
        };
    }
    // Contamination: test only ancestry under a production root.
    if root.tier == Tier::Production {
        let bad: Vec<String> = walker
            .graph
            .values()
            .filter(|r| r.tier == Tier::TestOnlyTrust)
            .map(|r| r.id.clone())
            .collect();
        if !bad.is_empty() {
            reasons.push(format!(
                "PRODUCTION receipt depends on TEST_ONLY_TRUST receipts: {}",
                bad.join(", ")
            ));
            return ChainReport {
                root: root_id,
                status: Verdict::Fail,
                lines: walker.lines,
                reasons,
            };
        }
    }

    let status = if !walker.failed.is_empty() {
        reasons.extend(walker.failed.clone());
        Verdict::Fail
    } else if root_report.status == Verdict::Fail {
        reasons.extend(root_report.reasons.clone());
        Verdict::Fail
    } else if !walker.missing.is_empty() {
        reasons.extend(walker.missing.clone());
        reasons.push("missing or incomplete dependencies never yield PASS".to_string());
        Verdict::Incomplete
    } else if !walker.blocked.is_empty() {
        reasons.extend(walker.blocked.clone());
        Verdict::Blocked
    } else {
        match root_report.status {
            Verdict::Pass => {
                reasons.push(format!(
                    "chain OK: root {root_id} plus {} reachable receipts all PASS",
                    walker.graph.len()
                ));
                Verdict::Pass
            }
            other => {
                reasons.extend(root_report.reasons.clone());
                other
            }
        }
    };
    ChainReport {
        root: root_id,
        status,
        lines: walker.lines,
        reasons,
    }
}

/// True when a receipt at `have` tier satisfies a requirement of `need` tier.
/// Physical requirements need physical evidence; test only trust satisfies
/// nothing but itself.
pub fn tier_satisfies(need: Tier, have: Tier) -> bool {
    if have == Tier::TestOnlyTrust {
        return need == Tier::TestOnlyTrust;
    }
    if need.is_physical() && !have.is_physical() {
        return false;
    }
    have.rank() >= need.rank()
}

pub fn chain_path_arg(path: &str) -> PathBuf {
    PathBuf::from(path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::fixtures::{sample, sealed};
    use crate::evidence::{store_receipt, Mutation, Report};

    fn qemu_receipt(kind: &str, deps: Vec<String>) -> Receipt {
        let mut r = sample();
        r.kind = kind.to_string();
        r.dependencies = deps;
        sealed(r)
    }

    fn put(store: &Path, r: &Receipt) -> String {
        store_receipt(store, r).unwrap()
    }

    fn write_root(dir: &Path, r: &Receipt) -> PathBuf {
        let path = dir.join("root.json");
        let text = serde_json::to_string_pretty(r).unwrap();
        std::fs::write(&path, format!("{text}\n")).unwrap();
        path
    }

    #[test]
    fn empty_chain_passes() {
        let store = crate::testutil::temp_dir("chain-empty");
        let r = qemu_receipt("m3", vec![]);
        let id = put(&store, &r);
        let dir = crate::testutil::temp_dir("chain-empty-root");
        let path = write_root(&dir, &r);
        let report = verify_chain(&path, &store).unwrap();
        assert_eq!(report.status, Verdict::Pass);
        assert_eq!(report.root, id);
    }

    #[test]
    fn missing_dependency_is_incomplete_never_pass() {
        let store = crate::testutil::temp_dir("chain-missing");
        let r = sealed(qemu_receipt("seed-0b-machine1", vec!["f".repeat(64)]));
        let dir = crate::testutil::temp_dir("chain-missing-root");
        let path = write_root(&dir, &r);
        let report = verify_chain(&path, &store).unwrap();
        assert_eq!(report.status, Verdict::Incomplete);
    }

    #[test]
    fn tampered_dependency_file_fails_closed() {
        let store = crate::testutil::temp_dir("chain-tamper");
        let dep = put(&store, &qemu_receipt("m3", vec![]));
        let root = sealed(qemu_receipt("seed", vec![dep.clone()]));
        let dir = crate::testutil::temp_dir("chain-tamper-root");
        let path = write_root(&dir, &root);
        assert_eq!(verify_chain(&path, &store).unwrap().status, Verdict::Pass);
        let dep_path = store.join("receipts").join(format!("{dep}.json"));
        let text = std::fs::read_to_string(&dep_path).unwrap().replacen(
            "artifact_signature_valid",
            "artifact_signature_tampered",
            1,
        );
        std::fs::write(&dep_path, text).unwrap();
        assert_eq!(
            verify_chain(&path, &store).unwrap().status,
            Verdict::Incomplete
        );
    }

    #[test]
    fn cyclic_dependencies_rejected() {
        // Hash linked IDs make real cycles infeasible to construct on disk,
        // which is itself a defense. The traversal guard is exercised here
        // through an in memory resolver with a genuine two node cycle.
        let store = crate::testutil::temp_dir("chain-cycle");
        let mut a = sample();
        a.id = "a".repeat(64);
        a.kind = "cycle-a".to_string();
        a.dependencies = vec!["b".repeat(64)];
        let mut b = sample();
        b.id = "b".repeat(64);
        b.kind = "cycle-b".to_string();
        b.dependencies = vec!["a".repeat(64)];
        let map: BTreeMap<String, (Receipt, Report)> = BTreeMap::from([
            ("a".repeat(64), (a.clone(), Report::pass("ok"))),
            ("b".repeat(64), (b.clone(), Report::pass("ok"))),
        ]);
        let resolve = |id: &str| map.get(id).cloned().ok_or_else(|| "missing".to_string());
        let mut root = sample();
        root.id = "r".repeat(64);
        root.kind = "cycle-root".to_string();
        root.dependencies = vec!["a".repeat(64)];
        let report = verify_graph(&root, &Report::pass("ok"), &store, &resolve);
        assert_eq!(report.status, Verdict::Fail);
        assert!(report.reasons.iter().any(|r| r.contains("cycle")));
    }

    #[test]
    fn blocked_dependency_blocks_chain_never_passes() {
        let store = crate::testutil::temp_dir("chain-blocked");
        let mut dep = sample();
        dep.kind = "qemu-qual".to_string();
        dep.result = Verdict::Blocked;
        let dep_id = put(&store, &sealed(dep));
        let root = sealed(qemu_receipt("seed", vec![dep_id]));
        let dir = crate::testutil::temp_dir("chain-blocked-root");
        let path = write_root(&dir, &root);
        let report = verify_chain(&path, &store).unwrap();
        assert_eq!(report.status, Verdict::Blocked);
    }

    #[test]
    fn failed_dependency_fails_chain() {
        let store = crate::testutil::temp_dir("chain-fail");
        let mut dep = sample();
        dep.kind = "qemu-qual".to_string();
        dep.result = Verdict::Fail;
        let dep_id = put(&store, &sealed(dep));
        let root = sealed(qemu_receipt("seed", vec![dep_id]));
        let dir = crate::testutil::temp_dir("chain-fail-root");
        let path = write_root(&dir, &root);
        assert_eq!(verify_chain(&path, &store).unwrap().status, Verdict::Fail);
    }

    #[test]
    fn illegal_tier_escalation_fails() {
        // PRODUCTION root with TEST_ONLY_TRUST ancestry.
        let store = crate::testutil::temp_dir("chain-tier");
        let mut t = sample();
        t.kind = "lab-note".to_string();
        t.tier = Tier::TestOnlyTrust;
        t.env_class = "TEST_ONLY_TRUST".to_string();
        let t_id = put(&store, &sealed(t));
        let mut p = sample();
        p.kind = "prod-release".to_string();
        p.tier = Tier::Production;
        p.env_class = "PRODUCTION".to_string();
        p.authority = "release board 2026-09".to_string();
        p.dependencies = vec![t_id];
        let p = sealed(p);
        let dir = crate::testutil::temp_dir("chain-tier-root");
        let path = write_root(&dir, &p);
        assert_eq!(verify_chain(&path, &store).unwrap().status, Verdict::Fail);
    }

    #[test]
    fn tier_satisfaction_matrix() {
        assert!(tier_satisfies(Tier::Qemu, Tier::Qemu));
        assert!(tier_satisfies(Tier::Qemu, Tier::Machine1Mutating));
        assert!(!tier_satisfies(Tier::Machine1Attended, Tier::Qemu));
        assert!(!tier_satisfies(Tier::Machine1Attended, Tier::QemuSecurity));
        assert!(tier_satisfies(
            Tier::Machine1ReadOnly,
            Tier::Machine1Mutating
        ));
        assert!(!tier_satisfies(Tier::HostTest, Tier::TestOnlyTrust));
        assert!(tier_satisfies(Tier::TestOnlyTrust, Tier::TestOnlyTrust));
    }

    #[test]
    fn mutation_import_used() {
        assert_eq!(Mutation::None.as_str(), "NONE");
    }
}
