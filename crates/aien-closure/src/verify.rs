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

//! The verification pipeline behind `verify-closure`.

use crate::declare;
use crate::digest::source_digest;
use crate::error::{Code, Finding};
use crate::graph::{cargo_packages, closure_of, find_cycle, graph_of, Package};
use crate::manifest::{self, Manifest};
use aien_proof::chain::{tier_satisfies, verify_chain};
use aien_proof::evidence::{receipt_id, Receipt, Tier, Verdict, RECEIPT_DIR};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};

pub const LOCK_FILE: &str = "closure.lock";
pub const LOCK_HEADER: &str = "closure-lock v1";

pub struct Options {
    pub root: PathBuf,
    pub store: PathBuf,
    /// Write `closure.lock` (only when no other finding exists) before
    /// comparing, so a fresh lock verifies.
    pub write_lock: bool,
}

pub struct Outcome {
    pub findings: Vec<Finding>,
    /// Number of components that carry a manifest.
    pub components: usize,
    /// Components with no findings.
    pub passed: Vec<(String, String)>,
}

impl Outcome {
    pub fn ok(&self) -> bool {
        self.findings.is_empty()
    }
}

/// Minimum receipt tier a verifier profile demands. Unknown profiles have no
/// entry, which the verifier reports as UNKNOWN_VERIFIER_PROFILE.
pub fn profile_min_tier(profile: &str) -> Option<Tier> {
    match profile {
        "host-v1" => Some(Tier::HostTest),
        "qemu-v1" => Some(Tier::Qemu),
        "production-v1" => Some(Tier::Production),
        _ => None,
    }
}

enum Load {
    Missing,
    Mismatch(String),
    Ok(Box<Receipt>),
}

fn receipt_path(store: &Path, id: &str) -> PathBuf {
    store.join(RECEIPT_DIR).join(format!("{id}.json"))
}

/// Read a receipt by claimed id and recompute its identity with aien-proof.
fn load(store: &Path, id: &str) -> Load {
    let Ok(bytes) = fs::read(receipt_path(store, id)) else {
        return Load::Missing;
    };
    let r: Receipt = match serde_json::from_slice(&bytes) {
        Ok(r) => r,
        Err(e) => return Load::Mismatch(format!("not a valid receipt: {e}")),
    };
    match receipt_id(&r) {
        Err(e) => Load::Mismatch(e),
        Ok(c) if c != id || r.id.to_ascii_lowercase() != id => Load::Mismatch(format!(
            "file {id} has canonical identity {c} and claims {}",
            r.id
        )),
        Ok(_) => Load::Ok(Box::new(r)),
    }
}

struct ChainFacts {
    findings: Vec<(Code, String)>,
    tainted: Vec<String>,
    root: Option<Box<Receipt>>,
}

/// Walk the receipt graph from `id`, classifying missing, mismatched and
/// tainted receipts. Tainted means recorded dirty or TEST_ONLY_TRUST.
fn walk_receipts(store: &Path, id: &str) -> ChainFacts {
    let mut facts = ChainFacts {
        findings: vec![],
        tainted: vec![],
        root: None,
    };
    let mut seen = BTreeSet::new();
    let mut todo = vec![id.to_string()];
    while let Some(cur) = todo.pop() {
        if !seen.insert(cur.clone()) {
            continue;
        }
        match load(store, &cur) {
            Load::Missing => facts
                .findings
                .push((Code::MissingReceipt, format!("receipt {cur} not in store"))),
            Load::Mismatch(m) => facts
                .findings
                .push((Code::ReceiptHashMismatch, format!("receipt {cur}: {m}"))),
            Load::Ok(r) => {
                if r.dirty {
                    facts
                        .tainted
                        .push(format!("{cur} recorded from a dirty tree"));
                }
                if r.tier == Tier::TestOnlyTrust {
                    facts.tainted.push(format!("{cur} is TEST_ONLY_TRUST"));
                }
                todo.extend(r.dependencies.iter().map(|d| d.to_ascii_lowercase()));
                if cur == id {
                    facts.root = Some(r);
                }
            }
        }
    }
    facts
}

fn check_component(
    m: &Manifest,
    pkg: &Package,
    manifests: &BTreeMap<String, Manifest>,
    store: &Path,
    out: &mut Vec<Finding>,
) {
    let name = m.component.as_str();
    let mut add = |code: Code, detail: String| out.push(Finding::new(code, name, detail));

    // Profile.
    let min_tier = profile_min_tier(&m.profile);
    if min_tier.is_none() {
        add(
            Code::UnknownVerifierProfile,
            format!("profile {:?} is not known to this verifier", m.profile),
        );
    }

    // Declarations (SPEC 6.2): an actual edge the manifest does not declare is
    // UNDECLARED_IMPORT; a declared dep that is not an actual edge is
    // UNVERIFIED_DEPENDENCY.
    let actual: BTreeSet<String> = pkg.deps.iter().cloned().collect();
    let declared: BTreeSet<String> = m.deps.keys().cloned().collect();
    for d in declare::compare(&actual, &declared) {
        let dep = &d.name;
        let detail = match d.code {
            Code::UndeclaredImport => {
                format!("edge to {dep} is not declared in {}", manifest::FILE_NAME)
            }
            _ => format!("declared dep {dep} is not an edge of the graph"),
        };
        add(d.code, detail);
    }

    // Edges: every declared actual edge pinned to the dep's own receipt.
    for dep in &pkg.deps {
        if let Some(pin) = m.deps.get(dep) {
            match manifests.get(dep) {
                None => add(
                    Code::UnverifiedDependency,
                    format!("dependency {dep} has no {}", manifest::FILE_NAME),
                ),
                Some(dm) => match pin {
                    None => add(
                        Code::DependencyNotPinned,
                        format!("dep {dep} declared without a receipt pin"),
                    ),
                    Some(p) if *p != dm.receipt => add(
                        Code::DependencyNotPinned,
                        format!(
                            "dep {dep} pinned to {p} but {dep} is qualified by {}",
                            dm.receipt
                        ),
                    ),
                    Some(_) => {}
                },
            }
        }
    }

    // Receipt and chain.
    let facts = walk_receipts(store, &m.receipt);
    let broken = !facts.findings.is_empty();
    for (code, detail) in facts.findings {
        add(code, detail);
    }
    for t in facts.tainted {
        add(Code::TaintedArtifact, t);
    }
    if let Some(root) = &facts.root {
        // Pins must be bound inside the receipt, so the chain covers them.
        let rdeps: BTreeSet<String> = root
            .dependencies
            .iter()
            .map(|d| d.to_ascii_lowercase())
            .collect();
        for (dep, pin) in &m.deps {
            if let Some(p) = pin {
                if !rdeps.contains(p) {
                    add(
                        Code::DependencyNotPinned,
                        format!(
                            "receipt {} does not depend on pinned receipt {p} of {dep}",
                            m.receipt
                        ),
                    );
                }
            }
        }
        // Source binding.
        if !root.input_artifacts.iter().any(|a| a == &m.source) {
            add(
                Code::StaleReceipt,
                format!("receipt {} was not taken at {}", m.receipt, m.source),
            );
        }
        if let Some(need) = min_tier {
            if root.tier != Tier::TestOnlyTrust && !tier_satisfies(need, root.tier) {
                add(
                    Code::UnverifiedDependency,
                    format!(
                        "receipt tier {} does not satisfy profile {}",
                        root.tier.as_str(),
                        m.profile
                    ),
                );
            }
        }
        if root.result != Verdict::Pass {
            add(
                Code::UnverifiedDependency,
                format!("receipt {} result is {}", m.receipt, root.result.as_str()),
            );
        }
    }
    if !broken {
        match verify_chain(&receipt_path(store, &m.receipt), store) {
            Err(e) => add(Code::UnverifiedDependency, format!("chain error: {e}")),
            Ok(rep) if rep.status != Verdict::Pass => {
                let why = rep.reasons.join("; ");
                let code = if why.contains("dependency cycle") {
                    Code::DependencyCycle
                } else {
                    Code::UnverifiedDependency
                };
                add(code, format!("chain {}: {why}", rep.status.as_str()));
            }
            Ok(_) => {}
        }
    }

    // Current source against the digest the manifest records.
    match source_digest(&pkg.dir) {
        Err(e) => add(Code::StaleReceipt, format!("cannot digest source: {e}")),
        Ok(d) => {
            let now = format!("blake3:{d}");
            if now != m.source {
                add(
                    Code::StaleReceipt,
                    format!("source is {now} but manifest records {}", m.source),
                );
            }
        }
    }
}

/// Render the lock: header, one `component` line each, then every transitive
/// dependency with its pinned receipt id. Sorted, so it is deterministic.
pub fn render_lock(manifests: &BTreeMap<String, Manifest>, graph: &crate::graph::Graph) -> String {
    let mut s = format!("{LOCK_HEADER}\n");
    for (name, m) in manifests {
        s.push_str(&format!("component {name} {} {}\n", m.receipt, m.source));
        for dep in closure_of(graph, name) {
            let id = manifests.get(&dep).map_or("-", |d| d.receipt.as_str());
            s.push_str(&format!("needs {name} {dep} {id}\n"));
        }
    }
    s
}

/// Run every check. `Err` is an internal error (cannot run cargo, cannot
/// read the repo); callers must treat it as failure too.
pub fn verify(opts: &Options) -> Result<Outcome, String> {
    let packages = cargo_packages(&opts.root)?;
    let graph = graph_of(&packages);
    let mut findings: Vec<Finding> = Vec::new();

    // Manifests.
    let mut manifests: BTreeMap<String, Manifest> = BTreeMap::new();
    for pkg in &packages {
        let path = pkg.dir.join(manifest::FILE_NAME);
        if !path.exists() {
            continue;
        }
        let parsed = fs::read_to_string(&path)
            .map_err(|e| e.to_string())
            .and_then(|t| manifest::parse(&t))
            .and_then(|m| {
                if m.component == pkg.name {
                    Ok(m)
                } else {
                    Err(format!(
                        "component {:?} does not match package {:?}",
                        m.component, pkg.name
                    ))
                }
            });
        match parsed {
            Ok(m) => {
                manifests.insert(pkg.name.clone(), m);
            }
            Err(e) => findings.push(Finding::new(
                Code::UnverifiedDependency,
                &pkg.name,
                format!("bad {}: {e}", manifest::FILE_NAME),
            )),
        }
    }
    if manifests.is_empty() && findings.is_empty() {
        findings.push(Finding::new(
            Code::UnverifiedDependency,
            "-",
            format!(
                "no {} found under {}",
                manifest::FILE_NAME,
                opts.root.display()
            ),
        ));
    }

    // Cycles.
    if let Some(cycle) = find_cycle(&graph) {
        findings.push(Finding::new(
            Code::DependencyCycle,
            &cycle[0],
            cycle.join(" -> "),
        ));
    }

    // Components with a manifest.
    let by_name: BTreeMap<&str, &Package> = packages.iter().map(|p| (p.name.as_str(), p)).collect();
    for (name, m) in &manifests {
        check_component(
            m,
            by_name[name.as_str()],
            &manifests,
            &opts.store,
            &mut findings,
        );
    }

    // Lock file, only meaningful when the rest is clean.
    if findings.is_empty() {
        let want = render_lock(&manifests, &graph);
        let path = opts.root.join(LOCK_FILE);
        if opts.write_lock {
            fs::write(&path, &want).map_err(|e| format!("cannot write {}: {e}", path.display()))?;
        }
        match fs::read_to_string(&path) {
            Ok(have) if have == want => {}
            Ok(have) => {
                // A pin for something the component does not depend on is
                // UNVERIFIED_DEPENDENCY (SPEC 6.2). Everything else that
                // differs (missing line, wrong id) is DEPENDENCY_NOT_PINNED.
                for (comp, dep) in declare::extra_needs(&have, &want) {
                    findings.push(Finding::new(
                        Code::UnverifiedDependency,
                        &comp,
                        format!("{LOCK_FILE} pins {dep} for {comp}, which does not depend on it"),
                    ));
                }
                if declare::without_extra_needs(&have, &want) != want {
                    findings.push(Finding::new(
                        Code::DependencyNotPinned,
                        "-",
                        format!("{LOCK_FILE} does not match the rebuilt closure"),
                    ));
                }
            }
            Err(_) => findings.push(Finding::new(
                Code::DependencyNotPinned,
                "-",
                format!("{LOCK_FILE} is missing"),
            )),
        }
    }

    findings.sort();
    findings.dedup();
    let bad: BTreeSet<&str> = findings.iter().map(|f| f.component.as_str()).collect();
    let passed = manifests
        .iter()
        .filter(|(n, _)| !bad.contains(n.as_str()))
        .map(|(n, m)| (n.clone(), m.receipt.clone()))
        .collect();
    Ok(Outcome {
        findings,
        components: manifests.len(),
        passed,
    })
}
