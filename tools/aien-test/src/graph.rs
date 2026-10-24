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

//! The gate dependency graph (ADR 0028 Decisions 3 and 6).
//!
//! `load` finds every `*.gate` file under a repository root (the D1
//! convention, via `runner::discover`), parses each one strictly, and builds
//! a DAG from the `depends_on` lists. A bad manifest, a repeated gate id, an
//! unknown dependency and a cycle are all hard errors that name what is wrong.
//! Nodes are sorted by gate id, so node numbers, the topological order and
//! every query below are deterministic.

use crate::manifest::{self, Manifest};
use crate::runner::discover;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone)]
pub struct Node {
    pub id: String,
    pub path: PathBuf,
    pub manifest: Manifest,
    /// Node numbers of the gates this gate depends on: sorted, no repeats.
    pub deps: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GraphError {
    Io(String),
    BadManifest {
        path: PathBuf,
        message: String,
    },
    DuplicateGate {
        id: String,
        paths: Vec<PathBuf>,
    },
    UnknownDependency {
        gate: String,
        dep: String,
    },
    /// Gate ids around the cycle; the first id is repeated at the end.
    Cycle(Vec<String>),
}

impl fmt::Display for GraphError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            GraphError::Io(m) => f.write_str(m),
            GraphError::BadManifest { path, message } => {
                write!(f, "BAD_MANIFEST {}: {message}", path.display())
            }
            GraphError::DuplicateGate { id, paths } => {
                let list: Vec<String> = paths.iter().map(|p| p.display().to_string()).collect();
                write!(f, "duplicate gate id {id} in: {}", list.join(", "))
            }
            GraphError::UnknownDependency { gate, dep } => {
                write!(f, "gate {gate} depends on unknown gate {dep}")
            }
            GraphError::Cycle(ids) => {
                write!(f, "dependency cycle: {}", ids.join(" -> "))
            }
        }
    }
}

#[derive(Debug, Clone)]
pub struct Graph {
    nodes: Vec<Node>,
    index: BTreeMap<String, usize>,
    dependents: Vec<Vec<usize>>,
    order: Vec<usize>,
}

/// Discover, parse and link every gate under `root`.
pub fn load(root: &Path) -> Result<Graph, GraphError> {
    let mut entries = Vec::new();
    for path in discover(root) {
        let bytes = fs::read(&path)
            .map_err(|e| GraphError::Io(format!("cannot read {}: {e}", path.display())))?;
        let m = manifest::parse(&bytes).map_err(|e| GraphError::BadManifest {
            path: path.clone(),
            message: e.0,
        })?;
        entries.push((path, m));
    }
    build(entries)
}

/// Link already parsed manifests (path, manifest) into a graph.
pub fn build(mut entries: Vec<(PathBuf, Manifest)>) -> Result<Graph, GraphError> {
    entries.sort_by(|a, b| a.1.gate.cmp(&b.1.gate).then_with(|| a.0.cmp(&b.0)));
    for pair in entries.windows(2) {
        if pair[0].1.gate == pair[1].1.gate {
            let id = pair[0].1.gate.clone();
            let paths = entries
                .iter()
                .filter(|e| e.1.gate == id)
                .map(|e| e.0.clone())
                .collect();
            return Err(GraphError::DuplicateGate { id, paths });
        }
    }
    let index: BTreeMap<String, usize> = entries
        .iter()
        .enumerate()
        .map(|(i, e)| (e.1.gate.clone(), i))
        .collect();
    let mut nodes: Vec<Node> = Vec::with_capacity(entries.len());
    for (path, m) in entries {
        let mut deps: BTreeSet<usize> = BTreeSet::new();
        for d in &m.depends_on {
            match index.get(d) {
                Some(&i) => {
                    deps.insert(i);
                }
                None => {
                    return Err(GraphError::UnknownDependency {
                        gate: m.gate.clone(),
                        dep: d.clone(),
                    })
                }
            }
        }
        nodes.push(Node {
            id: m.gate.clone(),
            path,
            manifest: m,
            deps: deps.into_iter().collect(),
        });
    }
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); nodes.len()];
    for (i, n) in nodes.iter().enumerate() {
        for &d in &n.deps {
            dependents[d].push(i);
        }
    }
    let order = topological_order(&nodes, &dependents);
    if order.len() != nodes.len() {
        let adj: Vec<&[usize]> = nodes.iter().map(|n| n.deps.as_slice()).collect();
        let cycle = find_cycle(&adj).unwrap_or_default();
        let ids = cycle.into_iter().map(|i| nodes[i].id.clone()).collect();
        return Err(GraphError::Cycle(ids));
    }
    Ok(Graph {
        nodes,
        index,
        dependents,
        order,
    })
}

/// Kahn's algorithm, always taking the lowest node number (lowest gate id)
/// among the ready ones. Gates on a cycle never become ready, so the result
/// is shorter than the node list exactly when a cycle exists.
fn topological_order(nodes: &[Node], dependents: &[Vec<usize>]) -> Vec<usize> {
    let mut waiting: Vec<usize> = nodes.iter().map(|n| n.deps.len()).collect();
    let mut ready: BTreeSet<usize> = waiting
        .iter()
        .enumerate()
        .filter(|(_, w)| **w == 0)
        .map(|(i, _)| i)
        .collect();
    let mut order = Vec::with_capacity(nodes.len());
    while let Some(i) = ready.pop_first() {
        order.push(i);
        for &j in &dependents[i] {
            waiting[j] -= 1;
            if waiting[j] == 0 {
                ready.insert(j);
            }
        }
    }
    order
}

/// Depth-first search for one cycle. Returns the node numbers along it with
/// the first one repeated at the end, or None if the graph is acyclic.
fn find_cycle(deps: &[&[usize]]) -> Option<Vec<usize>> {
    fn visit(
        n: usize,
        deps: &[&[usize]],
        state: &mut [u8],
        stack: &mut Vec<usize>,
    ) -> Option<Vec<usize>> {
        state[n] = 1;
        stack.push(n);
        for &d in deps[n] {
            if state[d] == 1 {
                let start = stack.iter().position(|&x| x == d).unwrap_or(0);
                let mut cycle: Vec<usize> = stack[start..].to_vec();
                cycle.push(d);
                return Some(cycle);
            }
            if state[d] == 0 {
                if let Some(c) = visit(d, deps, state, stack) {
                    return Some(c);
                }
            }
        }
        stack.pop();
        state[n] = 2;
        None
    }
    let mut state = vec![0u8; deps.len()];
    let mut stack = Vec::new();
    (0..deps.len()).find_map(|n| {
        if state[n] == 0 {
            visit(n, deps, &mut state, &mut stack)
        } else {
            None
        }
    })
}

/// Every node reachable from `start` by one or more steps of `next`.
fn reach<'a>(start: &[usize], next: impl Fn(usize) -> &'a [usize]) -> BTreeSet<usize> {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<usize> = start.to_vec();
    while let Some(n) = stack.pop() {
        for &m in next(n) {
            if seen.insert(m) {
                stack.push(m);
            }
        }
    }
    seen
}

impl Graph {
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn node(&self, i: usize) -> &Node {
        &self.nodes[i]
    }

    pub fn index_of(&self, id: &str) -> Option<usize> {
        self.index.get(id).copied()
    }

    /// Topological order: every gate comes after all the gates it depends on.
    pub fn order(&self) -> &[usize] {
        &self.order
    }

    /// Gates that list `i` directly in their `depends_on`.
    pub fn direct_dependents(&self, i: usize) -> &[usize] {
        &self.dependents[i]
    }

    /// All gates `i` rests on, directly or through other gates (not `i`).
    pub fn transitive_deps(&self, i: usize) -> BTreeSet<usize> {
        reach(&[i], |n| self.nodes[n].deps.as_slice())
    }

    /// All gates that rest on `i`, directly or through other gates (not `i`).
    pub fn transitive_dependents(&self, i: usize) -> BTreeSet<usize> {
        reach(&[i], |n| self.dependents[n].as_slice())
    }

    /// `targets` plus everything they depend on: the set to run to check them.
    pub fn with_dependencies(&self, targets: &[usize]) -> BTreeSet<usize> {
        let mut set = reach(targets, |n| self.nodes[n].deps.as_slice());
        set.extend(targets.iter().copied());
        set
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testutil::{write_file, TempDir};

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

    fn gate(id: &str, deps: &[&str]) -> (PathBuf, Manifest) {
        (
            PathBuf::from(format!("tests/{id}.gate")),
            manifest::parse(gate_text(id, deps).as_bytes()).unwrap(),
        )
    }

    fn graph(spec: &[(&str, &[&str])]) -> Result<Graph, GraphError> {
        build(spec.iter().map(|(id, deps)| gate(id, deps)).collect())
    }

    fn ids(g: &Graph, set: &BTreeSet<usize>) -> Vec<String> {
        set.iter().map(|&i| g.node(i).id.clone()).collect()
    }

    /// A -> B, A -> C, B -> D, C -> D (an arrow means "depends on").
    fn diamond() -> Graph {
        graph(&[("A", &["B", "C"]), ("B", &["D"]), ("C", &["D"]), ("D", &[])]).unwrap()
    }

    #[test]
    fn order_puts_dependencies_first() {
        let g = diamond();
        let order: Vec<&str> = g.order().iter().map(|&i| g.node(i).id.as_str()).collect();
        assert_eq!(order, vec!["D", "B", "C", "A"]);
        let pos = |id: &str| order.iter().position(|x| *x == id).unwrap();
        for n in g.nodes() {
            for &d in &n.deps {
                assert!(
                    pos(&g.node(d).id) < pos(&n.id),
                    "{} before {}",
                    g.node(d).id,
                    n.id
                );
            }
        }
    }

    #[test]
    fn order_is_independent_of_input_order() {
        let a = graph(&[("X", &[]), ("Y", &["X"]), ("Z", &[])]).unwrap();
        let b = graph(&[("Z", &[]), ("Y", &["X"]), ("X", &[])]).unwrap();
        assert_eq!(a.order(), b.order());
        let names: Vec<&str> = a.order().iter().map(|&i| a.node(i).id.as_str()).collect();
        assert_eq!(names, vec!["X", "Y", "Z"]);
    }

    #[test]
    fn unknown_dependency_names_both_gates() {
        let e = graph(&[("A", &["GHOST"])]).unwrap_err();
        assert_eq!(
            e,
            GraphError::UnknownDependency {
                gate: "A".to_string(),
                dep: "GHOST".to_string()
            }
        );
        let msg = e.to_string();
        assert!(msg.contains("A") && msg.contains("GHOST"), "{msg}");
    }

    #[test]
    fn cycle_is_named() {
        let e = graph(&[("A", &["B"]), ("B", &["C"]), ("C", &["A"])]).unwrap_err();
        assert_eq!(
            e,
            GraphError::Cycle(vec![
                "A".to_string(),
                "B".to_string(),
                "C".to_string(),
                "A".to_string()
            ])
        );
        assert_eq!(e.to_string(), "dependency cycle: A -> B -> C -> A");
    }

    #[test]
    fn cycle_behind_an_acyclic_prefix_is_found() {
        let e = graph(&[
            ("A", &["B"]),
            ("B", &[]),
            ("C", &["D"]),
            ("D", &["E"]),
            ("E", &["D"]),
        ])
        .unwrap_err();
        assert_eq!(e.to_string(), "dependency cycle: D -> E -> D");
    }

    #[test]
    fn self_dependency_is_a_cycle() {
        let e = graph(&[("A", &["A"])]).unwrap_err();
        assert_eq!(e.to_string(), "dependency cycle: A -> A");
    }

    #[test]
    fn duplicate_gate_ids_name_both_files() {
        let mut entries = vec![gate("A", &[]), gate("B", &[])];
        let mut second = gate("A", &[]);
        second.0 = PathBuf::from("other/a.gate");
        entries.push(second);
        match build(entries).unwrap_err() {
            GraphError::DuplicateGate { id, paths } => {
                assert_eq!(id, "A");
                assert_eq!(
                    paths,
                    vec![PathBuf::from("other/a.gate"), PathBuf::from("tests/A.gate")]
                );
            }
            other => panic!("expected DuplicateGate, got {other:?}"),
        }
    }

    #[test]
    fn repeated_depends_on_entries_collapse() {
        let g = graph(&[("A", &["B", "B"]), ("B", &[])]).unwrap();
        let a = g.index_of("A").unwrap();
        assert_eq!(g.node(a).deps.len(), 1);
        assert_eq!(g.direct_dependents(g.index_of("B").unwrap()), &[a]);
    }

    #[test]
    fn transitive_queries() {
        let g = diamond();
        let i = |id: &str| g.index_of(id).unwrap();
        assert_eq!(ids(&g, &g.transitive_deps(i("A"))), vec!["B", "C", "D"]);
        assert_eq!(ids(&g, &g.transitive_deps(i("B"))), vec!["D"]);
        assert!(g.transitive_deps(i("D")).is_empty());
        assert_eq!(
            ids(&g, &g.transitive_dependents(i("D"))),
            vec!["A", "B", "C"]
        );
        assert_eq!(ids(&g, &g.transitive_dependents(i("B"))), vec!["A"]);
        assert!(g.transitive_dependents(i("A")).is_empty());
        assert_eq!(ids(&g, &g.with_dependencies(&[i("B")])), vec!["B", "D"]);
        assert_eq!(
            ids(&g, &g.with_dependencies(&[i("B"), i("C")])),
            vec!["B", "C", "D"]
        );
        assert_eq!(ids(&g, &g.with_dependencies(&[i("D")])), vec!["D"]);
    }

    #[test]
    fn lookup_and_size() {
        let g = diamond();
        assert_eq!(g.len(), 4);
        assert!(!g.is_empty());
        assert_eq!(g.nodes().len(), 4);
        assert_eq!(g.index_of("C").map(|i| g.node(i).id.as_str()), Some("C"));
        assert_eq!(g.index_of("nope"), None);
        let empty = build(Vec::new()).unwrap();
        assert!(empty.is_empty() && empty.order().is_empty());
    }

    #[test]
    fn load_finds_gates_and_skips_build_dirs() {
        let t = TempDir::new("graph-load");
        write_file(t.path(), "tests/a/a.gate", &gate_text("A", &["B"]), false);
        write_file(t.path(), "tests/b/b.gate", &gate_text("B", &[]), false);
        write_file(
            t.path(),
            "target/ghost.gate",
            &gate_text("GHOST", &[]),
            false,
        );
        let g = load(t.path()).unwrap();
        assert_eq!(g.len(), 2);
        let order: Vec<&str> = g.order().iter().map(|&i| g.node(i).id.as_str()).collect();
        assert_eq!(order, vec!["B", "A"]);
        assert!(g
            .node(g.index_of("A").unwrap())
            .path
            .ends_with("tests/a/a.gate"));
    }

    #[test]
    fn load_fails_closed_on_a_bad_manifest() {
        let t = TempDir::new("graph-bad");
        write_file(t.path(), "tests/bad.gate", "gate: X\nbogus: 1\n", false);
        match load(t.path()).unwrap_err() {
            GraphError::BadManifest { path, message } => {
                assert!(path.ends_with("tests/bad.gate"));
                assert!(!message.is_empty());
            }
            other => panic!("expected BadManifest, got {other:?}"),
        }
    }

    #[test]
    fn error_messages_are_plain() {
        assert_eq!(
            GraphError::DuplicateGate {
                id: "A".to_string(),
                paths: vec![PathBuf::from("x.gate"), PathBuf::from("y.gate")]
            }
            .to_string(),
            "duplicate gate id A in: x.gate, y.gate"
        );
        assert_eq!(
            GraphError::UnknownDependency {
                gate: "A".to_string(),
                dep: "B".to_string()
            }
            .to_string(),
            "gate A depends on unknown gate B"
        );
        assert_eq!(GraphError::Io("boom".to_string()).to_string(), "boom");
        let bad = GraphError::BadManifest {
            path: PathBuf::from("p.gate"),
            message: "line 1: nope".to_string(),
        };
        assert_eq!(bad.to_string(), "BAD_MANIFEST p.gate: line 1: nope");
    }
}
