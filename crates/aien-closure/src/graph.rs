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

//! Dependency graph reconstruction: Rust (cargo metadata) and C (quoted
//! `#include` scanner), plus cycle detection shared by both.

use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

/// node -> set of nodes it depends on.
pub type Graph = BTreeMap<String, BTreeSet<String>>;

/// A workspace package that is a node of the internal graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Package {
    pub name: String,
    pub dir: PathBuf,
    /// Internal path dependencies (normal and build kinds), by package name.
    pub deps: BTreeSet<String>,
}

/// Parse `cargo metadata --no-deps` output. Only workspace members are nodes.
/// An edge is a dependency that has a `path` and names another member. Dev
/// dependencies are not part of the built artifact and are skipped.
pub fn packages_from_metadata(meta: &Value) -> Result<Vec<Package>, String> {
    let members: BTreeSet<&str> = meta["workspace_members"]
        .as_array()
        .ok_or("metadata has no workspace_members")?
        .iter()
        .filter_map(Value::as_str)
        .collect();
    let pkgs: Vec<&Value> = meta["packages"]
        .as_array()
        .ok_or("metadata has no packages")?
        .iter()
        .filter(|p| p["id"].as_str().is_some_and(|id| members.contains(id)))
        .collect();
    let names: BTreeSet<&str> = pkgs.iter().filter_map(|p| p["name"].as_str()).collect();
    let mut out = Vec::new();
    for p in &pkgs {
        let name = p["name"].as_str().ok_or("package without name")?;
        let manifest = p["manifest_path"]
            .as_str()
            .ok_or("package without manifest_path")?;
        let dir = Path::new(manifest)
            .parent()
            .ok_or("bad manifest_path")?
            .to_path_buf();
        let mut deps = BTreeSet::new();
        for d in p["dependencies"].as_array().into_iter().flatten() {
            if d["kind"].as_str() == Some("dev") || d["path"].is_null() {
                continue;
            }
            if let Some(dn) = d["name"].as_str() {
                if names.contains(dn) && dn != name {
                    deps.insert(dn.to_string());
                }
            }
        }
        out.push(Package {
            name: name.to_string(),
            dir,
            deps,
        });
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Run `cargo metadata --offline --locked --no-deps` in `root`.
pub fn cargo_packages(root: &Path) -> Result<Vec<Package>, String> {
    let out = Command::new("cargo")
        .args([
            "metadata",
            "--format-version",
            "1",
            "--offline",
            "--locked",
            "--no-deps",
        ])
        .current_dir(root)
        .output()
        .map_err(|e| format!("cannot run cargo: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "cargo metadata failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let meta: Value =
        serde_json::from_slice(&out.stdout).map_err(|e| format!("bad metadata JSON: {e}"))?;
    packages_from_metadata(&meta)
}

pub fn graph_of(packages: &[Package]) -> Graph {
    packages
        .iter()
        .map(|p| (p.name.clone(), p.deps.clone()))
        .collect()
}

/// Returns one cycle as a path `a -> b -> ... -> a` if the graph has any.
/// Iterative, so a deep graph cannot overflow the stack.
pub fn find_cycle(g: &Graph) -> Option<Vec<String>> {
    #[derive(Clone, Copy, PartialEq)]
    enum Mark {
        Open,
        Done,
    }
    let mut mark: BTreeMap<&str, Mark> = BTreeMap::new();
    for start in g.keys() {
        if mark.contains_key(start.as_str()) {
            continue;
        }
        let mut stack: Vec<(&str, Vec<&str>)> = vec![(
            start.as_str(),
            g[start].iter().map(String::as_str).collect(),
        )];
        mark.insert(start.as_str(), Mark::Open);
        while let Some((node, todo)) = stack.last_mut() {
            let node = *node;
            match todo.pop() {
                Some(next) => match mark.get(next) {
                    Some(Mark::Open) => {
                        let mut path: Vec<String> = stack
                            .iter()
                            .map(|(n, _)| n.to_string())
                            .skip_while(|n| n != next)
                            .collect();
                        path.push(next.to_string());
                        return Some(path);
                    }
                    Some(Mark::Done) => {}
                    None => {
                        mark.insert(next, Mark::Open);
                        let kids = g
                            .get(next)
                            .map(|s| s.iter().map(String::as_str).collect())
                            .unwrap_or_default();
                        stack.push((next, kids));
                    }
                },
                None => {
                    mark.insert(node, Mark::Done);
                    stack.pop();
                }
            }
        }
    }
    None
}

/// Transitive closure of `start` (excluding `start` itself unless cyclic).
pub fn closure_of(g: &Graph, start: &str) -> BTreeSet<String> {
    let mut seen = BTreeSet::new();
    let mut todo: Vec<&str> = g
        .get(start)
        .into_iter()
        .flatten()
        .map(String::as_str)
        .collect();
    while let Some(n) = todo.pop() {
        if seen.insert(n.to_string()) {
            todo.extend(g.get(n).into_iter().flatten().map(String::as_str));
        }
    }
    seen
}

/// Result of scanning C sources for quoted includes.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct CScan {
    /// Files (named relative to the include root) and the files they include.
    pub graph: Graph,
    /// (file, include text) pairs that resolved to no file. Callers must treat
    /// these as findings, never ignore them.
    pub unresolved: Vec<(String, String)>,
}

/// Extract the target of a quoted include from one source line, if any. Only
/// lines whose first token is `#include "..."` count (leading and inner
/// whitespace allowed). Block comments and `#if 0` regions are NOT understood;
/// a commented out include is still reported, which errs on the strict side.
pub fn quoted_include(line: &str) -> Option<&str> {
    let rest = line.trim_start().strip_prefix('#')?.trim_start();
    let rest = rest.strip_prefix("include")?.trim_start();
    let rest = rest.strip_prefix('"')?;
    let end = rest.find('"')?;
    Some(&rest[..end])
}

fn walk_c(dir: &Path, out: &mut Vec<PathBuf>) -> Result<(), String> {
    for entry in fs::read_dir(dir).map_err(|e| format!("cannot read {}: {e}", dir.display()))? {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.is_dir() {
            walk_c(&path, out)?;
        } else if matches!(path.extension().and_then(|e| e.to_str()), Some("c" | "h")) {
            out.push(path);
        }
    }
    Ok(())
}

/// Scan every `.c` and `.h` under `src_dir`. An include resolves first against
/// the including file's directory, then against `include_root` (the `-I`
/// directory). Nodes are named relative to `include_root`, or to `src_dir`
/// when a file lies outside it.
pub fn scan_c_includes(src_dir: &Path, include_root: &Path) -> Result<CScan, String> {
    let canon = |p: &Path| fs::canonicalize(p).map_err(|e| format!("{}: {e}", p.display()));
    let src_dir = canon(src_dir)?;
    let include_root = canon(include_root)?;
    let name_of = |p: &Path| -> String {
        let rel = p
            .strip_prefix(&include_root)
            .or_else(|_| p.strip_prefix(&src_dir))
            .unwrap_or(p);
        rel.to_string_lossy().replace('\\', "/")
    };
    let mut files = Vec::new();
    walk_c(&src_dir, &mut files)?;
    files.sort();
    let mut scan = CScan::default();
    for file in &files {
        let me = name_of(file);
        let text =
            fs::read_to_string(file).map_err(|e| format!("cannot read {}: {e}", file.display()))?;
        let deps = scan.graph.entry(me.clone()).or_default();
        for line in text.lines() {
            let Some(inc) = quoted_include(line) else {
                continue;
            };
            let here = file.parent().map(|d| d.join(inc));
            let there = include_root.join(inc);
            let hit = [here, Some(there)]
                .into_iter()
                .flatten()
                .find(|c| c.is_file())
                .and_then(|c| fs::canonicalize(c).ok());
            match hit {
                Some(p) => {
                    deps.insert(name_of(&p));
                }
                None => scan.unresolved.push((me.clone(), inc.to_string())),
            }
        }
    }
    Ok(scan)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn g(edges: &[(&str, &[&str])]) -> Graph {
        edges
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn cycle_found_and_absent() {
        assert!(find_cycle(&g(&[("a", &["b"]), ("b", &["c"]), ("c", &[])])).is_none());
        let c = find_cycle(&g(&[("a", &["b"]), ("b", &["c"]), ("c", &["b"])])).unwrap();
        assert_eq!(c, vec!["b", "c", "b"]);
        assert!(find_cycle(&g(&[("a", &["a"])])).is_some());
    }

    #[test]
    fn closure_is_transitive() {
        let gr = g(&[("a", &["b"]), ("b", &["c"]), ("c", &[])]);
        let c: Vec<String> = closure_of(&gr, "a").into_iter().collect();
        assert_eq!(c, vec!["b", "c"]);
    }

    #[test]
    fn include_line_parsing() {
        assert_eq!(quoted_include("#include \"a/b.h\""), Some("a/b.h"));
        assert_eq!(quoted_include("  #  include   \"x.h\" // c"), Some("x.h"));
        assert_eq!(quoted_include("#include <stdio.h>"), None);
        assert_eq!(quoted_include("int x; // #include \"no.h\""), None);
    }

    #[test]
    fn c_scanner_on_fixture_dir() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/c-include");
        let scan = scan_c_includes(&root.join("src"), &root.join("src")).unwrap();
        let deps = |n: &str| -> Vec<&str> { scan.graph[n].iter().map(String::as_str).collect() };
        assert_eq!(deps("main.c"), vec!["lib/a.h", "lib/b.h"]);
        assert_eq!(deps("lib/a.h"), vec!["lib/b.h"]);
        assert_eq!(deps("lib/b.h"), Vec::<&str>::new());
        assert_eq!(
            scan.unresolved,
            vec![("main.c".to_string(), "missing.h".to_string())]
        );
        assert!(find_cycle(&scan.graph).is_none());
    }

    #[test]
    fn c_scanner_sees_header_cycle() {
        let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/c-cycle");
        let scan = scan_c_includes(&root, &root).unwrap();
        assert!(scan.unresolved.is_empty());
        assert!(find_cycle(&scan.graph).is_some());
    }
}
