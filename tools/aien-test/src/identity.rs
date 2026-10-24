//! Check identity (ADR 0033 Decision 3).
//!
//! `check_id` is sha256 over the canonical JSON object
//! `{manifest_digest, input_blob_ids, compiler_identity, flags,
//! dependency_check_ids, golden_digests, scope_binding}`. Same inputs give the
//! same id; a different input byte, compiler, flag, declared dependency,
//! golden file, or (for the machine and hardware scopes) machine gives a
//! different id. It is ADR 0028 Decision 7's cache key made explicit about
//! scope.
//!
//! Input identity is the git blob id of every tracked file under each
//! `inputs` entry, read with `git ls-files -s` (no git library). The index is
//! what is read, so an uncommitted edit to a tracked file is not seen: the
//! runner already refuses dirty trees, and `aien-test identity` is meant for
//! clean ones. A declared input that matches no tracked file is recorded with
//! a null blob id rather than dropped, so a typo or a deletion changes the id.

use crate::evidence::{canonical, sha256_hex};
use crate::graph::Graph;
use crate::manifest::{CacheScope, Manifest};
use crate::process;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::Path;

/// What the `scope_binding` member is built from. Only the part the manifest's
/// `cache_scope` names is used.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Binding {
    /// The `arch` string (`std::env::consts::ARCH` on the real machine).
    pub arch: String,
    /// The ADR 0028 `machine_digest`.
    pub machine_digest: String,
    /// The hardware identity block for `cache_scope: hardware`. The ADR names
    /// it but does not define its content (UNVERIFIED); the caller supplies it.
    pub hardware_identity: String,
}

fn git_ls_files(root: &Path, spec: &str) -> Result<Vec<(String, String)>, String> {
    let mut cmd = process::git_command();
    cmd.env("GIT_OPTIONAL_LOCKS", "0")
        .arg("-C")
        .arg(root)
        .args(["ls-files", "-s", "-z", "--"])
        .arg(format!(":(literal){spec}"));
    let out = process::output_gated(&mut cmd).map_err(|e| format!("cannot run git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git ls-files failed: {}",
            String::from_utf8_lossy(&out.stderr).trim()
        ));
    }
    let text = String::from_utf8(out.stdout).map_err(|_| "git output is not UTF-8".to_string())?;
    let mut rows = Vec::new();
    for rec in text.split('\0').filter(|r| !r.is_empty()) {
        let (meta, path) = rec
            .split_once('\t')
            .ok_or_else(|| format!("unparsable ls-files record {rec:?}"))?;
        let f: Vec<&str> = meta.split(' ').collect();
        if f.len() != 3 || f[1].len() != 40 {
            return Err(format!("unparsable ls-files record {rec:?}"));
        }
        if f[2] != "0" {
            return Err(format!("{path} has an unmerged index entry"));
        }
        rows.push((path.to_string(), f[1].to_string()));
    }
    Ok(rows)
}

/// `[{path, blob_id}]` sorted by path, one row per tracked file under the
/// manifest's `inputs`; a path that matches nothing gets `blob_id: null`.
pub fn input_blob_ids(m: &Manifest, root: &Path) -> Result<Vec<Value>, String> {
    let mut rows: BTreeMap<String, Option<String>> = BTreeMap::new();
    for input in &m.inputs {
        let spec = input.trim_end_matches('/');
        let found = git_ls_files(root, spec)?;
        if found.is_empty() {
            rows.entry(spec.to_string()).or_insert(None);
        }
        for (path, blob) in found {
            rows.insert(path, Some(blob));
        }
    }
    Ok(rows
        .into_iter()
        .map(|(path, blob)| json!({ "path": path, "blob_id": blob }))
        .collect())
}

/// The scope binding string for the manifest's `cache_scope`.
pub fn scope_binding(scope: CacheScope, b: &Binding) -> Result<String, String> {
    Ok(match scope {
        CacheScope::Portable => String::new(),
        CacheScope::Arch => b.arch.clone(),
        CacheScope::Machine => b.machine_digest.clone(),
        CacheScope::Hardware => {
            if b.hardware_identity.is_empty() {
                return Err("cache_scope hardware needs a hardware identity".into());
            }
            format!("{}:{}", b.machine_digest, b.hardware_identity)
        }
    })
}

/// Gates this check declares it rests on: `depends_on` plus an `oracle`
/// `compare` gate, sorted, no repeats.
pub fn declared_dependencies(m: &Manifest) -> Vec<String> {
    let mut set: BTreeSet<String> = m.depends_on.iter().cloned().collect();
    if let Some(c) = &m.oracle.compare {
        set.insert(c.clone());
    }
    set.into_iter().collect()
}

/// `input_blob_ids` computed from rows already read from a tree
/// (`git ls-tree -r COMMIT`), so a planner can identify a check at a commit
/// that is not checked out. Same shape and order as `input_blob_ids`.
pub fn input_blob_ids_from_rows(m: &Manifest, tree: &BTreeMap<String, String>) -> Vec<Value> {
    let mut rows: BTreeMap<String, Option<String>> = BTreeMap::new();
    for input in &m.inputs {
        let spec = input.trim_end_matches('/');
        let prefix = format!("{spec}/");
        let mut found = false;
        for (path, blob) in tree.range(spec.to_string()..) {
            if path == spec || path.starts_with(&prefix) {
                rows.insert(path.clone(), Some(blob.clone()));
                found = true;
            } else if !path.starts_with(spec) {
                break;
            }
        }
        if !found {
            rows.entry(spec.to_string()).or_insert(None);
        }
    }
    rows.into_iter()
        .map(|(path, blob)| json!({ "path": path, "blob_id": blob }))
        .collect()
}

/// The identity object from parts already read: `inputs` as `input_blob_ids`
/// gives them, `golden` the bytes of the oracle golden file if the manifest
/// names one. `identity_object` and the planner both build on this so the
/// hashed shape has one definition.
pub fn identity_object_from_parts(
    m: &Manifest,
    input_blob_ids: Vec<Value>,
    golden: Option<&[u8]>,
    compiler_identity: &str,
    flags: &[String],
    dependency_check_ids: &[String],
    binding: &Binding,
) -> Result<Value, String> {
    let mut deps: Vec<String> = dependency_check_ids.to_vec();
    deps.sort();
    let mut goldens: Vec<String> = Vec::new();
    if m.oracle.golden.is_some() {
        match golden {
            Some(bytes) => goldens.push(sha256_hex(bytes)),
            None => {
                return Err(format!(
                    "golden file {} is missing",
                    m.oracle.golden.as_deref().unwrap_or("")
                ))
            }
        }
    }
    goldens.sort();
    Ok(json!({
        "manifest_digest": m.digest,
        "input_blob_ids": input_blob_ids,
        "compiler_identity": compiler_identity,
        "flags": flags,
        "dependency_check_ids": deps,
        "golden_digests": goldens,
        "scope_binding": scope_binding(m.cache_scope, binding)?,
    }))
}

/// The object that `check_id` hashes.
pub fn identity_object(
    m: &Manifest,
    root: &Path,
    compiler_identity: &str,
    flags: &[String],
    dependency_check_ids: &[String],
    binding: &Binding,
) -> Result<Value, String> {
    let golden = match &m.oracle.golden {
        Some(g) => {
            Some(fs::read(root.join(g)).map_err(|e| format!("cannot read golden {g}: {e}"))?)
        }
        None => None,
    };
    identity_object_from_parts(
        m,
        input_blob_ids(m, root)?,
        golden.as_deref(),
        compiler_identity,
        flags,
        dependency_check_ids,
        binding,
    )
}

pub fn id_of(object: &Value) -> Result<String, String> {
    let bytes = canonical(object).map_err(|e| e.to_string())?;
    Ok(sha256_hex(&bytes))
}

/// `check_id` (ADR 0033 Decision 3).
pub fn check_id(
    m: &Manifest,
    root: &Path,
    compiler_identity: &str,
    flags: &[String],
    dependency_check_ids: &[String],
    binding: &Binding,
) -> Result<String, String> {
    id_of(&identity_object(
        m,
        root,
        compiler_identity,
        flags,
        dependency_check_ids,
        binding,
    )?)
}

/// Ids and hashed objects for gate `target` and everything it rests on,
/// dependencies first. Dependency ids are computed with the same compiler,
/// flags and binding.
pub fn graph_identities(
    graph: &Graph,
    target: usize,
    root: &Path,
    compiler_identity: &str,
    flags: &[String],
    binding: &Binding,
) -> Result<BTreeMap<String, (String, Value)>, String> {
    let wanted = graph.with_dependencies(&[target]);
    let mut done: BTreeMap<String, (String, Value)> = BTreeMap::new();
    for &i in graph.order() {
        if !wanted.contains(&i) {
            continue;
        }
        let m = &graph.node(i).manifest;
        let mut dep_ids = Vec::new();
        for d in declared_dependencies(m) {
            match done.get(&d) {
                Some((id, _)) => dep_ids.push(id.clone()),
                None => return Err(format!("gate {} declares unknown dependency {d}", m.gate)),
            }
        }
        let obj = identity_object(m, root, compiler_identity, flags, &dep_ids, binding)?;
        let id = id_of(&obj)?;
        done.insert(m.gate.clone(), (id, obj));
    }
    Ok(done)
}

/// Planner-style path filter: the gates whose `inputs` contain a changed
/// path, plus everything each of them declares it depends on. A declared
/// dependency is never dropped by path filtering (ADR 0033 test 17).
pub fn select_changed(graph: &Graph, changed: &[String]) -> BTreeSet<usize> {
    let touched: Vec<usize> = (0..graph.len())
        .filter(|&i| {
            graph.node(i).manifest.inputs.iter().any(|input| {
                let dir = input.trim_end_matches('/');
                changed
                    .iter()
                    .any(|c| c == dir || c.starts_with(&format!("{dir}/")))
            })
        })
        .collect();
    let mut seeds = touched;
    for &i in &seeds.clone() {
        if let Some(j) = graph
            .node(i)
            .manifest
            .oracle
            .compare
            .as_deref()
            .and_then(|c| graph.index_of(c))
        {
            seeds.push(j);
        }
    }
    graph.with_dependencies(&seeds)
}

/// The binding for the machine this process runs on.
pub fn local_binding(hardware_identity: &str) -> Result<Binding, String> {
    let machine = crate::runner::machine(&crate::resources::Hardware::detect());
    let machine_digest = crate::evidence::canonical_digest(&machine).map_err(|e| e.to_string())?;
    Ok(Binding {
        arch: std::env::consts::ARCH.to_string(),
        machine_digest,
        hardware_identity: hardware_identity.to_string(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph;
    use crate::manifest;
    use crate::testutil::{git, init_repo, write_file, TempDir};
    use std::process::Command;

    fn gate(id: &str, extra: &str) -> String {
        format!(
            "gate: {id}\nmanifest_version: 2\nowner: t\nrequires:\n  - host\ninputs:\n  - data_{id}/\nrun:\n  exec: tools/run.sh\nexpects:\n  exit: 0\n  verdict: PASS\ntimeout: 5s\n{extra}"
        )
    }

    fn binding(machine: &str) -> Binding {
        Binding {
            arch: "aarch64".into(),
            machine_digest: machine.repeat(64),
            hardware_identity: "hw-1".into(),
        }
    }

    fn repo(extra: &str) -> TempDir {
        let g = gate("A", extra);
        init_repo(
            "ident",
            &[
                ("gates/a.gate", g.as_str(), false),
                ("tools/run.sh", "#!/bin/sh\nexit 0\n", true),
                ("data_A/one.txt", "one\n", false),
                ("data_A/sub/two.txt", "two\n", false),
                ("elsewhere.txt", "x\n", false),
            ],
        )
    }

    fn parsed(t: &TempDir) -> Manifest {
        manifest::parse(&std::fs::read(t.path().join("gates/a.gate")).unwrap()).unwrap()
    }

    fn id(t: &TempDir, compiler: &str, flags: &[&str], deps: &[&str], b: &Binding) -> String {
        let flags: Vec<String> = flags.iter().map(|s| s.to_string()).collect();
        let deps: Vec<String> = deps.iter().map(|s| s.to_string()).collect();
        check_id(&parsed(t), t.path(), compiler, &flags, &deps, b).unwrap()
    }

    fn commit(t: &TempDir, rel: &str, content: &str) {
        write_file(t.path(), rel, content, false);
        git(t.path(), &["add", "-A"]);
        git(t.path(), &["commit", "-q", "-m", "edit"]);
    }

    #[test]
    fn t1_same_inputs_twice_give_the_same_id() {
        let t = repo("");
        let b = binding("a");
        let first = id(&t, "rustc 1", &["-O"], &[], &b);
        assert_eq!(first.len(), 64);
        assert_eq!(first, id(&t, "rustc 1", &["-O"], &[], &b));
        // A second repository with byte-identical content has the same id.
        let u = repo("");
        assert_eq!(first, id(&u, "rustc 1", &["-O"], &[], &b));
    }

    #[test]
    fn t2_one_input_byte_changes_the_id_and_an_unrelated_file_does_not() {
        let t = repo("");
        let b = binding("a");
        let before = id(&t, "c", &[], &[], &b);
        commit(&t, "elsewhere.txt", "y\n");
        assert_eq!(before, id(&t, "c", &[], &[], &b));
        commit(&t, "data_A/sub/two.txt", "twp\n");
        let after = id(&t, "c", &[], &[], &b);
        assert_ne!(before, after);
        commit(&t, "data_A/sub/two.txt", "two\n");
        assert_eq!(before, id(&t, "c", &[], &[], &b));
    }

    #[test]
    fn input_blob_ids_are_sorted_git_blob_ids_and_missing_inputs_are_kept() {
        let t = repo("");
        let rows = input_blob_ids(&parsed(&t), t.path()).unwrap();
        let paths: Vec<&str> = rows.iter().map(|r| r["path"].as_str().unwrap()).collect();
        assert_eq!(paths, ["data_A/one.txt", "data_A/sub/two.txt"]);
        // git's own blob id of "one\n".
        let mut cmd = Command::new("git");
        cmd.args(["hash-object", "--stdin"])
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped());
        // Through the spawn gate like every other spawn in this binary (sc#603):
        // a raw spawn here left a child between fork and exec holding a copy of
        // another test's GPU lock descriptor, and that test saw LockHeld after
        // its own guard was dropped.
        let mut child = crate::process::spawn_gated(&mut cmd).unwrap();
        std::io::Write::write_all(child.stdin.as_mut().unwrap(), b"one\n").unwrap();
        let out = child.wait_with_output().unwrap();
        assert_eq!(
            rows[0]["blob_id"].as_str().unwrap(),
            String::from_utf8_lossy(&out.stdout).trim()
        );
        // A declared input that matches no tracked file is a null row.
        let g = gate("A", "").replace("data_A/", "nothing_here/");
        let m = manifest::parse(g.as_bytes()).unwrap();
        let rows = input_blob_ids(&m, t.path()).unwrap();
        assert_eq!(rows.len(), 1);
        assert!(rows[0]["blob_id"].is_null());
        // An untracked file under an input is not part of the identity.
        let before = id(&t, "c", &[], &[], &binding("a"));
        write_file(t.path(), "data_A/untracked.txt", "u\n", false);
        assert_eq!(before, id(&t, "c", &[], &[], &binding("a")));
    }

    #[test]
    fn t3_compiler_identity_and_flags_change_the_id() {
        let t = repo("");
        let b = binding("a");
        let base = id(&t, "rustc 1", &["-O"], &[], &b);
        assert_ne!(base, id(&t, "rustc 2", &["-O"], &[], &b));
        assert_ne!(base, id(&t, "rustc 1", &["-O3"], &[], &b));
        assert_ne!(base, id(&t, "rustc 1", &[], &[], &b));
        assert_ne!(base, id(&t, "rustc 1", &["-O", "-g"], &[], &b));
    }

    #[test]
    fn t4_machine_scope_follows_the_machine_and_portable_does_not() {
        let machine = repo("cache_scope: machine\n");
        let portable = repo("cache_scope: portable\n");
        let arch = repo("cache_scope: arch\n");
        let (a, z) = (binding("a"), binding("b"));
        assert_ne!(
            id(&machine, "c", &[], &[], &a),
            id(&machine, "c", &[], &[], &z)
        );
        assert_eq!(
            id(&portable, "c", &[], &[], &a),
            id(&portable, "c", &[], &[], &z)
        );
        assert_eq!(id(&arch, "c", &[], &[], &a), id(&arch, "c", &[], &[], &z));
        let mut other_arch = binding("a");
        other_arch.arch = "x86_64".into();
        assert_ne!(
            id(&arch, "c", &[], &[], &a),
            id(&arch, "c", &[], &[], &other_arch)
        );
        // The default scope is machine.
        let default = repo("");
        assert_eq!(parsed(&default).cache_scope, CacheScope::Machine);
    }

    #[test]
    fn hardware_scope_adds_the_hardware_identity_and_refuses_without_it() {
        let t = repo("cache_scope: hardware\n");
        let a = binding("a");
        let mut other = a.clone();
        other.hardware_identity = "hw-2".into();
        assert_ne!(id(&t, "c", &[], &[], &a), id(&t, "c", &[], &[], &other));
        let mut none = a.clone();
        none.hardware_identity = String::new();
        assert!(check_id(&parsed(&t), t.path(), "c", &[], &[], &none).is_err());
        // The portable scope binds to nothing at all.
        assert_eq!(scope_binding(CacheScope::Portable, &a).unwrap(), "");
        assert_eq!(scope_binding(CacheScope::Arch, &a).unwrap(), "aarch64");
    }

    #[test]
    fn golden_file_bytes_enter_the_identity() {
        let g = gate("A", "oracle:\n  golden: golden.txt\n");
        let t = init_repo(
            "ident-golden",
            &[
                ("gates/a.gate", g.as_str(), false),
                ("tools/run.sh", "#!/bin/sh\n", true),
                ("golden.txt", "v1\n", false),
            ],
        );
        let b = binding("a");
        let first = id(&t, "c", &[], &[], &b);
        write_file(t.path(), "golden.txt", "v2\n", false);
        assert_ne!(first, id(&t, "c", &[], &[], &b));
        std::fs::remove_file(t.path().join("golden.txt")).unwrap();
        assert!(check_id(&parsed(&t), t.path(), "c", &[], &[], &b).is_err());
    }

    #[test]
    fn dependency_ids_are_sorted_and_part_of_the_object() {
        let t = repo("");
        let b = binding("a");
        let x = "x".repeat(64);
        let y = "y".repeat(64);
        let obj =
            identity_object(&parsed(&t), t.path(), "c", &[], &[y.clone(), x.clone()], &b).unwrap();
        assert_eq!(obj["dependency_check_ids"], serde_json::json!([x, y]));
        assert_eq!(
            id(&t, "c", &[], &["x", "y"], &b),
            id(&t, "c", &[], &["y", "x"], &b)
        );
        assert_ne!(id(&t, "c", &[], &["x"], &b), id(&t, "c", &[], &[], &b));
        for k in [
            "manifest_digest",
            "input_blob_ids",
            "compiler_identity",
            "flags",
            "golden_digests",
            "scope_binding",
        ] {
            assert!(obj.get(k).is_some(), "{k}");
        }
        assert_eq!(obj.as_object().unwrap().len(), 7);
    }

    fn two_gate_repo() -> TempDir {
        let a = gate("A", "depends_on:\n  - B\n");
        let b = gate("B", "");
        init_repo(
            "ident-graph",
            &[
                ("gates/a.gate", a.as_str(), false),
                ("gates/b.gate", b.as_str(), false),
                ("tools/run.sh", "#!/bin/sh\n", true),
                ("data_A/a.txt", "a\n", false),
                ("data_B/b.txt", "b\n", false),
            ],
        )
    }

    #[test]
    fn t17_a_declared_dependency_is_part_of_the_identity_even_when_no_changed_path_touches_it() {
        let t = two_gate_repo();
        let b = binding("a");
        let g = graph::load(t.path()).unwrap();
        let ia = g.index_of("A").unwrap();
        let before = graph_identities(&g, ia, t.path(), "c", &[], &b).unwrap();
        // Only B's input changes. A's own inputs are untouched, but A's id moves.
        commit(&t, "data_B/b.txt", "b2\n");
        let g = graph::load(t.path()).unwrap();
        let after = graph_identities(&g, ia, t.path(), "c", &[], &b).unwrap();
        assert_ne!(before["B"].0, after["B"].0);
        assert_ne!(before["A"].0, after["A"].0);
        assert_eq!(
            before["A"].1["input_blob_ids"],
            after["A"].1["input_blob_ids"]
        );
        // A's recorded dependency id is exactly B's id.
        assert_eq!(
            after["A"].1["dependency_check_ids"],
            serde_json::json!([after["B"].0])
        );
    }

    #[test]
    fn t17_path_filtering_keeps_declared_dependencies() {
        let t = two_gate_repo();
        let g = graph::load(t.path()).unwrap();
        let (ia, ib) = (g.index_of("A").unwrap(), g.index_of("B").unwrap());
        // Only A's input changed; B is not touched by any changed path, yet
        // A cannot be checked without it.
        let picked = select_changed(&g, &["data_A/a.txt".to_string()]);
        assert!(picked.contains(&ia));
        assert!(picked.contains(&ib));
        // Only B changed: B alone is selected (dependents are not pulled in).
        let picked = select_changed(&g, &["data_B/b.txt".to_string()]);
        assert!(picked.contains(&ib) && !picked.contains(&ia));
        // Nothing relevant changed: nothing selected.
        assert!(select_changed(&g, &["elsewhere.txt".to_string()]).is_empty());
        // A path that merely shares a name prefix with an input does not match.
        assert!(select_changed(&g, &["data_A2/a.txt".to_string()]).is_empty());
    }

    #[test]
    fn an_oracle_compare_gate_is_a_declared_dependency() {
        let m =
            manifest::parse(gate("A", "oracle:\n  compare: B\ndepends_on:\n  - C\n").as_bytes())
                .unwrap();
        assert_eq!(declared_dependencies(&m), ["B", "C"]);
    }
}

#[cfg(test)]
mod rows_tests {
    use super::*;
    use crate::manifest;
    use crate::testutil::init_repo;
    use std::process::Command;

    #[test]
    fn rows_from_a_tree_listing_give_the_same_identity_as_the_index() {
        let g = "gate: A\nmanifest_version: 2\nowner: t\nrequires:\n  - host\ninputs:\n  - data/\n  - missing.txt\nrun:\n  exec: run.sh\nexpects:\n  exit: 0\n  verdict: PASS\ntimeout: 5s\n";
        let t = init_repo(
            "ident-rows",
            &[
                ("a.gate", g, false),
                ("run.sh", "#!/bin/sh\n", true),
                ("data/one.txt", "one\n", false),
                ("data/sub/two.txt", "two\n", false),
                ("data2/three.txt", "three\n", false),
            ],
        );
        let m = manifest::parse(g.as_bytes()).unwrap();
        let from_index = input_blob_ids(&m, t.path()).unwrap();
        let mut cmd = Command::new("git");
        cmd.arg("-C").arg(t.path()).args(["ls-tree", "-r", "HEAD"]);
        let out = process::output_gated(&mut cmd).unwrap();
        let mut tree = BTreeMap::new();
        for line in String::from_utf8_lossy(&out.stdout).lines() {
            let (meta, path) = line.split_once('\t').unwrap();
            let blob = meta.split(' ').nth(2).unwrap();
            tree.insert(path.to_string(), blob.to_string());
        }
        let from_rows = input_blob_ids_from_rows(&m, &tree);
        assert_eq!(from_index, from_rows);
        let paths: Vec<&str> = from_rows
            .iter()
            .map(|r| r["path"].as_str().unwrap())
            .collect();
        assert_eq!(paths, ["data/one.txt", "data/sub/two.txt", "missing.txt"]);
        assert!(from_rows[2]["blob_id"].is_null());
        let b = Binding {
            arch: "a".into(),
            machine_digest: "m".repeat(64),
            hardware_identity: String::new(),
        };
        let a = identity_object(&m, t.path(), "c", &[], &[], &b).unwrap();
        let r = identity_object_from_parts(&m, from_rows, None, "c", &[], &[], &b).unwrap();
        assert_eq!(a, r);
    }
}
