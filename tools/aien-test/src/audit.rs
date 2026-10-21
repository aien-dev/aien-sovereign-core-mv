//! Input audit (L335): does a gate read only what its manifest declares?
//!
//! A gate that says `cache: inputs_complete` lends its result to every head
//! whose declared inputs are byte-identical (`cache::cross_head`). If the gate
//! reads a file it does not declare, a head that changed only that file gets
//! another head's verdict: a false PASS. This module checks the promise
//! against a real run: `aien-audit-inputs` runs the gate once under
//! `strace -f -y -Y` (every file opened, with the path the kernel resolved and
//! the program that opened it) and lists each opened file the manifest does
//! not cover.
//!
//! What counts as covered:
//! - inside the repository: a declared input (file, directory or `*` pattern),
//!   a build file every key carries ([`runner::IMPLIED_BUILD_FILES`]), the
//!   manifest itself and `run.exec`;
//! - outside it: an `outside_inputs` entry, or the toolchain and system
//!   (`/usr`, `/lib`, `/etc`, `/proc`, `/sys`, `/dev`, `/run`, `~/.rustup`,
//!   `~/.cargo`). The toolchain is pinned by `rust-toolchain.toml` and the
//!   registry sources by `Cargo.lock`, both in every key; a system library
//!   upgrade is not (a limit of this audit, stated in the report).
//!
//! Not judged as reads (each reported with a count, never silently):
//! - `git` and `tar` opening the tree: the commit and the copy step. What the
//!   copy's consumers read is mapped back (next item). A script that branches
//!   on git output is not caught here; read the script.
//! - files under the private `TMPDIR` the audit gives the run: a path
//!   `TMPDIR/<dir>/REL` whose `REL` exists in the repository is judged as the
//!   repository file `REL` (the PREFILL mutants build a copy of the tree);
//!   anything else there is the run's own scratch.
//! - the cargo target dir, `.git/`, `/tmp`, and opens for writing only.
//!
//! An audit only sees what one run read: run it with a cold target dir, or
//! cargo skips reading the sources of crates it considers fresh.
use crate::manifest::Manifest;
use crate::runner;
use std::collections::{BTreeMap, BTreeSet};
use std::path::{Path, PathBuf};

/// One successful open (or exec) from an strace log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Opened {
    /// The program's name as the kernel reports it (15 characters at most).
    pub comm: String,
    /// The path the kernel resolved (the `-y` decoration), absolute.
    pub path: String,
    /// Opened only for writing (`O_WRONLY`): not a read.
    pub write_only: bool,
}

/// Parse `strace -f -qq -Y -y -e trace=open,openat,openat2,execve
/// -e status=successful` output. Lines that are not a successful open or
/// exec of a named file are skipped.
pub fn parse_strace(text: &str) -> Vec<Opened> {
    let mut out = Vec::new();
    for line in text.lines() {
        let Some(lt) = line.find('<') else { continue };
        let Some(gt) = line[lt..].find("> ") else {
            continue;
        };
        let comm = line[lt + 1..lt + gt].to_string();
        let call = &line[lt + gt + 2..];
        if call.starts_with("execve(") {
            // execve("/abs/path", ...) = 0: the program file itself is read.
            if let Some(p) = quoted(call) {
                if p.starts_with('/') && call.trim_end().ends_with("= 0") {
                    out.push(Opened {
                        comm,
                        path: p,
                        write_only: false,
                    });
                }
            }
            continue;
        }
        if !(call.starts_with("openat(")
            || call.starts_with("open(")
            || call.starts_with("openat2("))
        {
            continue;
        }
        // ... ) = 3</resolved/path>
        let Some(eq) = call.rfind(") = ") else {
            continue;
        };
        let ret = &call[eq + 4..];
        let (Some(a), Some(b)) = (ret.find('<'), ret.rfind('>')) else {
            continue;
        };
        if b <= a + 1 {
            continue;
        }
        let path = ret[a + 1..b].to_string();
        let write_only = call[..eq].contains("O_WRONLY");
        out.push(Opened {
            comm,
            path,
            write_only,
        });
    }
    out
}

/// The first double-quoted string of a syscall line (no escapes expected in
/// the paths we judge; a path with `"` is skipped).
fn quoted(call: &str) -> Option<String> {
    let a = call.find('"')?;
    let b = call[a + 1..].find('"')?;
    Some(call[a + 1..a + 1 + b].to_string())
}

/// Where the audited run lived.
#[derive(Debug, Clone)]
pub struct Places {
    /// The repository root (canonical).
    pub root: PathBuf,
    /// The repository-relative path of the manifest.
    pub manifest_rel: String,
    /// The cargo target dir of the run (canonical).
    pub target_dir: PathBuf,
    /// The private `TMPDIR` given to the run (canonical).
    pub tmpdir: PathBuf,
    /// The judge's home, for `~/` entries and `~/.rustup`, `~/.cargo`.
    pub home: PathBuf,
    /// The files git tracks at HEAD (repo-relative). A repository path that is
    /// neither one of them nor a directory holding one was made by the run
    /// (build outputs in the tree, say `build/`), so it is not an input. None:
    /// every path is judged.
    pub tracked: Option<BTreeSet<String>>,
    /// Sibling checkouts the judge provides (`../<repo>`, canonical). Their
    /// commits are in every key as pins, so reading them is covered.
    pub siblings: Vec<PathBuf>,
}

/// The audit's findings.
#[derive(Debug, Default, Clone)]
pub struct Report {
    /// Repository files read but not declared (repo-relative), with a reader.
    pub undeclared: BTreeMap<String, String>,
    /// Files outside the repository read but not declared, with a reader.
    pub undeclared_outside: BTreeMap<String, String>,
    /// Counts of opens not judged as reads, by reason.
    pub ignored: BTreeMap<&'static str, usize>,
    /// Declared repository files the run did read (for the record).
    pub covered: BTreeSet<String>,
    /// The run started cargo, rustc, clippy or rustdoc.
    pub uses_cargo: bool,
}

impl Report {
    /// The promise holds: nothing undeclared, and a `target_lock: none` gate
    /// never started cargo.
    pub fn ok(&self, m: &Manifest) -> bool {
        self.undeclared.is_empty()
            && self.undeclared_outside.is_empty()
            && (m.target_lock || !self.uses_cargo)
    }
}

const SYSTEM: [&str; 9] = [
    "/usr/", "/lib/", "/lib64/", "/bin/", "/sbin/", "/etc/", "/proc/", "/sys/", "/dev/",
];
const CARGO_PROGRAMS: [&str; 6] = [
    "cargo",
    "rustc",
    "clippy-driver",
    "cargo-clippy",
    "rustdoc",
    "cargo-fmt",
];

/// Is the repository path `rel` covered by the manifest's declarations?
pub fn covered(m: &Manifest, manifest_rel: &str, rel: &str) -> bool {
    let rel = rel.trim_end_matches('/');
    if rel == manifest_rel || rel == m.exec || runner::IMPLIED_BUILD_FILES.contains(&rel) {
        return true;
    }
    m.inputs.iter().any(|i| input_covers(i, rel))
}

/// One declared input against one path: equal, inside a declared directory,
/// or matched component by component by a `*` pattern (whose last component
/// may itself be a directory the path is inside).
fn input_covers(input: &str, rel: &str) -> bool {
    let i = input.trim().trim_start_matches("./").trim_end_matches('/');
    if i.is_empty() || i == "." {
        return true;
    }
    let ip: Vec<&str> = i.split('/').collect();
    let rp: Vec<&str> = rel.split('/').collect();
    if rp.len() < ip.len() {
        return false;
    }
    ip.iter()
        .zip(&rp)
        .all(|(a, b)| runner::wildcard_match(a, b))
}

/// Judge every open of an audited run.
pub fn judge(m: &Manifest, places: &Places, opened: &[Opened]) -> Report {
    let mut r = Report::default();
    let root = places.root.to_string_lossy().into_owned() + "/";
    let target = places.target_dir.to_string_lossy().into_owned() + "/";
    let tmpdir = places.tmpdir.to_string_lossy().into_owned() + "/";
    let home = places.home.to_string_lossy().into_owned();
    let toolchain = [
        format!("{home}/.rustup/"),
        format!("{home}/.cargo/"),
        "/var/lib/aien-run/rustup/".into(),
        "/var/lib/aien-run/cargo/".into(),
    ];
    let outside: Vec<String> = m
        .outside_inputs
        .iter()
        .map(|o| {
            let o = o.trim_end_matches('/');
            match o.strip_prefix("~/") {
                Some(rest) => format!("{home}/{rest}"),
                None => o.to_string(),
            }
        })
        .collect();
    for o in opened {
        if CARGO_PROGRAMS.contains(&o.comm.as_str()) {
            r.uses_cargo = true;
        }
        if o.write_only {
            bump(&mut r, "opened for writing only");
            continue;
        }
        let p = o.path.as_str();
        let p_dir = format!("{}/", p.trim_end_matches('/'));
        if p_dir.starts_with(&target) {
            bump(&mut r, "cargo target dir (outputs)");
            continue;
        }
        // A file of the private TMPDIR whose tail names a repository file is
        // that file (a copy of the tree); the rest is the run's scratch.
        if let Some(t) = p.strip_prefix(&tmpdir) {
            let tail = t.split_once('/').map(|x| x.1).unwrap_or("");
            if !tail.is_empty() && places.root.join(tail).exists() && !tail.starts_with(".git") {
                judge_repo(m, places, tail, &o.comm, &mut r);
            } else {
                bump(&mut r, "private TMPDIR scratch");
            }
            continue;
        }
        if places
            .siblings
            .iter()
            .any(|s| p_dir.starts_with(&format!("{}/", s.display())))
        {
            bump(&mut r, "sibling checkout (its commit is a pin in the key)");
            continue;
        }
        if p_dir.starts_with(&root) || p_dir == root {
            let rel = p_dir[root.len().min(p_dir.len())..]
                .trim_end_matches('/')
                .to_string();
            if rel.is_empty() {
                bump(&mut r, "the repository root directory itself");
            } else if rel == ".git" || rel.starts_with(".git/") {
                bump(&mut r, "git metadata (.git)");
            } else if o.comm == "git" || o.comm == "tar" {
                bump(&mut r, "git or tar reading the tree (commit, copy step)");
            } else {
                judge_repo(m, places, &rel, &o.comm, &mut r);
            }
            continue;
        }
        if outside
            .iter()
            .any(|d| p == d || p.starts_with(&format!("{d}/")))
        {
            bump(&mut r, "declared outside_inputs");
            continue;
        }
        if toolchain.iter().any(|t| p_dir.starts_with(t.as_str())) {
            bump(
                &mut r,
                "toolchain and registry (rust-toolchain.toml, Cargo.lock)",
            );
            continue;
        }
        if SYSTEM.iter().any(|s| p_dir.starts_with(s)) || p == "/" {
            bump(&mut r, "system (not in the key: a limit)");
            continue;
        }
        if p.starts_with("/tmp/") || p.starts_with("/run/") || p.starts_with("/var/tmp/") {
            bump(&mut r, "/tmp, /run (locks and scratch)");
            continue;
        }
        r.undeclared_outside
            .entry(p.to_string())
            .or_insert_with(|| o.comm.clone());
    }
    r
}

fn bump(r: &mut Report, why: &'static str) {
    *r.ignored.entry(why).or_insert(0) += 1;
}

fn judge_repo(m: &Manifest, places: &Places, rel: &str, comm: &str, r: &mut Report) {
    if let Some(t) = &places.tracked {
        let dir = format!("{rel}/");
        let in_tree = t.contains(rel)
            || t.range(dir.clone()..)
                .next()
                .is_some_and(|f| f.starts_with(&dir));
        if !in_tree {
            bump(r, "not in the tree at HEAD (made by the run)");
            return;
        }
    }
    if covered(m, &places.manifest_rel, rel) {
        r.covered.insert(rel.to_string());
    } else {
        r.undeclared
            .entry(rel.to_string())
            .or_insert_with(|| comm.to_string());
    }
}

/// The report as text, one finding per line; `ok` decides the last line.
pub fn render(m: &Manifest, r: &Report) -> String {
    let mut s = String::new();
    s.push_str(&format!(
        "AUDIT {} covered_repo_files={} uses_cargo={}\n",
        m.gate,
        r.covered.len(),
        r.uses_cargo
    ));
    for (why, n) in &r.ignored {
        s.push_str(&format!("IGNORED {n} {why}\n"));
    }
    for (p, who) in &r.undeclared {
        s.push_str(&format!("UNDECLARED {p} (read by {who})\n"));
    }
    for (p, who) in &r.undeclared_outside {
        s.push_str(&format!("UNDECLARED_OUTSIDE {p} (read by {who})\n"));
    }
    if !m.target_lock && r.uses_cargo {
        s.push_str("TARGET_LOCK_BROKEN target_lock: none but the run started cargo or rustc\n");
    }
    s.push_str(if r.ok(m) {
        "RESULT PASS inputs complete\n"
    } else {
        "RESULT FAIL inputs incomplete\n"
    });
    s
}

/// Canonical form of `p`, or `p` itself when it does not exist yet.
pub fn canon(p: &Path) -> PathBuf {
    std::fs::canonicalize(p).unwrap_or_else(|_| p.to_path_buf())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest;

    fn gate(extra: &str) -> Manifest {
        let text = format!(
            "gate: G\nmanifest_version: 1\nowner: t\nrequires:\n  - host\ninputs:\n  - crates/a/\n  - crates/*/Cargo.toml\n{extra}run:\n  exec: tests/run.sh\nexpects:\n  exit: 0\n  verdict: PASS\ntimeout: 1m\ncache: inputs_complete\n"
        );
        manifest::parse(text.as_bytes()).unwrap()
    }

    fn places(root: &Path) -> Places {
        Places {
            root: root.to_path_buf(),
            manifest_rel: "tests/g.gate".into(),
            target_dir: root.join("target"),
            tmpdir: PathBuf::from("/audit/tmp"),
            home: PathBuf::from("/home/op"),
            tracked: None,
            siblings: Vec::new(),
        }
    }

    fn line(comm: &str, path: &str) -> String {
        format!("7<{comm}> openat(AT_FDCWD</r>, \"x\", O_RDONLY|O_CLOEXEC) = 3<{path}>\n")
    }

    #[test]
    fn parses_opens_execs_and_writes() {
        let t = "12<sh> openat(AT_FDCWD</r>, \"a\", O_RDONLY) = 3</r/a>\n\
                 12<sh> openat(AT_FDCWD</r>, \"o\", O_WRONLY|O_CREAT|O_TRUNC, 0666) = 4</r/o>\n\
                 13<sh> execve(\"/usr/bin/cargo\", [\"cargo\"], 0xffff /* 3 vars */) = 0\n\
                 13<cargo> +++ exited with 0 +++\n";
        let o = parse_strace(t);
        assert_eq!(o.len(), 3);
        assert_eq!(
            o[0],
            Opened {
                comm: "sh".into(),
                path: "/r/a".into(),
                write_only: false
            }
        );
        assert!(o[1].write_only);
        assert_eq!(o[2].path, "/usr/bin/cargo");
    }

    #[test]
    fn a_dropped_input_is_caught_and_a_declared_one_is_not() {
        let dir = tempdir();
        let root = dir.as_path();
        for f in [
            "crates/a/src/lib.rs",
            "crates/b/src/lib.rs",
            "crates/b/Cargo.toml",
        ] {
            std::fs::create_dir_all(root.join(f).parent().unwrap()).unwrap();
            std::fs::write(root.join(f), "x").unwrap();
        }
        let r = root.display();
        let trace = line("rustc", &format!("{r}/crates/a/src/lib.rs"))
            + &line("rustc", &format!("{r}/crates/b/src/lib.rs"))
            + &line("cargo", &format!("{r}/crates/b/Cargo.toml"))
            + &line("rustc", &format!("{r}/target/release/deps/liba.rlib"))
            + &line("rustc", "/usr/lib/libc.so.6");
        let opened = parse_strace(&trace);
        // crates/b/ is read but only its Cargo.toml is declared: FAIL.
        let m = gate("");
        let rep = judge(&m, &places(root), &opened);
        assert_eq!(
            rep.undeclared.keys().collect::<Vec<_>>(),
            vec!["crates/b/src/lib.rs"]
        );
        assert!(!rep.ok(&m));
        // Declare it: PASS.
        let m2 = gate("  - crates/b/\n");
        let rep2 = judge(&m2, &places(root), &opened);
        assert!(rep2.ok(&m2), "{}", render(&m2, &rep2));
        assert!(
            rep2.covered.contains("crates/b/Cargo.toml"),
            "the * pattern covers it"
        );
    }

    #[test]
    fn outside_reads_need_outside_inputs() {
        let dir = tempdir();
        let opened = parse_strace(&line(
            "prefill_e2e2-4",
            "/home/op/models/T/model.safetensors",
        ));
        let m = gate("");
        assert_eq!(
            judge(&m, &places(&dir), &opened).undeclared_outside.len(),
            1
        );
        let m2 = gate("outside_inputs:\n  - ~/models/T/\n");
        let rep = judge(&m2, &places(&dir), &opened);
        assert!(rep.ok(&m2), "{}", render(&m2, &rep));
    }

    #[test]
    fn a_copy_in_the_private_tmpdir_is_judged_as_the_repository_file() {
        let dir = tempdir();
        std::fs::create_dir_all(dir.join("crates/z")).unwrap();
        std::fs::write(dir.join("crates/z/lib.rs"), "x").unwrap();
        let trace = line("tar", &format!("{}/crates/z/lib.rs", dir.display()))
            + &line("rustc", "/audit/tmp/tmp.Ab12/crates/z/lib.rs")
            + &line("rustc", "/audit/tmp/tmp.Ab12/scratch.o");
        let rep = judge(&gate(""), &places(&dir), &parse_strace(&trace));
        assert_eq!(
            rep.undeclared.keys().collect::<Vec<_>>(),
            vec!["crates/z/lib.rs"]
        );
        assert_eq!(rep.ignored.get("private TMPDIR scratch"), Some(&1));
    }

    #[test]
    fn target_lock_none_with_cargo_fails() {
        let dir = tempdir();
        let m = gate("target_lock: none\n");
        let opened = parse_strace("9<sh> execve(\"/usr/bin/cargo\", [\"cargo\"], 0x1 /* 1 vars */) = 0\n9<cargo> openat(AT_FDCWD</r>, \"/etc/x\", O_RDONLY) = 3</etc/x>\n");
        let rep = judge(&m, &places(&dir), &opened);
        assert!(rep.uses_cargo && !rep.ok(&m));
        assert!(render(&m, &rep).contains("TARGET_LOCK_BROKEN"));
    }

    #[test]
    fn files_the_run_made_in_the_tree_are_not_inputs() {
        let dir = tempdir();
        let mut p = places(&dir);
        p.tracked = Some(["src/x.c".to_string()].into_iter().collect());
        let r = dir.display();
        let trace = line("cc", &format!("{r}/src/x.c"))
            + &line("ld", &format!("{r}/build/x.o"))
            + &line("make", &format!("{r}/src"));
        let rep = judge(&gate(""), &p, &parse_strace(&trace));
        // src/x.c and the src listing are tracked and undeclared; build/x.o is the run's own.
        assert_eq!(
            rep.undeclared.keys().collect::<Vec<_>>(),
            vec!["src", "src/x.c"]
        );
        assert_eq!(
            rep.ignored.get("not in the tree at HEAD (made by the run)"),
            Some(&1)
        );
    }

    fn tempdir() -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aien-audit-{}-{:?}",
            std::process::id(),
            std::thread::current().id()
        ));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        canon(&d)
    }
}
