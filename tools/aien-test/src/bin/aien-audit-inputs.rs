//! aien-audit-inputs: run one gate under strace and list every file it read
//! that its manifest does not declare (see `aien_test::audit`).
//!
//! ```text
//! aien-audit-inputs [--root DIR] --target-dir DIR --work DIR GATE_FILE
//! aien-audit-inputs [--root DIR] --target-dir DIR --work DIR --trace FILE GATE_FILE
//! ```
//! The first form runs `run.exec` with the manifest's args from the
//! repository root, with `CARGO_TARGET_DIR=--target-dir` (use a cold one) and
//! `TMPDIR=<work>/tmp`, and keeps the trace in `<work>/strace.txt`. The second
//! judges a trace made earlier with the same layout. Exit 0: inputs complete;
//! 1: something read is not declared; 2: usage or setup error.
use aien_test::audit::{self, Places};
use aien_test::manifest;
use std::path::PathBuf;
use std::process::{exit, Command};

fn main() {
    let mut root = PathBuf::from(".");
    let (mut target, mut work, mut trace, mut gate): (
        Option<PathBuf>,
        Option<PathBuf>,
        Option<PathBuf>,
        Option<PathBuf>,
    ) = (None, None, None, None);
    let mut siblings: Vec<PathBuf> = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .map(PathBuf::from)
                .unwrap_or_else(|| usage(&format!("{a} needs a value")))
        };
        match a.as_str() {
            "--root" => root = val(),
            "--target-dir" => target = Some(val()),
            "--work" => work = Some(val()),
            "--trace" => trace = Some(val()),
            "--sibling" => siblings.push(audit::canon(&val())),
            s if s.starts_with('-') => usage(&format!("unknown flag {s}")),
            _ => gate = Some(PathBuf::from(&a)),
        }
    }
    let (Some(target), Some(work), Some(gate)) = (target, work, gate) else {
        usage("missing --target-dir, --work or GATE_FILE")
    };
    let root = audit::canon(&root);
    let gate_path = if gate.is_absolute() {
        gate.clone()
    } else {
        root.join(&gate)
    };
    let bytes = std::fs::read(&gate_path)
        .unwrap_or_else(|e| fail(&format!("cannot read {}: {e}", gate_path.display())));
    let m = manifest::parse(&bytes).unwrap_or_else(|e| fail(&format!("bad manifest: {e:?}")));
    let manifest_rel = audit::canon(&gate_path)
        .strip_prefix(&root)
        .map(|p| p.to_string_lossy().into_owned())
        .unwrap_or_else(|_| fail("the gate file is not inside --root"));
    // The tree at HEAD: what is not in it was made by the run. A dirty tree
    // is judged anyway (its own files count as made by the run), with a warning.
    let ls = Command::new("git")
        .args(["ls-files", "-z"])
        .current_dir(&root)
        .output()
        .unwrap_or_else(|e| fail(&format!("git ls-files: {e}")));
    let tracked: std::collections::BTreeSet<String> = String::from_utf8_lossy(&ls.stdout)
        .split('\0')
        .filter(|s| !s.is_empty())
        .map(str::to_string)
        .collect();
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .current_dir(&root)
        .output()
        .map(|o| !o.stdout.is_empty())
        .unwrap_or(true);
    if dirty {
        println!("WARNING the tree has uncommitted changes; audit a clean checkout for the record");
    }
    let tmpdir = work.join("tmp");
    std::fs::create_dir_all(&tmpdir)
        .unwrap_or_else(|e| fail(&format!("cannot make {}: {e}", tmpdir.display())));
    std::fs::create_dir_all(&target)
        .unwrap_or_else(|e| fail(&format!("cannot make {}: {e}", target.display())));
    let trace_path = match trace {
        Some(t) => t,
        None => {
            let t = work.join("strace.txt");
            let status = Command::new("strace")
                .args([
                    "-f",
                    "-qq",
                    "-Y",
                    "-y",
                    "-e",
                    "trace=open,openat,openat2,execve",
                    "-e",
                    "status=successful",
                    "-o",
                ])
                .arg(&t)
                .arg(root.join(&m.exec))
                .args(&m.args)
                .current_dir(&root)
                .env("TMPDIR", &tmpdir)
                .env("CARGO_TARGET_DIR", &target)
                .status()
                .unwrap_or_else(|e| fail(&format!("cannot start strace: {e}")));
            println!(
                "GATE_EXIT {}",
                status
                    .code()
                    .map(|c| c.to_string())
                    .unwrap_or_else(|| "signal".into())
            );
            t
        }
    };
    let text = std::fs::read_to_string(&trace_path)
        .unwrap_or_else(|e| fail(&format!("cannot read {}: {e}", trace_path.display())));
    let places = Places {
        root,
        manifest_rel,
        target_dir: audit::canon(&target),
        tmpdir: audit::canon(&tmpdir),
        home: std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default(),
        tracked: Some(tracked),
        siblings,
    };
    let report = audit::judge(&m, &places, &audit::parse_strace(&text));
    let out = audit::render(&m, &report);
    let _ = std::fs::write(work.join(format!("audit-{}.txt", m.gate)), &out);
    // Every repository file the run read, declared or not: the closure to
    // narrow `inputs` to.
    let reads: Vec<&str> = report
        .covered
        .iter()
        .map(String::as_str)
        .chain(report.undeclared.keys().map(String::as_str))
        .collect();
    let mut reads = reads;
    reads.sort();
    let _ = std::fs::write(
        work.join(format!("reads-{}.txt", m.gate)),
        reads.join("\n") + "\n",
    );
    print!("{out}");
    exit(if report.ok(&m) { 0 } else { 1 });
}

fn usage(msg: &str) -> ! {
    eprintln!("aien-audit-inputs: {msg}\nusage: aien-audit-inputs [--root DIR] --target-dir DIR --work DIR [--trace FILE] [--sibling DIR]... GATE_FILE");
    exit(2)
}

fn fail(msg: &str) -> ! {
    eprintln!("aien-audit-inputs: {msg}");
    exit(2)
}
