//! Judge once (Drake 2026-10-20): the cross-head store, its key, the rules
//! for what may leave a head, and the batch budget. Red/green: each reuse
//! test also shows the miss one byte away.

use crate::cache::{self, Local, Reach, Shared};
use crate::evidence::{Index, Store};
use crate::manifest;
use crate::resources::{Budget, GpuConfig, Hardware};
use crate::runner::{self, CacheNote, Options, Outcome};
use crate::testutil::{git, init_repo, write_file, TempDir};
use crate::verdict::Verdict;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Duration;

const PASS: &str = "#!/bin/sh\necho x >> \"$1\"\necho \"AIEN-OBS n 3\"\necho PASS\n";
const FAIL: &str = "#!/bin/sh\necho x >> \"$1\"\necho \"AIEN-OBS n 3\"\nexit 1\n";

fn gate(counter: &Path, version: u8, extra: &str) -> String {
    format!(
        "gate: J-A\nmanifest_version: {version}\nowner: t\nrequires:\n  - host\ninputs:\n  - data/\nrun:\n  exec: tools/run.sh\n  args: [\"{}\"]\nexpects:\n  exit: 0\n  observe: [\"n == 3\"]\n  verdict: PASS\ntimeout: 20s\n{extra}",
        counter.display()
    )
}

/// One repository whose commits stand for different heads; each head gets
/// its own evidence store, as the worker gives it, and all share one
/// cross-head store.
struct Fx {
    repo: TempDir,
    ev: TempDir,
    scratch: TempDir,
}

impl Fx {
    fn new(script: &str, version: u8, extra: &str) -> Fx {
        let scratch = TempDir::new("jo-scratch");
        let text = gate(&scratch.path().join("a.count"), version, extra);
        let repo = init_repo(
            "jo-repo",
            &[
                ("tools/run.sh", script, true),
                ("tests/a.gate", text.as_str(), false),
                ("data/in.txt", "one\n", false),
                ("README", "hello\n", false),
            ],
        );
        Fx {
            repo,
            ev: TempDir::new("jo-ev"),
            scratch,
        }
    }
    fn runs(&self) -> usize {
        fs::read_to_string(self.scratch.path().join("a.count")).map_or(0, |s| s.lines().count())
    }
    fn head(&self) -> String {
        runner::head(self.repo.path()).unwrap()
    }
    fn shared_dir(&self) -> PathBuf {
        self.ev.path().join("gate-cache")
    }
    /// The options the worker would give the head now checked out.
    fn opts(&self) -> Options {
        let mut o = Options::new(self.ev.path().join("evidence").join(self.head()));
        o.kill_grace = Duration::from_millis(300);
        o.hardware = Hardware {
            gb10_present: false,
            qemu_available: false,
        };
        o.gpu = GpuConfig::rooted_at(self.scratch.path());
        o.shared_cache = Some(Shared::new(&self.shared_dir()));
        o
    }
    fn run(&self) -> Outcome {
        runner::run_gate(
            self.repo.path(),
            &self.repo.path().join("tests/a.gate"),
            &self.opts(),
        )
        .unwrap()
    }
    fn commit(&self, rel: &str, content: &str) -> String {
        write_file(self.repo.path(), rel, content, false);
        git(self.repo.path(), &["add", "-A"]);
        git(self.repo.path(), &["commit", "-q", "-m", rel]);
        self.head()
    }
    /// A new commit with the very same tree (a re-push, a squash merge of an
    /// up-to-date PR).
    fn same_tree_commit(&self) -> String {
        git(
            self.repo.path(),
            &["commit", "-q", "--allow-empty", "-m", "same tree"],
        );
        self.head()
    }
    fn receipt(o: &Outcome) -> Value {
        serde_json::from_slice(&fs::read(o.receipt_path.as_ref().unwrap()).unwrap()).unwrap()
    }
}

fn reused(o: &Outcome) -> Option<(&str, &str)> {
    match &o.cache {
        CacheNote::Reused { from, commit } => Some((from, commit)),
        _ => None,
    }
}

fn parsed(text: &str) -> manifest::Manifest {
    manifest::parse(text.as_bytes()).unwrap_or_else(|e| panic!("{e:?}\n{text}"))
}

// ---- what may leave a head ----------------------------------------------

#[test]
fn cross_head_rules_name_what_stays_in_its_head_and_why() {
    let c = Path::new("/tmp/c");
    let v2 = |extra: &str| parsed(&gate(c, 2, extra));
    assert_eq!(cache::cross_head(&v2("")), Ok(Reach::Tree));
    assert_eq!(
        cache::cross_head(&v2("cache: inputs_complete\n")),
        Ok(Reach::Inputs)
    );
    assert_eq!(
        cache::cross_head(&v2("cache: never\n")),
        Err(Local::CacheNever)
    );
    assert_eq!(
        cache::cross_head(&v2("cache: inputs_complete\ncache_scope: hardware\n")),
        Err(Local::HardwareScope)
    );
    assert_eq!(
        cache::cross_head(&v2("cache: inputs_complete\nexactness: APPROXIMATE\n")),
        Err(Local::NotExact)
    );
    let swap =
        |from: &str, to: &str| parsed(&gate(c, 2, "cache: inputs_complete\n").replace(from, to));
    assert_eq!(
        cache::cross_head(&swap("  - host\n", "  - gb10\n")),
        Err(Local::ChipState)
    );
    assert_eq!(
        cache::cross_head(&swap("  - host\n", "  - host\n  - operator\n")),
        Err(Local::Operator)
    );
    // A promise without real inputs is no promise: the tree decides.
    assert_eq!(
        cache::cross_head(&swap("inputs:\n  - data/\n", "")),
        Ok(Reach::Tree)
    );
    assert_eq!(cache::slot("k", Reach::Inputs, "t"), "k");
    assert_eq!(cache::slot("k", Reach::Tree, "t"), "k-t");
}

// ---- the key ---------------------------------------------------------------

#[test]
fn key_flips_on_any_declared_input_or_build_file_byte_and_not_on_others() {
    let fx = Fx::new(PASS, 1, "");
    let m = parsed(&fs::read_to_string(fx.repo.path().join("tests/a.gate")).unwrap());
    let digest = || runner::build_inputs_digest(fx.repo.path(), None, &m);
    let d0 = digest();
    fx.commit("README", "changed\n");
    assert_eq!(digest(), d0, "an undeclared file is not in the key");
    fx.commit("data/in.txt", "onf\n");
    let d1 = digest();
    assert_ne!(d1, d0, "one declared byte flips the key");
    fx.commit("Cargo.lock", "# lock\n");
    let d2 = digest();
    assert_ne!(d2, d1, "a repository build file joins every key");
    fx.commit("rust-toolchain.toml", "[toolchain]\n");
    assert_ne!(digest(), d2, "so does the toolchain pin");
}

#[test]
fn the_judge_binary_is_part_of_the_machine_and_so_of_the_key() {
    let fx = Fx::new(PASS, 1, "");
    let m = parsed(&fs::read_to_string(fx.repo.path().join("tests/a.gate")).unwrap());
    let hw = Hardware {
        gb10_present: false,
        qemu_available: false,
    };
    let path = fx.repo.path().join("tests/a.gate");
    let id = runner::identity(
        fx.repo.path(),
        &path,
        &m,
        &hw,
        crate::sandbox::Sandbox::None,
    )
    .unwrap();
    let sha = runner::judge_sha256().expect("the test binary can be read");
    assert_eq!(id.machine["judge_sha256"], sha.as_str());
    let o = fx.run();
    assert_eq!(
        Fx::receipt(&o)["core"]["machine"]["judge_sha256"],
        sha.as_str()
    );
}

// ---- reuse across heads, red/green -------------------------------------------

#[test]
fn a_complete_inputs_result_is_reused_by_another_head_and_not_one_byte_away() {
    let fx = Fx::new(PASS, 1, "cache: inputs_complete\n");
    let h1 = fx.head();
    let first = fx.run();
    assert_eq!((first.verdict, fx.runs()), (Verdict::Pass, 1));
    assert_eq!(reused(&first), None);

    // Another head: only an undeclared file differs. Reused, not run.
    let h2 = fx.commit("README", "other head\n");
    let second = fx.run();
    assert_eq!((second.verdict, fx.runs()), (Verdict::Pass, 1));
    let src = first.receipt_digest.clone().unwrap();
    assert_eq!(reused(&second), Some((src.as_str(), h1.as_str())));
    let r = Fx::receipt(&second);
    assert_eq!(r["core"]["commit"], h2.as_str());
    assert_eq!(r["volatile"]["cache"]["reused"], true);
    assert_eq!(r["volatile"]["cache"]["reused_from"], src.as_str());
    assert_eq!(r["volatile"]["cache"]["reused_from_commit"], h1.as_str());
    // The source receipt and its logs came along into this head's own store.
    let store = Store::new(&fx.opts().evidence_dir);
    let index = Index::load(&store);
    let copied = index.get(&src).expect("source receipt imported");
    assert!(store.blob_exists(copied.value["core"]["stdout_sha256"].as_str().unwrap()));

    // One declared byte different: the gate runs again.
    fx.commit("data/in.txt", "onf\n");
    let third = fx.run();
    assert_eq!((third.verdict, fx.runs()), (Verdict::Pass, 2));
    assert_eq!(reused(&third), None);
    match &third.cache {
        CacheNote::Miss(why) => assert!(why.contains("across heads"), "{why}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn without_the_promise_only_an_identical_tree_reuses() {
    // The gate reads nothing undeclared here, but its manifest does not say
    // so: an undeclared change (README) must run it again (SC-TEST reads the
    // root README.md), the same tree under a new commit reuses it.
    let fx = Fx::new(PASS, 1, "");
    fx.run();
    assert_eq!(fx.runs(), 1);
    fx.commit("README", "other head\n");
    let second = fx.run();
    assert_eq!((fx.runs(), reused(&second)), (2, None));
    let h2 = fx.head();
    fx.same_tree_commit();
    let third = fx.run();
    assert_eq!(fx.runs(), 2);
    assert_eq!(reused(&third).map(|r| r.1), Some(h2.as_str()));
}

#[test]
fn a_fail_crosses_heads_as_a_fail_never_as_a_pass() {
    let fx = Fx::new(FAIL, 1, "cache: inputs_complete\n");
    let first = fx.run();
    assert_eq!(first.verdict, Verdict::Fail);
    fx.commit("README", "other head\n");
    let second = fx.run();
    assert_eq!((second.verdict, fx.runs()), (Verdict::Fail, 1));
    assert!(reused(&second).is_some());
}

#[test]
fn a_gate_that_stays_in_its_head_is_never_shared_even_on_the_same_tree() {
    let fx = Fx::new(PASS, 2, "exactness: APPROXIMATE\n");
    fx.run();
    fx.same_tree_commit();
    let second = fx.run();
    assert_eq!((fx.runs(), reused(&second)), (2, None));
    assert!(!fx.shared_dir().join("keys").exists());
}

#[test]
fn a_damaged_shared_receipt_is_a_miss_never_a_hit() {
    let fx = Fx::new(PASS, 1, "cache: inputs_complete\n");
    let first = fx.run();
    let src = first.receipt_digest.unwrap();
    let path = fx.shared_dir().join(format!("{src}.json"));
    let mut perm = fs::metadata(&path).unwrap().permissions();
    #[allow(clippy::permissions_set_readonly_false)]
    perm.set_readonly(false);
    fs::set_permissions(&path, perm).unwrap();
    let text = fs::read_to_string(&path).unwrap().replace("PASS", "FAIL");
    fs::write(&path, text).unwrap();
    fx.commit("README", "other head\n");
    let second = fx.run();
    assert_eq!(
        (second.verdict, fx.runs(), reused(&second)),
        (Verdict::Pass, 2, None)
    );
}

#[test]
fn two_heads_racing_for_one_key_run_it_once() {
    // Two clones at different commits with the same declared inputs, judged
    // at the same moment: the key lock makes the second wait and reuse.
    let fx = Fx::new(
        "#!/bin/sh\necho x >> \"$1\"\nsleep 1\necho \"AIEN-OBS n 3\"\necho PASS\n",
        1,
        "cache: inputs_complete\n",
    );
    let other = TempDir::new("jo-clone");
    let dst = other.path().join("r");
    git(
        other.path(),
        &[
            "clone",
            "-q",
            fx.repo.path().to_str().unwrap(),
            dst.to_str().unwrap(),
        ],
    );
    write_file(&dst, "README", "other head\n", false);
    git(&dst, &["add", "-A"]);
    git(&dst, &["commit", "-q", "-m", "other"]);
    let opts_b = {
        let mut o = fx.opts();
        o.evidence_dir = fx.ev.path().join("evidence").join("b");
        o
    };
    let (a, b) = std::thread::scope(|s| {
        let a = s.spawn(|| fx.run());
        let b = s.spawn(|| runner::run_gate(&dst, &dst.join("tests/a.gate"), &opts_b).unwrap());
        (a.join().unwrap(), b.join().unwrap())
    });
    assert_eq!(fx.runs(), 1, "one run for both heads");
    assert_eq!(
        [reused(&a).is_some(), reused(&b).is_some()]
            .iter()
            .filter(|x| **x)
            .count(),
        1
    );
    assert_eq!((a.verdict, b.verdict), (Verdict::Pass, Verdict::Pass));
}

// ---- the batch budget ----------------------------------------------------------

#[test]
fn the_batch_budget_serves_shared_gates_before_the_others() {
    let budget = Arc::new(Budget::new(1));
    let held = budget.acquire(false);
    let order = Arc::new(std::sync::Mutex::new(Vec::new()));
    let started = Arc::new(AtomicUsize::new(0));
    std::thread::scope(|s| {
        let plain = {
            let (b, o, st) = (budget.clone(), order.clone(), started.clone());
            s.spawn(move || {
                st.fetch_add(1, Ordering::SeqCst);
                let _p = b.acquire(false);
                o.lock().unwrap().push("unique");
            })
        };
        while started.load(Ordering::SeqCst) < 1 {
            std::thread::yield_now();
        }
        std::thread::sleep(Duration::from_millis(50));
        let shared = {
            let (b, o) = (budget.clone(), order.clone());
            s.spawn(move || {
                let _p = b.acquire(true);
                o.lock().unwrap().push("shared");
            })
        };
        std::thread::sleep(Duration::from_millis(50));
        drop(held);
        plain.join().unwrap();
        shared.join().unwrap();
    });
    assert_eq!(*order.lock().unwrap(), vec!["shared", "unique"]);
}

// ---- declared inputs outside the tree and by pattern (L335) ----------------

/// A model checkpoint is not in the repository; `outside_inputs` puts its
/// bytes in the key. Unchanged checkpoint: another head reuses. One byte of
/// the checkpoint changed: the gate runs again.
#[test]
fn an_outside_input_is_in_the_key_and_one_changed_byte_runs_the_gate_again() {
    let ckpt = TempDir::new("jo-ckpt");
    fs::write(ckpt.path().join("model.safetensors"), "weights-1").unwrap();
    let extra = format!(
        "outside_inputs:\n  - {}/\ncache: inputs_complete\n",
        ckpt.path().display()
    );
    let fx = Fx::new(PASS, 1, &extra);
    let first = fx.run();
    assert_eq!((first.verdict, fx.runs()), (Verdict::Pass, 1));
    fx.commit("README", "other head\n");
    let second = fx.run();
    assert_eq!(
        fx.runs(),
        1,
        "same checkpoint, same declared inputs: reused"
    );
    assert!(reused(&second).is_some());
    // Same length, other bytes (the memo is keyed on size, mtime and inode,
    // so a rewrite is seen).
    std::thread::sleep(Duration::from_millis(20));
    fs::write(ckpt.path().join("model.safetensors"), "weights-2").unwrap();
    fx.commit("README", "third head\n");
    let third = fx.run();
    assert_eq!((third.verdict, fx.runs()), (Verdict::Pass, 2));
    assert_eq!(reused(&third), None);
}

#[test]
fn an_outside_input_that_is_missing_is_a_null_row_not_a_match() {
    let gone = TempDir::new("jo-gone");
    let text = |p: &str| {
        parsed(&gate(
            Path::new("/c"),
            1,
            &format!("outside_inputs:\n  - {p}/x\n"),
        ))
    };
    let fx = Fx::new(PASS, 1, "");
    let d_missing = runner::build_inputs_digest(
        fx.repo.path(),
        None,
        &text(&gone.path().display().to_string()),
    );
    fs::write(gone.path().join("x"), "").unwrap();
    let d_empty = runner::build_inputs_digest(
        fx.repo.path(),
        None,
        &text(&gone.path().display().to_string()),
    );
    assert_ne!(
        d_missing, d_empty,
        "a missing file and an empty file are different keys"
    );
    // A manifest without outside_inputs keeps the digest it always had.
    let plain = parsed(&gate(Path::new("/c"), 1, ""));
    let before = runner::build_inputs_digest(fx.repo.path(), None, &plain);
    assert_eq!(
        before,
        runner::build_inputs_digest(fx.repo.path(), None, &plain)
    );
}

#[test]
fn outside_inputs_must_be_home_or_absolute_paths_without_dotdot() {
    for bad in ["models/x", "~/../etc", "/a/../b", "~/", "/"] {
        let t = gate(Path::new("/c"), 1, &format!("outside_inputs:\n  - {bad}\n"));
        assert!(
            manifest::parse(t.as_bytes()).is_err(),
            "{bad} must be refused"
        );
    }
    let t = gate(Path::new("/c"), 1, "outside_inputs:\n  - ~/models/T/\n");
    assert_eq!(parsed(&t).outside_inputs, vec!["~/models/T/".to_string()]);
}

/// `crates/*/Cargo.toml`: every member manifest is in the key (cargo reads
/// them all to load the workspace), their sources are not.
#[test]
fn a_star_pattern_input_covers_each_match_and_nothing_else() {
    let fx = Fx::new(PASS, 1, "");
    fx.commit("crates/b/Cargo.toml", "[package]\n");
    fx.commit("crates/b/src/lib.rs", "x\n");
    let m =
        parsed(&gate(Path::new("/c"), 1, "").replace("  - data/\n", "  - crates/*/Cargo.toml\n"));
    let digest = || runner::build_inputs_digest(fx.repo.path(), None, &m);
    let d0 = digest();
    fx.commit("crates/b/src/lib.rs", "y\n");
    assert_eq!(digest(), d0, "a source file the pattern does not name");
    fx.commit("crates/b/Cargo.toml", "[package]\nname = \"b\"\n");
    let d1 = digest();
    assert_ne!(d1, d0, "a matched member manifest");
    fx.commit("crates/c/Cargo.toml", "[package]\n");
    assert_ne!(digest(), d1, "a new member manifest");
    // A pattern at the top is the whole tree for the cross-head rule.
    let top = parsed(
        &gate(Path::new("/c"), 1, "cache: inputs_complete\n")
            .replace("  - data/\n", "  - */Cargo.toml\n"),
    );
    assert_eq!(cache::cross_head(&top), Ok(Reach::Tree));
}

/// `listed`: a file added to a listed directory flips the key; a changed
/// byte inside it does not (its contents are not declared by the listing).
#[test]
fn a_listed_directory_puts_its_names_in_the_key_not_its_bytes() {
    let fx = Fx::new(PASS, 1, "");
    fx.commit("crates/b/tests/one.rs", "x\n");
    let m = parsed(&gate(Path::new("/c"), 1, "listed:\n  - crates/*/tests\n"));
    let digest = || runner::build_inputs_digest(fx.repo.path(), None, &m);
    let d0 = digest();
    fx.commit("crates/b/tests/one.rs", "y\n");
    assert_eq!(digest(), d0, "same names");
    fx.commit("crates/b/tests/two.rs", "z\n");
    assert_ne!(digest(), d0, "a new test target");
}
