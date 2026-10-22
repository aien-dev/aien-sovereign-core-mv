//! The digest cache (ADR 0028 Decision 7, Slice C).
//!
//! A gate's result is reused, not rerun, only when an earlier fresh receipt
//! describes the very same experiment: the manifest bytes, the program, every
//! declared input, the machine and the receipts the gate depends on all have
//! identical digests (the "cache key"). Anything else runs the gate again.
//!
//! The cache is keyed on the receipt bytes and digests exactly as the runner
//! writes them today. It adds no receipt member and changes no receipt format
//! (ADR 0029 will own that). Where the ADR says "index" this slice reads the
//! receipts themselves: every receipt already carries `volatile.cache.key`,
//! so a separate `index.jsonl` would only be a second copy that could drift.
//!
//! A result is reused only for the same commit it was made at (sc#517). The
//! key covers a gate's declared inputs, but a gate may read files it did not
//! declare (SC-TEST reads README.md and COVENANT.md and searches the tree),
//! so an older commit's PASS could stand in for a commit that would fail.
//! Direct runs now follow the worker's rule: same commit, same pins, or the
//! gate runs again. Cross-commit reuse returns only once every gate's inputs
//! are proven complete (ADR 0028 cache key).
//!
//! Choices where the ADR is silent or contradicts itself (recorded in the PR):
//! - A reused receipt is the source receipt's `core` with its `commit` set to
//!   the commit being checked (the same commit, since reuse is bound to it).
//!   `why`, `run` and the dependency lookups all match receipts by commit.
//! - Dependencies enter the key as receipt digests, resolved to the original
//!   fresh receipt when the dependency itself was reused. Without that step a
//!   reused dependency would get a new digest and every dependent would miss.
//! - A source receipt must pass extra checks the key cannot express (see
//!   [`Unusable`]); failing any one of them is a miss, never a hit.
//! - `args` after `--` are not in the ADR key. A run with replacement args
//!   never consults the cache, and its key carries the args so it can never
//!   be mistaken for a run with the manifest's own args.

use crate::evidence::{canonical_digest, Index, Store, StoreError, Stored, SCHEMA};
use crate::runner;
use crate::verdict::Verdict;
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::path::Path;

/// `volatile.cache.mode` (ADR Decision 4).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// The cache may be consulted.
    Allow,
    /// The manifest says `cache: never`.
    Never,
    /// Lookups were switched off for this run (`--no-cache`, replacement
    /// args, or a tree with uncommitted changes).
    Disabled,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Allow => "allow",
            Mode::Never => "never",
            Mode::Disabled => "disabled",
        }
    }
}

/// The mode the manifest and the command line ask for. `cache: never` wins
/// over everything: it is the gate author's rule, not a flag.
pub fn policy(cache_allowed: bool, no_cache: bool) -> Mode {
    if !cache_allowed {
        Mode::Never
    } else if no_cache {
        Mode::Disabled
    } else {
        Mode::Allow
    }
}

/// The members the cache key is made of.
#[derive(Debug, Clone)]
pub struct Experiment {
    pub manifest_digest: String,
    pub binary_sha256: Option<String>,
    pub build_inputs_digest: String,
    pub machine_digest: String,
    /// `(gate, receipt digest)` as recorded in `core.dependencies`.
    pub dependencies: Vec<(String, String)>,
    pub pins: Value,
    /// Replacement arguments given after `--`, if any.
    pub args_override: Option<Vec<String>>,
}

/// A receipt digest, followed through `reused_from` to the fresh receipt it
/// stands for. A digest the store does not hold is returned as it is.
pub fn origin_digest(index: &Index, digest: &str) -> String {
    match index.get(digest) {
        Some(r) if r.value["volatile"]["cache"]["reused"] == true => r.value["volatile"]["cache"]
            ["reused_from"]
            .as_str()
            .unwrap_or(digest)
            .to_string(),
        _ => digest.to_string(),
    }
}

/// `cache.key`: sha256 over the canonical bytes of the key object.
pub fn key(e: &Experiment, index: &Index) -> Result<String, StoreError> {
    let deps: Vec<Value> = e
        .dependencies
        .iter()
        .map(|(gate, receipt)| json!({ "gate": gate, "receipt": origin_digest(index, receipt) }))
        .collect();
    let mut obj = json!({
        "manifest_digest": e.manifest_digest,
        "binary_sha256": e.binary_sha256,
        "build_inputs_digest": e.build_inputs_digest,
        "machine_digest": e.machine_digest,
        "dependencies": deps,
        "runner_schema": SCHEMA,
        "pins": e.pins,
    });
    if let Some(args) = &e.args_override {
        obj["args_override"] = json!(args);
    }
    canonical_digest(&obj)
}

/// The key members of a stored receipt, read from its `core`.
fn experiment_of(src: &Stored) -> Option<Experiment> {
    let core = &src.value["core"];
    let text = |k: &str| core[k].as_str().map(str::to_string);
    let deps = core["dependencies"]
        .as_array()?
        .iter()
        .map(|d| {
            Some((
                d["gate"].as_str()?.to_string(),
                d["receipt"].as_str()?.to_string(),
            ))
        })
        .collect::<Option<Vec<_>>>()?;
    Some(Experiment {
        manifest_digest: text("manifest_digest")?,
        binary_sha256: text("binary_sha256"),
        build_inputs_digest: text("build_inputs_digest")?,
        machine_digest: text("machine_digest")?,
        dependencies: deps,
        pins: core.get("pins")?.clone(),
        args_override: None,
    })
}

/// Why an earlier receipt that carries the right key still cannot be reused.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Unusable {
    /// It was itself a reuse; only fresh results are sources.
    AlreadyReused,
    /// Its verdict describes a situation, not an experiment (NOT_RUN,
    /// BLOCKED_*, MISSING_IMPLEMENTATION).
    NotAnExperiment,
    /// The tree was dirty before or after the run.
    UncleanTree,
    /// HEAD moved during the run.
    HeadMoved,
    /// It was made at another commit; a result stands only for its own
    /// commit (sc#517).
    OtherCommit,
    /// The run went past its time limit.
    TimedOut,
    /// The key written in the receipt is not what its own contents hash to.
    KeyMismatch,
    /// The machine descriptor does not hash to the machine digest.
    MachineMismatch,
    /// The captured environment differs from the current one.
    EnvDiffers,
    /// A stdout or stderr blob it names is missing from the store.
    BlobMissing,
}

impl Unusable {
    pub fn text(self) -> &'static str {
        match self {
            Unusable::AlreadyReused => "it was itself a reuse, not a fresh run",
            Unusable::NotAnExperiment => "its verdict is not PASS or FAIL",
            Unusable::UncleanTree => "the tree had uncommitted changes when it ran",
            Unusable::HeadMoved => "HEAD moved while it ran",
            Unusable::OtherCommit => {
                "it was made at another commit (a result stands only for its own commit)"
            }
            Unusable::TimedOut => "it ran past its time limit",
            Unusable::KeyMismatch => "its recorded key does not match its own contents",
            Unusable::MachineMismatch => "its machine descriptor does not match its machine digest",
            Unusable::EnvDiffers => "the captured environment differs from the current one",
            Unusable::BlobMissing => "a log it refers to is missing from the evidence store",
        }
    }
}

/// May `src` (which carries `key`) stand in for a run at `commit` now? Fail
/// closed: the first problem found makes it unusable.
pub fn usable(
    src: &Stored,
    key: &str,
    index: &Index,
    store: &Store,
    env: &Value,
    commit: &str,
) -> Result<(), Unusable> {
    let (core, vol) = (&src.value["core"], &src.value["volatile"]);
    if vol["cache"]["reused"] != false {
        return Err(Unusable::AlreadyReused);
    }
    if src.commit() != commit {
        return Err(Unusable::OtherCommit);
    }
    if !matches!(src.verdict(), Some(Verdict::Pass | Verdict::Fail)) {
        return Err(Unusable::NotAnExperiment);
    }
    if core["tree_clean_before"] != true || core["tree_clean_after"] != true {
        return Err(Unusable::UncleanTree);
    }
    if core["commit_unchanged_after"] != true {
        return Err(Unusable::HeadMoved);
    }
    if vol["timeout_exceeded"] != false {
        return Err(Unusable::TimedOut);
    }
    // Recompute the key from the receipt's own contents; never trust the
    // stored value alone.
    let recomputed = experiment_of(src).and_then(|e| self::key(&e, index).ok());
    if recomputed.as_deref() != Some(key) {
        return Err(Unusable::KeyMismatch);
    }
    let machine_digest = core.get("machine").and_then(|m| canonical_digest(m).ok());
    if machine_digest.as_deref() != core["machine_digest"].as_str() {
        return Err(Unusable::MachineMismatch);
    }
    if &core["env"] != env {
        return Err(Unusable::EnvDiffers);
    }
    for member in ["stdout_sha256", "stderr_sha256"] {
        match core[member].as_str() {
            Some(sha) if store.blob_exists(sha) => {}
            _ => return Err(Unusable::BlobMissing),
        }
    }
    Ok(())
}

/// Why there is no hit.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Miss {
    /// No receipt in the store carries this key.
    NoReceipt,
    /// The newest receipt with this key cannot be reused.
    Unusable { digest: String, why: Unusable },
}

impl Miss {
    pub fn text(&self) -> String {
        match self {
            Miss::NoReceipt => "no earlier receipt has this key".to_string(),
            Miss::Unusable { digest, why } => format!(
                "the newest receipt with this key ({}) cannot be reused: {}",
                digest.chars().take(12).collect::<String>(),
                why.text()
            ),
        }
    }
}

#[derive(Debug)]
pub enum Lookup<'a> {
    Hit(&'a Stored),
    Miss(Miss),
}

/// The newest usable fresh receipt whose `cache.key` is `key`, made at
/// `commit`.
pub fn lookup<'a>(
    index: &'a Index,
    store: &Store,
    key: &str,
    env: &Value,
    commit: &str,
) -> Lookup<'a> {
    let mut first_rejection: Option<Miss> = None;
    for r in index.receipts().iter().rev() {
        if r.value["volatile"]["cache"]["key"] != key {
            continue;
        }
        match usable(r, key, index, store, env, commit) {
            Ok(()) => return Lookup::Hit(r),
            Err(why) => {
                first_rejection.get_or_insert(Miss::Unusable {
                    digest: r.digest.clone(),
                    why,
                });
            }
        }
    }
    Lookup::Miss(first_rejection.unwrap_or(Miss::NoReceipt))
}

/// The `core` of a reuse receipt: the source's core, with `commit` set to the
/// commit being checked.
pub fn reused_core(src: &Stored, commit: &str) -> Value {
    let mut core = src.value["core"].clone();
    core["commit"] = json!(commit);
    core
}

/// The `volatile.cache` member.
pub fn cache_member(key: &str, reused_from: Option<&str>, mode: Mode) -> Value {
    json!({
        "key": key,
        "reused": reused_from.is_some(),
        "reused_from": reused_from,
        "mode": mode.as_str(),
    })
}

/// Which key members of `src` differ from `now`, in plain words.
pub fn differences(src: &Stored, now: &Experiment, index: &Index) -> Vec<&'static str> {
    let then = match experiment_of(src) {
        Some(t) => t,
        None => return vec!["the receipt itself (it is missing key members)"],
    };
    let resolve = |e: &Experiment| -> Vec<(String, String)> {
        e.dependencies
            .iter()
            .map(|(g, r)| (g.clone(), origin_digest(index, r)))
            .collect()
    };
    let mut out = Vec::new();
    if then.manifest_digest != now.manifest_digest {
        out.push("the gate manifest");
    }
    if then.binary_sha256 != now.binary_sha256 {
        out.push("the program (run.exec)");
    }
    if then.build_inputs_digest != now.build_inputs_digest {
        out.push("the declared input files");
    }
    if then.machine_digest != now.machine_digest {
        out.push("the machine");
    }
    if resolve(&then) != resolve(now) {
        out.push("a dependency's receipt");
    }
    if then.pins != now.pins {
        out.push("the pinned dependencies");
    }
    out
}

fn short(s: &str) -> String {
    s.chars().take(12).collect()
}

/// For `why`: what a run of `target` and of everything it depends on would do
/// about the cache right now, one list of plain lines per gate (node number).
/// Nothing is run and nothing is written.
pub fn explain_chain(
    root: &Path,
    graph: &crate::graph::Graph,
    index: &Index,
    opts: &runner::Options,
    commit: &str,
    target: usize,
) -> BTreeMap<usize, Vec<String>> {
    let store = Store::new(&opts.evidence_dir);
    let env = runner::captured_env();
    let dirty = runner::working_tree_dirty(root);
    let chain = graph.with_dependencies(&[target]);
    let mut out: BTreeMap<usize, Vec<String>> = BTreeMap::new();
    // The receipt a dependency would stand on after a run: Some only if that
    // dependency would itself be reused.
    let mut stands_on: BTreeMap<usize, Option<String>> = BTreeMap::new();
    for &i in graph.order() {
        if !chain.contains(&i) {
            continue;
        }
        let node = graph.node(i);
        let m = &node.manifest;
        let mut lines: Vec<String> = Vec::new();
        let mode = policy(m.cache_allowed, opts.no_cache);
        let mut reuse: Option<String> = None;
        match (mode, dirty.as_ref()) {
            (Mode::Never, _) => {
                lines.push("cache: never (the manifest says cache: never), so it always runs".to_string())
            }
            (Mode::Disabled, _) => lines.push("cache: switched off by --no-cache, so it would run".to_string()),
            (_, Err(e)) => lines.push(format!("cache: not used, the tree state is unknown ({e})")),
            (_, Ok(&true)) => lines.push(
                "cache: not used, the tree has uncommitted changes (a run is refused until it is clean)"
                    .to_string(),
            ),
            (Mode::Allow, Ok(&false)) => {
                let waiting: Vec<&str> = m
                    .depends_on
                    .iter()
                    .filter(|d| {
                        graph
                            .index_of(d)
                            .is_none_or(|di| stands_on.get(&di).is_none_or(|s| s.is_none()))
                    })
                    .map(String::as_str)
                    .collect();
                if let Some(first) = waiting.first() {
                    lines.push(format!(
                        "cache: miss, it depends on {first}, which would run again, so its receipt would change"
                    ));
                } else {
                    match runner::identity(root, &node.path, m, &opts.hardware, opts.sandbox) {
                        Err(e) => lines.push(format!("cache: miss, cannot digest the inputs ({e})")),
                        Ok(id) => {
                            let deps: Vec<(String, String)> = m
                                .depends_on
                                .iter()
                                .filter_map(|d| {
                                    let di = graph.index_of(d)?;
                                    Some((d.clone(), stands_on.get(&di)?.clone()?))
                                })
                                .collect();
                            let now = Experiment {
                                manifest_digest: m.digest.clone(),
                                binary_sha256: id.binary_sha.clone(),
                                build_inputs_digest: id.inputs_digest.clone(),
                                machine_digest: id.machine_digest.clone(),
                                dependencies: deps,
                                pins: opts.pins.clone(),
                                args_override: None,
                            };
                            match key(&now, index) {
                                Err(e) => lines.push(format!("cache: miss, cannot build the key ({e})")),
                                Ok(k) => match lookup(index, &store, &k, &env, commit) {
                                    Lookup::Hit(src) => {
                                        lines.push(format!(
                                            "cache: hit, a run would reuse receipt {} ({}, made at commit {}); key {}",
                                            short(&src.digest),
                                            src.verdict().map_or("UNKNOWN", |v| v.as_str()),
                                            short(src.commit()),
                                            short(&k)
                                        ));
                                        reuse = Some(src.digest.clone());
                                    }
                                    Lookup::Miss(miss) => {
                                        lines.push(format!("cache: miss, {} (key {})", miss.text(), short(&k)));
                                        if let Some(prev) = index.latest(&node.id) {
                                            let d = differences(prev, &now, index);
                                            if !d.is_empty() {
                                                lines.push(format!(
                                                    "cache: since its newest receipt {} these changed: {}",
                                                    short(&prev.digest),
                                                    d.join(", ")
                                                ));
                                            }
                                        }
                                    }
                                },
                            }
                        }
                    }
                }
            }
        }
        if let Some(r) = index.latest_at(&node.id, commit) {
            if r.value["volatile"]["cache"]["reused"] == true {
                lines.push(format!(
                    "the receipt for this commit ({}) was reused from {}, not run; it is not qualification evidence",
                    short(&r.digest),
                    short(r.value["volatile"]["cache"]["reused_from"].as_str().unwrap_or(""))
                ));
            }
        }
        stands_on.insert(i, reuse);
        out.insert(i, lines);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph;
    use crate::resources::{GpuConfig, Hardware};
    use crate::runner::{self, CacheNote, Options, Outcome};
    use crate::testutil::{git, init_repo, write_file, TempDir};
    use std::collections::BTreeSet;
    use std::fs;
    use std::path::PathBuf;
    use std::time::Duration;

    const SCRIPT: &str = "#!/bin/sh\necho x >> \"$1\"\necho \"AIEN-OBS n 3\"\necho PASS\n";
    const FAILING_SCRIPT: &str = "#!/bin/sh\necho x >> \"$1\"\necho \"AIEN-OBS n 3\"\nexit 1\n";

    fn gate_text(id: &str, deps: &[&str], counter: &Path, extra: &str) -> String {
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
            "gate: {id}\nmanifest_version: 1\nowner: t\nrequires:\n  - host\n{dep_block}inputs:\n  - data/\nrun:\n  exec: tools/run.sh\n  args: [\"{}\"]\nexpects:\n  exit: 0\n  observe: [\"n == 3\"]\n  verdict: PASS\ntimeout: 20s\n{extra}",
            counter.display()
        )
    }

    struct Fx {
        repo: TempDir,
        ev: TempDir,
        scratch: TempDir,
    }

    impl Fx {
        fn new(script: &str) -> Fx {
            let scratch = TempDir::new("c-scratch");
            let counter = scratch.path().join("a.count");
            let gate = gate_text("C-A", &[], &counter, "");
            let repo = init_repo(
                "c-repo",
                &[
                    ("tools/run.sh", script, true),
                    ("tests/a.gate", gate.as_str(), false),
                    ("data/in.txt", "one\n", false),
                    ("README", "hello\n", false),
                ],
            );
            Fx {
                repo,
                ev: TempDir::new("c-ev"),
                scratch,
            }
        }
        fn counter(&self, name: &str) -> PathBuf {
            self.scratch.path().join(format!("{name}.count"))
        }
        /// How many times the gate's program really ran.
        fn runs(&self, name: &str) -> usize {
            fs::read_to_string(self.counter(name)).map_or(0, |s| s.lines().count())
        }
        fn opts(&self) -> Options {
            let mut o = Options::new(self.ev.path().join("out"));
            o.kill_grace = Duration::from_millis(300);
            o.hardware = Hardware {
                gb10_present: false,
                qemu_available: false,
            };
            o.gpu = GpuConfig::rooted_at(self.scratch.path());
            o
        }
        fn store(&self) -> Store {
            Store::new(&self.ev.path().join("out"))
        }
        fn run(&self, opts: &Options) -> Outcome {
            runner::run_gate(
                self.repo.path(),
                &self.repo.path().join("tests/a.gate"),
                opts,
            )
            .unwrap()
        }
        fn commit(&self, rel: &str, content: &str) {
            write_file(self.repo.path(), rel, content, false);
            git(self.repo.path(), &["add", "-A"]);
            git(self.repo.path(), &["commit", "-q", "-m", rel]);
        }
        fn head(&self) -> String {
            runner::head(self.repo.path()).unwrap()
        }
        fn receipt(&self, o: &Outcome) -> Value {
            serde_json::from_slice(&fs::read(o.receipt_path.as_ref().unwrap()).unwrap()).unwrap()
        }
    }

    fn reused_from(o: &Outcome) -> Option<&str> {
        match &o.cache {
            CacheNote::Reused { from } => Some(from),
            _ => None,
        }
    }

    fn fresh_pass(fx: &Fx) -> Outcome {
        let o = fx.run(&fx.opts());
        assert_eq!(o.verdict, Verdict::Pass, "reason {}", o.reason);
        assert_eq!(reused_from(&o), None);
        o
    }

    // ---- hit -----------------------------------------------------------

    #[test]
    fn identical_experiment_is_reused_and_not_rerun() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        assert_eq!(fx.runs("a"), 1);
        let second = fx.run(&fx.opts());
        assert_eq!(second.verdict, Verdict::Pass);
        assert_eq!(
            reused_from(&second),
            first.receipt_digest.as_deref(),
            "must name the receipt it took the result from"
        );
        assert_eq!(fx.runs("a"), 1, "the program must not run again");
        let (r1, r2) = (fx.receipt(&first), fx.receipt(&second));
        assert_ne!(first.receipt_digest, second.receipt_digest);
        assert_eq!(r2["volatile"]["cache"]["reused"], true);
        assert_eq!(
            r2["volatile"]["cache"]["reused_from"],
            json!(first.receipt_digest)
        );
        assert_eq!(
            r2["volatile"]["cache"]["key"],
            r1["volatile"]["cache"]["key"]
        );
        assert_eq!(r2["volatile"]["cache"]["mode"], "allow");
        assert_eq!(r1["volatile"]["cache"]["reused"], false);
        assert_eq!(r1["volatile"]["cache"]["mode"], "allow");
        assert_eq!(r1["core"], r2["core"], "same commit: identical core");
    }

    #[test]
    fn a_result_is_bound_to_its_commit_and_never_reused_for_another() {
        // sc#517: the key covers only declared inputs, and a gate may read
        // files it did not declare, so an older commit's PASS must not stand
        // in for a new commit even when every declared input is unchanged.
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let old_head = fx.head();
        // A change to a file the gate does not declare.
        fx.commit("README", "changed\n");
        assert_ne!(fx.head(), old_head);
        let second = fx.run(&fx.opts());
        assert_eq!(reused_from(&second), None, "no reuse across commits");
        assert_eq!(second.verdict, Verdict::Pass);
        assert_eq!(fx.runs("a"), 2, "the gate ran again at the new commit");
        let (c1, c2) = (
            fx.receipt(&first)["core"].clone(),
            fx.receipt(&second)["core"].clone(),
        );
        assert_eq!(c1["commit"], old_head);
        assert_eq!(
            c2["commit"],
            fx.head(),
            "the fresh receipt is for this commit"
        );
        // The same key, so only the commit rule kept the second run fresh.
        let r1 = fx.receipt(&first);
        let r2 = fx.receipt(&second);
        assert_eq!(
            r1["volatile"]["cache"]["key"],
            r2["volatile"]["cache"]["key"]
        );
        // Back at the new commit a third run reuses the second (same commit).
        let third = fx.run(&fx.opts());
        assert_eq!(reused_from(&third), second.receipt_digest.as_deref());
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn a_failed_result_is_reused_too_and_still_fails() {
        let fx = Fx::new(FAILING_SCRIPT);
        let first = fx.run(&fx.opts());
        assert_eq!(first.verdict, Verdict::Fail);
        let second = fx.run(&fx.opts());
        assert_eq!(second.verdict, Verdict::Fail);
        assert_eq!(second.reason, first.reason);
        assert_eq!(reused_from(&second), first.receipt_digest.as_deref());
        assert_eq!(fx.runs("a"), 1);
    }

    // ---- misses --------------------------------------------------------

    #[test]
    fn miss_when_a_declared_input_changes() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        fx.commit("data/in.txt", "two\n");
        let second = fx.run(&fx.opts());
        assert_eq!(second.verdict, Verdict::Pass);
        assert!(
            matches!(second.cache, CacheNote::Miss(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2, "the program must run again");
        // And the new result is now the one that is reused.
        let third = fx.run(&fx.opts());
        assert_eq!(reused_from(&third), second.receipt_digest.as_deref());
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn miss_when_the_program_changes() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        fx.commit("tools/run.sh", &format!("{SCRIPT}# edited\n"));
        let second = fx.run(&fx.opts());
        assert!(
            matches!(second.cache, CacheNote::Miss(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn miss_when_the_manifest_changes_even_by_a_comment() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        let text = fs::read_to_string(fx.repo.path().join("tests/a.gate")).unwrap();
        fx.commit("tests/a.gate", &format!("# note\n{text}"));
        let second = fx.run(&fx.opts());
        assert!(
            matches!(second.cache, CacheNote::Miss(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn miss_when_the_machine_differs() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        let mut o = fx.opts();
        o.hardware.qemu_available = true;
        let second = fx.run(&o);
        assert!(
            matches!(second.cache, CacheNote::Miss(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn a_dirty_tree_is_refused_not_answered_from_the_cache() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        write_file(fx.repo.path(), "README", "dirty\n", false);
        let second = fx.run(&fx.opts());
        assert_eq!(second.verdict, Verdict::NotRun);
        assert_eq!(second.reason, "dirty_tree");
        assert_eq!(reused_from(&second), None);
        assert_eq!(fx.runs("a"), 1);
    }

    #[test]
    fn miss_on_a_dirty_tree_even_when_the_run_is_allowed() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        write_file(fx.repo.path(), "README", "dirty\n", false);
        let mut o = fx.opts();
        o.allow_dirty = true;
        let second = fx.run(&o);
        assert_eq!(second.verdict, Verdict::Pass);
        assert!(
            matches!(second.cache, CacheNote::Skipped(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2, "a dirty tree always runs");
        let r = fx.receipt(&second);
        assert_eq!(r["volatile"]["cache"]["mode"], "disabled");
        assert_eq!(r["volatile"]["cache"]["reused"], false);
    }

    #[test]
    fn a_receipt_from_a_dirty_tree_is_never_a_source() {
        let fx = Fx::new(SCRIPT);
        write_file(fx.repo.path(), "README", "dirty\n", false);
        let mut o = fx.opts();
        o.allow_dirty = true;
        let dirty_run = fx.run(&o);
        assert_eq!(dirty_run.verdict, Verdict::Pass);
        git(fx.repo.path(), &["checkout", "--", "README"]);
        let clean_run = fx.run(&fx.opts());
        assert!(
            matches!(clean_run.cache, CacheNote::Miss(_)),
            "{:?}",
            clean_run.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn miss_when_there_is_no_receipt_or_the_receipt_is_gone() {
        let fx = Fx::new(SCRIPT);
        // Nothing in the store at all.
        let first = fx.run(&fx.opts());
        assert!(
            matches!(first.cache, CacheNote::Miss(_)),
            "{:?}",
            first.cache
        );
        assert_eq!(fx.runs("a"), 1);
        // The only receipt is removed.
        fs::remove_file(first.receipt_path.as_ref().unwrap()).unwrap();
        let second = fx.run(&fx.opts());
        assert!(
            matches!(second.cache, CacheNote::Miss(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn miss_when_a_log_the_receipt_names_is_missing() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let sha = fx.receipt(&first)["core"]["stdout_sha256"]
            .as_str()
            .unwrap()
            .to_string();
        fs::remove_file(fx.ev.path().join("out/blobs").join(format!("{sha}.log"))).unwrap();
        let second = fx.run(&fx.opts());
        assert!(
            matches!(second.cache, CacheNote::Miss(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn a_receipt_that_is_itself_a_reuse_is_not_a_source() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let second = fx.run(&fx.opts());
        assert!(reused_from(&second).is_some());
        // Only the reuse receipt is left.
        fs::remove_file(first.receipt_path.as_ref().unwrap()).unwrap();
        let third = fx.run(&fx.opts());
        assert!(
            matches!(third.cache, CacheNote::Miss(_)),
            "{:?}",
            third.cache
        );
        assert_eq!(fx.runs("a"), 2);
    }

    // ---- forced rerun and never ---------------------------------------

    #[test]
    fn no_cache_forces_a_fresh_run_and_says_so_in_the_receipt() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let mut o = fx.opts();
        o.no_cache = true;
        let second = fx.run(&o);
        assert_eq!(second.verdict, Verdict::Pass);
        assert!(
            matches!(second.cache, CacheNote::Skipped(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
        let r = fx.receipt(&second);
        assert_eq!(r["volatile"]["cache"]["mode"], "disabled");
        assert_eq!(r["volatile"]["cache"]["reused"], false);
        assert_eq!(r["volatile"]["cache"]["reused_from"], Value::Null);
        // The fresh result is a valid source for a later ordinary run.
        let third = fx.run(&fx.opts());
        assert_eq!(reused_from(&third), second.receipt_digest.as_deref());
        assert_ne!(reused_from(&third), first.receipt_digest.as_deref());
        assert_eq!(fx.runs("a"), 2);
    }

    #[test]
    fn a_gate_that_says_cache_never_always_runs() {
        let fx = Fx::new(SCRIPT);
        let counter = fx.counter("a");
        let text = gate_text("C-A", &[], &counter, "cache: never\n");
        fx.commit("tests/a.gate", &text);
        let first = fx.run(&fx.opts());
        assert_eq!(first.verdict, Verdict::Pass);
        let second = fx.run(&fx.opts());
        assert!(
            matches!(second.cache, CacheNote::Skipped(_)),
            "{:?}",
            second.cache
        );
        assert_eq!(fx.runs("a"), 2);
        assert_eq!(fx.receipt(&second)["volatile"]["cache"]["mode"], "never");
    }

    #[test]
    fn replacement_args_never_use_or_feed_the_cache() {
        let fx = Fx::new(SCRIPT);
        let other = fx.counter("other");
        let mut o = fx.opts();
        o.args_override = Some(vec![other.display().to_string()]);
        // 1. A run with replacement args does not reuse an ordinary run.
        fresh_pass(&fx);
        let with_args = fx.run(&o);
        assert!(
            matches!(with_args.cache, CacheNote::Skipped(_)),
            "{:?}",
            with_args.cache
        );
        assert_eq!(fx.runs("other"), 1);
        // 2. And an ordinary run does not reuse the replacement-args run:
        //    remove the ordinary receipt, the other one must not stand in.
        let fx2 = Fx::new(SCRIPT);
        let other2 = fx2.counter("other");
        let mut o2 = fx2.opts();
        o2.args_override = Some(vec![other2.display().to_string()]);
        let first = fx2.run(&o2);
        assert_eq!(first.verdict, Verdict::Pass);
        let plain = fx2.run(&fx2.opts());
        assert!(
            matches!(plain.cache, CacheNote::Miss(_)),
            "{:?}",
            plain.cache
        );
        assert_eq!(fx2.runs("a"), 1, "the ordinary run really ran");
    }

    // ---- dependencies ---------------------------------------------------

    fn two_gate_fixture() -> Fx {
        let fx = Fx::new(SCRIPT);
        let b = gate_text("C-B", &["C-A"], &fx.counter("b"), "");
        fx.commit("tests/b.gate", &b);
        fx
    }

    fn run_all(fx: &Fx, opts: &Options) -> BTreeMap<String, Outcome> {
        let g = graph::load(fx.repo.path()).unwrap();
        let all: BTreeSet<usize> = (0..g.len()).collect();
        runner::run_graph(fx.repo.path(), &g, &all, opts)
            .unwrap()
            .results
            .into_iter()
            .collect()
    }

    #[test]
    fn a_whole_graph_is_reused_including_gates_that_rest_on_a_reused_gate() {
        let fx = two_gate_fixture();
        let first = run_all(&fx, &fx.opts());
        assert!(first.values().all(|o| o.verdict == Verdict::Pass));
        assert_eq!((fx.runs("a"), fx.runs("b")), (1, 1));
        let second = run_all(&fx, &fx.opts());
        assert_eq!(
            reused_from(&second["C-A"]),
            first["C-A"].receipt_digest.as_deref()
        );
        assert_eq!(
            reused_from(&second["C-B"]),
            first["C-B"].receipt_digest.as_deref(),
            "B rests on a reused A and must still be a hit"
        );
        assert_eq!((fx.runs("a"), fx.runs("b")), (1, 1));
    }

    #[test]
    fn a_new_commit_reruns_the_whole_chain_and_the_same_commit_reuses_it() {
        // sc#517: a result stands only for its own commit, so any new commit
        // reruns every gate, whatever its declared inputs did. Within one
        // commit the chain is reused as a whole.
        let fx = two_gate_fixture();
        run_all(&fx, &fx.opts());
        let b = gate_text("C-B", &["C-A"], &fx.counter("b"), "").replace("data/", "other/");
        fx.commit("other/x.txt", "x\n");
        fx.commit("tests/b.gate", &b);
        let after = run_all(&fx, &fx.opts());
        assert!(after.values().all(|o| reused_from(o).is_none()));
        assert_eq!(
            (fx.runs("a"), fx.runs("b")),
            (2, 2),
            "a new commit: A reruns although its declared inputs are unchanged, and B with it"
        );
        let again = run_all(&fx, &fx.opts());
        assert!(again.values().all(|o| reused_from(o).is_some()));
        assert_eq!((fx.runs("a"), fx.runs("b")), (2, 2), "same commit: reused");
        fx.commit("data/in.txt", "two\n");
        let after = run_all(&fx, &fx.opts());
        assert!(after.values().all(|o| reused_from(o).is_none()));
        assert_eq!((fx.runs("a"), fx.runs("b")), (3, 3));
    }

    // ---- the key itself -------------------------------------------------

    fn base() -> Experiment {
        Experiment {
            manifest_digest: "m".repeat(64),
            binary_sha256: Some("b".repeat(64)),
            build_inputs_digest: "i".repeat(64),
            machine_digest: "c".repeat(64),
            dependencies: vec![("G".to_string(), "d".repeat(64))],
            pins: json!([]),
            args_override: None,
        }
    }

    #[test]
    fn the_key_is_deterministic() {
        let idx = Index::default();
        assert_eq!(key(&base(), &idx).unwrap(), key(&base(), &idx).unwrap());
        assert_eq!(key(&base(), &idx).unwrap().len(), 64);
    }

    #[test]
    fn every_key_member_changes_the_key() {
        let idx = Index::default();
        let k0 = key(&base(), &idx).unwrap();
        let mut variants: Vec<(&str, Experiment)> = Vec::new();
        let mut e = base();
        e.manifest_digest = "x".repeat(64);
        variants.push(("manifest", e));
        let mut e = base();
        e.binary_sha256 = Some("x".repeat(64));
        variants.push(("binary", e));
        let mut e = base();
        e.binary_sha256 = None;
        variants.push(("no binary", e));
        let mut e = base();
        e.build_inputs_digest = "x".repeat(64);
        variants.push(("inputs", e));
        let mut e = base();
        e.machine_digest = "x".repeat(64);
        variants.push(("machine", e));
        let mut e = base();
        e.dependencies = vec![("G".to_string(), "x".repeat(64))];
        variants.push(("dependency receipt", e));
        let mut e = base();
        e.dependencies = vec![];
        variants.push(("no dependency", e));
        let mut e = base();
        e.pins = json!([{ "name": "p", "commit": "c" }]);
        variants.push(("pins", e));
        let mut e = base();
        e.args_override = Some(vec!["x".to_string()]);
        variants.push(("replacement args", e));
        for (what, e) in variants {
            assert_ne!(key(&e, &idx).unwrap(), k0, "{what} must change the key");
        }
    }

    #[test]
    fn policy_cache_never_beats_everything() {
        assert_eq!(policy(true, false), Mode::Allow);
        assert_eq!(policy(true, true), Mode::Disabled);
        assert_eq!(policy(false, false), Mode::Never);
        assert_eq!(policy(false, true), Mode::Never);
        assert_eq!(Mode::Allow.as_str(), "allow");
        assert_eq!(Mode::Never.as_str(), "never");
        assert_eq!(Mode::Disabled.as_str(), "disabled");
    }

    // ---- receipts that carry the right key but cannot be trusted ---------

    /// A store holding one forged copy of `first`'s receipt (and its logs), so
    /// the original cannot mask the result.
    fn forged(fx: &Fx, first: &Outcome, edit: impl FnOnce(&mut Value)) -> (TempDir, Value) {
        let original = fx.receipt(first);
        let dir = TempDir::new("c-forged");
        let blobs = dir.path().join("blobs");
        fs::create_dir_all(&blobs).unwrap();
        for e in fs::read_dir(fx.ev.path().join("out/blobs"))
            .unwrap()
            .flatten()
        {
            fs::copy(e.path(), blobs.join(e.file_name())).unwrap();
        }
        let mut v = original.clone();
        edit(&mut v);
        Store::new(dir.path()).put_receipt(&v).unwrap();
        (dir, original)
    }

    fn lookup_in(dir: &TempDir, original: &Value) -> Result<Unusable, ()> {
        let store = Store::new(dir.path());
        let index = Index::load(&store);
        let k = original["volatile"]["cache"]["key"].as_str().unwrap();
        let commit = original["core"]["commit"].as_str().unwrap();
        match lookup(&index, &store, k, &runner::captured_env(), commit) {
            Lookup::Hit(_) => Err(()),
            Lookup::Miss(Miss::Unusable { why, .. }) => Ok(why),
            Lookup::Miss(Miss::NoReceipt) => panic!("the forged receipt must be found by its key"),
        }
    }

    #[test]
    fn the_unedited_copy_is_a_hit_so_the_forgery_tests_mean_something() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let (dir, original) = forged(&fx, &first, |_| {});
        assert_eq!(lookup_in(&dir, &original), Err(()));
    }

    #[test]
    fn not_run_and_blocked_receipts_are_never_sources() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        for verdict in [
            "NOT_RUN",
            "BLOCKED_HARDWARE",
            "BLOCKED_OPERATOR",
            "MISSING_IMPLEMENTATION",
        ] {
            let (dir, o) = forged(&fx, &first, |v| {
                v["core"]["derived_verdict"] = json!(verdict)
            });
            assert_eq!(
                lookup_in(&dir, &o),
                Ok(Unusable::NotAnExperiment),
                "{verdict}"
            );
        }
    }

    #[test]
    fn unclean_moved_or_timed_out_receipts_are_never_sources() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["tree_clean_before"] = json!(false)
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::UncleanTree));
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["tree_clean_after"] = json!(false)
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::UncleanTree));
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["commit_unchanged_after"] = json!(false)
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::HeadMoved));
        let (dir, o) = forged(&fx, &first, |v| {
            v["volatile"]["timeout_exceeded"] = json!(true)
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::TimedOut));
    }

    #[test]
    fn a_stored_key_that_its_own_contents_do_not_hash_to_is_not_trusted() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        // The receipt claims the key but its inputs digest says otherwise.
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["build_inputs_digest"] = json!("0".repeat(64))
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::KeyMismatch));
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["manifest_digest"] = json!("0".repeat(64))
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::KeyMismatch));
    }

    #[test]
    fn a_machine_descriptor_that_does_not_match_its_digest_is_not_trusted() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["machine"]["cpu_count"] = json!(9999)
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::MachineMismatch));
    }

    #[test]
    fn a_different_captured_environment_is_not_the_same_experiment() {
        let fx = Fx::new(SCRIPT);
        let first = fresh_pass(&fx);
        let (dir, o) = forged(&fx, &first, |v| {
            v["core"]["env"]["RUSTFLAGS"] = json!("-C opt-level=3")
        });
        assert_eq!(lookup_in(&dir, &o), Ok(Unusable::EnvDiffers));
    }

    // ---- why --------------------------------------------------------------

    fn why_lines(fx: &Fx) -> String {
        let g = graph::load(fx.repo.path()).unwrap();
        let store = fx.store();
        let index = Index::load(&store);
        let target = g.index_of("C-A").unwrap();
        let facts = explain_chain(fx.repo.path(), &g, &index, &fx.opts(), &fx.head(), target);
        facts[&target].join("\n")
    }

    #[test]
    fn why_explains_a_hit_and_then_a_miss_and_what_changed() {
        let fx = Fx::new(SCRIPT);
        let before = why_lines(&fx);
        assert!(before.contains("cache: miss"), "{before}");
        assert!(
            before.contains("no earlier receipt has this key"),
            "{before}"
        );
        let first = fresh_pass(&fx);
        let hit = why_lines(&fx);
        assert!(hit.contains("cache: hit"), "{hit}");
        assert!(
            hit.contains(&first.receipt_digest.clone().unwrap()[..12]),
            "{hit}"
        );
        // sc#517: a new commit is a miss even when no declared input changed.
        fx.commit("README", "changed\n");
        let other = why_lines(&fx);
        assert!(other.contains("cache: miss"), "{other}");
        assert!(other.contains("made at another commit"), "{other}");
        assert!(!other.contains("these changed"), "{other}");
        fx.commit("data/in.txt", "two\n");
        let miss = why_lines(&fx);
        assert!(miss.contains("cache: miss"), "{miss}");
        assert!(
            miss.contains("these changed: the declared input files"),
            "{miss}"
        );
    }

    #[test]
    fn why_says_so_when_a_receipt_was_reused_and_when_the_tree_is_dirty() {
        let fx = Fx::new(SCRIPT);
        fresh_pass(&fx);
        let reused = fx.run(&fx.opts());
        assert!(reused_from(&reused).is_some());
        let text = why_lines(&fx);
        assert!(text.contains("was reused from"), "{text}");
        assert!(text.contains("not qualification evidence"), "{text}");
        write_file(fx.repo.path(), "README", "dirty\n", false);
        let dirty = why_lines(&fx);
        assert!(dirty.contains("uncommitted changes"), "{dirty}");
        assert!(!dirty.contains("cache: hit"), "{dirty}");
    }

    #[test]
    fn why_for_a_dependent_follows_a_reused_dependency() {
        let fx = two_gate_fixture();
        run_all(&fx, &fx.opts());
        // Same commit: B's dependency A is a hit, so B is one too.
        let g = graph::load(fx.repo.path()).unwrap();
        let index = Index::load(&fx.store());
        let b = g.index_of("C-B").unwrap();
        let facts = explain_chain(fx.repo.path(), &g, &index, &fx.opts(), &fx.head(), b);
        assert!(
            facts[&b].join("\n").contains("cache: hit"),
            "{:?}",
            facts[&b]
        );
        fx.commit("data/in.txt", "two\n");
        let index = Index::load(&fx.store());
        let facts = explain_chain(fx.repo.path(), &g, &index, &fx.opts(), &fx.head(), b);
        let text = facts[&b].join("\n");
        assert!(
            text.contains("depends on C-A, which would run again"),
            "{text}"
        );
    }
}
