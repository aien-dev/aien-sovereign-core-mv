//! aien-test: run gates and report verdicts (ADR 0028).
//!
//! - `run GATE`: run one gate. Exit 0 PASS, 1 FAIL or BAD_MANIFEST, 2 NOT_RUN
//!   or usage, 3 BLOCKED_HARDWARE, 4 BLOCKED_OPERATOR, 5 MISSING_IMPLEMENTATION.
//! - `test ./...` and `test gate:NAME`: run many gates, dependencies first,
//!   with worker threads that honour the resource pools. The exit code is the
//!   campaign verdict: any FAIL gives 1, else the worst of 2 to 5, else 0.
//! - `why GATE`: explain a gate's state from the receipts already written.
//! - `list`: list the gates and their dependencies.
//! - `identity GATE`: print the check id (ADR 0033 Decision 3) and the object it
//!   hashed, for the gate and everything it depends on.
//! - `plan`, `test changed`, `test crate:NAME`: the planner (ADR 0033 Decision 12).
//! - `status`, `verify COMMIT`, `worker`: the local pipeline (Decisions 9, 10).

use aien_test::bundle;
use aien_test::cli::{parse_args, Cli, Command, Selection, USAGE};
use aien_test::evidence::{Index, Store};
use aien_test::graph::{self, GraphError};
use aien_test::identity;
use aien_test::plan::{self, Request};
use aien_test::runner::{self, CacheNote, Options, RunError};
use aien_test::sandbox::Sandbox;
use aien_test::status;
use aien_test::verdict::Verdict;
use aien_test::verify;
use aien_test::why;
use aien_test::worker::{self, StateDir};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Duration;

fn main() {
    std::process::exit(real_main());
}

fn real_main() -> i32 {
    let argv: Vec<String> = std::env::args().skip(1).collect();
    // VerificationBundleV1 commands (ADR 0033 Decision 6) have their own small parser.
    if argv.first().is_some_and(|c| bundle::is_bundle_command(c)) {
        return bundle::cli_main(&argv);
    }
    let cli = match parse_args(&argv) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("aien-test: {e}");
            eprintln!("{USAGE}");
            return 2;
        }
    };
    let cwd = match std::env::current_dir() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("aien-test: cannot read current directory: {e}");
            return 2;
        }
    };
    match &cli.command {
        Command::Run { gate } => cmd_run(&cli, gate, &cwd),
        Command::Test(selection) => cmd_test(&cli, selection, &cwd),
        Command::Why { gate } => cmd_why(&cli, gate, &cwd),
        Command::List => cmd_list(&cwd),
        Command::Identity { gate } => cmd_identity(&cli, gate, &cwd),
        Command::Plan => cmd_plan(&cli, &cwd),
        Command::Status => cmd_status(&cli),
        Command::Verify { commit } => cmd_verify(&cli, commit, &cwd),
        Command::Worker => cmd_worker(&cli),
    }
}

/// Options for the real machine, with the command line applied.
fn options(cli: &Cli, root: &Path) -> Result<Options, i32> {
    let evidence_dir = match &cli.evidence_dir {
        Some(d) => d.clone(),
        None => match runner::default_evidence_dir(root) {
            Some(d) => d,
            None => {
                eprintln!("aien-test: HOME is not set and no --evidence-dir given");
                return Err(2);
            }
        },
    };
    let mut opts = Options::new(evidence_dir);
    opts.allow_dirty = cli.allow_dirty;
    opts.no_cache = cli.no_cache;
    opts.args_override = cli.args.clone();
    opts.pool_sizes = opts.pool_sizes.with(cli.jobs);
    opts.gpu.wait = cli.wait_gpu;
    opts.sandbox = cli.sandbox;
    opts.release = cli.release;
    opts.cargo_target_dir = std::env::var_os("CARGO_TARGET_DIR").map(PathBuf::from);
    Ok(opts)
}

fn run_error(e: &RunError) -> i32 {
    match e {
        RunError::BadManifest(m) => println!("AIEN_TEST: BAD_MANIFEST {m}"),
        RunError::Fatal(m) => println!("AIEN_TEST: FAIL {m}"),
    }
    1
}

fn graph_error(e: &GraphError) -> i32 {
    match e {
        // The message already starts with BAD_MANIFEST.
        GraphError::BadManifest { .. } => println!("AIEN_TEST: {e}"),
        _ => println!("AIEN_TEST: FAIL {e}"),
    }
    1
}

fn repo_root_or_report(cwd: &Path) -> Result<PathBuf, i32> {
    runner::repo_root(cwd).map_err(|e| {
        println!("AIEN_TEST: FAIL {e}");
        1
    })
}

/// Say plainly what the cache did. A reused result is never presented as a
/// fresh run.
fn print_cache(note: &CacheNote) {
    match note {
        CacheNote::Reused { from } => {
            println!("REUSED {from}");
            println!(
                "CACHE reused: an earlier run of exactly this experiment already has this result, so it was not run again (use --no-cache to force a fresh run)"
            );
        }
        CacheNote::Miss(why) => println!("CACHE miss: {why}; the gate was run"),
        CacheNote::Skipped(why) => println!("CACHE not used: {why}"),
    }
}

fn cmd_run(cli: &Cli, gate: &str, cwd: &Path) -> i32 {
    let manifest_path = match runner::locate_gate(cwd, gate) {
        Ok(p) => p,
        Err(e) => {
            println!("AIEN_TEST: BAD_MANIFEST {e}");
            return 1;
        }
    };
    let parent = manifest_path.parent().unwrap_or(cwd).to_path_buf();
    let root = match repo_root_or_report(&parent) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let opts = match options(cli, &root) {
        Ok(o) => o,
        Err(code) => return code,
    };
    match runner::run_gate(&root, &manifest_path, &opts) {
        Ok(o) => {
            if o.verdict == Verdict::NotRun && !o.reason.is_empty() {
                println!("REFUSED: {}", o.reason);
            }
            if let Some(p) = &o.receipt_path {
                println!("RECEIPT {}", p.display());
            }
            print_cache(&o.cache);
            if o.reason.is_empty() {
                println!("AIEN_TEST: {}", o.verdict.as_str());
            } else {
                println!("AIEN_TEST: {} {}", o.verdict.as_str(), o.reason);
            }
            o.verdict.exit_code()
        }
        Err(e) => run_error(&e),
    }
}

fn cmd_test(cli: &Cli, selection: &Selection, cwd: &Path) -> i32 {
    let root = match repo_root_or_report(cwd) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let graph = match graph::load(&root) {
        Ok(g) => g,
        Err(e) => return graph_error(&e),
    };
    let chosen: BTreeSet<usize> = match selection {
        Selection::All => (0..graph.len()).collect(),
        Selection::Gate(name) => match graph.index_of(name) {
            Some(i) => graph.with_dependencies(&[i]),
            None => {
                println!("AIEN_TEST: BAD_MANIFEST no gate named {name}");
                return 1;
            }
        },
        Selection::Changed => {
            let p = match plan_for(cli, &root) {
                Ok(p) => p,
                Err(code) => return code,
            };
            for line in plan::render(&plan::repository_name(&root, cli.repo.as_deref()), &p) {
                println!("{line}");
            }
            let code = plan::exit_code(&p);
            if code != 0 {
                println!(
                    "AIEN_TEST: {}",
                    if code == 4 {
                        "BLOCKED_OPERATOR"
                    } else {
                        "NOT_RUN"
                    }
                );
                return code;
            }
            p.entries
                .iter()
                .filter(|e| e.section != plan::Section::Blocked)
                .filter_map(|e| graph.index_of(&e.gate))
                .collect()
        }
        Selection::Crate(name) => match plan::crate_dirs(&root, name) {
            Ok(dirs) => {
                println!("CRATE {name} dirs {}", dirs.join(" "));
                plan::select_for_dirs(&graph, &dirs)
            }
            Err(e) => {
                println!("AIEN_TEST: FAIL {e}");
                return 1;
            }
        },
    };
    if chosen.is_empty() {
        println!("AIEN_TEST: NOT_RUN no gates found under {}", root.display());
        return 2;
    }
    let opts = match options(cli, &root) {
        Ok(o) => o,
        Err(code) => return code,
    };
    match runner::run_graph(&root, &graph, &chosen, &opts) {
        Ok(campaign) => {
            for (id, o) in &campaign.results {
                if o.reason.is_empty() {
                    println!("GATE {id} {}", o.verdict.as_str());
                } else {
                    println!("GATE {id} {} {}", o.verdict.as_str(), o.reason);
                }
                if let Some(p) = &o.receipt_path {
                    println!("RECEIPT {}", p.display());
                }
                print_cache(&o.cache);
            }
            println!("AIEN_TEST: {}", campaign.verdict().as_str());
            campaign.exit_code()
        }
        Err(e) => run_error(&e),
    }
}

fn cmd_why(cli: &Cli, gate: &str, cwd: &Path) -> i32 {
    let root = match repo_root_or_report(cwd) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let graph = match graph::load(&root) {
        Ok(g) => g,
        Err(e) => return graph_error(&e),
    };
    let target = match why::resolve_target(&graph, cwd, gate) {
        Ok(i) => i,
        Err(e) => {
            println!("AIEN_TEST: BAD_MANIFEST {e}");
            return 1;
        }
    };
    let opts = match options(cli, &root) {
        Ok(o) => o,
        Err(code) => return code,
    };
    let commit = match runner::head(&root) {
        Ok(c) => c,
        Err(e) => {
            println!("AIEN_TEST: FAIL {e}");
            return 1;
        }
    };
    let index = Index::load(&Store::new(&opts.evidence_dir));
    let cache_facts =
        aien_test::cache::explain_chain(&root, &graph, &index, &opts, &commit, target);
    for line in why::explain(&graph, &index, &commit, target, &cache_facts) {
        println!("{line}");
    }
    0
}

fn cmd_identity(cli: &Cli, gate: &str, cwd: &Path) -> i32 {
    let root = match repo_root_or_report(cwd) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let graph = match graph::load(&root) {
        Ok(g) => g,
        Err(e) => return graph_error(&e),
    };
    let target = match why::resolve_target(&graph, cwd, gate) {
        Ok(i) => i,
        Err(e) => {
            println!("AIEN_TEST: BAD_MANIFEST {e}");
            return 1;
        }
    };
    let node = graph.node(target);
    let binding = match identity::local_binding(cli.hardware_identity.as_deref().unwrap_or("")) {
        Ok(b) => b,
        Err(e) => {
            println!("AIEN_TEST: FAIL {e}");
            return 1;
        }
    };
    let ids = match identity::graph_identities(
        &graph,
        target,
        &root,
        &cli.compiler_identity,
        &cli.flags,
        &binding,
    ) {
        Ok(i) => i,
        Err(e) => {
            println!("AIEN_TEST: FAIL {e}");
            return 1;
        }
    };
    for (g, (id, _)) in &ids {
        if *g != node.id {
            println!("DEPENDENCY {g} CHECK_ID {id}");
        }
    }
    if let Some((id, obj)) = ids.get(&node.id) {
        println!("CHECK_ID {id}");
        println!("OBJECT {}", obj);
    }
    0
}

fn cmd_list(cwd: &Path) -> i32 {
    let root = match repo_root_or_report(cwd) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let graph = match graph::load(&root) {
        Ok(g) => g,
        Err(e) => return graph_error(&e),
    };
    for &i in graph.order() {
        let node = graph.node(i);
        let file = node.path.strip_prefix(&root).unwrap_or(&node.path);
        let deps: Vec<&str> = node
            .deps
            .iter()
            .map(|&d| graph.node(d).id.as_str())
            .collect();
        println!(
            "{} pool={} depends_on={} file={}",
            node.id,
            node.manifest.pool(),
            if deps.is_empty() {
                "-".to_string()
            } else {
                deps.join(",")
            },
            file.display()
        );
    }
    println!("{} gates", graph.len());
    0
}

/// The planner request for this checkout: BASE from `--base` (default
/// `origin/main`, then `main`), HEAD from the checkout; `--allow-dirty` adds
/// the uncommitted changes.
fn plan_for(cli: &Cli, root: &Path) -> Result<plan::Plan, i32> {
    let head = runner::head(root).map_err(|e| {
        println!("AIEN_TEST: FAIL {e}");
        1
    })?;
    let base = match &cli.base {
        Some(b) => plan::resolve(root, b),
        None => plan::resolve_first(root, &["origin/main", "main"]),
    }
    .map_err(|e| {
        println!("AIEN_TEST: FAIL {e}");
        1
    })?;
    let mut req = Request::new(&base, &head);
    req.operator = cli.operator;
    req.compiler_identity = cli.compiler_identity.clone();
    req.flags = cli.flags.clone();
    req.hardware_identity = cli.hardware_identity.clone().unwrap_or_default();
    let dirty = plan::uncommitted(root).map_err(|e| {
        println!("AIEN_TEST: FAIL {e}");
        1
    })?;
    if !dirty.is_empty() {
        if !cli.allow_dirty {
            println!(
                "AIEN_TEST: NOT_RUN dirty_tree ({} uncommitted paths; use --allow-dirty to plan them)",
                dirty.len()
            );
            return Err(2);
        }
        req.dirty = dirty;
    }
    plan::compute(root, &req).map_err(|e| {
        println!("AIEN_TEST: FAIL {e}");
        1
    })
}

fn cmd_plan(cli: &Cli, cwd: &Path) -> i32 {
    let root = match repo_root_or_report(cwd) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let mut p = match plan_for(cli, &root) {
        Ok(p) => p,
        Err(code) => return code,
    };
    // Say which selected gates the evidence cache already settles.
    if let Ok(opts) = options(cli, &root) {
        let cached = plan::probe_cached(&root, &p.graph, &opts);
        if !cached.is_empty() {
            for e in p.entries.iter_mut() {
                if e.section == plan::Section::Required && cached.contains(&e.gate) {
                    e.section = plan::Section::Cached;
                }
            }
        }
    }
    for line in plan::render(&plan::repository_name(&root, cli.repo.as_deref()), &p) {
        println!("{line}");
    }
    plan::exit_code(&p)
}

fn state_dir(cli: &Cli) -> Result<StateDir, i32> {
    match &cli.state_dir {
        Some(d) => Ok(StateDir::new(d)),
        None => StateDir::default_dir().ok_or_else(|| {
            eprintln!("aien-test: HOME is not set and no --state-dir given");
            2
        }),
    }
}

fn cmd_status(cli: &Cli) -> i32 {
    let state = match state_dir(cli) {
        Ok(s) => s,
        Err(code) => return code,
    };
    for line in status::lines(&state) {
        println!("{line}");
    }
    0
}

fn cmd_verify(cli: &Cli, commit: &str, cwd: &Path) -> i32 {
    let root = match repo_root_or_report(cwd) {
        Ok(r) => r,
        Err(code) => return code,
    };
    let state = match state_dir(cli) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let ask = verify::Ask {
        state,
        commit: commit.to_string(),
        // verify applies the worker's rule: a commit on this line is
        // checked against its first parent.
        base: cli
            .base
            .clone()
            .unwrap_or_else(|| "origin/main".to_string()),
        repository: cli.repo.clone(),
        release: cli.release,
        operator: cli.operator,
        hardware_identity: cli.hardware_identity.clone().unwrap_or_default(),
        crumb: Some("crumb".to_string()),
        keys_override: None,
    };
    let report = verify::run(&root, &ask);
    for line in &report.lines {
        println!("{line}");
    }
    if report.ready {
        0
    } else {
        1
    }
}

fn cmd_worker(cli: &Cli) -> i32 {
    let state = match state_dir(cli) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let home = std::env::var_os("HOME").map(PathBuf::from);
    let key = match (&cli.key, &home) {
        (Some(k), _) => k.clone(),
        (None, Some(h)) => h.join(".aien/verify.key"),
        (None, None) => {
            eprintln!("aien-test: HOME is not set and no --key given");
            return 2;
        }
    };
    let clones = match (&cli.clones, &home) {
        (Some(c), _) => c.clone(),
        (None, Some(h)) => h.join("workspace"),
        (None, None) => {
            eprintln!("aien-test: HOME is not set and no --clones given");
            return 2;
        }
    };
    let wt_root = cli
        .wt_root
        .clone()
        .unwrap_or_else(|| PathBuf::from("/var/lib/aien-run/wt"));
    let mut cfg = worker::Config::new(state, &key, &clones, &wt_root);
    // INFRA's worktree tool; the default is the copy in the trusted clone of
    // this repository. A missing tool makes `--sandbox user` refuse to run.
    cfg.wt_tool = match &cli.wt_tool {
        Some(t) => Some(t.clone()),
        None => Some(clones.join("aien-sovereign-core/tools/aien-local-ci/wt.sh"))
            .filter(|p| p.is_file()),
    };
    // The operator's sibling map, from the same trusted clone. A missing default
    // file means no siblings (the worker journals it); an explicit file that
    // cannot be read makes every run refuse.
    cfg.siblings = match &cli.siblings {
        Some(f) => Some(f.clone()),
        None => Some(clones.join("aien-sovereign-core/tools/aien-local-ci/siblings"))
            .filter(|p| p.is_file()),
    };
    // The worker runs other people's code: the user sandbox is its default.
    // `--sandbox none` is a development mode only, needs
    // `--dev-unsandboxed`, and is never allowed with `--github`.
    cfg.sandbox = if cli.sandbox_given {
        cli.sandbox
    } else {
        Sandbox::User
    };
    cfg.allow_unsandboxed = cli.dev_unsandboxed;
    if let Err(e) = cfg.check_sandbox_policy_for(cli.github) {
        eprintln!("aien-test: {e}");
        return 2;
    }
    if let Some(b) = &cli.base {
        cfg.base_ref = b.clone();
    }
    cfg.github = cli.github;
    cfg.poll = Duration::from_secs(cli.poll.unwrap_or(30));
    cfg.repos = cli.repos.clone();
    cfg.operator = cli.operator;
    cfg.jobs = cli.jobs;
    cfg.no_cache = cli.no_cache;
    cfg.poll_cap = Duration::from_secs(
        cli.poll_cap
            .unwrap_or(aien_test::idle::DEFAULT_POLL_CAP.as_secs()),
    );
    cfg.quiet_after = Duration::from_secs(
        cli.quiet_after
            .unwrap_or(aien_test::idle::DEFAULT_QUIET_AFTER.as_secs()),
    );
    cfg.cold = cli.cold;
    // The warm-standby prime map, beside the sibling map in the trusted clone.
    // A missing default file means no priming.
    cfg.prime = match &cli.prime {
        Some(f) => Some(f.clone()),
        None => Some(clones.join("aien-sovereign-core/tools/aien-local-ci/prime"))
            .filter(|p| p.is_file()),
    };
    cfg.hardware_identity = cli.hardware_identity.clone().unwrap_or_default();
    if let Some((repo, sha)) = &cli.once {
        if cli.prime_only {
            return match worker::prime_now(&cfg, repo, sha) {
                Ok(o) => {
                    println!("PRIME {repo} {sha} {o:?}");
                    i32::from(o != worker::PrimeOutcome::Warm)
                }
                Err(e) => {
                    println!("PRIME {repo} {sha} refused: {e}");
                    2
                }
            };
        }
        return match worker::run_once(&cfg, repo, sha) {
            Ok(s) => {
                for n in &s.notes {
                    println!("NOTE {n}");
                }
                for (g, st, v) in &s.gates {
                    println!("GATE {g} {st} {}", v.as_str());
                }
                println!("BUNDLE {} {}", s.bundle_path.display(), s.bundle_digest);
                println!("PUBLICATION {}", s.publication);
                if s.reason.is_empty() {
                    println!("AIEN_TEST: {}", s.verdict.as_str());
                } else {
                    println!("AIEN_TEST: {} {}", s.verdict.as_str(), s.reason);
                }
                s.verdict.exit_code()
            }
            Err(e) => {
                println!("AIEN_TEST: FAIL {e}");
                1
            }
        };
    }
    let iterations = match cli.iterations {
        Some(0) | None => None,
        Some(n) => Some(n),
    };
    let served = worker::run_loop(&cfg, iterations);
    println!("WORKER served {served}");
    0
}
