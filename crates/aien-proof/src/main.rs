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

//! aien-proof CLI.
//!
//!   aien-proof run --job NAME [--input PATH]... [--gpu] [--resource NAME]... [--agent ID] -- CMD...
//!   aien-proof hold --resource NAME... [--job NAME] [--agent ID] -- CMD...
//!   aien-proof crates [--dir WORKSPACE] [--agent ID] [--gpu-crate NAME]... [--only NAME]... [-- CARGO_TEST_ARGS...]
//!   aien-proof verify
//!   aien-proof status
//!   aien-proof receipt show <id> [--store DIR]
//!   aien-proof receipt verify <path> [--store DIR]
//!   aien-proof receipt import [options] --assert FILE [--store DIR]
//!   aien-proof qualify [same options as receipt import]
//!   aien-proof verify-chain <path> [--store DIR]
//!   aien-proof gate status <GATE> [--store DIR] [--gate FILE]
//!   aien-proof gate explain <GATE> [--store DIR] [--gate FILE]
//!   aien-proof gate examples [--out DIR]
//!   aien-proof formal digest|bind ...   (formal oracle bindings, see formal.rs)

use aien_proof::board::{mem_available, Board, Job, Outcome};
use aien_proof::{chain, evidence, formal, gate, import, ledger, workspace};
use std::path::PathBuf;
use std::process::{exit, Command};
use std::time::Instant;

const USAGE: &str = "usage:
  aien-proof run --job NAME [--input PATH]... [--gpu] [--resource NAME]... [--agent ID] -- CMD...
  aien-proof hold --resource NAME... [--job NAME] [--agent ID] -- CMD...
  aien-proof crates [--dir WORKSPACE] [--agent ID] [--gpu-crate NAME]... [--only NAME]... [-- CARGO_TEST_ARGS...]
  aien-proof verify
  aien-proof status
  aien-proof receipt show <id> [--store DIR]
  aien-proof receipt verify <path> [--store DIR]
  aien-proof receipt import --repo URL --commit SHA --gate KIND --tier TIER --procedure PROC --machine MACH --assert FILE [--store DIR] [options]
  aien-proof qualify --repo URL --commit SHA --gate KIND --tier TIER --procedure PROC --machine MACH --assert FILE [--store DIR] [options]
  aien-proof verify-chain <path> [--store DIR]
  aien-proof gate status <GATE> [--store DIR] [--gate FILE]
  aien-proof gate explain <GATE> [--store DIR] [--gate FILE]
  aien-proof gate examples [--out DIR]
  aien-proof formal digest|bind ...   (see: aien-proof formal)";

struct Flags {
    values: Vec<(String, String)>,
    switches: Vec<String>,
    rest: Vec<String>,
    positionals: Vec<String>,
}

impl Flags {
    fn parse(args: &[String], switches: &[&str]) -> Self {
        let mut f = Flags {
            values: vec![],
            switches: vec![],
            rest: vec![],
            positionals: vec![],
        };
        let mut i = 0;
        while i < args.len() {
            let a = &args[i];
            if a == "--" {
                f.rest = args[i + 1..].to_vec();
                break;
            } else if switches.contains(&a.as_str()) {
                f.switches.push(a.clone());
            } else if a.starts_with("--") && i + 1 < args.len() {
                f.values.push((a.clone(), args[i + 1].clone()));
                i += 1;
            } else if a.starts_with("--") {
                eprintln!("option {a} needs a value\n{USAGE}");
                exit(2);
            } else {
                f.positionals.push(a.clone());
            }
            i += 1;
        }
        f
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.values
            .iter()
            .rev()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn all(&self, name: &str) -> Vec<String> {
        self.values
            .iter()
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.clone())
            .collect()
    }

    fn has(&self, name: &str) -> bool {
        self.switches.iter().any(|s| s == name)
    }
}

fn agent(flags: &Flags) -> String {
    flags
        .get("--agent")
        .map(str::to_string)
        .or_else(|| std::env::var("AIEN_AGENT_ID").ok())
        .or_else(|| std::env::var("USER").ok())
        .unwrap_or_else(|| "agent".into())
}

/// Compiler identity plus flags that change what gets built.
fn toolchain() -> String {
    let mut t = Command::new("rustc")
        .arg("-vV")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).into_owned())
        .unwrap_or_default();
    for var in ["RUSTFLAGS", "RUSTDOCFLAGS", "CARGO_BUILD_TARGET"] {
        t.push_str(&format!(
            "{var}={}\n",
            std::env::var(var).unwrap_or_default()
        ));
    }
    t
}

fn cmd_run(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &["--gpu"]);
    if flags.rest.is_empty() || !flags.positionals.is_empty() {
        eprintln!("{USAGE}");
        return 2;
    }
    let job = Job {
        name: flags
            .get("--job")
            .map(str::to_string)
            .unwrap_or_else(|| flags.rest.join(" ")),
        agent: agent(&flags),
        base: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        inputs: flags
            .all("--input")
            .into_iter()
            .map(PathBuf::from)
            .collect(),
        cmd: flags.rest.clone(),
        gpu: flags.switches.iter().any(|s| s == "--gpu"),
        resources: flags.all("--resource"),
        toolchain: toolchain(),
    };
    match Board::from_env().run(&job) {
        Ok(outcome) => outcome.record().exit_code,
        Err(e) => {
            eprintln!("[aien-proof] {}: {e}", job.name);
            1
        }
    }
}

fn cmd_crates(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &[]);
    if !flags.positionals.is_empty() {
        eprintln!("{USAGE}");
        return 2;
    }
    let here = flags
        .get("--dir")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")));
    let (root, members) = match workspace::members(&here) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[aien-proof] cargo metadata failed: {e}");
            return 1;
        }
    };
    let only = flags.all("--only");
    let gpu_crates = flags.all("--gpu-crate");
    let board = Board::from_env();
    let (agent, toolchain) = (agent(&flags), toolchain());
    let started = Instant::now();
    let (mut stamped, mut joined, mut ran, mut failed) = (0, 0, 0, Vec::new());
    let selected: Vec<_> = members
        .iter()
        .filter(|m| only.is_empty() || only.contains(&m.name))
        .collect();

    for m in &selected {
        let mut cmd = vec![
            "cargo".to_string(),
            "test".into(),
            "-p".into(),
            m.name.clone(),
        ];
        cmd.extend(flags.rest.iter().cloned());
        let job = Job {
            name: format!("cargo-test:{}", m.name),
            agent: agent.clone(),
            base: root.clone(),
            inputs: m.inputs.clone(),
            cmd,
            gpu: m.gpu || gpu_crates.contains(&m.name),
            resources: vec![],
            toolchain: toolchain.clone(),
        };
        match board.run(&job) {
            Ok(outcome) => {
                let rec = outcome.record();
                eprintln!(
                    "[aien-proof] {:<32} {:<8} {:<4} {:>8} ms",
                    m.name,
                    outcome.label(),
                    if outcome.passed() { "pass" } else { "FAIL" },
                    rec.duration_ms
                );
                match outcome {
                    Outcome::Stamped(_) => stamped += 1,
                    Outcome::Joined(_) => joined += 1,
                    Outcome::Ran(_) => ran += 1,
                }
                if !outcome.passed() {
                    failed.push(m.name.clone());
                }
            }
            Err(e) => {
                eprintln!("[aien-proof] {}: {e}", m.name);
                failed.push(m.name.clone());
            }
        }
    }

    eprintln!(
        "[aien-proof] {} crates in {:.1}s: {ran} ran, {stamped} stamped, {joined} joined, {} failed",
        selected.len(),
        started.elapsed().as_secs_f64(),
        failed.len()
    );
    if failed.is_empty() {
        0
    } else {
        eprintln!("[aien-proof] failed: {}", failed.join(", "));
        1
    }
}

fn cmd_verify() -> i32 {
    let board = Board::from_env();
    match ledger::verify(&board.root.join(ledger::LEDGER_FILE)) {
        Ok((n, head)) => {
            println!("ledger OK: {n} events, head {}", ledger::hex(&head));
            0
        }
        Err(e) => {
            println!("ledger BROKEN: {e}");
            1
        }
    }
}

/// Run a command while holding exclusive keys (for example `machine-1`).
/// Always runs; recorded in the ledger as an `audit` event.
fn cmd_hold(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &[]);
    let resources = flags.all("--resource");
    if flags.rest.is_empty() || resources.is_empty() || !flags.positionals.is_empty() {
        eprintln!("{USAGE}");
        return 2;
    }
    let job = Job {
        name: flags
            .get("--job")
            .map(str::to_string)
            .unwrap_or_else(|| flags.rest.join(" ")),
        agent: agent(&flags),
        base: std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
        inputs: vec![],
        cmd: flags.rest.clone(),
        gpu: false,
        resources,
        toolchain: String::new(),
    };
    match Board::from_env().hold(&job) {
        Ok(rec) => {
            eprintln!(
                "[aien-proof] {}: held {} for {} ms, exit {}, ledger #{}",
                rec.job,
                job.resources.join(","),
                rec.duration_ms,
                rec.exit_code,
                rec.ledger_index
            );
            // The audit event hash on stdout is the hold ID that evidence
            // receipts bind with `--lease-hold`.
            println!("{}", rec.ledger_hash);
            rec.exit_code
        }
        Err(e) => {
            eprintln!("[aien-proof] {}: {e}", job.name);
            1
        }
    }
}

fn count(dir: PathBuf) -> usize {
    std::fs::read_dir(dir).map(|d| d.count()).unwrap_or(0)
}

fn cmd_status() -> i32 {
    let b = Board::from_env();
    println!("board:      {}", b.root.display());
    println!("stamps:     {}", count(b.root.join("stamps")));
    println!("failures:   {}", count(b.root.join("fails")));
    println!("receipts:   {}", count(b.root.join(evidence::RECEIPT_DIR)));
    println!("holds:      {}", count(b.root.join("holds")));
    println!("cpu slots:  {}", b.cpu_slots);
    println!("memory gate:{} GiB free required", b.min_free_bytes >> 30);
    if let Some(free) = mem_available() {
        println!("free now:   {} GiB", free >> 30);
    }
    for (name, who) in b.holders() {
        println!("key held:   {name} by {who}");
    }
    cmd_verify()
}

fn store_dir(flags: &Flags) -> PathBuf {
    flags
        .get("--store")
        .map(PathBuf::from)
        .unwrap_or_else(|| Board::from_env().root)
}

fn verdict_code(v: evidence::Verdict) -> i32 {
    match v {
        evidence::Verdict::Pass => 0,
        evidence::Verdict::Fail => 1,
        evidence::Verdict::Blocked => 2,
        evidence::Verdict::Incomplete => 3,
        evidence::Verdict::Skipped => 4,
    }
}

const IMPORT_USAGE: &str = "usage:
  aien-proof receipt import --repo URL --commit SHA --gate KIND --tier TIER --procedure PROC --machine MACH --assert FILE [options]
options:
  --toolchain TEXT            build/toolchain identity (default: rustc -vV)
  --artifact DIGEST           input artifact digest, repeatable
  --output-artifact DIGEST    output artifact digest, repeatable
  --dep RECEIPT_ID            dependency receipt ID, repeatable
  --declared-mutation CLASS   required for hardware tiers
  --observed-mutation CLASS   defaults to declared
  --authority REF             approval reference; required for PRODUCTION
  --output-digest HEX         BLAKE3 of complete captured output
  --output-file PATH          hashed for the output digest
  --external-ref REF          external evidence reference, repeatable
  --ledger INDEX:HASH         Crumb ledger event reference
  --lease-hold HOLD_ID        audit event hash from `aien-proof hold`
  --lease-resource NAME       resource the hold covered
  --dirty                     mark the source tree dirty
  --out PATH                  also write the receipt file here
  --store DIR                 receipt store (default: proof board dir)";

fn cmd_receipt(args: &[String]) -> i32 {
    let (sub, rest) = match args.split_first() {
        Some((s, r)) => (s.as_str(), r),
        None => {
            eprintln!("{USAGE}");
            return 2;
        }
    };
    match sub {
        "show" => cmd_receipt_show(rest),
        "verify" => cmd_receipt_verify(rest),
        "import" => cmd_receipt_import(rest),
        _ => {
            eprintln!("unknown receipt subcommand {sub}\n{USAGE}");
            2
        }
    }
}

fn cmd_receipt_show(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &[]);
    let store = store_dir(&flags);
    let id = match flags.positionals.first() {
        Some(id) => id.clone(),
        None => {
            eprintln!("usage: aien-proof receipt show <id> [--store DIR]");
            return 2;
        }
    };
    match evidence::load_receipt(&store, &id) {
        Ok(receipt) => {
            println!("{}", serde_json::to_string_pretty(&receipt).unwrap());
            0
        }
        Err(e) => {
            eprintln!("[aien-proof] {e}");
            1
        }
    }
}

fn cmd_receipt_verify(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &[]);
    let store = store_dir(&flags);
    let path = match flags.positionals.first() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: aien-proof receipt verify <path> [--store DIR]");
            return 2;
        }
    };
    let text = match std::fs::read(&path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[aien-proof] cannot read {}: {e}", path.display());
            return 2;
        }
    };
    match evidence::verify_with_store(&text, &store) {
        Ok((receipt, report)) => {
            println!("receipt {}: {}", receipt.id, report.status.as_str());
            for reason in &report.reasons {
                println!("  - {reason}");
            }
            verdict_code(report.status)
        }
        Err(e) => {
            println!("receipt BROKEN: {e}");
            1
        }
    }
}

fn cmd_receipt_import(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &["--dirty"]);
    let assert_path = match flags.get("--assert") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("{IMPORT_USAGE}");
            return 2;
        }
    };
    for required in [
        "--repo",
        "--commit",
        "--gate",
        "--tier",
        "--procedure",
        "--machine",
    ] {
        if flags.get(required).is_none() {
            eprintln!("missing {required}\n{IMPORT_USAGE}");
            return 2;
        }
    }
    let store = store_dir(&flags);
    let (assertions, result) = match import::read_assertion_file(&assert_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("[aien-proof] {e}");
            return 2;
        }
    };
    let (ledger_index, ledger_hash) = match flags.get("--ledger") {
        Some(pair) => match pair.split_once(':') {
            Some((index, hash)) => match index.parse::<u64>() {
                Ok(i) => (Some(i), Some(hash.to_string())),
                Err(_) => {
                    eprintln!("[aien-proof] bad --ledger value, want INDEX:HASH");
                    return 2;
                }
            },
            None => {
                eprintln!("[aien-proof] bad --ledger value, want INDEX:HASH");
                return 2;
            }
        },
        None => (None, None),
    };
    let req = import::ImportRequest {
        repo: flags.get("--repo").unwrap().to_string(),
        commit: flags.get("--commit").unwrap().to_string(),
        dirty: flags.has("--dirty"),
        kind: flags.get("--gate").unwrap().to_string(),
        tier: flags.get("--tier").unwrap().to_string(),
        toolchain: flags
            .get("--toolchain")
            .map(str::to_string)
            .unwrap_or_else(toolchain),
        procedure: flags.get("--procedure").unwrap().to_string(),
        machine: flags.get("--machine").unwrap().to_string(),
        input_artifacts: flags.all("--artifact"),
        output_artifacts: flags.all("--output-artifact"),
        dependencies: flags.all("--dep"),
        declared_mutation: flags.get("--declared-mutation").map(str::to_string),
        observed_mutation: flags.get("--observed-mutation").map(str::to_string),
        authority: flags
            .get("--authority")
            .map(str::to_string)
            .unwrap_or_default(),
        output_digest: flags.get("--output-digest").map(str::to_string),
        output_file: flags.get("--output-file").map(str::to_string),
        external_refs: flags.all("--external-ref"),
        ledger_index,
        ledger_hash,
        lease_hold: flags.get("--lease-hold").map(str::to_string),
        lease_resource: flags.get("--lease-resource").map(str::to_string),
        timestamp: None,
    };
    let receipt = match import::build_receipt(&req, assertions, result) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("[aien-proof] import refused: {e}");
            return 1;
        }
    };
    // Warn about dependencies that cannot be resolved yet. They stay
    // missing, and verification reports INCOMPLETE, never PASS.
    for dep in &receipt.dependencies {
        if evidence::load_receipt(&store, dep).is_err() {
            eprintln!("[aien-proof] warning: dependency {dep} not in store yet");
        }
    }
    match evidence::store_receipt(&store, &receipt) {
        Ok(id) => {
            println!("{id}");
            if let Some(out) = flags.get("--out") {
                let text = serde_json::to_string_pretty(&receipt).unwrap();
                if let Err(e) = std::fs::write(out, format!("{text}\n")) {
                    eprintln!("[aien-proof] stored as {id} but --out failed: {e}");
                    return 1;
                }
            }
            0
        }
        Err(e) => {
            eprintln!("[aien-proof] cannot store receipt: {e}");
            1
        }
    }
}

fn cmd_verify_chain(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &[]);
    let store = store_dir(&flags);
    let path = match flags.positionals.first() {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("usage: aien-proof verify-chain <path> [--store DIR]");
            return 2;
        }
    };
    match chain::verify_chain(&path, &store) {
        Ok(report) => {
            println!("chain {}: {}", report.root, report.status.as_str());
            for line in &report.lines {
                println!(
                    "  {} {:<24} {:<18} {} {}",
                    line.status.as_str(),
                    line.kind,
                    line.tier.as_str(),
                    &line.id[..16.min(line.id.len())],
                    line.note
                );
            }
            for reason in &report.reasons {
                println!("  - {reason}");
            }
            verdict_code(report.status)
        }
        Err(e) => {
            println!("chain BROKEN: {e}");
            1
        }
    }
}

fn cmd_gate(args: &[String]) -> i32 {
    let (sub, rest) = match args.split_first() {
        Some((s, r)) => (s.as_str(), r),
        None => {
            eprintln!("{USAGE}");
            return 2;
        }
    };
    match sub {
        "status" => cmd_gate_status(rest, false),
        "explain" => cmd_gate_status(rest, true),
        "examples" => cmd_gate_examples(rest),
        _ => {
            eprintln!("unknown gate subcommand {sub}\n{USAGE}");
            2
        }
    }
}

fn cmd_gate_status(args: &[String], verbose: bool) -> i32 {
    let flags = Flags::parse(args, &[]);
    let store = store_dir(&flags);
    let name = match flags.positionals.first() {
        Some(n) => n.clone(),
        None => {
            eprintln!("usage: aien-proof gate status <GATE> [--store DIR] [--gate FILE]");
            return 2;
        }
    };
    let explicit = flags.get("--gate").map(PathBuf::from);
    let manifest = match gate::load_manifest(&store, &name, explicit.as_deref()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("[aien-proof] {e}");
            return 2;
        }
    };
    let report = gate::evaluate(&manifest, &store);
    println!("{}: {}", report.gate, report.status.as_str());
    for line in &report.lines {
        let word = match line.status {
            evidence::Verdict::Pass => "PASS",
            evidence::Verdict::Fail => "FAIL",
            _ => line.status.as_str(),
        };
        println!("{word:<7} {} {}", line.name, line.detail);
        if verbose {
            explain_requirement(&store, &manifest, line, 1);
        }
    }
    verdict_code(report.status)
}

/// Verbose `explain` detail for one gate line: sub gate lines recurse with
/// indentation, receipt matches show their assertion table.
fn explain_requirement(
    store: &std::path::Path,
    manifest: &gate::Gate,
    line: &gate::GateLine,
    depth: usize,
) {
    if depth > 12 {
        println!(
            "{}gate explanation depth limit reached",
            "  ".repeat(depth * 2)
        );
        return;
    }
    let pad = "  ".repeat(depth * 2);
    let req = match manifest.requires.iter().find(|r| r.name == line.name) {
        Some(r) => r,
        None => return,
    };
    if let Some(child) = &req.gate_ref {
        match gate::load_manifest(store, child, None) {
            Ok(child_manifest) => {
                let report = gate::evaluate(&child_manifest, store);
                for subline in &report.lines {
                    println!(
                        "{pad}{} {} {}",
                        subline.status.as_str(),
                        subline.name,
                        subline.detail
                    );
                    explain_requirement(store, &child_manifest, subline, depth + 1);
                }
            }
            Err(e) => println!("{pad}BLOCKED {child}: {e}"),
        }
        return;
    }
    if let Some(receipt) = gate::best_candidate(store, &req.kind) {
        println!("{pad}id {} tier {}", receipt.id, receipt.tier.as_str());
        for a in &receipt.assertions {
            println!(
                "{pad}[{}] {} expected={:?} observed={:?} source={:?}",
                if a.pass { "x" } else { " " },
                a.id,
                a.expected,
                a.observed,
                a.source
            );
        }
    }
}

fn cmd_gate_examples(args: &[String]) -> i32 {
    let flags = Flags::parse(args, &[]);
    let default = Board::from_env().root.join(gate::GATE_DIR);
    let out = flags.get("--out").map(PathBuf::from).unwrap_or(default);
    match gate::write_examples(&out) {
        Ok(paths) => {
            for p in paths {
                println!("{}", p.display());
            }
            0
        }
        Err(e) => {
            eprintln!("[aien-proof] {e}");
            1
        }
    }
}

const FORMAL_USAGE: &str = "usage:
  aien-proof formal digest [--base DIR] --invariant P --model P... --proof-source P... --covered P...
  aien-proof formal bind [--base DIR] --invariant P --model P... --proof-source P... --covered P...
        --oracle-toolchain TEXT --theorem NAME... --result-prefix PREFIX --output-file FILE
        --repo URL --commit SHA --machine M --procedure PROC [--dirty] [--dep ID]... [--external-ref R]... [--store DIR]
  paths are relative to --base (default: current directory); bind refuses output that
  lacks the `inputs_digest:` line printed by `formal digest`, or whose last line is not
  `PREFIX: PASS|FAIL|NOT_RUN`";

fn formal_inputs(flags: &Flags) -> Option<formal::FormalInputs> {
    let one = flags.get("--invariant")?;
    Some(formal::FormalInputs {
        base: flags
            .get("--base")
            .map(PathBuf::from)
            .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."))),
        manifest: PathBuf::from(one),
        model: flags
            .all("--model")
            .into_iter()
            .map(PathBuf::from)
            .collect(),
        proof_source: flags
            .all("--proof-source")
            .into_iter()
            .map(PathBuf::from)
            .collect(),
        covered: flags
            .all("--covered")
            .into_iter()
            .map(PathBuf::from)
            .collect(),
    })
}

fn cmd_formal(args: &[String]) -> i32 {
    let Some(sub) = args.first() else {
        eprintln!("{FORMAL_USAGE}");
        return 2;
    };
    let flags = Flags::parse(&args[1..], &["--dirty"]);
    let Some(inputs) = formal_inputs(&flags) else {
        eprintln!("{FORMAL_USAGE}");
        return 2;
    };
    let run_digest = match inputs.run_digest() {
        Ok(d) => d,
        Err(e) => {
            eprintln!("[aien-proof] {e}");
            return 1;
        }
    };
    match sub.as_str() {
        "digest" => {
            println!("{run_digest}");
            0
        }
        "bind" => formal_bind(&flags, &inputs, &run_digest),
        _ => {
            eprintln!("{FORMAL_USAGE}");
            2
        }
    }
}

fn formal_bind(flags: &Flags, inputs: &formal::FormalInputs, run_digest: &str) -> i32 {
    for required in [
        "--oracle-toolchain",
        "--result-prefix",
        "--output-file",
        "--repo",
        "--commit",
        "--machine",
        "--procedure",
    ] {
        if flags.get(required).is_none() {
            eprintln!("missing {required}\n{FORMAL_USAGE}");
            return 2;
        }
    }
    let refuse = |e: String| {
        eprintln!("[aien-proof] formal bind refused: {e}");
        1
    };
    let bytes = match std::fs::read(flags.get("--output-file").unwrap()) {
        Ok(b) => b,
        Err(e) => return refuse(format!("cannot read output file: {e}")),
    };
    let text = String::from_utf8_lossy(&bytes);
    if let Err(e) = formal::check_output_binds_inputs(&text, run_digest) {
        return refuse(e);
    }
    let result = match formal::parse_result_line(&text, flags.get("--result-prefix").unwrap()) {
        Ok(v) => v,
        Err(e) => return refuse(e),
    };
    let binding = match formal::compute_binding(
        inputs,
        flags.get("--oracle-toolchain").unwrap(),
        &flags.all("--theorem"),
    ) {
        Ok(b) => b,
        Err(e) => return refuse(e),
    };
    let run = formal::FormalRun {
        repo: flags.get("--repo").unwrap().to_string(),
        commit: flags.get("--commit").unwrap().to_string(),
        dirty: flags.has("--dirty"),
        machine: flags.get("--machine").unwrap().to_string(),
        procedure: flags.get("--procedure").unwrap().to_string(),
        result,
        output_digest: blake3::hash(&bytes).to_hex().to_string(),
        dependencies: flags.all("--dep"),
        external_refs: flags.all("--external-ref"),
        timestamp: None,
    };
    let receipt = match formal::bind_receipt(&binding, &run) {
        Ok(r) => r,
        Err(e) => return refuse(e),
    };
    match evidence::store_receipt(&store_dir(flags), &receipt) {
        Ok(id) => {
            println!("{id}");
            println!("{}: {}", receipt.kind, receipt.result.as_str());
            verdict_code(receipt.result)
        }
        Err(e) => refuse(e),
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let code = match args.first().map(String::as_str) {
        Some("run") => cmd_run(&args[1..]),
        Some("crates") => cmd_crates(&args[1..]),
        Some("hold") => cmd_hold(&args[1..]),
        Some("verify") => cmd_verify(),
        Some("status") => cmd_status(),
        Some("receipt") => cmd_receipt(&args[1..]),
        Some("qualify") => cmd_receipt_import(&args[1..]),
        Some("verify-chain") => cmd_verify_chain(&args[1..]),
        Some("gate") => cmd_gate(&args[1..]),
        Some("formal") => cmd_formal(&args[1..]),
        _ => {
            eprintln!("{USAGE}");
            2
        }
    };
    exit(code);
}
