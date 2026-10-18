//! aien-hetjob: worker and coordinator front door (aien-sovereign-core#637).
//!
//!   aien-hetjob keygen --state DIR                 create the worker key (0600), print its public key
//!   aien-hetjob caps --state DIR                   claimed platform, key, program digest, local baseline
//!   aien-hetjob serve --state DIR [LIMITS]         serve PST1 on stdin/stdout until the peer closes
//!   aien-hetjob dispatch --roster FILE (--ssh HOST [--remote CMD] | --local-worker DIR) JOB [--deadline-s N] [--no-oracle] [--idem HEX --task HEX --cause HEX to retry]
//!   aien-hetjob local JOB [--threads N]            run the worker's algorithm and the oracle here, timed
//!   aien-hetjob put-blob --state DIR FILE          copy an input file into a worker's blob store, print its sha256
//!
//! JOB is `--matmul M K N SEED`, `--chain ROUNDS SEED_HEX64` or
//! `--tokenize TOKENIZER_JSON CORPUS [--bos]` (the worker must hold both files
//! in its blob store, see put-blob; the coordinator reads its own copies).
//! LIMITS: --max-memory-mib N, --max-work N, --max-wall-s N, --threads N.
//! Exit: 0 verified (or command done), 1 any other outcome, 2 usage error.

use aien_hetjob::coordinator::{self, Job, OracleCheck, Outcome, Policy};
use aien_hetjob::receipt::Receipt;
use aien_hetjob::store::{self, Store};
use aien_hetjob::task::{self, Control, Inputs, Limits, TaskSpec};
use aien_hetjob::worker::Worker;
use aien_hetjob::{host, wire};
use aien_test::bundle::{fingerprint, hex_decode, hex_encode, keygen, load_signing_key};
use ed25519_dalek::VerifyingKey;
use sha2::{Digest, Sha256};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode, Stdio};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

fn usage(msg: &str) -> ExitCode {
    eprintln!("aien-hetjob: {msg}\nsee the header of crates/aien-hetjob/src/main.rs for usage");
    ExitCode::from(2)
}

struct Args(Vec<String>);

impl Args {
    fn flag(&mut self, name: &str) -> bool {
        match self.0.iter().position(|a| a == name) {
            Some(i) => {
                self.0.remove(i);
                true
            }
            None => false,
        }
    }
    fn values(&mut self, name: &str, n: usize) -> Result<Option<Vec<String>>, String> {
        let Some(i) = self.0.iter().position(|a| a == name) else {
            return Ok(None);
        };
        if i + n >= self.0.len() {
            return Err(format!("{name} needs {n} value(s)"));
        }
        let v: Vec<String> = self.0.drain(i..i + 1 + n).skip(1).collect();
        Ok(Some(v))
    }
    fn one(&mut self, name: &str) -> Result<Option<String>, String> {
        Ok(self.values(name, 1)?.map(|mut v| v.remove(0)))
    }
    fn num<T: std::str::FromStr>(&mut self, name: &str) -> Result<Option<T>, String> {
        match self.one(name)? {
            Some(s) => s
                .parse()
                .map(Some)
                .map_err(|_| format!("{name}: not a number: {s}")),
            None => Ok(None),
        }
    }
    fn done(&self) -> Result<(), String> {
        match self.0.first() {
            Some(a) => Err(format!("unexpected argument {a}")),
            None => Ok(()),
        }
    }
}

fn job_spec(a: &mut Args) -> Result<(TaskSpec, Inputs), String> {
    if let Some(v) = a.values("--tokenize", 2)? {
        let bos = a.flag("--bos");
        let read = |p: &str| std::fs::read(p).map_err(|e| format!("--tokenize: {p}: {e}"));
        let inputs = Inputs {
            tokenizer: read(&v[0])?,
            corpus: read(&v[1])?,
        };
        let spec = TaskSpec::TokenizeBatch {
            tokenizer: Sha256::digest(&inputs.tokenizer).into(),
            corpus: Sha256::digest(&inputs.corpus).into(),
            bos,
        };
        return Ok((spec, inputs));
    }
    generated_spec(a).map(|s| (s, Inputs::default()))
}

fn generated_spec(a: &mut Args) -> Result<TaskSpec, String> {
    if let Some(v) = a.values("--matmul", 4)? {
        let p = |i: usize| {
            v[i].parse::<u64>()
                .map_err(|_| format!("--matmul: not a number: {}", v[i]))
        };
        let dim = |i: usize| {
            p(i).and_then(|x| u16::try_from(x).map_err(|_| format!("--matmul: {x} above 65535")))
        };
        return Ok(TaskSpec::MatmulU32 {
            m: dim(0)?,
            k: dim(1)?,
            n: dim(2)?,
            seed: p(3)?,
        });
    }
    if let Some(v) = a.values("--chain", 2)? {
        let rounds = v[0]
            .parse()
            .map_err(|_| format!("--chain: not a number: {}", v[0]))?;
        let seed = hex_decode(&v[1])
            .and_then(|b| <[u8; 32]>::try_from(b).ok())
            .ok_or("--chain: seed must be 64 hex digits")?;
        return Ok(TaskSpec::DigestChain { rounds, seed });
    }
    Err(
        "a job is required: --matmul M K N SEED, --chain ROUNDS SEED_HEX64 or --tokenize TOKENIZER CORPUS"
            .to_string(),
    )
}

fn limits(a: &mut Args) -> Result<Limits, String> {
    let mut l = Limits::default();
    if let Some(m) = a.num::<u64>("--max-memory-mib")? {
        l.max_memory_bytes = m << 20;
    }
    if let Some(w) = a.num("--max-work")? {
        l.max_work_units = w;
    }
    if let Some(s) = a.num("--max-wall-s")? {
        l.max_wall = Duration::from_secs(s);
    }
    if let Some(t) = a.num("--threads")? {
        l.max_threads = t;
    }
    Ok(l)
}

fn random<const N: usize>() -> std::io::Result<[u8; N]> {
    let mut b = [0u8; N];
    std::fs::File::open("/dev/urandom")?.read_exact(&mut b)?;
    Ok(b)
}

fn read_roster(path: &Path) -> Result<Vec<VerifyingKey>, String> {
    let txt = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read roster {}: {e}", path.display()))?;
    txt.lines()
        .map(|l| l.split('#').next().unwrap_or("").trim())
        .filter(|l| !l.is_empty())
        .map(|l| {
            hex_decode(l)
                .and_then(|b| <[u8; 32]>::try_from(b).ok())
                .and_then(|b| VerifyingKey::from_bytes(&b).ok())
                .ok_or_else(|| format!("roster line is not a 32-byte ed25519 public key: {l}"))
        })
        .collect()
}

fn ms(d: Duration) -> String {
    format!("{:.3}", d.as_secs_f64() * 1e3)
}

fn time_local(spec: &TaskSpec, inputs: &Inputs, threads: usize) -> ([u8; 32], Duration) {
    let cancel = AtomicBool::new(false);
    let t0 = Instant::now();
    let ctl = Control {
        cancel: &cancel,
        deadline: t0 + Duration::from_secs(3600),
        threads,
    };
    let d = task::run(spec, inputs, &ctl).expect("local run: not cancelled, inputs usable");
    (d, t0.elapsed())
}

fn main() -> ExitCode {
    let mut argv: Vec<String> = std::env::args().skip(1).collect();
    if argv.is_empty() {
        return usage("missing command");
    }
    let cmd = argv.remove(0);
    let mut a = Args(argv);
    match run(&cmd, &mut a) {
        Ok(code) => code,
        Err(e) if e.starts_with("usage:") => usage(&e[6..]),
        Err(e) => {
            eprintln!("aien-hetjob: {e}");
            ExitCode::from(1)
        }
    }
}

fn run(cmd: &str, a: &mut Args) -> Result<ExitCode, String> {
    let u = |e: String| format!("usage:{e}");
    match cmd {
        "keygen" => {
            let state = PathBuf::from(a.one("--state").map_err(u)?.ok_or("usage:--state DIR")?);
            a.done().map_err(u)?;
            std::fs::create_dir_all(&state).map_err(|e| e.to_string())?;
            let pk = keygen(&state.join(aien_hetjob::store::KEY_FILE))?;
            println!("{pk}");
            Ok(ExitCode::SUCCESS)
        }
        "caps" => {
            let state = PathBuf::from(a.one("--state").map_err(u)?.ok_or("usage:--state DIR")?);
            a.done().map_err(u)?;
            let key = load_signing_key(&state.join(aien_hetjob::store::KEY_FILE))?;
            let p = host::platform().map_err(|e| e.to_string())?;
            let vk = key.verifying_key();
            println!("pubkey={}", hex_encode(vk.as_bytes()));
            println!("key_fingerprint={}", fingerprint(&vk));
            println!(
                "claimed_os={}\nclaimed_arch={}\nclaimed_cpus={}",
                p.os, p.arch, p.cpus
            );
            println!(
                "claimed_memory_bytes={}\nclaimed_program_digest={}",
                p.memory_bytes,
                hex_encode(&p.program_digest)
            );
            println!(
                "claimed_version={}\nkinds=matmul_u32,digest_chain,tokenize_batch",
                p.version
            );
            let base = TaskSpec::MatmulU32 {
                m: 256,
                k: 256,
                n: 256,
                seed: 1,
            };
            for t in [1usize, Limits::default().max_threads] {
                let (_, d) = time_local(&base, &Inputs::default(), t);
                println!("baseline_matmul_256_threads_{t}_ms={}", ms(d));
            }
            Ok(ExitCode::SUCCESS)
        }
        "serve" => {
            let state = PathBuf::from(a.one("--state").map_err(u)?.ok_or("usage:--state DIR")?);
            let lim = limits(a).map_err(u)?;
            a.done().map_err(u)?;
            let store = Store::open(&state).map_err(|e| format!("REFUSED: {e}"))?;
            let key = load_signing_key(&store.key_path())?;
            // CPU seconds for one connection: every thread busy for the wall limit, plus slack.
            host::limit_cpu_seconds(lim.max_wall.as_secs() * lim.max_threads as u64 + 30)
                .map_err(|e| e.to_string())?;
            let platform = host::platform().map_err(|e| e.to_string())?;
            let w = Worker {
                key,
                limits: lim,
                store,
                platform,
            };
            let sum = w
                .serve(std::io::stdin(), std::io::stdout().lock())
                .map_err(|e| e.to_string())?;
            eprintln!("aien-hetjob serve: {sum:?}");
            Ok(ExitCode::SUCCESS)
        }
        "local" => {
            let (spec, inputs) = job_spec(a).map_err(u)?;
            let threads = a
                .num("--threads")
                .map_err(u)?
                .unwrap_or(Limits::default().max_threads);
            a.done().map_err(u)?;
            let sizes = [inputs.tokenizer.len() as u64, inputs.corpus.len() as u64];
            let (d, t) = time_local(&spec, &inputs, threads);
            let t0 = Instant::now();
            let o = task::oracle(&spec, &inputs)?;
            let to = t0.elapsed();
            println!(
                "kind={}\nwork_units={}\nadmission_memory_bytes={}",
                spec.kind_name(),
                spec.work_units(&sizes),
                spec.memory_bytes(&sizes)
            );
            if let TaskSpec::TokenizeBatch { .. } = spec {
                let s = task::tokenize_summary(&spec, &inputs)?;
                println!(
                    "lines={}\ntokens={}\nround_trips={}",
                    s.lines, s.tokens, s.round_trips
                );
            }
            println!(
                "worker_algorithm_digest={}\nworker_algorithm_ms={}\nthreads={threads}",
                hex_encode(&d),
                ms(t)
            );
            println!(
                "oracle_digest={}\noracle_ms={}\nagree={}",
                hex_encode(&o),
                ms(to),
                d == o
            );
            Ok(if d == o {
                ExitCode::SUCCESS
            } else {
                ExitCode::from(1)
            })
        }
        "put-blob" => {
            let state = PathBuf::from(a.one("--state").map_err(u)?.ok_or("usage:--state DIR")?);
            let file = a.0.pop().ok_or("usage:put-blob --state DIR FILE")?;
            a.done().map_err(u)?;
            let bytes = std::fs::read(&file).map_err(|e| format!("{file}: {e}"))?;
            let d = store::put_blob(&state, &bytes).map_err(|e| e.to_string())?;
            println!("{}  {} bytes", hex_encode(&d), bytes.len());
            Ok(ExitCode::SUCCESS)
        }
        "dispatch" => dispatch(a),
        other => Err(format!("usage:unknown command {other}")),
    }
}

fn dispatch(a: &mut Args) -> Result<ExitCode, String> {
    let u = |e: String| format!("usage:{e}");
    let roster = read_roster(Path::new(
        &a.one("--roster").map_err(u)?.ok_or("usage:--roster FILE")?,
    ))?;
    let ssh = a.one("--ssh").map_err(u)?;
    let remote = a
        .one("--remote")
        .map_err(u)?
        .unwrap_or_else(|| "aien-hetjob serve --state aien-hetjob/state".to_string());
    let local_worker = a.one("--local-worker").map_err(u)?;
    let deadline = Duration::from_secs(a.num("--deadline-s").map_err(u)?.unwrap_or(120));
    let no_oracle = a.flag("--no-oracle");
    let idem = match a.one("--idem").map_err(u)? {
        Some(h) => hex_decode(&h)
            .and_then(|b| <[u8; 16]>::try_from(b).ok())
            .ok_or("usage:--idem must be 32 hex digits")?,
        None => random().map_err(|e| e.to_string())?,
    };
    // A retry of the same request repeats both the task id and the
    // idempotence key (PST1 section 8); a new task id with an old key is a
    // different request and the worker refuses it as DUPLICATE_ID.
    let task_id = match a.one("--task").map_err(u)? {
        Some(h) => hex_decode(&h)
            .and_then(|b| <[u8; 16]>::try_from(b).ok())
            .ok_or("usage:--task must be 32 hex digits")?,
        None => random().map_err(|e| e.to_string())?,
    };
    let cause_id = match a.one("--cause").map_err(u)? {
        Some(h) => hex_decode(&h)
            .and_then(|b| <[u8; 32]>::try_from(b).ok())
            .ok_or("usage:--cause must be 64 hex digits")?,
        None => random().map_err(|e| e.to_string())?,
    };
    let (spec, inputs) = job_spec(a).map_err(u)?;
    a.done().map_err(u)?;

    let mut child = match (&ssh, &local_worker) {
        (Some(host), None) => Command::new("ssh")
            .args([
                "-T",
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=yes",
                host,
            ])
            .arg(&remote)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn(),
        (None, Some(dir)) => Command::new(std::env::current_exe().map_err(|e| e.to_string())?)
            .args(["serve", "--state", dir])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .spawn(),
        _ => return Err("usage:exactly one of --ssh HOST or --local-worker DIR".to_string()),
    }
    .map_err(|e| format!("cannot start the worker connection: {e}"))?;

    let objective = Sha256::digest(format!(
        "AIEN_E2E_OBJECTIVE_V1\naien-hetjob P1 {}",
        spec.kind_name()
    ))
    .into();
    let ids = wire::Ids {
        cause_id,
        origin_run_id: random().map_err(|e| e.to_string())?,
        objective_id: objective,
        subject_id: [0; 32],
        task_id,
        idempotence_key: idem,
    };
    let mut summary = None;
    let (oracle, oracle_time) = if no_oracle {
        (None, Duration::ZERO)
    } else {
        let t0 = Instant::now();
        let o = if let TaskSpec::TokenizeBatch { .. } = spec {
            let s = task::tokenize_summary(&spec, &inputs)?;
            summary = Some(s);
            s.digest
        } else {
            task::oracle(&spec, &inputs)?
        };
        (Some(o), t0.elapsed())
    };
    let policy = Policy {
        deadline,
        cancel_grace: Duration::from_secs(5),
        oracle,
    };
    let job = Job { spec, ids };
    let r = child.stdout.take().expect("piped stdout");
    let mut w = child.stdin.take().expect("piped stdin");
    let (outcome, stats) =
        coordinator::dispatch(r, &mut w, &job, &roster, &policy).map_err(|e| e.to_string())?;
    drop(w);
    if matches!(outcome, Outcome::Uncertain(_)) {
        let _ = child.kill();
    }
    let _ = child.wait();

    println!(
        "cause_id={}\ntask_id={}\nidempotence_key={}",
        hex_encode(&ids.cause_id),
        hex_encode(&ids.task_id),
        hex_encode(&ids.idempotence_key)
    );
    let sizes = [inputs.tokenizer.len() as u64, inputs.corpus.len() as u64];
    println!(
        "kind={}\nwork_units={}",
        spec.kind_name(),
        spec.work_units(&sizes)
    );
    if let Some(s) = summary {
        println!(
            "oracle_lines={}\noracle_tokens={}\noracle_round_trips={}",
            s.lines, s.tokens, s.round_trips
        );
    }
    println!(
        "round_trip_ms={}\nframes={}\nstale_ignored={}",
        ms(stats.round_trip),
        stats.frames,
        stats.stale_ignored
    );
    if oracle.is_some() {
        println!("oracle_ms={}", ms(oracle_time));
    }
    match outcome {
        Outcome::Verified(v) => {
            let r: &Receipt = &v.receipt;
            println!("OUTCOME VERIFIED");
            println!("output_digest={}", hex_encode(&v.output));
            println!(
                "oracle={}",
                match v.oracle {
                    OracleCheck::Matched => "matched",
                    OracleCheck::NotChecked => "not-checked",
                }
            );
            println!("worker_key_fingerprint={}", fingerprint(&v.worker_key));
            println!(
                "claimed_platform={} {} cpus={} memory_bytes={}",
                r.platform.os, r.platform.arch, r.platform.cpus, r.platform.memory_bytes
            );
            println!(
                "claimed_program_digest={}",
                hex_encode(&r.platform.program_digest)
            );
            println!(
                "claimed_remote_compute_ms={}",
                ms(Duration::from_nanos(r.compute_ns))
            );
            println!("claimed_threads={}", r.threads);
            Ok(ExitCode::SUCCESS)
        }
        Outcome::Failed(why) => {
            println!("OUTCOME FAILED {why}");
            Ok(ExitCode::from(1))
        }
        Outcome::Cancelled => {
            println!("OUTCOME CANCELLED");
            Ok(ExitCode::from(1))
        }
        Outcome::Uncertain(why) => {
            println!("OUTCOME UNCERTAIN {why}");
            Ok(ExitCode::from(1))
        }
        Outcome::Rejected(why) => {
            println!("OUTCOME REJECTED {why}");
            Ok(ExitCode::from(1))
        }
    }
}
