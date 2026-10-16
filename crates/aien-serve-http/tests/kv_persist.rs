//! Persistent KV park and restore on the omega door (sc#505), host only.
//!
//! The engine is the real `OmegaEngine` over the tiny in-memory checkpoint with a host bf16
//! pool (`host_test_engine`), the store is the real sealed host store in a private temp dir,
//! and the key is generated per test from /dev/urandom. The GPU lock is a private file, never
//! the hive's `/tmp/aien-gb10.lock`.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aien_inference_abi::qwen3_moe_omega::{host_test_engine, OmegaEngine, WeightFormat};
use aien_serve_http::codec::ByteCodec;
use aien_serve_http::engine::{Engine, GenerateParams};
use aien_serve_http::kv_persist::{self, OmegaBuild, PersistState};
use aien_serve_http::omega_engine::{GpuLock, OmegaDoor};
use aien_serve_http::sampling::sample_next;
use aien_serve_http::server::{serve, serve_until, ServerConfig};

const CONTEXT: usize = 256;
const WAIT: Duration = Duration::from_secs(20);

static NEXT: AtomicUsize = AtomicUsize::new(0);

/// A private scratch dir (0700) under the system temp dir, removed on drop.
struct Scratch(PathBuf);

impl Scratch {
    fn new(tag: &str) -> Self {
        use std::os::unix::fs::PermissionsExt;
        let n = NEXT.fetch_add(1, Ordering::SeqCst);
        let p = std::env::temp_dir().join(format!("pkv505-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&p);
        std::fs::create_dir_all(&p).unwrap();
        std::fs::set_permissions(&p, std::fs::Permissions::from_mode(0o700)).unwrap();
        Self(p)
    }
    fn store(&self) -> PathBuf {
        self.0.join("store")
    }
    fn lock(&self) -> PathBuf {
        self.0.join("gpu.lock")
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// 64 hex digits from /dev/urandom (a fresh key per test, never printed).
fn gen_key() -> String {
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut b)
        .unwrap();
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn test_build() -> OmegaBuild {
    OmegaBuild {
        omega_commit: "d0889a944b02066f84fec71af7b9bbb92f5c986f".into(),
        gpu_link: "pinned-source".into(),
        gpu_archive_sha256: "11".repeat(32),
        gpu_digest: "unrecorded".into(),
        gpu_target: "aarch64-unknown-linux-gnu".into(),
        sc_commit: "22".repeat(20),
        sc_dirty: false,
    }
}

/// Everything one door start depends on; a test changes one field.
#[derive(Clone)]
struct Setup {
    weights: WeightFormat,
    build: OmegaBuild,
    env: Vec<(String, String)>,
    store_vars: Vec<(String, String)>,
}

impl Setup {
    fn new(store: &Path, key: &str) -> Self {
        Self {
            weights: WeightFormat::Fp8All,
            build: test_build(),
            env: Vec::new(),
            store_vars: vec![
                ("AIEN_KV_STORE_DIR".into(), store.display().to_string()),
                ("AIEN_KV_STORE_KEY".into(), key.into()),
                ("AIEN_KV_CHUNK_BLOCKS".into(), "1".into()),
            ],
        }
    }
    fn without(mut self, var: &str) -> Self {
        self.store_vars.retain(|(k, _)| k != var);
        self
    }
    fn with_var(mut self, var: &str, v: &str) -> Self {
        self.store_vars.retain(|(k, _)| k != var);
        self.store_vars.push((var.into(), v.into()));
        self
    }
}

fn engine(weights: WeightFormat) -> OmegaEngine {
    host_test_engine::engine(weights, CONTEXT).unwrap()
}

/// A door with persistence started from `s` (the startup line is returned).
fn door(scratch: &Scratch, s: &Setup) -> (OmegaDoor, String) {
    door_with_fault(scratch, s, None)
}

/// [`door`] with the store's test-only stall (`AIEN_KV_TEST_FAULT`, sc#507).
fn door_with_fault(scratch: &Scratch, s: &Setup, fault: Option<&str>) -> (OmegaDoor, String) {
    door_at(scratch, s, fault, Duration::from_secs(5))
}

/// [`door_with_fault`] whose generations wait `lock_wait` for the GPU lock.
fn door_at(
    scratch: &Scratch,
    s: &Setup,
    fault: Option<&str>,
    lock_wait: Duration,
) -> (OmegaDoor, String) {
    let mut d = OmegaDoor::from_parts(engine(s.weights), scratch.lock(), lock_wait);
    let vars: HashMap<String, String> = s.store_vars.iter().cloned().collect();
    let settings =
        kv_persist::settings_from_lookup(|n| vars.get(n).cloned()).expect("store dir is set");
    let identity = kv_persist::OmegaKvIdentity::new(&kv_persist::IdentityInputs {
        model_id: "tiny-qwen3-moe",
        model_sha256: [7u8; 32],
        model_digest_kind: "file",
        config: &host_test_engine::config(),
        rope_scaling: false,
        weights: s.weights,
        switches: d.kv_switches(),
        build: &s.build,
        env: &s.env,
    });
    let fault = kv_persist::TestFault::from_lookup(|n| {
        (n == kv_persist::TEST_FAULT_ENV)
            .then(|| fault.map(str::to_string))
            .flatten()
    });
    let (state, line) = kv_persist::open_with_test_fault(settings, identity, fault);
    if let PersistState::On(p) = &state {
        assert!(p.wait_ready(WAIT), "store recovery did not finish");
    }
    d.set_persistence(state);
    (d, line)
}

fn plain_door(scratch: &Scratch) -> OmegaDoor {
    OmegaDoor::from_parts(
        engine(WeightFormat::Fp8All),
        scratch.lock(),
        Duration::from_secs(5),
    )
}

fn greedy(n: usize) -> GenerateParams {
    GenerateParams {
        max_new_tokens: n,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 1,
        stop_token_ids: vec![],
    }
}

fn turn(d: &OmegaDoor, prompt: &[u32]) -> Vec<u32> {
    let mut out = Vec::new();
    d.generate(prompt, &greedy(6), &mut |t| {
        out.push(t);
        true
    })
    .unwrap();
    out
}

fn prompt1() -> Vec<u32> {
    (0..40u32).map(|i| (i * 7 + 3) % 64).collect()
}

/// Second turn prompt: everything the door holds after turn 1 plus a few new tokens (a chat turn
/// that extends the conversation).
fn prompt2(history: &[u32]) -> Vec<u32> {
    let mut p = history.to_vec();
    p.extend([5, 6, 7, 9]);
    p
}

fn counters(d: &OmegaDoor) -> kv_persist::PersistCounters {
    match d.persistence() {
        PersistState::On(p) => p.counters(),
        _ => panic!("persistence is not on"),
    }
}

/// The uninterrupted reference: one plain door that keeps running, turn 2 reuses its RAM prefix.
fn uninterrupted(scratch: &Scratch) -> (Vec<u32>, Vec<u32>, Vec<u32>) {
    let d = plain_door(scratch);
    let p1 = prompt1();
    let o1 = turn(&d, &p1);
    let h1 = d.history();
    let o2 = turn(&d, &prompt2(&h1));
    (o1, h1, o2)
}

/// Turn 1 on a persistence door, wait for the parks, stop it (drains the store).
fn park_turn1(scratch: &Scratch, s: &Setup) -> (Vec<u32>, Vec<u32>) {
    let (a, line) = door(scratch, s);
    assert!(line.contains("persistent KV: on"), "{line}");
    let o1 = turn(&a, &prompt1());
    let h1 = a.history();
    match a.persistence() {
        PersistState::On(p) => assert!(p.wait_idle(WAIT), "parks did not finish"),
        _ => unreachable!(),
    }
    let c = counters(&a);
    assert!(
        c.parked_chunks >= 3 && c.park_bytes > 0 && c.park_failed == 0,
        "{c:?}"
    );
    drop(a);
    (o1, h1)
}

#[test]
fn park_then_restart_then_restore_equals_the_uninterrupted_run() {
    let scratch = Scratch::new("restore");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (o1, h1) = park_turn1(&scratch, &s);
    let (u1, uh1, u2) = uninterrupted(&scratch);
    assert_eq!(
        (&o1, &h1),
        (&u1, &uh1),
        "turn 1 differs with persistence on"
    );

    // "Restart": a new engine and a new store handle on the same directory and key.
    let (b, _) = door(&scratch, &s);
    let r2 = turn(&b, &prompt2(&h1));
    let c = counters(&b);
    assert_eq!(c.restores, 1, "{c:?}");
    assert_eq!(
        c.restored_tokens,
        h1.len() as u64,
        "restored up to the uninterrupted reuse point: {c:?}"
    );
    assert_eq!(
        r2, u2,
        "restored continuation differs from the uninterrupted run"
    );
    assert_eq!(
        b.reused_last(),
        h1.len(),
        "the engine reused the restored prefix"
    );
}

#[test]
fn the_last_prompt_token_is_always_recomputed() {
    let scratch = Scratch::new("last");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (_, h1) = park_turn1(&scratch, &s);
    // Prompt 2 is exactly the parked history: everything is stored, but the last token is stepped.
    let (b, _) = door(&scratch, &s);
    let _ = turn(&b, &h1);
    let c = counters(&b);
    assert_eq!(c.restores, 1, "{c:?}");
    assert_eq!(c.restored_tokens, (h1.len() - 1) as u64, "{c:?}");
    assert_eq!(b.reused_last(), h1.len() - 1);
}

#[test]
fn a_longer_ram_prefix_is_not_replaced_by_the_store() {
    let scratch = Scratch::new("ram");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (a, _) = door(&scratch, &s);
    let _ = turn(&a, &prompt1());
    let h1 = a.history();
    match a.persistence() {
        PersistState::On(p) => assert!(p.wait_idle(WAIT)),
        _ => unreachable!(),
    }
    let _ = turn(&a, &prompt2(&h1));
    let c = counters(&a);
    assert_eq!(
        c.restores, 0,
        "the RAM prefix is as long as the store's: {c:?}"
    );
    assert_eq!(a.reused_last(), h1.len());
}

/// Each identity input changed alone: the parked prefix is never served.
#[test]
fn each_identity_input_changed_alone_is_never_served() {
    let scratch = Scratch::new("ident");
    let key = gen_key();
    let base = Setup::new(&scratch.store(), &key);
    let (_, h1) = park_turn1(&scratch, &base);
    let mut variants: Vec<(&str, Setup)> = Vec::new();
    let mut v = base.clone();
    v.weights = WeightFormat::Fp8Experts;
    variants.push(("weights fp8-experts", v));
    let mut v = base.clone();
    v.weights = WeightFormat::Bf16;
    variants.push(("weights bf16", v));
    let mut v = base.clone();
    v.build.omega_commit = "e".repeat(40);
    variants.push(("omega commit", v));
    let mut v = base.clone();
    v.build.gpu_archive_sha256 = "33".repeat(32);
    variants.push(("gpu archive digest", v));
    let mut v = base.clone();
    v.build.sc_commit = "44".repeat(20);
    variants.push(("engine commit", v));
    let mut v = base.clone();
    v.env.push(("AIEN_OMEGA_FP8_GEMV".into(), "0".into()));
    variants.push(("omega fp8 gemv switch", v));
    let mut v = base.clone();
    v.env
        .push(("AIEN_OMEGA_DGRAPH_PREFETCH".into(), "1".into()));
    variants.push(("decode graph prefetch switch", v));
    variants.push((
        "sharing namespace",
        base.clone().with_var("AIEN_KV_STORE_NAMESPACE", "other"),
    ));
    for (what, v) in variants {
        let (b, line) = door(&scratch, &v);
        assert!(line.contains("persistent KV: on"), "{what}: {line}");
        let _ = turn(&b, &prompt2(&h1));
        let c = counters(&b);
        assert_eq!(
            c.restores, 0,
            "{what} changed, yet the parked prefix was served: {c:?}"
        );
        assert!(
            c.restore_misses.get("not-found").copied().unwrap_or(0) >= 1,
            "{what}: {c:?}"
        );
        assert!(b.reused_last() < h1.len(), "{what}");
    }
    // The base setup still restores (the store is intact).
    let (b, _) = door(&scratch, &base);
    let _ = turn(&b, &prompt2(&h1));
    assert_eq!(counters(&b).restores, 1);
}

#[test]
fn unknown_omega_switches_and_the_mutants_refuse_persistence() {
    let scratch = Scratch::new("mutant");
    let base = Setup::new(&scratch.store(), &gen_key());
    for (k, v) in [
        ("OWN30B_DECODE_GRAPH_MUTANT", "skip-norm"),
        ("AIEN_OMEGA_SOMETHING_NEW", "1"),
    ] {
        let mut s = base.clone();
        s.env.push((k.into(), v.into()));
        let (d, line) = door(&scratch, &s);
        assert!(line.contains("REFUSED") && line.contains(k), "{k}: {line}");
        assert!(matches!(d.persistence(), PersistState::Refused(_)));
        let _ = turn(&d, &prompt1());
    }
}

#[test]
fn switch_off_reads_only_the_store_dir_and_changes_nothing() {
    for dir in [None, Some("")] {
        let seen = Mutex::new(Vec::<String>::new());
        let got = kv_persist::settings_from_lookup(|n| {
            seen.lock().unwrap().push(n.to_string());
            if n == "AIEN_KV_STORE_DIR" {
                dir.map(String::from)
            } else {
                Some("garbage that must not be read".into())
            }
        });
        assert!(got.is_none());
        assert_eq!(*seen.lock().unwrap(), vec!["AIEN_KV_STORE_DIR".to_string()]);
    }

    // Tokens: the door with persistence off equals the engine called the way the door always
    // called it (its own sampler, prefix reuse across turns).
    let scratch = Scratch::new("off");
    let d = plain_door(&scratch);
    assert!(matches!(d.persistence(), PersistState::Off));
    assert!(d.persistence_status().is_none());
    let raw = engine(WeightFormat::Fp8All);
    let p = greedy(6);
    let q = aien_inference_abi::qwen3_moe_omega::GenerateParams {
        max_new_tokens: p.max_new_tokens,
        temperature: p.temperature,
        top_p: p.top_p,
        top_k: p.top_k,
        seed: p.seed,
        stop_token_ids: p.stop_token_ids.clone(),
    };
    let raw_turn = |prompt: &[u32]| {
        let mut out = Vec::new();
        raw.generate_with(
            prompt,
            &q,
            &mut |lg: &[f32], step| sample_next(lg, &p, step),
            &mut |t| {
                out.push(t);
                true
            },
        )
        .unwrap();
        out
    };
    let p1 = prompt1();
    assert_eq!(turn(&d, &p1), raw_turn(&p1));
    let p2 = prompt2(&d.history());
    assert_eq!(turn(&d, &p2), raw_turn(&p2));

    // Health: the same keys as before persistence existed.
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cfg = ServerConfig {
        port: 0,
        model_id: "m".into(),
        ..ServerConfig::default()
    };
    let (addr, _h) = rt
        .block_on(serve(cfg, Arc::new(d), Arc::new(ByteCodec)))
        .unwrap();
    let v = get_json(addr, "/health");
    let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(|s| s.as_str()).collect();
    keys.sort();
    assert_eq!(keys, ["busy", "engine", "model", "status", "uptime_s"]);
}

#[test]
fn health_reports_the_persistence_counters_when_on() {
    let scratch = Scratch::new("health");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (d, _) = door(&scratch, &s);
    let _ = turn(&d, &prompt1());
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cfg = ServerConfig {
        port: 0,
        model_id: "m".into(),
        ..ServerConfig::default()
    };
    let (addr, _h) = rt
        .block_on(serve(cfg, Arc::new(d), Arc::new(ByteCodec)))
        .unwrap();
    let v = get_json(addr, "/health");
    let pk = &v["persistent_kv"];
    assert_eq!(pk["state"], "on", "{v}");
    for k in [
        "parked_chunks",
        "park_bytes",
        "park_dropped",
        "restores",
        "restored_tokens",
        "restore_bytes",
        "restore_misses",
        "refusals",
        "seal",
        "recovery",
        "io",
    ] {
        assert!(!pk[k].is_null(), "missing {k}: {pk}");
    }
    for k in [
        "done",
        "failed",
        "corrupt_deleted",
        "tmp_deleted",
        "unlink_failures",
        "foreign_files",
    ] {
        assert!(!pk["recovery"][k].is_null(), "missing recovery.{k}: {pk}");
    }
    for k in [
        "read_bytes",
        "write_bytes",
        "published",
        "failed",
        "disk_full",
    ] {
        assert!(!pk["io"][k].is_null(), "missing io.{k}: {pk}");
    }
    assert!(
        pk["io"]["write_bytes"].as_u64().unwrap() > 0,
        "the park wrote: {pk}"
    );
}

#[test]
fn a_corrupted_store_file_is_never_served() {
    let scratch = Scratch::new("corrupt");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (_, h1) = park_turn1(&scratch, &s);
    let mut flipped = 0;
    for f in files_under(&scratch.store().join("objects")) {
        let mut b = std::fs::read(&f).unwrap();
        let mid = b.len() / 2;
        b[mid] ^= 0x40;
        std::fs::write(&f, &b).unwrap();
        flipped += 1;
    }
    assert!(flipped >= 3, "expected parked objects, found {flipped}");
    let (b, _) = door(&scratch, &s);
    let r2 = turn(&b, &prompt2(&h1));
    let c = counters(&b);
    assert_eq!(c.restores, 0, "{c:?}");
    assert_eq!(
        r2,
        plain_turn2(&scratch, &h1),
        "a refused restore must be a plain recompute"
    );
}

#[test]
fn a_wrong_key_is_never_served() {
    let scratch = Scratch::new("wrongkey");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (_, h1) = park_turn1(&scratch, &s);
    let other = s.clone().with_var("AIEN_KV_STORE_KEY", &gen_key());
    let (b, _) = door(&scratch, &other);
    let r2 = turn(&b, &prompt2(&h1));
    if let PersistState::On(_) = b.persistence() {
        assert_eq!(counters(&b).restores, 0);
    }
    assert_eq!(r2, plain_turn2(&scratch, &h1));
}

#[test]
fn a_missing_or_malformed_key_refuses_and_runs_plain() {
    let scratch = Scratch::new("nokey");
    let base = Setup::new(&scratch.store(), &gen_key());
    for s in [
        base.clone().without("AIEN_KV_STORE_KEY"),
        base.clone().with_var("AIEN_KV_STORE_KEY", "abc"),
    ] {
        let (d, line) = door(&scratch, &s);
        assert!(line.contains("REFUSED"), "{line}");
        assert!(matches!(d.persistence(), PersistState::Refused(_)));
        let st = d.persistence_status().unwrap();
        assert_eq!(st["state"], "refused");
        let _ = turn(&d, &prompt1());
    }
    assert!(
        !scratch.store().exists(),
        "a refused start must not create the store"
    );
}

#[test]
fn an_unsafe_store_dir_refuses_and_runs_plain() {
    use std::os::unix::fs::PermissionsExt;
    let scratch = Scratch::new("unsafe");
    let open = scratch.0.join("open");
    std::fs::create_dir_all(&open).unwrap();
    std::fs::set_permissions(&open, std::fs::Permissions::from_mode(0o777)).unwrap();
    let s = Setup::new(&open.join("store"), &gen_key());
    let (d, line) = door(&scratch, &s);
    assert!(line.contains("REFUSED"), "{line}");
    assert!(matches!(d.persistence(), PersistState::Refused(_)));
    let _ = turn(&d, &prompt1());
}

fn plain_turn2(scratch: &Scratch, h1: &[u32]) -> Vec<u32> {
    let d = plain_door(scratch);
    turn(&d, &prompt2(h1))
}

fn files_under(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(rd) = std::fs::read_dir(dir) else {
        return out;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            out.extend(files_under(&p));
        } else {
            out.push(p);
        }
    }
    out
}

fn get_json(addr: std::net::SocketAddr, path: &str) -> serde_json::Value {
    let mut s = TcpStream::connect(addr).unwrap();
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    write!(
        s,
        "GET {path} HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n"
    )
    .unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let text = String::from_utf8(raw).unwrap();
    let body = text.split_once("\r\n\r\n").unwrap().1;
    serde_json::from_str(body).unwrap_or_else(|e| panic!("{e}: {body}"))
}

fn pk_of(d: &OmegaDoor) -> &kv_persist::KvPersistence {
    match d.persistence() {
        PersistState::On(p) => p,
        _ => panic!("persistence is not on"),
    }
}

/// Copies `src` to `dst` with its modes (a crash image of the store taken while a write is in
/// flight: what a power cut or `kill -9` leaves on disk).
fn copy_tree(src: &Path, dst: &Path) {
    use std::os::unix::fs::PermissionsExt;
    std::fs::create_dir_all(dst).unwrap();
    let mode = std::fs::metadata(src).unwrap().permissions().mode();
    std::fs::set_permissions(dst, std::fs::Permissions::from_mode(mode)).unwrap();
    for e in std::fs::read_dir(src).unwrap().flatten() {
        let (p, q) = (e.path(), dst.join(e.file_name()));
        if p.is_dir() {
            copy_tree(&p, &q);
        } else {
            std::fs::copy(&p, &q).unwrap();
        }
    }
}

fn wait_for(what: &str, mut ok: impl FnMut() -> bool) {
    let t0 = std::time::Instant::now();
    while !ok() {
        assert!(t0.elapsed() < WAIT, "timed out waiting for {what}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

#[test]
fn the_test_fault_spec_is_parsed_strictly() {
    let parse = |v: &str| kv_persist::TestFault::from_lookup(|_| Some(v.to_string()));
    assert!(parse("").unwrap().is_none(), "empty = unset");
    for ok in ["p1:5", "p2:0", "p3:100:1", "p4:1:2", "p5:7", "load:12000:1"] {
        assert_eq!(parse(ok).unwrap().unwrap().spec(), ok);
    }
    for bad in [
        "p3", "p6:5", "p3:x", "p3:5:y", "p3:5:1:2", "load:-1", "P3:5",
    ] {
        assert!(parse(bad).is_err(), "{bad} must refuse");
    }
    // A malformed spec refuses persistence (fail closed); the door runs plain.
    let scratch = Scratch::new("badfault");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (d, line) = door_with_fault(&scratch, &s, Some("p9:1"));
    assert!(
        line.contains("REFUSED") && line.contains("AIEN_KV_TEST_FAULT"),
        "{line}"
    );
    assert!(matches!(d.persistence(), PersistState::Refused(_)));
}

#[test]
fn a_write_interrupted_before_publish_is_never_served() {
    // The park stalls before its rename; the store is imaged at that moment (the kill -9 of
    // the chip check), the temp file is torn in half, and a new door opens the image.
    let scratch = Scratch::new("torn");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (a, line) = door_with_fault(&scratch, &s, Some("p3:60000"));
    assert!(line.contains("TEST FAULT"), "{line}");
    let h1 = {
        let _ = turn(&a, &prompt1());
        a.history()
    };
    let tmp = scratch.store().join("tmp");
    wait_for("a temp file", || !files_under(&tmp).is_empty());
    let image = scratch.0.join("image");
    copy_tree(&scratch.store(), &image);
    // The stalled worker would keep the first door alive for a minute: leave it behind, as
    // the killed process would.
    std::mem::forget(a);
    let torn = files_under(&image.join("tmp"));
    assert!(!torn.is_empty());
    for t in &torn {
        let len = std::fs::metadata(t).unwrap().len();
        std::fs::OpenOptions::new()
            .write(true)
            .open(t)
            .unwrap()
            .set_len(len / 2)
            .unwrap();
    }
    assert!(
        files_under(&image.join("objects")).is_empty(),
        "nothing was published"
    );
    let si = Setup::new(&image, "").with_var("AIEN_KV_STORE_KEY", &s.store_vars[1].1);
    let (b, _) = door(&scratch, &si);
    let r2 = turn(&b, &prompt2(&h1));
    let c = counters(&b);
    assert_eq!(c.restores, 0, "{c:?}");
    assert_eq!(
        r2,
        plain_turn2(&scratch, &h1),
        "a torn write must be a plain recompute"
    );
    assert!(
        files_under(&image.join("tmp")).is_empty(),
        "recovery removes the torn temp"
    );
    if let PersistState::On(pk) = b.persistence() {
        let v = pk.status_json();
        assert!(
            v["recovery"]["tmp_deleted"].as_u64().unwrap() >= 1,
            "health counts the removed temp: {v}"
        );
        assert_eq!(v["recovery"]["done"], true, "{v}");
    } else {
        panic!("persistence should be on");
    }
}

#[test]
fn a_restore_that_completes_after_its_timeout_is_never_served() {
    // The load of turn 2 stalls past LOAD_TIMEOUT: the door gives up, prefills, and the late
    // completion that arrives afterwards is dropped. The store itself stays good.
    let scratch = Scratch::new("lateload");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (_, h1) = park_turn1(&scratch, &s);
    let late = kv_persist::LOAD_TIMEOUT + Duration::from_secs(2);
    let spec = format!("load:{}:1", late.as_millis());
    let (b, _) = door_with_fault(&scratch, &s, Some(&spec));
    let r2 = turn(&b, &prompt2(&h1));
    let c = counters(&b);
    assert_eq!(c.restores, 0, "{c:?}");
    assert_eq!(c.restore_misses.get("timeout"), Some(&1), "{c:?}");
    let plain = plain_turn2(&scratch, &h1);
    assert_eq!(r2, plain);
    std::thread::sleep(Duration::from_secs(3));
    let c = counters(&b);
    assert_eq!(
        c.restores, 0,
        "the late completion must not count as a restore: {c:?}"
    );
    // A new conversation on the same door is not touched by the stale completion.
    let fresh: Vec<u32> = (0..30u32).map(|i| (i * 5 + 1) % 64).collect();
    assert_eq!(turn(&b, &fresh), turn(&plain_door(&scratch), &fresh));
    drop(b);
    let (c2, _) = door(&scratch, &s);
    assert_eq!(turn(&c2, &prompt2(&h1)), plain);
    assert_eq!(
        counters(&c2).restores,
        1,
        "the store still serves after the stale load"
    );
}

#[test]
fn a_park_that_lands_after_the_conversation_moved_on_is_harmless() {
    // The first park of turn 1 stalls; turn 2 and a diverging turn run while it is in flight
    // (turn 2 resubmits the chunks the store does not hold yet). Every answer equals the
    // uninterrupted run, and after a restart the late parks restore turn 2 exactly.
    let scratch = Scratch::new("latepark");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (u1, uh1, u2) = uninterrupted(&scratch);
    let (a, _) = door_with_fault(&scratch, &s, Some("p3:1500:1"));
    assert_eq!(turn(&a, &prompt1()), u1);
    let h1 = a.history();
    assert_eq!(h1, uh1);
    assert_eq!(turn(&a, &prompt2(&h1)), u2);
    let mut other = h1.clone();
    other.extend([1, 2, 3, 4]);
    let plain_other = turn(&plain_door(&scratch), &other);
    assert_eq!(turn(&a, &other), plain_other);
    assert!(pk_of(&a).wait_idle(WAIT), "parks did not finish");
    let c = counters(&a);
    assert!(c.parked_chunks >= 3 && c.park_failed == 0, "{c:?}");
    drop(a);
    let (b, _) = door(&scratch, &s);
    assert_eq!(turn(&b, &prompt2(&h1)), u2);
    assert_eq!(counters(&b).restores, 1);
}

#[test]
fn a_graceful_shutdown_releases_the_engine_and_flushes_the_parks() {
    // serve_until: on shutdown the server stops and lets go of the engine; dropping the door
    // then drains the parks still in flight (each stalls 200 ms) and leaves no temp file.
    let scratch = Scratch::new("shutdown");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (d, _) = door_with_fault(&scratch, &s, Some("p3:200"));
    let _ = turn(&d, &prompt1());
    let h1 = d.history();
    let keep = Arc::new(d);
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cfg = ServerConfig {
        port: 0,
        model_id: "m".into(),
        ..ServerConfig::default()
    };
    let (tx, rx) = tokio::sync::oneshot::channel::<()>();
    let engine: Arc<dyn Engine> = keep.clone();
    let (addr, h) = rt
        .block_on(serve_until(cfg, engine, Arc::new(ByteCodec), async {
            let _ = rx.await;
        }))
        .unwrap();
    assert_eq!(get_json(addr, "/health")["status"], "ok");
    assert!(
        pk_of(&keep).counters().parked_chunks < 3,
        "the parks are still in flight"
    );
    tx.send(()).unwrap();
    rt.block_on(h).unwrap();
    assert!(TcpStream::connect(addr).is_err(), "the listener is closed");
    wait_for("the server to release the engine", || {
        Arc::strong_count(&keep) == 1
    });
    drop(keep);
    assert!(
        files_under(&scratch.store().join("tmp")).is_empty(),
        "no temp file left"
    );
    assert!(
        files_under(&scratch.store().join("objects")).len() >= 3,
        "the parks were flushed"
    );
    let (b, _) = door(&scratch, &s);
    let _ = turn(&b, &prompt2(&h1));
    assert_eq!(counters(&b).restores, 1);
}

/// #514 chip findings: on the 30B model one 256-token chunk is 25 MB. The storage worker's park
/// budget (a quarter of `AIEN_KV_STORE_MAX_INFLIGHT_BYTES`, 64 MB by default) holds two chunks,
/// so a 2079-token turn parked 2 of 9 chunks and refused the rest; its restore budget (the other
/// three quarters, 192 MB) is smaller than that turn's 204 MB, so the whole restore was refused.
/// Here the park budget holds exactly one chunk and the restore budget three, for a 7-chunk
/// history: every chunk must still be parked and the restart must restore the whole history.
#[test]
fn a_turn_larger_than_the_park_and_restore_budgets_is_parked_and_restored_in_full() {
    let long: Vec<u32> = (0..100u32).map(|i| (i * 11 + 5) % 64).collect();
    let probe = Scratch::new("budget-probe");
    let (a, line) = door(&probe, &Setup::new(&probe.store(), &gen_key()));
    assert!(line.contains("persistent KV: on"), "{line}");
    let _ = turn(&a, &long);
    assert!(pk_of(&a).wait_idle(WAIT));
    let c = counters(&a);
    assert!(c.parked_chunks >= 7 && c.park_refused == 0, "{c:?}");
    let (n, per_chunk) = (c.parked_chunks, c.park_bytes / c.parked_chunks);
    drop(a);

    let scratch = Scratch::new("budget");
    let budget = 4 * (per_chunk + aien_kv_store::record::HEADER_REGION_LEN as u64);
    let s = Setup::new(&scratch.store(), &gen_key())
        .with_var("AIEN_KV_STORE_MAX_INFLIGHT_BYTES", &budget.to_string());
    let (a, line) = door(&scratch, &s);
    assert!(line.contains("persistent KV: on"), "{line}");
    let _ = turn(&a, &long);
    let h1 = a.history();
    assert!(pk_of(&a).wait_idle(WAIT));
    let c = counters(&a);
    assert_eq!(
        (c.parked_chunks, c.park_refused, c.park_failed),
        (n, 0, 0),
        "every chunk parked through a one-chunk park budget: {c:?}"
    );
    drop(a);

    let plain = plain_door(&scratch);
    let _ = turn(&plain, &long);
    let u2 = turn(&plain, &prompt2(&plain.history()));
    let (b, _) = door(&scratch, &s);
    let r2 = turn(&b, &prompt2(&h1));
    let c = counters(&b);
    assert_eq!(
        (c.restores, c.restored_tokens),
        (1, h1.len() as u64),
        "the whole history restored through a three-chunk restore budget: {c:?}"
    );
    assert_eq!(
        r2, u2,
        "restored continuation differs from the uninterrupted run"
    );
}

/// What the storage worker charges for one chunk of these tests: its payload (one block at
/// `AIEN_KV_CHUNK_BLOCKS=1`, the same for every chunk) plus the header region. Probed by parking a
/// turn on a door with the default budgets.
fn chunk_charge() -> u64 {
    let probe = Scratch::new("charge-probe");
    let (a, line) = door(&probe, &Setup::new(&probe.store(), &gen_key()));
    assert!(line.contains("persistent KV: on"), "{line}");
    let _ = turn(&a, &prompt1());
    assert!(pk_of(&a).wait_idle(WAIT));
    let c = counters(&a);
    assert!(c.parked_chunks > 0 && c.park_refused == 0, "{c:?}");
    c.park_bytes / c.parked_chunks + aien_kv_store::record::HEADER_REGION_LEN as u64
}

/// sc#523 item 2: a chunk that can never fit the worker's park budget (a quarter of
/// `AIEN_KV_STORE_MAX_INFLIGHT_BYTES`) or restore budget (the rest) is refused at open, not
/// spun on at every turn until the credit wait runs out.
#[test]
fn a_chunk_larger_than_the_park_or_restore_budget_refuses_at_open() {
    let c = chunk_charge();
    let scratch = Scratch::new("oversize");
    let s = Setup::new(&scratch.store(), &gen_key());
    // Park budget one byte short of a chunk, restore budget three chunks: parking can never run.
    let short_park = s
        .clone()
        .with_var("AIEN_KV_STORE_MAX_INFLIGHT_BYTES", &(4 * c - 4).to_string());
    let (d, line) = door(&scratch, &short_park);
    assert!(
        line.contains("REFUSED") && line.contains("park budget"),
        "{line}"
    );
    assert!(matches!(d.persistence(), PersistState::Refused(_)));
    let t0 = Instant::now();
    assert_eq!(
        turn(&d, &prompt1()),
        turn(&plain_door(&scratch), &prompt1())
    );
    assert!(t0.elapsed() < Duration::from_secs(5), "{:?}", t0.elapsed());
    drop(d);
    // The same budgets with parking off: restore alone fits, so persistence is on.
    let (d, line) = door(&scratch, &short_park.clone().with_var("AIEN_KV_PARK", "0"));
    assert!(line.contains("persistent KV: on"), "{line}");
    drop(d);
    // Restore budget (three quarters of one chunk) short of a chunk: restoring can never run.
    let short_restore = s
        .with_var("AIEN_KV_PARK", "0")
        .with_var("AIEN_KV_STORE_MAX_INFLIGHT_BYTES", &c.to_string());
    let (d, line) = door(&scratch, &short_restore);
    assert!(
        line.contains("REFUSED") && line.contains("restore budget"),
        "{line}"
    );
    assert!(matches!(d.persistence(), PersistState::Refused(_)));
}

/// sc#523 item 3: one fetch holds the request for at most [`kv_persist::LOAD_TIMEOUT`] in all,
/// submit phase and wait phase together. The restore budget holds one chunk, so each of the three
/// loads of turn 2 waits for the one before; each load stalls 4 s, so the last is submitted at
/// about 8 s and lands at about 12 s.
#[test]
fn one_fetch_waits_at_most_the_load_timeout_over_submit_and_wait() {
    let c = chunk_charge();
    let scratch = Scratch::new("onedeadline");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (_, h1) = park_turn1(&scratch, &s);
    let narrow = s
        .clone()
        .with_var("AIEN_KV_PARK", "0")
        .with_var("AIEN_KV_STORE_MAX_INFLIGHT_BYTES", &(2 * c).to_string());
    let (b, line) = door_with_fault(&scratch, &narrow, Some("load:4000:3"));
    assert!(line.contains("persistent KV: on"), "{line}");
    let t0 = Instant::now();
    let r2 = turn(&b, &prompt2(&h1));
    let took = t0.elapsed();
    let c = counters(&b);
    assert_eq!(
        (c.restores, c.restore_misses.get("timeout").copied()),
        (0, Some(1)),
        "turn 2 took {took:?}: {c:?}"
    );
    assert!(
        took < kv_persist::LOAD_TIMEOUT + Duration::from_secs(2),
        "one fetch held the request {took:?}"
    );
    assert_eq!(r2, plain_turn2(&scratch, &h1));
}

/// sc#523 item 4: a refused park drops the rest of its chain (a chain with a hole cannot be
/// restored past it); the dropped chunks are counted. The park budget holds one chunk and the
/// first park stalls past [`kv_persist::PARK_CREDIT_WAIT`], so chunk 2 is refused and the rest
/// is dropped.
#[test]
fn a_refused_park_counts_the_chunks_it_drops() {
    let c = chunk_charge();
    let scratch = Scratch::new("parkdrop");
    let s = Setup::new(&scratch.store(), &gen_key())
        .with_var("AIEN_KV_STORE_MAX_INFLIGHT_BYTES", &(4 * c).to_string());
    let stall = kv_persist::PARK_CREDIT_WAIT + Duration::from_secs(2);
    let (a, line) = door_with_fault(&scratch, &s, Some(&format!("p3:{}:1", stall.as_millis())));
    assert!(line.contains("persistent KV: on"), "{line}");
    let _ = turn(&a, &prompt1());
    let chunks = a.history().len().div_ceil(16) as u64;
    assert!(chunks >= 3, "{chunks}");
    assert!(pk_of(&a).wait_idle(WAIT));
    let v = pk_of(&a).status_json();
    assert_eq!(v["chunk_tokens"], 16, "{v}");
    assert_eq!(
        (
            v["parked_chunks"].as_u64(),
            v["park_refused"].as_u64(),
            v["park_dropped"].as_u64()
        ),
        (Some(1), Some(1), Some(chunks - 2)),
        "every chunk of the turn is accounted for: {v}"
    );
}

/// sc#523 item 5: a prefix loaded from the store but never imported because the GPU lock stayed
/// busy is counted as a restore miss (`gpu-lock-busy`), not lost from the counters.
#[test]
fn a_fetched_prefix_left_unused_by_a_busy_gpu_lock_is_a_miss() {
    let scratch = Scratch::new("lockbusy");
    let s = Setup::new(&scratch.store(), &gen_key());
    let (_, h1) = park_turn1(&scratch, &s);
    let (b, _) = door_at(&scratch, &s, None, Duration::from_millis(200));
    let held = GpuLock::acquire_at(&scratch.lock(), Duration::from_secs(1)).unwrap();
    let r = b.generate(&prompt2(&h1), &greedy(6), &mut |_| true);
    assert!(
        r.is_err(),
        "the door must answer busy while the lock is held"
    );
    let c = counters(&b);
    assert_eq!(
        (c.restores, c.restore_misses.get("gpu-lock-busy").copied()),
        (0, Some(1)),
        "{c:?}"
    );
    drop(held);
    assert_eq!(turn(&b, &prompt2(&h1)), plain_turn2(&scratch, &h1));
    assert_eq!(counters(&b).restores, 1, "the store still serves");
}
