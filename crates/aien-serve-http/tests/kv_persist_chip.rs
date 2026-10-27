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

//! Chip-only acceptance check of persistent KV on the omega engine (sc#505, locked by #499).
//! NOT run by `cargo test`: it loads the 30B checkpoint onto the chip. Run it only through
//! `scripts/own30b-pkv-chip-check.sh` (which builds natively and checks the chip is free):
//!
//!   OWN30B_MODEL_DIR=... cargo test --release -p aien-serve-http --test kv_persist_chip -- --ignored
//!
//! In one process: turn 1, wait for the parks, turn 2 with the RAM prefix (the uninterrupted
//! run), then forget the RAM KV and run turn 2 again so it must restore from the store. The two
//! turn 2 replies and the K/V the engine holds afterwards must be identical, byte for byte.
//! The store key is generated in memory for the run; the store lives in a private temp dir.
//! Takes the hive GPU lock (/tmp/aien-gb10.lock) around each generation, as the server does.

use std::io::Read;
use std::path::PathBuf;
use std::time::Duration;

use aien_inference_abi::qwen3_moe_omega::WeightFormat;
use aien_serve_http::engine::{Engine, GenerateParams};
use aien_serve_http::kv_persist::{self, OmegaBuild, PersistState};
use aien_serve_http::omega_engine::{OmegaDoor, PrefillMode};

fn gen_key() -> String {
    let mut b = [0u8; 32];
    std::fs::File::open("/dev/urandom")
        .unwrap()
        .read_exact(&mut b)
        .unwrap();
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn turn(d: &OmegaDoor, prompt: &[u32], n: usize) -> Vec<u32> {
    let p = GenerateParams {
        max_new_tokens: n,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 1,
        stop_token_ids: vec![],
    };
    let mut out = Vec::new();
    d.generate(prompt, &p, &mut |t| {
        out.push(t);
        true
    })
    .unwrap();
    out
}

#[test]
#[ignore = "chip-only: loads the 30B checkpoint on omega; run via scripts/own30b-pkv-chip-check.sh"]
fn restored_turn_two_equals_the_uninterrupted_turn_two_on_the_chip() {
    let model = PathBuf::from(std::env::var("OWN30B_MODEL_DIR").expect("OWN30B_MODEL_DIR"));
    let scratch = std::env::temp_dir().join(format!("pkv505-chip-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&scratch, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut door = OmegaDoor::load(
        &model,
        8192,
        WeightFormat::Fp8All,
        Some(2000),
        PrefillMode::default(),
    )
    .expect("omega load");
    let store = scratch.join("store");
    let key = gen_key();
    let settings = kv_persist::settings_from_lookup(|n| match n {
        "AIEN_KV_STORE_DIR" => Some(store.display().to_string()),
        "AIEN_KV_STORE_KEY" => Some(key.clone()),
        _ => None,
    })
    .unwrap();
    let env: Vec<(String, String)> = std::env::vars().collect();
    let (sha, kind) = kv_persist::checkpoint_digest(&model).unwrap();
    let identity = kv_persist::OmegaKvIdentity::new(&kv_persist::IdentityInputs {
        model_id: "own30b-chip-check",
        model_sha256: sha,
        model_digest_kind: kind,
        config: &door.config(),
        rope_scaling: kv_persist::config_has_rope_scaling(&model).unwrap(),
        weights: WeightFormat::Fp8All,
        switches: door.kv_switches(),
        build: &OmegaBuild::linked(),
        env: &env,
    });
    let (state, line) = kv_persist::open(settings, identity);
    eprintln!("{line}");
    let PersistState::On(_) = &state else {
        panic!("persistence refused: {line}")
    };
    door.set_persistence(state);
    let pk = match door.persistence() {
        PersistState::On(p) => p,
        _ => unreachable!(),
    };
    assert!(pk.wait_ready(Duration::from_secs(60)));

    // A 600-token prompt (the same sentence ids repeated), then a continuation.
    let p1: Vec<u32> = (0..600u32).map(|i| 1000 + (i * 37) % 5000).collect();
    let _ = turn(&door, &p1, 32);
    assert!(
        pk.wait_idle(Duration::from_secs(60)),
        "parks did not finish"
    );
    let h1 = door.history();
    let mut p2 = h1.clone();
    p2.extend([2000u32, 2001, 2002, 2003, 2004]);

    // Freeze the store as it was after turn 1 (the state a restarted server would find).
    pk.set_park(false);
    let t_u = turn(&door, &p2, 64);
    let reused_u = door.reused_last();
    let h_u = door.history();
    let kv_u = door.export_kv_for_check(h_u.len()).unwrap();

    door.reset_kv();
    let t_r = turn(&door, &p2, 64);
    let c = pk.counters();
    eprintln!("counters after the restored turn: {c:?}");
    assert_eq!(c.restores, 1, "turn 2 did not restore: {c:?}");
    assert_eq!(
        c.restored_tokens as usize, reused_u,
        "restored up to the uninterrupted reuse point"
    );
    assert_eq!(door.reused_last(), reused_u);
    assert_eq!(
        t_r, t_u,
        "restored continuation differs from the uninterrupted run"
    );
    assert_eq!(door.history(), h_u);
    let kv_r = door.export_kv_for_check(h_u.len()).unwrap();
    assert!(
        kv_r == kv_u,
        "the K/V after the restored turn differs from the uninterrupted one"
    );
    drop(door);
    let _ = std::fs::remove_dir_all(&scratch);
}

/// `/proc/self/io` `read_bytes`: bytes this process caused to be read from the block layer.
fn proc_read_bytes() -> u64 {
    let s = std::fs::read_to_string("/proc/self/io").unwrap();
    s.lines()
        .find_map(|l| l.strip_prefix("read_bytes: "))
        .unwrap()
        .trim()
        .parse()
        .unwrap()
}

/// Drops the page cache (DGX Spark known issues: `sync; echo 3 > /proc/sys/vm/drop_caches`).
fn drop_caches() {
    let ok = std::process::Command::new("sudo")
        .args(["-n", "sh", "-c", "sync; echo 3 > /proc/sys/vm/drop_caches"])
        .status()
        .unwrap()
        .success();
    assert!(ok, "drop_caches needs sudo -n");
}

fn store_objects(dir: &std::path::Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            store_objects(&p, out);
        } else {
            out.push(p);
        }
    }
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn summary(v: &mut [f64]) -> String {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    format!(
        "min={:.3} median={:.3} max={:.3} n={}",
        v[0],
        v[v.len() / 2],
        v[v.len() - 1],
        v.len()
    )
}

/// Chip qualification of #514 (plan: docs/inference/evidence/pkv514-chip-20261014/PLAN.md):
/// the fused-cache export/import round trip byte for byte, the checkpoint cost, the NVMe restore
/// latency cold and warm, and the negative controls (a flipped payload byte must show, a
/// truncated payload and corrupted stored objects must be refused). Prints `SNAP` lines.
#[test]
#[ignore = "chip-only: loads the 30B checkpoint on omega; run by hand per the #514 plan"]
fn fused_cache_round_trip_cost_and_restore_latency_on_the_chip() {
    use std::time::Instant;
    const REPS: usize = 5;
    let model = PathBuf::from(std::env::var("OWN30B_MODEL_DIR").expect("OWN30B_MODEL_DIR"));
    // The store must be on the NVMe (not tmpfs) for a cold read.
    let scratch_root = std::env::var("PKV_CHIP_SCRATCH")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir());
    let scratch = scratch_root.join(format!("pkv514-chip-{}", std::process::id()));
    std::fs::create_dir_all(&scratch).unwrap();
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&scratch, std::fs::Permissions::from_mode(0o700)).unwrap();
    }
    let mut door = OmegaDoor::load(
        &model,
        8192,
        WeightFormat::Fp8All,
        Some(2000),
        PrefillMode::default(),
    )
    .expect("omega load");
    let sw = door.kv_switches();
    eprintln!("SNAP site=switches attention_path={}", sw.attention_path);
    assert_eq!(
        sw.attention_path, "fused",
        "the live server runs the fused cache"
    );
    let store = scratch.join("store");
    let key = gen_key();
    let settings = kv_persist::settings_from_lookup(|n| match n {
        "AIEN_KV_STORE_DIR" => Some(store.display().to_string()),
        "AIEN_KV_STORE_KEY" => Some(key.clone()),
        _ => None,
    })
    .unwrap();
    let env: Vec<(String, String)> = std::env::vars().collect();
    let (sha, kind) = kv_persist::checkpoint_digest(&model).unwrap();
    let identity = kv_persist::OmegaKvIdentity::new(&kv_persist::IdentityInputs {
        model_id: "own30b-chip-check",
        model_sha256: sha,
        model_digest_kind: kind,
        config: &door.config(),
        rope_scaling: kv_persist::config_has_rope_scaling(&model).unwrap(),
        weights: WeightFormat::Fp8All,
        switches: door.kv_switches(),
        build: &OmegaBuild::linked(),
        env: &env,
    });
    let (state, line) = kv_persist::open(settings, identity);
    eprintln!("{line}");
    let PersistState::On(_) = &state else {
        panic!("persistence refused: {line}")
    };
    door.set_persistence(state);
    let pk = match door.persistence() {
        PersistState::On(p) => p,
        _ => unreachable!(),
    };
    assert!(pk.wait_ready(Duration::from_secs(60)));

    // Turn 1: a 2048-token prompt and 32 generated tokens, parked by the door.
    let p1: Vec<u32> = (0..2048u32).map(|i| 1000 + (i * 37) % 5000).collect();
    let _ = turn(&door, &p1, 32);
    assert!(
        pk.wait_idle(Duration::from_secs(120)),
        "parks did not finish"
    );
    pk.set_park(false);
    let c = pk.counters();
    let h = door.history();
    let n = h.len();
    eprintln!(
        "SNAP site=park positions={n} parked_chunks={} park_bytes={} park_failed={} park_refused={}",
        c.parked_chunks, c.park_bytes, c.park_failed, c.park_refused
    );
    assert!(c.parked_chunks > 0 && c.park_failed == 0 && c.park_refused == 0);

    // Item 3: checkpoint cost, the export of the whole history under the GPU lock.
    let mut t_export = Vec::new();
    let mut a: Option<Vec<u8>> = None;
    for _ in 0..REPS {
        let t0 = Instant::now();
        let x = door.export_kv_for_check(n).unwrap();
        t_export.push(ms(t0.elapsed()));
        match &a {
            None => a = Some(x),
            Some(a) => assert!(*a == x, "two exports of the same KV differ"),
        }
    }
    let a = a.unwrap();
    assert!(a.iter().any(|&b| b != 0), "export is all zeros");
    eprintln!(
        "SNAP site=export positions={n} bytes={} ms_{}",
        a.len(),
        summary(&mut t_export)
    );

    // Item 1: export -> reset -> import -> export, byte for byte; import timed.
    let mut t_import = Vec::new();
    for _ in 0..REPS {
        door.reset_kv();
        assert!(door.history().is_empty());
        let t0 = Instant::now();
        door.restore_kv_for_check(&h, &a).unwrap();
        t_import.push(ms(t0.elapsed()));
        assert_eq!(door.history(), h);
        let b = door.export_kv_for_check(n).unwrap();
        assert!(b == a, "round trip export(import(A)) != A");
    }
    eprintln!(
        "SNAP site=round_trip identical=true reps={REPS} import_ms_{}",
        summary(&mut t_import)
    );

    // Negative control: one flipped byte in the middle of the payload must come back flipped.
    let mut bad = a.clone();
    let off = bad.len() / 2;
    bad[off] ^= 0x40;
    door.restore_kv_for_check(&h, &bad).unwrap();
    let b = door.export_kv_for_check(n).unwrap();
    let diff: Vec<usize> = (0..a.len()).filter(|&i| a[i] != b[i]).collect();
    eprintln!(
        "SNAP site=neg_flip differing_bytes={} at={:?}",
        diff.len(),
        diff.first()
    );
    assert_eq!(
        diff,
        vec![off],
        "a flipped byte must show at exactly its offset"
    );
    // Negative control: a truncated payload is refused and leaves an empty engine.
    let r = door.restore_kv_for_check(&h, &a[..a.len() - 1]);
    eprintln!(
        "SNAP site=neg_truncated refused={} err={:?}",
        r.is_err(),
        r.as_ref().err()
    );
    assert!(r.is_err() && door.history().is_empty());
    door.restore_kv_for_check(&h, &a).unwrap();
    assert!(door.export_kv_for_check(n).unwrap() == a);

    // Item 4: NVMe restore latency (lookup, read, decrypt, verify), warm then cold.
    let mut p2 = h.clone();
    p2.extend([2000u32, 2001, 2002, 2003, 2004]);
    let mut objs = Vec::new();
    store_objects(&store.join("objects"), &mut objs);
    let obj_bytes: u64 = objs.iter().map(|p| p.metadata().unwrap().len()).sum();
    eprintln!(
        "SNAP site=store objects={} object_bytes={obj_bytes}",
        objs.len()
    );
    let mut warm = Vec::new();
    for _ in 0..REPS {
        let t0 = Instant::now();
        let f = pk.fetch(&p2, 0).expect("warm fetch");
        warm.push(ms(t0.elapsed()));
        assert_eq!(f.tokens, n);
        assert!(f.payload == a, "warm fetched payload != A");
    }
    let mut cold = Vec::new();
    let mut cold_read = Vec::new();
    for _ in 0..REPS {
        drop_caches();
        let r0 = proc_read_bytes();
        let t0 = Instant::now();
        let f = pk.fetch(&p2, 0).expect("cold fetch");
        cold.push(ms(t0.elapsed()));
        let rb = proc_read_bytes() - r0;
        cold_read.push(rb);
        assert_eq!(f.tokens, n);
        assert!(f.payload == a, "cold fetched payload != A");
    }
    eprintln!("SNAP site=fetch_warm ms_{}", summary(&mut warm));
    eprintln!(
        "SNAP site=fetch_cold ms_{} block_read_bytes={cold_read:?} object_bytes={obj_bytes}",
        summary(&mut cold)
    );
    // Cold is proven only if the reads reached the block layer.
    assert!(
        cold_read.iter().all(|&r| r >= obj_bytes),
        "a cold fetch read less than the stored bytes from the device: not cold"
    );

    // Negative control: corrupt every stored object; the fetch must refuse.
    let before = pk.counters().restore_misses;
    for p in &objs {
        let mut v = std::fs::read(p).unwrap();
        let m = v.len() / 2;
        v[m] ^= 0x40;
        std::fs::write(p, v).unwrap();
    }
    drop_caches();
    let f = pk.fetch(&p2, 0);
    let after = pk.counters().restore_misses;
    eprintln!(
        "SNAP site=neg_corrupt refused={} misses_before={before:?} misses_after={after:?}",
        f.is_none()
    );
    assert!(f.is_none(), "a corrupted checkpoint was served");
    assert!(after != before, "the refusal was not counted");
    drop(door);
    let _ = std::fs::remove_dir_all(&scratch);
}
