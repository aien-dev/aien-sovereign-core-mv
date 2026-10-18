//! Expert Ribbon P1 (sc#629) read receipt: random verified expert reads through the bounded read adapter
//! from the real checkpoint on the NVMe. Read-only on the model; no GPU.
//!
//! ribbon_read_bench --model DIR [--manifest PATH] [--reads N] [--workers W] [--depth D] [--buffered] [--seed S]
//!
//! Each read = one whole expert (all six tensors, the two aligned segments), BLAKE3-verified before it counts.
//! Keeps `depth` reads in flight. Prints one JSON summary: payload GB/s, device GB/s, latency mean/p50/p95/max.
//! --buffered reads through the page cache: a warm cache makes it a memory copy, not an SSD figure.

use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use aien_heartbeat::{FileShards, Manifest, ReadAdapter, ReadConfig};

fn arg<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse().unwrap_or_else(|_| panic!("bad value for {name}")))
        .unwrap_or(default)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model: PathBuf = arg(&args, "--model", PathBuf::from("."));
    let manifest_path: PathBuf = arg(
        &args,
        "--manifest",
        std::env::temp_dir().join("aien-ribbon-manifest.json"),
    );
    let reads: usize = arg(&args, "--reads", 2000);
    let workers: usize = arg(&args, "--workers", 8);
    let depth: usize = arg(&args, "--depth", 16);
    let seed: u64 = arg(&args, "--seed", 2026);
    let direct = !args.iter().any(|a| a == "--buffered");

    let m = match Manifest::load(&manifest_path) {
        Ok(m) if m.model_dir == model => m,
        _ => {
            let t = Instant::now();
            let m = Manifest::build(&model).expect("manifest");
            m.save(&manifest_path).expect("save manifest");
            eprintln!(
                "manifest built in {:.1} s ({} experts, model digest {})",
                t.elapsed().as_secs_f64(),
                m.experts.len(),
                m.model_digest
            );
            m
        }
    };
    let m = Arc::new(m);
    let dev = Arc::new(FileShards::open(&m.shard_paths(), direct).expect("open shards"));
    let r = ReadAdapter::new(
        m.clone(),
        dev,
        ReadConfig {
            workers,
            queue_depth: depth,
        },
    );

    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (s >> 33) as usize % m.experts.len()
    };
    let mut lat: Vec<f64> = Vec::with_capacity(reads);
    let mut inflight = VecDeque::new();
    let mut payload = 0u64;
    let t0 = Instant::now();
    let mut submitted = 0usize;
    while lat.len() < reads {
        while submitted < reads && inflight.len() < depth {
            let i = next();
            inflight.push_back((Instant::now(), r.submit(i).expect("submit")));
            submitted += 1;
        }
        let (ts, t) = inflight.pop_front().expect("in flight");
        let v = t.wait(Duration::from_secs(30)).expect("verified read");
        payload += v.payload().len() as u64;
        lat.push(ts.elapsed().as_secs_f64() * 1e3);
    }
    let wall = t0.elapsed().as_secs_f64();
    let st = r.stats();
    lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let pct = |p: f64| lat[((lat.len() as f64 * p) as usize).min(lat.len() - 1)];
    println!(
        "{}",
        serde_json::json!({
            "tool": "ribbon_read_bench",
            "model_digest": m.model_digest,
            "direct": direct,
            "workers": workers,
            "depth": depth,
            "reads": reads,
            "seed": seed,
            "payload_bytes": payload,
            "device_bytes": st.device_bytes,
            "wall_s": wall,
            "payload_GBps": payload as f64 / wall / 1e9,
            "device_GBps": st.device_bytes as f64 / wall / 1e9,
            "read_worker_s": st.read_ns as f64 / 1e9,
            "digest_worker_s": st.digest_ns as f64 / 1e9,
            "latency_ms": {"mean": lat.iter().sum::<f64>() / lat.len() as f64, "p50": pct(0.5), "p95": pct(0.95), "max": lat[lat.len() - 1]},
        })
    );
}
