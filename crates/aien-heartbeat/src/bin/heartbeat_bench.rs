//! Heartbeat H1 rhythm bench: full rotations of the expert ring from NVMe.
//!
//! heartbeat_bench --model DIR [--manifest PATH] [--slots N] [--workers W]
//!                 [--rotations R] [--hot-per-layer K] [--raw] [--buffered]
//!
//! --raw       read + discard, no digest, no publish (raw device cost)
//! --buffered  no O_DIRECT (page cache allowed; not a valid ceiling run)
//! Prints one JSON object per rotation, then a summary object.

use std::collections::HashSet;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use aien_heartbeat::{Manifest, Ring, Schedule, StreamConfig, Streamer};

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
        std::env::temp_dir().join("aien-heartbeat-manifest.json"),
    );
    let slots: usize = arg(&args, "--slots", 64);
    let workers: usize = arg(&args, "--workers", 8);
    let rotations: usize = arg(&args, "--rotations", 3);
    let hot_per_layer: u16 = arg(&args, "--hot-per-layer", 0);
    let raw = args.iter().any(|a| a == "--raw");
    let direct = !args.iter().any(|a| a == "--buffered");

    let m = match Manifest::load(&manifest_path) {
        Ok(m) if m.model_dir == model => m,
        _ => {
            let t = Instant::now();
            let m = Manifest::build(&model).unwrap_or_else(|e| panic!("manifest: {e}"));
            m.save(&manifest_path)
                .unwrap_or_else(|e| panic!("save manifest: {e}"));
            eprintln!(
                "built manifest: {} experts, {} bytes, {:.1}s -> {}",
                m.experts.len(),
                m.total_bytes(),
                t.elapsed().as_secs_f64(),
                manifest_path.display()
            );
            m
        }
    };
    let hot: HashSet<(u16, u16)> = (0..m.num_layers)
        .flat_map(|l| (0..hot_per_layer).map(move |e| (l, e)))
        .collect();
    let sched = Schedule::layer_order(&m, &hot);
    let ring = Ring::new(slots, Streamer::slot_bytes(&m));
    let cfg = StreamConfig {
        slots,
        workers,
        direct,
        publish: !raw,
        lease_wait: Duration::from_secs(60),
    };
    let s = Streamer::open(&m.shard_paths(), cfg).unwrap_or_else(|e| panic!("open: {e}"));

    let mut walls = Vec::new();
    let (mut pay, mut dev, mut dig, mut rd) = (0u64, 0u64, Duration::ZERO, Duration::ZERO);
    for r in 0..rotations {
        let st = s
            .rotate(&ring, &m, &sched)
            .unwrap_or_else(|e| panic!("rotation {r}: {e}"));
        println!(
            "{}",
            serde_json::json!({
                "rotation": r, "experts": st.experts, "payload_bytes": st.payload_bytes,
                "device_bytes": st.device_bytes, "wall_s": st.wall.as_secs_f64(),
                "payload_gbps": st.payload_gbps(), "device_gbps": st.device_gbps(),
                "read_worker_s": st.read_time.as_secs_f64(),
                "digest_worker_s": st.digest_time.as_secs_f64(),
                "lease_wait_s": st.lease_wait_time.as_secs_f64(),
            })
        );
        walls.push(st.wall.as_secs_f64());
        pay += st.payload_bytes;
        dev += st.device_bytes;
        dig += st.digest_time;
        rd += st.read_time;
    }
    let total: f64 = walls.iter().sum();
    let min = walls.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = walls.iter().cloned().fold(0.0, f64::max);
    println!(
        "{}",
        serde_json::json!({
            "summary": true, "mode": if raw { "raw" } else { "verify+publish" }, "direct": direct,
            "slots": slots, "workers": workers, "rotations": rotations,
            "hot_per_layer": hot_per_layer, "experts_per_rotation": sched.items.len(),
            "payload_bytes_total": pay, "device_bytes_total": dev, "wall_s_total": total,
            "beat_s_min": min, "beat_s_max": max, "beat_s_mean": total / rotations as f64,
            "payload_gbps": pay as f64 / total / 1e9, "device_gbps": dev as f64 / total / 1e9,
            "read_worker_s_total": rd.as_secs_f64(), "digest_worker_s_total": dig.as_secs_f64(),
        })
    );
}
