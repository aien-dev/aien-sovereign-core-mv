//! linkprobe (L258): our own link receipts between two machines, no iperf. Std only: builds with the crate or alone
//! (`rustc -O --edition 2021 src/bin/linkprobe.rs`) on a machine without the engine's dependencies.
//!
//!   linkprobe serve ADDR                                     echo pings, ack bulk, until killed
//!   linkprobe run ADDR LABEL [--sizes 4096,8192,...] [--count N] [--warm W] [--bulk BYTES] [--chunk BYTES] [--reps R]
//!
//! `run` prints one JSON document: per size the round trip (us: min, median, p90, p99, max, mean), and the bulk rate
//! (GB/s = bytes / time until the far end acked, per rep, plus the median).
#[path = "../link.rs"]
#[allow(dead_code)]
mod link;

use std::net::TcpListener;

fn opt(a: &[String], k: &str) -> Option<String> {
    a.iter().position(|x| x == k).and_then(|i| a.get(i + 1).cloned())
}

fn num(a: &[String], k: &str, d: u64) -> u64 {
    opt(a, k).and_then(|v| v.parse().ok()).unwrap_or(d)
}

fn run(a: &[String]) -> Result<(), String> {
    let addr = a.first().ok_or("run: ADDR")?;
    let label = a.get(1).ok_or("run: LABEL")?;
    let sizes: Vec<usize> = opt(a, "--sizes").unwrap_or_else(|| "4096,8192,40960,65536,1048576".into()).split(',').map(|s| s.parse().map_err(|_| format!("bad size {s}"))).collect::<Result<_, _>>()?;
    let count = num(a, "--count", 2000) as usize;
    let warm = num(a, "--warm", 100) as usize;
    let total = num(a, "--bulk", 1 << 30);
    let chunk = num(a, "--chunk", 1 << 20) as usize;
    let reps = num(a, "--reps", 5) as usize;
    let mut l = link::Link::connect(addr)?;
    let mut rows = Vec::new();
    for &size in &sizes {
        // large pings get fewer rounds so a slow link finishes
        let n = if size >= 1 << 20 { (count / 20).max(20) } else { count };
        let rt = link::ping(&mut l, size, warm.min(n), n)?;
        let s = link::stats(&rt);
        let us = |x: f64| format!("{:.3}", x / 1e3);
        rows.push(format!(
            "{{\"bytes\":{size},\"n\":{n},\"rt_us\":{{\"min\":{},\"median\":{},\"p90\":{},\"p99\":{},\"max\":{},\"mean\":{}}}}}",
            us(s[0]), us(s[1]), us(s[2]), us(s[3]), us(s[4]), us(s[5])
        ));
    }
    let mut gbps = Vec::new();
    for _ in 0..reps {
        let ns = link::bulk(&mut l, total, chunk)?;
        gbps.push(total as f64 / ns as f64);
    }
    let mut g = gbps.clone();
    g.sort_by(|x, y| x.partial_cmp(y).unwrap());
    let med = g[g.len() / 2];
    println!(
        "{{\"schema\":\"aien-linkprobe/1\",\"label\":\"{label}\",\"addr\":\"{addr}\",\"transport\":\"tcp nodelay\",\"ping\":[{}],\"bulk\":{{\"bytes\":{total},\"chunk\":{chunk},\"gbps\":[{}],\"gbps_median\":{med:.4}}}}}",
        rows.join(","),
        gbps.iter().map(|x| format!("{x:.4}")).collect::<Vec<_>>().join(",")
    );
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let r = match a.first().map(|s| s.as_str()) {
        Some("serve") => a.get(1).ok_or("serve: ADDR".to_string()).and_then(|addr| TcpListener::bind(addr).map_err(|e| format!("bind {addr}: {e}"))).and_then(link::serve),
        Some("run") => run(&a[1..]),
        _ => Err("usage: linkprobe serve ADDR | linkprobe run ADDR LABEL [--sizes a,b] [--count N] [--warm W] [--bulk BYTES] [--chunk BYTES] [--reps R]".into()),
    };
    if let Err(e) = r {
        eprintln!("linkprobe: {e}");
        std::process::exit(1);
    }
}
