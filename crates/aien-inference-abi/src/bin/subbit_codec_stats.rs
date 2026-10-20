//! Synthetic check of the sub-4-bit codes (ledger L134): relative error of a trellis code on unit
//! Gaussian data against the rate-distortion limit `2^(-2 K)`, with the encoder's speed.
//!
//!   subbit_codec_stats <K> <L> [train-rounds]

use std::time::Instant;

use aien_inference_abi::qwen38_subbit::Trellis;

fn main() {
    let a: Vec<usize> = std::env::args().skip(1).filter_map(|s| s.parse().ok()).collect();
    if a.len() < 2 {
        eprintln!("usage: subbit_codec_stats <K> <L> [train-rounds]");
        std::process::exit(2);
    }
    let (k, l, rounds) = (a[0], a[1], a.get(2).copied().unwrap_or(8));
    let mut tr = Trellis::gaussian_init(k, l);
    let limit = 2f64.powi(-2 * k as i32);
    let e0 = tr.gaussian_error(256, 2048, 11);
    let t = Instant::now();
    let hist = tr.train_gaussian(256, 4096, rounds, 0x7EA1);
    let train_s = t.elapsed().as_secs_f64();
    let t = Instant::now();
    let e1 = tr.gaussian_error(256, 2048, 12);
    let s = t.elapsed().as_secs_f64();
    println!(
        "K {k} L {l}: untrained {e0:.5}, trained {e1:.5}, limit {limit:.5} (ratio {:.3}, {:.2} dB over); \
         bits/weight {:.4} (T 256); encode {:.1} ns per weight over all threads; training {train_s:.1} s; rounds {:?}",
        e1 / limit,
        10.0 * (e1 / limit).log10(),
        tr.bits_per_weight(256),
        s * 1e9 / (2048.0 * 256.0),
        hist.iter().map(|v| (v * 1e5).round() / 1e5).collect::<Vec<_>>()
    );
}
