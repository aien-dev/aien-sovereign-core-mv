//! OWN-30B loader identity and host timing, no chip (W-FL2).
//!
//!   own30b_load_digest <checkpoint> [--mode digest|copy|read] [--out <file>]
//!
//! Runs the real loader (`load_qwen3_moe_omega_paths`, `WeightFormat::Fp8All`, the server's format) over the
//! checkpoint shards with a host uploader in place of the chip:
//!   * `digest` (default): every upload, in the order the loader hands it out, is recorded as one line
//!     `<seq> <kind> <experts> <n> <k> <weights sha256> <scales sha256>` (sha256 of 16 MiB chunks, hashed again,
//!     so the hashing runs on all cores). The final `LOAD_DIGEST` is the sha256 of all lines. Two loaders that
//!     print the same `LOAD_DIGEST` handed the chip the same bytes, tensor by tensor, in the same order.
//!   * `copy`: each upload's bytes are copied into fresh host memory (page faults + one write pass, standing in
//!     for the allocation zero-fill and the copy into the weight buffer); no hashing. A host timing model.
//!   * `read`: the uploader does nothing; the time is the read side only.
//!
//! `AIEN_LOAD_THREADS` is read by the loader (see `load_threads`). Prints `LOAD_SECONDS`.
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use aien_inference_abi::qwen3_moe_omega::{
    load_qwen3_moe_omega_paths, DenseMatmul, DevicePaths, ExpertMatmul, Fp8Gemv, LmHeadGemv, ModelUploader, WeightFormat,
};
use rayon::prelude::*;
use sha2::{Digest, Sha256};

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Digest,
    Copy,
    Read,
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// sha256 over the sha256 of each 16 MiB chunk (the chunks hashed in parallel).
fn chunked_sha256(bytes: &[u8]) -> String {
    let parts: Vec<[u8; 32]> = bytes.par_chunks(16 << 20).map(|c| Sha256::digest(c).into()).collect();
    let mut h = Sha256::new();
    h.update((bytes.len() as u64).to_le_bytes());
    for p in &parts {
        h.update(p);
    }
    hex(&h.finalize())
}

fn u16_bytes(v: &[u16]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

struct Record {
    seq: usize,
    line: String,
}

struct HostUploader {
    mode: Mode,
    next: Mutex<usize>,
    records: Mutex<Vec<Record>>,
}

impl HostUploader {
    fn take_seq(&self) -> usize {
        let mut n = self.next.lock().unwrap();
        *n += 1;
        *n - 1
    }
    fn record(&self, seq: usize, kind: &str, shape: (usize, usize, usize), a: &[u8], b: &[u8]) {
        let line = match self.mode {
            Mode::Digest => format!("{seq} {kind} {} {} {} {} {}", shape.0, shape.1, shape.2, chunked_sha256(a), chunked_sha256(b)),
            Mode::Copy => {
                let mut v: Vec<u8> = Vec::with_capacity(a.len());
                v.extend_from_slice(a);
                std::hint::black_box(&v);
                format!("{seq} {kind} {} {} {}", shape.0, shape.1, shape.2)
            }
            Mode::Read => format!("{seq} {kind} {} {} {}", shape.0, shape.1, shape.2),
        };
        self.records.lock().unwrap().push(Record { seq, line });
    }
}

#[derive(Debug)]
struct Nop(usize, usize, usize);

impl DenseMatmul for Nop {
    fn k(&self) -> usize {
        self.0
    }
    fn n(&self) -> usize {
        self.1
    }
    fn matmul_f32(&self, _: usize, _: &[f32], _: &mut [f32]) -> Result<u64, String> {
        Err("host digest uploader: no matmul".into())
    }
}

impl Fp8Gemv for Nop {
    fn experts(&self) -> usize {
        self.0
    }
    fn n(&self) -> usize {
        self.1
    }
    fn k(&self) -> usize {
        self.2
    }
    fn grouped_f32(&self, _: &[u32], _: &[f32], _: &mut [f32]) -> Result<u64, String> {
        Err("host digest uploader: no GEMV".into())
    }
}

impl LmHeadGemv for Nop {
    fn logits_f32(&self, _: &[f32], _: &mut [f32]) -> Result<u64, String> {
        Err("host digest uploader: no lm_head".into())
    }
}

impl ModelUploader for HostUploader {
    fn upload_dense(&self, k: usize, n: usize, bits: &[u16]) -> Result<Arc<dyn DenseMatmul>, String> {
        let seq = self.take_seq();
        self.record(seq, "dense", (1, n, k), &u16_bytes(bits), &[]);
        Ok(Arc::new(Nop(k, n, 0)))
    }
    fn upload_experts(&self, _: usize, _: usize, _: &[u16]) -> Result<Arc<dyn ExpertMatmul>, String> {
        Err("host digest uploader: bf16 experts are not the server format".into())
    }
    fn upload_fp8(&self, experts: usize, n: usize, k: usize, w: &[u8], scales: &[u16]) -> Result<Arc<dyn Fp8Gemv>, String> {
        let seq = self.take_seq();
        self.record(seq, "fp8", (experts, n, k), w, &u16_bytes(scales));
        Ok(Arc::new(Nop(experts, n, k)))
    }
    fn upload_lm_head_gemv(&self, n: usize, k: usize, bits: &[u16]) -> Result<Option<Arc<dyn LmHeadGemv>>, String> {
        let seq = self.take_seq();
        self.record(seq, "lm_head", (1, n, k), &u16_bytes(bits), &[]);
        Ok(Some(Arc::new(Nop(n, k, 0))))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut dir: Option<PathBuf> = None;
    let mut mode = Mode::Digest;
    let mut out: Option<PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--mode" => {
                mode = match it.next().map(String::as_str) {
                    Some("digest") => Mode::Digest,
                    Some("copy") => Mode::Copy,
                    Some("read") => Mode::Read,
                    m => {
                        eprintln!("--mode must be digest, copy or read (got {m:?})");
                        std::process::exit(2);
                    }
                }
            }
            "--out" => out = it.next().map(PathBuf::from),
            s if dir.is_none() && !s.starts_with("--") => dir = Some(PathBuf::from(s)),
            s => {
                eprintln!("unknown argument {s}\nusage: own30b_load_digest <checkpoint> [--mode digest|copy|read] [--out <file>]");
                std::process::exit(2);
            }
        }
    }
    let Some(dir) = dir else {
        eprintln!("usage: own30b_load_digest <checkpoint> [--mode digest|copy|read] [--out <file>]");
        std::process::exit(2);
    };
    let up = HostUploader { mode, next: Mutex::new(0), records: Mutex::new(Vec::new()) };
    let t0 = Instant::now();
    let r = load_qwen3_moe_omega_paths(&dir, &up, WeightFormat::Fp8All, DevicePaths::default(), &mut |_| {});
    let secs = t0.elapsed().as_secs_f64();
    if let Err(e) = r {
        eprintln!("LOAD_FAILED {e}");
        std::process::exit(1);
    }
    drop(r);
    let mut recs = up.records.into_inner().unwrap();
    recs.sort_by_key(|r| r.seq);
    let text: String = recs.iter().map(|r| format!("{}\n", r.line)).collect();
    if let Some(p) = out {
        std::fs::write(&p, &text).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
    }
    let threads = std::env::var("AIEN_LOAD_THREADS").unwrap_or_else(|_| "default".into());
    println!("UPLOADS {}", recs.len());
    if mode == Mode::Digest {
        println!("LOAD_DIGEST {}", hex(&Sha256::digest(text.as_bytes())));
    }
    println!("LOAD_SECONDS {secs:.2} (mode {}, AIEN_LOAD_THREADS {threads})", match mode {
        Mode::Digest => "digest",
        Mode::Copy => "copy",
        Mode::Read => "read",
    });
}
