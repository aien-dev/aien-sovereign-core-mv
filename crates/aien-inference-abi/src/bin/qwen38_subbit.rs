//! Sub-4-bit drift study for Qwen3.8-27B on the host reference (ledger L134, SPEED-10). CPU only.
//!
//!   qwen38_subbit <checkpoint-dir> <seq-dir> <out-dir> <variant> [--ref-dir D] [--texts a,b] [--max N]
//!                 [--head keep|mxfp4] [--codebooks D]
//!
//! `variant` is `ref` (FP8 weights, bf16 head: writes `<text>.logits` into out-dir), or a code from
//! `qwen38_subbit.rs` (`mxfp4`, `rht-scalar:<bits>:<group>`, `trellis:<K>:<L>:<group>[:<T>]`),
//! which needs `--ref-dir` and writes `<text>.json` plus `summary.json`. All texts run through one
//! read of the checkpoint, teacher forced: the same token ids go through the reference and the
//! coded model, so only the weights differ. Per position: does the coded model pick the same next
//! word as the reference, the reference's top-two margin where it does not, KL(reference || coded),
//! and both models' negative log-likelihood of the text's actual next token.

use std::collections::BTreeMap;
use std::io::Write;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use aien_inference_abi::qwen38::{Matrix, Qwen38Reference};
use aien_inference_abi::qwen38_subbit::{relative_error, ActTable, Variant};
use serde_json::{json, Value};

const TEXTS: [&str; 6] = ["golden", "license", "prose", "codelong", "code", "arch"];
const CHUNK: usize = 16;

fn mem_available_gib() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find_map(|l| l.strip_prefix("MemAvailable:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        })
        .unwrap_or(0.0)
        / (1 << 20) as f64
}

fn read_ids(p: &Path) -> Result<Vec<u32>, String> {
    let v: Value = serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{}: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))?;
    v["ids"]
        .as_array()
        .ok_or_else(|| format!("{}: no ids", p.display()))?
        .iter()
        .map(|x| x.as_u64().map(|n| n as u32).ok_or_else(|| "id not a number".to_string()))
        .collect()
}

fn log_softmax(row: &[f32]) -> (f64, Vec<f64>) {
    let max = row.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let lse = row.iter().map(|&v| (v as f64 - max).exp()).sum::<f64>().ln() + max;
    (lse, row.iter().map(|&v| v as f64 - lse).collect())
}

fn top_two(row: &[f32]) -> (usize, f32) {
    let (mut a, mut av, mut bv) = (0usize, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for (i, &v) in row.iter().enumerate() {
        if v > av {
            bv = av;
            av = v;
            a = i;
        } else if v > bv {
            bv = v;
        }
    }
    (a, av - bv)
}

fn bytes_of(v: &[f32]) -> &[u8] {
    // SAFETY: f32 has no padding and any byte view of it is valid; the length is the exact size.
    unsafe { std::slice::from_raw_parts(v.as_ptr().cast::<u8>(), std::mem::size_of_val(v)) }
}

fn read_row_chunk(f: &std::fs::File, row0: usize, rows: usize, vocab: usize) -> Result<Vec<f32>, String> {
    let mut buf = vec![0f32; rows * vocab];
    // SAFETY: the byte view covers exactly the f32 buffer; every byte pattern is a valid f32.
    let bytes = unsafe { std::slice::from_raw_parts_mut(buf.as_mut_ptr().cast::<u8>(), buf.len() * 4) };
    f.read_exact_at(bytes, (row0 * vocab * 4) as u64)
        .map_err(|e| format!("reference logits: {e}"))?;
    Ok(buf)
}

fn run() -> Result<(), String> {
    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("calib") {
        return calibrate(&args);
    }
    if args.len() < 5 {
        return Err("usage: qwen38_subbit <ckpt> <seq-dir> <out-dir> <variant> [--ref-dir D] [--texts a,b] [--max N] [--head keep|mxfp4] [--codebooks D]".into());
    }
    let (dir, seq_dir, out_dir, variant) = (Path::new(&args[1]), Path::new(&args[2]), PathBuf::from(&args[3]), args[4].clone());
    let opt = |name: &str| -> Option<String> {
        args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).cloned()
    };
    let ref_dir = opt("--ref-dir").map(PathBuf::from);
    let texts: Vec<String> = match opt("--texts") {
        Some(s) => s.split(',').map(str::to_string).collect(),
        None => TEXTS.iter().map(|s| s.to_string()).collect(),
    };
    let max: usize = opt("--max").map(|s| s.parse().map_err(|_| "--max".to_string())).transpose()?.unwrap_or(usize::MAX);
    let head = opt("--head").unwrap_or_else(|| "keep".into());
    let is_ref = variant == "ref";
    if !is_ref && ref_dir.is_none() {
        return Err("a coded variant needs --ref-dir".into());
    }
    if mem_available_gib() < 16.0 {
        return Err(format!("memory guard: MemAvailable {:.1} GiB is under 16", mem_available_gib()));
    }
    std::fs::create_dir_all(&out_dir).map_err(|e| e.to_string())?;

    let mut idss = Vec::new();
    for t in &texts {
        let mut ids = read_ids(&seq_dir.join(format!("{t}.json")))?;
        ids.truncate(max);
        if ids.len() < 2 {
            return Err(format!("{t}: fewer than 2 ids"));
        }
        idss.push(ids);
    }

    let t0 = Instant::now();
    let mut model = Qwen38Reference::open(dir)?;
    let vocab = model.config.vocab;
    let hidden = model.config.hidden;
    eprintln!("opened in {:.1} s; texts {texts:?}, tokens {:?}", t0.elapsed().as_secs_f64(), idss.iter().map(Vec::len).collect::<Vec<_>>());

    // Weight error per projection role, shared with the transform closure.
    let stats: std::sync::Arc<Mutex<BTreeMap<String, (f64, usize)>>> = Default::default();
    let mut bits = f64::NAN;
    if !is_ref {
        let v = Variant::parse(&variant, opt("--codebooks").map(PathBuf::from).as_deref())?;
        bits = v.bits_per_weight(5120);
        eprintln!("variant {variant}: about {bits:.3} bits per weight (k 5120, scales as bf16)");
        let act = match opt("--act") {
            Some(p) => {
                let e: f32 = opt("--act-exp").map(|s| s.parse().map_err(|_| "--act-exp".to_string())).transpose()?.unwrap_or(0.25);
                eprintln!("activation-weighted column scaling from {p}, exponent {e}");
                Some(ActTable::read(Path::new(&p), e)?)
            }
            None => None,
        };
        let (st, vv) = (stats.clone(), v.clone());
        model.set_weight_transform(Box::new(move |name: &str, m: Matrix| {
            if !matches!(m, Matrix::Fp8 { .. }) {
                return m;
            }
            let (n, k) = (m.n(), m.k());
            let w = m.to_f32();
            let cs = act.as_ref().and_then(|a| a.colscale(name, k));
            let rebuilt = vv.apply(&w, n, k, cs.as_deref());
            let e = relative_error(&w, &rebuilt);
            let role = name.rsplit_once('.').map(|(a, _)| a).unwrap_or(name);
            let role = role.rsplit('.').next().unwrap_or(role).to_string();
            let mut s = st.lock().unwrap();
            let ent = s.entry(role).or_insert((0.0, 0));
            ent.0 += e;
            ent.1 += 1;
            Matrix::F32 { n, k, w: rebuilt }
        }));
        if head == "mxfp4" {
            let mx = Variant::Mxfp4;
            model.transform_head(&mut |_, m| {
                let (n, k) = (m.n(), m.k());
                let w = m.to_f32();
                Matrix::F32 { n, k, w: mx.apply(&w, n, k, None) }
            });
        }
    }

    let mut seqs: Vec<_> = idss.iter().map(|ids| (model.new_state(), ids.clone())).collect();
    let t1 = Instant::now();
    let hid = model.forward_hidden_multi(&mut seqs)?;
    eprintln!("body done in {:.1} s", t1.elapsed().as_secs_f64());

    let mut summary = Vec::new();
    for (ti, name) in texts.iter().enumerate() {
        let ids = &idss[ti];
        let n = ids.len();
        let h = &hid[ti];
        let ref_file = if is_ref {
            None
        } else {
            let p = ref_dir.as_ref().unwrap().join(format!("{name}.logits"));
            Some(std::fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?)
        };
        let mut out_file = if is_ref {
            Some(std::fs::File::create(out_dir.join(format!("{name}.logits"))).map_err(|e| e.to_string())?)
        } else {
            None
        };
        let (mut agree, mut kl_sum, mut nll_a, mut nll_b, mut scored) = (0usize, 0f64, 0f64, 0f64, 0usize);
        let (mut max_flip_margin, mut flips) = (0f32, 0usize);
        let mut kl_max = 0f64;
        let mut windows: Vec<Value> = Vec::new();
        let (mut w_agree, mut w_kl, mut w_nll_a, mut w_nll_b, mut w_n, mut w_start) = (0usize, 0f64, 0f64, 0f64, 0usize, 0usize);
        let mut pos = 0;
        while pos < n {
            let rows = CHUNK.min(n - pos);
            let logits = model.head().matmul(&h[pos * hidden..(pos + rows) * hidden], rows);
            if let Some(f) = out_file.as_mut() {
                f.write_all(bytes_of(&logits)).map_err(|e| e.to_string())?;
            }
            if let Some(rf) = ref_file.as_ref() {
                let reference = read_row_chunk(rf, pos, rows, vocab)?;
                for r in 0..rows {
                    let (a_row, b_row) = (&reference[r * vocab..(r + 1) * vocab], &logits[r * vocab..(r + 1) * vocab]);
                    let ((ia, margin), (ib, _)) = (top_two(a_row), (top_two(b_row).0, 0f32));
                    let ((_, la), (_, lb)) = (log_softmax(a_row), log_softmax(b_row));
                    let kl: f64 = la.iter().zip(&lb).map(|(&x, &y)| x.exp() * (x - y)).sum();
                    kl_sum += kl;
                    kl_max = kl_max.max(kl);
                    if ia == ib {
                        agree += 1;
                    } else {
                        flips += 1;
                        max_flip_margin = max_flip_margin.max(margin);
                    }
                    let i = pos + r;
                    if i + 1 < n {
                        let tgt = ids[i + 1] as usize;
                        nll_a -= la[tgt];
                        nll_b -= lb[tgt];
                        scored += 1;
                    }
                    w_agree += (ia == ib) as usize;
                    w_kl += kl;
                    w_n += 1;
                    if i + 1 < n {
                        w_nll_a -= la[ids[i + 1] as usize];
                        w_nll_b -= lb[ids[i + 1] as usize];
                    }
                    if w_n == 128 || i + 1 == n {
                        windows.push(json!({"from": w_start, "n": w_n, "agree": w_agree, "kl_mean": w_kl / w_n as f64, "nll_ref_sum": w_nll_a, "nll_coded_sum": w_nll_b}));
                        (w_agree, w_kl, w_nll_a, w_nll_b, w_n, w_start) = (0, 0.0, 0.0, 0.0, 0, i + 1);
                    }
                }
            } else {
                for r in 0..rows {
                    let i = pos + r;
                    if i + 1 < n {
                        let (_, la) = log_softmax(&logits[r * vocab..(r + 1) * vocab]);
                        nll_a -= la[ids[i + 1] as usize];
                        scored += 1;
                    }
                }
            }
            pos += rows;
        }
        let entry = json!({
            "text": name, "positions": n, "agree": agree, "flips": flips,
            "agree_pct": if is_ref { Value::Null } else { json!(100.0 * agree as f64 / n as f64) },
            "kl_mean": kl_sum / n as f64, "kl_max": kl_max,
            "largest_ref_margin_at_a_flip": max_flip_margin,
            "nll_ref_mean": nll_a / scored.max(1) as f64, "nll_coded_mean": nll_b / scored.max(1) as f64,
            "windows_of_128": windows,
        });
        eprintln!("{name}: {entry}");
        std::fs::write(out_dir.join(format!("{name}.json")), serde_json::to_string_pretty(&entry).unwrap())
            .map_err(|e| e.to_string())?;
        summary.push(entry);
    }
    let weight_err: BTreeMap<String, f64> = stats.lock().unwrap().iter().map(|(k, (s, c))| (k.clone(), s / *c as f64)).collect();
    let all = json!({
        "variant": variant, "head": head, "bits_per_weight_k5120": bits,
        "weight_relative_error_energy_by_role_mean_over_layers": weight_err,
        "texts": summary, "body_seconds": t1.elapsed().as_secs_f64(),
    });
    std::fs::write(out_dir.join("summary.json"), serde_json::to_string_pretty(&all).unwrap()).map_err(|e| e.to_string())?;
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("qwen38_subbit: {e}");
        std::process::exit(1);
    }
}

/// `qwen38_subbit calib <ckpt> <seq-dir> <out.act> <name,name,...>`: runs the FP8 model over the
/// named token files (held-out text, not the six test texts) and writes the per-channel mean
/// squares of every matmul input.
fn calibrate(args: &[String]) -> Result<(), String> {
    if args.len() != 6 {
        return Err("usage: qwen38_subbit calib <ckpt> <seq-dir> <out.act> <name,name,...>".into());
    }
    let (dir, seq_dir, out) = (Path::new(&args[2]), Path::new(&args[3]), Path::new(&args[4]));
    if mem_available_gib() < 16.0 {
        return Err(format!("memory guard: MemAvailable {:.1} GiB is under 16", mem_available_gib()));
    }
    let mut model = Qwen38Reference::open(dir)?;
    model.enable_calibration();
    let mut seqs = Vec::new();
    for name in args[5].split(',') {
        let ids = read_ids(&seq_dir.join(format!("{name}.json")))?;
        eprintln!("calibration text {name}: {} tokens", ids.len());
        seqs.push((model.new_state(), ids));
    }
    model.forward_hidden_multi(&mut seqs)?;
    let cb = model.take_calibration().ok_or("no calibration collected")?;
    let table = ActTable::from_means(cb.rows, [&cb.xn, &cb.mix, &cb.mlp_in, &cb.act]);
    table.write(out)?;
    eprintln!("wrote {} ({} rows)", out.display(), cb.rows);
    Ok(())
}
