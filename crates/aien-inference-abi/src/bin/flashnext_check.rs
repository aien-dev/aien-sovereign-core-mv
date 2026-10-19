//! flashnext_check: host checks of the Qwen3.8-Flash-Next reference on tensors fetched by HTTP range
//! (tools/flashnext/hf-range-fetch.sh), before the checkpoint can be downloaded.
//!
//!   flashnext_check formats BF16_DIR FP8_DIR NVFP4_DIR OUT.json LAYER:EXPERT...
//!     Per expert: weight error of every stored or convertible 4-bit / 8-bit form against the bf16 checkpoint
//!     (relative RMS over the three matrices) and the expert MLP output error on fixed inputs.
//!   flashnext_check nv4-fixture NVFP4_DIR OUT_DIR LAYER:EXPERT...
//!     Real NVFP4 experts as the chip's NV4 blocks (gate|up fused, down) plus the host reference's f32 outputs on fixed
//!     inputs, for omega tests/gpu_nv4_api_test.c --fixture. Derived weights: never commit OUT_DIR, only its SHA256SUMS.
//!   flashnext_check tokenize TOKENIZER_JSON TEXT
//!   flashnext_check golden DIR OUT_DIR LAYERS GEN ID...
//!     M1 golden on the real checkpoint DIR (first LAYERS layers, 48 = whole model): teacher-forced pass over the prompt IDs
//!     (top-5 and log-probability of the next given token at every position), the same tokens again one at a time
//!     (state handling check), then GEN greedy tokens. Writes OUT_DIR/golden.json, logits_last.f32, routing in golden.json.
//!   flashnext_check run CONFIG_JSON DIR LAYERS stored|mxfp4 OUT_DIR ID...
//!     The first LAYERS layers on DIR's tensors. Exit 3 with "MISSING ..." lines when experts or n-gram rows are
//!     not fetched yet (the driver fetches them and runs again). On success writes OUT_DIR/run.json (routing,
//!     indexer selections, lane norms) and OUT_DIR/moe/L<l>/ fixtures for the chip's MoE parity test.
use std::io::Write;
use std::path::{Path, PathBuf};

use aien_inference_abi::flashnext::*;
use aien_inference_abi::qwen38::Matrix;
use aien_inference_abi::qwen3_moe_omega::ShardSource;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("formats") if a.len() >= 7 => formats(&a[2], &a[3], &a[4], &a[5], &a[6..]),
        Some("tokenize") if a.len() == 4 => tokenize(&a[2], &a[3]),
        Some("run") if a.len() >= 8 => run(&a[2], &a[3], &a[4], &a[5], &a[6], &a[7..]),
        Some("golden") if a.len() >= 8 => golden(&a[2], &a[3], &a[4], &a[5], &a[6..]),
        Some("nv4-fixture") if a.len() >= 5 => nv4_fixture(&a[2], &a[3], &a[4..]),
        _ => Err("usage: see the file header".into()),
    };
    match r {
        Ok(()) => {}
        Err(e) if e.starts_with("MISSING") => {
            println!("{e}");
            std::process::exit(3);
        }
        Err(e) => {
            eprintln!("FAIL: {e}");
            std::process::exit(1);
        }
    }
}

fn prand(seed: u64, idx: usize) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64 + 1);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x & 0xFFFF) as f32 / 32768.0) - 1.0
}

fn rel(a: &[f32], b: &[f32]) -> (f64, f64) {
    let (mut e, mut r, mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64, 0f64, 0f64);
    for (&x, &y) in a.iter().zip(b) {
        let (x, y) = (x as f64, y as f64);
        e += (x - y).powi(2);
        r += y * y;
        ab += x * y;
        aa += x * x;
        bb += y * y;
    }
    ((e / r).sqrt(), ab / (aa.sqrt() * bb.sqrt()))
}

fn formats(bf: &str, f8: &str, nv: &str, out: &str, items: &[String]) -> Result<(), String> {
    let cfg = FlashNextConfig::from_json(include_str!("../../fixtures/flashnext/config.json"))?;
    let (mut sb, mut s8, mut sn) = (
        ShardSource::open(Path::new(bf))?,
        ShardSource::open(Path::new(f8))?,
        ShardSource::open(Path::new(nv))?,
    );
    let mut rows = Vec::new();
    for it in items {
        let (l, e) = it.split_once(':').ok_or("LAYER:EXPERT")?;
        let (l, e): (usize, usize) = (
            l.parse().map_err(|_| "layer")?,
            e.parse().map_err(|_| "expert")?,
        );
        let mlp = format!("model.language_model.layers.{l}.mlp");
        let mut st = QuantStats::default();
        let ref_ = read_expert(&mut sb, &mlp, e, &cfg, ExpertFormat::AsStored, &mut st)?;
        let mut forms: Vec<(&str, Expert, QuantStats)> = Vec::new();
        let mut q = QuantStats::default();
        forms.push((
            "fp8_block128",
            read_expert(&mut s8, &mlp, e, &cfg, ExpertFormat::AsStored, &mut q)?,
            q,
        ));
        let mut q = QuantStats::default();
        forms.push((
            "nvfp4",
            read_expert(&mut sn, &mlp, e, &cfg, ExpertFormat::AsStored, &mut q)?,
            q,
        ));
        let mut q = QuantStats::default();
        forms.push((
            "mxfp4_from_bf16",
            read_expert(&mut sb, &mlp, e, &cfg, ExpertFormat::Mxfp4, &mut q)?,
            q,
        ));
        let mut q = QuantStats::default();
        forms.push((
            "mxfp4_from_fp8",
            read_expert(&mut s8, &mlp, e, &cfg, ExpertFormat::Mxfp4, &mut q)?,
            q,
        ));
        let mut q = QuantStats::default();
        forms.push((
            "mxfp4_from_nvfp4",
            read_expert(&mut sn, &mlp, e, &cfg, ExpertFormat::Mxfp4, &mut q)?,
            q,
        ));
        let mut q = QuantStats::default();
        let mse = |p: &Proj, q: &mut QuantStats| mxfp4_from_f32_mse(p.n(), p.k(), &p.dense(), q);
        let ex = Expert {
            gate: mse(&ref_.gate, &mut q),
            up: mse(&ref_.up, &mut q),
            down: mse(&ref_.down, &mut q),
        };
        forms.push(("mxfp4_mse_from_bf16", ex, q));
        let rw: Vec<Vec<f32>> = [&ref_.gate, &ref_.up, &ref_.down]
            .iter()
            .map(|p| p.dense())
            .collect();
        // inputs: unit-RMS pseudo-random rows (no real activations without the earlier layers)
        let t = 8;
        let x: Vec<f32> = (0..t * cfg.hidden)
            .map(|i| prand(1000 + i as u64 / cfg.hidden as u64, i) * 1.732)
            .collect();
        let yref = ref_.forward(&x, t);
        let mut fr = serde_json::Map::new();
        for (name, ex, _) in &forms {
            let ws: Vec<Vec<f32>> = [&ex.gate, &ex.up, &ex.down]
                .iter()
                .map(|p| p.dense())
                .collect();
            let (mut e2, mut r2) = (0f64, 0f64);
            for (a, b) in ws.iter().zip(&rw) {
                for (&x, &y) in a.iter().zip(b) {
                    e2 += ((x - y) as f64).powi(2);
                    r2 += (y as f64).powi(2);
                }
            }
            let y = ex.forward(&x, t);
            let (oe, cos) = rel(&y, &yref);
            let bytes: usize = [&ex.gate, &ex.up, &ex.down]
                .iter()
                .map(|p| p.stored_bytes())
                .sum();
            fr.insert(
                name.to_string(),
                serde_json::json!({"weight_rel_rms": (e2 / r2).sqrt(), "out_rel_rms": oe, "out_cosine": cos, "bytes": bytes}),
            );
        }
        rows.push(serde_json::json!({"layer": l, "expert": e, "forms": fr}));
        println!(
            "L{l} E{e}: {}",
            serde_json::to_string(&rows.last().unwrap()["forms"]).unwrap()
        );
    }
    let doc = serde_json::json!({"what": "expert weight formats vs the bf16 checkpoint", "inputs": "8 pseudo-random rows, unit RMS", "rows": rows});
    std::fs::write(out, serde_json::to_string_pretty(&doc).unwrap()).map_err(|e| e.to_string())
}

fn tokenize(tok: &str, text: &str) -> Result<(), String> {
    let t = tokenizers::Tokenizer::from_file(tok).map_err(|e| e.to_string())?;
    let enc = t.encode(text, false).map_err(|e| e.to_string())?;
    let ids: Vec<String> = enc.get_ids().iter().map(|i| i.to_string()).collect();
    println!("{}", ids.join(" "));
    Ok(())
}

fn write_f32(p: &Path, v: &[f32]) -> Result<(), String> {
    let b: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}
fn write_bf16(p: &Path, m: &Matrix) -> Result<(), String> {
    match m {
        Matrix::Bf16 { w, .. } => {
            let b: Vec<u8> = w.iter().flat_map(|x| x.to_le_bytes()).collect();
            std::fs::write(p, b).map_err(|e| e.to_string())
        }
        Matrix::Fp8 { .. } => Err("expected bf16".into()),
    }
}
fn write_mx(dir: &Path, stem: &str, p: &Proj) -> Result<(), String> {
    let Proj::Mxfp4 { nib, e8, .. } = p else {
        return Err("expected mxfp4".into());
    };
    std::fs::write(dir.join(format!("{stem}.nib")), nib).map_err(|e| e.to_string())?;
    std::fs::write(dir.join(format!("{stem}.e8")), e8).map_err(|e| e.to_string())
}

/// Concatenates gate | up rows into one [2m][k] MXFP4 tensor (the chip's fused gate+up layout).
fn fuse(a: &Proj, b: &Proj) -> Result<Proj, String> {
    match (a, b) {
        (
            Proj::Mxfp4 { n, k, nib, e8 },
            Proj::Mxfp4 {
                n: n2,
                k: k2,
                nib: nb,
                e8: eb,
            },
        ) if n == n2 && k == k2 => Ok(Proj::Mxfp4 {
            n: 2 * n,
            k: *k,
            nib: [nib.clone(), nb.clone()].concat(),
            e8: [e8.clone(), eb.clone()].concat(),
        }),
        _ => Err("fuse: shapes".into()),
    }
}

fn run(
    cfg_path: &str,
    dir: &str,
    layers: &str,
    fmt: &str,
    out: &str,
    ids: &[String],
) -> Result<(), String> {
    let full =
        FlashNextConfig::from_json(&std::fs::read_to_string(cfg_path).map_err(|e| e.to_string())?)?;
    let n: usize = layers.parse().map_err(|_| "layers")?;
    let c = full.prefix(n).ok_or("layers")?;
    let tokens: Vec<u32> = ids
        .iter()
        .map(|s| s.parse().map_err(|_| "id"))
        .collect::<Result<_, _>>()?;
    let src = ShardSource::open(Path::new(dir))?;
    let mut r = FlashNextReference::new(c.clone(), src);
    r.with_head = false;
    r.expert_format = match fmt {
        "stored" => ExpertFormat::AsStored,
        "mxfp4" => ExpertFormat::Mxfp4,
        o => return Err(format!("format {o}")),
    };
    let mut st = r.new_state();
    let mut taps = Vec::new();
    let t0 = std::time::Instant::now();
    let res = r.forward(&mut st, &tokens, Some(&mut taps));
    let (mixed, _) = match res {
        Ok(v) => v,
        Err(e) if e.starts_with("missing experts") || e.starts_with("missing ngram") => {
            return Err(format!("MISSING {e}"))
        }
        Err(e) => return Err(e),
    };
    let wall = t0.elapsed().as_secs_f64();
    let outp = PathBuf::from(out);
    std::fs::create_dir_all(&outp).map_err(|e| e.to_string())?;
    let l = c.lanes();
    let lane_rms: Vec<Vec<f64>> = taps
        .iter()
        .map(|tp| {
            tp.chunks(l)
                .map(|row| (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / l as f64).sqrt())
                .collect()
        })
        .collect();
    let finite =
        mixed.iter().all(|v| v.is_finite()) && taps.iter().flatten().all(|v| v.is_finite());
    let traces: Vec<serde_json::Value> = r
        .traces
        .iter()
        .map(|t| serde_json::json!({"ids": t.ids, "weights": t.weights, "shared_gate": t.shared_gate}))
        .collect();
    let doc = serde_json::json!({
        "tokens": tokens, "layers": n, "expert_format": fmt, "finite": finite, "wall_s": wall,
        "lane_rms_entering_layer": lane_rms, "moe": traces,
        "indexer_selected_counts": r.selections.iter().map(|s| s.iter().map(|x| x.len()).collect::<Vec<_>>()).collect::<Vec<_>>(),
        "quant": {"weights": r.quant.weights, "rel_rms": if r.quant.sq_ref > 0.0 { (r.quant.sq_err / r.quant.sq_ref).sqrt() } else { 0.0 }},
    });
    std::fs::write(
        outp.join("run.json"),
        serde_json::to_string_pretty(&doc).unwrap(),
    )
    .map_err(|e| e.to_string())?;
    write_f32(&outp.join("mixer_out.f32"), &mixed)?;
    for (i, tp) in taps.iter().enumerate() {
        write_f32(&outp.join(format!("lanes_in_L{i}.f32")), tp)?;
    }
    // MoE fixtures (mxfp4 runs): router input, routing, outputs, and the weights in the chip's layout.
    if r.expert_format == ExpertFormat::Mxfp4 {
        moe_fixtures(&mut r, &c, &outp)?;
    }
    println!(
        "PASS run: {n} layers, {} tokens, finite {finite}, {wall:.1} s",
        tokens.len()
    );
    Ok(())
}

fn moe_fixtures(
    r: &mut FlashNextReference<ShardSource>,
    c: &FlashNextConfig,
    outp: &Path,
) -> Result<(), String> {
    let prefix = r.prefix.clone();
    for layer in 0..c.layers {
        let d = outp.join(format!("moe/L{layer}"));
        std::fs::create_dir_all(d.join("experts")).map_err(|e| e.to_string())?;
        let p = format!("{prefix}layers.{layer}");
        let trace = r.traces[layer].clone();
        let t = trace.ids.len();
        write_f32(&d.join("x.f32"), &r.moe_inputs[layer])?;
        write_f32(&d.join("y.f32"), &r.moe_outputs[layer])?;
        let mw = r.read_moe(&p)?;
        write_bf16(&d.join("router.bf16"), &mw.router)?;
        write_bf16(&d.join("shared_gate.bf16"), &mw.shared_gate)?;
        write_mx(&d, "shared_gu", &fuse(&mw.shared.gate, &mw.shared.up)?)?;
        write_mx(&d, "shared_dn", &mw.shared.down)?;
        let mut need: Vec<usize> = trace.ids.iter().flatten().copied().collect();
        need.sort_unstable();
        need.dedup();
        for &e in &need {
            let ex = r.expert(layer, e).ok_or("expert not loaded")?;
            write_mx(
                &d.join("experts"),
                &format!("{e}.gu"),
                &fuse(&ex.gate, &ex.up)?,
            )?;
            write_mx(&d.join("experts"), &format!("{e}.dn"), &ex.down)?;
        }
        let mut f = std::fs::File::create(d.join("routing.txt")).map_err(|e| e.to_string())?;
        writeln!(
            f,
            "# tokens {t} hidden {} experts {} top_k {}: ids | weights | sigmoid(shared gate)",
            c.hidden, c.experts, c.top_k
        )
        .map_err(|e| e.to_string())?;
        for i in 0..t {
            let ids: Vec<String> = trace.ids[i].iter().map(|v| v.to_string()).collect();
            let ws: Vec<String> = trace.weights[i]
                .iter()
                .map(|v| format!("{v:.9e}"))
                .collect();
            writeln!(
                f,
                "{} | {} | {:.9e}",
                ids.join(" "),
                ws.join(" "),
                trace.shared_gate[i]
            )
            .map_err(|e| e.to_string())?;
        }
    }
    Ok(())
}

fn nv4_parts(p: &Proj) -> Result<(&[u8], &[u8], f32), String> {
    match p {
        Proj::Nvfp4 {
            packed,
            scales,
            global,
            ..
        } => Ok((packed, scales, *global)),
        _ => Err("expected an NVFP4 projection".into()),
    }
}

fn nv4_fixture(dir: &str, out: &str, items: &[String]) -> Result<(), String> {
    let c = FlashNextConfig::from_json(include_str!("../../fixtures/flashnext/config.json"))?;
    let (h, m) = (c.hidden, c.moe_inter);
    let mut src = ShardSource::open(Path::new(dir))?;
    let outp = PathBuf::from(out);
    std::fs::create_dir_all(&outp).map_err(|e| e.to_string())?;
    let g = items.len();
    let x: Vec<f32> = (0..g * h)
        .map(|i| prand(2000 + (i / h) as u64, i) * 1.732)
        .collect();
    let (mut act, mut gu_ref, mut dn_ref) = (Vec::new(), Vec::new(), Vec::new());
    let mut cases = String::from("# case index per line; source layer:expert in the same order: ");
    cases.push_str(&items.join(" "));
    cases.push('\n');
    for (i, it) in items.iter().enumerate() {
        let (l, e) = it.split_once(':').ok_or("LAYER:EXPERT")?;
        let mlp = format!("model.language_model.layers.{l}.mlp");
        let mut q = QuantStats::default();
        let ex = read_expert(
            &mut src,
            &mlp,
            e.parse().map_err(|_| "expert")?,
            &c,
            ExpertFormat::AsStored,
            &mut q,
        )?;
        let xi = &x[i * h..(i + 1) * h];
        let gt = ex.gate.matmul(xi, 1);
        let up = ex.up.matmul(xi, 1);
        // act on fixed values (not silu(gate) * up, so down is tested on a well-scaled input of its own)
        let a: Vec<f32> = (0..m).map(|j| prand(3000 + i as u64, j) * 1.732).collect();
        let d = ex.down.matmul(&a, 1);
        gu_ref.extend_from_slice(&gt);
        gu_ref.extend_from_slice(&up);
        act.extend_from_slice(&a);
        dn_ref.extend_from_slice(&d);
        let (gn, gs, gg) = nv4_parts(&ex.gate)?;
        let (un, us, ug) = nv4_parts(&ex.up)?;
        let (dn, ds, dg) = nv4_parts(&ex.down)?;
        let w =
            |name: String, b: &[u8]| std::fs::write(outp.join(name), b).map_err(|e| e.to_string());
        w(format!("{i}.gu.nib"), &[gn, un].concat())?;
        w(format!("{i}.gu.sc"), &[gs, us].concat())?;
        let gl: Vec<f32> = (0..2 * m).map(|r| if r < m { gg } else { ug }).collect();
        write_f32(&outp.join(format!("{i}.gu.g")), &gl)?;
        w(format!("{i}.dn.nib"), dn)?;
        w(format!("{i}.dn.sc"), ds)?;
        write_f32(&outp.join(format!("{i}.dn.g")), &vec![dg; h])?;
        cases.push_str(&format!("{i}\n"));
    }
    write_f32(&outp.join("x.f32"), &x)?;
    write_f32(&outp.join("act.f32"), &act)?;
    write_f32(&outp.join("gu_ref.f32"), &gu_ref)?;
    write_f32(&outp.join("dn_ref.f32"), &dn_ref)?;
    std::fs::write(outp.join("cases.txt"), cases).map_err(|e| e.to_string())?;
    println!("PASS nv4-fixture: {g} experts -> {out}");
    Ok(())
}

fn argmax(v: &[f32]) -> u32 {
    let mut b = 0usize;
    for (i, &x) in v.iter().enumerate() {
        if x > v[b] {
            b = i;
        }
    }
    b as u32
}

fn log_softmax_at(v: &[f32], i: usize) -> f64 {
    let m = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max) as f64;
    let z: f64 = v.iter().map(|&x| ((x as f64) - m).exp()).sum();
    (v[i] as f64) - m - z.ln()
}

fn loadavg() -> String {
    std::fs::read_to_string("/proc/loadavg")
        .unwrap_or_default()
        .trim()
        .to_string()
}

fn golden(dir: &str, out: &str, layers: &str, gen: &str, ids: &[String]) -> Result<(), String> {
    let n: usize = layers.parse().map_err(|_| "layers")?;
    let gen: usize = gen.parse().map_err(|_| "gen")?;
    let tokens: Vec<u32> = ids
        .iter()
        .map(|s| s.parse().map_err(|_| "id"))
        .collect::<Result<_, _>>()?;
    if tokens.len() < 2 {
        return Err("golden needs at least 2 prompt ids".into());
    }
    let dirp = Path::new(dir);
    let tk = tokenizers::Tokenizer::from_file(dirp.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let text = |id: u32| tk.decode(&[id], false).unwrap_or_default();
    let mut r = FlashNextReference::open(dirp, if n == 48 { None } else { Some(n) })?;
    r.evict_experts = true;
    let c = r.config.clone();
    let v = c.vocab;
    let outp = PathBuf::from(out);
    std::fs::create_dir_all(&outp).map_err(|e| e.to_string())?;
    let load0 = loadavg();

    // pass A: the whole prompt in one call (teacher forcing)
    let t0 = std::time::Instant::now();
    let mut st = r.new_state();
    let mut taps = Vec::new();
    let (mixed_a, logits_a) = r.forward(&mut st, &tokens, Some(&mut taps))?;
    let wall_a = t0.elapsed().as_secs_f64();
    let routing: Vec<serde_json::Value> = r
        .traces
        .iter()
        .map(|t| serde_json::json!({"ids": t.ids, "weights": t.weights, "shared_gate": t.shared_gate}))
        .collect();
    let selected: Vec<Vec<usize>> = r
        .selections
        .iter()
        .map(|s| s.iter().map(|x| x.len()).collect())
        .collect();
    let finite = logits_a.iter().all(|x| x.is_finite()) && mixed_a.iter().all(|x| x.is_finite());
    let mut positions = Vec::new();
    let mut nll_sum = 0f64;
    let mut nll_n = 0usize;
    let mut top1_hits = 0usize;
    for i in 0..tokens.len() {
        let row = &logits_a[i * v..(i + 1) * v];
        let mut idx: Vec<usize> = (0..v).collect();
        idx.select_nth_unstable_by(4, |&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        let mut top: Vec<usize> = idx[..5].to_vec();
        top.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap());
        let mut e = serde_json::json!({
            "pos": i, "input": tokens[i], "input_text": text(tokens[i]),
            "top5": top.iter().map(|&j| serde_json::json!({"id": j, "logit": row[j], "text": text(j as u32)})).collect::<Vec<_>>(),
        });
        if i + 1 < tokens.len() {
            let nx = tokens[i + 1] as usize;
            let lp = log_softmax_at(row, nx);
            nll_sum -= lp;
            nll_n += 1;
            if top[0] == nx {
                top1_hits += 1;
            }
            e["next_given"] = serde_json::json!(nx);
            e["next_logprob"] = serde_json::json!(lp);
            e["next_is_top1"] = serde_json::json!(top[0] == nx);
        }
        positions.push(e);
    }

    // pass B: the same tokens one at a time (recurrent state, conv state, indexer and n-gram history)
    let t1 = std::time::Instant::now();
    let mut st = r.new_state();
    let mut logits_b: Vec<f32> = Vec::new();
    for &tok in &tokens {
        let (_, lg) = r.forward(&mut st, &[tok], None)?;
        logits_b.extend(lg);
    }
    let wall_b = t1.elapsed().as_secs_f64();
    let mut max_abs = 0f32;
    let mut argmax_same = 0usize;
    let (mut e2, mut r2) = (0f64, 0f64);
    for i in 0..tokens.len() {
        let (a, b) = (&logits_a[i * v..(i + 1) * v], &logits_b[i * v..(i + 1) * v]);
        for (&x, &y) in a.iter().zip(b) {
            max_abs = max_abs.max((x - y).abs());
            e2 += ((x - y) as f64).powi(2);
            r2 += (y as f64).powi(2);
        }
        if argmax(a) == argmax(b) {
            argmax_same += 1;
        }
    }

    // greedy continuation from the one-at-a-time state
    let t2 = std::time::Instant::now();
    let mut gen_ids: Vec<u32> = Vec::new();
    let mut next = argmax(&logits_b[(tokens.len() - 1) * v..]);
    for _ in 0..gen {
        gen_ids.push(next);
        let (_, lg) = r.forward(&mut st, &[next], None)?;
        next = argmax(&lg);
    }
    let wall_g = t2.elapsed().as_secs_f64();
    let gen_text: String = gen_ids.iter().map(|&i| text(i)).collect();
    let prompt_text: String = tokens.iter().map(|&i| text(i)).collect();

    let l = c.lanes();
    let lane_rms: Vec<Vec<f64>> = taps
        .iter()
        .map(|tp| {
            tp.chunks(l)
                .map(|row| (row.iter().map(|v| (*v as f64).powi(2)).sum::<f64>() / l as f64).sqrt())
                .collect()
        })
        .collect();
    let nll = if nll_n > 0 { nll_sum / nll_n as f64 } else { 0.0 };
    let doc = serde_json::json!({
        "what": "Qwen3.8-Flash-Next CPU reference, teacher-forced golden on real NVFP4 tensors",
        "checkpoint_dir": dir, "layers": n, "expert_format": "stored (NVFP4 dequantised per hf_quant_config.json)",
        "prompt_ids": tokens, "prompt_text": prompt_text,
        "finite": finite,
        "teacher_forced": {"mean_nll_nats": nll, "perplexity": nll.exp(), "top1_hits": top1_hits, "positions_scored": nll_n},
        "one_at_a_time_vs_whole_prompt": {"max_abs_logit_diff": max_abs, "rel_rms": (e2 / r2.max(1e-30)).sqrt(), "argmax_same_positions": argmax_same, "positions": tokens.len()},
        "greedy": {"ids": gen_ids, "text": gen_text},
        "positions": positions,
        "moe_routing_pass_a": routing,
        "indexer_selected_counts_pass_a": selected,
        "lane_rms_entering_layer": lane_rms,
        "wall_s": {"whole_prompt": wall_a, "one_at_a_time": wall_b, "greedy": wall_g},
        "loadavg_start": load0, "loadavg_end": loadavg(),
    });
    std::fs::write(outp.join("golden.json"), serde_json::to_string_pretty(&doc).unwrap())
        .map_err(|e| e.to_string())?;
    write_f32(&outp.join("logits_last.f32"), &logits_a[(tokens.len() - 1) * v..])?;
    write_f32(&outp.join("mixer_out_last.f32"), &mixed_a[(tokens.len() - 1) * c.hidden..])?;
    println!(
        "PASS golden: {n} layers, {} prompt tokens, finite {finite}, nll {nll:.3}, top1 {top1_hits}/{nll_n}, \
         one-at-a-time max|dlogit| {max_abs:.4}, greedy {:?} = {:?}, {wall_a:.0}s + {wall_b:.0}s + {wall_g:.0}s",
        tokens.len(),
        gen_ids,
        gen_text
    );
    Ok(())
}
