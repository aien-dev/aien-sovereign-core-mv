//! The Qwen3.6-35B-A3B MXFP4 store (L355): convert once, check, survey the scale rules.
//!
//!   q36_w4_store write <checkpoint dir> <out store> <ocp|fit|best|nv16> [threads]
//!   q36_w4_store gptq <checkpoint dir> <out store> <calibration.txt> <tokens> [threads]
//!   q36_w4_store check <checkpoint dir> <store>
//!   q36_w4_store survey <checkpoint dir> [experts per layer, default 16]
//!   q36_w4_store split <checkpoint dir> [experts per layer, default 16]
//!   q36_w4_store hdump <checkpoint dir> <calibration.txt> <tokens> <out prefix> [threads]
//!   q36_w4_store lowrank <checkpoint dir> <nv16 store> <r> <out sidecar|-> <H fit|-> <H eval|-> [experts per layer|all] [threads]
//!
//! `hdump` and `lowrank` are the low-rank arm (docs/inference/q36/w4-gate.md "Low-rank arm"): hdump writes the down-input H
//! of the first and second half of the calibration tokens (`<prefix>-a.hdn`, `<prefix>-b.hdn`); lowrank fits the rank-r
//! correction of every down matrix to nv16's rounding error (calibrated with H fit, plain with `-`), optionally writes the
//! sidecar, and prints the frozen go / no-go: squared error after <= 0.5 x before, weighted by H eval when given.
//! `survey` converts a fixed sample of each layer's routed experts (ids 0, 16, 32, .. for 16) plus the shared expert and
//! the trunk with every rule and prints one JSON line per rule: weights, saturated, relative RMS error. It picks no rule;
//! the choice is written in docs/inference/q36/w4-gate.md before a drift run.
use aien_inference_abi::q36_w4_store::{layer_projs, read_fused, write_store, W4Store};
use aien_inference_abi::qwen38::{text_prefix, Qwen38Config};
use aien_inference_abi::qwen3_moe_omega::ShardSource;
use aien_omega_q38::w4::{from_fp8, W4Stats, W4_RULE_BEST, W4_RULE_FIT, W4_RULE_OCP};
use std::path::Path;
use std::time::Instant;

fn rule_of(s: &str) -> Result<i32, String> {
    match s {
        "ocp" => Ok(W4_RULE_OCP),
        "fit" => Ok(W4_RULE_FIT),
        "best" => Ok(W4_RULE_BEST),
        r => Err(format!("rule {r} (ocp, fit or best)")),
    }
}

fn threads() -> usize {
    std::thread::available_parallelism().map_or(8, |n| n.get())
}

fn survey(dir: &Path, per_layer: usize) -> Result<(), String> {
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    let step = (s.experts / per_layer.max(1)).max(1);
    let rules = [("ocp", W4_RULE_OCP), ("fit", W4_RULE_FIT), ("best", W4_RULE_BEST)];
    let mut ex = [W4Stats::default(); 3];
    let mut tr = [W4Stats::default(); 3];
    for l in 0..c.layers {
        let projs = layer_projs(&c, &prefix, l, c.full_attention[l])?;
        for (pi, (name, parts)) in projs.iter().enumerate() {
            // experts: a sample of blocks (gate + up = 2 parts per expert, down = 1), always including the shared one
            let picked: Vec<_> = match pi {
                0 | 1 => parts.clone(),
                2 => parts.chunks(2).enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).flat_map(|(_, p)| p.to_vec()).collect(),
                _ => parts.iter().enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).map(|(_, p)| p.clone()).collect(),
            };
            for part in &picked {
                let (n, k, w, sc) = read_fused(&mut source, name, std::slice::from_ref(part))?;
                for (ri, (_, rule)) in rules.iter().enumerate() {
                    let (_, _, st) = from_fp8(n, k, &w, &sc, threads(), *rule).map_err(|e| format!("{name}: {e}"))?;
                    if pi < 2 { tr[ri].add(&st) } else { ex[ri].add(&st) }
                }
            }
        }
        eprintln!("survey: layer {l} of {}", c.layers);
    }
    for (ri, (rname, _)) in rules.iter().enumerate() {
        for (what, st) in [("experts", &ex[ri]), ("trunk", &tr[ri])] {
            println!(
                "{{\"rule\":\"{rname}\",\"tensors\":\"{what}\",\"weights\":{},\"saturated\":{},\"zero_groups\":{},\"rel_rms\":{:.6}}}",
                st.weights, st.saturated, st.zero_groups, st.rel_rms()
            );
        }
    }
    Ok(())
}

/// The survey's expert sample with BEST and nv16, gate|up and down apart (docs/inference/q36/w4-gate.md, NV16 arm item
/// 4): one JSON line per rule and matrix with sq_err and sq_ref, then the mixed go / no-go against 0.5 x BEST's sq.
fn split(dir: &Path, per_layer: usize) -> Result<(), String> {
    use aien_inference_abi::{q36_w4_gptq::widen_fp8, q36_w4_nv};
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    let step = (s.experts / per_layer.max(1)).max(1);
    // [rule][matrix]: rule 0 = best, 1 = nv16; matrix 0 = gate|up, 1 = down
    let mut st = [[W4Stats::default(); 2]; 2];
    for l in 0..c.layers {
        let projs = layer_projs(&c, &prefix, l, c.full_attention[l])?;
        for (pi, (name, parts)) in projs.iter().enumerate().skip(2) {
            let picked: Vec<_> = match pi {
                2 => parts.chunks(2).enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).flat_map(|(_, p)| p.to_vec()).collect(),
                _ => parts.iter().enumerate().filter(|(e, _)| e % step == 0 || *e == s.experts).map(|(_, p)| p.clone()).collect(),
            };
            for part in &picked {
                // one part = one linear (gate, up or down), so nv16 gets one global per linear as in its store
                let (n, k, w, sc) = read_fused(&mut source, name, std::slice::from_ref(part))?;
                let (_, _, b) = from_fp8(n, k, &w, &sc, threads(), W4_RULE_BEST).map_err(|e| format!("{name}: {e}"))?;
                let (_, _, _, v) = q36_w4_nv::quantize(&widen_fp8(n, k, &w, &sc), n, k, threads()).map_err(|e| format!("{name}: {e}"))?;
                st[0][pi - 2].add(&b);
                st[1][pi - 2].add(&v);
            }
        }
        eprintln!("split: layer {l} of {}", c.layers);
    }
    for (ri, rname) in ["best", "nv16"].iter().enumerate() {
        for (mi, what) in ["gate_up", "down"].iter().enumerate() {
            let x = &st[ri][mi];
            println!(
                "{{\"rule\":\"{rname}\",\"matrix\":\"{what}\",\"weights\":{},\"saturated\":{},\"sq_err\":{:e},\"sq_ref\":{:e},\"rel_rms\":{:.6}}}",
                x.weights, x.saturated, x.sq_err, x.sq_ref, x.rel_rms()
            );
        }
    }
    let r = st[0][0].sq_ref + st[0][1].sq_ref;
    let limit = 0.5 * (st[0][0].sq_err + st[0][1].sq_err) / r;
    let (gu, dn) = (st[1][0].sq_err / r, st[1][1].sq_err / r);
    let v = |x: f64| if x <= limit { "GO" } else { "NO-GO" };
    println!(
        "{{\"best_sq\":{:.7},\"nv16_sq\":{:.7},\"limit_sq\":{limit:.7},\"m_gu_sq\":{gu:.7},\"m_gu\":\"{}\",\"m_dn_sq\":{dn:.7},\"m_dn\":\"{}\"}}",
        2.0 * limit,
        (st[1][0].sq_err + st[1][1].sq_err) / r,
        v(gu),
        v(dn)
    );
    Ok(())
}

/// The first `tokens` ids of a calibration text. The tokenizer takes at most 2048 tokens per call: pieces of about 4000
/// characters cut at line ends.
fn calib_ids(dir: &Path, text_path: &Path, tokens: usize) -> Result<Vec<u32>, String> {
    use aien_inference_abi::tokenizer::ChatTokenizer;
    let text = std::fs::read_to_string(text_path).map_err(|e| format!("{}: {e}", text_path.display()))?;
    let tok = ChatTokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| e.to_string())?;
    let mut ids = Vec::new();
    let mut piece = String::new();
    for line in text.split_inclusive('\n') {
        piece.push_str(line);
        if piece.len() >= 4000 {
            ids.extend(tok.encode(&piece).map_err(|e| e.to_string())?);
            piece.clear();
        }
    }
    if !piece.is_empty() {
        ids.extend(tok.encode(&piece).map_err(|e| e.to_string())?);
    }
    if ids.len() < tokens {
        return Err(format!("{}: {} tokens, {tokens} asked", text_path.display(), ids.len()));
    }
    ids.truncate(tokens);
    Ok(ids)
}

/// Runs the FP8 host reference over the first `tokens` tokens of `text` with the calibration tap on, then writes the
/// error-feedback store (docs/inference/q36/w4-gate.md, GPTQ arm). The calibration text must not be gate text.
fn gptq(dir: &Path, out: &Path, text_path: &Path, tokens: usize, th: usize) -> Result<(), String> {
    use aien_inference_abi::qwen38::{Logits, Qwen38Reference};
    let t0 = Instant::now();
    let ids = calib_ids(dir, text_path, tokens)?;
    let mut model = Qwen38Reference::open(dir)?;
    let c = model.config.clone();
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    model.calib = Some(aien_inference_abi::q36_w4_gptq::Calib::new(c.layers, c.hidden, s.expert_inter, s.experts, th));
    // windows of 2048 tokens (a fresh context each, like separate conversations), fed in chunks of 512
    let mut st = model.new_state();
    for (i, chunk) in ids.chunks(512).enumerate() {
        if i % 4 == 0 {
            st = model.new_state();
        }
        model.forward(&mut st, chunk, Logits::Last)?;
        eprintln!("gptq calibration: {} of {tokens} tokens, {:.0} s", i * 512 + chunk.len(), t0.elapsed().as_secs_f64());
    }
    let calib = model.calib.take().ok_or("calibration lost")?;
    drop(model);
    let h = aien_inference_abi::q36_w4_gptq::write_store(dir, out, &calib, th, &mut |m| eprintln!("{m}"))?;
    println!(
        "{{\"store\":\"{}\",\"rule\":\"{}\",\"calibration\":\"{}\",\"tokens\":{tokens},\"tensors\":{},\"weights\":{},\"rel_rms\":{:.6},\"seconds\":{:.1}}}",
        out.display(), h.rule, text_path.display(), h.tensors.len(), h.weights, h.rel_rms, t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// The FP8 host reference over the first `tokens` calibration tokens with the down tap only (no gate|up H), as [`gptq`]
/// (windows of 2048, chunks of 512); H of the first half goes to `<prefix>-a.hdn`, of the second to `<prefix>-b.hdn`.
fn hdump(dir: &Path, text_path: &Path, tokens: usize, prefix: &str, th: usize) -> Result<(), String> {
    use aien_inference_abi::q36_w4_gptq::Calib;
    use aien_inference_abi::q36_w4_lowrank::Hdn;
    use aien_inference_abi::qwen38::{Logits, Qwen38Reference};
    if tokens % 4096 != 0 {
        return Err(format!("hdump: {tokens} tokens is not two halves of whole 2048-token windows"));
    }
    let t0 = Instant::now();
    let ids = calib_ids(dir, text_path, tokens)?;
    let mut model = Qwen38Reference::open(dir)?;
    let c = model.config.clone();
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    let fresh = || {
        let mut k = Calib::new(c.layers, c.hidden, s.expert_inter, s.experts, th);
        k.h_gu = Vec::new(); // add_gu skips every layer: only the down inputs are wanted
        k
    };
    let mut outs = Vec::new();
    for (half, part) in ids.chunks(tokens / 2).enumerate() {
        model.calib = Some(fresh());
        let mut st = model.new_state();
        for (i, chunk) in part.chunks(512).enumerate() {
            if i % 4 == 0 {
                st = model.new_state();
            }
            model.forward(&mut st, chunk, Logits::Last)?;
            eprintln!("hdump half {half}: {} of {} tokens, {:.0} s", i * 512 + chunk.len(), part.len(), t0.elapsed().as_secs_f64());
        }
        let k = model.calib.take().ok_or("calibration lost")?;
        let out = format!("{prefix}-{}.hdn", if half == 0 { "a" } else { "b" });
        Hdn::write(Path::new(&out), &k.h_dn, &k.n_dn, s.expert_inter)?;
        let routed = k.n_dn.iter().flatten().filter(|&&n| n > 0).count();
        outs.push(format!("{{\"file\":\"{out}\",\"tokens\":{},\"slots_reached\":{routed}}}", part.len()));
    }
    println!(
        "{{\"calibration\":\"{}\",\"tokens\":{tokens},\"halves\":[{}],\"seconds\":{:.1}}}",
        text_path.display(), outs.join(","), t0.elapsed().as_secs_f64()
    );
    Ok(())
}

/// Per-matrix sums of the screen.
#[derive(Default, Clone, Copy)]
struct LrStat {
    matrices: usize,
    sq_ref: f64,
    sq_before: f64,
    sq_after: f64,
    w_before: f64,
    w_after: f64,
    /// matrices with no H eval rows (not in the weighted sums)
    unweighted: usize,
    /// matrices fitted plain because H fit had no rows
    fit_plain: usize,
}

impl LrStat {
    fn add(&mut self, o: &LrStat) {
        self.matrices += o.matrices;
        self.sq_ref += o.sq_ref;
        self.sq_before += o.sq_before;
        self.sq_after += o.sq_after;
        self.w_before += o.w_before;
        self.w_after += o.w_after;
        self.unweighted += o.unweighted;
        self.fit_plain += o.fit_plain;
    }
    fn json(&self) -> String {
        format!(
            "\"matrices\":{},\"sq_ref\":{:e},\"sq_before\":{:e},\"sq_after\":{:e},\"ratio\":{:.6},\"w_before\":{:e},\"w_after\":{:e},\"w_ratio\":{:.6},\"unweighted\":{},\"fit_plain\":{}",
            self.matrices, self.sq_ref, self.sq_before, self.sq_after, self.sq_after / self.sq_before,
            self.w_before, self.w_after, if self.w_before > 0.0 { self.w_after / self.w_before } else { f64::NAN },
            self.unweighted, self.fit_plain
        )
    }
}

/// Fits the rank-`r` correction of every (or a sample of each layer's) down matrix to nv16's rounding error, writes the
/// sidecar when `out` is given (all experts only), and prints per-layer lines on stderr and the go / no-go on stdout.
#[allow(clippy::too_many_arguments)]
fn lowrank(dir: &Path, store: &Path, r: usize, out: Option<&Path>, hfit: Option<&Path>, heval: Option<&Path>, per_layer: Option<usize>, th: usize) -> Result<(), String> {
    use aien_inference_abi::q36_w4_gptq::widen_fp8;
    use aien_inference_abi::q36_w4_lowrank::{chol_damped, fit, frob, gram, residual, Hdn, Shape, Writer};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Mutex;
    let t0 = Instant::now();
    let c = Qwen38Config::from_checkpoint(dir)?;
    let mut source = ShardSource::open(dir)?;
    let prefix = text_prefix(&mut source)?;
    let s = c.moe.clone().ok_or("not a MoE checkpoint")?;
    let st = W4Store::open(store, dir)?;
    if st.group() != 16 {
        return Err(format!("{}: not an nv16 store (the m-dn down)", store.display()));
    }
    if out.is_some() && per_layer.is_some() {
        return Err("lowrank: the sidecar needs every expert (no sample)".into());
    }
    let (h, k, slots) = (c.hidden, s.expert_inter, s.experts + 1);
    let hf = hfit.map(Hdn::open).transpose()?;
    let he = heval.map(Hdn::open).transpose()?;
    for x in hf.iter().chain(he.iter()) {
        if x.layers != c.layers || x.slots != slots || x.k != k {
            return Err(format!("H file shape {}x{}x{} is not {}x{slots}x{k}", x.layers, x.slots, x.k, c.layers));
        }
    }
    let shape = Shape { layers: c.layers, slots, h, k, r, method: hf.is_some() as u32 };
    let w = out.map(|p| Writer::create(p, shape)).transpose()?;
    let step = per_layer.map_or(1, |n| (s.experts / n.max(1)).max(1));
    let mut total = LrStat::default();
    let mut layers = Vec::new();
    for l in 0..c.layers {
        let projs = layer_projs(&c, &prefix, l, c.full_attention[l])?;
        let (name, parts) = &projs[3];
        let (g, per) = st.globals(name).ok_or_else(|| format!("{name}: no nv16 globals"))?;
        let picked: Vec<usize> = (0..slots).filter(|e| e % step == 0 || *e == s.experts).collect();
        // E = W_fp8 - Q_nv16 per picked slot, with |W|^2
        let mut errs = Vec::with_capacity(picked.len());
        for &e in &picked {
            let (n, kk, w8, sc) = read_fused(&mut source, name, std::slice::from_ref(&parts[e]))?;
            if n != h || kk != k {
                return Err(format!("{name} block {e}: [{n}][{kk}], want [{h}][{k}]"));
            }
            let mut wf = widen_fp8(n, k, &w8, &sc);
            let sq_ref: f64 = wf.iter().map(|&x| x as f64 * x as f64).sum();
            let (w4, e8) = st.read_rows(name, e * h, h)?;
            let mut q = vec![0f32; k];
            for row in 0..h {
                aien_inference_abi::q36_w4_nv::widen_row(&w4, &e8, 1.0, k, row, &mut q);
                let gl = g[(e * h + row) / per];
                for (x, &y) in wf[row * k..(row + 1) * k].iter_mut().zip(&q) {
                    *x -= y * gl;
                }
            }
            errs.push((e, wf, sq_ref));
        }
        let next = AtomicUsize::new(0);
        let acc = Mutex::new((LrStat::default(), None::<String>));
        std::thread::scope(|sc| {
            for _ in 0..th.min(errs.len()).max(1) {
                sc.spawn(|| loop {
                    let i = next.fetch_add(1, Ordering::Relaxed);
                    if i >= errs.len() {
                        break;
                    }
                    let (e, em, sq_ref) = (errs[i].0, &errs[i].1, errs[i].2);
                    let one = (|| -> Result<LrStat, String> {
                        let mut o = LrStat { matrices: 1, sq_ref, ..Default::default() };
                        let lch = match &hf {
                            Some(f) => match f.get(l, e)? {
                                Some(hm) => chol_damped(&hm, k)?,
                                None => None,
                            },
                            None => None,
                        };
                        o.fit_plain = (hf.is_some() && lch.is_none()) as usize;
                        let ft = fit(em, h, k, r, lch.as_deref());
                        let d = residual(em, h, k, r, &ft);
                        o.sq_before = em.iter().map(|&x| x as f64 * x as f64).sum();
                        o.sq_after = d.iter().map(|&x| x as f64 * x as f64).sum();
                        if let Some(f) = &he {
                            match f.get(l, e)? {
                                Some(hm) => {
                                    o.w_before = frob(&gram(em, h, k), &hm);
                                    o.w_after = frob(&gram(&d, h, k), &hm);
                                }
                                None => o.unweighted = 1,
                            }
                        }
                        if let Some(w) = &w {
                            w.put(l, e, &ft)?;
                        }
                        Ok(o)
                    })();
                    let mut a = acc.lock().unwrap();
                    match one {
                        Ok(o) => a.0.add(&o),
                        Err(x) => a.1 = Some(format!("layer {l} slot {e}: {x}")),
                    }
                });
            }
        });
        let (ls, err) = acc.into_inner().unwrap();
        if let Some(x) = err {
            return Err(x);
        }
        eprintln!("lowrank: layer {l} {{{}}} {:.0} s", ls.json(), t0.elapsed().as_secs_f64());
        layers.push(format!("{{\"layer\":{l},{}}}", ls.json()));
        total.add(&ls);
    }
    let weighted = he.is_some();
    let ratio = if weighted { total.w_after / total.w_before } else { total.sq_after / total.sq_before };
    println!(
        "{{\"rule\":\"w4-gate.md Low-rank arm (b')\",\"store\":\"{}\",\"r\":{r},\"method\":\"{}\",\"h_fit\":{},\"h_eval\":{},\"sample\":{},\"sidecar\":{},\"sidecar_bytes\":{},{},\"metric\":\"{}\",\"gate_ratio\":{ratio:.6},\"limit\":0.5,\"verdict\":\"{}\",\"layers\":[{}],\"seconds\":{:.1}}}",
        store.display(),
        if hf.is_some() { "calibrated" } else { "plain" },
        hfit.map_or("null".into(), |p| format!("\"{}\"", p.display())),
        heval.map_or("null".into(), |p| format!("\"{}\"", p.display())),
        per_layer.map_or("\"all\"".into(), |n| n.to_string()),
        out.map_or("null".into(), |p| format!("\"{}\"", p.display())),
        if out.is_some() { 64 + (c.layers * slots * (h * r + r * k) * 2) as u64 } else { 0 },
        total.json(),
        if weighted { "weighted tr(D H_eval D^T)" } else { "unweighted |D|^2" },
        if ratio <= 0.5 { "GO" } else { "NO-GO" },
        layers.join(","),
        t0.elapsed().as_secs_f64()
    );
    Ok(())
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let t0 = Instant::now();
    let r = match a.get(1).map(String::as_str) {
        Some("write") if a.len() >= 5 && a[4] == "nv16" => {
            let th = a.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            aien_inference_abi::q36_w4_nv::write_store(Path::new(&a[2]), Path::new(&a[3]), th, &mut |m| eprintln!("{m}")).map(|h| {
                println!(
                    "{{\"store\":\"{}\",\"rule\":\"{}\",\"tensors\":{},\"weights\":{},\"saturated\":{},\"rel_rms\":{:.6},\"seconds\":{:.1}}}",
                    a[3], h.rule, h.tensors.len(), h.weights, h.saturated, h.rel_rms, t0.elapsed().as_secs_f64()
                );
            })
        }
        Some("write") if a.len() >= 5 => rule_of(&a[4]).and_then(|rule| {
            let th = a.get(5).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            let h = write_store(Path::new(&a[2]), Path::new(&a[3]), rule, th, &mut |m| eprintln!("{m}"))?;
            println!(
                "{{\"store\":\"{}\",\"rule\":\"{}\",\"tensors\":{},\"weights\":{},\"saturated\":{},\"rel_rms\":{:.6},\"seconds\":{:.1}}}",
                a[3], h.rule, h.tensors.len(), h.weights, h.saturated, h.rel_rms, t0.elapsed().as_secs_f64()
            );
            Ok(())
        }),
        Some("check") if a.len() >= 4 => W4Store::open(Path::new(&a[3]), Path::new(&a[2])).and_then(|s| {
            s.check_all()?;
            println!("{{\"store\":\"{}\",\"check\":\"PASS\",\"tensors\":{},\"seconds\":{:.1}}}", a[3], s.header.tensors.len(), t0.elapsed().as_secs_f64());
            Ok(())
        }),
        Some("gptq") if a.len() >= 6 => a[5].parse().map_err(|_| "tokens".to_string()).and_then(|n| {
            let th = a.get(6).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            gptq(Path::new(&a[2]), Path::new(&a[3]), Path::new(&a[4]), n, th)
        }),
        Some("survey") if a.len() >= 3 => survey(Path::new(&a[2]), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16)),
        Some("split") if a.len() >= 3 => split(Path::new(&a[2]), a.get(3).and_then(|s| s.parse().ok()).unwrap_or(16)),
        Some("hdump") if a.len() >= 6 => a[4].parse().map_err(|_| "tokens".to_string()).and_then(|n| {
            let th = a.get(6).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            hdump(Path::new(&a[2]), Path::new(&a[3]), n, &a[5], th)
        }),
        Some("lowrank") if a.len() >= 8 => a[4].parse().map_err(|_| "r".to_string()).and_then(|r| {
            let opt = |s: &String| (s != "-").then(|| std::path::PathBuf::from(s));
            let (out, hf, he) = (opt(&a[5]), opt(&a[6]), opt(&a[7]));
            let per = match a.get(8).map(String::as_str) {
                None | Some("all") => None,
                Some(n) => Some(n.parse().map_err(|_| "experts per layer".to_string())?),
            };
            let th = a.get(9).and_then(|s| s.parse().ok()).unwrap_or_else(threads);
            lowrank(Path::new(&a[2]), Path::new(&a[3]), r, out.as_deref(), hf.as_deref(), he.as_deref(), per, th)
        }),
        _ => Err("usage: q36_w4_store write <dir> <out> <ocp|fit|best|nv16> [threads] | gptq <dir> <out> <calibration.txt> <tokens> [threads] | check <dir> <store> | survey <dir> [n] | split <dir> [n] | hdump <dir> <calibration.txt> <tokens> <out prefix> [threads] | lowrank <dir> <nv16 store> <r> <sidecar|-> <H fit|-> <H eval|-> [n|all] [threads]".into()),
    };
    if let Err(e) = r {
        eprintln!("q36_w4_store: {e}");
        std::process::exit(1);
    }
}
