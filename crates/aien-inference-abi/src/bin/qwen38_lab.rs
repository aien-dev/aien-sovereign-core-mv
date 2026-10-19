//! Skipped-ideas lab for Qwen3.8-27B (branch research/skipped-ideas; handoff
//! ~/handoffs/2026-10-19-skipped-ideas-lab.md). Host experiments on the f32 reference
//! (`qwen38::lab`) and one chip timing.
//!
//!   qwen38_lab mkseq <prompt.json> <gen-out.json> <seq.json>
//!     ids = prompt `ids` + the run's `generated_ids`; score_start = prompt length.
//!   qwen38_lab mkseq-chat <checkpoint-dir> <prompt.txt> <reply.txt> <seq.json>
//!     the chat-rendered prompt (thinking off) + the reply's tokens.
//!   qwen38_lab run <checkpoint-dir> <seq.json> base|skip|teal|quant|smoke <out.json> [--base <base.json>]
//!     one pass, every variant of the experiment against the unchanged model (variant 0) on the
//!     same tokens: top-1 agreement over the scored rows, flips and the base margin there, KL,
//!     NLL of the fed text, residual cosine per layer, bytes per decoded token. `skip` reads the
//!     layer order (block influence) from a `base` run.
//!   qwen38_lab mtp <checkpoint-dir> <seq.json> <out-dir>
//!     main model over the session (argmax per row), then the MTP head over every (residual,
//!     next token) pair: writes mtp.f32 (head output after mtp.norm), trace.json (targets).
//!   qwen38_lab learn <checkpoint-dir> <trace-dir> <out.json> [--configs a,b,..]
//!     replays the k = 1 speculative loop over the trace: one draft per pass, the verify pass
//!     reveals the model's token, an adapter on the drafter's output learns from it online.
//!   qwen38_lab rows <checkpoint-dir> <layers> <reps> <out.json>
//!     chip (caller holds /tmp/aien-gb10.lock): the exact multi-row path (qwen38_omega spec) at
//!     1, 2, 4, 8 rows on the first `layers` layers, the bf16 lm_head per row, and the decode
//!     graph token, for "several answers per pass".
use std::path::Path;
use std::time::Instant;

use aien_inference_abi::qwen38::lab::{
    kl, layer_bytes, model_bytes, quiet_lock, top2, Act, Quant, Teal, TealMode, Variant, UNION_N,
};
use aien_inference_abi::qwen38::Qwen38Reference;
use aien_inference_abi::qwen38::{argmax, negative_log_likelihood, Matrix, Qwen38Config};
use aien_inference_abi::tokenizer::{ChatOptions, ChatTokenizer};
use rayon::prelude::*;
use serde_json::{json, Value};

fn read_json(p: &str) -> Result<Value, String> {
    serde_json::from_str(&std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?)
        .map_err(|e| format!("{p}: {e}"))
}

fn ids(v: &Value, key: &str) -> Result<Vec<u32>, String> {
    v.get(key)
        .and_then(|a| a.as_array())
        .ok_or(format!("no {key}"))?
        .iter()
        .map(|x| {
            x.as_u64()
                .map(|u| u as u32)
                .ok_or(format!("{key}: not ids"))
        })
        .collect()
}

fn write(p: &str, v: &Value) -> Result<(), String> {
    std::fs::write(p, serde_json::to_vec_pretty(v).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{p}: {e}"))
}

fn mem_available_gib() -> f64 {
    std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|m| {
            m.lines()
                .find_map(|l| l.strip_prefix("MemAvailable:"))
                .and_then(|v| v.trim().trim_end_matches("kB").trim().parse::<f64>().ok())
        })
        .map(|kb| kb / 1048576.0)
        .unwrap_or(-1.0)
}

fn mkseq(a: &[String]) -> Result<(), String> {
    let [p, g, o] = a else {
        return Err("mkseq <prompt.json> <gen-out.json> <seq.json>".into());
    };
    let mut v = ids(&read_json(p)?, "ids")?;
    let n = v.len();
    v.extend(ids(&read_json(g)?, "generated_ids")?);
    write(
        o,
        &json!({"ids": v, "score_start": n, "prompt": p, "gen": g}),
    )
}

fn mkseq_chat(a: &[String]) -> Result<(), String> {
    let [dir, p, r, o] = a else {
        return Err("mkseq-chat <ckpt> <prompt.txt> <reply.txt> <seq.json>".into());
    };
    let tok = ChatTokenizer::from_model_dir(Path::new(dir), None).map_err(|e| e.to_string())?;
    let prompt = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
    let reply = std::fs::read_to_string(r).map_err(|e| format!("{r}: {e}"))?;
    let opts = ChatOptions {
        enable_thinking: Some(false),
        reasoning_effort: None,
    };
    let s = tok
        .try_format_chat_with(&[("user", prompt.trim_end())], &opts)
        .map_err(|e| e.to_string())?;
    let mut v = tok
        .encode_with_special(&s, false)
        .map_err(|e| e.to_string())?;
    let n = v.len();
    v.extend(
        tok.encode_with_special(&reply, false)
            .map_err(|e| e.to_string())?,
    );
    write(
        o,
        &json!({"ids": v, "score_start": n, "prompt": p, "reply": r}),
    )
}

fn full(layers: usize) -> Vec<Act> {
    vec![Act::Full; layers]
}

/// Layers ordered by block influence (lowest first) within a pool.
fn order(bi: &[f64], pool: &[usize]) -> Vec<usize> {
    let mut p = pool.to_vec();
    p.sort_by(|&a, &b| bi[a].total_cmp(&bi[b]));
    p
}

fn variants(c: &Qwen38Config, exp: &str, base: Option<&Value>) -> Result<Vec<Variant>, String> {
    let n = c.layers;
    let mut v = vec![Variant {
        name: "base".into(),
        acts: full(n),
    }];
    let with = |name: String, f: &dyn Fn(usize) -> Act| Variant {
        name,
        acts: (0..n).map(f).collect(),
    };
    match exp {
        "base" => {}
        "smoke" => {
            v.push(with("dup".into(), &|_| Act::Full));
            v.push(with("skip-63".into(), &|l| {
                if l == n - 1 {
                    Act::Skip
                } else {
                    Act::Full
                }
            }));
            v.push(with("teal-row-0.5".into(), &|_| {
                Act::Teal(Teal {
                    s: 0.5,
                    group: 1,
                    mode: TealMode::Row,
                })
            }));
        }
        "skip" => {
            let b = base.ok_or("skip needs --base <base.json>")?;
            let bi: Vec<f64> = b["bi"]
                .as_array()
                .ok_or("base.json: no bi")?
                .iter()
                .map(|x| x.as_f64().unwrap_or(1.0))
                .collect();
            let attn: Vec<usize> = (0..n).filter(|&l| c.full_attention[l]).collect();
            let dn: Vec<usize> = (0..n).filter(|&l| !c.full_attention[l]).collect();
            let all: Vec<usize> = (0..n).collect();
            let pools = [
                ("any", order(&bi, &all)),
                ("attn", order(&bi, &attn)),
                ("dn", order(&bi, &dn)),
            ];
            for (pname, ord) in &pools {
                // Cut to fit the shared CPU (a stream costs one model pass): 2026-10-19T17:20Z.
                let ks: &[usize] = match *pname {
                    "attn" => &[1, 2, 4],
                    "dn" => &[1, 2, 4, 8],
                    _ => &[8],
                };
                for &k in ks {
                    let set: Vec<usize> = ord[..k].to_vec();
                    v.push(with(format!("static-{pname}-{k}"), &|l| {
                        if set.contains(&l) {
                            Act::Skip
                        } else {
                            Act::Full
                        }
                    }));
                }
            }
            for (pname, ord, ks) in [
                ("dn", &pools[2].1, &[4usize, 8][..]),
                ("attn", &pools[1].1, &[4][..]),
            ] {
                for &k in ks {
                    for cheap in [false, true] {
                        let set: Vec<usize> = ord[..k].to_vec();
                        let tag = if cheap { "cheap" } else { "nostate" };
                        v.push(with(format!("dyn-{pname}-{k}-{tag}"), &|l| {
                            if set.contains(&l) {
                                Act::DynSkip { cheap }
                            } else {
                                Act::Full
                            }
                        }));
                    }
                }
            }
        }
        "teal" => {
            for s in [0.2f32, 0.3, 0.4, 0.5] {
                v.push(with(format!("teal-row-{s}"), &|_| {
                    Act::Teal(Teal {
                        s,
                        group: 1,
                        mode: TealMode::Row,
                    })
                }));
            }
            for g in [4usize, 8] {
                for s in [0.3f32, 0.5] {
                    v.push(with(format!("teal-shared{g}-{s}"), &|_| {
                        Act::Teal(Teal {
                            s,
                            group: g,
                            mode: TealMode::Shared,
                        })
                    }));
                }
            }
            for (g, s) in [(4usize, 0.5f32)] {
                v.push(with(format!("teal-union{g}-{s}"), &|_| {
                    Act::Teal(Teal {
                        s,
                        group: g,
                        mode: TealMode::Union,
                    })
                }));
            }
        }
        "quant" => {
            for q in [
                Quant::Mxfp4,
                Quant::Int4G32,
                Quant::Mxint3,
                Quant::Mxfp3,
                Quant::Int3G32,
            ] {
                v.push(with(format!("quant-{}", q.name()), &|_| Act::Quant(q)));
            }
        }
        other => return Err(format!("exp {other}: base|skip|teal|quant|smoke")),
    }
    Ok(v)
}

/// Body + head bytes one decoded token reads under a variant (GB), from the config shapes.
fn bytes_gb(c: &Qwen38Config, var: &Variant, dyn_frac: f64, union_kept: Option<f64>) -> f64 {
    let (body, head) = model_bytes(c);
    let mut b = head;
    for l in 0..c.layers {
        let lb = layer_bytes(c, l);
        b += match &var.acts[l] {
            Act::Full => lb.total(),
            Act::Skip => 0.0,
            Act::DynSkip { cheap } => {
                let saved = if *cheap {
                    lb.mixer_rest + lb.mlp
                } else {
                    lb.total()
                };
                lb.total() - dyn_frac * saved
            }
            Act::Teal(t) => match t.mode {
                TealMode::Row => lb.total() * union_kept.unwrap_or(1.0 - t.s as f64),
                // Union rows: the kept share is not fixed; reported from the per-row unions.
                TealMode::Union => lb.total(),
                TealMode::Shared => lb.total() * (1.0 - t.s as f64),
            },
            Act::Quant(q) => lb.total() * q.bits() / 8.0,
        };
    }
    let _ = body;
    b / 1e9
}

fn run(a: &[String]) -> Result<(), String> {
    if a.len() < 4 {
        return Err("run <ckpt> <seq.json> <exp> <out.json> [--base base.json]".into());
    }
    let (dir, seqp, exp, outp) = (Path::new(&a[0]), &a[1], a[2].as_str(), &a[3]);
    let base = match a.get(4).map(String::as_str) {
        Some("--base") => Some(read_json(a.get(5).ok_or("--base needs a file")?)?),
        _ => None,
    };
    let seq = read_json(seqp)?;
    let tokens = ids(&seq, "ids")?;
    let ss = seq["score_start"].as_u64().ok_or("seq: no score_start")? as usize;
    let mem0 = mem_available_gib();
    let t0 = Instant::now();
    let mut model = Qwen38Reference::open(dir)?;
    let c = model.config.clone();
    let vocab = c.vocab;
    let vars = variants(&c, exp, base.as_ref())?;
    let t = tokens.len();
    let dyn_skip: Vec<bool> = (0..t).map(|p| p >= ss && (p - ss) % 2 == 1).collect();
    let dyn_frac = dyn_skip.iter().filter(|&&s| s).count() as f64 / (t - ss) as f64;
    let taps: Vec<usize> = if exp == "base" {
        vec![16, 32, 48, 56, 60, 62, 63]
    } else {
        vec![]
    };
    eprintln!(
        "{exp}: {} variants, {t} tokens, scored rows {}..{}, MemAvailable {mem0:.1} GiB",
        vars.len(),
        ss - 1,
        t - 1
    );
    let out = model.lab_pass(&tokens, ss, &vars, &dyn_skip, &taps)?;
    let h = c.hidden;
    let rows = t - ss;
    let base_logits = model.lab_logits(&out.vars[0].final_rows);
    let base_top = top2(&base_logits, vocab);
    let fed: Vec<usize> = (0..rows).map(|r| tokens[ss + r] as usize).collect();
    let nll_of = |lg: &[f32]| {
        lg.chunks_exact(vocab)
            .zip(&fed)
            .map(|(r, &f)| negative_log_likelihood(r, f))
            .sum::<f64>()
            / rows as f64
    };
    let base_nll = nll_of(&base_logits);
    let text_agree = base_top
        .iter()
        .zip(&fed)
        .filter(|(b, &f)| b.0 as usize == f)
        .count();
    let (body, head) = model_bytes(&c);
    let full_gb = (body + head) / 1e9;
    let mut results = Vec::new();
    for (vi, var) in vars.iter().enumerate() {
        let _q = quiet_lock()?;
        let vo = &out.vars[vi];
        let lg = if vi == 0 {
            base_logits.clone()
        } else {
            model.lab_logits(&vo.final_rows)
        };
        let top = top2(&lg, vocab);
        let mut flips = Vec::new();
        for (r, (b, v)) in base_top.iter().zip(&top).enumerate() {
            if b.0 != v.0 {
                flips.push(json!({"row": r, "base": b.0, "var": v.0, "base_margin": b.1}));
            }
        }
        let agree = rows - flips.len();
        let kls: Vec<f64> = lg
            .par_chunks_exact(vocab)
            .zip(base_logits.par_chunks_exact(vocab))
            .map(|(q, p)| kl(p, q))
            .collect();
        let kl_mean = kls.iter().sum::<f64>() / rows as f64;
        let kl_max = kls.iter().cloned().fold(0.0, f64::max);
        let first_flip = flips
            .first()
            .map(|f| f["row"].clone())
            .unwrap_or(json!(null));
        let union_kept: Option<Vec<f64>> = vo.union.as_ref().map(|u| {
            u.kept_bytes
                .iter()
                .map(|k| k / u.total_bytes.max(1.0))
                .collect()
        });
        let gb = bytes_gb(&c, var, dyn_frac, None);
        let cos_final = vo.cos_layer.last().cloned().unwrap_or((1.0, 1.0));
        let pick = |l: usize| vo.cos_layer.get(l).map(|x| json!([x.0, x.1]));
        let qerr = if vo.quant_err.1 > 0.0 {
            Some((vo.quant_err.0 / vo.quant_err.1).sqrt())
        } else {
            None
        };
        let skipped: Vec<usize> = (0..c.layers)
            .filter(|&l| matches!(var.acts[l], Act::Skip | Act::DynSkip { .. }))
            .collect();
        eprintln!(
            "{:22} agree {agree}/{rows} ({:.4})  KL {kl_mean:.3e}  cos final {:.5}/{:.5}  GB/token {gb:.2} ({:.1}% of {full_gb:.2}){}",
            var.name,
            agree as f64 / rows as f64,
            cos_final.0,
            cos_final.1,
            100.0 * gb / full_gb,
            union_kept
                .as_ref()
                .map(|u| format!("  union kept {u:.3?} for N {UNION_N:?}"))
                .unwrap_or_default()
        );
        results.push(json!({
            "name": var.name, "agree": agree, "rows": rows, "agree_frac": agree as f64 / rows as f64,
            "first_flip_row": first_flip, "flips": flips, "kl_mean": kl_mean, "kl_max": kl_max,
            "nll_fed_mean": nll_of(&lg), "cos_final_mean": cos_final.0, "cos_final_min": cos_final.1,
            "cos_layer_16_32_48_63": [pick(15), pick(31), pick(47), pick(63)],
            "gb_per_token": gb, "gb_frac_of_full": gb / full_gb,
            "union_kept_frac_by_rows": union_kept, "union_rows": UNION_N,
            "quant_rel_rms_err": qerr, "skipped_layers": skipped,
        }));
    }
    let mut extra = json!({});
    if exp == "base" {
        let mut ee = Vec::new();
        for (l, rows_l) in &out.taps {
            let _q = quiet_lock()?;
            let lg = model.lab_logits(rows_l);
            let top = top2(&lg, vocab);
            let agree = top
                .iter()
                .zip(&base_top)
                .filter(|(a, b)| a.0 == b.0)
                .count();
            eprintln!("early exit after {l} layers: agree {agree}/{rows}");
            ee.push(json!({"layers": l, "agree": agree, "rows": rows}));
        }
        extra = json!({"early_exit": ee});
    }
    let _ = h;
    let rec = json!({
        "tool": "qwen38_lab run", "exp": exp, "seq": seqp, "tokens": t, "score_start": ss,
        "scored_rows": rows, "pass_seconds": out.seconds, "total_seconds": t0.elapsed().as_secs_f64(),
        "mem_available_gib_start": mem0, "mem_available_gib_end": mem_available_gib(),
        "base_nll_fed_mean": base_nll, "base_argmax_equals_fed": text_agree,
        "base_argmax": base_top.iter().map(|x| x.0).collect::<Vec<_>>(),
        "base_margin": base_top.iter().map(|x| x.1).collect::<Vec<_>>(),
        "bi": out.bi, "dyn_skip_frac_of_decoded": dyn_frac, "full_gb_per_token": full_gb,
        "variants": results, "extra": extra,
    });
    write(outp, &rec)
}

fn write_f32(p: &Path, v: &[f32]) -> Result<(), String> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(p, b).map_err(|e| format!("{}: {e}", p.display()))
}

fn read_f32(p: &Path) -> Result<Vec<f32>, String> {
    let b = std::fs::read(p).map_err(|e| format!("{}: {e}", p.display()))?;
    Ok(b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect())
}

fn argmax_rows(
    model: &Qwen38Reference<impl aien_inference_abi::qwen3_moe_omega::TensorSource>,
    rows: &[f32],
    h: usize,
    normed: bool,
) -> Vec<u32> {
    let vocab = model.config.vocab;
    let mut out = Vec::new();
    for chunk in rows.chunks(64 * h) {
        let lg = if normed {
            model.lab_lm_head().matmul(chunk, chunk.len() / h)
        } else {
            model.lab_logits(chunk)
        };
        out.extend(lg.chunks_exact(vocab).map(|r| argmax(r) as u32));
    }
    out
}

fn mtp(a: &[String]) -> Result<(), String> {
    let [dir, seqp, outd] = a else {
        return Err("mtp <ckpt> <seq.json> <out-dir>".into());
    };
    let outd = Path::new(outd);
    std::fs::create_dir_all(outd).map_err(|e| e.to_string())?;
    let seq = read_json(seqp)?;
    let tokens = ids(&seq, "ids")?;
    let ss = seq["score_start"].as_u64().ok_or("seq: no score_start")? as usize;
    let mut model = Qwen38Reference::open(Path::new(dir))?;
    let c = model.config.clone();
    let h = c.hidden;
    let t = tokens.len();
    let t0 = Instant::now();
    let vars = vec![Variant {
        name: "base".into(),
        acts: full(c.layers),
    }];
    // Score from row 1 so every row's residual comes back (rows 0..t-1).
    let out = model.lab_pass(&tokens, 1, &vars, &vec![false; t], &[])?;
    let _q = quiet_lock()?;
    let resid = &out.vars[0].final_rows; // rows 0 .. t-1 (t - 1 rows)
    let main = argmax_rows(&model, resid, h, false);
    let pass_s = t0.elapsed().as_secs_f64();
    eprintln!("main pass {t} tokens in {pass_s:.0} s");
    // MTP pairs: residual at i with token i + 1, i in 0..t-1.
    let next: Vec<u32> = tokens[1..].to_vec();
    let m = model.lab_mtp(resid, &next[..resid.len() / h])?;
    let draft = argmax_rows(&model, &m, h, true);
    write_f32(&outd.join("mtp.f32"), &m)?;
    // draft[i] guesses token i + 2; the verifier's token there is main[i + 1].
    let rows = main.len();
    let mut hits = 0;
    let mut n = 0;
    for i in ss.saturating_sub(1)..rows - 1 {
        n += 1;
        hits += usize::from(draft[i] == main[i + 1]);
    }
    let text_agree = (0..rows).filter(|&i| main[i] == tokens[i + 1]).count();
    eprintln!("MTP draft == verifier on {hits}/{n} decoded rows; main argmax == text on {text_agree}/{rows}");
    write(
        &outd.join("trace.json").to_string_lossy(),
        &json!({"seq": seqp, "tokens": tokens, "score_start": ss, "main_argmax": main, "mtp_draft": draft,
                "draft_hits": hits, "draft_rows": n, "main_equals_text": text_agree, "rows": rows,
                "pass_seconds": pass_s, "total_seconds": t0.elapsed().as_secs_f64(),
                "mem_available_gib_end": mem_available_gib()}),
    )
}

/// Online adapter on the drafter's output (after mtp.norm): logits = W (o + delta(o)) + b.
#[derive(Clone, Debug)]
struct LearnCfg {
    name: String,
    kind: String, // none | vbias | hbias | lora
    lr: f32,
    rank: usize,
    miss_only: bool,
}

fn parse_cfg(s: &str) -> Result<LearnCfg, String> {
    // kind:lr[:rank][:miss]
    let p: Vec<&str> = s.split(':').collect();
    let kind = p[0].to_string();
    let lr = p.get(1).map(|x| x.parse().unwrap_or(0.0)).unwrap_or(0.0);
    let rank = p.get(2).and_then(|x| x.parse().ok()).unwrap_or(16);
    let miss_only = s.ends_with(":miss");
    if !["none", "vbias", "hbias", "lora"].contains(&kind.as_str()) {
        return Err(format!("config {s}: none|vbias|hbias|lora"));
    }
    Ok(LearnCfg {
        name: s.into(),
        kind,
        lr,
        rank,
        miss_only,
    })
}

struct Adam {
    m: Vec<f32>,
    v: Vec<f32>,
    t: i32,
}

impl Adam {
    fn new(n: usize) -> Self {
        Self {
            m: vec![0.0; n],
            v: vec![0.0; n],
            t: 0,
        }
    }
    fn step(&mut self, p: &mut [f32], g: &[f32], lr: f32, idx: Option<&[usize]>) {
        self.t += 1;
        let (b1, b2) = (0.9f32, 0.999f32);
        let c1 = 1.0 - b1.powi(self.t);
        let c2 = 1.0 - b2.powi(self.t);
        let mut upd = |i: usize, gi: f32| {
            self.m[i] = b1 * self.m[i] + (1.0 - b1) * gi;
            self.v[i] = b2 * self.v[i] + (1.0 - b2) * gi * gi;
            p[i] -= lr * (self.m[i] / c1) / ((self.v[i] / c2).sqrt() + 1e-8);
        };
        match idx {
            Some(ix) => {
                for (j, &i) in ix.iter().enumerate() {
                    upd(i, g[j]);
                }
            }
            None => {
                for (i, &gi) in g.iter().enumerate() {
                    upd(i, gi);
                }
            }
        }
    }
}

struct CfgState {
    cfg: LearnCfg,
    vb: Vec<f32>,
    hb: Vec<f32>,
    lu: Vec<f32>,
    lv: Vec<f32>,
    ad: [Adam; 4],
    p: usize,
    passes: usize,
    hits: usize,
    produced: usize,
    win: (usize, usize),
    next_mark: usize,
    windows: Vec<(usize, usize, usize)>,
    hx: Vec<f32>,
    vr: Vec<f32>,
}

fn learn(a: &[String]) -> Result<(), String> {
    if a.len() < 3 {
        return Err("learn <ckpt> <trace-dir> <out.json> [--configs ...]".into());
    }
    let (dir, td, outp) = (Path::new(&a[0]), Path::new(&a[1]), &a[2]);
    let cfgs: Vec<LearnCfg> = match a.get(3).map(String::as_str) {
        Some("--configs") => a
            .get(4)
            .ok_or("--configs needs a list")?
            .split(',')
            .map(parse_cfg)
            .collect::<Result<_, _>>()?,
        _ => [
            "none",
            "vbias:0.1",
            "vbias:0.5",
            "hbias:0.001",
            "hbias:0.01",
            "lora:0.001:16",
            "lora:0.01:16",
        ]
        .iter()
        .map(|s| parse_cfg(s))
        .collect::<Result<_, _>>()?,
    };
    let trace = read_json(&td.join("trace.json").to_string_lossy())?;
    let main = ids(&trace, "main_argmax")?;
    let ss = trace["score_start"].as_u64().ok_or("trace: score_start")? as usize;
    let model = Qwen38Reference::open(dir)?;
    let (h, vocab) = (model.config.hidden, model.config.vocab);
    let o_all = read_f32(&td.join("mtp.f32"))?;
    let w = model.lab_lm_head();
    let widen = |row: usize| -> Vec<f32> {
        match w {
            Matrix::Bf16 { k, w, .. } => w[row * k..(row + 1) * k]
                .iter()
                .map(|&b| aien_inference_abi::qwen3_moe::bf16_to_f32(b))
                .collect(),
            _ => panic!("lm_head expected bf16"),
        }
    };
    let rows = main.len();
    let t0 = Instant::now();
    let mut seed = 0x9E37_79B9u64;
    let mut rnd = || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        ((seed >> 11) as f64 / (1u64 << 53) as f64) as f32 - 0.5
    };
    let mut sts: Vec<CfgState> = cfgs
        .iter()
        .map(|cfg| {
            let r = cfg.rank;
            CfgState {
                cfg: cfg.clone(),
                vb: vec![0.0; vocab],
                hb: vec![0.0; h],
                lu: vec![0.0; h * r],
                lv: (0..h * r)
                    .map(|_| rnd() * 2.0 / (h as f32).sqrt())
                    .collect(),
                ad: [
                    Adam::new(vocab),
                    Adam::new(h),
                    Adam::new(h * r),
                    Adam::new(h * r),
                ],
                p: ss.saturating_sub(1),
                passes: 0,
                hits: 0,
                produced: 0,
                win: (0, 0),
                next_mark: 250,
                windows: Vec::new(),
                hx: Vec::new(),
                vr: Vec::new(),
            }
        })
        .collect();
    // The k = 1 loop: at verified position p the draft guesses token p + 2 from o[p]; the
    // verifier's token there is main[p + 1]. A hit advances 2 tokens, a miss 1. Every config
    // runs its own loop; their drafts at one tick share one lm_head read.
    let mut tick = 0usize;
    loop {
        let _q = quiet_lock()?;
        let active: Vec<usize> = (0..sts.len()).filter(|&i| sts[i].p + 1 < rows).collect();
        if active.is_empty() {
            break;
        }
        let mut batch = Vec::with_capacity(active.len() * h);
        for &i in &active {
            let s = &mut sts[i];
            let o = &o_all[s.p * h..(s.p + 1) * h];
            let r = s.cfg.rank;
            let mut hx = o.to_vec();
            let mut vr = vec![0f32; r];
            if s.cfg.kind == "hbias" {
                hx.iter_mut().zip(&s.hb).for_each(|(a, b)| *a += b);
            } else if s.cfg.kind == "lora" {
                for (j, v) in vr.iter_mut().enumerate() {
                    *v = (0..h).map(|i| s.lv[i * r + j] * o[i]).sum();
                }
                for (i, x) in hx.iter_mut().enumerate() {
                    *x += (0..r).map(|j| s.lu[i * r + j] * vr[j]).sum::<f32>();
                }
            }
            batch.extend_from_slice(&hx);
            s.hx = hx;
            s.vr = vr;
        }
        let lg_all = w.matmul(&batch, active.len());
        for (bi, &i) in active.iter().enumerate() {
            let s = &mut sts[i];
            let mut lg = lg_all[bi * vocab..(bi + 1) * vocab].to_vec();
            if s.cfg.kind == "vbias" {
                lg.iter_mut().zip(&s.vb).for_each(|(a, b)| *a += b);
            }
            let d = argmax(&lg) as u32;
            let target = main[s.p + 1];
            let hit = d == target;
            s.passes += 1;
            s.win.0 += 1;
            if hit {
                s.hits += 1;
                s.win.1 += 1;
            }
            let adv = if hit { 2 } else { 1 };
            s.produced += adv;
            if s.produced >= s.next_mark {
                s.windows.push((s.produced, s.win.0, s.win.1));
                s.win = (0, 0);
                s.next_mark += 250;
            }
            if s.cfg.kind != "none" && (!s.cfg.miss_only || !hit) {
                let mx = lg.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mut idx: Vec<usize> = (0..vocab).collect();
                let kk = 1024;
                idx.select_nth_unstable_by(kk, |&x, &y| lg[y].total_cmp(&lg[x]));
                idx.truncate(kk);
                if !idx.contains(&(target as usize)) {
                    idx.push(target as usize);
                }
                let z: f32 = lg.iter().map(|&v| (v - mx).exp()).sum();
                let g: Vec<f32> = idx
                    .iter()
                    .map(|&j| (lg[j] - mx).exp() / z - if j == target as usize { 1.0 } else { 0.0 })
                    .collect();
                let lr = s.cfg.lr;
                if s.cfg.kind == "vbias" {
                    s.ad[0].step(&mut s.vb, &g, lr, Some(&idx));
                } else {
                    let parts: Vec<Vec<f32>> = idx
                        .par_iter()
                        .zip(g.par_iter())
                        .map(|(&j, &gj)| widen(j).into_iter().map(|x| x * gj).collect())
                        .collect();
                    let mut gh = vec![0f32; h];
                    for pv in &parts {
                        gh.iter_mut().zip(pv).for_each(|(a, b)| *a += b);
                    }
                    if s.cfg.kind == "hbias" {
                        s.ad[1].step(&mut s.hb, &gh, lr, None);
                    } else {
                        let r = s.cfg.rank;
                        let o = &o_all[s.p * h..(s.p + 1) * h];
                        let mut ug = vec![0f32; r];
                        for i2 in 0..h {
                            for (j, u) in ug.iter_mut().enumerate() {
                                *u += s.lu[i2 * r + j] * gh[i2];
                            }
                        }
                        let mut gu = vec![0f32; h * r];
                        let mut gv = vec![0f32; h * r];
                        for i2 in 0..h {
                            for j in 0..r {
                                gu[i2 * r + j] = gh[i2] * s.vr[j];
                                gv[i2 * r + j] = o[i2] * ug[j];
                            }
                        }
                        let (a2, rest) = s.ad.split_at_mut(3);
                        a2[2].step(&mut s.lu, &gu, lr, None);
                        rest[0].step(&mut s.lv, &gv, lr, None);
                    }
                }
            }
            s.p += adv;
        }
        tick += 1;
        if tick % 200 == 0 {
            eprintln!("tick {tick}, {:.0} s", t0.elapsed().as_secs_f64());
        }
    }
    let mut results = Vec::new();
    for s in &mut sts {
        if s.win.0 > 0 {
            s.windows.push((s.produced, s.win.0, s.win.1));
        }
        let wpp = s.produced as f64 / s.passes as f64;
        eprintln!(
            "{:18} passes {} hits {} ({:.4}) words/pass {wpp:.4}  windows {:?}",
            s.cfg.name,
            s.passes,
            s.hits,
            s.hits as f64 / s.passes as f64,
            s.windows
                .iter()
                .map(|w| format!("{:.3}", w.2 as f64 / w.1 as f64))
                .collect::<Vec<_>>()
        );
        results.push(json!({
            "config": s.cfg.name, "passes": s.passes, "hits": s.hits,
            "accept": s.hits as f64 / s.passes as f64, "words_per_pass": wpp, "produced": s.produced,
            "windows": s.windows.iter().map(|w| json!({"tokens_end": w.0, "passes": w.1, "hits": w.2,
                "accept": w.2 as f64 / w.1 as f64})).collect::<Vec<_>>(),
        }));
    }
    write(
        outp,
        &json!({"tool": "qwen38_lab learn", "trace": td, "rows": rows, "score_start": ss,
                "seconds": t0.elapsed().as_secs_f64(), "results": results}),
    )
}

fn rows_chip(a: &[String]) -> Result<(), String> {
    use aien_inference_abi::qwen38_omega::spec::RowStats;
    use aien_inference_abi::qwen38_omega::{need_bytes, Qwen38Omega, Qwen38Snapshot};
    let [dir, layers, reps, outp] = a else {
        return Err("rows <ckpt> <layers> <reps> <out.json>".into());
    };
    let dir = Path::new(dir);
    let layers: usize = layers.parse().map_err(|e| format!("layers: {e}"))?;
    let reps: usize = reps.parse().map_err(|e| format!("reps: {e}"))?;
    let full_c = Qwen38Config::from_checkpoint(dir)?;
    let c = full_c.prefix(layers).ok_or("layers out of range")?;
    let max_pos = 256;
    let need = need_bytes(&c, max_pos, 0) as f64 / (1u64 << 30) as f64;
    let mem0 = mem_available_gib();
    if mem0 < need + 12.0 {
        return Err(format!(
            "MemAvailable {mem0:.1} GiB < model {need:.1} + 12 GiB"
        ));
    }
    let mut m = Qwen38Omega::open(dir, Some(layers), max_pos)?;
    let golden = read_json(
        &(std::env::var("HOME").unwrap_or_default() + "/handoffs/qwen38-ref/gen-chat-max-24.json"),
    )?;
    let mut toks = ids(&golden, "prompt_ids")?;
    toks.extend(ids(&golden, "generated_ids")?);
    let base = toks.len() - 9;
    let h = c.hidden;
    // History through one-row passes, then snapshot.
    let mut st = RowStats::default();
    for &tk in &toks[..base] {
        m.exact_rows(&[tk], None, &mut st)?;
    }
    let mut snap = Qwen38Snapshot::default();
    m.snapshot(&mut snap)?;
    let ms = |ns: u128| ns as f64 / 1e6;
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.total_cmp(b));
        v[v.len() / 2]
    };
    let mut per_n = Vec::new();
    for _warm in 0..1 {
        m.restore(&snap)?;
        m.exact_rows(&toks[base..base + 8], None, &mut RowStats::default())?;
    }
    for n in [1usize, 2, 4, 8] {
        let (mut pass, mut head) = (Vec::new(), Vec::new());
        for _ in 0..reps {
            m.restore(&snap)?;
            let t = Instant::now();
            let x = m.exact_rows(&toks[base..base + n], None, &mut RowStats::default())?;
            pass.push(ms(t.elapsed().as_nanos()));
            let t = Instant::now();
            for r in 0..n {
                m.head_token(&x[r * h..(r + 1) * h], &mut RowStats::default())?;
            }
            head.push(ms(t.elapsed().as_nanos()));
        }
        let (pm, hm) = (med(&mut pass), med(&mut head));
        eprintln!("rows {n}: pass ({layers} layers) median {pm:.2} ms, lm_head x{n} {hm:.2} ms");
        per_n.push(json!({"rows": n, "pass_ms_median": pm, "pass_ms": pass, "head_ms_median": hm, "head_ms": head}));
    }
    m.reset()?;
    let mut graph = Vec::new();
    for &tk in &toks[..base + 8] {
        let (_, gs) = m.step(tk)?;
        graph.push(gs.total_ns as f64 / 1e6);
    }
    let gm = med(&mut graph[base..].to_vec());
    eprintln!("decode graph token ({layers} layers) median {gm:.2} ms");
    let p1 = per_n[0]["pass_ms_median"].as_f64().unwrap_or(1.0);
    let h1 = per_n[0]["head_ms_median"].as_f64().unwrap_or(0.0);
    let scale = full_c.layers as f64 / layers as f64;
    let mut table = Vec::new();
    for e in &per_n {
        let n = e["rows"].as_u64().unwrap_or(1) as f64;
        let pm = e["pass_ms_median"].as_f64().unwrap_or(0.0);
        let hm = e["head_ms_median"].as_f64().unwrap_or(0.0);
        let whole = pm * scale + hm;
        let whole1 = p1 * scale + h1;
        table.push(
            json!({"rows": n, "body_ratio_vs_1": pm / p1, "whole_model_ms_est": whole,
            "whole_ratio_vs_1": whole / whole1, "ms_per_answer_token_est": whole / n}),
        );
        eprintln!("rows {n}: body x{:.3} of 1 row; whole-model est {whole:.1} ms per step = x{:.3}, {:.1} ms per answer token", pm / p1, whole / whole1, whole / n);
    }
    write(
        outp,
        &json!({"tool": "qwen38_lab rows", "layers": layers, "reps": reps, "positions_before": base,
        "per_rows": per_n, "graph_token_ms_median": gm, "scaled_to_layers": full_c.layers, "table": table,
        "mem_available_gib_start": mem0, "mem_available_gib_end": mem_available_gib()}),
    )
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let r = match a.get(1).map(String::as_str) {
        Some("mkseq") => mkseq(&a[2..]),
        Some("mkseq-chat") => mkseq_chat(&a[2..]),
        Some("run") => run(&a[2..]),
        Some("mtp") => mtp(&a[2..]),
        Some("learn") => learn(&a[2..]),
        Some("rows") => rows_chip(&a[2..]),
        _ => Err("qwen38_lab mkseq|mkseq-chat|run|mtp|learn|rows ... (see the file header)".into()),
    };
    if let Err(e) = r {
        eprintln!("qwen38_lab: {e}");
        std::process::exit(1);
    }
}
