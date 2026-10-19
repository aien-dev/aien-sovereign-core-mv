use super::*;
use crate::qwen3_moe::{f32_to_bf16, f32_to_fp8_e4m3fn};

fn prand(seed: u64, idx: usize) -> f32 {
    let mut x = seed
        .wrapping_mul(0x9E37_79B9_7F4A_7C15)
        .wrapping_add(idx as u64 + 1);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 32;
    ((x & 0xFFFF) as f32 / 32768.0) - 1.0
}

#[derive(Default)]
struct Mem {
    tensors: HashMap<String, (String, Vec<usize>, Vec<u8>)>,
    seed: u64,
}

impl TensorSource for Mem {
    fn read_into(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[usize],
        dst: &mut [u8],
    ) -> Result<(), String> {
        let (d, s, b) = self.tensors.get(name).ok_or(format!("{name}: missing"))?;
        if d != dtype || s != shape || b.len() != dst.len() {
            return Err(format!("{name}: {d} {s:?} asked {dtype} {shape:?}"));
        }
        dst.copy_from_slice(b);
        Ok(())
    }
    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        Ok(self.tensors.get(name).map(|t| t.0.clone()))
    }
}

impl Mem {
    fn bf16(&mut self, name: &str, shape: &[usize], scale: f32, offset: f32) {
        self.seed += 1;
        let n: usize = shape.iter().product();
        let bytes = (0..n)
            .flat_map(|i| f32_to_bf16(offset + prand(self.seed, i) * scale).to_le_bytes())
            .collect();
        self.tensors
            .insert(name.into(), ("BF16".into(), shape.to_vec(), bytes));
    }
    fn raw(&mut self, name: &str, dtype: &str, shape: &[usize], bytes: Vec<u8>) {
        self.tensors
            .insert(name.into(), (dtype.into(), shape.to_vec(), bytes));
    }
}

fn tiny() -> FlashNextConfig {
    FlashNextConfig {
        hidden: 64,
        layers: 4,
        vocab: 50,
        full_attention: vec![false, false, false, true],
        heads: 4,
        kv_heads: 2,
        head_dim: 32,
        rotary_dim: 8,
        rope_theta: 1e7,
        lin_k_heads: 2,
        lin_v_heads: 4,
        lin_k_dim: 16,
        lin_v_dim: 16,
        conv_kernel: 4,
        gdn_gate: GateAct::Sigmoid,
        eps: 1e-6,
        eos: 1,
        hc: 4,
        hc_rank: 8,
        experts: 8,
        top_k: 3,
        moe_inter: 32,
        shared_inter: 32,
        norm_topk: true,
        idx_heads: 2,
        idx_head_dim: 16,
        idx_budget: 8,
        idx_ratio: 4,
        ple_layers: vec![1],
        ngram: 3,
        heads_per_ngram: 2,
        ngram_base: 50,
        ngram_divisor: 8,
        ple_conv_kernel: 4,
        seed: 1234,
    }
}

fn tiny_source(c: &FlashNextConfig) -> Mem {
    let mut m = Mem::default();
    let (h, l, p) = (c.hidden, c.lanes(), "model.language_model.");
    m.bf16(&format!("{p}embed_tokens.weight"), &[c.vocab, h], 1.0, 0.0);
    m.bf16("lm_head.weight", &[c.vocab, h], 0.1, 0.0);
    let hcw = |m: &mut Mem, q: &str, inject: bool| {
        m.bf16(&format!("{q}.hc_norm.weight"), &[l], 0.1, 0.0);
        m.bf16(
            &format!("{q}.input_mix_weight_down.weight"),
            &[c.hc_rank, l],
            0.1,
            0.0,
        );
        m.bf16(
            &format!("{q}.input_mix_weight_up.weight"),
            &[l, c.hc_rank],
            0.3,
            0.0,
        );
        if inject {
            m.bf16(
                &format!("{q}.block_inject_weight.weight"),
                &[c.hc, l],
                0.1,
                0.0,
            );
        }
    };
    hcw(&mut m, &format!("{p}hyper_connection_mixer"), false);
    for layer in 0..c.layers {
        let lp = format!("{p}layers.{layer}");
        hcw(&mut m, &format!("{lp}.attn_hyper_connection"), true);
        hcw(&mut m, &format!("{lp}.mlp_hyper_connection"), true);
        if c.full_attention[layer] {
            let a = format!("{lp}.self_attn");
            m.bf16(
                &format!("{a}.q_proj.weight"),
                &[2 * c.heads * c.head_dim, h],
                0.2,
                0.0,
            );
            m.bf16(&format!("{a}.k_proj.weight"), &[c.kv_dim(), h], 0.2, 0.0);
            m.bf16(&format!("{a}.v_proj.weight"), &[c.kv_dim(), h], 0.2, 0.0);
            m.bf16(
                &format!("{a}.o_proj.weight"),
                &[h, c.heads * c.head_dim],
                0.1,
                0.0,
            );
            m.bf16(&format!("{a}.q_norm.weight"), &[c.head_dim], 0.1, 0.0);
            m.bf16(&format!("{a}.k_norm.weight"), &[c.head_dim], 0.1, 0.0);
            m.bf16(
                &format!("{a}.indexer.index_qk_proj.weight"),
                &[(c.idx_heads + 1) * c.idx_head_dim, h],
                0.3,
                0.0,
            );
            m.bf16(
                &format!("{a}.indexer.q_layernorm.weight"),
                &[c.idx_head_dim],
                0.1,
                0.0,
            );
            m.bf16(
                &format!("{a}.indexer.k_layernorm.weight"),
                &[c.idx_head_dim],
                0.1,
                0.0,
            );
        } else {
            let a = format!("{lp}.linear_attn");
            m.bf16(
                &format!("{a}.in_proj_qkv.weight"),
                &[c.conv_dim(), h],
                0.2,
                0.0,
            );
            m.bf16(
                &format!("{a}.in_proj_z.weight"),
                &[c.lin_value_dim(), h],
                0.2,
                0.0,
            );
            m.bf16(
                &format!("{a}.in_proj_a.weight"),
                &[c.lin_v_heads, h],
                0.05,
                0.0,
            );
            m.bf16(
                &format!("{a}.in_proj_b.weight"),
                &[c.lin_v_heads, h],
                0.05,
                0.0,
            );
            m.bf16(
                &format!("{a}.conv1d.weight"),
                &[c.conv_dim(), 1, c.conv_kernel],
                0.5,
                0.0,
            );
            m.bf16(&format!("{a}.A_log"), &[c.lin_v_heads], 0.5, 0.0);
            m.bf16(&format!("{a}.dt_bias"), &[c.lin_v_heads], 0.5, 0.0);
            m.bf16(&format!("{a}.norm.weight"), &[c.lin_v_dim], 0.1, 1.0);
            m.bf16(
                &format!("{a}.out_proj.weight"),
                &[h, c.lin_value_dim()],
                0.1,
                0.0,
            );
        }
        let mp = format!("{lp}.mlp");
        m.bf16(&format!("{mp}.gate.weight"), &[c.experts, h], 0.5, 0.0);
        m.bf16(
            &format!("{mp}.shared_expert_gate.weight"),
            &[1, h],
            0.2,
            0.0,
        );
        for (n, k, name) in [
            (c.shared_inter, h, "gate_proj"),
            (c.shared_inter, h, "up_proj"),
            (h, c.shared_inter, "down_proj"),
        ] {
            m.bf16(
                &format!("{mp}.shared_expert.{name}.weight"),
                &[n, k],
                0.2,
                0.0,
            );
        }
        for e in 0..c.experts {
            m.bf16(
                &format!("{mp}.experts.{e}.gate_up_proj"),
                &[2 * c.moe_inter, h],
                0.2,
                0.0,
            );
            m.bf16(
                &format!("{mp}.experts.{e}.down_proj"),
                &[h, c.moe_inter],
                0.2,
                0.0,
            );
        }
        if c.ple_layers.contains(&layer) {
            let q = format!("{lp}.ple");
            let pi = c.ple_layers.iter().position(|&x| x == layer).unwrap();
            let (sizes, offs) = ngram_heads(c, pi);
            let mult = layer_multipliers(c.vocab as u64, c.ngram, pi as u64, c.seed);
            let i64b = |v: Vec<i64>| {
                v.into_iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<u8>>()
            };
            let nh = c.ngram_heads();
            m.raw(
                &format!("{q}.ple_embedding.layer_multipliers"),
                "I64",
                &[c.ngram],
                i64b(mult),
            );
            m.raw(
                &format!("{q}.ple_embedding.ngram_heads_vocab_sizes"),
                "I64",
                &[nh],
                i64b(sizes.iter().map(|&v| v as i64).collect()),
            );
            m.raw(
                &format!("{q}.ple_embedding.ngram_heads_offsets"),
                "I64",
                &[nh],
                i64b(offs.iter().map(|&v| v as i64).collect()),
            );
            m.bf16(
                &format!("{q}.ple_embedding.ngram_embedding.weight_scale"),
                &[1],
                0.0,
                0.05,
            );
            let total: u64 = sizes.iter().sum();
            for id in 0..total {
                let row = (0..c.ngram_row())
                    .map(|j| f32_to_fp8_e4m3fn(prand(id + 77, j) * 8.0))
                    .collect();
                m.raw(&format!("ngram_row.{id}"), "F8_E4M3", &[c.ngram_row()], row);
            }
            m.bf16(&format!("{q}.key_proj.weight"), &[l, h], 0.2, 0.0);
            m.bf16(&format!("{q}.value_proj.weight"), &[h, h], 0.2, 0.0);
            for n in ["norm_key", "norm_query", "norm_conv"] {
                m.bf16(&format!("{q}.{n}.weight"), &[l], 0.1, 0.0);
            }
            // the model initialises this conv to zero; random here so the test sees it
            m.bf16(
                &format!("{q}.conv1d.weight"),
                &[l, 1, c.ple_conv_kernel],
                0.5,
                0.0,
            );
        }
    }
    m
}

#[test]
fn reads_the_flash_next_config() {
    let c =
        FlashNextConfig::from_json(include_str!("../../fixtures/flashnext/config.json")).unwrap();
    assert_eq!(
        (c.hidden, c.layers, c.vocab, c.lanes()),
        (2560, 48, 248320, 10240)
    );
    assert_eq!(c.full_attention.iter().filter(|&&f| f).count(), 12);
    assert!(c.full_attention[3] && !c.full_attention[2] && c.full_attention[47]);
    assert_eq!(
        (c.heads, c.kv_heads, c.head_dim, c.rotary_dim),
        (24, 2, 256, 64)
    );
    assert_eq!(
        (c.experts, c.top_k, c.moe_inter, c.shared_inter),
        (512, 10, 640, 640)
    );
    assert!(c.norm_topk);
    assert_eq!(c.gdn_gate, GateAct::Sigmoid);
    assert_eq!(c.ple_layers, vec![1]);
    assert_eq!(
        (c.ngram_heads(), c.ngram_row(), c.ple_state_len()),
        (16, 160, 9)
    );
    assert_eq!(c.idx_budget / c.idx_ratio, 512);
    assert_eq!(c.seed, 1234);
    assert_eq!(c.conv_dim(), 10240);
}

/// The hash constants this code derives must equal the ones the checkpoint ships (fetched by HTTP range from
/// Qwen/Qwen3.8-Flash-Next-FP8 @ 236dfdf2, values in fixtures/flashnext/ple-constants.json).
#[test]
fn derived_hash_constants_equal_the_shipped_ones() {
    let c =
        FlashNextConfig::from_json(include_str!("../../fixtures/flashnext/config.json")).unwrap();
    let shipped: serde_json::Value =
        serde_json::from_str(include_str!("../../fixtures/flashnext/ple-constants.json")).unwrap();
    let get = |k: &str| -> Vec<i64> {
        shipped[k]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap())
            .collect()
    };
    let mult = layer_multipliers(c.vocab as u64, c.ngram, 0, c.seed);
    assert_eq!(mult, get("layer_multipliers"));
    let (sizes, offs) = ngram_heads(&c, 0);
    assert_eq!(
        sizes.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        get("ngram_heads_vocab_sizes")
    );
    assert_eq!(
        offs.iter().map(|&v| v as i64).collect::<Vec<_>>(),
        get("ngram_heads_offsets")
    );
    let total: u64 = sizes.iter().sum();
    let padded = total.div_ceil(c.ngram_divisor) * c.ngram_divisor;
    assert_eq!(padded, 128 * 2_500_012);
}

#[test]
fn ngram_ids_reset_at_eos() {
    let c = tiny();
    let (sizes, offs) = ngram_heads(&c, 0);
    let mult = layer_multipliers(c.vocab as u64, c.ngram, 0, c.seed);
    let e = c.eos;
    let a = ngram_ids(&c, &mult, &sizes, &offs, &[e, e], &[7, 9, e, 4, 5]);
    let b = ngram_ids(&c, &mult, &sizes, &offs, &[e, e], &[4, 5]);
    let nh = c.ngram_heads();
    assert_eq!(&a[3 * nh..], &b[..]);
    // and the history carries across calls
    let whole = ngram_ids(&c, &mult, &sizes, &offs, &[e, e], &[7, 9, 11, 4]);
    let tail = ngram_ids(&c, &mult, &sizes, &offs, &[9, 11], &[4]);
    assert_eq!(&whole[3 * nh..], &tail[..]);
    // every id inside its head's range
    for (j, &id) in whole.iter().enumerate() {
        let h = j % nh;
        assert!(id >= offs[h] && id < offs[h] + sizes[h]);
    }
    // a bigram head differs when only the previous token differs
    let x = ngram_ids(&c, &mult, &sizes, &offs, &[e, 3], &[4]);
    let y = ngram_ids(&c, &mult, &sizes, &offs, &[e, 5], &[4]);
    assert_ne!(x[0], y[0]);
}

#[test]
fn e2m1_rounding_and_mxfp4_round_trip() {
    assert_eq!(e2m1_rne(0.25), 0); // tie 0 / 0.5 -> even code 0
    assert_eq!(e2m1_rne(0.75), 2); // tie 0.5 / 1 -> code 2
    assert_eq!(e2m1_rne(2.5), 4); // tie 2 / 3 -> code 4
    assert_eq!(e2m1_rne(5.0), 6); // tie 4 / 6 -> code 6
    assert_eq!(e2m1_rne(100.0), 7);
    // values that are exactly e2m1 * 2^e round-trip with zero error
    let mut v = vec![0f32; 2 * 64];
    for (i, x) in v.iter_mut().enumerate() {
        *x = E2M1[i % 16] * if i < 64 { 0.25 } else { 8.0 };
    }
    v[0] = 6.0 * 0.25; // the group max sets X = 2^(floor(log2 1.5) - 2)
    let mut st = QuantStats::default();
    let p = mxfp4_from_f32(2, 64, &v, &mut st);
    let mut row = vec![0f32; 64];
    for r in 0..2 {
        p.row(r, &mut row);
        for c in 0..64 {
            assert_eq!(row[c], v[r * 64 + c], "{r} {c}");
        }
    }
    assert_eq!(st.sq_err, 0.0);
    // an all-zero group: E 127, zero nibbles
    let z = mxfp4_from_f32(1, 32, &[0.0; 32], &mut st);
    let Proj::Mxfp4 { e8, nib, .. } = &z else {
        panic!()
    };
    assert_eq!(e8[0], 127);
    assert!(nib.iter().all(|&b| b == 0));
}

#[test]
fn nvfp4_decodes_by_the_scalar_formula() {
    let (n, k) = (3usize, 64usize);
    let packed: Vec<u8> = (0..n * k / 2).map(|i| (i * 37 % 256) as u8).collect();
    let scales: Vec<u8> = (0..n * k / 16)
        .map(|i| f32_to_fp8_e4m3fn(0.5 + i as f32 * 0.25))
        .collect();
    let p = Proj::Nvfp4 {
        n,
        k,
        packed: packed.clone(),
        scales: scales.clone(),
        global: 0.125,
    };
    let x: Vec<f32> = (0..k).map(|i| prand(3, i)).collect();
    let y = p.matmul(&x, 1);
    for r in 0..n {
        let want: f64 = (0..k)
            .map(|c| {
                let b = packed[r * k / 2 + c / 2];
                let nib = if c % 2 == 0 { b & 15 } else { b >> 4 };
                E2M1[nib as usize] as f64
                    * fp8_e4m3fn_to_f32(scales[r * k / 16 + c / 16]) as f64
                    * 0.125
                    * x[c] as f64
            })
            .sum();
        assert!(
            (y[r] as f64 - want).abs() < 1e-4 * (1.0 + want.abs()),
            "{r}"
        );
    }
}

#[test]
fn gated_residual_matches_the_scalar_formula() {
    let c = tiny();
    let mut m = tiny_source(&c);
    let q = "model.language_model.layers.0.attn_hyper_connection";
    let gr = GatedResidual::read(&mut m, q, &c, true).unwrap();
    let (h, l, hc) = (c.hidden, c.lanes(), c.hc);
    let lanes: Vec<f32> = (0..l).map(|i| prand(11, i) * 3.0).collect();
    let (out, inj) = gr.forward(&c, &lanes, 1);
    let norm = read_vec(&mut m, &format!("{q}.hc_norm.weight"), l).unwrap();
    let w = |m: &mut Mem, n: &str, s: &[usize]| -> Vec<f32> {
        read_bf16_words(m, n, s)
            .unwrap()
            .into_iter()
            .map(bf16_to_f32)
            .collect()
    };
    let down = w(
        &mut m,
        &format!("{q}.input_mix_weight_down.weight"),
        &[c.hc_rank, l],
    );
    let up = w(
        &mut m,
        &format!("{q}.input_mix_weight_up.weight"),
        &[l, c.hc_rank],
    );
    let injw = w(&mut m, &format!("{q}.block_inject_weight.weight"), &[hc, l]);
    let mut nrm = vec![0f64; l];
    for lane in 0..hc {
        let g = &lanes[lane * h..(lane + 1) * h];
        let ms: f64 = g.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / h as f64;
        for j in 0..h {
            nrm[lane * h + j] =
                g[j] as f64 / (ms + 1e-6).sqrt() * (1.0 + norm[lane * h + j] as f64);
        }
    }
    let d: Vec<f64> = (0..c.hc_rank)
        .map(|r| {
            let s: f64 = (0..l).map(|j| down[r * l + j] as f64 * nrm[j]).sum::<f64>() / hc as f64;
            s / (1.0 + (-s).exp())
        })
        .collect();
    for j in 0..h {
        let mut acc = 0f64;
        for lane in 0..hc {
            let i = lane * h + j;
            let u: f64 = (0..c.hc_rank)
                .map(|r| up[i * c.hc_rank + r] as f64 * d[r])
                .sum();
            acc += nrm[i] / (1.0 + (-u).exp());
        }
        let want = acc / hc as f64;
        assert!(
            (out[j] as f64 - want).abs() < 1e-4 * (1.0 + want.abs()),
            "{j} {} {want}",
            out[j]
        );
    }
    for lane in 0..hc {
        let s: f64 = (0..l)
            .map(|j| injw[lane * l + j] as f64 * nrm[j])
            .sum::<f64>()
            / hc as f64;
        let want = 2.0 / (1.0 + (-s).exp());
        assert!((inj[lane] as f64 - want).abs() < 1e-5, "{lane}");
    }
}

#[test]
fn route_takes_top_k_and_renormalises() {
    let c = tiny();
    let logits = [0.1f32, 2.0, -1.0, 2.0, 0.5, 3.0, -2.0, 0.0];
    let (ids, w) = route(&c, &logits);
    assert_eq!(ids, vec![5, 1, 3]); // tie 1/3: the lower id first
    assert!((w.iter().sum::<f32>() - 1.0).abs() < 1e-6);
    let e: Vec<f32> = [3.0f32, 2.0, 2.0].iter().map(|v| v.exp()).collect();
    let s: f32 = e.iter().sum();
    assert!((w[0] - e[0] / s).abs() < 1e-6);
    let mut off = c.clone();
    off.norm_topk = false;
    let (_, w2) = route(&off, &logits);
    assert!(w2.iter().sum::<f32>() < 1.0);
}

/// Under the budget the indexer selects every visible token; past it exactly budget + tail.
#[test]
fn indexer_selects_all_under_the_budget_and_budget_past_it() {
    let c = tiny();
    let d = c.idx_head_dim;
    let kn = vec![0.0f32; d];
    let q: Vec<f32> = (0..c.idx_heads * d).map(|i| prand(5, i)).collect();
    let keys: Vec<f32> = (0..40 * d).map(|i| prand(6, i)).collect();
    for p in 0..10 {
        // visible p+1 <= budget + ratio - 1 = 11 tokens
        assert_eq!(
            indexer_select(&c, &kn, &q, &keys, p),
            (0..=p).collect::<Vec<_>>()
        );
    }
    for p in [11usize, 17, 30, 39] {
        let s = indexer_select(&c, &kn, &q, &keys, p);
        let tail = (p + 1) % c.idx_ratio;
        assert_eq!(s.len(), c.idx_budget + tail, "{p}");
        assert!(s.windows(2).all(|w| w[0] < w[1]));
        if tail > 0 {
            assert_eq!(*s.last().unwrap(), p);
        }
    }
}

/// The indexer drops the blocks a query does not score: a block whose keys align with q wins.
#[test]
fn indexer_picks_the_aligned_blocks() {
    let mut c = tiny();
    c.idx_budget = 4; // one block
    c.rotary_dim = 0; // no turning, so alignment is plain
    let d = c.idx_head_dim;
    let kn = vec![0.0f32; d];
    let q: Vec<f32> = (0..c.idx_heads * d)
        .map(|i| if i % d == 0 { 1.0 } else { 0.0 })
        .collect();
    let mut keys = vec![0f32; 16 * d];
    for tok in 0..16 {
        keys[tok * d + 1] = 1.0; // every key has some norm
    }
    for tok in 8..12 {
        keys[tok * d] = 5.0; // block 2 aligns with q
    }
    let s = indexer_select(&c, &kn, &q, &keys, 14);
    assert_eq!(s, vec![8, 9, 10, 11, 12, 13, 14]);
}

/// One call and token-by-token through the carried state (DeltaNet conv + state, KV + indexer keys, PLE history
/// and dilated conv, positions) give the same logits. The tiny budget makes the indexer mask bite past 11 tokens.
#[test]
fn one_call_equals_token_by_token() {
    let c = tiny();
    let mut whole = FlashNextReference::new(c.clone(), tiny_source(&c));
    let mut steps = FlashNextReference::new(c.clone(), tiny_source(&c));
    let tokens: Vec<u32> = vec![5, 17, 3, 42, 8, 8, 1, 30, 2, 9, 11, 12, 44, 7, 19, 23];
    let mut s1 = whole.new_state();
    let (_, all) = whole.forward(&mut s1, &tokens, None).unwrap();
    let past = whole.selections[0]
        .iter()
        .filter(|s| s.len() < s.last().unwrap() + 1)
        .count();
    assert!(past > 0, "the indexer never masked anything");
    let mut s2 = steps.new_state();
    let (_, mut rows) = steps.forward(&mut s2, &tokens[..3], None).unwrap();
    for &tok in &tokens[3..] {
        rows.extend(steps.forward(&mut s2, &[tok], None).unwrap().1);
    }
    let max = all
        .iter()
        .zip(&rows)
        .map(|(a, b)| (a - b).abs())
        .fold(0f32, f32::max);
    assert!(max < 1e-3, "max logit difference {max}");
    let spread =
        all.iter().cloned().fold(f32::MIN, f32::max) - all.iter().cloned().fold(f32::MAX, f32::min);
    assert!(spread > 1e-3);
    assert!(all.iter().all(|v| v.is_finite()));
}

/// Negative control for the test above: dropping the PLE conv state between calls must change the logits.
#[test]
fn dropping_the_ple_state_is_caught() {
    let c = tiny();
    let mut a = FlashNextReference::new(c.clone(), tiny_source(&c));
    let mut b = FlashNextReference::new(c.clone(), tiny_source(&c));
    let tokens = [5u32, 17, 3, 42, 8];
    let mut s1 = a.new_state();
    let (_, all) = a.forward(&mut s1, &tokens, None).unwrap();
    let mut s2 = b.new_state();
    b.forward(&mut s2, &tokens[..4], None).unwrap();
    s2.ple_conv[0].iter_mut().for_each(|v| *v = 0.0);
    let (_, last) = b.forward(&mut s2, &tokens[4..], None).unwrap();
    let v = c.vocab;
    let max = all[4 * v..]
        .iter()
        .zip(&last)
        .map(|(x, y)| (x - y).abs())
        .fold(0f32, f32::max);
    assert!(max > 1e-3, "PLE state loss not visible ({max})");
}

/// The MXFP4 expert path (the chip's numbers) stays close to bf16 on the tiny model and reports its error.
#[test]
fn mxfp4_experts_stay_close() {
    let c = tiny();
    let mut a = FlashNextReference::new(c.clone(), tiny_source(&c));
    let mut b = FlashNextReference::new(c.clone(), tiny_source(&c));
    b.expert_format = ExpertFormat::Mxfp4;
    let tokens = [5u32, 17, 3];
    let (mut sa, mut sb) = (a.new_state(), b.new_state());
    let (oa, _) = a.forward(&mut sa, &tokens, None).unwrap();
    let (ob, _) = b.forward(&mut sb, &tokens, None).unwrap();
    let num: f64 = oa
        .iter()
        .zip(&ob)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum();
    let na: f64 = oa.iter().map(|x| (*x as f64).powi(2)).sum();
    let nb: f64 = ob.iter().map(|x| (*x as f64).powi(2)).sum();
    let cos = num / (na.sqrt() * nb.sqrt());
    assert!(cos > 0.98, "cosine {cos}");
    let rel = (b.quant.sq_err / b.quant.sq_ref).sqrt();
    assert!(rel > 0.01 && rel < 0.2, "relative rms {rel}");
}

#[test]
fn missing_experts_are_listed() {
    let c = tiny();
    let mut m = tiny_source(&c);
    m.tensors
        .retain(|k, _| !k.contains("layers.0.mlp.experts."));
    let mut r = FlashNextReference::new(c.clone(), m);
    let mut st = r.new_state();
    let err = r.forward(&mut st, &[5, 6], None).unwrap_err();
    assert!(err.starts_with("missing experts: layer 0:"), "{err}");
}
