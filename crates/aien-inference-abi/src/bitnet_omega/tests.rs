use super::test_support::*;
use super::*;
use crate::qwen3_moe::f32_to_bf16;
use crate::qwen3_moe_omega::test_support::MemorySource;

/// `~/models/BitNet-b1.58-2B-4T-bf16/config.json` (sha256 9dbdfeeb..., revision 2766813), verbatim.
const BITNET_2B4T_CONFIG: &str = r#"{
    "architectures": [
      "BitNetForCausalLM"
    ],
    "auto_map": {
      "AutoConfig": "configuration_bitnet.BitNetConfig",
      "AutoModelForCausalLM": "modeling_bitnet.BitNetForCausalLM"
    },
    "bos_token_id": 128000,
    "eos_token_id": 128001,
    "hidden_act": "relu2",
    "hidden_size": 2560,
    "initializer_range": 0.02,
    "intermediate_size": 6912,
    "max_position_embeddings": 4096,
    "model_type": "bitnet",
    "rms_norm_eps": 1e-05,
    "num_attention_heads": 20,
    "num_hidden_layers": 30,
    "num_key_value_heads": 5,
    "rope_theta": 500000.0,
    "tie_word_embeddings": true,
    "torch_dtype": "bfloat16",
    "use_cache": true,
    "vocab_size": 128256,
    "quantization_config": {
      "quant_method": "bitnet",
      "linear_class": "autobitlinear",
      "quantization_mode": "online"
    }
}"#;

/// Its `generation_config.json`, verbatim.
const BITNET_2B4T_GENERATION: &str = r#"{
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128009
  ],
  "do_sample": true,
  "temperature": 0.6,
  "max_length": 4096,
  "top_p": 0.9,
  "transformers_version": "4.40.0.dev0"
}"#;

fn real_config() -> BitNetConfig {
    BitNetConfig::from_json(BITNET_2B4T_CONFIG, Some(BITNET_2B4T_GENERATION)).unwrap()
}

fn variant(
    edit: impl FnOnce(&mut serde_json::Map<String, serde_json::Value>),
) -> Result<BitNetConfig, String> {
    let mut v: serde_json::Value = serde_json::from_str(BITNET_2B4T_CONFIG).unwrap();
    edit(v.as_object_mut().unwrap());
    BitNetConfig::from_json(&v.to_string(), None)
}

fn assert_refused(r: Result<BitNetConfig, String>, needles: &[&str]) {
    let e = r.expect_err("must be refused");
    assert!(e.starts_with("config refused"), "{e}");
    for n in needles {
        assert!(e.contains(n), "refusal {e:?} does not name {n:?}");
    }
}

fn load_tiny(up: &RecordingTernUploader) -> Result<(BitNetOmegaWeights, BitNetLoadReport), String> {
    let c = tiny_config();
    load_bitnet_omega_from(&c, &mut memory_source(&c), up, &mut |_| {})
}

/// The full byte test: every upload against the checkpoint, then the norms the host keeps.
fn byte_test(up: &RecordingTernUploader, w: &BitNetOmegaWeights) -> Result<(), String> {
    let c = tiny_config();
    let recs = up.records.lock().unwrap();
    check_upload_bytes(&c, &mut memory_source(&c), &recs)?;
    check_host_norms(&c, &mut memory_source(&c), w)
}

fn scratch(name: &str) -> std::path::PathBuf {
    let d = std::env::temp_dir().join(format!("aien-bitnet-{name}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

// -------------------------------------------------------------------------------------------
// Config.
// -------------------------------------------------------------------------------------------

#[test]
fn reads_the_bitnet_2b4t_config() {
    let c = real_config();
    assert_eq!(
        (
            c.layers,
            c.hidden,
            c.q_heads,
            c.kv_heads,
            c.head_dim,
            c.intermediate,
            c.vocab
        ),
        (30, 2560, 20, 5, 128, 6912, 128256)
    );
    assert_eq!(
        (c.rms_eps, c.rope_theta, c.max_position),
        (1e-5, 500000.0, 4096)
    );
    assert_eq!(
        (c.bos_token_id, c.eos_token_ids.clone()),
        (128000, vec![128001, 128009])
    );
    assert_eq!(c.shape(TernTensor::Qkv), (3840, 2560));
    assert_eq!(c.shape(TernTensor::O), (2560, 2560));
    assert_eq!(c.shape(TernTensor::GateUp), (13824, 2560));
    assert_eq!(c.shape(TernTensor::Down), (2560, 6912));
    let g = KvGeometry::from(&c);
    assert_eq!(
        (g.layers, g.q_heads, g.kv_heads, g.head_dim, g.rope_theta),
        (30, 20, 5, 128, 500000.0)
    );
    // Device bytes: trits 2,084,044,800 / 4 = 521,011,200 (spec 5) + 210 bf16 scales (420) +
    // e4m3 table 328,335,360 + its 1002 x 20 bf16 scales (40,080): 849,387,060
    // (calc:f379a7e8).
    assert_eq!(c.model_resident_bytes(), 849_387_060);
}

#[test]
fn refuses_other_families_by_name() {
    assert_refused(
        variant(|o| {
            o.insert("model_type".into(), "llama".into());
        }),
        &["model_type", "\"llama\"", "bitnet"],
    );
    assert_refused(
        variant(|o| {
            o.insert("hidden_act".into(), "silu".into());
        }),
        &["hidden_act", "\"silu\"", "relu2"],
    );
    assert_refused(
        variant(|o| {
            o.insert("tie_word_embeddings".into(), false.into());
        }),
        &["tie_word_embeddings", "false"],
    );
    assert_refused(
        variant(|o| {
            o.remove("tie_word_embeddings");
        }),
        &["tie_word_embeddings", "absent"],
    );
    assert_refused(
        variant(|o| {
            o.insert("torch_dtype".into(), "float16".into());
        }),
        &["torch_dtype", "\"float16\"", "bfloat16"],
    );
    assert_refused(
        variant(|o| {
            o["quantization_config"]["quantization_mode"] = "offline".into();
        }),
        &["quantization_config", "offline"],
    );
    assert_refused(
        variant(|o| {
            o["quantization_config"]["quant_method"] = "gptq".into();
        }),
        &["quantization_config", "gptq"],
    );
    assert_refused(
        variant(|o| {
            o.remove("quantization_config");
        }),
        &["quantization_config", "absent"],
    );
    assert_refused(
        variant(|o| {
            o.insert(
                "architectures".into(),
                serde_json::json!(["LlamaForCausalLM"]),
            );
        }),
        &["architectures", "LlamaForCausalLM"],
    );
}

#[test]
fn refuses_what_the_engine_does_not_implement() {
    assert_refused(
        variant(|o| {
            o.insert("sliding_window".into(), 4096.into());
        }),
        &["\"sliding_window\"", "not implemented"],
    );
    assert_refused(
        variant(|o| {
            o.insert(
                "rope_scaling".into(),
                serde_json::json!({"type": "linear", "factor": 2.0}),
            );
        }),
        &["rope_scaling"],
    );
    assert_refused(
        variant(|o| {
            o.insert("attention_bias".into(), true.into());
        }),
        &["attention_bias"],
    );
    assert_refused(
        variant(|o| {
            o.insert("head_dim".into(), 64.into());
        }),
        &["head_dim"],
    );
    assert_refused(
        variant(|o| {
            o.insert("num_key_value_heads".into(), 3.into());
        }),
        &["GQA"],
    );
    assert_refused(
        variant(|o| {
            o.insert("vocab_size".into(), 128255.into());
        }),
        &["word table"],
    );
    assert_refused(
        variant(|o| {
            o.insert("intermediate_size".into(), 6920.into());
        }),
        &["intermediate_size", "16"],
    );
    // Neutral values of the optional keys are taken.
    variant(|o| {
        o.insert("rope_scaling".into(), serde_json::Value::Null);
        o.insert("attention_bias".into(), false.into());
        o.insert("head_dim".into(), 128.into());
    })
    .unwrap();
}

// -------------------------------------------------------------------------------------------
// The ternariser.
// -------------------------------------------------------------------------------------------

fn bf(v: &[f32]) -> Vec<u16> {
    v.iter().map(|&x| f32_to_bf16(x)).collect()
}

#[test]
fn ternariser_is_pytorch_bf16_absmean() {
    // mean |w| = 1: s = 1, w_scale = 1. 0.5 is a tie and rounds to even (0); 1.5 rounds to 2,
    // clamped to 1; -0.5 to 0; -0.75 to -1.
    let w = bf(&[0.5, 1.5, -0.5, -0.75, 1.25, 0.25, 2.0, -1.25]);
    let mut t = vec![0i8; w.len()];
    let s = absmean_ternarise(&w, &mut t).unwrap();
    assert_eq!((s.mean, s.s, s.w_scale), (1.0, 1.0, 1.0));
    assert_eq!(t, [0, 1, 0, -1, 1, 0, 1, -1]);

    // mean 0.3 is not a bf16 value: every step rounds. bf16(0.3) = 0.30078125,
    // s = bf16(1 / 0.30078125) = 3.328125, w_scale = bf16(1 / 3.328125) = 0.30078125.
    let w = bf(&[0.3, -0.3, 0.3, -0.3]);
    let s = absmean_ternarise(&w, &mut [0i8; 4]).unwrap();
    assert_eq!((s.mean, s.s, s.w_scale), (0.30078125, 3.328125, 0.30078125));
    assert_eq!(s.w_scale_bf16(), f32_to_bf16(0.30078125));

    // The clamp: an all-zero projection has mean bf16(1e-5), trits 0.
    let mut t = vec![5i8; 4];
    let s = absmean_ternarise(&[0u16; 4], &mut t).unwrap();
    assert_eq!(s.mean, bf16_to_f32(f32_to_bf16(1e-5)));
    assert_eq!(t, [0; 4]);

    // NaN and infinity are refused by index; lengths must agree.
    let mut w = bf(&[1.0, 2.0, 3.0]);
    w[2] = 0x7fc0;
    assert!(absmean_ternarise(&w, &mut [0; 3])
        .unwrap_err()
        .contains("weight 2"));
    w[2] = 0x7f80;
    assert!(absmean_ternarise(&w, &mut [0; 3])
        .unwrap_err()
        .contains("weight 2"));
    assert!(absmean_ternarise(&bf(&[1.0]), &mut [0; 2]).is_err());
}

/// The trit of `bf16(w * s)` decides at the bf16 product, not at the exact product: with
/// mean 3 (bf16 exact), s = bf16(1/3) = 0.333984375; w = 1.5 gives 0.5009765625 exactly in
/// f32, which bf16 rounds to 0.5, a tie, so the trit is 0 although the exact |w|/mean is 0.5.
#[test]
fn ternariser_rounds_the_product_to_bf16_first() {
    let s = bf16_to_f32(f32_to_bf16(1.0 / 3.0));
    assert_eq!(s, 171.0 / 512.0);
    assert_eq!(bf16_to_f32(f32_to_bf16(1.5 * s)), 0.5);
    assert_eq!(trit_of(f32_to_bf16(1.5), s), 0);
    assert_eq!(trit_of(f32_to_bf16(1.5078125), s), 1);
    assert_eq!(trit_of(f32_to_bf16(-1.5078125), s), -1);
    // The test's own reference agrees on a sweep of every bf16 in [-4, 4] at three scales.
    for s in [171.0f32 / 512.0, 1.0, 3.328125] {
        for b in 0u16..=0xffff {
            let v = bf16_to_f32(b);
            if v.is_finite() && v.abs() <= 4.0 {
                assert_eq!(trit_of(b, s), reference_trit(b, s), "w {v} s {s}");
            }
        }
    }
}

/// omega's `oma_quant_absmean` (`omega/src/algebra/oma_quant.c:6-33` at origin/main 6c1827ba),
/// ported line for line: f32 input, double mean, `scale = (float)mean`, `q = round(w / scale)`
/// half away from zero, clamped.
fn oma_quant_absmean(w: &[f32]) -> (Vec<i8>, f32) {
    let sum: f64 = w.iter().map(|&x| (x as f64).abs()).sum();
    if sum == 0.0 {
        return (vec![0; w.len()], 0.0);
    }
    let s = (sum / w.len() as f64) as f32;
    let q = w
        .iter()
        .map(|&x| ((x as f64) / (s as f64)).round().clamp(-1.0, 1.0) as i8)
        .collect();
    (q, s)
}

/// Where the rules coincide, the trits and the scale equal omega's: the mean is a power of two
/// (so the three bf16 roundings are exact) and no weight sits on the |w| = mean/2 boundary.
/// Where they do not, the difference is pinned: (1) on the boundary omega rounds away from
/// zero, PyTorch (and we) to even, so 0; (2) a mean that is not a bf16 value gives a different
/// scale (omega keeps the f32 mean; we store bf16(1 / bf16(1 / bf16(mean))) as the reference
/// checkpoint does).
#[test]
fn equals_oma_quant_absmean_where_the_rules_coincide() {
    use crate::qwen3_moe_omega::test_support::prand;
    for seed in 0..20u64 {
        // Pairs (a, 4 - a) with a = 2 + m/64, m in 0..128, each with its own random sign: both
        // are bf16 values, |a| + |4 - a| = 4, so the mean is exactly 2 (a power of two). The
        // mean/2 boundary |w| = 1 is a = 3, m = 64: skipped.
        let mut w = Vec::new();
        for i in 0..2048 {
            let m = ((prand(seed, 3 * i) + 1.0) * 64.0) as u32 % 128;
            if m == 64 {
                continue;
            }
            let a = 2.0 + m as f32 / 64.0;
            let sign = |j| if prand(seed, j) < 0.0 { -1.0f32 } else { 1.0 };
            w.push(sign(3 * i + 1) * a);
            w.push(sign(3 * i + 2) * (4.0 - a));
        }
        let bits = bf(&w);
        assert!(
            bits.iter().zip(&w).all(|(&b, &x)| bf16_to_f32(b) == x),
            "the pairs are on the bf16 grid"
        );
        let mut ours = vec![0i8; w.len()];
        let s = absmean_ternarise(&bits, &mut ours).unwrap();
        let (theirs, oma_scale) = oma_quant_absmean(&w);
        assert_eq!(
            (s.mean, s.w_scale, oma_scale),
            (2.0, 2.0, 2.0),
            "seed {seed}"
        );
        assert_eq!(ours, theirs, "seed {seed}");
        assert!(ours.contains(&0) && ours.contains(&1) && ours.contains(&-1));
    }
    // (1) The boundary: mean 2, w = 1 is |w| = mean/2.
    let w = [1.0f32, 3.0, -1.0, -3.0];
    let mut ours = [0i8; 4];
    absmean_ternarise(&bf(&w), &mut ours).unwrap();
    let (theirs, _) = oma_quant_absmean(&w);
    assert_eq!(ours, [0, 1, 0, -1]);
    assert_eq!(theirs, [1, 1, -1, -1]);
    // (2) A mean that is not a bf16 value.
    let w = [0.3f32, -0.3, 0.3, -0.3];
    let s = absmean_ternarise(&bf(&w), &mut [0i8; 4]).unwrap();
    let (_, oma_scale) =
        oma_quant_absmean(&bf(&w).iter().map(|&b| bf16_to_f32(b)).collect::<Vec<_>>());
    assert_eq!(oma_scale, 0.30078125);
    assert_eq!(s.w_scale, 0.30078125);
    let w = [0.1f32, -0.2, 0.3, -0.45];
    let s = absmean_ternarise(&bf(&w), &mut [0i8; 4]).unwrap();
    let wf: Vec<f32> = bf(&w).iter().map(|&b| bf16_to_f32(b)).collect();
    let (_, oma_scale) = oma_quant_absmean(&wf);
    assert_ne!(s.w_scale, oma_scale, "bf16 steps change the scale here");
}

// -------------------------------------------------------------------------------------------
// The packer.
// -------------------------------------------------------------------------------------------

#[test]
fn packs_omegas_trit_code_sixteen_per_word() {
    let mut trits = vec![0i8; 32];
    trits[0] = 1; // bits 0..1 = 0b01
    trits[1] = -1; // bits 2..3 = 0b10
    trits[15] = -1; // bits 30..31 = 0b10
    trits[16] = 1; // second word, bits 0..1
    let mut words = [0u32; 2];
    pack_trits(&trits, 32, &mut words).unwrap();
    assert_eq!(words, [0b1001 | (0b10 << 30), 0b01]);
    // Every word decodes back.
    let m = HostTernMatrix::new(
        1,
        32,
        words.to_vec(),
        vec![TernSegment {
            part: TernPart::O,
            rows: 1,
            scale_bf16: f32_to_bf16(1.0),
        }],
    )
    .unwrap();
    assert_eq!(m.row(0), trits);
    assert_eq!(stored_row(&m, 0), trits);
    assert!(pack_trits(&trits, 24, &mut words).is_err());
    assert!(pack_trits(&trits, 32, &mut [0u32; 3]).is_err());
}

#[test]
fn trit_grid_is_validated_no_0b11() {
    assert_eq!(
        find_invalid_code(&[0x9999_9999, 0x5555_5555, 0xaaaa_aaaa]),
        None
    );
    assert_eq!(find_invalid_code(&[0, 0b11 << 6]), Some((1, 3)));
    let seg = |rows| TernSegment {
        part: TernPart::O,
        rows,
        scale_bf16: f32_to_bf16(0.5),
    };
    // 2 rows of 32 trits = 4 words; word 3 trit 5 is row 1 column 21.
    let mut words = vec![0u32; 4];
    check_ternary_upload(2, 32, &words, &[seg(2)]).unwrap();
    words[3] = 0b11 << 10;
    let e = check_ternary_upload(2, 32, &words, &[seg(2)]).unwrap_err();
    assert!(e.contains("row 1 column 21") && e.contains("0b11"), "{e}");
    words[3] = 0;
    assert!(check_ternary_upload(2, 32, &words[..3], &[seg(2)]).is_err());
    assert!(check_ternary_upload(2, 32, &words, &[seg(1)])
        .unwrap_err()
        .contains("cover 1 rows"));
    let zero = TernSegment {
        scale_bf16: 0,
        ..seg(2)
    };
    assert!(check_ternary_upload(2, 32, &words, &[zero])
        .unwrap_err()
        .contains("finite and positive"));
    let nan = TernSegment {
        scale_bf16: 0x7fc0,
        ..seg(2)
    };
    assert!(check_ternary_upload(2, 32, &words, &[nan]).is_err());
    // The host matrix refuses the same way.
    words[0] = 0b11;
    assert!(HostTernMatrix::new(2, 32, words, vec![seg(2)]).is_err());
}

#[test]
fn host_gemv_is_the_exact_ternary_sum() {
    use crate::qwen3_moe_omega::test_support::prand;
    let (n, k) = (48, 64);
    let trits: Vec<i8> = (0..n * k)
        .map(|i| (prand(7, i) * 1.5).round() as i8)
        .collect();
    let mut words = vec![0u32; n * k / 16];
    pack_trits(&trits, k, &mut words).unwrap();
    let m = HostTernMatrix::new(
        n,
        k,
        words,
        vec![TernSegment {
            part: TernPart::Down,
            rows: n,
            scale_bf16: f32_to_bf16(1.0),
        }],
    )
    .unwrap();
    let x: Vec<i8> = (0..k).map(|i| (prand(8, i) * 127.0) as i8).collect();
    let mut acc = vec![0i32; n];
    m.gemv_i8(&x, &mut acc).unwrap();
    for r in 0..n {
        let want: i32 = (0..k).map(|c| trits[r * k + c] as i32 * x[c] as i32).sum();
        assert_eq!(acc[r], want, "row {r}");
    }
    assert!(m.gemv_i8(&x[..k - 1], &mut acc).is_err());
}

// -------------------------------------------------------------------------------------------
// The 8-bit word table.
// -------------------------------------------------------------------------------------------

#[test]
fn fp8_table_is_the_omega_study_rule() {
    let (n, k) = (128, 256);
    let mut t = vec![0u16; n * k];
    // Block (0, 0): amax 448 -> scale 1.0; values land on e4m3 codes directly.
    t[0] = f32_to_bf16(448.0);
    t[1] = f32_to_bf16(-1.0);
    t[2] = f32_to_bf16(-0.0);
    t[3] = f32_to_bf16(1.0625); // between 1.0 and 1.125: a tie, to the even code (1.0)
                                // Block (0, 1) stays all zero: scale 1.0, codes 0.
    let (mut w, mut s) = (vec![0xffu8; n * k], vec![0u16; 2]);
    build_fp8_table(&t, n, k, &mut w, &mut s).unwrap();
    assert_eq!(s, [f32_to_bf16(1.0), f32_to_bf16(1.0)]);
    assert_eq!(&w[..4], &[0x7e, 0xb8, 0x00, 0x38]);
    assert!(w[128..256].iter().all(|&b| b == 0));
    // amax 1.0: scale bf16(1/448) = 0.002227783203125; 1.0 / scale = 448.88 saturates at 448.
    let mut t2 = vec![0u16; n * k];
    t2[0] = f32_to_bf16(1.0);
    build_fp8_table(&t2, n, k, &mut w, &mut s).unwrap();
    assert_eq!(bf16_to_f32(s[0]), 73.0 / 32768.0);
    assert_eq!(w[0], 0x7e);
    // Shape and NaN refusals.
    assert!(build_fp8_table(&t, n, k + 1, &mut w, &mut s).is_err());
    t[5] = 0x7fc0;
    assert!(build_fp8_table(&t, n, k, &mut w, &mut s)
        .unwrap_err()
        .contains("element 5"));
}

// -------------------------------------------------------------------------------------------
// The loader on the tiny model.
// -------------------------------------------------------------------------------------------

#[test]
fn tiny_load_shapes_byte_counts_and_order() {
    let up = RecordingTernUploader::keeping();
    let (w, report) = load_tiny(&up).unwrap();
    let c = tiny_config();
    assert_eq!(report.uploads, 4 * c.layers + 1);
    assert_eq!(report.resident_bytes, c.model_resident_bytes());
    // 2 layers x (qkv 512x256 + o 256x256 + gate_up 768x256 + down 256x384) / 4 + 7 scales x 2,
    // plus the 256x256 e4m3 table and its 2x2 scales.
    let per_layer = (512 * 256 + 256 * 256 + 768 * 256 + 256 * 384) / 4 + 14;
    assert_eq!(report.resident_bytes, 2 * per_layer + 256 * 256 + 8);
    assert_eq!(
        report.host_bytes,
        2 * 256 * 256 + 4 * 256 + 2 * 4 * (256 + 256 + 256 + 384)
    );
    assert_eq!(w.host_bytes(), report.host_bytes);
    assert_eq!(report.scales.len(), 2);
    let recs = up.records.lock().unwrap();
    let order: Vec<_> = recs.iter().map(|r| (r.layer, r.what, r.n, r.k)).collect();
    use TernTensor::*;
    assert_eq!(
        order,
        [
            (Some(0), Some(Qkv), 512, 256),
            (Some(0), Some(O), 256, 256),
            (Some(0), Some(GateUp), 768, 256),
            (Some(0), Some(Down), 256, 384),
            (Some(1), Some(Qkv), 512, 256),
            (Some(1), Some(O), 256, 256),
            (Some(1), Some(GateUp), 768, 256),
            (Some(1), Some(Down), 256, 384),
            (None, None, 256, 256),
        ]
    );
    assert_eq!(w.embed_row(5).unwrap().len(), 256);
    assert!(w.embed_row(256).is_err());
}

#[test]
fn byte_test_passes_and_stacks_trits_and_scales_q_k_v_and_gate_up() {
    let up = RecordingTernUploader::keeping();
    let (w, report) = load_tiny(&up).unwrap();
    byte_test(&up, &w).unwrap();
    // One scale per projection, in stacking order, each its own value.
    let segs = w.layers[0].qkv.segments();
    let parts: Vec<_> = segs.iter().map(|s| (s.part, s.rows)).collect();
    assert_eq!(
        parts,
        [(TernPart::Q, 256), (TernPart::K, 128), (TernPart::V, 128)]
    );
    let gu: Vec<_> = w.layers[0]
        .gate_up
        .segments()
        .iter()
        .map(|s| (s.part, s.rows))
        .collect();
    assert_eq!(gu, [(TernPart::Gate, 384), (TernPart::Up, 384)]);
    let sc = report.scales[0];
    assert!(sc[0].w_scale != sc[1].w_scale && sc[1].w_scale != sc[2].w_scale);
    assert!(sc[4].w_scale != sc[5].w_scale);
    for (seg, s) in segs.iter().zip(&sc[..3]) {
        assert_eq!(seg.scale_bf16, s.w_scale_bf16());
    }
}

#[test]
fn negative_control_kv_row_swap_fails_the_byte_test() {
    let up = RecordingTernUploader::swapping_kv_rows(&tiny_config());
    let (w, _) = load_tiny(&up).unwrap();
    let e = byte_test(&up, &w).unwrap_err();
    assert!(e.contains("qkv") && e.contains("stored row 256"), "{e}");
}

#[test]
fn negative_control_plane_swap_fails_the_byte_test() {
    let up = RecordingTernUploader::with(TernMutant::PlaneSwap);
    let (w, _) = load_tiny(&up).unwrap();
    let e = byte_test(&up, &w).unwrap_err();
    assert!(e.contains("stored row 0"), "{e}");
}

#[test]
fn negative_control_kv_scale_swap_fails_the_byte_test() {
    let up = RecordingTernUploader::with(TernMutant::SwapKvScales);
    let (w, _) = load_tiny(&up).unwrap();
    let e = byte_test(&up, &w).unwrap_err();
    assert!(e.contains("segment K") && e.contains("scale"), "{e}");
}

#[test]
fn negative_control_word_table_byte_flip_fails_the_byte_test() {
    let up = RecordingTernUploader::with(TernMutant::FlipTableByte);
    let (w, _) = load_tiny(&up).unwrap();
    let e = byte_test(&up, &w).unwrap_err();
    assert!(e.contains("table"), "{e}");
}

#[test]
fn negative_control_input_post_norm_swap_fails_the_byte_test() {
    let up = RecordingTernUploader::keeping();
    let (mut w, _) = load_tiny(&up).unwrap();
    byte_test(&up, &w).unwrap();
    let l = &mut w.layers[1];
    std::mem::swap(&mut l.input_norm, &mut l.post_norm);
    let e = byte_test(&up, &w).unwrap_err();
    assert!(e.contains("model.layers.1.input_layernorm.weight"), "{e}");
}

#[test]
fn malformed_or_empty_eos_lists_are_refused_by_name() {
    let gen = |eos: serde_json::Value| {
        let mut g: serde_json::Value = serde_json::from_str(BITNET_2B4T_GENERATION).unwrap();
        g["eos_token_id"] = eos;
        BitNetConfig::from_json(BITNET_2B4T_CONFIG, Some(&g.to_string()))
    };
    assert_refused(
        gen(serde_json::json!(["128001"])),
        &["generation_config.json eos_token_id", "128001"],
    );
    assert_refused(
        gen(serde_json::json!("eot")),
        &["generation_config.json eos_token_id"],
    );
    assert_refused(gen(serde_json::json!([])), &["eos_token_id", "empty"]);
    assert_refused(
        variant(|o| {
            o.insert("eos_token_id".into(), serde_json::json!(-1));
        }),
        &["eos_token_id", "-1"],
    );
    assert_eq!(
        gen(serde_json::json!(128009)).unwrap().eos_token_ids,
        [128009]
    );
}

#[test]
fn unexpected_or_missing_tensors_are_refused_by_name() {
    let c = tiny_config();
    assert_eq!(real_config().tensor_names().len(), 332);
    let base = tensors(&c);
    for (label, edit, needle) in [
        ("bias", 0, "model.layers.0.self_attn.q_proj.bias"),
        ("scale", 1, "model.layers.1.mlp.down_proj.weight_scale"),
        (
            "missing",
            2,
            "model.layers.1.self_attn.attn_sub_norm.weight",
        ),
    ] {
        let dir = scratch(&format!("names-{label}"));
        let mut all = base.clone();
        match edit {
            0 | 1 => {
                let last = all[all.len() - 1].clone();
                all.push((needle.to_string(), last.1, last.2, last.3));
            }
            _ => all.retain(|t| t.0 != needle),
        }
        write_checkpoint_of(&dir, &c, all).unwrap();
        let e = load_bitnet_omega(&dir, &RecordingTernUploader::record_only(), &mut |_| {})
            .unwrap_err();
        assert!(
            e.contains("checkpoint tensors refused") && e.contains(needle),
            "{label}: {e}"
        );
        std::fs::remove_dir_all(&dir).unwrap();
    }
}

#[test]
fn tiny_round_trip_from_disk_equals_memory() {
    let c = tiny_config();
    let dir = scratch("roundtrip");
    write_checkpoint(&dir, &c).unwrap();
    assert_eq!(BitNetConfig::from_checkpoint(&dir).unwrap(), c);
    let disk_up = RecordingTernUploader::keeping();
    let (disk, _) = load_bitnet_omega(&dir, &disk_up, &mut |_| {}).unwrap();
    let mem_up = RecordingTernUploader::keeping();
    let (mem, _) = load_tiny(&mem_up).unwrap();
    // The disk load passes the byte test against the file itself.
    let mut file = SafetensorsFile::open(&dir.join("model.safetensors")).unwrap();
    check_upload_bytes(&c, &mut file, &disk_up.records.lock().unwrap()).unwrap();
    assert_eq!(disk.embed_bf16, mem.embed_bf16);
    assert_eq!(disk.final_norm, mem.final_norm);
    let (dr, mr) = (
        disk_up.records.lock().unwrap(),
        mem_up.records.lock().unwrap(),
    );
    for (a, b) in dr.iter().zip(mr.iter()) {
        assert_eq!(a.segments, b.segments);
        match (&a.matrix, &b.matrix) {
            (Some(x), Some(y)) => assert_eq!(x.words(), y.words()),
            _ => {
                let (x, y) = (a.table.as_ref().unwrap(), b.table.as_ref().unwrap());
                assert_eq!((x.weights(), x.scales()), (y.weights(), y.scales()));
            }
        }
    }
    std::fs::remove_dir_all(&dir).unwrap();
}

fn load_err(edit: impl FnOnce(&mut MemorySource)) -> String {
    let c = tiny_config();
    let mut m = memory_source(&c);
    edit(&mut m);
    load_bitnet_omega_from(&c, &mut m, &RecordingTernUploader::keeping(), &mut |_| {}).unwrap_err()
}

#[test]
fn bad_checkpoints_are_refused_by_name() {
    let q = TernPart::K.tensor_name(1);
    // A tensor stored in another dtype.
    let e = load_err(|m| {
        let t = m.tensors.get_mut(&q).unwrap();
        t.0 = "F16".into();
    });
    assert!(e.contains(&q) && e.contains("F16"), "{e}");
    // A NaN weight.
    let e = load_err(|m| {
        let t = m.tensors.get_mut(&q).unwrap();
        t.2[6] = 0xc0;
        t.2[7] = 0x7f;
    });
    assert!(e.contains(&q) && e.contains("weight 3"), "{e}");
    // A missing sub-norm.
    let sub = "model.layers.0.mlp.ffn_sub_norm.weight";
    let e = load_err(|m| {
        m.tensors.remove(sub);
    });
    assert!(e.contains(sub), "{e}");
    // An untied head in a tied checkpoint.
    let e = load_err(|m| {
        let embed = m.tensors["model.embed_tokens.weight"].clone();
        m.tensors.insert("lm_head.weight".into(), embed);
    });
    assert!(e.contains("lm_head.weight"), "{e}");
    // A wrong shape.
    let e = load_err(|m| {
        m.tensors.get_mut(&q).unwrap().1 = vec![256, 128];
    });
    assert!(e.contains(&q), "{e}");
}

// -------------------------------------------------------------------------------------------
// Memory: the RSS gate, measured.
// -------------------------------------------------------------------------------------------

/// Fills every read with a finite pattern (0x3c bytes: bf16 0x3c3c, about 0.0115), so a
/// full-width layer loads without a checkpoint on disk.
struct PatternSource;

impl TensorSource for PatternSource {
    fn read_into(
        &mut self,
        _: &str,
        dtype: &str,
        _: &[usize],
        dst: &mut [u8],
    ) -> Result<(), String> {
        assert_eq!(dtype, "BF16");
        dst.fill(0x3c);
        Ok(())
    }
    fn dtype_of(&mut self, name: &str) -> Result<Option<String>, String> {
        Ok((name != "lm_head.weight").then(|| "BF16".into()))
    }
}

/// Three layers at the 2B-4T widths with a 16k vocabulary (embedding 84 MB).
fn rss_config() -> BitNetConfig {
    let mut c = real_config();
    c.layers = 3;
    c.vocab = 16384;
    (c.bos_token_id, c.eos_token_ids) = (1, vec![2]);
    c
}

const RSS_CHILD_ENV: &str = "AIEN_BITNET_RSS_CHILD";
const RSS_CHILD_TEST: &str = "bitnet_omega::tests::rss_probe_child";

/// The child side of [`measured_peak_growth_in_child`]; does nothing unless
/// `AIEN_BITNET_RSS_CHILD=<streaming|f32image|f32qkv>` is set, then prints `BITNET-RSS-GROWTH <bytes>`.
#[test]
#[ignore = "child process of the measured peak RSS gate; started by it"]
fn rss_probe_child() {
    let Ok(case) = std::env::var(RSS_CHILD_ENV) else {
        return;
    };
    let up = match case.as_str() {
        "streaming" => RecordingTernUploader::record_only(),
        "f32image" => RecordingTernUploader::keeping_f32_image(),
        "f32qkv" => RecordingTernUploader {
            keep: Keep::F32CopyOf(TernTensor::Qkv),
            ..Default::default()
        },
        other => panic!("unknown case {other}"),
    };
    if std::fs::write("/proc/self/clear_refs", "5").is_err() {
        println!("BITNET-RSS-GROWTH skipped");
        return;
    }
    let before = crate::qwen3_moe_omega::current_rss_bytes().unwrap();
    let (w, report) =
        load_bitnet_omega_from(&rss_config(), &mut PatternSource, &up, &mut |_| {}).unwrap();
    drop(w);
    println!(
        "BITNET-RSS-GROWTH {}",
        report.peak_rss_bytes.unwrap().saturating_sub(before)
    );
}

fn measured_peak_growth_in_child(case: &str) -> Option<u64> {
    let out = std::process::Command::new(std::env::current_exe().unwrap())
        .args([
            RSS_CHILD_TEST,
            "--ignored",
            "--exact",
            "--nocapture",
            "--test-threads=1",
        ])
        .env(RSS_CHILD_ENV, case)
        .output()
        .unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success(),
        "RSS child {case} failed:\n{stdout}\n{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let value = stdout
        .lines()
        .find_map(|l| l.split_once("BITNET-RSS-GROWTH ").map(|(_, v)| v.trim()))
        .unwrap_or_else(|| panic!("RSS child {case} printed no measurement:\n{stdout}"));
    (value != "skipped").then(|| value.parse().unwrap())
}

/// The memory gate measured: three full-width layers must raise peak RSS by at most the
/// embedding plus the larger loader phase ([`bitnet_loader_transient_bytes`]) plus 24 MB, so a
/// per-layer leak or one stray f32 copy of a matrix shows. Controls, each must break the same
/// bound: an uploader that keeps an f32 image of every matrix and of the table (what a naive
/// host reference would hold), and one that keeps an f32 copy of layer 0's qkv only (39 MB).
#[test]
fn measured_peak_rss_is_the_embedding_plus_the_buffers() {
    let c = rss_config();
    // Slack 24 MB: below the 39 MB of one stray f32 copy of qkv, the second-smallest matrix.
    let bound = (c.embed_host_bytes() + bitnet_loader_transient_bytes(&c) + (24 << 20)) as u64;
    let Some(streaming) = measured_peak_growth_in_child("streaming") else {
        assert!(
            std::env::var_os("CI").is_none(),
            "no /proc/self/clear_refs under CI: the measured peak RSS gate cannot run"
        );
        println!("no /proc/self/clear_refs: measured peak RSS check skipped");
        return;
    };
    println!("BITNET-RSS streaming growth {streaming} bound {bound}");
    assert!(
        streaming <= bound,
        "streaming load grew peak RSS by {streaming}, bound {bound}"
    );
    for case in ["f32image", "f32qkv"] {
        let grown = measured_peak_growth_in_child(case).unwrap();
        println!("BITNET-RSS {case} control growth {grown} bound {bound}");
        assert!(
            grown > bound,
            "control {case}: grew peak RSS by only {grown}, bound {bound}"
        );
    }
}

// -------------------------------------------------------------------------------------------
// The real checkpoint (host only, ignored by default).
// -------------------------------------------------------------------------------------------

fn real_dir() -> std::path::PathBuf {
    std::env::var_os("AIEN_BITNET_DIR")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| Path::new(env!("HOME")).join("models/BitNet-b1.58-2B-4T-bf16"))
}

/// The reference int2 checkpoint's stored scales (spec 6 step 2, read from
/// `converted/model_state_int2.pt`; printed there rounded to 6 decimals from exact bf16):
/// `[layer][q, k, v, o, gate, up, down]`.
const SPEC_SCALES: [(usize, [f32; 7]); 3] = [
    (
        0,
        [
            1.21875, 1.804688, 2.296875, 0.960938, 1.554688, 1.828125, 2.15625,
        ],
    ),
    (
        15,
        [1.6875, 2.71875, 4.5, 2.8125, 2.171875, 2.125, 2.609375],
    ),
    (
        29,
        [
            1.125, 1.648438, 1.78125, 0.957031, 2.828125, 2.609375, 2.28125,
        ],
    ),
];

/// The real BitNet 2B-4T master through the streaming loader and the record-only fake: sha256
/// pinned, per-tensor bytes, total resident, peak RSS, wall time, and the 21 scales the spec
/// lists from the reference int2 checkpoint. Run with
/// `cargo test -p aien-inference-abi --release --lib bitnet_omega::tests::real_bitnet -- --ignored --nocapture --test-threads=1`
/// (`AIEN_BITNET_DIR` overrides `~/models/BitNet-b1.58-2B-4T-bf16`).
#[test]
#[ignore = "reads the 4.8 GB BitNet 2B-4T master"]
fn real_bitnet_streaming_load_host_only() {
    let dir = real_dir();
    let sha = sha256_file(&dir.join("model.safetensors")).unwrap();
    assert_eq!(sha, BITNET_2B4T_SAFETENSORS_SHA256, "not the pinned master");
    println!("BITNET-LOAD safetensors_sha256={sha}");
    assert_eq!(
        checkpoint_tensor_names(&dir).unwrap().len(),
        332,
        "spec 1.1: 332 tensors"
    );
    let rss_before = crate::qwen3_moe_omega::current_rss_bytes().unwrap();
    let up = RecordingTernUploader::record_only();
    let (w, report) = load_bitnet_omega(&dir, &up, &mut |_| {}).unwrap();
    let c = &w.config;
    assert_eq!(*c, real_config());
    for what in TernTensor::ALL {
        let (n, k) = c.shape(what);
        println!(
            "BITNET-LOAD tensor {:<8} [{n}][{k}] bytes_each={} count={}",
            what.name(),
            c.ternary_bytes(what),
            c.layers
        );
    }
    let peak = report.peak_rss_bytes.unwrap();
    println!(
        "BITNET-LOAD layout={TERN_LAYOUT_ID} resident_bytes={} uploads={} host_bytes={} embed_bytes={} \
         transient_bytes={} rss_before={rss_before} peak_rss={peak} peak_over_start={} seconds={:.2}",
        report.resident_bytes,
        report.uploads,
        report.host_bytes,
        c.embed_host_bytes(),
        report.transient_bytes,
        peak - rss_before,
        report.seconds
    );
    for (l, s) in report.scales.iter().enumerate() {
        let v: Vec<String> = s.iter().map(|x| format!("{}", x.w_scale)).collect();
        println!(
            "BITNET-SCALES layer {l} q,k,v,o,gate,up,down = {}",
            v.join(",")
        );
    }
    // About 0.85 GB on the device (design calc:f4fe034c; exact count calc:f379a7e8).
    assert_eq!(report.resident_bytes, 849_387_060);
    assert_eq!(report.uploads, 4 * 30 + 1);
    assert!(
        (peak - rss_before) as usize
            <= c.embed_host_bytes() + bitnet_loader_transient_bytes(c) + (64 << 20),
        "peak RSS grew {} bytes",
        peak - rss_before
    );
    for (layer, want) in SPEC_SCALES {
        for (j, (got, want)) in report.scales[layer].iter().zip(want).enumerate() {
            assert!(
                (got.w_scale - want).abs() <= 1e-6,
                "layer {layer} part {j}: our scale {} is not the reference {want}",
                got.w_scale
            );
        }
    }
}

/// Informational, test tool only: how many of our trits agree with Microsoft's I2_S gguf
/// (parsed here, nothing of theirs is run). Prints `BITNET-GGUF` lines per projection kind and
/// in total, for Q and K both in checkpoint row order and in the interleaved-RoPE row order,
/// with the gguf block size that decodes (128 or 64). Run like the test above
/// (`AIEN_BITNET_GGUF` overrides `~/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf`).
#[test]
#[ignore = "reads the 4.8 GB master and the 1.2 GB gguf"]
fn real_gguf_trit_agreement() {
    let gguf = std::env::var_os("AIEN_BITNET_GGUF")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| {
            Path::new(env!("HOME")).join("models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf")
        });
    let sha = sha256_file(&gguf).unwrap();
    println!("BITNET-GGUF sha256={sha}");
    let header = super::gguf_ref::read_header(&gguf).unwrap();
    let up = RecordingTernUploader::keeping();
    let (w, report) = load_bitnet_omega(&real_dir(), &up, &mut |_| {}).unwrap();
    let c = &w.config;
    let hd = c.head_dim;
    let gguf_name = |part: TernPart, l: usize| {
        let n = match part {
            TernPart::Q => "attn_q",
            TernPart::K => "attn_k",
            TernPart::V => "attn_v",
            TernPart::O => "attn_output",
            TernPart::Gate => "ffn_gate",
            TernPart::Up => "ffn_up",
            TernPart::Down => "ffn_down",
        };
        format!("blk.{l}.{n}.weight")
    };
    // Block size: whichever decodes layer 0 q closer to ours.
    let mut totals: std::collections::BTreeMap<&str, (u64, u64, u64)> = Default::default();
    let mut block = 0usize;
    for l in 0..c.layers {
        for what in TernTensor::ALL {
            let m = w.layers[l].matrix(what);
            let rec_words = up.records.lock().unwrap()[l * 4 + what as usize]
                .matrix
                .clone()
                .unwrap();
            let k = m.k();
            let mut row0 = 0;
            for (si, (part, rows)) in c.parts(what).into_iter().enumerate() {
                let t = &header[&gguf_name(part, l)];
                let ours: Vec<i8> = (0..rows).flat_map(|r| rec_words.row(row0 + r)).collect();
                if block == 0 {
                    let agree = |b| {
                        let (g, _) = super::gguf_ref::read_i2s(&gguf, t, b).unwrap();
                        g.iter().zip(&ours).filter(|(a, b)| a == b).count()
                    };
                    let (a128, a64) = (agree(128), agree(64));
                    println!(
                        "BITNET-GGUF block choice: agree@128={a128} agree@64={a64} of {}",
                        ours.len()
                    );
                    block = if a128 >= a64 { 128 } else { 64 };
                }
                let (g, gscale) = super::gguf_ref::read_i2s(&gguf, t, block).unwrap();
                let count = |perm: bool| {
                    let mut same = 0u64;
                    for r in 0..rows {
                        // Interleaved order: gguf row j of a head is HF row (j % 2) * hd/2 + j / 2.
                        let src = if perm {
                            let (h, j) = (r / hd, r % hd);
                            h * hd + (j % 2) * (hd / 2) + j / 2
                        } else {
                            r
                        };
                        same += g[r * k..(r + 1) * k]
                            .iter()
                            .zip(&ours[src * k..(src + 1) * k])
                            .filter(|(a, b)| a == b)
                            .count() as u64;
                    }
                    same
                };
                let plain = count(false);
                let permuted = if matches!(part, TernPart::Q | TernPart::K) {
                    count(true)
                } else {
                    plain
                };
                let e = totals.entry(part.name()).or_default();
                e.0 += plain;
                e.1 += permuted;
                e.2 += (rows * k) as u64;
                if l == 0 {
                    println!(
                        "BITNET-GGUF layer 0 {} gguf_scale={gscale} our_w_scale={} plain={plain} rope_interleaved={permuted} of {}",
                        part.name(),
                        report.scales[0][[0, 1, 2, 3, 4, 5, 6][match what {
                            TernTensor::Qkv => si,
                            TernTensor::O => 3,
                            TernTensor::GateUp => 4 + si,
                            TernTensor::Down => 6,
                        }]]
                        .w_scale,
                        rows * k
                    );
                }
                row0 += rows;
            }
        }
    }
    let (mut best, mut all) = (0u64, 0u64);
    for (part, (plain, permuted, n)) in &totals {
        println!("BITNET-GGUF {part}: plain={plain} rope_interleaved={permuted} of {n}");
        best += plain.max(permuted);
        all += n;
    }
    println!(
        "BITNET-GGUF block={block} total_agree={best} of {all} disagree={}",
        all - best
    );
    // What NOTES.md states from this run: at least 97 % agree overall (measured 97.2514 %,
    // calc:736bea70), the I2_S block is 128, and Q and K agree in checkpoint row order, not in
    // the interleaved-RoPE order (the gguf keeps the Hugging Face rows).
    assert_eq!(block, 128);
    assert!(
        best * 100 >= all * 97,
        "agreement {best} of {all} is under 97 %"
    );
    for part in ["q", "k"] {
        let (plain, permuted, _) = totals[part];
        assert!(
            plain > permuted,
            "{part}: plain {plain} not above interleaved {permuted}"
        );
    }
}
