//! FB-1 cut 3b: OmegaGb10Backend against ReferenceCpuBackend on TinyLlama layer shapes.
//!
//! Host test (runs everywhere, stub-aware): in a stub build the backend must
//! report itself unavailable. Chip test (`#[ignore]`, run through the heavy
//! queue with a native build):
//!   AIEN_OMEGA_DIR=... AIEN_PHYSICS_DIR=... cargo test -p aien-inference-abi --release \
//!     --test omega_backend_parity -- --ignored --nocapture
//!
//! Error metric: |chip - ref| divided by sum_i |x_i * w_i| (the accumulation
//! scale). Both sides see bf16-rounded inputs on the chip and f32 on the
//! reference, so each product carries at most ~2 * 2^-8 relative error; the bound
//! asserted is 8e-3 of the scale, and the typical value is far lower (random signs).
use aien_inference_abi::{OmegaGb10Backend, ReferenceCpuBackend, TensorBackend};

fn lcg(s: &mut u32) -> f32 {
    *s = s.wrapping_mul(1664525).wrapping_add(1013904223);
    ((*s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
}

/// Max over outputs of |got - want| / sum|x*w|, for out = X * W^T.
fn worst_ratio(
    got: &[f32],
    want: &[f32],
    x: &[f32],
    w: &[f32],
    m: usize,
    k: usize,
    n: usize,
) -> f64 {
    let mut worst = 0.0f64;
    for r in 0..m {
        for o in 0..n {
            let mut scale = 0.0f64;
            for i in 0..k {
                scale += (x[r * k + i] as f64 * w[o * k + i] as f64).abs();
            }
            let d = (got[r * n + o] as f64 - want[r * n + o] as f64).abs();
            worst = worst.max(d / scale.max(1e-30));
        }
    }
    worst
}

#[test]
fn stub_backend_reports_unavailable() {
    let b = OmegaGb10Backend::new();
    assert_eq!(b.is_available(), aien_omega_gpu::is_native());
    assert_eq!(b.native_ops().native_ops().len(), 9);
}

#[test]
#[ignore = "chip test: needs a native build and the GB10; run through the heavy queue"]
fn chip_parity_on_tinyllama_shapes() {
    let omega = OmegaGb10Backend::new();
    assert!(omega.is_available(), "native build required");
    let reference = ReferenceCpuBackend::new();
    // (name, out_dim, in_dim): TinyLlama 1.1B hidden 2048, kv 256, inter 5632, vocab 32000.
    let shapes = [
        ("q_proj/o_proj", 2048usize, 2048usize),
        ("k_proj/v_proj", 256, 2048),
        ("gate/up_proj", 5632, 2048),
        ("down_proj", 2048, 5632),
        ("lm_head", 32000, 2048),
    ];
    let mut seed = 12345u32;
    let mut overall = 0.0f64;
    for (name, out_dim, in_dim) in shapes {
        let w: Vec<f32> = (0..out_dim * in_dim).map(|_| lcg(&mut seed)).collect();
        // matmul_vec (and compute_logits for lm_head), batch 1
        let x: Vec<f32> = (0..in_dim).map(|_| lcg(&mut seed)).collect();
        let mut got = vec![0.0f32; out_dim];
        let mut want = vec![0.0f32; out_dim];
        if name == "lm_head" {
            omega.compute_logits(&mut got, &x, &w, out_dim, in_dim);
            reference.compute_logits(&mut want, &x, &w, out_dim, in_dim);
        } else {
            omega.matmul_vec(&mut got, &x, &w, out_dim, in_dim);
            reference.matmul_vec(&mut want, &x, &w, out_dim, in_dim);
        }
        let r1 = worst_ratio(&got, &want, &x, &w, 1, in_dim, out_dim);
        // matmul_batch, batch 7 (same resident weights, different m)
        let batch = 7usize;
        let xb: Vec<f32> = (0..batch * in_dim).map(|_| lcg(&mut seed)).collect();
        let mut gotb = vec![0.0f32; batch * out_dim];
        let mut wantb = vec![0.0f32; batch * out_dim];
        omega.matmul_batch(&mut gotb, &xb, &w, batch, in_dim, out_dim);
        reference.matmul_batch(&mut wantb, &xb, &w, batch, in_dim, out_dim);
        let rb = worst_ratio(&gotb, &wantb, &xb, &w, batch, in_dim, out_dim);
        println!(
            "PARITY {name} {out_dim}x{in_dim}: vec max_err/scale={r1:.3e} batch7 max_err/scale={rb:.3e}"
        );
        assert!(r1 < 8e-3 && rb < 8e-3, "{name}: parity bound exceeded");
        overall = overall.max(r1).max(rb);
    }
    assert_eq!(omega.fallback_count(), 0, "{}", omega.last_error());
    assert_eq!(omega.chip_errors(), 0, "{}", omega.last_error());
    println!("PARITY overall worst max_err/scale = {overall:.3e}");
    println!("PARITY resident weights = {}", omega.resident_count());
    println!("PARITY {}", omega.op_report().line());
}

/// Worst of |got - want| / (1e-6 + 1e-5 * |want|) over all elements; <= 1 means inside the
/// omega cut-4 elementwise tolerance (1e-5 relative + 1e-6 absolute).
fn worst_tol_ratio(got: &[f32], want: &[f32]) -> f64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - *w as f64).abs() / (1e-6 + 1e-5 * (*w as f64).abs()))
        .fold(0.0, f64::max)
}

/// FB-1 cut 3c: rmsnorm, apply_rope and swiglu on the chip at TinyLlama shapes
/// (hidden 2048, 32 q heads, 4 kv heads, head_dim 64, intermediate 5632, eps 1e-5,
/// theta 10000) against ReferenceCpuBackend. Rope must be bit-exact (omega emits
/// FMUL/FMUL/FADD, no fused multiply-add); rmsnorm and swiglu within the tolerance
/// stated in omega docs/numeric/FB1_CUT4_ELEMENTWISE.md (1e-5 relative + 1e-6 absolute).
#[test]
#[ignore = "chip test: needs a native build and the GB10; run through the heavy queue"]
fn chip_elementwise_parity_on_tinyllama_shapes() {
    let omega = OmegaGb10Backend::new();
    assert!(omega.is_available(), "native build required");
    let reference = ReferenceCpuBackend::new();
    let mut seed = 777u32;

    // rmsnorm, dim 2048 (hidden), 20 different rows of data incl. a large-magnitude one.
    let dim = 2048usize;
    let w: Vec<f32> = (0..dim).map(|_| lcg(&mut seed) * 4.0 + 1.0).collect();
    let mut rms_worst = 0.0f64;
    for trial in 0..20 {
        let scale = if trial == 19 { 50.0 } else { 2.0 };
        let x: Vec<f32> = (0..dim).map(|_| lcg(&mut seed) * scale).collect();
        let (mut got, mut want) = (vec![0.0f32; dim], vec![0.0f32; dim]);
        omega.rmsnorm(&mut got, &x, &w, 1e-5);
        reference.rmsnorm(&mut want, &x, &w, 1e-5);
        rms_worst = rms_worst.max(worst_tol_ratio(&got, &want));
    }
    println!("PARITY rmsnorm 2048 worst err/tolerance = {rms_worst:.3e} (must be <= 1)");
    assert!(rms_worst <= 1.0, "rmsnorm outside tolerance");

    // rope, 32 q heads + 4 kv heads of 64, several positions.
    let (hd, nq, nkv, theta) = (64usize, 32usize, 4usize, 10000.0f32);
    let mut rope_mismatch = 0usize;
    for pos in [0usize, 1, 2, 37, 511, 1500, 2047] {
        let q0: Vec<f32> = (0..nq * hd).map(|_| lcg(&mut seed) * 3.0).collect();
        let k0: Vec<f32> = (0..nkv * hd).map(|_| lcg(&mut seed) * 3.0).collect();
        let (mut qg, mut kg) = (q0.clone(), k0.clone());
        let (mut qw, mut kw) = (q0.clone(), k0.clone());
        omega.apply_rope(&mut qg, &mut kg, pos, hd, nq, nkv, theta);
        reference.apply_rope(&mut qw, &mut kw, pos, hd, nq, nkv, theta);
        rope_mismatch += qg
            .iter()
            .zip(&qw)
            .chain(kg.iter().zip(&kw))
            .filter(|(a, b)| a.to_bits() != b.to_bits())
            .count();
    }
    println!("PARITY rope 32+4 heads x 64, 7 positions: bit mismatches = {rope_mismatch}");
    assert_eq!(rope_mismatch, 0, "rope must be bit-exact");

    // swiglu, n = 5632 (intermediate), gate in a wide range.
    let n = 5632usize;
    let mut sw_worst = 0.0f64;
    for _ in 0..10 {
        let gate: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 16.0).collect();
        let up: Vec<f32> = (0..n).map(|_| lcg(&mut seed) * 4.0).collect();
        let (mut got, mut want) = (vec![0.0f32; n], vec![0.0f32; n]);
        omega.swiglu(&mut got, &gate, &up);
        reference.swiglu(&mut want, &gate, &up);
        sw_worst = sw_worst.max(worst_tol_ratio(&got, &want));
    }
    println!("PARITY swiglu 5632 worst err/tolerance = {sw_worst:.3e} (must be <= 1)");
    assert!(sw_worst <= 1.0, "swiglu outside tolerance");

    assert_eq!(omega.fallback_count(), 0, "{}", omega.last_error());
    assert_eq!(omega.chip_errors(), 0, "{}", omega.last_error());
    println!("PARITY elementwise {}", omega.op_report().line());
}

/// Diagnostic for the cut-3c finding: elementwise launches after the matmul device is
/// open. Run with AIEN_DEV_FALLBACK=1 so a failure is reported instead of a strict panic.
#[test]
#[ignore = "chip diagnostic: heavy queue only"]
fn chip_mixed_matmul_then_elementwise() {
    let omega = OmegaGb10Backend::new();
    assert!(omega.is_available(), "native build required");
    let mut seed = 4242u32;
    let (k, n) = (256usize, 64usize);
    let w: Vec<f32> = (0..n * k).map(|_| lcg(&mut seed)).collect();
    let x: Vec<f32> = (0..k).map(|_| lcg(&mut seed)).collect();
    let mut out = vec![0.0f32; n];
    let wn: Vec<f32> = (0..256).map(|_| lcg(&mut seed)).collect();
    let mut no = vec![0.0f32; 256];
    omega.rmsnorm(&mut no, &x, &wn, 1e-5);
    println!(
        "MIXED rmsnorm before matmul: chip_errors={}",
        omega.chip_errors()
    );
    omega.matmul_vec(&mut out, &x, &w, n, k);
    println!("MIXED matmul: chip_errors={}", omega.chip_errors());
    omega.rmsnorm(&mut no, &x, &wn, 1e-5);
    println!(
        "MIXED rmsnorm after matmul: chip_errors={} last_error={}",
        omega.chip_errors(),
        omega.last_error()
    );
}

fn worst_attn_ratio(got: &[f32], want: &[f32]) -> f64 {
    // omega tests/gpu_attention_test.c tolerances (FB-1 cut 5): rel 2e-4, abs 2e-5.
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .map(|(g, w)| (*g as f64 - *w as f64).abs() / (2e-5 + 2e-4 * (*w as f64).abs()))
        .fold(0.0, f64::max)
}

/// Fill `n_tokens` tokens of `layer` into a fresh TinyLlama-shaped pool (block_size 16),
/// blocks used in a scrambled order so the block table is not the identity.
fn filled_pool(
    dtype: aien_kv_cache::KvDType,
    n_tokens: usize,
    layer: usize,
    seed: &mut u32,
) -> (aien_kv_cache::UnifiedKvTensorPool, Vec<usize>) {
    let (num_blocks, bs) = (160usize, 16usize);
    let cfg = aien_kv_cache::KvPoolConfig::for_tinyllama(num_blocks, bs, dtype);
    let mut pool = aien_kv_cache::UnifiedKvTensorPool::allocate(cfg).expect("pool");
    let need = n_tokens.div_ceil(bs);
    let blocks: Vec<usize> = (0..need).map(|i| (i * 37 + 11) % num_blocks).collect();
    let kv_dim = 4 * 64;
    for t in 0..n_tokens {
        let k: Vec<f32> = (0..kv_dim).map(|_| lcg(seed) * 2.0).collect();
        let v: Vec<f32> = (0..kv_dim).map(|_| lcg(seed) * 2.0).collect();
        pool.write_token_kv(blocks[t / bs], layer, t % bs, &k, &v);
    }
    (pool, blocks)
}

/// FB-1 cut 3d: the three attention ops on the chip at TinyLlama shapes (32 q heads,
/// 4 kv heads, head_dim 64) against ReferenceCpuBackend: contiguous gqa, paged over an
/// Fp32 pool (the model's pool; host gather + f32 kernel), paged over a bf16 pool (omega's
/// paged kernel), and the batch op with ragged rows, a negative entry and an empty row.
#[test]
#[ignore = "chip test: needs a native build and the GB10; run through the heavy queue"]
fn chip_attention_parity_on_tinyllama_shapes() {
    let omega = OmegaGb10Backend::new();
    assert!(omega.is_available(), "native build required");
    let reference = ReferenceCpuBackend::new();
    let mut seed = 4242u32;
    let (nq, nkv, hd) = (32usize, 4usize, 64usize);

    let mut gqa_worst = 0.0f64;
    for seq in [1usize, 7, 64, 300, 2048] {
        let q: Vec<f32> = (0..nq * hd).map(|_| lcg(&mut seed) * 2.0).collect();
        let k: Vec<f32> = (0..seq * nkv * hd).map(|_| lcg(&mut seed) * 2.0).collect();
        let v: Vec<f32> = (0..seq * nkv * hd).map(|_| lcg(&mut seed) * 2.0).collect();
        let (mut got, mut want) = (vec![0.0f32; nq * hd], vec![0.0f32; nq * hd]);
        omega.gqa_attention(&mut got, &q, &k, &v, seq, nq, nkv, hd);
        reference.gqa_attention(&mut want, &q, &k, &v, seq, nq, nkv, hd);
        let r = worst_attn_ratio(&got, &want);
        let at = got
            .iter()
            .zip(&want)
            .map(|(g, w)| (*g as f64 - *w as f64).abs())
            .enumerate()
            .fold(
                (0usize, 0.0f64),
                |m, (i, d)| if d > m.1 { (i, d) } else { m },
            );
        println!(
            "PARITY gqa seq={seq} err/tol={r:.3e} worst at {} got={} want={}",
            at.0, got[at.0], want[at.0]
        );
        gqa_worst = gqa_worst.max(r);
    }
    println!(
        "PARITY gqa_attention seq 1..2048 worst err/tolerance = {gqa_worst:.3e} (must be <= 1)"
    );
    assert!(gqa_worst <= 1.0, "gqa_attention outside tolerance");

    for dtype in [aien_kv_cache::KvDType::Fp32, aien_kv_cache::KvDType::Bf16] {
        let mut worst = 0.0f64;
        for (ctx, layer) in [(1usize, 0usize), (17, 3), (250, 21), (1000, 9)] {
            let (pool, blocks) = filled_pool(dtype, ctx, layer, &mut seed);
            let q: Vec<f32> = (0..nq * hd).map(|_| lcg(&mut seed) * 2.0).collect();
            let (mut got, mut want) = (vec![0.0f32; nq * hd], vec![0.0f32; nq * hd]);
            omega.paged_attention(&mut got, &q, &pool, &blocks, ctx, layer, nq, nkv, hd);
            reference.paged_attention(&mut want, &q, &pool, &blocks, ctx, layer, nq, nkv, hd);
            worst = worst.max(worst_attn_ratio(&got, &want));
        }
        println!(
            "PARITY paged_attention {dtype:?} worst err/tolerance = {worst:.3e} (must be <= 1)"
        );
        assert!(worst <= 1.0, "paged_attention {dtype:?} outside tolerance");

        // batch: 3 rows (ctx 40, 0 = empty, 90 with a trailing -1), max_blocks 8.
        let (pool, blocks) = filled_pool(dtype, 90, 5, &mut seed);
        let mb = 8usize;
        let mut tables = vec![-1i32; 3 * mb];
        for (i, b) in blocks.iter().take(3).enumerate() {
            tables[i] = *b as i32;
        }
        for (i, b) in blocks.iter().enumerate() {
            tables[2 * mb + i] = *b as i32;
        }
        let lens = [40i32, 0, 90];
        let q: Vec<f32> = (0..3 * nq * hd).map(|_| lcg(&mut seed) * 2.0).collect();
        let (mut got, mut want) = (vec![0.0f32; 3 * nq * hd], vec![0.0f32; 3 * nq * hd]);
        omega.paged_attention_batch(&mut got, &q, &pool, &tables, &lens, mb, 3, 5, nq, nkv, hd);
        reference
            .paged_attention_batch(&mut want, &q, &pool, &tables, &lens, mb, 3, 5, nq, nkv, hd);
        let bw = worst_attn_ratio(&got, &want);
        println!(
            "PARITY paged_attention_batch {dtype:?} worst err/tolerance = {bw:.3e} (must be <= 1)"
        );
        assert!(
            bw <= 1.0,
            "paged_attention_batch {dtype:?} outside tolerance"
        );
    }

    assert_eq!(omega.fallback_count(), 0, "{}", omega.last_error());
    assert_eq!(omega.chip_errors(), 0, "{}", omega.last_error());
    println!("PARITY attention {}", omega.op_report().line());
}

/// Negative control on the chip build: an attention geometry the native engine does not
/// support (head_dim 32; omega refuses head_dim != 64 with OMEGA_GPU_ATTN_TOO_LARGE in
/// src/omega_gpu_attention_api.c before any launch) goes through the native library, is
/// refused there, and in a strict process the refusal is fatal: no CPU answer appears behind
/// the GPU backend. In a dev process it is a counted fallback. This is a library refusal, not
/// an injected device fault; a device fault takes the same `fail` -> `reference_path` route.
#[test]
#[ignore = "chip test: needs a native build and the GB10; run through the heavy queue"]
fn chip_unsupported_attention_is_refused() {
    let omega = OmegaGb10Backend::new();
    assert!(omega.is_available(), "native build required");
    let (nq, nkv, hd, seq) = (4usize, 2usize, 32usize, 3usize);
    let mut seed = 99u32;
    let q: Vec<f32> = (0..nq * hd).map(|_| lcg(&mut seed)).collect();
    let k: Vec<f32> = (0..seq * nkv * hd).map(|_| lcg(&mut seed)).collect();
    let v: Vec<f32> = (0..seq * nkv * hd).map(|_| lcg(&mut seed)).collect();
    let mut out = vec![0.0f32; nq * hd];
    let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        omega.gqa_attention(&mut out, &q, &k, &v, seq, nq, nkv, hd)
    }));
    println!(
        "NEGATIVE unsupported head_dim={hd}: chip_errors={} fallback_count={} last_error={:?} strict={}",
        omega.chip_errors(),
        omega.fallback_count(),
        omega.last_error(),
        aien_inference_abi::strict::production_strict()
    );
    assert_eq!(omega.chip_errors(), 1, "the native library must refuse");
    assert_eq!(omega.fallback_count(), 1);
    if aien_inference_abi::strict::production_strict() {
        let err = r.expect_err("strict: an unsupported native op must be fatal");
        let msg = err
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| err.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_default();
        assert!(
            msg.contains(aien_inference_abi::strict::STRICT_VIOLATION_PREFIX)
                && msg.contains("gqa_attention"),
            "{msg}"
        );
        println!("NEGATIVE strict refusal: {msg}");
    } else {
        r.expect("dev: counted fallback, run continues");
    }
}
