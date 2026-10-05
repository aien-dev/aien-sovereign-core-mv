//! SC-2: deterministic wiring qualification of the attention reference paths (Drake's attention
//! hardening cut). Every case uses head-distinct magnitude bands so a wrong kv-head mapping, a
//! wrong query-head row, a wrong output row or a transposed output cannot pass by accident, and
//! every negative is a concrete wrong wiring whose answer is shown to differ.
//!
//! Values: V for token t, kv head j, dim d is `1000 (j+1) + 20 t + d` (bands never overlap for
//! t < 16, d < 16), so a value decodes back to (kv head, token, dim). K for token t is
//! `BIG * e_t` in every kv head and q for query head h is `BIG * e_target(h)`, so head h attends
//! to token target(h) with weight 1 - O(e^-25).

use aien_abi_core::AttentionGeometry;
use aien_inference_abi::backend::{ReferenceCpuBackend, TensorBackend};
use aien_inference_abi::omega_backend::OmegaGb10Backend;
use aien_inference_abi::tensor;
use aien_kv_cache::{
    bf16_bits_to_f32, f32_to_bf16_bits, KvDType, KvPoolConfig, UnifiedKvTensorPool,
};
use std::panic::{catch_unwind, AssertUnwindSafe};

const HD: usize = 16;
const BIG: f32 = 10.0;
const THETA: f32 = 10000.0;

fn v_val(t: usize, j: usize, d: usize) -> f32 {
    assert!(t < 16 && d < 16);
    (1000 * (j + 1) + 20 * t + d) as f32
}
fn band_head(x: f32) -> usize {
    x.round() as usize / 1000 - 1
}
fn band_token(x: f32) -> usize {
    (x.round() as usize % 1000) / 20
}
fn band_dim(x: f32) -> usize {
    x.round() as usize % 20
}

/// Contiguous `[t][j][d]` K and V for `t_count` tokens and `nkv` kv heads.
fn banded_kv(t_count: usize, nkv: usize) -> (Vec<f32>, Vec<f32>) {
    let mut k = vec![0.0f32; t_count * nkv * HD];
    let mut v = vec![0.0f32; t_count * nkv * HD];
    for t in 0..t_count {
        for j in 0..nkv {
            let base = (t * nkv + j) * HD;
            k[base + t] = BIG;
            for d in 0..HD {
                v[base + d] = v_val(t, j, d);
            }
        }
    }
    (k, v)
}

/// q with head h pointing at token `target(h)`.
fn pointed_q(nq: usize, target: impl Fn(usize) -> usize) -> Vec<f32> {
    let mut q = vec![0.0f32; nq * HD];
    for h in 0..nq {
        q[h * HD + target(h)] = BIG;
    }
    q
}

fn pool_with(
    cfg: KvPoolConfig,
    block_ids: &[usize],
    layer: usize,
    k: &[f32],
    v: &[f32],
    t_count: usize,
) -> UnifiedKvTensorPool {
    let mut pool = UnifiedKvTensorPool::allocate(cfg.clone()).unwrap();
    let kv_dim = cfg.num_kv_heads * cfg.head_dim;
    for t in 0..t_count {
        let (b, s) = (t / cfg.block_size, t % cfg.block_size);
        if b >= block_ids.len() {
            break;
        }
        pool.write_token_kv(
            block_ids[b],
            layer,
            s,
            &k[t * kv_dim..(t + 1) * kv_dim],
            &v[t * kv_dim..(t + 1) * kv_dim],
        );
    }
    pool
}

fn assert_close(got: &[f32], want: &[f32], tol: f32, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(
            (g - w).abs() <= tol,
            "{what}: index {i} got {g} want {w} (tol {tol})"
        );
    }
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn panic_message(r: Result<(), Box<dyn std::any::Any + Send>>) -> String {
    let e = r.expect_err("expected a refusal (panic)");
    if let Some(s) = e.downcast_ref::<String>() {
        s.clone()
    } else if let Some(s) = e.downcast_ref::<&str>() {
        s.to_string()
    } else {
        String::from("<non-string panic payload>")
    }
}

// ---------------------------------------------------------------- A: GQA mapping

#[test]
fn a_gqa_mapping_mha_ratio2_ratio4_ratio8() {
    let be = ReferenceCpuBackend::new();
    for (nq, nkv) in [(4, 4), (8, 4), (8, 2), (16, 2), (8, 1)] {
        let geom = AttentionGeometry::new(nq, nkv, HD).unwrap();
        // one token: softmax weight is exactly 1, out[h] is V[0][kv_head_of(h)] exactly
        let (k, v) = banded_kv(1, nkv);
        let q = vec![1.0f32; nq * HD];
        let mut out = vec![f32::NAN; nq * HD];
        be.gqa_attention(&mut out, &q, &k, &v, 1, nq, nkv, HD);
        let mut wrong_rows = 0;
        for h in 0..nq {
            let j = geom.kv_head_of(h).unwrap();
            assert_eq!(j, h / (nq / nkv));
            for d in 0..HD {
                assert_eq!(
                    out[h * HD + d],
                    v_val(0, j, d),
                    "gqa {nq}q/{nkv}kv head {h} dim {d}"
                );
                assert_eq!(band_head(out[h * HD + d]), j);
            }
            // the modulo mapping h % nkv is the classic wrong wiring; it must give a different
            // row for some head whenever ratio > 1
            if h % nkv != j {
                wrong_rows += 1;
                assert_ne!(out[h * HD], v_val(0, h % nkv, 0));
            }
        }
        if nq / nkv > 1 && nkv > 1 {
            assert!(wrong_rows > 0, "{nq}q/{nkv}kv: negative not exercised");
        }
        // same through the paged path (one block, Fp32)
        let cfg = KvPoolConfig {
            num_blocks: 3,
            block_size: 4,
            num_layers: 2,
            num_kv_heads: nkv,
            head_dim: HD,
            dtype: KvDType::Fp32,
        };
        let pool = pool_with(cfg, &[2], 1, &k, &v, 1);
        let mut pout = vec![f32::NAN; nq * HD];
        be.paged_attention(&mut pout, &q, &pool, &[2], 1, 1, nq, nkv, HD);
        assert_eq!(pout, out, "paged {nq}q/{nkv}kv differs from gqa");
    }
}

// ---------------------------------------------------------------- B + C: q rows, out rows

#[test]
fn b_query_head_rows_and_c_head_major_output() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv, t_count) = (8usize, 2usize, 4usize);
    let geom = AttentionGeometry::new(nq, nkv, HD).unwrap();
    let target = |h: usize| h % t_count;
    let (k, v) = banded_kv(t_count, nkv);
    let q = pointed_q(nq, target);
    let mut out = vec![f32::NAN; nq * HD];
    be.gqa_attention(&mut out, &q, &k, &v, t_count, nq, nkv, HD);

    let mut want = vec![0.0f32; nq * HD];
    let mut want_swapped_q = vec![0.0f32; nq * HD]; // q row read from head h ^ 1
    let mut want_swapped_out = vec![0.0f32; nq * HD]; // head h's answer stored in row h ^ 1
    for h in 0..nq {
        let j = geom.kv_head_of(h).unwrap();
        for d in 0..HD {
            want[h * HD + d] = v_val(target(h), j, d);
            want_swapped_q[h * HD + d] = v_val(target(h ^ 1), j, d);
            want_swapped_out[(h ^ 1) * HD + d] = v_val(target(h), j, d);
        }
    }
    // B: every head read its own q row (token digit) and its own kv head (band)
    assert_close(&out, &want, 1e-2, "B/C wiring");
    for h in 0..nq {
        for d in 0..HD {
            let x = out[h * HD + d];
            assert_eq!(band_head(x), geom.kv_head_of(h).unwrap(), "head {h} band");
            assert_eq!(band_token(x), target(h), "head {h} token digit");
            assert_eq!(band_dim(x), d, "head {h} dim digit");
        }
    }
    // negatives are real alternatives with different answers
    assert!(
        max_abs_diff(&want, &want_swapped_q) >= 10.0,
        "swapped q rows must differ"
    );
    assert!(
        max_abs_diff(&want, &want_swapped_out) >= 10.0,
        "swapped out rows must differ"
    );
    assert!(max_abs_diff(&out, &want_swapped_q) >= 9.0);
    assert!(max_abs_diff(&out, &want_swapped_out) >= 9.0);
    // C: head-major flattening; the dim-major reinterpretation is a different vector
    let mut dim_major = vec![0.0f32; nq * HD];
    for h in 0..nq {
        for d in 0..HD {
            dim_major[d * nq + h] = want[h * HD + d];
        }
    }
    assert!(
        max_abs_diff(&out, &dim_major) >= 9.0,
        "dim-major layout must not match"
    );
}

// ---------------------------------------------------------------- D: dense W_o mixing

#[test]
fn d_dense_wo_mixes_every_head_and_catches_a_permutation() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv, hidden) = (8usize, 2usize, 24usize);
    let geom = AttentionGeometry::new(nq, nkv, HD).unwrap();
    let q_dim = geom.q_dim();
    assert_ne!(
        q_dim, hidden,
        "this case deliberately has q_dim != hidden_dim"
    );
    // attention output with distinct bands per head (from case B's wiring)
    let mut attn = vec![0.0f32; q_dim];
    for h in 0..nq {
        for d in 0..HD {
            attn[h * HD + d] = v_val(h % 4, geom.kv_head_of(h).unwrap(), d) / 1000.0;
        }
    }
    // dense W_o [hidden][q_dim]: deterministic pseudo-random entries in [-1, 1], so every
    // (row, head) pair contributes and no head permutation is a symmetry
    let mut seed = 0xd0_u64;
    let w: Vec<f32> = (0..hidden * q_dim).map(|_| lcg(&mut seed)).collect();
    let mut y = vec![f32::NAN; hidden];
    be.matmul_vec(&mut y, &attn, &w, hidden, q_dim);
    // hand sum over every head
    let mut y_ref = vec![0.0f32; hidden];
    for r in 0..hidden {
        let mut heads_contributing = 0;
        for h in 0..nq {
            let mut part = 0.0f64;
            for d in 0..HD {
                part += (w[r * q_dim + h * HD + d] as f64) * (attn[h * HD + d] as f64);
            }
            if part != 0.0 {
                heads_contributing += 1;
            }
            y_ref[r] += part as f32;
        }
        assert!(
            heads_contributing >= 2,
            "row {r}: W_o must mix several heads"
        );
    }
    assert_close(&y, &y_ref, 1e-3, "W_o dense product");
    // permutation negative: swapping two heads of the attention output changes y
    let mut attn_perm = attn.clone();
    for d in 0..HD {
        attn_perm.swap(d, HD + d);
    }
    let mut y_perm = vec![0.0f32; hidden];
    be.matmul_vec(&mut y_perm, &attn_perm, &w, hidden, q_dim);
    assert!(
        max_abs_diff(&y, &y_perm) > 0.05,
        "a head permutation before W_o must change the output"
    );
    // full h -> h ^ 1 permutation (the Q_ROW / OUT_ROW shape of error) also caught
    let mut attn_x = vec![0.0f32; q_dim];
    for h in 0..nq {
        attn_x[(h ^ 1) * HD..(h ^ 1) * HD + HD].copy_from_slice(&attn[h * HD..h * HD + HD]);
    }
    let mut y_x = vec![0.0f32; hidden];
    be.matmul_vec(&mut y_x, &attn_x, &w, hidden, q_dim);
    assert!(max_abs_diff(&y, &y_x) > 0.05);
}

// ---------------------------------------------------------------- E: RoPE

fn lcg(seed: &mut u64) -> f32 {
    *seed = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*seed >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
}

fn rope_q(be: &dyn TensorBackend, q0: &[f32], pos: usize, nq: usize, nkv: usize) -> Vec<f32> {
    let mut q = q0.to_vec();
    let mut k = vec![0.0f32; nkv * HD];
    be.apply_rope(&mut q, &mut k, pos, HD, nq, nkv, THETA);
    q
}

fn rope_k(be: &dyn TensorBackend, k0: &[f32], pos: usize, nq: usize, nkv: usize) -> Vec<f32> {
    let mut q = vec![0.0f32; nq * HD];
    let mut k = k0.to_vec();
    be.apply_rope(&mut q, &mut k, pos, HD, nq, nkv, THETA);
    k
}

fn dot(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (*x as f64) * (*y as f64))
        .sum()
}

#[test]
fn e_rope_depends_on_relative_position_only() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv) = (4usize, 2usize);
    let mut seed = 0x5eed_u64;
    let q0: Vec<f32> = (0..nq * HD).map(|_| lcg(&mut seed)).collect();
    let k0: Vec<f32> = (0..nkv * HD).map(|_| lcg(&mut seed)).collect();
    // the trait's argument order (head_dim, num_q_heads, num_kv_heads) must reach the
    // implementation (num_heads, num_kv_heads, head_dim) correctly: compare bitwise
    let (mut tq, mut tk) = (q0.clone(), k0.clone());
    tensor::apply_rope(&mut tq, &mut tk, 7, nq, nkv, HD, THETA);
    assert_eq!(
        rope_q(&be, &q0, 7, nq, nkv),
        tq,
        "trait rope q != tensor rope q"
    );
    assert_eq!(
        rope_k(&be, &k0, 7, nq, nkv),
        tk,
        "trait rope k != tensor rope k"
    );
    assert_ne!(tq, q0, "rope at pos 7 must change q");
    // same position on both sides cancels: dot unchanged
    for h in 0..nq {
        let j = h / (nq / nkv);
        let d0 = dot(&q0[h * HD..(h + 1) * HD], &k0[j * HD..(j + 1) * HD]);
        let d1 = dot(
            &rope_q(&be, &q0, 11, nq, nkv)[h * HD..(h + 1) * HD],
            &rope_k(&be, &k0, 11, nq, nkv)[j * HD..(j + 1) * HD],
        );
        assert!(
            (d0 - d1).abs() < 1e-4,
            "head {h}: same-position rope changed the dot"
        );
    }
    // (5,2) and (13,10) share the offset 3; (5,3) has offset 2
    for h in 0..nq {
        let j = h / (nq / nkv);
        let at = |pq: usize, pk: usize| {
            dot(
                &rope_q(&be, &q0, pq, nq, nkv)[h * HD..(h + 1) * HD],
                &rope_k(&be, &k0, pk, nq, nkv)[j * HD..(j + 1) * HD],
            )
        };
        let (a, b, c) = (at(5, 2), at(13, 10), at(5, 3));
        assert!(
            (a - b).abs() < 1e-4,
            "head {h}: offset 3 at two bases differs: {a} vs {b}"
        );
        assert!(
            (a - c).abs() > 1e-3,
            "head {h}: offsets 3 and 2 must differ: {a} vs {c}"
        );
    }
    // through attention: V one-hot per token so the output IS the softmax weight vector
    let t_count = 6usize;
    let weights = |base: usize, q_pos: usize| -> Vec<f32> {
        let mut k = vec![0.0f32; t_count * nkv * HD];
        let mut v = vec![0.0f32; t_count * nkv * HD];
        for t in 0..t_count {
            let kt = rope_k(&be, &k0, base + t, nq, nkv);
            for j in 0..nkv {
                let b = (t * nkv + j) * HD;
                k[b..b + HD].copy_from_slice(&kt[j * HD..(j + 1) * HD]);
                v[b + t] = 1.0;
            }
        }
        let q = rope_q(
            &be,
            &q0.iter().map(|x| x * 4.0).collect::<Vec<_>>(),
            base + q_pos,
            nq,
            nkv,
        );
        let mut out = vec![f32::NAN; nq * HD];
        be.gqa_attention(&mut out, &q, &k, &v, t_count, nq, nkv, HD);
        out
    };
    let w0 = weights(0, t_count - 1);
    let w37 = weights(37, t_count - 1);
    assert_close(&w0, &w37, 1e-4, "softmax weights must be shift invariant");
    for h in 0..nq {
        let s: f32 = w0[h * HD..h * HD + t_count].iter().sum();
        assert!((s - 1.0).abs() < 1e-4, "head {h} weights sum {s}");
        assert!(w0[h * HD + t_count..(h + 1) * HD]
            .iter()
            .all(|x| x.abs() < 1e-6));
    }
    let w_shifted_q = weights(0, t_count + 4); // q five positions later than the last token
    assert!(
        max_abs_diff(&w0, &w_shifted_q) > 1e-3,
        "moving q relative to K must change the weights"
    );
}

// ---------------------------------------------------------------- F: boundaries

fn cfg(nkv: usize, block_size: usize, num_blocks: usize, dtype: KvDType) -> KvPoolConfig {
    KvPoolConfig {
        num_blocks,
        block_size,
        num_layers: 3,
        num_kv_heads: nkv,
        head_dim: HD,
        dtype,
    }
}

#[test]
fn f_empty_context_and_no_blocks_write_zeros() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv) = (8usize, 2usize);
    let (k, v) = banded_kv(3, nkv);
    let pool = pool_with(cfg(nkv, 4, 4, KvDType::Fp32), &[1], 2, &k, &v, 3);
    let q = pointed_q(nq, |h| h % 3);
    let mut out = vec![f32::NAN; nq * HD];
    be.paged_attention(&mut out, &q, &pool, &[1], 0, 2, nq, nkv, HD);
    assert!(out.iter().all(|x| *x == 0.0), "context 0 must write zeros");
    out.fill(f32::NAN);
    be.paged_attention(&mut out, &q, &pool, &[], 3, 2, nq, nkv, HD);
    assert!(out.iter().all(|x| *x == 0.0), "no blocks must write zeros");
    out.fill(f32::NAN);
    be.gqa_attention(&mut out, &q, &k, &v, 0, nq, nkv, HD);
    assert!(out.iter().all(|x| *x == 0.0), "seq_len 0 must write zeros");
}

#[test]
fn f_one_token_is_exactly_v() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv) = (8usize, 4usize);
    let (k, v) = banded_kv(1, nkv);
    let pool = pool_with(cfg(nkv, 16, 2, KvDType::Fp32), &[1], 0, &k, &v, 1);
    let q: Vec<f32> = (0..nq * HD).map(|i| (i % 7) as f32 - 3.0).collect();
    let mut out = vec![f32::NAN; nq * HD];
    be.paged_attention(&mut out, &q, &pool, &[1], 1, 0, nq, nkv, HD);
    for h in 0..nq {
        for d in 0..HD {
            assert_eq!(out[h * HD + d], v_val(0, h / 2, d));
        }
    }
}

/// Partial last block, several blocks, scattered ids, Fp32 and Bf16, against the contiguous
/// reference over the test's own arrays (not the pool's gather).
#[test]
fn f_partial_block_multiple_scattered_blocks_fp32_and_bf16() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv, bs) = (8usize, 2usize, 4usize);
    let ids = [7usize, 2, 11, 29];
    let ctx = 14; // 3 full blocks + 2 tokens of the 4th
    let mut seed = 99u64;
    let kv_dim = nkv * HD;
    let k: Vec<f32> = (0..ctx * kv_dim).map(|_| lcg(&mut seed)).collect();
    let v: Vec<f32> = (0..ctx * kv_dim).map(|_| lcg(&mut seed) * 50.0).collect();
    let q: Vec<f32> = (0..nq * HD).map(|_| lcg(&mut seed) * 3.0).collect();
    for dtype in [KvDType::Fp32, KvDType::Bf16] {
        let pool = pool_with(cfg(nkv, bs, 32, dtype), &ids, 1, &k, &v, ctx);
        // what the pool holds after its own rounding
        let round = |x: f32| match dtype {
            KvDType::Bf16 => bf16_bits_to_f32(f32_to_bf16_bits(x)),
            _ => x,
        };
        let kr: Vec<f32> = k.iter().map(|x| round(*x)).collect();
        let vr: Vec<f32> = v.iter().map(|x| round(*x)).collect();
        let mut want = vec![0.0f32; nq * HD];
        be.gqa_attention(&mut want, &q, &kr, &vr, ctx, nq, nkv, HD);
        let mut got = vec![f32::NAN; nq * HD];
        be.paged_attention(&mut got, &q, &pool, &ids, ctx, 1, nq, nkv, HD);
        assert_close(&got, &want, 1e-4, &format!("paged vs contiguous {dtype:?}"));
        // the other layers of the same blocks were never written: layer 0 must not see this data
        let mut other = vec![f32::NAN; nq * HD];
        be.paged_attention(&mut other, &q, &pool, &ids, ctx, 0, nq, nkv, HD);
        assert!(
            max_abs_diff(&other, &want) > 1.0,
            "layer 0 read layer 1's data"
        );
        // tokens past the block table are dropped (documented oracle behaviour): the first two
        // blocks only
        let mut want8 = vec![0.0f32; nq * HD];
        be.gqa_attention(&mut want8, &q, &kr, &vr, 2 * bs, nq, nkv, HD);
        let mut got8 = vec![f32::NAN; nq * HD];
        be.paged_attention(&mut got8, &q, &pool, &ids[..2], ctx, 1, nq, nkv, HD);
        assert_close(&got8, &want8, 1e-4, "tokens past the table dropped");
        assert!(
            max_abs_diff(&got8, &want) > 1e-3,
            "8 and 14 tokens must differ"
        );
    }
}

#[test]
fn f_wrong_q_length_is_refused() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv) = (8usize, 2usize);
    let (k, v) = banded_kv(2, nkv);
    let pool = pool_with(cfg(nkv, 4, 4, KvDType::Fp32), &[0], 0, &k, &v, 2);
    let q = vec![1.0f32; nq * HD + HD]; // one head too many
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD];
        be.paged_attention(&mut out, &q, &pool, &[0], 2, 0, nq, nkv, HD);
    })));
    assert!(
        msg.contains("refused") && msg.contains("q has 144"),
        "{msg}"
    );
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD];
        be.gqa_attention(&mut out, &q[..nq * HD - 1], &k, &v, 2, nq, nkv, HD);
    })));
    assert!(
        msg.contains("gqa_attention refused") && msg.contains("q has 127"),
        "{msg}"
    );
}

#[test]
fn f_wrong_out_length_is_refused() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv) = (8usize, 2usize);
    let (k, v) = banded_kv(2, nkv);
    let pool = pool_with(cfg(nkv, 4, 4, KvDType::Fp32), &[0], 0, &k, &v, 2);
    let q = vec![1.0f32; nq * HD];
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD - HD]; // one head short
        be.paged_attention(&mut out, &q, &pool, &[0], 2, 0, nq, nkv, HD);
    })));
    assert!(
        msg.contains("refused") && msg.contains("out has 112"),
        "{msg}"
    );
    // batch: 2 sequences need 2 x q_dim
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD];
        let q2 = vec![1.0f32; 2 * nq * HD];
        be.paged_attention_batch(&mut out, &q2, &pool, &[0, 0], &[2, 2], 1, 2, 0, nq, nkv, HD);
    })));
    assert!(msg.contains("paged_attention_batch refused"), "{msg}");
}

#[test]
fn f_incompatible_kv_geometry_is_refused_before_reading() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv) = (8usize, 2usize);
    let (k, v) = banded_kv(2, nkv);
    // pool laid out for 2 kv heads, caller claims 4
    let pool = pool_with(cfg(nkv, 4, 4, KvDType::Fp32), &[0], 0, &k, &v, 2);
    let q = vec![1.0f32; nq * HD];
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD];
        be.paged_attention(&mut out, &q, &pool, &[0], 2, 0, nq, 4, HD);
    })));
    assert!(
        msg.contains("4 kv heads x head_dim 16") && msg.contains("pool is laid out for 2 kv heads"),
        "{msg}"
    );
    // head_dim mismatch (caller says 8), even with an empty context
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * 8];
        be.paged_attention(&mut out, &q[..nq * 8], &pool, &[0], 0, 0, nq, nkv, 8);
    })));
    assert!(msg.contains("head_dim 16"), "{msg}");
    // non-divisible heads
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; 6 * HD];
        be.gqa_attention(&mut out, &q[..6 * HD], &k, &v, 2, 6, 4, HD);
    })));
    assert!(
        msg.contains("6 is not a multiple of num_kv_heads 4"),
        "{msg}"
    );
    // a short K cache for the claimed seq_len
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD];
        be.gqa_attention(&mut out, &q, &k, &v, 5, nq, nkv, HD);
    })));
    assert!(
        msg.contains("k/v caches hold 64/64 values") && msg.contains("need 160"),
        "{msg}"
    );
}

/// The chip backend refuses the same mismatches before any native call; in the CPU-stub build
/// this is the only path that runs, and the refusal is counted as a chip error. In a strict
/// process the refusal surfaces as the strict violation for the op (the fallback itself is
/// fatal); in a dev process the reference path then refuses with the geometry message.
#[test]
fn f_omega_backend_refuses_pool_mismatch_before_native_code() {
    let strict = aien_inference_abi::strict::production_strict();
    let refused = |msg: &str, op: &str, dev_text: &[&str]| {
        if strict {
            assert!(
                msg.contains(aien_inference_abi::strict::STRICT_VIOLATION_PREFIX)
                    && msg.contains(op),
                "{msg}"
            );
        } else {
            assert!(dev_text.iter().all(|t| msg.contains(t)), "{msg}");
        }
    };
    let be = OmegaGb10Backend::new();
    let (nq, nkv) = (8usize, 2usize);
    let (k, v) = banded_kv(2, nkv);
    let pool = pool_with(cfg(nkv, 4, 4, KvDType::Fp32), &[0], 0, &k, &v, 2);
    let q = vec![1.0f32; nq * HD];
    let before = be.chip_errors();
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; nq * HD];
        be.paged_attention(&mut out, &q, &pool, &[0], 2, 0, nq, 4, HD);
    })));
    refused(
        &msg,
        "paged_attention",
        &["pool is laid out for 2 kv heads"],
    );
    assert!(be.chip_errors() > before, "the refusal must be counted");
    let msg = panic_message(catch_unwind(AssertUnwindSafe(|| {
        let mut out = vec![0.0f32; 6 * HD];
        be.gqa_attention(&mut out, &q[..6 * HD], &k, &v, 2, 6, 4, HD);
    })));
    refused(
        &msg,
        "gqa_attention",
        &["6 is not a multiple of num_kv_heads 4"],
    );
}

/// Batched call with a shared prefix and negative (padding) table entries equals the
/// per-sequence calls; a sequence whose private tail differs gets a different answer.
#[test]
fn f_batch_shared_prefix_scattered_ids_matches_single_calls() {
    let be = ReferenceCpuBackend::new();
    let (nq, nkv, bs) = (8usize, 2usize, 4usize);
    let kv_dim = nkv * HD;
    let mut seed = 7u64;
    let mut pool = UnifiedKvTensorPool::allocate(cfg(nkv, bs, 32, KvDType::Fp32)).unwrap();
    // shared prefix blocks 9 and 3 (8 tokens), private tails: seq0 block 20 (3 tokens),
    // seq1 block 5 (2 tokens), seq2 no tail (prefix only)
    let mut fill = |block: usize, tokens: usize| {
        for s in 0..tokens {
            let k: Vec<f32> = (0..kv_dim).map(|_| lcg(&mut seed)).collect();
            let v: Vec<f32> = (0..kv_dim).map(|_| lcg(&mut seed) * 20.0).collect();
            pool.write_token_kv(block, 0, s, &k, &v);
        }
    };
    fill(9, bs);
    fill(3, bs);
    fill(20, 3);
    fill(5, 2);
    let tables: [&[i32]; 3] = [&[9, 3, 20], &[9, 3, 5], &[9, 3, -1]];
    let ctx = [11i32, 10, 8];
    let maxb = 3;
    let q: Vec<f32> = (0..3 * nq * HD).map(|_| lcg(&mut seed) * 2.0).collect();
    let flat: Vec<i32> = tables.concat();
    let mut batch = vec![f32::NAN; 3 * nq * HD];
    be.paged_attention_batch(&mut batch, &q, &pool, &flat, &ctx, maxb, 3, 0, nq, nkv, HD);
    for s in 0..3 {
        let ids: Vec<usize> = tables[s]
            .iter()
            .filter(|b| **b >= 0)
            .map(|b| *b as usize)
            .collect();
        let mut single = vec![f32::NAN; nq * HD];
        be.paged_attention(
            &mut single,
            &q[s * nq * HD..(s + 1) * nq * HD],
            &pool,
            &ids,
            ctx[s] as usize,
            0,
            nq,
            nkv,
            HD,
        );
        assert_close(
            &batch[s * nq * HD..(s + 1) * nq * HD],
            &single,
            1e-6,
            &format!("seq {s}"),
        );
    }
    // the three sequences see different histories, so with the same q they must differ
    let q0 = &q[..nq * HD];
    let mut same_q = vec![0.0f32; 3 * nq * HD];
    for s in 0..3 {
        same_q[s * nq * HD..(s + 1) * nq * HD].copy_from_slice(q0);
    }
    be.paged_attention_batch(
        &mut batch, &same_q, &pool, &flat, &ctx, maxb, 3, 0, nq, nkv, HD,
    );
    assert!(max_abs_diff(&batch[..nq * HD], &batch[nq * HD..2 * nq * HD]) > 1e-3);
    assert!(max_abs_diff(&batch[..nq * HD], &batch[2 * nq * HD..]) > 1e-3);
    // context_len <= 0 rows are zero
    be.paged_attention_batch(
        &mut batch,
        &same_q,
        &pool,
        &flat,
        &[11, 0, -4],
        maxb,
        3,
        0,
        nq,
        nkv,
        HD,
    );
    assert!(batch[nq * HD..].iter().all(|x| *x == 0.0));
}
