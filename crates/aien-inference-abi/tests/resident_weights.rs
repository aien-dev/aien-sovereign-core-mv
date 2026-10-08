//! GB10 weight ownership (sovereign-core #277 cut C): resident bf16 upload straight from the
//! checkpoint, no host f32 copy of any matmul weight.
//!
//! No chip is used. The "device" is a fake that keeps the uploaded `[k][n]` bf16 bits and
//! computes with the same host arithmetic as the reference backend, so a Host load and a
//! Resident load of the same checkpoint must give bit-identical logits if (and only if) the
//! streaming, the transpose and the dispatch are right.

use aien_abi_core::RopeParams;
use aien_inference_abi::{
    bf16_bits_as_uploaded_today, load_model_config, load_resident_weights, resident_load_wanted,
    transpose_bf16_for_upload, CheckpointError, MatrixRef, MatrixWeight, NativeTransformerBackend,
    ReferenceCpuBackend, ResidentHandle, ResidentUploader, ResidentWeight, TensorBackend,
    TransformerWeights,
};
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

fn tiny_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/qwen3-tiny")
}

// ---- numerics: bf16 -> f32 -> bf16 over every pattern ----

/// Independent software model of `(__bf16)f` on aarch64 (round to nearest even, NaN quieted),
/// not sharing code with the loader. It was checked against gcc's own `(__bf16)` cast for all
/// 65536 patterns on this host: 126 differences, all signaling NaNs (see the PR text).
fn narrow_like_omega(f: f32) -> u16 {
    let bits = f.to_bits();
    if f.is_nan() {
        return ((bits >> 16) as u16) | 0x0040;
    }
    let lsb = (bits >> 16) & 1;
    (bits.wrapping_add(0x7fff + lsb) >> 16) as u16
}

#[test]
fn bf16_widen_then_narrow_is_exact_except_signaling_nans_which_omega_quiets() {
    let mut changed = Vec::new();
    let (mut denormals, mut nans, mut infs) = (0u32, 0u32, 0u32);
    for p in 0..=u16::MAX {
        let widened = f32::from_bits((p as u32) << 16);
        let via_omega = narrow_like_omega(widened);
        // What the loader sends must equal what omega's f32 path stores today, for every pattern.
        assert_eq!(
            bf16_bits_as_uploaded_today(p),
            via_omega,
            "pattern {p:#06x}"
        );
        if via_omega != p {
            changed.push(p);
        }
        let exp = (p >> 7) & 0xff;
        let man = p & 0x7f;
        if exp == 0 && man != 0 {
            denormals += 1;
        }
        if exp == 0xff && man != 0 {
            nans += 1;
        }
        if exp == 0xff && man == 0 {
            infs += 1;
        }
    }
    // Every NaN payload, every denormal, both infinities and both zeros are in the sweep.
    assert_eq!((denormals, nans, infs), (254, 254, 2));
    // The only patterns that change: signaling NaNs (quiet bit clear), 63 per sign.
    assert_eq!(changed.len(), 126);
    for p in &changed {
        assert_eq!(p & 0x7f80, 0x7f80);
        assert_eq!(p & 0x0040, 0, "{p:#06x} is not a signaling NaN");
        assert_ne!(p & 0x007f, 0);
        assert_eq!(bf16_bits_as_uploaded_today(*p), p | 0x0040);
    }
    // Everything else, including all denormals and the quiet NaN payloads, is bit-identical.
    assert!(!changed.contains(&0x0001) && !changed.contains(&0x7fc1) && !changed.contains(&0x8000));
}

#[test]
fn transpose_puts_w_out_in_into_k_by_n() {
    let (n, k) = (3usize, 5usize);
    let src: Vec<u16> = (0..(n * k) as u16).map(|i| 0x3f80 + i).collect(); // finite bf16 values
    let raw: Vec<u8> = src.iter().flat_map(|b| b.to_le_bytes()).collect();
    let mut dst = vec![9u16; 4];
    transpose_bf16_for_upload(&raw, n, k, &mut dst);
    assert_eq!(dst.len(), n * k);
    for o in 0..n {
        for i in 0..k {
            assert_eq!(dst[i * n + o], src[o * k + i], "o={o} i={i}");
        }
    }
    // Larger than one 64x64 tile in both directions, odd sizes.
    let (n, k) = (131usize, 70usize);
    let src: Vec<u16> = (0..(n * k)).map(|i| (i as u16) | 0x4000).collect();
    let raw: Vec<u8> = src.iter().flat_map(|b| b.to_le_bytes()).collect();
    transpose_bf16_for_upload(&raw, n, k, &mut dst);
    for o in 0..n {
        for i in 0..k {
            assert_eq!(dst[i * n + o], src[o * k + i]);
        }
    }
}

// ---- a fake device ----

#[derive(Debug)]
struct FakeDevice {
    k: usize,
    n: usize,
    /// W as `[n][k]` f32, rebuilt from the uploaded `[k][n]` bf16 bits.
    w: Vec<f32>,
    calls: Mutex<u64>,
}

impl ResidentHandle for FakeDevice {
    fn matmul_f32(&self, m: usize, a: &[f32], c: &mut [f32]) -> Result<u64, String> {
        assert_eq!(a.len(), m * self.k);
        assert_eq!(c.len(), m * self.n);
        for r in 0..m {
            aien_inference_abi::tensor::matmul_vec(
                &a[r * self.k..(r + 1) * self.k],
                &self.w,
                &mut c[r * self.n..(r + 1) * self.n],
                self.k,
                self.n,
            );
        }
        *self.calls.lock().unwrap() += 1;
        Ok(1)
    }
}

#[derive(Default)]
struct FakeUploader {
    shapes: Mutex<Vec<(usize, usize)>>,
    /// Zero the first uploaded matrix (negative control).
    corrupt_first: bool,
    /// Fail on the Nth upload (0-based).
    fail_at: Option<usize>,
}

impl ResidentUploader for FakeUploader {
    fn upload_bf16(&self, k: usize, n: usize, bits: &[u16]) -> Result<ResidentWeight, String> {
        let idx = {
            let mut s = self.shapes.lock().unwrap();
            s.push((k, n));
            s.len() - 1
        };
        if self.fail_at == Some(idx) {
            return Err("NV_ERR_NO_MEMORY (fake)".to_string());
        }
        assert_eq!(bits.len(), k * n);
        let mut w = vec![0.0f32; n * k];
        for i in 0..k {
            for o in 0..n {
                w[o * k + i] = f32::from_bits((bits[i * n + o] as u32) << 16);
            }
        }
        if self.corrupt_first && idx == 0 {
            w.iter_mut().for_each(|v| *v = 0.0);
        }
        Ok(ResidentWeight::new(
            k,
            n,
            Arc::new(FakeDevice {
                k,
                n,
                w,
                calls: Mutex::new(0),
            }),
        ))
    }
}

/// The reference backend plus the one thing the real GB10 backend adds: it can run a resident
/// weight. Everything else delegates, so a difference can only come from the weights.
struct DeviceLikeBackend(ReferenceCpuBackend);

impl TensorBackend for DeviceLikeBackend {
    fn name(&self) -> &'static str {
        "DeviceLikeBackend (test)"
    }
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32) {
        self.0.rmsnorm(out, x, weight, eps)
    }
    fn rmsnorm_heads(&self, x: &mut [f32], weight: &[f32], head_dim: usize, eps: f32) {
        self.0.rmsnorm_heads(x, weight, head_dim, eps)
    }
    fn apply_rope(
        &self,
        q: &mut [f32],
        k: &mut [f32],
        pos: usize,
        head_dim: usize,
        nq: usize,
        nkv: usize,
        rope: &RopeParams,
    ) {
        self.0.apply_rope(q, k, pos, head_dim, nq, nkv, rope)
    }
    fn matmul_vec(&self, out: &mut [f32], x: &[f32], w: &[f32], out_dim: usize, in_dim: usize) {
        self.0.matmul_vec(out, x, w, out_dim, in_dim)
    }
    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]) {
        self.0.swiglu(out, gate, up)
    }
    fn gqa_attention(
        &self,
        out: &mut [f32],
        q: &[f32],
        kc: &[f32],
        vc: &[f32],
        seq_len: usize,
        nq: usize,
        nkv: usize,
        head_dim: usize,
    ) {
        self.0
            .gqa_attention(out, q, kc, vc, seq_len, nq, nkv, head_dim)
    }
    fn compute_logits(&self, l: &mut [f32], h: &[f32], e: &[f32], v: usize, hd: usize) {
        self.0.compute_logits(l, h, e, v, hd)
    }
    fn matmul_batch_w(
        &self,
        out: &mut [f32],
        x: &[f32],
        w: MatrixRef<'_>,
        batch: usize,
        in_dim: usize,
        out_dim: usize,
    ) {
        match w {
            MatrixRef::Host(h) => self.0.matmul_batch(out, x, h, batch, in_dim, out_dim),
            MatrixRef::Resident(r) => {
                assert_eq!((r.k(), r.n()), (in_dim, out_dim));
                r.matmul_f32(batch, x, out).unwrap();
            }
        }
    }
    fn matmul_vec_w(
        &self,
        out: &mut [f32],
        x: &[f32],
        w: MatrixRef<'_>,
        out_dim: usize,
        in_dim: usize,
    ) {
        self.matmul_batch_w(out, x, w, 1, in_dim, out_dim)
    }
    fn compute_logits_w(
        &self,
        l: &mut [f32],
        h: &[f32],
        w: MatrixRef<'_>,
        vocab: usize,
        hidden: usize,
    ) {
        self.matmul_batch_w(l, h, w, 1, hidden, vocab)
    }
}

fn host_weights() -> TransformerWeights {
    let dir = tiny_dir();
    let config = load_model_config(&dir).unwrap();
    TransformerWeights::load_from_safetensors(dir.join("model.safetensors"), &config).unwrap()
}

fn resident_weights(up: &FakeUploader) -> Result<TransformerWeights, CheckpointError> {
    let dir = tiny_dir();
    let config = load_model_config(&dir).unwrap();
    load_resident_weights(dir.join("model.safetensors"), &config, up, &mut |_| {})
}

fn max_abs(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

const IDS: [u32; 6] = [3, 17, 5, 9, 1, 22];

fn prefill_logits(weights: TransformerWeights) -> Vec<f32> {
    let mut be = NativeTransformerBackend::with_backend(
        weights,
        Arc::new(DeviceLikeBackend(Default::default())),
    );
    be.prefill_sequence(1, &IDS).expect("prefill")
}

#[test]
fn resident_load_holds_no_host_matmul_weight_and_matches_host_logits_exactly() {
    let up = FakeUploader::default();
    let w = resident_weights(&up).expect("resident load");
    let layers = w.layers.len();
    // Every one of the seven matmul weights per layer, plus the (tied) head, is resident.
    for l in &w.layers {
        for m in [
            &l.q_proj,
            &l.k_proj,
            &l.v_proj,
            &l.o_proj,
            &l.gate_proj,
            &l.up_proj,
            &l.down_proj,
        ] {
            assert!(m.is_resident());
        }
    }
    assert!(w.config.tie_word_embeddings);
    assert!(matches!(w.lm_head, Some(MatrixWeight::Resident(_))));
    assert_eq!(up.shapes.lock().unwrap().len(), layers * 7 + 1);
    // The host keeps what the host reads: embedding rows and norms (readers: see the PR text).
    assert_eq!(w.embed_tokens, host_weights().embed_tokens);
    assert_eq!(w.final_norm, host_weights().final_norm);

    let got = prefill_logits(w);
    let want = prefill_logits(host_weights());
    assert_eq!(
        got, want,
        "resident logits must be bit-identical to host logits"
    );
}

#[test]
fn negative_control_a_wrong_resident_matrix_moves_the_logits() {
    let up = FakeUploader {
        corrupt_first: true,
        ..Default::default()
    };
    let got = prefill_logits(resident_weights(&up).unwrap());
    let want = prefill_logits(host_weights());
    assert!(max_abs(&got, &want) > 1e-3, "comparison would be vacuous");
}

#[test]
fn uploads_run_layer_by_layer_then_the_head() {
    let dir = tiny_dir();
    let config = load_model_config(&dir).unwrap();
    let mut names = Vec::new();
    load_resident_weights(
        dir.join("model.safetensors"),
        &config,
        &FakeUploader::default(),
        &mut |n| names.push(n.to_string()),
    )
    .unwrap();
    assert_eq!(
        names.first().unwrap(),
        "model.layers.0.self_attn.q_proj.weight"
    );
    assert_eq!(names.last().unwrap(), "model.embed_tokens.weight");
    assert_eq!(names.len(), config.num_layers * 7 + 1);
}

#[test]
fn an_upload_failure_is_a_named_error_never_a_host_copy() {
    let up = FakeUploader {
        fail_at: Some(3),
        ..Default::default()
    };
    match resident_weights(&up) {
        Err(CheckpointError::ResidentUpload { tensor, message }) => {
            assert_eq!(tensor, "model.layers.0.self_attn.o_proj.weight");
            assert!(message.contains("NV_ERR_NO_MEMORY"));
        }
        other => panic!("expected ResidentUpload, got {:?}", other.map(|_| ())),
    }
}

#[test]
fn a_missing_tensor_is_refused_before_any_upload() {
    let dir = tiny_dir();
    let mut config = load_model_config(&dir).unwrap();
    config.num_layers += 1; // the file has no layer N
    let up = FakeUploader::default();
    let r = load_resident_weights(dir.join("model.safetensors"), &config, &up, &mut |_| {});
    assert!(matches!(r, Err(CheckpointError::MissingTensor(_))));
    assert!(
        up.shapes.lock().unwrap().is_empty(),
        "uploaded before validating"
    );
}

#[test]
#[should_panic(expected = "RESIDENT_WEIGHT_REFUSED")]
fn the_cpu_reference_backend_refuses_a_resident_weight_by_name() {
    let w = resident_weights(&FakeUploader::default()).unwrap();
    let be = ReferenceCpuBackend::new();
    let mut out = vec![0.0; w.config.num_heads * w.config.head_dim];
    let x = vec![0.0; w.config.hidden_dim()];
    let (od, id) = (out.len(), x.len());
    be.matmul_vec_w(&mut out, &x, w.layers[0].q_proj.as_ref(), od, id);
}

#[test]
#[should_panic(expected = "RESIDENT_WEIGHT_REFUSED")]
fn host_slice_access_to_a_resident_weight_is_refused_by_name() {
    let w = resident_weights(&FakeUploader::default()).unwrap();
    let _ = w.layers[0].q_proj.len();
}

#[test]
fn resident_load_is_chosen_only_on_the_native_strict_opted_in_qwen3_path() {
    let dir = tiny_dir();
    let tiny = load_model_config(&dir).unwrap(); // qk_norm, head_dim 16
    let mut q4b = tiny.clone();
    q4b.head_dim = 128; // the head_dim the GB10 engine accepts for Qwen3
                        // (native, strict, opted in, accepted Qwen3) -> resident
    assert!(resident_load_wanted(true, true, true, &q4b));
    // each condition off -> host f32 load, unchanged
    assert!(!resident_load_wanted(false, true, true, &q4b)); // stub build / no GB10
    assert!(!resident_load_wanted(true, false, true, &q4b)); // dev-fallback needs host values
    assert!(!resident_load_wanted(true, true, false, &q4b)); // Qwen3 switched off (=0): refused
    assert!(!resident_load_wanted(true, true, true, &tiny)); // engine refuses head_dim 16
    let mut llama = q4b.clone();
    llama.qk_norm = false; // other models keep their path
    assert!(!resident_load_wanted(true, true, true, &llama));
}

#[test]
fn streamed_checkpoint_reads_the_same_bytes_as_the_whole_file_loader() {
    let dir = tiny_dir();
    let config = load_model_config(&dir).unwrap();
    let catalog = aien_inference_abi::llama_catalog(&config);
    let path = dir.join("model.safetensors");
    let whole = aien_inference_abi::load_checkpoint_with_catalog(&path, &catalog).unwrap();
    let streamed = aien_inference_abi::StreamedCheckpoint::open(&path, &catalog).unwrap();
    let mut buf = Vec::new();
    for (name, shape) in &catalog {
        let n = streamed.read_bf16_bytes(name, &mut buf).unwrap();
        assert_eq!(n, shape.iter().product::<usize>(), "{name}");
        assert_eq!(buf.as_slice(), whole.tensor_bytes(name).unwrap(), "{name}");
        assert_eq!(streamed.shape(name).unwrap(), shape.as_slice());
    }
    // A catalog tensor the file lacks is refused when the checkpoint is opened.
    let mut bad = catalog.clone();
    bad.push(("model.layers.9.mlp.up_proj.weight".to_string(), vec![4, 4]));
    assert!(matches!(
        aien_inference_abi::StreamedCheckpoint::open(&path, &bad),
        Err(CheckpointError::MissingTensor(_))
    ));
}

fn fake_resident(k: usize, n: usize) -> (ResidentWeight, Vec<f32>) {
    // W [n][k] with bf16-exact values, and its resident twin built through the real transpose.
    let w: Vec<f32> = (0..n * k)
        .map(|i| f32::from_bits((0x3f00 + (i % 200) as u32) << 16))
        .collect();
    let raw: Vec<u8> = w
        .iter()
        .flat_map(|v| ((v.to_bits() >> 16) as u16).to_le_bytes())
        .collect();
    let mut bits = Vec::new();
    transpose_bf16_for_upload(&raw, n, k, &mut bits);
    (FakeUploader::default().upload_bf16(k, n, &bits).unwrap(), w)
}

#[test]
fn omega_backend_runs_a_resident_weight_through_the_resident_handle() {
    let (k, n) = (16usize, 24usize);
    let (r, w) = fake_resident(k, n);
    let be = aien_inference_abi::OmegaGb10Backend::new(); // stub build: no chip, no host path used
    let x: Vec<f32> = (0..3 * k).map(|i| (i % 7) as f32 * 0.25).collect();
    let mut got = vec![0.0; 3 * n];
    be.matmul_batch_w(&mut got, &x, MatrixRef::Resident(&r), 3, k, n);
    let mut want = vec![0.0; 3 * n];
    ReferenceCpuBackend::new().matmul_batch(&mut want, &x, &w, 3, k, n);
    assert_eq!(got, want);
    assert_eq!(be.chip_calls(), 1);
    assert_eq!(be.fallback_count(), 0);
    let mut one = vec![0.0; n];
    be.matmul_vec_w(&mut one, &x[..k], MatrixRef::Resident(&r), n, k);
    assert_eq!(one, want[..n]);
    be.compute_logits_w(&mut one, &x[..k], MatrixRef::Resident(&r), n, k);
    assert_eq!(one, want[..n]);
    assert_eq!(be.chip_calls(), 3);
}

#[test]
#[should_panic(expected = "RESIDENT_WEIGHT_REFUSED")]
fn omega_backend_refuses_a_resident_weight_of_the_wrong_shape() {
    let (r, _) = fake_resident(16, 24);
    let be = aien_inference_abi::OmegaGb10Backend::new();
    let mut out = vec![0.0; 32];
    be.matmul_vec_w(&mut out, &[0.0; 16], MatrixRef::Resident(&r), 32, 16);
}
