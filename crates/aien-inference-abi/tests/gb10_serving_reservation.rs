//! sovereign-core#277 (cut toward it, not a fix): when the daemon serves Qwen3 on the GB10
//! (the already opt-in path), start-up reserves the serving buffers from declared bounds,
//! prepares every matmul kernel, seals, and refuses to serve by name when omega refuses.
//! CPU only: the omega calls are injected, nothing touches a driver. The daemon-level ordering
//! tests (prepare of every rows x shape, seal last) live in aien-cli `commands.rs`.
use aien_inference_abi::gb10_serving::{
    reserve_gb10_serving_with, serving_matmul_shapes, ServingLimits, ServingOps,
};
use aien_inference_abi::model_config_from_hf_json;
use aien_omega_gpu::ServingBounds;
use std::cell::{Cell, RefCell};
use std::path::PathBuf;

fn qwen3_4b() -> aien_inference_abi::ModelConfig {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join("qwen3-4b-instruct-2507-config");
    let cfg = std::fs::read_to_string(dir.join("config.json")).unwrap();
    let gen = std::fs::read_to_string(dir.join("generation_config.json")).unwrap();
    model_config_from_hf_json("Qwen/Qwen3-4B-Instruct-2507", &cfg, Some(&gen)).unwrap()
}

/// The daemon's declared limits (max_batch_size 256, prefill chunk 128, arena 4096).
fn daemon_limits() -> ServingLimits {
    ServingLimits {
        context_tokens: 4096,
        max_batch_rows: 256,
        prefill_chunk_rows: 128,
    }
}

#[derive(Default)]
struct Ops {
    reserved: RefCell<Vec<ServingBounds>>,
    prepares: Cell<u32>,
    seals: Cell<u32>,
    refuse_reserve: bool,
}

impl ServingOps for Ops {
    fn reserve(&self, b: &ServingBounds) -> Result<(), String> {
        self.reserved.borrow_mut().push(*b);
        if self.refuse_reserve {
            Err("omega_gpu rc=-4 (CHIP_FAIL) (stage: nvrm_alloc scratch)".to_string())
        } else {
            Ok(())
        }
    }
    fn prepare(&self, _m: u32, _k: u32, _n: u32) -> Result<(), String> {
        self.prepares.set(self.prepares.get() + 1);
        Ok(())
    }
    fn seal(&self) {
        self.seals.set(self.seals.get() + 1);
    }
}

#[test]
fn qwen3_on_gb10_reserves_once_with_the_declared_bounds() {
    let ops = Ops::default();
    let got = reserve_gb10_serving_with(&qwen3_4b(), &daemon_limits(), true, true, &ops)
        .expect("reserved")
        .expect("Qwen3 on GB10 must reserve");
    let calls = ops.reserved.borrow();
    assert_eq!(calls.len(), 1, "exactly one reservation call");
    let b = calls[0];
    assert_eq!(got.bounds, b);
    assert_eq!((b.num_q_heads, b.num_kv_heads, b.head_dim), (32, 8, 128));
    assert_eq!(b.max_context, 4096);
    assert_eq!(
        b.max_seqs, 1,
        "the GB10 backend runs attention one sequence at a time"
    );
    assert_eq!(b.kv_block_size as usize, qwen3_4b().block_size);
    assert_eq!(
        b.max_rows, 256,
        "decode batch rows (256) exceed the prefill chunk (128)"
    );
    assert_eq!(b.max_k, 9728, "down_proj input is the intermediate size");
    assert_eq!(b.max_n_one_row, 151936, "the logits projection");
    assert!(b.max_n >= 9728);
    // omega's own sweep (omega#338, fake layer): Qwen3-4B needs 95 distinct kernels, so the 32
    // slots of omega#333 are not enough; 128 is omega's maximum.
    assert_eq!(b.kernel_slots, 128);
    assert_eq!(
        got.bytes.total(),
        194_170_880,
        "bytes asked for the daemon bounds"
    );
    assert_eq!((ops.prepares.get(), ops.seals.get()), (256 * 6, 1));
    assert_eq!(got.prepare_calls, 256 * 6);
}

#[test]
fn shapes_come_from_the_model_config() {
    let c = qwen3_4b();
    let s = serving_matmul_shapes(&c).unwrap();
    assert_eq!(
        s,
        vec![
            (2560, 1024),
            (2560, 4096),
            (2560, 9728),
            (2560, 151936),
            (4096, 2560),
            (9728, 2560)
        ]
    );
}

#[test]
fn context_is_the_smaller_of_the_kv_plan_and_the_attention_limit() {
    let mut seen = Vec::new();
    for ctx in [2048usize, 262144] {
        let ops = Ops::default();
        let mut l = daemon_limits();
        l.context_tokens = ctx;
        reserve_gb10_serving_with(&qwen3_4b(), &l, true, true, &ops).unwrap();
        seen.push(ops.reserved.borrow()[0].max_context);
    }
    assert_eq!(seen, vec![2048, 4096]);
}

#[test]
fn a_refusal_stops_serving_by_name_with_no_fallback() {
    let ops = Ops {
        refuse_reserve: true,
        ..Ops::default()
    };
    let err = reserve_gb10_serving_with(&qwen3_4b(), &daemon_limits(), true, true, &ops)
        .expect_err("a refused reservation must refuse to serve");
    assert!(err.contains("GB10_SERVING_RESERVATION refused"), "{err}");
    assert!(err.contains("CHIP_FAIL"), "names omega's reason: {err}");
    assert!(err.contains("sovereign-core#277"), "{err}");
    assert!(err.contains("no on-demand fallback"), "{err}");
    assert_eq!((ops.prepares.get(), ops.seals.get()), (0, 0));
}

#[test]
fn nothing_changes_off_the_enabled_qwen3_gb10_path() {
    let none = |native: bool, opted: bool, cfg: &aien_inference_abi::ModelConfig| {
        let ops = Ops::default();
        let r = reserve_gb10_serving_with(cfg, &daemon_limits(), native, opted, &ops).unwrap();
        assert!(r.is_none());
        assert!(
            ops.reserved.borrow().is_empty() && ops.prepares.get() == 0 && ops.seals.get() == 0
        );
    };
    none(false, true, &qwen3_4b()); // not native (CPU daemon)
    none(true, false, &qwen3_4b()); // switched off (=0): the refusal still applies
    let mut llama = qwen3_4b();
    llama.qk_norm = false; // a non-Qwen3 model on the GB10 is untouched
    none(true, true, &llama);
}

#[test]
fn the_alloc_probe_logs_the_baseline_and_shows_any_nonzero_delta() {
    use aien_inference_abi::gb10_serving::{AllocProbe, ServingProbe};
    use aien_omega_gpu::AllocStats;
    use std::sync::{Arc, Mutex};
    let now = Arc::new(Mutex::new(AllocStats {
        allocs: 600,
        alloc_bytes: 7_000_000_000,
        alloc_failures: 0,
        frees: 12,
    }));
    let n2 = now.clone();
    let p = AllocProbe::new(Box::new(move || Some(*n2.lock().unwrap())));
    assert_eq!(
        p.after_request(),
        None,
        "no line before the warm-up baseline"
    );
    assert_eq!(
        p.after_warm_up().unwrap(),
        "GB10_SERVING_ALLOC after_warmup allocs=600 bytes=7000000000 failures=0 frees=12"
    );
    let quiet = p.after_request().unwrap();
    assert!(
        quiet.contains("allocs_since_warmup=0") && !quiet.contains("NONZERO"),
        "{quiet}"
    );
    // frees alone (teardown) are flagged under their own name, never as an allocation
    now.lock().unwrap().frees += 33;
    let freed = p.after_request().unwrap();
    assert!(
        freed.contains("frees_since_warmup=33")
            && freed.contains("FREES:")
            && !freed.contains("NONZERO"),
        "{freed}"
    );
    now.lock().unwrap().allocs += 3;
    now.lock().unwrap().alloc_bytes += 12288;
    now.lock().unwrap().alloc_failures += 1;
    let loud = p.after_request().unwrap();
    assert!(loud.contains("allocs_since_warmup=3"), "{loud}");
    assert!(loud.contains("bytes_since_warmup=12288"), "{loud}");
    assert!(loud.contains("failures_since_warmup=1"), "{loud}");
    assert!(
        loud.contains("NONZERO"),
        "a nonzero delta must be visible: {loud}"
    );
    // the stub build has no counters: the probe stays silent rather than inventing zeros
    let none = AllocProbe::new(Box::new(|| None));
    assert_eq!((none.after_warm_up(), none.after_request()), (None, None));
}
