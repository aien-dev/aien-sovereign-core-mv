//! Checkpoint indexer and the file-backed read path: random access to one
//! expert, byte-for-byte round trip against the checkpoint bytes, determinism,
//! read-only source, negative controls and on-disk damage.

mod common;

use aien_expert_store::format::hex;
use aien_expert_store::indexer::{index_qwen3_moe, read_expert, Checkpoint, Qwen3MoeExperts};
use aien_expert_store::store::INDEX_FILE;
use aien_expert_store::{
    ExpertStore, IndexEntry, IoMode, Outcome, Projection, ReadAdapter, ReadError, ReadLimits,
    StagingBuffer, StoreError, StoreLimits,
};
use common::*;
use std::sync::Arc;
use std::time::{Duration, Instant};

fn shape() -> Qwen3MoeExperts {
    Qwen3MoeExperts {
        model_digest: MODEL,
        hidden: H,
        moe_intermediate: I,
        experts: EXPERTS,
    }
}

fn read_one(a: &ReadAdapter, e: IndexEntry) -> Outcome {
    let b = StagingBuffer::new(e.object_len as usize).unwrap();
    a.submit(e, b, Instant::now() + Duration::from_secs(30))
        .unwrap();
    a.wait(Duration::from_secs(10)).unwrap().outcome
}

fn dir_digest(dir: &std::path::Path) -> Vec<(String, [u8; 32])> {
    let mut v: Vec<_> = std::fs::read_dir(dir)
        .unwrap()
        .map(|e| {
            let p = e.unwrap().path();
            let d = aien_expert_store::format::sha256(&std::fs::read(&p).unwrap());
            (p.file_name().unwrap().to_string_lossy().into_owned(), d)
        })
        .collect();
    v.sort();
    v
}

#[test]
fn index_then_random_access_round_trips_byte_for_byte() {
    let ck = TempDir::new("ckpt");
    let expected = write_checkpoint(ck.path());
    let before = dir_digest(ck.path());
    let st = TempDir::new("st");
    let mut store = ExpertStore::open(st.path(), StoreLimits::default()).unwrap();
    let mut ckpt = Checkpoint::open(ck.path()).unwrap();
    let entries = index_qwen3_moe(&mut ckpt, &shape(), 0..LAYERS, &mut store).unwrap();
    assert_eq!(entries.len(), (LAYERS * EXPERTS * 3) as usize);
    assert_eq!(
        dir_digest(ck.path()),
        before,
        "checkpoint must be untouched"
    );

    let store = ExpertStore::open(st.path(), StoreLimits::default()).unwrap();
    let a = ReadAdapter::new(
        Arc::new(store.file_source(IoMode::Buffered)),
        ReadLimits::default(),
    )
    .unwrap();
    // Random access order, one expert at a time.
    for i in [7usize, 0, 17, 3, 11, 16, 1] {
        let (k, w, s) = &expected[i];
        let e = store.get(k).expect("indexed");
        match read_one(&a, e) {
            Outcome::Ready(r) => {
                assert_eq!(r.weights(), &w[..], "weights of {k:?}");
                assert_eq!(r.scales(), &s[..], "scales of {k:?}");
            }
            o => panic!("{o:?}"),
        }
        // Same bytes as a direct read of the checkpoint tensor.
        let mut c = Checkpoint::open(ck.path()).unwrap();
        let (dw, ds, _) =
            read_expert(&mut c, &shape(), k.layer, k.expert, k.projection, u64::MAX).unwrap();
        assert_eq!((&dw, &ds), (w, s));
    }
}

#[test]
fn indexing_is_deterministic() {
    let ck = TempDir::new("det");
    write_checkpoint(ck.path());
    let (s1, s2) = (TempDir::new("d1"), TempDir::new("d2"));
    for s in [&s1, &s2] {
        let mut store = ExpertStore::open(s.path(), StoreLimits::default()).unwrap();
        let mut ckpt = Checkpoint::open(ck.path()).unwrap();
        index_qwen3_moe(&mut ckpt, &shape(), 0..LAYERS, &mut store).unwrap();
    }
    assert_eq!(
        std::fs::read(s1.path().join(INDEX_FILE)).unwrap(),
        std::fs::read(s2.path().join(INDEX_FILE)).unwrap()
    );
    assert_eq!(
        dir_digest(&s1.path().join("objects")),
        dir_digest(&s2.path().join("objects"))
    );
}

#[test]
fn negative_controls_missing_dtype_shape_expert_and_store_location() {
    let ck = TempDir::new("neg");
    write_checkpoint(ck.path());
    let st = TempDir::new("negst");
    let mut store = ExpertStore::open(st.path(), StoreLimits::default()).unwrap();
    let mut ckpt = Checkpoint::open(ck.path()).unwrap();

    // Missing layer.
    assert!(matches!(
        index_qwen3_moe(&mut ckpt, &shape(), 0..LAYERS + 1, &mut store),
        Err(StoreError::MissingTensor(_))
    ));
    assert!(
        !st.path().join(INDEX_FILE).exists(),
        "nothing committed on failure"
    );

    // Shape mismatch (wrong hidden size).
    let mut bad = shape();
    bad.hidden = H + 128;
    assert!(matches!(
        read_expert(&mut ckpt, &bad, 0, 0, Projection::Gate, u64::MAX),
        Err(StoreError::ShapeMismatch { .. })
    ));
    // Expert id out of range.
    assert!(matches!(
        read_expert(&mut ckpt, &shape(), 0, EXPERTS, Projection::Gate, u64::MAX),
        Err(StoreError::MissingTensor(_))
    ));
    // Read bound.
    assert!(matches!(
        read_expert(&mut ckpt, &shape(), 0, 0, Projection::Gate, 100),
        Err(StoreError::ObjectTooLarge { .. })
    ));
    // Store inside the checkpoint directory.
    let inner = ck.path().join("store");
    let mut s2 = ExpertStore::open(&inner, StoreLimits::default()).unwrap();
    assert_eq!(
        index_qwen3_moe(&mut ckpt, &shape(), 0..1, &mut s2).unwrap_err(),
        StoreError::StoreInsideCheckpoint
    );
}

#[test]
fn dtype_mismatch_and_hostile_shard_names_are_refused() {
    let ck = TempDir::new("dt");
    let w = pattern((I * H) as usize, 1);
    write_shard(
        &ck.path().join("a.safetensors"),
        &[
            T {
                name: "model.layers.0.mlp.experts.0.gate_proj.weight".into(),
                dtype: "BF16",
                shape: vec![I, H],
                data: w,
            },
            T {
                name: "model.layers.0.mlp.experts.0.up_proj.weight".into(),
                dtype: "F8_E4M3",
                shape: vec![I, H],
                data: pattern((I * H) as usize, 2),
            },
            T {
                name: "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv".into(),
                dtype: "F16",
                shape: vec![2, 2],
                data: vec![0; 8],
            },
        ],
    );
    let index = serde_json::json!({"weight_map": {
        "model.layers.0.mlp.experts.0.gate_proj.weight": "a.safetensors",
        "model.layers.0.mlp.experts.0.up_proj.weight": "a.safetensors",
        "model.layers.0.mlp.experts.0.up_proj.weight_scale_inv": "a.safetensors",
        "model.layers.0.mlp.experts.0.down_proj.weight": "../escape.safetensors",
    }});
    std::fs::write(
        ck.path().join("model.safetensors.index.json"),
        serde_json::to_vec(&index).unwrap(),
    )
    .unwrap();
    let mut c = Checkpoint::open(ck.path()).unwrap();
    assert!(matches!(
        read_expert(&mut c, &shape(), 0, 0, Projection::Gate, u64::MAX),
        Err(StoreError::DtypeMismatch { .. })
    ));
    assert!(matches!(
        read_expert(&mut c, &shape(), 0, 0, Projection::Up, u64::MAX),
        Err(StoreError::DtypeMismatch { .. })
    ));
    assert!(matches!(
        read_expert(&mut c, &shape(), 0, 0, Projection::Down, u64::MAX),
        Err(StoreError::Checkpoint(_))
    ));
}

#[test]
fn on_disk_damage_and_symlinks_fail_closed() {
    let st = TempDir::new("dmg");
    let mut store = ExpertStore::open(st.path(), StoreLimits::default()).unwrap();
    let k = key(0, 0, Projection::Down);
    let (_, _, w, s) = artifact(k, 256, 192, 4);
    let e = store
        .put(k, 256, 192, aien_expert_store::ScaleDtype::Bf16, &w, &s)
        .unwrap();
    let a = ReadAdapter::new(
        Arc::new(store.file_source(IoMode::Buffered)),
        ReadLimits::default(),
    )
    .unwrap();
    assert!(matches!(read_one(&a, e), Outcome::Ready(_)));
    let p = st.path().join("objects").join(hex(&e.data_digest));
    let good = std::fs::read(&p).unwrap();

    let mut flipped = good.clone();
    flipped[4096 + 10] ^= 0x80;
    std::fs::write(&p, &flipped).unwrap();
    assert!(matches!(
        read_one(&a, e),
        Outcome::Failed {
            error: ReadError::DigestMismatch,
            ..
        }
    ));

    std::fs::write(&p, &good[..good.len() - 4096]).unwrap();
    assert!(matches!(
        read_one(&a, e),
        Outcome::Failed {
            error: ReadError::LengthMismatch { .. },
            ..
        }
    ));

    // A planted symlink is not followed.
    let elsewhere = st.path().join("elsewhere");
    std::fs::write(&elsewhere, &good).unwrap();
    std::fs::remove_file(&p).unwrap();
    std::os::unix::fs::symlink(&elsewhere, &p).unwrap();
    match read_one(&a, e) {
        Outcome::Failed {
            error: ReadError::Io(_),
            buf,
        } => {
            assert!(buf.as_slice().iter().all(|&b| b == 0))
        }
        o => panic!("{o:?}"),
    }

    std::fs::remove_file(&p).unwrap();
    assert!(matches!(
        read_one(&a, e),
        Outcome::Failed {
            error: ReadError::NotFound,
            ..
        }
    ));
}

#[test]
fn direct_mode_reads_verify_or_refuse_explicitly() {
    let st = TempDir::new("direct");
    let mut store = ExpertStore::open(st.path(), StoreLimits::default()).unwrap();
    let k = key(0, 1, Projection::Gate);
    let (_, _, w, s) = artifact(k, 192, 256, 5);
    let e = store
        .put(k, 192, 256, aien_expert_store::ScaleDtype::Bf16, &w, &s)
        .unwrap();
    let a = ReadAdapter::new(
        Arc::new(store.file_source(IoMode::Direct)),
        ReadLimits::default(),
    )
    .unwrap();
    match read_one(&a, e) {
        Outcome::Ready(r) => {
            assert_eq!(r.weights(), &w[..]);
            eprintln!("direct mode: verified read");
        }
        // Not Linux, or a file system without O_DIRECT (tmpfs): an explicit
        // refusal, never a silent buffered fallback.
        Outcome::Failed {
            error: ReadError::DirectIoUnsupported,
            ..
        } => eprintln!("direct mode: refused (DirectIoUnsupported)"),
        o => panic!("{o:?}"),
    }
}
