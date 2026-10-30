//! sc#629 acceptance: random-access expert weight/scale reads from the
//! content-addressed expert store equal, byte for byte, what the resident
//! OWN-30B loader's shard reader (`ShardSource`, the source behind
//! `load_qwen3_moe_omega`) returns for the same FP8 checkpoint tensors.

use aien_expert_store::indexer::{index_qwen3_moe, Checkpoint, Qwen3MoeExperts};
use aien_expert_store::{
    ExpertKey, ExpertStore, IoMode, Outcome, Projection, QuantFormat, ReadAdapter, ReadLimits,
    StagingBuffer, StoreLimits,
};
use aien_inference_abi::qwen3_moe_omega::{ShardSource, TensorSource};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

const H: usize = 256;
const I: usize = 192;
const LAYERS: u32 = 2;
const EXPERTS: u32 = 4;
const MODEL: [u8; 32] = [0x30; 32];

fn fill(len: usize, seed: u32) -> Vec<u8> {
    let mut x = seed.wrapping_mul(2_654_435_761) | 1;
    (0..len)
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            (x >> 8) as u8
        })
        .collect()
}

fn dims(p: Projection) -> (usize, usize) {
    match p {
        Projection::Gate | Projection::Up => (I, H),
        Projection::Down => (H, I),
    }
}

fn write_checkpoint(dir: &Path) {
    let mut weight_map = serde_json::Map::new();
    for l in 0..LAYERS {
        let shard = format!("model-{:05}-of-{LAYERS:05}.safetensors", l + 1);
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for e in 0..EXPERTS {
            for p in Projection::ALL {
                let (r, c) = dims(p);
                let base = format!("model.layers.{l}.mlp.experts.{e}.{}", p.checkpoint_name());
                let seed = l * 1000 + e * 10 + p as u32;
                for (name, dtype, shape, bytes) in [
                    (
                        format!("{base}.weight"),
                        "F8_E4M3",
                        vec![r, c],
                        fill(r * c, seed),
                    ),
                    (
                        format!("{base}.weight_scale_inv"),
                        "BF16",
                        vec![r.div_ceil(128), c.div_ceil(128)],
                        fill(r.div_ceil(128) * c.div_ceil(128) * 2, seed + 7),
                    ),
                ] {
                    let start = data.len();
                    data.extend_from_slice(&bytes);
                    header.insert(
                        name.clone(),
                        serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [start, data.len()]}),
                    );
                    weight_map.insert(name, serde_json::Value::String(shard.clone()));
                }
            }
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut out = (h.len() as u64).to_le_bytes().to_vec();
        out.extend_from_slice(&h);
        out.extend_from_slice(&data);
        std::fs::write(dir.join(shard), out).unwrap();
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"weight_map": weight_map})).unwrap(),
    )
    .unwrap();
}

fn temp(tag: &str) -> PathBuf {
    let p = std::env::temp_dir().join(format!("sc629-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&p);
    std::fs::create_dir_all(&p).unwrap();
    p
}

#[test]
fn expert_store_reads_equal_the_resident_loader_bytes() {
    let ck = temp("ckpt");
    let st = temp("store");
    write_checkpoint(&ck);
    let shape = Qwen3MoeExperts {
        model_digest: MODEL,
        hidden: H as u64,
        moe_intermediate: I as u64,
        experts: EXPERTS,
    };
    let mut store = ExpertStore::open(&st, StoreLimits::default()).unwrap();
    let mut ckpt = Checkpoint::open(&ck).unwrap();
    index_qwen3_moe(&mut ckpt, &shape, 0..LAYERS, &mut store).unwrap();

    let reader = ReadAdapter::new(
        Arc::new(store.file_source(IoMode::Buffered)),
        ReadLimits::default(),
    )
    .unwrap();
    let mut resident = ShardSource::open(&ck).unwrap();
    let mut checked = 0;
    // A scattered order: each read touches one expert only.
    for (l, e) in [(1u32, 3u32), (0, 0), (1, 0), (0, 2), (1, 1), (0, 3)] {
        for p in Projection::ALL {
            let key = ExpertKey {
                model_digest: MODEL,
                layer: l,
                expert: e,
                projection: p,
                quant: QuantFormat::Fp8E4m3Block128,
            };
            let entry = store.get(&key).expect("indexed");
            let buf = StagingBuffer::new(entry.object_len as usize).unwrap();
            reader
                .submit(entry, buf, Instant::now() + Duration::from_secs(30))
                .unwrap();
            let Outcome::Ready(ready) = reader.wait(Duration::from_secs(10)).unwrap().outcome
            else {
                panic!("read of {key:?} did not verify");
            };
            let (r, c) = dims(p);
            let base = format!("model.layers.{l}.mlp.experts.{e}.{}", p.checkpoint_name());
            let mut w = vec![0u8; r * c];
            resident
                .read_into(&format!("{base}.weight"), "F8_E4M3", &[r, c], &mut w)
                .unwrap();
            let sshape = [r.div_ceil(128), c.div_ceil(128)];
            let mut s = vec![0u8; sshape[0] * sshape[1] * 2];
            resident
                .read_into(&format!("{base}.weight_scale_inv"), "BF16", &sshape, &mut s)
                .unwrap();
            assert_eq!(ready.weights(), &w[..], "weights {key:?}");
            assert_eq!(ready.scales(), &s[..], "scales {key:?}");
            checked += 1;
        }
    }
    assert_eq!(checked, 18);
    let _ = std::fs::remove_dir_all(&ck);
    let _ = std::fs::remove_dir_all(&st);
}
