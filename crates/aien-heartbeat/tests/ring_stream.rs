//! End-to-end on a small synthetic safetensors checkpoint (no O_DIRECT, so it
//! runs on tmpfs too).

use std::collections::{BTreeMap, HashSet};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

use aien_heartbeat::{HeartbeatError, Manifest, Ring, Schedule, StreamConfig, Streamer};

const LAYERS: u16 = 2;
const EXPERTS: u16 = 3;
const PROJ: [&str; 3] = ["down_proj", "gate_proj", "up_proj"];

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aien-heartbeat-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

/// Writes one shard holding every expert tensor (plus one non-expert tensor)
/// and the index. Returns the shard path.
fn write_model(dir: &Path) -> PathBuf {
    write_shards(dir, |_| 0).remove(0)
}

/// Writes shards with every tensor placed by `pick(name) -> shard index`
/// (0 or 1), plus the index. Returns the shard paths.
fn write_shards(dir: &Path, pick: impl Fn(&str) -> usize) -> Vec<PathBuf> {
    let names = [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    ];
    let mut headers = [serde_json::Map::new(), serde_json::Map::new()];
    let mut datas = [Vec::new(), Vec::new()];
    let mut weight_map = BTreeMap::new();
    let mut push = |name: String, len: usize, fill: u8| {
        let s = pick(&name);
        let data = &mut datas[s];
        let a = data.len();
        data.extend((0..len).map(|k| fill.wrapping_add(k as u8)));
        headers[s].insert(
            name.clone(),
            serde_json::json!({"dtype": "U8", "shape": [len], "data_offsets": [a, data.len()]}),
        );
        weight_map.insert(name, names[s].to_string());
    };
    push("model.embed_tokens.weight".into(), 333, 1);
    for l in 0..LAYERS {
        for e in 0..EXPERTS {
            for p in PROJ {
                let base = format!("model.layers.{l}.mlp.experts.{e}.{p}");
                push(
                    format!("{base}.weight"),
                    5000 + 17 * e as usize,
                    (l * 31 + e) as u8,
                );
                push(format!("{base}.weight_scale_inv"), 24, 9);
            }
        }
    }
    let mut paths = Vec::new();
    for s in 0..2 {
        if headers[s].is_empty() {
            continue;
        }
        let h = serde_json::to_vec(&headers[s]).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(&h);
        file.extend(&datas[s]);
        let path = dir.join(names[s]);
        std::fs::write(&path, file).unwrap();
        paths.push(path);
    }
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": weight_map})).unwrap(),
    )
    .unwrap();
    paths
}

#[test]
fn expert_split_across_shards_streams_as_two_segments() {
    let d = tmpdir("split");
    // Expert (1,0): down + gate in shard 0, up in shard 1; everything after it
    // in shard 1 too (as at a real shard boundary).
    let boundary = |n: &str| {
        let after = [
            "model.layers.1.mlp.experts.0.up_proj",
            "model.layers.1.mlp.experts.1",
            "model.layers.1.mlp.experts.2",
        ];
        usize::from(after.iter().any(|p| n.starts_with(p)))
    };
    let paths = write_shards(&d, boundary);
    let m = Manifest::build(&d).unwrap();
    let split = m
        .experts
        .iter()
        .find(|e| (e.layer, e.expert) == (1, 0))
        .unwrap();
    assert_eq!(split.segments.len(), 2);
    let raws: Vec<Vec<u8>> = paths.iter().map(|p| std::fs::read(p).unwrap()).collect();
    let want: Vec<u8> = split
        .segments
        .iter()
        .flat_map(|s| raws[s.shard as usize][s.start as usize..(s.start + s.len) as usize].to_vec())
        .collect();
    let ring = Ring::new(m.experts.len(), Streamer::slot_bytes(&m));
    let s = Streamer::open(&m.shard_paths(), cfg(m.experts.len(), 2)).unwrap();
    s.rotate(&ring, &m, &Schedule::layer_order(&m, &HashSet::new()))
        .unwrap();
    let l = ring.lease(1, 0).unwrap().unwrap();
    assert_eq!(l.segments().count(), 2);
    assert_eq!(l.to_vec(), want);
}

fn cfg(slots: usize, workers: usize) -> StreamConfig {
    StreamConfig {
        slots,
        workers,
        direct: false,
        publish: true,
        lease_wait: Duration::from_secs(5),
    }
}

#[test]
fn manifest_round_trip_and_contiguous_spans() {
    let d = tmpdir("manifest");
    write_model(&d);
    let m = Manifest::build(&d).unwrap();
    assert_eq!((m.num_layers, m.experts_per_layer), (LAYERS, EXPERTS));
    assert_eq!(m.experts.len(), (LAYERS * EXPERTS) as usize);
    for e in &m.experts {
        assert_eq!(e.tensors.len(), 6);
        assert_eq!(e.tensors.iter().map(|t| t.len).sum::<u64>(), e.len);
    }
    let p = d.join("manifest.json");
    m.save(&p).unwrap();
    assert_eq!(Manifest::load(&p).unwrap(), m);
}

#[test]
fn rotation_publishes_exact_bytes_in_layer_order() {
    let d = tmpdir("rotate");
    let shard = write_model(&d);
    let raw = std::fs::read(&shard).unwrap();
    let m = Manifest::build(&d).unwrap();
    // As many slots as experts so the whole rotation stays resident to check.
    let n = m.experts.len();
    let ring = Ring::new(n, Streamer::slot_bytes(&m));
    let s = Streamer::open(&m.shard_paths(), cfg(n, 3)).unwrap();
    let sched = Schedule::layer_order(&m, &HashSet::new());
    let st = s.rotate(&ring, &m, &sched).unwrap();
    assert_eq!(st.experts as usize, n);
    let mut last = 0;
    for e in &m.experts {
        let l = ring.lease(e.layer, e.expert).unwrap().expect("published");
        assert_eq!(
            l.to_vec(),
            &raw[e.segments[0].start as usize..(e.segments[0].start + e.len) as usize]
        );
        last = last.max(l.tag().version);
    }
    assert_eq!(last as usize, n);
}

#[test]
fn hot_experts_are_skipped() {
    let d = tmpdir("hot");
    write_model(&d);
    let m = Manifest::build(&d).unwrap();
    let hot: HashSet<_> = [(0u16, 1u16), (1, 2)].into_iter().collect();
    let sched = Schedule::layer_order(&m, &hot);
    assert_eq!(sched.items.len(), m.experts.len() - 2);
    let ring = Ring::new(8, Streamer::slot_bytes(&m));
    Streamer::open(&m.shard_paths(), cfg(8, 2))
        .unwrap()
        .rotate(&ring, &m, &sched)
        .unwrap();
    assert!(ring.lease(0, 1).unwrap().is_none());
    assert!(ring.lease(1, 2).unwrap().is_none());
    assert!(ring.lease(1, 1).unwrap().is_some());
}

#[test]
fn small_ring_rotates_with_slot_reuse() {
    let d = tmpdir("reuse");
    write_model(&d);
    let m = Manifest::build(&d).unwrap();
    let ring = Ring::new(2, Streamer::slot_bytes(&m));
    let s = Streamer::open(&m.shard_paths(), cfg(2, 2)).unwrap();
    let sched = Schedule::layer_order(&m, &HashSet::new());
    for _ in 0..3 {
        s.rotate(&ring, &m, &sched).unwrap();
    }
    // Last two schedule items are what remains resident.
    assert!(ring.lease(1, 2).unwrap().is_some());
    assert!(ring.lease(1, 1).unwrap().is_some());
    assert!(ring.lease(0, 0).unwrap().is_none());
}

#[test]
fn corrupted_bytes_fail_closed() {
    let d = tmpdir("corrupt");
    let shard = write_model(&d);
    let m = Manifest::build(&d).unwrap();
    let victim = &m.experts[4];
    std::fs::OpenOptions::new()
        .write(true)
        .open(&shard)
        .unwrap()
        .write_all_at(&[0xEE], victim.segments[0].start + 10)
        .unwrap();
    let ring = Ring::new(6, Streamer::slot_bytes(&m));
    let s = Streamer::open(&m.shard_paths(), cfg(6, 2)).unwrap();
    let err = s
        .rotate(&ring, &m, &Schedule::layer_order(&m, &HashSet::new()))
        .unwrap_err();
    assert!(
        matches!(err, HeartbeatError::DigestMismatch { expert } if expert == (victim.layer, victim.expert)),
        "{err}"
    );
    assert!(ring.is_failed());
    assert!(matches!(ring.lease(0, 0), Err(HeartbeatError::RingFailed)));
}

#[test]
fn truncated_file_fails_closed() {
    let d = tmpdir("trunc");
    let shard = write_model(&d);
    let m = Manifest::build(&d).unwrap();
    let last = m.experts.last().unwrap();
    std::fs::OpenOptions::new()
        .write(true)
        .open(&shard)
        .unwrap()
        .set_len(last.segments[0].start + 7)
        .unwrap();
    let ring = Ring::new(2, Streamer::slot_bytes(&m));
    let s = Streamer::open(&m.shard_paths(), cfg(2, 1)).unwrap();
    let err = s
        .rotate(&ring, &m, &Schedule::layer_order(&m, &HashSet::new()))
        .unwrap_err();
    assert!(matches!(err, HeartbeatError::ShortRead { .. }), "{err}");
    assert!(ring.is_failed());
}

#[test]
fn reader_lease_holds_back_the_writer() {
    let d = tmpdir("leasehold");
    write_model(&d);
    let m = Manifest::build(&d).unwrap();
    let ring = Ring::new(1, Streamer::slot_bytes(&m));
    let mut c = cfg(1, 1);
    c.lease_wait = Duration::from_millis(30);
    let s = Streamer::open(&m.shard_paths(), c).unwrap();
    let one = Schedule { items: vec![0] };
    s.rotate(&ring, &m, &one).unwrap();
    let held = ring.lease(0, 0).unwrap().unwrap();
    let before = held.to_vec();
    // Writer must not overwrite the leased slot; it times out and fails closed.
    let err = s
        .rotate(&ring, &m, &Schedule { items: vec![1] })
        .unwrap_err();
    assert!(matches!(err, HeartbeatError::SlotBusy(0)), "{err}");
    assert_eq!(held.to_vec(), before);
}
