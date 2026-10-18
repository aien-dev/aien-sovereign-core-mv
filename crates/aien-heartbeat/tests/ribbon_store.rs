//! Expert Ribbon P1/P2 host tests (sc#629, sc#630): FP8 expert layout, bounded verified read adapter with
//! injected faults, residency through the pager with a fake GPU backend. Synthetic checkpoint with the real
//! Qwen3 FP8 names, dtypes and shapes (hidden 256, moe intermediate 128), F8 tensors first then the BF16
//! scales, as safetensors groups them, so every expert is two segments.

use std::collections::{BTreeMap, HashMap};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use aien_heartbeat::{
    ExpertId, ExpertLayout, ExpertPager, ExpertParts, ExpertShape, FakeShards, Fault, FileShards,
    Manifest, PagerError, ReadAdapter, ReadConfig, ReadError, SlotBackend, SlotVersion,
};

const L: u16 = 3;
const E: u16 = 4;
const SHAPE: ExpertShape = ExpertShape {
    hidden: 256,
    inter: 128,
};

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aien-ribbon-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn byte(l: u16, e: u16, p: usize, k: usize) -> u8 {
    ((l as usize * 131 + e as usize * 17 + p * 7 + k * 13) % 251) as u8
}

/// (name, dtype, shape, bytes) for one expert's six tensors.
fn expert_tensors(l: u16, e: u16) -> Vec<(String, &'static str, Vec<u64>, Vec<u8>)> {
    let (h, i) = (SHAPE.hidden, SHAPE.inter);
    let mut v = Vec::new();
    for (p, (proj, shape)) in [
        ("gate_proj", [i, h]),
        ("up_proj", [i, h]),
        ("down_proj", [h, i]),
    ]
    .into_iter()
    .enumerate()
    {
        let n = (shape[0] * shape[1]) as usize;
        v.push((
            format!("model.layers.{l}.mlp.experts.{e}.{proj}.weight"),
            "F8_E4M3",
            shape.to_vec(),
            (0..n).map(|k| byte(l, e, p, k)).collect(),
        ));
        let sshape = [shape[0] / 128, shape[1] / 128];
        let sn = (sshape[0] * sshape[1]) as usize;
        v.push((
            format!("model.layers.{l}.mlp.experts.{e}.{proj}.weight_scale_inv"),
            "BF16",
            sshape.to_vec(),
            (0..sn)
                .flat_map(|k| (0x3C00u16 + (l * 8 + e) + k as u16 + p as u16).to_le_bytes())
                .collect(),
        ));
    }
    v
}

/// One shard: all F8 tensors sorted by name, then all BF16 tensors sorted by name (safetensors' grouping).
fn write_model(dir: &Path) -> Vec<u8> {
    let mut all: Vec<(String, &str, Vec<u64>, Vec<u8>)> = Vec::new();
    for l in 0..L {
        for e in 0..E {
            all.extend(expert_tensors(l, e));
        }
    }
    all.push((
        "model.norm.weight".into(),
        "BF16",
        vec![256],
        vec![1u8; 512],
    ));
    all.sort_by(|a, b| (a.1 != "F8_E4M3", &a.0).cmp(&(b.1 != "F8_E4M3", &b.0)));
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    let mut weight_map = BTreeMap::new();
    for (name, dtype, shape, bytes) in &all {
        let a = data.len();
        data.extend(bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [a, data.len()]}),
        );
        weight_map.insert(name.clone(), "model-00001-of-00001.safetensors".to_string());
    }
    let h = serde_json::to_vec(&header).unwrap();
    let mut file = (h.len() as u64).to_le_bytes().to_vec();
    file.extend(&h);
    file.extend(&data);
    std::fs::write(dir.join("model-00001-of-00001.safetensors"), &file).unwrap();
    std::fs::write(
        dir.join("model.safetensors.index.json"),
        serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": weight_map})).unwrap(),
    )
    .unwrap();
    file
}

/// What the resident loader uploads for one expert, built straight from the tensors (the oracle).
fn oracle(l: u16, e: u16) -> (Vec<u8>, Vec<u16>, Vec<u8>, Vec<u16>) {
    let t = expert_tensors(l, e);
    let get = |suffix: &str| t.iter().find(|x| x.0.ends_with(suffix)).unwrap().3.clone();
    let words = |b: Vec<u8>| -> Vec<u16> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .collect()
    };
    let mut gu = get("gate_proj.weight");
    gu.extend(get("up_proj.weight"));
    let mut gus = words(get("gate_proj.weight_scale_inv"));
    gus.extend(words(get("up_proj.weight_scale_inv")));
    (
        gu,
        gus,
        get("down_proj.weight"),
        words(get("down_proj.weight_scale_inv")),
    )
}

fn setup(tag: &str) -> (PathBuf, Arc<Manifest>, Vec<u8>) {
    let d = tmpdir(tag);
    let file = write_model(&d);
    let m = Arc::new(Manifest::build(&d).unwrap());
    (d, m, file)
}

fn fake(file: &[u8], latency: Duration) -> Arc<FakeShards> {
    Arc::new(FakeShards::new(vec![file.to_vec()], latency))
}

fn idx(m: &Manifest, id: ExpertId) -> usize {
    m.experts
        .iter()
        .position(|e| (e.layer, e.expert) == id)
        .unwrap()
}

// ---------------- sc#629: layout, index, verified reads ----------------

#[test]
fn manifest_has_a_content_model_digest_and_two_segments_per_expert() {
    let (d, m, _) = setup("digest");
    assert_eq!(m.experts.len(), (L * E) as usize);
    assert!(m.experts.iter().all(|e| e.segments.len() == 2));
    assert_eq!(m.model_digest.len(), 64);
    // Same bytes -> same digest; one weight byte changed -> different model digest.
    assert_eq!(Manifest::build(&d).unwrap().model_digest, m.model_digest);
    let p = d.join("model-00001-of-00001.safetensors");
    let mut f = std::fs::read(&p).unwrap();
    let off = m.experts[5].segments[0].start as usize + 3;
    f[off] ^= 1;
    std::fs::write(&p, &f).unwrap();
    let m2 = Manifest::build(&d).unwrap();
    assert_ne!(m2.model_digest, m.model_digest);
    assert_ne!(m2.experts[5].digest, m.experts[5].digest);
}

#[test]
fn every_expert_round_trips_byte_for_byte_through_a_verified_random_read() {
    let (d, m, _) = setup("roundtrip");
    let dev = Arc::new(FileShards::open(&m.shard_paths(), false).unwrap());
    let r = ReadAdapter::new(
        m.clone(),
        dev,
        ReadConfig {
            workers: 3,
            queue_depth: 4,
        },
    );
    // Random order, not layer order: (2,1), (0,3), (1,0), ...
    let order: Vec<ExpertId> = (0..L * E)
        .map(|k| ((k * 7 % (L * E)) / E, (k * 7 % (L * E)) % E))
        .collect();
    for id in order {
        let i = idx(&m, id);
        let v = r.submit(i).unwrap().wait(Duration::from_secs(5)).unwrap();
        assert_eq!((v.layer, v.expert), id);
        let layout = ExpertLayout::for_entry(&m.experts[i], SHAPE).unwrap();
        let parts = layout.parts(&v.payload()).unwrap();
        let (gu, gus, dn, dns) = oracle(id.0, id.1);
        assert_eq!(&*parts.gate_up_w, &gu[..], "{id:?} gate|up weights");
        assert!(
            matches!(parts.gate_up_w, std::borrow::Cow::Borrowed(_)),
            "gate and up adjacent: no copy"
        );
        assert_eq!(parts.gate_up_s, gus);
        assert_eq!(&*parts.down_w, &dn[..]);
        assert_eq!(parts.down_s, dns);
    }
    assert_eq!(r.stats().reads_ok, (L * E) as u64);
    let _ = std::fs::remove_dir_all(d);
}

#[test]
fn layout_refuses_wrong_shape_and_missing_tensor() {
    let (_, m, _) = setup("layout");
    let e = &m.experts[0];
    let bad = ExpertShape {
        hidden: 256,
        inter: 256,
    };
    assert!(ExpertLayout::for_entry(e, bad).is_err());
    let mut e2 = e.clone();
    e2.tensors.retain(|t| !t.name.ends_with("up_proj.weight"));
    let err = ExpertLayout::for_entry(&e2, SHAPE).unwrap_err().to_string();
    assert!(err.contains("5 tensors"), "{err}");
    let mut e3 = e.clone();
    e3.tensors[0].dtype = "BF16".into();
    assert!(ExpertLayout::for_entry(&e3, SHAPE).is_err());
}

fn adapter(m: &Arc<Manifest>, dev: Arc<FakeShards>, depth: usize) -> ReadAdapter {
    ReadAdapter::new(
        m.clone(),
        dev,
        ReadConfig {
            workers: 2,
            queue_depth: depth,
        },
    )
}

#[test]
fn eio_short_read_eof_and_corruption_return_explicit_errors_and_free_the_buffer() {
    let (_, m, file) = setup("faults");
    let dev = fake(&file, Duration::ZERO);
    let r = adapter(&m, dev.clone(), 1);
    let w = Duration::from_secs(5);
    dev.fault_at(0, Fault::Eio);
    assert!(matches!(
        r.submit(0).unwrap().wait(w),
        Err(ReadError::Io(_))
    ));
    // Short read then EOF on the same segment.
    let c = dev.calls();
    dev.fault_at(c, Fault::Short(100));
    assert!(matches!(
        r.submit(1).unwrap().wait(w),
        Err(ReadError::ShortRead { .. })
    ));
    let c = dev.calls();
    dev.fault_at(c + 1, Fault::Flip); // second segment (scales) corrupted
    assert!(matches!(
        r.submit(2).unwrap().wait(w),
        Err(ReadError::DigestMismatch { expert: (0, 2) })
    ));
    // Depth 1: the single staging buffer came back after every failure.
    let ok = r.submit(3).unwrap().wait(w).unwrap();
    assert_eq!((ok.layer, ok.expert), (0, 3));
    assert_eq!(r.stats().errors, 3);
}

#[test]
fn bounded_queue_refuses_when_every_staging_buffer_is_in_use() {
    let (_, m, file) = setup("bounded");
    let r = adapter(&m, fake(&file, Duration::ZERO), 2);
    let w = Duration::from_secs(5);
    let a = r.submit(0).unwrap().wait(w).unwrap();
    let b = r.submit(1).unwrap().wait(w).unwrap();
    assert_eq!(r.submit(2).err(), Some(ReadError::QueueFull));
    drop(a);
    let c = r.submit(2).unwrap().wait(w).unwrap();
    assert_eq!(c.expert, 2);
    drop((b, c));
    assert!(r.stats().queue_full >= 1);
}

#[test]
fn stalled_read_hits_the_deadline_and_cancel_skips_queued_work() {
    let (_, m, file) = setup("stall");
    let dev = fake(&file, Duration::ZERO);
    let r = ReadAdapter::new(
        m.clone(),
        dev.clone(),
        ReadConfig {
            workers: 1,
            queue_depth: 3,
        },
    );
    let (tx, rx) = mpsc::channel();
    dev.fault_at(0, Fault::Stall(rx));
    let stalled = r.submit(0).unwrap();
    let queued = r.submit(1).unwrap(); // behind the stalled one (one worker)
    queued.cancel();
    assert_eq!(
        stalled.wait(Duration::from_millis(50)).err(),
        Some(ReadError::Deadline)
    );
    tx.send(()).unwrap(); // the NVMe "recovers"; its result is discarded
    assert_eq!(
        queued.wait(Duration::from_secs(5)).err(),
        Some(ReadError::Cancelled)
    );
    // Every buffer returns: three more reads fit.
    let w = Duration::from_secs(5);
    let held: Vec<_> = (2..5)
        .map(|i| r.submit_wait(i, w).unwrap().wait(w).unwrap())
        .collect();
    assert_eq!(held.len(), 3);
}

#[test]
fn shutdown_with_reads_in_flight_is_deterministic() {
    let (_, m, file) = setup("shutdown");
    let r = adapter(&m, fake(&file, Duration::from_millis(5)), 4);
    let ts: Vec<_> = (0..4).map(|i| r.submit(i).unwrap()).collect();
    drop(r); // joins the workers after the queue drains
    for t in ts {
        // finished before shutdown, or the reply channel closed: never a hang, never a partial result
        match t.wait(Duration::from_secs(5)) {
            Ok(v) => assert!(!v.payload().is_empty()),
            Err(e) => assert_eq!(e, ReadError::Shutdown),
        }
    }
}

// ---------------- sc#630: pager + residency with a fake GPU backend ----------------

/// (expert, gate|up weights, gate|up scales, down weights, down scales) held by one fake slot pair.
type SlotBytes = (ExpertId, Vec<u8>, Vec<u16>, Vec<u8>, Vec<u16>);

/// Host stand-in for the two omega slot pools: keeps the bytes per slot and versions like the chip API.
struct FakeBackend {
    slots: usize,
    data: Mutex<HashMap<u32, SlotBytes>>,
    next: AtomicU64,
    fail_on: Mutex<Option<ExpertId>>,
    loads: AtomicU64,
}

impl FakeBackend {
    fn new(slots: usize) -> Arc<Self> {
        Arc::new(Self {
            slots,
            data: Mutex::new(HashMap::new()),
            next: AtomicU64::new(1),
            fail_on: Mutex::new(None),
            loads: AtomicU64::new(0),
        })
    }
}

impl SlotBackend for FakeBackend {
    fn slots(&self) -> usize {
        self.slots
    }
    fn load(&self, slot: u32, id: ExpertId, p: &ExpertParts<'_>) -> Result<SlotVersion, String> {
        if *self.fail_on.lock().unwrap() == Some(id) {
            return Err("injected GPU copy failure".into());
        }
        self.loads.fetch_add(1, Ordering::Relaxed);
        self.data.lock().unwrap().insert(
            slot,
            (
                id,
                p.gate_up_w.to_vec(),
                p.gate_up_s.clone(),
                p.down_w.to_vec(),
                p.down_s.clone(),
            ),
        );
        let v = self.next.fetch_add(2, Ordering::Relaxed);
        Ok(SlotVersion {
            gate_up: v,
            down: v + 1,
        })
    }
}

fn pager(m: &Arc<Manifest>, file: &[u8], slots: usize) -> (ExpertPager, Arc<FakeBackend>) {
    let be = FakeBackend::new(slots);
    let r = ReadAdapter::new(
        m.clone(),
        fake(file, Duration::ZERO),
        ReadConfig {
            workers: 4,
            queue_depth: 8,
        },
    );
    let p = ExpertPager::new(m.clone(), SHAPE, r, be.clone(), Duration::from_secs(5)).unwrap();
    (p, be)
}

#[test]
fn pager_pins_exactly_the_routed_experts_and_slots_hold_their_bytes() {
    let (_, m, file) = setup("pager");
    let (p, be) = pager(&m, &file, 6);
    let want = [3u16, 0, 2];
    let pin = p.ensure(1, &want).unwrap();
    assert_eq!(pin.slots.len(), 3);
    {
        let d = be.data.lock().unwrap();
        for (k, &s) in pin.slots.iter().enumerate() {
            let (id, gu, gus, dn, dns) = &d[&s];
            assert_eq!(*id, (1, want[k]), "router order kept");
            assert_eq!(
                (gu, gus, dn, dns),
                (
                    &oracle(1, want[k]).0,
                    &oracle(1, want[k]).1,
                    &oracle(1, want[k]).2,
                    &oracle(1, want[k]).3
                )
            );
        }
    }
    assert_eq!(p.pinned(), 3);
    p.release(&pin);
    assert_eq!(p.pinned(), 0);
    // Same request again: all hits, no new loads, same slots and versions.
    let again = p.ensure(1, &want).unwrap();
    assert_eq!(again, pin);
    p.release(&again);
    assert_eq!(be.loads.load(Ordering::Relaxed), 3);
    let s = p.stats();
    assert_eq!((s.experts_missed, s.experts_hit), (3, 3));
}

#[test]
fn cache_pressure_evicts_lru_never_a_pinned_slot() {
    let (_, m, file) = setup("pressure");
    let (p, be) = pager(&m, &file, 4);
    let a = p.ensure(0, &[0, 1]).unwrap(); // held
    let b = p.ensure(1, &[0, 1]).unwrap();
    p.release(&b);
    // 2 free-able slots left (layer 1); layer 2 needs 2: evicts layer 1, not layer 0 (pinned)
    let c = p.ensure(2, &[2, 3]).unwrap();
    assert!(c.slots.iter().all(|s| !a.slots.contains(s)));
    assert_eq!(p.residency_stats().evictions, 2);
    // Everything pinned now: a third request cannot be served, and says so.
    match p.ensure(0, &[3]) {
        Err(PagerError::Exhausted { capacity: 4, .. }) => {}
        o => panic!("{o:?}"),
    }
    assert_eq!(p.pinned(), 4, "the failed request left nothing pinned");
    p.release(&a);
    p.release(&c);
    // Repeated layer-crossing churn keeps bytes right.
    for round in 0..20u16 {
        let l = round % L;
        let ex = [(round % E), ((round + 1) % E)];
        let pin = p.ensure(l, &ex).unwrap();
        let d = be.data.lock().unwrap();
        for (k, s) in pin.slots.iter().enumerate() {
            assert_eq!(d[s].0, (l, ex[k]));
            assert_eq!(d[s].1, oracle(l, ex[k]).0);
        }
        drop(d);
        p.release(&pin);
    }
    assert_eq!(p.pinned(), 0);
}

#[test]
fn a_failed_fetch_fails_closed_and_recovers() {
    let (_, m, file) = setup("failclosed");
    let (p, be) = pager(&m, &file, 4);
    *be.fail_on.lock().unwrap() = Some((2, 1));
    match p.ensure(2, &[0, 1, 2]) {
        Err(PagerError::Backend { id: (2, 1), .. }) => {}
        o => panic!("{o:?}"),
    }
    assert_eq!(
        p.pinned(),
        0,
        "the good experts of the failed request were released"
    );
    assert_eq!(
        p.resident(),
        2,
        "experts that did load stay resident and correct"
    );
    *be.fail_on.lock().unwrap() = None;
    let ok = p.ensure(2, &[0, 1, 2]).unwrap();
    p.release(&ok);
    assert_eq!(p.resident(), 3);
}

#[test]
fn corrupt_checkpoint_bytes_never_reach_a_slot() {
    let (_, m, file) = setup("corrupt");
    let be = FakeBackend::new(4);
    let dev = fake(&file, Duration::ZERO);
    dev.fault_at(0, Fault::Flip);
    let r = ReadAdapter::new(
        m.clone(),
        dev,
        ReadConfig {
            workers: 1,
            queue_depth: 2,
        },
    );
    let p = ExpertPager::new(m.clone(), SHAPE, r, be.clone(), Duration::from_secs(5)).unwrap();
    match p.ensure(0, &[1]) {
        Err(PagerError::Read {
            err: ReadError::DigestMismatch { .. },
            ..
        }) => {}
        o => panic!("{o:?}"),
    }
    assert_eq!(be.loads.load(Ordering::Relaxed), 0);
    assert_eq!(p.resident(), 0);
}

#[test]
fn simultaneous_requests_share_slots_and_wait_for_in_flight_fetches() {
    let (_, m, file) = setup("concurrent");
    let (p, be) = pager(&m, &file, 8);
    let p = Arc::new(p);
    let hs: Vec<_> = (0..4)
        .map(|_| {
            let p = p.clone();
            std::thread::spawn(move || {
                let pin = p.ensure(1, &[0, 1, 2, 3]).unwrap();
                std::thread::sleep(Duration::from_millis(5));
                p.release(&pin);
                pin
            })
        })
        .collect();
    let pins: Vec<_> = hs.into_iter().map(|h| h.join().unwrap()).collect();
    assert!(
        pins.windows(2).all(|w| w[0] == w[1]),
        "everyone got the same slots and versions"
    );
    assert_eq!(
        be.loads.load(Ordering::Relaxed),
        4,
        "each expert loaded once"
    );
    assert_eq!(p.pinned(), 0);
}
