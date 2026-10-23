//! Expert Ribbon in the OWN-30B engine (aienix#4, sc#632): FP8 experts paged from the NVMe into a
//! fixed pool of GPU slots instead of all 6144 experts resident.
//!
//! Per layer and token, [`PagedExperts::ffn`] takes the router's exact top-k, pins those experts in
//! slots through the [`ExpertPager`] (hits pin; misses are read from the checkpoint, BLAKE3-verified,
//! split into the chip layout and loaded into a free slot), runs one gate|up and one down grouped
//! launch over the pinned (slot, version) pairs (`omega_gpu_fp8_slots_gemv_grouped_f32`, the same
//! kernel as the resident tensor) and releases the pins. Nothing is substituted: a miss that cannot be
//! served fails the step.
//!
//! Exactness against the resident path: the activation row is the same bits. With the device router
//! the resident path reads the router's normed row on the chip (`grouped_dev`); here it is read back
//! (`read_post`: the router's row times 2^-64) and packed again (times 2^64), both exact powers of two.
//! The expert bytes and scales are the checkpoint's, as the resident loader uploads them.
//!
//! The router still writes its group tables from a directory; the paged path never reads them, so
//! [`directory_placeholder`] fills the directory with fixed non-zero words (the router refuses zero
//! words). A device-side slot table that would let the router address slots directly is P3, not here.

use aien_heartbeat::weight_flow::{self, FlowSettings, FlowTrace};
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aien_heartbeat::reader::ReadStats;
use aien_heartbeat::{
    ExpertId, ExpertPager, ExpertParts, ExpertShape, Manifest, PagerStats, ReadAdapter, ReadConfig,
    ResidencyStats, ShardDevice, SlotBackend, SlotVersion,
};

/// One pool of FP8 slots of a single projection shape `[n][k]` (gate|up or down).
pub trait SlotGemv: Send + Sync + std::fmt::Debug {
    fn slots(&self) -> usize;
    fn n(&self) -> usize;
    fn k(&self) -> usize;
    /// Copies one expert projection into `slot` and returns its new version. Refused while pinned.
    fn load(&self, slot: u32, key: u64, w: &[u8], scales: &[u16]) -> Result<u64, String>;
    /// One grouped launch over `(slots[g], versions[g])`; `x` is one shared row or one row per group.
    /// Returns the chip-reported kernel ns.
    fn grouped(
        &self,
        slots: &[u32],
        versions: &[u64],
        x: &[f32],
        y: &mut [f32],
    ) -> Result<u64, String>;
}

/// The chip pool (`omega_gpu_fp8_slots`, omega#533).
#[derive(Debug)]
pub struct OmegaSlotGemv(aien_omega_fp8::Fp8Slots);

impl OmegaSlotGemv {
    pub fn alloc(slots: usize, n: usize, k: usize) -> Result<Self, String> {
        aien_omega_fp8::Fp8Slots::alloc(slots, n, k)
            .map(Self)
            .map_err(|e| format!("fp8 slot pool {slots} x [{n}][{k}]: {e}"))
    }
    pub fn pool_stats(&self) -> aien_omega_fp8::SlotPoolStats {
        self.0.stats()
    }
    /// Device bytes of one slot (weights + f32 scales).
    pub fn slot_bytes(&self) -> usize {
        self.0.slot_bytes()
    }
}

impl SlotGemv for OmegaSlotGemv {
    fn slots(&self) -> usize {
        self.0.slots()
    }
    fn n(&self) -> usize {
        self.0.n()
    }
    fn k(&self) -> usize {
        self.0.k()
    }
    fn load(&self, slot: u32, key: u64, w: &[u8], scales: &[u16]) -> Result<u64, String> {
        self.0
            .load(slot as usize, key, w, scales)
            .map_err(|e| e.to_string())
    }
    fn grouped(
        &self,
        slots: &[u32],
        versions: &[u64],
        x: &[f32],
        y: &mut [f32],
    ) -> Result<u64, String> {
        self.0
            .grouped_f32(slots, versions, x, y)
            .map(|i| i.elapsed_ns)
            .map_err(|e| e.to_string())
    }
}

/// Content key of one projection of one expert in a slot (omega records it with the slot).
pub fn slot_key(id: ExpertId, down: bool) -> u64 {
    ((id.0 as u64) << 17) | ((id.1 as u64) << 1) | down as u64
}

/// The two pools as the pager's slot pair: slot `s` of gate|up and slot `s` of down hold one expert.
#[derive(Debug)]
struct SlotPair {
    gate_up: Arc<dyn SlotGemv>,
    down: Arc<dyn SlotGemv>,
}

impl SlotBackend for SlotPair {
    fn slots(&self) -> usize {
        self.gate_up.slots()
    }
    fn load(
        &self,
        slot: u32,
        id: ExpertId,
        parts: &ExpertParts<'_>,
    ) -> Result<SlotVersion, String> {
        let gate_up = self.gate_up.load(
            slot,
            slot_key(id, false),
            &parts.gate_up_w,
            &parts.gate_up_s,
        )?;
        let down = self
            .down
            .load(slot, slot_key(id, true), &parts.down_w, &parts.down_s)?;
        Ok(SlotVersion { gate_up, down })
    }
}

/// Where one paged FFN call spent its time.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FfnTiming {
    /// Wall time in `ensure` (hits are near zero; misses read, verify and load).
    pub page_ns: u64,
    pub chip_ns: u64,
    pub host_ns: u64,
    pub launches: u64,
}

/// The paged expert weights of every layer: one slot pool pair shared by all layers.
pub struct PagedExperts {
    pager: ExpertPager,
    flow: Mutex<Option<FlowSettings>>,
    last_flow: Mutex<Option<FlowTrace>>,
    gate_up: Arc<dyn SlotGemv>,
    down: Arc<dyn SlotGemv>,
    inter: usize,
    hidden: usize,
}

impl std::fmt::Debug for PagedExperts {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PagedExperts {{ slots: {}, resident: {}, gate_up: {:?}, down: {:?} }}",
            self.pager.capacity(),
            self.pager.resident(),
            self.gate_up,
            self.down
        )
    }
}

/// Fixed non-zero directory words for the router in paged mode (never dereferenced: see the module doc).
pub fn directory_placeholder(experts: usize) -> Vec<u64> {
    (0..2 * experts as u64)
        .map(|j| 0xDEAD_0000_0000_0000 | (j + 1))
        .collect()
}

impl PagedExperts {
    /// `manifest` must describe the same checkpoint the rest of the model is loaded from; every
    /// expert's layout is checked against `hidden` x `inter` here.
    pub fn new(
        manifest: Arc<Manifest>,
        hidden: usize,
        inter: usize,
        dev: Arc<dyn ShardDevice>,
        read: ReadConfig,
        gate_up: Arc<dyn SlotGemv>,
        down: Arc<dyn SlotGemv>,
        read_deadline: Duration,
    ) -> Result<Self, String> {
        if gate_up.slots() != down.slots() || gate_up.slots() == 0 {
            return Err(format!(
                "slot pools differ or are empty: gate|up {} slots, down {}",
                gate_up.slots(),
                down.slots()
            ));
        }
        if (gate_up.n(), gate_up.k()) != (2 * inter, hidden)
            || (down.n(), down.k()) != (hidden, inter)
        {
            return Err(format!(
                "slot shapes gate|up [{}][{}] down [{}][{}] do not fit hidden {hidden} inter {inter}",
                gate_up.n(),
                gate_up.k(),
                down.n(),
                down.k()
            ));
        }
        let reader = ReadAdapter::new(manifest.clone(), dev, read);
        let backend = Arc::new(SlotPair {
            gate_up: gate_up.clone(),
            down: down.clone(),
        });
        let shape = ExpertShape {
            hidden: hidden as u64,
            inter: inter as u64,
        };
        let pager = ExpertPager::new(manifest, shape, reader, backend, read_deadline)
            .map_err(|e| e.to_string())?;
        Ok(Self {
            pager,
            flow: Mutex::new(None),
            last_flow: Mutex::new(None),
            gate_up,
            down,
            inter,
            hidden,
        })
    }

    /// Opt-in only; default retains the original grouped FFN. No environment lookup.
    pub fn set_flow(&self, settings: FlowSettings) -> Result<(), String> {
        if !(1..=32).contains(&settings.max_inflight) {
            return Err("flow bound must be 1..32".into());
        }
        *self.flow.lock().map_err(|_| "flow state poisoned")? = Some(settings);
        Ok(())
    }
    pub fn take_flow_trace(&self) -> Option<FlowTrace> {
        self.last_flow.lock().ok()?.take()
    }

    pub fn capacity(&self) -> usize {
        self.pager.capacity()
    }
    pub fn resident(&self) -> usize {
        self.pager.resident()
    }
    pub fn pinned(&self) -> u32 {
        self.pager.pinned()
    }
    pub fn stats(&self) -> PagerStats {
        self.pager.stats()
    }
    pub fn residency_stats(&self) -> ResidencyStats {
        self.pager.residency_stats()
    }
    pub fn read_stats(&self) -> ReadStats {
        self.pager.read_stats()
    }
    pub fn manifest(&self) -> &Manifest {
        self.pager.manifest()
    }

    fn flow_ffn(
        &self,
        layer: u16,
        ids: &[u16],
        x: &[f32],
        gu_out: &mut [f32],
        act_all: &mut [f32],
        down_all: &mut [f32],
        settings: &mut FlowSettings,
    ) -> Result<FfnTiming, String> {
        // Validation precedes allocations and all outputs remain unchanged on any stage failure.
        if ids.len() > 32 {
            return Err("weight flow supports at most 32 selected experts".into());
        }
        *self.last_flow.lock().map_err(|_| "flow trace poisoned")? = None;
        let (i, h, k) = (self.inter, self.hidden, ids.len());
        let mut gu = vec![0.0; k * 2 * i];
        let mut act = vec![0.0; k * i];
        let mut down = vec![0.0; k * h];
        let mut timing = FfnTiming::default();
        let trace = weight_flow::run(
            &self.pager,
            layer,
            ids,
            settings,
            &AtomicBool::new(false),
            |g, pin| {
                let ver = pin.versions[0];
                timing.chip_ns += self.gate_up.grouped(
                    &pin.slots,
                    &[ver.gate_up],
                    x,
                    &mut gu[g * 2 * i..(g + 1) * 2 * i],
                )?;
                timing.launches += 1;
                let th = Instant::now();
                super::swiglu(
                    &gu[g * 2 * i..(g + 1) * 2 * i],
                    i,
                    &mut act[g * i..(g + 1) * i],
                );
                timing.host_ns += th.elapsed().as_nanos() as u64;
                timing.chip_ns += self.down.grouped(
                    &pin.slots,
                    &[ver.down],
                    &act[g * i..(g + 1) * i],
                    &mut down[g * h..(g + 1) * h],
                )?;
                timing.launches += 1;
                Ok(())
            },
        )?;
        timing.page_ns = trace.wait_ns;
        // Acquire the fallible trace lock before committing caller-visible results.
        let mut recorded = self.last_flow.lock().map_err(|_| "flow trace poisoned")?;
        gu_out[..gu.len()].copy_from_slice(&gu);
        act_all[..act.len()].copy_from_slice(&act);
        down_all[..down.len()].copy_from_slice(&down);
        *recorded = Some(trace);
        Ok(timing)
    }

    /// The expert FFN of one token in one layer: `gu_out = gate|up(x)`, SwiGLU into `act_all`,
    /// `down_all = down(act_all)`, one row per expert in router order. The combine stays with the caller.
    pub fn ffn(
        &self,
        layer: usize,
        experts: &[u32],
        x: &[f32],
        gu_out: &mut [f32],
        act_all: &mut [f32],
        down_all: &mut [f32],
    ) -> Result<FfnTiming, String> {
        let (i, h, k) = (self.inter, self.hidden, experts.len());
        if x.len() != h
            || gu_out.len() < k * 2 * i
            || act_all.len() < k * i
            || down_all.len() < k * h
        {
            return Err(format!(
                "paged ffn layer {layer}: buffer sizes do not fit {k} experts"
            ));
        }
        let ids: Vec<u16> = experts
            .iter()
            .map(|&e| u16::try_from(e).map_err(|_| format!("expert id {e}")))
            .collect::<Result<_, _>>()?;
        let layer16 = u16::try_from(layer).map_err(|_| format!("layer {layer}"))?;
        let mut flow = self.flow.lock().map_err(|_| "flow state poisoned")?;
        if let Some(settings) = flow.as_mut() {
            return self.flow_ffn(layer16, &ids, x, gu_out, act_all, down_all, settings);
        }
        drop(flow);
        let mut t = FfnTiming::default();
        let t0 = Instant::now();
        let pinned = self
            .pager
            .ensure(layer16, &ids)
            .map_err(|e| format!("layer {layer} expert pager: {e}"))?;
        t.page_ns = t0.elapsed().as_nanos() as u64;
        // The pins are dropped on every exit from here on.
        struct Release<'a>(&'a ExpertPager, &'a aien_heartbeat::Pinned);
        impl Drop for Release<'_> {
            fn drop(&mut self) {
                self.0.release(self.1);
            }
        }
        let _release = Release(&self.pager, &pinned);
        let gu_ver: Vec<u64> = pinned.versions.iter().map(|v| v.gate_up).collect();
        let dn_ver: Vec<u64> = pinned.versions.iter().map(|v| v.down).collect();
        t.chip_ns += self
            .gate_up
            .grouped(&pinned.slots, &gu_ver, x, &mut gu_out[..k * 2 * i])
            .map_err(|e| format!("layer {layer} gate_up slots: {e}"))?;
        t.launches += 1;
        let th = Instant::now();
        for g in 0..k {
            super::swiglu(
                &gu_out[g * 2 * i..(g + 1) * 2 * i],
                i,
                &mut act_all[g * i..(g + 1) * i],
            );
        }
        t.host_ns += th.elapsed().as_nanos() as u64;
        t.chip_ns += self
            .down
            .grouped(
                &pinned.slots,
                &dn_ver,
                &act_all[..k * i],
                &mut down_all[..k * h],
            )
            .map_err(|e| format!("layer {layer} down slots: {e}"))?;
        t.launches += 1;
        Ok(t)
    }
}

#[cfg(test)]
mod tests {
    //! Paged vs resident on the tiny in-memory model, with host fakes for the slot pools: the same
    //! logits bit for bit at every position, at the smallest capacity (top_k slots: every layer
    //! evicts) and with room for everything; a corrupted slot load changes them (the check has teeth);
    //! a capacity below top_k fails the step instead of computing something else.
    use super::super::test_support::*;
    use super::super::*;
    use super::*;
    use std::collections::BTreeMap;
    use std::sync::Mutex;

    /// Host slot pool: the resident host GEMV over `slots` expert-sized regions, with omega's
    /// version rule (a launch names the version it pinned; a reloaded slot refuses the old one).
    #[derive(Debug)]
    struct HostSlots {
        inner: Mutex<(HostFp8, Vec<u64>, u64)>,
        corrupt: bool,
    }

    impl HostSlots {
        fn new(slots: usize, n: usize, k: usize, corrupt: bool) -> Arc<Self> {
            let sc = (n / FP8_BLOCK) * (k / FP8_BLOCK);
            Arc::new(Self {
                inner: Mutex::new((
                    HostFp8 {
                        experts: slots,
                        n,
                        k,
                        w: vec![0; slots * n * k],
                        scales: vec![0; slots * sc],
                        dev: None,
                        gemm: None,
                    },
                    vec![0; slots],
                    0,
                )),
                corrupt,
            })
        }
    }

    impl SlotGemv for HostSlots {
        fn slots(&self) -> usize {
            self.inner.lock().unwrap().1.len()
        }
        fn n(&self) -> usize {
            self.inner.lock().unwrap().0.n
        }
        fn k(&self) -> usize {
            self.inner.lock().unwrap().0.k
        }
        fn load(&self, slot: u32, _key: u64, w: &[u8], scales: &[u16]) -> Result<u64, String> {
            let mut g = self.inner.lock().unwrap();
            let (n, k) = (g.0.n, g.0.k);
            let sc = (n / FP8_BLOCK) * (k / FP8_BLOCK);
            let s = slot as usize;
            assert_eq!((w.len(), scales.len()), (n * k, sc));
            g.0.w[s * n * k..(s + 1) * n * k].copy_from_slice(w);
            if self.corrupt {
                g.0.w[s * n * k] ^= 0x01;
            }
            g.0.scales[s * sc..(s + 1) * sc].copy_from_slice(scales);
            g.2 += 1;
            let v = g.2;
            g.1[s] = v;
            Ok(v)
        }
        fn grouped(
            &self,
            slots: &[u32],
            versions: &[u64],
            x: &[f32],
            y: &mut [f32],
        ) -> Result<u64, String> {
            let g = self.inner.lock().unwrap();
            for (&s, &v) in slots.iter().zip(versions) {
                if g.1[s as usize] != v {
                    return Err(format!(
                        "slot {s}: version {v} is stale (now {})",
                        g.1[s as usize]
                    ));
                }
            }
            g.0.grouped_f32(slots, x, y)
        }
    }

    /// The in-memory checkpoint as one safetensors shard + index (F8 tensors first, then BF16, each
    /// sorted by name, as safetensors groups them), so the manifest and reader see real files.
    fn write_checkpoint(src: &MemorySource, dir: &std::path::Path) {
        let mut all: Vec<_> = src.tensors.iter().collect();
        all.sort_by(|a, b| (a.1 .0 != "F8_E4M3", a.0).cmp(&(b.1 .0 != "F8_E4M3", b.0)));
        let shard = "model-00001-of-00001.safetensors";
        let (mut header, mut data, mut map) = (serde_json::Map::new(), Vec::new(), BTreeMap::new());
        for (name, (dtype, shape, bytes)) in all {
            let a = data.len();
            data.extend_from_slice(bytes);
            header.insert(
                name.clone(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [a, data.len()]}),
            );
            map.insert(name.clone(), shard.to_string());
        }
        let h = serde_json::to_vec(&header).unwrap();
        let mut file = (h.len() as u64).to_le_bytes().to_vec();
        file.extend(&h);
        file.extend(&data);
        std::fs::write(dir.join(shard), file).unwrap();
        std::fs::write(
            dir.join("model.safetensors.index.json"),
            serde_json::to_vec(&serde_json::json!({"metadata": {}, "weight_map": map})).unwrap(),
        )
        .unwrap();
    }

    fn tmpdir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "aien-ribbon-{tag}-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn paged_engine(
        c: &Qwen3MoeConfig,
        dir: &std::path::Path,
        slots: usize,
        fakes: bool,
        corrupt: bool,
    ) -> (Qwen3MoeOmegaEngine, Arc<PagedExperts>) {
        let m = Arc::new(Manifest::build(dir).unwrap());
        let dev = Arc::new(aien_heartbeat::FileShards::open(&m.shard_paths(), false).unwrap());
        let (h, i) = (c.hidden, c.moe_intermediate);
        let paged = Arc::new(
            PagedExperts::new(
                m,
                h,
                i,
                dev,
                ReadConfig {
                    workers: 2,
                    queue_depth: 4,
                },
                HostSlots::new(slots, 2 * i, h, corrupt),
                HostSlots::new(slots, h, i, false),
                Duration::from_secs(10),
            )
            .unwrap(),
        );
        let mut src = MemorySource::for_config(c);
        let up = RecordingUploader {
            device_fakes: fakes,
            ..Default::default()
        };
        let (w, _) = load_qwen3_moe_omega_paged_from(
            c,
            &mut src,
            &up,
            WeightFormat::Fp8All,
            DevicePaths::default(),
            paged.clone(),
            &mut |_| {},
        )
        .unwrap();
        assert!(
            !src.reads.iter().any(|n| n.contains(".mlp.experts.")),
            "a paged load must not read expert bytes"
        );
        let kv = HostKv::new(&KvGeometry::from(c), 32);
        (Qwen3MoeOmegaEngine::new(w, Box::new(kv)), paged)
    }

    fn resident_engine(c: &Qwen3MoeConfig, fakes: bool) -> Qwen3MoeOmegaEngine {
        let mut src = MemorySource::for_config(c);
        let up = RecordingUploader {
            device_fakes: fakes,
            ..Default::default()
        };
        let (w, _) = load_qwen3_moe_omega_from_paths(
            c,
            &mut src,
            &up,
            WeightFormat::Fp8All,
            DevicePaths::default(),
            &mut |_| {},
        )
        .unwrap();
        Qwen3MoeOmegaEngine::new(w, Box::new(HostKv::new(&KvGeometry::from(c), 32)))
    }

    /// Prompt then greedy decode; logits bits at every position.
    fn run(e: &mut Qwen3MoeOmegaEngine, steps: usize) -> Vec<Vec<u32>> {
        let mut out = Vec::new();
        let mut tok = 5u32;
        for p in 0..steps {
            let hidden = e.step(if p < 4 { [5, 9, 17, 2][p] } else { tok }).unwrap();
            let lg = e.logits(&hidden).unwrap();
            tok = lg
                .iter()
                .enumerate()
                .fold((0, f32::MIN), |b, (j, &v)| if v > b.1 { (j, v) } else { b })
                .0 as u32;
            out.push(lg.iter().map(|v| v.to_bits()).collect());
        }
        out
    }

    #[derive(Debug)]
    struct FailSecondDown {
        base: Arc<HostSlots>,
        calls: std::sync::atomic::AtomicUsize,
    }
    impl SlotGemv for FailSecondDown {
        fn slots(&self) -> usize {
            self.base.slots()
        }
        fn n(&self) -> usize {
            self.base.n()
        }
        fn k(&self) -> usize {
            self.base.k()
        }
        fn load(&self, slot: u32, key: u64, w: &[u8], s: &[u16]) -> Result<u64, String> {
            self.base.load(slot, key, w, s)
        }
        fn grouped(
            &self,
            slots: &[u32],
            versions: &[u64],
            x: &[f32],
            y: &mut [f32],
        ) -> Result<u64, String> {
            if self
                .calls
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                == 1
            {
                return Err("second down failed".into());
            }
            self.base.grouped(slots, versions, x, y)
        }
    }
    #[test]
    fn weight_flow_late_compute_failure_is_transactional() {
        let c = tiny_config();
        let dir = tmpdir("flow-transaction");
        write_checkpoint(&MemorySource::for_config(&c), &dir);
        let m = Arc::new(Manifest::build(&dir).unwrap());
        let dev = Arc::new(aien_heartbeat::FileShards::open(&m.shard_paths(), false).unwrap());
        let p = PagedExperts::new(
            m,
            c.hidden,
            c.moe_intermediate,
            dev,
            ReadConfig {
                workers: 2,
                queue_depth: 2,
            },
            HostSlots::new(2, 2 * c.moe_intermediate, c.hidden, false),
            Arc::new(FailSecondDown {
                base: HostSlots::new(2, c.hidden, c.moe_intermediate, false),
                calls: std::sync::atomic::AtomicUsize::new(0),
            }),
            Duration::from_secs(2),
        )
        .unwrap();
        p.set_flow(FlowSettings::fixed(2)).unwrap();
        let mut gu = vec![-0.0; 4 * c.moe_intermediate];
        let mut act = vec![-0.0; 2 * c.moe_intermediate];
        let mut dn = vec![-0.0; 2 * c.hidden];
        let err = p
            .ffn(
                0,
                &[0, 1],
                &vec![0.25; c.hidden],
                &mut gu,
                &mut act,
                &mut dn,
            )
            .unwrap_err();
        assert!(err.contains("second down failed"));
        assert!(gu
            .iter()
            .chain(&act)
            .chain(&dn)
            .all(|x| x.to_bits() == (-0.0f32).to_bits()));
        assert_eq!(p.pinned(), 0);
        assert!(p.take_flow_trace().is_none());
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn weight_flow_all_modes_match_resident_logits_and_record_real_execution() {
        let c = tiny_config();
        let dir = tmpdir("flow-model");
        write_checkpoint(&MemorySource::for_config(&c), &dir);
        let binary = std::env::var_os("AIEN_WEIGHT_FLOW_POLICY")
            .expect("build native/weight-flow policy and set AIEN_WEIGHT_FLOW_POLICY");
        for fakes in [false, true] {
            let want = run(&mut resident_engine(&c, fakes), 10);
            for cap in [1, 2, c.top_k, c.layers * c.experts] {
                for mode in 0..3 {
                    let (e, p) = paged_engine(&c, &dir, cap, fakes, false);
                    let mut e = e;
                    let cfg = match mode {
                        0 => FlowSettings::synchronous(),
                        1 => FlowSettings::fixed(2),
                        _ => FlowSettings::adaptive(8, std::path::Path::new(&binary)).unwrap(),
                    };
                    p.set_flow(cfg).unwrap();
                    assert_eq!(
                        run(&mut e, 10),
                        want,
                        "mode {mode} cap {cap} router {fakes}"
                    );
                    let trace = p
                        .take_flow_trace()
                        .expect("flow execution must be attached, not silently ignored");
                    assert_eq!(trace.experts, c.top_k);
                    assert_eq!(trace.events.len(), c.top_k);
                    assert!(trace.peak_inflight <= cap.min(8));
                    assert_eq!(p.pinned(), 0);
                }
            }
        }
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn paged_experts_give_the_resident_logits_bit_for_bit_at_any_capacity() {
        let c = tiny_config();
        let dir = tmpdir("parity");
        write_checkpoint(&MemorySource::for_config(&c), &dir);
        let all = c.layers * c.experts;
        for fakes in [false, true] {
            let want = run(&mut resident_engine(&c, fakes), 10);
            for slots in [c.top_k, c.top_k + 1, all] {
                let (mut e, paged) = paged_engine(&c, &dir, slots, fakes, false);
                let got = run(&mut e, 10);
                assert_eq!(got, want, "device router {fakes}, {slots} slots");
                let s = paged.stats();
                assert_eq!(s.requests, (10 * c.layers) as u64);
                assert_eq!(
                    s.experts_hit + s.experts_missed,
                    (10 * c.layers * c.top_k) as u64
                );
                assert!(
                    s.experts_missed >= (c.layers * c.top_k) as u64,
                    "the first token misses everything"
                );
                if slots == all {
                    // Room for everything: each expert is read at most once.
                    assert!(s.experts_missed <= all as u64, "{s:?}");
                } else {
                    assert!(
                        s.experts_missed > all as u64,
                        "smallest pools must evict: {s:?}"
                    );
                }
                assert_eq!(paged.pinned(), 0, "every pin released");
                assert!(e.last_timing.page_ns > 0 || s.experts_missed > 0);
            }
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn a_corrupted_slot_load_changes_the_logits_and_too_few_slots_fail_the_step() {
        let c = tiny_config();
        let dir = tmpdir("neg");
        write_checkpoint(&MemorySource::for_config(&c), &dir);
        let want = run(&mut resident_engine(&c, true), 4);
        let (mut bad, _) = paged_engine(&c, &dir, c.layers * c.experts, true, true);
        assert_ne!(
            run(&mut bad, 4),
            want,
            "negative control: one flipped weight byte per load"
        );

        let (mut small, paged) = paged_engine(&c, &dir, c.top_k - 1, true, false);
        let err = small.step(5).unwrap_err();
        assert!(err.contains("pinned or loading"), "{err}");
        assert_eq!(paged.pinned(), 0, "a failed step leaves nothing pinned");
        std::fs::remove_dir_all(&dir).ok();
    }
}

/// What [`open_omega`] needs: the checkpoint's expert manifest, the slot count and the read queue.
#[derive(Debug, Clone)]
pub struct RibbonOptions {
    /// Saved manifest (built from the checkpoint and saved here when missing or for another model).
    pub manifest: std::path::PathBuf,
    pub slots: usize,
    pub read: ReadConfig,
    /// O_DIRECT reads (the SSD figure); false reads through the page cache.
    pub direct: bool,
    pub read_deadline: Duration,
}

/// The chip slot pools for a checkpoint: gate|up `[2i][h]` and down `[h][i]`, `slots` each.
/// Every expert read is BLAKE3-checked against the manifest, so a stale manifest fails closed.
pub fn open_omega(
    dir: &std::path::Path,
    hidden: usize,
    inter: usize,
    o: &RibbonOptions,
    progress: &mut dyn FnMut(&str),
) -> Result<Arc<PagedExperts>, String> {
    let m = match Manifest::load(&o.manifest) {
        Ok(m) if m.model_dir == dir => m,
        _ => {
            let t = Instant::now();
            let m = Manifest::build(dir).map_err(|e| format!("expert manifest: {e}"))?;
            m.save(&o.manifest)
                .map_err(|e| format!("save manifest {}: {e}", o.manifest.display()))?;
            progress(&format!(
                "ribbon manifest built in {:.1} s ({} experts)",
                t.elapsed().as_secs_f64(),
                m.experts.len()
            ));
            m
        }
    };
    progress(&format!(
        "ribbon manifest {} experts, model digest {}",
        m.experts.len(),
        m.model_digest
    ));
    let m = Arc::new(m);
    let dev = Arc::new(
        aien_heartbeat::FileShards::open(&m.shard_paths(), o.direct)
            .map_err(|e| format!("open shards: {e}"))?,
    );
    let gate_up = Arc::new(OmegaSlotGemv::alloc(o.slots, 2 * inter, hidden)?);
    let down = Arc::new(OmegaSlotGemv::alloc(o.slots, hidden, inter)?);
    progress(&format!(
        "ribbon slots {} x {} B = {} B device",
        o.slots,
        gate_up.slot_bytes() + down.slot_bytes(),
        o.slots * (gate_up.slot_bytes() + down.slot_bytes())
    ));
    PagedExperts::new(
        m,
        hidden,
        inter,
        dev,
        o.read,
        gate_up,
        down,
        o.read_deadline,
    )
    .map(Arc::new)
}
