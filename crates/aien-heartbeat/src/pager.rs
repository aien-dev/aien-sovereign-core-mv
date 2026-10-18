//! Expert pager (Expert Ribbon P2, sc#630/#632): turns the router's exact top-k for one layer into pinned,
//! resident GPU slots. Hits pin; misses read the expert from the checkpoint (bounded, digest-verified
//! [`ReadAdapter`]), split it into the chip layout and load it into a slot pair through a [`SlotBackend`].
//! Misses of one layer are read and loaded in parallel (each on its own thread; the read adapter bounds how
//! many reads are in the kernel at once).
//!
//! Exactness: the experts returned are exactly the ones asked for, in the same order. A miss that cannot be
//! served (read error, digest mismatch, every slot pinned) is an error; nothing is substituted or skipped.

use std::collections::HashMap;
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use crate::layout::{ExpertLayout, ExpertParts, ExpertShape};
use crate::manifest::Manifest;
use crate::reader::{ReadAdapter, ReadError};
use crate::residency::{Acquire, ExpertId, Residency, ResidencyStats, SlotVersion};

/// The GPU side of a slot pair. The real one wraps two omega_gpu_fp8_slots pools (gate|up and down);
/// tests use a host fake. Must be callable from several threads at once (one load per slot at a time).
pub trait SlotBackend: Send + Sync {
    fn slots(&self) -> usize;
    /// Load `parts` into slot pair `slot`, returning the published versions. Refused (error) if the slot is
    /// still read by a launch: the residency manager never asks for that, so a refusal is a bug, fail closed.
    fn load(&self, slot: u32, id: ExpertId, parts: &ExpertParts<'_>)
        -> Result<SlotVersion, String>;
}

#[derive(Debug)]
pub enum PagerError {
    Read {
        id: ExpertId,
        err: ReadError,
    },
    Layout(String),
    Backend {
        id: ExpertId,
        err: String,
    },
    /// Every slot is pinned or being fetched: the capacity is too small for this request.
    Exhausted {
        id: ExpertId,
        capacity: usize,
    },
    UnknownExpert(ExpertId),
}

impl std::fmt::Display for PagerError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Read { id, err } => write!(f, "expert {id:?}: read failed: {err}"),
            Self::Layout(s) => write!(f, "expert layout: {s}"),
            Self::Backend { id, err } => write!(f, "expert {id:?}: slot load failed: {err}"),
            Self::Exhausted { id, capacity } => write!(
                f,
                "expert {id:?}: all {capacity} slots are pinned or loading (capacity too small for one request)"
            ),
            Self::UnknownExpert(id) => write!(f, "expert {id:?} is not in the manifest"),
        }
    }
}
impl std::error::Error for PagerError {}

#[derive(Debug, Clone, Default)]
pub struct PagerStats {
    pub requests: u64,
    pub experts_hit: u64,
    pub experts_missed: u64,
    pub bytes_loaded: u64,
    /// Wall time inside `ensure` spent on misses (read + verify + slot load), summed per call.
    pub miss_wall_ns: u64,
    /// Summed over miss threads: waiting for the verified read, and loading the slot pair.
    pub read_wait_ns: u64,
    pub slot_load_ns: u64,
}

/// Slots pinned for one layer's launch, in the order the router gave the experts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pinned {
    pub slots: Vec<u32>,
    pub versions: Vec<SlotVersion>,
}

pub struct ExpertPager {
    manifest: Arc<Manifest>,
    layouts: Vec<ExpertLayout>,
    index_of: HashMap<ExpertId, usize>,
    reader: ReadAdapter,
    backend: Arc<dyn SlotBackend>,
    res: Mutex<Residency>,
    fetch_done: Condvar,
    stats: Mutex<PagerStats>,
    read_deadline: Duration,
}

impl ExpertPager {
    /// Validates every expert's layout against `shape` up front (a format mismatch fails here, not mid-run).
    pub fn new(
        manifest: Arc<Manifest>,
        shape: ExpertShape,
        reader: ReadAdapter,
        backend: Arc<dyn SlotBackend>,
        read_deadline: Duration,
    ) -> Result<Self, PagerError> {
        let layouts = manifest
            .experts
            .iter()
            .map(|e| ExpertLayout::for_entry(e, shape))
            .collect::<Result<Vec<_>, _>>()
            .map_err(|e| PagerError::Layout(e.to_string()))?;
        let index_of = manifest
            .experts
            .iter()
            .enumerate()
            .map(|(i, e)| ((e.layer, e.expert), i))
            .collect();
        let slots = backend.slots();
        Ok(Self {
            manifest,
            layouts,
            index_of,
            reader,
            backend,
            res: Mutex::new(Residency::new(slots)),
            fetch_done: Condvar::new(),
            stats: Mutex::new(PagerStats::default()),
            read_deadline,
        })
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }
    pub fn capacity(&self) -> usize {
        self.res.lock().unwrap().capacity()
    }
    pub fn resident(&self) -> usize {
        self.res.lock().unwrap().resident()
    }
    pub fn stats(&self) -> PagerStats {
        self.stats.lock().unwrap().clone()
    }
    pub fn residency_stats(&self) -> ResidencyStats {
        self.res.lock().unwrap().stats()
    }
    pub fn read_stats(&self) -> crate::reader::ReadStats {
        self.reader.stats()
    }
    pub fn pinned(&self) -> u32 {
        self.res.lock().unwrap().pinned()
    }

    /// Pins exactly `experts` of `layer` (router order). On error nothing stays pinned.
    pub fn ensure(&self, layer: u16, experts: &[u16]) -> Result<Pinned, PagerError> {
        let mut out: Vec<Option<(u32, SlotVersion)>> = vec![None; experts.len()];
        let mut fetch: Vec<(usize, u32, ExpertId)> = Vec::new(); // (position, slot, id)
        {
            let mut r = self.res.lock().unwrap();
            for (pos, &ex) in experts.iter().enumerate() {
                let id = (layer, ex);
                if !self.index_of.contains_key(&id) {
                    self.undo(&mut r, &out, &fetch);
                    return Err(PagerError::UnknownExpert(id));
                }
                loop {
                    match r.acquire(id) {
                        Acquire::Hit { slot, ver } => {
                            out[pos] = Some((slot, ver));
                            break;
                        }
                        Acquire::Fetch { slot, .. } => {
                            fetch.push((pos, slot, id));
                            break;
                        }
                        Acquire::Pending => {
                            r = self.fetch_done.wait(r).unwrap();
                        }
                        Acquire::Exhausted => {
                            let capacity = r.capacity();
                            self.undo(&mut r, &out, &fetch);
                            return Err(PagerError::Exhausted { id, capacity });
                        }
                    }
                }
            }
        }
        let hits = (experts.len() - fetch.len()) as u64;
        let t0 = Instant::now();
        let results: Vec<Result<(SlotVersion, u64, u64, u64), PagerError>> = if fetch.is_empty() {
            Vec::new()
        } else {
            std::thread::scope(|sc| {
                let hs: Vec<_> = fetch
                    .iter()
                    .map(|&(_, slot, id)| sc.spawn(move || self.fetch_one(slot, id)))
                    .collect();
                hs.into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(|_| {
                            Err(PagerError::Layout("miss thread panicked".into()))
                        })
                    })
                    .collect()
            })
        };
        let miss_wall = t0.elapsed().as_nanos() as u64;
        let mut first_err = None;
        {
            let mut r = self.res.lock().unwrap();
            for (&(pos, slot, _), res) in fetch.iter().zip(&results) {
                match res {
                    Ok((ver, ..)) => {
                        r.fetched(slot, *ver);
                        out[pos] = Some((slot, *ver));
                    }
                    Err(_) => r.fetch_failed(slot),
                }
            }
            for res in &results {
                if let Err(e) = res {
                    first_err.get_or_insert_with(|| format!("{e}"));
                }
            }
            if first_err.is_some() {
                for (slot, _) in out.iter().flatten() {
                    r.release(*slot);
                }
            }
        }
        self.fetch_done.notify_all();
        let mut s = self.stats.lock().unwrap();
        s.requests += 1;
        s.experts_hit += hits;
        s.experts_missed += fetch.len() as u64;
        s.miss_wall_ns += miss_wall;
        for (_, rw, sl, b) in results.iter().flatten() {
            s.read_wait_ns += rw;
            s.slot_load_ns += sl;
            s.bytes_loaded += b;
        }
        drop(s);
        if let Some(e) = results.into_iter().find_map(|r| r.err()) {
            return Err(e);
        }
        let (slots, versions) = out
            .into_iter()
            .map(|o| o.expect("every position filled"))
            .unzip();
        Ok(Pinned { slots, versions })
    }

    /// Drops the pins of a finished launch.
    pub fn release(&self, p: &Pinned) {
        let mut r = self.res.lock().unwrap();
        for &s in &p.slots {
            r.release(s);
        }
    }

    fn undo(
        &self,
        r: &mut Residency,
        out: &[Option<(u32, SlotVersion)>],
        fetch: &[(usize, u32, ExpertId)],
    ) {
        for (slot, _) in out.iter().flatten() {
            r.release(*slot);
        }
        for &(_, slot, _) in fetch {
            r.fetch_failed(slot);
        }
    }

    /// Read + verify + split + slot load. Returns (version, read wait ns, slot load ns, bytes).
    fn fetch_one(
        &self,
        slot: u32,
        id: ExpertId,
    ) -> Result<(SlotVersion, u64, u64, u64), PagerError> {
        let idx = self.index_of[&id];
        let t0 = Instant::now();
        let v = self
            .reader
            .submit_wait(idx, self.read_deadline)
            .and_then(|t| t.wait(self.read_deadline))
            .map_err(|err| PagerError::Read { id, err })?;
        let rw = t0.elapsed().as_nanos() as u64;
        let t1 = Instant::now();
        let payload = v.payload();
        let parts = self.layouts[idx]
            .parts(&payload)
            .map_err(|e| PagerError::Layout(e.to_string()))?;
        let bytes = (parts.gate_up_w.len()
            + parts.down_w.len()
            + 2 * (parts.gate_up_s.len() + parts.down_s.len())) as u64;
        let ver = self
            .backend
            .load(slot, id, &parts)
            .map_err(|err| PagerError::Backend { id, err })?;
        Ok((ver, rw, t1.elapsed().as_nanos() as u64, bytes))
    }
}
