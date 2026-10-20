//! Expert cache on the Heartbeat ring: a fixed set of versioned slots (aien-heartbeat `Ring`) holding
//! Flash-Next routed experts. A slot is readable only after its whole expert was read from the NVMe and
//! its BLAKE3 digest matched the manifest (publish after digest; the ring fails closed on any mismatch).
//! Misses are filled on demand into the least recently used slot; `prefetch` fills a list in parallel
//! on the streamer's worker pool (one writer per slot), which is the heartbeat beat.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use aien_heartbeat::manifest::TensorRange;
use aien_heartbeat::{ExpertEntry, Manifest, Ring, StreamConfig, Streamer};

#[derive(Debug, Clone, Default, serde::Serialize)]
pub struct CacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub fill_bytes: u64,
    pub fill_s: f64,
    pub prefetched: u64,
}

pub struct ExpertCache {
    pub manifest: Manifest,
    index: HashMap<(u16, u16), usize>,
    ring: Ring,
    streamer: Streamer,
    /// expert -> slot, slot -> (expert, last use)
    slot_of: HashMap<(u16, u16), usize>,
    owner: Vec<Option<((u16, u16), u64)>>,
    tick: u64,
    pub stats: CacheStats,
}

impl ExpertCache {
    pub fn new(manifest: Manifest, slots: usize, workers: usize, direct: bool) -> Result<Self, String> {
        let slot_bytes = Streamer::slot_bytes(&manifest);
        let streamer = Streamer::open(
            &manifest.shard_paths(),
            StreamConfig {
                slots,
                workers: workers.min(slots).max(1),
                direct,
                publish: true,
                lease_wait: Duration::from_secs(30),
            },
        )
        .map_err(|e| e.to_string())?;
        let index = manifest
            .experts
            .iter()
            .enumerate()
            .map(|(i, e)| ((e.layer, e.expert), i))
            .collect();
        Ok(Self {
            manifest,
            index,
            ring: Ring::new(slots, slot_bytes),
            streamer,
            slot_of: HashMap::new(),
            owner: vec![None; slots],
            tick: 0,
            stats: CacheStats::default(),
        })
    }

    pub fn slots(&self) -> usize {
        self.owner.len()
    }

    pub fn entry(&self, layer: u16, expert: u16) -> Option<&ExpertEntry> {
        self.index.get(&(layer, expert)).map(|&i| &self.manifest.experts[i])
    }

    /// Test hook: replace an expert's expected digest (the negative control must fail closed).
    pub fn corrupt_digest(&mut self, layer: u16, expert: u16) {
        if let Some(&i) = self.index.get(&(layer, expert)) {
            self.manifest.experts[i].digest = blake3::hash(b"not this expert").to_hex().to_string();
        }
    }

    /// Forget every resident expert (the slots stay allocated): the next lookup of any expert refills it
    /// from the SSD and checks its digest again (L190 fail-closed control).
    pub fn evict_all(&mut self) {
        self.slot_of.clear();
        self.owner.iter_mut().for_each(|o| *o = None);
    }

    fn victim(&self) -> usize {
        if let Some(s) = self.owner.iter().position(|o| o.is_none()) {
            return s;
        }
        let mut best = 0;
        for (s, o) in self.owner.iter().enumerate() {
            if o.unwrap().1 < self.owner[best].unwrap().1 {
                best = s;
            }
        }
        best
    }

    fn fill(&mut self, key: (u16, u16)) -> Result<usize, String> {
        let i = *self.index.get(&key).ok_or(format!("no expert {key:?} in the manifest"))?;
        let slot = self.victim();
        if let Some((old, _)) = self.owner[slot] {
            self.slot_of.remove(&old);
        }
        self.owner[slot] = None;
        let t0 = Instant::now();
        self.streamer
            .load_one(&self.ring, slot, &self.manifest.experts[i])
            .map_err(|e| format!("heartbeat fill {key:?}: {e}"))?;
        self.stats.fill_s += t0.elapsed().as_secs_f64();
        self.stats.fill_bytes += self.manifest.experts[i].len;
        self.tick += 1;
        self.owner[slot] = Some((key, self.tick));
        self.slot_of.insert(key, slot);
        Ok(slot)
    }

    /// Fill every listed expert not yet resident, in parallel on the worker pool (distinct slots, one
    /// writer each). Returns how many were read.
    pub fn prefetch(&mut self, keys: &[(u16, u16)]) -> Result<usize, String> {
        // victims: empty slots first, then least recently used; never a slot holding a listed expert
        let wanted: std::collections::HashSet<(u16, u16)> = keys.iter().copied().collect();
        let mut victims: Vec<usize> = (0..self.owner.len())
            .filter(|&s| self.owner[s].is_none_or(|(k, _)| !wanted.contains(&k)))
            .collect();
        victims.sort_by_key(|&s| self.owner[s].map(|x| x.1).unwrap_or(0));
        let mut todo: Vec<(usize, usize)> = Vec::new(); // (slot, manifest index)
        let mut seen = std::collections::HashSet::new();
        for &k in keys {
            if self.slot_of.contains_key(&k) || !seen.insert(k) {
                continue;
            }
            let i = *self.index.get(&k).ok_or(format!("no expert {k:?}"))?;
            let Some(&s) = victims.get(todo.len()) else { break };
            todo.push((s, i));
        }
        for &(s, _) in &todo {
            if let Some((old, _)) = self.owner[s] {
                self.slot_of.remove(&old);
            }
            self.owner[s] = None;
        }
        let workers = self.streamer.config().workers;
        let t0 = Instant::now();
        let (ring, streamer, m) = (&self.ring, &self.streamer, &self.manifest);
        let res: Vec<Result<(), String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..workers)
                .map(|w| {
                    let todo = &todo;
                    sc.spawn(move || -> Result<(), String> {
                        for (j, &(s, i)) in todo.iter().enumerate() {
                            if j % workers == w {
                                streamer.load_one(ring, s, &m.experts[i]).map_err(|e| e.to_string())?;
                            }
                        }
                        Ok(())
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(Err("worker panicked".into()))).collect()
        });
        for r in res {
            r?;
        }
        self.stats.fill_s += t0.elapsed().as_secs_f64();
        for &(s, i) in &todo {
            let e = &self.manifest.experts[i];
            self.tick += 1;
            self.owner[s] = Some(((e.layer, e.expert), self.tick));
            self.slot_of.insert((e.layer, e.expert), s);
            self.stats.fill_bytes += e.len;
        }
        self.stats.prefetched += todo.len() as u64;
        Ok(todo.len())
    }

    /// Copy one tensor of a routed expert out of its published slot (filling the slot on a miss).
    /// Counts one lookup per expert (on its first tensor read), not per tensor.
    pub fn read_tensor(&mut self, layer: u16, expert: u16, name: &str, dtype: &str, shape: &[usize], dst: &mut [u8]) -> Result<(), String> {
        let key = (layer, expert);
        let tr: TensorRange = self
            .entry(layer, expert)
            .ok_or(format!("{name}: not in the manifest"))?
            .tensors
            .iter()
            .find(|t| t.name == name)
            .cloned()
            .ok_or(format!("{name}: not in the expert's manifest entry"))?;
        if tr.dtype != dtype {
            return Err(format!("{name}: dtype {} in the checkpoint, {dtype} asked", tr.dtype));
        }
        let want: Vec<u64> = shape.iter().map(|&x| x as u64).collect();
        if tr.shape != want || tr.len as usize != dst.len() {
            return Err(format!("{name}: shape {:?} / {} bytes, asked {:?} / {}", tr.shape, tr.len, want, dst.len()));
        }
        let first = self.entry(layer, expert).unwrap().tensors[0].name == name;
        if first {
            self.stats.lookups += 1;
        }
        let slot = match self.slot_of.get(&key) {
            Some(&s) if self.ring.tag(s).map(|t| (t.layer, t.expert)) == Some(key) => s,
            _ => {
                // one demand fill per missed expert, whichever of its tensors is read first (L190: the 30B
                // drafter reads gate_proj first, not tensors[0]; counting on tensors[0] reported 0 misses)
                self.stats.misses += 1;
                self.fill(key)?
            }
        };
        self.stats.hits = self.stats.lookups.saturating_sub(self.stats.misses);
        self.tick += 1;
        self.owner[slot] = Some((key, self.tick));
        let lease = self
            .ring
            .lease(layer, expert)
            .map_err(|e| e.to_string())?
            .ok_or(format!("{name}: slot vanished after publish"))?;
        // the payload is the concatenation of the lease's segments; copy [offset, offset+len)
        let (mut pos, mut out) = (0u64, 0usize);
        let (start, end) = (tr.offset, tr.offset + tr.len);
        for seg in lease.segments() {
            let (s0, s1) = (pos, pos + seg.len() as u64);
            let (a, b) = (start.max(s0), end.min(s1));
            if a < b {
                let n = (b - a) as usize;
                dst[out..out + n].copy_from_slice(&seg[(a - s0) as usize..(b - s0) as usize]);
                out += n;
            }
            pos = s1;
        }
        if out != dst.len() {
            return Err(format!("{name}: copied {out} of {} bytes", dst.len()));
        }
        Ok(())
    }
}
