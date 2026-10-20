//! The movement engines: deliver bytes along the schedule and never compute. Built on the heartbeat ring
//! (`aien-heartbeat`): aligned slots, O_DIRECT reads, BLAKE3 digest checked before a slot is published, any error
//! fails the whole ring closed. Buffer `u` of the schedule is the slot group `u * slots .. (u + 1) * slots` of the
//! cylinder ring.
//!
//! Engine count (Drake 2026-10-20: "three, or more"): each tier boundary is its own engine with its own buffer pair.
//! * [`Engines::Direct`] (1 engine): the NVMe engine reads straight into the cylinder and verifies there.
//! * [`Engines::Staged`] (2 engines): the NVMe engine reads into a staging pair (host RAM, verified on arrival); the
//!   RAM engine copies a verified staging slot into the cylinder and verifies again before publishing. The two run on
//!   their own threads with the staging pair between them, so the next beat's disk read overlaps this beat's copy.
//! On the GB10 the host RAM and the GPU region are one LPDDR5X pool, so the RAM engine is a same-pool copy; the
//! fast-memory engine (L2/SMEM) and the math engine exist only on the chip (next cut).
use crate::schedule::{Fill, Schedule};
use aien_heartbeat::manifest::{ExpertEntry, Manifest};
use aien_heartbeat::ring::Ring;
use aien_heartbeat::stream::{StreamConfig, Streamer};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum Engines {
    Direct,
    Staged,
}

impl Engines {
    pub fn count(self) -> usize {
        match self {
            Engines::Direct => 1,
            Engines::Staged => 2,
        }
    }
    pub fn parse(s: &str) -> Result<Self, String> {
        match s {
            "1" | "direct" => Ok(Engines::Direct),
            "2" | "staged" => Ok(Engines::Staged),
            x => Err(format!("engines {x}: 1|2")),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FillStamp {
    pub for_beat: u32,
    pub buffer: u32,
    pub bytes: u64,
    pub issue_ns: i64,
    pub done_ns: i64,
    /// NVMe engine done (staging full, verified). Equals `done_ns` on the direct path.
    pub nvme_done_ns: i64,
    /// RAM engine start (first staging lease taken). Equals `nvme_done_ns` on the direct path.
    pub ram_start_ns: i64,
    /// Sum over parts of the worker's read time (overlapping across workers, so not wall time).
    pub read_ns_sum: u64,
    pub digest_ns_sum: u64,
    pub lease_wait_ns_sum: u64,
    /// Parts of this fill served from the resident tier (a copy from host RAM) instead of the disk (H4 head swap).
    #[serde(default)]
    pub hits: u32,
}

pub struct MovementEngine {
    ring: Arc<Ring>,
    streamer: Streamer,
    staging: Option<Arc<Ring>>,
    sched: Arc<Schedule>,
    entries: Vec<ExpertEntry>,
    workers: usize,
    engines: Engines,
    t0: Instant,
    /// H4: the resident tier (a head held in host RAM). A part published there is copied from it, not read from disk.
    resident: Option<Arc<Ring>>,
    last_hits: AtomicU32,
}

const LEASE_WAIT: Duration = Duration::from_secs(60);

impl MovementEngine {
    pub fn open(sched: Arc<Schedule>, m: &Manifest, workers: usize, direct: bool, engines: Engines, t0: Instant) -> Result<Self, String> {
        let num_slots = (sched.buffers * sched.slots) as usize;
        let slot_bytes = Streamer::slot_bytes(m);
        let ring = Arc::new(Ring::new(num_slots, slot_bytes));
        let cfg = StreamConfig { slots: num_slots, workers, direct, publish: true, lease_wait: LEASE_WAIT };
        let streamer = Streamer::open(&m.shard_paths(), cfg).map_err(|e| e.to_string())?;
        let staging = match engines {
            Engines::Direct => None,
            Engines::Staged => Some(Arc::new(Ring::new(2 * sched.slots as usize, slot_bytes))),
        };
        let mut entries = Vec::with_capacity(sched.parts.len());
        for p in &sched.parts {
            let e = m.experts.iter().find(|e| e.layer == p.layer && e.expert == p.expert).ok_or(format!("part {}: layer {} expert {} not in the manifest", p.id, p.layer, p.expert))?;
            if e.digest != p.digest || e.len != p.bytes {
                return Err(format!("part {}: schedule and manifest disagree (digest or length); rebuild the schedule", p.id));
            }
            entries.push(e.clone());
        }
        Ok(Self { ring, streamer, staging, sched, entries, workers: workers.max(1), engines, t0, resident: None, last_hits: AtomicU32::new(0) })
    }

    /// H4: serve parts from `r` when they are published there (the resident head); misses go to the disk.
    pub fn set_resident(&mut self, r: Arc<Ring>) {
        self.resident = Some(r);
    }

    pub fn ring(&self) -> Arc<Ring> {
        self.ring.clone()
    }

    pub fn engines(&self) -> Engines {
        self.engines
    }

    fn ns(&self) -> i64 {
        self.t0.elapsed().as_nanos() as i64
    }

    /// Stage 0, the NVMe engine: every part of `f` from disk into `ring` (the cylinder on the direct path, the
    /// staging pair on the staged path; staging buffer = for_beat % 2). Returns (read, digest, lease-wait, bytes) sums.
    fn nvme_stage(&self, f: &Fill, ring: &Ring, slot_of: &(dyn Fn(usize) -> usize + Sync)) -> Result<(u64, u64, u64, u64), String> {
        let w = self.workers.min(f.parts.len()).max(1);
        let hits = AtomicU32::new(0);
        let results: Vec<Result<(u64, u64, u64, u64), String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..w)
                .map(|wi| {
                    let (streamer, entries, resident, hits) = (&self.streamer, &self.entries, &self.resident, &hits);
                    sc.spawn(move || {
                        let (mut rd, mut dg, mut wt, mut by) = (0u64, 0u64, 0u64, 0u64);
                        for (i, &p) in f.parts.iter().enumerate() {
                            if i % w != wi {
                                continue;
                            }
                            let slot = slot_of(i);
                            let e = &entries[p as usize];
                            // H4: a part the resident head holds is copied from host RAM and verified again on publish
                            if let Some(res) = resident {
                                if let Some(lease) = res.lease(e.layer, e.expert).map_err(|x| x.to_string())? {
                                    let t = Instant::now();
                                    let (mut g, waited) = ring.begin_fill(slot, LEASE_WAIT).map_err(|x| format!("beat {} part {p} slot {slot}: {x}", f.for_beat))?;
                                    let mut off = 0usize;
                                    {
                                        let dst = g.buf_mut();
                                        for seg in lease.segments() {
                                            if off + seg.len() > dst.len() {
                                                return Err(format!("beat {} part {p}: payload {} > slot {}", f.for_beat, off + seg.len(), dst.len()));
                                            }
                                            dst[off..off + seg.len()].copy_from_slice(seg);
                                            off += seg.len();
                                        }
                                    }
                                    drop(lease);
                                    let expected = e.digest_bytes().map_err(|x| x.to_string())?;
                                    let (_tag, d) = g.publish(e.layer, e.expert, &[(0, off)], &expected).map_err(|x| format!("beat {} part {p}: {x}", f.for_beat))?;
                                    rd += (t.elapsed() - d).as_nanos() as u64;
                                    dg += d.as_nanos() as u64;
                                    wt += waited.as_nanos() as u64;
                                    by += e.len;
                                    hits.fetch_add(1, Ordering::Relaxed);
                                    continue;
                                }
                            }
                            let (r, d, waited, _dev) = streamer.load_one(ring, slot, &entries[p as usize]).map_err(|e| format!("beat {} part {p} slot {slot}: {e}", f.for_beat))?;
                            rd += r.as_nanos() as u64;
                            dg += d.as_nanos() as u64;
                            wt += waited.as_nanos() as u64;
                            by += entries[p as usize].len;
                        }
                        Ok((rd, dg, wt, by))
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(Err("nvme engine worker panicked".into()))).collect()
        });
        let mut acc = (0u64, 0u64, 0u64, 0u64);
        for r in results {
            let (a, b, c, d) = r?;
            acc = (acc.0 + a, acc.1 + b, acc.2 + c, acc.3 + d);
        }
        self.last_hits.store(hits.load(Ordering::Relaxed), Ordering::Relaxed);
        Ok(acc)
    }

    /// Stage 1, the RAM engine: every part of `f` from its verified staging slot into the cylinder slot, verified
    /// again before publishing (the ring's own rule: nothing unverified is ever leased). Returns digest-ns sum.
    fn ram_stage(&self, f: &Fill, staging: &Ring) -> Result<u64, String> {
        let w = self.workers.min(f.parts.len()).max(1);
        let results: Vec<Result<u64, String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..w)
                .map(|wi| {
                    let (ring, sched, entries) = (&self.ring, &self.sched, &self.entries);
                    sc.spawn(move || {
                        let mut dg = 0u64;
                        for (i, &p) in f.parts.iter().enumerate() {
                            if i % w != wi {
                                continue;
                            }
                            let e = &entries[p as usize];
                            let lease = staging.lease(e.layer, e.expert).map_err(|x| x.to_string())?.ok_or_else(|| format!("beat {} part {p}: staging slot not published", f.for_beat))?;
                            let slot = sched.slot(f.buffer, i);
                            let (mut g, _waited) = ring.begin_fill(slot, LEASE_WAIT).map_err(|x| format!("beat {} part {p} slot {slot}: {x}", f.for_beat))?;
                            let mut off = 0usize;
                            {
                                let dst = g.buf_mut();
                                for seg in lease.segments() {
                                    if off + seg.len() > dst.len() {
                                        return Err(format!("beat {} part {p}: payload {} > slot {}", f.for_beat, off + seg.len(), dst.len()));
                                    }
                                    dst[off..off + seg.len()].copy_from_slice(seg);
                                    off += seg.len();
                                }
                            }
                            drop(lease);
                            let expected = e.digest_bytes().map_err(|x| x.to_string())?;
                            let (_tag, d) = g.publish(e.layer, e.expert, &[(0, off)], &expected).map_err(|x| format!("beat {} part {p}: {x}", f.for_beat))?;
                            dg += d.as_nanos() as u64;
                        }
                        Ok(dg)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().unwrap_or(Err("ram engine worker panicked".into()))).collect()
        });
        let mut dg = 0u64;
        for r in results {
            dg += r?;
        }
        Ok(dg)
    }

    /// One fill on the direct path (1 engine). Returns when the last cylinder slot is published.
    pub fn fill(&self, f: &Fill) -> Result<FillStamp, String> {
        let issue = self.ns();
        let sched = &self.sched;
        let (rd, dg, wt, by) = self.nvme_stage(f, &self.ring, &|i| sched.slot(f.buffer, i))?;
        let done = self.ns();
        Ok(FillStamp { for_beat: f.for_beat, buffer: f.buffer, bytes: by, issue_ns: issue, done_ns: done, nvme_done_ns: done, ram_start_ns: done, read_ns_sum: rd, digest_ns_sum: dg, lease_wait_ns_sum: wt, hits: self.last_hits.load(Ordering::Relaxed) })
    }

    /// Run as the intake: fills arrive in beat order on `jobs`; each stamp goes back on `stamps`. An error is sent
    /// once and everything stops: the ring is failed closed so every later lease refuses. On the staged path two
    /// threads run (NVMe engine, RAM engine) with the staging pair between them.
    pub fn serve(self, jobs: Receiver<Fill>, stamps: Sender<Result<FillStamp, String>>) {
        match self.staging.clone() {
            None => {
                for f in jobs {
                    let r = self.fill(&f);
                    let stop = r.is_err();
                    if stop {
                        self.ring.fail();
                    }
                    if stamps.send(r).is_err() || stop {
                        break;
                    }
                }
            }
            Some(staging) => {
                let me = Arc::new(self);
                let (htx, hrx) = mpsc::channel::<(Fill, FillStamp)>();
                let nvme = {
                    let (me, staging, stamps) = (me.clone(), staging.clone(), stamps.clone());
                    std::thread::spawn(move || {
                        for f in jobs {
                            let issue = me.ns();
                            let slots = me.sched.slots as usize;
                            let sbuf = (f.for_beat % 2) as usize;
                            match me.nvme_stage(&f, &staging, &|i| sbuf * slots + i) {
                                Ok((rd, dg, wt, by)) => {
                                    let st = FillStamp { for_beat: f.for_beat, buffer: f.buffer, bytes: by, issue_ns: issue, nvme_done_ns: me.ns(), read_ns_sum: rd, digest_ns_sum: dg, lease_wait_ns_sum: wt, ..Default::default() };
                                    if htx.send((f, st)).is_err() {
                                        break;
                                    }
                                }
                                Err(e) => {
                                    staging.fail();
                                    me.ring.fail();
                                    let _ = stamps.send(Err(format!("nvme engine: {e}")));
                                    break;
                                }
                            }
                        }
                    })
                };
                for (f, mut st) in hrx {
                    st.ram_start_ns = me.ns();
                    match me.ram_stage(&f, &staging) {
                        Ok(dg) => {
                            st.digest_ns_sum += dg;
                            st.done_ns = me.ns();
                            if stamps.send(Ok(st)).is_err() {
                                break;
                            }
                        }
                        Err(e) => {
                            staging.fail();
                            me.ring.fail();
                            let _ = stamps.send(Err(format!("ram engine: {e}")));
                            break;
                        }
                    }
                }
                let _ = nvme.join();
            }
        }
    }
}
