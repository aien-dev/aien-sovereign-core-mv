//! The movement engine: delivers bytes along the schedule and never computes. Built on the heartbeat ring
//! (`aien-heartbeat`): N aligned slots, O_DIRECT reads, BLAKE3 digest checked before a slot is published, any
//! error fails the whole ring closed. Buffer `u` of the schedule is the slot group `u * slots .. (u + 1) * slots`.
use crate::schedule::{Fill, Schedule};
use aien_heartbeat::manifest::{ExpertEntry, Manifest};
use aien_heartbeat::ring::Ring;
use aien_heartbeat::stream::{StreamConfig, Streamer};
use serde::{Deserialize, Serialize};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::Arc;
use std::time::{Duration, Instant};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub struct FillStamp {
    pub for_beat: u32,
    pub buffer: u32,
    pub bytes: u64,
    pub issue_ns: i64,
    pub done_ns: i64,
    /// Sum over parts of the worker's read time (overlapping across workers, so not wall time).
    pub read_ns_sum: u64,
    pub digest_ns_sum: u64,
    pub lease_wait_ns_sum: u64,
}

pub struct MovementEngine {
    ring: Arc<Ring>,
    streamer: Streamer,
    sched: Arc<Schedule>,
    entries: Vec<ExpertEntry>,
    workers: usize,
    t0: Instant,
}

impl MovementEngine {
    pub fn open(sched: Arc<Schedule>, m: &Manifest, workers: usize, direct: bool, t0: Instant) -> Result<Self, String> {
        let num_slots = (sched.buffers * sched.slots) as usize;
        let slot_bytes = Streamer::slot_bytes(m);
        let ring = Arc::new(Ring::new(num_slots, slot_bytes));
        let cfg = StreamConfig { slots: num_slots, workers, direct, publish: true, lease_wait: Duration::from_secs(30) };
        let streamer = Streamer::open(&m.shard_paths(), cfg).map_err(|e| e.to_string())?;
        let mut entries = Vec::with_capacity(sched.parts.len());
        for p in &sched.parts {
            let e = m.experts.iter().find(|e| e.layer == p.layer && e.expert == p.expert).ok_or(format!("part {}: layer {} expert {} not in the manifest", p.id, p.layer, p.expert))?;
            if e.digest != p.digest || e.len != p.bytes {
                return Err(format!("part {}: schedule and manifest disagree (digest or length); rebuild the schedule", p.id));
            }
            entries.push(e.clone());
        }
        Ok(Self { ring, streamer, sched, entries, workers: workers.max(1), t0 })
    }

    pub fn ring(&self) -> Arc<Ring> {
        self.ring.clone()
    }

    pub fn slot_bytes(&self) -> usize {
        self.ring.slot_capacity()
    }

    /// One fill: every part of `f` into its slot of buffer `f.buffer`, `workers` threads, part `i` on worker
    /// `i % workers`. Returns when the last slot is published (digest verified). Fails closed through the ring.
    pub fn fill(&self, f: &Fill) -> Result<FillStamp, String> {
        let issue = Instant::now();
        let w = self.workers.min(f.parts.len()).max(1);
        let results: Vec<Result<(u64, u64, u64, u64), String>> = std::thread::scope(|sc| {
            let hs: Vec<_> = (0..w)
                .map(|wi| {
                    let (ring, streamer, sched, entries) = (&self.ring, &self.streamer, &self.sched, &self.entries);
                    sc.spawn(move || {
                        let (mut rd, mut dg, mut wt, mut by) = (0u64, 0u64, 0u64, 0u64);
                        for (i, &p) in f.parts.iter().enumerate() {
                            if i % w != wi {
                                continue;
                            }
                            let slot = sched.slot(f.buffer, i);
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
            hs.into_iter().map(|h| h.join().unwrap_or(Err("fill worker panicked".into()))).collect()
        });
        let mut st = FillStamp { for_beat: f.for_beat, buffer: f.buffer, issue_ns: (issue - self.t0).as_nanos() as i64, ..Default::default() };
        for r in results {
            let (rd, dg, wt, by) = r?;
            st.read_ns_sum += rd;
            st.digest_ns_sum += dg;
            st.lease_wait_ns_sum += wt;
            st.bytes += by;
        }
        st.done_ns = self.t0.elapsed().as_nanos() as i64;
        Ok(st)
    }

    /// Run as the intake thread: fills arrive in beat order on `jobs`; each stamp goes back on `stamps`.
    /// An error is sent once and the thread stops: the ring is failed closed so every later lease refuses.
    pub fn serve(self, jobs: Receiver<Fill>, stamps: Sender<Result<FillStamp, String>>) {
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
}
