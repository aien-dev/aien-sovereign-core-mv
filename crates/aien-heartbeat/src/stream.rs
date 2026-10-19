//! The streamer: a fixed pool of workers that read expert spans with
//! O_DIRECT `pread` into ring slots in layer order, verify, then publish.
//!
//! Schedule item `i` always lands in slot `i % slots`, and slot `s` is owned
//! by worker `s % workers`, so each slot has exactly one writer and slots are
//! refilled in schedule order.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::aligned::{round_down, round_up};
use crate::error::HeartbeatError;
use crate::manifest::{ExpertEntry, Manifest};
use crate::ring::Ring;
use crate::IO_ALIGN;

/// Rotation order: indices into `Manifest::experts`, in (layer, expert)
/// order, hot-resident experts removed.
#[derive(Debug, Clone)]
pub struct Schedule {
    pub items: Vec<usize>,
}

impl Schedule {
    pub fn layer_order(m: &Manifest, hot: &HashSet<(u16, u16)>) -> Self {
        let mut items: Vec<usize> = (0..m.experts.len())
            .filter(|&i| !hot.contains(&(m.experts[i].layer, m.experts[i].expert)))
            .collect();
        items.sort_by_key(|&i| (m.experts[i].layer, m.experts[i].expert));
        Self { items }
    }

    pub fn bytes(&self, m: &Manifest) -> u64 {
        self.items.iter().map(|&i| m.experts[i].len).sum()
    }
}

#[derive(Debug, Clone)]
pub struct StreamConfig {
    pub slots: usize,
    pub workers: usize,
    /// Open shards with O_DIRECT (bypass page cache).
    pub direct: bool,
    /// Verify digest and publish (true), or read and discard without
    /// publishing (false; used only to measure raw read cost).
    pub publish: bool,
    /// How long a writer waits for readers to release a slot before failing.
    pub lease_wait: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct RotationStats {
    pub experts: u64,
    /// Expert payload bytes (what a reader sees).
    pub payload_bytes: u64,
    /// Bytes actually read from the device (aligned windows).
    pub device_bytes: u64,
    pub wall: Duration,
    /// Summed over workers.
    pub read_time: Duration,
    pub digest_time: Duration,
    pub lease_wait_time: Duration,
}

impl RotationStats {
    pub fn payload_gbps(&self) -> f64 {
        self.payload_bytes as f64 / self.wall.as_secs_f64() / 1e9
    }
    pub fn device_gbps(&self) -> f64 {
        self.device_bytes as f64 / self.wall.as_secs_f64() / 1e9
    }
}

pub struct Streamer {
    files: Vec<File>,
    cfg: StreamConfig,
}

impl Streamer {
    pub fn open(shards: &[PathBuf], cfg: StreamConfig) -> Result<Self, HeartbeatError> {
        assert!(cfg.slots > 0 && cfg.workers > 0 && cfg.workers <= cfg.slots);
        let mut files = Vec::with_capacity(shards.len());
        for p in shards {
            let mut o = OpenOptions::new();
            o.read(true);
            if cfg.direct {
                o.custom_flags(libc::O_DIRECT);
            }
            files.push(
                o.open(p)
                    .map_err(|e| HeartbeatError::io(p.display().to_string(), e))?,
            );
        }
        Ok(Self { files, cfg })
    }

    pub fn config(&self) -> &StreamConfig {
        &self.cfg
    }

    /// Bytes each slot needs for the largest aligned window in `m`.
    pub fn slot_bytes(m: &Manifest) -> usize {
        // Each segment gets its own aligned window; worst case it starts one
        // byte before an alignment boundary.
        m.experts
            .iter()
            .map(|e| {
                e.segments
                    .iter()
                    .map(|s| round_up(s.len as usize + IO_ALIGN, IO_ALIGN))
                    .sum::<usize>()
            })
            .max()
            .unwrap_or(IO_ALIGN)
    }

    /// Run one full rotation over `schedule`. Any error fails the ring closed
    /// and stops every worker.
    pub fn rotate(
        &self,
        ring: &Ring,
        m: &Manifest,
        schedule: &Schedule,
    ) -> Result<RotationStats, HeartbeatError> {
        assert_eq!(ring.num_slots(), self.cfg.slots);
        let read_ns = AtomicU64::new(0);
        let digest_ns = AtomicU64::new(0);
        let wait_ns = AtomicU64::new(0);
        let device = AtomicU64::new(0);
        let t0 = Instant::now();
        let results: Vec<Result<(), HeartbeatError>> = std::thread::scope(|sc| {
            let handles: Vec<_> = (0..self.cfg.workers)
                .map(|w| {
                    let (read_ns, digest_ns, wait_ns, device) =
                        (&read_ns, &digest_ns, &wait_ns, &device);
                    sc.spawn(move || -> Result<(), HeartbeatError> {
                        for (i, &ei) in schedule.items.iter().enumerate() {
                            let slot = i % self.cfg.slots;
                            if slot % self.cfg.workers != w {
                                continue;
                            }
                            if ring.is_failed() {
                                return Err(HeartbeatError::RingFailed);
                            }
                            let e = &m.experts[ei];
                            let (r, d, wt, dev) = self.load_one(ring, slot, e)?;
                            read_ns.fetch_add(r.as_nanos() as u64, Ordering::Relaxed);
                            digest_ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
                            wait_ns.fetch_add(wt.as_nanos() as u64, Ordering::Relaxed);
                            device.fetch_add(dev, Ordering::Relaxed);
                        }
                        Ok(())
                    })
                })
                .collect();
            handles
                .into_iter()
                .map(|h| h.join().unwrap_or(Err(HeartbeatError::RingFailed)))
                .collect()
        });
        let wall = t0.elapsed();
        // Report the root cause, not the RingFailed echoes from other workers.
        let mut first_err = None;
        for r in results {
            if let Err(e) = r {
                if first_err.is_none() || matches!(first_err, Some(HeartbeatError::RingFailed)) {
                    first_err = Some(e);
                }
            }
        }
        if let Some(e) = first_err {
            ring.fail();
            return Err(e);
        }
        Ok(RotationStats {
            experts: schedule.items.len() as u64,
            payload_bytes: schedule.bytes(m),
            device_bytes: device.into_inner(),
            wall,
            read_time: Duration::from_nanos(read_ns.into_inner()),
            digest_time: Duration::from_nanos(digest_ns.into_inner()),
            lease_wait_time: Duration::from_nanos(wait_ns.into_inner()),
        })
    }

    /// Read, digest-check and publish one expert into `slot` (the demand-fill path of a cache on
    /// this ring; `rotate` uses the same call for every schedule item).
    pub fn load_one(
        &self,
        ring: &Ring,
        slot: usize,
        e: &ExpertEntry,
    ) -> Result<(Duration, Duration, Duration, u64), HeartbeatError> {
        let expected = e.digest_bytes()?;
        let (mut g, waited) = ring.begin_fill(slot, self.cfg.lease_wait)?;
        let cap = g.buf_mut().len();
        let t_read = Instant::now();
        let mut ranges = Vec::with_capacity(e.segments.len());
        let mut base = 0usize; // aligned position of this segment's window in the slot
        let mut device = 0u64;
        for seg in &e.segments {
            let astart = round_down(seg.start, IO_ALIGN as u64);
            let off = (seg.start - astart) as usize;
            let need = off + seg.len as usize;
            let window = round_up(need, IO_ALIGN);
            if base + window > cap {
                g.fail();
                return Err(HeartbeatError::PayloadTooLarge {
                    slot,
                    need: base + window,
                    cap,
                });
            }
            let file = &self.files[seg.shard as usize];
            let buf = &mut g.buf_mut()[base..base + window];
            let mut got = 0usize;
            while got < need {
                match file.read_at(&mut buf[got..], astart + got as u64) {
                    Ok(0) => break,
                    Ok(n) => got += n,
                    Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                    Err(err) => {
                        g.fail();
                        return Err(HeartbeatError::io(
                            format!("pread expert ({},{})", e.layer, e.expert),
                            err,
                        ));
                    }
                }
            }
            if got < need {
                g.fail();
                return Err(HeartbeatError::ShortRead {
                    expert: (e.layer, e.expert),
                    wanted: need,
                    got,
                });
            }
            ranges.push((base + off, seg.len as usize));
            device += got as u64;
            base += window;
        }
        let read = t_read.elapsed();
        let digest = if self.cfg.publish {
            g.publish(e.layer, e.expert, &ranges, &expected)?.1
        } else {
            g.discard();
            Duration::ZERO
        };
        Ok((read, digest, waited, device))
    }
}
