//! Bounded expert read adapter (Expert Ribbon P1, sc#629): random reads of one selected expert from the
//! checkpoint shards into a fixed pool of aligned staging buffers, verified against the manifest digest
//! before anyone can see the bytes.
//!
//! * Bounded: `queue_depth` staging buffers, allocated once. `submit` refuses with [`ReadError::QueueFull`]
//!   when all are in use (no unbounded queue, no hidden allocation).
//! * Verified: a [`VerifiedExpert`] exists only after the whole expert was read and its BLAKE3 matched. A short
//!   read, EOF, I/O error or digest mismatch returns an explicit error and publishes nothing.
//! * Cancellation and deadlines: [`Ticket::cancel`] and [`Ticket::wait`]'s deadline. A read that is already in
//!   the kernel cannot be interrupted; its result is discarded and the buffer returns to the pool when it ends.
//! * Shutdown: dropping the adapter closes the queue and joins every worker (outstanding tickets see
//!   [`ReadError::Shutdown`]).
//! * Device: [`ShardDevice`]. [`FileShards`] is the Linux one (O_DIRECT or buffered, chosen by measurement, see
//!   the README); [`FakeShards`] is a deterministic in-memory one with injectable latency and faults for tests.

use std::collections::VecDeque;
use std::fs::{File, OpenOptions};
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::aligned::{round_down, round_up, AlignedBuf};
use crate::layout::Payload;
use crate::manifest::{ExpertEntry, Manifest};

/// Positional reads from the checkpoint shards.
pub trait ShardDevice: Send + Sync {
    fn read_at(&self, shard: u32, buf: &mut [u8], off: u64) -> std::io::Result<usize>;
    /// Alignment of buffers, offsets and lengths the device needs (4096 for O_DIRECT, 1 otherwise).
    fn align(&self) -> usize;
}

/// The shard files, read-only, O_DIRECT or buffered.
pub struct FileShards {
    files: Vec<File>,
    direct: bool,
}

impl FileShards {
    pub fn open(paths: &[PathBuf], direct: bool) -> Result<Self, ReadError> {
        let mut files = Vec::with_capacity(paths.len());
        for p in paths {
            let mut o = OpenOptions::new();
            o.read(true);
            if direct {
                o.custom_flags(libc::O_DIRECT);
            }
            files.push(
                o.open(p)
                    .map_err(|e| ReadError::Io(format!("open {}: {e}", p.display())))?,
            );
        }
        Ok(Self { files, direct })
    }
}

impl ShardDevice for FileShards {
    fn read_at(&self, shard: u32, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
        let f = self
            .files
            .get(shard as usize)
            .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidInput, "shard index"))?;
        f.read_at(buf, off)
    }
    fn align(&self) -> usize {
        if self.direct {
            crate::IO_ALIGN
        } else {
            1
        }
    }
}

/// A fault the fake device applies to one read call (counted from 0 over the device's life).
#[derive(Debug)]
pub enum Fault {
    /// The call returns EIO.
    Eio,
    /// The call returns at most this many bytes, then every later call on the same request returns 0 (EOF).
    Short(usize),
    /// The call succeeds but flips one byte of what it returns.
    Flip,
    /// The call blocks until the sender is dropped or sends (a stalled NVMe), then reads normally.
    Stall(Receiver<()>),
}

/// Deterministic in-memory shards for tests: optional fixed latency per read and a fault plan by call index.
pub struct FakeShards {
    pub shards: Vec<Vec<u8>>,
    pub latency: Duration,
    faults: Mutex<VecDeque<(u64, Fault)>>,
    calls: AtomicU64,
    eof_after_short: Mutex<Option<u64>>,
}

impl FakeShards {
    pub fn new(shards: Vec<Vec<u8>>, latency: Duration) -> Self {
        Self {
            shards,
            latency,
            faults: Mutex::new(VecDeque::new()),
            calls: AtomicU64::new(0),
            eof_after_short: Mutex::new(None),
        }
    }
    /// Apply `f` to read call number `call` (0-based over the device's life).
    pub fn fault_at(&self, call: u64, f: Fault) {
        self.faults.lock().unwrap().push_back((call, f));
    }
    pub fn calls(&self) -> u64 {
        self.calls.load(Ordering::Relaxed)
    }
}

impl ShardDevice for FakeShards {
    fn read_at(&self, shard: u32, buf: &mut [u8], off: u64) -> std::io::Result<usize> {
        let call = self.calls.fetch_add(1, Ordering::Relaxed);
        if !self.latency.is_zero() {
            std::thread::sleep(self.latency);
        }
        if let Some(c) = *self.eof_after_short.lock().unwrap() {
            if call == c + 1 {
                return Ok(0);
            }
        }
        let fault = {
            let mut q = self.faults.lock().unwrap();
            q.iter()
                .position(|(c, _)| *c == call)
                .and_then(|i| q.remove(i))
                .map(|(_, f)| f)
        };
        let data = &self.shards[shard as usize];
        let start = (off as usize).min(data.len());
        let mut n = buf.len().min(data.len() - start);
        match fault {
            Some(Fault::Eio) => return Err(std::io::Error::from_raw_os_error(libc::EIO)),
            Some(Fault::Short(m)) => {
                n = n.min(m);
                *self.eof_after_short.lock().unwrap() = Some(call);
            }
            Some(Fault::Stall(ref rx)) => {
                let _ = rx.recv();
            }
            _ => {}
        }
        buf[..n].copy_from_slice(&data[start..start + n]);
        if let (Some(Fault::Flip), true) = (&fault, n > 0) {
            buf[n / 2] ^= 0x5A;
        }
        Ok(n)
    }
    fn align(&self) -> usize {
        1
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReadError {
    /// Every staging buffer is in use (bounded queue).
    QueueFull,
    /// The expert does not fit one staging buffer.
    TooLarge {
        need: usize,
        cap: usize,
    },
    Io(String),
    /// Fewer bytes than the manifest span (EOF or a short read that never completed).
    ShortRead {
        expert: (u16, u16),
        wanted: usize,
        got: usize,
    },
    DigestMismatch {
        expert: (u16, u16),
    },
    Cancelled,
    /// The waiter's deadline passed before the read finished (a stalled read).
    Deadline,
    Shutdown,
    Manifest(String),
}

impl std::fmt::Display for ReadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ReadError {}

#[derive(Debug, Clone, Copy)]
pub struct ReadConfig {
    /// Reader threads (parallel reads in flight).
    pub workers: usize,
    /// Staging buffers = the most requests in flight or waiting to be consumed.
    pub queue_depth: usize,
}

#[derive(Debug, Clone, Default)]
pub struct ReadStats {
    pub reads_ok: u64,
    pub payload_bytes: u64,
    pub device_bytes: u64,
    pub read_ns: u64,
    pub digest_ns: u64,
    pub errors: u64,
    pub queue_full: u64,
}

struct Pool {
    bufs: Mutex<Vec<AlignedBuf>>,
    returned: Condvar,
    cap: usize,
}

impl Pool {
    fn give_back(&self, b: AlignedBuf) {
        self.bufs.lock().unwrap().push(b);
        self.returned.notify_all();
    }
}

/// A whole, digest-checked expert in a staging buffer. The buffer returns to the pool on drop.
pub struct VerifiedExpert {
    pub index: usize,
    pub layer: u16,
    pub expert: u16,
    buf: Option<AlignedBuf>,
    ranges: Vec<(usize, usize)>,
    pool: Arc<Pool>,
}

impl VerifiedExpert {
    pub fn payload(&self) -> Payload<'_> {
        let b = self.buf.as_ref().expect("buffer").as_slice();
        Payload::new(self.ranges.iter().map(|&(o, l)| &b[o..o + l]).collect())
    }
}

impl Drop for VerifiedExpert {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            self.pool.give_back(b);
        }
    }
}

type Reply = Result<VerifiedExpert, ReadError>;

struct Job {
    index: usize,
    buf: AlignedBuf,
    cancel: Arc<AtomicBool>,
    reply: SyncSender<Reply>,
}

/// One submitted read.
pub struct Ticket {
    rx: Receiver<Reply>,
    cancel: Arc<AtomicBool>,
}

impl Ticket {
    /// Request cancellation: a read not yet started is skipped; one in progress is discarded.
    pub fn cancel(&self) {
        self.cancel.store(true, Ordering::Release);
    }
    /// Wait up to `deadline` for the verified expert. On timeout the request is cancelled.
    pub fn wait(self, deadline: Duration) -> Result<VerifiedExpert, ReadError> {
        match self.rx.recv_timeout(deadline) {
            Ok(r) => r,
            Err(RecvTimeoutError::Timeout) => {
                self.cancel();
                Err(ReadError::Deadline)
            }
            Err(RecvTimeoutError::Disconnected) => Err(ReadError::Shutdown),
        }
    }
}

pub struct ReadAdapter {
    manifest: Arc<Manifest>,
    pool: Arc<Pool>,
    tx: Option<SyncSender<Job>>,
    workers: Vec<JoinHandle<()>>,
    stats: Arc<Mutex<ReadStats>>,
}

/// Bytes one staging buffer needs for any expert of `m` on a device of alignment `align`.
pub fn staging_bytes(m: &Manifest, align: usize) -> usize {
    m.experts
        .iter()
        .map(|e| {
            e.segments
                .iter()
                .map(|s| round_up(s.len as usize + align, align.max(crate::IO_ALIGN)))
                .sum::<usize>()
        })
        .max()
        .unwrap_or(crate::IO_ALIGN)
}

impl ReadAdapter {
    pub fn new(manifest: Arc<Manifest>, dev: Arc<dyn ShardDevice>, cfg: ReadConfig) -> Self {
        assert!(cfg.workers > 0 && cfg.queue_depth > 0);
        let cap = staging_bytes(&manifest, dev.align());
        let pool = Arc::new(Pool {
            bufs: Mutex::new((0..cfg.queue_depth).map(|_| AlignedBuf::new(cap)).collect()),
            returned: Condvar::new(),
            cap,
        });
        let (tx, rx) = mpsc::sync_channel::<Job>(cfg.queue_depth);
        let rx = Arc::new(Mutex::new(rx));
        let stats = Arc::new(Mutex::new(ReadStats::default()));
        let workers = (0..cfg.workers)
            .map(|_| {
                let (rx, dev, m, pool, stats) = (
                    rx.clone(),
                    dev.clone(),
                    manifest.clone(),
                    pool.clone(),
                    stats.clone(),
                );
                std::thread::spawn(move || loop {
                    let job = match rx.lock().unwrap().recv() {
                        Ok(j) => j,
                        Err(_) => return,
                    };
                    let r = run_job(&*dev, &m, &pool, &stats, job.index, job.buf, &job.cancel);
                    // A waiter that gave up (deadline) dropped its receiver: the buffer comes back via drop.
                    let _ = job.reply.send(r);
                })
            })
            .collect();
        Self {
            manifest,
            pool,
            tx: Some(tx),
            workers,
            stats,
        }
    }

    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    pub fn staging_capacity(&self) -> usize {
        self.pool.cap
    }

    pub fn stats(&self) -> ReadStats {
        self.stats.lock().unwrap().clone()
    }

    /// Queue a read of `manifest.experts[index]`. Refuses (QueueFull) when every staging buffer is in use.
    pub fn submit(&self, index: usize) -> Result<Ticket, ReadError> {
        if index >= self.manifest.experts.len() {
            return Err(ReadError::Manifest(format!("expert index {index}")));
        }
        let buf = match self.pool.bufs.lock().unwrap().pop() {
            Some(b) => b,
            None => {
                self.stats.lock().unwrap().queue_full += 1;
                return Err(ReadError::QueueFull);
            }
        };
        let (rtx, rrx) = mpsc::sync_channel(1);
        let cancel = Arc::new(AtomicBool::new(false));
        let job = Job {
            index,
            buf,
            cancel: cancel.clone(),
            reply: rtx,
        };
        let tx = self.tx.as_ref().ok_or(ReadError::Shutdown)?;
        if let Err(mpsc::SendError(j)) = tx.send(job) {
            self.pool.give_back(j.buf);
            return Err(ReadError::Shutdown);
        }
        Ok(Ticket { rx: rrx, cancel })
    }

    /// Submit, waiting (up to `wait`) for a free staging buffer instead of refusing.
    pub fn submit_wait(&self, index: usize, wait: Duration) -> Result<Ticket, ReadError> {
        let t0 = Instant::now();
        loop {
            match self.submit(index) {
                Err(ReadError::QueueFull) => {
                    let left = wait.saturating_sub(t0.elapsed());
                    if left.is_zero() {
                        return Err(ReadError::QueueFull);
                    }
                    let g = self.pool.bufs.lock().unwrap();
                    if g.is_empty() {
                        let _ = self.pool.returned.wait_timeout(g, left).unwrap();
                    }
                }
                r => return r,
            }
        }
    }
}

impl Drop for ReadAdapter {
    fn drop(&mut self) {
        self.tx.take();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn run_job(
    dev: &dyn ShardDevice,
    m: &Manifest,
    pool: &Arc<Pool>,
    stats: &Mutex<ReadStats>,
    index: usize,
    mut buf: AlignedBuf,
    cancel: &AtomicBool,
) -> Reply {
    let e = &m.experts[index];
    let fail = |buf: AlignedBuf, err: ReadError| {
        pool.give_back(buf);
        if err != ReadError::Cancelled {
            stats.lock().unwrap().errors += 1;
        }
        Err(err)
    };
    if cancel.load(Ordering::Acquire) {
        return fail(buf, ReadError::Cancelled);
    }
    let t0 = Instant::now();
    let (ranges, device) = match read_expert(dev, e, buf.as_mut_slice()) {
        Ok(v) => v,
        Err(err) => return fail(buf, err),
    };
    let read_ns = t0.elapsed().as_nanos() as u64;
    if cancel.load(Ordering::Acquire) {
        return fail(buf, ReadError::Cancelled);
    }
    let t1 = Instant::now();
    let expected = match e.digest_bytes() {
        Ok(d) => d,
        Err(err) => return fail(buf, ReadError::Manifest(err.to_string())),
    };
    let mut h = blake3::Hasher::new();
    for &(o, l) in &ranges {
        h.update(&buf.as_slice()[o..o + l]);
    }
    if h.finalize().as_bytes() != &expected {
        return fail(
            buf,
            ReadError::DigestMismatch {
                expert: (e.layer, e.expert),
            },
        );
    }
    {
        let mut s = stats.lock().unwrap();
        s.reads_ok += 1;
        s.payload_bytes += e.len;
        s.device_bytes += device;
        s.read_ns += read_ns;
        s.digest_ns += t1.elapsed().as_nanos() as u64;
    }
    Ok(VerifiedExpert {
        index,
        layer: e.layer,
        expert: e.expert,
        buf: Some(buf),
        ranges,
        pool: pool.clone(),
    })
}

/// Reads every segment of `e` into its own aligned window of `buf`. Returns the payload ranges and the device bytes.
pub fn read_expert(
    dev: &dyn ShardDevice,
    e: &ExpertEntry,
    buf: &mut [u8],
) -> Result<(Vec<(usize, usize)>, u64), ReadError> {
    let align = dev.align().max(1);
    let window_align = align.max(crate::IO_ALIGN);
    let mut ranges = Vec::with_capacity(e.segments.len());
    let (mut base, mut device) = (0usize, 0u64);
    for seg in &e.segments {
        let astart = round_down(seg.start, align as u64);
        let off = (seg.start - astart) as usize;
        let need = off + seg.len as usize;
        let window = round_up(need, align);
        if base + window > buf.len() {
            return Err(ReadError::TooLarge {
                need: base + window,
                cap: buf.len(),
            });
        }
        let w = &mut buf[base..base + window];
        let mut got = 0usize;
        while got < need {
            match dev.read_at(seg.shard, &mut w[got..], astart + got as u64) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(err) if err.kind() == std::io::ErrorKind::Interrupted => {}
                Err(err) => {
                    return Err(ReadError::Io(format!(
                        "read expert ({},{}): {err}",
                        e.layer, e.expert
                    )))
                }
            }
        }
        if got < need {
            return Err(ReadError::ShortRead {
                expert: (e.layer, e.expert),
                wanted: need,
                got,
            });
        }
        ranges.push((base + off, seg.len as usize));
        device += got as u64;
        base += round_up(window, window_align);
    }
    Ok((ranges, device))
}
