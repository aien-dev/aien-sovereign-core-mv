//! The bounded asynchronous read adapter.
//!
//! Rules, in one place:
//!
//! * `submit` is non-blocking and bounded. It refuses (handing the staging
//!   buffer back) when the queue depth (`max_in_flight`), the in-flight byte
//!   budget (`max_inflight_bytes`), the object size bound
//!   (`max_object_bytes`) or the buffer capacity would be exceeded, when the
//!   deadline has already passed, or after shutdown.
//! * Every accepted read yields exactly one terminal [`ReadCompletion`]:
//!   `Ready`, `Failed` or `Cancelled`. The staging buffer travels back in it.
//! * A read is `Ready` only after the whole object was read, its SHA-256
//!   compared in constant time with the index digest, and its header matched
//!   against the requested key and length. Otherwise the buffer is scrubbed
//!   before it is returned; no partial or unverified bytes are ever visible.
//! * Cancellation and the deadline are checked before every chunk and once
//!   more before the read is marked ready. A single read call that never
//!   returns cannot be preempted; `chunk_bytes` bounds how long one call can
//!   delay detection.

use crate::error::ReadError;
use crate::format::{digest_eq, sha256, ArtifactHeader, IndexEntry, ALIGN, HEADER_REGION};
use crate::source::BlockSource;
use crate::staging::StagingBuffer;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver, Sender};
use std::sync::{Arc, Condvar, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// Explicit bounds of one adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ReadLimits {
    /// Queue depth: reads queued or running at once (also the worker count).
    pub max_in_flight: usize,
    /// Sum of the object lengths of reads queued or running.
    pub max_inflight_bytes: u64,
    /// Largest object a single read may fetch.
    pub max_object_bytes: u64,
    /// Bytes per read call; a 4096 multiple so direct reads stay aligned.
    pub chunk_bytes: usize,
}

impl Default for ReadLimits {
    fn default() -> Self {
        Self {
            max_in_flight: 4,
            max_inflight_bytes: 256 << 20,
            max_object_bytes: 64 << 20,
            chunk_bytes: 1 << 20,
        }
    }
}

/// Never reused within one adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticket(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CancelOutcome {
    /// The read will end `Cancelled` unless it was already verified.
    Requested,
    /// No such read is in flight (unknown, or its terminal was already sent).
    Unknown,
}

/// A verified artifact in its staging buffer.
#[derive(Debug)]
pub struct ReadyExpert {
    pub header: ArtifactHeader,
    pub entry: IndexEntry,
    buf: StagingBuffer,
}

impl ReadyExpert {
    /// The FP8 weight bytes, exactly as in the source checkpoint.
    pub fn weights(&self) -> &[u8] {
        &self.buf.as_slice()[self.header.weight_range()]
    }

    /// The scale bytes, exactly as in the source checkpoint.
    pub fn scales(&self) -> &[u8] {
        &self.buf.as_slice()[self.header.scale_range()]
    }

    /// The whole verified artifact.
    pub fn artifact(&self) -> &[u8] {
        &self.buf.as_slice()[..self.header.object_len as usize]
    }

    /// Gives the buffer back for reuse (it still holds the verified bytes).
    pub fn into_buffer(self) -> StagingBuffer {
        self.buf
    }
}

#[derive(Debug)]
pub enum Outcome {
    Ready(ReadyExpert),
    /// The buffer has been scrubbed.
    Failed {
        error: ReadError,
        buf: StagingBuffer,
    },
    /// The buffer has been scrubbed.
    Cancelled {
        buf: StagingBuffer,
    },
}

#[derive(Debug)]
pub struct ReadCompletion {
    pub ticket: Ticket,
    pub outcome: Outcome,
    pub elapsed: Duration,
}

/// A refused submit: the reason and the caller's buffer, untouched.
#[derive(Debug)]
pub struct Refused {
    pub error: ReadError,
    pub buf: StagingBuffer,
}

struct Job {
    ticket: Ticket,
    entry: IndexEntry,
    deadline: Instant,
    buf: StagingBuffer,
    cancel: Arc<AtomicBool>,
    submitted: Instant,
}

#[derive(Default)]
struct State {
    queue: VecDeque<Job>,
    in_flight: usize,
    in_flight_bytes: u64,
    next_ticket: u64,
    cancels: HashMap<u64, Arc<AtomicBool>>,
    shutdown: bool,
}

struct Shared {
    limits: ReadLimits,
    source: Arc<dyn BlockSource>,
    state: Mutex<State>,
    work: Condvar,
}

pub struct ReadAdapter {
    shared: Arc<Shared>,
    workers: Vec<JoinHandle<()>>,
    completions: Mutex<Receiver<ReadCompletion>>,
}

impl ReadAdapter {
    /// Starts `limits.max_in_flight` workers over `source`. Limits that are
    /// zero, or a `chunk_bytes` that is not a 4096 multiple, are refused.
    pub fn new(source: Arc<dyn BlockSource>, limits: ReadLimits) -> Result<Self, ReadError> {
        if limits.max_in_flight == 0
            || limits.max_inflight_bytes == 0
            || limits.max_object_bytes == 0
            || limits.chunk_bytes == 0
            || !(limits.chunk_bytes as u64).is_multiple_of(ALIGN)
        {
            return Err(ReadError::InvalidLimits);
        }
        let shared = Arc::new(Shared {
            limits,
            source,
            state: Mutex::new(State::default()),
            work: Condvar::new(),
        });
        let (tx, rx) = channel();
        let workers = (0..limits.max_in_flight)
            .map(|_| {
                let shared = Arc::clone(&shared);
                let tx: Sender<ReadCompletion> = tx.clone();
                std::thread::spawn(move || worker(shared, tx))
            })
            .collect();
        Ok(Self {
            shared,
            workers,
            completions: Mutex::new(rx),
        })
    }

    pub fn limits(&self) -> ReadLimits {
        self.shared.limits
    }

    /// Reads queued or running, and their bytes.
    pub fn in_flight(&self) -> (usize, u64) {
        let st = self.shared.state.lock().unwrap();
        (st.in_flight, st.in_flight_bytes)
    }

    /// Queues one verified read of `entry` into `buf`. Non-blocking.
    pub fn submit(
        &self,
        entry: IndexEntry,
        buf: StagingBuffer,
        deadline: Instant,
    ) -> Result<Ticket, Refused> {
        let l = &self.shared.limits;
        let refuse = |error, buf| Err(Refused { error, buf });
        let len = entry.object_len;
        if len <= HEADER_REGION as u64 || !len.is_multiple_of(ALIGN) {
            return refuse(ReadError::InvalidEntry, buf);
        }
        if len > l.max_object_bytes {
            return refuse(ReadError::ObjectTooLarge, buf);
        }
        if len > buf.capacity() as u64 {
            return refuse(
                ReadError::InsufficientCapacity {
                    need: len,
                    have: buf.capacity() as u64,
                },
                buf,
            );
        }
        let now = Instant::now();
        if deadline <= now {
            return refuse(ReadError::DeadlineExceeded, buf);
        }
        let mut st = self.shared.state.lock().unwrap();
        if st.shutdown {
            return refuse(ReadError::ShutDown, buf);
        }
        if st.in_flight >= l.max_in_flight {
            return refuse(ReadError::QueueFull, buf);
        }
        if st.in_flight_bytes.saturating_add(len) > l.max_inflight_bytes {
            return refuse(ReadError::InflightBytesExceeded, buf);
        }
        let ticket = Ticket(st.next_ticket);
        st.next_ticket += 1;
        st.in_flight += 1;
        st.in_flight_bytes += len;
        let cancel = Arc::new(AtomicBool::new(false));
        st.cancels.insert(ticket.0, Arc::clone(&cancel));
        st.queue.push_back(Job {
            ticket,
            entry,
            deadline,
            buf,
            cancel,
            submitted: now,
        });
        drop(st);
        self.shared.work.notify_one();
        Ok(ticket)
    }

    /// Requests cancellation. Returns immediately; the read still yields its
    /// one terminal completion.
    pub fn cancel(&self, ticket: Ticket) -> CancelOutcome {
        let st = self.shared.state.lock().unwrap();
        match st.cancels.get(&ticket.0) {
            Some(flag) => {
                flag.store(true, Ordering::SeqCst);
                CancelOutcome::Requested
            }
            None => CancelOutcome::Unknown,
        }
    }

    /// Non-blocking: at most `max` completions, oldest first.
    pub fn poll(&self, max: usize) -> Vec<ReadCompletion> {
        let rx = self.completions.lock().unwrap();
        let mut out = Vec::new();
        while out.len() < max {
            match rx.try_recv() {
                Ok(c) => out.push(c),
                Err(_) => break,
            }
        }
        out
    }

    /// Blocks up to `timeout` for the next completion.
    pub fn wait(&self, timeout: Duration) -> Option<ReadCompletion> {
        let rx = self.completions.lock().unwrap();
        rx.recv_timeout(timeout).ok()
    }
}

impl Drop for ReadAdapter {
    /// Stops intake, cancels every queued and running read and joins the workers.
    fn drop(&mut self) {
        {
            let mut st = self.shared.state.lock().unwrap();
            st.shutdown = true;
            for flag in st.cancels.values() {
                flag.store(true, Ordering::SeqCst);
            }
        }
        self.shared.work.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

fn worker(shared: Arc<Shared>, tx: Sender<ReadCompletion>) {
    loop {
        let job = {
            let mut st = shared.state.lock().unwrap();
            loop {
                if let Some(job) = st.queue.pop_front() {
                    break job;
                }
                if st.shutdown {
                    return;
                }
                st = shared.work.wait(st).unwrap();
            }
        };
        let ticket = job.ticket;
        let len = job.entry.object_len;
        let submitted = job.submitted;
        let outcome = run(&shared, job);
        {
            let mut st = shared.state.lock().unwrap();
            st.in_flight -= 1;
            st.in_flight_bytes -= len;
            st.cancels.remove(&ticket.0);
        }
        // A dropped receiver only means nobody is listening any more.
        let _ = tx.send(ReadCompletion {
            ticket,
            outcome,
            elapsed: submitted.elapsed(),
        });
    }
}

enum Stop {
    Error(ReadError),
    Cancelled,
}

fn run(shared: &Shared, job: Job) -> Outcome {
    let Job {
        entry,
        deadline,
        mut buf,
        cancel,
        ..
    } = job;
    match fill_and_verify(shared, &entry, deadline, &cancel, &mut buf) {
        Ok(header) => Outcome::Ready(ReadyExpert { header, entry, buf }),
        Err(stop) => {
            buf.scrub();
            match stop {
                Stop::Cancelled => Outcome::Cancelled { buf },
                Stop::Error(error) => Outcome::Failed { error, buf },
            }
        }
    }
}

fn checkpoint(deadline: Instant, cancel: &AtomicBool) -> Result<(), Stop> {
    if cancel.load(Ordering::SeqCst) {
        return Err(Stop::Cancelled);
    }
    if Instant::now() >= deadline {
        return Err(Stop::Error(ReadError::DeadlineExceeded));
    }
    Ok(())
}

fn fill_and_verify(
    shared: &Shared,
    entry: &IndexEntry,
    deadline: Instant,
    cancel: &AtomicBool,
    buf: &mut StagingBuffer,
) -> Result<ArtifactHeader, Stop> {
    let e = Stop::Error;
    checkpoint(deadline, cancel)?;
    let expected = entry.object_len;
    let mut reader = shared.source.open(&entry.data_digest).map_err(e)?;
    let found = reader.size().map_err(e)?;
    if found != expected {
        return Err(e(ReadError::LengthMismatch { expected, found }));
    }
    let n = expected as usize;
    let dst = &mut buf.as_mut_slice()[..n];
    let mut done = 0usize;
    while done < n {
        checkpoint(deadline, cancel)?;
        let end = (done + shared.limits.chunk_bytes).min(n);
        let got = reader
            .read_at(done as u64, &mut dst[done..end])
            .map_err(e)?;
        if got == 0 {
            return Err(e(ReadError::UnexpectedEof {
                read: done as u64,
                expected,
            }));
        }
        done += got.min(end - done);
    }
    if !digest_eq(&sha256(dst), &entry.data_digest) {
        return Err(e(ReadError::DigestMismatch));
    }
    let header = ArtifactHeader::decode(dst).map_err(|_| e(ReadError::HeaderMismatch))?;
    if header.key != entry.key || header.object_len != expected {
        return Err(e(ReadError::HeaderMismatch));
    }
    checkpoint(deadline, cancel)?;
    Ok(header)
}
