//! A tiny deterministic in-memory [`BlockSource`] with injectable faults, for
//! concurrency, latency, failure, corruption and cancellation tests. No
//! randomness, no device access.

use crate::error::ReadError;
use crate::source::{BlockReader, BlockSource};
use std::collections::HashMap;
use std::io::ErrorKind;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::Duration;

/// A fault applied to every read of one object.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Fault {
    /// XOR 0x01 into the byte at this offset as it is read.
    CorruptByte(u64),
    /// The object is this many bytes long (length and data both shortened).
    Truncate(u64),
    /// `len` reports the full length but reads return end of file from this offset.
    EofAt(u64),
    /// Each read returns at most this many bytes.
    ShortReads(usize),
    /// Every read fails with this error kind.
    IoError(ErrorKind),
    /// Each read sleeps this long first (a slow or stalled device).
    Delay(Duration),
}

#[derive(Default)]
struct State {
    objects: HashMap<[u8; 32], Arc<Vec<u8>>>,
    faults: HashMap<[u8; 32], Fault>,
    held: bool,
}

#[derive(Default)]
struct Inner {
    state: Mutex<State>,
    gate: Condvar,
    active: AtomicUsize,
    peak: AtomicUsize,
    reads: AtomicUsize,
}

/// Cloning shares the same objects, faults and counters.
#[derive(Clone, Default)]
pub struct FakeSource(Arc<Inner>);

impl FakeSource {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&self, digest: [u8; 32], bytes: Vec<u8>) {
        self.0
            .state
            .lock()
            .unwrap()
            .objects
            .insert(digest, Arc::new(bytes));
    }

    pub fn set_fault(&self, digest: [u8; 32], fault: Fault) {
        self.0.state.lock().unwrap().faults.insert(digest, fault);
    }

    pub fn clear_fault(&self, digest: &[u8; 32]) {
        self.0.state.lock().unwrap().faults.remove(digest);
    }

    /// Blocks every read at entry until [`FakeSource::release`].
    pub fn hold(&self) {
        self.0.state.lock().unwrap().held = true;
    }

    pub fn release(&self) {
        self.0.state.lock().unwrap().held = false;
        self.0.gate.notify_all();
    }

    /// Reads currently inside `read_at`.
    pub fn active(&self) -> usize {
        self.0.active.load(Ordering::SeqCst)
    }

    /// Most reads ever inside `read_at` at once.
    pub fn peak(&self) -> usize {
        self.0.peak.load(Ordering::SeqCst)
    }

    /// Total `read_at` calls.
    pub fn reads(&self) -> usize {
        self.0.reads.load(Ordering::SeqCst)
    }
}

struct FakeReader {
    inner: Arc<Inner>,
    bytes: Arc<Vec<u8>>,
    fault: Option<Fault>,
}

struct ActiveGuard<'a>(&'a Inner);

impl Drop for ActiveGuard<'_> {
    fn drop(&mut self) {
        self.0.active.fetch_sub(1, Ordering::SeqCst);
    }
}

impl BlockReader for FakeReader {
    fn size(&self) -> Result<u64, ReadError> {
        let full = self.bytes.len() as u64;
        Ok(match self.fault {
            Some(Fault::Truncate(n)) => n.min(full),
            _ => full,
        })
    }

    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<usize, ReadError> {
        self.inner.reads.fetch_add(1, Ordering::SeqCst);
        let now = self.inner.active.fetch_add(1, Ordering::SeqCst) + 1;
        self.inner.peak.fetch_max(now, Ordering::SeqCst);
        let _active = ActiveGuard(&self.inner);
        {
            let mut st = self.inner.state.lock().unwrap();
            while st.held {
                st = self.inner.gate.wait(st).unwrap();
            }
        }
        let mut end = self.bytes.len() as u64;
        let mut max = dst.len();
        match self.fault {
            Some(Fault::IoError(k)) => return Err(ReadError::Io(k)),
            Some(Fault::Delay(d)) => std::thread::sleep(d),
            Some(Fault::Truncate(n)) | Some(Fault::EofAt(n)) => end = end.min(n),
            Some(Fault::ShortReads(n)) => max = max.min(n.max(1)),
            Some(Fault::CorruptByte(_)) | None => {}
        }
        if offset >= end {
            return Ok(0);
        }
        let n = ((end - offset) as usize).min(max);
        let start = offset as usize;
        dst[..n].copy_from_slice(&self.bytes[start..start + n]);
        if let Some(Fault::CorruptByte(at)) = self.fault {
            if at >= offset && at < offset + n as u64 {
                dst[(at - offset) as usize] ^= 0x01;
            }
        }
        Ok(n)
    }
}

impl BlockSource for FakeSource {
    fn open(&self, digest: &[u8; 32]) -> Result<Box<dyn BlockReader>, ReadError> {
        let st = self.0.state.lock().unwrap();
        let bytes = st.objects.get(digest).cloned().ok_or(ReadError::NotFound)?;
        let fault = st.faults.get(digest).copied();
        Ok(Box::new(FakeReader {
            inner: Arc::clone(&self.0),
            bytes,
            fault,
        }))
    }
}
