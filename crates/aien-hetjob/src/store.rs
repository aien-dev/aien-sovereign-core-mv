//! The worker's state folder: signing key, an exclusive lock and the durable
//! results that make a repeated request idempotent.
//!
//! A result is keyed by the idempotence key and bound to the task id and the
//! request digest inside its receipt (PST1 section 8: same key and same task
//! is the same request). It is written to a temporary file, synced, renamed
//! and the folder synced, before the worker sends `completed`.

use crate::receipt::{Receipt, ReceiptError};
use aien_test::bundle::hex_encode;
use ed25519_dalek::VerifyingKey;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::os::unix::io::AsRawFd;
use std::path::{Path, PathBuf};

pub const KEY_FILE: &str = "worker.key";
const LOCK_FILE: &str = "LOCK";
const RESULTS: &str = "results";

/// An open state folder. Holding it holds the lock.
pub struct Store {
    dir: PathBuf,
    _lock: File,
}

/// What a lookup found under one idempotence key.
pub enum Lookup {
    Missing,
    /// Same task id and request digest: the stored receipt bytes.
    Same(Vec<u8>),
    /// The key is already bound to a different request.
    Conflict,
}

impl Store {
    /// Open (creating if needed) and take the exclusive lock. A second
    /// opener, in this or another process, gets `WouldBlock`.
    pub fn open(dir: &Path) -> io::Result<Store> {
        fs::create_dir_all(dir.join(RESULTS))?;
        let lock = OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(dir.join(LOCK_FILE))?;
        // SAFETY: flock on a file descriptor we own; no memory is shared.
        let rc = unsafe { libc::flock(lock.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
        if rc != 0 {
            return Err(io::Error::new(
                io::ErrorKind::WouldBlock,
                "another worker holds this state folder",
            ));
        }
        Ok(Store {
            dir: dir.to_path_buf(),
            _lock: lock,
        })
    }

    pub fn key_path(&self) -> PathBuf {
        self.dir.join(KEY_FILE)
    }

    fn result_path(&self, idem: &[u8; 16]) -> PathBuf {
        self.dir
            .join(RESULTS)
            .join(format!("{}.receipt", hex_encode(idem)))
    }

    /// Look up a stored result. The stored receipt is re-verified against the
    /// worker's own key, so a damaged file is never replayed.
    pub fn lookup(
        &self,
        idem: &[u8; 16],
        task_id: &[u8; 16],
        input_digest: &[u8; 32],
        own_key: &VerifyingKey,
    ) -> io::Result<Lookup> {
        let bytes = match fs::read(self.result_path(idem)) {
            Ok(b) => b,
            Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(Lookup::Missing),
            Err(e) => return Err(e),
        };
        let (r, _) =
            Receipt::verify(&bytes, std::slice::from_ref(own_key)).map_err(|e: ReceiptError| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("stored receipt refused: {}", e.code()),
                )
            })?;
        if &r.task_id == task_id && &r.input_digest == input_digest {
            Ok(Lookup::Same(bytes))
        } else {
            Ok(Lookup::Conflict)
        }
    }

    /// Store a receipt durably under its idempotence key.
    pub fn put(&self, idem: &[u8; 16], receipt: &[u8]) -> io::Result<()> {
        let path = self.result_path(idem);
        let tmp = path.with_extension("tmp");
        {
            let mut f = File::create(&tmp)?;
            f.write_all(receipt)?;
            f.sync_all()?;
        }
        fs::rename(&tmp, &path)?;
        File::open(self.dir.join(RESULTS))?.sync_all()
    }
}
