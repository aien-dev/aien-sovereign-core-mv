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
use sha2::{Digest, Sha256};
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

    pub fn dir(&self) -> &Path {
        &self.dir
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

const BLOBS: &str = "blobs";

/// Where the input file with this sha256 lives under a state folder. Input
/// files are content addressed: the name is the digest of the bytes.
pub fn blob_path(state: &Path, digest: &[u8; 32]) -> PathBuf {
    state.join(BLOBS).join(hex_encode(digest))
}

/// Copy bytes into the state folder's blob store and return their sha256.
/// Needs no lock (the name is the content, written to a temporary file and
/// renamed), so it works while a worker serves the same folder.
pub fn put_blob(state: &Path, bytes: &[u8]) -> io::Result<[u8; 32]> {
    let digest: [u8; 32] = Sha256::digest(bytes).into();
    let path = blob_path(state, &digest);
    if path.is_file() {
        return Ok(digest);
    }
    fs::create_dir_all(state.join(BLOBS))?;
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    fs::rename(&tmp, &path)?;
    File::open(state.join(BLOBS))?.sync_all()?;
    Ok(digest)
}

/// Why an input file could not be used.
#[derive(Debug)]
pub enum BlobError {
    Missing,
    /// The file's bytes do not hash to its name.
    DigestMismatch,
    Io(io::Error),
}

impl BlobError {
    pub fn code(&self, digest: &[u8; 32]) -> String {
        let d = hex_encode(digest);
        match self {
            BlobError::Missing => format!("MISSING_INPUT:{d}"),
            BlobError::DigestMismatch => format!("INPUT_DIGEST_MISMATCH:{d}"),
            BlobError::Io(e) => format!("INPUT_IO:{d}:{e}"),
        }
    }
}

/// Size of a stored input file, for admission before it is read.
pub fn blob_size(state: &Path, digest: &[u8; 32]) -> Result<u64, BlobError> {
    match fs::metadata(blob_path(state, digest)) {
        Ok(m) => Ok(m.len()),
        Err(e) if e.kind() == io::ErrorKind::NotFound => Err(BlobError::Missing),
        Err(e) => Err(BlobError::Io(e)),
    }
}

/// Read a stored input file and check its bytes against its digest.
pub fn load_blob(state: &Path, digest: &[u8; 32]) -> Result<Vec<u8>, BlobError> {
    let bytes = match fs::read(blob_path(state, digest)) {
        Ok(b) => b,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(BlobError::Missing),
        Err(e) => return Err(BlobError::Io(e)),
    };
    if <[u8; 32]>::from(Sha256::digest(&bytes)) != *digest {
        return Err(BlobError::DigestMismatch);
    }
    Ok(bytes)
}
