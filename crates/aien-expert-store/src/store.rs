//! The on-disk expert store.
//!
//! ```text
//! <root>/objects/<sha256 hex>   one artifact per file, named by its content address
//! <root>/index-v1               the committed index (see `format`)
//! <root>/tmp/                   private temporaries, removed on open
//! ```
//!
//! Crash safety: every file is written to a fresh temporary (`O_EXCL`, mode
//! 0600), synced, renamed into place and its directory synced. An object is
//! durable before `put` returns; the index is replaced atomically by
//! `commit`, so after a crash the store holds either the old or the new index
//! and every object either index names. Orphan objects are harmless.
//!
//! Bounded memory: the store holds only the index (`max_entries` entries);
//! `put` holds one artifact (`max_object_bytes`).

use crate::error::StoreError;
use crate::format::{
    build_artifact, decode_index, digest_eq, encode_index, hex, max_index_bytes, sha256, ExpertKey,
    IndexEntry, ScaleDtype,
};
use crate::source::{FileSource, IoMode};
use std::collections::BTreeMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

pub const INDEX_FILE: &str = "index-v1";
static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StoreLimits {
    pub max_entries: usize,
    pub max_object_bytes: u64,
}

impl Default for StoreLimits {
    fn default() -> Self {
        // One Qwen3-30B-A3B checkpoint: 48 layers x 128 experts x 3 projections.
        Self {
            max_entries: 48 * 128 * 3,
            max_object_bytes: 64 << 20,
        }
    }
}

pub struct ExpertStore {
    root: PathBuf,
    limits: StoreLimits,
    entries: BTreeMap<ExpertKey, IndexEntry>,
}

const OPEN_FLAGS: i32 = libc::O_NOFOLLOW | libc::O_CLOEXEC;

fn sync_dir(dir: &Path) -> Result<(), StoreError> {
    File::open(dir)?.sync_all()?;
    Ok(())
}

fn ensure_dir(dir: &Path) -> Result<(), StoreError> {
    match std::fs::DirBuilder::new().mode(0o700).create(dir) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
            if std::fs::symlink_metadata(dir)?.is_dir() {
                Ok(())
            } else {
                Err(StoreError::Io(std::io::ErrorKind::NotADirectory))
            }
        }
        Err(e) => Err(e.into()),
    }
}

/// Reads a whole file of at most `max` bytes without following a symlink.
fn read_bounded(path: &Path, max: u64) -> Result<Vec<u8>, StoreError> {
    let f = OpenOptions::new()
        .read(true)
        .custom_flags(OPEN_FLAGS)
        .open(path)?;
    let len = f.metadata()?.len();
    if len > max {
        return Err(StoreError::ObjectTooLarge { len, max });
    }
    let mut out = Vec::with_capacity(len as usize);
    f.take(max + 1).read_to_end(&mut out)?;
    if out.len() as u64 > max {
        return Err(StoreError::ObjectTooLarge {
            len: out.len() as u64,
            max,
        });
    }
    Ok(out)
}

impl ExpertStore {
    /// Opens (creating if needed) the store at `root`, removes temporaries
    /// left by a crash, and loads and validates the committed index.
    pub fn open(root: &Path, limits: StoreLimits) -> Result<Self, StoreError> {
        ensure_dir(root)?;
        ensure_dir(&root.join("objects"))?;
        let tmp = root.join("tmp");
        ensure_dir(&tmp)?;
        for e in std::fs::read_dir(&tmp)? {
            let e = e?;
            if e.file_type()?.is_file() {
                std::fs::remove_file(e.path())?;
            }
        }
        let mut entries = BTreeMap::new();
        match read_bounded(&root.join(INDEX_FILE), max_index_bytes(limits.max_entries)) {
            Ok(bytes) => {
                for e in decode_index(&bytes, limits.max_entries)? {
                    if e.object_len > limits.max_object_bytes {
                        return Err(StoreError::ObjectTooLarge {
                            len: e.object_len,
                            max: limits.max_object_bytes,
                        });
                    }
                    entries.insert(e.key, e);
                }
            }
            Err(StoreError::Io(std::io::ErrorKind::NotFound)) => {}
            Err(StoreError::ObjectTooLarge { .. }) => return Err(StoreError::TooManyEntries),
            Err(e) => return Err(e),
        }
        Ok(Self {
            root: root.to_path_buf(),
            limits,
            entries,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn limits(&self) -> StoreLimits {
        self.limits
    }

    pub fn objects_dir(&self) -> PathBuf {
        self.root.join("objects")
    }

    /// A read source over this store's objects.
    pub fn file_source(&self, mode: IoMode) -> FileSource {
        FileSource::new(self.objects_dir(), mode)
    }

    /// Entries in key order (committed plus staged by `put`).
    pub fn entries(&self) -> impl Iterator<Item = &IndexEntry> {
        self.entries.values()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Index lookup only, no I/O.
    pub fn get(&self, key: &ExpertKey) -> Option<IndexEntry> {
        self.entries.get(key).copied()
    }

    /// Writes one artifact durably and stages its index entry (persisted by
    /// [`ExpertStore::commit`]). Putting the same key with the same content is
    /// idempotent; the same key with other content is
    /// [`StoreError::DuplicateKey`].
    pub fn put(
        &mut self,
        key: ExpertKey,
        rows: u64,
        cols: u64,
        scale_dtype: ScaleDtype,
        weights: &[u8],
        scales: &[u8],
    ) -> Result<IndexEntry, StoreError> {
        let (hdr, bytes) = build_artifact(
            key,
            rows,
            cols,
            scale_dtype,
            weights,
            scales,
            self.limits.max_object_bytes,
        )?;
        let entry = IndexEntry {
            key,
            data_digest: sha256(&bytes),
            object_len: hdr.object_len,
        };
        match self.entries.get(&key) {
            Some(old) if old == &entry => {}
            Some(_) => return Err(StoreError::DuplicateKey(key)),
            None if self.entries.len() >= self.limits.max_entries => {
                return Err(StoreError::TooManyEntries)
            }
            None => {}
        }
        let name = hex(&entry.data_digest);
        let objects = self.objects_dir();
        // An existing object is reused only if it still hashes to its name;
        // a damaged one is replaced.
        let present = match read_bounded(&objects.join(&name), self.limits.max_object_bytes) {
            Ok(existing) => digest_eq(&sha256(&existing), &entry.data_digest),
            Err(StoreError::Io(std::io::ErrorKind::NotFound))
            | Err(StoreError::ObjectTooLarge { .. }) => false,
            Err(e) => return Err(e),
        };
        if !present {
            self.write_atomic(&objects, &name, &bytes)?;
        }
        self.entries.insert(key, entry);
        Ok(entry)
    }

    /// Atomically replaces the on-disk index with the current entries.
    pub fn commit(&self) -> Result<(), StoreError> {
        let entries: Vec<IndexEntry> = self.entries.values().copied().collect();
        let bytes = encode_index(&entries)?;
        self.write_atomic(&self.root, INDEX_FILE, &bytes)
    }

    fn write_atomic(&self, dir: &Path, name: &str, bytes: &[u8]) -> Result<(), StoreError> {
        let tmp = self.root.join("tmp").join(format!(
            "{name}.{}.{}",
            std::process::id(),
            TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
        ));
        let result = (|| {
            let mut f = OpenOptions::new()
                .write(true)
                .create_new(true)
                .mode(0o600)
                .custom_flags(OPEN_FLAGS)
                .open(&tmp)?;
            f.write_all(bytes)?;
            f.sync_all()?;
            drop(f);
            std::fs::rename(&tmp, dir.join(name))?;
            sync_dir(dir)
        })();
        if result.is_err() {
            let _ = std::fs::remove_file(&tmp);
        }
        result
    }
}
