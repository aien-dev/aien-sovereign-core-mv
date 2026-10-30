//! Byte sources behind the read adapter. Buffered and direct file reads share
//! one interface so the mode can be chosen by measurement (sc#629 open item:
//! the GB10 read-only receipt), not assumed.

use crate::error::ReadError;
use crate::format::hex;
use std::fs::File;
use std::os::unix::fs::{FileExt, OpenOptionsExt};
use std::path::PathBuf;

/// Opens objects by content address.
pub trait BlockSource: Send + Sync + 'static {
    fn open(&self, digest: &[u8; 32]) -> Result<Box<dyn BlockReader>, ReadError>;
}

/// One open object. `size` is the object length in bytes. `read_at` may return fewer bytes than asked (a short read);
/// `Ok(0)` means end of file.
pub trait BlockReader: Send {
    fn size(&self) -> Result<u64, ReadError>;
    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<usize, ReadError>;
}

/// How file objects are read.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IoMode {
    /// `pread` through the page cache.
    Buffered,
    /// `O_DIRECT` (Linux): offsets, lengths and the staging buffer are all
    /// 4096-aligned by construction. Refused elsewhere, never silently
    /// downgraded to buffered.
    Direct,
}

/// Objects under `<store>/objects/<hex digest>`.
#[derive(Debug, Clone)]
pub struct FileSource {
    objects_dir: PathBuf,
    mode: IoMode,
}

impl FileSource {
    pub fn new(objects_dir: PathBuf, mode: IoMode) -> Self {
        Self { objects_dir, mode }
    }

    pub fn mode(&self) -> IoMode {
        self.mode
    }
}

struct FileReader(File);

impl BlockReader for FileReader {
    fn size(&self) -> Result<u64, ReadError> {
        Ok(self.0.metadata()?.len())
    }

    fn read_at(&mut self, offset: u64, dst: &mut [u8]) -> Result<usize, ReadError> {
        loop {
            match self.0.read_at(dst, offset) {
                Ok(n) => return Ok(n),
                Err(e) if e.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e.into()),
            }
        }
    }
}

impl BlockSource for FileSource {
    fn open(&self, digest: &[u8; 32]) -> Result<Box<dyn BlockReader>, ReadError> {
        let base = libc::O_NOFOLLOW | libc::O_CLOEXEC;
        let flags = match self.mode {
            IoMode::Buffered => base,
            #[cfg(target_os = "linux")]
            IoMode::Direct => base | libc::O_DIRECT,
            #[cfg(not(target_os = "linux"))]
            IoMode::Direct => return Err(ReadError::DirectIoUnsupported),
        };
        let file = std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(flags)
            .open(self.objects_dir.join(hex(digest)))
            .map_err(|e| match e.raw_os_error() {
                Some(libc::EINVAL) if self.mode == IoMode::Direct => ReadError::DirectIoUnsupported,
                _ => ReadError::from(e),
            })?;
        Ok(Box::new(FileReader(file)))
    }
}
