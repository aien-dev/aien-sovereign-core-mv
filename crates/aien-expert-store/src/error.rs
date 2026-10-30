//! Structured errors. [`StoreError`] covers the store, the formats and the
//! indexer; [`ReadError`] covers the read adapter and is a plain value.

use crate::format::ExpertKey;
use std::io::ErrorKind;

/// Why a store, index, artifact or checkpoint operation was refused.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum StoreError {
    /// An I/O call failed (the kind only, no path, so nothing private leaks).
    Io(ErrorKind),
    BadMagic,
    UnsupportedVersion,
    /// An artifact header field is out of range or disagrees with the others.
    BadHeader(&'static str),
    /// The index file is malformed (size, ordering, reserved bytes, enums).
    BadIndex(&'static str),
    /// The index trailer digest does not match its contents.
    IndexChecksum,
    /// A key is already bound to different content.
    DuplicateKey(ExpertKey),
    /// Weight or scale bytes do not have the length the geometry requires.
    PayloadLength {
        expected: u64,
        found: u64,
    },
    /// More entries than `StoreLimits::max_entries`.
    TooManyEntries,
    /// An artifact larger than `StoreLimits::max_object_bytes`.
    ObjectTooLarge {
        len: u64,
        max: u64,
    },
    /// An existing object file does not hash to its name.
    ObjectCorrupt,
    /// A size computation overflowed.
    Overflow,
    /// The store root would be inside the source checkpoint directory.
    StoreInsideCheckpoint,
    /// The checkpoint index or a shard header is malformed.
    Checkpoint(String),
    /// A tensor the index or the expert layout requires is missing.
    MissingTensor(String),
    /// A tensor has the wrong dtype.
    DtypeMismatch {
        name: String,
        found: String,
    },
    /// A tensor has the wrong shape.
    ShapeMismatch {
        name: String,
    },
}

impl From<std::io::Error> for StoreError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e.kind())
    }
}

impl core::fmt::Display for StoreError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "expert store: {self:?}")
    }
}

impl std::error::Error for StoreError {}

/// Why a read was refused at submit or failed before it became ready.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadError {
    /// `max_in_flight` jobs are already queued or running.
    QueueFull,
    /// Admitting this read would exceed `max_inflight_bytes`.
    InflightBytesExceeded,
    /// The object is larger than `max_object_bytes`.
    ObjectTooLarge,
    /// The staging buffer is smaller than the object.
    InsufficientCapacity { need: u64, have: u64 },
    /// The index entry cannot describe a valid artifact (length not aligned or
    /// smaller than the header region).
    InvalidEntry,
    /// The deadline passed before submit, or before the read finished.
    DeadlineExceeded,
    /// The adapter is shutting down.
    ShutDown,
    /// No object with that digest exists in the source.
    NotFound,
    /// An I/O call failed.
    Io(ErrorKind),
    /// The object has a different length than the index records (truncated or
    /// extended file).
    LengthMismatch { expected: u64, found: u64 },
    /// The source returned end of file before the object was complete.
    UnexpectedEof { read: u64, expected: u64 },
    /// SHA-256 of the bytes read differs from the index digest.
    DigestMismatch,
    /// The bytes hash correctly but their header does not describe the
    /// requested key and length.
    HeaderMismatch,
    /// Adapter limits that are zero, or a chunk size that is not a 4096 multiple.
    InvalidLimits,
    /// Direct I/O is not available on this platform or file system.
    DirectIoUnsupported,
}

impl From<std::io::Error> for ReadError {
    fn from(e: std::io::Error) -> Self {
        match e.kind() {
            ErrorKind::NotFound => Self::NotFound,
            k => Self::Io(k),
        }
    }
}

impl core::fmt::Display for ReadError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "expert read: {self:?}")
    }
}

impl std::error::Error for ReadError {}
