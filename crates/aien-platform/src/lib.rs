//! Pure no_std platform traits and capability abstractions for the AIEN and SparkOS runtime.
//! Eliminates POSIX/Linux concepts (PathBuf, Command, sockets, threads, procfs) from core cognition and scheduling.

#![no_std]

extern crate alloc;

use alloc::string::String;

pub mod compute;
pub mod memory;
pub mod objects;
pub mod queue;
pub mod runtime;
pub mod task;
pub mod telemetry;

pub use compute::{
    BufferLayout, BufferRegion, ComputeDevice, ComputeWork, DeviceAddress, Fence, MemoryDevice,
    UnifiedBuffer,
};
pub use memory::{MemoryEntity, MemoryService, RecallQuery, RecallResult};
pub use objects::{ObjectId, ObjectRef, ObjectStore, ObjectWrite};
pub use queue::{
    InferenceWork, KvHandle, ModelHandle, Priority, QueueError, SequenceId, Ticks, WorkQueue,
};
pub use runtime::Runtime;
pub use task::{TaskExecutor, TaskRequest, TaskResult};
pub use telemetry::{TelemetrySnapshot, TelemetrySource};

/// Fundamental platform error enumeration free of host OS dependencies.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PlatformError {
    OutOfMemory,
    DeviceUnavailable,
    InvalidLayout,
    QueueFull,
    QueueEmpty,
    TaskRejected(String),
    ObjectNotFound(ObjectId),
    AccessDenied,
    HardwareFault(String),
    UnsupportedOperation,
}

impl core::fmt::Display for PlatformError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::OutOfMemory => write!(f, "Platform error: out of physical/unified memory"),
            Self::DeviceUnavailable => write!(f, "Platform error: compute device unavailable"),
            Self::InvalidLayout => write!(f, "Platform error: invalid buffer layout or alignment"),
            Self::QueueFull => write!(f, "Platform error: work queue capacity exceeded"),
            Self::QueueEmpty => write!(f, "Platform error: work queue empty"),
            Self::TaskRejected(msg) => {
                write!(f, "Platform error: task execution rejected: {}", msg)
            }
            Self::ObjectNotFound(id) => write!(f, "Platform error: object {:?} not found", id),
            Self::AccessDenied => write!(f, "Platform error: capability access denied"),
            Self::HardwareFault(msg) => write!(f, "Platform error: hardware fault: {}", msg),
            Self::UnsupportedOperation => write!(
                f,
                "Platform error: unsupported operation on target substrate"
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::string::ToString;

    #[test]
    fn test_platform_error_display() {
        assert_eq!(
            PlatformError::OutOfMemory.to_string(),
            "Platform error: out of physical/unified memory"
        );
        assert_eq!(
            PlatformError::DeviceUnavailable.to_string(),
            "Platform error: compute device unavailable"
        );
        assert_eq!(
            PlatformError::QueueFull.to_string(),
            "Platform error: work queue capacity exceeded"
        );
        assert_eq!(
            PlatformError::AccessDenied.to_string(),
            "Platform error: capability access denied"
        );
        assert_eq!(
            PlatformError::TaskRejected("timeout".into()).to_string(),
            "Platform error: task execution rejected: timeout"
        );
    }

    #[test]
    fn test_queue_priority_ordering() {
        assert!(Priority::Realtime > Priority::Interactive);
        assert!(Priority::Interactive > Priority::Normal);
        assert!(Priority::Normal > Priority::Background);
    }

    #[test]
    fn test_buffer_layout() {
        let layout = BufferLayout::new(4096, 128);
        assert_eq!(layout.size, 4096);
        assert_eq!(layout.align, 128);
    }

    #[test]
    fn test_telemetry_snapshot() {
        let snap = TelemetrySnapshot {
            architecture: "aarch64".into(),
            target: "gb10".into(),
            clock_ticks: 1_000_000,
            memory_used_bytes: 1024 * 1024 * 100,
            memory_total_bytes: 1024 * 1024 * 1024 * 128,
            active_sequences: 8,
        };
        assert_eq!(snap.architecture, "aarch64");
        assert_eq!(snap.target, "gb10");
        assert_eq!(snap.active_sequences, 8);
    }
}
