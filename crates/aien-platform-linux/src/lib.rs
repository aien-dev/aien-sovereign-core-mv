//! Linux and Grace Blackwell GB10 platform driver cell for the AIEN runtime.

pub mod buffer;
pub mod device;
pub mod queue;
pub mod telemetry;

pub use buffer::{LinuxMemoryKind, LinuxUnifiedBuffer, ResidencyPolicy};
pub use device::{LinuxComputeDevice, LinuxCudaCapabilities};
pub use queue::LinuxWorkQueue;
pub use telemetry::LinuxTelemetrySource;

#[cfg(test)]
mod tests {
    use super::*;
    use aien_platform::{
        BufferLayout, ComputeDevice, InferenceWork, KvHandle, ModelHandle, Priority,
        TelemetrySource, UnifiedBuffer, WorkQueue,
    };

    #[test]
    fn test_linux_unified_buffer_allocation_and_read_write() {
        let layout = BufferLayout::new(65536, 128);
        let mut buf = LinuxUnifiedBuffer::allocate(
            layout,
            LinuxMemoryKind::AtsSystem,
            ResidencyPolicy::FaultIn,
        )
        .expect("Failed to allocate LinuxUnifiedBuffer");

        assert_eq!(buf.len(), 65536);
        assert!(buf.device_address().0 != 0);
        assert_eq!(buf.device_address().0 % 128, 0);

        unsafe {
            let p = buf.as_mut_ptr();
            *p = 42;
            *p.add(65535) = 99;
            assert_eq!(*buf.as_ptr(), 42);
            assert_eq!(*buf.as_ptr().add(65535), 99);
        }
    }

    #[test]
    fn test_linux_compute_device_allocation() {
        let device = LinuxComputeDevice::new();
        let layout = BufferLayout::new(1024 * 1024, 256);
        let buf = device
            .alloc(layout)
            .expect("Failed to allocate from LinuxComputeDevice");

        assert_eq!(buf.len(), 1024 * 1024);
        assert_eq!(buf.device_address().0 % 256, 0);
    }

    #[test]
    fn test_linux_memory_device_copy_and_zero() {
        use aien_platform::{BufferRegion, MemoryDevice};

        let device = LinuxComputeDevice::new();
        let layout = BufferLayout::new(4096, 128);
        let mut buf_src = device.alloc(layout).expect("alloc src");
        let mut buf_dst = device.alloc(layout).expect("alloc dst");

        unsafe {
            for i in 0..4096 {
                *buf_src.as_mut_ptr().add(i) = (i % 251) as u8;
                *buf_dst.as_mut_ptr().add(i) = 0xFF;
            }
        }

        let src_region = BufferRegion {
            address: buf_src.device_address(),
            offset: 0,
            len: 4096,
        };
        let dst_region = BufferRegion {
            address: buf_dst.device_address(),
            offset: 0,
            len: 4096,
        };

        let fence = device.copy(src_region, dst_region).expect("device copy");
        device.synchronize(fence).expect("synchronize copy");

        unsafe {
            for i in 0..4096 {
                assert_eq!(
                    *buf_dst.as_ptr().add(i),
                    (i % 251) as u8,
                    "copy mismatch at index {}",
                    i
                );
            }
        }

        let fence_zero = device.zero(dst_region).expect("device zero");
        device.synchronize(fence_zero).expect("synchronize zero");

        unsafe {
            for i in 0..4096 {
                assert_eq!(*buf_dst.as_ptr().add(i), 0, "zero mismatch at index {}", i);
            }
        }
    }

    #[test]
    fn test_linux_work_queue_priority_and_deadline() {
        let queue = LinuxWorkQueue::new(64);
        let w_bg = InferenceWork {
            sequence: 1,
            model: ModelHandle(1),
            kv: KvHandle(1),
            priority: Priority::Background,
            deadline: None,
            branch_parent: None,
            next_token_budget: 16,
        };
        let w_rt = InferenceWork {
            sequence: 2,
            model: ModelHandle(1),
            kv: KvHandle(2),
            priority: Priority::Realtime,
            deadline: Some(100),
            branch_parent: None,
            next_token_budget: 16,
        };
        let w_rt_earlier = InferenceWork {
            sequence: 3,
            model: ModelHandle(1),
            kv: KvHandle(3),
            priority: Priority::Realtime,
            deadline: Some(50),
            branch_parent: None,
            next_token_budget: 16,
        };

        queue.push(w_bg).expect("Push background failed");
        queue.push(w_rt).expect("Push realtime failed");
        queue
            .push(w_rt_earlier)
            .expect("Push earlier realtime failed");

        assert_eq!(queue.len(), 3);

        let first = queue.pop().expect("Pop first");
        assert_eq!(first.sequence, 3);

        let second = queue.pop().expect("Pop second");
        assert_eq!(second.sequence, 2);

        let third = queue.pop().expect("Pop third");
        assert_eq!(third.sequence, 1);

        assert!(queue.is_empty());
    }

    #[test]
    fn test_linux_gb10_ats_capabilities_if_available() {
        let device = LinuxComputeDevice::new();
        let caps = device.capabilities();
        eprintln!("LinuxComputeDevice detected capabilities: {:?}", caps);
        eprintln!(
            "Preferred memory kind: {:?}",
            device.preferred_memory_kind()
        );

        #[cfg(has_blackwell_cuda)]
        {
            if caps.is_ats_hardware_coherent() {
                assert_eq!(device.preferred_memory_kind(), LinuxMemoryKind::AtsSystem);
            }
        }
    }

    #[test]
    fn test_linux_telemetry_source_snapshot() {
        let source = LinuxTelemetrySource::new();
        let snap = source.snapshot().expect("Snapshot failed");
        assert!(!snap.architecture.is_empty());
        assert!(!snap.target.is_empty());
        assert!(snap.clock_ticks > 0);
    }
}
