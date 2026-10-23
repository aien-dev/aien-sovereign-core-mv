use crate::PlatformError;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceAddress(pub u64);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferLayout {
    pub size: usize,
    pub align: usize,
}

impl BufferLayout {
    pub const fn new(size: usize, align: usize) -> Self {
        Self { size, align }
    }
}

pub trait UnifiedBuffer {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn device_address(&self) -> DeviceAddress;
    fn as_ptr(&self) -> *const u8;
    fn as_mut_ptr(&mut self) -> *mut u8;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fence(pub u64);

pub struct ComputeWork<'a> {
    pub kernel_id: u32,
    pub input_buffer: DeviceAddress,
    pub output_buffer: DeviceAddress,
    pub params: &'a [u8],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BufferRegion {
    pub address: DeviceAddress,
    pub offset: usize,
    pub len: usize,
}

pub trait MemoryDevice {
    fn copy(&self, src: BufferRegion, dst: BufferRegion) -> Result<Fence, PlatformError>;
    fn zero(&self, dst: BufferRegion) -> Result<Fence, PlatformError>;
}

pub trait ComputeDevice {
    type Buffer: UnifiedBuffer;

    fn alloc(&self, layout: BufferLayout) -> Result<Self::Buffer, PlatformError>;
    fn submit(&self, work: ComputeWork<'_>) -> Result<Fence, PlatformError>;
    fn synchronize(&self, fence: Fence) -> Result<(), PlatformError>;
}

impl<T: ?Sized + UnifiedBuffer> UnifiedBuffer for alloc::boxed::Box<T> {
    fn len(&self) -> usize {
        (**self).len()
    }
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }
    fn device_address(&self) -> DeviceAddress {
        (**self).device_address()
    }
    fn as_ptr(&self) -> *const u8 {
        (**self).as_ptr()
    }
    fn as_mut_ptr(&mut self) -> *mut u8 {
        (**self).as_mut_ptr()
    }
}
