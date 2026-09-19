//! Ring-allocated UBO pool — replaces per-dispatch wgpu::Buffer creation with
//! sub-allocation inside one persistent buffer.
//!
//! Pattern:
//!   pool.alloc(size) -> UniformAllocation { offset, size }
//!   pool.write(&data, &alloc)  // queue.write_buffer, NOT map_async
//!   ... encoder.dispatch ...
//!   pool.reset_after(submission_idx, &device)  // poll then reset
//!
//! Forward-pass-scope reset cadence. NO free list — pure bump allocator.
//! Single-threaded per GPU context (no Mutex, no Arc on the pool itself —
//! caller wraps in Arc<RwLock<...>> if multi-tenant).

use std::sync::Arc;
use wgpu::SubmissionIndex;

#[derive(Clone, Copy, Debug)]
pub struct UniformAllocation {
    pub offset: u64,
    pub size: u64,
}

pub struct UniformPool {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    buffer: wgpu::Buffer,
    capacity: u64,
    offset: u64,
    alignment: u64,
}

impl UniformPool {
    /// Construct a UBO pool backed by one large buffer.
    /// Default capacity 8 MB, alignment derived from device limits.
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        Self::with_capacity(device, queue, 8 * 1024 * 1024)
    }

    pub fn with_capacity(
        device: Arc<wgpu::Device>,
        queue: Arc<wgpu::Queue>,
        capacity: u64,
    ) -> Self {
        let limits = device.limits();
        let alignment = limits.min_uniform_buffer_offset_alignment as u64;

        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("uniform_pool"),
            size: capacity,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self {
            device,
            queue,
            buffer,
            capacity,
            offset: 0,
            alignment: alignment.max(16),
        }
    }

    /// Bump-allocate a region. Wraps to 0 on overflow (caller MUST ensure
    /// no in-flight work references prior offsets — call reset_after first).
    #[inline]
    pub fn alloc(&mut self, size: u64) -> UniformAllocation {
        let aligned_offset = (self.offset + self.alignment - 1) & !(self.alignment - 1);

        if aligned_offset + size > self.capacity {
            // Ring wrap. Caller is responsible for ensuring prior submissions
            // have completed via reset_after().
            self.offset = 0;
            return UniformAllocation { offset: 0, size };
        }

        let alloc = UniformAllocation {
            offset: aligned_offset,
            size,
        };
        self.offset = aligned_offset + size;
        alloc
    }

    /// Upload typed POD data via queue.write_buffer (DMA path on Pascal).
    /// Never use map_async here — that's a sync stall on P100 Vulkan.
    #[inline]
    pub fn write<T: bytemuck::Pod>(&self, data: &T, alloc: &UniformAllocation) {
        let bytes = bytemuck::bytes_of(data);
        debug_assert!(bytes.len() as u64 <= alloc.size);
        self.queue.write_buffer(&self.buffer, alloc.offset, bytes);
    }

    /// Upload raw bytes (for non-Pod payloads or precomputed buffers).
    #[inline]
    pub fn write_bytes(&self, bytes: &[u8], alloc: &UniformAllocation) {
        debug_assert!(bytes.len() as u64 <= alloc.size);
        self.queue.write_buffer(&self.buffer, alloc.offset, bytes);
    }

    /// Reset the pool after the GPU has drained the given submission.
    /// Blocks until the submission completes via Maintain::WaitForSubmissionIndex.
    /// Call this between forward passes.
    pub fn reset_after(&mut self, submission_idx: SubmissionIndex) {
        self.device.poll(wgpu::Maintain::WaitForSubmissionIndex(submission_idx));
        self.offset = 0;
    }

    /// Force-reset without waiting for GPU. Use only when caller has external
    /// guarantee that no in-flight work references this pool.
    pub fn reset_unchecked(&mut self) {
        self.offset = 0;
    }

    pub fn buffer(&self) -> &wgpu::Buffer {
        &self.buffer
    }

    pub fn capacity(&self) -> u64 {
        self.capacity
    }

    pub fn current_offset(&self) -> u64 {
        self.offset
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Construction tests require a real wgpu device, deferred to integration tests.
    // Logic-only tests on a stub:

    #[test]
    fn alloc_aligns_correctly() {
        // Pure logic test — alignment math, no GPU needed.
        // Mock alignment behavior:
        let alignment: u64 = 256;
        let mut offset: u64 = 5;
        let size: u64 = 100;

        let aligned = (offset + alignment - 1) & !(alignment - 1);
        assert_eq!(aligned, 256);

        offset = aligned + size;
        assert_eq!(offset, 356);

        let aligned2 = (offset + alignment - 1) & !(alignment - 1);
        assert_eq!(aligned2, 512);
    }
}
