//! GPU-side stats buffer for NaN/min/max without per-token CPU readback.
//!
//! Pattern:
//!   1. Bind a 16-byte stats buffer to your kernel at @group(0) @binding(5)
//!   2. Kernel calls update_stats(x) (defined in shaders/stats_helper.wgsl)
//!      after computing each output element
//!   3. CPU reads stats only every N tokens AND only on Debug mode
//!
//! Uses sign-flip-bit-cast trick: WGSL has no atomic-on-f32, so we
//! transform f32 -> u32 such that bitwise atomic min/max preserves
//! float ordering, then transform back on readback.
//!
//! 16-byte struct matches WGSL layout (3 atomics + 1 pad u32).

use std::sync::Arc;
use bytemuck::{Pod, Zeroable};
use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Copy, Clone, Pod, Zeroable, Debug, Default)]
pub struct LogitStats {
    pub min_bits: u32,
    pub max_bits: u32,
    pub has_nan: u32,
    pub _pad: u32,
}

impl LogitStats {
    /// Initial state for a fresh stats buffer.
    /// min_bits is set to f32::INFINITY's ordered representation so atomicMin
    /// will replace it on first real value. max_bits to f32::NEG_INFINITY's.
    pub fn empty() -> Self {
        Self {
            min_bits: float_to_ordered_u32(f32::INFINITY),
            max_bits: float_to_ordered_u32(f32::NEG_INFINITY),
            has_nan: 0,
            _pad: 0,
        }
    }

    pub fn decode(&self) -> (f32, f32, bool) {
        let min = ordered_u32_to_float(self.min_bits);
        let max = ordered_u32_to_float(self.max_bits);
        (min, max, self.has_nan != 0)
    }
}

/// Sign-flip-bit-cast trick for atomic float ordering on u32.
/// For non-negative f32: bit pattern sorts correctly as u32.
/// For negative f32: bit pattern sorts in REVERSE — flip sign bit
/// so they fall below positive bit patterns and sort correctly amongst
/// themselves. Symmetric inverse on readback.
#[inline]
pub fn float_to_ordered_u32(x: f32) -> u32 {
    let bits = x.to_bits();
    if bits & 0x8000_0000 != 0 {
        // Negative: flip ALL bits (so smaller magnitude negatives map to
        // larger u32 within the negative range, but still below positives)
        !bits
    } else {
        // Non-negative: flip just the sign bit so it goes above negatives
        bits | 0x8000_0000
    }
}

#[inline]
pub fn ordered_u32_to_float(x: u32) -> f32 {
    if x & 0x8000_0000 != 0 {
        // Was non-negative: unflip sign bit
        f32::from_bits(x & 0x7fff_ffff)
    } else {
        // Was negative: unflip all bits
        f32::from_bits(!x)
    }
}

/// Holder for a GPU-resident stats buffer.
pub struct GpuStatsBuffer {
    pub buffer: wgpu::Buffer,
    pub readback: wgpu::Buffer,
    queue: Arc<wgpu::Queue>,
    device: Arc<wgpu::Device>,
}

impl GpuStatsBuffer {
    pub fn new(device: Arc<wgpu::Device>, queue: Arc<wgpu::Queue>) -> Self {
        let initial = LogitStats::empty();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("gpu_stats"),
            contents: bytemuck::bytes_of(&initial),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("gpu_stats_readback"),
            size: std::mem::size_of::<LogitStats>() as u64,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        Self { buffer, readback, queue, device }
    }

    /// Reset stats buffer to empty state. Cheap — single write_buffer.
    pub fn reset(&self) {
        let initial = LogitStats::empty();
        self.queue.write_buffer(&self.buffer, 0, bytemuck::bytes_of(&initial));
    }

    /// Submit a copy from stats -> readback in the given encoder.
    pub fn record_copy(&self, encoder: &mut wgpu::CommandEncoder) {
        encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &self.readback,
            0,
            std::mem::size_of::<LogitStats>() as u64,
        );
    }

    /// Read the stats buffer. BLOCKS on device.poll. Call only when
    /// you've already submitted the recording encoder.
    pub fn read_blocking(&self) -> LogitStats {
        let slice = self.readback.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |res| {
            let _ = tx.send(res);
        });
        self.device.poll(wgpu::Maintain::Wait);
        let _ = rx.recv();

        let data = slice.get_mapped_range();
        let stats: LogitStats = *bytemuck::from_bytes(&data);
        drop(data);
        self.readback.unmap();
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ordered_roundtrip_positive() {
        for &x in &[0.0_f32, 1.0, 1e10, f32::INFINITY] {
            let u = float_to_ordered_u32(x);
            let back = ordered_u32_to_float(u);
            assert_eq!(x.to_bits(), back.to_bits(), "roundtrip failed for {}", x);
        }
    }

    #[test]
    fn ordered_roundtrip_negative() {
        for &x in &[-0.0_f32, -1.0, -1e10, f32::NEG_INFINITY] {
            let u = float_to_ordered_u32(x);
            let back = ordered_u32_to_float(u);
            assert_eq!(x.to_bits(), back.to_bits(), "roundtrip failed for {}", x);
        }
    }

    #[test]
    fn ordered_preserves_ordering() {
        // For atomic min/max correctness: if a < b in float, then
        // ordered_u32(a) < ordered_u32(b) in u32 comparison.
        let pairs = [
            (-100.0_f32, -1.0),
            (-1.0, 0.0),
            (0.0, 1.0),
            (1.0, 100.0),
            (-1e10, 1e10),
        ];
        for (a, b) in pairs {
            let ua = float_to_ordered_u32(a);
            let ub = float_to_ordered_u32(b);
            assert!(ua < ub, "ordering broken: {} -> {} >= {} -> {}", a, ua, b, ub);
        }
    }

    #[test]
    fn empty_state_is_extremes() {
        let s = LogitStats::empty();
        let (min, max, nan) = s.decode();
        assert_eq!(min, f32::INFINITY);
        assert_eq!(max, f32::NEG_INFINITY);
        assert!(!nan);
    }
}
