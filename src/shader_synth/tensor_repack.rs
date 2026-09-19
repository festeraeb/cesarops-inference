//! GGUF tensor re-layout compiler — GPU warp-coalesced packing
//!
//! Converts CPU-optimal GGUF layout to GPU-optimal warp-interleaved microtiles.
//! This alone can give 1.3x-2.5x speedup on memory-bound kernels.

/// GPU tensor layout descriptor
#[derive(Debug, Clone)]
pub struct GpuTensorLayout {
    pub warp_tile: usize,
    pub block_size: usize,
    pub stride: usize,
}

/// Repack tensor data from CPU row-major to GPU warp-coalesced layout.
///
/// Instead of: thread reads scattered memory
/// We get: warp reads contiguous memory → coalesced load
///
/// This is the #1 performance optimization for Pascal (memory-bound).
pub fn repack_for_warp_coalescing(data: &[u8], warp_size: usize) -> Vec<u8> {
    let tile = 16;
    let chunk_size = warp_size * tile;

    if data.len() < chunk_size {
        return data.to_vec();
    }

    let mut out = vec![0u8; data.len()];
    let num_chunks = data.len() / chunk_size;

    for w in 0..num_chunks {
        for lane in 0..warp_size {
            for i in 0..tile {
                let src = w * chunk_size + lane * tile + i;
                let dst = w * chunk_size + i * warp_size + lane;

                if src < data.len() && dst < out.len() {
                    out[dst] = data[src];
                }
            }
        }
    }

    // Copy remainder (not aligned to chunk_size)
    let remainder_start = num_chunks * chunk_size;
    if remainder_start < data.len() {
        out[remainder_start..].copy_from_slice(&data[remainder_start..]);
    }

    out
}

/// Repack specifically for IQ4_XS blocks (24 bytes per 32 weights)
/// Ensures blocks are warp-aligned for coalesced access in the fused matvec shader.
pub fn repack_iq4xs_for_gpu(data: &[u8], num_rows: usize) -> Vec<u8> {
    let block_bytes = 24;
    let blocks_per_row = data.len() / (num_rows * block_bytes);

    // For IQ4_XS, the key optimization is ensuring that consecutive warps
    // access consecutive blocks in memory (row-major block order is already good).
    // The main win is aligning row starts to cache line boundaries (64 bytes).
    let row_bytes = blocks_per_row * block_bytes;
    let aligned_row = ((row_bytes + 63) / 64) * 64;

    if aligned_row == row_bytes {
        return data.to_vec(); // Already aligned
    }

    // Pad each row to cache line boundary
    let mut out = vec![0u8; num_rows * aligned_row];
    for row in 0..num_rows {
        let src_start = row * row_bytes;
        let dst_start = row * aligned_row;
        let src_end = (src_start + row_bytes).min(data.len());
        let copy_len = src_end - src_start;
        out[dst_start..dst_start + copy_len].copy_from_slice(&data[src_start..src_end]);
    }

    out
}
