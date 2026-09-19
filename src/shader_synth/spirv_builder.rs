//! Direct IR → SPIR-V assembler (bypasses GLSL entirely)
//!
//! Emits binary SPIR-V words from our shader IR.
//! Eliminates GLSL compiler unpredictability and vendor differences.
//!
//! ──────────────────────────────────────────────────────────────────────
//! HONEST STATUS
//!
//! What this currently emits: SPIR-V header + a handful of OpFAdd/OpFMul
//! word pairs. It is not a full IRBuilder. It cannot represent a complete
//! shader module on its own (no type table, no entry point decoration,
//! no capability declarations, no decoration tables).
//!
//! What it does NOT yet model that the cooperative-matrix guidance asked
//! for:
//!   - cooperative matrix type class (no `OpTypeCooperativeMatrixKHR`)
//!   - load/store/mul-add helpers
//!   - capability auto-injection (`CooperativeMatrixKHR`,
//!     `SPV_KHR_cooperative_matrix`)
//!   - memory-class legality enforcement (no rule preventing
//!     `coopmat` in a uniform/SSBO/shared pointer)
//!
//! Real cooperative-matrix shaders we ship today live in `shaders/*.glsl`
//! and are compiled by `glslc`, then loaded via
//! `wgpu::ShaderSource::SpirV`. Naga (wgpu 24) does not yet accept
//! `OpTypeCooperativeMatrixKHR` from front-end code, which means a
//! native IRBuilder path here would have to skip naga validation and
//! patch the binary directly — that's the right work to do, but it's a
//! standalone project, not a five-line addition to this file.
//! ──────────────────────────────────────────────────────────────────────

pub struct SpirvBuilder {
    pub words: Vec<u32>,
    id: u32,
}

impl SpirvBuilder {
    pub fn new() -> Self {
        Self {
            words: vec![
                0x07230203, // SPIR-V magic
                0x00010000, // version 1.0
                0,          // generator
                0,          // bound (patched later)
                0,          // reserved
            ],
            id: 1,
        }
    }

    pub fn alloc_id(&mut self) -> u32 {
        let id = self.id;
        self.id += 1;
        id
    }

    /// OpFAdd (opcode 131)
    pub fn op_add(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.words.extend([(5 << 16) | 131, result_type, r, a, b]);
        r
    }

    /// OpFMul (opcode 132)
    pub fn op_mul(&mut self, result_type: u32, a: u32, b: u32) -> u32 {
        let r = self.alloc_id();
        self.words.extend([(5 << 16) | 132, result_type, r, a, b]);
        r
    }

    /// OpLoad (opcode 61)
    pub fn op_load(&mut self, result_type: u32, pointer: u32) -> u32 {
        let r = self.alloc_id();
        self.words.extend([(4 << 16) | 61, result_type, r, pointer]);
        r
    }

    /// OpStore (opcode 62)
    pub fn op_store(&mut self, pointer: u32, value: u32) {
        self.words.extend([(3 << 16) | 62, pointer, value]);
    }

    /// Finalize — patch the bound field
    pub fn finalize(&mut self) -> &[u32] {
        self.words[3] = self.id; // bound = max ID used
        &self.words
    }

    /// Write to file
    pub fn write_to_file(&mut self, path: &str) -> std::io::Result<()> {
        let data = self.finalize();
        let bytes: Vec<u8> = data.iter().flat_map(|w| w.to_le_bytes()).collect();
        std::fs::write(path, bytes)
    }
}

use super::benchmark_db::ShaderGenome;

/// Lower IR ops to SPIR-V instructions
pub fn lower_ir_to_spirv(ir: &[super::IrOp], spv: &mut SpirvBuilder) {
    let float_type = spv.alloc_id(); // placeholder type ID
    let mut stack: Vec<u32> = vec![];

    for op in ir {
        match op {
            super::IrOp::Load => {
                let ptr = spv.alloc_id();
                let val = spv.op_load(float_type, ptr);
                stack.push(val);
            }
            super::IrOp::Mul => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(spv.op_mul(float_type, a, b));
                }
            }
            super::IrOp::Add => {
                if stack.len() >= 2 {
                    let b = stack.pop().unwrap();
                    let a = stack.pop().unwrap();
                    stack.push(spv.op_add(float_type, a, b));
                }
            }
            _ => {}
        }
    }
}
