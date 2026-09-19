//! Shader Synthesizer — hardware-adaptive kernel generation pipeline
//!
//! Full GPU compiler research platform:
//! - Detect GPU class, scan existing shaders
//! - Reverse-engineer GGML quant layouts
//! - Build prompts for LLM shader generation
//! - Direct IR → SPIR-V assembly (no GLSL dependency)
//! - GPU-aware tensor re-layout (warp-coalesced packing)
//! - JIT compile + hot-swap pipelines at runtime
//! - Benchmark farm + evolutionary shader search
//! - Cross-GPU shader distillation (Turing → Pascal)

pub mod gpu_probe;
pub mod shader_scan;
pub mod quant_policy;
pub mod prompt_builder;
pub mod compiler;
pub mod benchmark_db;
pub mod layout_reverse;
pub mod jit_runtime;
pub mod spirv_builder;
pub mod tensor_repack;
pub mod ir_graph;
pub mod microbench;
pub mod simt_sim;
pub mod hil_trainer;
pub mod coopmat_probe;

/// Shader IR operations — the intermediate representation
/// between high-level intent and low-level GPU instructions.
#[derive(Debug, Clone)]
pub enum IrOp {
    Load,
    Store,
    Mul,
    Add,
    Dot,
    Unpack4Bit,
    SubgroupReduce,
    FusedMulAdd,
}
