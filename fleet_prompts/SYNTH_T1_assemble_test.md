# Task: Assemble and verify the shader_synth module compiles

We have 14 Rust source files in `cesarops-inference/src/shader_synth/`. They need to be wired into the main crate and compile cleanly.

## Current module list (src/shader_synth/mod.rs):
- gpu_probe.rs
- shader_scan.rs
- quant_policy.rs
- prompt_builder.rs
- compiler.rs
- benchmark_db.rs
- layout_reverse.rs
- jit_runtime.rs
- spirv_builder.rs
- tensor_repack.rs
- ir_graph.rs
- microbench.rs
- simt_sim.rs
- hil_trainer.rs

## What to do:

1. Add `pub mod shader_synth;` to `src/lib.rs` (if not already there)
2. Fix any compilation errors across all 14 files:
   - Missing imports
   - Type mismatches
   - Unused variable warnings (prefix with `_`)
   - Any `rand::random()` calls need `rand` in Cargo.toml
3. Ensure `cargo check -p cesarops-inference` passes with no errors (warnings OK)

## Known issues to fix:
- `benchmark_db.rs` uses `rand::random()` — need `rand = "0.8"` in Cargo.toml
- `hil_trainer.rs` uses `rand::random()` — same
- `compiler.rs` references `super::gpu_probe::GpuClass` — verify path
- `prompt_builder.rs` references `super::quant_policy` — verify path
- `spirv_builder.rs` references `super::IrOp` — the IrOp enum is in mod.rs
- `ir_graph.rs` uses `std::array::from_fn` — needs Rust 1.63+ (we have 2021 edition, fine)

## Output:
List of changes needed to make it compile. For each file that needs a fix, show the exact line change. If Cargo.toml needs `rand`, show that too.
