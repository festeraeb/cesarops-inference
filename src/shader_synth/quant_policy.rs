use super::gpu_probe::GpuClass;

#[derive(Debug, Clone)]
pub enum QuantMode {
    FP32,
    FP16,
    INT8,
    Q4_0,
    Q4PascalSafe,  // Register-heavy Q4, no subgroup dependency
    IQ4_XS,        // Importance-weighted 4-bit
}

pub fn choose_quant_mode(gpu: &GpuClass) -> QuantMode {
    match gpu {
        GpuClass::Pascal => QuantMode::Q4PascalSafe,
        GpuClass::Turing => QuantMode::IQ4_XS,
        GpuClass::AmpereOrNewer => QuantMode::INT8,
        GpuClass::Unknown => QuantMode::FP16,
    }
}

/// Pascal-specific constraints to inject into prompts
pub fn pascal_constraints() -> &'static str {
    r#"PASCAL HARDWARE CONSTRAINTS:
- NO subgroup-heavy reliance (partial support, inconsistent)
- NO tensor cores (none available)
- NO heavy shared memory tiling (often slower than register reuse)
- NO INT8 dot pipelines (inconsistent acceleration)
- PREFER register-resident loops
- PREFER FP16 where possible
- PREFER Q4 packing with manual unpack
- PREFER warp-independent execution
- PREFER minimal branching
- PREFER coalesced SSBO access patterns"#
}

/// Turing-specific advantages to inject
pub fn turing_advantages() -> &'static str {
    r#"TURING HARDWARE ADVANTAGES:
- Good INT8 throughput via tensor cores
- FP16 tensor core acceleration available
- Subgroup shuffle (__shfl_sync / subgroupAdd) is fast
- 64KB configurable shared memory per SM
- Use cooperative matrix if available
- Warp-shuffle reductions preferred over shared memory"#
}
