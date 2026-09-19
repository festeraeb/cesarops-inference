use super::gpu_probe::GpuClass;
use super::quant_policy::{self, QuantMode};
use super::shader_scan::ShaderEntry;

/// Build a structured shader-generation prompt for Gemma or any LLM.
pub fn build_shader_prompt(
    gpu: &GpuClass,
    quant: &QuantMode,
    existing: &[ShaderEntry],
    task: &str,
) -> String {
    let existing_list = existing
        .iter()
        .map(|s| format!("- {} ({})", s.name, s.kind))
        .collect::<Vec<_>>()
        .join("\n");

    let hw_constraints = match gpu {
        GpuClass::Pascal => quant_policy::pascal_constraints(),
        GpuClass::Turing => quant_policy::turing_advantages(),
        _ => "No specific hardware constraints.",
    };

    format!(
r#"You are a GPU compute shader expert.

TARGET TASK: {task}

HARDWARE TARGET: {gpu:?}

QUANTIZATION STRATEGY: {quant:?}

EXISTING SHADERS IN PROJECT:
{existing_list}

{hw_constraints}

REQUIREMENTS:
- Generate a Vulkan GLSL compute shader (.comp) OR WGSL for wgpu
- Optimize for {gpu:?} hardware constraints
- Avoid unsupported features for this GPU class
- Use coalesced SSBO access patterns
- Minimize register pressure (target 64-96 regs/thread)
- Include layout bindings and push constants

OUTPUT FORMAT:
- Full shader source code
- Include comments explaining optimization choices
"#
    )
}

/// Build an evolution improvement prompt (feeds benchmark results back)
pub fn build_evolution_prompt(
    shader_src: &str,
    gpu: &GpuClass,
    ms_per_token: f32,
) -> String {
    format!(
r#"Improve this Vulkan compute shader for {gpu:?}.

Current performance: {ms_per_token} ms/token.

Focus on:
- Memory coalescing (biggest real win)
- Warp efficiency
- Quantized GEMV performance
- Minimal divergence
- Reduce ALU ops where possible

CURRENT SHADER:
{shader_src}

OUTPUT: Improved shader with comments explaining changes.
"#
    )
}
