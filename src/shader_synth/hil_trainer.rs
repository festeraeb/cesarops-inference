//! Hardware-in-the-loop compiler training system
//!
//! The GPU is the teacher. The compiler is the student.
//! The optimizer is learned, not written.
//! IR evolves based on real hardware behavior.

/// GPU telemetry collected per kernel dispatch
#[derive(Clone, Debug, Default)]
pub struct GpuTelemetry {
    pub time_ms: f32,
    pub memory_bw_gbps: f32,
    pub occupancy: f32,
    pub branch_divergence: f32,
    pub l2_hit_rate: f32,
}

/// Compiler state observation (input to policy)
#[derive(Clone, Debug)]
pub struct CompilerState {
    pub ir_complexity: f32,
    pub memory_intensity: f32,
    pub divergence_risk: f32,
    pub gpu_class: String,
}

/// Actions the compiler can take
#[derive(Clone, Debug)]
pub enum CompilerAction {
    FuseOps,
    ReorderMemory,
    ChangeQuantization,
    InlineKernel,
    SplitWarpWork,
    NoOp,
}

/// Compiler genome — evolved configuration
#[derive(Clone, Debug)]
pub struct CompilerGenome {
    pub fusion_level: f32,
    pub warp_split_strategy: f32,
    pub quant_mode_bias: f32,
    pub memory_reorder_strength: f32,
}

impl Default for CompilerGenome {
    fn default() -> Self {
        Self {
            fusion_level: 0.5,
            warp_split_strategy: 0.5,
            quant_mode_bias: 0.5,
            memory_reorder_strength: 0.5,
        }
    }
}

/// Experience record for training
#[derive(Clone, Debug)]
pub struct Experience {
    pub state: CompilerState,
    pub action: CompilerAction,
    pub telemetry: GpuTelemetry,
    pub reward: f32,
}

/// Reward function — hardware-grounded, multi-objective
pub fn compute_reward(t: &GpuTelemetry) -> f32 {
    if t.time_ms <= 0.0 { return 0.0; }

    let speed = 1.0 / t.time_ms;
    let efficiency = t.occupancy * t.memory_bw_gbps;
    let stability = 1.0 - t.branch_divergence;

    speed * 0.5 + efficiency * 0.3 + stability * 0.2
}

/// Policy function — decides compiler action from state
/// (Starts as heuristic, evolves via training)
pub fn policy(state: &CompilerState) -> CompilerAction {
    if state.memory_intensity > 0.7 {
        return CompilerAction::ReorderMemory;
    }
    if state.divergence_risk > 0.5 {
        return CompilerAction::SplitWarpWork;
    }
    if state.ir_complexity > 0.8 {
        return CompilerAction::FuseOps;
    }
    CompilerAction::InlineKernel
}

/// Mutate a compiler genome
pub fn mutate_genome(genome: &mut CompilerGenome) {
    let r: f32 = rand::random();
    let delta = (rand::random::<f32>() - 0.5) * 0.1;

    if r < 0.25 {
        genome.fusion_level = (genome.fusion_level + delta).clamp(0.0, 1.0);
    } else if r < 0.5 {
        genome.warp_split_strategy = (genome.warp_split_strategy + delta).clamp(0.0, 1.0);
    } else if r < 0.75 {
        genome.quant_mode_bias = (genome.quant_mode_bias + delta).clamp(0.0, 1.0);
    } else {
        genome.memory_reorder_strength = (genome.memory_reorder_strength + delta).clamp(0.0, 1.0);
    }
}

/// Selection — keep top half by reward
pub fn select_population(pop: &mut Vec<CompilerGenome>, scores: &[f32]) {
    let mut paired: Vec<(CompilerGenome, f32)> = pop.drain(..)
        .zip(scores.iter().cloned())
        .collect();

    paired.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    let keep = paired.len() / 2;
    *pop = paired.into_iter().take(keep).map(|(g, _)| g).collect();
}

/// Learned transformation rule (replaces hand-written passes)
#[derive(Clone, Debug)]
pub struct TransformRule {
    pub pattern: String,
    pub replacement: String,
    pub weight: f32,
}

/// Apply learned rules to IR
pub fn apply_learned_rules(ir: &[String], rules: &[TransformRule]) -> Vec<String> {
    let mut out = ir.to_vec();
    for rule in rules.iter().filter(|r| r.weight > 0.1) {
        for op in out.iter_mut() {
            if op.contains(&rule.pattern) {
                *op = op.replace(&rule.pattern, &rule.replacement);
            }
        }
    }
    out
}

/// Evolve rules from training samples
pub fn evolve_rules(rules: &mut Vec<TransformRule>, samples: &[Experience]) {
    for s in samples {
        if s.reward > 0.5 {
            // Good experience → reinforce the action
            rules.push(TransformRule {
                pattern: format!("{:?}", s.state.gpu_class),
                replacement: format!("{:?}", s.action),
                weight: s.reward,
            });
        }
    }
    // Prune weak rules
    rules.retain(|r| r.weight > 0.05);
}
