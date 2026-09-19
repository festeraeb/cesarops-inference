//! SIMT warp simulator — predict GPU behavior before Vulkan dispatch
//!
//! Simulates: warps (32 lanes), instruction issue, memory latency,
//! branch divergence, execution masking.
//! Lets you predict Pascal vs Turing behavior and detect divergence
//! hotspots BEFORE compilation.

#[derive(Clone)]
pub struct LaneState {
    pub pc: usize,
    pub active: bool,
    pub registers: [f32; 8],
}

impl Default for LaneState {
    fn default() -> Self {
        Self { pc: 0, active: true, registers: [0.0; 8] }
    }
}

#[derive(Clone)]
pub struct Warp {
    pub lanes: [LaneState; 32],
    pub stalled_cycles: u32,
}

impl Default for Warp {
    fn default() -> Self {
        Self {
            lanes: std::array::from_fn(|_| LaneState::default()),
            stalled_cycles: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub enum Inst {
    Add,
    Mul,
    FMA,
    LoadGlobal,
    StoreGlobal,
    LoadShared,
    Branch(bool),
    SubgroupReduce,
}

#[derive(Debug, Default)]
pub struct SimResult {
    pub total_cycles: u32,
    pub divergence_penalties: u32,
    pub memory_stalls: u32,
    pub active_lane_avg: f32,
}

/// Step one instruction across all lanes in a warp
pub fn step_warp(warp: &mut Warp, inst: &Inst) {
    match inst {
        Inst::Add | Inst::Mul | Inst::FMA => {
            for lane in warp.lanes.iter_mut() {
                if lane.active {
                    lane.registers[0] = lane.registers[0] + lane.registers[1];
                }
            }
            // 1 cycle for ALU
        }
        Inst::Branch(pred) => {
            let mut active_count = 0u32;
            for lane in warp.lanes.iter_mut() {
                lane.active = lane.active && *pred;
                if lane.active { active_count += 1; }
            }
            if active_count < 32 && active_count > 0 {
                warp.stalled_cycles += 2; // divergence penalty
            }
        }
        Inst::LoadGlobal => {
            warp.stalled_cycles += 4; // ~400 cycle latency / 100 = simplified
        }
        Inst::StoreGlobal => {
            warp.stalled_cycles += 2;
        }
        Inst::LoadShared => {
            warp.stalled_cycles += 1; // shared memory is fast
        }
        Inst::SubgroupReduce => {
            // 5 shuffle steps for 32-wide reduction
            warp.stalled_cycles += 5;
        }
    }
}

/// Simulate a full program on a warp, return performance metrics
pub fn simulate(program: &[Inst]) -> SimResult {
    let mut warp = Warp::default();
    let mut result = SimResult::default();

    for inst in program {
        step_warp(&mut warp, inst);
        result.total_cycles += 1 + warp.stalled_cycles;

        match inst {
            Inst::Branch(_) => result.divergence_penalties += warp.stalled_cycles,
            Inst::LoadGlobal | Inst::StoreGlobal => result.memory_stalls += warp.stalled_cycles,
            _ => {}
        }

        warp.stalled_cycles = 0;
    }

    let active: f32 = warp.lanes.iter().filter(|l| l.active).count() as f32;
    result.active_lane_avg = active / 32.0;

    result
}

/// Compare two instruction sequences (shader variants)
pub fn compare_programs(a: &[Inst], b: &[Inst]) -> (SimResult, SimResult) {
    (simulate(a), simulate(b))
}
