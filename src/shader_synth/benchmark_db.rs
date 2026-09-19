use std::collections::HashMap;

#[derive(Clone, Debug)]
pub struct BenchmarkResult {
    pub shader_id: String,
    pub gpu: String,
    pub ms_per_token: f32,
    pub memory_bw_util: f32,
    pub generation: u32,
}

#[derive(Clone, Debug)]
pub struct ShaderGenome {
    pub code: String,
    pub score: f32,
    pub mutations: u32,
    pub shader_id: String,
}

pub struct ShaderDB {
    pub results: HashMap<String, BenchmarkResult>,
}

impl ShaderDB {
    pub fn new() -> Self {
        Self { results: HashMap::new() }
    }

    pub fn insert(&mut self, r: BenchmarkResult) {
        self.results.insert(r.shader_id.clone(), r);
    }

    pub fn best_for_gpu(&self, gpu: &str) -> Option<&BenchmarkResult> {
        self.results
            .values()
            .filter(|r| r.gpu == gpu)
            .min_by(|a, b| a.ms_per_token.partial_cmp(&b.ms_per_token).unwrap())
    }
}

/// Fitness function — prioritize latency, penalize errors
pub fn fitness(ms_per_kernel: f32, memory_bw: f32, errors: u32) -> f32 {
    if errors > 0 {
        return 0.0;
    }
    (1.0 / ms_per_kernel) * 0.7 + memory_bw * 0.3
}

/// Simple evolution step — sort by score, keep best, mutate
pub fn evolve(mut population: Vec<ShaderGenome>) -> Vec<ShaderGenome> {
    population.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

    let mut next_gen = Vec::new();

    // Elitism — keep top 2
    next_gen.extend(population.iter().take(2).cloned());

    // Mutation — mutate top 5
    for parent in population.iter().take(5) {
        let mut child = parent.clone();
        child.code = mutate_shader(&parent.code);
        child.mutations += 1;
        next_gen.push(child);
    }

    next_gen
}

/// Shader-level mutations (real transformations, not random noise)
fn mutate_shader(code: &str) -> String {
    let mut out = code.to_string();

    let r: f32 = rand::random();

    if r < 0.25 {
        // Unroll hint
        out = out.replace("for (", "for ( /* unrolled */ ");
    } else if r < 0.5 {
        // Precision downgrade (Pascal optimization)
        out = out.replacen("float acc", "half acc", 1);
    } else if r < 0.75 {
        // Workgroup size exploration
        out = out.replace("local_size_x = 256", "local_size_x = 128");
    }
    // else: no mutation (stability)

    out
}
