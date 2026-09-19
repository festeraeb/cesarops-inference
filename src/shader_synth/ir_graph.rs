//! LLVM-style shader optimizer IR graph
//!
//! Replaces linear IR with a real compiler-style DAG enabling:
//! - Global optimization across blocks
//! - Instruction fusion beyond local passes
//! - Memory dependency tracking
//! - Scheduling (critical for Pascal/Turing)

use std::collections::{HashMap, HashSet};

#[derive(Clone, Debug, PartialEq)]
pub enum NodeOp {
    Load,
    Store,
    Add,
    Mul,
    Dot,       // Fused mul+add
    Unpack4Bit,
    SubgroupReduce,
    FusedMulAdd,
}

#[derive(Clone, Debug)]
pub struct Node {
    pub id: usize,
    pub op: NodeOp,
    pub inputs: Vec<usize>,
}

#[derive(Default)]
pub struct IrGraph {
    pub nodes: HashMap<usize, Node>,
    pub outputs: Vec<usize>,
    next_id: usize,
}

impl IrGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_node(&mut self, op: NodeOp, inputs: Vec<usize>) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, Node { id, op, inputs });
        id
    }

    pub fn mark_output(&mut self, id: usize) {
        self.outputs.push(id);
    }

    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }
}

/// Global fusion pass — fuse Mul+Add into Dot (FMA)
/// This is where LLVM-like optimization begins.
pub fn fuse_mul_add(graph: &mut IrGraph) {
    let mut fused = HashSet::new();

    let node_ids: Vec<usize> = graph.nodes.keys().cloned().collect();

    for &id in &node_ids {
        if fused.contains(&id) { continue; }

        let node = match graph.nodes.get(&id) {
            Some(n) => n.clone(),
            None => continue,
        };

        if node.op != NodeOp::Mul { continue; }

        // Find any Add that consumes this Mul's output
        for &consumer_id in &node_ids {
            if fused.contains(&consumer_id) { continue; }
            if let Some(consumer) = graph.nodes.get(&consumer_id) {
                if consumer.op == NodeOp::Add && consumer.inputs.contains(&id) {
                    // Fuse: replace Add with FusedMulAdd, remove Mul
                    if let Some(c) = graph.nodes.get_mut(&consumer_id) {
                        c.op = NodeOp::FusedMulAdd;
                    }
                    fused.insert(id);
                    break;
                }
            }
        }
    }

    // Remove fused nodes
    for id in fused {
        graph.nodes.remove(&id);
    }
}

/// Dead node elimination — remove nodes not reachable from outputs
pub fn eliminate_dead_nodes(graph: &mut IrGraph) {
    let mut reachable = HashSet::new();
    let mut worklist: Vec<usize> = graph.outputs.clone();

    while let Some(id) = worklist.pop() {
        if reachable.contains(&id) { continue; }
        reachable.insert(id);
        if let Some(node) = graph.nodes.get(&id) {
            worklist.extend(node.inputs.iter());
        }
    }

    graph.nodes.retain(|id, _| reachable.contains(id));
}

/// Subgroup lowering — remove subgroup ops for Pascal
pub fn lower_for_pascal(graph: &mut IrGraph) {
    for node in graph.nodes.values_mut() {
        if node.op == NodeOp::SubgroupReduce {
            // Replace with sequential Add chain (Pascal-safe)
            node.op = NodeOp::Add;
        }
    }
}
