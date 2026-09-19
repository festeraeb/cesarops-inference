//! Dump per-layer scalar weights and norm magnitudes from Gemma-4 MoE GGUF
//! so we can sanity-check what the runner is loading.

use std::path::Path;

use cesarops_inference::bridge;
use cesarops_inference::hardware;
use cesarops_inference::loader;

const MODEL: &str = "/codebase/models/Gemma-4-26B-MoE-IQ4_XS.gguf";

#[test]
#[ignore = "metadata + tensor stats only"]
fn dump_per_layer_scalars() {
    let profile = hardware::audit_system();
    let weights = loader::load(Path::new(MODEL), &profile).expect("load gguf");

    // Collect 4 numeric fingerprints per layer:
    //   layer_output_scale       (1-element scalar)
    //   attn_norm[0..4]          (first 4 floats, first norm in block)
    //   ffn_norm[0..4]
    //   post_attention_norm[0..4]
    let layers_to_probe = [0usize, 1, 14, 15, 28, 29];
    for &li in &layers_to_probe {
        let mut row = String::new();
        row.push_str(&format!("layer {:>3}: ", li));

        // layer_output_scale
        let key = format!("blk.{li}.layer_output_scale.weight");
        if let (Some(region), Some(bytes)) = (
            weights.tensors.get(&key),
            weights.tensor_bytes(&key),
        ) {
            let data = bridge::dequantize_tensor(bytes, region.quant_type, 1);
            row.push_str(&format!("scale={:>10.5} qt={} ", data[0], region.quant_type));
        } else {
            row.push_str("scale=MISSING ");
        }

        for which in ["attn_norm", "ffn_norm", "post_attention_norm", "post_ffw_norm"] {
            let key = format!("blk.{li}.{which}.weight");
            if let (Some(region), Some(bytes)) = (
                weights.tensors.get(&key),
                weights.tensor_bytes(&key),
            ) {
                let n: usize = region.shape.iter().product();
                let data = bridge::dequantize_tensor(bytes, region.quant_type, n.min(8));
                let mean: f32 = data.iter().take(8).sum::<f32>() / 8.0;
                let max = data.iter().take(8).fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                row.push_str(&format!("{}: μ={:>7.4} max={:>7.4} qt={}  ", which, mean, max, region.quant_type));
            } else {
                row.push_str(&format!("{}: MISSING  ", which));
            }
        }
        println!("{row}");
    }
}
