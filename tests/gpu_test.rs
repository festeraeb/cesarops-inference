use cesarops_inference::gpu_context::GpuContext;

#[tokio::main]
async fn main() {
    println!("Initializing GPU context on GPU 0...");
    match GpuContext::init(0).await {
        Ok(gpu) => {
            println!("GPU init success!");
            // Test: 2x3 * 3x2 matmul (B is transposed, so B_T is 2x3)
            let a: Vec<f32> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]; // 2x3
            let b_t: Vec<f32> = vec![1.0, 3.0, 5.0, 2.0, 4.0, 6.0]; // 2x3 (rows of B^T)
            println!("Running 2x3 * 3x2 matmul on GPU...");
            let result = gpu.matmul_gpu(&a, &b_t, 2, 3, 2);
            println!("Result: {:?}", result);
            println!("Expected: [22.0, 28.0, 49.0, 64.0]");

            // Larger test: 1x1536 * 1536x1536 (simulates a projection)
            let big_a: Vec<f32> = (0..1536).map(|i| (i as f32) * 0.001).collect();
            let big_b: Vec<f32> = (0..1536*1536).map(|i| ((i % 7) as f32 - 3.0) * 0.01).collect();
            println!("\nRunning 1x1536 * 1536x1536 matmul on GPU...");
            let start = std::time::Instant::now();
            let big_result = gpu.matmul_gpu(&big_a, &big_b, 1, 1536, 1536);
            let elapsed = start.elapsed();
            println!("Done in {:?}, output[0..5] = {:?}", elapsed, &big_result[..5]);
        }
        Err(e) => {
            println!("GPU init failed: {}", e);
            println!("This is expected if no Vulkan GPU is available or SHADER_F16 is missing.");
        }
    }
}
