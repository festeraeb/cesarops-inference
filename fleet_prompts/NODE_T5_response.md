```rust
struct SpawnConfig {
    model_path: String,
    gpu_layers: u32,
    context_size: u32,
}

fn load_worker_spawn_config(name: &str) -> Option<SpawnConfig> {
    let config_str = std::fs::read_to_string("/codebase/repos/wreckhunter2000-1/cesarops-forge-v2/cluster_config.toml").ok()?;
    let config: serde_json::Value = toml::from_str(&config_str).ok()?;
    let worker = config.get("worker")?.as_array()?.iter().find(|v| v.get("name")?.as_str()? == name)?;
    Some(SpawnConfig {
        model_path: worker.get("model_path")?.as_str()?.to_string(),
        gpu_layers: worker.get("gpu_layers").and_then(|v| v.as_u64()).unwrap_or(999) as u32,
        context_size: worker.get("context_size").and_then(|v| v.as_u64()).unwrap_or(8192) as u32,
    })
}

async fn bring_up_remote(w: &SecondaryWorker) -> BringUpOutcome {
    let client = reqwest::Client::new();
    let base_url = format!("http://{}:5500", w.host);

    let status_res = tokio::time::timeout(std::time::Duration::from_secs(3), client.get(format!("{}/status", base_url)).send()).await;
    let status_res = match status_res {
        Ok(Ok(r)) if r.status().is_success() => r.json::<serde_json::Value>().await.unwrap_or_default(),
        _ => return BringUpOutcome::HostUnreachable,
    };

    if status_res.get("state").and_then(|s| s.as_str()) == Some("serving") {
        return BringUpOutcome::Started;
    }

    let cfg = match load_worker_spawn_config(&w.name) {
        Some(c) => c,
        None => return BringUpOutcome::Failed("Config not found".to_string()),
    };

    let payload = serde_json::json!({
        "model_path": cfg.model_path,
        "port": w.port,
        "gpu_layers": cfg.gpu_layers,
        "context_size": cfg.context_size
    });

    let spawn_res = tokio::time::timeout(std::time::Duration::from_secs(120), client.post(format!("{}/spawn", base_url)).json(&payload).send()).await;
    match spawn_res {
        Ok(Ok(r)) if r.status().is_success() => BringUpOutcome::Started,
        Ok(Ok(r)) => {
            let err = r.json::<serde_json::Value>().await.unwrap_or_default();
            BringUpOutcome::Failed(err.get("error").and_then(|e| e.as_str()).unwrap_or("Unknown error").to_string())
        }
        _ => BringUpOutcome::Failed("Spawn request timed out or failed".to_string()),
    }
}
```
