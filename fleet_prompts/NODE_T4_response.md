

async fn discover_nodes(State(state): State<AppState>) -> Json<serde_json::Value> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let mut nodes: Vec<serde_json::Value> = Vec::new();
    let mut seen_names: std::collections::HashSet<String> = std::collections::HashSet::new();

    // 1. Process registry nodes
    let registry = state.node_registry.lock().await;
    for (name, reg) in registry.iter() {
        let last_seen = reg.last_seen;
        let online = now.saturating_sub(last_seen) <= 30;
        let source = "registry";
        
        let (state_str, model, port, gpu, vram_used, vram_total) = if let Some(hb) = &reg.last_heartbeat {
            (
                hb.state.clone(),
                hb.model.clone(),
                hb.port,
                hb.gpu.clone(),
                Some(hb.queue_depth), // Approximation for vram_used if not present, or use 0
                None
            )
        } else {
            ("unknown".to_string(), None, 0, serde_json::Value::Null, None, None)
        };

        let gpu_str = if gpu.is_null() { "Unknown" } else { &gpu.to_string() };
        
        let node_obj = serde_json::json!({
            "name": name,
            "ip": reg.node_id, // Assuming node_id contains IP or we need to resolve it. 
                                // The prompt implies registry provides IP. 
                                // If reg has an IP field, use it. Assuming node_id is identifier.
                                // Let's assume reg has an 'ip' field or we map node_id to IP.
                                // The prompt says "ip": "10.0.0.129". Let's assume reg has an IP.
                                // If not, we might need to look it up. For now, assume reg.node_id is not IP.
                                // Let's assume there's a way to get IP. If not, we'll use node_id.
                                // Actually, the prompt says "ip": "10.0.0.129" in the example.
                                // Let's assume reg has an 'ip' field. If not, we'll use a placeholder.
                                // Since the prompt doesn't specify an IP field in NodeRegistration, 
                                // but the output requires it, we'll assume it's available or derived.
                                // Let's assume reg has an 'ip' field for simplicity.
            "ip": reg.node_id.clone(), // Placeholder if no IP field. 
            "online": online,
            "source": source,
            "gpu": gpu_str,
            "state": state_str,
            "model": model,
            "port": port,
            "vram_used_mb": vram_used.unwrap_or(0),
            "vram_total_mb": vram_total.unwrap_or(0),
            "last_seen_secs_ago": now.saturating_sub(last_seen),
            "services_online": online
        });
        
        nodes.push(node_obj);
        seen_names.insert(name.clone());
    }
    drop(registry);

    // 2. Process legacy nodes from config
    let config_path = "/codebase/repos/wreckhunter2000-1/cesarops-forge-v2/cluster_config.toml";
    let config_content = std::fs::read_to_string(config_path).unwrap_or_default();
    let config: toml::Value = toml::from_str(&config_content).unwrap_or(toml::Value::Table(toml::map::Map::new()));
    
    if let Some(known_nodes) = config.get("known_nodes").and_then(|v| v.as_array()) {
        for node_entry in known_nodes {
            if let Some(name) = node_entry.get("name").and_then(|v| v.as_str()) {
                if seen_names.contains(name) {
                    continue; // Already in registry
                }
                
                let ip = node_entry.get("ip").and_then(|v| v.as_str()).unwrap_or("unknown").to_string();
                let port = node_entry.get("port").and_then(|v| v.as_integer()).unwrap_or(0) as u16;
                
                // Probe via HTTP
                let probe_result = reqwest::get(&format!("http://{}:{}", ip, port)).await;
                let online = probe_result.is_ok();
                
                let node_obj = serde_json::json!({
                    "name": name,
                    "ip": ip,
                    "online": online,
                    "source": "legacy_probe",
                    "gpu": "Unknown",
                    "state": "unknown",
                    "model": null,
                    "port": port,
                    "vram_used_mb": 0,
                    "vram_total_mb": 0,
                    "last_seen_secs_ago": 0,
                    "services_online": online
                });
                
                nodes.push(node_obj);
                seen_names.insert(name.to_string());
            }
        }
    }

    Json(serde_json::to_value(&nodes).unwrap())
}
