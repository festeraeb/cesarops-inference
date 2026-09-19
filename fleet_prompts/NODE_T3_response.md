```rust
// === node_register ===
async fn node_register(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let node_id = payload["node_id"].as_str().unwrap_or("unknown").to_string();
    
    let registration = NodeRegistration {
        node_id: node_id.clone(),
        hardware: payload["hardware"].clone(),
        available_models: serde_json::from_value(payload["available_models"].clone()).unwrap_or_default(),
        listen_port: payload["listen_port"].as_u64().unwrap_or(0) as u16,
        last_heartbeat: None,
        last_seen: now,
    };

    state.node_registry.lock().await.insert(node_id.clone(), registration);
    tracing::info!(node_id = %node_id, "Node registered");
    
    Json(serde_json::json!({ "status": "registered", "node_id": node_id }))
}

// === node_heartbeat ===
async fn node_heartbeat(
    State(state): State<AppState>,
    Json(payload): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let node_id = payload["node_id"].as_str().unwrap_or("");
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let mut registry = state.node_registry.lock().await;

    if let Some(node) = registry.get_mut(node_id) {
        node.last_heartbeat = Some(NodeHeartbeat {
            state: payload["state"].as_str().unwrap_or("unknown").to_string(),
            model: payload["model"].as_str().map(|s| s.to_string()),
            port: payload["port"].as_u64().map(|p| p as u16),
            gpu: payload["gpu"].clone(),
            queue_depth: payload["queue_depth"].as_u64().unwrap_or(0) as u32,
        });
        node.last_seen = now;
        Json(serde_json::json!({ "status": "ok" }))
    } else {
        Json(serde_json::json!({ "error": "unknown node, register first" }))
    }
}

// === list_registered_nodes ===
async fn list_registered_nodes(State(state): State<AppState>) -> Json<serde_json::Value> {
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();
    let registry = state.node_registry.lock().await;
    
    let nodes: Vec<serde_json::Value> = registry.values().map(|n| {
        let mut val = serde_json::to_value(n).unwrap();
        val["online"] = serde_json::json!(now > n.last_seen && (now - n.last_seen) < 30);
        val
    }).collect();

    Json(serde_json::Value::Array(nodes))
}
```
