

```rust
#[derive(Clone, Serialize, Deserialize)]
struct MissionRecord {
    id: String,
    source: String,
    scenario_text: String,
    status: String,
    submitted_at: u64,
    completed_at: Option<u64>,
    report: Option<MissionReport>,
}

async fn webhook_mission(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let id = format!("{:016x}", std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos() & 0xFFFFFFFFFFFFFFFF);
    let scenario = body["scenario"].as_str().unwrap_or("").to_string();
    let bbox = body["bbox"].as_array().unwrap_or(&vec![]).clone();
    let priority = body["priority"].as_i64().unwrap_or(0) as u8;
    let callback = body["callback_url"].as_str().map(|s| s.to_string());
    let source = body["source"].as_str().unwrap_or("unknown").to_string();
    let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs();

    let record = MissionRecord {
        id: id.clone(), source, scenario_text: scenario, status: "running".into(),
        submitted_at: now, completed_at: None, report: None,
    };

    {
        let mut missions = state.missions.lock().unwrap();
        missions.push(record);
        if missions.len() > 50 { missions.remove(0); }
    }

    let orchestrator = state.orchestrator.clone();
    let missions = state.missions.clone();
    tokio::spawn(async move {
        run_mission_background(id, scenario, bbox, priority, callback, orchestrator, missions).await;
    });

    Json(json!({"status": "accepted", "mission_id": id}))
}

async fn list_missions(
    State(state): State<AppState>,
) -> Json<serde_json::Value> {
    let missions = state.missions.lock().unwrap();
    Json(json!({"missions": missions}))
}

async fn run_mission_background(
    id: String,
    scenario: String,
    bbox: Vec<serde_json::Value>,
    priority: u8,
    callback: Option<String>,
    orchestrator: Arc<Orchestrator>,
    missions: Arc<Mutex<Vec<MissionRecord>>>,
) {
    let report = orchestrator.execute_mission(OperatorScenario { id: id.clone(), scenario, bbox, priority }).await;
    let mut missions = missions.lock().unwrap();
    if let Some(rec) = missions.iter_mut().find(|r| r.id == id) {
        rec.status = "completed".into();
        rec.completed_at = Some(std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_secs());
        rec.report = Some(report);
    }
    if let Some(url) = callback {
        let _ = reqwest::Client::new().post(&url).timeout(std::time::Duration::from_secs(10)).json(&report).send().await;
    }
}
```
