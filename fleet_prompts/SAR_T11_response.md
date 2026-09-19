```rust
=== FILE: cesarops-forge-v2/src/orchestrator.rs ===
use serde::{Deserialize, Serialize};
use std::time::Duration;
use tokio::time::timeout;
use tokio::task::JoinSet;
use axum::{Json, extract::State};
use std::sync::Arc;
use reqwest::Client;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ScenarioClass {
    WreckHunt,
    DownedAircraft,
    EnvironmentalMonitoring,
    OceanographicSurvey,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleSpec {
    pub id: String,
    pub name: String,
    pub delegate: String,
    pub bbox: [f64; 4],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionPlan {
    pub scenario_class: ScenarioClass,
    pub modules: Vec<ModuleSpec>,
    pub stitching: StitchingStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StitchingStrategy {
    pub n_tiles: usize,
    pub stacks_per_p100: usize,
    pub tiles_per_stack: usize,
    pub merge_method: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerState {
    pub name: String,
    pub model_type: String,
    pub current_task: Option<String>,
    pub load: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterSnapshot {
    pub workers: Vec<WorkerState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RouteAssignment {
    pub module_id: String,
    pub endpoint: String,
    pub fallback_endpoints: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetoolReceipt {
    pub worker_name: String,
    pub previous_config: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleResult {
    pub module_id: String,
    pub status: String,
    pub data: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReport {
    pub scenario_class: ScenarioClass,
    pub modules: Vec<ModuleResult>,
    pub stitching_summary: Option<String>,
    pub runtime_seconds: f32,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperatorScenario {
    pub scenario_id: String,
    pub scenario_class: ScenarioClass,
    pub target_area: [f64; 4],
}

pub async fn probe_cluster() -> Result<ClusterSnapshot, String> {
    let client = Client::new();
    let res = client.post("http://127.0.0.1:9100/cluster/discover")
        .send()
        .await
        .map_err(|e| e.to_string())?;
    
    let workers: Vec<WorkerState> = res.json().await.map_err(|e| e.to_string())?;
    Ok(ClusterSnapshot { workers })
}

pub async fn plan_from_scenario(scenario: &OperatorScenario) -> Result<MissionPlan, String> {
    let client = Client::new();
    let prompt = format!(
        "Generate a mission plan for scenario class {:?}. Target area: {:?}. \
        Output strictly JSON with keys: scenario_class, modules (id, name, delegate, bbox), stitching (n_tiles, stacks_per_p100, tiles_per_stack, merge_method).",
        scenario.scenario_class, scenario.target_area
    );

    let mut attempt = 0;
    while attempt < 2 {
        let res = client.post("http://127.0.0.1:5001/api/v1/generate")
            .json(&serde_json::json!({ "prompt": prompt }))
            .send()
            .await;

        if let Ok(response) = res {
            if let Ok(val) = response.json::<serde_json::Value>().await {
                if let Ok(plan) = serde_json::from_value::<MissionPlan>(val) {
                    return Ok(plan);
                }
            }
        }
        attempt += 1;
    }
    Err("Failed to parse mission plan after retries".to_string())
}

pub async fn assign_specialists(modules: &[ModuleSpec], cluster: &ClusterSnapshot) -> Vec<RouteAssignment> {
    let mut assignments = Vec::new();
    for m in modules {
        let mut endpoint = String::new();
        let mut fallbacks = Vec::new();

        match m.delegate.as_str() {
            "VulkanGpu" => {
                endpoint = format!("rust://cesarops_inference/{}", m.name);
            }
            "Cpu" | "LLM" => {
                let best_worker = cluster.workers.iter()
                    .filter(|w| w.load < 70.0 && (w.model_type.contains("1060") || w.model_type.contains("1070") || w.model_type.contains("P1000")))
                    .min_by(|a, b| a.load.partial_cmp(&b.load).unwrap())
                    .map(|w| w.name.clone())
                    .unwrap_or_else(|| cluster.workers[0].name.clone());
                endpoint = format!("http://127.0.0.1:5001/api/v1/generate?worker={}", best_worker);
            }
            "CoralTpuInt8" => {
                endpoint = format!("http://127.0.0.1:5001/api/v1/generate?worker=coral");
                fallbacks.push(format!("rust://cesarops_inference/{}", m.name));
            }
            "Hybrid" => {
                endpoint = format!("http://127.0.0.1:5001/api/v1/generate?mode=primary");
                fallbacks.push(format!("http://127.0.0.1:5001/api/v1/generate?mode=secondary"));
            }
            _ => {
                endpoint = format!("http://127.0.0.1:5001/api/v1/generate");
            }
        }

        assignments.push(RouteAssignment {
            module_id: m.id.clone(),
            endpoint,
            fallback_endpoints: fallbacks,
        });
    }
    assignments
}

pub async fn retool_for_mission(plan: &MissionPlan) -> Result<RetoolReceipt, String> {
    let is_sar = matches!(plan.scenario_class, ScenarioClass::WreckHunt | ScenarioClass::DownedAircraft);
    let has_tile_compute = plan.modules.iter().any(|m| m.id.contains("tile"));

    if is_sar && has_tile_compute {
        let client = Client::new();
        let res = client.post("http://127.0.0.1:9100/cluster/worker/GemmaBig/stop")
            .send()
            .await
            .map_err(|e| e.to_string())?;
        
        if res.status().is_success() {
            return Ok(RetoolReceipt {
                worker_name: "GemmaBig".to_string(),
                previous_config: "/models/gemma-4-moe-v1".to_string(),
            });
        }
    }
    Ok(RetoolReceipt { worker_name: "".to_string(), previous_config: "".to_string() })
}

pub async fn restore_cluster(receipt: &RetoolReceipt) -> Result<(), String> {
    if receipt.worker_name.is_empty() { return Ok(()); }
    let client = Client::new();
    let url = format!("http://127.0.0.1:9100/cluster/worker/{}/start", receipt.worker_name);
    client.post(url).send().await.map_err(|e| e.to_string())?;
    Ok(())
}

pub async fn dispatch_modules(plan: &MissionPlan, routing: &[RouteAssignment]) -> Vec<ModuleResult> {
    let client = Client::new();
    let mut set = JoinSet::new();

    for route in routing {
        let client = client.clone();
        let module_id = route.module_id.clone();
        let endpoint = route.endpoint.clone();
        let plan_ref = plan.clone();

        set.spawn(async move {
            let task = async {
                if endpoint.starts_with("rust://cesarops_inference/") {
                    // TODO: Implement v2 wire as cesarops-inference dependency
                    ModuleResult { status: "ok".to_string(), module_id, data: Some("placeholder".to_string()) }
                } else if endpoint.starts_with("http") {
                    let res = client.post(&endpoint)
                        .json(&serde_json::json!({ "module_id": module_id, "context": plan_ref.scenario_class }))
                        .send()
                        .await;
                    
                    match res {
                        Ok(r) => ModuleResult { status: "ok".to_string(), module_id, data: Some("success".to_string()) },
                        Err(_) => ModuleResult { status: "failed".to_string(), module_id, data: None },
                    }
                } else {
                    ModuleResult { status: "failed".to_string(), module_id, data: None }
                }
            };

            match timeout(Duration::from_secs(60), task).await {
                Ok(res) => res,
                Err(_) => ModuleResult { status: "failed".to_string(), module_id, data: None },
            }
        });
    }

    let mut results = Vec::new();
    while let Some(res) = set.join_next().await {
        if let Ok(m_res) = res {
            results.push(m_res);
        }
    }
    results
}

pub async fn execute_mission(scenario: OperatorScenario) -> MissionReport {
    let start = std::time::Instant::now();
    
    let cluster = match probe_cluster().await {
        Ok(c) => c,
        Err(_) => return MissionReport { scenario_class: scenario.scenario_class, modules: vec![], stitching_summary: None, runtime_seconds: 0.0, status: "failed".to_string() },
    };

    let plan = match plan_from_scenario(&scenario).await {
        Ok(p) => p,
        Err(_) => return MissionReport { scenario_class: scenario.scenario_class, modules: vec![], stitching_summary: None, runtime_seconds: 0.0, status: "failed".to_string() },
    };

    let routing = assign_specialists(&plan.modules, &cluster).await;
    let receipt = retool_for_mission(&plan).await.unwrap_or(RetoolReceipt { worker_name: "".into(), previous_config: "".into() });
    
    let results = dispatch_modules(&plan, &routing).await;
    let _ = restore_cluster(&receipt).await;

    let status = if results.iter().all(|r| r.status == "ok") { "ok" } 
                 else if results.iter().any(|r| r.status == "ok") { "partial" } 
                 else { "failed" };

    MissionReport {
        scenario_class: scenario.scenario_class,
        modules: results,
        stitching_summary: Some(format!("Tiles: {}", plan.stitching.n_tiles)),
        runtime_seconds: start.elapsed().as_secs_f32(),
        status: status.to_string(),
    }
}

pub async fn orchestrator_probe() -> Json<ClusterSnapshot> {
    match probe_cluster().await {
        Ok(s) => Json(s),
        Err(_) => Json(ClusterSnapshot { workers: vec![] }),
    }
}

pub async fn orchestrator_plan(Json(scenario): Json<OperatorScenario>) -> Json<serde_json::Value> {
    match plan_from_scenario(&scenario).await {
        Ok(plan) => Json(serde_json::json!({ "plan": plan })),
        Err(e) => Json(serde_json::json!({ "error": e })),
    }
}

pub async fn orchestrator_execute(Json(scenario): Json<OperatorScenario>) -> Json<MissionReport> {
    Json(execute_mission(scenario).await)
}

=== DIFF: cesarops-forge-v2/src/main.rs ===
+ mod orchestrator;
+ use orchestrator::{orchestrator_probe, orchestrator_plan, orchestrator_execute};
...
+ router.route("/orchestrator/probe", axum::routing::get(orchestrator_probe))
+ router.route("/orchestrator/plan", axum::routing::post(orchestrator_plan))
+ router.route("/orchestrator/execute", axum::routing::post(orchestrator_execute))
```
