```rust
async fn llm_refine_plan(heuristic_plan: MissionPlan, scenario: &OperatorScenario) -> MissionPlan {
    let client = reqwest::Client::builder().timeout(std::time::Duration::from_secs(15)).build().unwrap();
    let endpoints = [INTAKE_ENDPOINT, INTAKE_FALLBACK];
    
    let prompt = format!(
        "You are a SAR mission planner. Given this scenario and initial plan, output ONLY a JSON object with optional overrides.\n\nScenario: {}\nBBox: {:?}\n\nCurrent plan:\n- Class: {:?}\n- Modules: {:?}\n- Stitching: {:?}\n\nIf the plan looks correct, output: {{\"action\": \"accept\"}}\nIf you want to add a module, output: {{\"action\": \"add_module\", \"module\": {{\"id\": \"...\", \"name\": \"...\", \"tool_name\": \"...\", \"tool_args\": {{...}}}}}}\nIf you want to change the scenario class, output: {{\"action\": \"reclassify\", \"class\": \"WreckHunt|DownedAircraft|SearchRescue|...\"}}\n\nOutput ONLY valid JSON, nothing else.",
        scenario.raw_text, scenario.bbox, heuristic_plan.class, 
        heuristic_plan.modules.iter().map(|m| &m.name).collect::<Vec<_>>(),
        heuristic_plan.stitching_summary
    );

    let payload = serde_json::json!({
        "prompt": prompt,
        "max_length": 256,
        "temperature": 0.1,
        "stop_sequence": ["\n\n", "}}\n"]
    });

    for url in endpoints {
        if let Ok(resp) = client.post(format!("{}/api/v1/generate", url)).json(&payload).send().await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                if let Some(text) = json["results"][0]["text"].as_str() {
                    if let Ok(action) = serde_json::from_str::<serde_json::Value>(text.trim()) {
                        let mut refined = heuristic_plan.clone();
                        match action["action"].as_str() {
                            Some("accept") => return refined,
                            Some("add_module") => {
                                if let Ok(m) = serde_json::from_value::<Module>(action["module"].clone()) {
                                    refined.modules.push(m);
                                }
                            },
                            Some("reclassify") => {
                                if let Some(new_class) = action["class"].as_str() {
                                    refined.class = new_class.to_string();
                                }
                            },
                            _ => return heuristic_plan,
                        }
                        return refined;
                    }
                }
            }
        }
    }
    heuristic_plan
}

// Updated execute_mission snippet:
let mut plan = retry_plan(&scenario).await?;
let mut llm_applied = false;

let original_plan = plan.clone();
plan = llm_refine_plan(plan, &scenario).await;
if plan != original_plan {
    llm_applied = true;
}

if llm_applied {
    notes.push("Mission plan refined by LLM intelligence.".to_string());
} else {
    notes.push("Heuristic planning used (LLM refinement skipped/failed).".to_string());
}

assign_specialists(&mut plan, &scenario).await?;
```
