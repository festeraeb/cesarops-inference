You are a HTML + JavaScript specialist. Add wreck-detection control buttons to the cluster_panel.html for the cesarops-forge-v2 web UI.

## Context

The forge cluster panel at /cluster currently has worker-control buttons (start/stop/configure each worker). We just added 7 new wreck-detection forge tools (scan_region, magnetic_dipole_detect, download_satellite_window, weather_window, detection_health, detection_scan, detection_poll) plus a coding-mode toggle.

We need a wreck-detection control area on the cluster panel — quick-access buttons for common operator workflows.

## Existing pattern

The panel uses vanilla JS fetch() POST to forge endpoints. For example the existing worker start button:

```html
<button onclick="startWorker(0)">Start GemmaBig</button>
<script>
async function startWorker(idx) {
    const r = await fetch(`/cluster/worker/${idx}/start`, {method: 'POST'});
    const d = await r.json();
    alert(d.message || d.error);
}
</script>
```

## Deliverable

Append a "Wreck Detection / SAR" section at the END of cluster_panel.html (just before `</body>`). Should include:

### Section 1: Detection Service Status
- Button: "Check Detection Health"
  - Calls GET /tool/detection_health (no args)
  - Shows result in a `<pre id="detection-health-output">` block
  - Updates the status display every 30s when the panel is open

### Section 2: Quick Region Scan
- Form fields (text inputs):
  - bbox: "lat_min,lon_min,lat_max,lon_max" (default: "41.0,-83.5,42.5,-82.0" — Lake Erie central)
  - days: number (default: 14)
  - mode: dropdown (wreck | sar | downed_aircraft) — default "wreck"
- Button: "Scan Region"
  - POST to /tool/scan_region with form values
  - Shows progress + truncated result in `<pre id="scan-output">` block

### Section 3: Weather Window Check
- Form field: bbox (same default)
- Dropdown: check (post_storm | calm | any)
- Button: "Check Weather Window"
  - POST to /tool/weather_window
  - Shows result + recommendation in `<pre id="weather-output">` block

### Section 4: Triple-Lock Detection Submission
- Form fields:
  - region: text input
  - tiles: textarea for JSON array of {lat, lon, image_b64}
- Button: "Submit Detection Job"
  - POST to /tool/detection_scan
  - Shows job_id in `<pre id="detection-submit-output">` block
  - On success, store job_id in a hidden field
- Button: "Poll Last Job"
  - Reads stored job_id, calls /tool/detection_poll
  - Shows result in `<pre id="detection-poll-output">` block

## IMPORTANT

The forge does NOT currently have a /tool/<name> endpoint that takes JSON args and returns the tool result. The chat /send is the only way to invoke tools today, and that goes through the AI orchestration loop.

For the panel buttons to work, we need a NEW endpoint:
  POST /tool/<name>
    body: { "arguments": { ... tool-specific args ... } }
    returns: { "result": "<tool-output-string>" }

This bypasses the AI loop and calls tools::execute() directly with the supplied args. Useful for human-driven tool invocation from the panel without going through chat.

So the deliverable is THREE parts:

### Part 1: Add /tool/{name} endpoint to cesarops-forge-v2/src/main.rs

```rust
async fn invoke_tool(
    State(state): State<AppState>,
    axum::extract::Path(name): axum::extract::Path<String>,
    Json(body): Json<serde_json::Value>,
) -> Json<serde_json::Value> {
    let args = body.get("arguments").cloned().unwrap_or(serde_json::json!({}));
    info!("Direct tool invocation: {} args={}", name, args);
    let result = tools::execute(&name, &args, &state).await;
    Json(serde_json::json!({"result": result, "tool": name}))
}
```

Register: `.route("/tool/{name}", post(invoke_tool))`

### Part 2: HTML + JS append block to cluster_panel.html

Use existing CSS classes if any (look at existing buttons for styling). Group sections in a single `<div class="wreck-detection-panel">` with a heading.

### Part 3: NO backend Python changes needed

The four Python tools (scan_engine, universal_downloader, weather_service) are out-of-scope for this task — they're handled separately. Just wire the HTTP layer.

## Constraints

- Vanilla JS only — no new frameworks, no jQuery
- All fetches use async/await
- Errors get displayed (don't swallow them)
- The panel must still work with existing worker controls (don't break anything)
- Use `<details>` collapsible sections so the panel doesn't get visually overwhelming

## Output format

```
=== DIFF: cesarops-forge-v2/src/main.rs ===
// the invoke_tool fn + route registration

=== APPEND: cesarops-forge-v2/src/cluster_panel.html ===
// the HTML + JS block to add at end of panel
```

Code only. No preamble.
