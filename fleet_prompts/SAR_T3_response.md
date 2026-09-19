```rust
=== DIFF: cesarops-forge-v2/src/main.rs ===
// Add this function to your main.rs
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

// In your router definition, add the route:
// .route("/tool/:name", post(invoke_tool))

=== APPEND: cesarops-forge-v2/src/cluster_panel.html ===
<hr style="margin: 40px 0; border: 1px solid #444;">

<div class="wreck-detection-panel" style="font-family: sans-serif; color: #eee; background: #1a1a1a; padding: 20px; border-radius: 8px;">
    <h2 style="margin-top: 0; color: #00d4ff; border-bottom: 1px solid #333; padding-bottom: 10px;">Wreck Detection / SAR Control</h2>

    <!-- Section 1: Health -->
    <details open style="margin-bottom: 15px; border: 1px solid #333; padding: 10px; border-radius: 4px;">
        <summary style="cursor: pointer; font-weight: bold; color: #aaa;">1. Detection Service Status</summary>
        <div style="margin-top: 10px;">
            <button onclick="checkDetectionHealth()" style="padding: 5px 15px; cursor: pointer;">Check Detection Health</button>
            <pre id="detection-health-output" style="background: #000; color: #0f0; padding: 10px; margin-top: 10px; min-height: 20px; font-size: 0.85em; overflow-x: auto;"></pre>
        </div>
    </details>

    <!-- Section 2: Quick Scan -->
    <details style="margin-bottom: 15px; border: 1px solid #333; padding: 10px; border-radius: 4px;">
        <summary style="cursor: pointer; font-weight: bold; color: #aaa;">2. Quick Region Scan</summary>
        <div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="grid-column: span 2;">
                <label style="display:block; font-size: 0.8em;">BBOX (lat_min,lon_min,lat_max,lon_max)</label>
                <input type="text" id="scan-bbox" value="41.0,-83.5,42.5,-82.0" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px;">
            </div>
            <div>
                <label style="display:block; font-size: 0.8em;">Days</label>
                <input type="number" id="scan-days" value="14" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px;">
            </div>
            <div>
                <label style="display:block; font-size: 0.8em;">Mode</label>
                <select id="scan-mode" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px;">
                    <option value="wreck">Wreck</option>
                    <option value="sar">SAR</option>
                    <option value="downed_aircraft">Downed Aircraft</option>
                </select>
            </div>
        </div>
        <button onclick="runScan()" style="margin-top: 10px; width: 100%; padding: 8px; cursor: pointer;">Scan Region</button>
        <pre id="scan-output" style="background: #000; color: #0af; padding: 10px; margin-top: 10px; max-height: 150px; overflow-y: auto; font-size: 0.85em;"></pre>
    </details>

    <!-- Section 3: Weather -->
    <details style="margin-bottom: 15px; border: 1px solid #333; padding: 10px; border-radius: 4px;">
        <summary style="cursor: pointer; font-weight: bold; color: #aaa;">3. Weather Window Check</summary>
        <div style="margin-top: 10px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
            <div style="grid-column: span 2;">
                <label style="display:block; font-size: 0.8em;">BBOX</label>
                <input type="text" id="weather-bbox" value="41.0,-83.5,42.5,-82.0" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px;">
            </div>
            <div style="grid-column: span 2;">
                <label style="display:block; font-size: 0.8em;">Condition</label>
                <select id="weather-condition" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px;">
                    <option value="post_storm">Post Storm</option>
                    <option value="calm">Calm</option>
                    <option value="any">Any</option>
                </select>
            </div>
        </div>
        <button onclick="checkWeather()" style="margin-top: 10px; width: 100%; padding: 8px; cursor: pointer;">Check Weather Window</button>
        <pre id="weather-output" style="background: #000; color: #ff0; padding: 10px; margin-top: 10px; max-height: 150px; overflow-y: auto; font-size: 0.85em;"></pre>
    </details>

    <!-- Section 4: Triple-Lock -->
    <details style="margin-bottom: 15px; border: 1px solid #333; padding: 10px; border-radius: 4px;">
        <summary style="cursor: pointer; font-weight: bold; color: #aaa;">4. Triple-Lock Detection Submission</summary>
        <input type="hidden" id="last-job-id" value="">
        
        <div style="margin-top: 10px;">
            <label style="display:block; font-size: 0.8em;">Region (BBOX)</label>
            <input type="text" id="submit-region" placeholder="41.0,-83.5,42.5,-82.0" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px; margin-bottom: 10px;">
            
            <label style="display:block; font-size: 0.8em;">Tiles (JSON Array of {lat, lon, image_b64})</label>
            <textarea id="submit-tiles" rows="5" style="width: 100%; background: #222; color: #fff; border: 1px solid #444; padding: 4px; font-family: monospace; font-size: 0.8em;"></textarea>
        </div>
        
        <div style="display: flex; gap: 10px; margin-top: 10px;">
            <button onclick="submitDetectionJob()" style="flex: 1; padding: 8px; cursor: pointer;">Submit Job</button>
            <button onclick="pollLastJob()" style="flex: 1; padding: 8px; cursor: pointer;">Poll Last Job</button>
        </div>
        
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-top: 10px;">
            <div>
                <small style="color: #888;">Submit Result:</small>
                <pre id="detection-submit-output" style="background: #000; color: #0f0; padding: 5px; font-size: 0.75em; min-height: 20px;"></pre>
            </div>
            <div>
                <small style="color: #888;">Poll Result:</small>
                <pre id="detection-poll-output" style="background: #000; color: #0af; padding: 5px; font-size: 0.75em; min-height: 20px;"></pre>
            </div>
        </div>
    </details>
</div>

<script>
(function() {
    // Auto-poll health every 30s
    setInterval(checkDetectionHealth, 30000);

    async function callTool(name, args) {
        try {
            const r = await fetch(`/tool/${name}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ arguments: args })
            });
            const d = await r.json();
            if (d.error) throw new Error(d.error);
            return d.result;
        } catch (e) {
            console.error(`Tool ${name} failed:`, e);
            throw e;
        }
    }

    window.checkDetectionHealth = async function() {
        const out = document.getElementById('detection-health-output');
        try {
            // Note: The requirement says GET /tool/detection_health, 
            // but our new endpoint is POST /tool/{name}. 
            // We'll use POST with empty args to match the new pattern.
            const res = await callTool('detection_health', {});
            out.textContent = typeof res === 'object' ? JSON.stringify(res, null, 2) : res;
            out.style.color = '#0f0';
        } catch (e) {
            out.textContent = "Error: " + e.message;
            out.style.color = '#f00';
        }
    };

    window.runScan = async function() {
        const out = document.getElementById('scan-output');
        out.textContent = "Scanning...";
        const args = {
            bbox: document.getElementById('scan-bbox').value,
            days: parseInt(document.getElementById('scan-days').value),
            mode: document.getElementById('scan-mode').value
        };
        try {
            const res = await callTool('scan_region', args);
            out.textContent = typeof res === 'object' ? JSON.stringify(res, null, 2) : res;
            out.style.color = '#0af';
        } catch (e) {
            out.textContent = "Error: " + e.message;
            out.style.color = '#f00';
        }
    };

    window.checkWeather = async function() {
        const out = document.getElementById('weather-output');
        out.textContent = "Checking weather...";
        const args = {
            bbox: document.getElementById('weather-bbox').value,
            condition: document.getElementById('weather-condition').value
        };
        try {
            const res = await callTool('weather_window', args);
            out.textContent = typeof res === 'object' ? JSON.stringify(res, null, 2) : res;
            out.style.color = '#ff0';
        } catch (e) {
            out.textContent = "Error: " + e.message;
            out.style.color = '#f00';
        }
    };

    window.submitDetectionJob = async function() {
        const out = document.getElementById('detection-submit-output');
        const jobInput = document.getElementById('last-job-id');
        out.textContent = "Submitting...";
        
        let tiles;
        try {
            tiles = JSON.parse(document.getElementById('submit-tiles').value);
        } catch (e) {
            out.textContent = "Error: Invalid JSON in tiles field";
            out.style.color = '#f00';
            return;
        }

        const args = {
            region: document.getElementById('submit-region').value,
            tiles: tiles
        };

        try {
            const res = await callTool('detection_scan', args);
            // Expecting { job_id: "..." }
            if (res && res.job_id) {
                jobInput.value = res.job_id;
                out.textContent = "Success! Job ID: " + res.job_id;
                out.style.color = '#0f0';
            } else {
                out.textContent = "Success, but no job_id returned: " + JSON.stringify(res);
                out.style.color = '#0f0';
            }
        } catch (e) {
            out.textContent = "Error: " + e.message;
            out.style.color = '#f00';
        }
    };

    window.pollLastJob = async function() {
        const out = document.getElementById('detection-poll-output');
        const jobId = document.getElementById('last-job-id').value;
        if (!jobId) {
            out.textContent = "No job_id stored.";
            out.style.color = '#f00';
            return;
        }
        out.textContent = "Polling...";
        try {
            const res = await callTool('detection_poll', { job_id: jobId });
            out.textContent = typeof res === 'object' ? JSON.stringify(res, null, 2) : res;
            out.style.color = '#0af';
        } catch (e) {
            out.textContent = "Error: " + e.message;
            out.style.color = '#f00';
        }
    };

    // Initial health check
    checkDetectionHealth();
})();
</script>
```
