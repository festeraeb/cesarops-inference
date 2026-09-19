=== FILE: index.html ===
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>NautiInferer</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/styles/github-dark.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.8.0/highlight.min.js"></script>
    <style>
        :root {
            --bg: #0a0a1a; --card: #1a1a2e; --accent: #00d4ff; --text: #e0e0e0;
            --text-dim: #a0a0a0; --green: #00ff88; --red: #ff4b4b;
        }
        body, html { margin: 0; padding: 0; height: 100%; font-family: 'Inter', -apple-system, sans-serif; background: var(--bg); color: var(--text); overflow: hidden; }
        #app { display: grid; grid-template-columns: 300px 1fr; grid-template-rows: 60px 1fr 60px; height: 100vh; }
        
        /* Header */
        header { grid-column: 1 / -1; background: var(--card); display: flex; align-items: center; justify-content: space-between; padding: 0 20px; border-bottom: 1px solid #333; }
        .logo { font-weight: 800; font-size: 1.2rem; letter-spacing: 1px; color: var(--accent); }
        .controls { display: flex; gap: 15px; align-items: center; }
        select { background: #0a0a1a; color: white; border: 1px solid #444; padding: 5px 10px; border-radius: 4px; outline: none; }

        /* Sidebar */
        aside { background: #121225; border-right: 1px solid #333; padding: 20px; display: flex; flex-direction: column; gap: 20px; }
        .stat-card { background: var(--card); padding: 15px; border-radius: 8px; border-left: 3px solid var(--accent); }
        .stat-label { font-size: 0.75rem; color: var(--text-dim); text-transform: uppercase; }
        .stat-value { font-size: 1.1rem; font-weight: bold; margin-top: 5px; }
        .toggle-container { display: flex; align-items: center; justify-content: space-between; cursor: pointer; }
        .switch { width: 40px; height: 20px; background: #444; border-radius: 10px; position: relative; transition: 0.3s; }
        .switch.active { background: var(--green); box-shadow: 0 0 10px var(--green); }
        .switch::after { content: ''; position: absolute; width: 16px; height: 16px; background: white; border-radius: 50%; top: 2px; left: 2px; transition: 0.3s; }
        .switch.active::after { left: 22px; }

        /* Chat Area */
        main { display: flex; flex-direction: column; position: relative; }
        #chat-container { flex: 1; overflow-y: auto; padding: 20px; display: flex; flex-direction: column; gap: 20px; scroll-behavior: smooth; }
        .msg { max-width: 85%; padding: 12px 16px; border-radius: 12px; line-height: 1.5; animation: fadeIn 0.3s ease; }
        .msg.user { align-self: flex-end; background: var(--accent); color: #000; }
        .msg.ai { align-self: flex-start; background: var(--card); border: 1px solid #333; }
        pre { background: #000 !important; padding: 10px; border-radius: 6px; overflow-x: auto; }
        code { font-family: 'Fira Code', monospace; font-size: 0.9rem; }

        /* Input Area */
        footer { grid-column: 2; padding: 20px; background: var(--bg); display: flex; gap: 10px; }
        #user-input { flex: 1; background: var(--card); border: 1px solid #444; color: white; padding: 12px; border-radius: 8px; outline: none; }
        #send-btn { background: var(--accent); border: none; padding: 0 20px; border-radius: 8px; cursor: pointer; font-weight: bold; }

        /* Status Bar */
        #status-bar { grid-column: 1 / -1; background: #050510; border-top: 1px solid #333; display: flex; align-items: center; padding: 0 20px; font-size: 0.8rem; color: var(--text-dim); gap: 20px; }
        .status-item span { color: var(--accent); font-weight: bold; }

        @keyframes fadeIn { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }
    </style>
</head>
<body>
    <div id="app">
        <header>
            <div class="logo">NAUTI<span style="color:white">INFERER</span></div>
            <div class="controls">
                <select id="model-select"><option>Loading models...</option></select>
                <div id="contributing-status" style="font-size: 0.8rem; color: var(--text-dim);">● Idle</div>
            </div>
        </header>

        <aside>
            <div class="toggle-container" onclick="toggleContribution()">
                <span style="font-size: 0.9rem;">Donate GPU</span>
                <div id="gpu-switch" class="switch"></div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Hardware</div>
                <div id="gpu-name" class="stat-value">Detecting...</div>
                <div id="gpu-vram" style="font-size: 0.8rem; color: var(--text-dim);">-- MB VRAM</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Credits Earned</div>
                <div id="credits-today" class="stat-value">0.00</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Mission Progress</div>
                <div id="mission-stat" style="font-size: 0.8rem; color: var(--text-dim); margin-top:5px;">0 tiles scanned</div>
            </div>
        </aside>

        <main>
            <div id="chat-container"></div>
            <footer>
                <input type="text" id="user-input" placeholder="Type your message..." onkeypress="if(event.key==='Enter') sendMessage()">
                <button id="send-btn" onclick="sendMessage()">SEND ▶</button>
            </footer>
        </main>

        <div id="status-bar">
            <div class="status-item">Fleet: <span id="fleet-nodes">0</span> nodes</div>
            <div class="status-item">Avg Speed: <span id="fleet-speed">0</span> tok/s</div>
            <div class="status-item">Active Jobs: <span id="fleet-jobs">0</span></div>
        </div>
    </div>

    <script>
        const { invoke } = window.__TAURI__.tauri;
        const { listen } = window.__TAURI__.event;

        let isContributing = false;

        async function init() {
            updateModels();
            updateStats();
            updateFleet();
            setInterval(updateStats, 5000);
            setInterval(updateFleet, 10000);
        }

        async function updateModels() {
            const models = await invoke('get_models');
            const select = document.getElementById('model-select');
            select.innerHTML = models.map(m => `<option value="${m.id}">${m.name} (${m.gpu_info})</option>`).join('');
        }

        async function updateStats() {
            const stats = await invoke('get_stats');
            document.getElementById('gpu-name').innerText = stats.gpu_name;
            document.getElementById('gpu-vram').innerText = `${stats.vram} MB VRAM | ${stats.temp}°C`;
            document.getElementById('credits-today').innerText = `${stats.credits_today.toFixed(2)} CR`;
            document.getElementById('mission-stat').innerText = `${stats.tiles_scanned} tiles scanned today`;
        }

        async function updateFleet() {
            const status = await invoke('get_fleet_status');
            document.getElementById('fleet-nodes').innerText = status.nodes;
            document.getElementById('fleet-speed').innerText = status.avg_speed;
            document.getElementById('fleet-jobs').innerText = status.active_jobs;
        }

        async function toggleContribution() {
            isContributing = !isContributing;
            const sw = document.getElementById('gpu-switch');
            const status = document.getElementById('contributing-status');
            
            await invoke('toggle_contribution', { enabled: isContributing });
            
            if(isContributing) {
                sw.classList.add('active');
                status.innerText = '● Contributing';
                status.style.color = 'var(--green)';
            } else {
                sw.classList.remove('active');
                status.innerText = '● Idle';
                status.style.color = 'var(--text-dim)';
            }
        }

        async function sendMessage() {
            const input = document.getElementById('user-input');
            const container = document.getElementById('chat-container');
            const prompt = input.value.trim();
            const model = document.getElementById('model-select').value;
            if(!prompt) return;

            // User Message
            const userDiv = document.createElement('div');
            userDiv.className = 'msg user';
            userDiv.innerText = prompt;
            container.appendChild(userDiv);
            input.value = '';
            container.scrollTop = container.scrollHeight;

            // AI Message Placeholder
            const aiDiv = document.createElement('div');
            aiDiv.className = 'msg ai';
            container.appendChild(aiDiv);

            try {
                // We use the Tauri command which handles the streaming via events
                await invoke('send_message', { prompt, model, endpoint: "https://api.cesarops.org/v1/inference" });
            } catch (e) {
                aiDiv.innerText = "Error: " + e;
            }
            
            container.scrollTop = container.scrollHeight;
        }

        // Listen for streaming tokens from Rust
        listen('stream-token', (event) => {
            const aiDiv = document.querySelector('.msg.ai:last-child');
            if (event.payload.finished) {
                aiDiv.innerHTML = aiDiv.innerHTML; // Finalize HTML
                hljs.highlightAll();
            } else {
                aiDiv.innerText += event.payload.delta;
                // Simple way to handle code blocks during stream
                const codeBlocks = aiDiv.querySelectorAll('pre code');
                codeBlocks.forEach(block => hljs.highlightElement(block));
            }
            const container = document.getElementById('chat-container');
            container.scrollTop = container.scrollHeight;
        });

        init();
    </script>
</body>
</html>

=== FILE: src-tauri/src/main.rs ===
#![cfg_ some_feature_flags]
use tauri::{Manager, Emitter};
use serde::{Serialize, Deserialize};
use reqwest::Client;
use futures_util::StreamExt;

#[derive(Serialize, Clone)]
struct StreamPayload {
    delta: String,
    finished: bool,
}

#[derive(Serialize)]
struct Model { id: String, name: String, gpu_info: String }

#[derive(Serialize)]
struct Stats { gpu_name: String, vram: u32, temp: u32, credits_today: f64, tiles_scanned: u32 }

#[derive(Serialize)]
struct FleetStatus { nodes: u32, avg_speed: f32, active_jobs: u32 }

#[tauri::command]
async fn get_models() -> Vec<Model> {
    // Mocking API response for demo
    vec![
        Model { id: "qwen-7b".into(), name: "Qwen-7B".into(), gpu_info: "2060S, 37 tok/s".into() },
        Model { id: "llama-3".into(), name: "Llama-3-8B".into(), gpu_info: "P100, 45 tok/s".into() },
    ]
}

#[tauri::command]
async fn get_stats() -> Stats {
    Stats { gpu_name: "NVIDIA RTX 3080".into(), vram: 10240, temp: 54, credits_today: 12.45, tiles_scanned: 142 }
}

#[tauri::command]
async fn get_fleet_status() -> FleetStatus {
    FleetStatus { nodes: 5, avg_speed: 37.2, active_jobs: 3 }
}

#[tauri::command]
async fn toggle_contribution(enabled: bool) -> Result<(), String> {
    println!("Contribution status: {}", enabled);
    Ok(())
}

#[tauri::command]
async fn send_message(window: tauri::Window, prompt: String, model: String, endpoint: String) -> Result<(), String> {
    let client = Client::new();
    let body = serde_json::json!({
        "model": model,
        "prompt": prompt,
        "max_tokens": 2048,
        "temperature": 0.4,
        "stream": true
    });

    let res = client.post(endpoint)
        .header("Accept", "text/event-stream")
        .json(&body)
        .send()
        .await
        .map_err(|e| e.to_string())?;

    let mut stream = res.bytes_stream();
    
    while let Some(item) = stream.next().await {
        let chunk = item.map_err(|e| e.to_string())?;
        let text = String::from_utf8_lossy(&chunk);
        
        // Parse SSE format: "data: {"delta": "...", "finished": false}"
        for line in text.lines() {
            if line.starts_with("data: ") {
                let json_str = &line[6..];
                if let Ok(payload) = serde_json::from_str::<StreamPayload>(json_str) {
                    window.emit("stream-token", &payload).unwrap();
                    if payload.finished { break; }
                }
            }
        }
    }
    Ok(())
}

fn main() {
    tauri::Builder::default()
        .invoke_handler(tauri::generate_handler![
            get_models, 
            get_stats, 
            get_fleet_status, 
            toggle_contribution, 
            send_message
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

=== FILE: src-tauri/Cargo.toml ===
[package]
name = "nauti-inferer"
version = "0.1.0"
edition = "2021"

[dependencies]
tauri = { version = "2.0.0-rc", features = [] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["json", "stream"] }
tokio = { version = "1", features = ["full"] }
futures-util = "0.3"

[build-dependencies]
tauri-build = "2.0.0-rc"

=== FILE: src-tauri/tauri.conf.json ===
{
  "productName": "NautiInferer",
  "version": "0.1.0",
  "identifier": "com.nauti.inferer",
  "bundle": {
    "active": true,
    "targets": "all",
    "icon": ["icons/32x32.png", "icons/128x128.png", "icons/icon.icns", "icons/icon.ico"]
  },
  "app": {
    "windows": [
      {
        "title": "NautiInferer",
        "width": 1200,
        "height": 800,
        "resizable": true,
        "decorations": true,
        "transparent": false
      }
    ],
    "security": {
      "csp": null
    }
  },
  "bundle": {
    "identifier": "com.nauti.inferer"
  }
}
