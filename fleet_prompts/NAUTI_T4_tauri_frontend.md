# Task: NautiInferer Tauri Desktop App — Production Chat + GPU Contributor

Write the complete frontend for a Tauri desktop application. This is a production-ready inference client that connects to the NautiInferer API.

## What it is:
A desktop app (like ChatGPT desktop or Claude desktop) that:
1. Lets users chat with AI models hosted on the NautiInferer fleet
2. Optionally donates their idle GPU to the fleet (SETI@home style)
3. Shows their contribution stats

## Tech stack:
- Tauri 2.x (Rust backend + web frontend)
- Frontend: vanilla HTML/CSS/JS (no React — keep it simple and fast)
- Backend: Rust (handles API calls, GPU monitoring, node daemon integration)

## Files to produce:

### 1. `src-tauri/src/main.rs` — Tauri backend
- Window setup (1200x800, dark theme, title "NautiInferer")
- Tauri commands:
  - `send_message(prompt, model, endpoint)` → streams response back via events
  - `get_models()` → fetches available models from API
  - `get_stats()` → returns contribution stats
  - `toggle_contribution(enabled)` → starts/stops local cesarops-node
  - `get_fleet_status()` → online nodes, active jobs

### 2. `src/index.html` — Main UI
Layout:
```
┌─────────────────────────────────────────────────┐
│  NautiInferer          [Model ▼] [⚡ Contributing]│
├─────────────────────────────────────────────────┤
│                                                 │
│  Chat messages (scrollable)                     │
│                                                 │
│  [AI]: Hello! I'm running on a P100 in the     │
│        CESAROPS fleet. How can I help?          │
│                                                 │
│  [You]: Write fibonacci in Rust                 │
│                                                 │
│  [AI]: ```rust                                  │
│        fn fibonacci(n: u32) -> u32 {            │
│            match n {                            │
│                0 => 0,                          │
│                ...                              │
│                                                 │
├─────────────────────────────────────────────────┤
│  [Type your message...              ] [Send ▶]  │
├─────────────────────────────────────────────────┤
│  ⚡ Fleet: 5 GPUs | 🔥 37 tok/s | 📊 142 tiles │
└─────────────────────────────────────────────────┘
```

### 3. Styling requirements:
- Dark theme: #0a0a1a bg, #1a1a2e cards, #00d4ff accent
- Smooth animations on message appear
- Code blocks with syntax highlighting (use highlight.js from CDN)
- Streaming tokens appear character by character
- Model selector dropdown with GPU info (e.g. "Qwen-7B (2060S, 37 tok/s)")
- Status bar at bottom showing fleet health
- "Contributing" toggle with green glow when active
- Professional — looks like a real product, not a hackathon demo

### 4. API integration:
- Endpoint: `https://api.cesarops.org/v1/inference` (or configurable)
- Streaming: POST with `Accept: text/event-stream`
- Request body: `{"model": "...", "prompt": "...", "max_tokens": 2048, "temperature": 0.4, "stream": true}`
- Response: SSE with `data: {"delta": "token", "finished": false}`
- Models list: `GET https://api.cesarops.org/v1/models`

### 5. GPU Contribution panel (collapsible sidebar):
- Toggle: "Donate GPU cycles when idle"
- Shows: GPU name, VRAM, temperature
- Credits earned today / total
- "Your GPU helped scan 12 satellite tiles today"
- "3 SAR missions supported this week"

## Output format:
```
=== FILE: index.html ===
(complete HTML with embedded CSS and JS)

=== FILE: src-tauri/src/main.rs ===
(Tauri backend with commands)

=== FILE: src-tauri/Cargo.toml ===
(dependencies)

=== FILE: src-tauri/tauri.conf.json ===
(window config)
```

## Constraints:
- index.html must be COMPLETELY self-contained (all CSS/JS inline)
- No npm, no build step for the frontend
- Use highlight.js from CDN for code blocks
- Streaming via fetch() ReadableStream (not EventSource — more control)
- Under 500 lines for index.html
- Under 100 lines for main.rs
- Must look professional — this is for a Wednesday demo
