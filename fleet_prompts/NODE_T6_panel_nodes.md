# Task: Add DII Node Registry section to cluster_panel.html

You are a frontend developer. Add a new section to the CESAROPS cluster panel that shows all registered DII nodes with their GPU info.

## Context

The forge serves `cluster_panel.html` at GET /cluster. It already has sections for workers, presets, and wreck detection. Add a new section at the TOP (before existing content) that shows the live node registry.

## API endpoint

`GET /cluster/nodes` returns:
```json
[
  {
    "node_id": "cesarops2",
    "hardware": {
      "gpu": "Quadro P1000",
      "vram_mb": 4096,
      "backend": "cuda",
      "all_gpus": [
        {"id": 0, "name": "NVIDIA GeForce GTX 1070", "vram_total_mb": 8192},
        {"id": 1, "name": "Quadro P1000", "vram_total_mb": 4096}
      ]
    },
    "available_models": ["TinyLlama-1.1B-Chat-v1.0-Q4_K_M.gguf", ...],
    "listen_port": 5500,
    "last_heartbeat": {
      "state": "serving",
      "model": "TinyLlama...",
      "port": 5571,
      "gpu": {"vram_used_mb": 397, "vram_total_mb": 4096, "temp_c": 31, "util_pct": 0},
      "all_gpus": [
        {"id": 0, "name": "GTX 1070", "vram_used_mb": 654, "vram_total_mb": 8192, "temp_c": 43, "util_pct": 0},
        {"id": 1, "name": "P1000", "vram_used_mb": 397, "vram_total_mb": 4096, "temp_c": 31, "util_pct": 0}
      ]
    },
    "last_seen_secs_ago": 5,
    "online": true
  }
]
```

## Output: HTML section

Write a self-contained HTML section (div) with:
1. Title: "🖥️ DII Node Fleet"
2. Auto-refresh every 10 seconds
3. For each node, show a card with:
   - Node name + online/offline badge (green/red dot)
   - Last seen: "5s ago"
   - State badge (idle=gray, serving=green, loading=yellow, error=red)
   - Active model (if serving)
   - For EACH GPU on the node: name, VRAM bar (used/total), temperature, utilization %
   - Available models count
4. Style: dark theme (bg #1a1a2e, cards #16213e, text #e0e0e0, accent #0f3460)
5. Use vanilla JS fetch(), no frameworks

## Output format

Output ONLY the HTML section (a single `<div id="dii-fleet">...</div>` with embedded `<style>` and `<script>`). No full page, no doctype, no body tags.

Keep it under 150 lines.
