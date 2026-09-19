# Task: Build CESAROPS Forge IDE — Single HTML File with Monaco + Twin Chat

Write a complete single-file HTML application that serves as the CESAROPS Forge IDE. It will be served by the forge at `GET /ide`.

## Layout (flexbox, dark theme):

```
┌──────────────────────────────────────────────────────────────┐
│ Header: CESAROPS FORGE IDE  [Mode: CESAROPS/CODING] [Nodes:3]│
├────────┬─────────────────────────────┬───────────────────────┤
│ File   │  Monaco Editor              │  Chat Panel A         │
│ Tree   │  (center, 50% width)        │  [GPU selector ▼]    │
│        │                             │  [streaming tokens]   │
│        │                             ├───────────────────────┤
│        │                             │  Chat Panel B         │
│        │                             │  [GPU selector ▼]     │
│        │                             │  [streaming tokens]   │
├────────┴─────────────────────────────┴───────────────────────┤
│ Terminal / Build Output (bottom 20%)                          │
└──────────────────────────────────────────────────────────────┘
```

## Features:

### 1. Monaco Editor (center)
- Load from CDN: `https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs`
- Supports: rust, python, typescript, json, markdown, toml
- File tabs (click file tree to open)
- Read files via: `GET /ide/file?path=/home/cesarops/wreckhunter2000-1/src/main.rs`
- Save files via: `POST /ide/file` body: `{"path": "...", "content": "..."}`

### 2. File Tree (left sidebar, 15% width)
- Load via: `GET /ide/tree?root=/home/cesarops/wreckhunter2000-1`
- Returns: `[{"name": "src", "type": "dir", "children": [...]}, {"name": "Cargo.toml", "type": "file"}]`
- Click file → opens in Monaco
- Click dir → expands/collapses

### 3. Twin Chat Panels (right, stacked vertically)
Each panel has:
- GPU/model selector dropdown (populated from `GET /cluster/nodes`)
- Message input + Send button
- Conversation history with streaming token display
- **Streaming**: Use `POST /ide/chat/stream` with `Accept: text/event-stream`
  - Body: `{"prompt": "...", "endpoint": "http://127.0.0.1:5001", "max_length": 2048, "temperature": 0.4}`
  - Server proxies to koboldcpp `/api/extra/generate/stream` and forwards SSE tokens
- "Send to Both" button fires same prompt to both panels simultaneously
- "Compare" button shows diff of both responses in Monaco
- "Apply" button writes selected response to current file in editor

### 4. Terminal (bottom)
- WebSocket to `ws://localhost:9100/ide/terminal`
- Basic xterm.js-style output (just show stdout/stderr, no full PTY needed for v1)
- Run button sends command via: `POST /ide/exec` body: `{"command": "cargo build --release"}`
- Shows output streamed back

### 5. Header Bar
- Current mode badge (from `GET /mode`)
- Online nodes count (from `GET /cluster/nodes`)
- "Switch Mode" button (POST /mode/cesarops or /mode/coding)

## Styling:
- Dark theme: bg #0f0f1a, panels #1a1a2e, borders #2d2d44, text #e0e0e0
- Accent: #00d4ff (cyan)
- Success: #4caf50, Warning: #ff9800, Error: #ff4b2b
- Font: 'JetBrains Mono', 'Fira Code', monospace for code; system-ui for UI
- Responsive: min-width 1200px

## CDN dependencies (load in <head>):
- Monaco: `https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs`
- No React, no build step. Vanilla JS + DOM manipulation.

## Output:
A single complete HTML file. Include all CSS in a `<style>` block and all JS in a `<script>` block. The file should be self-contained and work when served at any URL.

## Constraints:
- Under 800 lines total
- No external JS frameworks (no React, no Vue, no jQuery)
- Monaco loaded via AMD loader from CDN
- All API calls use fetch()
- Streaming uses EventSource or fetch with ReadableStream
- File tree is lazy-loaded (only expand dirs on click)
- Chat history stored in JS arrays (not persisted)
- Each chat panel maintains independent conversation context
