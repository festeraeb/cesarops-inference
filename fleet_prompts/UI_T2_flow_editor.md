# Task: Visual Flow Editor for CESAROPS Forge (Node-RED style)

Write a self-contained HTML/JS component that provides a visual pipeline editor for the CESAROPS cluster. This will be embedded in the forge IDE page.

## What it does:

The operator drags GPU/model nodes onto a canvas, connects them with lines to define the inference pipeline, then hits "Run" to execute.

## Node types (draggable from a palette):

1. **GPU Node** — represents a physical GPU in the cluster
   - Shows: name, model loaded, VRAM bar, temp, tok/s
   - Populated from `GET /cluster/nodes`
   - Color-coded: green=online, red=offline, yellow=loading

2. **Model Node** — a model that can be assigned to a GPU
   - Shows: model name, size, quant type
   - Draggable onto a GPU node to assign it

3. **Task Node** — a processing step
   - Types: "Code", "Review", "Correct", "Compile", "Think", "Scan", "Detect"
   - Has input/output ports for connecting

4. **Input Node** — where the prompt/task enters
   - Has a text area for the prompt
   - Or "From webhook" / "From n8n" source selector

5. **Output Node** — where results go
   - "To file", "To chat panel", "To webhook callback"

## Connections (lines between nodes):

- Drag from output port → input port
- Lines show data flow direction (arrow)
- Line color indicates: blue=idle, green=active, red=error
- Click a line to see what data passed through it

## Example pipeline the operator would build:

```
[Input: "Write fibonacci in Rust"]
    ↓
[Task: Code] → assigned to [GPU: P100#0 / Gemma]
    ↓
[Task: Review] → assigned to [GPU: 2060 Super / DeepSeek-R1]
    ↓
[Task: Correct] → assigned to [GPU: P100#1 / Qwen]
    ↓
[Task: Compile] → runs `cargo build`
    ↓
[Output: To file → src/fibonacci.rs]
```

## Execution:

When "Run Pipeline" is clicked:
1. Walk the graph from Input → Output
2. For each Task node, POST to the assigned GPU's endpoint with the prompt + context from previous node
3. Stream tokens back to the node (show them live on the canvas)
4. Pass output to next node in the chain
5. Final output goes to the Output node's destination

## API calls:
- `GET /cluster/nodes` — populate GPU nodes
- `POST /ide/chat/stream` — execute inference on a specific endpoint
- `POST /ide/exec` — run compile/shell commands
- `POST /ide/file` — write output to file

## Implementation:

Use vanilla JS with a `<canvas>` element for the flow editor. No React Flow (too heavy for CDN-only). Implement:
- Draggable nodes (mousedown/mousemove/mouseup)
- Connection ports (small circles on node edges)
- Bezier curve lines between ports
- Node palette (sidebar with draggable node templates)
- Pipeline state stored as JSON: `{nodes: [...], connections: [...]}`
- Save/Load pipeline: `POST /ide/pipeline/save` / `GET /ide/pipeline/load`

## Styling:
- Dark theme matching the IDE (bg #0f0f1a, nodes #1a1a2e, borders #2d2d44)
- Node width: 200px, height varies by type
- Canvas: full width of its container, 600px height
- Grid background (subtle dots)
- Zoom: scroll wheel, Pan: middle-click drag

## Output:

A single `<div id="flow-editor">` with embedded `<style>` and `<script>`. Self-contained, no external dependencies except the forge API.

Keep under 500 lines. Focus on the core: draggable nodes, connections, and pipeline execution. Polish can come later.
