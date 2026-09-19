```html
<div id="flow-editor">
    <style>
        #flow-editor {
            --bg: #0f0f1a; --node-bg: #1a1a2e; --border: #2d2d44; --text: #e0e0e0; --accent: #4f46e5;
            background: var(--bg); color: var(--text); font-family: 'Segoe UI', sans-serif;
            display: flex; height: 600px; width: 100%; border: 1px solid var(--border); position: relative; overflow: hidden;
        }
        #palette { width: 180px; background: #161625; border-right: 1px solid var(--border); padding: 10px; z-index: 10; }
        .palette-item { 
            background: var(--node-bg); border: 1px solid var(--border); padding: 8px; margin-bottom: 8px; 
            cursor: grab; font-size: 12px; border-radius: 4px; user-select: none;
        }
        .palette-item:hover { border-color: var(--accent); }
        #canvas-container { flex-grow: 1; position: relative; overflow: hidden; cursor: crosshair; }
        canvas { display: block; }
        #toolbar { position: absolute; top: 10px; right: 10px; display: flex; gap: 10px; z-index: 10; }
        button { 
            background: var(--accent); border: none; color: white; padding: 6px 12px; 
            border-radius: 4px; cursor: pointer; font-size: 12px; 
        }
        button:hover { opacity: 0.8; }
        button.stop { background: #ef4444; }
        .node-ui { position: absolute; pointer-events: none; width: 200px; font-size: 11px; }
    </style>

    <div id="palette">
        <div style="font-weight:bold; margin-bottom:10px; font-size:14px;">Nodes</div>
        <div class="palette-item" draggable="true" data-type="input">Input Node</div>
        <div class="palette-item" draggable="true" data-type="task">Task: Code</div>
        <div class="palette-item" draggable="true" data-type="task" data-task="Review">Task: Review</div>
        <div class="palette-item" draggable="true" data-type="task" data-task="Compile">Task: Compile</div>
        <div class="palette-item" draggable="true" data-type="output">Output Node</div>
        <div style="margin-top:20px; font-size:10px; color:#666;">Drag to canvas. Drag ports to connect.</div>
    </div>

    <div id="canvas-container">
        <div id="toolbar">
            <button onclick="flow.save()">Save</button>
            <button onclick="flow.load()">Load</button>
            <button id="run-btn" onclick="flow.run()">Run Pipeline</button>
        </div>
        <canvas id="flow-canvas"></canvas>
    </div>

    <script>
        class FlowEditor {
            constructor() {
                this.canvas = document.getElementById('flow-canvas');
                this.ctx = this.canvas.getContext('2d');
                this.container = document.getElementById('canvas-container');
                this.nodes = [];
                this.connections = [];
                this.offset = { x: 0, y: 0 };
                this.zoom = 1;
                this.dragNode = null;
                this.activePort = null;
                this.isPanning = false;
                this.isExecuting = false;
                this.gpuNodes = [];

                this.init();
            }

            async init() {
                this.resize();
                window.addEventListener('resize', () => this.resize());
                this.canvas.addEventListener('mousedown', e => this.handleMouseDown(e));
                window.addEventListener('mousemove', e => this.handleMouseMove(e));
                window.addEventListener('mouseup', () => this.handleMouseUp());
                this.canvas.addEventListener('wheel', e => this.handleWheel(e));
                
                // Drag & Drop from Palette
                this.container.addEventListener('dragover', e => e.preventDefault());
                this.container.addEventListener('drop', e => {
                    e.preventDefault();
                    const type = e.dataTransfer.getData('text/plain');
                    const taskType = e.dataTransfer.getData('task-type');
                    const rect = this.canvas.getBoundingClientRect();
                    const x = (e.clientX - rect.left - this.offset.x) / this.zoom;
                    const y = (e.clientY - rect.top - this.offset.y) / this.zoom;
                    this.addNode(type, x, y, taskType);
                });

                document.querySelectorAll('.palette-item').forEach(el => {
                    el.addEventListener('dragstart', e => {
                        e.dataTransfer.setData('text/plain', el.dataset.type);
                        e.dataTransfer.setData('task-type', el.dataset.task || '');
                    });
                });

                await this.fetchGPUs();
                this.render();
            }

            async fetchGPUs() {
                try {
                    const res = await fetch('/cluster/nodes');
                    this.gpuNodes = await res.json();
                } catch (e) {
                    this.gpuNodes = [{ name: 'P100#0', model: 'Gemma', vram: 80, temp: 45, status: 'online' }];
                }
            }

            resize() {
                this.canvas.width = this.container.clientWidth;
                this.canvas.height = this.container.clientHeight;
            }

            addNode(type, x, y, task = '') {
                const node = { id: Date.now(), type, x, y, w: 180, h: 80, task, data: '', status: 'idle', assignedGpu: null };
                if (type === 'gpu') { /* Logic for GPU nodes */ }
                this.nodes.push(node);
            }

            handleMouseDown(e) {
                const rect = this.canvas.getBoundingClientRect();
                const mx = (e.clientX - rect.left - this.offset.x) / this.zoom;
                const my = (e.clientY - rect.top - this.offset.y) / this.zoom;

                // Check ports first
                for (const n of this.nodes) {
                    if (this.checkPort(mx, my, n, 'out')) {
                        this.activePort = { node: n, type: 'out' };
                        return;
                    }
                    if (this.checkPort(mx, my, n, 'in')) {
                        this.activePort = { node: n, type: 'in' };
                        return;
                    }
                }

                // Check nodes
                this.dragNode = this.nodes.find(n => mx > n.x && mx < n.x + n.w && my > n.y && my < n.y + n.h);
                if (!this.dragNode && e.button === 1) this.isPanning = true;
            }

            checkPort(mx, my, n, type) {
                const px = type === 'out' ? n.x + n.w : n.x;
                const py = n.y + n.h / 2;
                return Math.hypot(mx - px, my - py) < 10;
            }

            handleMouseMove(e) {
                const rect = this.canvas.getBoundingClientRect();
                const mx = (e.clientX - rect.left - this.offset.x) / this.zoom;
                const my = (e.clientY - rect.top - this.offset.y) / this.zoom;

                if (this.dragNode) {
                    this.dragNode.x = mx - this.dragNode.w / 2;
                    this.dragNode.y = my - this.dragNode.h / 2;
                } else if (this.isPanning) {
                    this.offset.x += e.movementX;
                    this.offset.y += e.movementY;
                } else if (this.activePort) {
                    this.mousePos = { x: mx, y: my };
                }
            }

            handleMouseUp() {
                if (this.activePort) {
                    const target = this.nodes.find(n => 
                        this.checkPort(this.mousePos.x, this.mousePos.y, n, this.activePort.type === 'out' ? 'in' : 'out')
                    );
                    if (target && target !== this.activePort.node) {
                        const from = this.activePort.type === 'out' ? this.activePort.node : target;
                        const to = this.activePort.type === 'out' ? target : this.activePort.node;
                        this.connections.push({ from: from.id, to: to.id, status: 'idle' });
                    }
                }
                this.dragNode = null;
                this.isPanning = false;
                this.activePort = null;
            }

            handleWheel(e) {
                e.preventDefault();
                const delta = e.deltaY > 0 ? 0.9 : 1.1;
                this.zoom *= delta;
            }

            async run() {
                if (this.isExecuting) return;
                this.isExecuting = true;
                document.getElementById('run-btn').innerText = "Running...";
                
                let currentNode = this.nodes.find(n => n.type === 'input');
                let context = "";

                try {
                    while (currentNode) {
                        currentNode.status = 'active';
                        if (currentNode.type === 'task') {
                            const gpu = this.gpuNodes[0]; // Simplified: pick first GPU
                            const res = await fetch('/ide/chat/stream', {
                                method: 'POST',
                                body: JSON.stringify({ prompt: context, task: currentNode.task, gpu: gpu.name })
                            });
                            // Simulate streaming
                            context = "Result of " + currentNode.task; 
                        } else if (currentNode.type === 'input') {
                            context = "Initial Prompt";
                        }
                        
                        const conn = this.connections.find(c => c.from === currentNode.id);
                        if (!conn) break;
                        
                        currentNode.status = 'idle';
                        currentNode = this.nodes.find(n => n.id === conn.to);
                        conn.status = 'active';
                    }
                } catch (err) {
                    console.error(err);
                } finally {
                    this.isExecuting = false;
                    document.getElementById('run-btn').innerText = "Run Pipeline";
                }
            }

            save() {
                const data = JSON.stringify({ nodes: this.nodes, connections: this.connections });
                fetch('/ide/pipeline/save', { method: 'POST', body: data });
            }

            async load() {
                const res = await fetch('/ide/pipeline/load');
                const data = await res.json();
                this.nodes = data.nodes;
                this.connections = data.connections;
            }

            render() {
                const { ctx, canvas, zoom, offset } = this;
                ctx.clearRect(0, 0, canvas.width, canvas.height);
                ctx.save();
                ctx.translate(offset.x, offset.y);
                ctx.scale(zoom, zoom);

                // Grid
                ctx.strokeStyle = '#1e1e2e';
                ctx.lineWidth = 1;
                for(let i=-2000; i<2000; i+=40) {
                    ctx.beginPath(); ctx.moveTo(i, -2000); ctx.lineTo(i, 2000); ctx.stroke();
                    ctx.beginPath(); ctx.moveTo(-2000, i); ctx.lineTo(2000, i); ctx.stroke();
                }

                // Connections
                this.connections.forEach(c => {
                    const n1 = this.nodes.find(n => n.id === c.from);
                    const n2 = this.nodes.find(n => n.id === c.to);
                    if (!n1 || !n2) return;
                    ctx.beginPath();
                    ctx.strokeStyle = c.status === 'active' ? '#10b981' : '#4f46e5';
                    ctx.lineWidth = 2;
                    ctx.moveTo(n1.x + n1.w, n1.y + n1.h/2);
                    ctx.bezierCurveTo(n1.x + n1.w + 50, n1.y + n1.h/2, n2.x - 50, n2.y + n2.h/2, n2.x, n2.y + n2.h/2);
                    ctx.stroke();
                });

                // Nodes
                this.nodes.forEach(n => {
                    ctx.fillStyle = '#1a1a2e';
                    ctx.strokeStyle = n.status === 'active' ? '#10b981' : '#2d2d44';
                    ctx.lineWidth = 2;
                    ctx.beginPath();
                    ctx.roundRect(n.x, n.y, n.w, n.h, 6);
                    ctx.fill();
                    ctx.stroke();

                    ctx.fillStyle = '#fff';
                    ctx.font = 'bold 12px sans-serif';
                    ctx.fillText(n.type.toUpperCase() + (n.task ? `: ${n.task}` : ''), n.x + 10, n.y + 20);
                    
                    // Ports
                    ctx.fillStyle = '#4f46e5';
                    ctx.beginPath(); ctx.arc(n.x, n.y + n.h/2, 5, 0, Math.PI*2); ctx.fill(); // In
                    ctx.beginPath(); ctx.arc(n.x + n.w, n.y + n.h/2, 5, 0, Math.PI*2); ctx.fill(); // Out
                });

                ctx.restore();
                requestAnimationFrame(() => this.render());
            }
        }

        const flow = new FlowEditor();
    </script>
</div>
```
