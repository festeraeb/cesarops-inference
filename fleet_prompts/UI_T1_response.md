```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CESAROPS FORGE IDE</title>
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/xterm@5.3.0/css/xterm.css">
    <style>
        :root {
            --bg: #0f0f1a; --panel: #1a1a2e; --border: #2d2d44; --text: #e0e0e0;
            --accent: #00d4ff; --success: #4caf50; --warn: #ff9800; --err: #ff4b2b;
        }
        * { box-sizing: border-box; }
        body, html { margin: 0; padding: 0; height: 100%; background: var(--bg); color: var(--text); font-family: system-ui, -apple-system, sans-serif; overflow: hidden; }
        
        /* Layout */
        .app-container { display: flex; flex-direction: column; height: 100vh; }
        header { height: 45px; background: var(--panel); border-bottom: 1px solid var(--border); display: flex; align-items: center; padding: 0 15px; justify-content: space-between; font-size: 14px; }
        .main-view { flex: 1; display: flex; overflow: hidden; }
        .file-tree { width: 15%; border-right: 1px solid var(--border); overflow-y: auto; padding: 10px; font-size: 13px; }
        .editor-area { flex: 1; position: relative; display: flex; flex-direction: column; }
        #monaco-container { flex: 1; }
        .chat-sidebar { width: 300px; border-left: 1px solid var(--border); display: flex; flex-direction: column; background: var(--panel); }
        .chat-panel { flex: 1; display: flex; flex-direction: column; border-bottom: 1px solid var(--border); overflow: hidden; }
        .chat-header { padding: 8px; background: #161625; border-bottom: 1px solid var(--border); display: flex; justify-content: space-between; align-items: center; }
        .chat-messages { flex: 1; overflow-y: auto; padding: 10px; font-size: 13px; }
        .chat-input-area { padding: 10px; border-top: 1px solid var(--border); display: flex; gap: 5px; }
        .chat-input-area input { flex: 1; background: #0f0f1a; border: 1px solid var(--border); color: white; padding: 5px; border-radius: 4px; }
        .terminal-area { height: 20%; border-top: 2px solid var(--border); background: #000; display: flex; flex-direction: column; }
        .terminal-toolbar { height: 30px; background: var(--panel); display: flex; align-items: center; padding: 0 10px; gap: 10px; font-size: 12px; }

        /* UI Elements */
        .badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: bold; text-transform: uppercase; }
        .badge-mode { background: var(--accent); color: #000; }
        .badge-nodes { background: var(--success); color: #fff; }
        button { cursor: pointer; background: #2d2d44; color: white; border: none; padding: 4px 8px; border-radius: 3px; font-size: 12px; }
        button:hover { background: var(--accent); color: #000; }
        select { background: #0f0f1a; color: white; border: 1px solid var(--border); font-size: 11px; }
        .tree-item { cursor: pointer; padding: 2px 5px; white-space: nowrap; }
        .tree-item:hover { background: #2d2d44; }
        .msg { margin-bottom: 10px; line-height: 1.4; }
        .msg.user { color: var(--accent); font-weight: bold; }
        .msg.ai { color: #ccc; }
        .msg.stream { color: #fff; font-style: italic; }
        .tree-dir { color: var(--accent); font-weight: bold; }
        .tree-file { color: #aaa; }
        #terminal-output { flex: 1; overflow-y: auto; padding: 5px; font-family: 'JetBrains Mono', monospace; font-size: 12px; white-space: pre-wrap; }
    </style>
</head>
<body>

<div class="app-container">
    <header>
        <div><strong>CESAROPS FORGE IDE</strong> <span id="mode-badge" class="badge badge-mode">MODE</span></div>
        <div>
            <span id="node-count" class="badge badge-nodes">Nodes: 0</span>
            <button onclick="switchMode()">Switch Mode</button>
        </div>
    </header>

    <div class="main-view">
        <div class="file-tree" id="file-tree"></div>
        <div class="editor-area">
            <div id="monaco-container"></div>
        </div>
        <div class="chat-sidebar">
            <div class="chat-panel" id="panel-a">
                <div class="chat-header">
                    <select id="gpu-a"><option>Select GPU...</option></select>
                    <button onclick="applyToEditor('a')">Apply</button>
                </div>
                <div class="chat-messages" id="msgs-a"></div>
                <div class="chat-input-area">
                    <input type="text" id="input-a" placeholder="Ask...">
                    <button onclick="sendChat('a')">Send</button>
                </div>
            </div>
            <div class="chat-panel" id="panel-b">
                <div class="chat-header">
                    <select id="gpu-b"><option>Select GPU...</option></select>
                    <button onclick="applyToEditor('b')">Apply</button>
                </div>
                <div class="chat-messages" id="msgs-b"></div>
                <div class="chat-input-area">
                    <input type="text" id="input-b" placeholder="Ask...">
                    <button onclick="sendChat('b')">Send</button>
                </div>
            </div>
            <div style="padding: 5px; display: flex; gap: 5px; background: var(--panel);">
                <button style="flex:1" onclick="sendBoth()">Send to Both</button>
                <button style="flex:1" onclick="compareChats()">Compare</button>
            </div>
        </div>
    </div>

    <div class="terminal-area">
        <div class="terminal-toolbar">
            <button onclick="runCommand('cargo build --release')">Build</button>
            <input type="text" id="cmd-input" placeholder="Custom command..." style="flex:1; background:#000; border:1px solid #333; color:#fff; font-size:11px;">
            <button onclick="runCustomCommand()">Run</button>
        </div>
        <div id="terminal-output"></div>
    </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs/loader.js"></script>
<script>
    let editor, currentFilePath = null;
    let chatContext = { a: [], b: [] };
    let lastResponses = { a: "", b: "" };

    require.config({ paths: { vs: 'https://cdn.jsdelivr.net/npm/monaco-editor@0.45.0/min/vs' } });
    require(['vs/editor/editor.main'], function () {
        editor = monaco.editor.create(document.getElementById('monaco-container'), {
            theme: 'vs-dark',
            automaticLayout: true,
            fontSize: 14,
            fontFamily: 'JetBrains Mono, Fira Code, monospace'
        });
        initApp();
    });

    async function initApp() {
        await refreshHeader();
        await loadTree('/home/cesarops/wreckhunter2000-1');
        await loadGpus();
        setupTerminal();
    }

    async function refreshHeader() {
        const modeRes = await fetch('/mode');
        const mode = await modeRes.text();
        document.getElementById('mode-badge').innerText = mode;
        const nodeRes = await fetch('/cluster/nodes');
        const nodes = await nodeRes.json();
        document.getElementById('node-count').innerText = `Nodes: ${nodes.length}`;
        const gpuSelects = [document.getElementById('gpu-a'), document.getElementById('gpu-b')];
        gpuSelects.forEach(s => {
            s.innerHTML = nodes.map(n => `<option value="${n.url}">${n.name}</option>`).join('');
        });
    }

    async function switchMode() {
        const current = document.getElementById('mode-badge').innerText;
        const target = current === 'CESAROPS' ? '/mode/coding' : '/mode/cesarops';
        await fetch(target, { method: 'POST' });
        initApp();
    }

    async function loadTree(root) {
        const res = await fetch(`/ide/tree?root=${encodeURIComponent(root)}`);
        const data = await res.json();
        const container = document.getElementById('file-tree');
        container.innerHTML = '';
        container.appendChild(renderTree(data, root));
    }

    function renderTree(nodes, root) {
        const ul = document.createElement('div');
        nodes.forEach(node => {
            const div = document.createElement('div');
            div.className = `tree-item ${node.type === 'dir' ? 'tree-dir' : 'tree-file'}`;
            div.innerText = (node.type === 'dir' ? '📁 ' : '📄 ') + node.name;
            div.onclick = async (e) => {
                e.stopPropagation();
                if (node.type === 'dir') {
                    // Simple toggle logic could be added here
                } else {
                    const fullPath = root + '/' + (node.path || node.name); // Simplified path logic
                    openFile(fullPath);
                }
            };
            ul.appendChild(div);
            if (node.children) ul.appendChild(renderTree(node.children, root + '/' + node.name));
        });
        return ul;
    }

    async function openFile(path) {
        const res = await fetch(`/ide/file?path=${encodeURIComponent(path)}`);
        const content = await res.text();
        currentFilePath = path;
        editor.setValue(content);
    }

    async function saveFile() {
        if (!currentFilePath) return;
        await fetch('/ide/file', {
            method: 'POST',
            body: JSON.stringify({ path: currentFilePath, content: editor.getValue() })
        });
    }

    // Auto-save on change
    editor.onDidChangeModelContent(() => {
        // Debounced save would be better
    });

    async function sendChat(panelId) {
        const input = document.getElementById(`input-${panelId}`);
        const prompt = input.value.trim();
        if (!prompt) return;

        const msgDiv = document.createElement('div');
        msgDiv.className = 'msg user';
        msgDiv.innerText = `> ${prompt}`;
        const container = document.getElementById(`msgs-${panelId}`);
        container.appendChild(msgDiv);
        input.value = '';

        const aiDiv = document.createElement('div');
        aiDiv.className = 'msg ai stream';
        container.appendChild(aiDiv);
        container.scrollTop = container.scrollHeight;

        const endpoint = document.getElementById(`gpu-${panelId}`).value;
        
        try {
            const response = await fetch('/ide/chat/stream', {
                method: 'POST',
                headers: { 'Accept': 'text/event-stream' },
                body: JSON.stringify({ prompt, endpoint, max_length: 2048, temperature: 0.4 })
            });

            const reader = response.body.getReader();
            const decoder = new TextDecoder();
            let fullText = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                const chunk = decoder.decode(value);
                const lines = chunk.split('\n');
                for (let line of lines) {
                    if (line.startsWith('data: ')) {
                        const data = line.substring(6);
                        if (data === '[DONE]') break;
                        try {
                            const parsed = JSON.parse(data);
                            const token = parsed.text || "";
                            fullText += token;
                            aiDiv.innerText = fullText;
                            aiDiv.classList.remove('stream');
                            container.scrollTop = container.scrollHeight;
                        } catch(e) {}
                    }
                }
            }
            lastResponses[panelId] = fullText;
            chatContext[panelId].push({ role: 'user', content: prompt }, { role: 'assistant', content: fullText });
        } catch (err) {
            aiDiv.innerText = "Error: " + err.message;
            aiDiv.className = 'msg err';
        }
    }

    async function sendBoth() {
        const prompt = document.getElementById('input-a').value;
        if (!prompt) return;
        await sendChat('a');
        await sendChat('b');
    }

    function applyToEditor(panelId) {
        const text = lastResponses[panelId];
        if (text && editor) {
            const current = editor.getValue();
            editor.setValue(current + "\n" + text);
        }
    }

    function compareChats() {
        const diff = `--- PANEL A ---\n${lastResponses.a}\n\n--- PANEL B ---\n${lastResponses.b}`;
        editor.setValue(diff);
    }

    async function runCommand(cmd) {
        const res = await fetch('/ide/exec', {
            method: 'POST',
            body: JSON.stringify({ command: cmd })
        });
        const out = document.getElementById('terminal-output');
        out.innerText += `\n$ ${cmd}\n`;
    }

    async function runCustomCommand() {
        const cmd = document.getElementById('cmd-input').value;
        if (cmd) runCommand(cmd);
    }

    function setupTerminal() {
        const out = document.getElementById('terminal-output');
        const ws = new WebSocket('ws://localhost:9100/ide/terminal');
        ws.onmessage = (e) => {
            out.innerText += e.data;
            out.scrollTop = out.scrollHeight;
        };
    }

    async function loadGpus() {
        const res = await fetch('/cluster/nodes');
        const nodes = await res.json();
        const selects = [document.getElementById('gpu-a'), document.getElementById('gpu-b')];
        selects.forEach(s => {
            s.innerHTML = nodes.map(n => `<option value="${n.url}">${n.name}</option>`).join('');
        });
    }
</script>
</body>
</html>
```
