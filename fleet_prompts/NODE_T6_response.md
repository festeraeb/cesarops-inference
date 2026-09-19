

<div id="dii-fleet">
<style>
  #dii-fleet { background: #1a1a2e; color: #e0e0e0; padding: 20px; font-family: system-ui, -apple-system, sans-serif; }
  #dii-fleet h2 { margin: 0 0 16px; color: #fff; font-size: 1.4em; }
  .node-card { background: #16213e; border: 1px solid #0f3460; border-radius: 8px; padding: 14px; margin-bottom: 12px; }
  .node-header { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
  .node-name { font-weight: 700; font-size: 1.1em; }
  .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
  .dot.online { background: #00ff88; box-shadow: 0 0 6px #00ff88; }
  .dot.offline { background: #ff4444; box-shadow: 0 0 6px #ff4444; }
  .last-seen { margin-left: auto; font-size: 0.8em; color: #888; }
  .state-badge { display: inline-block; padding: 3px 8px; border-radius: 4px; font-size: 0.8em; font-weight: 600; text-transform: uppercase; margin-bottom: 6px; }
  .state-idle { background: #4a4a4a; color: #ccc; }
  .state-serving { background: #00b894; color: #fff; }
  .state-loading { background: #fdcb6e; color: #2d3436; }
  .state-error { background: #d63031; color: #fff; }
  .active-model { color: #74b9ff; font-size: 0.9em; margin-bottom: 8px; }
  .gpu-card { background: #0f3460; border-radius: 6px; padding: 8px 10px; margin-top: 6px; }
  .gpu-name { font-size: 0.9em; font-weight: 500; margin-bottom: 4px; }
  .vram-bar { height: 6px; background: #2d3748; border-radius: 3px; overflow: hidden; margin-bottom: 4px; }
  .vram-fill { height: 100%; background: #0984e3; transition: width 0.3s; }
  .gpu-stats { font-size: 0.8em; color: #b2bec3; display: flex; justify-content: space-between; }
  .models-count { font-size: 0.85em; color: #a0a0a0; margin-top: 8px; }
</style>
<h2>🖥️ DII Node Fleet</h2>
<div id="nodes-container"></div>
<script>
  async function fetchNodes() {
    try {
      const res = await fetch('/cluster/nodes');
      const nodes = await res.json();
      const container = document.getElementById('nodes-container');
      container.innerHTML = '';
      nodes.forEach(node => {
        const hb = node.last_heartbeat || {};
        const state = hb.state || 'idle';
        const stateClass = `state-${state}`;
        const isOnline = !!node.online;
        let gpusHtml = '';
        (hb.all_gpus || []).forEach(gpu => {
          const pct = gpu.vram_total_mb ? (gpu.vram_used_mb / gpu.vram_total_mb * 100) : 0;
          gpusHtml += `
            <div class="gpu-card">
              <div class="gpu-name">${gpu.name}</div>
              <div class="vram-bar"><div class="vram-fill" style="width:${pct}%"></div></div>
              <div class="gpu-stats">
                <span>${gpu.vram_used_mb}/${gpu.vram_total_mb} MB</span>
                <span>${gpu.temp_c}°C</span>
                <span>${gpu.util_pct}%</span>
              </div>
            </div>`;
        });
        const card = document.createElement('div');
        card.className = 'node-card';
        card.innerHTML = `
          <div class="node-header">
            <span class="node-name">${node.node_id}</span>
            <span class="dot ${isOnline ? 'online' : 'offline'}"></span>
            <span class="last-seen">${node.last_seen_secs_ago}s ago</span>
          </div>
          <span class="state-badge ${stateClass}">${state}</span>
          ${hb.model ? `<div class="active-model">📂 ${hb.model}</div>` : ''}
          ${gpusHtml}
          <div class="models-count">📦 ${node.available_models?.length || 0} available models</div>
        `;
        container.appendChild(card);
      });
    } catch (e) { console.error('DII Fleet fetch failed', e); }
  }
  fetchNodes();
  setInterval(fetchNodes, 10000);
</script>
</div>
