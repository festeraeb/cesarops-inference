# Task: n8n Integration Guide for CESAROPS Orchestrator

You are a systems integration engineer. Write a complete n8n workflow JSON that connects n8n to the CESAROPS forge orchestrator.

## Context

The CESAROPS forge runs on `http://10.0.0.61:9100` and exposes:
- `POST /webhook/mission` — submit a mission (returns immediately with mission_id)
- `GET /webhook/missions` — list recent missions with status
- `POST /orchestrator/execute` — synchronous mission execution (blocks until done)
- `GET /orchestrator/probe` — cluster status
- `GET /cluster/nodes` — registered DII nodes

n8n runs on the same T440 server at `http://localhost:5678`.

## Write an n8n workflow that:

1. **Trigger**: Webhook node listening on `/n8n/mission` (POST)
2. **Validate**: Check that the incoming payload has `scenario` field
3. **Submit**: POST to `http://10.0.0.61:9100/webhook/mission` with the payload
4. **Wait**: Poll `GET /webhook/missions` every 10s until the mission_id shows status != "running" (max 5 min)
5. **Respond**: Return the MissionReport to the original webhook caller

## Also write a second workflow:

**"Cluster Health Monitor"**
1. **Trigger**: Cron every 5 minutes
2. **Check**: GET `http://10.0.0.61:9100/cluster/nodes`
3. **Alert**: If any node has `online: false`, send a notification (just log it to a Set node for now)
4. **Store**: Write the cluster state to a file `/tmp/cluster_health.json`

## Output format

Output as two complete n8n workflow JSON objects:

```json
// === WORKFLOW 1: Mission Intake ===
{ "name": "CESAROPS Mission Intake", "nodes": [...], "connections": {...} }

// === WORKFLOW 2: Cluster Health Monitor ===
{ "name": "CESAROPS Cluster Health", "nodes": [...], "connections": {...} }
```

## Constraints:
- Use n8n node types: n8n-nodes-base.webhook, n8n-nodes-base.httpRequest, n8n-nodes-base.if, n8n-nodes-base.wait, n8n-nodes-base.set, n8n-nodes-base.respondToWebhook, n8n-nodes-base.scheduleTrigger, n8n-nodes-base.writeFile (or n8n-nodes-base.code for file write)
- Keep it practical — these should import directly into n8n
- Use n8n v1 workflow format
