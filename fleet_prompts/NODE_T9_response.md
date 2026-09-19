```json
// === WORKFLOW 1: Mission Intake ===
{
  "name": "CESAROPS Mission Intake",
  "nodes": [
    {
      "parameters": {
        "httpMethod": "POST",
        "path": "n8n/mission",
        "responseMode": "responseNode",
        "options": {}
      },
      "id": "1e6e7a8b-9c0d-4e1f-a2b3-c4d5e6f7a8b9",
      "name": "Webhook Trigger",
      "type": "n8n-nodes-base.webhook",
      "typeVersion": 1,
      "position": [100, 300]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{ $json.body.scenario }}",
              "operation": "isNotEmpty"
            }
          ]
        }
      },
      "id": "2f7e8a9b-0c1d-4e2f-b3c4-d5e6f7a8b9c0",
      "name": "Validate Scenario",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [320, 300]
    },
    {
      "parameters": {
        "method": "POST",
        "url": "http://10.0.0.61:9100/webhook/mission",
        "sendBody": true,
        "bodyParameters": {
          "parameters": [
            {
              "name": "scenario",
              "value": "={{ $json.body.scenario }}"
            }
          ]
        },
        "options": {}
      },
      "id": "3a1b2c3d-4e5f-6a7b-8c9d-0e1f2a3b4c5d",
      "name": "Submit Mission",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [540, 280]
    },
    {
      "parameters": {
        "url": "http://10.0.0.61:9100/webhook/missions",
        "options": {}
      },
      "id": "4b2c3d4e-5f6a-7b8c-9d0e-1f2a3b4c5d6e",
      "name": "Poll Status",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [760, 280]
    },
    {
      "parameters": {
        "conditions": {
          "string": [
            {
              "value1": "={{ $json.missions.find(m => m.mission_id === $node[\"Submit Mission\"].json.mission_id).status }}",
              "operation": "notEqual",
              "value2": "running"
            }
          ]
        }
      },
      "id": "5c3d4e5f-6a7b-8c9d-0e1f-2a3b4c5d6e7f",
      "name": "Is Finished?",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [980, 280]
    },
    {
      "parameters": {
        "amount": 10,
        "unit": "seconds"
      },
      "id": "6d4e5f6a-7b8c-9d0e-1f2a-3b4c5d6e7f8a",
      "name": "Wait 10s",
      "type": "n8n-nodes-base.wait",
      "typeVersion": 1,
      "position": [980, 450]
    },
    {
      "parameters": {
        "options": {}
      },
      "id": "7e5f6a7b-8c9d-0e1f-2a3b-4c5d6e7f8a9b",
      "name": "Respond to Webhook",
      "type": "n8n-nodes-base.respondToWebhook",
      "typeVersion": 1,
      "position": [1250, 260]
    },
    {
      "parameters": {
        "jsCode": "// Check if mission is still running in the list\nconst missions = $node[\"Poll Status\"].json.missions;\nconst targetId = $node[\"Submit Mission\"].json.mission_id;\nconst mission = missions.find(m => m.mission_id === targetId);\n\nif (!mission || mission.status !== 'running') {\n  return mission;\n} else {\n  throw new Error('STILL_RUNNING');\n}"
      },
      "id": "8f6a7b8c-9d0e-1f2a-3b4c-5d6e7f8a9b0c",
      "name": "Check Status Logic",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [1200, 280]
    }
  ],
  "connections": {
    "Webhook Trigger": {
      "main": [
        [
          {
            "node": "Validate Scenario",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Validate Scenario": {
      "main": [
        [
          {
            "node": "Submit Mission",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Submit Mission": {
      "main": [
        [
          {
            "node": "Poll Status",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Poll Status": {
      "main": [
        [
          {
            "node": "Is Finished?",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Is Finished?": {
      "main": [
        [
          {
            "node": "Respond to Webhook",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Wait 10s",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Wait 10s": {
      "main": [
        [
          {
            "node": "Poll Status",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}

// === WORKFLOW 2: Cluster Health Monitor ===
{
  "name": "CESAROPS Cluster Health",
  "nodes": [
    {
      "parameters": {
        "rule": {
          "interval": [
            {
              "field": "minutes",
              "minutesInterval": 5
            }
          ]
        }
      },
      "id": "a1b2c3d4-e5f6-4a7b-8c9d-0e1f2a3b4c5d",
      "name": "Schedule Trigger",
      "type": "n8n-nodes-base.scheduleTrigger",
      "typeVersion": 1.1,
      "position": [100, 300]
    },
    {
      "parameters": {
        "url": "http://10.0.0.61:9100/cluster/nodes",
        "options": {}
      },
      "id": "b2c3d4e5-f6a7-4b8c-9d0e-1f2a3b4c5d6e",
      "name": "Get Nodes",
      "type": "n8n-nodes-base.httpRequest",
      "typeVersion": 4.1,
      "position": [320, 300]
    },
    {
      "parameters": {
        "jsCode": "// Find nodes where online is false\nconst nodes = $input.all().map(i => i.json);\nconst offlineNodes = nodes.filter(n => n.online === false);\n\nreturn {\n  offline_count: offlineNodes.length,\n  offline_nodes: offlineNodes,\n  all_nodes: nodes,\n  timestamp: new Date().toISOString()\n};"
      },
      "id": "c3d4e5f6-a7b8-4c9d-0e1f-2a3b4c5d6e7f",
      "name": "Analyze Health",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [540, 300]
    },
    {
      "parameters": {
        "conditions": {
          "number": [
            {
              "value1": "={{ $json.offline_count }}",
              "operation": "larger",
              "value2": 0
            }
          ]
        }
      },
      "id": "d4e5f6a7-b8c9-4d0e-1f2a-3b4c5d6e7f8a",
      "name": "Any Offline?",
      "type": "n8n-nodes-base.if",
      "typeVersion": 1,
      "position": [760, 300]
    },
    {
      "parameters": {
        "values": {
          "string": [
            {
              "name": "alert_msg",
              "value": "={{ 'ALERT: ' + $json.offline_count + ' nodes are offline!' }}"
            }
          ]
        },
        "options": {}
      },
      "id": "e5f6a7b8-c9d0-4e1f-2a3b-4c5d6e7f8a9b",
      "name": "Log Alert",
      "type": "n8n-nodes-base.set",
      "typeVersion": 3.2,
      "position": [980, 200]
    },
    {
      "parameters": {
        "fileContent": "={{ JSON.stringify($node[\"Analyze Health\"].json, null, 2) }}",
        "fileName": "/tmp/cluster_health.json",
        "options": {}
      },
      "id": "f6a7b8c9-d0e1-4f2a-3b4c-5d6e7f8a9b0c",
      "name": "Write to File",
      "type": "n8n-nodes-base.code",
      "typeVersion": 2,
      "position": [1200, 300]
    }
  ],
  "connections": {
    "Schedule Trigger": {
      "main": [
        [
          {
            "node": "Get Nodes",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Get Nodes": {
      "main": [
        [
          {
            "node": "Analyze Health",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Analyze Health": {
      "main": [
        [
          {
            "node": "Any Offline?",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Any Offline?": {
      "main": [
        [
          {
            "node": "Log Alert",
            "type": "main",
            "index": 0
          }
        ],
        [
          {
            "node": "Write to File",
            "type": "main",
            "index": 0
          }
        ]
      ]
    },
    "Log Alert": {
      "main": [
        [
          {
            "node": "Write to File",
            "type": "main",
            "index": 0
          }
        ]
      ]
    }
  }
}
```
