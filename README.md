# cesarops-inference

Public **distributed inference program**: scheduler, API, fleet adapters (control plane).

> **Monorepo note:** Today this lives at `wreckhunter2000-1/backend/nauti-inferer/`. Product name for the distributed program is **cesarops-inference**; the engine is [`nauti-inferer`](https://github.com/festeraeb/nauti-inferer).

## Intended contents

| Monorepo path | Role |
|---|---|
| `backend/nauti-inferer/` | Distributed control plane |
| `scripts/nauti-inferer-*.sh`, `scripts/cesarops-nauti-inferer.service` | Ops |
| Hetero / cake topology helpers (non-secret) | Multi-node layout |

Pairs with [`nauti-inferer-client`](https://github.com/festeraeb/nauti-inferer-client).
