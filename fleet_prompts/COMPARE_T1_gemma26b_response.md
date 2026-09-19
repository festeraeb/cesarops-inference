# CESAROPS Codebase Architectural Assessment

## 1. Project Map & Completeness Assessment

| Project / Subsystem | Primary Components | Completeness (1-5) |
| :--- | :--- | :---: |
| **CESAROPS Core Engine** | `cesarops_engine.py`, `cesarops_orchestrator.py`, `cesarops-core/` | 4 |
| **Satellite Data Pipeline** | `universal_downloader.py`, `swot_ssh_extractor.py`, `cmr_search.py`, `tile_geometry.py` | 5 |
| **ML Detection Suite** | `wreck_ml_trainer.py`, `wreck_ml_predictor.py`, `triple_lock_fusion.py`, `tpu_client.py` | 4 |
| **Wreck Intelligence** | `wreck_scraper.py`, `wreck_web_scraper.py`, `smart_search_planner.py`, `wrecks_api/` | 3 |
| **Data Processing (Slicing/Sonar)** | `cesarops-slicer/` (Rust), `SonarSniffer/`, `Garmin/`, `nauticuvs-publish/` | 3 |
| **User Interface** | `tauri/` (React/Tauri desktop app) | 2 |

## 2. Dependency Graph
*   **Data Ingest $\rightarrow$ Processing:** `universal_downloader` $\rightarrow$ `tile_selector` $\rightarrow$ `cesarops-slicer` (Rust).
*   **Processing $\rightarrow$ Intelligence:** `cesarops-slicer` $\rightarrow$ `wreck_ml_predictor` $\rightarrow$ `triple_lock_fusion`.
*   **Intelligence $\rightarrow$ Orchestration:** `wreck_scraper` + `ml_predictor` $\rightarrow$ `ai_director` $\rightarrow$ `cesarops_orchestrator`.
*   **Orchestration $\rightarrow$ Deployment:** `cesarops_orchestrator` $\rightarrow$ `remote_dispatch` $\rightarrow$ `deploy_and_scan`.

## 3. Duplicates & Variants
*   **Scrapers:** `wreck_scraper.py` vs `wreck_web_scraper.py` (Likely a refactor or specialized version).
*   **Geometry:** `tile_geometry.py` vs `andaste_geometry_test.py` (Test-driven implementation).
*   **Core Logic:** `cesarops_engine.py` vs `cesarops-core/` (Potential split between monolithic script and library).

## 4. Integration Priority
1.  **Foundation:** `cesarops-core` and `database_connector`.
2.  **Data Ingest:** `universal_downloader.py` and `tile_geometry.py`.
3.  **Inference:** `wreck_ml_predictor.py` and `tpu_client.py`.
4.  **Orchestration:** `cesarops_orchestrator.py` and `ai_director.py`.

## 5. Missing Pieces
*   **Unified API Layer:** `wrecks_api/` exists, but no clear gateway connecting the ML output to the API.
*   **CI/CD & Environment:** No visible Dockerfiles, `requirements.txt`, or `Cargo.toml` in the top-level dump.
*   **Testing Suite:** Only `andaste_geometry_test.py` is visible; a global test framework is missing.

## 6. Recommended Monorepo Architecture
```text
/cesarops-monorepo
├── /apps
│   ├── /desktop (Tauri/React)
│   └── /api (Wrecks API)
├── /libs
│   ├── /core (cesarops-core, engine, orchestrator)
│   ├── /ml (trainer, predictor, fusion)
│   └── /geo (tile_geometry, slicer-rust)
├── /services
│   ├── /ingest (downloader, scraper, search)
│   └── /compute (tpu-server, remote-dispatch)
├── /tools (audit, probe, search-planner)
└── /infrastructure (deployment, database-connector)
```
