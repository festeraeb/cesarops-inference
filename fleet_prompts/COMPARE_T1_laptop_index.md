# Task: Analyze and Organize CESAROPS Codebase from Laptop Dump

You are a senior software architect. Analyze the following file inventory from a laptop programming directory and produce a structured assessment.

## File Inventory (top 50 files by size, from /data/laptopdump/programming/):

### Python tools (top-level):
- universal_downloader.py (1711 lines) — satellite data downloader
- lake_michigan_scan.py (1175 lines) — Lake Michigan scanning pipeline
- ai_director.py (816 lines) — AI orchestration director
- hard_pixel_audit.py (658 lines) — pixel-level audit tool
- wreck_ml_trainer.py (619 lines) — ML model training for wreck detection
- triple_lock_fusion.py (575 lines) — triple-lock detection fusion
- wreck_scraper.py (573 lines) — web scraping for wreck data
- wreck_web_scraper.py (483 lines) — web scraping variant
- tile_geometry.py (476 lines) — satellite tile geometry
- andaste_geometry_test.py (452 lines) — Andaste wreck geometry testing
- background_probe.py (450 lines) — background probing tool
- wreck_ml_predictor.py (431 lines) — ML prediction for wrecks
- cesarops_orchestrator.py (425 lines) — mission orchestrator
- swot_ssh_extractor.py (402 lines) — SWOT satellite data extraction
- cesarops_engine.py (375 lines) — core engine
- smart_search_planner.py (370 lines) — intelligent search planning
- remote_dispatch.py (362 lines) — remote job dispatch
- database_connector.py (309 lines) — database interface
- deploy_and_scan.py (286 lines) — deployment + scanning
- llm_context_injector.py (284 lines) — LLM context injection
- tile_selector.py (263 lines) — satellite tile selection
- cmr_search.py (230 lines) — NASA CMR search
- tpu_client.py (225 lines) — TPU inference client
- tpu_server.py (152 lines) — TPU inference server

### Directories:
- cesarops/ — main project
- cesarops-core/ — core library (lake_erie_scan.py, cesarops_mission.py)
- cesarops-slicer/ — GeoTIFF tile slicer (Rust)
- cesarops-wreckhunter build/ — build artifacts + analysis tools
- cesarops-db-connector/ — database connector
- CesaropsSARBuild/ — SAR build
- sentinel_hunt_src/ — Sentinel satellite hunting (Rust)
- nauticuvs-publish/ — NautiCUVs curvelet library
- SonarSniffer/ — sonar file processor
- sonarsniffer_core/ — sonar core library
- tauri/ — desktop app (Tauri + React)
- wayfinder/ — ADHD helper app
- wreckhunter2000/ — main wreck hunting project
- wrecks_api/ — wreck database API
- pipelines/ — data processing pipelines
- ml/ — machine learning models
- models/ — GGUF model files
- news_search/ — news article search
- sensors/ — sensor data processing
- scripts/ — utility scripts
- Garmin/ — Garmin sonar data tools

## Your task:

Produce a structured report with:

1. **Project Map** — group files into logical projects/subsystems
2. **Completeness Assessment** — for each project, rate 1-5 (1=stub, 5=production)
3. **Dependencies** — what depends on what
4. **Duplicates/Variants** — files that appear to be different versions of the same thing
5. **Integration Priority** — what should be brought into the active workspace first
6. **Missing Pieces** — what's referenced but not present
7. **Recommended Architecture** — how these should be organized in a clean monorepo

Be concise. Use tables where possible. Under 500 words.
