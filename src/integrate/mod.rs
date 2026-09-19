//! Laptop-dump integration — Rust ports of high-value Python pipeline scripts.

pub mod fuel_leak;
pub mod tile_zscore;
pub mod great_lakes;
pub mod sensor_scan;
pub mod cuda_stats;
pub mod forensic_scan;
pub mod full_basin;
pub mod triple_lock;
pub mod daily_pull;
pub mod cli_orchestrator;
pub mod agent_presets;
pub mod ai_director;
pub mod crossref;
pub mod drive_identity;
pub mod dynamic_key;
pub mod lake_michigan_dual_scan;
pub mod detection_sorter;
pub mod tiff_fast;
pub mod resolution_comparison;
pub mod anchor_lock_display;
pub mod xenon_cuda_checker;
pub mod daily_scan;
pub mod gpu_diagnostic;
pub mod lake_michigan_scan;
pub mod hard_pixel_audit;
pub mod monster_material_audit;
pub mod andaste_geometry;
pub mod cesarops_orchestrator;
pub mod cuda_env;
pub mod cuda_verification;
pub mod db_master_key;
pub mod find_swot_dates;
pub mod global_controls;
pub mod repeatability_check;
pub mod run_zero_baseline;
pub mod three_tile_offset;
pub mod tpu_client;
pub mod database_connector;
pub mod init_database;
pub mod audit_wrecks_db;
pub mod fetcher;
pub mod bridge_calibrate;
pub mod extract_oil_spills_kmz;
pub mod gpu_health;
pub mod gpu_stress_test;
pub mod sync_xenon;
pub mod xenon_sync;
pub mod tpu_server;
pub mod geotiff_inventory;
pub mod file_inventory;
pub mod inspect_scan;
pub mod download_straits_2024;
pub mod bridge_proximity;
pub mod mb2_crosscheck;
pub mod wipe_database;
pub mod query_wrecks_db;
pub mod scan_wreck_db;
pub mod match_wrecks;
pub mod hls_download;
pub mod hls_dl;
pub mod hls_dl2;
pub mod hls_download2;
pub mod hls_download3;
pub mod cleanup_and_organize;
pub mod inspect_geometry_metadata;
pub mod wreck_pixel_probe;
pub mod generate_report;
pub mod crossref_scans;
pub mod fix_and_add_wrecks;
pub mod fetch_geometry_metadata;
pub mod tile_geometry;
pub mod tile_selector;
pub mod deep_wreck_validation;
pub mod llm_context_injector;
pub mod deploy_and_scan;
pub mod remote_dispatch;
pub mod hls_b02_download;
pub mod batch_download_manager;
pub mod cuda_test_kmz;
pub mod db_ingestor;
pub mod erie_multiyear_downloader;
pub mod full_lake_michigan_run;
pub mod full_scan;
pub mod iowa_202_analysis;
pub mod live_feed_server;
pub mod monster_analysis;
pub mod monster_candidate;
pub mod lake_michigan;
pub mod prioritized_satellite_pull;
pub mod process_tiles;
pub mod process_with_coordinates;
pub mod pull_altimetry_anonymous;
pub mod raw_scan_reprocess;
pub mod run_configured_pipeline;
pub mod straits_fox_pipeline;
pub mod small_batch_test;
pub mod smart_daily_scan;
pub mod gpu_detection;
pub mod m2200_gpu_test;
pub mod gpu_test;
pub mod pipeline_test;
pub mod test_repeatability;
pub mod thermal_validation;
pub mod viirs_multi_year_scan;
pub mod zion_trench_squeeze;

pub use fuel_leak::{analyze_detection, classify_pixel, leak_index, DetectionFuelAnalysis, PixelClass};
pub use tile_zscore::{process_tile_gray, TileZscoreResult};
pub use great_lakes::{great_lakes_catalog, lakes_by_priority, Bbox, LakeConfig};
pub use sensor_scan::{
    anchor_calibration_offset, anchor_points, sensor_configs, AnchorPoint, SensorConfig,
};
pub use cuda_stats::{tile_anomaly_stats, CudaTileStats};
pub use forensic_scan::{
    apply_depth_scaling, apply_straits_correction, finalize_detection, ForensicDetection,
    DEPTH_THRESHOLD_FT, STRAITS_LAT_THRESHOLD, ZION_CONSTANT,
};
pub use full_basin::{
    known_wrecks, lake_michigan_bounds, signature_filter_name, KnownWreck, LakeBounds,
    SignatureFilter,
};
pub use triple_lock::{
    fuse_hits, fusion_cell, is_cold_sink, thermal_zscore, SensorHit, TripleLockTarget,
};
pub use daily_pull::{
    all_sources, lake_michigan_bbox, plan_date_range, LakeBbox, PullPlan, SatelliteSource,
};
pub use cli_orchestrator::{build_batch_plan, find_thermal_tiffs, BatchScanPlan, TiffJob};
pub use agent_presets::{default_presets, ProcessingPreset};
pub use ai_director::{available_tools, bounding_boxes, parse_request_fallback, BBoxPreset, DirectorPlan, ToolMeta};
pub use crossref::{confidence_flag, haversine_km, nearest_wrecks, AnomalySite, NearestWreck, Wreck};
pub use drive_identity::{
    build_registration_payload, build_session_record, default_admin_permissions, DriveIdentity,
    DrivePermissions, DriveRegistrationPayload, SessionRecord,
};
pub use dynamic_key::{
    derive_access_level, detect_network_type, generate_dynamic_key, Connectivity, DynamicDbKey,
    HdInfo, NetworkInfo,
};
pub use lake_michigan_dual_scan::{
    calculate_tile_quality, generate_landsat_path_rows,
    lake_michigan_bounds as dual_scan_lake_michigan_bounds, low_water_years,
    sentinel_tiles, DateWindow, LakeBounds as DualScanBounds, QualityScore, YearWindows,
};
pub use detection_sorter::{apply_filters as apply_detection_filters, DetectionSite, SorterFilters};
pub use tiff_fast::{process_tiff_fast, AnomalyPoint, FastScanResult, RasterShape};
pub use resolution_comparison::{analyze_resolution_comparison, ResolutionMetrics, ResolutionRun};
pub use anchor_lock_display::{
    anchor_network, candidate_length_match, zion_targets, Anchor, ZionTarget,
};
pub use xenon_cuda_checker::{
    parse_cuda_device_count, summarize_xenon_status, CommandProbe, XenonCudaStatus,
};
pub use daily_scan::{find_tiffs, summarize_detection_counts, DailyScanSummary};
pub use gpu_diagnostic::{
    nvidia_status_from_output, rust_gpu_status_from_output, summarize_diagnostic,
    vulkan_status_from_output, DiagnosticSummary,
};
pub use lake_michigan_scan::{
    anchor_points as lake_scan_anchor_points, apply_anchor_calibration as lake_scan_anchor_calibration,
    top_abs_zscores as lake_scan_top_abs_zscores, AnomalyDetection, GeoPoint,
};
pub use hard_pixel_audit::{
    apply_zion_constant as apply_depth_zion_constant, estimate_length_ft as estimate_length_from_pixels_ft,
    pixel_distance_m as hard_pixel_distance_m, zscores as hard_pixel_zscores, RegionAnomaly,
    DEPTH_THRESHOLD_FT as HARD_PIXEL_DEPTH_THRESHOLD_FT, ZION_CONSTANT as HARD_PIXEL_ZION_CONSTANT,
};
pub use monster_material_audit::{
    best_profile_for_decay, calculate_thermal_decay, cargo_profiles, compare_thermal_decay, CargoProfile,
};
pub use andaste_geometry::{
    haversine_m as andaste_haversine_m, scan_island_count, tumblehome_classification, IslandAnalysis,
    ThermalPeak,
};
pub use cesarops_orchestrator::{
    build_plan as build_orchestrator_plan, default_thresholds as orchestrator_default_thresholds,
    known_areas as orchestrator_known_areas, AreaConfig, OrchestratorPlan, Thresholds,
};
pub use cuda_env::{configure_cuda_environment, cuda_config_from_path, resolve_cuda_path, CudaConfig};
pub use cuda_verification::{
    parse_nvidia_smi_query, summarize_verification, CudaVerificationReport, CudaVerificationStep,
};
pub use db_master_key::{
    build_session as build_master_key_session, fallback_serial, generate_master_key, ExternalHdInfo,
    MasterKeySession, SALT_FILE,
};
pub use find_swot_dates::{
    bbox_query_string, build_swot_catalog, extract_dates_from_cmr_entries, group_consecutive_dates,
    lake_michigan_bbox as swot_lake_michigan_bbox, DateRange, LakeBbox as SwotLakeBbox, SwotDateCatalog,
    SWOT_PRODUCT,
};
pub use global_controls::{
    GlobalScannerSettings, LakePreset, TargetPreset, VramSettings,
};
pub use repeatability_check::{
    count_identical_detections, parse_kml_detections, KmlDetection, RepeatabilitySummary,
};
pub use run_zero_baseline::{parse_gpu_from_scanner_line, RunZeroRecord, SystemInfo};
pub use three_tile_offset::{
    default_triple_lock_tile, offset_meters, thermal_stats, PixelAnomaly, ThermalStats, TileSpec,
};
pub use tpu_client::{parse_infer_response, TpuClientConfig, TpuInferRequest, TpuInferResponse};
pub use database_connector::{
    census_exists, runs_exists, sql_log_scan_run, sql_new_arrivals, sql_stationary_anchors,
    sql_update_triple_lock, CensusStatus, DbPaths, NewArrival, StationaryAnchor,
};
pub use init_database::{
    expected_tables, metadata_bootstrap_rows, InitDatabaseConfig, MINIMAL_SCHEMA, DEFAULT_DB,
    SCHEMA_FILE,
};
pub use audit_wrecks_db::{
    find_singletons, round_coord, top_coordinate_clusters, wrecks_in_high_density_clusters,
    CoordCluster, WreckSingleton, WrecksAuditReport,
};
pub use fetcher::{
    all_great_lakes_bounds as fetcher_all_great_lakes_bounds,
    credentials_from_env,
    ice_break_windows,
    lake_michigan_bounds as fetcher_lake_michigan_bounds,
    lake_superior_bounds,
    low_silt_windows,
    usgs_scene_search_payload,
    DateWindow as FetcherDateWindow,
    LakeBounds as FetcherLakeBounds,
    SENTINEL_HUB_BASE,
    USGS_API_BASE,
};
pub use bridge_calibrate::{haversine_m, lat_lon_offset_to_metres, GeoRef, MACKINAC_REFS, RoundTripError};
pub use extract_oil_spills_kmz::{
    extract_oil_pixels, parse_leaking_boat_from_tile, LeakingBoatSensor, OilSpillFeature, PixelSizeM,
};
pub use gpu_health::{evaluate_transfer, is_success, CudaMinimalTestResult, GpuDeviceInfo};
pub use gpu_stress_test::{likely_gpu_used, StressTestJob, StressTestResult, DEFAULT_PIXELS, TILE_SIDE};
pub use sync_xenon::{plan_deploy_jobs, xenon_scp_command, ScpJob, DEPLOY_FILES, XENON_HOST, XENON_USER};
pub use xenon_sync::{build_sync_plan, XenonSyncPlan, XenonSyncSummary, DB_SYNC_FILES};
pub use tpu_server::{validate_infer_payload, TpuHealthResponse, TpuServerConfig};
pub use geotiff_inventory::{
    parse_hls_filename, record_tile, GeotiffInventory, ParsedTiffName, MIN_TIFF_BYTES,
};
pub use file_inventory::{classify_path, is_core_script, FileInventoryReport, CORE_SCRIPTS};
pub use hls_b02_download::{b02_dest_name, granule_query_params, pick_b02_href, product_for_title, B02DownloadJob};
pub use batch_download_manager::{build_download_tasks, filter_chunks_for_node, lake_catalog, summer_fall_dates, LakeDownloadSpec};
pub use cuda_test_kmz::{census_log_payload, parse_gpu_line, CudaBenchmarkResults, CudaDeviceInfo as CudaTestDeviceInfo};
pub use db_ingestor::{marker_path_for, parse_probe_json, ProbeFeature, ANOMALY_HITS_DDL as INGESTOR_DDL};
pub use erie_multiyear_downloader::{build_erie_tasks, max_results_for, ERIE_BBOX, ErieMonthTask};
pub use full_lake_michigan_run::{process_thermal_pair, ThermalRunStats, TileRunResult};
pub use full_scan::{anchor_points as full_scan_anchors, is_thermal_band, AnchorPoint as FullScanAnchor};
pub use iowa_202_analysis::{cargo_type_from_mass, un_squeeze_length, TargetProfile};
pub use live_feed_server::{group_sites_by_confidence, LiveFeedRoutes, LiveFeedSite};
pub use monster_analysis::{corrected_length_ft, estimate_mass_tons, inverse_projection_length, HullTarget};
pub use monster_candidate::{best_match, is_monster_sized, MonsterCandidate};
pub use lake_michigan::populate_database::{parse_results_json, RunTileRecord, CENSUS_DB};
pub use prioritized_satellite_pull::{great_lakes_pull_config, sort_lakes_by_priority, PullMode};
pub use process_tiles::{process_tile_values, TileProcessResult};
pub use process_with_coordinates::{build_anomaly, parse_gpu_anomaly_line, PixelAnomalyOut};
pub use pull_altimetry_anonymous::{great_lakes_altimetry_bboxes, satellite_ftp_path, AVISO_FTP_HOST};
pub use raw_scan_reprocess::{parse_mode, parse_tiles_arg, ReprocessMode, ReprocessPlan};
pub use run_configured_pipeline::{gpu_command, load_config_json, PipelineConfig};
pub use straits_fox_pipeline::{missing_packages, straits_fox_scripts, REQUIRED_PACKAGES};
pub use small_batch_test::{analyze_tile_stats, take_first_n, TileBatchResult};
pub use smart_daily_scan::{lake_scan_order, point_in_bbox, LakeScanBbox};
pub use gpu_detection::{evaluate_rust_gpu_stdout, GpuValidationResult};
pub use m2200_gpu_test::{scale_to_u16, HardwareTestSpec};
pub use gpu_test::{default_anomaly_centers, gpu_confirmed, SyntheticAnomaly};
pub use pipeline_test::{evaluate_gpu_detect, evaluate_tiff_process, PipelineTestReport};
pub use test_repeatability::{passes_repeatability, RepeatabilityCriteria, DetectionSnapshot};
pub use thermal_validation::{count_anomalies, thermal_zscore_stats, TARGET_LAT, TARGET_LON};
pub use viirs_multi_year_scan::{parse_viirs_filename, same_target, ViirsDetection};
pub use zion_trench_squeeze::{andaste_center, generate_trench_grid, TrenchGridCell};
