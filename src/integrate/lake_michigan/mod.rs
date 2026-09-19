//! Lake Michigan DB helpers.
pub mod populate_database;

pub use populate_database::{
    parse_results_json, sql_insert_hit, RunTileRecord, ANOMALY_HITS_DDL, CENSUS_DB,
};
