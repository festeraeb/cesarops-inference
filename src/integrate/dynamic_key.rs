//! Dynamic DB key derivation from `dynamic_db_key.py`.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HdInfo {
    pub found: bool,
    pub mount_point: Option<String>,
    pub serial: Option<String>,
    pub identity_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NetworkInfo {
    pub hostname: String,
    pub ip: String,
    pub network_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Connectivity {
    pub internet: bool,
    pub intranet: bool,
    pub database: bool,
    pub xenon: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicDbKey {
    pub key: String,
    pub access_level: String,
    pub user_type: String,
    pub signature: String,
}

pub fn detect_network_type(ip: &str) -> &'static str {
    if ip.starts_with("10.") || ip.starts_with("192.168.") || ip.starts_with("172.") {
        "intranet"
    } else if ip == "unknown" || ip.is_empty() {
        "offline"
    } else {
        "internet"
    }
}

fn stable_hash_hex(input: &str) -> String {
    let mut hasher = DefaultHasher::new();
    input.hash(&mut hasher);
    format!("{:016x}", hasher.finish())
}

pub fn derive_access_level(hd_found: bool, c: &Connectivity) -> &'static str {
    if hd_found && c.database {
        if c.intranet || c.xenon {
            "full"
        } else if c.internet {
            "write"
        } else {
            "cached"
        }
    } else {
        "read-only"
    }
}

pub fn generate_dynamic_key(
    hd: &HdInfo,
    net: &NetworkInfo,
    c: &Connectivity,
    timestamp_iso: &str,
) -> DynamicDbKey {
    let key_material = format!(
        "{}|{}|{}|{}|{}|{}|{}|{}",
        hd.serial.clone().unwrap_or_else(|| "unknown".into()),
        net.network_type,
        net.ip,
        timestamp_iso,
        c.internet,
        c.intranet,
        c.database,
        c.xenon
    );
    let key = stable_hash_hex(&key_material);
    DynamicDbKey {
        key: key.clone(),
        access_level: derive_access_level(hd.found, c).into(),
        user_type: if hd.found { "agent" } else { "guest" }.into(),
        signature: key,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_access_when_hd_and_db_and_intranet() {
        let c = Connectivity {
            internet: false,
            intranet: true,
            database: true,
            xenon: false,
        };
        assert_eq!(derive_access_level(true, &c), "full");
    }
}
