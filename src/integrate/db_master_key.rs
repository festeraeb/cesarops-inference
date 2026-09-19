//! Dynamic database master key — port of `wreckhunter/tools/generate_db_master_key.py`.

use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

pub const SALT_FILE: &str = "cesarops_salt.bin";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ExternalHdInfo {
    pub found: bool,
    pub mount_point: Option<String>,
    pub salt_file: Option<String>,
    pub serial: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct MasterKeySession {
    pub master_key: String,
    pub hd_serial: String,
    pub generated_at: String,
    pub valid_until: String,
}

pub fn fallback_serial(mount_point: &str) -> String {
    let mut hasher = DefaultHasher::new();
    mount_point.hash(&mut hasher);
    format!("{:012x}", hasher.finish())
}

pub fn generate_master_key(hd_serial: &str, salt_hex: &str, date_yyyymmdd: &str) -> String {
    let key_material = format!("{hd_serial}:{salt_hex}:{date_yyyymmdd}");
    let mut hasher = DefaultHasher::new();
    key_material.hash(&mut hasher);
    let digest = hasher.finish();
    format!("{:016x}{:016x}", digest, digest.rotate_left(17))
}

pub fn build_session(hd_serial: &str, salt: &[u8], generated_at: &str, valid_until: &str) -> MasterKeySession {
    let salt_hex = salt.iter().map(|b| format!("{b:02x}")).collect::<String>();
    let date = generated_at.get(0..10).unwrap_or("1970-01-01").replace('-', "");
    MasterKeySession {
        master_key: generate_master_key(hd_serial, &salt_hex, &date),
        hd_serial: hd_serial.to_string(),
        generated_at: generated_at.to_string(),
        valid_until: valid_until.to_string(),
    }
}

pub fn linux_mount_candidates() -> &'static [&'static str] {
    &[
        "/media/usb",
        "/mnt/usb",
        "/media/external",
        "/mnt/external",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_stable_for_same_inputs() {
        let a = generate_master_key("ABC123", "deadbeef", "20260526");
        let b = generate_master_key("ABC123", "deadbeef", "20260526");
        assert_eq!(a, b);
        assert_ne!(a, generate_master_key("ABC124", "deadbeef", "20260526"));
    }

    #[test]
    fn builds_session_record() {
        let s = build_session("SER123", &[1, 2, 3, 4], "2026-05-26T10:00:00", "2026-05-26T23:59:59");
        assert_eq!(s.hd_serial, "SER123");
        assert!(!s.master_key.is_empty());
    }
}
