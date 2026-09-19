//! Portable drive identity primitives from `drive_identity.py`.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DrivePermissions {
    pub draw_bbox: bool,
    pub preprocess_level: String,
    pub upload_anomalies: bool,
    pub generate_kmz: bool,
    pub push_to_github: bool,
    pub approve_others: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveIdentity {
    pub drive_id: String,
    pub owner: String,
    pub created_at: String,
    pub last_seen: String,
    pub hostname: String,
    pub tier: String,
    pub permissions: DrivePermissions,
    pub app_id: Option<String>,
    pub webpage_registered: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DriveRegistrationPayload {
    pub drive_id: String,
    pub owner: String,
    pub hostname: String,
    pub platform: String,
    pub registered_at: String,
    pub drive_path: String,
    pub database_exists: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionRecord {
    pub drive_id: String,
    pub app_id: Option<String>,
    pub database_path: String,
    pub kmz_path: String,
    pub session_started: String,
}

pub fn default_admin_permissions() -> DrivePermissions {
    DrivePermissions {
        draw_bbox: true,
        preprocess_level: "full_cuda".into(),
        upload_anomalies: true,
        generate_kmz: true,
        push_to_github: true,
        approve_others: true,
    }
}

pub fn build_registration_payload(
    identity: &DriveIdentity,
    hostname: &str,
    platform: &str,
    registered_at: &str,
    drive_path: &str,
    database_exists: bool,
) -> DriveRegistrationPayload {
    DriveRegistrationPayload {
        drive_id: identity.drive_id.clone(),
        owner: identity.owner.clone(),
        hostname: hostname.into(),
        platform: platform.into(),
        registered_at: registered_at.into(),
        drive_path: drive_path.into(),
        database_exists,
    }
}

pub fn build_session_record(
    identity: &DriveIdentity,
    database_path: &str,
    kmz_path: &str,
    started_at: &str,
) -> SessionRecord {
    SessionRecord {
        drive_id: identity.drive_id.clone(),
        app_id: identity.app_id.clone(),
        database_path: database_path.into(),
        kmz_path: kmz_path.into(),
        session_started: started_at.into(),
    }
}
