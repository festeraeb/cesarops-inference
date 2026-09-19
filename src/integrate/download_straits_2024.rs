//! Straits 2024 targeted band downloads — port of `download_straits_2024.py`.

use serde::{Deserialize, Serialize};

pub const BASE_S2: &str =
    "https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/16/T/FR/2024/9/S2B_16TFR_20240903_0_L2A";
pub const SCENE_L9: &str = "LC09_L2SP_022028_20240903_20240904_02_T1";
pub const PC_SIGN_API: &str = "https://planetarycomputer.microsoft.com/api/sas/v1/sign";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct BandDownload {
    pub url: String,
    pub filename: String,
    pub needs_pc_sign: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SkipReason {
    pub filename: String,
    pub reason: String,
}

pub fn straits_2024_download_plan() -> Vec<BandDownload> {
    let base_l9 = format!(
        "https://landsateuwest.blob.core.windows.net/landsat-c2/level-2/standard/oli-tirs/2024/022/028/{SCENE_L9}"
    );
    vec![
        BandDownload {
            url: format!("{BASE_S2}/B02.tif"),
            filename: "S2B_16TFR_20240903_0_L2A.blue.tif".into(),
            needs_pc_sign: false,
        },
        BandDownload {
            url: format!("{BASE_S2}/B03.tif"),
            filename: "S2B_16TFR_20240903_0_L2A.green.tif".into(),
            needs_pc_sign: false,
        },
        BandDownload {
            url: format!("{BASE_S2}/B04.tif"),
            filename: "S2B_16TFR_20240903_0_L2A.red.tif".into(),
            needs_pc_sign: false,
        },
        BandDownload {
            url: format!("{BASE_S2}/B11.tif"),
            filename: "S2B_16TFR_20240903_0_L2A.swir16.tif".into(),
            needs_pc_sign: false,
        },
        BandDownload {
            url: format!("{base_l9}/{SCENE_L9}_ST_B10.TIF"),
            filename: format!("{SCENE_L9}.lwir11.tif"),
            needs_pc_sign: true,
        },
        BandDownload {
            url: format!("{base_l9}/{SCENE_L9}_SR_B2.TIF"),
            filename: format!("{SCENE_L9}.blue.tif"),
            needs_pc_sign: true,
        },
        BandDownload {
            url: format!("{base_l9}/{SCENE_L9}_SR_B6.TIF"),
            filename: format!("{SCENE_L9}.swir16.tif"),
            needs_pc_sign: true,
        },
    ]
}

pub fn should_skip_existing(size_bytes: u64, min_bytes: u64) -> bool {
    size_bytes > min_bytes
}

pub fn pc_sign_request_url(unsigned_url: &str) -> String {
    format!("{PC_SIGN_API}?href={unsigned_url}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn plan_has_seven_bands() {
        assert_eq!(straits_2024_download_plan().len(), 7);
    }
}
