#[derive(Debug, Clone, PartialEq)]
pub enum GpuClass {
    Pascal,        // GTX 10xx, P100, P106
    Turing,        // RTX 20xx
    AmpereOrNewer, // RTX 30xx+
    Unknown,
}

pub fn detect_gpu_class(vendor_id: u32, device_id: u32) -> GpuClass {
    if vendor_id != 0x10DE {
        return GpuClass::Unknown;
    }
    match device_id {
        0x1B00..=0x1D00 => GpuClass::Pascal,
        0x1F00..=0x1FBF => GpuClass::Turing,
        0x2200..=0x2600 => GpuClass::AmpereOrNewer,
        _ => GpuClass::Unknown,
    }
}

/// Detect from nvidia-smi output (runtime detection)
pub fn detect_from_gpu_name(name: &str) -> GpuClass {
    let n = name.to_lowercase();
    if n.contains("p100") || n.contains("p106") || n.contains("1060")
        || n.contains("1070") || n.contains("1080") || n.contains("p1000")
    {
        GpuClass::Pascal
    } else if n.contains("2060") || n.contains("2070") || n.contains("2080") {
        GpuClass::Turing
    } else if n.contains("3060") || n.contains("3070") || n.contains("3080")
        || n.contains("3090") || n.contains("4060") || n.contains("4070")
        || n.contains("4080") || n.contains("4090")
    {
        GpuClass::AmpereOrNewer
    } else {
        GpuClass::Unknown
    }
}
