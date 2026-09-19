//! Cooperative-matrix probe.
//!
//! Two paths:
//!
//! 1. `probe_by_name(name)` — heuristic fallback by GPU class.
//!    Pascal → unsupported, Turing/Ampere → assume 16×16×16 fp16.
//!    Used when the runtime can't reach Vulkan directly.
//!
//! 2. `probe_via_vulkan()` — authoritative. Spins up a Vulkan instance via
//!    `ash`, walks every physical device, and reports the actual
//!    `VkCooperativeMatrixPropertiesKHR` array the driver advertises.
//!    A `usable` flag is set when at least one supported configuration
//!    matches what we plan to dispatch (fp16 A/B, fp32 accumulator,
//!    subgroup scope).
//!
//! Both paths return `CoopMatSupport`; the runtime picks one based on
//! whether `ash` initialization succeeded.

use std::ffi::CStr;

use ash::vk;
use tracing::{debug, info, warn};

use super::gpu_probe::{detect_from_gpu_name, GpuClass};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CoopMatVariant {
    None,
    Khr,
    Nv,
}

#[derive(Debug, Clone)]
pub struct CoopMatSupport {
    pub class: GpuClass,
    pub usable: bool,
    pub fp16_tile: (u32, u32, u32),
    /// Raw property entries reported by the driver. Empty for the heuristic path.
    pub raw: Vec<CoopMatProp>,
    /// GPU display name from `VkPhysicalDeviceProperties` (when probed via Vulkan).
    pub device_name: String,
    /// Which Vulkan extension exposed cooperative-matrix on this device.
    pub variant: CoopMatVariant,
}

#[derive(Debug, Clone, Copy)]
pub struct CoopMatProp {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub a_type: i32,
    pub b_type: i32,
    pub c_type: i32,
    pub result_type: i32,
    pub saturating_accumulation: bool,
    pub scope: i32,
}

pub fn probe_by_name(gpu_name: &str) -> CoopMatSupport {
    let class = detect_from_gpu_name(gpu_name);
    let (usable, fp16_tile) = match class {
        GpuClass::Pascal => (false, (0, 0, 0)),
        GpuClass::Turing | GpuClass::AmpereOrNewer => (true, (16, 16, 16)),
        GpuClass::Unknown => (false, (0, 0, 0)),
    };
    CoopMatSupport {
        class,
        usable,
        fp16_tile,
        raw: Vec::new(),
        device_name: gpu_name.to_string(),
        variant: CoopMatVariant::None,
    }
}

/// Authoritative probe via Vulkan. Returns per-physical-device support
/// reports. Unsafe FFI is contained inside this function.
///
/// On any unrecoverable failure (no Vulkan, no extension exposed, etc.)
/// returns an empty Vec — callers should fall back to `probe_by_name`.
pub fn probe_via_vulkan() -> Vec<CoopMatSupport> {
    let entry = match unsafe { ash::Entry::load() } {
        Ok(e) => e,
        Err(err) => {
            warn!("ash entry load failed: {} — falling back to heuristic probe", err);
            return Vec::new();
        }
    };

    // Instance with VK_KHR_get_physical_device_properties2 (core in 1.1)
    let app_info = vk::ApplicationInfo::default()
        .application_name(c"cesarops-coopmat-probe")
        .api_version(vk::API_VERSION_1_3);

    let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);

    let instance = match unsafe { entry.create_instance(&instance_create_info, None) } {
        Ok(i) => i,
        Err(err) => {
            warn!("vkCreateInstance failed: {:?} — falling back", err);
            return Vec::new();
        }
    };

    let physical_devices = match unsafe { instance.enumerate_physical_devices() } {
        Ok(p) => p,
        Err(err) => {
            warn!("enumerate_physical_devices failed: {:?}", err);
            unsafe { instance.destroy_instance(None) };
            return Vec::new();
        }
    };

    let mut results = Vec::new();

    for pd in physical_devices {
        let props = unsafe { instance.get_physical_device_properties(pd) };
        let device_name = unsafe { CStr::from_ptr(props.device_name.as_ptr()) }
            .to_string_lossy()
            .into_owned();

        let class = detect_from_gpu_name(&device_name);

        // Confirm VK_KHR_cooperative_matrix is exposed by this device.
        let exts = match unsafe { instance.enumerate_device_extension_properties(pd) } {
            Ok(e) => e,
            Err(err) => {
                debug!("enumerate_device_extension_properties({:?}) failed: {:?}", device_name, err);
                results.push(CoopMatSupport {
                    class,
                    usable: false,
                    fp16_tile: (0, 0, 0),
                    raw: Vec::new(),
                    device_name,
                    variant: CoopMatVariant::None,
                });
                continue;
            }
        };
        let mut variant = CoopMatVariant::None;
        for e in &exts {
            let name = unsafe { CStr::from_ptr(e.extension_name.as_ptr()) }
                .to_string_lossy();
            if name == "VK_KHR_cooperative_matrix" {
                variant = CoopMatVariant::Khr;
                break;
            }
            if name == "VK_NV_cooperative_matrix" && variant == CoopMatVariant::None {
                variant = CoopMatVariant::Nv;
            }
        }
        if variant == CoopMatVariant::None {
            results.push(CoopMatSupport {
                class,
                usable: false,
                fp16_tile: (0, 0, 0),
                raw: Vec::new(),
                device_name,
                variant: CoopMatVariant::None,
            });
            continue;
        }

        // Both KHR and NV variants expose a per-physical-device property list,
        // but with DIFFERENT struct layouts:
        //
        //   KHR: m,n,k, a,b,c,result types, saturating(Bool32), scope
        //   NV : m,n,k, a,b,c,d types, scope (no saturating field, d == result)
        //
        // We dispatch separately so we read the right number of bytes and
        // map the fields to a uniform CoopMatProp.
        let raw: Vec<CoopMatProp>;
        match variant {
            CoopMatVariant::Khr => {
                let raw_fn = unsafe { entry.get_instance_proc_addr(
                    instance.handle(),
                    c"vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR".as_ptr(),
                ) };
                let Some(raw_fn) = raw_fn else {
                    warn!("KHR variant advertised but entry point missing");
                    results.push(CoopMatSupport {
                        class, usable: false, fp16_tile: (0, 0, 0),
                        raw: Vec::new(), device_name, variant,
                    });
                    continue;
                };
                type GetKhr = unsafe extern "system" fn(
                    vk::PhysicalDevice, *mut u32, *mut vk::CooperativeMatrixPropertiesKHR,
                ) -> vk::Result;
                let f: GetKhr = unsafe { std::mem::transmute(raw_fn) };
                let mut count: u32 = 0;
                let r = unsafe { f(pd, &mut count, std::ptr::null_mut()) };
                if r != vk::Result::SUCCESS {
                    warn!("KHR coopmat count failed: {:?}", r);
                    results.push(CoopMatSupport {
                        class, usable: false, fp16_tile: (0, 0, 0),
                        raw: Vec::new(), device_name, variant,
                    });
                    continue;
                }
                let mut props_list = vec![vk::CooperativeMatrixPropertiesKHR::default(); count as usize];
                unsafe { f(pd, &mut count, props_list.as_mut_ptr()) };
                props_list.truncate(count as usize);
                raw = props_list.iter().map(|p| CoopMatProp {
                    m: p.m_size, n: p.n_size, k: p.k_size,
                    a_type: p.a_type.as_raw(),
                    b_type: p.b_type.as_raw(),
                    c_type: p.c_type.as_raw(),
                    result_type: p.result_type.as_raw(),
                    saturating_accumulation: p.saturating_accumulation == vk::TRUE,
                    scope: p.scope.as_raw(),
                }).collect();
            }
            CoopMatVariant::Nv => {
                let raw_fn = unsafe { entry.get_instance_proc_addr(
                    instance.handle(),
                    c"vkGetPhysicalDeviceCooperativeMatrixPropertiesNV".as_ptr(),
                ) };
                let Some(raw_fn) = raw_fn else {
                    warn!("NV variant advertised but entry point missing");
                    results.push(CoopMatSupport {
                        class, usable: false, fp16_tile: (0, 0, 0),
                        raw: Vec::new(), device_name, variant,
                    });
                    continue;
                };
                type GetNv = unsafe extern "system" fn(
                    vk::PhysicalDevice, *mut u32, *mut vk::CooperativeMatrixPropertiesNV,
                ) -> vk::Result;
                let f: GetNv = unsafe { std::mem::transmute(raw_fn) };
                let mut count: u32 = 0;
                let r = unsafe { f(pd, &mut count, std::ptr::null_mut()) };
                if r != vk::Result::SUCCESS {
                    warn!("NV coopmat count failed: {:?}", r);
                    results.push(CoopMatSupport {
                        class, usable: false, fp16_tile: (0, 0, 0),
                        raw: Vec::new(), device_name, variant,
                    });
                    continue;
                }
                let mut props_list = vec![vk::CooperativeMatrixPropertiesNV::default(); count as usize];
                unsafe { f(pd, &mut count, props_list.as_mut_ptr()) };
                props_list.truncate(count as usize);
                raw = props_list.iter().map(|p| CoopMatProp {
                    m: p.m_size, n: p.n_size, k: p.k_size,
                    // ComponentTypeNV / KHR share the same enum integer values
                    // for the variants we look at (FLOAT16, FLOAT32, etc.).
                    a_type: p.a_type.as_raw(),
                    b_type: p.b_type.as_raw(),
                    c_type: p.c_type.as_raw(),
                    result_type: p.d_type.as_raw(),
                    saturating_accumulation: false,
                    scope: p.scope.as_raw(),
                }).collect();
            }
            CoopMatVariant::None => unreachable!(),
        }

        // "Usable" = at least one entry with fp16 A/B + fp32 accumulator + subgroup scope.
        let want_fp16 = vk::ComponentTypeKHR::FLOAT16.as_raw();
        let want_fp32 = vk::ComponentTypeKHR::FLOAT32.as_raw();
        let want_subgroup = vk::ScopeKHR::SUBGROUP.as_raw();
        let usable_entry = raw.iter().find(|p| {
            p.a_type == want_fp16
                && p.b_type == want_fp16
                && p.c_type == want_fp32
                && p.scope == want_subgroup
        }).cloned();
        let (usable, fp16_tile) = match usable_entry {
            Some(p) => (true, (p.m, p.n, p.k)),
            None => (false, (0, 0, 0)),
        };

        info!(
            "coopmat probe: {} class={:?} variant={:?} usable={} tile={:?} entries={}",
            device_name, class, variant, usable, fp16_tile, raw.len()
        );
        results.push(CoopMatSupport {
            class,
            usable,
            fp16_tile,
            raw,
            device_name,
            variant,
        });
    }

    unsafe { instance.destroy_instance(None) };
    results
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heuristic_pascal_unsupported() {
        let p = probe_by_name("Tesla P100-PCIE-16GB");
        assert!(!p.usable);
    }

    #[test]
    fn heuristic_turing_supported() {
        let p = probe_by_name("NVIDIA GeForce RTX 2060 SUPER");
        assert!(p.usable);
        assert_eq!(p.fp16_tile, (16, 16, 16));
    }
}
