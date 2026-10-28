// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! GPU discovery from kernel sysfs (`/sys/bus/pci/devices`), read-only.
//! Replaces `nvidia-smi --query-gpu=name` probes: no NVIDIA userspace is run.
//!
//! MIGRATION-PENDING: Rust stop-gap. Its successor is a seeded-Omega (.om)
//! sysfs reader of the same files (PROPOSED, not yet written). Delete this
//! module when that lands.

use std::fs;
use std::path::{Path, PathBuf};

pub const NVIDIA_VENDOR: u16 = 0x10de;

/// PCI (vendor, device) IDs whose name has been checked against nvidia-smi
/// on real hardware. pci.ids does not list 10de:2e12.
const KNOWN_GPUS: &[(u16, u16, &str)] = &[(NVIDIA_VENDOR, 0x2e12, "NVIDIA GB10")];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PciGpu {
    pub path: PathBuf,
    pub vendor: u16,
    pub device: u16,
    /// Name of the bound kernel driver (`driver` symlink target), if any
    /// can be read.
    pub driver: Option<String>,
}

impl PciGpu {
    /// Checked name for known IDs, else `PCI vvvv:dddd`. Never guessed.
    pub fn name(&self) -> String {
        KNOWN_GPUS
            .iter()
            .find(|(v, d, _)| *v == self.vendor && *d == self.device)
            .map(|(_, _, n)| (*n).to_string())
            .unwrap_or_else(|| format!("PCI {:04x}:{:04x}", self.vendor, self.device))
    }

    pub fn is_gb10(&self) -> bool {
        self.vendor == NVIDIA_VENDOR && self.device == 0x2e12
    }
}

/// First driver-bound PCI display controller (class 0x03xxxx) under
/// `pci_devices`, NVIDIA first, then by PCI address so the choice is
/// deterministic.
pub fn find_display_gpu(pci_devices: &Path) -> Option<PciGpu> {
    let mut found: Vec<PciGpu> = fs::read_dir(pci_devices)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| read_hex(&p.join("class")).is_some_and(|c| c >> 16 == 0x03))
        .filter(|p| p.join("driver").exists())
        .filter_map(|p| {
            let vendor = u16::try_from(read_hex(&p.join("vendor"))?).ok()?;
            let device = u16::try_from(read_hex(&p.join("device"))?).ok()?;
            let driver = fs::read_link(p.join("driver"))
                .ok()
                .and_then(|l| l.file_name().map(|n| n.to_string_lossy().into_owned()));
            Some(PciGpu {
                path: p,
                vendor,
                device,
                driver,
            })
        })
        .collect();
    found.sort_by(|a, b| {
        (a.vendor != NVIDIA_VENDOR, &a.path).cmp(&(b.vendor != NVIDIA_VENDOR, &b.path))
    });
    found.into_iter().next()
}

fn read_hex(p: &Path) -> Option<u32> {
    let s = fs::read_to_string(p).ok()?;
    let s = s.trim();
    u32::from_str_radix(s.strip_prefix("0x").unwrap_or(s), 16).ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ExecutionSurface;
    use std::os::unix::fs::symlink;

    struct Tree(PathBuf);
    impl Drop for Tree {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn tree(tag: &str) -> Tree {
        let p = std::env::temp_dir().join(format!("pci_gpu-{}-{}", std::process::id(), tag));
        let _ = fs::remove_dir_all(&p);
        fs::create_dir_all(p.join("drivers/nvidia")).unwrap();
        fs::create_dir_all(p.join("drivers/vfio-pci")).unwrap();
        fs::create_dir_all(p.join("devices")).unwrap();
        Tree(p)
    }

    fn dev(t: &Tree, addr: &str, class: &str, vendor: &str, device: &str, driver: Option<&str>) {
        let d = t.0.join("devices").join(addr);
        fs::create_dir_all(&d).unwrap();
        fs::write(d.join("class"), format!("{class}\n")).unwrap();
        fs::write(d.join("vendor"), format!("{vendor}\n")).unwrap();
        fs::write(d.join("device"), format!("{device}\n")).unwrap();
        if let Some(drv) = driver {
            symlink(t.0.join("drivers").join(drv), d.join("driver")).unwrap();
        }
    }

    /// DGX Spark as read on 2026-10-27: GB10 at 000f:01:00.0 bound to
    /// `nvidia`, plus a bound non-display NIC.
    fn spark(tag: &str) -> Tree {
        let t = tree(tag);
        dev(
            &t,
            "000f:01:00.0",
            "0x030000",
            "0x10de",
            "0x2e12",
            Some("nvidia"),
        );
        dev(
            &t,
            "0000:01:00.0",
            "0x020000",
            "0x15b3",
            "0x1021",
            Some("vfio-pci"),
        );
        t
    }

    #[test]
    fn spark_gb10_found_with_driver_name() {
        let t = spark("found");
        let g = find_display_gpu(&t.0.join("devices")).unwrap();
        assert_eq!((g.vendor, g.device), (0x10de, 0x2e12));
        assert_eq!(g.driver.as_deref(), Some("nvidia"));
        assert_eq!(g.name(), "NVIDIA GB10");
        assert!(g.is_gb10());
    }

    #[test]
    fn spark_surface_is_grace_blackwell() {
        let t = spark("surface");
        assert_eq!(
            ExecutionSurface::nvidia_surface(&t.0.join("devices")),
            Some(ExecutionSurface::NvidiaGraceBlackwell {
                model: "NVIDIA GB10".into(),
                unified_memory_gb: 121,
            })
        );
    }

    #[test]
    fn gpu_on_vfio_is_not_an_accelerated_surface() {
        let t = tree("vfio");
        dev(
            &t,
            "000f:01:00.0",
            "0x030000",
            "0x10de",
            "0x2e12",
            Some("vfio-pci"),
        );
        assert!(find_display_gpu(&t.0.join("devices")).is_some());
        assert_eq!(ExecutionSurface::nvidia_surface(&t.0.join("devices")), None);
    }

    #[test]
    fn unbound_or_absent_gpu_is_none() {
        let t = tree("unbound");
        dev(&t, "000f:01:00.0", "0x030000", "0x10de", "0x2e12", None);
        assert_eq!(find_display_gpu(&t.0.join("devices")), None);
        assert_eq!(find_display_gpu(&t.0.join("missing")), None);
    }

    #[test]
    fn unknown_nvidia_gpu_has_unknown_vram_not_a_guess() {
        let t = tree("unknown");
        dev(
            &t,
            "0000:41:00.0",
            "0x030000",
            "0x10de",
            "0x2b85",
            Some("nvidia"),
        );
        let s = ExecutionSurface::nvidia_surface(&t.0.join("devices")).unwrap();
        assert_eq!(
            s,
            ExecutionSurface::NvidiaCuda {
                gpu_name: "PCI 10de:2b85".into(),
                vram_gb: None,
            }
        );
        assert_eq!(s.display_name(), "PCI 10de:2b85 (VRAM size unknown)");
        assert!(s.is_accelerated_gpu());
    }

    #[test]
    fn nvidia_preferred_over_lower_addressed_display_device() {
        let t = spark("prefer");
        dev(
            &t,
            "0000:00:01.0",
            "0x030000",
            "0x1a03",
            "0x2000",
            Some("vfio-pci"),
        );
        assert_eq!(
            find_display_gpu(&t.0.join("devices")).unwrap().vendor,
            NVIDIA_VENDOR
        );
    }

    #[test]
    fn detection_never_spawns_nvidia_smi() {
        // Pieces are joined at run time so this test's own text cannot match.
        let needle = ["Command::", "new(\"nvidia", "-smi\")"].concat();
        assert!(!include_str!("lib.rs").contains(&needle));
        assert!(!include_str!("pci_gpu.rs").contains(&["Command", "::new("].concat()));
    }
}
