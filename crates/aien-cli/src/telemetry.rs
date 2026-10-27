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

//! GPU status read from kernel sysfs only. No NVIDIA userspace (nvidia-smi,
//! NVML, libcuda) is executed or linked. Every read is a plain file read;
//! nothing under /sys is written and the device power state is never touched.
//!
//! Field sources:
//! - `available`: a PCI display-class device (class 0x03xxxx) with a bound
//!   kernel driver, from `/sys/bus/pci/devices/*/{class,driver}`
//!   (`aien_inference_abi::pci_gpu`).
//! - `name`: `vendor`/`device` IDs of that device, mapped through the
//!   checked table in `pci_gpu`; unknown IDs are shown as `PCI vvvv:dddd`.
//! - `temp_c`: the device's own hwmon (`<dev>/hwmon/hwmon*/temp1_input`) when
//!   the driver exports one, else the firmware ACPI GPU thermal zone
//!   (`/sys/class/thermal/thermal_zone*` whose `device/path` is `\_TZ_.TGPU`).
//!   On the DGX Spark the NVIDIA driver exports no hwmon, so the value is the
//!   ACPI zone, a different sensor from the one nvidia-smi reports.
//! - `used_mb`, `total_mb`, `util_pct`: no kernel source exists without the
//!   NVIDIA resource manager, so they are always `None`. nvidia-smi already
//!   reported memory as `[N/A]` on the GB10 (unified memory).
//!
//! MIGRATION-PENDING: this Rust reader is a stop-gap. Its successor is a
//! seeded-Omega (.om) sysfs reader of the same files with the same
//! unavailable-field contract (PROPOSED, not yet written). Delete this module
//! when that lands.

use aien_inference_abi::pci_gpu::find_display_gpu;
use std::fs;
use std::path::{Path, PathBuf};

const ACPI_GPU_THERMAL_ZONE: &str = "\\_TZ_.TGPU";

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct GpuTelemetry {
    pub name: Option<String>,
    pub temp_c: Option<u32>,
    pub used_mb: Option<u64>,
    pub total_mb: Option<u64>,
    pub util_pct: Option<u32>,
    pub available: bool,
}

impl GpuTelemetry {
    pub fn display_name(&self) -> &str {
        self.name
            .as_deref()
            .unwrap_or("Unavailable / Sensor Offline")
    }

    pub fn display_temp(&self) -> String {
        self.temp_c
            .map(|t| format!("{}°C", t))
            .unwrap_or_else(|| "N/A".to_string())
    }

    pub fn display_vram(&self) -> String {
        match (self.used_mb, self.total_mb) {
            (Some(u), Some(t)) => format!("{} MB / {} MB", u, t),
            _ => "N/A".to_string(),
        }
    }

    pub fn display_util(&self) -> String {
        self.util_pct
            .map(|u| format!("{}%", u))
            .unwrap_or_else(|| "N/A".to_string())
    }
}

pub fn get_gpu_telemetry() -> GpuTelemetry {
    read_gpu_telemetry(Path::new("/"))
}

/// Read telemetry from a filesystem rooted at `root` (`/` in production, a
/// fixture tree in tests).
fn read_gpu_telemetry(root: &Path) -> GpuTelemetry {
    // Epistemic honesty: return explicit sensor unavailability, never fake synthetic data
    let Some(gpu) = find_display_gpu(&root.join("sys/bus/pci/devices")) else {
        return GpuTelemetry::default();
    };
    let temp_c = device_hwmon_temp(&gpu.path)
        .or_else(|| acpi_gpu_zone_temp(&root.join("sys/class/thermal")))
        .and_then(millicelsius_to_celsius);
    GpuTelemetry {
        name: Some(gpu.name()),
        temp_c,
        used_mb: None,
        total_mb: None,
        util_pct: None,
        available: true,
    }
}

fn device_hwmon_temp(dev: &Path) -> Option<i64> {
    let mut hwmons: Vec<PathBuf> = fs::read_dir(dev.join("hwmon"))
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .collect();
    hwmons.sort();
    hwmons.iter().find_map(|h| read_int(&h.join("temp1_input")))
}

fn acpi_gpu_zone_temp(thermal: &Path) -> Option<i64> {
    let mut zones: Vec<PathBuf> = fs::read_dir(thermal)
        .ok()?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("thermal_zone"))
        })
        .collect();
    zones.sort();
    zones
        .iter()
        .find(|z| {
            fs::read_to_string(z.join("device/path"))
                .is_ok_and(|s| s.trim() == ACPI_GPU_THERMAL_ZONE)
        })
        .and_then(|z| read_int(&z.join("temp")))
}

/// sysfs temperatures are millidegrees Celsius; round to the nearest degree.
/// A negative or out-of-range reading cannot be shown in `u32` and is
/// reported as unavailable.
fn millicelsius_to_celsius(m: i64) -> Option<u32> {
    if m < 0 {
        return None;
    }
    u32::try_from((m + 500) / 1000).ok()
}

fn read_int(p: &Path) -> Option<i64> {
    fs::read_to_string(p).ok()?.trim().parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn put(root: &Path, rel: &str, body: &str) {
        let p = root.join(rel);
        fs::create_dir_all(p.parent().unwrap()).unwrap();
        fs::write(p, body).unwrap();
    }

    /// The DGX Spark layout observed read-only on 2026-10-27: GB10 at
    /// 000f:01:00.0 bound to `nvidia`, no device hwmon, seven ACPI zones with
    /// `\_TZ_.TGPU` at zone 5.
    fn spark_tree() -> tempfile::TempDir {
        let t = tempfile::tempdir().unwrap();
        let r = t.path();
        let gpu = "sys/bus/pci/devices/000f:01:00.0";
        put(r, &format!("{gpu}/class"), "0x030000\n");
        put(r, &format!("{gpu}/vendor"), "0x10de\n");
        put(r, &format!("{gpu}/device"), "0x2e12\n");
        put(r, &format!("{gpu}/driver/bind"), "");
        let nic = "sys/bus/pci/devices/0000:01:00.0";
        put(r, &format!("{nic}/class"), "0x020000\n");
        put(r, &format!("{nic}/vendor"), "0x15b3\n");
        put(r, &format!("{nic}/device"), "0x1021\n");
        put(r, &format!("{nic}/driver/bind"), "");
        let zones = [
            ("TSOC", 51800),
            ("TS0E", 43200),
            ("TS0P", 40900),
            ("TS1E", 49400),
            ("TS1P", 51800),
            ("TGPU", 41700),
            ("TUNC", 42700),
        ];
        for (i, (name, mc)) in zones.iter().enumerate() {
            let z = format!("sys/class/thermal/thermal_zone{i}");
            put(r, &format!("{z}/temp"), &format!("{mc}\n"));
            put(r, &format!("{z}/device/path"), &format!("\\_TZ_.{name}\n"));
        }
        t
    }

    #[test]
    fn spark_layout_reads_gb10_and_acpi_gpu_zone() {
        let t = spark_tree();
        let g = read_gpu_telemetry(t.path());
        assert_eq!(
            g,
            GpuTelemetry {
                name: Some("NVIDIA GB10".into()),
                temp_c: Some(42),
                used_mb: None,
                total_mb: None,
                util_pct: None,
                available: true,
            }
        );
        assert_eq!(g.display_vram(), "N/A");
        assert_eq!(g.display_util(), "N/A");
    }

    #[test]
    fn device_hwmon_wins_over_acpi_zone() {
        let t = spark_tree();
        put(
            t.path(),
            "sys/bus/pci/devices/000f:01:00.0/hwmon/hwmon9/temp1_input",
            "38499\n",
        );
        assert_eq!(read_gpu_telemetry(t.path()).temp_c, Some(38));
    }

    #[test]
    fn no_gpu_zone_means_no_temperature() {
        let t = spark_tree();
        put(
            t.path(),
            "sys/class/thermal/thermal_zone5/device/path",
            "\\_TZ_.TXXX\n",
        );
        let g = read_gpu_telemetry(t.path());
        assert!(g.available);
        assert_eq!(g.temp_c, None);
        assert_eq!(g.display_temp(), "N/A");
    }

    #[test]
    fn unbound_gpu_is_unavailable() {
        let t = spark_tree();
        fs::remove_dir_all(t.path().join("sys/bus/pci/devices/000f:01:00.0/driver")).unwrap();
        assert_eq!(read_gpu_telemetry(t.path()), GpuTelemetry::default());
    }

    #[test]
    fn empty_root_is_unavailable() {
        let t = tempfile::tempdir().unwrap();
        let g = read_gpu_telemetry(t.path());
        assert_eq!(g, GpuTelemetry::default());
        assert_eq!(g.display_name(), "Unavailable / Sensor Offline");
    }

    #[test]
    fn unknown_device_id_is_named_by_pci_ids_not_guessed() {
        let t = spark_tree();
        put(
            t.path(),
            "sys/bus/pci/devices/000f:01:00.0/device",
            "0x2b85\n",
        );
        assert_eq!(
            read_gpu_telemetry(t.path()).name.as_deref(),
            Some("PCI 10de:2b85")
        );
    }

    #[test]
    fn nvidia_preferred_over_lower_addressed_display_device() {
        let t = spark_tree();
        let bmc = "sys/bus/pci/devices/0000:00:01.0";
        put(t.path(), &format!("{bmc}/class"), "0x030000\n");
        put(t.path(), &format!("{bmc}/vendor"), "0x1a03\n");
        put(t.path(), &format!("{bmc}/device"), "0x2000\n");
        put(t.path(), &format!("{bmc}/driver/bind"), "");
        assert_eq!(
            read_gpu_telemetry(t.path()).name.as_deref(),
            Some("NVIDIA GB10")
        );
    }

    #[test]
    fn millicelsius_rounding_and_bounds() {
        assert_eq!(millicelsius_to_celsius(0), Some(0));
        assert_eq!(millicelsius_to_celsius(41499), Some(41));
        assert_eq!(millicelsius_to_celsius(41500), Some(42));
        assert_eq!(millicelsius_to_celsius(-1), None);
    }

    #[test]
    fn module_never_spawns_a_process() {
        // Pieces are joined at run time so this test's own text cannot match.
        let src = include_str!("telemetry.rs");
        assert!(!src.contains(&["Command", "::new("].concat()));
        assert!(!src.contains(&["std::", "process"].concat()));
    }
}
