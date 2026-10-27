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

use std::process::Command;

#[derive(Debug, Clone, Default)]
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
}

pub fn get_gpu_telemetry() -> GpuTelemetry {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=name,temperature.gpu,memory.used,memory.total,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output();

    if let Ok(out) = output {
        if out.status.success() {
            let text = String::from_utf8_lossy(&out.stdout);
            let parts: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
            if parts.len() >= 5 {
                return GpuTelemetry {
                    name: Some(parts[0].to_string()),
                    temp_c: parts[1].parse().ok(),
                    used_mb: parts[2].parse().ok(),
                    total_mb: parts[3].parse().ok(),
                    util_pct: parts[4].parse().ok(),
                    available: true,
                };
            }
        }
    }

    // Epistemic honesty: return explicit sensor unavailability, never fake synthetic data
    GpuTelemetry {
        name: None,
        temp_c: None,
        used_mb: None,
        total_mb: None,
        util_pct: None,
        available: false,
    }
}
