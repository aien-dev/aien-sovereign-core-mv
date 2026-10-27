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

//! Telemetry source for Linux and DGX Spark GB10.

use aien_platform::{PlatformError, TelemetrySnapshot, TelemetrySource};
use std::fs;

pub struct LinuxTelemetrySource;

impl LinuxTelemetrySource {
    pub fn new() -> Self {
        Self
    }
}

impl Default for LinuxTelemetrySource {
    fn default() -> Self {
        Self::new()
    }
}

impl TelemetrySource for LinuxTelemetrySource {
    fn snapshot(&self) -> Result<TelemetrySnapshot, PlatformError> {
        let mut total_mem = 0usize;
        let mut avail_mem = 0usize;

        if let Ok(meminfo) = fs::read_to_string("/proc/meminfo") {
            for line in meminfo.lines() {
                if line.starts_with("MemTotal:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        total_mem = val.parse::<usize>().unwrap_or(0) * 1024;
                    }
                } else if line.starts_with("MemAvailable:") {
                    if let Some(val) = line.split_whitespace().nth(1) {
                        avail_mem = val.parse::<usize>().unwrap_or(0) * 1024;
                    }
                }
            }
        }

        let used_mem = total_mem.saturating_sub(avail_mem);
        let target = if cfg!(target_arch = "aarch64") {
            "gb10"
        } else {
            "x86_64-linux"
        };

        Ok(TelemetrySnapshot {
            architecture: std::env::consts::ARCH.to_string(),
            target: target.to_string(),
            clock_ticks: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_millis() as u64)
                .unwrap_or(0),
            memory_used_bytes: used_mem,
            memory_total_bytes: total_mem,
            active_sequences: 0,
        })
    }
}
