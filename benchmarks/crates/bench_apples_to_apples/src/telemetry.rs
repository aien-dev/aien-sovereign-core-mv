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

//! Hardware and process telemetry samplers: GPU power, temperature, RSS, and thermal cool-down.

use std::fs;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::Mutex;

/// Instantaneous GPU telemetry sample.
#[derive(Debug, Clone, Copy, Default)]
pub struct GpuSample {
    pub power_watts: f64,
    pub temperature_c: f64,
    #[allow(dead_code)]
    pub gpu_util_pct: f64,
}

/// Query instantaneous GPU power, temperature, and utilization from nvidia-smi.
pub fn query_gpu_telemetry() -> Option<GpuSample> {
    let output = Command::new("nvidia-smi")
        .args([
            "--query-gpu=power.draw,temperature.gpu,utilization.gpu",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .ok()?;

    if !output.status.success() {
        return None;
    }

    let text = String::from_utf8_lossy(&output.stdout);
    let parts: Vec<&str> = text.trim().split(',').map(|s| s.trim()).collect();
    if parts.len() < 3 {
        return None;
    }

    let power_watts = parts[0].parse::<f64>().unwrap_or(0.0);
    let temperature_c = parts[1].parse::<f64>().unwrap_or(0.0);
    let gpu_util_pct = parts[2].parse::<f64>().unwrap_or(0.0);

    Some(GpuSample {
        power_watts,
        temperature_c,
        gpu_util_pct,
    })
}

/// Query resident set size (RSS) in gigabytes for a target PID, or self if pid is None.
pub fn query_process_rss_gb(pid: Option<u32>) -> f64 {
    let status_path = match pid {
        Some(p) => format!("/proc/{}/status", p),
        None => "/proc/self/status".to_string(),
    };

    let content = match fs::read_to_string(&status_path) {
        Ok(c) => c,
        Err(_) => return 0.0,
    };

    for line in content.lines() {
        if line.starts_with("VmRSS:") {
            let parts: Vec<&str> = line.split_whitespace().collect();
            if parts.len() >= 2 {
                if let Ok(kb) = parts[1].parse::<f64>() {
                    return kb / 1_048_576.0; // kB to GB
                }
            }
        }
    }

    0.0
}

/// Background hardware monitor collecting periodic telemetry samples during execution.
pub struct HardwareMonitor {
    stop_signal: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<GpuSample>>>,
    peak_rss_gb: Arc<Mutex<f64>>,
    target_pid: Option<u32>,
}

impl HardwareMonitor {
    pub fn start(target_pid: Option<u32>) -> Self {
        let stop_signal = Arc::new(AtomicBool::new(false));
        let samples = Arc::new(Mutex::new(Vec::new()));
        let peak_rss_gb = Arc::new(Mutex::new(0.0));

        let stop_clone = Arc::clone(&stop_signal);
        let samples_clone = Arc::clone(&samples);
        let peak_rss_clone = Arc::clone(&peak_rss_gb);

        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_millis(100));
            while !stop_clone.load(Ordering::Relaxed) {
                interval.tick().await;

                if let Some(sample) = query_gpu_telemetry() {
                    let mut s = samples_clone.lock().await;
                    s.push(sample);
                }

                let current_rss = query_process_rss_gb(target_pid);
                let mut peak = peak_rss_clone.lock().await;
                if current_rss > *peak {
                    *peak = current_rss;
                }
            }
        });

        Self {
            stop_signal,
            samples,
            peak_rss_gb,
            target_pid,
        }
    }

    pub async fn stop(self) -> (f64, f64, f64) {
        self.stop_signal.store(true, Ordering::Relaxed);
        tokio::time::sleep(Duration::from_millis(150)).await;

        let samples = self.samples.lock().await;
        let mut peak_rss = *self.peak_rss_gb.lock().await;
        let final_rss = query_process_rss_gb(self.target_pid);
        if final_rss > peak_rss {
            peak_rss = final_rss;
        }

        if samples.is_empty() {
            return (0.0, 0.0, peak_rss);
        }

        let mut sum_power = 0.0;
        let mut max_power = 0.0f64;
        for s in samples.iter() {
            sum_power += s.power_watts;
            if s.power_watts > max_power {
                max_power = s.power_watts;
            }
        }

        let avg_power = sum_power / (samples.len() as f64);
        (avg_power, max_power, peak_rss)
    }
}

/// Flushes kernel filesystem caches and drops memory to release unified coherent memory.
pub fn flush_system_caches() {
    let _ = Command::new("sync").status();
    // Attempt dropping page caches via sudo if available
    let _ = Command::new("sudo")
        .args(["-n", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"])
        .status();
}

/// Enforces the thermal stabilization envelope and cache flush between engines.
/// Waits for GPU power draw to settle into the idle envelope (<= target_watts, ~10-12W)
/// and temperature <= max_temp_c.
pub async fn enforce_thermal_cooldown(target_watts: f64, max_temp_c: f64) {
    eprintln!("Initiating thermal cool-down and cache flush sequence...");
    flush_system_caches();

    let start = Instant::now();
    let max_wait = Duration::from_secs(60);

    while start.elapsed() < max_wait {
        if let Some(sample) = query_gpu_telemetry() {
            eprintln!(
                "Thermal envelope: Power={:.1}W (target<={:.1}W), Temp={:.1}C (target<={:.1}C)",
                sample.power_watts, target_watts, sample.temperature_c, max_temp_c
            );

            if sample.power_watts <= target_watts && sample.temperature_c <= max_temp_c {
                eprintln!(
                    "Thermal envelope satisfied in {:.1}s.",
                    start.elapsed().as_secs_f64()
                );
                return;
            }
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }

    eprintln!("Cool-down timeout reached. Proceeding with benchmark.");
}
