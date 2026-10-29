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

//! Host-only measurement of the checkpoint load path (no GPU, no chip): peak and final resident
//! memory of the process for the unchanged host f32 load and for the GB10 resident load
//! (sovereign-core #277 cut C) with a stub uploader that only checksums the bf16 bits.
//!
//!   load_rss host|resident <model_dir>
use aien_inference_abi::{
    load_model_config, load_resident_weights, ResidentHandle, ResidentUploader, ResidentWeight,
    TransformerWeights,
};
use std::sync::{Arc, Mutex};

fn status_kib(key: &str) -> u64 {
    std::fs::read_to_string("/proc/self/status")
        .unwrap()
        .lines()
        .find_map(|l| l.strip_prefix(key))
        .and_then(|v| v.trim().trim_end_matches("kB").trim().parse().ok())
        .unwrap_or(0)
}

fn gib(kib: u64) -> f64 {
    kib as f64 / 1024.0 / 1024.0
}

#[derive(Debug)]
struct Stub;
impl ResidentHandle for Stub {
    fn matmul_f32(&self, _: usize, _: &[f32], _: &mut [f32]) -> Result<u64, String> {
        Err("stub".into())
    }
}

struct StubUploader {
    log: Mutex<Vec<(usize, u64)>>,
    bytes: Mutex<u64>,
}
impl ResidentUploader for StubUploader {
    fn upload_bf16(&self, k: usize, n: usize, bits: &[u16]) -> Result<ResidentWeight, String> {
        let sum = bits.iter().fold(0u64, |a, b| a.wrapping_add(*b as u64));
        std::hint::black_box(sum);
        *self.bytes.lock().unwrap() += (k * n * 2) as u64;
        self.log.lock().unwrap().push((k * n, status_kib("VmRSS:")));
        Ok(ResidentWeight::new(k, n, Arc::new(Stub)))
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let (mode, dir) = (args[1].as_str(), std::path::PathBuf::from(&args[2]));
    let config = load_model_config(&dir).expect("config.json");
    let t = std::time::Instant::now();
    let weights = match mode {
        "host" => TransformerWeights::load_from_safetensors(&dir, &config).expect("host load"),
        "resident" => {
            let up = StubUploader {
                log: Mutex::new(Vec::new()),
                bytes: Mutex::new(0),
            };
            let w = load_resident_weights(&dir, &config, &up, &mut |_| {}).expect("resident load");
            let log = up.log.lock().unwrap();
            let max_during_uploads = log.iter().map(|x| x.1).max().unwrap_or(0);
            println!(
                "uploads={} bf16_uploaded={:.3} GiB max_VmRSS_during_uploads={:.3} GiB",
                log.len(),
                gib(*up.bytes.lock().unwrap() / 1024),
                gib(max_during_uploads)
            );
            w
        }
        other => panic!("mode must be host or resident, got {other}"),
    };
    let resident = weights
        .layers
        .iter()
        .filter(|l| l.q_proj.is_resident())
        .count();
    println!(
        "mode={mode} load_secs={:.1} VmHWM(peak)={:.3} GiB VmRSS(after load)={:.3} GiB resident_layers={resident}/{}",
        t.elapsed().as_secs_f64(),
        gib(status_kib("VmHWM:")),
        gib(status_kib("VmRSS:")),
        weights.layers.len()
    );
}
