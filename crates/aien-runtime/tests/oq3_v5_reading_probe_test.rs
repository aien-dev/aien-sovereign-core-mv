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

//! Prints how the requirement reader reads every goal of the Qwen3 v5 campaign
//! (docs/campaigns/open-model-qwen3/tasks-oq3-v5.json), so a reader change can
//! be compared before and after: `cargo test --test oq3_v5_reading_probe_test
//! -- --nocapture`. It pins nothing; it only fails if the file lists no goals.
use aien_runtime::requirements::analyze;

#[test]
fn print_how_each_oq3_v5_goal_is_read() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../docs/campaigns/open-model-qwen3/tasks-oq3-v5.json");
    let v: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let tasks = v["tasks"].as_array().expect("tasks array");
    assert!(!tasks.is_empty());
    for (k, t) in tasks.iter().enumerate() {
        let goal = t["goal"].as_str().unwrap_or("");
        let ex = analyze(goal);
        println!(
            "OQ3V5 {k} {:?} | uncertain {:?}",
            ex.requirements, ex.uncertain
        );
    }
}
