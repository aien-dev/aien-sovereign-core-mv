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

use serde_json::{json, Value};
use std::process::Command;

fn get_hive_bin() -> std::path::PathBuf {
    let platform = crate::platform::PlatformContext::detect();
    let local = platform.home_dir.join("aien-hive/target/release/aien-hive");
    if local.exists() {
        local
    } else {
        std::path::PathBuf::from("aien-hive")
    }
}

pub fn hive_roster() -> String {
    let out = Command::new(get_hive_bin()).arg("roster").output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("Failed to run aien-hive roster: {}", e),
    }
}

pub fn hive_spawn(role: &str, task: &str) -> String {
    let out = Command::new(get_hive_bin())
        .args(["spawn", role, task])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("Failed to spawn cell: {}", e),
    }
}

pub fn hive_swarm(task: &str, n: usize) -> String {
    let out = Command::new(get_hive_bin())
        .args(["swarm", "--task", task, "--n", &n.to_string()])
        .output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("Failed to run swarm: {}", e),
    }
}

pub fn hive_read(name: &str) -> String {
    let out = Command::new(get_hive_bin()).args(["read", name]).output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("Failed to read cell {}: {}", name, e),
    }
}

pub fn hive_kill(name: &str) -> String {
    let out = Command::new(get_hive_bin()).args(["kill", name]).output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("Failed to kill cell {}: {}", name, e),
    }
}

pub fn hive_sense(text: &str) -> String {
    let out = Command::new(get_hive_bin()).args(["sense", text]).output();
    match out {
        Ok(o) => String::from_utf8_lossy(&o.stdout).to_string(),
        Err(e) => format!("Failed to sense: {}", e),
    }
}

pub fn hive_dispatch_tool(args: &Value) -> Value {
    let action = args
        .get("action")
        .and_then(Value::as_str)
        .unwrap_or("roster");
    match action {
        "roster" => json!({"roster": hive_roster()}),
        "spawn" => {
            let role = args
                .get("role")
                .and_then(Value::as_str)
                .unwrap_or("researcher");
            let task = args.get("task").and_then(Value::as_str).unwrap_or("");
            json!({"result": hive_spawn(role, task)})
        }
        "swarm" => {
            let task = args.get("task").and_then(Value::as_str).unwrap_or("");
            let n = args.get("n").and_then(Value::as_u64).unwrap_or(2) as usize;
            json!({"result": hive_swarm(task, n)})
        }
        "read" => {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("");
            json!({"output": hive_read(name)})
        }
        "kill" => {
            let name = args.get("name").and_then(Value::as_str).unwrap_or("");
            json!({"result": hive_kill(name)})
        }
        "sense" => {
            let text = args.get("text").and_then(Value::as_str).unwrap_or("");
            json!({"salience": hive_sense(text)})
        }
        other => json!({"error": format!("Unknown hive action: {}", other)}),
    }
}
