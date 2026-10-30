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

pub fn get_vault_secret(key: &str) -> Option<String> {
    if let Ok(val) = std::env::var(key) {
        if !val.trim().is_empty() {
            return Some(val.trim().to_string());
        }
    }

    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    let vault_candidate = format!("{}/.local/bin/atlas-vault", home);
    let bin = if std::path::Path::new(&vault_candidate).is_file() {
        &vault_candidate
    } else {
        "atlas-vault"
    };
    let output = Command::new(bin).arg("get").arg(key).output().ok()?;

    if output.status.success() {
        let val = String::from_utf8_lossy(&output.stdout).trim().to_string();
        if !val.is_empty() {
            return Some(val);
        }
    }
    None
}
