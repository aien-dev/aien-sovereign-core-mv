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

use std::path::PathBuf;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlatformKind {
    DgxSpark,
    Linux,
    MacOS,
    Windows,
}

#[derive(Debug, Clone)]
pub struct PlatformContext {
    pub kind: PlatformKind,
    pub hostname: String,
    pub current_user: String,
    pub home_dir: PathBuf,
    pub default_workspaces: Vec<PathBuf>,
}

impl PlatformContext {
    pub fn detect() -> Self {
        let hostname = std::env::var("HOSTNAME")
            .or_else(|_| std::fs::read_to_string("/etc/hostname").map(|s| s.trim().to_string()))
            .unwrap_or_else(|_| "localhost".to_string());

        let current_user = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_else(|_| "user".to_string());

        let home_dir = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/"));

        let is_spark = hostname.to_lowercase().contains("spark")
            || std::path::Path::new("/etc/spark-release").exists()
            || std::env::var("SPARK_HARDWARE")
                .map(|v| v == "1")
                .unwrap_or(false);

        let kind = if is_spark {
            PlatformKind::DgxSpark
        } else if cfg!(target_os = "macos") {
            PlatformKind::MacOS
        } else if cfg!(target_os = "windows") {
            PlatformKind::Windows
        } else {
            PlatformKind::Linux
        };

        let mut default_workspaces = Vec::new();

        if let Ok(env_ws) = std::env::var("AIEN_WORKSPACES") {
            for part in env_ws.split(':') {
                if !part.trim().is_empty() {
                    default_workspaces.push(PathBuf::from(part.trim()));
                }
            }
        }

        default_workspaces.push(home_dir.join("workspace"));
        default_workspaces.push(home_dir.join("atlas-prime-workspace"));
        default_workspaces.push(home_dir.join("basecamp"));
        default_workspaces.push(home_dir.join("spark-neural-os"));
        default_workspaces.push(home_dir.join("skills"));

        if let Ok(cwd) = std::env::current_dir() {
            if !default_workspaces.iter().any(|p| p == &cwd) {
                default_workspaces.push(cwd);
            }
        }

        Self {
            kind,
            hostname,
            current_user,
            home_dir,
            default_workspaces,
        }
    }

    pub fn is_dgx_spark(&self) -> bool {
        self.kind == PlatformKind::DgxSpark
    }
}
