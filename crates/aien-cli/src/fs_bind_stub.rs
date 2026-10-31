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

use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum BindError {
    Denied(String),
    NotFound(String),
    Exists(String),
    Failed(String),
}

impl BindError {
    pub fn message(&self) -> &str {
        match self {
            Self::Denied(m) | Self::NotFound(m) | Self::Exists(m) | Self::Failed(m) => m,
        }
    }
}

fn closed(op: &str) -> BindError {
    BindError::Denied(format!(
        "refusing {op}: filesystem binding is unavailable on this platform"
    ))
}

/// Same allow-list the Linux binder uses. Opening still fails closed.
pub fn permitted_workspaces() -> Vec<PathBuf> {
    let mut allowed = crate::platform::PlatformContext::detect().default_workspaces;
    allowed.push(PathBuf::from("/tmp"));
    allowed
}

#[derive(Debug)]
pub struct ListedEntry {
    pub name: String,
    pub is_dir: bool,
    pub size_bytes: u64,
}

#[derive(Debug)]
pub struct HeldFile;

impl HeldFile {
    pub fn authorized(&self) -> &Path {
        Path::new("")
    }

    pub fn display(&self) -> &str {
        ""
    }

    pub fn read_checkpoint(&self) -> Result<Option<Vec<u8>>, String> {
        Err(closed("read").message().to_string())
    }

    pub fn write_bytes(&self, _content: &[u8], _overwrite: bool) -> Result<bool, BindError> {
        Err(closed("write"))
    }

    pub fn overwrite_existing(&self, _content: &[u8]) -> Result<(), BindError> {
        Err(closed("replace"))
    }
}

pub fn hold_for_write(_path: &str) -> Result<HeldFile, BindError> {
    Err(closed("write"))
}

pub fn read_text(_path: &str) -> Result<(PathBuf, String), BindError> {
    Err(closed("read"))
}

pub fn list_dir(_path: &str) -> Result<(PathBuf, Vec<ListedEntry>), BindError> {
    Err(closed("list"))
}

pub fn make_dir(_path: &str) -> Result<PathBuf, BindError> {
    Err(closed("mkdir"))
}

pub fn grep_search(_path: &str, _query: &str, _limit: usize) -> Result<Vec<String>, BindError> {
    Err(closed("grep"))
}

pub fn find_names(_path: &str, _pattern: &str, _limit: usize) -> Result<Vec<String>, BindError> {
    Err(closed("find"))
}
