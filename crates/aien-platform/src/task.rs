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

use crate::PlatformError;
use alloc::string::String;
use alloc::vec::Vec;

pub struct TaskRequest<'a> {
    pub capability_token: u64,
    pub command_tag: &'a str,
    pub payload: &'a [u8],
    pub timeout_ticks: Option<u64>,
}

pub struct TaskResult {
    pub exit_code: i32,
    pub output: Vec<u8>,
    pub error_log: Option<String>,
}

pub trait TaskExecutor {
    fn execute(&self, task: TaskRequest<'_>) -> Result<TaskResult, PlatformError>;
}
