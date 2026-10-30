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

//! Filesystem binding for CLI tools.
//!
//! Linux opens the authorized object with `openat2`. Every other target
//! fails closed. A pathname allow-list is not confinement.

#[cfg(target_os = "linux")]
include!("fs_bind_linux.rs");

#[cfg(not(target_os = "linux"))]
include!("fs_bind_stub.rs");
