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

//! Persistent KV benchmark harness (cut C9, sc#485; ADR 0003 sections 10,
//! 15.1 and 16.5). `report` is the output contract (labels, required metrics,
//! NOT_RUN rows, no mixed cache states); `run` measures rows on the host CPU
//! reference backend through the real runtime spine.

pub mod report;
pub mod run;
