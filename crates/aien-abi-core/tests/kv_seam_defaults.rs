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

//! C6a (sc#484, ADR 0003 sections 6.1 to 6.3): the backend trait gains two
//! hooks for persistent KV, `computed_prefix` and `restore_sequence`. A backend
//! that implements neither must behave as it does today: it reports no
//! computed prefix (so nothing is ever parked from it) and refuses a restore
//! (so a restored prefix can never be admitted on a backend that cannot honour
//! it).

use aien_abi_core::{AienInferenceBackend, DecodeOutput, ModelConfig, ScheduledBatch, StepMetrics};
use async_trait::async_trait;

struct Bare;

#[async_trait]
impl AienInferenceBackend for Bare {
    async fn load_model(&mut self, _config: &ModelConfig) -> Result<(), String> {
        Ok(())
    }

    async fn execute_step(
        &mut self,
        _batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        Ok((Vec::new(), StepMetrics::default()))
    }
}

#[test]
fn restore_sequence_defaults_to_err() {
    let mut backend = Bare;
    let refusal = backend
        .restore_sequence(0x1_0000_0001, &[1, 2, 3])
        .expect_err("a backend that does not override restore_sequence must refuse");
    assert!(
        refusal.contains("restore_sequence"),
        "the refusal names the hook so a log says what is missing: {refusal}"
    );
}

#[test]
fn computed_prefix_defaults_to_none() {
    let backend = Bare;
    assert_eq!(
        backend.computed_prefix(0x1_0000_0001),
        None,
        "no override means no claim about which positions hold K/V"
    );
}
