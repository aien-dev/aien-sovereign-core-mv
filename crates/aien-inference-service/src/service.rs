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

use super::events::InferenceEvent;
use super::request::InferenceRequest;
use async_trait::async_trait;
use tokio::sync::mpsc;

#[async_trait]
pub trait InferenceService: Send + Sync {
    fn model_id(&self) -> String;
    fn endpoint(&self) -> String;
    async fn generate(&self, request: &InferenceRequest) -> Result<String, String>;
    async fn stream_events(
        &self,
        request: &InferenceRequest,
    ) -> Result<mpsc::Receiver<InferenceEvent>, String>;
}
