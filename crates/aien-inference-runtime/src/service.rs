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

use super::model::EmbeddedModel;
use aien_inference_service::{InferenceEvent, InferenceRequest, InferenceService};
use async_trait::async_trait;
use parking_lot::Mutex;
use std::sync::Arc;
use tokio::sync::mpsc;

#[derive(Clone)]
pub struct EmbeddedInferenceService {
    model: Arc<Mutex<EmbeddedModel>>,
    model_id: String,
}

impl EmbeddedInferenceService {
    pub fn new(model: EmbeddedModel) -> Self {
        let model_id = model.config.model_id.clone();
        Self {
            model: Arc::new(Mutex::new(model)),
            model_id,
        }
    }

    pub fn with_reference_weights() -> Result<Self, String> {
        let config = aien_inference_abi::ModelConfig::tinyllama_1_1b();
        let model = EmbeddedModel::with_reference_weights(&config)?;
        Ok(Self::new(model))
    }

    pub fn load_default_or_fallback() -> Result<Self, String> {
        let model = EmbeddedModel::load_default_or_fallback()?;
        Ok(Self::new(model))
    }
}

#[async_trait]
impl InferenceService for EmbeddedInferenceService {
    fn model_id(&self) -> String {
        self.model_id.clone()
    }

    fn endpoint(&self) -> String {
        "embedded://sovereign-core".to_string()
    }

    async fn generate(&self, request: &InferenceRequest) -> Result<String, String> {
        let model = Arc::clone(&self.model);
        let prompt = if let Some(ref sys) = request.system_prompt {
            format!(
                "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                sys.trim(),
                request.prompt.trim()
            )
        } else {
            format!("<|user|>\n{}</s>\n<|assistant|>\n", request.prompt.trim())
        };
        let max_tokens = request.limits.max_tokens;
        let temp = request.limits.temperature;

        tokio::task::spawn_blocking(move || {
            let mut guard = model.lock();
            guard.generate(&prompt, max_tokens, temp)
        })
        .await
        .map_err(|e| format!("TaskJoin error: {}", e))?
    }

    async fn stream_events(
        &self,
        request: &InferenceRequest,
    ) -> Result<mpsc::Receiver<InferenceEvent>, String> {
        let model = Arc::clone(&self.model);
        let prompt = if let Some(ref sys) = request.system_prompt {
            format!(
                "<|system|>\n{}</s>\n<|user|>\n{}</s>\n<|assistant|>\n",
                sys.trim(),
                request.prompt.trim()
            )
        } else {
            format!("<|user|>\n{}</s>\n<|assistant|>\n", request.prompt.trim())
        };
        let max_tokens = request.limits.max_tokens;
        let temp = request.limits.temperature;

        let (tx, rx) = mpsc::channel(64);

        tokio::task::spawn_blocking(move || {
            let mut token_count = 0;
            let res = {
                let mut guard = model.lock();
                guard.generate_stream(&prompt, max_tokens, temp, |piece| {
                    token_count += 1;
                    tx.blocking_send(InferenceEvent::Token(piece.to_string()))
                        .is_ok()
                })
            };

            match res {
                Ok(()) => {
                    let _ = tx.blocking_send(InferenceEvent::Completed {
                        total_tokens: token_count,
                        prompt_tokens: 0,
                        completion_tokens: token_count,
                    });
                }
                Err(e) => {
                    let _ = tx.blocking_send(InferenceEvent::Error(e));
                }
            }
        });

        Ok(rx)
    }
}
