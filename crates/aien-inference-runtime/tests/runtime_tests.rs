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

use aien_inference_runtime::EmbeddedInferenceService;
use aien_inference_service::{InferenceEvent, InferenceRequest, InferenceService};

#[tokio::test]
async fn test_embedded_inference_service_reference_weights() {
    let svc = EmbeddedInferenceService::with_reference_weights().expect("load reference weights");
    assert_eq!(svc.model_id(), "TinyLlama/TinyLlama-1.1B-Chat-v1.0");
    assert_eq!(svc.endpoint(), "embedded://sovereign-core");

    let req = InferenceRequest::new("Testing protocol").with_max_tokens(8);
    let output = svc.generate(&req).await.expect("generate");
    assert!(!output.is_empty());

    let mut rx = svc.stream_events(&req).await.expect("stream events");
    let mut received_tokens = Vec::new();
    let mut completed = false;

    while let Some(evt) = rx.recv().await {
        match evt {
            InferenceEvent::Token(tok) => received_tokens.push(tok),
            InferenceEvent::Completed { total_tokens, .. } => {
                completed = true;
                assert_eq!(total_tokens, received_tokens.len());
            }
            InferenceEvent::Error(err) => panic!("Unexpected error: {err}"),
            _ => {}
        }
    }

    assert!(completed);
    assert!(!received_tokens.is_empty());
}
