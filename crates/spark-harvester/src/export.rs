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

use crate::extractor::ExtractedPair;
use serde_json::json;
use std::fs::OpenOptions;
use std::io::{self, Write};
use std::path::Path;

pub struct DatasetExporter;

impl DatasetExporter {
    #[allow(dead_code)]
    pub(crate) fn append_to_jsonl(path: &Path, pairs: &[ExtractedPair]) -> io::Result<usize> {
        let mut file = OpenOptions::new().create(true).append(true).open(path)?;

        let mut written = 0;
        for pair in pairs {
            let record = json!({
                "instruction": pair.prompt,
                "reasoning": pair.reasoning,
                "response": pair.completion,
                "metadata": {
                    "source_provider": pair.provider,
                    "source_model": pair.model,
                    "tokens": pair.token_count,
                    "quality": pair.quality_score,
                    "license": "PolyForm-Noncommercial-1.0.0 Sovereign Defense Covenant"
                }
            });
            writeln!(file, "{}", record)?;
            written += 1;
        }
        Ok(written)
    }

    pub fn to_sharegpt_format(pair: &ExtractedPair) -> serde_json::Value {
        let human_text = pair.prompt.clone();
        let gpt_text = if let Some(r) = &pair.reasoning {
            format!("<think>\n{}\n</think>\n\n{}", r, pair.completion)
        } else {
            pair.completion.clone()
        };

        json!({
            "conversations": [
                {"from": "human", "value": human_text},
                {"from": "gpt", "value": gpt_text}
            ],
            "source": pair.provider
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharegpt_format() {
        let pair = ExtractedPair {
            prompt: "Say hi".to_string(),
            reasoning: Some("Greeting thought".to_string()),
            completion: "Hello".to_string(),
            provider: "gemini".to_string(),
            model: "gemini-2.5-pro".to_string(),
            token_count: 20,
            quality_score: 0.9,
        };
        let val = DatasetExporter::to_sharegpt_format(&pair);
        let convos = val["conversations"].as_array().unwrap();
        assert_eq!(convos.len(), 2);
        assert!(convos[1]["value"].as_str().unwrap().contains("<think>"));
    }
}
