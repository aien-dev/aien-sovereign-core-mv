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

use crate::models::{ChatMessage, DistillTask, DistillationRecord, Rollout};
use crate::providers::call_provider_unary;
use crate::router::AdapterRouter;
use crate::verifier::Verifier;
use chrono::Utc;
use reqwest::Client;
use serde_json::json;
use std::fs::{create_dir_all, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

pub struct DistillationEngine {
    pub client: Client,
    pub router: AdapterRouter,
    pub dataset_dir: PathBuf,
    pub cortex_endpoint: String,
}

impl Default for DistillationEngine {
    fn default() -> Self {
        Self::new()
    }
}

impl DistillationEngine {
    pub const CORTEX_QUALIFICATION_THRESHOLD: f32 = 0.85;

    pub fn new() -> Self {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("."));
        let default_dir = home.join("workspace/distillation-data");
        let _ = create_dir_all(&default_dir);

        let client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()
            .unwrap_or_default();

        Self {
            client,
            router: AdapterRouter::new(),
            dataset_dir: default_dir,
            cortex_endpoint: "http://127.0.0.1:18080".to_string(),
        }
    }

    pub fn with_client(mut self, client: Client) -> Self {
        self.client = client;
        self
    }

    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.client = Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(timeout)
            .build()
            .unwrap_or_default();
        self
    }

    pub fn with_dataset_dir<P: AsRef<Path>>(mut self, dir: P) -> Self {
        self.dataset_dir = dir.as_ref().to_path_buf();
        let _ = create_dir_all(&self.dataset_dir);
        self
    }

    pub fn qualifies_for_cortex(commit_to_cortex: bool, passed: bool, score: f32) -> bool {
        commit_to_cortex && passed && score >= Self::CORTEX_QUALIFICATION_THRESHOLD
    }

    pub fn build_cortex_payload(
        name: &str,
        canonical_name: &str,
        content: &str,
        task_id: &str,
        confidence: f64,
    ) -> serde_json::Value {
        serde_json::json!({
            "kind": "entity",
            "value": {
                "space": "atlas-memory",
                "canonicalName": canonical_name,
                "entityType": "learned_procedure",
                "content": content,
                "aliases": [name],
                "confidence": confidence,
                "metadata": {
                    "source": "spark-distill",
                    "taskId": task_id
                }
            }
        })
    }

    pub fn parse_cortex_receipt(body: &serde_json::Value, fallback: &str) -> String {
        body.pointer("/receipt/targetId")
            .or_else(|| body.pointer("/receipt/id"))
            .or_else(|| body.pointer("/receipt/target_id"))
            .and_then(|v| v.as_str())
            .unwrap_or(fallback)
            .to_string()
    }

    pub fn filter_dpo_candidate<'a>(
        chosen: &'a str,
        rejected: Option<&'a str>,
        preference_delta: f32,
    ) -> Option<&'a str> {
        rejected.filter(|rej| preference_delta > 0.0 && *rej != chosen)
    }

    pub async fn distill(&self, mut task: DistillTask) -> Result<DistillationRecord, String> {
        task.prompt = crate::vault::sanitize_outbound_prompt(&task.prompt);
        if let Some(ref sys) = task.system_prompt {
            task.system_prompt = Some(crate::vault::sanitize_outbound_prompt(sys));
        }

        let mut messages = Vec::new();
        if let Some(ref sys) = task.system_prompt {
            messages.push(ChatMessage {
                role: "system".to_string(),
                content: crate::vault::sanitize_outbound_prompt(sys),
                reasoning: None,
            });
        }
        messages.push(ChatMessage {
            role: "user".to_string(),
            content: crate::vault::sanitize_outbound_prompt(&task.prompt),
            reasoning: None,
        });

        // 1. Query Teacher Oracle
        let (teacher_provider, teacher_endpoint, teacher_model_id, teacher_key) =
            self.router.resolve_route(&task.teacher_model);

        let t0 = Instant::now();
        let (teacher_content, teacher_reasoning, teacher_tokens) = call_provider_unary(
            &self.client,
            teacher_provider,
            &teacher_endpoint,
            &teacher_model_id,
            teacher_key.as_deref(),
            &messages,
            Some(0.7),
        )
        .await
        .map_err(|e| format!("Teacher query failed ({}): {}", task.teacher_model, e))?;
        let teacher_duration = t0.elapsed().as_millis() as u64;

        let teacher_verif = Verifier::verify(&teacher_content, task.verification_strategy);

        let teacher_rollout = Rollout {
            provider: teacher_provider,
            model_id: task.teacher_model.clone(),
            content: teacher_content.clone(),
            reasoning: teacher_reasoning.clone(),
            duration_ms: teacher_duration,
            token_count: teacher_tokens,
            verification: teacher_verif.clone(),
        };

        // 2. Query Student (Local Open-Weight Resident Model)
        let mut student_rollout = None;
        let student_target = task
            .student_model
            .clone()
            .unwrap_or_else(|| "atlas-lightning-omni".to_string());

        let (student_provider, student_endpoint, student_model_id, student_key) =
            self.router.resolve_route(&student_target);

        let s0 = Instant::now();
        match call_provider_unary(
            &self.client,
            student_provider,
            &student_endpoint,
            &student_model_id,
            student_key.as_deref(),
            &messages,
            Some(0.7),
        )
        .await
        {
            Ok((student_content, student_reasoning, student_tokens)) => {
                let student_duration = s0.elapsed().as_millis() as u64;
                let student_verif = Verifier::verify(&student_content, task.verification_strategy);

                student_rollout = Some(Rollout {
                    provider: student_provider,
                    model_id: student_target.clone(),
                    content: student_content,
                    reasoning: student_reasoning,
                    duration_ms: student_duration,
                    token_count: student_tokens,
                    verification: student_verif,
                });
            }
            Err(e) => {
                eprintln!(
                    "Warning: Student rollout failed ({}): {}",
                    student_target, e
                );
            }
        }

        // 3. Selection and Preference Logic
        #[derive(Copy, Clone, PartialEq)]
        enum ChosenSeat {
            Teacher,
            Student,
        }

        let mut chosen_seat = ChosenSeat::Teacher;
        let mut chosen = teacher_content.clone();
        let mut chosen_reasoning = teacher_reasoning.clone();
        let mut chosen_score = teacher_rollout.verification.score;
        let mut chosen_passed = teacher_rollout.verification.passed;
        let mut rejected = None;
        let mut rejected_reasoning = None;
        let mut preference_delta = 0.0;

        if let Some(ref s_roll) = student_rollout {
            let t_pass = teacher_rollout.verification.passed;
            let s_pass = s_roll.verification.passed;
            let t_score = teacher_rollout.verification.score;
            let s_score = s_roll.verification.score;

            if s_pass && (!t_pass || s_score > t_score) {
                // Student strictly outperformed teacher
                chosen_seat = ChosenSeat::Student;
                chosen = s_roll.content.clone();
                chosen_reasoning = s_roll.reasoning.clone();
                chosen_score = s_score;
                chosen_passed = s_pass;
                rejected = Some(teacher_content.clone());
                rejected_reasoning = teacher_reasoning.clone();
                preference_delta = s_score - t_score;
            } else if t_pass && (!s_pass || t_score > s_score) {
                // Teacher strictly outperformed student
                chosen_seat = ChosenSeat::Teacher;
                chosen = teacher_content.clone();
                chosen_reasoning = teacher_reasoning.clone();
                chosen_score = t_score;
                chosen_passed = t_pass;
                rejected = Some(s_roll.content.clone());
                rejected_reasoning = s_roll.reasoning.clone();
                preference_delta = t_score - s_score;
            } else {
                // Tie or both failed: default to teacher solution, record exact delta
                chosen_seat = ChosenSeat::Teacher;
                chosen = teacher_content.clone();
                chosen_reasoning = teacher_reasoning.clone();
                chosen_score = t_score;
                chosen_passed = t_pass;
                rejected = Some(s_roll.content.clone());
                rejected_reasoning = s_roll.reasoning.clone();
                preference_delta = (t_score - s_score).abs();
            }
        }

        // 4. Persist to SFT and DPO Datasets
        let valid_dpo_rejected =
            Self::filter_dpo_candidate(&chosen, rejected.as_deref(), preference_delta);

        self.persist_training_pair(
            &task,
            &chosen,
            chosen_reasoning.as_deref(),
            valid_dpo_rejected,
        )?;

        // 5. Commit Durable Knowledge to Spark Cortex if Verified
        let mut durable_committed = false;
        let mut cortex_id = None;

        if Self::qualifies_for_cortex(task.commit_to_cortex, chosen_passed, chosen_score) {
            let entity_name = format!("distill_lesson:{}_{}", task.id, Utc::now().timestamp());
            let canonical = format!("distill_{}", task.id.replace('-', "_"));
            let source_label = match chosen_seat {
                ChosenSeat::Teacher => format!("Teacher Oracle ({})", task.teacher_model),
                ChosenSeat::Student => format!("Student Model ({})", student_target),
            };
            let content_summary = format!(
                "Distilled from {}. Task: {}. Reasoning: {}. Verified Solution: {}",
                source_label,
                task.prompt,
                chosen_reasoning.as_deref().unwrap_or("<direct>"),
                chosen
            );

            match self
                .commit_to_cortex(
                    &entity_name,
                    &canonical,
                    &content_summary,
                    &task.id,
                    chosen_score as f64,
                )
                .await
            {
                Ok(id) => {
                    durable_committed = true;
                    cortex_id = Some(id);
                }
                Err(e) => {
                    eprintln!("Warning: Cortex commit failed: {}", e);
                }
            }
        }

        Ok(DistillationRecord {
            task_id: task.id,
            task_type: task.task_type,
            prompt: task.prompt,
            system_prompt: task.system_prompt,
            teacher_rollout,
            student_rollout,
            chosen,
            chosen_reasoning,
            rejected,
            rejected_reasoning,
            preference_delta,
            durable_memory_committed: durable_committed,
            cortex_entity_id: cortex_id,
            created_at: Utc::now().to_rfc3339(),
        })
    }

    pub fn persist_training_pair(
        &self,
        task: &DistillTask,
        chosen: &str,
        chosen_reasoning: Option<&str>,
        rejected: Option<&str>,
    ) -> Result<bool, String> {
        let _ = create_dir_all(&self.dataset_dir);

        // Append SFT JSONL (ChatML format)
        let sft_file = self.dataset_dir.join("sft.jsonl");
        let mut sft_entry = json!({
            "task_id": task.id,
            "timestamp": Utc::now().to_rfc3339(),
            "messages": [
                { "role": "user", "content": task.prompt }
            ]
        });

        if let Some(ref sys) = task.system_prompt {
            if let Some(arr) = sft_entry.get_mut("messages").and_then(|m| m.as_array_mut()) {
                arr.insert(0, json!({ "role": "system", "content": sys }));
            }
        }

        let mut assistant_obj = json!({
            "role": "assistant",
            "content": chosen
        });
        if let Some(r) = chosen_reasoning {
            assistant_obj["reasoning"] = json!(r);
        }
        if let Some(arr) = sft_entry.get_mut("messages").and_then(|m| m.as_array_mut()) {
            arr.push(assistant_obj);
        }

        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&sft_file)
            .map_err(|e| format!("Failed to open sft.jsonl: {}", e))?;
        writeln!(file, "{}", serde_json::to_string(&sft_entry).unwrap())
            .map_err(|e| format!("Failed to write to sft.jsonl: {}", e))?;

        // Append DPO JSONL if rejected pair exists
        let mut dpo_written = false;
        if let Some(rej) = rejected {
            let dpo_file = self.dataset_dir.join("dpo.jsonl");
            let dpo_entry = json!({
                "task_id": task.id,
                "prompt": task.prompt,
                "system": task.system_prompt,
                "chosen": chosen,
                "rejected": rej,
                "timestamp": Utc::now().to_rfc3339()
            });

            let mut d_file = OpenOptions::new()
                .create(true)
                .append(true)
                .open(&dpo_file)
                .map_err(|e| format!("Failed to open dpo.jsonl: {}", e))?;
            writeln!(d_file, "{}", serde_json::to_string(&dpo_entry).unwrap())
                .map_err(|e| format!("Failed to write to dpo.jsonl: {}", e))?;
            dpo_written = true;
        }

        Ok(dpo_written)
    }

    pub async fn commit_to_cortex(
        &self,
        name: &str,
        canonical_name: &str,
        content: &str,
        task_id: &str,
        confidence: f64,
    ) -> Result<String, String> {
        let token = match crate::vault::resolve_secret("CORTEX_TOKEN") {
            Some(tok) if !tok.is_empty() => tok,
            _ => {
                return Err(
                    "CORTEX_TOKEN not found: configure CORTEX_TOKEN in environment or register in atlas-vault"
                        .to_string(),
                );
            }
        };

        let endpoint =
            std::env::var("CORTEX_ENDPOINT").unwrap_or_else(|_| self.cortex_endpoint.clone());
        let url = format!("{}/api/cortex/write", endpoint);

        let payload =
            Self::build_cortex_payload(name, canonical_name, content, task_id, confidence);

        let res = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", token))
            .json(&payload)
            .send()
            .await
            .map_err(|e| format!("Cortex request error: {}", e))?;

        if !res.status().is_success() {
            let status = res.status();
            let body = res.text().await.unwrap_or_default();
            return Err(format!("Cortex returned status {}: {}", status, body));
        }

        let body: serde_json::Value = res
            .json()
            .await
            .map_err(|e| format!("Failed to parse Cortex response JSON: {}", e))?;

        let entity_id = Self::parse_cortex_receipt(&body, canonical_name);

        Ok(entity_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_commit_to_cortex_requires_token() {
        let engine = DistillationEngine::new();
        let res = engine
            .commit_to_cortex("test", "test_canonical", "content", "task-1", 0.9)
            .await;
        if let Err(e) = res {
            assert!(
                e.contains("CORTEX_TOKEN not found")
                    || e.contains("Cortex request error")
                    || e.contains("Cortex returned status")
            );
        }
    }
}
