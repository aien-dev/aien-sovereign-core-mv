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

use crate::models::{DistillTask, TaskType, VerificationStrategy};
use std::path::Path;
use std::process::Command;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurriculumTrack {
    NativeSystemsAndGpu,
    AgentAgency,
    ScientificReasoning,
    GeneralFrontier,
    DynamicMining,
}

impl CurriculumTrack {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "systems" | "native" | "gpu" | "cuda" => Some(CurriculumTrack::NativeSystemsAndGpu),
            "agent" | "agency" | "tool" => Some(CurriculumTrack::AgentAgency),
            "science" | "scientific" | "chemistry" => Some(CurriculumTrack::ScientificReasoning),
            "frontier" | "general" | "algo" => Some(CurriculumTrack::GeneralFrontier),
            "dynamic" | "mining" | "git" | "git_mining" | "git-mining" | "refactor" => {
                Some(CurriculumTrack::DynamicMining)
            }
            _ => None,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            CurriculumTrack::NativeSystemsAndGpu => "Native Systems & GPU Engineering",
            CurriculumTrack::AgentAgency => "Autonomous Agent Agency & Tool Contracts",
            CurriculumTrack::ScientificReasoning => "Advanced Scientific & Technical Reasoning",
            CurriculumTrack::GeneralFrontier => "General Frontier Capabilities & Systems Design",
            CurriculumTrack::DynamicMining => "Dynamic Repository & Trace Mining",
        }
    }
}

pub struct CurriculumEngine;

impl CurriculumEngine {
    /// Generate a batch of distillation tasks for the selected curriculum track.
    pub fn generate_tasks(
        track: CurriculumTrack,
        limit: usize,
        teacher: &str,
        student: Option<&str>,
        commit_to_cortex: bool,
    ) -> Vec<DistillTask> {
        let (task_type, strategy, prompts) = match track {
            CurriculumTrack::NativeSystemsAndGpu => (
                TaskType::CodeSynthesis,
                VerificationStrategy::CompilerCheck,
                vec![
                    "Implement an allocation-free single-producer single-consumer ring buffer in Rust with acquire-release memory ordering.",
                    "Write a Rust FFI interface and layout descriptor for Blackwell Paged Attention key-value blocks.",
                    "Implement a zero-copy Ethernet and IPv4 packet header parser in pure safe Rust.",
                    "Write a thread-safe memory object pool with RAII reuse guard in Rust.",
                    "Implement a high-throughput bounded MPMC channel using atomic sequence indexes in Rust.",
                ],
            ),
            CurriculumTrack::AgentAgency => (
                TaskType::VerificationHarness,
                VerificationStrategy::CompilerCheck,
                vec![
                    "Design an autonomous agent state machine in Rust with states for Idle, Planning, ToolExecution, Verifying, and ErrorRecovery.",
                    "Implement a robust retry mechanism with exponential backoff, jitter, and classification of transient versus permanent errors.",
                    "Write a cryptographic execution receipt validator that verifies SHA-256 digests and provenance metadata.",
                    "Implement an audit logger for agent tool invocations enforcing least privilege and parameter masking.",
                    "Write a deterministic verification harness for testing multi-agent handoff receipts.",
                ],
            ),
            CurriculumTrack::ScientificReasoning => (
                TaskType::ReasoningTrace,
                VerificationStrategy::UnslopStrict,
                vec![
                    "Formulate the reaction kinetics and differential rate equations for a multi-step photocatalytic water splitting process on titanium dioxide.",
                    "Design a chemical formulation model for a moisture-curing room-temperature vulcanizing silicone sealant, detailing polymer backbone, crosslinker, and catalyst.",
                    "Derive the mathematical proof for the convergence of gradient descent on strongly convex functions with Lipschitz continuous gradients.",
                    "Model the diffusion dynamics of graphene nanoparticles in an aqueous dispersion using the Stokes-Einstein equation.",
                    "Formulate the thermodynamics of phase separation in polymer blends using Flory-Huggins solution theory.",
                ],
            ),
            CurriculumTrack::GeneralFrontier => (
                TaskType::CodeSynthesis,
                VerificationStrategy::CompilerCheck,
                vec![
                    "Implement a concurrent lock-free skip list in Rust supporting concurrent search, insert, and delete.",
                    "Write an implementation of the Raft leader election algorithm and heartbeat timer in async Rust.",
                    "Implement an LRU cache with O(1) get and put operations and thread-safe read-write locking.",
                    "Write a high-performance regex pattern matcher using non-deterministic finite automata (NFA) simulation in Rust.",
                    "Implement a B-tree indexing node with splitting and binary search in Rust.",
                ],
            ),
            CurriculumTrack::DynamicMining => {
                let dynamic_prompts = Self::mine_recent_commit_prompts(limit.max(5));
                return dynamic_prompts
                    .into_iter()
                    .take(limit)
                    .map(|prompt| DistillTask {
                        id: Uuid::new_v4().to_string(),
                        task_type: TaskType::RefactorLogic,
                        prompt,
                        system_prompt: Some("You are AIEN Sovereign Architect. Produce concise, verified, unslop Rust code.".to_string()),
                        teacher_model: teacher.to_string(),
                        student_model: student.map(String::from),
                        verification_strategy: VerificationStrategy::CompilerCheck,
                        commit_to_cortex,
                    })
                    .collect();
            }
        };

        prompts
            .into_iter()
            .take(limit)
            .map(|prompt| DistillTask {
                id: Uuid::new_v4().to_string(),
                task_type,
                prompt: prompt.to_string(),
                system_prompt: Some("You are a verified sovereign systems specialist. Output clear, compilable native code with zero unslop.".to_string()),
                teacher_model: teacher.to_string(),
                student_model: student.map(String::from),
                verification_strategy: strategy,
                commit_to_cortex,
            })
            .collect()
    }

    /// Extract real-world problem statements by inspecting recent git commits in the repo.
    fn mine_recent_commit_prompts(limit: usize) -> Vec<String> {
        let repo_path = std::env::current_dir().unwrap_or_else(|_| Path::new(".").to_path_buf());
        if let Ok(output) = Command::new("git")
            .args([
                "-C",
                repo_path.to_str().unwrap_or("."),
                "log",
                "-n",
                "10",
                "--oneline",
            ])
            .output()
        {
            if output.status.success() {
                let log = String::from_utf8_lossy(&output.stdout);
                let prompts: Vec<String> = log
                    .lines()
                    .filter(|l| !l.is_empty())
                    .take(limit)
                    .map(|l| {
                        let subject = l.split_whitespace().skip(1).collect::<Vec<_>>().join(" ");
                        format!("Analyze and implement the core engineering requirements for the following system change: '{}'. Produce verified, robust Rust code.", subject)
                    })
                    .collect();

                if !prompts.is_empty() {
                    return prompts;
                }
            }
        }

        vec![
            "Implement a robust asynchronous event bus with decoupled publishers and subscribers in Rust.".to_string(),
            "Design a thread-safe state store with transaction rollback capabilities in Rust.".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_track_parsing() {
        assert_eq!(
            CurriculumTrack::parse("systems"),
            Some(CurriculumTrack::NativeSystemsAndGpu)
        );
        assert_eq!(
            CurriculumTrack::parse("agent"),
            Some(CurriculumTrack::AgentAgency)
        );
        assert_eq!(
            CurriculumTrack::parse("science"),
            Some(CurriculumTrack::ScientificReasoning)
        );
        assert_eq!(
            CurriculumTrack::parse("frontier"),
            Some(CurriculumTrack::GeneralFrontier)
        );
        assert_eq!(
            CurriculumTrack::parse("dynamic"),
            Some(CurriculumTrack::DynamicMining)
        );
    }

    #[test]
    fn test_track_parsing_aliases() {
        assert_eq!(
            CurriculumTrack::parse("cuda"),
            Some(CurriculumTrack::NativeSystemsAndGpu)
        );
        assert_eq!(
            CurriculumTrack::parse("git"),
            Some(CurriculumTrack::DynamicMining)
        );
        assert_eq!(
            CurriculumTrack::parse("git_mining"),
            Some(CurriculumTrack::DynamicMining)
        );
        assert_eq!(
            CurriculumTrack::parse("git-mining"),
            Some(CurriculumTrack::DynamicMining)
        );
        assert_eq!(
            CurriculumTrack::parse("refactor"),
            Some(CurriculumTrack::DynamicMining)
        );
    }

    #[test]
    fn test_task_generation() {
        let tasks = CurriculumEngine::generate_tasks(
            CurriculumTrack::NativeSystemsAndGpu,
            2,
            "anthropic/claude-3-7-sonnet",
            Some("atlas-lightning-omni"),
            true,
        );

        assert_eq!(tasks.len(), 2);
        assert!(tasks[0].prompt.contains("Rust"));
        assert_eq!(tasks[0].teacher_model, "anthropic/claude-3-7-sonnet");
        assert_eq!(
            tasks[0].student_model.as_deref(),
            Some("atlas-lightning-omni")
        );
        assert!(tasks[0].commit_to_cortex);
    }
}
