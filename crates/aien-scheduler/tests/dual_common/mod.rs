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

//! Deterministic `serve.admit_preempt` workloads and a canonical, serializable
//! trace of every decision they produce (DUAL-3a instrumentation, ADR 0031).
//!
//! Everything here uses only APIs that exist on `main` before the observer
//! landed, so `dual_decision_golden.rs` compiles and runs on the base commit
//! and the golden fixture can be produced there. The observer tests reuse
//! the same workloads and compare traces produced with and without a tap.
//!
//! Determinism: `MockInferenceBackend` emits fixed tokens, never sleeps
//! (`simulated_step_latency_us` only adds to a reported figure), and every
//! timing field is excluded from the trace.

#![allow(dead_code)]

use aien_abi_core::{
    AienInferenceBackend, DecodeOutput, ModelConfig, SamplingParams, ScheduledBatch,
    SequenceRequest, StepMetrics,
};
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::{create_shared_kv_manager, KvMetrics, SharedKvManager};
use aien_scheduler::{AienScheduler, SchedulerConfig, SequenceId};
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Backend wrapper that keeps a copy of every `ScheduledBatch` the scheduler
/// handed it. The batch is the production decision; the wrapper changes
/// nothing about how the mock answers.
pub struct BatchRecordingBackend {
    inner: MockInferenceBackend,
    pub batches: Vec<ScheduledBatch>,
}

impl BatchRecordingBackend {
    pub fn new() -> Self {
        Self {
            inner: MockInferenceBackend::new(0),
            batches: Vec::new(),
        }
    }
}

impl Default for BatchRecordingBackend {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl AienInferenceBackend for BatchRecordingBackend {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String> {
        self.inner.load_model(config).await
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        self.batches.push(batch.clone());
        self.inner.execute_step(batch).await
    }

    fn manages_kv_cache(&self) -> bool {
        self.inner.manages_kv_cache()
    }
}

/// One action of a scripted workload.
#[derive(Debug, Clone)]
pub enum Action {
    /// `submit_request` with an explicit `request_id` (slot in the high 32
    /// bits, generation 1) so the KV id is known to the script.
    Submit {
        slot: u32,
        prompt_len: usize,
        priority: u8,
        max_tokens: usize,
    },
    /// KV-level `fork_prefilled(parent, child)` for each child slot, then
    /// `submit_request` for the child with the same prompt as the parent:
    /// the scheduler must admit it straight into decode on its existing table.
    ForkPrefilled {
        parent_slot: u32,
        child_slots: Vec<u32>,
        prompt_len: usize,
        max_tokens: usize,
    },
    /// `scheduler.step` this many times. An `Err` is recorded and the loop
    /// goes on (the request stays queued, as production does).
    Step(usize),
    /// `scheduler.step` until waiting, preempted and running are all empty or
    /// `max` steps ran, whichever comes first.
    Drain { max: usize },
}

#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: &'static str,
    pub kv_blocks: usize,
    pub block_size: usize,
    pub config: SchedulerConfig,
    pub actions: Vec<Action>,
    /// When true the scenario is expected to end with nothing queued, nothing
    /// running and zero KV blocks held (the leak check applies).
    pub expect_drained: bool,
}

pub fn seq(slot: u32) -> SequenceId {
    SequenceId::new(slot, 1).expect("generation 1 is valid")
}

pub fn request(slot: u32, prompt_len: usize, priority: u8, max_tokens: usize) -> SequenceRequest {
    SequenceRequest {
        request_id: seq(slot).to_u64(),
        prompt_tokens: (0..prompt_len as u32).map(|t| 1000 + t).collect(),
        sampling_params: SamplingParams {
            temperature: 0.0,
            top_p: 1.0,
            max_tokens,
            stop_token_ids: vec![],
        },
        arrival_time_ns: 0,
        priority,
    }
}

/// Canonical form of a batch: `block_tables` sorted by sequence id so two
/// equal decisions serialize to equal bytes (`HashMap` order is not stable).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CanonBatch {
    pub step_id: u64,
    pub prefill: Vec<(u64, Vec<u32>, u8)>,
    pub decode: Vec<u64>,
    pub block_tables: BTreeMap<u64, Vec<usize>>,
}

impl CanonBatch {
    pub fn from(batch: &ScheduledBatch) -> Self {
        Self {
            step_id: batch.step_id,
            prefill: batch
                .prefill_requests
                .iter()
                .map(|r| (r.request_id, r.prompt_tokens.clone(), r.priority))
                .collect(),
            decode: batch.decode_requests.clone(),
            block_tables: batch
                .block_tables
                .iter()
                .map(|(k, v)| (*k, v.clone()))
                .collect(),
        }
    }
}

/// KV accounting as the manager reports it, read after each step.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KvState {
    pub total_blocks: usize,
    pub free_blocks: usize,
    pub allocated_blocks: usize,
    pub active_tables: usize,
    pub physical_pages: usize,
    pub logical_pages: usize,
    pub shared_pages: usize,
    pub private_pages: usize,
    pub cow_faults: usize,
    pub used_blocks: usize,
    /// `(block id, ref_count, is_shared)` for every block with ref_count > 0.
    pub active_blocks: Vec<(usize, usize, bool)>,
}

impl KvState {
    pub fn read(kv: &SharedKvManager) -> Self {
        let kv = kv.read();
        let m: KvMetrics = kv.metrics();
        Self {
            total_blocks: kv.total_block_count(),
            free_blocks: kv.available_blocks(),
            allocated_blocks: kv.allocated_block_count(),
            active_tables: kv.active_sequence_count(),
            physical_pages: m.physical_pages,
            logical_pages: m.logical_pages,
            shared_pages: m.shared_pages,
            private_pages: m.private_pages,
            cow_faults: m.cow_faults,
            used_blocks: m.used_blocks,
            active_blocks: kv.debug_active_blocks(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchedulerState {
    pub waiting: usize,
    pub preempted: usize,
    pub running: usize,
    pub arena_active: usize,
    pub total_steps: u64,
    pub admitted_requests: u64,
    pub finished_requests: u64,
    pub preempted_requests: u64,
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
    pub chunked_prefill_steps: u64,
}

impl SchedulerState {
    pub fn read(s: &AienScheduler) -> Self {
        let m = s.metrics();
        Self {
            waiting: s.waiting_count(),
            preempted: s.preempted_count(),
            running: s.running_count(),
            arena_active: s.arena().active_count(),
            total_steps: m.total_steps,
            admitted_requests: m.admitted_requests,
            finished_requests: m.finished_requests,
            preempted_requests: m.preempted_requests,
            total_prefill_tokens: m.total_prefill_tokens,
            total_decode_tokens: m.total_decode_tokens,
            chunked_prefill_steps: m.chunked_prefill_steps,
        }
    }
}

/// Everything observable about one `scheduler.step`, timing excluded.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StepTrace {
    pub index: usize,
    /// The batch the scheduler handed the backend (`None` when the step
    /// returned `Ok(None)` or `Err`).
    pub batch: Option<CanonBatch>,
    pub error: Option<String>,
    pub outputs: Vec<String>,
    pub prefill_tokens_processed: Option<usize>,
    pub decode_tokens_emitted: Option<usize>,
    pub active_kv_blocks: Option<usize>,
    pub kv: KvState,
    pub scheduler: SchedulerState,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ScenarioTrace {
    pub name: String,
    pub steps: Vec<StepTrace>,
    /// Number of `scheduler.step` calls, each of which is exactly one
    /// `build_scheduled_batch` decision.
    pub decisions: usize,
    pub final_kv: KvState,
    pub final_scheduler: SchedulerState,
}

/// Builds the scheduler and KV manager for a scenario. The caller may install
/// an observer on the scheduler before running.
pub fn build(scenario: &Scenario) -> (AienScheduler, SharedKvManager) {
    let kv = create_shared_kv_manager(scenario.kv_blocks, scenario.block_size);
    let scheduler = AienScheduler::new(scenario.config.clone(), kv.clone());
    (scheduler, kv)
}

/// Runs the scenario's actions on a prepared scheduler and returns the trace.
pub fn run(
    scenario: &Scenario,
    scheduler: &mut AienScheduler,
    kv: &SharedKvManager,
) -> ScenarioTrace {
    let rt = tokio::runtime::Builder::new_current_thread()
        .build()
        .expect("tokio current-thread runtime");
    let mut backend = BatchRecordingBackend::new();
    let mut steps: Vec<StepTrace> = Vec::new();

    let mut one_step = |scheduler: &mut AienScheduler, backend: &mut BatchRecordingBackend| {
        let before = backend.batches.len();
        let result = rt.block_on(scheduler.step(backend));
        let batch = if backend.batches.len() > before {
            Some(CanonBatch::from(&backend.batches[before]))
        } else {
            None
        };
        let (error, outputs, pm, dm, akb) = match &result {
            Ok(Some((outputs, metrics))) => (
                None,
                outputs.iter().map(|o| format!("{o:?}")).collect(),
                Some(metrics.prefill_tokens_processed),
                Some(metrics.decode_tokens_emitted),
                Some(metrics.active_kv_blocks),
            ),
            Ok(None) => (None, Vec::new(), None, None, None),
            Err(e) => (Some(e.clone()), Vec::new(), None, None, None),
        };
        steps.push(StepTrace {
            index: steps.len(),
            batch,
            error,
            outputs,
            prefill_tokens_processed: pm,
            decode_tokens_emitted: dm,
            active_kv_blocks: akb,
            kv: KvState::read(kv),
            scheduler: SchedulerState::read(scheduler),
        });
    };

    for action in &scenario.actions {
        match action {
            Action::Submit {
                slot,
                prompt_len,
                priority,
                max_tokens,
            } => {
                scheduler.submit_request(request(*slot, *prompt_len, *priority, *max_tokens));
            }
            Action::ForkPrefilled {
                parent_slot,
                child_slots,
                prompt_len,
                max_tokens,
            } => {
                for child in child_slots {
                    kv.write()
                        .fork_prefilled(seq(*parent_slot).to_u64(), seq(*child).to_u64())
                        .expect("fork_prefilled of a ready parent");
                    scheduler.submit_request(request(*child, *prompt_len, 1, *max_tokens));
                }
            }
            Action::Step(n) => {
                for _ in 0..*n {
                    one_step(scheduler, &mut backend);
                }
            }
            Action::Drain { max } => {
                let mut ran = 0;
                while ran < *max
                    && (scheduler.waiting_count() > 0
                        || scheduler.running_count() > 0
                        || scheduler.preempted_count() > 0)
                {
                    one_step(scheduler, &mut backend);
                    ran += 1;
                }
            }
        }
    }

    let decisions = steps.len();
    ScenarioTrace {
        name: scenario.name.to_string(),
        steps,
        decisions,
        final_kv: KvState::read(kv),
        final_scheduler: SchedulerState::read(scheduler),
    }
}

fn cfg(
    max_batch_size: usize,
    max_batch_tokens: usize,
    max_prefill_tokens: usize,
    prefill_chunk_size: usize,
    chunk_prefill: bool,
    watermark_blocks: usize,
) -> SchedulerConfig {
    SchedulerConfig {
        max_batch_size,
        max_batch_tokens,
        max_prefill_tokens,
        prefill_chunk_size,
        chunk_prefill,
        watermark_blocks,
    }
}

/// The pre-registered workload set. Names are stable: the golden fixture and
/// the receipts refer to them.
#[allow(clippy::vec_init_then_push)]
pub fn scenarios() -> Vec<Scenario> {
    let mut v = Vec::new();

    // Admission: four short requests fit the batch and the pool; they run to
    // their length limit and the pool returns to empty.
    v.push(Scenario {
        name: "admission_basic",
        kv_blocks: 64,
        block_size: 16,
        config: cfg(8, 1024, 512, 128, true, 2),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 20,
                priority: 1,
                max_tokens: 3,
            },
            Action::Submit {
                slot: 2,
                prompt_len: 7,
                priority: 2,
                max_tokens: 2,
            },
            Action::Submit {
                slot: 3,
                prompt_len: 33,
                priority: 0,
                max_tokens: 4,
            },
            Action::Submit {
                slot: 4,
                prompt_len: 16,
                priority: 3,
                max_tokens: 1,
            },
            Action::Drain { max: 32 },
        ],
        expect_drained: true,
    });

    // No admission: nothing queued, the step must return Ok(None) and touch
    // nothing.
    v.push(Scenario {
        name: "no_admission_empty_queues",
        kv_blocks: 16,
        block_size: 16,
        config: cfg(8, 1024, 512, 128, true, 2),
        actions: vec![Action::Step(3)],
        expect_drained: true,
    });

    // Chunked prefill: a 300-token prompt with a 64-token chunk and a
    // 100-token prefill budget per step spreads over several steps while a
    // second sequence decodes beside it.
    v.push(Scenario {
        name: "chunked_prefill_continuation",
        kv_blocks: 64,
        block_size: 16,
        config: cfg(8, 1024, 100, 64, true, 2),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 10,
                priority: 1,
                max_tokens: 6,
            },
            Action::Step(1),
            Action::Submit {
                slot: 2,
                prompt_len: 300,
                priority: 1,
                max_tokens: 2,
            },
            Action::Drain { max: 40 },
        ],
        expect_drained: true,
    });

    // Preemption: 6-block pool, watermark 3, two 32-token prompts (2 blocks
    // each). After admission 2 blocks are free (< 3): the lower-priority
    // sequence is preempted on the next step, then readmitted later.
    v.push(Scenario {
        name: "preemption_watermark",
        kv_blocks: 6,
        block_size: 16,
        config: cfg(4, 1024, 512, 128, false, 3),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 32,
                priority: 3,
                max_tokens: 4,
            },
            Action::Submit {
                slot: 2,
                prompt_len: 32,
                priority: 1,
                max_tokens: 4,
            },
            Action::Drain { max: 40 },
        ],
        expect_drained: true,
    });

    // Ties: same pool pressure, equal priorities. `sort_by_key` is stable, so
    // the victim is the first in `running_sequences` order.
    v.push(Scenario {
        name: "preemption_tie_equal_priority",
        kv_blocks: 6,
        block_size: 16,
        config: cfg(4, 1024, 512, 128, false, 3),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 32,
                priority: 1,
                max_tokens: 4,
            },
            Action::Submit {
                slot: 2,
                prompt_len: 32,
                priority: 1,
                max_tokens: 4,
            },
            Action::Drain { max: 40 },
        ],
        expect_drained: true,
    });

    // No preemption under sufficient resources: a large pool, the same two
    // requests, no victim is ever ranked.
    v.push(Scenario {
        name: "no_preemption_sufficient_pool",
        kv_blocks: 256,
        block_size: 16,
        config: cfg(4, 1024, 512, 128, false, 3),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 32,
                priority: 3,
                max_tokens: 4,
            },
            Action::Submit {
                slot: 2,
                prompt_len: 32,
                priority: 1,
                max_tokens: 4,
            },
            Action::Drain { max: 40 },
        ],
        expect_drained: true,
    });

    // Queue exhaustion: twelve requests against a batch of 4. The admission
    // loop stops on the batch-size limit every step until the queue empties.
    let mut actions: Vec<Action> = (1..=12)
        .map(|slot| Action::Submit {
            slot,
            prompt_len: 8 + slot as usize,
            priority: (slot % 4) as u8,
            max_tokens: 2,
        })
        .collect();
    actions.push(Action::Drain { max: 64 });
    v.push(Scenario {
        name: "queue_exhaustion_batch_limit",
        kv_blocks: 128,
        block_size: 16,
        config: cfg(4, 1024, 512, 128, true, 1),
        actions,
        expect_drained: true,
    });

    // Capacity boundary, one below: 8-block pool, a prompt needing 7 blocks.
    v.push(Scenario {
        name: "capacity_one_below",
        kv_blocks: 8,
        block_size: 16,
        config: cfg(4, 4096, 4096, 4096, true, 0),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 7 * 16,
                priority: 1,
                max_tokens: 1,
            },
            Action::Drain { max: 8 },
        ],
        expect_drained: true,
    });

    // Capacity boundary, exactly at: a prompt needing all 8 blocks.
    v.push(Scenario {
        name: "capacity_exactly_at",
        kv_blocks: 8,
        block_size: 16,
        config: cfg(4, 4096, 4096, 4096, true, 0),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 8 * 16,
                priority: 1,
                max_tokens: 1,
            },
            Action::Drain { max: 8 },
        ],
        expect_drained: true,
    });

    // Capacity boundary, one above: 9 blocks needed from an 8-block pool with
    // nothing running. The step returns the KV_POOL_EXHAUSTED error and the
    // request stays queued; nothing is allocated.
    v.push(Scenario {
        name: "capacity_one_above_refused",
        kv_blocks: 8,
        block_size: 16,
        config: cfg(4, 4096, 4096, 4096, true, 0),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 9 * 16,
                priority: 1,
                max_tokens: 1,
            },
            Action::Step(2),
        ],
        expect_drained: false,
    });

    // Shortfall while something runs: the second request waits (no error)
    // until the first finishes and frees its blocks.
    v.push(Scenario {
        name: "capacity_wait_while_running",
        kv_blocks: 8,
        block_size: 16,
        config: cfg(4, 4096, 4096, 4096, true, 0),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 5 * 16,
                priority: 1,
                max_tokens: 2,
            },
            Action::Step(1),
            Action::Submit {
                slot: 2,
                prompt_len: 4 * 16,
                priority: 1,
                max_tokens: 1,
            },
            Action::Drain { max: 16 },
        ],
        expect_drained: true,
    });

    v.push(fanout(32));
    v.push(fanout(500));
    v
}

/// Branch fan-out: a 64-token parent is prefilled (2 steps), then `n`
/// children share its blocks via `fork_prefilled` and are admitted straight
/// into decode on their existing tables, each reserving its copy-on-write
/// block. The parent and every child run to their length limit.
pub fn fanout(n: u32) -> Scenario {
    let name: &'static str = match n {
        32 => "fanout_32_way",
        500 => "fanout_500_way",
        _ => "fanout_custom",
    };
    Scenario {
        name,
        kv_blocks: 4096,
        block_size: 16,
        config: cfg(1024, 16384, 4096, 512, true, 8),
        actions: vec![
            Action::Submit {
                slot: 1,
                prompt_len: 64,
                priority: 1,
                max_tokens: 2,
            },
            Action::Step(1),
            Action::ForkPrefilled {
                parent_slot: 1,
                child_slots: (2..2 + n).collect(),
                prompt_len: 64,
                max_tokens: 3,
            },
            Action::Drain { max: 64 },
        ],
        expect_drained: true,
    }
}

pub fn scenario(name: &str) -> Scenario {
    scenarios()
        .into_iter()
        .find(|s| s.name == name)
        .unwrap_or_else(|| panic!("no scenario named {name}"))
}

/// Canonical JSON (serde_json preserves struct field order; maps are
/// `BTreeMap`), the bytes the parity tests compare.
pub fn canonical_json<T: Serialize>(value: &T) -> Vec<u8> {
    serde_json::to_vec(value).expect("trace serializes")
}
