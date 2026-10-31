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

//! DUAL observation: a read-only macroscopic snapshot of the serve path
//! (ARCH-0031 DUAL-3a instrumentation).
//!
//! [`ServeObservationSnapshot::capture`] copies existing exact counters from
//! the scheduler, the KV manager and (when the caller has them) the last
//! `StepMetrics` and `ScheduledBatch` into one immutable struct, and derives
//! normalized densities from them. [`ObservationDelta::between`] subtracts two
//! snapshots to give arrival, completion and preemption counts over the
//! interval.
//!
//! Rules this crate keeps:
//!
//! - Every field documents the counter it is copied from. There is no second
//!   ledger: nothing here is incremented, remembered between calls or
//!   estimated. A snapshot is a pure function of the counters at the instant
//!   `capture` ran.
//! - Every quotient is a [`Density`] with its exact numerator and denominator;
//!   `Density::value` is `None` when the denominator is zero. No field is
//!   interpolated or guessed.
//! - This crate reads through `&AienScheduler` and `&AienKvManager` only. It
//!   cannot change a scheduling, admission, preemption or KV decision, and no
//!   field of a snapshot is an authority input anywhere (ADR 0031 §6).
//!
//! The scheduler's internal step counter is private; the generation here is
//! `SchedulerMetrics::total_steps`, which counts completed steps that executed
//! a batch. Two snapshots with equal `generation` were taken without a step
//! completing between them.

use aien_abi_core::{ScheduledBatch, StepMetrics};
use aien_kv_cache::{AienKvManager, KvMetrics};
use aien_scheduler::AienScheduler;
use serde::{Deserialize, Serialize};

/// An exact ratio kept as its two integers. `value()` is the normalized
/// density `numerator / denominator`, or `None` when the denominator is zero
/// (the zero-denominator rule for every density in this crate).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Density {
    pub numerator: usize,
    pub denominator: usize,
}

impl Density {
    pub fn new(numerator: usize, denominator: usize) -> Self {
        Self {
            numerator,
            denominator,
        }
    }

    /// `numerator / denominator`, `None` when `denominator == 0`.
    pub fn value(&self) -> Option<f64> {
        if self.denominator == 0 {
            None
        } else {
            Some(self.numerator as f64 / self.denominator as f64)
        }
    }
}

/// Copy of the last `StepMetrics` the caller received from
/// `AienScheduler::step`. `step_latency_us` is left out on purpose: it is a
/// timing, not a count, and DUAL observation records counts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StepReading {
    /// `StepMetrics::prefill_tokens_processed`.
    pub prefill_tokens_processed: usize,
    /// `StepMetrics::decode_tokens_emitted`.
    pub decode_tokens_emitted: usize,
    /// `StepMetrics::active_kv_blocks` (`AienKvManager::allocated_block_count`
    /// after the step, as the scheduler fills it).
    pub active_kv_blocks: usize,
}

impl From<&StepMetrics> for StepReading {
    fn from(m: &StepMetrics) -> Self {
        Self {
            prefill_tokens_processed: m.prefill_tokens_processed,
            decode_tokens_emitted: m.decode_tokens_emitted,
            active_kv_blocks: m.active_kv_blocks,
        }
    }
}

/// Sizes of the last `ScheduledBatch` the caller saw.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchReading {
    /// `ScheduledBatch::step_id`.
    pub step_id: u64,
    /// `ScheduledBatch::prefill_requests.len()`.
    pub prefill_requests: usize,
    /// `ScheduledBatch::decode_requests.len()`.
    pub decode_requests: usize,
    /// Sum of `prompt_tokens.len()` over `prefill_requests` plus one per
    /// decode request: the quantity the scheduler compares against
    /// `max_batch_tokens`.
    pub batch_tokens: usize,
}

impl From<&ScheduledBatch> for BatchReading {
    fn from(b: &ScheduledBatch) -> Self {
        Self {
            step_id: b.step_id,
            prefill_requests: b.prefill_requests.len(),
            decode_requests: b.decode_requests.len(),
            batch_tokens: b
                .prefill_requests
                .iter()
                .map(|r| r.prompt_tokens.len())
                .sum::<usize>()
                + b.decode_requests.len(),
        }
    }
}

/// One read-only macroscopic snapshot of the serve path.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ServeObservationSnapshot {
    /// `SchedulerMetrics::total_steps`: completed steps that executed a batch.
    pub generation: u64,

    // Scheduler queues and arena.
    /// `AienScheduler::waiting_count()`.
    pub waiting: usize,
    /// `AienScheduler::preempted_count()`.
    pub preempted: usize,
    /// `AienScheduler::running_count()`.
    pub running: usize,
    /// `SequenceArena::active_count()` via `AienScheduler::arena()`.
    pub arena_active: usize,

    // Scheduler lifetime counters (`SchedulerMetrics`).
    /// `SchedulerMetrics::admitted_requests` (counts `submit_request` and
    /// `submit_work`; a `fork_sequence` child is not counted here).
    pub admitted_requests: u64,
    /// `SchedulerMetrics::finished_requests`.
    pub finished_requests: u64,
    /// `SchedulerMetrics::preempted_requests` (lifetime count of preemption
    /// events, not the current queue length).
    pub preempted_requests: u64,
    /// `SchedulerMetrics::total_prefill_tokens`.
    pub total_prefill_tokens: u64,
    /// `SchedulerMetrics::total_decode_tokens`.
    pub total_decode_tokens: u64,
    /// `SchedulerMetrics::chunked_prefill_steps`.
    pub chunked_prefill_steps: u64,

    // Scheduler limits (`SchedulerConfig`).
    /// `SchedulerConfig::max_batch_size`.
    pub max_batch_size: usize,
    /// `SchedulerConfig::max_batch_tokens`.
    pub max_batch_tokens: usize,
    /// `SchedulerConfig::watermark_blocks`.
    pub watermark_blocks: usize,

    // KV pool (`AienKvManager`).
    /// `AienKvManager::total_block_count()`.
    pub kv_total_blocks: usize,
    /// `AienKvManager::available_blocks()` (free-list length).
    pub kv_free_blocks: usize,
    /// `AienKvManager::allocated_block_count()` (`total - free`).
    pub kv_allocated_blocks: usize,
    /// `AienKvManager::active_sequence_count()` (block tables held).
    pub kv_active_tables: usize,
    /// `KvMetrics::physical_pages` (blocks with `ref_count > 0`).
    pub kv_physical_pages: usize,
    /// `KvMetrics::logical_pages` (sum of table lengths).
    pub kv_logical_pages: usize,
    /// `KvMetrics::shared_pages` (blocks with `ref_count > 1`).
    pub kv_shared_pages: usize,
    /// `KvMetrics::private_pages` (blocks with `ref_count == 1`).
    pub kv_private_pages: usize,
    /// `KvMetrics::cow_faults`.
    pub kv_cow_faults: usize,
    /// `KvMetrics::used_blocks` (equals `physical_pages` by its definition).
    pub kv_used_blocks: usize,
    /// `AienKvManager::available_blocks() < SchedulerConfig::watermark_blocks`,
    /// the pool half of the scheduler's preemption predicate (the other half,
    /// "something is running", is `running > 0`).
    pub kv_below_watermark: bool,

    // Last step and batch, when the caller had them.
    pub last_step: Option<StepReading>,
    pub last_batch: Option<BatchReading>,

    // Normalized densities (used / capacity). Denominator zero gives None.
    /// `kv_allocated_blocks / kv_total_blocks`.
    pub kv_block_density: Density,
    /// `(kv_logical_pages - kv_physical_pages) / kv_logical_pages`: the share
    /// of logical pages served by sharing. `None` when no table holds a page.
    pub kv_sharing_density: Density,
    /// `running / max_batch_size`.
    pub running_slot_density: Density,
    /// `(prefill_requests + decode_requests) / max_batch_size` of the last
    /// batch; numerator 0 when no batch was given.
    pub batch_slot_density: Density,
    /// `batch_tokens / max_batch_tokens` of the last batch; numerator 0 when
    /// no batch was given.
    pub batch_token_density: Density,
}

impl ServeObservationSnapshot {
    /// Copies the counters. `kv` must be the manager the scheduler was built
    /// with; the caller holds the read guard so both reads see one instant.
    pub fn capture(
        scheduler: &AienScheduler,
        kv: &AienKvManager,
        last_step: Option<&StepMetrics>,
        last_batch: Option<&ScheduledBatch>,
    ) -> Self {
        let m = scheduler.metrics();
        let cfg = scheduler.config();
        let km: KvMetrics = kv.metrics();

        let kv_total_blocks = kv.total_block_count();
        let kv_free_blocks = kv.available_blocks();
        let kv_allocated_blocks = kv.allocated_block_count();
        let running = scheduler.running_count();
        let last_step = last_step.map(StepReading::from);
        let last_batch = last_batch.map(BatchReading::from);
        let (batch_slots, batch_tokens) = match &last_batch {
            Some(b) => (b.prefill_requests + b.decode_requests, b.batch_tokens),
            None => (0, 0),
        };

        Self {
            generation: m.total_steps,
            waiting: scheduler.waiting_count(),
            preempted: scheduler.preempted_count(),
            running,
            arena_active: scheduler.arena().active_count(),
            admitted_requests: m.admitted_requests,
            finished_requests: m.finished_requests,
            preempted_requests: m.preempted_requests,
            total_prefill_tokens: m.total_prefill_tokens,
            total_decode_tokens: m.total_decode_tokens,
            chunked_prefill_steps: m.chunked_prefill_steps,
            max_batch_size: cfg.max_batch_size,
            max_batch_tokens: cfg.max_batch_tokens,
            watermark_blocks: cfg.watermark_blocks,
            kv_total_blocks,
            kv_free_blocks,
            kv_allocated_blocks,
            kv_active_tables: kv.active_sequence_count(),
            kv_physical_pages: km.physical_pages,
            kv_logical_pages: km.logical_pages,
            kv_shared_pages: km.shared_pages,
            kv_private_pages: km.private_pages,
            kv_cow_faults: km.cow_faults,
            kv_used_blocks: km.used_blocks,
            kv_below_watermark: kv_free_blocks < cfg.watermark_blocks,
            last_step,
            last_batch,
            kv_block_density: Density::new(kv_allocated_blocks, kv_total_blocks),
            kv_sharing_density: Density::new(
                km.logical_pages.saturating_sub(km.physical_pages),
                km.logical_pages,
            ),
            running_slot_density: Density::new(running, cfg.max_batch_size),
            batch_slot_density: Density::new(batch_slots, cfg.max_batch_size),
            batch_token_density: Density::new(batch_tokens, cfg.max_batch_tokens),
        }
    }
}

/// Why two snapshots could not be subtracted.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeltaError {
    /// `later.generation < earlier.generation`: the order is reversed or the
    /// snapshots come from different schedulers.
    GenerationWentBackwards { earlier: u64, later: u64 },
    /// A lifetime counter decreased, which the scheduler never does; the
    /// snapshots are not from one scheduler.
    CounterWentBackwards {
        counter: &'static str,
        earlier: u64,
        later: u64,
    },
    /// The pool size differs; the snapshots are not from one KV manager.
    PoolSizeChanged { earlier: usize, later: usize },
}

/// Exact differences between two snapshots of the same scheduler.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationDelta {
    /// `later.generation - earlier.generation`: completed steps in between.
    pub steps: u64,
    /// `admitted_requests` difference: requests that arrived (were submitted).
    pub arrivals: u64,
    /// `finished_requests` difference.
    pub completions: u64,
    /// `preempted_requests` difference: preemption events.
    pub preemptions: u64,
    /// `total_prefill_tokens` difference.
    pub prefill_tokens: u64,
    /// `total_decode_tokens` difference.
    pub decode_tokens: u64,
    /// `later.kv_allocated_blocks - earlier.kv_allocated_blocks`, signed.
    pub kv_allocated_change: i64,
    /// `later.running - earlier.running`, signed.
    pub running_change: i64,
    /// `later.waiting - earlier.waiting`, signed.
    pub waiting_change: i64,
}

impl ObservationDelta {
    /// `later - earlier`. Refuses reversed order and snapshots that cannot
    /// come from one scheduler and one pool.
    pub fn between(
        earlier: &ServeObservationSnapshot,
        later: &ServeObservationSnapshot,
    ) -> Result<Self, DeltaError> {
        if later.generation < earlier.generation {
            return Err(DeltaError::GenerationWentBackwards {
                earlier: earlier.generation,
                later: later.generation,
            });
        }
        if later.kv_total_blocks != earlier.kv_total_blocks {
            return Err(DeltaError::PoolSizeChanged {
                earlier: earlier.kv_total_blocks,
                later: later.kv_total_blocks,
            });
        }
        let diff = |counter: &'static str, a: u64, b: u64| -> Result<u64, DeltaError> {
            b.checked_sub(a).ok_or(DeltaError::CounterWentBackwards {
                counter,
                earlier: a,
                later: b,
            })
        };
        Ok(Self {
            steps: later.generation - earlier.generation,
            arrivals: diff(
                "admitted_requests",
                earlier.admitted_requests,
                later.admitted_requests,
            )?,
            completions: diff(
                "finished_requests",
                earlier.finished_requests,
                later.finished_requests,
            )?,
            preemptions: diff(
                "preempted_requests",
                earlier.preempted_requests,
                later.preempted_requests,
            )?,
            prefill_tokens: diff(
                "total_prefill_tokens",
                earlier.total_prefill_tokens,
                later.total_prefill_tokens,
            )?,
            decode_tokens: diff(
                "total_decode_tokens",
                earlier.total_decode_tokens,
                later.total_decode_tokens,
            )?,
            kv_allocated_change: later.kv_allocated_blocks as i64
                - earlier.kv_allocated_blocks as i64,
            running_change: later.running as i64 - earlier.running as i64,
            waiting_change: later.waiting as i64 - earlier.waiting as i64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn density_zero_denominator_is_none() {
        assert_eq!(Density::new(0, 0).value(), None);
        assert_eq!(Density::new(5, 0).value(), None);
        assert_eq!(Density::new(0, 4).value(), Some(0.0));
        assert_eq!(Density::new(2, 4).value(), Some(0.5));
        assert_eq!(Density::new(4, 4).value(), Some(1.0));
    }
}
