pub mod sequence;

pub use sequence::*;

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, FinishReason, ScheduledBatch, SequenceRequest, StepMetrics,
};
use aien_kv_cache::AienKvManager;
use aien_platform::{InferenceWork, Priority};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SchedulerConfig {
    pub max_batch_size: usize,
    pub max_batch_tokens: usize,
    pub max_prefill_tokens: usize,
    pub prefill_chunk_size: usize,
    pub chunk_prefill: bool,
    pub watermark_blocks: usize,
}

impl Default for SchedulerConfig {
    fn default() -> Self {
        Self {
            max_batch_size: 64,
            max_batch_tokens: 4096,
            max_prefill_tokens: 2048,
            prefill_chunk_size: 512,
            chunk_prefill: true,
            watermark_blocks: 4,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct SchedulerMetrics {
    pub total_steps: u64,
    pub admitted_requests: u64,
    pub finished_requests: u64,
    pub preempted_requests: u64,
    pub total_prefill_tokens: u64,
    pub total_decode_tokens: u64,
    pub avg_step_latency_us: f64,
    pub chunked_prefill_steps: u64,
}

/// Native continuous batching LLM scheduler managing SequenceArena,
/// decoupled CompletionRouter, and zero-token-clone queues.
pub struct AienScheduler {
    config: SchedulerConfig,
    kv_manager: Arc<RwLock<AienKvManager>>,
    arena: SequenceArena,
    completion_router: CompletionRouter,
    waiting_queue: VecDeque<SequenceId>,
    preempted_queue: VecDeque<SequenceId>,
    running_sequences: Vec<SequenceId>,
    step_id: u64,
    metrics: SchedulerMetrics,
}

impl AienScheduler {
    pub fn new(config: SchedulerConfig, kv_manager: Arc<RwLock<AienKvManager>>) -> Self {
        Self {
            config,
            kv_manager,
            arena: SequenceArena::new(),
            completion_router: CompletionRouter::new(),
            waiting_queue: VecDeque::new(),
            preempted_queue: VecDeque::new(),
            running_sequences: Vec::new(),
            step_id: 0,
            metrics: SchedulerMetrics::default(),
        }
    }

    pub fn config(&self) -> &SchedulerConfig {
        &self.config
    }

    pub fn metrics(&self) -> &SchedulerMetrics {
        &self.metrics
    }

    pub fn arena(&self) -> &SequenceArena {
        &self.arena
    }

    pub fn arena_mut(&mut self) -> &mut SequenceArena {
        &mut self.arena
    }

    pub fn completion_router(&self) -> &CompletionRouter {
        &self.completion_router
    }

    pub fn completion_router_mut(&mut self) -> &mut CompletionRouter {
        &mut self.completion_router
    }

    pub fn waiting_count(&self) -> usize {
        self.waiting_queue.len()
    }

    pub fn preempted_count(&self) -> usize {
        self.preempted_queue.len()
    }

    pub fn running_count(&self) -> usize {
        self.running_sequences.len()
    }

    /// Submits a SequenceRequest, creating an immutable PromptHandle and allocating a SequenceId.
    pub fn submit_request(&mut self, request: SequenceRequest) -> SequenceId {
        let prompt_slice: Box<[u32]> = request.prompt_tokens.into_boxed_slice();
        let prompt_handle: PromptHandle = Arc::from(prompt_slice);
        let priority = match request.priority {
            0 => Priority::Background,
            1 => Priority::Normal,
            2 => Priority::Interactive,
            _ => Priority::Realtime,
        };

        let seq_id = self
            .arena
            .insert_with(
                prompt_handle,
                aien_platform::ModelHandle(0),
                aien_platform::KvHandle(0),
                priority,
                None,
                None,
                request.sampling_params.max_tokens as u32,
                request.sampling_params,
                None,
            )
            .expect("Arena allocation failed");

        self.waiting_queue.push_back(seq_id);
        self.metrics.admitted_requests += 1;
        seq_id
    }

    /// Submits structured platform InferenceWork without cloning prompt tokens.
    pub fn submit_work(
        &mut self,
        work: InferenceWork,
        prompt: PromptHandle,
        sampling_params: Option<aien_inference_abi::SamplingParams>,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        let sampling = sampling_params.unwrap_or_default();
        let seq_id = self.arena.insert_with(
            prompt,
            work.model,
            work.kv,
            work.priority,
            work.deadline,
            work.branch_parent
                .and_then(|p| SequenceId::from_u64(p).ok()),
            work.next_token_budget,
            sampling,
            sink_id,
        )?;

        self.waiting_queue.push_back(seq_id);
        self.metrics.admitted_requests += 1;
        Ok(seq_id)
    }

    /// Zero-copy sequence fork preserving immutable PromptHandle.
    pub fn fork_sequence(
        &mut self,
        parent_id: SequenceId,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        if self.arena.is_stale(parent_id) {
            return Err(format!(
                "Parent sequence {} is stale or inactive",
                parent_id
            ));
        }

        let child_id = self.arena.fork(parent_id, sink_id)?;

        // Fork block table in KV manager
        {
            let mut kv = self.kv_manager.write();
            kv.fork_sequence(parent_id.to_u64(), child_id.to_u64())?;
        }

        self.running_sequences.push(child_id);
        Ok(child_id)
    }

    /// Legacy backward compatibility for fork_subagent(parent_id, child_id).
    pub fn fork_subagent(&mut self, parent_raw: u64, child_raw: u64) -> Result<(), String> {
        let parent_id = self
            .running_sequences
            .iter()
            .copied()
            .find(|id| id.to_u64() == parent_raw || id.slot as u64 == parent_raw)
            .or_else(|| SequenceId::from_u64(parent_raw).ok())
            .ok_or_else(|| format!("Parent sequence {} not found in running pool", parent_raw))?;

        if self.arena.is_stale(parent_id) {
            return Err(format!("Parent sequence {} is stale", parent_id));
        }

        let child_id = self.arena.fork(parent_id, None)?;

        {
            let mut kv = self.kv_manager.write();
            kv.fork_sequence(parent_id.to_u64(), child_id.to_u64())
                .or_else(|_| kv.fork_sequence(parent_raw, child_raw))?;
        }

        self.running_sequences.push(child_id);
        Ok(())
    }

    /// Discards stale items from queue heads and internal lists without panics.
    fn purge_stale_work(&mut self) {
        while let Some(&id) = self.waiting_queue.front() {
            if self.arena.is_stale(id) {
                self.waiting_queue.pop_front();
            } else {
                break;
            }
        }

        while let Some(&id) = self.preempted_queue.front() {
            if self.arena.is_stale(id) {
                self.preempted_queue.pop_front();
            } else {
                break;
            }
        }

        let arena = &self.arena;
        self.running_sequences.retain(|&id| !arena.is_stale(id));
    }

    /// Builds the next scheduled batch enforcing chunked prefill budgets and watermark preemption.
    pub fn build_scheduled_batch(&mut self) -> Result<Option<ScheduledBatch>, String> {
        self.step_id += 1;
        self.purge_stale_work();

        let mut prefill_requests = Vec::new();
        let mut decode_requests = Vec::new();
        let mut block_tables = HashMap::new();

        let mut current_tokens = 0;
        let mut prefill_budget = self.config.max_prefill_tokens;
        let mut preempted_this_step = false;

        // 1. Watermark Memory Pressure Check: Preempt lowest priority sequence if below watermark
        {
            let kv = self.kv_manager.read();
            let available = kv.available_blocks();
            if available < self.config.watermark_blocks && !self.running_sequences.is_empty() {
                let mut candidates: Vec<(SequenceId, Priority)> = self
                    .running_sequences
                    .iter()
                    .filter_map(|&id| self.arena.get(id).map(|rec| (id, rec.priority)))
                    .collect();
                candidates.sort_by_key(|c| c.1 as u8);

                if let Some((preempt_id, _)) = candidates.first() {
                    let preempt_id = *preempt_id;
                    drop(kv);
                    if let Some(pos) = self.running_sequences.iter().position(|&x| x == preempt_id)
                    {
                        self.running_sequences.remove(pos);
                        let _ = self.kv_manager.write().free_sequence(preempt_id.to_u64());
                        if let Some(rec) = self.arena.get_mut(preempt_id) {
                            rec.phase = SequencePhase::Preempted;
                            rec.prompt_tokens_prefilled = 0;
                            rec.is_prefilled = false;
                        }
                        self.preempted_queue.push_back(preempt_id);
                        self.metrics.preempted_requests += 1;
                        preempted_this_step = true;
                    }
                }
            }
        }

        // 2. Schedule active decode sequences and in-flight prefill chunks
        let mut running_ids = self.running_sequences.clone();
        running_ids.sort();

        for seq_id in running_ids {
            if decode_requests.len() + prefill_requests.len() >= self.config.max_batch_size {
                break;
            }
            if current_tokens >= self.config.max_batch_tokens {
                break;
            }

            let seq = match self.arena.get_mut(seq_id) {
                Some(s) => s,
                None => continue,
            };

            if seq.is_prefilled {
                // Active decode step
                if let Some(table) = self.kv_manager.read().get_block_table(seq_id.to_u64()) {
                    block_tables.insert(seq_id.to_u64(), table.block_ids.clone());
                    decode_requests.push(seq_id.to_u64());
                    current_tokens += 1;
                }
            } else {
                // Continuing chunked prefill
                let total_prompt_len = seq.prompt.len();
                let remaining = total_prompt_len.saturating_sub(seq.prompt_tokens_prefilled);
                let chunk_size = std::cmp::min(remaining, self.config.prefill_chunk_size);
                let chunk_size = std::cmp::min(chunk_size, prefill_budget);

                if chunk_size > 0 {
                    let start = seq.prompt_tokens_prefilled;
                    let end = start + chunk_size;
                    let chunk_tokens = seq.prompt[start..end].to_vec();

                    let chunk_req = SequenceRequest {
                        request_id: seq_id.to_u64(),
                        prompt_tokens: chunk_tokens,
                        sampling_params: seq.sampling_params.clone(),
                        arrival_time_ns: seq.arrival_time_ns,
                        priority: seq.priority as u8,
                    };

                    if let Some(table) = self.kv_manager.read().get_block_table(seq_id.to_u64()) {
                        block_tables.insert(seq_id.to_u64(), table.block_ids.clone());
                    }

                    prefill_requests.push(chunk_req);
                    current_tokens += chunk_size;
                    prefill_budget = prefill_budget.saturating_sub(chunk_size);
                }
            }
        }

        // 3. Admit waiting or preempted requests if capacity permits and not under preemption pressure
        if !preempted_this_step {
            while decode_requests.len() + prefill_requests.len() < self.config.max_batch_size
                && current_tokens < self.config.max_batch_tokens
                && prefill_budget > 0
            {
                let (next_seq_id, is_preempted) = if let Some(&id) = self.preempted_queue.front() {
                    (id, true)
                } else if let Some(&id) = self.waiting_queue.front() {
                    (id, false)
                } else {
                    break;
                };

                if self.arena.is_stale(next_seq_id) {
                    if is_preempted {
                        self.preempted_queue.pop_front();
                    } else {
                        self.waiting_queue.pop_front();
                    }
                    continue;
                }

                let prompt_len = match self.arena.get(next_seq_id) {
                    Some(s) => s.prompt.len(),
                    None => {
                        if is_preempted {
                            self.preempted_queue.pop_front();
                        } else {
                            self.waiting_queue.pop_front();
                        }
                        continue;
                    }
                };

                let required_blocks = (prompt_len + 15) / 16;
                let available = self.kv_manager.read().available_blocks();

                // Preempted sequences require headroom above watermark to prevent thrashing
                let min_needed = if is_preempted {
                    required_blocks + self.config.watermark_blocks
                } else {
                    required_blocks
                };

                if available < min_needed {
                    break;
                }

                if is_preempted {
                    self.preempted_queue.pop_front();
                } else {
                    self.waiting_queue.pop_front();
                }

                let seq = match self.arena.get_mut(next_seq_id) {
                    Some(s) => s,
                    None => continue,
                };

                let chunk_size = if self.config.chunk_prefill {
                    std::cmp::min(prompt_len, self.config.prefill_chunk_size)
                } else {
                    prompt_len
                };
                let chunk_size = std::cmp::min(chunk_size, prefill_budget);

                if chunk_size == 0 {
                    if is_preempted {
                        self.preempted_queue.push_front(next_seq_id);
                    } else {
                        self.waiting_queue.push_front(next_seq_id);
                    }
                    break;
                }

                // Allocate blocks for admitted sequence
                let assigned_blocks = self
                    .kv_manager
                    .write()
                    .allocate_sequence(next_seq_id.to_u64(), &seq.prompt)?;

                block_tables.insert(next_seq_id.to_u64(), assigned_blocks);

                let chunk_tokens = seq.prompt[0..chunk_size].to_vec();
                let chunk_req = SequenceRequest {
                    request_id: next_seq_id.to_u64(),
                    prompt_tokens: chunk_tokens,
                    sampling_params: seq.sampling_params.clone(),
                    arrival_time_ns: seq.arrival_time_ns,
                    priority: seq.priority as u8,
                };

                prefill_requests.push(chunk_req);
                current_tokens += chunk_size;
                prefill_budget = prefill_budget.saturating_sub(chunk_size);

                seq.phase = SequencePhase::Prefill;
                self.running_sequences.push(next_seq_id);
            }
        }

        if prefill_requests.is_empty() && decode_requests.is_empty() {
            return Ok(None);
        }

        Ok(Some(ScheduledBatch {
            prefill_requests,
            decode_requests,
            block_tables,
            step_id: self.step_id,
        }))
    }

    /// Builds a structured zero-copy BatchPlan for native batch engines.
    pub fn build_batch_plan(&mut self) -> Result<Option<BatchPlan>, String> {
        let batch = match self.build_scheduled_batch()? {
            Some(b) => b,
            None => return Ok(None),
        };

        let mut prefill_spans = Vec::new();
        let mut decode_items = Vec::new();
        let mut total_tokens = 0;

        for req in &batch.prefill_requests {
            if let Ok(seq_id) = SequenceId::from_u64(req.request_id) {
                if let Some(record) = self.arena.get(seq_id) {
                    let length = req.prompt_tokens.len();
                    prefill_spans.push(PrefillSpan {
                        seq_id,
                        start_pos: record.prompt_tokens_prefilled,
                        length,
                    });
                    total_tokens += length;
                }
            }
        }

        for &req_id in &batch.decode_requests {
            if let Ok(seq_id) = SequenceId::from_u64(req_id) {
                if let Some(record) = self.arena.get(seq_id) {
                    decode_items.push(DecodeItem {
                        seq_id,
                        token_pos: record.total_tokens(),
                    });
                    total_tokens += 1;
                }
            }
        }

        Ok(Some(BatchPlan {
            step_id: batch.step_id,
            prefill_spans,
            decode_items,
            total_tokens,
        }))
    }

    /// Advances the engine by one step using the configured backend.
    pub async fn step(
        &mut self,
        backend: &mut (dyn AienInferenceBackend + '_),
    ) -> Result<Option<(Vec<DecodeOutput>, StepMetrics)>, String> {
        let batch = match self.build_scheduled_batch()? {
            Some(b) => b,
            None => return Ok(None),
        };

        let t0 = Instant::now();
        let (outputs, metrics) = backend.execute_step(&batch).await?;
        let step_latency = t0.elapsed().as_micros() as u64;

        // Update prefill progress for chunked sequences
        for prefill_req in &batch.prefill_requests {
            if let Ok(seq_id) = SequenceId::from_u64(prefill_req.request_id) {
                if let Some(seq) = self.arena.get_mut(seq_id) {
                    if !seq.is_prefilled {
                        seq.prompt_tokens_prefilled += prefill_req.prompt_tokens.len();
                        if seq.prompt_tokens_prefilled >= seq.prompt.len() {
                            seq.is_prefilled = true;
                            seq.phase = SequencePhase::Decode;
                        }
                    }
                }
            }
        }

        let mut final_outputs = Vec::new();

        for output in outputs {
            match output {
                DecodeOutput::Token {
                    request_id,
                    token_id,
                    logprob,
                } => {
                    let mut is_finished = false;
                    let mut finish_reason = FinishReason::StopToken;
                    let mut total_tokens = 0;
                    let mut should_emit_token = false;

                    if let Ok(seq_id) = SequenceId::from_u64(request_id) {
                        if let Some(seq) = self.arena.get_mut(seq_id) {
                            if seq.is_prefilled {
                                should_emit_token = true;
                                seq.tokens_generated += 1;
                                seq.generated_tokens.push(token_id);
                                total_tokens = seq.tokens_generated;

                                if seq.sampling_params.stop_token_ids.contains(&token_id) {
                                    is_finished = true;
                                    finish_reason = FinishReason::StopToken;
                                } else if seq.tokens_generated >= seq.sampling_params.max_tokens {
                                    is_finished = true;
                                    finish_reason = FinishReason::LengthLimit;
                                } else {
                                    let append_result =
                                        self.kv_manager.write().append_token(request_id);
                                    if append_result.is_err() {
                                        is_finished = true;
                                        finish_reason = FinishReason::Preempted;
                                    }
                                }

                                if let Some(sink_id) = seq.sink_id {
                                    self.completion_router.emit(
                                        sink_id,
                                        CompletionEvent::Token {
                                            seq_id,
                                            token: token_id,
                                        },
                                    );
                                }
                            }
                        }

                        if is_finished {
                            if let Some(pos) =
                                self.running_sequences.iter().position(|&x| x == seq_id)
                            {
                                self.running_sequences.remove(pos);
                            }
                            if finish_reason == FinishReason::Preempted {
                                let _ = self.kv_manager.write().free_sequence(request_id);
                                if let Some(seq) = self.arena.get_mut(seq_id) {
                                    seq.phase = SequencePhase::Preempted;
                                    seq.prompt_tokens_prefilled = 0;
                                    seq.is_prefilled = false;
                                }
                                self.preempted_queue.push_back(seq_id);
                                self.metrics.preempted_requests += 1;
                            } else {
                                let _ = self.kv_manager.write().free_sequence(request_id);
                                self.arena.free_sequence(seq_id);
                                self.metrics.finished_requests += 1;
                            }

                            final_outputs.push(DecodeOutput::Finished {
                                request_id,
                                reason: finish_reason,
                                total_tokens,
                            });
                        } else if should_emit_token {
                            final_outputs.push(DecodeOutput::Token {
                                request_id,
                                token_id,
                                logprob,
                            });
                        }
                    }
                }
                DecodeOutput::Finished {
                    request_id,
                    reason,
                    total_tokens,
                } => {
                    if let Ok(seq_id) = SequenceId::from_u64(request_id) {
                        if let Some(pos) = self.running_sequences.iter().position(|&x| x == seq_id)
                        {
                            self.running_sequences.remove(pos);
                        }
                        let _ = self.kv_manager.write().free_sequence(request_id);
                        self.arena.free_sequence(seq_id);
                        self.metrics.finished_requests += 1;
                    }
                    final_outputs.push(DecodeOutput::Finished {
                        request_id,
                        reason,
                        total_tokens,
                    });
                }
            }
        }

        self.metrics.total_steps += 1;
        self.metrics.total_prefill_tokens += metrics.prefill_tokens_processed as u64;
        self.metrics.total_decode_tokens += metrics.decode_tokens_emitted as u64;

        if metrics.prefill_tokens_processed > 0 && self.config.chunk_prefill {
            self.metrics.chunked_prefill_steps += 1;
        }

        let total_steps = self.metrics.total_steps as f64;
        self.metrics.avg_step_latency_us = (self.metrics.avg_step_latency_us * (total_steps - 1.0)
            + step_latency as f64)
            / total_steps;

        let active_kv_blocks = self.kv_manager.read().allocated_block_count();

        Ok(Some((
            final_outputs,
            StepMetrics {
                prefill_tokens_processed: metrics.prefill_tokens_processed,
                decode_tokens_emitted: metrics.decode_tokens_emitted,
                step_latency_us: step_latency,
                active_kv_blocks,
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aien_inference_abi::{MockInferenceBackend, SamplingParams};
    use aien_kv_cache::create_shared_kv_manager;

    #[tokio::test]
    async fn test_scheduler_lifecycle_and_stale_work_rejection() {
        let kv_manager = create_shared_kv_manager(64, 16);
        let config = SchedulerConfig::default();
        let mut scheduler = AienScheduler::new(config, kv_manager);
        let mut backend = MockInferenceBackend::new(10);

        let req = SequenceRequest {
            request_id: 1,
            prompt_tokens: vec![1, 2, 3, 4],
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 3,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        };

        let seq_id = scheduler.submit_request(req);
        assert_eq!(scheduler.waiting_count(), 1);

        // Step 1: Admits and prefills (emits token 1)
        let res1 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res1.0.len(), 1);
        assert_eq!(scheduler.running_count(), 1);
        assert_eq!(res1.1.prefill_tokens_processed, 4);

        // Manually simulate a stale work ticket in waiting queue:
        let fake_stale_id = SequenceId::new(seq_id.slot, seq_id.generation + 10).unwrap();
        scheduler.waiting_queue.push_back(fake_stale_id);

        // Step 2: Executes decode token 2 and silently discards fake_stale_id without error or panic
        let res2 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res2.0.len(), 1);
        assert_eq!(scheduler.waiting_count(), 0); // Stale work was discarded

        // Step 3: Hits max_tokens (3) and finishes
        let res3 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res3.0.len(), 1);
        match &res3.0[0] {
            DecodeOutput::Finished {
                request_id,
                reason,
                total_tokens,
            } => {
                assert_eq!(*request_id, seq_id.to_u64());
                assert_eq!(*reason, FinishReason::LengthLimit);
                assert_eq!(*total_tokens, 3);
            }
            _ => panic!("Expected Finished output"),
        }
        assert_eq!(scheduler.running_count(), 0);
        assert_eq!(scheduler.metrics().finished_requests, 1);
    }

    #[tokio::test]
    async fn test_chunked_prefill_segmentation() {
        let kv_manager = create_shared_kv_manager(100, 16);
        let config = SchedulerConfig {
            max_batch_size: 16,
            max_batch_tokens: 1024,
            max_prefill_tokens: 512,
            prefill_chunk_size: 64,
            chunk_prefill: true,
            watermark_blocks: 2,
        };
        let mut scheduler = AienScheduler::new(config, kv_manager);
        let mut backend = MockInferenceBackend::new(10);

        let req = SequenceRequest {
            request_id: 100,
            prompt_tokens: (0..128).collect(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 2,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        };

        scheduler.submit_request(req);

        // Step 1: Processes chunk 1 (64 tokens)
        let res1 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res1.1.prefill_tokens_processed, 64);
        assert_eq!(scheduler.running_count(), 1);

        // Step 2: Processes chunk 2 (64 tokens, completes prompt)
        let res2 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res2.1.prefill_tokens_processed, 64);

        // Step 3: Decode begins
        let res3 = scheduler.step(&mut backend).await.unwrap().unwrap();
        assert_eq!(res3.1.decode_tokens_emitted, 1);
    }

    #[tokio::test]
    async fn test_watermark_pressure_preemption() {
        let kv_manager = create_shared_kv_manager(6, 16);
        let config = SchedulerConfig {
            max_batch_size: 4,
            max_batch_tokens: 1024,
            max_prefill_tokens: 512,
            prefill_chunk_size: 128,
            chunk_prefill: false,
            watermark_blocks: 3,
        };
        let mut scheduler = AienScheduler::new(config, kv_manager.clone());
        let mut backend = MockInferenceBackend::new(10);

        scheduler.submit_request(SequenceRequest {
            request_id: 1,
            prompt_tokens: (0..32).collect(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 10,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 5,
        });

        scheduler.submit_request(SequenceRequest {
            request_id: 2,
            prompt_tokens: (0..32).collect(),
            sampling_params: SamplingParams {
                temperature: 0.0,
                top_p: 1.0,
                max_tokens: 10,
                stop_token_ids: vec![],
            },
            arrival_time_ns: 0,
            priority: 1,
        });

        // Step 1: Admits both
        scheduler.step(&mut backend).await.unwrap();

        // Step 2: Memory pressure triggers preemption of lower priority
        scheduler.step(&mut backend).await.unwrap();
        assert_eq!(scheduler.metrics().preempted_requests, 1);
        assert_eq!(scheduler.preempted_count(), 1);
        assert_eq!(scheduler.running_count(), 1);
    }
}
