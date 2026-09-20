use aien_inference_abi::{FinishReason, SamplingParams, SequenceRequest};
use aien_platform::{InferenceWork, KvHandle, ModelHandle, Priority, Ticks};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// 64-bit Sequence Identifier composed of a slot index and a generation counter.
/// Layout: [slot: u32 (high 32 bits)][generation: u32 (low 32 bits)]
/// Generation zero is strictly invalid ab initio.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct SequenceId {
    pub slot: u32,
    pub generation: u32,
}

impl SequenceId {
    pub const INVALID: Self = Self {
        slot: u32::MAX,
        generation: 0,
    };

    pub fn new(slot: u32, generation: u32) -> Result<Self, &'static str> {
        if generation == 0 {
            return Err("Generation zero is strictly invalid");
        }
        Ok(Self { slot, generation })
    }

    #[inline]
    pub fn is_valid(&self) -> bool {
        self.generation != 0
    }

    #[inline]
    pub fn to_u64(&self) -> u64 {
        ((self.slot as u64) << 32) | (self.generation as u64)
    }

    #[inline]
    pub fn from_u64(val: u64) -> Result<Self, &'static str> {
        let slot = (val >> 32) as u32;
        let generation = (val & 0xFFFF_FFFF) as u32;
        if generation == 0 {
            return Err("Generation zero is strictly invalid");
        }
        Ok(Self { slot, generation })
    }
}

impl From<SequenceId> for u64 {
    fn from(id: SequenceId) -> Self {
        id.to_u64()
    }
}

impl TryFrom<u64> for SequenceId {
    type Error = &'static str;
    fn try_from(val: u64) -> Result<Self, Self::Error> {
        SequenceId::from_u64(val)
    }
}

impl std::fmt::Display for SequenceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Seq(slot:{},gen:{})", self.slot, self.generation)
    }
}

/// Immutable, reference-counted prompt token buffer.
/// Guarantees zero allocation overhead across sequence forks and subagent branches.
pub type PromptHandle = Arc<[u32]>;

/// Opaque identifier for a decoupled completion sink.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CompletionSinkId(pub u64);

/// Structured lifecycle event emitted for a sequence during execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompletionEvent {
    Token {
        seq_id: SequenceId,
        token: u32,
    },
    Finished {
        seq_id: SequenceId,
        finish_reason: FinishReason,
        total_tokens: usize,
    },
    Error {
        seq_id: SequenceId,
        message: String,
    },
}

/// Decoupled event consumer trait free from sockets, threads, and processes.
pub trait CompletionSink: Send + Sync {
    fn emit(&self, event: CompletionEvent);
}

/// Channel-based completion sink implementation for async task integration.
pub struct ChannelCompletionSink {
    sender: tokio::sync::mpsc::UnboundedSender<CompletionEvent>,
}

impl ChannelCompletionSink {
    pub fn new(sender: tokio::sync::mpsc::UnboundedSender<CompletionEvent>) -> Self {
        Self { sender }
    }

    pub fn channel() -> (Self, tokio::sync::mpsc::UnboundedReceiver<CompletionEvent>) {
        let (tx, rx) = tokio::sync::mpsc::unbounded_channel();
        (Self::new(tx), rx)
    }
}

impl CompletionSink for ChannelCompletionSink {
    fn emit(&self, event: CompletionEvent) {
        let _ = self.sender.send(event);
    }
}

/// Event router managing active completion sinks.
#[derive(Default)]
pub struct CompletionRouter {
    sinks: HashMap<CompletionSinkId, Arc<dyn CompletionSink>>,
    next_id: u64,
}

impl CompletionRouter {
    pub fn new() -> Self {
        Self {
            sinks: HashMap::new(),
            next_id: 1,
        }
    }

    pub fn register(&mut self, sink: Arc<dyn CompletionSink>) -> CompletionSinkId {
        let id = CompletionSinkId(self.next_id);
        self.next_id += 1;
        self.sinks.insert(id, sink);
        id
    }

    pub fn unregister(&mut self, id: CompletionSinkId) -> Option<Arc<dyn CompletionSink>> {
        self.sinks.remove(&id)
    }

    pub fn emit(&self, sink_id: CompletionSinkId, event: CompletionEvent) {
        if let Some(sink) = self.sinks.get(&sink_id) {
            sink.emit(event);
        }
    }

    pub fn broadcast(&self, event: CompletionEvent) {
        for sink in self.sinks.values() {
            sink.emit(event.clone());
        }
    }
}

/// Span of prompt tokens scheduled for prefill execution.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PrefillSpan {
    pub seq_id: SequenceId,
    pub start_pos: usize,
    pub length: usize,
}

/// Single autoregressive decode step scheduled for an active sequence.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DecodeItem {
    pub seq_id: SequenceId,
    pub token_pos: usize,
}

/// Structured hardware execution plan dividing work into prefill spans and decode steps.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPlan {
    pub step_id: u64,
    pub prefill_spans: Vec<PrefillSpan>,
    pub decode_items: Vec<DecodeItem>,
    pub total_tokens: usize,
}

/// Explicit lifecycle phases for a sequence record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SequencePhase {
    Waiting,
    Prefill,
    PrefillInFlight,
    Decode,
    DecodeInFlight,
    Preempted,
    Finished,
    Cancelled,
}

/// Authoritative runtime state record for an individual sequence.
#[derive(Debug, Clone)]
pub struct SequenceRecord {
    pub seq_id: SequenceId,
    pub prompt: PromptHandle,
    pub model: ModelHandle,
    pub kv: KvHandle,
    pub priority: Priority,
    pub deadline: Option<Ticks>,
    pub branch_parent: Option<SequenceId>,
    pub next_token_budget: u32,
    pub phase: SequencePhase,
    pub tokens_generated: usize,
    pub prompt_tokens_prefilled: usize,
    pub is_prefilled: bool,
    pub sink_id: Option<CompletionSinkId>,
    pub sampling_params: SamplingParams,
    pub arrival_time_ns: u64,
    pub generated_tokens: Vec<u32>,
}

impl SequenceRecord {
    pub fn new(
        seq_id: SequenceId,
        prompt: PromptHandle,
        model: ModelHandle,
        kv: KvHandle,
        priority: Priority,
        deadline: Option<Ticks>,
        branch_parent: Option<SequenceId>,
        next_token_budget: u32,
        sampling_params: SamplingParams,
        sink_id: Option<CompletionSinkId>,
    ) -> Self {
        Self {
            seq_id,
            prompt,
            model,
            kv,
            priority,
            deadline,
            branch_parent,
            next_token_budget,
            phase: SequencePhase::Waiting,
            tokens_generated: 0,
            prompt_tokens_prefilled: 0,
            is_prefilled: false,
            sink_id,
            sampling_params,
            arrival_time_ns: 0,
            generated_tokens: Vec::new(),
        }
    }

    pub fn from_work(
        work: InferenceWork,
        seq_id: SequenceId,
        prompt: PromptHandle,
        sampling_params: SamplingParams,
        sink_id: Option<CompletionSinkId>,
    ) -> Self {
        Self {
            seq_id,
            prompt,
            model: work.model,
            kv: work.kv,
            priority: work.priority,
            deadline: work.deadline,
            branch_parent: work
                .branch_parent
                .and_then(|p| SequenceId::from_u64(p).ok()),
            next_token_budget: work.next_token_budget,
            phase: SequencePhase::Waiting,
            tokens_generated: 0,
            prompt_tokens_prefilled: 0,
            is_prefilled: false,
            sink_id,
            sampling_params,
            arrival_time_ns: 0,
            generated_tokens: Vec::new(),
        }
    }

    pub fn remaining_prefill_tokens(&self) -> usize {
        self.prompt
            .len()
            .saturating_sub(self.prompt_tokens_prefilled)
    }

    pub fn total_tokens(&self) -> usize {
        self.prompt.len() + self.tokens_generated
    }

    pub fn is_finished(&self) -> bool {
        self.phase == SequencePhase::Finished
            || self.tokens_generated >= self.sampling_params.max_tokens
            || (self.next_token_budget > 0
                && self.tokens_generated >= self.next_token_budget as usize)
    }

    pub fn to_request(&self) -> SequenceRequest {
        SequenceRequest {
            request_id: self.seq_id.to_u64(),
            prompt_tokens: self.prompt.to_vec(),
            sampling_params: self.sampling_params.clone(),
            arrival_time_ns: self.arrival_time_ns,
            priority: self.priority as u8,
        }
    }
}

/// Slot entry within SequenceArena.
#[derive(Debug, Clone)]
pub struct SlotEntry {
    pub generation: u32,
    pub retired: bool,
    pub record: Option<SequenceRecord>,
}

/// Central state arena managing active sequence records with slot/generation tracking.
#[derive(Debug, Default)]
pub struct SequenceArena {
    slots: Vec<SlotEntry>,
    free_slots: Vec<u32>,
    retired_slots_count: usize,
    total_tokens_generated: usize,
    total_prefilled_tokens: usize,
}

impl SequenceArena {
    pub fn new() -> Self {
        Self {
            slots: Vec::new(),
            free_slots: Vec::new(),
            retired_slots_count: 0,
            total_tokens_generated: 0,
            total_prefilled_tokens: 0,
        }
    }

    pub fn with_capacity(capacity: usize) -> Self {
        let mut slots = Vec::with_capacity(capacity);
        let mut free_slots = Vec::with_capacity(capacity);
        for i in 0..capacity {
            slots.push(SlotEntry {
                generation: 1, // Generation 0 is strictly invalid
                retired: false,
                record: None,
            });
            free_slots.push(i as u32);
        }
        free_slots.reverse();

        Self {
            slots,
            free_slots,
            retired_slots_count: 0,
            total_tokens_generated: 0,
            total_prefilled_tokens: 0,
        }
    }

    /// Allocates an active sequence slot with a non-zero generation.
    /// Retires slot if generation wraps to u32::MAX.
    pub fn allocate_slot(&mut self) -> Result<SequenceId, String> {
        while let Some(slot_idx) = self.free_slots.pop() {
            let slot = &mut self.slots[slot_idx as usize];
            if slot.retired {
                continue;
            }
            if slot.generation == 0 {
                slot.generation = 1;
            }
            return Ok(SequenceId {
                slot: slot_idx,
                generation: slot.generation,
            });
        }

        let slot_idx = self.slots.len() as u32;
        let generation = 1; // Generation 0 is invalid
        self.slots.push(SlotEntry {
            generation,
            retired: false,
            record: None,
        });

        Ok(SequenceId {
            slot: slot_idx,
            generation,
        })
    }

    /// Inserts a sequence into an allocated slot.
    pub fn insert_with(
        &mut self,
        prompt: PromptHandle,
        model: ModelHandle,
        kv: KvHandle,
        priority: Priority,
        deadline: Option<Ticks>,
        branch_parent: Option<SequenceId>,
        next_token_budget: u32,
        sampling_params: SamplingParams,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        let seq_id = self.allocate_slot()?;
        let record = SequenceRecord::new(
            seq_id,
            prompt,
            model,
            kv,
            priority,
            deadline,
            branch_parent,
            next_token_budget,
            sampling_params,
            sink_id,
        );
        self.slots[seq_id.slot as usize].record = Some(record);
        Ok(seq_id)
    }

    /// Inserts a sequence record with an explicit SequenceId (e.g. from upstream caller/runtime).
    pub fn insert_with_id(
        &mut self,
        seq_id: SequenceId,
        prompt: PromptHandle,
        model: ModelHandle,
        kv: KvHandle,
        priority: Priority,
        deadline: Option<Ticks>,
        branch_parent: Option<SequenceId>,
        next_token_budget: u32,
        sampling_params: SamplingParams,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        let slot_idx = seq_id.slot;
        if (slot_idx as usize) >= self.slots.len() {
            return Err(format!(
                "Slot index {} exceeds arena capacity {}",
                slot_idx,
                self.slots.len()
            ));
        }
        if let Some(pos) = self.free_slots.iter().position(|&s| s == slot_idx) {
            self.free_slots.remove(pos);
        }
        let slot = &mut self.slots[slot_idx as usize];
        slot.generation = seq_id.generation;
        slot.retired = false;
        let record = SequenceRecord::new(
            seq_id,
            prompt,
            model,
            kv,
            priority,
            deadline,
            branch_parent,
            next_token_budget,
            sampling_params,
            sink_id,
        );
        slot.record = Some(record);
        Ok(seq_id)
    }

    /// Validates whether a SequenceId is stale:
    /// - generation 0
    /// - slot index out of bounds
    /// - generation doesn't match current slot generation
    /// - slot has no active record
    pub fn is_stale(&self, id: SequenceId) -> bool {
        if id.generation == 0 || (id.slot as usize) >= self.slots.len() {
            return true;
        }
        let slot = &self.slots[id.slot as usize];
        slot.generation != id.generation || slot.record.is_none()
    }

    pub fn get(&self, id: SequenceId) -> Option<&SequenceRecord> {
        if id.generation == 0 || (id.slot as usize) >= self.slots.len() {
            return None;
        }
        let slot = &self.slots[id.slot as usize];
        if slot.generation == id.generation {
            slot.record.as_ref()
        } else {
            None
        }
    }

    pub fn get_mut(&mut self, id: SequenceId) -> Option<&mut SequenceRecord> {
        if id.generation == 0 || (id.slot as usize) >= self.slots.len() {
            return None;
        }
        let slot = &mut self.slots[id.slot as usize];
        if slot.generation == id.generation {
            slot.record.as_mut()
        } else {
            None
        }
    }

    /// Deallocates a sequence, advancing its generation to invalidate stale queued IDs.
    /// Retires the slot if generation reaches u32::MAX.
    pub fn free_sequence(&mut self, id: SequenceId) -> bool {
        if id.generation == 0 || (id.slot as usize) >= self.slots.len() {
            return false;
        }
        let slot = &mut self.slots[id.slot as usize];
        if slot.generation != id.generation || slot.record.is_none() {
            return false;
        }

        slot.record = None;
        if slot.generation == u32::MAX {
            slot.retired = true;
            self.retired_slots_count += 1;
        } else {
            slot.generation += 1;
            self.free_slots.push(id.slot);
        }
        true
    }

    /// Zero-copy sequence fork preserving immutable PromptHandle.
    pub fn fork(
        &mut self,
        parent_id: SequenceId,
        sink_id: Option<CompletionSinkId>,
    ) -> Result<SequenceId, String> {
        let parent = self
            .get(parent_id)
            .ok_or_else(|| format!("Parent sequence {} not found in arena or stale", parent_id))?
            .clone();

        let child_id = self.allocate_slot()?;
        let child_record = SequenceRecord {
            seq_id: child_id,
            prompt: Arc::clone(&parent.prompt),
            model: parent.model,
            kv: KvHandle(child_id.to_u64()),
            priority: parent.priority,
            deadline: parent.deadline,
            branch_parent: Some(parent_id),
            next_token_budget: parent.next_token_budget,
            phase: SequencePhase::Decode,
            tokens_generated: parent.tokens_generated,
            prompt_tokens_prefilled: parent.prompt_tokens_prefilled,
            is_prefilled: parent.is_prefilled,
            sink_id,
            sampling_params: parent.sampling_params.clone(),
            arrival_time_ns: parent.arrival_time_ns,
            generated_tokens: parent.generated_tokens.clone(),
        };

        self.slots[child_id.slot as usize].record = Some(child_record);
        Ok(child_id)
    }

    pub fn active_ids(&self) -> Vec<SequenceId> {
        let mut ids = Vec::new();
        for (idx, slot) in self.slots.iter().enumerate() {
            if slot.record.is_some() {
                ids.push(SequenceId {
                    slot: idx as u32,
                    generation: slot.generation,
                });
            }
        }
        ids
    }

    pub fn active_count(&self) -> usize {
        self.slots.iter().filter(|s| s.record.is_some()).count()
    }

    pub fn len(&self) -> usize {
        self.active_count()
    }

    pub fn is_empty(&self) -> bool {
        self.active_count() == 0
    }

    pub fn retired_slots(&self) -> usize {
        self.retired_slots_count
    }

    pub fn record_generated_token(&mut self, seq_id: SequenceId, token: u32) {
        if let Some(record) = self.get_mut(seq_id) {
            record.tokens_generated += 1;
            record.generated_tokens.push(token);
            self.total_tokens_generated += 1;
        }
    }

    pub fn record_prefilled_tokens(&mut self, seq_id: SequenceId, count: usize) {
        if let Some(record) = self.get_mut(seq_id) {
            record.prompt_tokens_prefilled += count;
            if record.prompt_tokens_prefilled >= record.prompt.len() {
                record.is_prefilled = true;
                record.phase = SequencePhase::Decode;
            } else {
                record.phase = SequencePhase::Prefill;
            }
            self.total_prefilled_tokens += count;
        }
    }

    pub fn total_tokens_generated(&self) -> usize {
        self.total_tokens_generated
    }

    pub fn total_prefilled_tokens(&self) -> usize {
        self.total_prefilled_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sequence_id_generation_zero_invalid() {
        assert!(SequenceId::new(0, 0).is_err());
        assert!(!SequenceId::INVALID.is_valid());
        assert!(SequenceId::from_u64(0).is_err());

        let valid = SequenceId::new(42, 1).unwrap();
        assert!(valid.is_valid());
        let raw = valid.to_u64();
        let recovered = SequenceId::from_u64(raw).unwrap();
        assert_eq!(valid, recovered);
        assert_eq!(recovered.slot, 42);
        assert_eq!(recovered.generation, 1);
    }

    #[test]
    fn test_slot_retirement_on_generation_wrap() {
        let mut arena = SequenceArena::new();
        let id1 = arena.allocate_slot().unwrap();
        assert_eq!(id1.slot, 0);
        assert_eq!(id1.generation, 1);

        // Manually simulate generation approaching u32::MAX
        arena.slots[0].generation = u32::MAX;
        arena.slots[0].record = Some(SequenceRecord::new(
            SequenceId {
                slot: 0,
                generation: u32::MAX,
            },
            Arc::from(vec![1, 2, 3].into_boxed_slice()),
            ModelHandle(0),
            KvHandle(0),
            Priority::Normal,
            None,
            None,
            10,
            SamplingParams::default(),
            None,
        ));

        // Free the sequence -> should retire slot 0
        let target_id = SequenceId {
            slot: 0,
            generation: u32::MAX,
        };
        assert!(arena.free_sequence(target_id));
        assert_eq!(arena.retired_slots(), 1);
        assert!(arena.slots[0].retired);

        // Next allocation should NOT reuse retired slot 0
        let id2 = arena.allocate_slot().unwrap();
        assert_eq!(id2.slot, 1);
        assert_eq!(id2.generation, 1);
    }

    #[test]
    fn test_stale_work_rejection() {
        let mut arena = SequenceArena::new();
        let prompt: PromptHandle = Arc::from(vec![10, 20, 30].into_boxed_slice());
        let id1 = arena
            .insert_with(
                prompt,
                ModelHandle(0),
                KvHandle(0),
                Priority::Normal,
                None,
                None,
                10,
                SamplingParams::default(),
                None,
            )
            .unwrap();

        assert!(!arena.is_stale(id1));

        // Free sequence 1 -> generation bumps to 2
        assert!(arena.free_sequence(id1));

        // Old id1 is now stale!
        assert!(arena.is_stale(id1));
        assert!(arena.get(id1).is_none());

        // New allocation in slot 0 gets generation 2
        let prompt2: PromptHandle = Arc::from(vec![40, 50].into_boxed_slice());
        let id2 = arena
            .insert_with(
                prompt2,
                ModelHandle(0),
                KvHandle(0),
                Priority::Normal,
                None,
                None,
                10,
                SamplingParams::default(),
                None,
            )
            .unwrap();

        assert_eq!(id2.slot, id1.slot);
        assert_eq!(id2.generation, id1.generation + 1);
        assert!(!arena.is_stale(id2));
        assert!(arena.is_stale(id1)); // id1 still stale!
    }

    #[test]
    fn test_prompt_handle_sharing_and_zero_copy_fork() {
        let prompt_tokens = vec![1, 2, 3, 4, 5];
        let handle: PromptHandle = Arc::from(prompt_tokens.into_boxed_slice());

        let mut arena = SequenceArena::new();
        let parent_id = arena
            .insert_with(
                handle.clone(),
                ModelHandle(0),
                KvHandle(1),
                Priority::Normal,
                None,
                None,
                128,
                SamplingParams::default(),
                None,
            )
            .unwrap();

        assert_eq!(arena.active_count(), 1);

        let child_id = arena.fork(parent_id, None).unwrap();
        assert_eq!(arena.active_count(), 2);

        let child = arena.get(child_id).unwrap();
        assert_eq!(child.branch_parent, Some(parent_id));
        assert!(Arc::ptr_eq(&child.prompt, &handle));
    }

    #[test]
    fn test_completion_router_registration_and_dispatch() {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let sink = Arc::new(ChannelCompletionSink::new(tx));

        let mut router = CompletionRouter::new();
        let sink_id = router.register(sink);

        let seq_id = SequenceId::new(1, 1).unwrap();
        router.emit(sink_id, CompletionEvent::Token { seq_id, token: 42 });

        let event = rx.try_recv().unwrap();
        match event {
            CompletionEvent::Token { seq_id: id, token } => {
                assert_eq!(id, seq_id);
                assert_eq!(token, 42);
            }
            _ => panic!("Unexpected event"),
        }
    }
}
