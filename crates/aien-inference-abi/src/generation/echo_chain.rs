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

//! The host fake forward pass `EchoChainModel` (ADR 0004 section 4), behind `host-test-engine`.
//!
//! No weights, vocabulary 64, the tiny KV geometry (2 layers, 1 KV head of 128), on a real
//! [`HostPoolKv`], so export, restore and park move real `HostPagedPoolV1` bytes. For token `t` at
//! position `p` and layer `l` it appends K and V rows of small integers (exact in bf16), then
//! attends with a fixed query over `0..=p` at every layer and folds every layer's attention
//! output into one hash `h` together with `p`; `logits[i] = ((h + 37 i) % 64) / 64`, a
//! permutation, so the argmax is unique. It keeps no history of its own: every logit depends on
//! every stored row in every layer, so a stale, missing or wrong row changes the tokens.

use std::sync::{Arc, Mutex};

use super::{gather_rows, ForwardPass, KvIdentityInputs, Next, Want};
use crate::qwen3_moe::{bf16_to_f32, f32_to_bf16};
use crate::qwen3_moe_omega::{
    HostPoolKv, KvAttention, KvGeometry, KvStore, StepTiming, KV_BLOCK_SIZE,
};
use crate::tensor::sample_argmax;

/// The fake's vocabulary.
pub const ECHO_VOCAB: usize = 64;
const LAYERS: usize = 2;
const HEAD_DIM: usize = 128;

/// The fake's KV geometry: 2 layers, 1 query head, 1 KV head of 128.
pub fn echo_geometry() -> KvGeometry {
    KvGeometry {
        layers: LAYERS,
        q_heads: 1,
        kv_heads: 1,
        head_dim: HEAD_DIM,
        rope_theta: 1_000_000.0,
        block_size: KV_BLOCK_SIZE,
    }
}

/// One entry of the fake's call log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Call {
    Prefill {
        pos: usize,
        len: usize,
    },
    Decode {
        pos: usize,
        tokens: Vec<u32>,
    },
    /// Pushed by a test's `on_token` through [`EchoChainModel::log_handle`], to check ordering.
    Emitted(u32),
}

/// The fake's cache: a [`KvAttention`] whose every layer can be inspected.
pub trait EchoKv: KvAttention {
    /// Positions held in `layer` alone.
    fn layer_len(&self, layer: usize) -> usize;
    /// `Some(reason)` when this cache deliberately damages K/V (a negative control).
    fn mutant(&self) -> Option<&'static str> {
        None
    }
}

impl EchoKv for HostPoolKv {
    fn layer_len(&self, layer: usize) -> usize {
        HostPoolKv::layer_len(self, layer)
    }
}

/// Negative control 1 (ADR 0004 section 4): a [`HostPoolKv`] that, on the first truncate that
/// keeps position `pos` (a rewind), overwrites the V row of (`pos`, `layer`) with different
/// bytes. A warm run that reuses that row must then differ from a cold run.
pub struct CorruptRowKv {
    inner: HostPoolKv,
    pos: usize,
    layer: usize,
    fired: bool,
}

impl CorruptRowKv {
    pub fn new(max_context: usize, pos: usize, layer: usize) -> Result<Self, String> {
        Ok(Self {
            inner: HostPoolKv::new(&echo_geometry(), max_context)?,
            pos,
            layer,
            fired: false,
        })
    }

    /// Whether the row has been overwritten.
    pub fn fired(&self) -> bool {
        self.fired
    }

    /// Rewrites positions `0..len` through export and import with one V row changed (every
    /// element plus 7, exact in bf16). `append` refuses out-of-order writes, so the payload is
    /// the only door to a surviving row.
    fn corrupt(&mut self, len: usize) {
        let mut bytes = match self.inner.export_blocks(0, len) {
            Some(Ok(b)) => b,
            other => panic!("corrupt row: export of {len} positions failed: {other:?}"),
        };
        let g = self.inner.geometry().pool_block();
        let off = g.row_offset(self.pos, self.layer, true, 0);
        for e in bytes[off..off + g.token_stride()].chunks_exact_mut(2) {
            let v = bf16_to_f32(u16::from_le_bytes([e[0], e[1]])) + 7.0;
            e.copy_from_slice(&f32_to_bf16(v).to_le_bytes());
        }
        self.inner.truncate(0);
        match self.inner.import_blocks(0, len, &bytes) {
            Some(Ok(())) => {}
            other => panic!("corrupt row: import of {len} positions failed: {other:?}"),
        }
        self.fired = true;
    }
}

impl KvStore for CorruptRowKv {
    fn geometry(&self) -> KvGeometry {
        self.inner.geometry()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn path_name(&self) -> &'static str {
        self.inner.path_name()
    }
    fn truncate(&mut self, len: usize) {
        self.inner.truncate(len);
        if !self.fired && self.pos < len && self.inner.len() == len {
            self.corrupt(len);
        }
    }
    fn export_blocks(&mut self, start: usize, len: usize) -> Option<Result<Vec<u8>, String>> {
        self.inner.export_blocks(start, len)
    }
    fn import_blocks(
        &mut self,
        start: usize,
        len: usize,
        bytes: &[u8],
    ) -> Option<Result<(), String>> {
        self.inner.import_blocks(start, len, bytes)
    }
}

impl KvAttention for CorruptRowKv {
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        self.inner.append(layer, pos, k, v)
    }
    fn attend(
        &mut self,
        layer: usize,
        q: &[f32],
        context_len: usize,
        out: &mut [f32],
    ) -> Result<u64, String> {
        self.inner.attend(layer, q, context_len, out)
    }
    fn resident(&self) -> bool {
        false
    }
}

impl EchoKv for CorruptRowKv {
    fn layer_len(&self, layer: usize) -> usize {
        self.inner.layer_len(layer)
    }
    fn mutant(&self) -> Option<&'static str> {
        Some("the corrupt-row negative control damages a stored K/V row")
    }
}

/// The cache as the fake lends it through `kv()`. Lending forgets nothing: pending heads stay
/// pending until `clear_pending`, `prefill` or `decode` replaces them (like the 30B's held graph
/// token), so a test sees every `clear_pending` the controller makes. A head is still refused
/// after the cache changed under it (I2): a truncate that drops a row or an import moves `epoch`.
struct Lent {
    inner: Box<dyn EchoKv>,
    epoch: u64,
}

impl KvStore for Lent {
    fn geometry(&self) -> KvGeometry {
        self.inner.geometry()
    }
    fn capacity(&self) -> usize {
        self.inner.capacity()
    }
    fn len(&self) -> usize {
        self.inner.len()
    }
    fn path_name(&self) -> &'static str {
        self.inner.path_name()
    }
    fn truncate(&mut self, len: usize) {
        if (0..LAYERS).any(|l| self.inner.layer_len(l) > len) {
            self.epoch += 1;
        }
        self.inner.truncate(len);
    }
    fn export_blocks(&mut self, start: usize, len: usize) -> Option<Result<Vec<u8>, String>> {
        self.inner.export_blocks(start, len)
    }
    fn import_blocks(
        &mut self,
        start: usize,
        len: usize,
        bytes: &[u8],
    ) -> Option<Result<(), String>> {
        self.epoch += 1;
        self.inner.import_blocks(start, len, bytes)
    }
}

/// One pending output.
struct Head {
    h: u64,
    /// The cache epoch it was made in: a truncate that drops a row, or an import, makes it stale.
    epoch: u64,
    /// Graph mode: the argmax taken at `decode` and held, like the 30B's one-list graph.
    chip_token: Option<u32>,
}

/// The host fake forward pass. Switches and instruments are public fields.
pub struct EchoChainModel {
    kv: Lent,
    heads: Vec<Option<Head>>,
    logits: Vec<f32>,
    log: Arc<Mutex<Vec<Call>>>,
    /// Full-logits read-backs of a graph head (the 30B's `dgraph_take_logits`).
    pub chip_reads: usize,
    /// Host output heads (the 30B's `logits()`): every prefill head, every decode head off the graph.
    pub lm_heads: usize,
    /// `next_rows` calls.
    pub rows_reads: usize,
    /// Decode heads hold the argmax, taken eagerly, and answer `Want::Greedy` with no read.
    pub graph_mode: bool,
    /// Writing this position fails after layer 0 (a deliberate half-write).
    pub fail_at: Option<usize>,
    /// Added to every K/V element: an identity entry.
    pub salt: u32,
    /// What `max_decode_batch()` reports.
    pub max_decode_batch: usize,
}

impl EchoChainModel {
    /// The fake on a [`HostPoolKv`] of `max_context` tokens (rounded up to whole blocks).
    pub fn new(max_context: usize) -> Self {
        let kv = HostPoolKv::new(&echo_geometry(), max_context).expect("echo-chain host pool");
        Self::with_kv(Box::new(kv))
    }

    /// The fake on any inspectable cache built for [`echo_geometry`] (a negative control).
    pub fn with_kv(kv: Box<dyn EchoKv>) -> Self {
        Self {
            kv: Lent {
                inner: kv,
                epoch: 0,
            },
            heads: Vec::new(),
            logits: vec![0.0; ECHO_VOCAB],
            log: Arc::new(Mutex::new(Vec::new())),
            chip_reads: 0,
            lm_heads: 0,
            rows_reads: 0,
            graph_mode: false,
            fail_at: None,
            salt: 0,
            max_decode_batch: 1,
        }
    }

    /// The call log so far.
    pub fn calls(&self) -> Vec<Call> {
        self.log.lock().expect("echo-chain log").clone()
    }

    /// The shared call log, for a test's `on_token` to push [`Call::Emitted`] into.
    pub fn log_handle(&self) -> Arc<Mutex<Vec<Call>>> {
        Arc::clone(&self.log)
    }

    /// Positions held by each layer.
    pub fn layer_lens(&self) -> Vec<usize> {
        (0..LAYERS).map(|l| self.kv.inner.layer_len(l)).collect()
    }

    /// The cache's mutant reason, if any.
    pub fn kv_mutant(&self) -> Option<&'static str> {
        self.kv.inner.mutant()
    }

    /// The `d`th element of the K (`s` 0) or V (`s` 1) row of token `t` at (`p`, `l`).
    fn element(&self, t: u32, p: usize, l: usize, d: usize, s: usize) -> f32 {
        let n = t as usize * 131 + p * 17 + l * 7 + d + 3 * s + self.salt as usize;
        (n % 61) as f32 - 30.0
    }

    /// Writes `tokens` from `pos` in every layer; the hash of each position's attention outputs.
    fn run(&mut self, pos: usize, tokens: &[u32]) -> Result<Vec<u64>, String> {
        self.heads.clear();
        if tokens.is_empty() {
            return Err("echo-chain: no tokens".into());
        }
        if let Some(t) = tokens.iter().find(|&&t| t as usize >= ECHO_VOCAB) {
            return Err(format!("echo-chain: token {t} outside the vocabulary"));
        }
        // I1: the ledger's position must be what every layer holds.
        if self.kv.inner.len() != pos {
            return Err(format!(
                "echo-chain: asked to write at {pos}, the cache holds {}",
                self.kv.inner.len()
            ));
        }
        let query: Vec<f32> = (0..HEAD_DIM)
            .map(|d| ((d * 5) % 7) as f32 / 1024.0 - 3.0 / 1024.0)
            .collect();
        let mut out = vec![0.0f32; HEAD_DIM];
        let mut hashes = Vec::with_capacity(tokens.len());
        for (i, &t) in tokens.iter().enumerate() {
            let p = pos + i;
            let mut h = mix(0xcbf2_9ce4_8422_2325, p as u64);
            for l in 0..LAYERS {
                if l > 0 && self.fail_at == Some(p) {
                    return Err(format!(
                        "echo-chain: injected failure at position {p} after layer 0"
                    ));
                }
                let k: Vec<f32> = (0..HEAD_DIM).map(|d| self.element(t, p, l, d, 0)).collect();
                let v: Vec<f32> = (0..HEAD_DIM).map(|d| self.element(t, p, l, d, 1)).collect();
                self.kv.inner.append(l, p, &k, &v)?;
                self.kv.inner.attend(l, &query, p + 1, &mut out)?;
                for x in &out {
                    h = mix(h, u64::from(x.to_bits()));
                }
            }
            hashes.push(h);
        }
        Ok(hashes)
    }

    fn fill_logits(&mut self, h: u64) {
        for (i, l) in self.logits.iter_mut().enumerate() {
            *l = (h.wrapping_add(37 * i as u64) % ECHO_VOCAB as u64) as f32 / ECHO_VOCAB as f32;
        }
    }

    fn push(&self, call: Call) {
        self.log.lock().expect("echo-chain log").push(call);
    }
}

/// FNV-1a step over one 64-bit word.
fn mix(h: u64, x: u64) -> u64 {
    (h ^ x).wrapping_mul(0x0000_0100_0000_01b3)
}

impl ForwardPass for EchoChainModel {
    fn kind(&self) -> &'static str {
        "echo-chain"
    }
    fn vocab(&self) -> usize {
        ECHO_VOCAB
    }
    fn kv(&mut self) -> &mut dyn KvStore {
        &mut self.kv
    }
    fn max_decode_batch(&self) -> usize {
        self.max_decode_batch
    }
    fn prefill(
        &mut self,
        pos: usize,
        tokens: &[u32],
        timing: &mut StepTiming,
    ) -> Result<(), String> {
        self.push(Call::Prefill {
            pos,
            len: tokens.len(),
        });
        let hashes = self.run(pos, tokens)?;
        timing.launches += tokens.len() as u64;
        let h = *hashes.last().expect("non-empty prefill");
        self.heads = vec![Some(Head {
            h,
            epoch: self.kv.epoch,
            chip_token: None,
        })];
        Ok(())
    }
    fn decode(
        &mut self,
        pos: usize,
        tokens: &[u32],
        timing: &mut StepTiming,
    ) -> Result<(), String> {
        self.push(Call::Decode {
            pos,
            tokens: tokens.to_vec(),
        });
        if tokens.len() > self.max_decode_batch {
            return Err(format!(
                "echo-chain: decode of {} tokens, batch is {}",
                tokens.len(),
                self.max_decode_batch
            ));
        }
        let hashes = self.run(pos, tokens)?;
        timing.launches += tokens.len() as u64;
        if self.graph_mode {
            timing.graph_tokens += tokens.len() as u64;
            timing.graph_submits += 1;
        }
        let mut heads = Vec::with_capacity(hashes.len());
        for h in hashes {
            let chip_token = if self.graph_mode {
                self.fill_logits(h);
                Some(sample_argmax(&self.logits).0)
            } else {
                None
            };
            heads.push(Some(Head {
                h,
                epoch: self.kv.epoch,
                chip_token,
            }));
        }
        self.heads = heads;
        Ok(())
    }
    fn next(
        &mut self,
        head: usize,
        want: Want,
        timing: &mut StepTiming,
    ) -> Result<Next<'_>, String> {
        let hd = self
            .heads
            .get_mut(head)
            .and_then(Option::take)
            .ok_or_else(|| format!("echo-chain: head {head} is not pending"))?;
        if hd.epoch != self.kv.epoch {
            return Err(format!(
                "echo-chain: head {head} is stale: the cache changed under it"
            ));
        }
        match (hd.chip_token, want) {
            (Some(t), Want::Greedy) => return Ok(Next::Token(t)),
            (Some(_), Want::Logits) => self.chip_reads += 1,
            (None, _) => {
                self.lm_heads += 1;
                timing.launches += 1;
            }
        }
        self.fill_logits(hd.h);
        Ok(Next::Logits(&self.logits))
    }
    fn next_rows(
        &mut self,
        head: usize,
        rows: &[u32],
        out: &mut [f32],
        timing: &mut StepTiming,
    ) -> Result<(), String> {
        self.rows_reads += 1;
        gather_rows(self, head, rows, out, timing)
    }
    fn clear_pending(&mut self) {
        self.heads.clear();
    }
    fn kv_identity(&self) -> Result<KvIdentityInputs, String> {
        if let Some(why) = self.kv.inner.mutant() {
            return Err(why.into());
        }
        // The fake's K/V bytes depend on the cache path and the salt only (graph_mode, the
        // instruments and the decode batch change no byte).
        Ok(KvIdentityInputs {
            kind: self.kind(),
            entries: vec![
                ("attention_path", self.kv.inner.path_name().to_string()),
                ("salt", self.salt.to_string()),
            ],
        })
    }
}
