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

//! PREFILL-I25: a KV pool that runs out during a prefill append or a
//! single-token decode step must fail LOUDLY and leave a retryable state.
//!
//! Before the fix, `prefill_prompt_layer_by_layer_paged` turned an
//! out-of-blocks error from `append_token_with_slot` into `None` (`.ok()`),
//! dropped the K/V write result, and returned Ok: the appended tokens were
//! silently not stored in the pool. The single-token path
//! (`forward_token_impl_paged`) had the same `.ok()` on its slot.
//!
//! Runs on the CPU reference backend with synthetic reference weights and a
//! tiny paged pool (block size 4), so it needs no checkpoint and takes
//! seconds. It is not ignored.
//!
//! Receipt: written as JSON to the path in env `AIEN_I25_RECEIPT` (when set)
//! BEFORE any assertion fires. Every check has its own marker in its failure
//! message; all check outcomes are printed first so every marker that
//! applies is in the log.
//!
//! Checks and markers:
//! - x_prefill_exhaustion_is_loud    POOL_SILENT_SUCCESS_VIOLATION
//!   (POOL_ERROR_PREFIX_VIOLATION when the Err lacks the stable prefix or
//!   the needed/available numbers)
//! - x_prefill_exhaustion_is_atomic  POOL_PARTIAL_STATE_VIOLATION
//! - x_prefill_clean_path_unchanged  CLEAN_PATH_VIOLATION
//! - x_exact_fit_ok                  BOUNDARY_VIOLATION
//! - x_decode_exhaustion_is_loud     POOL_SILENT_SUCCESS_VIOLATION
//! - x_recovers_after_free           RETRY_VIOLATION
//!
//! A check whose precondition is not met (for example the atomicity check
//! when the append did not fail at all) is recorded with `applicable: false`
//! and passes vacuously; the check that owns that precondition fails instead.
//!
//! Scenarios: the prompt is 6 tokens (one full block of 4 and a tail with 2
//! tokens). The second prefill chunk appends 11 tokens (ends mid-block) or
//! 10 tokens (ends exactly on a block boundary), with the prompt's tail
//! either private or shared with a second sequence (so the first append
//! needs a copy-on-write block). The number of new blocks each scenario
//! needs is computed by `oracle_new_blocks` from first principles, not by
//! the backend's own helper.

use aien_inference_abi::{
    AienInferenceBackend, DecodeOutput, FinishReason, ModelConfig, NativeTransformerBackend,
    ScheduledBatch, TransformerWeights,
};
use std::collections::HashMap;

const BLOCK_SIZE: usize = 4;
/// Prompt tokens: 6 = one full block of 4 plus a partly filled tail of 2.
const PROMPT_LEN: usize = 6;
/// Second-chunk lengths: 11 ends mid-block (17 tokens), 10 ends exactly on a
/// block boundary (16 tokens).
const APPEND_LENS: [usize; 2] = [11, 10];
/// Pool used for the exhaustion and boundary scenarios.
const SMALL_POOL: usize = 24;
/// Pool that is always large enough.
const BIG_POOL: usize = 64;

const SEQ: u64 = 7001;
const WITNESS: u64 = 7002;
const FILLER: u64 = 7003;
const CONTROL: u64 = 7004;

/// The stable prefix of every pool-exhaustion error. Kept as a literal here
/// (not imported from the crate) so a renamed or reworded prefix is caught.
const PREFIX: &str = "KV pool exhausted:";

const M_SILENT: &str = "POOL_SILENT_SUCCESS_VIOLATION";
const M_PREFIX: &str = "POOL_ERROR_PREFIX_VIOLATION";
const M_PARTIAL: &str = "POOL_PARTIAL_STATE_VIOLATION";
const M_CLEAN: &str = "CLEAN_PATH_VIOLATION";
const M_BOUNDARY: &str = "BOUNDARY_VIOLATION";
const M_RETRY: &str = "RETRY_VIOLATION";

type Planes = Vec<(Vec<f32>, Vec<f32>)>;

/// One scenario: is the prompt's tail shared with `WITNESS`, and how many
/// tokens does the second prefill chunk append.
#[derive(Clone, Copy)]
struct Scenario {
    shared_tail: bool,
    len: usize,
}

impl Scenario {
    fn label(&self) -> String {
        format!("shared_tail={} append_len={}", self.shared_tail, self.len)
    }

    /// New blocks this scenario's append needs.
    fn need(&self) -> usize {
        oracle_new_blocks(PROMPT_LEN, self.len, self.shared_tail)
    }
}

fn scenarios() -> Vec<Scenario> {
    let mut all = Vec::new();
    for len in APPEND_LENS {
        for shared_tail in [false, true] {
            all.push(Scenario { shared_tail, len });
        }
    }
    all
}

fn config() -> ModelConfig {
    ModelConfig {
        num_layers: 2,
        num_heads: 4,
        num_kv_heads: 2,
        head_dim: 16,
        hidden_dim: 64,
        intermediate_dim: 128,
        vocab_size: 256,
        block_size: BLOCK_SIZE,
        ..Default::default()
    }
}

fn backend(total_blocks: usize) -> NativeTransformerBackend {
    let weights = TransformerWeights::reference_test_weights(&config());
    NativeTransformerBackend::with_paged_kv(weights, total_blocks, BLOCK_SIZE)
        .expect("small paged backend")
}

/// Deterministic tokens `start..start+n` of one fixed stream (ids 3..=255).
fn tokens(start: usize, n: usize) -> Vec<u32> {
    (start..start + n)
        .map(|i| ((i * 37 + 11) % 253 + 3) as u32)
        .collect()
}

fn free_blocks(be: &NativeTransformerBackend) -> usize {
    let kv = be.kv_manager.as_ref().expect("kv manager").read();
    kv.free_block_count()
}

fn table_of(be: &NativeTransformerBackend, id: u64) -> (Vec<usize>, usize) {
    let kv = be.kv_manager.as_ref().expect("kv manager").read();
    let table = kv.get_block_table(id).expect("block table");
    (table.block_ids.clone(), table.total_tokens)
}

/// K and V of every layer over the table's tokens, gathered from the pool.
fn kv_planes(be: &NativeTransformerBackend, id: u64) -> Planes {
    let kv = be.kv_manager.as_ref().expect("kv manager").read();
    let mut planes: Planes = Vec::new();
    for layer in 0..config().num_layers {
        planes.push(kv.gather_layer_kv(id, layer).expect("gather layer kv"));
    }
    planes
}

/// Everything the atomicity check compares before and after a failed call.
struct Snap {
    ids: Vec<usize>,
    total: usize,
    free: usize,
    seq_tokens: Vec<u32>,
    planes: Planes,
}

fn snap(be: &NativeTransformerBackend, id: u64) -> Snap {
    let (ids, total) = table_of(be, id);
    Snap {
        ids,
        total,
        free: free_blocks(be),
        seq_tokens: be.sequences[&id].tokens.clone(),
        planes: kv_planes(be, id),
    }
}

/// Field-by-field differences between two snapshots (empty = identical).
fn snap_diff(a: &Snap, b: &Snap) -> Vec<String> {
    let mut d = Vec::new();
    if a.ids != b.ids {
        d.push(format!("block ids {:?} -> {:?}", a.ids, b.ids));
    }
    if a.total != b.total {
        d.push(format!("table total_tokens {} -> {}", a.total, b.total));
    }
    if a.free != b.free {
        d.push(format!("free blocks {} -> {}", a.free, b.free));
    }
    if a.seq_tokens != b.seq_tokens {
        d.push(format!(
            "backend sequence tokens {} -> {}",
            a.seq_tokens.len(),
            b.seq_tokens.len()
        ));
    }
    if !planes_equal(&a.planes, &b.planes) {
        d.push("pool K/V of the sequence changed".to_string());
    }
    d
}

/// Bitwise equality of two gathered K/V sets (tolerance 0.0).
fn planes_equal(a: &Planes, b: &Planes) -> bool {
    a.len() == b.len()
        && a.iter().zip(b.iter()).all(|(x, y)| {
            x.0.len() == y.0.len()
                && x.1.len() == y.1.len()
                && x.0
                    .iter()
                    .zip(y.0.iter())
                    .all(|(p, q)| p.to_bits() == q.to_bits())
                && x.1
                    .iter()
                    .zip(y.1.iter())
                    .all(|(p, q)| p.to_bits() == q.to_bits())
        })
}

/// Largest absolute K or V difference (for the message only; equality is bitwise).
fn planes_max_abs_diff(a: &Planes, b: &Planes) -> f32 {
    if a.len() != b.len() {
        return f32::INFINITY;
    }
    let mut max = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.0.len() != y.0.len() || x.1.len() != y.1.len() {
            return f32::INFINITY;
        }
        for (p, q) in x.0.iter().zip(y.0.iter()).chain(x.1.iter().zip(y.1.iter())) {
            max = max.max((p - q).abs());
        }
    }
    max
}

/// One-shot control: the first `len` tokens of the stream prefilled in one
/// call on a backend with a large pool.
fn control_planes(len: usize) -> Planes {
    let mut control = backend(BIG_POOL);
    control
        .prefill_sequence(CONTROL, &tokens(0, len))
        .expect("control prefill");
    kv_planes(&control, CONTROL)
}

/// Independent oracle for how many new blocks the append needs. It does not
/// use the backend's helper: blocks after minus blocks before, plus one
/// copy-on-write block when the table's partly filled tail is shared.
fn oracle_new_blocks(prompt: usize, appended: usize, shared_partial_tail: bool) -> usize {
    let before = prompt.div_ceil(BLOCK_SIZE);
    let after = (prompt + appended).div_ceil(BLOCK_SIZE);
    let cow =
        usize::from(shared_partial_tail && !prompt.is_multiple_of(BLOCK_SIZE) && appended > 0);
    after - before + cow
}

/// Allocates `FILLER` so that exactly `target` blocks stay free.
fn fill_pool_to(be: &NativeTransformerBackend, target: usize) {
    let free = free_blocks(be);
    assert!(
        free >= target,
        "setup: only {free} blocks free, cannot leave {target}"
    );
    if free > target {
        let filler = vec![1u32; (free - target) * BLOCK_SIZE];
        be.kv_manager
            .as_ref()
            .expect("kv manager")
            .write()
            .allocate_sequence(FILLER, &filler)
            .expect("filler allocation");
    }
    assert_eq!(free_blocks(be), target, "setup: free blocks after filler");
}

/// Backend with a pool of `total_blocks`, the prompt prefilled for `SEQ`, the
/// tail optionally shared with `WITNESS`, and (when `free_target` is given)
/// the pool filled by `FILLER` until exactly `free_target` blocks are free.
fn setup(
    total_blocks: usize,
    shared_tail: bool,
    free_target: Option<usize>,
) -> NativeTransformerBackend {
    let mut be = backend(total_blocks);
    be.prefill_sequence(SEQ, &tokens(0, PROMPT_LEN))
        .expect("prompt prefill");
    if shared_tail {
        be.kv_manager
            .as_ref()
            .expect("kv manager")
            .write()
            .fork_sequence(SEQ, WITNESS)
            .expect("witness fork");
    }
    if let Some(target) = free_target {
        fill_pool_to(&be, target);
    }
    be
}

/// The append under test: a second prefill chunk of `len` tokens for `SEQ`,
/// at positions `PROMPT_LEN..PROMPT_LEN + len`.
fn append(be: &mut NativeTransformerBackend, len: usize) -> Result<Vec<f32>, String> {
    be.prefill_sequence(SEQ, &tokens(PROMPT_LEN, len))
}

struct Check {
    name: &'static str,
    marker: &'static str,
    pass: bool,
    applicable: bool,
    message: String,
}

fn check_loud() -> Check {
    let mut notes = Vec::new();
    let mut silent = false;
    let mut bad_error = false;
    for sc in scenarios() {
        let need = sc.need();
        let mut be = setup(SMALL_POOL, sc.shared_tail, Some(need - 1));
        match append(&mut be, sc.len) {
            Ok(_) => {
                silent = true;
                notes.push(format!(
                    "{}: append returned Ok with {} free blocks but {need} needed",
                    sc.label(),
                    need - 1
                ));
            }
            Err(e) => {
                let needs = format!("needs {need} new KV blocks");
                let only = format!("only {} are free", need - 1);
                if e.starts_with(PREFIX) && e.contains(&needs) && e.contains(&only) {
                    notes.push(format!("{}: loud Err: {e}", sc.label()));
                } else {
                    bad_error = true;
                    notes.push(format!(
                        "{}: Err must start with {PREFIX:?} and contain {needs:?} and \
                         {only:?}, got: {e}",
                        sc.label()
                    ));
                }
            }
        }
    }
    Check {
        name: "x_prefill_exhaustion_is_loud",
        marker: if silent { M_SILENT } else { M_PREFIX },
        pass: !silent && !bad_error,
        applicable: true,
        message: notes.join("; "),
    }
}

fn check_atomic() -> Check {
    let mut notes = Vec::new();
    let mut applied = false;
    let mut pass = true;
    for sc in scenarios() {
        let need = sc.need();
        let mut be = setup(SMALL_POOL, sc.shared_tail, Some(need - 1));
        let before = snap(&be, SEQ);
        match append(&mut be, sc.len) {
            Ok(_) => notes.push(format!(
                "{}: append did not fail, atomicity not applicable",
                sc.label()
            )),
            Err(_) => {
                applied = true;
                let diff = snap_diff(&before, &snap(&be, SEQ));
                if diff.is_empty() {
                    notes.push(format!("{}: state unchanged", sc.label()));
                } else {
                    pass = false;
                    notes.push(format!(
                        "{}: failed append left partial state: {}",
                        sc.label(),
                        diff.join(", ")
                    ));
                }
            }
        }
    }
    Check {
        name: "x_prefill_exhaustion_is_atomic",
        marker: M_PARTIAL,
        pass,
        applicable: applied,
        message: notes.join("; "),
    }
}

/// Appends `sc.len` tokens on `be` (pool large enough) and compares the
/// result with the one-shot control. Returns notes and whether everything
/// matched exactly.
fn append_matches_control(
    be: &mut NativeTransformerBackend,
    sc: Scenario,
    stage: &str,
) -> (Vec<String>, bool) {
    let label = format!("{} {stage}", sc.label());
    if let Err(e) = append(be, sc.len) {
        return (vec![format!("{label}: append returned Err: {e}")], false);
    }
    let mut notes = Vec::new();
    let mut ok = true;
    let full = PROMPT_LEN + sc.len;
    let (_, total) = table_of(be, SEQ);
    if total != full {
        ok = false;
        notes.push(format!(
            "{label}: table holds {total} tokens, expected {full}"
        ));
    }
    if be.sequences[&SEQ].tokens != tokens(0, full) {
        ok = false;
        notes.push(format!(
            "{label}: backend sequence tokens differ from the stream"
        ));
    }
    let got = kv_planes(be, SEQ);
    let want = control_planes(full);
    if planes_equal(&got, &want) {
        notes.push(format!("{label}: K/V equals the one-shot control exactly"));
    } else {
        ok = false;
        notes.push(format!(
            "{label}: pool K/V differs from the one-shot control (max abs diff {})",
            planes_max_abs_diff(&got, &want)
        ));
    }
    (notes, ok)
}

/// The witness shares the old blocks and must be untouched by the append: its
/// K/V equals a prompt-only control.
fn witness_untouched(be: &NativeTransformerBackend) -> bool {
    planes_equal(&kv_planes(be, WITNESS), &control_planes(PROMPT_LEN))
}

fn check_clean_path() -> Check {
    let mut notes = Vec::new();
    let mut pass = true;
    for sc in scenarios() {
        let mut be = setup(BIG_POOL, sc.shared_tail, None);
        let (n, ok) = append_matches_control(&mut be, sc, "clean");
        notes.extend(n);
        pass &= ok;
        if sc.shared_tail {
            if witness_untouched(&be) {
                notes.push(format!("{}: witness K/V untouched", sc.label()));
            } else {
                pass = false;
                notes.push(format!("{}: witness K/V changed by the append", sc.label()));
            }
        }
    }
    Check {
        name: "x_prefill_clean_path_unchanged",
        marker: M_CLEAN,
        pass,
        applicable: true,
        message: notes.join("; "),
    }
}

/// Exactly as many free blocks as the append needs: it must succeed, store
/// the right K/V, and drain the pool to zero.
fn check_exact_fit() -> Check {
    let mut notes = Vec::new();
    let mut pass = true;
    for sc in scenarios() {
        let need = sc.need();
        let mut be = setup(SMALL_POOL, sc.shared_tail, Some(need));
        let (n, ok) = append_matches_control(&mut be, sc, "exact fit");
        notes.extend(n);
        pass &= ok;
        let free_after = free_blocks(&be);
        if free_after != 0 {
            pass = false;
            notes.push(format!(
                "{} exact fit: {free_after} blocks free after the append, expected 0",
                sc.label()
            ));
        }
    }
    Check {
        name: "x_exact_fit_ok",
        marker: M_BOUNDARY,
        pass,
        applicable: true,
        message: notes.join("; "),
    }
}

/// Single-token path (`forward_token_impl_paged`, reached through
/// `decode_branch_step`): a branch whose shared tail block is full needs a new
/// block, and the pool has none.
fn decode_single_token_leg() -> (bool, String) {
    let mut be = backend(SMALL_POOL);
    let ctx = be
        .create_context(&tokens(0, 2 * BLOCK_SIZE))
        .expect("context");
    let branch = be.fork_context(ctx).expect("branch");
    fill_pool_to(&be, 0);
    let before = table_of(&be, branch.0);
    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        be.decode_branch_step(branch)
    }));
    match outcome {
        Ok(Ok((tok, _))) => (
            false,
            format!("decode_branch_step returned Ok (token {tok}) with no free block"),
        ),
        Ok(Err(e)) if e.starts_with(PREFIX) => {
            let after = table_of(&be, branch.0);
            if after == before && free_blocks(&be) == 0 {
                (true, format!("single-token path loud Err: {e}"))
            } else {
                (
                    false,
                    format!("single-token Err left partial state: {before:?} -> {after:?}"),
                )
            }
        }
        Ok(Err(e)) => (
            false,
            format!("single-token Err lacks the prefix {PREFIX:?}: {e}"),
        ),
        Err(payload) => {
            let msg = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string());
            (
                false,
                format!("decode_branch_step panicked instead of returning a loud Err: {msg}"),
            )
        }
    }
}

/// Batched path (`execute_step` decode request, `forward_decode_batch`): the
/// slot is reserved before compute and a full pool already ends the sequence
/// as `Preempted`; it must never emit a token.
async fn decode_batched_leg() -> (bool, String) {
    let mut be = backend(SMALL_POOL);
    be.prefill_sequence(SEQ, &tokens(0, 2 * BLOCK_SIZE))
        .expect("prefill");
    fill_pool_to(&be, 0);
    let before = table_of(&be, SEQ);
    let batch = ScheduledBatch {
        prefill_requests: Vec::new(),
        decode_requests: vec![SEQ],
        block_tables: HashMap::new(),
        step_id: 0,
    };
    match be.execute_step(&batch).await {
        Ok((outputs, _metrics)) => {
            let token = outputs
                .iter()
                .any(|o| matches!(o, DecodeOutput::Token { request_id, .. } if *request_id == SEQ));
            let preempted = outputs.iter().any(|o| {
                matches!(
                    o,
                    DecodeOutput::Finished { request_id, reason: FinishReason::Preempted, .. }
                        if *request_id == SEQ
                )
            });
            let unchanged = table_of(&be, SEQ) == before;
            (
                !token && preempted && unchanged,
                format!(
                    "batched decode outputs: token_emitted={token}, preempted={preempted}, \
                     table_unchanged={unchanged}"
                ),
            )
        }
        Err(e) if e.starts_with(PREFIX) => (true, format!("batched decode loud Err: {e}")),
        Err(e) => (false, format!("batched decode Err lacks the prefix: {e}")),
    }
}

async fn check_decode() -> Check {
    let (single_ok, single_msg) = decode_single_token_leg();
    let (batched_ok, batched_msg) = decode_batched_leg().await;
    Check {
        name: "x_decode_exhaustion_is_loud",
        marker: M_SILENT,
        pass: single_ok && batched_ok,
        applicable: true,
        message: format!("{single_msg}; {batched_msg}"),
    }
}

/// After a failed append, freeing the other sequence and repeating the very
/// same append must succeed with the exact K/V of the one-shot control.
fn check_recovers() -> Check {
    let mut notes = Vec::new();
    let mut applied = false;
    let mut pass = true;
    for sc in scenarios() {
        let need = sc.need();
        let mut be = setup(SMALL_POOL, sc.shared_tail, Some(need - 1));
        if append(&mut be, sc.len).is_ok() {
            notes.push(format!(
                "{}: first append did not fail, retry not applicable",
                sc.label()
            ));
            continue;
        }
        applied = true;
        be.kv_manager
            .as_ref()
            .expect("kv manager")
            .write()
            .free_sequence(FILLER)
            .expect("free filler");
        let (n, ok) = append_matches_control(&mut be, sc, "retry");
        notes.extend(n);
        pass &= ok;
        if sc.shared_tail && !witness_untouched(&be) {
            pass = false;
            notes.push(format!(
                "{}: witness K/V changed across the failed and retried append",
                sc.label()
            ));
        }
    }
    Check {
        name: "x_recovers_after_free",
        marker: M_RETRY,
        pass,
        applicable: applied,
        message: notes.join("; "),
    }
}

fn write_receipt(checks: &[Check], finished: bool) {
    let Ok(path) = std::env::var("AIEN_I25_RECEIPT") else {
        return;
    };
    let all_pass = finished && checks.iter().all(|c| c.pass);
    let receipt = serde_json::json!({
        "test": "prefill_pool_exhaustion",
        "block_size": BLOCK_SIZE,
        "prompt_len": PROMPT_LEN,
        "append_lens": APPEND_LENS,
        "small_pool_blocks": SMALL_POOL,
        "finished": finished,
        "pass": all_pass,
        "checks": checks
            .iter()
            .map(|c| serde_json::json!({
                "name": c.name,
                "marker": c.marker,
                "pass": c.pass,
                "applicable": c.applicable,
                "message": c.message,
            }))
            .collect::<Vec<_>>(),
    });
    if let Some(parent) = std::path::Path::new(&path).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    std::fs::write(
        &path,
        serde_json::to_string_pretty(&receipt).expect("receipt json"),
    )
    .expect("write receipt");
}

#[tokio::test]
async fn prefill_pool_exhaustion_is_loud_atomic_and_retryable() {
    // A not-finished receipt exists from the start so a crash still leaves
    // evidence (pass is false until every check ran).
    write_receipt(&[], false);

    let checks = [
        check_loud(),
        check_atomic(),
        check_clean_path(),
        check_exact_fit(),
        check_decode().await,
        check_recovers(),
    ];
    write_receipt(&checks, true);

    // Print every check's outcome before any assert fires, so every marker
    // that applies is in the log.
    for c in &checks {
        if c.pass {
            println!(
                "PREFILL_I25 check {} PASS (applicable={}): {}",
                c.name, c.applicable, c.message
            );
        } else {
            println!(
                "PREFILL_I25 check {} FAIL: {}: {}",
                c.name, c.marker, c.message
            );
        }
    }
    for c in &checks {
        assert!(c.pass, "{}: {}", c.marker, c.message);
    }
}
