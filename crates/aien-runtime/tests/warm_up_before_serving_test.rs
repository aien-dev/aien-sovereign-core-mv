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

//! T4 fix cut, part (a): the declared warm-up (ACCEPTANCE-v4 Section 2(2),
//! `server.rs` `run_warm_up`) runs before the daemon serves anything, so the
//! skill budget B (started inside `propose_task_checked`, compose/bridge.rs) never pays
//! the first-use cost, and the warm-up does not change a later turn's tokens.
//!
//! 1. `warm_up_runs_before_a_waiting_request_is_served` (always runs, mock
//!    backend): a client that connects while the warm-up is still running is
//!    served only after the warm-up's step finished, and the warm-up prompt is
//!    the first prefill the backend sees.
//! 2. `warm_up_does_not_change_the_served_reply` (always runs, mock backend):
//!    the same turn gives the same reply with and without the warm-up.
//! 3. `cpu_reference_tokens_unchanged_by_warm_up` (real model, CPU reference
//!    backend, only when `AIEN_LLAMA3_DIR` is set): the greedy tokens of the T4
//!    compose prompt are identical on a fresh runtime and on one that ran the
//!    warm-up turn first (shared KV reuse after the warm-up leaks nothing).
use aien_inference_abi::{
    AienInferenceBackend, ChatTokenizer, DecodeOutput, MockInferenceBackend, ModelConfig,
    SamplingParams, ScheduledBatch, StepMetrics,
};
use aien_kv_cache::create_shared_kv_manager;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::{format_chat, ChatTurn, WARM_UP_TEXT};
use aien_scheduler::SchedulerConfig;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Simulated first-use cost (weight upload, kernel generation) paid by the
/// first step the backend runs.
const FIRST_STEP_COST: Duration = Duration::from_millis(600);

/// Word-level tokenizer: the mock emits ids 100 and 101, decoded as x100/x101.
fn toy_tokenizer() -> ChatTokenizer {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,
        "decoder":null,"model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<unk>":0,"<s>":1,"</s>":2,"Warm":3,"up":4,"turn":5,"hello":6,"x100":100,"x101":101}}}"#;
    ChatTokenizer::from_bytes(json.as_bytes()).expect("toy tokenizer")
}

#[derive(Debug, Clone)]
struct StepRecord {
    prefill: Vec<(u64, usize)>,
    decode: Vec<u64>,
    started: Instant,
    ended: Instant,
}

/// The mock backend, recording every step; the first step costs FIRST_STEP_COST.
struct RecordingBackend {
    inner: MockInferenceBackend,
    log: Arc<Mutex<Vec<StepRecord>>>,
}

#[async_trait::async_trait]
impl AienInferenceBackend for RecordingBackend {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String> {
        self.inner.load_model(config).await
    }

    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let started = Instant::now();
        if self.log.lock().unwrap().is_empty() {
            std::thread::sleep(FIRST_STEP_COST);
        }
        let out = self.inner.execute_step(batch).await;
        self.log.lock().unwrap().push(StepRecord {
            prefill: batch
                .prefill_requests
                .iter()
                .map(|r| (r.request_id, r.prompt_tokens.len()))
                .collect(),
            decode: batch.decode_requests.clone(),
            started,
            ended: Instant::now(),
        });
        out
    }
}

/// This binary's own runtime state dir (#299): never the machine-wide
/// `/tmp/aien-runtime-processed-ops.json` other daemons and tests write.
fn private_state_dir() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        let dir = std::env::temp_dir().join(format!("aien-warm-up-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::env::set_var("AIEN_RUNTIME_STATE_DIR", &dir);
        // sc#328: the daemon refuses to start without the (default) desk key.
        std::env::set_var("AIEN_COMPOSE_DIR", dir.join("compose"));
        let key = aien_runtime::approved_auth::desk_key_path(&dir.join("compose"));
        if !key.exists() {
            aien_runtime::approved_auth::DeskKey::create(&key).unwrap();
        }
    });
}

fn spine() -> AienRuntimeSpine {
    private_state_dir();
    // Small prefill chunks: the warm-up prompt spans several steps, so a
    // request accepted during the warm-up would interleave with its chunks.
    let cfg = SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 4096,
        prefill_chunk_size: 8,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 4096,
    };
    AienRuntimeSpine::new(64, cfg, create_shared_kv_manager(1024, 16))
}

fn user(content: &str) -> Vec<ChatTurn> {
    vec![ChatTurn {
        role: "user".into(),
        content: content.into(),
    }]
}

fn prompt_len(content: &str) -> usize {
    let tok = toy_tokenizer();
    tok.encode(&format_chat(tok.template(), &user(content)).expect("chat"))
        .expect("encode")
        .len()
}

/// Start a server (warm-up on or off), send one turn as soon as the socket
/// exists, and return (reply, step log, instant the client connected).
async fn serve_one_turn(warm_up: bool, tag: &str) -> (String, Vec<StepRecord>, Instant) {
    let tmp = tempfile::tempdir().unwrap();
    let socket = tmp.path().join(format!("{tag}.sock"));
    let log = Arc::new(Mutex::new(Vec::new()));
    let server = AienRuntimeServer::new(spine(), &socket);
    server.set_tokenizer(toy_tokenizer());
    if warm_up {
        server.enable_warm_up();
    }
    let backend = RecordingBackend {
        inner: MockInferenceBackend::new(1),
        log: log.clone(),
    };
    let handle = tokio::spawn(async move { server.run(backend).await });
    // The socket is bound before the warm-up runs; connect as soon as it exists.
    let t_wait = Instant::now();
    while !socket.exists() {
        assert!(
            t_wait.elapsed() < Duration::from_secs(10),
            "socket never bound"
        );
        tokio::time::sleep(Duration::from_millis(2)).await;
    }
    let connected = Instant::now();
    let client = AienRuntimeClient::new(&socket);
    let reply = client
        .stream_turn(user("hello hello hello"), 4, 0.0)
        .await
        .expect("turn served");
    let _ = client.shutdown().await;
    let _ = tokio::time::timeout(Duration::from_secs(5), handle).await;
    let records = log.lock().unwrap().clone();
    (reply, records, connected)
}

fn touches(r: &StepRecord, pred: impl Fn(u64) -> bool) -> bool {
    r.prefill.iter().any(|&(id, _)| pred(id)) || r.decode.iter().any(|&id| pred(id))
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn warm_up_runs_before_a_waiting_request_is_served() {
    let (_reply, log, connected) = serve_one_turn(true, "warm").await;
    let warm_len = prompt_len(WARM_UP_TEXT);
    let turn_len = prompt_len("hello hello hello");
    assert!(warm_len > 8, "the warm-up must span several prefill chunks");

    let first = log.first().expect("at least one step");
    assert_eq!(
        first.prefill.len(),
        1,
        "warm-up prefill runs alone: {first:?}"
    );
    let warm_id = first.prefill[0].0;
    let warm_steps: Vec<&StepRecord> = log
        .iter()
        .filter(|r| touches(r, |id| id == warm_id))
        .collect();
    let turn_steps: Vec<&StepRecord> = log
        .iter()
        .filter(|r| touches(r, |id| id != warm_id))
        .collect();
    let warm_prefilled: usize = warm_steps
        .iter()
        .flat_map(|r| {
            r.prefill
                .iter()
                .filter(|&&(id, _)| id == warm_id)
                .map(|&(_, n)| n)
        })
        .sum();
    assert_eq!(
        warm_prefilled, warm_len,
        "the first sequence is the whole warm-up prompt"
    );
    assert!(
        warm_steps.len() > 1,
        "warm-up spans several steps: {}",
        warm_steps.len()
    );
    let warm_end = warm_steps.last().unwrap().ended;
    // Precondition: the client was already waiting while the warm-up ran.
    assert!(
        connected < warm_end,
        "client connected after the warm-up ended; the test proves nothing"
    );
    let turn_first = turn_steps
        .first()
        .expect("the client's turn reached the backend");
    assert!(
        turn_first.started >= warm_end,
        "the served turn reached the backend before the warm-up finished"
    );
    let turn_prefilled: usize = turn_steps
        .iter()
        .flat_map(|r| {
            r.prefill
                .iter()
                .filter(|&&(id, _)| id != warm_id)
                .map(|&(_, n)| n)
        })
        .sum();
    assert_eq!(turn_prefilled, turn_len, "turn prompt");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn warm_up_does_not_change_the_served_reply() {
    let (with_warm, log_w, _) = serve_one_turn(true, "with").await;
    let (without, log_n, _) = serve_one_turn(false, "without").await;
    assert_eq!(with_warm, without, "reply changed by the warm-up");
    assert!(!with_warm.is_empty());
    // With the warm-up, the first-use cost lands on a warm-up step; without
    // it, the served turn's first step pays it (what the warm-up avoids).
    let warm_id = log_w[0].prefill[0].0;
    assert!(log_w[0].ended - log_w[0].started >= FIRST_STEP_COST);
    let turn_w: Vec<&StepRecord> = log_w
        .iter()
        .filter(|r| touches(r, |id| id != warm_id))
        .collect();
    assert!(!turn_w.is_empty());
    assert!(turn_w.iter().all(|r| r.ended - r.started < FIRST_STEP_COST));
    assert!(log_n[0].ended - log_n[0].started >= FIRST_STEP_COST);
}

/// One greedy turn on the daemon's path (as `examples/np1_reference.rs`).
async fn cpu_turn(
    spine: &mut AienRuntimeSpine,
    backend: &mut aien_inference_abi::NativeTransformerBackend,
    ids: &[u32],
    max_tokens: usize,
    stop: &[u32],
) -> Vec<u32> {
    use aien_scheduler::{ChannelCompletionSink, CompletionEvent};
    let (sink, mut ev) = ChannelCompletionSink::channel();
    let sid = spine.register_completion_sink(Arc::new(sink));
    let sampling = SamplingParams {
        temperature: 0.0,
        top_p: 0.95,
        max_tokens,
        stop_token_ids: stop.to_vec(),
    };
    spine
        .submit_work(Arc::from(ids), sampling, 2, Some(sid))
        .expect("submit");
    let mut out = Vec::new();
    loop {
        if spine.scheduler.running_count() > 0 || spine.scheduler.waiting_count() > 0 {
            spine.step(backend).await.expect("step");
        }
        while let Ok(e) = ev.try_recv() {
            match e {
                CompletionEvent::Token { token, .. } => out.push(token),
                CompletionEvent::Finished { .. } => return out,
                CompletionEvent::Error { message, .. } => panic!("turn error: {message}"),
            }
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn cpu_reference_tokens_unchanged_by_warm_up() {
    let Ok(dir) = std::env::var("AIEN_LLAMA3_DIR") else {
        eprintln!("AIEN_LLAMA3_DIR not set: CPU reference warm-up token check skipped");
        return;
    };
    let dir = std::path::PathBuf::from(dir);
    let tok = ChatTokenizer::from_model_dir(&dir, None).expect("tokenizer");
    let stop = tok.stop_token_ids().to_vec();
    let goal = "Create the file docs/GUIDE.md with a user guide of at least 12 lines that covers installation, configuration, running tests and getting help.";
    let text = format!(
        "{}{}",
        format_chat(
            tok.template(),
            &user(&aien_runtime::compose::rederive::proposal_prompt(goal))
        )
        .expect("chat"),
        aien_runtime::compose::rederive::COMPOSE_ASSISTANT_PREFIX
    );
    let ids = tok.encode(&text).expect("encode");
    let warm = tok
        .encode(&format_chat(tok.template(), &user(WARM_UP_TEXT)).expect("chat"))
        .expect("encode");
    const N: usize = 8;

    let mut runs = Vec::new();
    for with_warm_up in [false, true] {
        let config = aien_inference_abi::load_model_config(&dir).expect("config.json");
        let weights = aien_inference_abi::TransformerWeights::load_from_safetensors(&dir, &config)
            .expect("weights");
        let tb: Arc<dyn aien_inference_abi::TensorBackend> =
            Arc::new(aien_inference_abi::ReferenceCpuBackend::new());
        // The daemon's scheduler config and KV sizing (aien-cli run_daemon_server).
        let cfg = SchedulerConfig {
            max_batch_size: 256,
            max_batch_tokens: 16384,
            max_prefill_tokens: 8192,
            prefill_chunk_size: 128,
            chunk_prefill: true,
            watermark_blocks: 64,
        };
        let sizing = aien_runtime::shared_kv::SharedKvSizing {
            arena_capacity: 4096,
            total_blocks: 8192,
        };
        let (mut spine, mut backend) =
            aien_runtime::shared_kv::build_shared_kv_runtime(weights, tb, cfg, sizing)
                .expect("shared KV");
        if with_warm_up {
            let w = cpu_turn(&mut spine, &mut backend, &warm, 1, &stop).await;
            assert_eq!(w.len(), 1, "warm-up emits exactly one token");
        }
        let t0 = Instant::now();
        let out = cpu_turn(&mut spine, &mut backend, &ids, N, &stop).await;
        eprintln!(
            "CPU reference turn (warm_up={with_warm_up}): {} prompt tokens, {} generated, {} ms, ids {:?}",
            ids.len(),
            out.len(),
            t0.elapsed().as_millis(),
            out
        );
        runs.push(out);
    }
    assert!(!runs[0].is_empty());
    assert_eq!(runs[0], runs[1], "the warm-up changed the turn's tokens");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn plain_model_warm_up_is_skipped_not_failed() {
    // A plain (base) model has no chat template; the chat warm-up is skipped.
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../aien-inference-abi/fixtures/openwaldo-byte");
    let plain = ChatTokenizer::from_model_dir(&dir, None).expect("plain fixture loads");
    let tmp = tempfile::tempdir().unwrap();
    let server = AienRuntimeServer::new(spine(), tmp.path().join("plain.sock"));
    server.set_tokenizer(plain);
    // No step loop is running: an attempted warm-up would hang or fail, a skip returns at once.
    let line = tokio::time::timeout(Duration::from_secs(5), server.run_warm_up())
        .await
        .expect("a skipped warm-up returns immediately");
    assert_eq!(line, "Warm-up: skipped (plain model, no chat template)");
    assert!(!line.contains("FAILED"), "{line}");
}
