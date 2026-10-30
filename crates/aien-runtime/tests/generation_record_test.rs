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

//! The daemon's generation record (docs/DAEMON_GENERATION_RECORD.md): evidence
//! that this daemon generated this text from the model files it loaded.
//! Socket-level, mock backend, toy tokenizer. Every test needs the linked
//! composition archive and is IGNORED in a stub build, never passed (the
//! no-record cases that hold in a stub build are unit tests in server.rs).
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_inference_abi::{
    AienInferenceBackend, ChatTokenizer, DecodeObservation, DecodeOutput, MockInferenceBackend,
    ModelConfig, ScheduledBatch, StepMetrics,
};
use aien_kv_cache::create_shared_kv_manager;
use aien_omega_compose::hex;
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::compose::{token_ids_sha256, ComposeBridge, ComposeProposer};
use aien_runtime::control::{ControlCommand, ControlResponse};
use aien_runtime::generation::ModelIdentity;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_runtime::{format_chat, ChatTurn};
use aien_scheduler::SchedulerConfig;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::sync::Arc;
use std::time::Duration;

/// The composition engine holds one home at a time per process.
static ONE_HOME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

fn toy_tokenizer() -> ChatTokenizer {
    toy_tokenizer_with(r#""x100":100,"x101":101"#)
}

/// The toy vocabulary with the spelling of the two mock tokens chosen by the test.
fn toy_tokenizer_with(mock_tokens: &str) -> ChatTokenizer {
    let json = r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],
        "normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,
        "decoder":null,"model":{"type":"WordLevel","unk_token":"<unk>","vocab":{
        "<unk>":0,"<s>":1,"</s>":2,"hello":6,MOCK}}}"#
        .replace("MOCK", mock_tokens);
    ChatTokenizer::from_bytes(json.as_bytes()).expect("toy tokenizer")
}

fn identity() -> ModelIdentity {
    ModelIdentity {
        model_sha256: "ab".repeat(32),
        model_digest_kind: "index+shards".into(),
        model_path: "/models/toy/model.safetensors".into(),
        tokenizer_sha256: "cd".repeat(32),
        tokenizer_path: "/models/toy/tokenizer.json".into(),
    }
}

fn user(content: &str) -> Vec<ChatTurn> {
    vec![ChatTurn {
        role: "user".into(),
        content: content.into(),
    }]
}

struct Daemon {
    client: AienRuntimeClient,
    handle: tokio::task::JoinHandle<Result<(), String>>,
    _tmp: tempfile::TempDir,
}

async fn start(identity: Option<ModelIdentity>) -> Daemon {
    start_with(identity, toy_tokenizer()).await
}

async fn start_with(identity: Option<ModelIdentity>, tokenizer: ChatTokenizer) -> Daemon {
    start_on(identity, tokenizer, MockInferenceBackend::new(1)).await
}

// Success-path evidence fixtures must actually stop, not merely reach a cap.
struct EosBackend<B>(B);
#[async_trait::async_trait]
impl<B: AienInferenceBackend + Send> AienInferenceBackend for EosBackend<B> {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String> {
        self.0.load_model(config).await
    }
    async fn execute_step(&mut self, batch: &ScheduledBatch)
        -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        let (mut outputs, metrics) = self.0.execute_step(batch).await?;
        for output in &mut outputs {
            if let DecodeOutput::Token { token_id, .. } = output {
                if *token_id == 101 { *token_id = 2; }
            }
        }
        Ok((outputs, metrics))
    }
    fn take_decode_observation(&mut self, request_id: u64) -> Option<DecodeObservation> {
        self.0.take_decode_observation(request_id)
    }
}

async fn start_on<B: AienInferenceBackend + Send + 'static>(
    identity: Option<ModelIdentity>,
    tokenizer: ChatTokenizer,
    backend: B,
) -> Daemon {
    let tmp = tempfile::tempdir().unwrap();
    let socket = tmp.path().join("runtime.sock");
    let cfg = SchedulerConfig {
        max_batch_size: 8,
        max_batch_tokens: 1024,
        prefill_chunk_size: 64,
        watermark_blocks: 4,
        chunk_prefill: true,
        max_prefill_tokens: 1024,
    };
    let spine = AienRuntimeSpine::new(64, cfg, create_shared_kv_manager(256, 16));
    let proposer: ComposeProposer =
        Arc::new(|_p: &str, _l: Duration| Err("not used in this test".to_string()));
    // sc#328: the bridge requires the desk by default; the daemon will not
    // start without its key.
    aien_runtime::approved_auth::DeskKey::create(&aien_runtime::approved_auth::desk_key_path(
        &tmp.path().join("compose"),
    ))
    .unwrap();
    let bridge = Arc::new(ComposeBridge::new(
        tmp.path().join("compose"),
        proposer,
        "test:generation-record",
    ));
    let server = AienRuntimeServer::new(spine, &socket).with_compose_bridge(bridge);
    server.set_tokenizer(tokenizer);
    if let Some(id) = identity {
        server.set_model_identity(id);
    }
    let handle = tokio::spawn(async move { server.run(EosBackend(backend)).await });
    let client = AienRuntimeClient::new(&socket);
    for _ in 0..200 {
        if client.is_alive().await {
            return Daemon {
                client,
                handle,
                _tmp: tmp,
            };
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    panic!("daemon did not come up");
}

impl Daemon {
    async fn turn(&self, content: &str) -> (String, Option<u64>) {
        self.client
            .stream_turn_recorded(user(content), 4, 0.0)
            .await
            .expect("turn")
    }

    async fn records(&self) -> u64 {
        match self.recall(vec![]).await {
            ControlResponse::ComposeRecalled(r) => r.records_total,
            other => panic!("recall: {other:?}"),
        }
    }

    async fn recall(&self, ids: Vec<u64>) -> ControlResponse {
        self.client
            .send_command(ControlCommand::ComposeRecall { ids, prefix: None })
            .await
            .unwrap()
    }

    async fn record(&self, id: u64) -> Value {
        match self.recall(vec![id]).await {
            ControlResponse::ComposeRecalled(r) => {
                assert_eq!(r.cited.len(), 1, "record {id} missing");
                assert!(r.cited[0].verified, "record {id} digest does not verify");
                assert_eq!(r.cited[0].note.as_deref(), Some("effect"));
                serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap()
            }
            other => panic!("recall: {other:?}"),
        }
    }

    async fn note(&self, kind: &str, text: &str) -> ControlResponse {
        self.client
            .send_command(ControlCommand::ComposeNote {
                kind: kind.into(),
                text: text.into(),
                links: vec![],
            })
            .await
            .unwrap()
    }

    async fn stop(self) {
        let _ = self.client.send_command(ControlCommand::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), self.handle).await;
    }
}

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
}

/// The stopping mock emits ordinary x100 (100), then EOS </s> (2).
/// This toy tokenizer does not register special added tokens, so both decode.
fn ids_of(text: &str) -> Vec<u32> {
    text.split_whitespace()
        .map(|w| match w {
            "x100" => 100,
            "</s>" => 2,
            other => panic!("unexpected word {other:?} in {text:?}"),
        })
        .collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn record_carries_the_digests_of_the_turn() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    let d = start(Some(identity())).await;
    let tok = toy_tokenizer();
    let prompt = tok
        .encode(&format_chat(tok.template(), &user("hello hello")).expect("chat"))
        .unwrap();
    let (text, id) = d.turn("hello hello").await;
    assert!(!text.is_empty());
    let id = id.expect("a record id");
    let r = d.record(id).await;
    let want = identity();
    assert_eq!(r["generation"], 1);
    assert_eq!(r["v"], 1);
    assert_eq!(r["model_sha256"], want.model_sha256);
    assert_eq!(r["model_digest_kind"], want.model_digest_kind);
    assert_eq!(r["model_path"], want.model_path);
    assert_eq!(r["tokenizer_sha256"], want.tokenizer_sha256);
    assert_eq!(r["tokenizer_path"], want.tokenizer_path);
    assert_eq!(r["prompt_ids_sha256"], token_ids_sha256(&prompt));
    // The digest of the exact text TurnFinished returned.
    assert_eq!(r["output_text_sha256"], sha(text.as_bytes()));
    let out = ids_of(&text);
    assert_eq!(r["output_token_ids_sha256"], token_ids_sha256(&out));
    assert_eq!(r["output_tokens"], out.len());
    assert!(r["total_tokens"].as_u64().unwrap() > 0);
    assert_eq!(r["finish_reason"], "eos");
    assert_eq!(r["daemon"]["pid"], std::process::id());
    assert!(r["daemon"]["started_unix_ms"].as_str().unwrap().len() > 8);
    // sc#294: the mock backend observes nothing, so the record makes no
    // decoding claim (absent, never a default "greedy").
    assert!(r.get("decoding").is_none(), "{r}");
    // The envelope ids are the client's own, recorded as asserted.
    assert!(r["request_id"].as_u64().unwrap() > 0);
    assert!(r["operation_id"].as_str().unwrap().len() > 3);
    d.stop().await;
}

/// The record hashes the text exactly as returned, whitespace included (a
/// `trim()` before hashing would break the match with what the client holds).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn text_is_hashed_byte_exact_with_surrounding_whitespace() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    // Decoded text of the mock tokens: leading and trailing spaces.
    let d = start_with(
        Some(identity()),
        toy_tokenizer_with(r#""  x100 ":100,"x101  ":101"#),
    )
    .await;
    let (text, id) = d.turn("hello").await;
    assert_ne!(
        text,
        text.trim(),
        "test needs surrounding whitespace: {text:?}"
    );
    let r = d.record(id.expect("a record id")).await;
    assert_eq!(r["output_text_sha256"], sha(text.as_bytes()));
    assert_ne!(r["output_text_sha256"], sha(text.trim().as_bytes()));
    d.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn two_identical_turns_make_two_records() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    let d = start(Some(identity())).await;
    let (t1, a) = d.turn("hello").await;
    let (t2, b) = d.turn("hello").await;
    let (a, b) = (a.expect("first record"), b.expect("second record"));
    assert_ne!(a, b, "one record per turn");
    assert_eq!(t1, t2);
    let (ra, rb) = (d.record(a).await, d.record(b).await);
    assert_eq!(ra["output_text_sha256"], rb["output_text_sha256"]);
    assert_eq!(ra["prompt_ids_sha256"], rb["prompt_ids_sha256"]);
    d.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn no_model_identity_means_no_record_and_the_turn_succeeds() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    let d = start(None).await;
    let before = d.records().await;
    let (text, id) = d.turn("hello").await;
    assert!(!text.is_empty());
    assert_eq!(id, None);
    assert_eq!(d.records().await, before, "nothing was written");
    d.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn a_caller_cannot_write_a_generation_record() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    let d = start(Some(identity())).await;
    let (_, id) = d.turn("hello").await;
    assert!(id.is_some());
    let before = d.records().await;
    let forged = r#"{"generation":1,"v":1,"model_sha256":"00","output_text_sha256":"00"}"#;
    for (kind, text) in [
        ("generation", forged),
        ("generation", "plain text"),
        ("effect", forged),
    ] {
        match d.note(kind, text).await {
            ControlResponse::Error(e) => assert!(e.contains("generation"), "{kind}: {e}"),
            other => panic!("FORGED generation record accepted ({kind}): {other:?}"),
        }
    }
    assert_eq!(d.records().await, before, "ledger count unchanged");
    d.stop().await;
}

#[test]
fn turn_finished_without_a_record_serializes_as_before() {
    // Older clients and servers: the field is absent when there is no record.
    let none = ControlResponse::TurnFinished {
        text: "x".into(),
        total_tokens: 1,
        generation_record: None,
    };
    assert_eq!(
        serde_json::to_string(&none).unwrap(),
        r#"{"TurnFinished":{"text":"x","total_tokens":1}}"#
    );
    let old: ControlResponse =
        serde_json::from_str(r#"{"TurnFinished":{"text":"x","total_tokens":1}}"#).unwrap();
    assert!(matches!(
        old,
        ControlResponse::TurnFinished {
            generation_record: None,
            ..
        }
    ));
    let some = ControlResponse::TurnFinished {
        text: "x".into(),
        total_tokens: 1,
        generation_record: Some(9),
    };
    assert!(serde_json::to_string(&some)
        .unwrap()
        .contains(r#""generation_record":9"#));
}

/// The mock backend, plus a fixed report of how it decoded: stands in for a
/// backend that observes its sampling branch (sc#294).
struct ObservingMock {
    inner: MockInferenceBackend,
    report: DecodeObservation,
}

#[async_trait::async_trait]
impl AienInferenceBackend for ObservingMock {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String> {
        self.inner.load_model(config).await
    }
    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<(Vec<DecodeOutput>, StepMetrics), String> {
        self.inner.execute_step(batch).await
    }
    fn take_decode_observation(&mut self, _request_id: u64) -> Option<DecodeObservation> {
        Some(self.report.clone())
    }
}

/// sc#294: the StreamTurn record states the decoding the backend reported at
/// finish (from `CompletionEvent::Finished`), not the request's temperature:
/// the client asks for 0.0 here and the backend reports sampling, so a record
/// built from the settings would say greedy.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn record_states_the_decoding_the_backend_reported() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    let report = DecodeObservation {
        greedy_tokens: 1,
        sampled_tokens: 3,
        temperature: Some(0.7),
        top_p: Some(0.95),
        seed_request_id: Some(42),
    };
    let backend = ObservingMock {
        inner: MockInferenceBackend::new(1),
        report,
    };
    let d = start_on(Some(identity()), toy_tokenizer(), backend).await;
    let (_, id) = d.turn("hello hello").await;
    let r = d.record(id.expect("a record id")).await;
    assert_eq!(
        r["decoding"],
        serde_json::json!({
            "mode": "mixed",
            "greedy_tokens": 1,
            "sampled_tokens": 3,
            "temperature": 0.7,
            "top_p": 0.95,
            "seed_request_id": 42
        }),
        "{r}"
    );
    d.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn capped_turn_is_refused_but_its_evidence_is_retained() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    let d = start(Some(identity())).await;
    let before = d.records().await;
    let error = d.client.stream_turn_recorded(user("hello"), 1, 0.0)
        .await.expect_err("a token cap is not completion");
    assert!(error.contains("max_tokens"), "{error}");
    assert_eq!(d.records().await, before + 1);
    let ControlResponse::ComposeRecalled(report) = d.recall(vec![]).await else { panic!("recall") };
    let record = report.host.iter().rev().find_map(|record| {
        let value: Value = serde_json::from_str(record.text.as_deref()?).ok()?;
        (value["generation"] == 1).then_some((record, value))
    }).expect("generation evidence");
    assert!(record.0.verified);
    assert_eq!(record.1["finish_reason"], "max_tokens");
    assert_eq!(record.1["output_text_sha256"], sha(b"x100"));
    assert_eq!(record.1["output_token_ids_sha256"], token_ids_sha256(&[100]));
    d.stop().await;
}
