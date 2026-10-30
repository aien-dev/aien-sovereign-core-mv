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

//! Provenance link (arch#162, ALLEN end-to-end demo v1 steps S4 and S8): a
//! commit the model produced through the compose path names the generation
//! record of that proposal and the ALLEN the task ran under.
//!
//! The real compose library, the real daemon server on its socket and the
//! ordinary operator flow: propose (RunComposeTask) -> authorize -> intent ->
//! write -> ack. Only the model is replaced by a fixed proposer that returns
//! the token ids the real inference path returns; the daemon is told the
//! "loaded model" digest with `set_model_identity`, exactly as the CLI does
//! after it loads weights. The live SmolLM2 version of this flow is
//! `crates/aien-cli/tests/provenance_link_live_test.rs` (ignored: it needs a
//! prebuilt binary and the model, like recovery_matrix_test).
//!
//! Everything here reads the ledger through the socket (`ComposeRecall`) and
//! the report as JSON, so the test is the same on a build without the link.
//! Every test needs the linked composition archive and is IGNORED in a stub
//! build, never passed.
#[path = "support/home_guard.rs"]
mod home_guard;

#[path = "../../aien-allen/tests/support/mod.rs"]
mod support;

use aien_allen::{hex as ahex, ENV_ADOPT, ENV_SUBJECT};
use aien_inference_abi::MockInferenceBackend;
use aien_kv_cache::create_shared_kv_manager;
use aien_omega_compose::{hex, Compose, RootKind};
use aien_runtime::approved_auth::{desk_key_path, AuthorizeBinding, DeskKey};
use aien_runtime::client::AienRuntimeClient;
use aien_runtime::compose::{token_ids_sha256, ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::{ControlCommand, ControlResponse, DeskProof};
use aien_runtime::generation::ModelIdentity;
use aien_runtime::server::AienRuntimeServer;
use aien_runtime::spine::AienRuntimeSpine;
use aien_scheduler::SchedulerConfig;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// The composition engine holds one home at a time per process, and the ALLEN
/// variables are process-wide.
static ONE_HOME: tokio::sync::Mutex<()> = tokio::sync::Mutex::const_new(());

const PROPOSAL: &str = "filename: NOTES.md\nkeep every change inside the workspace\n";
const CONTENT: &str = "keep every change inside the workspace\n";
const OUT_IDS: [u32; 3] = [11, 12, 13];
const PROMPT_IDS: [u32; 5] = [1, 2, 3, 4, 5];

fn sha(b: &[u8]) -> String {
    hex(&Sha256::digest(b))
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

/// How the backend chose OUT_IDS: all greedy (sc#294).
fn greedy_observation() -> aien_inference_abi::DecodeObservation {
    aien_inference_abi::DecodeObservation {
        greedy_tokens: OUT_IDS.len() as u64,
        ..Default::default()
    }
}

/// The tensor backend and op counters the scheduler reports (sc#337).
fn cpu_ops() -> aien_inference_abi::OpEvidence {
    aien_inference_abi::OpEvidence {
        backend: "reference-cpu".into(),
        native_fallbacks: 0,
        reference_runs: 0,
        report: "OP_REPORT native=[] reference=[] native_fallbacks=[] reference_runs=[]".into(),
        gpu_executions: None,
    }
}

/// The reply the real inference path gives: text, generated ids, prompt ids hash.
fn proposer() -> ComposeProposer {
    Arc::new(|_p: &str, _l: Duration| {
        Ok(Generation {
            text: PROPOSAL.to_string(),
            tokens: OUT_IDS.len(),
            finish_reason: Some("eos".into()),
            token_ids: Some(OUT_IDS.to_vec()),
            prompt_tokens: Some(PROMPT_IDS.len()),
            prompt_ids_sha256: Some(token_ids_sha256(&PROMPT_IDS)),
            // sc#294: what the scheduler reports for a greedy run of these ids.
            decoding: Some(greedy_observation()),
            // sc#337: what the scheduler reports for a production CPU run.
            ops: Some(cpu_ops()),
        })
    })
}

struct Daemon {
    client: AienRuntimeClient,
    bridge: Arc<ComposeBridge>,
    handle: tokio::task::JoinHandle<Result<(), String>>,
    _tmp: tempfile::TempDir,
    ws: PathBuf,
}

impl Daemon {
    /// `home`: an existing compose home (ALLEN tests prepare one), else a fresh one.
    async fn start(
        model: Option<ModelIdentity>,
        home: Option<(tempfile::TempDir, PathBuf)>,
    ) -> Self {
        let (tmp, home) = home.unwrap_or_else(|| {
            let t = tempfile::tempdir().unwrap();
            let h = t.path().join("compose");
            (t, h)
        });
        let ws = std::fs::canonicalize({
            let w = tmp.path().join("ws");
            std::fs::create_dir_all(&w).unwrap();
            w
        })
        .unwrap();
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
        // sc#328: the desk is required (the default); every authorize here is
        // signed with this home's desk key, as `aien compose authorize --desk 1`.
        let key = desk_key_path(&home);
        if !key.exists() {
            DeskKey::create(&key).unwrap();
        }
        let bridge = Arc::new(ComposeBridge::new(home, proposer(), "test:provenance-link"));
        let server = AienRuntimeServer::new(spine, &socket).with_compose_bridge(bridge.clone());
        if let Some(m) = model {
            server.set_model_identity(m);
        }
        let handle = tokio::spawn(async move { server.run(MockInferenceBackend::new(1)).await });
        let client = AienRuntimeClient::new(&socket);
        for _ in 0..200 {
            if client.is_alive().await {
                return Self {
                    client,
                    bridge,
                    handle,
                    _tmp: tmp,
                    ws,
                };
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        panic!("daemon did not come up");
    }

    /// The approval desk's proof for an authorize of commit `cx` (path and
    /// content as the daemon's commit record names them).
    fn desk(
        &self,
        cx: u64,
        proposal_sha256: &str,
        path: &str,
        content_sha256: &str,
    ) -> Option<DeskProof> {
        let key = DeskKey::load(&desk_key_path(self.bridge.dir())).unwrap();
        let nonce = format!("prov-{cx}");
        let mac = key.sign_authorize(&AuthorizeBinding {
            cx_promotion: cx,
            proposal_sha256: proposal_sha256.into(),
            path: path.into(),
            content_sha256: content_sha256.into(),
            workspace: self.ws.display().to_string(),
            approver: "drake".into(),
            constraints: vec![],
            nonce: nonce.clone(),
            desk_key_id: key.id().into(),
        });
        Some(DeskProof { nonce, mac })
    }

    async fn send(&self, c: ControlCommand) -> ControlResponse {
        self.client.send_command(c).await.unwrap()
    }

    /// Text of record `id` as JSON, digest-verified.
    async fn record(&self, id: u64) -> Value {
        match self
            .send(ControlCommand::ComposeRecall {
                ids: vec![id],
                prefix: None,
            })
            .await
        {
            ControlResponse::ComposeRecalled(r) => {
                assert_eq!(r.cited.len(), 1, "record {id} missing");
                assert!(r.cited[0].verified, "record {id} does not verify");
                serde_json::from_str(r.cited[0].text.as_deref().unwrap()).unwrap()
            }
            other => panic!("recall: {other:?}"),
        }
    }

    async fn host_records(&self) -> usize {
        match self
            .send(ControlCommand::ComposeRecall {
                ids: vec![],
                prefix: None,
            })
            .await
        {
            ControlResponse::ComposeRecalled(r) => r.host.len(),
            other => panic!("recall: {other:?}"),
        }
    }

    async fn stop(self) {
        let _ = self.client.send_command(ControlCommand::Shutdown).await;
        let _ = tokio::time::timeout(Duration::from_secs(5), self.handle).await;
    }
}

/// Every record id of one ordinary flow, and its report as JSON.
struct Flow {
    report: Value,
    commit: u64,
    grant: u64,
    intent: u64,
    ack: u64,
}

fn noted(r: ControlResponse, what: &str) -> u64 {
    match r {
        ControlResponse::ComposeNoted(n) => n.id,
        other => panic!("{what}: {other:?}"),
    }
}

/// propose -> authorize -> intent -> write -> ack, through the socket.
async fn ordinary_flow(d: &Daemon) -> Flow {
    let report = match d
        .send(ControlCommand::RunComposeTask {
            context: None,
            goal: "write the note".into(),
            workspace: d.ws.to_str().unwrap().into(),
        })
        .await
    {
        ControlResponse::ComposeTaskResult(r) => serde_json::to_value(&*r).unwrap(),
        other => panic!("run: {other:?}"),
    };
    assert_eq!(report["committed"], true, "{report}");
    let commit = report["compose_commit"].as_u64().expect("commit record");
    let grant = noted(
        d.send(ControlCommand::ComposeAuthorize {
            cx_promotion: report["cx_promotion"].as_u64().unwrap(),
            proposal_sha256: sha(PROPOSAL.as_bytes()),
            workspace: d.ws.display().to_string(),
            approver: "drake".into(),
            constraints: vec![],
            desk_proof: d.desk(
                report["cx_promotion"].as_u64().unwrap(),
                &sha(PROPOSAL.as_bytes()),
                "NOTES.md",
                &sha(CONTENT.as_bytes()),
            ),
            objective_id: None,
        })
        .await,
        "authorize",
    );
    let target = d.ws.join("NOTES.md").display().to_string();
    let (pid, start) = aien_runtime::effects::self_executor();
    let intent = noted(
        d.send(ControlCommand::ComposeEffectIntent {
            authorization: grant,
            proposal_sha256: sha(PROPOSAL.as_bytes()),
            path: "NOTES.md".into(),
            target: target.clone(),
            content_sha256: sha(CONTENT.as_bytes()),
            executor_pid: pid,
            executor_start: start,
        })
        .await,
        "intent",
    );
    std::fs::write(&target, CONTENT).unwrap();
    let ack = noted(
        d.send(ControlCommand::ComposeEffectAck {
            intent,
            reported: json!({"executor": "test"}),
        })
        .await,
        "ack",
    );
    assert_eq!(d.record(ack).await["state"], "DONE");
    Flow {
        report,
        commit,
        grant,
        intent,
        ack,
    }
}

/// The assertions of the demo's S4 and S8: the commit's records name a
/// generation record whose model digest is the loaded model's, and the agent.
async fn assert_linked(d: &Daemon, f: &Flow, want_agent: &str) {
    let gen_id = f.report["generation_record"]
        .as_u64()
        .expect("the report names the generation record");
    for (name, id) in [
        ("compose_commit", f.commit),
        ("grant", f.grant),
        ("intent", f.intent),
        ("ack", f.ack),
    ] {
        let r = d.record(id).await;
        assert_eq!(r["provenance"]["generation_record"], gen_id, "{name}: {r}");
        assert_eq!(r["provenance"]["allen_agent"], want_agent, "{name}: {r}");
    }
    let g = d.record(gen_id).await;
    let want = identity();
    assert_eq!(g["generation"], 1);
    assert_eq!(g["origin"], "compose_proposal");
    assert_eq!(g["model_sha256"], want.model_sha256);
    assert_eq!(g["model_digest_kind"], want.model_digest_kind);
    assert_eq!(g["tokenizer_sha256"], want.tokenizer_sha256);
    assert_eq!(g["task"], f.report["task"]);
    assert_eq!(g["attempt"], 1);
    assert_eq!(g["prompt_ids_sha256"], token_ids_sha256(&PROMPT_IDS));
    assert_eq!(g["output_token_ids_sha256"], token_ids_sha256(&OUT_IDS));
    assert_eq!(g["output_text_sha256"], sha(PROPOSAL.as_bytes()));
    assert_eq!(g["output_tokens"], OUT_IDS.len());
    assert_eq!(g["finish_reason"], "eos");
    assert_eq!(g["daemon"]["pid"], std::process::id());
    // sc#294: the decoding the backend reported travels into the record.
    assert_eq!(
        g["decoding"],
        json!({"mode": "greedy", "greedy_tokens": OUT_IDS.len(), "sampled_tokens": 0}),
        "{g}"
    );
    // sc#337: the backend and op counters travel into the record.
    assert_eq!(
        g["ops"],
        json!({
            "backend": "reference-cpu",
            "native_fallbacks": 0,
            "reference_runs": 0,
            "scope": "process",
            "report": cpu_ops().report,
        }),
        "{g}"
    );
    // The grant, intent and commit still say what they said before.
    assert_eq!(
        d.record(f.commit).await["proposal_sha256"],
        sha(PROPOSAL.as_bytes())
    );
    assert_eq!(
        d.record(f.grant).await["content_sha256"],
        sha(CONTENT.as_bytes())
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn commit_names_generation_record_and_no_agent_when_none_is_attached() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    std::env::remove_var(ENV_SUBJECT);
    std::env::remove_var(ENV_ADOPT);
    let d = Daemon::start(Some(identity()), None).await;
    let f = ordinary_flow(&d).await;
    assert_linked(&d, &f, "none").await;
    d.stop().await;
}

/// (machine id, digest of Cortex record 1) of a home the daemon already opened.
fn home_lineage(home: &Path) -> [u8; 32] {
    let mut name = home.as_os_str().to_owned();
    name.push(".machine-root");
    let root = std::fs::read(PathBuf::from(name)).unwrap();
    let (mut c, _) = Compose::open(home, RootKind::Provisioned, &root, 0xA1E4_0001).unwrap();
    c.record(1).unwrap().digest
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn commit_names_generation_record_and_the_allen_logical_agent_id() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    std::env::remove_var(ENV_SUBJECT);
    std::env::remove_var(ENV_ADOPT);
    // A home that already exists (an ALLEN is bound to a journal's lineage).
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("compose");
    {
        let p = ComposeBridge::new(home.clone(), proposer(), "test:prepare");
        assert!(matches!(
            p.note("constraint", "prepare the home", &[]),
            ControlResponse::ComposeNoted(_)
        ));
        p.close();
    }
    let mut fx = support::fx("provenance");
    fx.cortex = home_lineage(&home);
    let chain = support::chain(&fx, 2);
    let subject = support::write_head(&tmp.path().join("subject.bin"), chain.last().unwrap());
    let agent = ahex(&fx.agent);
    std::env::set_var(ENV_SUBJECT, &subject);
    std::env::set_var(ENV_ADOPT, &agent);
    let d = Daemon::start(Some(identity()), Some((tmp, home))).await;
    let f = ordinary_flow(&d).await;
    assert_linked(&d, &f, &agent).await;
    d.stop().await;
    std::env::remove_var(ENV_SUBJECT);
    std::env::remove_var(ENV_ADOPT);
}

/// No model identity (a stub or reference-weights run): no generation record,
/// an explicit null in the chain, the flow itself unchanged.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn no_model_identity_means_no_generation_claim_and_the_flow_still_works() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    std::env::remove_var(ENV_SUBJECT);
    std::env::remove_var(ENV_ADOPT);
    let d = Daemon::start(None, None).await;
    let f = ordinary_flow(&d).await;
    assert_eq!(f.report["generation_record"], Value::Null);
    for id in [f.commit, f.grant, f.intent, f.ack] {
        let r = d.record(id).await;
        assert_eq!(r["provenance"]["generation_record"], Value::Null, "{r}");
        assert_eq!(r["provenance"]["allen_agent"], "none");
    }
    d.stop().await;
}

/// A ledger written before this change has no provenance anywhere: it opens,
/// verifies and runs the same flow; what the daemon writes next says "none".
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn an_old_ledger_without_provenance_still_opens_and_runs() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    std::env::remove_var(ENV_SUBJECT);
    std::env::remove_var(ENV_ADOPT);
    let d = Daemon::start(Some(identity()), None).await;
    // The old commit record exactly as #261 wrote it (no provenance field),
    // appended below the note guard, as an old journal would hold it.
    let target = d.ws.join("OLD.md");
    let old = json!({"compose_commit": 1, "task": 77, "cx_promotion": 777, "cx_evidence": 778,
        "proposal_sha256": sha(b"old proposal"), "path": "OLD.md",
        "content_sha256": sha(b"old\n"), "workspace": d.ws.display().to_string()});
    let commit = noted(
        d.bridge.note_unchecked("effect", &old.to_string(), &[]),
        "old commit",
    );
    let before = d.record(commit).await;
    assert!(before.get("provenance").is_none());
    let grant = noted(
        d.send(ControlCommand::ComposeAuthorize {
            cx_promotion: 777,
            proposal_sha256: sha(b"old proposal"),
            workspace: d.ws.display().to_string(),
            approver: "drake".into(),
            constraints: vec![],
            desk_proof: d.desk(777, &sha(b"old proposal"), "OLD.md", &sha(b"old\n")),
            objective_id: None,
        })
        .await,
        "authorize on an old commit",
    );
    let (pid, start) = aien_runtime::effects::self_executor();
    let intent = noted(
        d.send(ControlCommand::ComposeEffectIntent {
            authorization: grant,
            proposal_sha256: sha(b"old proposal"),
            path: "OLD.md".into(),
            target: target.display().to_string(),
            content_sha256: sha(b"old\n"),
            executor_pid: pid,
            executor_start: start,
        })
        .await,
        "intent on an old commit",
    );
    std::fs::write(&target, "old\n").unwrap();
    let ack = noted(
        d.send(ControlCommand::ComposeEffectAck {
            intent,
            reported: json!({}),
        })
        .await,
        "ack",
    );
    assert_eq!(d.record(ack).await["state"], "DONE");
    for id in [grant, intent, ack] {
        let r = d.record(id).await;
        assert_eq!(r["provenance"]["generation_record"], Value::Null, "{r}");
        assert_eq!(r["provenance"]["allen_agent"], "none");
    }
    d.stop().await;
}

/// A caller cannot set the fields: ComposeNote refuses a provenance field on
/// any record, and a forged "generation" a record that names one; a grant
/// request carries only ids and digests, so the copy comes from the daemon's
/// own commit record whatever the request says.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
#[cfg_attr(not(compose_linked), ignore = "needs librx_compose.a: stub build")]
async fn a_client_cannot_set_the_provenance_fields() {
    let _home = home_guard::home_slot();
    let _g = ONE_HOME.lock().await;
    std::env::remove_var(ENV_SUBJECT);
    std::env::remove_var(ENV_ADOPT);
    let d = Daemon::start(Some(identity()), None).await;
    let before = d.host_records().await;
    for (kind, text) in [
        (
            "effect",
            r#"{"provenance":{"generation_record":3,"allen_agent":"none"}}"#,
        ),
        ("constraint", r#"{"provenance":{"generation_record":3}}"#),
        (
            "effect",
            r#"{"compose_commit":1,"provenance":{"generation_record":3}}"#,
        ),
        ("effect", r#"{"generation":1,"origin":"compose_proposal"}"#),
        ("authorization", r#"{"provenance":{"generation_record":3}}"#),
    ] {
        match d
            .send(ControlCommand::ComposeNote {
                kind: kind.into(),
                text: text.into(),
                links: vec![],
            })
            .await
        {
            ControlResponse::Error(e) => assert!(
                e.contains("provenance") || e.contains("generation") || e.contains("authorization"),
                "{e}"
            ),
            other => panic!("FORGED ({kind} {text}): {other:?}"),
        }
    }
    assert_eq!(d.host_records().await, before, "nothing was written");
    // The flow's records carry the daemon's values; the request has no field to say otherwise.
    let f = ordinary_flow(&d).await;
    assert_linked(&d, &f, "none").await;
    d.stop().await;
}
