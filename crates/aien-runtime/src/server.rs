//! Local UNIX domain socket server for AienRuntimeSpine
//! Exposes typed control RPC over local IPC to operator CLI tools.

use crate::compose::{compose_dir_from_env, ComposeBridge, ComposeProposer};
use crate::control::{ControlCommand, ControlEnvelope, ControlResponse};
use crate::generation::{DaemonStart, ModelIdentity, TurnEvidence};
use crate::spine::AienRuntimeSpine;
use aien_inference_abi::{AienInferenceBackend, ChatTokenizer, SamplingParams};
use aien_scheduler::{ChannelCompletionSink, CompletionEvent};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, RwLock};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tokio::sync::{Mutex, Notify};

/// True when the process on the other end of `stream` runs as this process's
/// effective user. The default socket lives in shared /tmp, so both ends check
/// peer credentials instead of trusting whoever created the path.
pub(crate) fn peer_is_current_user(stream: &UnixStream) -> bool {
    // SAFETY: geteuid has no preconditions and cannot fail.
    let euid = unsafe { libc::geteuid() };
    stream.peer_cred().map(|c| c.uid() == euid).unwrap_or(false)
}

/// The persistent-KV counters of the `AIEN_STEP_LOG=1` step line, under their
/// METRICS_AND_SETTINGS names (cut C7).
pub fn kv_step_log_fields(s: &crate::kv_residency::KvResidencyStatus) -> String {
    let t = &s.stats;
    format!(
        "kv_park_jobs_total={} kv_park_failed_total={} kv_park_dedup_total={} \
         kv_park_indeterminate_total={} kv_restore_jobs_total={} kv_restore_failed_total={} \
         kv_restore_fallback_prefill_total={} kv_restore_retry_suppressed_total={} \
         kv_store_stale_completions_total={} kv_pool_holder_blocks={} kv_restore_inflight_blocks={}",
        t.park_jobs_total,
        t.park_failed_total,
        t.park_dedup_total,
        t.park_indeterminate_total,
        t.restore_jobs_total,
        t.restore_failed_total,
        t.restore_fallback_prefill_total,
        t.restore_retry_suppressed_total,
        t.store_stale_completions_total,
        s.holder_blocks,
        s.restore_inflight_blocks,
    )
}

/// The shutdown log line of the persistent-KV drain (ADR 7.5): the counters
/// `kv_store_shutdown_unfinished_jobs` and `kv_shutdown_unfinished_bytes`, then
/// every unfinished job by key, size and stage.
pub fn kv_shutdown_log_line(r: &aien_kv_store::DrainReport) -> String {
    let bytes: u64 = r.unfinished.iter().map(|u| u.bytes).sum();
    let mut line = format!(
        "persistent KV drain: published={} cancelled={} kv_store_shutdown_unfinished_jobs={} kv_shutdown_unfinished_bytes={}",
        r.published,
        r.cancelled,
        r.unfinished.len(),
        bytes
    );
    for u in &r.unfinished {
        let key: String = u.key.0 .0.iter().map(|b| format!("{b:02x}")).collect();
        line.push_str(&format!(
            "\n  unfinished job {} key {key} bytes {} stage {:?}",
            u.job.0, u.bytes, u.stage
        ));
    }
    line
}

pub struct AienRuntimeServer {
    socket_path: PathBuf,
    spine: Arc<Mutex<AienRuntimeSpine>>,
    shutdown_notify: Arc<Notify>,
    is_running: Arc<AtomicBool>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    warm_up: AtomicBool,
    /// Optional serving-time probe (GB10 allocation counters), see `set_serving_probe`.
    probe: std::sync::Mutex<Option<Arc<dyn aien_inference_abi::gb10_serving::ServingProbe>>>,
    /// Test seam: a ready-made compose bridge (see `with_compose_bridge`).
    compose_override: std::sync::Mutex<Option<Arc<ComposeBridge>>>,
    /// The loaded model files' digests (set by the daemon after load; absent
    /// = no generation record, see crate::generation).
    identity: std::sync::Mutex<Option<Arc<ModelIdentity>>>,
    started: DaemonStart,
    /// Headstart M3 (#537): the provisional preparation adapter whose fence
    /// bound compose commit requests pass (see `with_prep_adapter`).
    #[cfg(feature = "provisional-prep")]
    prep: std::sync::Mutex<Option<Arc<crate::provisional::PrepAdapter>>>,
}

/// NEXT-PHASE-1 v4 declared warm-up prompt (ACCEPTANCE-v4 Section 2(2)):
/// fixed text, more than one 128-token prefill chunk once templated, so
/// both the full-chunk and the remainder prefill paths run before serving.
/// Log line when the warm-up is skipped for a plain model (no chat template).
pub const WARM_UP_SKIPPED_PLAIN: &str = "Warm-up: skipped (plain model, no chat template)";

pub const WARM_UP_TEXT: &str = "Warm-up turn before serving requests. This text is fixed and its one generated token is discarded. \
Warm-up turn before serving requests. This text is fixed and its one generated token is discarded. \
Warm-up turn before serving requests. This text is fixed and its one generated token is discarded. \
Warm-up turn before serving requests. This text is fixed and its one generated token is discarded. \
Warm-up turn before serving requests. This text is fixed and its one generated token is discarded. \
Warm-up turn before serving requests. This text is fixed and its one generated token is discarded.";

impl AienRuntimeServer {
    pub fn new(spine: AienRuntimeSpine, socket_path: impl AsRef<Path>) -> Self {
        Self {
            socket_path: socket_path.as_ref().to_path_buf(),
            spine: Arc::new(Mutex::new(spine)),
            shutdown_notify: Arc::new(Notify::new()),
            is_running: Arc::new(AtomicBool::new(false)),
            tokenizer: Arc::new(RwLock::new(None)),
            warm_up: AtomicBool::new(false),
            probe: std::sync::Mutex::new(None),
            compose_override: std::sync::Mutex::new(None),
            identity: std::sync::Mutex::new(None),
            started: DaemonStart::now(),
            #[cfg(feature = "provisional-prep")]
            prep: std::sync::Mutex::new(None),
        }
    }

    /// Headstart M3 (#537): send `ComposeApprovedProposal` and
    /// `ComposeAuthorize` requests that `adapter` has bound
    /// (`PrepAdapter::bind_approval`) through its final commit fence. Unbound
    /// requests run the original call unchanged. Without this call (the
    /// default) nothing is fenced.
    #[cfg(feature = "provisional-prep")]
    pub fn with_prep_adapter(self, adapter: Arc<crate::provisional::PrepAdapter>) -> Self {
        *self.prep.lock().expect("prep adapter") = Some(adapter);
        self
    }

    /// Test seam: serve compose commands from `bridge` (for example one with a
    /// fixed proposer) instead of the bridge built from the environment and
    /// the loaded model. The daemon binary never calls this.
    #[doc(hidden)]
    #[cfg(feature = "test-support")]
    pub fn with_compose_bridge(self, bridge: Arc<ComposeBridge>) -> Self {
        *self.compose_override.lock().expect("compose override") = Some(bridge);
        self
    }

    /// Installs the digests of the model and tokenizer files the daemon loaded.
    /// Without them (stub or reference weights) no generation record is written.
    pub fn set_model_identity(&self, identity: ModelIdentity) {
        *self.identity.lock().expect("identity lock") = Some(Arc::new(identity));
    }

    /// Installs the tokenizer that `StreamTurn` uses to encode prompts and decode tokens.
    pub fn set_tokenizer(&self, tokenizer: ChatTokenizer) {
        *self.tokenizer.write().expect("tokenizer lock") = Some(tokenizer);
    }

    /// Run the declared warm-up turn (1 token, discarded) at the start of
    /// `run`, before any request is served (ACCEPTANCE-v4 Section 2(2)).
    pub fn enable_warm_up(&self) {
        self.warm_up.store(true, Ordering::SeqCst);
    }

    /// Log the GB10 allocation counters once after warm-up and the change since warm-up after
    /// every served connection (sovereign-core#277). Unset = nothing is logged.
    pub fn set_serving_probe(
        &self,
        probe: Arc<dyn aien_inference_abi::gb10_serving::ServingProbe>,
    ) {
        *self.probe.lock().expect("probe lock") = Some(probe);
    }

    pub fn spine(&self) -> Arc<Mutex<AienRuntimeSpine>> {
        self.spine.clone()
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Runs the runtime server, accepting control connections and advancing the engine step loop.
    pub async fn run<B: AienInferenceBackend + Send + 'static>(
        &self,
        mut backend: B,
    ) -> Result<(), String> {
        // Clean up stale socket file if it exists and refuses connection
        if self.socket_path.exists() {
            if UnixStream::connect(&self.socket_path).await.is_err() {
                let _ = std::fs::remove_file(&self.socket_path);
            } else {
                return Err(format!(
                    "Runtime socket already active at {}",
                    self.socket_path.display()
                ));
            }
        }

        if let Some(parent) = self.socket_path.parent() {
            if !parent.exists() {
                let _ = std::fs::create_dir_all(parent);
            }
        }

        // Bad compose limits stop the daemon here, loudly, before anything is served.
        crate::compose::compose_budgets_from_env().map_err(|e| format!("compose limits: {e}"))?;
        crate::compose::compose_doc_max_tokens_from_env()
            .map_err(|e| format!("compose limits: {e}"))?;

        let authorize_desk = crate::compose::authorize_requires_desk_from_env()?;
        // sc#328: a bridge that requires the approval desk refuses to start
        // without a loadable desk key, before the socket exists. A bridge with
        // the desk off (code that embeds the runtime) is the same dev-only
        // opt-out as the env switch: a strict run refuses it.
        let desk_home = match self
            .compose_override
            .lock()
            .expect("compose override")
            .as_ref()
        {
            Some(b) if !b.authorize_requires_desk() => {
                crate::compose::authorize_desk_setting(
                    Some("0"),
                    aien_inference_abi::strict::dev_fallback_active(),
                )
                .map_err(|e| format!("compose bridge built with the approval desk off: {e}"))?;
                None
            }
            Some(b) => Some(b.dir().to_path_buf()),
            None if authorize_desk => compose_dir_from_env().ok(),
            None => None,
        };
        if let Some(dir) = desk_home {
            crate::compose::check_desk_key_at_start(&dir)?;
        }
        let listener = UnixListener::bind(&self.socket_path).map_err(|e| {
            format!(
                "Failed to bind runtime UNIX domain socket at {}: {}",
                self.socket_path.display(),
                e
            )
        })?;
        {
            use std::os::unix::fs::PermissionsExt;
            std::fs::set_permissions(&self.socket_path, std::fs::Permissions::from_mode(0o600))
                .map_err(|e| {
                    format!(
                        "Failed to restrict runtime socket {} to owner: {}",
                        self.socket_path.display(),
                        e
                    )
                })?;
        }

        self.is_running.store(true, Ordering::SeqCst);

        // Background worker advancing engine steps whenever requests are active
        let spine_worker = self.spine.clone();
        let is_running_worker = self.is_running.clone();
        let shutdown_notify_worker = self.shutdown_notify.clone();

        let step_log = std::env::var("AIEN_STEP_LOG").is_ok_and(|v| v == "1");
        let worker_handle = tokio::spawn(async move {
            while is_running_worker.load(Ordering::Relaxed) {
                let has_work = {
                    let s = spine_worker.lock().await;
                    s.scheduler.running_count() > 0 || s.scheduler.waiting_count() > 0
                };

                if has_work {
                    let mut s = spine_worker.lock().await;
                    let t_step = std::time::Instant::now();
                    match s.step(&mut backend).await {
                        // NEXT-PHASE-1 v3 observation: one line per engine step
                        // (AIEN_STEP_LOG=1) from the metrics the step already reports.
                        // Persistent-KV counters are appended when it is on (C7).
                        Ok(Some(m)) if step_log => eprintln!(
                            "step: prefill_tokens={} emitted={} backend_us={} spine_us={}{}",
                            m.prefill_tokens_processed,
                            m.decode_tokens_emitted,
                            m.step_latency_us,
                            t_step.elapsed().as_micros(),
                            s.kv_residency_status()
                                .map(|k| format!(" {}", kv_step_log_fields(&k)))
                                .unwrap_or_default()
                        ),
                        Ok(_) => {}
                        // A swallowed step error makes the daemon look alive
                        // while its decode loop is dead. Print it.
                        Err(e) => eprintln!("runtime step error: {}", e),
                    }
                } else {
                    tokio::select! {
                        _ = tokio::time::sleep(tokio::time::Duration::from_millis(5)) => {},
                        _ = shutdown_notify_worker.notified() => break,
                    }
                }
            }
        });

        // NEXT-PHASE-1 v4: the declared warm-up, before the accept loop
        // serves anything (clients that connect meanwhile wait in the backlog).
        if self.warm_up.load(Ordering::SeqCst) {
            println!("  {}", self.run_warm_up().await);
        }
        let probe = self.probe.lock().expect("probe lock").clone();
        if let Some(line) = probe.as_ref().and_then(|p| p.after_warm_up()) {
            println!("  {line}");
        }

        // NEXT-PHASE-1: the compose bridge (opened on the first RunComposeTask).
        let preset = self
            .compose_override
            .lock()
            .expect("compose override")
            .clone();
        let identity = self.identity.lock().expect("identity lock").clone();
        #[cfg(feature = "provisional-prep")]
        let prep = self.prep.lock().expect("prep adapter").clone();
        // Correlation trace exporter (#398): on only when AIEN_TRACE_JSONL names
        // a file. Off, the bridge keeps a NullSink and emits nothing. The flusher
        // handle lives until the end of `run`, after the compose home closes,
        // so the last events are written at shutdown.
        let (trace_sink, _trace_flusher) = trace_sink_from_env();
        let compose = match compose_dir_from_env() {
            _ if preset.is_some() => preset,
            Ok(dir) => Some(Arc::new(
                ComposeBridge::new(
                    dir,
                    model_proposer(
                        self.spine.clone(),
                        self.tokenizer.clone(),
                        tokio::runtime::Handle::current(),
                    ),
                    "model:StreamTurn-path",
                )
                .with_doc_proposer(doc_model_proposer(
                    self.spine.clone(),
                    self.tokenizer.clone(),
                    tokio::runtime::Handle::current(),
                )?)
                .with_authorize_requires_desk(authorize_desk)
                .with_trace_sink(trace_sink),
            )),
            Err(e) => {
                tracing::warn!("compose bridge disabled: {e}");
                None
            }
        };
        // Provenance (arch#162): the compose task writes the generation record of
        // the proposal it commits, so the bridge needs the model identity too.
        if let (Some(b), Some(id)) = (compose.as_ref(), identity.as_ref()) {
            b.set_model_identity((**id).clone(), self.started);
        }
        // NEXT-PHASE-2 (ACCEPTANCE-v2 2.4): settle effects a previous process
        // left open, before serving anything. Reads the world, never re-runs.
        // A failed or refused reconcile does not stop the daemon, but every
        // effect command refuses until an operator reconcile succeeds
        // (ACCEPTANCE-v3 2.5).
        if let Some(b) = compose.as_ref() {
            let line = crate::compose::authorize_mac_line(b.authorize_requires_desk());
            println!("{line}");
            if !b.authorize_requires_desk() {
                tracing::warn!("{line}");
            }
            if aien_omega_compose::LINKED {
                println!(
                    "Compose: native (omega {})",
                    aien_omega_compose::EXPECTED_OMEGA_SHA
                );
            } else {
                println!("Compose: STUB (every compose call is Unavailable)");
            }
        }
        if let Some(b) = compose.clone() {
            let gate = b.clone();
            let line =
                match tokio::task::spawn_blocking(move || crate::effects::reconcile_at_start(&b))
                    .await
                {
                    Ok(line) => line,
                    Err(e) => {
                        gate.set_reconcile_failed(format!("failed: {e}"));
                        format!(
                            "Reconcile: failed: {e}; {}",
                            crate::effects::RECONCILE_GATE_NOTE
                        )
                    }
                };
            println!("{line}");
            // sovereign-core #249: settle approved-proposal claims a previous
            // process left open (never re-run). A failure gates effect commands
            // and approved proposals like a failed effect reconcile.
            let b = compose.clone().expect("compose is Some here");
            let gate = b.clone();
            let line = match tokio::task::spawn_blocking(move || {
                crate::approved_replay::reconcile_at_start(&b)
            })
            .await
            {
                Ok(Ok(line)) => line,
                Ok(Err(e)) => {
                    gate.set_reconcile_failed(format!("replay reconcile refused: {e}"));
                    format!(
                        "Replay reconcile: refused: {e}; {}",
                        crate::effects::RECONCILE_GATE_NOTE
                    )
                }
                Err(e) => {
                    gate.set_reconcile_failed(format!("replay reconcile failed: {e}"));
                    format!(
                        "Replay reconcile: failed: {e}; {}",
                        crate::effects::RECONCILE_GATE_NOTE
                    )
                }
            };
            println!("{line}");
        }

        // Connection accept loop
        loop {
            tokio::select! {
                accept_res = listener.accept() => {
                    match accept_res {
                        Ok((stream, _)) => {
                            if !peer_is_current_user(&stream) {
                                tracing::warn!("rejected runtime connection from another user");
                                continue;
                            }
                            let spine_conn = self.spine.clone();
                            let is_running_conn = self.is_running.clone();
                            let notify_conn = self.shutdown_notify.clone();

                            let tokenizer_conn = self.tokenizer.clone();
                            let compose_conn = compose.clone();
                            let identity_conn = identity.clone();
                            let started = self.started;
                            let probe_conn = probe.clone();
                            #[cfg(feature = "provisional-prep")]
                            let prep_conn = prep.clone();
                            tokio::spawn(async move {
                                #[cfg(not(feature = "provisional-prep"))]
                                handle_connection(
                                    stream,
                                    spine_conn,
                                    is_running_conn,
                                    notify_conn,
                                    tokenizer_conn,
                                    compose_conn,
                                    (identity_conn, started),
                                )
                                .await;
                                #[cfg(feature = "provisional-prep")]
                                handle_connection(
                                    stream,
                                    spine_conn,
                                    is_running_conn,
                                    notify_conn,
                                    tokenizer_conn,
                                    compose_conn,
                                    (identity_conn, started),
                                    prep_conn,
                                )
                                .await;
                                if let Some(line) = probe_conn.as_ref().and_then(|p| p.after_request()) {
                                    println!("  {line}");
                                }
                            });
                        }
                        Err(e) => {
                            tracing::warn!("Failed to accept runtime connection: {}", e);
                        }
                    }
                }
                _ = self.shutdown_notify.notified() => {
                    break;
                }
            }
        }

        self.is_running.store(false, Ordering::SeqCst);
        let _ = worker_handle.await;

        // Persistent KV (ADR 7.5): drain the store with its deadline after the
        // step loop stopped. The worker keeps every buffer of an unfinished job
        // until its terminal; nothing is freed to meet the deadline.
        let drain = self.spine.lock().await.kv_drain_handle();
        if let Some(handle) = drain {
            if let Ok(report) = tokio::task::spawn_blocking(move || handle.drain()).await {
                eprintln!("{}", kv_shutdown_log_line(&report));
            }
        }

        // Close the compose home now, not whenever the last connection task drops
        // its Arc (sovereign-core #306).
        if let Some(c) = compose.clone() {
            let _ = tokio::task::spawn_blocking(move || c.close()).await;
        }

        // Clean up socket file on clean shutdown
        if self.socket_path.exists() {
            let _ = std::fs::remove_file(&self.socket_path);
        }

        Ok(())
    }
}

async fn write_response(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    response: &ControlResponse,
) -> bool {
    match serde_json::to_string(response) {
        Ok(mut serialized) => {
            serialized.push('\n');
            writer.write_all(serialized.as_bytes()).await.is_ok()
        }
        Err(_) => false,
    }
}

fn text_suffix(previous: &str, decoded: &str) -> String {
    decoded.strip_prefix(previous).unwrap_or("").to_string()
}

impl AienRuntimeServer {
    /// One greedy 1-token turn on `WARM_UP_TEXT`; the token is discarded.
    /// Returns the log line: `Warm-up: 1 token in <ms> ms over <n> prompt tokens (discarded)`,
    /// `Warm-up: FAILED ...` (a failure does not stop the daemon), or [`WARM_UP_SKIPPED_PLAIN`]
    /// when the loaded tokenizer has no chat template (a plain model refuses chat turns by
    /// design, so the chat warm-up is not attempted).
    pub async fn run_warm_up(&self) -> String {
        if self
            .tokenizer
            .read()
            .expect("tokenizer lock")
            .as_ref()
            .is_some_and(|t| matches!(t.template(), aien_inference_abi::ChatTemplate::None))
        {
            return WARM_UP_SKIPPED_PLAIN.to_string();
        }
        let messages = vec![crate::control::ChatTurn {
            role: "user".into(),
            content: WARM_UP_TEXT.into(),
        }];
        let prompt_tokens = {
            let guard = self.tokenizer.read().expect("tokenizer lock");
            guard.as_ref().and_then(|t| {
                crate::control::try_format_chat(t.template(), &messages)
                    .ok()
                    .and_then(|text| t.encode(&text).ok())
                    .map(|ids| ids.len())
            })
        };
        let t0 = std::time::Instant::now();
        let out = generate_text(
            self.spine.clone(),
            self.tokenizer.clone(),
            messages,
            1,
            0.0,
            std::time::Duration::from_secs(120),
            "",
        )
        .await;
        let ms = t0.elapsed().as_millis();
        match out {
            Ok(g) => format!(
                "Warm-up: {} token in {ms} ms over {} prompt tokens (discarded)",
                g.tokens,
                prompt_tokens.unwrap_or(0)
            ),
            Err(e) => format!("Warm-up: FAILED after {ms} ms: {e}"),
        }
    }
}

/// Encode the chat and submit it to the spine (the `StreamTurn` path). Returns
/// the tokenizer and the completion event stream of the submitted sequence.
async fn submit_turn(
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    messages: Vec<crate::control::ChatTurn>,
    max_tokens: usize,
    temperature: f32,
    assistant_prefix: &str,
) -> Result<
    (
        ChatTokenizer,
        tokio::sync::mpsc::UnboundedReceiver<CompletionEvent>,
        Vec<u32>,
    ),
    String,
> {
    let tokenizer = {
        let guard = tokenizer.read().expect("tokenizer lock");
        guard.clone()
    };
    let Some(tokenizer) = tokenizer else {
        return Err("tokenizer is not loaded; native chat cannot encode the prompt".into());
    };
    // The template ends with the assistant marker; an assistant-response
    // prefix (compose Skill, ACCEPTANCE-v4 2(1)) follows it directly. The
    // tokenizer adds the one BOS token.
    let prompt = format!(
        "{}{assistant_prefix}",
        crate::control::try_format_chat(tokenizer.template(), &messages)
            .map_err(|error| format!("{error}"))?
    );
    let tokens = tokenizer
        .encode(&prompt)
        .map_err(|error| format!("tokenizer encode failed: {error}"))?;
    let (sink, events) = ChannelCompletionSink::channel();
    let mut spine = spine.lock().await;
    let sink_id = spine.register_completion_sink(std::sync::Arc::new(sink));
    let sampling = SamplingParams {
        temperature,
        top_p: 0.95,
        max_tokens: max_tokens.max(1),
        stop_token_ids: tokenizer.stop_token_ids().to_vec(),
    };
    spine.submit_work(
        std::sync::Arc::from(tokens.as_slice()),
        sampling,
        2,
        Some(sink_id),
    )?;
    Ok((tokenizer, events, tokens))
}

/// Receipt label of a finish reason (ACCEPTANCE-v5 Q3): the stop token
/// (end of sequence) is "eos", the token limit is "max_tokens".
pub fn finish_reason_label(reason: &aien_inference_abi::FinishReason) -> &'static str {
    match reason {
        aien_inference_abi::FinishReason::StopToken => "eos",
        aien_inference_abi::FinishReason::LengthLimit => "max_tokens",
        aien_inference_abi::FinishReason::Aborted => "aborted",
        aien_inference_abi::FinishReason::Preempted => "preempted",
    }
}

/// A compose command, run on a blocking thread against the bridge.
type ComposeJob = Box<dyn FnOnce(&Arc<ComposeBridge>) -> ControlResponse + Send>;

/// NEXT-PHASE-1: the whole turn as one string (no streaming) and its token
/// count, for the compose "model" Skill. Same submission path as
/// `StreamTurn`; fails after `limit`.
async fn generate_text(
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    messages: Vec<crate::control::ChatTurn>,
    max_tokens: usize,
    temperature: f32,
    limit: std::time::Duration,
    assistant_prefix: &str,
) -> Result<crate::compose::Generation, String> {
    let (tokenizer, mut events, prompt_ids) = submit_turn(
        spine,
        tokenizer,
        messages,
        max_tokens,
        temperature,
        assistant_prefix,
    )
    .await?;
    let collect = async {
        let mut produced = Vec::new();
        loop {
            match events.recv().await {
                Some(CompletionEvent::Token { token, .. }) => produced.push(token),
                Some(CompletionEvent::Finished {
                    finish_reason,
                    decoding,
                    ops,
                    ..
                }) => {
                    if let Some(line) = crate::generation::op_report_line(ops.as_ref()) {
                        eprintln!("{line}");
                    }
                    return tokenizer
                        .decode_opts(&produced, true)
                        .map(|text| crate::compose::Generation {
                            text,
                            tokens: produced.len(),
                            finish_reason: Some(finish_reason_label(&finish_reason).to_string()),
                            token_ids: Some(produced.clone()),
                            prompt_tokens: Some(prompt_ids.len()),
                            prompt_ids_sha256: Some(crate::compose::token_ids_sha256(&prompt_ids)),
                            decoding: decoding.clone(),
                            ops: ops.clone(),
                        })
                        .map_err(|e| format!("tokenizer decode failed: {e}"));
                }
                Some(CompletionEvent::Error { message, .. }) => return Err(message),
                None => return Err("native runtime stopped before the turn finished".into()),
            }
        }
    };
    tokio::time::timeout(limit, collect)
        .await
        .map_err(|_| crate::compose::wall_clock_reason(limit.as_millis()))?
}

/// The compose "model" Skill: real inference through `generate_text`, run
/// from an omega World worker thread (not a tokio thread) via `block_on`.
/// Greedy, at most `AIEN_COMPOSE_MAX_TOKENS` (default 48) tokens per reply;
/// each call gets the limit `propose_task_checked` passes (what is left of
/// the task budget: 29 s for an edit, 120 s for a document; omega waits that
/// plus 1 s), so a slow model fails the Skill (no proposal) instead of failing the run.
pub fn model_proposer(
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    handle: tokio::runtime::Handle,
) -> ComposeProposer {
    let max_tokens = std::env::var("AIEN_COMPOSE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(48);
    proposer_with_cap(spine, tokenizer, handle, max_tokens)
}

/// The proposer for full-document tasks: the same inference path with its own
/// token cap, `AIEN_COMPOSE_DOC_MAX_TOKENS` (default 1024). A bad value is refused.
pub fn doc_model_proposer(
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    handle: tokio::runtime::Handle,
) -> Result<ComposeProposer, String> {
    let cap = crate::compose::compose_doc_max_tokens_from_env()?;
    Ok(proposer_with_cap(spine, tokenizer, handle, cap))
}

fn proposer_with_cap(
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    handle: tokio::runtime::Handle,
    max_tokens: usize,
) -> ComposeProposer {
    Arc::new(move |prompt: &str, limit: std::time::Duration| {
        let messages = vec![crate::control::ChatTurn {
            role: "user".into(),
            content: prompt.to_string(),
        }];
        // The assistant turn starts with the template's fixed prefix; the
        // reply the parser reads is prefix + generated text (ACCEPTANCE-v4 2(1)).
        let prefix = tokenizer
            .read()
            .expect("tokenizer lock")
            .as_ref()
            .map(|t| crate::compose::compose_assistant_prefix(&t.template()))
            .unwrap_or(crate::compose::COMPOSE_ASSISTANT_PREFIX);
        handle
            .block_on(generate_text(
                spine.clone(),
                tokenizer.clone(),
                messages,
                max_tokens,
                0.0,
                limit,
                prefix,
            ))
            .map(|g| crate::compose::Generation {
                text: format!("{prefix}{}", g.text),
                ..g
            })
    })
}

/// What a turn needs to write its generation record.
struct RecordCtx {
    compose: Option<Arc<ComposeBridge>>,
    identity: Option<Arc<ModelIdentity>>,
    started: DaemonStart,
    request_id: u64,
    operation_id: u128,
}

/// Owned inputs of one finished turn's generation record.
struct GenerationOwned {
    prompt_ids: Vec<u32>,
    output_ids: Vec<u32>,
    text: String,
    total_tokens: usize,
    finish_reason: &'static str,
    request_id: u64,
    operation_id: u128,
    decoding: Option<aien_abi_core::DecodeObservation>,
    ops: Option<aien_abi_core::OpEvidence>,
}

/// Write the daemon's generation record for a finished turn and return its
/// ledger id. `None` when there is no compose ledger or no model identity (a
/// stub or reference-weights run), or when the append failed; the failure is
/// logged and the turn still succeeds, but an id is returned only for a record
/// that was written.
async fn record_generation(
    compose: Option<Arc<ComposeBridge>>,
    identity: Option<Arc<ModelIdentity>>,
    started: DaemonStart,
    g: GenerationOwned,
) -> Option<u64> {
    let (bridge, identity) = (compose?, identity?);
    let written = tokio::task::spawn_blocking(move || {
        let record = crate::generation::build_record(
            &identity,
            &TurnEvidence {
                prompt_ids: &g.prompt_ids,
                output_ids: &g.output_ids,
                text: &g.text,
                total_tokens: g.total_tokens,
                finish_reason: g.finish_reason,
                request_id: g.request_id,
                operation_id: g.operation_id,
                decoding: g.decoding.as_ref(),
                ops: g.ops.as_ref(),
            },
            started,
        );
        bridge.record_generation(&record)
    })
    .await;
    match written {
        Ok(Ok(id)) => Some(id),
        Ok(Err(e)) => {
            tracing::warn!("generation record not written: {e}");
            eprintln!("generation record not written: {e}");
            None
        }
        Err(e) => {
            tracing::warn!("generation record task failed: {e}");
            eprintln!("generation record task failed: {e}");
            None
        }
    }
}

async fn stream_turn(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    messages: Vec<crate::control::ChatTurn>,
    max_tokens: usize,
    temperature: f32,
    rec: RecordCtx,
) {
    let RecordCtx {
        compose,
        identity,
        started,
        request_id,
        operation_id,
    } = rec;
    let (tokenizer, mut events, prompt_ids) =
        match submit_turn(spine, tokenizer, messages, max_tokens, temperature, "").await {
            Ok(x) => x,
            Err(error) => {
                let _ = write_response(writer, &ControlResponse::Error(error)).await;
                return;
            }
        };

    let mut produced = Vec::new();
    let mut text = String::new();
    loop {
        let next = tokio::time::timeout(std::time::Duration::from_secs(120), events.recv()).await;
        match next {
            Ok(Some(CompletionEvent::Token { token, .. })) => {
                produced.push(token);
                let decoded = tokenizer
                    .decode_opts(&produced, true)
                    .unwrap_or_else(|_| text.clone());
                let delta = text_suffix(&text, &decoded);
                if decoded.len() >= text.len() && decoded.starts_with(&text) {
                    text = decoded;
                }
                if !delta.is_empty()
                    && !write_response(writer, &ControlResponse::TurnDelta { text: delta }).await
                {
                    return;
                }
            }
            Ok(Some(CompletionEvent::Finished {
                total_tokens,
                finish_reason,
                decoding,
                ops,
                ..
            })) => {
                if let Some(line) = crate::generation::op_report_line(ops.as_ref()) {
                    eprintln!("{line}");
                }
                let generation_record = record_generation(
                    compose,
                    identity,
                    started,
                    GenerationOwned {
                        prompt_ids,
                        output_ids: produced,
                        text: text.clone(),
                        total_tokens,
                        finish_reason: finish_reason_label(&finish_reason),
                        request_id,
                        operation_id,
                        decoding,
                        ops,
                    },
                )
                .await;
                let _ = write_response(
                    writer,
                    &ControlResponse::TurnFinished {
                        text,
                        total_tokens,
                        generation_record,
                    },
                )
                .await;
                return;
            }
            Ok(Some(CompletionEvent::Error { message, .. })) => {
                let _ = write_response(writer, &ControlResponse::Error(message)).await;
                return;
            }
            Ok(None) | Err(_) => {
                let _ = write_response(
                    writer,
                    &ControlResponse::Error(
                        "native runtime stopped streaming before the turn finished".into(),
                    ),
                )
                .await;
                return;
            }
        }
    }
}

#[cfg_attr(feature = "provisional-prep", allow(clippy::too_many_arguments))]
async fn handle_connection(
    stream: UnixStream,
    spine: Arc<Mutex<AienRuntimeSpine>>,
    is_running: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    compose: Option<Arc<ComposeBridge>>,
    (identity, started): (Option<Arc<ModelIdentity>>, DaemonStart),
    #[cfg(feature = "provisional-prep")] prep: Option<Arc<crate::provisional::PrepAdapter>>,
) {
    let (reader, mut writer) = stream.into_split();
    let mut buf_reader = BufReader::new(reader);
    let mut line = String::new();

    while let Ok(n) = buf_reader.read_line(&mut line).await {
        if n == 0 {
            break;
        }
        let trimmed = line.trim();
        if trimmed.is_empty() {
            line.clear();
            continue;
        }

        let parsed = serde_json::from_str::<ControlEnvelope>(trimmed);
        let envelope = match parsed {
            Ok(envelope) => envelope,
            Err(e) => {
                if !write_response(
                    &mut writer,
                    &ControlResponse::Error(format!("Invalid control envelope JSON: {}", e)),
                )
                .await
                {
                    break;
                }
                line.clear();
                continue;
            }
        };

        if let ControlCommand::StreamTurn {
            messages,
            max_tokens,
            temperature,
        } = envelope.command
        {
            stream_turn(
                &mut writer,
                spine.clone(),
                tokenizer.clone(),
                messages,
                max_tokens,
                temperature,
                RecordCtx {
                    compose: compose.clone(),
                    identity: identity.clone(),
                    started,
                    request_id: envelope.request_id,
                    operation_id: envelope.operation_id,
                },
            )
            .await;
            line.clear();
            continue;
        }

        let compose_job: Option<ComposeJob> = match envelope.command {
            ControlCommand::RunComposeTask {
                ref goal,
                ref workspace,
                ref context,
            } => {
                let (g, w, c) = (goal.clone(), workspace.clone(), context.clone());
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    b.run_task_in(&g, &w, c.as_deref())
                }))
            }
            ControlCommand::ComposeNote {
                ref kind,
                ref text,
                ref links,
            } => {
                let (k, t, l) = (kind.clone(), text.clone(), links.clone());
                // Gated records (every authorization, effect phases, replay and
                // commit records) are refused inside `note` itself (#261).
                Some(Box::new(move |b: &Arc<ComposeBridge>| b.note(&k, &t, &l)))
            }
            ControlCommand::ComposeAuthorize {
                cx_promotion,
                ref proposal_sha256,
                ref workspace,
                ref approver,
                ref constraints,
                ref desk_proof,
                ref objective_id,
            } => {
                let req = crate::effects::MintRequest {
                    cx_promotion,
                    proposal_sha256: proposal_sha256.clone(),
                    workspace: workspace.clone(),
                    approver: approver.clone(),
                    constraints: constraints.clone(),
                };
                let proof = desk_proof.clone();
                let objective = objective_id.clone();
                #[cfg(feature = "provisional-prep")]
                let fence = prep.clone();
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    // Headstart M3: a bound request passes the fence instead.
                    #[cfg(feature = "provisional-prep")]
                    if let Some(r) = fence.as_ref().and_then(|a| {
                        a.route_authorize(b, &req, proof.as_ref(), objective.as_deref())
                    }) {
                        return r;
                    }
                    crate::effects::authorize_with_objective(
                        b,
                        &req,
                        proof.as_ref(),
                        objective.as_deref(),
                    )
                }))
            }
            ControlCommand::ComposeRecall { ref ids, prefix } => {
                let i = ids.clone();
                Some(Box::new(move |b: &Arc<ComposeBridge>| b.recall(&i, prefix)))
            }
            ControlCommand::RecoverComposeHome => {
                Some(Box::new(|b: &Arc<ComposeBridge>| b.recover()))
            }
            // NEXT-PHASE-2: effect intents, acks, reconcile, operator control.
            ControlCommand::ComposeEffectIntent {
                authorization,
                ref proposal_sha256,
                ref path,
                ref target,
                ref content_sha256,
                executor_pid,
                executor_start,
            } => {
                let req = crate::effects::IntentRequest {
                    authorization,
                    proposal_sha256: proposal_sha256.clone(),
                    path: path.clone(),
                    target: target.clone(),
                    content_sha256: content_sha256.clone(),
                    executor_pid,
                    executor_start,
                };
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    crate::effects::open_intent(b, &req)
                }))
            }
            ControlCommand::ComposeEffectAck {
                intent,
                ref reported,
            } => {
                let r = reported.clone();
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    crate::effects::ack(b, intent, &r)
                }))
            }
            ControlCommand::ComposeReconcile { ref declare } => {
                let d = declare.clone();
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    crate::effects::reconcile(b, d.as_ref(), "reconcile")
                }))
            }
            ControlCommand::ComposeApprovedProposal {
                ref proposal,
                ref workspace,
            } => {
                let (p, w) = (proposal.clone(), workspace.clone());
                #[cfg(feature = "provisional-prep")]
                let fence = prep.clone();
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    // Headstart M3: a bound request passes the fence instead.
                    #[cfg(feature = "provisional-prep")]
                    if let Some(r) = fence.as_ref().and_then(|a| a.route_approved(b, &p, &w)) {
                        return r;
                    }
                    match crate::approved::ProposerHook::new(b.clone()).submit(&p, &w) {
                        Ok(r) => ControlResponse::ComposeApprovedResult(Box::new(r)),
                        Err(r) => ControlResponse::ComposeApprovedRefused(r),
                    }
                }))
            }
            ControlCommand::ComposeControl {
                ref action,
                ref approver,
                authorization,
            } => {
                let (a, p) = (action.clone(), approver.clone());
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    crate::effects::control(b, &a, &p, authorization)
                }))
            }
            ControlCommand::AllenStatus
            | ControlCommand::AllenProfileShow
            | ControlCommand::AllenProfileSet { .. }
            | ControlCommand::AllenProfileHistory
            | ControlCommand::AllenProfileRevert { .. }
            | ControlCommand::AllenMemoryPut { .. }
            | ControlCommand::AllenMemoryRecall { .. }
            | ControlCommand::AllenMemoryInspect { .. }
            | ControlCommand::AllenMemoryCorrect { .. }
            | ControlCommand::AllenMemoryForget { .. }
            | ControlCommand::AllenMemoryExport { .. }
            | ControlCommand::AllenGoalsList { .. }
            | ControlCommand::AllenGoalAdd { .. }
            | ControlCommand::AllenGoalClose { .. }
            | ControlCommand::AllenProfileReset { .. } => {
                let c = envelope.command.clone();
                Some(Box::new(move |b: &Arc<ComposeBridge>| b.allen_command(&c)))
            }
            _ => None,
        };
        if let Some(job) = compose_job {
            let response = match compose.clone() {
                Some(bridge) => tokio::task::spawn_blocking(move || job(&bridge))
                    .await
                    .unwrap_or_else(|e| {
                        ControlResponse::Error(format!("compose command failed: {e}"))
                    }),
                None => ControlResponse::Error(
                    "compose: no compose home (set AIEN_COMPOSE_DIR or AIEN_RUNTIME_STATE_DIR)"
                        .into(),
                ),
            };
            if !write_response(&mut writer, &response).await {
                break;
            }
            line.clear();
            continue;
        }

        let is_shutdown = matches!(envelope.command, ControlCommand::Shutdown);
        let response = {
            let mut s = spine.lock().await;
            s.handle_control_command(envelope)
        };
        if is_shutdown {
            is_running.store(false, Ordering::SeqCst);
            shutdown_notify.notify_waiters();
        }
        if !write_response(&mut writer, &response).await {
            break;
        }
        line.clear();
    }
}

/// The correlation trace sink for this daemon (#398). `AIEN_TRACE_JSONL=<file>`
/// turns on a bounded append-only JSON-lines exporter (4096 queued events,
/// drops counted and reported as `trace_dropped`, flushed every 250 ms and
/// once more when the returned handle drops). Unset or empty: `NullSink`.
/// Never fails the daemon: an unwritable file only loses trace lines.
pub fn trace_sink_from_env() -> (
    Arc<dyn aien_trace::TraceSink>,
    Option<aien_trace::FlusherHandle>,
) {
    match std::env::var("AIEN_TRACE_JSONL") {
        Ok(path) if !path.trim().is_empty() => {
            let sink = Arc::new(aien_trace::BoundedJsonlSink::new(path.trim()));
            let flusher = sink.spawn_flusher(std::time::Duration::from_millis(250));
            tracing::info!(path = %sink.path().display(), "trace exporter on (AIEN_TRACE_JSONL)");
            (sink, Some(flusher))
        }
        _ => (Arc::new(aien_trace::NullSink), None),
    }
}
#[cfg(test)]
mod generation_record_tests {
    use super::*;

    fn identity() -> Arc<ModelIdentity> {
        Arc::new(ModelIdentity {
            model_sha256: "a".repeat(64),
            model_digest_kind: "file".into(),
            model_path: "/m/model.safetensors".into(),
            tokenizer_sha256: "b".repeat(64),
            tokenizer_path: "/m/tokenizer.json".into(),
        })
    }

    fn turn() -> GenerationOwned {
        GenerationOwned {
            prompt_ids: vec![1, 2],
            output_ids: vec![3],
            text: "t".into(),
            total_tokens: 1,
            finish_reason: "eos",
            request_id: 1,
            operation_id: 1,
            decoding: None,
            ops: None,
        }
    }

    fn bridge(dir: std::path::PathBuf) -> Arc<ComposeBridge> {
        let proposer: ComposeProposer =
            Arc::new(|_p: &str, _l: std::time::Duration| Err("not used".to_string()));
        Arc::new(ComposeBridge::new(dir, proposer, "test:generation-record"))
    }

    #[tokio::test]
    async fn no_ledger_means_no_record() {
        let got = record_generation(None, Some(identity()), DaemonStart::now(), turn()).await;
        assert_eq!(got, None);
    }

    #[tokio::test]
    async fn no_model_identity_means_no_record() {
        let tmp = tempfile::tempdir().unwrap();
        let b = bridge(tmp.path().join("compose"));
        let got = record_generation(Some(b), None, DaemonStart::now(), turn()).await;
        assert_eq!(got, None);
        // Nothing was opened or written.
        assert!(!tmp.path().join("compose").exists());
    }

    /// A ledger that cannot be written (here: its directory is a regular
    /// file, and in a stub build the engine is absent) yields no id.
    #[tokio::test]
    async fn a_failed_append_returns_no_id() {
        let _home = crate::home_guard::home_slot();
        let tmp = tempfile::tempdir().unwrap();
        let blocked = tmp.path().join("compose");
        std::fs::write(&blocked, b"not a directory").unwrap();
        let got = record_generation(
            Some(bridge(blocked)),
            Some(identity()),
            DaemonStart::now(),
            turn(),
        )
        .await;
        assert_eq!(got, None);
    }
}
