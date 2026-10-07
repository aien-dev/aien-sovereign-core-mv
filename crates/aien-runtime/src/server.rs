//! Local UNIX domain socket server for AienRuntimeSpine
//! Exposes typed control RPC over local IPC to operator CLI tools.

use crate::control::{ControlCommand, ControlEnvelope, ControlResponse};
use crate::spine::{compose_dir_from_env, AienRuntimeSpine, ComposeBridge, ComposeProposer};
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

pub struct AienRuntimeServer {
    socket_path: PathBuf,
    spine: Arc<Mutex<AienRuntimeSpine>>,
    shutdown_notify: Arc<Notify>,
    is_running: Arc<AtomicBool>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    warm_up: AtomicBool,
}

/// NEXT-PHASE-1 v4 declared warm-up prompt (ACCEPTANCE-v4 Section 2(2)):
/// fixed text, more than one 128-token prefill chunk once templated, so
/// both the full-chunk and the remainder prefill paths run before serving.
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
        }
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
                        Ok(Some(m)) if step_log => eprintln!(
                            "step: prefill_tokens={} emitted={} backend_us={} spine_us={}",
                            m.prefill_tokens_processed,
                            m.decode_tokens_emitted,
                            m.step_latency_us,
                            t_step.elapsed().as_micros()
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
            self.run_warm_up().await;
        }

        // NEXT-PHASE-1: the compose bridge (opened on the first RunComposeTask).
        let compose = match compose_dir_from_env() {
            Ok(dir) => Some(Arc::new(ComposeBridge::new(
                dir,
                model_proposer(
                    self.spine.clone(),
                    self.tokenizer.clone(),
                    tokio::runtime::Handle::current(),
                ),
                "model:StreamTurn-path",
            ))),
            Err(e) => {
                tracing::warn!("compose bridge disabled: {e}");
                None
            }
        };
        // NEXT-PHASE-2 (ACCEPTANCE-v2 2.4): settle effects a previous process
        // left open, before serving anything. Reads the world, never re-runs.
        // A failed or refused reconcile does not stop the daemon, but every
        // effect command refuses until an operator reconcile succeeds
        // (ACCEPTANCE-v3 2.5).
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
                            tokio::spawn(async move {
                                handle_connection(
                                    stream,
                                    spine_conn,
                                    is_running_conn,
                                    notify_conn,
                                    tokenizer_conn,
                                    compose_conn,
                                )
                                .await;
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
    /// Prints `Warm-up: 1 token in <ms> ms over <n> prompt tokens (discarded)`
    /// or `Warm-up: FAILED ...`; a failure does not stop the daemon.
    async fn run_warm_up(&self) {
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
            Ok(g) => println!(
                "  Warm-up: {} token in {ms} ms over {} prompt tokens (discarded)",
                g.tokens,
                prompt_tokens.unwrap_or(0)
            ),
            Err(e) => println!("  Warm-up: FAILED after {ms} ms: {e}"),
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
) -> Result<crate::spine::Generation, String> {
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
                Some(CompletionEvent::Finished { finish_reason, .. }) => {
                    return tokenizer
                        .decode_opts(&produced, true)
                        .map(|text| crate::spine::Generation {
                            text,
                            tokens: produced.len(),
                            finish_reason: Some(finish_reason_label(&finish_reason).to_string()),
                            token_ids: Some(produced.clone()),
                            prompt_tokens: Some(prompt_ids.len()),
                            prompt_ids_sha256: Some(crate::spine::token_ids_sha256(&prompt_ids)),
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
        .map_err(|_| format!("model proposal exceeded {} ms", limit.as_millis()))?
}

/// The compose "model" Skill: real inference through `generate_text`, run
/// from an omega World worker thread (not a tokio thread) via `block_on`.
/// Greedy, at most `AIEN_COMPOSE_MAX_TOKENS` (default 48) tokens per reply;
/// each call gets the limit `propose_with_retries` passes (what is left of
/// the 29 s budget, under rx_compose_run's 30 s quiescence wait), so a slow
/// model fails the Skill (no proposal) instead of failing the run.
pub fn model_proposer(
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    handle: tokio::runtime::Handle,
) -> ComposeProposer {
    let max_tokens = std::env::var("AIEN_COMPOSE_MAX_TOKENS")
        .ok()
        .and_then(|v| v.trim().parse::<usize>().ok())
        .unwrap_or(48);
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
            .map(|t| crate::spine::compose_assistant_prefix(&t.template()))
            .unwrap_or(crate::spine::COMPOSE_ASSISTANT_PREFIX);
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
            .map(|g| crate::spine::Generation {
                text: format!("{prefix}{}", g.text),
                ..g
            })
    })
}

async fn stream_turn(
    writer: &mut tokio::net::unix::OwnedWriteHalf,
    spine: Arc<Mutex<AienRuntimeSpine>>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    messages: Vec<crate::control::ChatTurn>,
    max_tokens: usize,
    temperature: f32,
) {
    let (tokenizer, mut events, _prompt_ids) =
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
            Ok(Some(CompletionEvent::Finished { total_tokens, .. })) => {
                let _ = write_response(
                    writer,
                    &ControlResponse::TurnFinished { text, total_tokens },
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

async fn handle_connection(
    stream: UnixStream,
    spine: Arc<Mutex<AienRuntimeSpine>>,
    is_running: Arc<AtomicBool>,
    shutdown_notify: Arc<Notify>,
    tokenizer: Arc<RwLock<Option<ChatTokenizer>>>,
    compose: Option<Arc<ComposeBridge>>,
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
            )
            .await;
            line.clear();
            continue;
        }

        let compose_job: Option<ComposeJob> = match envelope.command {
            ControlCommand::RunComposeTask {
                ref goal,
                ref workspace,
            } => {
                let (g, w) = (goal.clone(), workspace.clone());
                Some(Box::new(move |b: &Arc<ComposeBridge>| b.run_task(&g, &w)))
            }
            ControlCommand::ComposeNote {
                ref kind,
                ref text,
                ref links,
            } => {
                let (k, t, l) = (kind.clone(), text.clone(), links.clone());
                // NEXT-PHASE-2: gated effect/control records cannot be forged here.
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
                    match crate::effects::check_reserved_note(&k, &t) {
                        Ok(()) => b.note(&k, &t, &l),
                        Err(e) => ControlResponse::Error(e),
                    }
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
                Some(Box::new(move |b: &Arc<ComposeBridge>| {
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
