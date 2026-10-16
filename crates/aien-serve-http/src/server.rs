//! The HTTP surface: `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE),
//! `GET /health`. Every error is OpenAI error JSON. One generation at a time: the
//! second request while one runs gets 503 `server_busy` with `Retry-After: 1`.

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use tokio::sync::{mpsc, Mutex, OwnedMutexGuard};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use crate::codec::{Codec, CodecError};
use crate::engine::{Engine, EngineError, FinishReason, GenerateParams};
use crate::openai::{
    chunk_json, completion_json, ChatRequest, ErrorBody, ModelCard, ModelList, Usage,
};
use crate::sampling::SamplingRecord;
use crate::text::{Push, TextAssembler};

#[derive(Debug, Clone)]
pub struct ServerConfig {
    /// Must be a loopback address; `serve` refuses anything else.
    pub bind: IpAddr,
    /// `0` picks a free port (tests).
    pub port: u16,
    pub model_id: String,
    pub max_body_bytes: usize,
    pub max_messages: usize,
    pub max_stop_strings: usize,
    /// Upper bound on `max_tokens`; larger requests are clamped to it.
    pub max_new_tokens_cap: usize,
    pub default_max_new_tokens: usize,
    pub default_temperature: f32,
    pub default_top_p: f32,
    pub default_top_k: usize,
    /// Optional server-side context cap (min with the engine's own limit).
    pub max_context_tokens: Option<usize>,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            bind: IpAddr::V4(Ipv4Addr::LOCALHOST),
            port: 18091,
            model_id: "qwen3-30b-a3b-instruct-2507".into(),
            max_body_bytes: 8 * 1024 * 1024,
            max_messages: 512,
            max_stop_strings: 8,
            max_new_tokens_cap: 4096,
            default_max_new_tokens: 1024,
            default_temperature: 0.7,
            default_top_p: 0.8,
            default_top_k: 20,
            max_context_tokens: None,
        }
    }
}

pub struct AppState {
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    cfg: ServerConfig,
    gate: Arc<Mutex<()>>,
    started: Instant,
    counter: AtomicU64,
}

/// An OpenAI-shaped error with its HTTP status.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    body: ErrorBody,
    retry_after: Option<u64>,
}

impl ApiError {
    fn bad_request(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::BAD_REQUEST,
            body: ErrorBody::new("invalid_request_error", msg),
            retry_after: None,
        }
    }

    fn context_exceeded(msg: impl Into<String>) -> Self {
        let mut e = Self::bad_request(msg);
        e.body.error.code = Some("context_length_exceeded".into());
        e
    }

    fn not_found(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::NOT_FOUND,
            body: ErrorBody::new("not_found_error", msg),
            retry_after: None,
        }
    }

    fn too_large(max: usize) -> Self {
        Self {
            status: StatusCode::PAYLOAD_TOO_LARGE,
            body: ErrorBody::new(
                "invalid_request_error",
                format!("request body exceeds the limit of {max} bytes"),
            ),
            retry_after: None,
        }
    }

    fn busy() -> Self {
        Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: ErrorBody::new(
                "server_busy",
                "one generation at a time: the engine is busy with another request, retry shortly",
            ),
            retry_after: Some(1),
        }
    }

    fn internal(msg: impl Into<String>) -> Self {
        Self {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            body: ErrorBody::new("server_error", msg),
            retry_after: None,
        }
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let mut resp = (self.status, Json(self.body)).into_response();
        if let Some(s) = self.retry_after {
            resp.headers_mut().insert(
                header::RETRY_AFTER,
                HeaderValue::from_str(&s.to_string()).expect("digits"),
            );
        }
        resp
    }
}

struct Prepared {
    prompt_tokens: Vec<u32>,
    params: GenerateParams,
    stop_strings: Vec<String>,
    stream: bool,
}

/// What the generation thread sends to the HTTP side.
enum Ev {
    Delta(String),
    Done {
        finish: &'static str,
        completion_tokens: usize,
        sampling: Option<SamplingRecord>,
    },
    Error(String),
    /// The engine could not get the shared chip: 503 before any token.
    Busy(String),
}

fn unix_now() -> (u64, u128) {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    (d.as_secs(), d.as_nanos())
}

pub fn router(state: Arc<AppState>) -> Router {
    let limit = state.cfg.max_body_bytes;
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/:id", get(get_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .fallback(not_found)
        .layer(DefaultBodyLimit::max(limit))
        .with_state(state)
}

/// Binds `cfg.bind:cfg.port` (loopback only) and serves in a background task.
/// Returns the bound address (useful with port 0) and the task handle.
pub async fn serve(
    cfg: ServerConfig,
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
) -> std::io::Result<(SocketAddr, JoinHandle<()>)> {
    serve_until(cfg, engine, codec, std::future::pending()).await
}

/// [`serve`] until `shutdown` resolves: then no new connection is accepted, the requests in
/// flight finish (a streaming generation runs to its end) and the task ends, releasing the
/// server's hold on `engine`.
pub async fn serve_until(
    cfg: ServerConfig,
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    shutdown: impl std::future::Future<Output = ()> + Send + 'static,
) -> std::io::Result<(SocketAddr, JoinHandle<()>)> {
    if !cfg.bind.is_loopback() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            format!("refusing to bind {}: loopback addresses only", cfg.bind),
        ));
    }
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(cfg.bind, cfg.port)).await?;
    let addr = listener.local_addr()?;
    let state = Arc::new(AppState {
        engine,
        codec,
        cfg,
        gate: Arc::new(Mutex::new(())),
        started: Instant::now(),
        counter: AtomicU64::new(0),
    });
    let app = router(state);
    let handle = tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app)
            .with_graceful_shutdown(shutdown)
            .await
        {
            eprintln!("aien-serve-http: server stopped: {e}");
        }
    });
    Ok((addr, handle))
}

fn model_card(state: &AppState) -> ModelCard {
    ModelCard {
        id: state.cfg.model_id.clone(),
        object: "model",
        created: unix_now().0 - state.started.elapsed().as_secs(),
        owned_by: format!("aien/{}", state.engine.kind()),
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: vec![model_card(&state)],
    })
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ModelCard>, ApiError> {
    if id != state.cfg.model_id {
        return Err(ApiError::not_found(format!(
            "model '{id}' not found; this server serves '{}'",
            state.cfg.model_id
        )));
    }
    Ok(Json(model_card(&state)))
}

async fn health(State(state): State<Arc<AppState>>) -> Json<Value> {
    let mut v = serde_json::json!({
        "status": "ok",
        "model": state.cfg.model_id,
        "engine": state.engine.kind(),
        "busy": state.gate.try_lock().is_err(),
        "uptime_s": state.started.elapsed().as_secs(),
    });
    // Persistent KV (sc#505): present only when AIEN_KV_STORE_DIR asked for it.
    if let (Some(pk), Some(obj)) = (state.engine.persistence_status(), v.as_object_mut()) {
        obj.insert("persistent_kv".into(), pk);
    }
    Json(v)
}

async fn not_found() -> ApiError {
    ApiError::not_found(
        "no such route; this server exposes /v1/models, /v1/chat/completions and /health",
    )
}

async fn chat_completions(
    State(state): State<Arc<AppState>>,
    body: Result<Bytes, BytesRejection>,
) -> Response {
    match handle_chat(state, body).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

async fn handle_chat(
    state: Arc<AppState>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Response, ApiError> {
    let body = body.map_err(|r| {
        if r.status() == StatusCode::PAYLOAD_TOO_LARGE {
            ApiError::too_large(state.cfg.max_body_bytes)
        } else {
            ApiError::bad_request(format!("could not read request body: {}", r.body_text()))
        }
    })?;
    let req: ChatRequest = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON request: {e}")))?;
    // Rendering and tokenizing a long prompt is CPU work: run it on a blocking thread so it never
    // stalls the async runtime (sc#522).
    let prepared = {
        let state = Arc::clone(&state);
        tokio::task::spawn_blocking(move || prepare(&state, req))
            .await
            .map_err(|e| ApiError::internal(format!("request preparation failed: {e}")))??
    };
    let guard = Arc::clone(&state.gate)
        .try_lock_owned()
        .map_err(|_| ApiError::busy())?;

    let (created, nanos) = unix_now();
    let n = state.counter.fetch_add(1, Ordering::Relaxed);
    let id = format!("chatcmpl-{nanos:x}{n:04x}");
    let prompt_tokens = prepared.prompt_tokens.len();
    let stream = prepared.stream;
    let (tx, mut rx) = mpsc::channel::<Ev>(64);
    let engine = Arc::clone(&state.engine);
    let codec = Arc::clone(&state.codec);
    let model = state.cfg.model_id.clone();
    let log_id = id.clone();
    tokio::task::spawn_blocking(move || {
        run_generation(engine, codec, prepared, tx, guard, &log_id);
    });

    // The first event decides the status: a busy chip is a 503 with a plain body, before any
    // headers go out (streaming included). Everything else is forwarded unchanged.
    let first = rx.recv().await;
    if let Some(Ev::Busy(_)) = first {
        let mut resp = (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({"error": "GPU busy with another job"})),
        )
            .into_response();
        resp.headers_mut()
            .insert(header::RETRY_AFTER, HeaderValue::from_static("5"));
        return Ok(resp);
    }
    let (ftx, frx) = mpsc::channel::<Ev>(64);
    tokio::spawn(async move {
        if let Some(ev) = first {
            if ftx.send(ev).await.is_err() {
                return;
            }
        }
        while let Some(ev) = rx.recv().await {
            if ftx.send(ev).await.is_err() {
                return;
            }
        }
    });
    let rx = frx;
    if stream {
        Ok(stream_response(id, created, model, prompt_tokens, rx))
    } else {
        collect_response(id, created, &model, prompt_tokens, rx).await
    }
}

fn text_of_content(content: &Value) -> Result<String, ApiError> {
    match content {
        Value::Null => Ok(String::new()),
        Value::String(s) => Ok(s.clone()),
        Value::Array(parts) => {
            let mut out = String::new();
            for part in parts {
                match part.get("type").and_then(Value::as_str) {
                    Some("text") => out.push_str(
                        part.get("text")
                            .and_then(Value::as_str)
                            .ok_or_else(|| ApiError::bad_request("text part without 'text'"))?,
                    ),
                    other => {
                        return Err(ApiError::bad_request(format!(
                            "unsupported content part type {:?}: only text parts are accepted",
                            other.unwrap_or("missing")
                        )))
                    }
                }
            }
            Ok(out)
        }
        _ => Err(ApiError::bad_request(
            "message content must be a string or an array of text parts",
        )),
    }
}

fn prepare(state: &AppState, req: ChatRequest) -> Result<Prepared, ApiError> {
    let cfg = &state.cfg;
    if let Some(m) = req.model.as_deref() {
        if m != cfg.model_id {
            return Err(ApiError::not_found(format!(
                "model '{m}' not found; this server serves '{}'",
                cfg.model_id
            )));
        }
    }
    if matches!(req.n, Some(n) if n != 1) {
        return Err(ApiError::bad_request("n must be 1"));
    }
    if req.tools.as_ref().is_some_and(|t| !t.is_empty())
        || req.functions.as_ref().is_some_and(|f| !f.is_empty())
    {
        return Err(ApiError::bad_request(
            "tool calling is not supported by this server (v1 is plain text)",
        ));
    }
    if req.messages.is_empty() {
        return Err(ApiError::bad_request("messages must be a non-empty array"));
    }
    if req.messages.len() > cfg.max_messages {
        return Err(ApiError::bad_request(format!(
            "too many messages: {} (limit {})",
            req.messages.len(),
            cfg.max_messages
        )));
    }
    let mut turns: Vec<(String, String)> = Vec::with_capacity(req.messages.len());
    for m in &req.messages {
        let role = match m.role.trim().to_ascii_lowercase().as_str() {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            "tool" | "function" => {
                return Err(ApiError::bad_request(
                    "role tool is not supported by this server (v1 is plain text)",
                ))
            }
            other => {
                return Err(ApiError::bad_request(format!(
                    "unknown message role '{other}'"
                )))
            }
        };
        turns.push((role.to_string(), text_of_content(&m.content)?));
    }
    let turn_refs: Vec<(&str, &str)> = turns
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();
    let prompt_text = state.codec.render_chat(&turn_refs).map_err(|e| match e {
        CodecError::Unrenderable(m) => ApiError::bad_request(m),
        CodecError::Failed(m) => ApiError::internal(m),
    })?;
    let prompt_tokens = state
        .codec
        .encode(&prompt_text)
        .map_err(|e| ApiError::context_exceeded(format!("could not tokenize prompt: {e}")))?;
    if prompt_tokens.is_empty() {
        return Err(ApiError::bad_request("the rendered prompt is empty"));
    }

    let ctx = cfg
        .max_context_tokens
        .unwrap_or(usize::MAX)
        .min(state.engine.max_context_tokens());
    if prompt_tokens.len() >= ctx {
        return Err(ApiError::context_exceeded(format!(
            "prompt is {} tokens; the context window is {} tokens and at least one token must be left for the reply",
            prompt_tokens.len(),
            ctx
        )));
    }
    let requested = match req.max_tokens.or(req.max_completion_tokens) {
        Some(v) if v < 1 => return Err(ApiError::bad_request("max_tokens must be at least 1")),
        Some(v) => v as usize,
        None => cfg.default_max_new_tokens,
    };
    let max_new_tokens = requested
        .min(cfg.max_new_tokens_cap)
        .min(ctx - prompt_tokens.len());

    let temperature = req.temperature.unwrap_or(cfg.default_temperature);
    if !(0.0..=2.0).contains(&temperature) {
        return Err(ApiError::bad_request("temperature must be between 0 and 2"));
    }
    let top_p = req.top_p.unwrap_or(cfg.default_top_p);
    if !(top_p > 0.0 && top_p <= 1.0) {
        return Err(ApiError::bad_request("top_p must be in (0, 1]"));
    }
    let top_k = match req.top_k {
        Some(k) if k < 0 => return Err(ApiError::bad_request("top_k must be >= 0")),
        Some(k) => k as usize,
        None => cfg.default_top_k,
    };
    let seed = match req.seed {
        Some(s) if s < 0 => return Err(ApiError::bad_request("seed must be >= 0")),
        Some(s) => s as u64,
        None => (unix_now().1 as u64) ^ state.counter.load(Ordering::Relaxed).rotate_left(32),
    };
    let stop_strings = match req.stop {
        None | Some(Value::Null) => Vec::new(),
        Some(Value::String(s)) => vec![s],
        Some(Value::Array(items)) => items
            .into_iter()
            .map(|v| match v {
                Value::String(s) => Ok(s),
                _ => Err(ApiError::bad_request(
                    "stop must be a string or an array of strings",
                )),
            })
            .collect::<Result<Vec<_>, _>>()?,
        Some(_) => {
            return Err(ApiError::bad_request(
                "stop must be a string or an array of strings",
            ))
        }
    };
    if stop_strings.iter().any(String::is_empty) {
        return Err(ApiError::bad_request("stop strings must be non-empty"));
    }
    if stop_strings.len() > cfg.max_stop_strings {
        return Err(ApiError::bad_request(format!(
            "at most {} stop strings are accepted",
            cfg.max_stop_strings
        )));
    }

    Ok(Prepared {
        prompt_tokens,
        params: GenerateParams {
            max_new_tokens,
            temperature,
            top_p,
            top_k,
            seed,
            stop_token_ids: state.codec.stop_token_ids(),
        },
        stop_strings,
        stream: req.stream,
    })
}

/// Runs on a blocking thread holding the generation lock (`guard`). The lock is released as soon
/// as the engine returns, before the final event is sent, so a client that sends its next request
/// the moment it has the reply never meets a stale 503 `server_busy`.
fn run_generation(
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    prepared: Prepared,
    tx: mpsc::Sender<Ev>,
    guard: OwnedMutexGuard<()>,
    id: &str,
) {
    let mut asm = TextAssembler::new(codec, prepared.stop_strings);
    let mut stopped_by_string = false;
    let mut client_gone = false;
    let result = engine.generate(
        &prepared.prompt_tokens,
        &prepared.params,
        &mut |tok| match asm.push(tok) {
            Push::Hold => true,
            Push::Delta(d) => {
                if tx.blocking_send(Ev::Delta(d)).is_err() {
                    client_gone = true;
                    false
                } else {
                    true
                }
            }
            Push::StopAt(d) => {
                stopped_by_string = true;
                if !d.is_empty() {
                    let _ = tx.blocking_send(Ev::Delta(d));
                }
                false
            }
        },
    );
    drop(guard);
    match result {
        Err(EngineError::Busy(m)) => {
            eprintln!("aien-serve-http: {id}: GPU busy: {m}");
            let _ = tx.blocking_send(Ev::Busy(m));
        }
        Err(e) => {
            eprintln!("aien-serve-http: {id}: {e}");
            let _ = tx.blocking_send(Ev::Error(e.to_string()));
        }
        Ok(out) => {
            if client_gone {
                eprintln!(
                    "aien-serve-http: {id}: client left after {} tokens",
                    out.emitted_tokens
                );
                return;
            }
            if let Some(rest) = asm.finish() {
                let _ = tx.blocking_send(Ev::Delta(rest));
            }
            let finish = if stopped_by_string {
                "stop"
            } else {
                match out.finish {
                    FinishReason::Stop | FinishReason::Cancelled => "stop",
                    FinishReason::Length => "length",
                }
            };
            let tps = if out.decode_ms > 0 {
                out.emitted_tokens as f64 * 1000.0 / out.decode_ms as f64
            } else {
                0.0
            };
            let sampling_fields = out
                .sampling
                .as_ref()
                .map(|s| format!(", {}", s.log_fields()))
                .unwrap_or_default();
            eprintln!(
                "aien-serve-http: {id}: prompt {} tok, reply {} tok, finish {finish}, prefill {} ms, decode {} ms ({tps:.2} tok/s){sampling_fields}",
                prepared.prompt_tokens.len(),
                out.emitted_tokens,
                out.prefill_ms,
                out.decode_ms
            );
            let _ = tx.blocking_send(Ev::Done {
                finish,
                completion_tokens: out.emitted_tokens,
                sampling: out.sampling,
            });
        }
    }
}

async fn collect_response(
    id: String,
    created: u64,
    model: &str,
    prompt_tokens: usize,
    mut rx: mpsc::Receiver<Ev>,
) -> Result<Response, ApiError> {
    let mut content = String::new();
    while let Some(ev) = rx.recv().await {
        match ev {
            Ev::Delta(d) => content.push_str(&d),
            Ev::Done {
                finish,
                completion_tokens,
                sampling,
            } => {
                let usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    aien_sampling: sampling,
                };
                return Ok(Json(completion_json(
                    &id, created, model, &content, finish, &usage,
                ))
                .into_response());
            }
            Ev::Error(m) => return Err(ApiError::internal(m)),
            Ev::Busy(m) => return Err(ApiError::internal(m)),
        }
    }
    Err(ApiError::internal("generation ended without a result"))
}

fn stream_response(
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    mut rx: mpsc::Receiver<Ev>,
) -> Response {
    let (etx, erx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        let data = |v: Value| Ok(Event::default().data(v.to_string()));
        let first = chunk_json(
            &id,
            created,
            &model,
            Some("assistant"),
            Some(""),
            None,
            None,
        );
        if etx.send(data(first)).await.is_err() {
            return;
        }
        while let Some(ev) = rx.recv().await {
            let item = match ev {
                Ev::Delta(d) => data(chunk_json(&id, created, &model, None, Some(&d), None, None)),
                Ev::Done {
                    finish,
                    completion_tokens,
                    sampling,
                } => {
                    let usage = Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        aien_sampling: sampling,
                    };
                    let last =
                        chunk_json(&id, created, &model, None, None, Some(finish), Some(&usage));
                    if etx.send(data(last)).await.is_err() {
                        return;
                    }
                    let _ = etx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
                Ev::Error(m) | Ev::Busy(m) => {
                    let err = serde_json::to_value(ErrorBody::new("server_error", m))
                        .expect("error serializes");
                    if etx.send(data(err)).await.is_err() {
                        return;
                    }
                    let _ = etx.send(Ok(Event::default().data("[DONE]"))).await;
                    return;
                }
            };
            if etx.send(item).await.is_err() {
                return;
            }
        }
    });
    Sse::new(ReceiverStream::new(erx))
        .keep_alive(
            KeepAlive::new()
                .interval(Duration::from_secs(15))
                .text("keep-alive"),
        )
        .into_response()
}
