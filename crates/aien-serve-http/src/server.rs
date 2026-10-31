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

//! The HTTP surface: `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE),
//! `GET /health`. Every error is OpenAI error JSON. As many generations at a time as the
//! engine says it takes ([`Engine::concurrency`], 1 unless the engine batches): the next
//! request while all those run gets 503 `server_busy` with `Retry-After: 1`.

use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::{to_bytes, Body, Bytes};
use axum::extract::{Path, State};
use axum::http::{header, HeaderMap, HeaderValue, Request, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use aien_inference_abi::tokenizer::ChatOptions;

use crate::codec::{Codec, CodecError};
use crate::engine::{Engine, EngineError, FinishReason, GenerateParams, TestSwitches};
use crate::openai::{
    chunk_json, chunk_json_with, completion_json, ChatRequest, ErrorBody, ModelCard, ModelList,
    Usage,
};
use crate::sampling::SamplingRecord;
use crate::text::{Push, TextAssembler};

#[derive(Clone)]
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
    /// `enable_thinking` for requests that do not set it (`None`: the chat template's own
    /// default). Only templates with a thinking switch read it (Qwen3.8-27B).
    pub default_enable_thinking: Option<bool>,
    /// The eight readers (sc L371): a second model id that reads a long input in parts at once.
    /// `None`: one id, as before.
    pub readers: Option<crate::readers::ReadersConfig>,
    /// Secret supplied out of band; an absent or empty credential refuses startup.
    pub bearer_token: String,
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
            default_enable_thinking: None,
            readers: None,
            bearer_token: String::new(),
        }
    }
}

pub struct AppState {
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    cfg: ServerConfig,
    /// One permit per generation the engine runs at once.
    gate: Arc<Semaphore>,
    allowed_hosts: [String; 3],
    started: Instant,
    counter: AtomicU64,
    /// Stable-first tool order and tool boundary marks for the door's prompt cache (sc L375).
    tool_cache: crate::tool_cache::ToolCache,
}

/// An OpenAI-shaped error with its HTTP status.
#[derive(Debug)]
pub struct ApiError {
    status: StatusCode,
    body: ErrorBody,
    retry_after: Option<u64>,
}

impl ApiError {
    fn unauthorized() -> Self {
        Self {
            status: StatusCode::UNAUTHORIZED,
            body: ErrorBody::new(
                "authentication_error",
                "valid bearer authorization required",
            ),
            retry_after: None,
        }
    }

    fn forbidden(msg: &'static str) -> Self {
        Self {
            status: StatusCode::FORBIDDEN,
            body: ErrorBody::new("invalid_request_error", msg),
            retry_after: None,
        }
    }

    fn unsupported_media_type() -> Self {
        Self {
            status: StatusCode::UNSUPPORTED_MEDIA_TYPE,
            body: ErrorBody::new(
                "invalid_request_error",
                "Content-Type must be application/json",
            ),
            retry_after: None,
        }
    }
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
                "every generation slot is taken (the engine's concurrency): retry shortly",
            ),
            retry_after: Some(1),
        }
    }

    /// The engine could not get the shared chip in time (another job holds the GPU lock).
    fn gpu_busy(detail: &str) -> Self {
        let mut e = Self {
            status: StatusCode::SERVICE_UNAVAILABLE,
            body: ErrorBody::new(
                "server_busy",
                format!("GPU busy with another job, retry shortly: {detail}"),
            ),
            retry_after: Some(5),
        };
        e.body.error.code = Some("gpu_busy".into());
        e
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
    /// Prompt cache checkpoint hints (tool boundaries) for the engine.
    marks: Vec<usize>,
    params: GenerateParams,
    stop_strings: Vec<String>,
    stream: bool,
    /// The prompt opens a thinking block: reply tokens up to this id are reasoning.
    reasoning_end: Option<u32>,
    return_ids: bool,
    /// The request carried tools: parse `<tool_call>` blocks out of the reply.
    tools: bool,
    /// The merge train's switches for this request (empty: none; `aien_test` absent).
    test: TestSwitches,
}

/// What the generation thread sends to the HTTP side. `Done` is sent once per request, so its size
/// (past clippy's line since `aien_test` joined it) costs nothing.
#[allow(clippy::large_enum_variant)]
enum Ev {
    Delta(String),
    /// Text of the thinking block (`reasoning_content`).
    Reasoning(String),
    Done {
        finish: &'static str,
        completion_tokens: usize,
        sampling: Option<SamplingRecord>,
        token_ids: Option<Vec<u32>>,
        /// The `aien_test` switches applied (`usage.aien_test.applied`), when the request set any.
        test: Option<Vec<&'static str>>,
    },
    Error(String),
    /// The engine could not get the shared chip: 503 before any token.
    Busy(String),
    /// Readers, streaming: the reading has started (the response goes out; nothing is sent).
    Reading,
    /// Readers: what each reader found (`aien_readers`) and the connect prompt's length.
    Readers {
        info: Value,
        prompt_tokens: usize,
    },
}

fn unix_now() -> (u64, u128) {
    let d = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or(Duration::ZERO);
    (d.as_secs(), d.as_nanos())
}

pub fn router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/models", get(list_models))
        .route("/v1/models/:id", get(get_model))
        .route("/v1/chat/completions", post(chat_completions))
        .route("/health", get(health))
        .fallback(not_found)
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
    if cfg.bearer_token.is_empty() || !cfg.bearer_token.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "a nonempty printable bearer token is required before binding",
        ));
    }
    let listener = tokio::net::TcpListener::bind(SocketAddr::new(cfg.bind, cfg.port)).await?;
    let addr = listener.local_addr()?;
    let gate = Arc::new(Semaphore::new(engine.concurrency().max(1)));
    let tool_cache = crate::tool_cache::ToolCache::for_engine(engine.kind());
    let state = Arc::new(AppState {
        tool_cache,
        engine,
        codec,
        cfg,
        gate,
        allowed_hosts: [
            "localhost".into(),
            format!("localhost:{}", addr.port()),
            addr.to_string(),
        ],
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
    model_card_for(state, &state.cfg.model_id)
}

fn model_card_for(state: &AppState, id: &str) -> ModelCard {
    ModelCard {
        id: id.to_string(),
        object: "model",
        created: unix_now().0 - state.started.elapsed().as_secs(),
        owned_by: format!("aien/{}", state.engine.kind()),
    }
}

async fn list_models(State(state): State<Arc<AppState>>) -> Json<ModelList> {
    Json(ModelList {
        object: "list",
        data: std::iter::once(model_card(&state))
            .chain(
                state
                    .cfg
                    .readers
                    .as_ref()
                    .map(|r| model_card_for(&state, &r.model_id)),
            )
            .collect(),
    })
}

async fn get_model(
    State(state): State<Arc<AppState>>,
    Path(id): Path<String>,
) -> Result<Json<ModelCard>, ApiError> {
    if let Some(r) = state.cfg.readers.as_ref().filter(|r| r.model_id == id) {
        return Ok(Json(model_card_for(&state, &r.model_id)));
    }
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
        "busy": state.gate.available_permits() == 0,
        "uptime_s": state.started.elapsed().as_secs(),
    });
    // A batching engine says how many it runs at once; one-at-a-time engines leave `/health`
    // exactly as it was.
    let n = state.engine.concurrency();
    if let (true, Some(obj)) = (n > 1, v.as_object_mut()) {
        obj.insert("concurrency".into(), n.into());
        obj.insert(
            "in_flight".into(),
            (n - state.gate.available_permits().min(n)).into(),
        );
    }
    // Persistent KV (sc#505): present only when AIEN_KV_STORE_DIR asked for it.
    if let (Some(pk), Some(obj)) = (state.engine.persistence_status(), v.as_object_mut()) {
        obj.insert("persistent_kv".into(), pk);
    }
    // The merge train (L384): which `aien_test` switches this build honors, so the train marks a
    // lever NOT-ON-TRAIN instead of PASS when its switch is not in the candidate. Test builds only.
    // Absent for an engine with no switches, so its /health stays exactly as it was.
    #[cfg(feature = "aien-test-switches")]
    if let (false, Some(obj)) = (
        state.engine.test_switches_honored().is_empty(),
        v.as_object_mut(),
    ) {
        obj.insert(
            "aien_test".into(),
            serde_json::json!({ "honored": state.engine.test_switches_honored() }),
        );
    }
    Json(v)
}

async fn not_found() -> ApiError {
    ApiError::not_found(
        "no such route; this server exposes /v1/models, /v1/chat/completions and /health",
    )
}

/// Keep all authorization and slot decisions before the first body poll.
async fn chat_completions(State(state): State<Arc<AppState>>, request: Request<Body>) -> Response {
    match handle_chat(state, request).await {
        Ok(r) => r,
        Err(e) => e.into_response(),
    }
}

fn authorized(headers: &HeaderMap, token: &str) -> bool {
    if headers.get_all(header::AUTHORIZATION).iter().count() != 1 {
        return false;
    }
    let Some(value) = headers
        .get(header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok())
    else {
        return false;
    };
    let Some(provided) = value.strip_prefix("Bearer ") else {
        return false;
    };
    let (a, b) = (provided.as_bytes(), token.as_bytes());
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (&x, &y) in a.iter().zip(b) {
        diff |= x ^ y;
    }
    diff == 0
}

fn check_headers(state: &AppState, headers: &HeaderMap) -> Result<(), ApiError> {
    if !authorized(headers, &state.cfg.bearer_token) {
        return Err(ApiError::unauthorized());
    }
    if headers.get_all(header::HOST).iter().count() != 1 {
        return Err(ApiError::forbidden("unrecognized Host"));
    }
    let host = headers
        .get(header::HOST)
        .and_then(|h| h.to_str().ok())
        .ok_or_else(|| ApiError::forbidden("unrecognized Host"))?;
    if !state
        .allowed_hosts
        .iter()
        .any(|h| host.eq_ignore_ascii_case(h))
    {
        return Err(ApiError::forbidden("unrecognized Host"));
    }
    if headers.get_all(header::ORIGIN).iter().count() > 1 {
        return Err(ApiError::forbidden("unrecognized Origin"));
    }
    if let Some(origin) = headers.get(header::ORIGIN) {
        let origin = origin
            .to_str()
            .map_err(|_| ApiError::forbidden("unrecognized Origin"))?;
        let origin_host = origin
            .strip_prefix("http://")
            .ok_or_else(|| ApiError::forbidden("unrecognized Origin"))?;
        if !state.allowed_hosts[1..]
            .iter()
            .any(|h| origin_host.eq_ignore_ascii_case(h))
        {
            return Err(ApiError::forbidden("unrecognized Origin"));
        }
    }
    if headers.get_all(header::CONTENT_TYPE).iter().count() != 1 {
        return Err(ApiError::unsupported_media_type());
    }
    let media = headers
        .get(header::CONTENT_TYPE)
        .and_then(|h| h.to_str().ok())
        .unwrap_or("");
    if !media
        .split(';')
        .next()
        .is_some_and(|kind| kind.eq_ignore_ascii_case("application/json"))
    {
        return Err(ApiError::unsupported_media_type());
    }
    Ok(())
}

async fn handle_chat(state: Arc<AppState>, request: Request<Body>) -> Result<Response, ApiError> {
    check_headers(&state, request.headers())?;
    // A request owns its engine slot while the bounded body is read and while parsing or
    // tokenizing. Readers may reserve additional slots only after their initial admission.
    let guard = Arc::clone(&state.gate)
        .try_acquire_owned()
        .map_err(|_| ApiError::busy())?;
    if request
        .headers()
        .get(header::CONTENT_LENGTH)
        .and_then(|h| h.to_str().ok())
        .and_then(|s| s.parse::<usize>().ok())
        .is_some_and(|n| n > state.cfg.max_body_bytes)
    {
        return Err(ApiError::too_large(state.cfg.max_body_bytes));
    }
    let limit = state.cfg.max_body_bytes;
    let body = tokio::time::timeout(
        Duration::from_secs(15),
        to_bytes(request.into_body(), limit),
    )
    .await
    .map_err(|_| ApiError::bad_request("request body read timed out"))?
    .map_err(|_| ApiError::too_large(limit))?;
    let pending = tokio::task::spawn_blocking(move || prepare_chat(state, body, guard))
        .await
        .map_err(|_| ApiError::internal("request preparation failed"))??;
    respond(
        pending.rx,
        pending.stream,
        pending.id,
        pending.created,
        pending.model,
        pending.prompt_tokens,
        pending.tools,
    )
    .await
}

struct PendingResponse {
    rx: mpsc::Receiver<Ev>,
    stream: bool,
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    tools: bool,
}

fn prepare_chat(
    state: Arc<AppState>,
    body: Bytes,
    guard: OwnedSemaphorePermit,
) -> Result<PendingResponse, ApiError> {
    let mut guard = Some(guard);
    let mut raw: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON request: {e}")))?;
    let mut model = state.cfg.model_id.clone();
    if let Some(rc) = state.cfg.readers.as_ref() {
        if raw.get("model").and_then(Value::as_str) == Some(rc.model_id.as_str()) {
            raw["model"] = Value::String(state.cfg.model_id.clone());
            if let Some(r) = handle_readers(&state, rc, &raw, &mut guard)? {
                return Ok(r);
            }
            model = rc.model_id.clone();
        }
    }
    let req: ChatRequest = serde_json::from_value(raw)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON request: {e}")))?;
    let prepared = prepare(&state, req)?;
    let (created, id) = new_id(&state);
    let prompt_tokens = prepared.prompt_tokens.len();
    let stream = prepared.stream;
    let tools = prepared.tools;
    let (tx, rx) = mpsc::channel::<Ev>(64);
    let engine = Arc::clone(&state.engine);
    let codec = Arc::clone(&state.codec);
    let log_id = id.clone();
    tokio::task::spawn_blocking(move || {
        run_generation(
            engine,
            codec,
            prepared,
            tx,
            guard.expect("admitted slot"),
            &log_id,
        );
    });
    Ok(PendingResponse {
        rx,
        stream,
        id,
        created,
        model,
        prompt_tokens,
        tools,
    })
}

fn new_id(state: &AppState) -> (u64, String) {
    let (created, nanos) = unix_now();
    let n = state.counter.fetch_add(1, Ordering::Relaxed);
    (created, format!("chatcmpl-{nanos:x}{n:04x}"))
}

/// A request to the readers id: `Ok(None)` when its long input is under the threshold (the caller
/// serves it on the plain path). Otherwise the rows that are free now (up to `rc.readers`; one
/// free row of a batching engine is 503 `server_busy`) each read one part, and one of them writes
/// the answer from the notes.
fn handle_readers(
    state: &Arc<AppState>,
    rc: &crate::readers::ReadersConfig,
    raw: &Value,
    first: &mut Option<OwnedSemaphorePermit>,
) -> Result<Option<PendingResponse>, ApiError> {
    use crate::readers;
    let msgs: &[Value] = raw
        .get("messages")
        .and_then(Value::as_array)
        .map_or(&[], Vec::as_slice);
    let count = |s: &str| state.codec.encode(s).map_or(s.len() / 4, |t| t.len());
    let Some(mut plan) = readers::plan(msgs, rc, &count) else {
        return Ok(None);
    };
    let slots = state.engine.concurrency().max(1);
    let want = plan.parts.len().min(slots);
    let (extra, got) = (1..want)
        .rev()
        .find_map(|k| {
            Arc::clone(&state.gate)
                .try_acquire_many_owned(k as u32)
                .ok()
                .map(|p| (Some(p), k + 1))
        })
        .unwrap_or((None, 1));
    if slots > 1 && got < 2 {
        return Err(ApiError::busy());
    }
    if slots > 1 && got < plan.parts.len() {
        let fewer = readers::ReadersConfig {
            readers: got,
            ..rc.clone()
        };
        plan = readers::plan(msgs, &fewer, &count)
            .ok_or_else(|| ApiError::internal("readers: the plan vanished with fewer readers"))?;
    }
    let base = state.cfg.model_id.as_str();
    let from = |v: Value| -> Result<Prepared, ApiError> {
        let req: ChatRequest = serde_json::from_value(v)
            .map_err(|e| ApiError::bad_request(format!("invalid JSON request: {e}")))?;
        prepare(state, req)
    };
    let maps = (0..plan.parts.len())
        .map(|i| {
            // The readers ride the request's merge-train switches too.
            let mut b = readers::map_body(msgs, &plan, i, base, rc);
            if let Some(t) = raw.get("aien_test") {
                b["aien_test"] = t.clone();
            }
            from(b)
        })
        .collect::<Result<Vec<_>, _>>()?;
    // The connect request must be valid before any row reads (same checks, notes not yet known).
    let blank = vec![String::new(); plan.parts.len()];
    let check = from(readers::connect_body(raw, &plan, &blank, base))?;
    let (stream, tools) = (check.stream, check.tools);
    let show_notes = raw
        .get("aien_show_notes")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let (created, id) = new_id(state);
    let (tx, rx) = mpsc::channel::<Ev>(64);
    let st = Arc::clone(state);
    let raw = raw.clone();
    let log_id = id.clone();
    let mut permit = first.take().expect("admitted slot");
    if let Some(extra) = extra {
        permit.merge(extra);
    }
    tokio::task::spawn_blocking(move || {
        run_readers(st, raw, plan, maps, permit, tx, stream, show_notes, &log_id);
    });
    Ok(Some(PendingResponse {
        rx,
        stream,
        id,
        created,
        model: rc.model_id.clone(),
        prompt_tokens: 0,
        tools,
    }))
}

/// Phase 1 (map: every part read at once through [`Engine::generate_group`]) and phase 2
/// (connect: the notes into one answer on the usual generation path), on a blocking thread
/// holding the readers' permits; all but one are given back when the notes are in.
#[allow(clippy::too_many_arguments)]
fn run_readers(
    state: Arc<AppState>,
    raw: Value,
    plan: crate::readers::Plan,
    maps: Vec<Prepared>,
    mut permit: OwnedSemaphorePermit,
    tx: mpsc::Sender<Ev>,
    stream: bool,
    show_notes: bool,
    id: &str,
) {
    use crate::readers;
    use std::sync::atomic::AtomicBool;
    let n = maps.len();
    let t0 = Instant::now();
    if stream {
        // The response goes out now; the reading takes a while.
        let mut s = format!("Reading {} tokens in {n} parts at once:", plan.total_tokens);
        for (k, p) in plan.parts.iter().enumerate() {
            s.push_str(&format!(
                " {}: lines {}-{} ({} tok);",
                k + 1,
                p.first_line,
                p.last_line,
                p.tokens
            ));
        }
        s.push('\n');
        if tx.blocking_send(Ev::Reading).is_err() || tx.blocking_send(Ev::Reasoning(s)).is_err() {
            return;
        }
    }
    let cancel = Arc::new(AtomicBool::new(false));
    let jobs: Vec<crate::engine::GroupJob> = maps
        .iter()
        .map(|p| crate::engine::GroupJob {
            prompt: p.prompt_tokens.clone(),
            marks: p.marks.clone(),
            params: p.params.clone(),
            cancel: Some(Arc::clone(&cancel)),
            test: p.test.clone(),
        })
        .collect();
    // A client that leaves stops every reader, even one still taking in its part.
    let reading = Arc::new(AtomicBool::new(true));
    {
        let (tx, cancel, reading) = (tx.clone(), Arc::clone(&cancel), Arc::clone(&reading));
        std::thread::spawn(move || {
            while reading.load(Ordering::SeqCst) {
                if tx.is_closed() {
                    cancel.store(true, Ordering::SeqCst);
                    return;
                }
                std::thread::sleep(Duration::from_millis(200));
            }
        });
    }
    let mut toks: Vec<Vec<u32>> = vec![Vec::new(); n];
    let outs = state.engine.generate_group(&jobs, &mut |i, t| {
        toks[i].push(t);
        !cancel.load(Ordering::SeqCst)
    });
    reading.store(false, Ordering::SeqCst);
    let map_ms = t0.elapsed().as_millis() as u64;
    if cancel.load(Ordering::SeqCst) && tx.is_closed() {
        eprintln!("aien-serve-http: {id}: readers: client left while reading");
        return;
    }
    let mut parts = Vec::with_capacity(n);
    let mut notes = Vec::with_capacity(n);
    for (k, (out, t)) in outs.into_iter().zip(&toks).enumerate() {
        let out = match out {
            Ok(o) => o,
            Err(EngineError::Busy(m)) => {
                let _ = tx.blocking_send(Ev::Busy(m));
                return;
            }
            Err(e) => {
                eprintln!("aien-serve-http: {id}: reader {} failed: {e}", k + 1);
                let _ = tx.blocking_send(Ev::Error(format!("reader {} of {n}: {e}", k + 1)));
                return;
            }
        };
        let text = state.codec.decode(t).unwrap_or_default();
        let note = readers::clean_notes(&text);
        let p = &plan.parts[k];
        parts.push(serde_json::json!({
            "part": k + 1,
            "lines": [p.first_line, p.last_line],
            "tokens": p.tokens,
            "prompt_tokens": maps[k].prompt_tokens.len(),
            "note_tokens": out.emitted_tokens,
            "finish": match out.finish {
                FinishReason::Length => "length",
                _ => "stop",
            },
            "notes": note,
        }));
        notes.push(note);
    }
    // Keep one row for the answer, give the others back.
    let Some(one) = permit.split(1) else {
        let _ = tx.blocking_send(Ev::Error("readers: no permit left for the answer".into()));
        return;
    };
    drop(permit);
    let body = readers::connect_body(&raw, &plan, &notes, &state.cfg.model_id);
    let connect = match serde_json::from_value::<ChatRequest>(body)
        .map_err(|e| e.to_string())
        .and_then(|r| prepare(&state, r).map_err(|e| e.body.error.message))
    {
        Ok(p) => p,
        Err(m) => {
            let _ = tx.blocking_send(Ev::Error(format!("readers: connect request: {m}")));
            return;
        }
    };
    let info = serde_json::json!({
        "readers": n,
        "input_tokens": plan.total_tokens,
        "input_lines": plan.total_lines,
        "question_is_excerpt": plan.question_is_excerpt,
        "map_ms": map_ms,
        "map_prompt_tokens": maps.iter().map(|m| m.prompt_tokens.len()).sum::<usize>(),
        "connect_prompt_tokens": connect.prompt_tokens.len(),
        "parts": parts,
    });
    eprintln!(
        "aien-serve-http: {id}: readers: {n} parts of {} tokens read in {map_ms} ms, notes {} tok, connect prompt {} tok",
        plan.total_tokens,
        parts.iter().map(|p| p["note_tokens"].as_u64().unwrap_or(0)).sum::<u64>(),
        connect.prompt_tokens.len()
    );
    if stream {
        let s = format!(
            "Notes from {n} readers in {:.1} s; connecting them.\n",
            map_ms as f64 / 1000.0
        );
        if tx.blocking_send(Ev::Reasoning(s)).is_err() {
            return;
        }
    }
    let prompt_tokens = connect.prompt_tokens.len();
    if tx
        .blocking_send(Ev::Readers {
            info,
            prompt_tokens,
        })
        .is_err()
    {
        return;
    }
    if show_notes
        && tx
            .blocking_send(Ev::Delta(readers::notes_preface(&plan, &notes)))
            .is_err()
    {
        return;
    }
    run_generation(
        Arc::clone(&state.engine),
        Arc::clone(&state.codec),
        connect,
        tx,
        one,
        id,
    );
}

/// The response for a generation's events: the first decides the status (a busy chip is 503).
async fn respond(
    mut rx: mpsc::Receiver<Ev>,
    stream: bool,
    id: String,
    created: u64,
    model: String,
    prompt_tokens: usize,
    tools: bool,
) -> Result<Response, ApiError> {
    // The first event decides the status: a busy chip is a 503 in the OpenAI error envelope,
    // before any headers go out (streaming included). Everything else is forwarded unchanged.
    let first = rx.recv().await;
    if let Some(Ev::Busy(m)) = &first {
        return Err(ApiError::gpu_busy(m));
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
        Ok(stream_response(
            id,
            created,
            model,
            prompt_tokens,
            rx,
            tools,
        ))
    } else {
        collect_response(id, created, &model, prompt_tokens, rx, tools).await
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
    // Tool calling (Qwen3.6 qwen3_coder XML, as the model's own template renders it): legacy
    // `functions` are accepted as tools.
    let tool_defs: Vec<Value> = match (&req.tools, &req.functions) {
        (Some(t), _) if !t.is_empty() => t.clone(),
        (_, Some(f)) if !f.is_empty() => f
            .iter()
            .map(|f| serde_json::json!({"type": "function", "function": f}))
            .collect(),
        _ => Vec::new(),
    };
    let tool_defs = state.tool_cache.arrange(tool_defs);
    let has_tools = !tool_defs.is_empty();
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
    let mut turns: Vec<(String, String)> = Vec::with_capacity(req.messages.len() + 1);
    let mut pending_tool: Vec<String> = Vec::new();
    for m in &req.messages {
        let role = match m.role.trim().to_ascii_lowercase().as_str() {
            "system" | "developer" => "system",
            "user" => "user",
            "assistant" => "assistant",
            "tool" | "function" => "tool",
            other => {
                return Err(ApiError::bad_request(format!(
                    "unknown message role '{other}'"
                )))
            }
        };
        let text = text_of_content(&m.content)?;
        if role == "tool" {
            pending_tool.push(text);
            continue;
        }
        if !pending_tool.is_empty() {
            turns.push(("user".into(), crate::tools::tool_responses(&pending_tool)));
            pending_tool.clear();
        }
        let text = match (role, &m.tool_calls) {
            ("assistant", Some(calls)) if !calls.is_empty() => {
                crate::tools::assistant_with_calls(&text, calls)
            }
            _ => text,
        };
        turns.push((role.to_string(), text));
    }
    if !pending_tool.is_empty() {
        turns.push(("user".into(), crate::tools::tool_responses(&pending_tool)));
    }
    if has_tools {
        match turns.first_mut() {
            Some((r, c)) if r == "system" => {
                *c = crate::tools::system_with_tools(&tool_defs, Some(c.as_str()));
            }
            _ => turns.insert(
                0,
                (
                    "system".into(),
                    crate::tools::system_with_tools(&tool_defs, None),
                ),
            ),
        }
    }
    let turn_refs: Vec<(&str, &str)> = turns
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();
    let mut opts = chat_options(&req)?;
    if opts.enable_thinking.is_none() {
        opts.enable_thinking = cfg.default_enable_thinking;
    }
    let prompt_text = state
        .codec
        .render_chat_with(&turn_refs, &opts)
        .map_err(|e| match e {
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
    let marks = if has_tools {
        state
            .tool_cache
            .marks(&*state.codec, &prompt_text, &tool_defs, &prompt_tokens)
    } else {
        Vec::new()
    };

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
        marks,
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
        reasoning_end: if prompt_text.ends_with("<think>\n") {
            state.codec.reasoning_end_id()
        } else {
            None
        },
        return_ids: req.aien_return_token_ids,
        tools: has_tools,
        test: test_switches(state, req.aien_test.as_ref())?,
    })
}

/// The merge train's `aien_test` field. A build without the `aien-test-switches` feature (the
/// live server) refuses the field outright; a test build parses it and refuses any switch the
/// engine does not honor, so no switch is ever ignored in silence and no verdict rests on one.
fn test_switches(state: &AppState, v: Option<&Value>) -> Result<TestSwitches, ApiError> {
    let Some(v) = v.filter(|v| !v.is_null()) else {
        return Ok(TestSwitches::default());
    };
    #[cfg(not(feature = "aien-test-switches"))]
    {
        let _ = (state, v);
        Err(ApiError::bad_request(
            "aien_test: this server was built without the merge train's test switches (cargo feature aien-test-switches); it never serves a test case",
        ))
    }
    #[cfg(feature = "aien-test-switches")]
    {
        let t = TestSwitches::from_json(v).map_err(ApiError::bad_request)?;
        let honored = state.engine.test_switches_honored();
        let unhonored: Vec<&str> = t
            .names()
            .into_iter()
            .filter(|n| !honored.contains(n))
            .collect();
        if !unhonored.is_empty() {
            return Err(ApiError::bad_request(format!(
                "aien_test: switches {unhonored:?} are not honored by this server's {} engine (it honors {honored:?}); a load-only switch needs its own server",
                state.engine.kind()
            )));
        }
        Ok(t)
    }
}

/// The template switches of a request: `chat_template_kwargs.enable_thinking` /
/// `.reasoning_effort` / `.preserve_thinking`, and the top-level `reasoning_effort`.
fn chat_options(req: &ChatRequest) -> Result<ChatOptions, ApiError> {
    let mut o = ChatOptions::default();
    if let Some(kw) = req.chat_template_kwargs.as_ref().filter(|v| !v.is_null()) {
        let Some(map) = kw.as_object() else {
            return Err(ApiError::bad_request(
                "chat_template_kwargs must be an object",
            ));
        };
        if let Some(v) = map.get("enable_thinking").filter(|v| !v.is_null()) {
            o.enable_thinking = Some(v.as_bool().ok_or_else(|| {
                ApiError::bad_request("chat_template_kwargs.enable_thinking must be a boolean")
            })?);
        }
        if let Some(v) = map.get("preserve_thinking").filter(|v| !v.is_null()) {
            o.preserve_thinking = Some(v.as_bool().ok_or_else(|| {
                ApiError::bad_request("chat_template_kwargs.preserve_thinking must be a boolean")
            })?);
        }
        if let Some(v) = map.get("reasoning_effort").filter(|v| !v.is_null()) {
            o.reasoning_effort = Some(
                v.as_str()
                    .ok_or_else(|| {
                        ApiError::bad_request(
                            "chat_template_kwargs.reasoning_effort must be a string",
                        )
                    })?
                    .to_string(),
            );
        }
    }
    if let Some(e) = req.reasoning_effort.as_deref() {
        o.reasoning_effort = Some(e.to_string());
    }
    Ok(o)
}

/// Runs on a blocking thread holding the generation lock (`guard`). The lock is released as soon
/// as the engine returns, before the final event is sent, so a client that sends its next request
/// the moment it has the reply never meets a stale 503 `server_busy`.
fn run_generation(
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    prepared: Prepared,
    tx: mpsc::Sender<Ev>,
    guard: OwnedSemaphorePermit,
    id: &str,
) {
    let mut asm = TextAssembler::new(Arc::clone(&codec), prepared.stop_strings);
    // The thinking block (prompt ends in `<think>\n`): its text goes out as reasoning, the
    // closing token is dropped, and the answer's leading newlines are trimmed (the template
    // puts `\n</think>\n\n` between the two).
    let mut thinking = prepared
        .reasoning_end
        .map(|end| (end, TextAssembler::new(codec, Vec::new())));
    let mut answer_started = prepared.reasoning_end.is_none();
    let mut ids = prepared.return_ids.then(Vec::new);
    let mut stopped_by_string = false;
    let mut client_gone = false;
    let test_report = (!prepared.test.is_empty()).then(|| prepared.test.names());
    let result = engine.generate_marked(
        &prepared.prompt_tokens,
        &prepared.marks,
        &prepared.params,
        &prepared.test,
        &mut |tok| {
            if let Some(ids) = ids.as_mut() {
                ids.push(tok);
            }
            if let Some((end, rasm)) = thinking.as_mut() {
                let ev = if tok == *end {
                    let rest = rasm.finish();
                    thinking = None;
                    rest
                } else {
                    match rasm.push(tok) {
                        Push::Delta(d) | Push::StopAt(d) => Some(d),
                        Push::Hold => None,
                    }
                };
                if let Some(d) = ev.filter(|d| !d.is_empty()) {
                    if tx.blocking_send(Ev::Reasoning(d)).is_err() {
                        client_gone = true;
                        return false;
                    }
                }
                return true;
            }
            let trim = |d: String, started: &mut bool| {
                if *started {
                    return d;
                }
                let t = d.trim_start_matches('\n');
                *started = !t.is_empty();
                t.to_string()
            };
            match asm.push(tok) {
                Push::Hold => true,
                Push::Delta(d) => {
                    let d = trim(d, &mut answer_started);
                    if d.is_empty() {
                        return true;
                    }
                    if tx.blocking_send(Ev::Delta(d)).is_err() {
                        client_gone = true;
                        false
                    } else {
                        true
                    }
                }
                Push::StopAt(d) => {
                    stopped_by_string = true;
                    let d = trim(d, &mut answer_started);
                    if !d.is_empty() {
                        let _ = tx.blocking_send(Ev::Delta(d));
                    }
                    false
                }
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
            if let Some((_, mut rasm)) = thinking.take() {
                if let Some(rest) = rasm.finish().filter(|r| !r.is_empty()) {
                    let _ = tx.blocking_send(Ev::Reasoning(rest));
                }
            }
            if let Some(rest) = asm.finish() {
                let rest = if answer_started {
                    rest
                } else {
                    rest.trim_start_matches('\n').to_string()
                };
                if !rest.is_empty() {
                    let _ = tx.blocking_send(Ev::Delta(rest));
                }
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
                token_ids: ids,
                test: test_report,
            });
        }
    }
}

async fn collect_response(
    id: String,
    created: u64,
    model: &str,
    mut prompt_tokens: usize,
    mut rx: mpsc::Receiver<Ev>,
    tools: bool,
) -> Result<Response, ApiError> {
    let mut content = String::new();
    let mut reasoning: Option<String> = None;
    let mut readers: Option<Value> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            Ev::Delta(d) => content.push_str(&d),
            Ev::Reading => {}
            Ev::Readers {
                info,
                prompt_tokens: p,
            } => {
                readers = Some(info);
                prompt_tokens = p;
            }
            Ev::Reasoning(d) => reasoning.get_or_insert_with(String::new).push_str(&d),
            Ev::Done {
                finish,
                completion_tokens,
                sampling,
                token_ids,
                test,
            } => {
                let usage = Usage {
                    prompt_tokens,
                    completion_tokens,
                    total_tokens: prompt_tokens + completion_tokens,
                    aien_sampling: sampling,
                    aien_token_ids: token_ids,
                    aien_test: test.map(|n| serde_json::json!({ "applied": n })),
                };
                let reasoning = reasoning.as_deref().map(str::trim);
                let (text, calls) = if tools {
                    crate::tools::parse_reply(&content, &id)
                } else {
                    (content.clone(), Vec::new())
                };
                let mut body =
                    completion_json(&id, created, model, &text, reasoning, finish, &usage);
                if !calls.is_empty() {
                    let msg = &mut body["choices"][0]["message"];
                    msg["tool_calls"] = Value::Array(calls);
                    if text.trim().is_empty() {
                        msg["content"] = Value::Null;
                    }
                    body["choices"][0]["finish_reason"] = Value::String("tool_calls".into());
                }
                if let Some(info) = readers {
                    body["aien_readers"] = info;
                }
                return Ok(Json(body).into_response());
            }
            Ev::Error(m) => return Err(ApiError::internal(m)),
            // Not the first event on the readers' path (`Ev::Readers` goes before the connect
            // generation); nothing has been sent yet, so it is the same 503 as a busy first event.
            Ev::Busy(m) => return Err(ApiError::gpu_busy(&m)),
        }
    }
    Err(ApiError::internal("generation ended without a result"))
}

fn stream_response(
    id: String,
    created: u64,
    model: String,
    mut prompt_tokens: usize,
    mut rx: mpsc::Receiver<Ev>,
    tools: bool,
) -> Response {
    let (etx, erx) = mpsc::channel::<Result<Event, Infallible>>(64);
    tokio::spawn(async move {
        // Tool requests: content streams until a `<tool_call>` opens (a possible split tag is held
        // back); from there the reply is buffered and sent as one `tool_calls` delta at the end.
        let mut held = String::new();
        let mut in_call = false;
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
                Ev::Reading => continue,
                Ev::Readers {
                    info,
                    prompt_tokens: p,
                } => {
                    prompt_tokens = p;
                    let mut v = chunk_json(&id, created, &model, None, None, None, None);
                    v["aien_readers"] = info;
                    data(v)
                }
                Ev::Delta(d) if tools => {
                    held.push_str(&d);
                    if in_call {
                        continue;
                    }
                    let out = if let Some(p) = held.find(crate::tools::OPEN) {
                        in_call = true;
                        let out = held[..p].to_string();
                        held = held[p..].to_string();
                        out
                    } else {
                        let keep = crate::tools::partial_open_len(&held);
                        let cut = held.len() - keep;
                        let out = held[..cut].to_string();
                        held = held[cut..].to_string();
                        out
                    };
                    if out.is_empty() {
                        continue;
                    }
                    data(chunk_json(
                        &id,
                        created,
                        &model,
                        None,
                        Some(&out),
                        None,
                        None,
                    ))
                }
                Ev::Delta(d) => data(chunk_json(&id, created, &model, None, Some(&d), None, None)),
                Ev::Reasoning(d) => data(chunk_json_with(
                    &id,
                    created,
                    &model,
                    None,
                    None,
                    Some(&d),
                    None,
                    None,
                )),
                Ev::Done {
                    finish,
                    completion_tokens,
                    sampling,
                    token_ids,
                    test,
                } => {
                    let usage = Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                        aien_sampling: sampling,
                        aien_token_ids: token_ids,
                        aien_test: test.map(|n| serde_json::json!({ "applied": n })),
                    };
                    let mut finish = finish;
                    if tools && !held.is_empty() {
                        let (text, calls) = crate::tools::parse_reply(&held, &id);
                        if !text.is_empty()
                            && etx
                                .send(data(chunk_json(
                                    &id,
                                    created,
                                    &model,
                                    None,
                                    Some(&text),
                                    None,
                                    None,
                                )))
                                .await
                                .is_err()
                        {
                            return;
                        }
                        if !calls.is_empty() {
                            let deltas: Vec<Value> = calls
                                .into_iter()
                                .enumerate()
                                .map(|(i, mut c)| {
                                    c["index"] = Value::from(i);
                                    c
                                })
                                .collect();
                            let mut ch = chunk_json(&id, created, &model, None, None, None, None);
                            ch["choices"][0]["delta"]["tool_calls"] = Value::Array(deltas);
                            if etx.send(data(ch)).await.is_err() {
                                return;
                            }
                            finish = "tool_calls";
                        }
                    }
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

#[cfg(test)]
mod tests {
    use super::*;

    async fn answer(events: Vec<Ev>, stream: bool) -> Response {
        let (tx, rx) = mpsc::channel(events.len().max(1));
        for ev in events {
            tx.send(ev).await.expect("room for every event");
        }
        drop(tx);
        respond(rx, stream, "id".into(), 0, "m".into(), 3, false)
            .await
            .unwrap_or_else(IntoResponse::into_response)
    }

    fn assert_chip_busy(resp: &Response) {
        assert_eq!(resp.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert_eq!(
            resp.headers()
                .get(header::RETRY_AFTER)
                .map(|v| v.as_bytes()),
            Some(&b"5"[..])
        );
    }

    /// A busy chip before anything else is a 503 with Retry-After 5.
    #[tokio::test]
    async fn busy_first_is_a_503_with_retry_after() {
        assert_chip_busy(&answer(vec![Ev::Busy("held".into())], false).await);
    }

    /// The readers' non-streaming path sends `Ev::Readers` before the connect generation, so a busy
    /// chip for that generation is not the first event: it must still be the same 503 with
    /// Retry-After 5, not a 500.
    #[tokio::test]
    async fn busy_after_the_readers_event_is_the_same_503() {
        let events = vec![
            Ev::Readers {
                info: serde_json::json!({ "readers": 2 }),
                prompt_tokens: 7,
            },
            Ev::Busy("GPU lock held by other jobs for 300 s".into()),
        ];
        assert_chip_busy(&answer(events, false).await);
    }
}
