//! The HTTP surface: `GET /v1/models`, `POST /v1/chat/completions` (JSON or SSE),
//! `GET /health`. Every error is OpenAI error JSON. As many generations at a time as the
//! engine says it takes ([`Engine::concurrency`], 1 unless the engine batches): the next
//! request while all those run gets 503 `server_busy` with `Retry-After: 1`.

use std::collections::HashMap;
use std::convert::Infallible;
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use axum::body::Bytes;
use axum::extract::rejection::BytesRejection;
use axum::extract::{DefaultBodyLimit, Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::Value;
use tokio::sync::{mpsc, OwnedSemaphorePermit, Semaphore};
use tokio::task::JoinHandle;
use tokio_stream::wrappers::ReceiverStream;

use aien_inference_abi::tokenizer::ChatOptions;

use crate::board::{self, Board, Conversation, List, Patch};
use crate::codec::{Codec, CodecError};
use crate::engine::{
    Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams, TestSwitches,
};
use crate::openai::{
    chunk_json, chunk_json_with, completion_json, ChatRequest, ErrorBody, ModelCard, ModelList,
    Usage,
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
    /// `enable_thinking` for requests that do not set it (`None`: the chat template's own
    /// default). Only templates with a thinking switch read it (Qwen3.8-27B).
    pub default_enable_thinking: Option<bool>,
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
        }
    }
}

pub struct AppState {
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    cfg: ServerConfig,
    /// One permit per generation the engine runs at once.
    gate: Arc<Semaphore>,
    started: Instant,
    counter: AtomicU64,
    /// The team board (L376): `None` when `AIEN_BOARD=0` or the caller passed none.
    board: Option<Arc<Board>>,
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
                "every generation slot is taken (the engine's concurrency): retry shortly",
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
    /// The prompt opens a thinking block: reply tokens up to this id are reasoning.
    reasoning_end: Option<u32>,
    return_ids: bool,
    /// The request carried tools: parse `<tool_call>` blocks out of the reply.
    tools: bool,
    /// The built-in board tools were offered: a round whose calls are all built-in is answered in
    /// the server and the same turn continues in place.
    builtin_tools: bool,
    /// The context this request was checked against (the smaller of the door's and the engine's):
    /// a built-in round's continuation stays under the same ceiling.
    ctx: usize,
    /// The template switches, for the in-place tool round's tail.
    opts: ChatOptions,
    /// The conversation's board state to commit when the request is admitted.
    board: Option<BoardPending>,
    /// The merge train's switches for this request (empty: none; `aien_test` absent).
    test: TestSwitches,
}

/// A conversation's cursors and patches after this request's delivery, committed on admission.
struct BoardPending {
    key: String,
    conv: Conversation,
    /// The block appended at this request's newest user turn, if one was: what to undo when the
    /// prompt would not fit (then the turn is rendered without it and the cursors stay).
    delivered: Option<Delivered>,
}

struct Delivered {
    turn: usize,
    original: String,
    board_seen: u64,
    memory_seen: u64,
}

impl BoardPending {
    /// Takes this request's block back off the turn and rolls the cursors back.
    fn undo(&mut self, turns: &mut [(String, String)], shift: usize) {
        if let Some(d) = self.delivered.take() {
            if let Some(t) = turns.get_mut(d.turn + shift) {
                t.1 = d.original;
            }
            self.conv.patches.pop();
            self.conv.board_seen = d.board_seen;
            self.conv.memory_seen = d.memory_seen;
        }
    }
}

/// Most built-in tool rounds one request runs (each continues the same turn in place).
const MAX_TOOL_ROUNDS: usize = 4;

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
        .route("/v1/aien/:list", get(board_get).post(board_post))
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
    serve_with_board(cfg, engine, codec, None, shutdown).await
}

/// [`serve_until`] with the team board (L376): every conversation gets the board's new entries
/// appended at its next turn, tool requests get `board_post` / `board_read` / `memory_note`, and
/// `GET|POST /v1/aien/board` and `/v1/aien/memory` read and post from outside.
pub async fn serve_with_board(
    cfg: ServerConfig,
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    board: Option<Arc<Board>>,
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
    let gate = Arc::new(Semaphore::new(engine.concurrency().max(1)));
    let state = Arc::new(AppState {
        engine,
        codec,
        cfg,
        gate,
        started: Instant::now(),
        counter: AtomicU64::new(0),
        board,
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
    // Team board (L376): present only when the board is on.
    if let (Some(b), Some(obj)) = (state.board.as_ref(), v.as_object_mut()) {
        let (entries, memory, conversations) = b.counts();
        obj.insert(
            "board".into(),
            serde_json::json!({
                "entries": entries,
                "memory": memory,
                "conversations": conversations,
                "tools": b.offers_tools(),
            }),
        );
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

fn board_list(name: &str) -> Result<List, ApiError> {
    match name {
        "board" => Ok(List::Board),
        "memory" => Ok(List::Memory),
        other => Err(ApiError::not_found(format!(
            "no such list '{other}': /v1/aien/board and /v1/aien/memory"
        ))),
    }
}

fn board_of(state: &AppState) -> Result<&Board, ApiError> {
    state
        .board
        .as_deref()
        .ok_or_else(|| ApiError::not_found("the team board is off on this server (AIEN_BOARD=0)"))
}

/// `GET /v1/aien/board?since=N&limit=M` (and `/memory`): entries after `since`, the newest `limit`.
async fn board_get(
    State(state): State<Arc<AppState>>,
    Path(list): Path<String>,
    Query(q): Query<HashMap<String, String>>,
) -> Result<Json<Value>, ApiError> {
    let list = board_list(&list)?;
    let b = board_of(&state)?;
    let num = |k: &str, d: u64| -> Result<u64, ApiError> {
        match q.get(k) {
            None => Ok(d),
            Some(v) => v
                .trim()
                .parse()
                .map_err(|_| ApiError::bad_request(format!("{k} must be a whole number"))),
        }
    };
    let since = num("since", 0)?;
    let limit = num("limit", 100)?.clamp(1, 1000) as usize;
    let (entries, newest) = b.read(list, since, limit);
    Ok(Json(serde_json::json!({
        "object": "list",
        "list": list.name(),
        "entries": entries,
        "newest": newest,
    })))
}

/// `POST /v1/aien/board {"author": "...", "text": "..."}` (and `/memory`): appends one entry.
async fn board_post(
    State(state): State<Arc<AppState>>,
    Path(list): Path<String>,
    body: Result<Bytes, BytesRejection>,
) -> Result<Json<Value>, ApiError> {
    let list = board_list(&list)?;
    let b = board_of(&state)?;
    let body = body.map_err(|r| ApiError::bad_request(r.body_text()))?;
    let v: Value = serde_json::from_slice(&body)
        .map_err(|e| ApiError::bad_request(format!("invalid JSON request: {e}")))?;
    let text = v
        .get("text")
        .and_then(Value::as_str)
        .ok_or_else(|| ApiError::bad_request("'text' (a string) is required"))?;
    let author = v.get("author").and_then(Value::as_str).unwrap_or("http");
    let entry = b.post(list, author, text, None).map_err(|e| match e {
        board::PostError::Invalid(m) => ApiError::bad_request(m),
        board::PostError::Io(m) => ApiError::internal(m),
    })?;
    Ok(Json(serde_json::to_value(entry).expect("entry serializes")))
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
    let prepared = prepare(&state, req)?;
    let guard = Arc::clone(&state.gate)
        .try_acquire_owned()
        .map_err(|_| ApiError::busy())?;
    // Admitted: the conversation's board cursors and this turn's block are now the record (a
    // 503 retry above renders the same block again).
    if let (Some(b), Some(p)) = (state.board.as_ref(), prepared.board.as_ref()) {
        match b.commit(&p.key, p.conv.clone()) {
            Ok(took) if took.as_millis() >= 20 => eprintln!(
                "aien-serve-http: board: {}: cursor write took {} ms",
                p.key,
                took.as_millis()
            ),
            Ok(_) => {}
            Err(e) => eprintln!("aien-serve-http: board: {e}"),
        }
    }
    let board = state.board.clone();

    let (created, nanos) = unix_now();
    let n = state.counter.fetch_add(1, Ordering::Relaxed);
    let id = format!("chatcmpl-{nanos:x}{n:04x}");
    let prompt_tokens = prepared.prompt_tokens.len();
    let stream = prepared.stream;
    let tools = prepared.tools;
    let (tx, mut rx) = mpsc::channel::<Ev>(64);
    let engine = Arc::clone(&state.engine);
    let codec = Arc::clone(&state.codec);
    let model = state.cfg.model_id.clone();
    let log_id = id.clone();
    tokio::task::spawn_blocking(move || {
        run_generation(engine, codec, prepared, tx, guard, &log_id, board);
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
    let mut tool_defs: Vec<Value> = match (&req.tools, &req.functions) {
        (Some(t), _) if !t.is_empty() => t.clone(),
        (_, Some(f)) if !f.is_empty() => f
            .iter()
            .map(|f| serde_json::json!({"type": "function", "function": f}))
            .collect(),
        _ => Vec::new(),
    };
    let has_tools = !tool_defs.is_empty();
    // Team board tools (L376): offered to every tool request, rendered FIRST (stable-first order:
    // the shared prefix of every chat holds them), when the layout has a tool-response turn.
    let builtin_tools = has_tools
        && state.board.as_ref().is_some_and(|b| b.offers_tools())
        && state
            .codec
            .tool_round_tail("", &ChatOptions::default())
            .is_some();
    if builtin_tools {
        let mut defs = board::tool_defs();
        defs.append(&mut tool_defs);
        tool_defs = defs;
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
    // Team board (L376): re-apply the blocks earlier turns of this conversation already carry
    // (the prompt stays append-only), then append what it has not seen yet to the newest user turn.
    let mut board_pending = state
        .board
        .as_ref()
        .filter(|_| req.aien_board != Some(false))
        .map(|b| board_deliver(b, &req, &mut turns));
    let mut system_inserted = 0usize;
    if has_tools {
        match turns.first_mut() {
            Some((r, c)) if r == "system" => {
                *c = crate::tools::system_with_tools(&tool_defs, Some(c.as_str()));
            }
            _ => {
                turns.insert(
                    0,
                    (
                        "system".into(),
                        crate::tools::system_with_tools(&tool_defs, None),
                    ),
                );
                system_inserted = 1;
            }
        }
    }
    let mut opts = chat_options(&req)?;
    if opts.enable_thinking.is_none() {
        opts.enable_thinking = cfg.default_enable_thinking;
    }
    let render = |turns: &[(String, String)]| -> Result<(String, Vec<u32>), ApiError> {
        let turn_refs: Vec<(&str, &str)> = turns
            .iter()
            .map(|(r, c)| (r.as_str(), c.as_str()))
            .collect();
        let text = state
            .codec
            .render_chat_with(&turn_refs, &opts)
            .map_err(|e| match e {
                CodecError::Unrenderable(m) => ApiError::bad_request(m),
                CodecError::Failed(m) => ApiError::internal(m),
            })?;
        let ids = state
            .codec
            .encode(&text)
            .map_err(|e| ApiError::context_exceeded(format!("could not tokenize prompt: {e}")))?;
        Ok((text, ids))
    };
    let ctx = cfg
        .max_context_tokens
        .unwrap_or(usize::MAX)
        .min(state.engine.max_context_tokens());
    let (mut prompt_text, mut prompt_tokens) = render(&turns)?;
    if prompt_tokens.len() >= ctx {
        // The board block would not fit: this turn goes without it and the cursors stay, so the
        // entries come at a later turn (or a shorter history).
        if let Some(p) = board_pending.as_mut().filter(|p| p.delivered.is_some()) {
            eprintln!(
                "aien-serve-http: board: {}: prompt {} tokens with the block, context {ctx}: delivered without it",
                p.key,
                prompt_tokens.len()
            );
            p.undo(&mut turns, system_inserted);
            (prompt_text, prompt_tokens) = render(&turns)?;
        }
    }
    if prompt_tokens.is_empty() {
        return Err(ApiError::bad_request("the rendered prompt is empty"));
    }
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
        reasoning_end: if prompt_text.ends_with("<think>\n") {
            state.codec.reasoning_end_id()
        } else {
            None
        },
        return_ids: req.aien_return_token_ids,
        tools: has_tools,
        builtin_tools,
        opts,
        ctx,
        board: board_pending,
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

/// The conversation key and author name of a request. Key: `aien_conversation` (our agents send
/// their ledger id), else llama.cpp `session_id`, else a fingerprint of the system text and the
/// first user text (the same chat resends both unchanged), namespaced by OpenAI `user` when it is
/// sent (one user runs many chats: `user` alone would merge them). Author: `aien_agent`, else
/// `user`, else `conv-<key tail>`.
fn conversation_key(req: &ChatRequest, turns: &[(String, String)]) -> (String, String) {
    fn clean(s: Option<&str>) -> Option<&str> {
        s.map(str::trim).filter(|s| !s.is_empty())
    }
    let named = clean(req.aien_conversation.as_deref())
        .or(clean(req.session_id.as_ref().and_then(Value::as_str)));
    let key = match named {
        Some(n) => format!("id-{}", board::digest(n)),
        None => {
            let system = turns
                .first()
                .filter(|(r, _)| r == "system")
                .map(|(_, c)| c.as_str())
                .unwrap_or("");
            let first_user = turns
                .iter()
                .find(|(r, _)| r == "user")
                .map(|(_, c)| c.as_str())
                .unwrap_or("");
            let fp = board::fingerprint(system, first_user);
            match clean(req.user.as_ref().and_then(Value::as_str)) {
                Some(u) => format!("u-{}-{fp}", board::digest(u)),
                None => fp,
            }
        }
    };
    let author = clean(req.aien_agent.as_deref())
        .or(clean(req.user.as_ref().and_then(Value::as_str)))
        .map(str::to_string)
        .unwrap_or_else(|| format!("conv-{}", &key[key.len().saturating_sub(8)..]));
    (key, author)
}

/// How many of a record's patches still match the turns (each at its turn, same original text).
fn matching_patches(conv: &Conversation, turns: &[(String, String)]) -> usize {
    conv.patches
        .iter()
        .take_while(|p| {
            turns
                .get(p.turn)
                .is_some_and(|(r, c)| r == "user" && board::digest(c) == p.hash)
        })
        .count()
}

/// Re-applies the conversation's earlier blocks at their turns (the prompt stays append-only),
/// then appends the entries it has not seen to the newest user turn and advances its cursors.
/// A turn whose text changed since its patch (the client rewrote the history) forks the
/// conversation: a new key from the rewritten text, the cursors as of the last patch still
/// matching, the old record untouched. Nothing is written here; the caller commits on admission.
fn board_deliver(board: &Board, req: &ChatRequest, turns: &mut [(String, String)]) -> BoardPending {
    let (mut key, author) = conversation_key(req, turns);
    let mut conv = board.conversation(&key);
    let mut kept = matching_patches(&conv, turns);
    for _hop in 0..8 {
        if kept == conv.patches.len() {
            break;
        }
        let turn = conv.patches[kept].turn;
        let rewritten = turns.get(turn).map(|(_, c)| c.as_str()).unwrap_or("");
        let fork = board::fork_key(&key, rewritten);
        eprintln!(
            "aien-serve-http: board: {key} rewrote turn {turn}: patch dropped, fork {fork} ({kept} patches kept)"
        );
        let existing = board.conversation(&fork);
        conv = if existing == Conversation::default() {
            conv.fork(kept, &key)
        } else {
            existing
        };
        key = fork;
        kept = matching_patches(&conv, turns);
    }
    conv.patches.truncate(kept);
    conv.author = author;
    for p in &conv.patches {
        let (_, c) = &mut turns[p.turn];
        *c = board::apply_block(c, &p.text);
    }
    let last = turns.len() - 1;
    let fresh_turn = turns[last].0 == "user" && conv.patches.last().is_none_or(|p| p.turn < last);
    let carried: usize = conv.patches.iter().map(|p| p.text.len()).sum();
    let mut delivered = None;
    if fresh_turn && carried >= board::PATCH_BUDGET {
        eprintln!(
            "aien-serve-http: board: {key}: paused: {carried} chars of blocks in this conversation (budget {}); a new conversation key gets the board again",
            board::PATCH_BUDGET
        );
    } else if fresh_turn {
        let unseen = board.unseen(&key, &conv);
        if let Some(r) = unseen.render() {
            let original = std::mem::take(&mut turns[last].1);
            turns[last].1 = board::apply_block(&original, &r.text);
            conv.patches.push(Patch {
                turn: last,
                hash: board::digest(&original),
                text: r.text,
                board_seen: r.board_seen,
                memory_seen: r.memory_seen,
            });
            delivered = Some(Delivered {
                turn: last,
                original,
                board_seen: conv.board_seen,
                memory_seen: conv.memory_seen,
            });
            conv.board_seen = r.board_seen;
            conv.memory_seen = r.memory_seen;
        } else {
            conv.board_seen = unseen.board_newest;
            conv.memory_seen = unseen.memory_newest;
        }
    }
    BoardPending {
        key,
        conv,
        delivered,
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
///
/// Built-in board tools (L376): when the request was offered them, text from `<tool_call>` on is
/// held back; a round whose calls are all built-in is answered here and the same turn continues
/// in place (the prompt, the reply's own tokens, the template's tool-response tail), up to
/// [`MAX_TOOL_ROUNDS`] rounds. A round with any client call is forwarded as before, its built-in
/// calls answered and stripped out of it.
fn run_generation(
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    prepared: Prepared,
    tx: mpsc::Sender<Ev>,
    guard: OwnedSemaphorePermit,
    id: &str,
    board: Option<Arc<Board>>,
) {
    let conv = match (&board, &prepared.board) {
        (Some(b), Some(p)) if prepared.builtin_tools => {
            Some((Arc::clone(b), p.key.clone(), p.conv.author.clone()))
        }
        _ => None,
    };
    let builtin = conv.is_some();
    let max_ctx = prepared.ctx;
    let first_prompt_len = prepared.prompt_tokens.len();
    let mut prompt = prepared.prompt_tokens;
    let mut params = prepared.params;
    let mut ids = prepared.return_ids.then(Vec::new);
    let mut total_emitted = 0usize;
    let mut prefill_ms = 0u64;
    let mut decode_ms = 0u64;
    let mut sent = SendState::default();
    let mut rounds = 0usize;
    let mut per_round: Vec<usize> = Vec::new();
    let test_report = (!prepared.test.is_empty()).then(|| prepared.test.names());
    let mut stopped_by_string;
    let last: GenerateOutcome;
    loop {
        rounds += 1;
        let mut asm = TextAssembler::new(Arc::clone(&codec), prepared.stop_strings.clone());
        // The thinking block (prompt ends in `<think>\n`): its text goes out as reasoning, the
        // closing token is dropped, and the answer's leading newlines are trimmed (the template
        // puts `\n</think>\n\n` between the two).
        let mut thinking = prepared
            .reasoning_end
            .map(|end| (end, TextAssembler::new(Arc::clone(&codec), Vec::new())));
        let mut answer_started = prepared.reasoning_end.is_none();
        let mut round_stop = false;
        let mut client_gone = false;
        let mut round_ids: Vec<u32> = Vec::new();
        let mut held = String::new();
        let mut in_call = false;
        sent.sent_this_round = false;
        let on_tok = &mut |tok: u32| {
            round_ids.push(tok);
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
                    let out = if builtin {
                        hold_back(&mut held, &mut in_call, d)
                    } else {
                        d
                    };
                    if out.is_empty() {
                        return true;
                    }
                    if !send_delta(&tx, out, &mut sent) {
                        client_gone = true;
                        return false;
                    }
                    true
                }
                Push::StopAt(d) => {
                    round_stop = true;
                    let d = trim(d, &mut answer_started);
                    if !d.is_empty() {
                        let out = if builtin {
                            hold_back(&mut held, &mut in_call, d)
                        } else {
                            d
                        };
                        if !out.is_empty() {
                            let _ = send_delta(&tx, out, &mut sent);
                        }
                    }
                    false
                }
            }
        };
        let result = if rounds == 1 {
            engine.generate_with_test(&prompt, &params, &prepared.test, on_tok)
        } else {
            engine.generate_continuation(&prompt, &params, &prepared.test, on_tok)
        };
        let out = match result {
            Err(EngineError::Busy(m)) => {
                drop(guard);
                eprintln!("aien-serve-http: {id}: GPU busy: {m}");
                let _ = tx.blocking_send(Ev::Busy(m));
                return;
            }
            Err(e) => {
                drop(guard);
                eprintln!("aien-serve-http: {id}: {e}");
                let _ = tx.blocking_send(Ev::Error(e.to_string()));
                return;
            }
            Ok(out) => out,
        };
        total_emitted += out.emitted_tokens;
        per_round.push(out.emitted_tokens);
        prefill_ms += out.prefill_ms;
        decode_ms += out.decode_ms;
        if let Some(ids) = ids.as_mut() {
            ids.extend_from_slice(&round_ids);
        }
        if client_gone {
            drop(guard);
            eprintln!("aien-serve-http: {id}: client left after {total_emitted} tokens");
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
                if builtin {
                    held.push_str(&rest);
                } else {
                    let _ = send_delta(&tx, rest, &mut sent);
                }
            }
        }
        stopped_by_string = round_stop;
        // Built-in round end: what was held is a tool call, or a split tag that never became one.
        let Some((b, key, author)) = conv.as_ref().filter(|_| !held.is_empty()) else {
            last = out;
            break;
        };
        let (before, calls) = crate::tools::parse_reply(&held, id);
        // A reply cut by the length cap may hold a half-written call: not run, not forwarded as a call.
        if calls.is_empty() || out.finish != FinishReason::Stop {
            let _ = send_delta(&tx, std::mem::take(&mut held), &mut sent);
            last = out;
            break;
        }
        if !before.is_empty() {
            let _ = send_delta(&tx, before, &mut sent);
        }
        let mut responses: Vec<String> = Vec::new();
        let mut any_client = false;
        for c in &calls {
            let name = c["function"]["name"].as_str().unwrap_or("");
            if !board::is_builtin(name) {
                any_client = true;
                continue;
            }
            let args: Value = c["function"]["arguments"]
                .as_str()
                .and_then(|a| serde_json::from_str(a).ok())
                .unwrap_or(Value::Null);
            let r = board::call(b, key, author, name, &args);
            eprintln!(
                "aien-serve-http: {id}: {name}: {}",
                r.lines().next().unwrap_or("")
            );
            responses.push(r);
        }
        if any_client {
            // The client runs its own calls; the built-in ones are already answered.
            let kept = crate::tools::strip_calls(&held, |n| !board::is_builtin(n));
            let _ = send_delta(&tx, kept, &mut sent);
            last = out;
            break;
        }
        // Every call was built-in: answer in place and continue the same turn.
        let tail = codec
            .tool_round_tail(&crate::tools::tool_responses(&responses), &prepared.opts)
            .and_then(|t| codec.encode(&t).ok());
        let remaining = params.max_new_tokens.saturating_sub(out.emitted_tokens);
        let Some(tail) = tail else {
            last = out;
            break;
        };
        if rounds >= MAX_TOOL_ROUNDS
            || remaining == 0
            || out.finish != FinishReason::Stop
            || prompt.len() + round_ids.len() + tail.len() + 1 >= max_ctx
        {
            eprintln!(
                "aien-serve-http: {id}: built-in tool round {rounds} ends the reply ({remaining} tokens left, finish {:?})",
                out.finish
            );
            last = out;
            break;
        }
        prompt.extend_from_slice(&round_ids);
        prompt.extend_from_slice(&tail);
        params.max_new_tokens = remaining;
    }
    drop(guard);
    let out = last;
    let finish = if stopped_by_string {
        "stop"
    } else {
        match out.finish {
            FinishReason::Stop | FinishReason::Cancelled => "stop",
            FinishReason::Length => "length",
        }
    };
    let tps = if decode_ms > 0 {
        total_emitted as f64 * 1000.0 / decode_ms as f64
    } else {
        0.0
    };
    let sampling_fields = out
        .sampling
        .as_ref()
        .map(|s| format!(", {}", s.log_fields()))
        .unwrap_or_default();
    let rounds_field = if rounds > 1 {
        let per: Vec<String> = per_round.iter().map(|n| n.to_string()).collect();
        format!(", {rounds} rounds (built-in tools: {} tok)", per.join("+"))
    } else {
        String::new()
    };
    eprintln!(
        "aien-serve-http: {id}: prompt {first_prompt_len} tok, reply {total_emitted} tok, finish {finish}, prefill {prefill_ms} ms, decode {decode_ms} ms ({tps:.2} tok/s){sampling_fields}{rounds_field}"
    );
    let _ = tx.blocking_send(Ev::Done {
        finish,
        completion_tokens: total_emitted,
        sampling: out.sampling,
        token_ids: ids,
        test: test_report,
    });
}

#[derive(Default)]
struct SendState {
    sent_any: bool,
    sent_this_round: bool,
}

/// Sends an answer delta; the first one of a later round gets a blank line before it when an
/// earlier round already answered. `false`: the client is gone.
fn send_delta(tx: &mpsc::Sender<Ev>, d: String, s: &mut SendState) -> bool {
    let d = if s.sent_any && !s.sent_this_round {
        format!("\n\n{}", d.trim_start_matches('\n'))
    } else {
        d
    };
    s.sent_any = true;
    s.sent_this_round = true;
    tx.blocking_send(Ev::Delta(d)).is_ok()
}

/// Built-in tools: `held` keeps the text from `<tool_call>` on (and a possible split tag); the
/// part safe to stream is returned.
fn hold_back(held: &mut String, in_call: &mut bool, d: String) -> String {
    held.push_str(&d);
    if *in_call {
        return String::new();
    }
    if let Some(p) = held.find(crate::tools::OPEN) {
        *in_call = true;
        let out = held[..p].to_string();
        *held = held[p..].to_string();
        return out;
    }
    let keep = crate::tools::partial_open_len(held);
    let cut = held.len() - keep;
    let out = held[..cut].to_string();
    *held = held[cut..].to_string();
    out
}

async fn collect_response(
    id: String,
    created: u64,
    model: &str,
    prompt_tokens: usize,
    mut rx: mpsc::Receiver<Ev>,
    tools: bool,
) -> Result<Response, ApiError> {
    let mut content = String::new();
    let mut reasoning: Option<String> = None;
    while let Some(ev) = rx.recv().await {
        match ev {
            Ev::Delta(d) => content.push_str(&d),
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
                let (text, calls) = if tools && finish == "stop" {
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
                return Ok(Json(body).into_response());
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
                        // A reply cut by the length cap may hold a half-written call: it goes
                        // out as text, never as a call.
                        let (text, calls) = if finish == "stop" {
                            crate::tools::parse_reply(&held, &id)
                        } else {
                            (std::mem::take(&mut held), Vec::new())
                        };
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
