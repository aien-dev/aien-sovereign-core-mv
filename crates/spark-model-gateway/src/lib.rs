use async_trait::async_trait;
use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, HeaderMap, HeaderValue, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use bytes::Bytes;
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::{
    collections::HashMap,
    convert::Infallible,
    path::PathBuf,
    sync::{
        atomic::{AtomicBool, AtomicUsize, Ordering},
        Arc,
    },
    time::{Duration, Instant},
};
use thiserror::Error;
use tokenizers::Tokenizer;
use tokio::sync::{mpsc, oneshot, Mutex};
use uuid::Uuid;

pub const GATEWAY_ADDR: &str = "127.0.0.1:18007";
pub const DEFAULT_UPSTREAM: &str = "http://127.0.0.1:18006";
pub const CONTEXT_WINDOW: usize = 49_152;
pub const MAX_INPUT_TOKENS: usize = 36_864;
pub const OUTPUT_RESERVE: usize = 8_192;
pub const PROTOCOL_MARGIN: usize = 4_096;
pub const QUEUE_CAPACITY: usize = 4;

#[derive(Clone)]
pub struct AppState {
    pub config: Arc<Config>,
    pub metrics: Arc<Metrics>,
    queue: mpsc::Sender<Job>,
}

#[derive(Clone)]
pub struct Config {
    pub upstream: String,
    pub tokenizer: Arc<Tokenizer>,
    pub tokenizer_path: PathBuf,
    pub model: String,
    pub tokens: Arc<HashMap<Principal, String>>,
    pub chat_template: ChatTemplate,
    pub upstream_timeout: Duration,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Principal {
    OpenCode,
    AienRuntime,
    Rsi,
}

impl Principal {
    fn as_str(self) -> &'static str {
        match self {
            Self::OpenCode => "opencode",
            Self::AienRuntime => "aien-runtime",
            Self::Rsi => "rsi",
        }
    }
    fn from_token(token: &str, tokens: &HashMap<Principal, String>) -> Option<Self> {
        tokens
            .iter()
            .find_map(|(principal, configured)| (configured == token).then_some(*principal))
    }
}

#[derive(Clone, Copy, Debug)]
pub enum ChatTemplate {
    NemotronQwen,
    Plain,
}

impl ChatTemplate {
    fn render(self, request: &ChatRequest) -> String {
        let mut out = String::new();
        match self {
            Self::NemotronQwen => {
                for message in &request.messages {
                    out.push_str("<|im_start|>");
                    out.push_str(&message.role);
                    out.push('\n');
                    out.push_str(&message.text());
                    out.push_str("<|im_end|>\n");
                }
                if request.tools.as_ref().is_some_and(|v| !v.is_empty()) {
                    out.push_str("<|im_start|>system\nTools:\n");
                    out.push_str(
                        &serde_json::to_string(request.tools.as_ref().unwrap()).unwrap_or_default(),
                    );
                    out.push_str("<|im_end|>\n");
                }
                out.push_str("<|im_start|>assistant\n");
            }
            Self::Plain => {
                for message in &request.messages {
                    out.push_str(&message.role);
                    out.push_str(": ");
                    out.push_str(&message.text());
                    out.push('\n');
                }
            }
        }
        out
    }
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct ChatRequest {
    pub model: Option<String>,
    pub messages: Vec<Message>,
    pub stream: Option<bool>,
    pub max_tokens: Option<usize>,
    pub max_completion_tokens: Option<usize>,
    pub temperature: Option<f32>,
    pub tools: Option<Vec<Value>>,
    pub n: Option<usize>,
    pub best_of: Option<usize>,
    #[serde(flatten)]
    pub extra: HashMap<String, Value>,
}

#[derive(Debug, Deserialize, Serialize, Clone)]
pub struct Message {
    pub role: String,
    pub content: Value,
}

impl Message {
    fn text(&self) -> String {
        match &self.content {
            Value::String(s) => s.clone(),
            Value::Array(parts) => parts
                .iter()
                .filter_map(|part| part.get("text").and_then(Value::as_str))
                .collect::<Vec<_>>()
                .join(""),
            Value::Null => String::new(),
            other => other.to_string(),
        }
    }
}

#[derive(Clone)]
pub struct Metrics {
    inner: Arc<Mutex<HashMap<&'static str, ClientMetrics>>>,
    queue_depth: Arc<AtomicUsize>,
    active: Arc<AtomicBool>,
}
#[derive(Default, Clone, Copy)]
struct ClientMetrics {
    requests: u64,
    rejected: u64,
    upstream_errors: u64,
    cancellations: u64,
    prompt_rejections: u64,
    queue_wait_ms: u128,
    ttft_ms: u128,
    generation_ms: u128,
    input_tokens: u64,
    output_tokens: u64,
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(HashMap::new())),
            queue_depth: Arc::new(AtomicUsize::new(0)),
            active: Arc::new(AtomicBool::new(false)),
        }
    }
    fn queued(&self) {
        self.queue_depth.fetch_add(1, Ordering::Relaxed);
    }
    fn dequeued(&self) {
        self.queue_depth.fetch_sub(1, Ordering::Relaxed);
    }
    fn set_active(&self, value: bool) {
        self.active.store(value, Ordering::Relaxed);
    }
    async fn record(&self, principal: Principal, f: impl FnOnce(&mut ClientMetrics)) {
        let mut map = self.inner.lock().await;
        f(map.entry(principal.as_str()).or_default());
    }
    pub async fn render(&self) -> String {
        let map = self.inner.lock().await;
        let mut out = format!("# TYPE spark_gateway_active_slots gauge\nspark_gateway_active_slots {}\n# TYPE spark_gateway_queue_depth gauge\nspark_gateway_queue_depth {}\n", self.active.load(Ordering::Relaxed) as u8, self.queue_depth.load(Ordering::Relaxed));
        for (client, m) in map.iter() {
            out.push_str(&format!("spark_gateway_requests_total{{client=\"{}\"}} {}\nspark_gateway_rejected_total{{client=\"{}\"}} {}\nspark_gateway_upstream_errors_total{{client=\"{}\"}} {}\nspark_gateway_cancellations_total{{client=\"{}\"}} {}\nspark_gateway_prompt_rejections_total{{client=\"{}\"}} {}\nspark_gateway_input_tokens_total{{client=\"{}\"}} {}\nspark_gateway_output_tokens_total{{client=\"{}\"}} {}\nspark_gateway_queue_wait_ms_total{{client=\"{}\"}} {}\nspark_gateway_ttft_ms_total{{client=\"{}\"}} {}\nspark_gateway_generation_ms_total{{client=\"{}\"}} {}\n", client,m.requests,client,m.rejected,client,m.upstream_errors,client,m.cancellations,client,m.prompt_rejections,client,m.input_tokens,client,m.output_tokens,client,m.queue_wait_ms,client,m.ttft_ms,client,m.generation_ms));
        }
        out
    }
}

struct Job {
    principal: Principal,
    request: ChatRequest,
    accepted: Instant,
    request_id: String,
    response: oneshot::Sender<Result<Response, GatewayError>>,
}

/// Provider boundary for the compatibility membrane. MAX is the current provider;
/// a native AIEN runtime can implement this trait without changing admission logic.
#[async_trait]
pub trait InferenceProvider: Send + Sync {
    async fn start(&self, request: &ChatRequest) -> Result<reqwest::Response, GatewayError>;
}

pub struct MaxProvider {
    client: Client,
    endpoint: String,
}

impl MaxProvider {
    fn new(config: &Config) -> Self {
        Self {
            client: Client::builder()
                .timeout(config.upstream_timeout)
                .build()
                .unwrap_or_default(),
            endpoint: format!(
                "{}/v1/chat/completions",
                config.upstream.trim_end_matches('/')
            ),
        }
    }
}

#[async_trait]
impl InferenceProvider for MaxProvider {
    async fn start(&self, request: &ChatRequest) -> Result<reqwest::Response, GatewayError> {
        self.client
            .post(&self.endpoint)
            .json(request)
            .send()
            .await
            .map_err(|e| GatewayError::Upstream(e.to_string()))
    }
}

#[derive(Debug, Error)]
pub enum GatewayError {
    #[error("queue is full")]
    QueueFull,
    #[error("upstream request failed: {0}")]
    Upstream(String),
    #[error("upstream returned HTTP {0}")]
    UpstreamStatus(u16),
    #[error("request body is invalid: {0}")]
    Invalid(String),
}

#[derive(Serialize)]
struct ErrorBody {
    error: ErrorDetail,
}
#[derive(Serialize)]
struct ErrorDetail {
    message: String,
    r#type: &'static str,
    code: &'static str,
}

impl IntoResponse for GatewayError {
    fn into_response(self) -> Response {
        let (status, code) = match self {
            Self::QueueFull => (StatusCode::TOO_MANY_REQUESTS, "queue_full"),
            Self::Invalid(_) => (StatusCode::BAD_REQUEST, "invalid_request"),
            Self::UpstreamStatus(s) => (
                StatusCode::from_u16(s).unwrap_or(StatusCode::BAD_GATEWAY),
                "upstream_error",
            ),
            Self::Upstream(_) => (StatusCode::BAD_GATEWAY, "upstream_error"),
        };
        (
            status,
            Json(ErrorBody {
                error: ErrorDetail {
                    message: self.to_string(),
                    r#type: "gateway_error",
                    code,
                },
            }),
        )
            .into_response()
    }
}

pub fn build_app(state: AppState) -> Router {
    Router::new()
        .route("/v1/chat/completions", post(chat_completions))
        .route("/v1/models", get(models))
        .route("/healthz", get(healthz))
        .route("/readyz", get(readyz))
        .route("/metrics", get(metrics))
        .layer(axum::middleware::from_fn_with_state(
            state.clone(),
            authenticate,
        ))
        .with_state(state)
}

async fn authenticate(State(state): State<AppState>, req: Request, next: Next) -> Response {
    let public = req.uri().path() == "/healthz" || req.uri().path() == "/readyz";
    if public {
        return next.run(req).await;
    }
    let token = req
        .headers()
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match token.and_then(|t| Principal::from_token(t, &state.config.tokens)) {
        Some(principal) => {
            let mut req = req;
            req.extensions_mut().insert(principal);
            next.run(req).await
        }
        None => (
            StatusCode::UNAUTHORIZED,
            Json(ErrorBody {
                error: ErrorDetail {
                    message: "missing or invalid bearer token".into(),
                    r#type: "authentication_error",
                    code: "invalid_token",
                },
            }),
        )
            .into_response(),
    }
}

async fn chat_completions(
    State(state): State<AppState>,
    req: Request,
) -> Result<Response, GatewayError> {
    let principal = *req
        .extensions()
        .get::<Principal>()
        .ok_or_else(|| GatewayError::Invalid("authenticated principal missing".into()))?;
    let body = axum::body::to_bytes(req.into_body(), 16 * 1024 * 1024)
        .await
        .map_err(|e| GatewayError::Invalid(e.to_string()))?;
    let request: ChatRequest =
        serde_json::from_slice(&body).map_err(|e| GatewayError::Invalid(e.to_string()))?;
    validate_request(&state, principal, &request).await?;
    let (response_tx, response_rx) = oneshot::channel();
    let job = Job {
        principal,
        request,
        accepted: Instant::now(),
        request_id: format!("gw-{}", Uuid::new_v4()),
        response: response_tx,
    };
    if state.queue.try_send(job).is_err() {
        state.metrics.record(principal, |m| m.rejected += 1).await;
        return Err(GatewayError::QueueFull);
    }
    state.metrics.queued();
    response_rx
        .await
        .map_err(|_| GatewayError::Upstream("dispatcher stopped".into()))?
}

async fn validate_request(
    state: &AppState,
    principal: Principal,
    request: &ChatRequest,
) -> Result<(), GatewayError> {
    if request.messages.is_empty() {
        return Err(GatewayError::Invalid("messages must not be empty".into()));
    }
    if let (Some(max_tokens), Some(max_completion_tokens)) =
        (request.max_tokens, request.max_completion_tokens)
    {
        if max_tokens != max_completion_tokens {
            return Err(GatewayError::Invalid(
                "max_tokens and max_completion_tokens must match when both are supplied".into(),
            ));
        }
    }
    if request.n.unwrap_or(1) != 1 || request.best_of.unwrap_or(1) != 1 {
        return Err(GatewayError::Invalid("n and best_of must be 1".into()));
    }
    let output = request
        .max_completion_tokens
        .or(request.max_tokens)
        .unwrap_or(OUTPUT_RESERVE);
    if output > OUTPUT_RESERVE {
        return Err(GatewayError::Invalid(format!(
            "requested output {} exceeds reserve {}",
            output, OUTPUT_RESERVE
        )));
    }
    let rendered = state.config.chat_template.render(request);
    let input = state
        .config
        .tokenizer
        .encode(rendered, false)
        .map_err(|e| GatewayError::Invalid(format!("tokenization failed: {}", e)))?
        .len();
    if input > MAX_INPUT_TOKENS || input + output + PROTOCOL_MARGIN > CONTEXT_WINDOW {
        state
            .metrics
            .record(principal, |m| m.prompt_rejections += 1)
            .await;
        return Err(GatewayError::Invalid(format!(
            "prompt has {} tokens, budget is {} with {} output tokens and {} protocol margin",
            input, MAX_INPUT_TOKENS, output, PROTOCOL_MARGIN
        )));
    }
    Ok(())
}

async fn models(State(state): State<AppState>) -> Json<Value> {
    Json(
        serde_json::json!({"object":"list","data":[{"id":state.config.model,"object":"model","owned_by":"aien"}]}),
    )
}
async fn healthz() -> &'static str {
    "ok\n"
}
async fn readyz(State(state): State<AppState>) -> Response {
    match state.config.upstream.parse::<reqwest::Url>() {
        Ok(url) => match state
            .config
            .configured_client()
            .get(format!("{}/v1/models", url.as_str().trim_end_matches('/')))
            .timeout(Duration::from_secs(2))
            .send()
            .await
        {
            Ok(r) if r.status().is_success() => (StatusCode::OK, "ready\n").into_response(),
            _ => (StatusCode::SERVICE_UNAVAILABLE, "upstream unavailable\n").into_response(),
        },
        Err(_) => (StatusCode::SERVICE_UNAVAILABLE, "invalid upstream\n").into_response(),
    }
}
async fn metrics(State(state): State<AppState>) -> Response {
    let body = state.metrics.render().await;
    ([(header::CONTENT_TYPE, "text/plain; version=0.0.4")], body).into_response()
}

impl Config {
    fn configured_client(&self) -> Client {
        Client::builder()
            .timeout(self.upstream_timeout)
            .build()
            .unwrap_or_default()
    }
}

async fn run_dispatcher(mut rx: mpsc::Receiver<Job>, config: Arc<Config>, metrics: Arc<Metrics>) {
    let provider = MaxProvider::new(&config);
    while let Some(job) = rx.recv().await {
        metrics.dequeued();
        if job.response.is_closed() {
            continue;
        }
        metrics.set_active(true);
        let principal = job.principal;
        let queue_wait = job.accepted.elapsed();
        metrics
            .record(principal, |m| {
                m.requests += 1;
                m.queue_wait_ms += queue_wait.as_millis();
            })
            .await;
        let started = Instant::now();
        let upstream = match provider.start(&job.request).await {
            Ok(response) => response,
            Err(e) => {
                metrics.record(principal, |m| m.upstream_errors += 1).await;
                metrics.set_active(false);
                let _ = job.response.send(Err(e));
                continue;
            }
        };
        if !upstream.status().is_success() {
            let status = upstream.status().as_u16();
            metrics.record(principal, |m| m.upstream_errors += 1).await;
            metrics.set_active(false);
            let _ = job.response.send(Err(GatewayError::UpstreamStatus(status)));
            continue;
        }
        let is_stream = job.request.stream.unwrap_or(false);
        let mut headers = HeaderMap::new();
        if is_stream {
            headers.insert(
                header::CONTENT_TYPE,
                HeaderValue::from_static("text/event-stream"),
            );
            headers.insert(header::CACHE_CONTROL, HeaderValue::from_static("no-cache"));
        }
        let (tx, rx_body) = mpsc::channel::<Result<Bytes, Infallible>>(16);
        let body = Body::from_stream(tokio_stream_from(rx_body));
        let mut response = Response::new(body);
        *response.status_mut() = StatusCode::OK;
        response.headers_mut().extend(headers);
        if let Ok(value) = HeaderValue::from_str(&job.request_id) {
            response.headers_mut().insert("x-request-id", value);
        }
        if job.response.send(Ok(response)).is_err() {
            metrics.record(principal, |m| m.cancellations += 1).await;
            metrics.set_active(false);
            continue;
        }
        let first = Instant::now();
        let mut stream = upstream.bytes_stream();
        let mut first_seen = false;
        let mut output_text = String::new();
        while let Some(chunk) = stream.next().await {
            match chunk {
                Ok(bytes) => {
                    if !first_seen {
                        metrics
                            .record(principal, |m| m.ttft_ms += first.elapsed().as_millis())
                            .await;
                        first_seen = true;
                    }
                    if !tx.send(Ok(bytes.clone())).await.is_ok() {
                        metrics.record(principal, |m| m.cancellations += 1).await;
                        break;
                    }
                    if let Ok(text) = std::str::from_utf8(&bytes) {
                        output_text.push_str(text);
                    }
                }
                Err(e) => {
                    metrics.record(principal, |m| m.upstream_errors += 1).await;
                    let _ = tx
                        .send(Ok(Bytes::from(format!(
                            "data: {{\"error\":\"{}\"}}\n\n",
                            e
                        ))))
                        .await;
                    break;
                }
            }
        }
        drop(tx);
        let output_tokens = config
            .tokenizer
            .encode(output_text.as_str(), false)
            .map(|e| e.len())
            .unwrap_or(0);
        metrics
            .record(principal, |m| {
                m.output_tokens += output_tokens as u64;
                m.generation_ms += started.elapsed().as_millis();
            })
            .await;
        metrics.set_active(false);
    }
}

fn tokio_stream_from(
    rx: mpsc::Receiver<Result<Bytes, Infallible>>,
) -> impl futures_util::Stream<Item = Result<Bytes, Infallible>> {
    futures_util::stream::unfold(rx, |mut rx| async {
        rx.recv().await.map(|item| (item, rx))
    })
}

pub fn load_config() -> Result<Config, Box<dyn std::error::Error + Send + Sync>> {
    let path = std::env::var("AIEN_GATEWAY_TOKENIZER_PATH")
        .or_else(|_| std::env::var("AIEN_TOKENIZER_PATH"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("crates/aien-inference-abi/fixtures/tokenizer.json"));
    let tokenizer = Tokenizer::from_file(&path)
        .map_err(|e| format!("failed to load tokenizer {}: {}", path.display(), e))?;
    let mut tokens = HashMap::new();
    for (principal, env) in [
        (Principal::OpenCode, "AIEN_GATEWAY_TOKEN_OPENCODE"),
        (Principal::AienRuntime, "AIEN_GATEWAY_TOKEN_AIEN_RUNTIME"),
        (Principal::Rsi, "AIEN_GATEWAY_TOKEN_RSI"),
    ] {
        if let Ok(token) = std::env::var(env) {
            if !token.is_empty() {
                tokens.insert(principal, token);
            }
        }
    }
    if tokens.is_empty() {
        return Err("at least one AIEN_GATEWAY_TOKEN_* must be configured".into());
    }
    Ok(Config {
        upstream: std::env::var("AIEN_GATEWAY_UPSTREAM")
            .unwrap_or_else(|_| DEFAULT_UPSTREAM.into()),
        tokenizer: Arc::new(tokenizer),
        tokenizer_path: path,
        model: std::env::var("AIEN_GATEWAY_MODEL")
            .unwrap_or_else(|_| "atlas-lightning-omni".into()),
        tokens: Arc::new(tokens),
        chat_template: match std::env::var("AIEN_GATEWAY_CHAT_TEMPLATE").as_deref() {
            Ok("plain") => ChatTemplate::Plain,
            _ => ChatTemplate::NemotronQwen,
        },
        upstream_timeout: Duration::from_secs(
            std::env::var("AIEN_GATEWAY_UPSTREAM_TIMEOUT_SECS")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(300),
        ),
    })
}

pub async fn serve(config: Config) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let config = Arc::new(config);
    let metrics = Arc::new(Metrics::new());
    let (tx, rx) = mpsc::channel(QUEUE_CAPACITY);
    let state = AppState {
        config: config.clone(),
        metrics: metrics.clone(),
        queue: tx,
    };
    tokio::spawn(run_dispatcher(rx, config, metrics));
    let listener = tokio::net::TcpListener::bind(GATEWAY_ADDR).await?;
    axum::serve(listener, build_app(state)).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{body::to_bytes, http::Request as HttpRequest};
    use std::time::Duration;
    use tower::ServiceExt;

    fn config(upstream: &str) -> Config {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../aien-inference-abi/fixtures/tokenizer.json");
        let tokenizer = Tokenizer::from_file(&path).unwrap();
        let mut tokens = HashMap::new();
        tokens.insert(Principal::OpenCode, "open-code-test-token".into());
        Config {
            upstream: upstream.into(),
            tokenizer: Arc::new(tokenizer),
            tokenizer_path: path,
            model: "test-model".into(),
            tokens: Arc::new(tokens),
            chat_template: ChatTemplate::NemotronQwen,
            upstream_timeout: Duration::from_secs(5),
        }
    }

    #[test]
    fn template_includes_all_messages_and_tools() {
        let req = ChatRequest {
            model: None,
            messages: vec![Message {
                role: "user".into(),
                content: Value::String("hello".into()),
            }],
            stream: Some(true),
            max_tokens: Some(8),
            max_completion_tokens: None,
            temperature: None,
            tools: Some(vec![serde_json::json!({"type":"function"})]),
            n: None,
            best_of: None,
            extra: HashMap::new(),
        };
        let rendered = ChatTemplate::NemotronQwen.render(&req);
        assert!(
            rendered.contains("hello")
                && rendered.contains("function")
                && rendered.ends_with("<|im_start|>assistant\n")
        );
    }
    #[test]
    fn message_content_parts_are_counted() {
        let m = Message {
            role: "user".into(),
            content: serde_json::json!([{"type":"text","text":"a"},{"type":"text","text":"b"}]),
        };
        assert_eq!(m.text(), "ab");
    }
    #[test]
    fn principals_are_distinct() {
        let mut map = HashMap::new();
        map.insert(Principal::OpenCode, "one".into());
        map.insert(Principal::Rsi, "two".into());
        assert_eq!(Principal::from_token("two", &map), Some(Principal::Rsi));
    }

    #[tokio::test]
    async fn protected_routes_require_a_principal_token() {
        let metrics = Arc::new(Metrics::new());
        let (queue, _rx) = mpsc::channel(QUEUE_CAPACITY);
        let state = AppState {
            config: Arc::new(config("http://127.0.0.1:9")),
            metrics,
            queue,
        };
        let response = build_app(state)
            .oneshot(
                HttpRequest::builder()
                    .uri("/v1/models")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn queue_capacity_is_four_waiting_requests() {
        let (queue, mut rx) = mpsc::channel(QUEUE_CAPACITY);
        for _ in 0..QUEUE_CAPACITY {
            let (response, _receiver) = oneshot::channel();
            queue
                .try_send(Job {
                    principal: Principal::OpenCode,
                    request: ChatRequest {
                        model: None,
                        messages: vec![],
                        stream: None,
                        max_tokens: None,
                        max_completion_tokens: None,
                        temperature: None,
                        tools: None,
                        n: None,
                        best_of: None,
                        extra: HashMap::new(),
                    },
                    accepted: Instant::now(),
                    request_id: "test".into(),
                    response,
                })
                .unwrap();
        }
        let (response, _receiver) = oneshot::channel();
        assert!(queue
            .try_send(Job {
                principal: Principal::OpenCode,
                request: ChatRequest {
                    model: None,
                    messages: vec![],
                    stream: None,
                    max_tokens: None,
                    max_completion_tokens: None,
                    temperature: None,
                    tools: None,
                    n: None,
                    best_of: None,
                    extra: HashMap::new()
                },
                accepted: Instant::now(),
                request_id: "overflow".into(),
                response
            })
            .is_err());
        assert!(rx.recv().await.is_some());
    }

    #[tokio::test]
    async fn output_reserve_rejects_amplifying_requests() {
        let state = AppState {
            config: Arc::new(config("http://127.0.0.1:9")),
            metrics: Arc::new(Metrics::new()),
            queue: mpsc::channel(QUEUE_CAPACITY).0,
        };
        let request = ChatRequest {
            model: None,
            messages: vec![Message {
                role: "user".into(),
                content: Value::String("hello".into()),
            }],
            stream: None,
            max_tokens: Some(OUTPUT_RESERVE + 1),
            max_completion_tokens: None,
            temperature: None,
            tools: None,
            n: None,
            best_of: None,
            extra: HashMap::new(),
        };
        assert!(validate_request(&state, Principal::OpenCode, &request)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn streamed_upstream_body_is_forwarded() {
        let upstream = Router::new().route(
            "/v1/chat/completions",
            post(|| async {
                (
                    [(header::CONTENT_TYPE, "text/event-stream")],
                    r#"data: {"choices":[{"delta":{"content":"hi"}}]}

data: [DONE]

"#,
                )
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let address = listener.local_addr().unwrap();
        tokio::spawn(async move {
            axum::serve(listener, upstream).await.unwrap();
        });
        let metrics = Arc::new(Metrics::new());
        let (queue, rx) = mpsc::channel(QUEUE_CAPACITY);
        let state = AppState {
            config: Arc::new(config(&format!("http://{}", address))),
            metrics: metrics.clone(),
            queue,
        };
        tokio::spawn(run_dispatcher(rx, state.config.clone(), metrics));
        let response = build_app(state).oneshot(HttpRequest::builder().method("POST").uri("/v1/chat/completions").header("authorization", "Bearer open-code-test-token").header("content-type", "application/json").body(Body::from(r#"{"messages":[{"role":"user","content":"hello"}],"stream":true,"max_tokens":8}"#)).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        assert!(response.headers().get("x-request-id").is_some());
        let body = to_bytes(response.into_body(), 1024 * 1024).await.unwrap();
        assert!(String::from_utf8_lossy(&body).contains("[DONE]"));
    }
}
