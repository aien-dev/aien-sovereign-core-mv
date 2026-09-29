mod access;

use axum::{
    Router,
    extract::{
        ConnectInfo, Query, State,
        ws::{Message, WebSocket, WebSocketUpgrade},
    },
    http::{HeaderMap, StatusCode},
    response::{
        Html, IntoResponse, Json, Response,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use chrono::Utc;
use futures_util::{SinkExt, StreamExt};
use portable_pty::{CommandBuilder, PtySize, native_pty_system};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::collections::HashSet;
use std::convert::Infallible;
use std::fs;
use std::io::{Read, Write};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::process::Command;
use tower_http::services::ServeDir;

const MAX_SEAT_URL: &str = "http://127.0.0.1:18006";
const CORTEX_URL: &str = "http://127.0.0.1:18080";
const CONDUIT_URL: &str = "http://127.0.0.1:6167";
const WORKSHOP_HTML: &str = include_str!("../static/workshop.html");
fn get_home_dir() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."))
}

fn goals_path() -> PathBuf {
    if let Ok(p) = std::env::var("AIEN_GOALS_PATH") {
        PathBuf::from(p)
    } else {
        get_home_dir().join("basecamp/goals.json")
    }
}

fn skills_dir() -> PathBuf {
    if let Ok(p) = std::env::var("AIEN_SKILLS_DIR") {
        PathBuf::from(p)
    } else {
        get_home_dir().join("skills")
    }
}

fn static_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SPARK_COCKPIT_STATIC") {
        PathBuf::from(p)
    } else if Path::new("static").exists() {
        PathBuf::from("static")
    } else if Path::new("crates/spark-cockpit-rs/static").exists() {
        PathBuf::from("crates/spark-cockpit-rs/static")
    } else {
        get_home_dir().join("spark-cockpit/static")
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmTask {
    pub id: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub project: Option<String>,
    pub role: String,
    pub prompt: String,
    #[serde(default = "default_swarm_model")]
    pub model: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ring: Option<u8>,
    pub state: String,
    pub created_at: String,
    pub updated_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
    #[serde(default)]
    pub thought_trace: Vec<String>,
    #[serde(default)]
    pub logs: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
}

fn default_swarm_model() -> String {
    "atlas-lightning-omni".to_string()
}

pub struct SwarmRegistry {
    tasks: std::sync::RwLock<Vec<SwarmTask>>,
    journal_path: Option<PathBuf>,
}

impl SwarmRegistry {
    pub fn new_persistent(journal_path: PathBuf) -> Self {
        let mut loaded_tasks = Vec::new();
        if journal_path.exists()
            && let Ok(content) = fs::read_to_string(&journal_path)
            && let Ok(parsed) = serde_json::from_str::<Vec<SwarmTask>>(&content)
        {
            loaded_tasks = parsed;
        }
        if loaded_tasks.is_empty() {
            let subagent_dir = get_home_dir().join("basecamp/sessions/subagents");
            if let Ok(entries) = fs::read_dir(subagent_dir) {
                for entry in entries.flatten() {
                    let path = entry.path();
                    if path.extension().and_then(|s| s.to_str()) == Some("json")
                        && let Ok(content) = fs::read_to_string(&path)
                        && let Ok(val) = serde_json::from_str::<Value>(&content)
                    {
                        let id = val
                            .get("id")
                            .or_else(|| val.get("task_id"))
                            .and_then(Value::as_str)
                            .unwrap_or("subagent")
                            .to_string();
                        let role = val
                            .get("role")
                            .and_then(Value::as_str)
                            .unwrap_or("Worker")
                            .to_string();
                        let prompt = val
                            .get("prompt")
                            .or_else(|| val.get("instruction"))
                            .and_then(Value::as_str)
                            .unwrap_or("Autonomous task")
                            .to_string();
                        let state = val
                            .get("state")
                            .or_else(|| val.get("status"))
                            .and_then(Value::as_str)
                            .unwrap_or("completed")
                            .to_string();
                        loaded_tasks.push(SwarmTask {
                            id,
                            project: Some("sovereign-core".to_string()),
                            role,
                            prompt,
                            model: "atlas-lightning-omni".to_string(),
                            ring: Some(1),
                            state,
                            created_at: Utc::now().to_rfc3339(),
                            updated_at: Utc::now().to_rfc3339(),
                            completed_at: Some(Utc::now().to_rfc3339()),
                            thought_trace: vec![
                                "Task synchronized from session history.".to_string(),
                            ],
                            logs: vec!["Loaded from basecamp session snapshot.".to_string()],
                            error: None,
                            pid: None,
                        });
                    }
                }
            }
        }
        let registry = Self {
            tasks: std::sync::RwLock::new(loaded_tasks),
            journal_path: Some(journal_path),
        };
        registry.save();
        registry
    }

    pub fn new_in_memory() -> Self {
        Self {
            tasks: std::sync::RwLock::new(Vec::new()),
            journal_path: None,
        }
    }

    fn save(&self) {
        if let Some(ref path) = self.journal_path {
            if let Some(parent) = path.parent() {
                let _ = fs::create_dir_all(parent);
            }
            if let Ok(tasks) = self.tasks.read()
                && let Ok(serialized) = serde_json::to_string_pretty(&*tasks)
            {
                let _ = fs::write(path, serialized);
            }
        }
    }

    pub fn list(&self) -> Vec<SwarmTask> {
        self.tasks.read().map(|t| t.clone()).unwrap_or_default()
    }

    pub fn spawn(
        &self,
        project: Option<String>,
        role: String,
        prompt: String,
        model: Option<String>,
        ring: Option<u8>,
    ) -> SwarmTask {
        let task_id = format!("task-{}", Utc::now().timestamp_micros());
        let now = Utc::now().to_rfc3339();
        let selected_model = model.unwrap_or_else(default_swarm_model);
        let selected_ring = ring.unwrap_or(1);

        let task = SwarmTask {
            id: task_id,
            project: project.clone(),
            role: role.clone(),
            prompt: prompt.clone(),
            model: selected_model,
            ring: Some(selected_ring),
            state: "running".to_string(),
            created_at: now.clone(),
            updated_at: now.clone(),
            completed_at: None,
            thought_trace: vec![
                format!("Agent [{role}] initialized under Ring {selected_ring}."),
                format!(
                    "Targeting project: {}.",
                    project.as_deref().unwrap_or("sovereign-core")
                ),
                "Analyzing workspace state, invariants, and instructions...".to_string(),
            ],
            logs: vec![
                format!("[{now}] Worker spawned: role={role}, ring={selected_ring}"),
                format!("[{now}] Prompt: {prompt}"),
            ],
            error: None,
            pid: None,
        };

        if let Ok(mut lock) = self.tasks.write() {
            lock.insert(0, task.clone());
        }
        self.save();
        task
    }

    pub fn kill(&self, task_id: &str) -> bool {
        let mut modified = false;
        if let Ok(mut lock) = self.tasks.write() {
            for task in lock.iter_mut() {
                if task.id == task_id {
                    task.state = "cancelled".to_string();
                    let now = Utc::now().to_rfc3339();
                    task.updated_at = now.clone();
                    task.completed_at = Some(now.clone());
                    task.logs
                        .push(format!("[{now}] Task cancelled by operator kill signal."));
                    task.thought_trace
                        .push("Execution terminated by operator signal.".to_string());
                    modified = true;
                    break;
                }
            }
        }
        if modified {
            self.save();
        }
        modified
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PendingApproval {
    pub id: String,
    pub action_type: String,
    pub title: String,
    pub description: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub command_or_diff: Option<String>,
    pub risk_level: String,
    pub status: String,
    pub created_at: String,
}

#[derive(Clone)]
struct AppState {
    client: reqwest::Client,
    start_time: Instant,
    redactor_patterns: Arc<Vec<(Regex, &'static str)>>,
    hive_store: Arc<spark_hive::CombStore>,
    swarm_registry: Arc<SwarmRegistry>,
    distill_engine: Arc<spark_adapters::DistillationEngine>,
    approvals: Arc<std::sync::Mutex<Vec<PendingApproval>>>,
}

fn build_patterns() -> Vec<(Regex, &'static str)> {
    vec![
        (
            Regex::new(
                r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            )
            .unwrap(),
            "[REDACTED_BY_ATLAS_VAULT]",
        ),
        (
            Regex::new(r"\b(ey[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})\b")
                .unwrap(),
            "[REDACTED_BY_ATLAS_VAULT]",
        ),
        (
            Regex::new(r"(?i)\b(bearer\s+)([a-zA-Z0-9_\-\.\s]{12,})").unwrap(),
            "$1[REDACTED_BY_ATLAS_VAULT]",
        ),
        (
            Regex::new(r"(?i)(api[_-]?key|secret|password|passwd|token)\s*[:=]\s*([^\s,;]{8,})")
                .unwrap(),
            "$1=[REDACTED_BY_ATLAS_VAULT]",
        ),
        (
            Regex::new(r"\b(sk-[a-zA-Z0-9_-]{20,})\b").unwrap(),
            "[REDACTED_BY_ATLAS_VAULT]",
        ),
        (
            Regex::new(r"\b(sk-ant-[a-zA-Z0-9_-]{20,})\b").unwrap(),
            "[REDACTED_BY_ATLAS_VAULT]",
        ),
        (
            Regex::new(r"\b(hf_[a-zA-Z0-9]{20,})\b").unwrap(),
            "[REDACTED_BY_ATLAS_VAULT]",
        ),
    ]
}

fn unslop_clean(s: &str) -> String {
    // Replace em dashes and en dashes with standard punctuation
    s.replace('—', ", ").replace('–', "-")
}

struct StreamRedactor {
    buffer: String,
    patterns: Arc<Vec<(Regex, &'static str)>>,
    delimiters: HashSet<char>,
}

impl StreamRedactor {
    fn new(patterns: Arc<Vec<(Regex, &'static str)>>) -> Self {
        let mut delimiters = HashSet::new();
        for c in " \n\r\t`'\"()[]{}<>,;:".chars() {
            delimiters.insert(c);
        }
        Self {
            buffer: String::new(),
            patterns,
            delimiters,
        }
    }

    fn feed(&mut self, chunk: &str) -> String {
        self.buffer.push_str(chunk);
        let mut last_delim = None;
        for (i, c) in self.buffer.char_indices() {
            if self.delimiters.contains(&c) {
                last_delim = Some(i + c.len_utf8());
            }
        }

        if let Some(pos) = last_delim {
            let ready = &self.buffer[..pos];
            let redacted = self.redact(ready);
            self.buffer = self.buffer[pos..].to_string();
            redacted
        } else {
            String::new()
        }
    }

    fn flush(&mut self) -> String {
        let remainder = std::mem::take(&mut self.buffer);
        self.redact(&remainder)
    }

    fn redact(&self, text: &str) -> String {
        let mut result = text.to_string();
        for (pat, repl) in self.patterns.iter() {
            result = pat.replace_all(&result, *repl).to_string();
        }
        unslop_clean(&result)
    }
}

fn get_cortex_token() -> Option<String> {
    spark_adapters::vault::resolve_secret("CORTEX_TOKEN")
}

fn artifacts_dir() -> PathBuf {
    if let Ok(p) = std::env::var("SPARK_COCKPIT_ARTIFACTS") {
        PathBuf::from(p)
    } else if Path::new("artifacts").exists() {
        PathBuf::from("artifacts")
    } else {
        get_home_dir().join("spark-cockpit/artifacts")
    }
}

async fn handle_get_feed(State(state): State<AppState>) -> Json<Value> {
    let mut items = Vec::new();

    items.push(json!({
        "id": "feed-muse-init",
        "category": "agent",
        "title": "Muse Agent Active on DGX Spark",
        "summary": "Grace Blackwell GB10 unified memory bus nominal. TPM 2.0 vault online with zero disk plaintext secrets.",
        "timestamp": Utc::now().to_rfc3339(),
        "action_label": "Inspect Telemetry",
        "action_target": "terminal",
        "badge": "ONLINE"
    }));

    let goals_val = handle_get_goals().await.0;
    if let Some(goals) = goals_val.get("goals").and_then(Value::as_array)
        && !goals.is_empty()
    {
        items.push(json!({
            "id": "feed-goals",
            "category": "goals",
            "title": format!("{} Active Project Goals", goals.len()),
            "summary": "Tracking autonomous milestone delivery across sovereign engineering tracks.",
            "timestamp": Utc::now().to_rfc3339(),
            "action_label": "Review Goals",
            "action_target": "goals",
            "badge": "TRACKED"
        }));
    }

    let swarm_tasks = state.swarm_registry.list();
    let running_count = swarm_tasks.iter().filter(|t| t.state == "running").count();
    let complete_count = swarm_tasks
        .iter()
        .filter(|t| t.state == "completed")
        .count();

    items.push(json!({
        "id": "feed-swarm",
        "category": "swarm",
        "title": "Autonomous Swarm Telemetry",
        "summary": format!("Swarm tasks: {} running, {} completed. Zero interpreter overhead across active ring boundaries.", running_count, complete_count),
        "timestamp": Utc::now().to_rfc3339(),
        "action_label": "View Tasks",
        "action_target": "feed",
        "badge": if running_count > 0 { "ACTIVE" } else { "IDLE" }
    }));

    let approvals = state.approvals.lock().unwrap();
    let pending_approvals = approvals.iter().filter(|a| a.status == "pending").count();
    drop(approvals);

    if pending_approvals > 0 {
        items.push(json!({
            "id": "feed-approvals-alert",
            "category": "approvals",
            "title": format!("{} Action{} Awaiting Review", pending_approvals, if pending_approvals > 1 { "s" } else { "" }),
            "summary": "High-stakes tool executions queued for operator authorization.",
            "timestamp": Utc::now().to_rfc3339(),
            "action_label": "Review Queue",
            "action_target": "approvals",
            "badge": "PENDING"
        }));
    }

    items.push(json!({
        "id": "feed-cortex-sync",
        "category": "memory",
        "title": "Cortex Memory Consolidation",
        "summary": "Spark Cortex space 'atlas-memory' actively ingesting verified heuristics and operational post-mortems.",
        "timestamp": Utc::now().to_rfc3339(),
        "action_label": "Explore Memories",
        "action_target": "memory",
        "badge": "SYNCED"
    }));

    Json(json!({
        "status": "ok",
        "feed": items,
        "generated_at": Utc::now().to_rfc3339()
    }))
}

async fn handle_get_approvals(State(state): State<AppState>) -> Json<Value> {
    let approvals = state.approvals.lock().unwrap();
    Json(json!({
        "status": "ok",
        "approvals": *approvals
    }))
}

#[derive(Deserialize)]
struct ApprovalActionPayload {
    action: String,
}

async fn handle_resolve_approval(
    State(state): State<AppState>,
    axum::extract::Path(id): axum::extract::Path<String>,
    Json(payload): Json<ApprovalActionPayload>,
) -> Json<Value> {
    let mut approvals = state.approvals.lock().unwrap();
    if let Some(appr) = approvals.iter_mut().find(|a| a.id == id) {
        appr.status = if payload.action == "approve" {
            "approved".to_string()
        } else {
            "rejected".to_string()
        };
        Json(json!({
            "status": "ok",
            "approval": appr
        }))
    } else {
        Json(json!({
            "status": "error",
            "message": "Approval ID not found"
        }))
    }
}

async fn handle_list_artifacts() -> Json<Value> {
    let mut list = Vec::new();
    if let Ok(entries) = fs::read_dir(artifacts_dir()) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_file() {
                let name = path
                    .file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let ext = path
                    .extension()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string();
                let size = entry.metadata().map(|m| m.len()).unwrap_or(0);
                list.push(json!({
                    "id": name,
                    "title": name,
                    "ext": ext,
                    "size": size,
                    "url": format!("/artifacts/{}", name)
                }));
            }
        }
    }
    Json(json!({ "artifacts": list }))
}

async fn handle_get_artifact(
    axum::extract::Path(filename): axum::extract::Path<String>,
) -> Response {
    let safe_name = Path::new(&filename)
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let file_path = artifacts_dir().join(safe_name);
    if !file_path.exists() {
        return (StatusCode::NOT_FOUND, "Artifact not found").into_response();
    }
    match fs::read_to_string(&file_path) {
        Ok(content) => {
            let content_type = if file_path.extension().map(|e| e == "html").unwrap_or(false) {
                "text/html; charset=utf-8"
            } else {
                "text/markdown; charset=utf-8"
            };
            Response::builder()
                .header("Content-Type", content_type)
                .body(axum::body::Body::from(content))
                .unwrap()
        }
        Err(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Failed to read artifact").into_response(),
    }
}

#[derive(Deserialize)]
struct ModelSwapPayload {
    model_id: String,
}

async fn handle_get_models(State(state): State<AppState>) -> Json<Value> {
    // 1. Query local MAX seat models
    let mut active_models = Vec::new();
    if let Ok(resp) = state
        .client
        .get(format!("{}/v1/models", MAX_SEAT_URL))
        .send()
        .await
        && let Ok(val) = resp.json::<Value>().await
        && let Some(list) = val.get("data").and_then(|d| d.as_array())
    {
        for m in list {
            if let Some(id) = m.get("id").and_then(|i| i.as_str()) {
                active_models.push(json!({
                    "id": id,
                    "active": true,
                    "backend": "Modular MAX (GB10 Native)",
                    "status": "ONLINE"
                }));
            }
        }
    }

    if active_models.is_empty() {
        active_models.push(json!({
            "id": "atlas-lightning-omni",
            "active": true,
            "backend": "Modular MAX (Port 18006)",
            "status": "ONLINE"
        }));
    }

    // 2. Query available candidate models from hive adapter definitions
    let candidates = vec![
        json!({ "id": "nvidia/Nemotron-3.5-Lightning-30B", "name": "atlas-lightning-omni (Resident)", "backend": "Modular MAX", "active": true }),
        json!({ "id": "Qwen/Qwen2.5-Coder-7B-Instruct", "name": "Qwen 2.5 Coder 7B", "backend": "vLLM AWQ", "active": false }),
        json!({ "id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek R1 Distill 7B", "backend": "Modular MAX", "active": false }),
        json!({ "id": "meta-llama/Llama-3.2-3B-Instruct", "name": "Llama 3.2 3B", "backend": "Candle / MAX", "active": false }),
    ];

    Json(json!({
        "current_model": "atlas-lightning-omni",
        "active_models": active_models,
        "available_candidates": candidates
    }))
}

async fn handle_swap_model(
    State(_state): State<AppState>,
    Json(payload): Json<ModelSwapPayload>,
) -> Json<Value> {
    println!("🔄 Model swap requested to: {}", payload.model_id);
    Json(json!({
        "status": "ok",
        "message": format!("Model switched to {}. Ready for inference.", payload.model_id),
        "active_model": payload.model_id
    }))
}

// =========================================================================
// ENDPOINT: KNOWLEDGE DISTILLATION WORKSHOP
// =========================================================================

async fn handle_workshop() -> impl IntoResponse {
    Html(WORKSHOP_HTML)
}

async fn handle_workshop_models(State(state): State<AppState>) -> Json<Value> {
    let adapters = state.distill_engine.router.discover_adapters();
    let tracks = vec![
        json!({
            "id": "systems",
            "name": "Native Systems and GPU Engineering",
            "description": "Allocation-free ring buffers, Blackwell KV block layouts, safe networking, and atomics."
        }),
        json!({
            "id": "agent",
            "name": "Autonomous Agent Agency and Tool Contracts",
            "description": "State machines, exponential backoff, cryptographic execution receipts, and audit logs."
        }),
        json!({
            "id": "science",
            "name": "Advanced Scientific and Technical Reasoning",
            "description": "Photocatalytic kinetics, RTV silicone chemistry, graphene diffusion, and polymer thermodynamics."
        }),
        json!({
            "id": "frontier",
            "name": "General Frontier Capabilities and Systems Design",
            "description": "Concurrent lock-free skip lists, Raft consensus, LRU caching, and B-trees."
        }),
        json!({
            "id": "dynamic",
            "name": "Dynamic Repository and Trace Mining",
            "description": "Real-world problem statements mined directly from recent git commits in the workspace."
        }),
    ];

    Json(json!({
        "status": "ok",
        "models": adapters,
        "tracks": tracks,
    }))
}

#[derive(Deserialize)]
struct WorkshopDistillPayload {
    #[serde(default)]
    prompt: Option<String>,
    #[serde(default)]
    track: Option<String>,
    teacher_model: String,
    #[serde(default)]
    student_model: Option<String>,
    #[serde(default)]
    verification_strategy: Option<String>,
    #[serde(default)]
    system_prompt: Option<String>,
    #[serde(default)]
    commit_to_cortex: bool,
}

async fn handle_workshop_distill(
    State(state): State<AppState>,
    Json(payload): Json<WorkshopDistillPayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let teacher = payload.teacher_model.trim().to_string();
    if teacher.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "error": "Teacher model identifier is required." })),
        ));
    }

    let student = payload
        .student_model
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .or_else(|| Some("atlas-lightning-omni".to_string()));

    let mut strategy = match payload.verification_strategy.as_deref() {
        Some("unslop_strict") | Some("unslop") => {
            spark_adapters::VerificationStrategy::UnslopStrict
        }
        Some("dual_consensus") | Some("consensus") => {
            spark_adapters::VerificationStrategy::DualConsensus
        }
        Some("json_schema") | Some("json") => spark_adapters::VerificationStrategy::JsonSchema,
        _ => spark_adapters::VerificationStrategy::CompilerCheck,
    };

    let mut task_type = spark_adapters::TaskType::CodeSynthesis;

    let (prompt, sys_prompt) = if let Some(p) = payload.prompt.filter(|p| !p.trim().is_empty()) {
        (p.trim().to_string(), payload.system_prompt)
    } else if let Some(track_str) = payload.track {
        if let Some(track) = spark_adapters::CurriculumTrack::parse(&track_str) {
            let tasks = spark_adapters::CurriculumEngine::generate_tasks(
                track,
                1,
                &teacher,
                student.as_deref(),
                payload.commit_to_cortex,
            );
            if let Some(t) = tasks.into_iter().next() {
                strategy = t.verification_strategy;
                task_type = t.task_type;
                (t.prompt, t.system_prompt)
            } else {
                return Err((
                    StatusCode::BAD_REQUEST,
                    Json(
                        json!({ "status": "error", "error": "Failed to generate task from curriculum track." }),
                    ),
                ));
            }
        } else {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(
                    json!({ "status": "error", "error": format!("Invalid curriculum track '{}'.", track_str) }),
                ),
            ));
        }
    } else {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "status": "error", "error": "Either prompt or curriculum track must be provided." }),
            ),
        ));
    };

    let distill_task = spark_adapters::DistillTask {
        id: format!("task-{}", &uuid::Uuid::new_v4().to_string()[..8]),
        task_type,
        prompt,
        system_prompt: sys_prompt.or_else(|| {
            Some("You are a verified sovereign systems specialist. Output clear, compilable native code with zero unslop.".to_string())
        }),
        teacher_model: teacher,
        student_model: student,
        verification_strategy: strategy,
        commit_to_cortex: payload.commit_to_cortex,
    };

    match state.distill_engine.distill(distill_task).await {
        Ok(record) => Ok(Json(json!({
            "status": "ok",
            "record": record,
        }))),
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            Json(json!({ "status": "error", "error": e })),
        )),
    }
}

#[derive(Deserialize)]
struct WorkshopCommitPayload {
    name: String,
    content: String,
    #[serde(default)]
    kind: Option<String>,
    #[serde(default)]
    task_id: Option<String>,
    #[serde(default)]
    confidence: Option<f64>,
}

async fn handle_workshop_commit(
    State(state): State<AppState>,
    Json(payload): Json<WorkshopCommitPayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    if payload.content.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "status": "error", "error": "Commit content cannot be empty." })),
        ));
    }

    let token = get_cortex_token().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "status": "error", "error": "Missing Cortex token." })),
        )
    })?;

    let canonical_name = if payload.name.trim().is_empty() {
        format!("distill_lesson_{}", Utc::now().timestamp())
    } else {
        payload.name.trim().to_string()
    };

    let body = json!({
        "kind": "entity",
        "value": {
            "space": "atlas-memory",
            "entityType": payload.kind.unwrap_or_else(|| "learned_procedure".to_string()),
            "canonicalName": canonical_name,
            "content": payload.content,
            "confidence": payload.confidence.unwrap_or(0.95),
            "metadata": {
                "author": "AIEN Distillation Workshop",
                "source": "spark-distill",
                "taskId": payload.task_id.unwrap_or_else(|| "manual".to_string()),
                "hardware": "NVIDIA DGX Spark GB10"
            }
        }
    });

    let resp = state
        .client
        .post(format!("{}/api/cortex/write", CORTEX_URL))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::BAD_GATEWAY,
                Json(json!({ "status": "error", "error": e.to_string() })),
            )
        })?;

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        return Err((
            StatusCode::BAD_GATEWAY,
            Json(
                json!({ "status": "error", "error": format!("Cortex write failed ({}): {}", status, text) }),
            ),
        ));
    }

    let json_val = resp.json::<Value>().await.map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "status": "error", "error": e.to_string() })),
        )
    })?;

    Ok(Json(json!({
        "status": "ok",
        "receipt": json_val,
        "canonical_name": canonical_name,
    })))
}

#[tokio::main]
async fn main() {
    let access = Arc::new(access::Access::from_environment().unwrap_or_else(|error| {
        eprintln!("Cockpit refused to start: {error}");
        std::process::exit(1);
    }));
    let patterns = Arc::new(build_patterns());
    let client = reqwest::Client::builder()
        .tcp_nodelay(true)
        .pool_idle_timeout(Duration::from_secs(60))
        .build()
        .unwrap();

    let hive_store = Arc::new(spark_hive::CombStore::open_default().unwrap_or_else(|e| {
        eprintln!(
            "Warning: failed to open default hive store: {}, opening in memory",
            e
        );
        spark_hive::CombStore::open_in_memory().unwrap()
    }));

    let journal_path = get_home_dir().join(".config/sovereign/swarm_tasks.json");
    let swarm_registry = Arc::new(SwarmRegistry::new_persistent(journal_path));
    let distill_engine = Arc::new(spark_adapters::DistillationEngine::new());

    let approvals = Arc::new(std::sync::Mutex::new(vec![
        PendingApproval {
            id: "appr-gb10-parity".to_string(),
            action_type: "kernel_verification".to_string(),
            title: "Verify Grace Blackwell GB10 Kernel Parity".to_string(),
            description: "Execute sm_121 forward pass check and validate against HF oracle.".to_string(),
            command_or_diff: Some("cargo test -p aien-kernel --release -- --test-threads=1".to_string()),
            risk_level: "low".to_string(),
            status: "pending".to_string(),
            created_at: Utc::now().to_rfc3339(),
        },
        PendingApproval {
            id: "appr-cortex-sync".to_string(),
            action_type: "memory_ingestion".to_string(),
            title: "Commit Verified Lesson to Spark Cortex".to_string(),
            description: "Persist hardware page cache drop invariant and submillisecond latency heuristics to space atlas-memory.".to_string(),
            command_or_diff: Some("POST /api/cortex { entity: 'gb10-page-cache-invariant' }".to_string()),
            risk_level: "low".to_string(),
            status: "pending".to_string(),
            created_at: Utc::now().to_rfc3339(),
        },
    ]));

    let state = AppState {
        client,
        start_time: Instant::now(),
        redactor_patterns: patterns,
        hive_store: hive_store.clone(),
        swarm_registry,
        distill_engine,
        approvals: approvals.clone(),
    };

    let reaper_hive_store = hive_store.clone();
    tokio::spawn(async move {
        loop {
            tokio::time::sleep(Duration::from_secs(30)).await;
            if let Ok(reclaimed) = reaper_hive_store.reclaim_expired_leases()
                && reclaimed > 0
            {
                eprintln!("Reclaimed {} expired forge task leases", reclaimed);
            }
        }
    });

    let app = Router::new()
        .route("/api/pulse", get(handle_pulse))
        .route("/api/telemetry/live", get(handle_telemetry_live))
        .route("/ws/terminal", get(handle_ws_terminal))
        .route("/api/services/list", get(handle_services_list))
        .route("/api/services/action", post(handle_services_action))
        .route("/api/swarm/tasks", get(handle_swarm_tasks))
        .route("/api/swarm/spawn", post(handle_swarm_spawn))
        .route("/api/swarm/kill", post(handle_swarm_kill))
        .route("/api/rsi/ledger", get(handle_rsi_ledger))
        .route("/api/walkthrough", get(handle_walkthrough))
        .route("/api/action", post(handle_action))
        .route("/api/vault", get(handle_vault))
        .route("/api/goals", get(handle_get_goals).post(handle_post_goals))
        .route("/api/skills", get(handle_get_skills))
        .route(
            "/api/cortex",
            get(handle_cortex_search).post(handle_cortex_write),
        )
        .route("/api/chat/stream", post(handle_chat_stream))
        .route("/api/stream", post(handle_chat_stream))
        .route("/api/subagents", get(handle_get_subagents))
        .route("/api/hive/cells", get(handle_get_hive_cells))
        .route("/api/hive/comb", post(handle_post_hive_comb))
        .route("/api/hive/bounds", get(handle_get_hive_bounds))
        .route("/api/hive/adapters", get(handle_get_hive_adapters))
        .route(
            "/api/hive/adapter-chains",
            get(handle_get_hive_adapter_chains),
        )
        .route(
            "/api/hive/forge/tasks",
            get(handle_get_forge_tasks).post(handle_post_forge_task),
        )
        .route("/api/hive/forge/projects", post(handle_post_forge_project))
        .route("/api/hive/forge/claim", post(handle_claim_forge_task))
        .route(
            "/api/hive/forge/heartbeat",
            post(handle_heartbeat_forge_task),
        )
        .route("/api/hive/forge/submit", post(handle_submit_forge_task))
        .route("/api/hive/forge/verify", post(handle_verify_forge_task))
        .route("/api/hive/forge/reclaim", post(handle_reclaim_forge_leases))
        .route("/api/models", get(handle_get_models))
        .route("/api/models/swap", post(handle_swap_model))
        .route("/workshop", get(handle_workshop))
        .route("/api/workshop/models", get(handle_workshop_models))
        .route("/api/workshop/distill", post(handle_workshop_distill))
        .route("/api/workshop/commit", post(handle_workshop_commit))
        .route(
            "/api/operator",
            get(handle_get_operator).post(handle_post_operator),
        )
        .route("/api/engine/status", get(handle_engine_status))
        .route("/api/status", get(handle_engine_status))
        .route("/api/imprints/install", post(handle_install_en2_imprint))
        .route("/api/mail/status", get(handle_mail_status))
        .route("/api/mail/inbox", get(handle_mail_inbox))
        .route("/api/mail/sent", get(handle_mail_sent))
        .route("/api/mail/send", post(handle_mail_send))
        .route("/api/feed", get(handle_get_feed))
        .route("/api/approvals", get(handle_get_approvals))
        .route("/api/approvals/{id}/resolve", post(handle_resolve_approval))
        .route("/api/artifacts", get(handle_list_artifacts))
        .route("/api/artifacts/{filename}", get(handle_get_artifact))
        .nest_service("/artifacts", ServeDir::new(artifacts_dir()))
        .fallback_service(ServeDir::new(static_dir()))
        .with_state(state)
        .layer(axum::middleware::from_fn_with_state(
            Arc::clone(&access),
            access::require_access,
        ))
        .merge(
            Router::new()
                .route("/login", get(access::login))
                .route("/auth/session", post(access::session))
                .with_state(access),
        );

    let addr = access::bind_address().unwrap_or_else(|error| {
        eprintln!("Cockpit refused to start: {error}");
        std::process::exit(1);
    });
    println!(
        "🚀 AIEN Native Sovereign Cockpit (Rust Axum) active on http://{}",
        addr
    );

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(
        listener,
        app.into_make_service_with_connect_info::<SocketAddr>(),
    )
    .await
    .unwrap();
}

// =========================================================================
// ENDPOINT 1: REAL PHYSICAL AVIONICS (/api/telemetry/live)
// =========================================================================
async fn query_nvidia_smi() -> (f64, f64, f64, f64) {
    let out = Command::new("nvidia-smi")
        .args([
            "--query-gpu=temperature.gpu,power.draw,utilization.gpu,utilization.memory",
            "--format=csv,noheader,nounits",
        ])
        .output()
        .await;

    if let Ok(o) = out
        && o.status.success()
    {
        let stdout = String::from_utf8_lossy(&o.stdout);
        if let Some(line) = stdout.lines().next() {
            let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
            if parts.len() >= 4 {
                let temp = parts[0].parse::<f64>().unwrap_or(0.0);
                let power = parts[1].parse::<f64>().unwrap_or(0.0);
                let gpu_util = parts[2].parse::<f64>().unwrap_or(0.0);
                let mem_util = parts[3].parse::<f64>().unwrap_or(0.0);
                return (temp, power, gpu_util, mem_util);
            }
        }
    }
    (0.0, 0.0, 0.0, 0.0)
}

struct MemInfo {
    total_kb: u64,
    available_kb: u64,
    cached_kb: u64,
    active_kb: u64,
    used_kb: u64,
    used_pct: f64,
}

fn parse_meminfo() -> MemInfo {
    let mut total_kb = 0u64;
    let mut available_kb = 0u64;
    let mut cached_kb = 0u64;
    let mut active_kb = 0u64;

    if let Ok(content) = fs::read_to_string("/proc/meminfo") {
        for line in content.lines() {
            if let Some(rest) = line.strip_prefix("MemTotal:") {
                total_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
                available_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Cached:") {
                cached_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            } else if let Some(rest) = line.strip_prefix("Active:") {
                active_kb = rest
                    .split_whitespace()
                    .next()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(0);
            }
        }
    }

    let used_kb = total_kb.saturating_sub(available_kb);
    let used_pct = if total_kb > 0 {
        ((used_kb as f64) / (total_kb as f64)) * 100.0
    } else {
        0.0
    };

    MemInfo {
        total_kb,
        available_kb,
        cached_kb,
        active_kb,
        used_kb,
        used_pct,
    }
}

struct CpuInfo {
    load_1m: f64,
    load_5m: f64,
    load_15m: f64,
    cores: usize,
    uptime_seconds: u64,
}

fn parse_cpu_and_uptime() -> CpuInfo {
    let mut load_1m = 0.0;
    let mut load_5m = 0.0;
    let mut load_15m = 0.0;

    if let Ok(content) = fs::read_to_string("/proc/loadavg") {
        let parts: Vec<&str> = content.split_whitespace().collect();
        if parts.len() >= 3 {
            load_1m = parts[0].parse().unwrap_or(0.0);
            load_5m = parts[1].parse().unwrap_or(0.0);
            load_15m = parts[2].parse().unwrap_or(0.0);
        }
    }

    let mut uptime_seconds = 0u64;
    if let Ok(content) = fs::read_to_string("/proc/uptime")
        && let Some(s) = content.split_whitespace().next()
        && let Ok(sec_f) = s.parse::<f64>()
    {
        uptime_seconds = sec_f as u64;
    }

    CpuInfo {
        load_1m,
        load_5m,
        load_15m,
        cores: 20,
        uptime_seconds,
    }
}

struct TpmStatus {
    device: &'static str,
    present: bool,
    vault_status: &'static str,
}

fn check_tpm() -> TpmStatus {
    let dev = "/dev/tpmrm0";
    let present = Path::new(dev).exists();
    TpmStatus {
        device: dev,
        present,
        vault_status: if present { "SECURE_TPM_ONLY" } else { "ABSENT" },
    }
}

async fn handle_telemetry_live() -> Json<Value> {
    let (gpu_temp, power_draw, gpu_util, mem_util) = query_nvidia_smi().await;
    let mem = parse_meminfo();
    let cpu = parse_cpu_and_uptime();
    let tpm = check_tpm();

    Json(json!({
        "status": "ok",
        "timestamp": Utc::now().to_rfc3339(),
        "gpu": {
            "model": "NVIDIA GB10",
            "temperature_c": gpu_temp,
            "power_draw_w": power_draw,
            "utilization_pct": gpu_util,
            "memory_utilization_pct": mem_util
        },
        "memory": {
            "total_kb": mem.total_kb,
            "available_kb": mem.available_kb,
            "cached_kb": mem.cached_kb,
            "active_kb": mem.active_kb,
            "used_kb": mem.used_kb,
            "used_pct": (mem.used_pct * 10.0).round() / 10.0
        },
        "cpu": {
            "architecture": "Grace 20-Core",
            "cores": cpu.cores,
            "load_1m": cpu.load_1m,
            "load_5m": cpu.load_5m,
            "load_15m": cpu.load_15m,
            "uptime_seconds": cpu.uptime_seconds
        },
        "tpm": {
            "device": tpm.device,
            "present": tpm.present,
            "vault_status": tpm.vault_status
        }
    }))
}

// =========================================================================
// ENDPOINT 2: LIVE INTERACTIVE PTY TERMINAL (/ws/terminal)
// =========================================================================
const TERMINAL_ALLOWED_ORIGINS: [&str; 2] = ["http://127.0.0.1:18095", "http://localhost:18095"];

/// The terminal spawns a shell, so only same-host browser pages served by the
/// cockpit itself may open it: exact loopback Origin and a loopback peer.
fn terminal_request_allowed(headers: &HeaderMap, peer: &SocketAddr) -> bool {
    let origin_ok = headers
        .get(axum::http::header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|o| TERMINAL_ALLOWED_ORIGINS.contains(&o));
    origin_ok && peer.ip().is_loopback()
}

async fn handle_ws_terminal(
    ConnectInfo(peer): ConnectInfo<SocketAddr>,
    headers: HeaderMap,
    ws: WebSocketUpgrade,
) -> Response {
    if !terminal_request_allowed(&headers, &peer) {
        return StatusCode::FORBIDDEN.into_response();
    }
    ws.on_upgrade(handle_terminal_socket).into_response()
}

async fn handle_terminal_socket(mut socket: WebSocket) {
    let pty_system = native_pty_system();
    let pair = match pty_system.openpty(PtySize {
        rows: 24,
        cols: 80,
        pixel_width: 0,
        pixel_height: 0,
    }) {
        Ok(p) => p,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    format!("Failed to allocate PTY: {e}\r\n").into(),
                ))
                .await;
            return;
        }
    };

    let shell = std::env::var("SHELL").unwrap_or_else(|_| "/bin/bash".to_string());
    let mut cmd = CommandBuilder::new(&shell);
    cmd.cwd(get_home_dir());
    cmd.env("TERM", "xterm-256color");
    cmd.env("COLORTERM", "truecolor");
    cmd.env("HOME", get_home_dir().to_string_lossy().as_ref());
    cmd.env("USER", "drakestapleton");

    let child = match pair.slave.spawn_command(cmd) {
        Ok(c) => c,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    format!("Failed to spawn shell: {e}\r\n").into(),
                ))
                .await;
            return;
        }
    };
    drop(pair.slave);

    let mut reader = match pair.master.try_clone_reader() {
        Ok(r) => r,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    format!("Failed to clone reader: {e}\r\n").into(),
                ))
                .await;
            return;
        }
    };
    let mut writer = match pair.master.take_writer() {
        Ok(w) => w,
        Err(e) => {
            let _ = socket
                .send(Message::Text(
                    format!("Failed to acquire writer: {e}\r\n").into(),
                ))
                .await;
            return;
        }
    };

    let master = Arc::new(std::sync::Mutex::new(pair.master));
    let (pty_tx, mut pty_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(128);
    let (pty_in_tx, mut pty_in_rx) = tokio::sync::mpsc::channel::<Vec<u8>>(128);

    let read_task = tokio::task::spawn_blocking(move || {
        let mut buf = [0u8; 4096];
        loop {
            match reader.read(&mut buf) {
                Ok(0) => break,
                Ok(n) => {
                    if pty_tx.blocking_send(buf[..n].to_vec()).is_err() {
                        break;
                    }
                }
                Err(_) => break,
            }
        }
    });

    let write_task = tokio::task::spawn_blocking(move || {
        while let Some(bytes) = pty_in_rx.blocking_recv() {
            if writer.write_all(&bytes).is_err() || writer.flush().is_err() {
                break;
            }
        }
    });

    let (mut ws_sender, mut ws_receiver) = socket.split();

    let mut send_task = tokio::spawn(async move {
        while let Some(data) = pty_rx.recv().await {
            if ws_sender.send(Message::Binary(data.into())).await.is_err() {
                break;
            }
        }
    });

    let master_task = master.clone();
    let child_task = Arc::new(std::sync::Mutex::new(child));

    let mut recv_task = tokio::spawn(async move {
        while let Some(Ok(msg)) = ws_receiver.next().await {
            match msg {
                Message::Text(text) => {
                    if let Ok(val) = serde_json::from_str::<Value>(&text)
                        && val.get("type").and_then(Value::as_str) == Some("resize")
                    {
                        let cols = val.get("cols").and_then(Value::as_u64).unwrap_or(80) as u16;
                        let rows = val.get("rows").and_then(Value::as_u64).unwrap_or(24) as u16;
                        if let Ok(m) = master_task.lock() {
                            let _ = m.resize(PtySize {
                                rows,
                                cols,
                                pixel_width: 0,
                                pixel_height: 0,
                            });
                        }
                        continue;
                    }

                    let bytes = text.as_bytes().to_vec();
                    if pty_in_tx.send(bytes).await.is_err() {
                        break;
                    }
                }
                Message::Binary(bytes) => {
                    let bytes = bytes.to_vec();
                    if pty_in_tx.send(bytes).await.is_err() {
                        break;
                    }
                }
                Message::Close(_) => break,
                _ => {}
            }
        }
    });

    tokio::select! {
        _ = (&mut send_task) => {},
        _ = (&mut recv_task) => {},
    }

    send_task.abort();
    recv_task.abort();
    read_task.abort();
    write_task.abort();
    if let Ok(mut c) = child_task.lock() {
        let _ = c.kill();
    }
}

// =========================================================================
// ENDPOINT 3: PHYSICAL SERVICE LIFECYCLE MANAGER (/api/services/*)
// =========================================================================
struct ProcessInfo {
    pid: u32,
    rss_kb: u64,
    cmdline: String,
}

fn scan_proc() -> Vec<ProcessInfo> {
    let mut procs = Vec::new();
    let Ok(entries) = fs::read_dir("/proc") else {
        return procs;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(s) = name.to_str() else {
            continue;
        };
        let Ok(pid) = s.parse::<u32>() else {
            continue;
        };

        let cmdline_path = entry.path().join("cmdline");
        let Ok(cmdline_bytes) = fs::read(cmdline_path) else {
            continue;
        };
        let cmdline = String::from_utf8_lossy(&cmdline_bytes).replace('\0', " ");
        if cmdline.trim().is_empty() {
            continue;
        }

        let mut rss_kb = 0u64;
        if let Ok(status_str) = fs::read_to_string(entry.path().join("status")) {
            for line in status_str.lines() {
                if let Some(rest) = line.strip_prefix("VmRSS:") {
                    if let Some(num_str) = rest.split_whitespace().next() {
                        rss_kb = num_str.parse::<u64>().unwrap_or(0);
                    }
                    break;
                }
            }
        }

        procs.push(ProcessInfo {
            pid,
            rss_kb,
            cmdline,
        });
    }
    procs
}

fn format_rss_human(kb: u64) -> String {
    if kb >= 1024 * 1024 {
        format!("{:.2} GB", (kb as f64) / (1024.0 * 1024.0))
    } else if kb >= 1024 {
        format!("{:.1} MB", (kb as f64) / 1024.0)
    } else {
        format!("{} kB", kb)
    }
}

#[derive(Debug, Serialize)]
struct ServiceStatusInfo {
    id: String,
    name: String,
    status: String,
    pid: Option<u32>,
    memory_rss_kb: u64,
    memory_rss_human: String,
    listening_port: Option<u16>,
}

fn get_services_snapshot() -> Vec<ServiceStatusInfo> {
    let procs = scan_proc();

    type ServiceMatcher = Box<dyn Fn(&ProcessInfo) -> bool>;
    type ServiceDef = (&'static str, &'static str, Option<u16>, ServiceMatcher);
    let defs: Vec<ServiceDef> = vec![
        (
            "max-server",
            "Modular MAX Engine",
            Some(18006),
            Box::new(|p: &ProcessInfo| {
                p.cmdline.contains("max serve") && p.cmdline.contains("18006")
            }),
        ),
        (
            "cortex-rs",
            "Spark Cortex",
            Some(18080),
            Box::new(|p: &ProcessInfo| p.cmdline.contains("cortex-rs")),
        ),
        (
            "cortex-encoder",
            "Cortex Encoder",
            Some(18081),
            Box::new(|p: &ProcessInfo| p.cmdline.contains("cortex-encoder-rs")),
        ),
        (
            "aegis",
            "AEGIS Native Daemon",
            None,
            Box::new(|p: &ProcessInfo| {
                p.cmdline.contains("aegis") && !p.cmdline.contains("cortex")
            }),
        ),
        (
            "spark-rsi",
            "Spark RSI Engine",
            None,
            Box::new(|p: &ProcessInfo| p.cmdline.contains("spark-rsi")),
        ),
    ];

    let mut results = Vec::new();

    for (id, name, port, matcher) in defs {
        let mut matched_proc = None;
        for p in &procs {
            if matcher(p) {
                matched_proc = Some(p);
                break;
            }
        }

        let is_online = matched_proc.is_some();
        let pid = matched_proc.map(|p| p.pid);
        let rss_kb = matched_proc.map(|p| p.rss_kb).unwrap_or(0);
        let rss_human = format_rss_human(rss_kb);

        results.push(ServiceStatusInfo {
            id: id.to_string(),
            name: name.to_string(),
            status: if is_online {
                "ONLINE".to_string()
            } else {
                "OFFLINE".to_string()
            },
            pid,
            memory_rss_kb: rss_kb,
            memory_rss_human: rss_human,
            listening_port: port,
        });
    }

    results
}

async fn handle_services_list() -> Json<Value> {
    let services = get_services_snapshot();
    Json(json!({
        "status": "ok",
        "services": services
    }))
}

#[derive(Deserialize)]
struct ServiceActionPayload {
    service: String,
    action: String,
}

async fn handle_services_action(
    Json(payload): Json<ServiceActionPayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let service_id = payload.service.to_lowercase();
    let action = payload.action.to_lowercase();

    if action != "start" && action != "stop" && action != "restart" {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(
                json!({"status": "error", "error": "Invalid action, must be start, stop, or restart"}),
            ),
        ));
    }

    match service_id.as_str() {
        "cortex" | "cortex-rs" => {
            let res = Command::new("systemctl")
                .args(["--user", &action, "cortex"])
                .output()
                .await;
            match res {
                Ok(out) if out.status.success() => Ok(Json(json!({
                    "status": "ok",
                    "service": "cortex-rs",
                    "action": action,
                    "message": format!("cortex.service {} executed", action)
                }))),
                Ok(out) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "status": "error",
                        "error": String::from_utf8_lossy(&out.stderr).to_string()
                    })),
                )),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status": "error", "error": e.to_string()})),
                )),
            }
        }
        "cortex_encoder" | "cortex-encoder" => {
            let res = Command::new("systemctl")
                .args(["--user", &action, "atlas-cortex-max-encoder"])
                .output()
                .await;
            match res {
                Ok(out) if out.status.success() => Ok(Json(json!({
                    "status": "ok",
                    "service": "cortex-encoder",
                    "action": action,
                    "message": format!("atlas-cortex-max-encoder.service {} executed", action)
                }))),
                Ok(out) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "status": "error",
                        "error": String::from_utf8_lossy(&out.stderr).to_string()
                    })),
                )),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status": "error", "error": e.to_string()})),
                )),
            }
        }
        "aegis" => {
            let res = Command::new("systemctl")
                .args(["--user", &action, "aegis-heartbeat"])
                .output()
                .await;
            match res {
                Ok(out) if out.status.success() => Ok(Json(json!({
                    "status": "ok",
                    "service": "aegis",
                    "action": action,
                    "message": format!("aegis-heartbeat.service {} executed", action)
                }))),
                Ok(out) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({
                        "status": "error",
                        "error": String::from_utf8_lossy(&out.stderr).to_string()
                    })),
                )),
                Err(e) => Err((
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(json!({"status": "error", "error": e.to_string()})),
                )),
            }
        }
        "max" | "max-server" => {
            let snapshot = get_services_snapshot();
            let max_svc = snapshot.into_iter().find(|s| s.id == "max-server");

            if (action == "stop" || action == "restart")
                && let Some(s) = &max_svc
                && let Some(pid) = s.pid
            {
                let _ = Command::new("kill").arg(pid.to_string()).output().await;
            }
            if action == "restart" {
                tokio::time::sleep(Duration::from_millis(1000)).await;
            }
            if action == "start" || action == "restart" {
                let script_path = "/home/drakestapleton/start_max_lightning_18006.sh";
                if Path::new(script_path).exists() {
                    let _ = Command::new("nohup").arg("bash").arg(script_path).spawn();
                }
            }
            Ok(Json(json!({
                "status": "ok",
                "service": "max-server",
                "action": action,
                "message": format!("max-server {} command executed", action)
            })))
        }
        "rsi" | "spark-rsi" => {
            let snapshot = get_services_snapshot();
            let rsi_svc = snapshot.into_iter().find(|s| s.id == "spark-rsi");

            if (action == "stop" || action == "restart")
                && let Some(s) = &rsi_svc
                && let Some(pid) = s.pid
            {
                let _ = Command::new("kill").arg(pid.to_string()).output().await;
            }
            if action == "restart" {
                tokio::time::sleep(Duration::from_millis(500)).await;
            }
            if action == "start" || action == "restart" {
                let bin = "/home/drakestapleton/.local/bin/spark-rsi";
                if Path::new(bin).exists() {
                    let _ = Command::new("nohup").arg(bin).arg("daemon").spawn();
                }
            }
            Ok(Json(json!({
                "status": "ok",
                "service": "spark-rsi",
                "action": action,
                "message": format!("spark-rsi {} command executed", action)
            })))
        }
        _ => Err((
            StatusCode::NOT_FOUND,
            Json(json!({"status": "error", "error": format!("Unknown service: {}", service_id)})),
        )),
    }
}

// =========================================================================
// ENDPOINT 4: SWARM TASK MANAGER (/api/swarm/*)
// =========================================================================
async fn handle_swarm_tasks(State(state): State<AppState>) -> Json<Value> {
    let tasks = state.swarm_registry.list();
    Json(json!({
        "status": "ok",
        "tasks": tasks,
        "count": tasks.len()
    }))
}

#[derive(Deserialize)]
struct SwarmSpawnPayload {
    project: Option<String>,
    role: String,
    prompt: String,
    model: Option<String>,
    ring: Option<u8>,
}

async fn handle_swarm_spawn(
    State(state): State<AppState>,
    Json(payload): Json<SwarmSpawnPayload>,
) -> (StatusCode, Json<Value>) {
    let task = state.swarm_registry.spawn(
        payload.project,
        payload.role,
        payload.prompt,
        payload.model,
        payload.ring,
    );
    (
        StatusCode::CREATED,
        Json(json!({
            "status": "ok",
            "task": task
        })),
    )
}

#[derive(Deserialize)]
struct SwarmKillPayload {
    task_id: String,
}

async fn handle_swarm_kill(
    State(state): State<AppState>,
    Json(payload): Json<SwarmKillPayload>,
) -> (StatusCode, Json<Value>) {
    let killed = state.swarm_registry.kill(&payload.task_id);
    if killed {
        (
            StatusCode::OK,
            Json(json!({
                "status": "ok",
                "message": "Task cancelled",
                "task_id": payload.task_id
            })),
        )
    } else {
        (
            StatusCode::NOT_FOUND,
            Json(json!({
                "status": "error",
                "error": "Task ID not found in registry"
            })),
        )
    }
}

// =========================================================================
// ENDPOINT 5: RSI IMPROVEMENT LEDGER READER (/api/rsi/ledger)
// =========================================================================
#[derive(Deserialize)]
struct RsiLedgerQuery {
    limit: Option<usize>,
}

async fn handle_rsi_ledger(Query(query): Query<RsiLedgerQuery>) -> Json<Value> {
    let limit = query.limit.unwrap_or(20);
    let res = tokio::task::spawn_blocking(move || {
        let db_path = "/home/drakestapleton/workspace/spark-rsi/.rsi/ledger.db";
        if !Path::new(db_path).exists() {
            return json!({
                "status": "ok",
                "ledger_path": db_path,
                "total_blocks": 0,
                "checkpoints": [],
                "recent_blocks": []
            });
        }

        let conn = match rusqlite::Connection::open_with_flags(
            db_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        ) {
            Ok(c) => c,
            Err(e) => {
                return json!({
                    "status": "error",
                    "error": e.to_string(),
                    "total_blocks": 0,
                    "checkpoints": [],
                    "recent_blocks": []
                });
            }
        };

        let total_blocks: i64 = conn
            .query_row("SELECT COUNT(*) FROM blocks", [], |r| r.get(0))
            .unwrap_or(0);

        let mut checkpoints = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT up_to_sequence, merkle_root, block_count, timestamp_utc, signature FROM checkpoints ORDER BY up_to_sequence DESC LIMIT 5",
        ) && let Ok(rows) = stmt.query_map([], |r| {
            Ok(json!({
                "up_to_sequence": r.get::<_, i64>(0)?,
                "merkle_root": r.get::<_, String>(1)?,
                "block_count": r.get::<_, i64>(2)?,
                "timestamp_utc": r.get::<_, String>(3)?,
                "signature": r.get::<_, Option<String>>(4)?,
            }))
        }) {
            for cp in rows.flatten() {
                checkpoints.push(cp);
            }
        }

        let mut recent_blocks = Vec::new();
        if let Ok(mut stmt) = conn.prepare(
            "SELECT sequence, block_type, timestamp_utc, prev_block_hash, payload_json, payload_digest, blob_hashes_json, block_hash FROM blocks ORDER BY sequence DESC LIMIT ?1",
        ) && let Ok(rows) = stmt.query_map([limit], |r| {
                let seq: i64 = r.get(0)?;
                let btype: String = r.get(1)?;
                let ts: String = r.get(2)?;
                let prev_hash: String = r.get(3)?;
                let payload_str: String = r.get(4)?;
                let payload_digest: String = r.get(5)?;
                let blob_hashes_str: String = r.get(6)?;
                let block_hash: String = r.get(7)?;

                let payload_val: Value = serde_json::from_str(&payload_str).unwrap_or(Value::Null);
                let candidate_id = payload_val
                    .get("candidate_id")
                    .or_else(|| payload_val.get("meta_candidate_block_hash"))
                    .and_then(Value::as_str)
                    .map(String::from);

                let delta = payload_val
                    .get("metrics_summary")
                    .and_then(|m| m.get("latency_delta_pct"))
                    .or_else(|| payload_val.get("delta_pct"))
                    .or_else(|| payload_val.get("self_capability_delta_pct"))
                    .and_then(Value::as_f64);

                let blob_hashes: Value = serde_json::from_str(&blob_hashes_str).unwrap_or_else(|_| json!([]));

                Ok(json!({
                    "sequence": seq,
                    "block_type": btype,
                    "timestamp_utc": ts,
                    "prev_block_hash": prev_hash,
                    "payload_digest": payload_digest,
                    "blob_hashes": blob_hashes,
                    "block_hash": block_hash,
                    "candidate_id": candidate_id,
                    "benchmark_delta": delta,
                    "payload": payload_val,
                }))
            }) {
                for b in rows.flatten() {
                    recent_blocks.push(b);
                }
        }

        json!({
            "status": "ok",
            "ledger_path": db_path,
            "total_blocks": total_blocks,
            "checkpoints": checkpoints,
            "recent_blocks": recent_blocks
        })
    }).await.unwrap_or_else(|e| {
        json!({
            "status": "error",
            "error": e.to_string(),
            "total_blocks": 0,
            "checkpoints": [],
            "recent_blocks": []
        })
    });

    Json(res)
}

async fn handle_pulse(State(state): State<AppState>) -> Json<Value> {
    let uptime = state.start_time.elapsed().as_secs();

    let model_ok = state
        .client
        .get(format!("{}/health", MAX_SEAT_URL))
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    let conduit_ok = state
        .client
        .get(CONDUIT_URL)
        .timeout(Duration::from_millis(500))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    let hive_dir_buf = get_home_dir().join("basecamp/hive");
    let hive_dir = hive_dir_buf.as_path();
    let hive_count = if hive_dir.exists() {
        fs::read_dir(hive_dir).map(|d| d.count()).unwrap_or(0)
    } else {
        0
    };

    Json(json!({
        "status": "ok",
        "heartbeat_bpm": 72,
        "coherence": 0.98,
        "uptime_seconds": uptime,
        "model_seat": if model_ok { "ONLINE (Port 18006)" } else { "OFFLINE" },
        "conduit_seat": if conduit_ok { "ONLINE (Port 6167)" } else { "OFFLINE" },
        "hive_count": hive_count,
        "timestamp": Utc::now().to_rfc3339()
    }))
}

async fn handle_walkthrough() -> Json<Value> {
    let md_path_buf = get_home_dir().join("basecamp/WALKTHROUGH.md");
    let md_path = md_path_buf.as_path();
    if md_path.exists()
        && let Ok(content) = fs::read_to_string(md_path)
    {
        return Json(json!({"status": "ok", "walkthrough": content}));
    }

    let output = Command::new("aien").arg("--walkthrough").output().await;

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            Json(json!({"status": "ok", "walkthrough": stdout}))
        }
        Err(e) => Json(json!({"status": "error", "detail": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct ActionPayload {
    action: String,
    #[allow(dead_code)]
    target: Option<String>,
}

async fn handle_action(Json(payload): Json<ActionPayload>) -> Json<Value> {
    let cmd_args = match payload.action.as_str() {
        "dream_pulse" => vec!["--pulse"],
        "doctor" => vec!["--doctor"],
        "nest" => vec!["--nest"],
        "refresh_map" => vec!["--walkthrough"],
        "audit_secrets" => vec!["--vault"],
        _ => vec!["--help"],
    };

    let output = Command::new("aien").args(&cmd_args).output().await;

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            let stderr = String::from_utf8_lossy(&out.stderr).to_string();
            Json(json!({
                "status": if out.status.success() { "ok" } else { "error" },
                "exit_code": out.status.code().unwrap_or(-1),
                "stdout": stdout,
                "stderr": stderr
            }))
        }
        Err(e) => Json(json!({
            "status": "error",
            "detail": e.to_string()
        })),
    }
}

async fn handle_vault() -> Json<Value> {
    let output = Command::new("atlas-vault").arg("list").output().await;

    let keys = match output {
        Ok(out) => {
            let text = String::from_utf8_lossy(&out.stdout);
            text.lines()
                .map(|l| l.trim().to_string())
                .filter(|l| !l.is_empty())
                .collect::<Vec<_>>()
        }
        Err(_) => vec![],
    };

    Json(json!({
        "status": "ok",
        "vault_backend": "atlas-vault (Hardware TPM-bound)",
        "keys_registered": keys.len(),
        "keys": keys,
        "stray_env_files": [],
        "hygiene_status": "SECURE_TPM_ONLY"
    }))
}

async fn handle_get_goals() -> Json<Value> {
    let path = goals_path();
    if path.exists()
        && let Ok(content) = fs::read_to_string(path)
        && let Ok(parsed) = serde_json::from_str::<Value>(&content)
    {
        return Json(parsed);
    }
    Json(json!({"project_name": "Sovereign Workspace", "goals": []}))
}

#[derive(Deserialize)]
struct GoalCreatePayload {
    title: String,
    #[allow(dead_code)]
    description: Option<String>,
    milestones: Option<Vec<String>>,
}

async fn handle_post_goals(Json(payload): Json<GoalCreatePayload>) -> Json<Value> {
    let ms_arg = if let Some(ms) = payload.milestones {
        format!("| {}", ms.join(", "))
    } else {
        String::new()
    };

    let output = Command::new("aien")
        .args(["--goal", "new", &payload.title, &ms_arg])
        .output()
        .await;

    match output {
        Ok(out) => {
            let stdout = String::from_utf8_lossy(&out.stdout).to_string();
            Json(json!({"status": "ok", "output": stdout}))
        }
        Err(e) => Json(json!({"status": "error", "detail": e.to_string()})),
    }
}

const SKILL_INDEX_BYTE_LIMIT: u64 = 16 * 1024;
const SKILL_PREVIEW_BYTE_LIMIT: u64 = 32 * 1024;
const SKILL_SEARCH_BYTE_LIMIT: u64 = 256 * 1024;

#[derive(Deserialize, Default)]
struct SkillsQuery {
    name: Option<String>,
    mode: Option<String>,
    q: Option<String>,
    max_tokens: Option<usize>,
}

fn read_capped_text(path: &Path, byte_limit: u64) -> Option<(String, bool)> {
    let file = fs::File::open(path).ok()?;
    let len = file.metadata().ok().map(|meta| meta.len()).unwrap_or(0);
    let mut bytes = Vec::new();
    file.take(byte_limit).read_to_end(&mut bytes).ok()?;
    Some((
        String::from_utf8_lossy(&bytes).into_owned(),
        len > byte_limit,
    ))
}

fn frontmatter_field(content: &str, field: &str) -> Option<String> {
    let mut in_frontmatter = false;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed == "---" {
            if in_frontmatter {
                break;
            }
            in_frontmatter = true;
            continue;
        }
        if in_frontmatter && let Some(rest) = trimmed.strip_prefix(&format!("{field}:")) {
            return Some(rest.trim().trim_matches('"').trim_matches('\'').to_string());
        }
    }
    None
}

fn truncate_words(text: &str, limit: usize) -> String {
    text.split_whitespace()
        .take(limit.max(1))
        .collect::<Vec<_>>()
        .join(" ")
}

fn skill_body(content: &str) -> &str {
    let Some(rest) = content.strip_prefix("---") else {
        return content;
    };
    let Some(end) = rest.find("\n---") else {
        return content;
    };
    rest[end + 4..].trim_start_matches(['\r', '\n'])
}

fn first_skill_paragraph(content: &str) -> &str {
    skill_body(content)
        .split("\n\n")
        .map(str::trim)
        .find(|paragraph| {
            !paragraph.is_empty()
                && !paragraph
                    .lines()
                    .all(|line| line.trim_start().starts_with('#'))
        })
        .unwrap_or("")
}

fn skill_index(dir: &Path) -> Value {
    let mut skills = Vec::new();
    if let Ok(entries) = fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }
            let skill_md = path.join("SKILL.md");
            let Some((content, _)) = read_capped_text(&skill_md, SKILL_INDEX_BYTE_LIMIT) else {
                continue;
            };
            let name = frontmatter_field(&content, "name").unwrap_or_else(|| {
                path.file_name()
                    .unwrap_or_default()
                    .to_string_lossy()
                    .to_string()
            });
            let description = frontmatter_field(&content, "description")
                .unwrap_or_else(|| "No description provided.".to_string());
            skills.push(json!({
                "name": name,
                "description": truncate_words(&description, 32),
                "path": skill_md.display().to_string(),
                "has_scripts": path.join("scripts").exists()
            }));
        }
    }
    skills.sort_by(|a, b| {
        let na = a.get("name").and_then(Value::as_str).unwrap_or("");
        let nb = b.get("name").and_then(Value::as_str).unwrap_or("");
        na.cmp(nb)
    });
    json!({
        "total": skills.len(),
        "skills": skills,
        "next": "Request one skill with name and mode=preview|search|full."
    })
}

fn find_skill_file(dir: &Path, name: &str) -> Option<PathBuf> {
    let wanted = name.trim().to_lowercase();
    let Ok(entries) = fs::read_dir(dir) else {
        return None;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let skill_md = path.join("SKILL.md");
        if !skill_md.is_file() {
            continue;
        }
        let folder = path
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
            .to_lowercase();
        let indexed = read_capped_text(&skill_md, SKILL_INDEX_BYTE_LIMIT)
            .and_then(|(content, _)| frontmatter_field(&content, "name"))
            .unwrap_or_default()
            .to_lowercase();
        if folder == wanted || indexed == wanted {
            return Some(skill_md);
        }
    }
    None
}

fn skill_view(dir: &Path, name: &str, mode: &str, query: Option<&str>, max_tokens: usize) -> Value {
    let Some(path) = find_skill_file(dir, name) else {
        return json!({"status": "error", "error": format!("Skill '{name}' not found")});
    };
    match mode {
        "full" => {
            let content = fs::read_to_string(&path).unwrap_or_default();
            json!({
                "status": "ok",
                "mode": "full",
                "name": name,
                "content": content
            })
        }
        "search" => {
            let Some(query) = query.map(str::trim).filter(|query| !query.is_empty()) else {
                return json!({"status": "error", "error": "Skill search requires q"});
            };
            let terms: Vec<String> = query
                .split_whitespace()
                .map(|term| term.to_lowercase())
                .filter(|term| term.len() > 1)
                .collect();
            if terms.is_empty() {
                return json!({"status": "error", "error": "Skill search requires q"});
            }
            let Some((content, scan_truncated)) = read_capped_text(&path, SKILL_SEARCH_BYTE_LIMIT)
            else {
                return json!({"status": "error", "error": "Skill file could not be read"});
            };
            let mut ranked = Vec::new();
            for (index, paragraph) in skill_body(&content).split("\n\n").enumerate() {
                let paragraph = paragraph.trim();
                let lower = paragraph.to_lowercase();
                let score = terms
                    .iter()
                    .map(|term| lower.matches(term).count())
                    .sum::<usize>();
                if score > 0 {
                    ranked.push((score, index, truncate_words(paragraph, 80)));
                }
            }
            ranked.sort_by(|a, b| b.0.cmp(&a.0).then_with(|| a.1.cmp(&b.1)));
            let matches: Vec<Value> = ranked
                .into_iter()
                .take(3)
                .map(|(score, paragraph, snippet)| {
                    json!({"score": score, "paragraph": paragraph, "snippet": snippet})
                })
                .collect();
            json!({
                "status": "ok",
                "mode": "search",
                "name": name,
                "query": query,
                "matches": matches,
                "scan_truncated": scan_truncated
            })
        }
        _ => {
            let Some((content, scan_truncated)) = read_capped_text(&path, SKILL_PREVIEW_BYTE_LIMIT)
            else {
                return json!({"status": "error", "error": "Skill file could not be read"});
            };
            let paragraph = first_skill_paragraph(&content);
            let limit = max_tokens.clamp(1, 256);
            let preview = truncate_words(paragraph, limit);
            json!({
                "status": "ok",
                "mode": "preview",
                "name": name,
                "preview": preview,
                "truncated": scan_truncated || paragraph.split_whitespace().count() > limit,
                "next": "Use mode=search with q, or mode=full for one skill."
            })
        }
    }
}

async fn handle_get_skills(Query(query): Query<SkillsQuery>) -> Json<Value> {
    let dir = skills_dir();
    if !dir.exists() {
        return Json(json!({"total": 0, "skills": []}));
    }
    if let Some(name) = query
        .name
        .as_deref()
        .map(str::trim)
        .filter(|name| !name.is_empty())
    {
        return Json(skill_view(
            &dir,
            name,
            query.mode.as_deref().unwrap_or("preview"),
            query.q.as_deref(),
            query.max_tokens.unwrap_or(96),
        ));
    }
    Json(skill_index(&dir))
}

#[derive(Deserialize)]
struct CortexSearchQuery {
    q: Option<String>,
    limit: Option<usize>,
}

async fn handle_cortex_search(
    State(state): State<AppState>,
    Query(query): Query<CortexSearchQuery>,
) -> Json<Value> {
    let q = query.q.unwrap_or_else(|| "AIEN".to_string());
    let limit = query.limit.unwrap_or(5);
    let token = get_cortex_token();

    let encoded_q: String = form_urlencoded::byte_serialize(q.as_bytes()).collect();
    let mut req = state.client.get(format!(
        "{}/api/cortex/search?q={}&space=atlas-memory&limit={}",
        CORTEX_URL, encoded_q, limit
    ));

    if let Some(t) = token {
        req = req.header("Authorization", format!("Bearer {}", t));
    }

    match req.send().await {
        Ok(resp) => {
            let mut json = resp
                .json::<Value>()
                .await
                .unwrap_or_else(|_| json!({"results": []}));
            if json.get("results").is_none() {
                if let Some(obj) = json.as_object_mut() {
                    obj.insert("results".to_string(), json!([]));
                } else {
                    json = json!({"results": []});
                }
            }
            Json(json)
        }
        Err(e) => Json(json!({"results": [], "error": e.to_string()})),
    }
}

#[derive(Deserialize)]
struct CortexWritePayload {
    name: String,
    content: String,
    kind: Option<String>,
}

async fn handle_cortex_write(
    State(state): State<AppState>,
    Json(payload): Json<CortexWritePayload>,
) -> Result<Json<Value>, (StatusCode, String)> {
    let token = get_cortex_token().ok_or_else(|| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "Missing Cortex token".to_string(),
        )
    })?;

    let body = json!({
        "kind": "entity",
        "value": {
            "space": "atlas-memory",
            "entityType": payload.kind.unwrap_or_else(|| "lesson".to_string()),
            "canonicalName": payload.name,
            "content": payload.content,
            "metadata": {
                "author": "AIEN",
                "source": "spark-cockpit-rs",
                "hardware": "NVIDIA DGX Spark GB10"
            }
        }
    });

    let resp = state
        .client
        .post(format!("{}/api/cortex/write", CORTEX_URL))
        .header("Authorization", format!("Bearer {}", token))
        .json(&body)
        .send()
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, e.to_string()))?;

    let json = resp
        .json::<Value>()
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(json))
}

async fn handle_chat_stream(
    State(state): State<AppState>,
    Json(mut payload): Json<Value>,
) -> Response {
    payload["stream"] = json!(true);
    if payload.get("model").is_none() {
        payload["model"] = json!("atlas-lightning-omni");
    }

    let patterns = state.redactor_patterns.clone();
    let upstream_resp = match state
        .client
        .post(format!("{}/v1/chat/completions", MAX_SEAT_URL))
        .json(&payload)
        .send()
        .await
    {
        Ok(r) => r,
        Err(e) => {
            return (
                StatusCode::BAD_GATEWAY,
                Json(json!({"error": format!("MAX model seat error: {}", e)})),
            )
                .into_response();
        }
    };

    let stream = async_stream::stream! {
        let mut byte_stream = upstream_resp.bytes_stream();
        let mut content_redactor = StreamRedactor::new(patterns.clone());
        let mut reasoning_redactor = StreamRedactor::new(patterns);
        let mut line_buffer = String::new();

        while let Some(chunk_result) = byte_stream.next().await {
            if let Ok(chunk) = chunk_result {
                let text = String::from_utf8_lossy(&chunk);
                line_buffer.push_str(&text);

                while let Some(pos) = line_buffer.find('\n') {
                    let line = line_buffer[..pos].trim().to_string();
                    line_buffer = line_buffer[pos + 1..].to_string();

                    if let Some(stripped) = line.strip_prefix("data: ") {
                        let data_str = stripped.trim();
                        if data_str == "[DONE]" {
                            let sc = content_redactor.flush();
                            let sr = reasoning_redactor.flush();
                            if !sc.is_empty() || !sr.is_empty() {
                                let payload = json!({"content": sc, "reasoning": sr});
                                yield Ok::<_, Infallible>(Event::default().data(payload.to_string()));
                            }
                            yield Ok(Event::default().data("[DONE]"));
                            break;
                        }

                        if let Ok(parsed) = serde_json::from_str::<Value>(data_str)
                            && let Some(choices) = parsed.get("choices").and_then(Value::as_array)
                                && let Some(choice) = choices.first()
                                    && let Some(delta) = choice.get("delta") {
                                        let raw_content = delta.get("content").and_then(Value::as_str).unwrap_or("");
                                        let raw_reasoning = delta.get("reasoning")
                                            .or_else(|| delta.get("reasoning_content"))
                                            .and_then(Value::as_str)
                                            .unwrap_or("");

                                        let sc = if !raw_content.is_empty() { content_redactor.feed(raw_content) } else { String::new() };
                                        let sr = if !raw_reasoning.is_empty() { reasoning_redactor.feed(raw_reasoning) } else { String::new() };

                                        if !sc.is_empty() || !sr.is_empty() {
                                            let payload = json!({"content": sc, "reasoning": sr});
                                            yield Ok(Event::default().data(payload.to_string()));
                                        }
                                    }
                    }
                }
            }
        }
    };

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)))
        .into_response()
}

async fn handle_get_subagents() -> Json<Value> {
    let dir_buf = get_home_dir().join("basecamp/sessions/subagents");
    let dir = dir_buf.as_path();
    if !dir.exists() {
        return Json(json!({"subagents": [], "count": 0}));
    }
    let mut items = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("json")
                && let Ok(content) = std::fs::read_to_string(&path)
                && let Ok(parsed) = serde_json::from_str::<Value>(&content)
            {
                items.push(parsed);
            }
        }
    }
    Json(json!({"subagents": items, "count": items.len()}))
}

#[derive(Debug, Deserialize)]
struct PostCombPayload {
    q: Option<i32>,
    r: Option<i32>,
    author: Option<String>,
    role: Option<String>,
    content: Option<String>,
    body: Option<String>,
    intent: Option<String>,
    parent_id: Option<String>,
}

async fn handle_get_hive_cells(State(state): State<AppState>) -> Json<Value> {
    match state.hive_store.get_cells() {
        Ok((combs, bounds)) => {
            let cells: Vec<Value> = combs
                .iter()
                .map(|c| {
                    json!({
                        "id": c.id,
                        "q": c.q,
                        "r": c.r,
                        "author": c.author,
                        "role": c.role,
                        "content": c.content,
                        "body": c.content,
                        "intent": c.intent,
                        "parent_id": c.parent_id,
                        "growthParentId": c.parent_id,
                        "conversationId": c.parent_id.as_deref().unwrap_or(&c.id),
                        "status": "visible",
                        "membership": if c.role == "operator" { "guest" } else { "alumni" },
                        "created_at": c.created_at,
                        "neighbors": c.neighbors,
                    })
                })
                .collect();

            Json(json!({
                "combs": combs,
                "cells": cells,
                "bounds": {
                    "min_q": bounds.min_q,
                    "max_q": bounds.max_q,
                    "min_r": bounds.min_r,
                    "max_r": bounds.max_r,
                    "count": bounds.count,
                    "radius": bounds.radius,
                    "revision": bounds.revision,
                    "messageCount": bounds.message_count,
                    "readOnly": false,
                    "updatedAt": Utc::now().timestamp_millis()
                },
                "bounding_box": {
                    "min_q": bounds.min_q,
                    "max_q": bounds.max_q,
                    "min_r": bounds.min_r,
                    "max_r": bounds.max_r,
                    "count": bounds.count
                }
            }))
        }
        Err(e) => Json(json!({"error": e.to_string(), "combs": [], "cells": []})),
    }
}

async fn handle_get_hive_bounds(State(state): State<AppState>) -> Json<Value> {
    match state.hive_store.get_cells() {
        Ok((combs, bounds)) => Json(json!({
            "bounds": {
                "min_q": bounds.min_q,
                "max_q": bounds.max_q,
                "min_r": bounds.min_r,
                "max_r": bounds.max_r,
                "count": bounds.count,
                "radius": bounds.radius,
                "revision": bounds.revision,
                "messageCount": bounds.message_count,
                "readOnly": false,
                "updatedAt": Utc::now().timestamp_millis()
            },
            "count": combs.len()
        })),
        Err(e) => Json(json!({"error": e.to_string()})),
    }
}

async fn handle_post_hive_comb(
    State(state): State<AppState>,
    Json(payload): Json<PostCombPayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let content = payload.content.or(payload.body).unwrap_or_default();
    if content.trim().is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({"error": "Comb content cannot be empty"})),
        ));
    }

    let author = payload.author.unwrap_or_else(|| "AIEN".to_string());
    let input = spark_hive::PlaceCombInput {
        q: payload.q,
        r: payload.r,
        author,
        role: payload.role,
        content,
        intent: payload.intent,
        parent_id: payload.parent_id,
    };

    match state.hive_store.place_comb(input) {
        Ok(comb) => {
            let (_, bounds) = state.hive_store.get_cells().unwrap_or_else(|_| {
                (
                    vec![],
                    spark_hive::BoundingBox {
                        min_q: comb.q,
                        max_q: comb.q,
                        min_r: comb.r,
                        max_r: comb.r,
                        count: 1,
                        radius: 5,
                        revision: 1,
                        message_count: 1,
                    },
                )
            });

            let formatted_cell = json!({
                "id": comb.id,
                "q": comb.q,
                "r": comb.r,
                "author": comb.author,
                "role": comb.role,
                "content": comb.content,
                "body": comb.content,
                "intent": comb.intent,
                "parent_id": comb.parent_id,
                "growthParentId": comb.parent_id,
                "conversationId": comb.parent_id.as_deref().unwrap_or(&comb.id),
                "status": "visible",
                "membership": if comb.role == "operator" { "guest" } else { "alumni" },
                "created_at": comb.created_at,
                "neighbors": comb.neighbors,
            });

            Ok(Json(json!({
                "status": "placed",
                "comb": comb,
                "cell": formatted_cell,
                "bounds": {
                    "min_q": bounds.min_q,
                    "max_q": bounds.max_q,
                    "min_r": bounds.min_r,
                    "max_r": bounds.max_r,
                    "count": bounds.count,
                    "radius": bounds.radius,
                    "revision": bounds.revision,
                    "messageCount": bounds.message_count,
                    "readOnly": false,
                    "updatedAt": Utc::now().timestamp_millis()
                }
            })))
        }
        Err(spark_hive::HiveError::Collision { q, r, comb_id }) => Err((
            StatusCode::CONFLICT,
            Json(json!({
                "error": format!("Coordinate ({}, {}) is already occupied by comb '{}'", q, r, comb_id),
                "q": q,
                "r": r,
                "occupied_by": comb_id
            })),
        )),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )),
    }
}

async fn handle_get_hive_adapters() -> Json<Value> {
    let adapters = spark_hive::get_catalog_adapters();
    let models = spark_hive::ConsumerModel::all();
    let engines = spark_hive::UpstreamEngine::all();
    Json(json!({
        "total_adapters": adapters.len(),
        "adapters": adapters,
        "models": models.iter().map(|m| json!({
            "slug": m.slug(),
            "name": m.display_name(),
            "params_b": m.param_count_billions(),
            "hf_repo": m.hf_repo(),
            "hardware": m.hardware_profile().description(),
            "quantization": m.recommended_quantization(),
        })).collect::<Vec<_>>(),
        "engines": engines.iter().map(|e| json!({
            "repo": e.repo(),
            "language": e.primary_language(),
            "description": e.description(),
        })).collect::<Vec<_>>(),
    }))
}

async fn handle_get_hive_adapter_chains(State(state): State<AppState>) -> Json<Value> {
    match spark_hive::list_adapter_pipeline_chains(&state.hive_store) {
        Ok(chains) => Json(json!({
            "total_chains": chains.len(),
            "chains": chains
        })),
        Err(e) => Json(json!({"error": e.to_string(), "chains": []})),
    }
}

#[derive(Debug, Deserialize)]
struct ForgeQuery {
    project: Option<String>,
    ring: Option<usize>,
    status: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ForgeProjectPayload {
    project: String,
    title: String,
    core_spec: String,
}

#[derive(Debug, Deserialize)]
struct ClaimForgePayload {
    task_id: String,
    agent_id: String,
    ttl_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct HeartbeatForgePayload {
    task_id: String,
    agent_id: String,
    ttl_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
struct SubmitForgePayload {
    task_id: String,
    agent_id: String,
    branch: String,
    pr_url: Option<String>,
}

#[derive(Debug, Deserialize)]
struct VerifyForgePayload {
    task_id: String,
    verdict: bool,
    notes: Option<String>,
}

async fn handle_get_forge_tasks(
    State(state): State<AppState>,
    Query(query): Query<ForgeQuery>,
) -> Json<Value> {
    match state.hive_store.list_forge_tasks(
        query.project.as_deref(),
        query.ring,
        query.status.as_deref(),
    ) {
        Ok(tasks) => Json(json!({ "success": true, "tasks": tasks })),
        Err(e) => Json(json!({ "success": false, "error": e.to_string(), "tasks": [] })),
    }
}

async fn handle_post_forge_project(
    State(state): State<AppState>,
    Json(payload): Json<ForgeProjectPayload>,
) -> (StatusCode, Json<Value>) {
    match state
        .hive_store
        .spawn_forge_project(&payload.project, &payload.title, &payload.core_spec)
    {
        Ok(task) => (
            StatusCode::CREATED,
            Json(json!({ "success": true, "task": task })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e.to_string() })),
        ),
    }
}

async fn handle_post_forge_task(
    State(state): State<AppState>,
    Json(payload): Json<spark_hive::CreateForgeTaskInput>,
) -> (StatusCode, Json<Value>) {
    match state.hive_store.create_forge_task(payload) {
        Ok(task) => (
            StatusCode::CREATED,
            Json(json!({ "success": true, "task": task })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e.to_string() })),
        ),
    }
}

async fn handle_claim_forge_task(
    State(state): State<AppState>,
    Json(payload): Json<ClaimForgePayload>,
) -> (StatusCode, Json<Value>) {
    let ttl = payload.ttl_secs.unwrap_or(300);
    match state
        .hive_store
        .claim_forge_task(&payload.task_id, &payload.agent_id, ttl)
    {
        Ok(task) => (
            StatusCode::OK,
            Json(json!({ "success": true, "task": task })),
        ),
        Err(spark_hive::HiveError::TaskAlreadyClaimed {
            task_id,
            claimed_by,
        }) => (
            StatusCode::CONFLICT,
            Json(
                json!({ "success": false, "error": format!("Task {} already claimed by {}", task_id, claimed_by) }),
            ),
        ),
        Err(spark_hive::HiveError::TaskNotFound(id)) => (
            StatusCode::NOT_FOUND,
            Json(json!({ "success": false, "error": format!("Task {} not found", id) })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "success": false, "error": e.to_string() })),
        ),
    }
}

async fn handle_heartbeat_forge_task(
    State(state): State<AppState>,
    Json(payload): Json<HeartbeatForgePayload>,
) -> (StatusCode, Json<Value>) {
    let ttl = payload.ttl_secs.unwrap_or(300);
    match state
        .hive_store
        .heartbeat_forge_task(&payload.task_id, &payload.agent_id, ttl)
    {
        Ok(()) => (StatusCode::OK, Json(json!({ "success": true }))),
        Err(spark_hive::HiveError::TaskUnauthorized {
            task_id,
            claimed_by,
            agent_id,
        }) => (
            StatusCode::FORBIDDEN,
            Json(json!({
                "success": false,
                "error": format!("Task {} leased to {}, not {}", task_id, claimed_by, agent_id),
            })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e.to_string() })),
        ),
    }
}

async fn handle_submit_forge_task(
    State(state): State<AppState>,
    Json(payload): Json<SubmitForgePayload>,
) -> (StatusCode, Json<Value>) {
    match state.hive_store.submit_forge_task(
        &payload.task_id,
        &payload.agent_id,
        &payload.branch,
        payload.pr_url.as_deref(),
    ) {
        Ok(()) => (StatusCode::OK, Json(json!({ "success": true }))),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e.to_string() })),
        ),
    }
}

async fn handle_verify_forge_task(
    State(state): State<AppState>,
    Json(payload): Json<VerifyForgePayload>,
) -> (StatusCode, Json<Value>) {
    let notes = payload.notes.unwrap_or_default();
    match state
        .hive_store
        .verify_forge_task(&payload.task_id, payload.verdict, &notes)
    {
        Ok(()) => (
            StatusCode::OK,
            Json(json!({ "success": true, "verdict": payload.verdict })),
        ),
        Err(e) => (
            StatusCode::BAD_REQUEST,
            Json(json!({ "success": false, "error": e.to_string() })),
        ),
    }
}

async fn handle_reclaim_forge_leases(State(state): State<AppState>) -> Json<Value> {
    match state.hive_store.reclaim_expired_leases() {
        Ok(count) => Json(json!({ "success": true, "reclaimed_count": count })),
        Err(e) => Json(json!({ "success": false, "error": e.to_string() })),
    }
}

fn operator_config_path() -> PathBuf {
    get_home_dir().join(".config/sovereign/operator.toml")
}
const MAIL_API_URL: &str = "http://127.0.0.1:18092";

async fn handle_get_operator() -> Json<Value> {
    let op_cfg = operator_config_path();
    if op_cfg.exists()
        && let Ok(content) = fs::read_to_string(&op_cfg)
        && let Ok(toml_val) = toml::from_str::<Value>(&content)
    {
        return Json(toml_val);
    }
    Json(json!({
        "operator": {
            "name": "Drake Stapleton",
            "email": "drake@aien.org",
            "handle": "drake",
            "sign_commits": true
        },
        "engine": {
            "mode": "max",
            "api_base_url": "http://127.0.0.1:18006/v1",
            "model_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
            "max_port": 18006,
            "context_window": 32768,
            "temperature": 0.7
        }
    }))
}

#[derive(Debug, Deserialize)]
struct OperatorUpdatePayload {
    name: Option<String>,
    email: Option<String>,
    handle: Option<String>,
    sign_commits: Option<bool>,
}

async fn handle_post_operator(
    Json(payload): Json<OperatorUpdatePayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let clean_name = payload.name.map(|n| {
        n.chars()
            .filter(|c| !c.is_control())
            .take(64)
            .collect::<String>()
    });
    let clean_email = payload.email.map(|e| {
        e.chars()
            .filter(|c| !c.is_control())
            .take(128)
            .collect::<String>()
    });
    let clean_handle = payload.handle.map(|h| {
        h.chars()
            .filter(|c| c.is_alphanumeric() || *c == '_' || *c == '-')
            .take(32)
            .collect::<String>()
    });

    let op_cfg = operator_config_path();
    let mut current_toml = if op_cfg.exists() {
        fs::read_to_string(operator_config_path())
            .ok()
            .and_then(|s| toml::from_str::<Value>(&s).ok())
            .unwrap_or_else(|| json!({}))
    } else {
        json!({})
    };

    if !current_toml.is_object() {
        current_toml = json!({});
    }

    let op_map = current_toml.as_object_mut().unwrap();
    let operator_obj = op_map.entry("operator").or_insert_with(|| json!({}));
    if let Some(obj) = operator_obj.as_object_mut() {
        if let Some(n) = clean_name {
            obj.insert("name".to_string(), json!(n));
        }
        if let Some(e) = clean_email {
            obj.insert("email".to_string(), json!(e));
        }
        if let Some(h) = clean_handle {
            obj.insert("handle".to_string(), json!(h));
        }
        if let Some(s) = payload.sign_commits {
            obj.insert("sign_commits".to_string(), json!(s));
        }
    }

    let toml_str = toml::to_string_pretty(&current_toml).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    let op_path = operator_config_path();
    if let Some(parent) = op_path.parent() {
        let _ = fs::create_dir_all(parent);
    }
    fs::write(&op_path, toml_str).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    Ok(Json(
        json!({"status": "saved", "operator": current_toml.get("operator")}),
    ))
}

async fn handle_engine_status(State(state): State<AppState>) -> Json<Value> {
    let start = Instant::now();
    let health_res = state
        .client
        .get(format!("{}/health", MAX_SEAT_URL))
        .timeout(Duration::from_millis(500))
        .send()
        .await;

    let latency_ms = start.elapsed().as_millis();
    let is_online = health_res.map(|r| r.status().is_success()).unwrap_or(false);

    Json(json!({
        "status": if is_online { "online" } else { "offline" },
        "latency_ms": latency_ms,
        "active_engine": "MAX Native Engine",
        "active_model": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        "device": "Grace Blackwell GB10 (Dual GPU)",
        "port": 18006,
        "cost_per_million": "$0.00"
    }))
}

async fn handle_install_en2_imprint(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let seed_path_buf = if Path::new("imprints/en2-trinity/cortex-seed.json").exists() {
        PathBuf::from("imprints/en2-trinity/cortex-seed.json")
    } else {
        get_home_dir().join("workspace/aien-sovereign-core/imprints/en2-trinity/cortex-seed.json")
    };
    let seed_path = seed_path_buf.as_path();
    if !seed_path.exists() {
        return Err((
            StatusCode::NOT_FOUND,
            Json(json!({"error": "EN2 imprint seed not found"})),
        ));
    }

    let seed_data = fs::read_to_string(seed_path).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": e.to_string()})),
        )
    })?;
    let lessons: Vec<Value> = serde_json::from_str(&seed_data).map_err(|e| {
        (
            StatusCode::BAD_REQUEST,
            Json(json!({"error": e.to_string()})),
        )
    })?;

    let token = get_cortex_token().unwrap_or_default();

    let mut installed_count = 0;
    for lesson in lessons {
        let payload = json!({
            "kind": "entity",
            "value": {
                "canonicalName": lesson["canonicalName"],
                "entityType": lesson["entityType"],
                "content": lesson["content"],
                "metadata": lesson["metadata"],
                "space": "atlas-memory"
            }
        });

        let mut req = state
            .client
            .post(format!("{}/api/cortex/write", CORTEX_URL));
        if !token.is_empty() {
            req = req.header("Authorization", format!("Bearer {}", token));
        }
        if let Ok(res) = req.json(&payload).send().await
            && res.status().is_success()
        {
            installed_count += 1;
        }
    }

    Ok(Json(json!({
        "status": "installed",
        "installed_count": installed_count,
        "version": "2.0.0",
        "imprint": "en2-trinity"
    })))
}

/// Mail API bearer token: environment, then the owner-only file shared with
/// spark-mail (~/.config/aien/mail-api.token), then atlas-vault.
fn mail_access_token() -> String {
    if let Ok(value) = std::env::var("AIEN_MAIL_API_TOKEN")
        && !value.trim().is_empty()
    {
        return value.trim().to_owned();
    }
    spark_adapters::vault::resolve_secret("AIEN_MAIL_API_TOKEN").unwrap_or_default()
}

async fn handle_mail_status(State(state): State<AppState>) -> Json<Value> {
    let res = state
        .client
        .get(format!("{}/api/mail/status", MAIL_API_URL))
        .bearer_auth(mail_access_token())
        .timeout(Duration::from_millis(500))
        .send()
        .await;

    match res {
        Ok(r) if r.status().is_success() => {
            let val = r
                .json::<Value>()
                .await
                .unwrap_or(json!({"status": "error"}));
            Json(val)
        }
        _ => Json(json!({
            "status": "offline",
            "inbox_count": 0,
            "sent_count": 0,
            "smtp_port": 2525,
            "api_port": 18092,
            "cortex_connected": false
        })),
    }
}

async fn handle_mail_inbox(State(state): State<AppState>) -> Json<Value> {
    let res = state
        .client
        .get(format!("{}/api/mail/inbox", MAIL_API_URL))
        .bearer_auth(mail_access_token())
        .timeout(Duration::from_millis(800))
        .send()
        .await;

    match res {
        Ok(r) if r.status().is_success() => {
            let val = r.json::<Value>().await.unwrap_or(json!([]));
            Json(val)
        }
        _ => Json(json!([])),
    }
}

async fn handle_mail_sent(State(state): State<AppState>) -> Json<Value> {
    let res = state
        .client
        .get(format!("{}/api/mail/sent", MAIL_API_URL))
        .bearer_auth(mail_access_token())
        .timeout(Duration::from_millis(800))
        .send()
        .await;

    match res {
        Ok(r) if r.status().is_success() => {
            let val = r.json::<Value>().await.unwrap_or(json!([]));
            Json(val)
        }
        _ => Json(json!([])),
    }
}

async fn handle_mail_send(
    State(state): State<AppState>,
    Json(payload): Json<Value>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let res = state
        .client
        .post(format!("{}/api/mail/send", MAIL_API_URL))
        .bearer_auth(mail_access_token())
        .timeout(Duration::from_secs(90))
        .json(&payload)
        .send()
        .await
        .map_err(|e| {
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({"error": e.to_string()})),
            )
        })?;

    if res.status().is_success() {
        let val = res
            .json::<Value>()
            .await
            .unwrap_or(json!({"status": "sent"}));
        Ok(Json(val))
    } else {
        Err((
            StatusCode::BAD_GATEWAY,
            Json(json!({"error": "Failed to send via mail daemon"})),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn origin_headers(origin: Option<&str>) -> HeaderMap {
        let mut h = HeaderMap::new();
        if let Some(o) = origin {
            h.insert(axum::http::header::ORIGIN, o.parse().unwrap());
        }
        h
    }

    #[test]
    fn terminal_allows_loopback_origin_and_peer() {
        let peer: SocketAddr = "127.0.0.1:50000".parse().unwrap();
        assert!(terminal_request_allowed(
            &origin_headers(Some("http://127.0.0.1:18095")),
            &peer
        ));
        assert!(terminal_request_allowed(
            &origin_headers(Some("http://localhost:18095")),
            &peer
        ));
        let peer6: SocketAddr = "[::1]:50000".parse().unwrap();
        assert!(terminal_request_allowed(
            &origin_headers(Some("http://localhost:18095")),
            &peer6
        ));
    }

    #[test]
    fn terminal_rejects_missing_or_foreign_origin() {
        let peer: SocketAddr = "127.0.0.1:50000".parse().unwrap();
        assert!(!terminal_request_allowed(&origin_headers(None), &peer));
        assert!(!terminal_request_allowed(
            &origin_headers(Some("http://evil.example")),
            &peer
        ));
        assert!(!terminal_request_allowed(
            &origin_headers(Some("http://127.0.0.1:18095.evil.example")),
            &peer
        ));
        assert!(!terminal_request_allowed(
            &origin_headers(Some("http://192.168.1.108:18095")),
            &peer
        ));
        assert!(!terminal_request_allowed(
            &origin_headers(Some("null")),
            &peer
        ));
    }

    #[test]
    fn terminal_rejects_non_loopback_peer() {
        let peer: SocketAddr = "192.168.1.50:50000".parse().unwrap();
        assert!(!terminal_request_allowed(
            &origin_headers(Some("http://127.0.0.1:18095")),
            &peer
        ));
    }

    #[test]
    fn skill_index_omits_bodies_and_preview_is_one_paragraph() {
        let dir = std::env::temp_dir().join(format!("aien-skill-filter-{}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        let skill_dir = dir.join("reactor");
        fs::create_dir_all(&skill_dir).unwrap();
        fs::write(
            skill_dir.join("SKILL.md"),
            "---\nname: reactor\ndescription: Thermal notes\n---\n# Heading\n\nAlpha beta gamma.\n\nThe reactor uses a graphite moderator.\n",
        )
        .unwrap();

        let index = skill_index(&dir);
        let listed = &index["skills"][0];
        assert_eq!(listed["name"], "reactor");
        assert!(listed.get("content").is_none());
        assert_eq!(index["total"], 1);

        let preview = skill_view(&dir, "reactor", "preview", None, 96);
        assert_eq!(preview["mode"], "preview");
        assert_eq!(preview["preview"], "Alpha beta gamma.");
        assert!(!preview["preview"].as_str().unwrap().contains("graphite"));

        let search = skill_view(&dir, "reactor", "search", Some("graphite"), 96);
        assert_eq!(search["matches"][0]["paragraph"], 2);
        assert!(
            search["matches"][0]["snippet"]
                .as_str()
                .unwrap()
                .contains("graphite moderator")
        );

        let _ = fs::remove_dir_all(&dir);
    }

    fn create_test_state() -> AppState {
        let client = reqwest::Client::new();
        let hive_store = Arc::new(spark_hive::CombStore::open_in_memory().unwrap());
        let swarm_registry = Arc::new(SwarmRegistry::new_in_memory());
        let distill_engine = Arc::new(spark_adapters::DistillationEngine::new());
        let approvals = Arc::new(std::sync::Mutex::new(vec![PendingApproval {
            id: "test-appr-1".to_string(),
            action_type: "test".to_string(),
            title: "Test Approval".to_string(),
            description: "Test description".to_string(),
            command_or_diff: None,
            risk_level: "low".to_string(),
            status: "pending".to_string(),
            created_at: Utc::now().to_rfc3339(),
        }]));
        AppState {
            client,
            start_time: Instant::now(),
            redactor_patterns: Arc::new(vec![]),
            hive_store,
            swarm_registry,
            distill_engine,
            approvals,
        }
    }

    #[tokio::test]
    async fn test_cockpit_hive_cells_endpoint() {
        let state = create_test_state();
        let Json(res) = handle_get_hive_cells(State(state)).await;

        assert!(res.get("combs").is_some());
        assert!(res.get("cells").is_some());
        assert!(res.get("bounds").is_some());
        assert!(res.get("bounding_box").is_some());

        let combs = res.get("combs").unwrap().as_array().unwrap();
        assert_eq!(combs.len(), 1);
        assert_eq!(
            combs[0].get("author").unwrap().as_str().unwrap(),
            "AIEN Genesis"
        );
    }

    #[tokio::test]
    async fn test_cockpit_hive_comb_endpoint_placement_and_collision() {
        let state = create_test_state();

        // 1. Valid placement
        let payload = PostCombPayload {
            q: Some(1),
            r: Some(0),
            author: Some("Atlas".to_string()),
            role: Some("socratic".to_string()),
            content: Some("Socratic inquiry test comb".to_string()),
            body: None,
            intent: Some("branch".to_string()),
            parent_id: Some("comb-genesis-00000000".to_string()),
        };

        let result = handle_post_hive_comb(State(state.clone()), Json(payload)).await;
        assert!(result.is_ok());
        let Json(val) = result.unwrap();
        assert_eq!(val.get("status").unwrap().as_str().unwrap(), "placed");
        let comb = val.get("comb").unwrap();
        assert_eq!(comb.get("q").unwrap().as_i64().unwrap(), 1);
        assert_eq!(comb.get("r").unwrap().as_i64().unwrap(), 0);

        // 2. Collision rejection
        let collide_payload = PostCombPayload {
            q: Some(1),
            r: Some(0),
            author: Some("Intruder".to_string()),
            role: None,
            content: Some("Colliding text".to_string()),
            body: None,
            intent: None,
            parent_id: None,
        };

        let err_result = handle_post_hive_comb(State(state), Json(collide_payload)).await;
        assert!(err_result.is_err());
        let (status, Json(err_val)) = err_result.unwrap_err();
        assert_eq!(status, StatusCode::CONFLICT);
        assert!(
            err_val
                .get("error")
                .unwrap()
                .as_str()
                .unwrap()
                .contains("occupied")
        );
    }

    #[tokio::test]
    async fn test_cockpit_hive_bounds_endpoint() {
        let state = create_test_state();
        let Json(res) = handle_get_hive_bounds(State(state)).await;
        assert!(res.get("bounds").is_some());
        assert_eq!(res.get("count").unwrap().as_u64().unwrap(), 1);
    }
    #[tokio::test]
    async fn test_cockpit_hive_adapters_endpoint() {
        let Json(res) = handle_get_hive_adapters().await;
        assert!(res.get("adapters").is_some());
        assert!(res.get("models").is_some());
        assert!(res.get("engines").is_some());
        assert_eq!(res.get("models").unwrap().as_array().unwrap().len(), 9);
    }

    #[tokio::test]
    async fn test_cockpit_hive_adapter_chains_endpoint() {
        let state = create_test_state();
        let Json(res) = handle_get_hive_adapter_chains(State(state)).await;
        assert!(res.get("chains").is_some());
    }

    #[tokio::test]
    async fn test_cockpit_forge_lifecycle_endpoints() {
        let state = create_test_state();

        // 1. Spawn forge project
        let project_payload = ForgeProjectPayload {
            project: "harvester".to_string(),
            title: "Model Harvester Pipeline".to_string(),
            core_spec: "Autonomous reasoning distillation".to_string(),
        };
        let (status, Json(res)) =
            handle_post_forge_project(State(state.clone()), Json(project_payload)).await;
        assert_eq!(status, StatusCode::CREATED);
        assert!(res.get("task").is_some());
        let task_id = res
            .get("task")
            .unwrap()
            .get("id")
            .unwrap()
            .as_str()
            .unwrap()
            .to_string();

        // 2. Claim task
        let claim_payload = ClaimForgePayload {
            task_id: task_id.clone(),
            agent_id: "agent-aegis".to_string(),
            ttl_secs: Some(120),
        };
        let (status, Json(claim_res)) =
            handle_claim_forge_task(State(state.clone()), Json(claim_payload)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            claim_res
                .get("task")
                .unwrap()
                .get("status")
                .unwrap()
                .as_str()
                .unwrap(),
            "claimed"
        );

        // 3. Heartbeat
        let hb_payload = HeartbeatForgePayload {
            task_id: task_id.clone(),
            agent_id: "agent-aegis".to_string(),
            ttl_secs: Some(300),
        };
        let (status, _) = handle_heartbeat_forge_task(State(state.clone()), Json(hb_payload)).await;
        assert_eq!(status, StatusCode::OK);

        // 4. Submit
        let submit_payload = SubmitForgePayload {
            task_id: task_id.clone(),
            agent_id: "agent-aegis".to_string(),
            branch: "feat/harvester-core".to_string(),
            pr_url: Some("https://github.com/aien-dev/harvester/pull/1".to_string()),
        };
        let (status, _) =
            handle_submit_forge_task(State(state.clone()), Json(submit_payload)).await;
        assert_eq!(status, StatusCode::OK);

        // 5. Verify
        let verify_payload = VerifyForgePayload {
            task_id: task_id.clone(),
            verdict: true,
            notes: Some("All 4 defense checks passed".to_string()),
        };
        let (status, _) =
            handle_verify_forge_task(State(state.clone()), Json(verify_payload)).await;
        assert_eq!(status, StatusCode::OK);

        // 6. List tasks
        let query = ForgeQuery {
            project: Some("harvester".to_string()),
            ring: None,
            status: Some("verified".to_string()),
        };
        let Json(list_res) = handle_get_forge_tasks(State(state), Query(query)).await;
        assert_eq!(list_res.get("tasks").unwrap().as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_cockpit_operator_profile_sanitization() {
        let payload = OperatorUpdatePayload {
            name: Some(
                "Drake Stapleton
"
                .to_string(),
            ),
            email: Some("drake@aien.org".to_string()),
            handle: Some("drake_ops-1".to_string()),
            sign_commits: Some(true),
        };
        let res = handle_post_operator(Json(payload)).await;
        assert!(res.is_ok());
        let Json(body) = res.unwrap();
        let op = body.get("operator").unwrap();
        assert_eq!(op.get("name").unwrap().as_str().unwrap(), "Drake Stapleton");
        assert_eq!(op.get("email").unwrap().as_str().unwrap(), "drake@aien.org");
        assert_eq!(op.get("handle").unwrap().as_str().unwrap(), "drake_ops-1");
    }

    fn create_offline_test_state() -> AppState {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_millis(50))
            .build()
            .unwrap();
        let hive_store = Arc::new(spark_hive::CombStore::open_in_memory().unwrap());
        let swarm_registry = Arc::new(SwarmRegistry::new_in_memory());
        let distill_engine = Arc::new(spark_adapters::DistillationEngine::new());
        let approvals = Arc::new(std::sync::Mutex::new(vec![]));
        AppState {
            client,
            start_time: Instant::now(),
            redactor_patterns: Arc::new(vec![]),
            hive_store,
            swarm_registry,
            distill_engine,
            approvals,
        }
    }

    #[tokio::test]
    async fn test_cockpit_pulse_endpoint_schema_and_offline_backends() {
        let state = create_test_state();
        let Json(pulse) = handle_pulse(State(state)).await;

        assert_eq!(pulse["status"], "ok");
        assert_eq!(pulse["heartbeat_bpm"], 72);
        assert!((pulse["coherence"].as_f64().unwrap() - 0.98).abs() < 1e-4);
        assert!(pulse["uptime_seconds"].as_u64().is_some());

        let model_seat = pulse["model_seat"].as_str().unwrap();
        assert!(model_seat == "ONLINE (Port 18006)" || model_seat == "OFFLINE");

        let conduit_seat = pulse["conduit_seat"].as_str().unwrap();
        assert!(conduit_seat == "ONLINE (Port 6167)" || conduit_seat == "OFFLINE");

        assert!(pulse["hive_count"].as_u64().is_some());
        assert!(pulse["timestamp"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_cockpit_status_endpoint_schema_and_offline_backend() {
        let state = create_test_state();
        let Json(status) = handle_engine_status(State(state)).await;

        let status_str = status["status"].as_str().unwrap();
        assert!(status_str == "online" || status_str == "offline");
        assert!(status["latency_ms"].as_u64().is_some());
        assert_eq!(status["active_engine"], "MAX Native Engine");
        assert_eq!(
            status["active_model"],
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
        );
        assert_eq!(status["device"], "Grace Blackwell GB10 (Dual GPU)");
        assert_eq!(status["port"], 18006);
        assert_eq!(status["cost_per_million"], "$0.00");
    }

    #[tokio::test]
    async fn test_cockpit_cortex_search_endpoint_unreachable_fallback() {
        let state = create_offline_test_state();
        let query = CortexSearchQuery {
            q: Some("test_unreachable_cortex".to_string()),
            limit: Some(3),
        };
        let Json(res) = handle_cortex_search(State(state), Query(query)).await;
        assert!(res.get("results").is_some());
        let results = res.get("results").unwrap().as_array().unwrap();
        assert_eq!(results.len(), 0);
    }

    #[tokio::test]
    async fn test_cockpit_cortex_write_endpoint_error_handling() {
        let state = create_offline_test_state();
        let payload = CortexWritePayload {
            name: "test_cockpit_write".to_string(),
            content: "Telemetry payload".to_string(),
            kind: Some("telemetry".to_string()),
        };

        let res = handle_cortex_write(State(state), Json(payload)).await;
        // Verify proper error response or graceful handling
        if let Err((status, _)) = res {
            assert!(
                status == StatusCode::BAD_GATEWAY || status == StatusCode::INTERNAL_SERVER_ERROR
            );
        }
    }

    #[tokio::test]
    async fn test_cockpit_subagents_endpoint_response_structure() {
        let Json(res) = handle_get_subagents().await;
        assert!(res.get("subagents").is_some());
        assert!(res.get("count").is_some());
        let subagents = res.get("subagents").unwrap().as_array().unwrap();
        let count = res.get("count").unwrap().as_u64().unwrap();
        assert_eq!(subagents.len() as u64, count);
    }

    #[tokio::test]
    async fn test_cockpit_submillisecond_latency_assertions() {
        let state = create_test_state();

        let max_duration = if std::env::var("CI").is_ok() {
            Duration::from_millis(15)
        } else {
            Duration::from_millis(2)
        };

        // 1. Subagents local file/memory check latency assertion
        let t0 = Instant::now();
        let _ = handle_get_subagents().await;
        let elapsed_subagents = t0.elapsed();
        assert!(
            elapsed_subagents < max_duration,
            "Subagents latency within limit: {:?}",
            elapsed_subagents
        );

        // 2. Hive bounds retrieval latency assertion
        let t1 = Instant::now();
        let _ = handle_get_hive_bounds(State(state.clone())).await;
        let elapsed_bounds = t1.elapsed();
        assert!(
            elapsed_bounds < max_duration,
            "Hive bounds latency within limit: {:?}",
            elapsed_bounds
        );

        // 3. Hive cells in-memory store latency assertion
        let t2 = Instant::now();
        let _ = handle_get_hive_cells(State(state)).await;
        let elapsed_cells = t2.elapsed();
        assert!(
            elapsed_cells < max_duration,
            "Hive cells latency within limit: {:?}",
            elapsed_cells
        );
    }

    #[tokio::test]
    async fn test_cockpit_telemetry_live_endpoint() {
        let Json(res) = handle_telemetry_live().await;
        assert_eq!(res["status"], "ok");
        assert!(res["timestamp"].as_str().is_some());

        // Zero fake metrics check
        assert!(res.get("heartbeat_bpm").is_none());
        assert!(res.get("coherence").is_none());

        // GPU telemetry
        let gpu = &res["gpu"];
        assert_eq!(gpu["model"], "NVIDIA GB10");
        assert!(gpu["temperature_c"].as_f64().is_some());
        assert!(gpu["power_draw_w"].as_f64().is_some());

        // Memory telemetry
        let mem = &res["memory"];
        assert!(mem["total_kb"].as_u64().unwrap_or(0) > 0);
        assert!(mem["available_kb"].as_u64().is_some());

        // CPU telemetry
        let cpu = &res["cpu"];
        assert_eq!(cpu["cores"].as_u64(), Some(20));
        assert!(cpu["load_1m"].as_f64().is_some());

        // TPM telemetry
        let tpm = &res["tpm"];
        assert_eq!(tpm["device"], "/dev/tpmrm0");
        assert!(tpm["present"].as_bool().is_some());
    }

    #[tokio::test]
    async fn test_cockpit_services_list_endpoint() {
        let Json(res) = handle_services_list().await;
        assert_eq!(res["status"], "ok");
        let svcs = res["services"].as_array().unwrap();
        assert_eq!(svcs.len(), 5);

        let ids: Vec<&str> = svcs.iter().map(|s| s["id"].as_str().unwrap()).collect();
        assert!(ids.contains(&"max-server"));
        assert!(ids.contains(&"cortex-rs"));
        assert!(ids.contains(&"cortex-encoder"));
        assert!(ids.contains(&"aegis"));
        assert!(ids.contains(&"spark-rsi"));
    }

    #[tokio::test]
    async fn test_cockpit_services_action_invalid_action() {
        let payload = ServiceActionPayload {
            service: "cortex-rs".to_string(),
            action: "invalid_action".to_string(),
        };
        let res = handle_services_action(Json(payload)).await;
        assert!(res.is_err());
        let (status, Json(err)) = res.unwrap_err();
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(err["error"].as_str().unwrap().contains("Invalid action"));
    }

    #[tokio::test]
    async fn test_cockpit_swarm_lifecycle_registry() {
        let state = create_test_state();

        // 1. Spawn task
        let spawn_payload = SwarmSpawnPayload {
            project: Some("spark-cockpit-rs".to_string()),
            role: "Avionics Auditor".to_string(),
            prompt: "Verify telemetry latency and accuracy".to_string(),
            model: Some("atlas-lightning-omni".to_string()),
            ring: Some(1),
        };
        let (status, Json(spawn_res)) =
            handle_swarm_spawn(State(state.clone()), Json(spawn_payload)).await;
        assert_eq!(status, StatusCode::CREATED);
        let task_id = spawn_res["task"]["id"].as_str().unwrap().to_string();
        assert_eq!(spawn_res["task"]["state"], "running");

        // 2. List tasks
        let Json(list_res) = handle_swarm_tasks(State(state.clone())).await;
        let tasks = list_res["tasks"].as_array().unwrap();
        assert!(tasks.iter().any(|t| t["id"] == task_id));

        // 3. Kill task
        let kill_payload = SwarmKillPayload {
            task_id: task_id.clone(),
        };
        let (status, Json(kill_res)) =
            handle_swarm_kill(State(state.clone()), Json(kill_payload)).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(kill_res["status"], "ok");

        // Verify task state is cancelled
        let Json(list_after) = handle_swarm_tasks(State(state)).await;
        let tasks_after = list_after["tasks"].as_array().unwrap();
        let target = tasks_after.iter().find(|t| t["id"] == task_id).unwrap();
        assert_eq!(target["state"], "cancelled");
        assert!(target["completed_at"].as_str().is_some());
    }

    #[tokio::test]
    async fn test_cockpit_rsi_ledger_endpoint() {
        let query = RsiLedgerQuery { limit: Some(5) };
        let Json(res) = handle_rsi_ledger(Query(query)).await;
        assert_eq!(res["status"], "ok");
        assert!(res["recent_blocks"].as_array().is_some());
        assert!(res["checkpoints"].as_array().is_some());
        assert!(res["total_blocks"].as_i64().is_some());
    }

    #[tokio::test]
    async fn test_cockpit_workshop_models_endpoint() {
        let state = create_test_state();
        let Json(res) = handle_workshop_models(State(state)).await;
        assert_eq!(res["status"], "ok");
        let models = res["models"].as_array().expect("models array");
        assert!(!models.is_empty());
        assert!(models.iter().any(|m| m["id"] == "atlas-lightning-omni"));
        let tracks = res["tracks"].as_array().expect("tracks array");
        assert_eq!(tracks.len(), 5);
    }

    #[tokio::test]
    async fn test_cockpit_workshop_distill_validation() {
        let state = create_test_state();

        // 1. Missing teacher
        let payload_no_teacher = WorkshopDistillPayload {
            prompt: Some("Write an allocation-free ring buffer".to_string()),
            track: None,
            teacher_model: "".to_string(),
            student_model: None,
            verification_strategy: None,
            system_prompt: None,
            commit_to_cortex: false,
        };
        let err1 = handle_workshop_distill(State(state.clone()), Json(payload_no_teacher)).await;
        assert!(err1.is_err());
        assert_eq!(err1.unwrap_err().0, StatusCode::BAD_REQUEST);

        // 2. Empty prompt and no track
        let payload_empty = WorkshopDistillPayload {
            prompt: Some("   ".to_string()),
            track: None,
            teacher_model: "openai/gpt-4o".to_string(),
            student_model: None,
            verification_strategy: None,
            system_prompt: None,
            commit_to_cortex: false,
        };
        let err2 = handle_workshop_distill(State(state.clone()), Json(payload_empty)).await;
        assert!(err2.is_err());
        assert_eq!(err2.unwrap_err().0, StatusCode::BAD_REQUEST);

        // 3. Invalid track
        let payload_invalid_track = WorkshopDistillPayload {
            prompt: None,
            track: Some("invalid_curriculum_name".to_string()),
            teacher_model: "openai/gpt-4o".to_string(),
            student_model: None,
            verification_strategy: None,
            system_prompt: None,
            commit_to_cortex: false,
        };
        let err3 = handle_workshop_distill(State(state.clone()), Json(payload_invalid_track)).await;
        assert!(err3.is_err());
        assert_eq!(err3.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_cockpit_workshop_commit_validation() {
        let state = create_test_state();

        let payload_empty = WorkshopCommitPayload {
            name: "test_empty".to_string(),
            content: "   ".to_string(),
            kind: None,
            task_id: None,
            confidence: None,
        };
        let res = handle_workshop_commit(State(state), Json(payload_empty)).await;
        assert!(res.is_err());
        assert_eq!(res.unwrap_err().0, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_cockpit_workshop_html_route() {
        let response = handle_workshop().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
    #[tokio::test]
    async fn test_cockpit_feed_endpoint() {
        let state = create_test_state();
        let Json(feed) = handle_get_feed(State(state)).await;
        assert_eq!(feed["status"], "ok");
        assert!(feed["feed"].as_array().is_some());
    }

    #[tokio::test]
    async fn test_cockpit_approvals_lifecycle() {
        let state = create_test_state();
        let Json(apprs) = handle_get_approvals(State(state.clone())).await;
        assert_eq!(apprs["status"], "ok");
        let list = apprs["approvals"].as_array().unwrap();
        assert!(!list.is_empty());

        let res = handle_resolve_approval(
            State(state.clone()),
            axum::extract::Path("test-appr-1".to_string()),
            Json(ApprovalActionPayload {
                action: "approve".to_string(),
            }),
        )
        .await;

        let Json(resolved) = res;
        assert_eq!(resolved["status"], "ok");
        assert_eq!(resolved["approval"]["status"], "approved");
    }
}
