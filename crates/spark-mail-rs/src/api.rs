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

use crate::cortex_sync::CortexSync;
use crate::models::{EmailMessage, MailboxStatus, SendEmailRequest};
use crate::relay::MailRelay;
use crate::store::MailStore;
use axum::{
    Router,
    extract::{Path as AxumPath, State},
    http::StatusCode,
    response::Json,
    routing::{get, post},
};
use axum::{
    extract::Request,
    middleware::{self, Next},
    response::{IntoResponse, Response},
};
use serde_json::{Value, json};
use std::sync::Arc;

#[derive(Clone)]
pub struct ApiState {
    pub store: Arc<MailStore>,
    pub cortex: Arc<CortexSync>,
    pub relay: Arc<MailRelay>,
    pub smtp_port: u16,
    pub api_port: u16,
    pub operator_email: String,
}

pub fn create_router(state: ApiState) -> Router {
    Router::new()
        .route("/api/mail/status", get(handle_status))
        .route("/api/mail/inbox", get(handle_list_inbox))
        .route("/api/mail/sent", get(handle_list_sent))
        .route("/api/mail/archive", get(handle_list_archive))
        .route("/api/mail/message/{id}", get(handle_get_message))
        .route("/api/mail/send", post(handle_send_mail))
        .route("/api/mail/ingest-test", post(handle_ingest_test))
        .layer(middleware::from_fn(require_mail_access))
        .with_state(state)
        .merge(Router::new().route("/health", get(handle_health)))
}

async fn require_mail_access(request: Request, next: Next) -> Response {
    // Browser access goes through the authenticated cockpit, never directly to mail.
    if request.headers().contains_key(axum::http::header::ORIGIN) {
        return StatusCode::FORBIDDEN.into_response();
    }
    let token = tokio::task::spawn_blocking(resolve_mail_token)
        .await
        .unwrap_or_default();
    if let Err(status) = validate_mail_token(
        token.trim(),
        request
            .headers()
            .get(axum::http::header::AUTHORIZATION)
            .and_then(|v| v.to_str().ok()),
    ) {
        return status.into_response();
    }
    let mut response = next.run(request).await;
    response.headers_mut().insert(
        axum::http::header::CACHE_CONTROL,
        "no-store".parse().unwrap(),
    );
    response
}

/// Resolve through process memory or the TPM-bound vault; never a plaintext file.
/// A vault hit is cached for 60 seconds so each request does not pay a TPM
/// unseal (about 2 s); failures are not cached, so provisioning takes effect
/// on the next request.
fn resolve_mail_token() -> String {
    use std::sync::Mutex;
    use std::time::{Duration, Instant};
    static CACHE: Mutex<Option<(Instant, String)>> = Mutex::new(None);

    if let Ok(value) = std::env::var("AIEN_MAIL_API_TOKEN")
        && !value.trim().is_empty()
    {
        return value;
    }
    if let Ok(guard) = CACHE.lock()
        && let Some((at, value)) = guard.as_ref()
        && at.elapsed() < Duration::from_secs(60)
    {
        return value.clone();
    }
    let value = vault_get("AIEN_MAIL_API_TOKEN");
    if !value.trim().is_empty()
        && let Ok(mut guard) = CACHE.lock()
    {
        *guard = Some((Instant::now(), value.clone()));
    }
    value
}

/// Runs `atlas-vault get`. The mail unit's PATH lacks ~/.local/bin, so fall
/// back to the operator's install location when PATH lookup finds nothing.
fn vault_get(name: &str) -> String {
    let run = |program: &std::ffi::OsStr| {
        std::process::Command::new(program)
            .args(["get", name])
            .stdin(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .output()
    };
    let output = match run(std::ffi::OsStr::new("atlas-vault")) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
            let home = std::env::var_os("HOME").unwrap_or_default();
            run(std::path::Path::new(&home)
                .join(".local/bin/atlas-vault")
                .as_os_str())
        }
        other => other,
    };
    match output {
        Ok(out) if out.status.success() => String::from_utf8(out.stdout).unwrap_or_default(),
        _ => String::new(),
    }
}

fn validate_mail_token(expected: &str, authorization: Option<&str>) -> Result<(), StatusCode> {
    if expected.len() < 32 {
        return Err(StatusCode::SERVICE_UNAVAILABLE);
    }
    let supplied = authorization.and_then(|v| v.strip_prefix("Bearer "));
    let valid = supplied.is_some_and(|t| {
        t.len() == expected.len()
            && t.bytes()
                .zip(expected.bytes())
                .fold(0u8, |d, (a, b)| d | (a ^ b))
                == 0
    });
    if valid {
        Ok(())
    } else {
        Err(StatusCode::UNAUTHORIZED)
    }
}

async fn handle_health() -> Json<Value> {
    Json(json!({
        "status": "ok",
        "service": "spark-mail-rs",
        "version": "0.1.0"
    }))
}

async fn handle_status(State(state): State<ApiState>) -> Json<MailboxStatus> {
    let cortex_ok = state.cortex.check_health().await;
    let bridge_ok = state.relay.check_bridge().await;

    let status = state.store.get_status(
        &state.operator_email,
        state.smtp_port,
        state.api_port,
        cortex_ok,
        bridge_ok,
    );
    Json(status)
}

async fn handle_list_inbox(State(state): State<ApiState>) -> Json<Vec<EmailMessage>> {
    let messages = state.store.list_folder("inbox");
    Json(messages)
}

async fn handle_list_sent(State(state): State<ApiState>) -> Json<Vec<EmailMessage>> {
    let messages = state.store.list_folder("sent");
    Json(messages)
}

async fn handle_list_archive(State(state): State<ApiState>) -> Json<Vec<EmailMessage>> {
    let messages = state.store.list_folder("archive");
    Json(messages)
}

async fn handle_get_message(
    State(state): State<ApiState>,
    AxumPath(id): AxumPath<String>,
) -> Result<Json<EmailMessage>, StatusCode> {
    state
        .store
        .get_message(&id)
        .map(Json)
        .ok_or(StatusCode::NOT_FOUND)
}

async fn handle_send_mail(
    State(state): State<ApiState>,
    Json(req): Json<SendEmailRequest>,
) -> Result<Json<EmailMessage>, (StatusCode, String)> {
    state
        .relay
        .send_email(req)
        .await
        .map(Json)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))
}

async fn handle_ingest_test(
    State(state): State<ApiState>,
) -> Result<Json<EmailMessage>, (StatusCode, String)> {
    let test_req = SendEmailRequest {
        to: vec![state.operator_email.clone()],
        subject: "Sovereign Mail Node Online".to_string(),
        body: "Your sovereign mail server is running directly on Spark hardware. Messages are stored on local NVMe disk with zero cloud telemetry and synced to Atlas Cortex memory.".to_string(),
        from: Some("atlas@sovereign.spark".to_string()),
    };

    let id = uuid::Uuid::new_v4().to_string();
    let now = chrono::Utc::now().to_rfc3339();
    let mut headers = std::collections::HashMap::new();
    headers.insert("From".to_string(), "atlas@sovereign.spark".to_string());
    headers.insert("To".to_string(), state.operator_email.clone());
    headers.insert("Subject".to_string(), test_req.subject.clone());

    let mut msg = EmailMessage {
        id: id.clone(),
        from: "atlas@sovereign.spark".to_string(),
        to: test_req.to,
        subject: test_req.subject,
        body: test_req.body,
        headers,
        received_at: now,
        folder: "inbox".to_string(),
        cortex_indexed: false,
        untrusted: false,
    };

    state
        .store
        .save_message(&msg)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e))?;

    if state.cortex.commit_mail(&msg).await.is_ok() {
        msg.cortex_indexed = true;
        let _ = state.store.save_message(&msg);
    }

    Ok(Json(msg))
}

#[cfg(test)]
mod access_tests {
    use super::*;
    #[test]
    fn mail_access_fails_closed() {
        let token = "a".repeat(64);
        assert_eq!(
            validate_mail_token("", None),
            Err(StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(
            validate_mail_token(&token, None),
            Err(StatusCode::UNAUTHORIZED)
        );
        assert_eq!(
            validate_mail_token(&token, Some("Bearer wrong")),
            Err(StatusCode::UNAUTHORIZED)
        );
        assert_eq!(
            validate_mail_token(&token, Some(&format!("Bearer {token}"))),
            Ok(())
        );
    }
}
