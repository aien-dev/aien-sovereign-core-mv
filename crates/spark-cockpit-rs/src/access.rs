//! Operator access boundary. Browser sessions never put credentials in URLs.
use axum::{
    extract::{Form, Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::{Html, IntoResponse, Response},
};
use serde::Deserialize;
use std::{
    collections::HashMap,
    net::SocketAddr,
    sync::{Arc, Mutex},
    time::{Duration, Instant},
};

const COOKIE: &str = "aien_cockpit_session";
const LOGIN: &str = r#"<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>AIEN sign in</title><h1>Sign in to AIEN</h1><form method="post" action="/auth/session"><label>Operator access token <input type="password" name="token" autocomplete="off" required></label><button>Sign in</button></form></html>"#;

pub struct Access {
    token: String,
    origins: Vec<String>,
    sessions: Mutex<HashMap<String, Instant>>,
}

impl Access {
    pub fn new(token: String, origins: Vec<String>) -> Result<Self, String> {
        if token.len() < 32
            || token.len() > 256
            || !token
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || b"-_+=/.".contains(&b))
        {
            return Err("Cockpit token must contain 32 to 256 cookie-safe ASCII characters".into());
        }
        if origins.is_empty()
            || origins.iter().any(|o| {
                o.parse::<axum::http::Uri>().map_or(true, |uri| {
                    uri.authority().is_none() || uri.path() != "/" || uri.query().is_some()
                }) || !(o.starts_with("http://") || o.starts_with("https://"))
                    || o.ends_with('/')
            })
        {
            return Err(
                "Cockpit origins must be explicit HTTP(S) origins without a trailing slash".into(),
            );
        }
        Ok(Self {
            token,
            origins,
            sessions: Mutex::new(HashMap::new()),
        })
    }

    pub fn from_environment() -> Result<Self, String> {
        let from_env = std::env::var("AIEN_COCKPIT_TOKEN")
            .ok()
            .filter(|t| !t.trim().is_empty());
        let token = from_env
            .or_else(|| {
                let result = std::process::Command::new("atlas-vault")
                    .args(["get", "AIEN_COCKPIT_TOKEN"])
                    .output()
                    .ok()?;
                result
                    .status
                    .success()
                    .then(|| String::from_utf8(result.stdout).ok())
                    .flatten()
            })
            .ok_or("AIEN_COCKPIT_TOKEN is unavailable from memory or atlas-vault")?;
        let origins = std::env::var("AIEN_COCKPIT_ORIGINS")
            .unwrap_or_else(|_| "http://127.0.0.1:18095,http://localhost:18095".into());
        Self::new(
            token.trim().to_owned(),
            origins.split(',').map(|o| o.trim().to_owned()).collect(),
        )
    }

    fn matches(&self, supplied: &str) -> bool {
        if supplied.len() != self.token.len() {
            return false;
        }
        self.token
            .bytes()
            .zip(supplied.bytes())
            .fold(0u8, |diff, (a, b)| diff | (a ^ b))
            == 0
    }

    fn create_session(&self) -> Option<String> {
        let mut sessions = self.sessions.lock().ok()?;
        sessions.retain(|_, expires| *expires > Instant::now());
        if sessions.len() >= 1024 {
            return None;
        }
        let id = format!(
            "{}{}",
            uuid::Uuid::new_v4().simple(),
            uuid::Uuid::new_v4().simple()
        );
        sessions.insert(id.clone(), Instant::now() + Duration::from_secs(3600));
        Some(id)
    }

    fn session_valid(&self, id: &str) -> bool {
        self.sessions.lock().ok().is_some_and(|sessions| {
            sessions
                .get(id)
                .is_some_and(|expires| *expires > Instant::now())
        })
    }

    fn origin_allowed(&self, headers: &HeaderMap) -> bool {
        headers
            .get(header::ORIGIN)
            .and_then(|v| v.to_str().ok())
            .is_some_and(|origin| self.origins.iter().any(|o| o == origin))
    }
}

/// Read-only health routes that local monitors (aien-cli, spark-debugger,
/// spark-supervisor, bench_inference_stack) poll without credentials. GET/HEAD
/// only, and only when the Host header names the loopback listener, so a
/// DNS-rebinding page cannot read them through a foreign hostname.
const PUBLIC_READ_ROUTES: &[&str] = &["/api/pulse", "/api/status"];

fn loopback_host(headers: &HeaderMap) -> bool {
    headers
        .get(header::HOST)
        .and_then(|v| v.to_str().ok())
        .and_then(|host| host.parse::<axum::http::uri::Authority>().ok())
        .is_some_and(|authority| matches!(authority.host(), "127.0.0.1" | "localhost" | "[::1]"))
}

pub fn bind_address() -> Result<SocketAddr, String> {
    std::env::var("AIEN_COCKPIT_ADDR")
        .unwrap_or_else(|_| "127.0.0.1:18095".into())
        .parse()
        .map_err(|_| "AIEN_COCKPIT_ADDR must be a socket address".into())
}

pub async fn login() -> Response {
    let mut response = Html(LOGIN).into_response();
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response.headers_mut().insert(
        header::CONTENT_SECURITY_POLICY,
        "default-src 'none'; form-action 'self'; frame-ancestors 'none'"
            .parse()
            .unwrap(),
    );
    response
}

#[derive(Deserialize)]
pub struct LoginForm {
    token: String,
}

pub async fn session(
    State(access): State<Arc<Access>>,
    headers: HeaderMap,
    Form(form): Form<LoginForm>,
) -> Response {
    if !access.origin_allowed(&headers) {
        return StatusCode::FORBIDDEN.into_response();
    }
    if !access.matches(&form.token) {
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let secure = headers
        .get(header::ORIGIN)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.starts_with("https://"));
    let Some(id) = access.create_session() else {
        return StatusCode::SERVICE_UNAVAILABLE.into_response();
    };
    let cookie = format!(
        "{COOKIE}={}; Path=/; HttpOnly; SameSite=Strict; Max-Age=3600{}",
        id,
        if secure { "; Secure" } else { "" }
    );
    let mut response = StatusCode::SEE_OTHER.into_response();
    response
        .headers_mut()
        .insert(header::LOCATION, "/".parse().unwrap());
    response
        .headers_mut()
        .insert(header::SET_COOKIE, cookie.parse().unwrap());
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}

pub async fn require_access(
    State(access): State<Arc<Access>>,
    request: Request,
    next: Next,
) -> Response {
    let headers = request.headers();
    let origin_present = headers.contains_key(header::ORIGIN);
    let websocket = headers
        .get(header::UPGRADE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|v| v.eq_ignore_ascii_case("websocket"));
    if (origin_present && !access.origin_allowed(headers))
        || (websocket && !access.origin_allowed(headers))
    {
        return StatusCode::FORBIDDEN.into_response();
    }
    let read_only =
        request.method() == axum::http::Method::GET || request.method() == axum::http::Method::HEAD;
    if read_only
        && !websocket
        && PUBLIC_READ_ROUTES.contains(&request.uri().path())
        && loopback_host(headers)
    {
        return next.run(request).await;
    }
    let bearer = headers
        .get(header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .is_some_and(|t| access.matches(t));
    let cookie = headers
        .get(header::COOKIE)
        .and_then(|v| v.to_str().ok())
        .is_some_and(|cookies| {
            cookies.split(';').any(|v| {
                v.trim()
                    .strip_prefix(&format!("{COOKIE}="))
                    .is_some_and(|t| access.session_valid(t))
            })
        });
    // Cookie credentials require an allowed browser Origin for every mutation.
    let safe =
        request.method() == axum::http::Method::GET || request.method() == axum::http::Method::HEAD;
    if !bearer && (!cookie || (!safe && !access.origin_allowed(headers))) {
        if safe
            && headers
                .get(header::ACCEPT)
                .and_then(|v| v.to_str().ok())
                .is_some_and(|v| v.contains("text/html"))
        {
            return axum::response::Redirect::to("/login").into_response();
        }
        return StatusCode::UNAUTHORIZED.into_response();
    }
    let mut response = next.run(request).await;
    response
        .headers_mut()
        .insert(header::CACHE_CONTROL, "no-store".parse().unwrap());
    response
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::{
        Router, middleware,
        routing::{get, post},
    };

    #[tokio::test]
    async fn boundary_rejects_missing_tokens_and_cross_origin_sessions() {
        let access =
            Arc::new(Access::new("a".repeat(64), vec!["http://localhost:18095".into()]).unwrap());
        let app = Router::new()
            .route(
                "/private",
                get(|| async { "private" }).post(|| async { "changed" }),
            )
            .route("/ws/terminal", get(|| async { "terminal" }))
            .layer(middleware::from_fn_with_state(
                Arc::clone(&access),
                require_access,
            ))
            .merge(
                Router::new()
                    .route("/auth/session", post(session))
                    .with_state(Arc::clone(&access)),
            );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = reqwest::Client::builder()
            .redirect(reqwest::redirect::Policy::none())
            .build()
            .unwrap();
        for path in ["/private", "/ws/terminal"] {
            assert_eq!(
                client
                    .get(format!("{url}{path}"))
                    .send()
                    .await
                    .unwrap()
                    .status(),
                401
            );
        }
        assert_eq!(
            client
                .get(format!("{url}/private"))
                .bearer_auth("wrong")
                .send()
                .await
                .unwrap()
                .status(),
            401
        );
        assert_eq!(
            client
                .get(format!("{url}/private"))
                .bearer_auth("a".repeat(64))
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        let form = format!("token={}", "a".repeat(64));
        assert_eq!(
            client
                .post(format!("{url}/auth/session"))
                .header("Origin", "http://evil.example")
                .header("Content-Type", "application/x-www-form-urlencoded")
                .body(form.clone())
                .send()
                .await
                .unwrap()
                .status(),
            403
        );
        let login = client
            .post(format!("{url}/auth/session"))
            .header("Origin", "http://localhost:18095")
            .header("Content-Type", "application/x-www-form-urlencoded")
            .body(form)
            .send()
            .await
            .unwrap();
        assert_eq!(login.status(), 303);
        let cookie = login
            .headers()
            .get("set-cookie")
            .unwrap()
            .to_str()
            .unwrap()
            .split(';')
            .next()
            .unwrap()
            .to_owned();
        assert!(!cookie.contains(&"a".repeat(64)));
        assert_eq!(
            client
                .get(format!("{url}/private"))
                .header("Cookie", &cookie)
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        assert_eq!(
            client
                .post(format!("{url}/private"))
                .header("Cookie", &cookie)
                .send()
                .await
                .unwrap()
                .status(),
            401
        );
        assert_eq!(
            client
                .post(format!("{url}/private"))
                .header("Cookie", &cookie)
                .header("Origin", "http://evil.example")
                .send()
                .await
                .unwrap()
                .status(),
            403
        );
        assert_eq!(
            client
                .post(format!("{url}/private"))
                .header("Cookie", &cookie)
                .header("Origin", "http://localhost:18095")
                .send()
                .await
                .unwrap()
                .status(),
            200
        );
        for expires in access.sessions.lock().unwrap().values_mut() {
            *expires = Instant::now() - Duration::from_secs(1);
        }
        assert_eq!(
            client
                .get(format!("{url}/private"))
                .header("Cookie", &cookie)
                .send()
                .await
                .unwrap()
                .status(),
            401
        );
        server.abort();
    }

    #[test]
    fn refuses_empty_credentials_and_invalid_origins() {
        assert!(Access::new(String::new(), vec!["http://localhost:18095".into()]).is_err());
        assert!(Access::new("a".repeat(64), vec!["http://localhost:18095/path".into()]).is_err());
    }

    #[tokio::test]
    async fn health_reads_stay_open_on_loopback_only() {
        let access =
            Arc::new(Access::new("a".repeat(64), vec!["http://localhost:18095".into()]).unwrap());
        let app = Router::new()
            .route("/api/pulse", get(|| async { "pulse" }))
            .route("/api/status", get(|| async { "status" }))
            .route("/api/goals", get(|| async { "goals" }))
            .layer(middleware::from_fn_with_state(
                Arc::clone(&access),
                require_access,
            ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let url = format!("http://{}", listener.local_addr().unwrap());
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        let client = reqwest::Client::new();
        for path in ["/api/pulse", "/api/status"] {
            let status = client
                .get(format!("{url}{path}"))
                .send()
                .await
                .unwrap()
                .status();
            assert_eq!(status, 200, "{path} must stay readable for local monitors");
        }
        let not_public = client
            .get(format!("{url}/api/goals"))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(not_public, 401);
        let mutation = client
            .post(format!("{url}/api/pulse"))
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(mutation, 401);
        let rebound = client
            .get(format!("{url}/api/pulse"))
            .header("Host", "evil.example:18095")
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(rebound, 401);
        let foreign_origin = client
            .get(format!("{url}/api/pulse"))
            .header("Origin", "http://evil.example")
            .send()
            .await
            .unwrap()
            .status();
        assert_eq!(foreign_origin, 403);
        server.abort();
    }
}
