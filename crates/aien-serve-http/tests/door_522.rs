//! Host checks for the sc#522 review follow-ups of the HTTP door: request preparation
//! (chat rendering and tokenizing) must not run on the async runtime, and a busy chip must
//! answer in the same OpenAI error envelope as every other error.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Duration;

use aien_serve_http::codec::{ByteCodec, Codec, CodecError};
use aien_serve_http::engine::{Engine, EngineError, GenerateOutcome, GenerateParams, StubEngine};
use aien_serve_http::server::{serve, ServerConfig};
use serde_json::Value;

const PROMPT: &str =
    r#"{"model":"test-model","messages":[{"role":"user","content":"hi"}],"max_tokens":4}"#;

/// Serves on a CURRENT-THREAD runtime driven by its own thread: anything that blocks the
/// runtime thread stalls every other request, which is what the off-runtime test detects.
fn start_single_thread(engine: Arc<dyn Engine>, codec: Arc<dyn Codec>) -> SocketAddr {
    let (tx, rx) = mpsc::channel();
    std::thread::spawn(move || {
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("runtime");
        rt.block_on(async move {
            let cfg = ServerConfig {
                port: 0,
                model_id: "test-model".into(),
                ..ServerConfig::default()
            };
            let (addr, handle) = serve(cfg, engine, codec).await.expect("bind loopback");
            tx.send(addr).expect("report addr");
            let _ = handle.await;
        });
    });
    rx.recv_timeout(Duration::from_secs(10)).expect("server up")
}

struct Reply {
    status: u16,
    headers: String,
    body: String,
}

/// One HTTP/1.1 exchange; `Err` when the server does not answer within `wait`.
fn request(
    addr: SocketAddr,
    method: &str,
    path: &str,
    body: &str,
    wait: Duration,
) -> std::io::Result<Reply> {
    let mut s = TcpStream::connect(addr)?;
    s.set_read_timeout(Some(wait))?;
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes())?;
    let mut raw = Vec::new();
    s.read_to_end(&mut raw)?;
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header end");
    let headers = String::from_utf8_lossy(&raw[..split]).to_string();
    let status = headers
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status code");
    Ok(Reply {
        status,
        headers,
        body: String::from_utf8_lossy(&raw[split + 4..]).to_string(),
    })
}

/// [`ByteCodec`] whose `encode` reports that it started, then blocks until released (or 20 s).
struct GateCodec {
    entered: Mutex<mpsc::Sender<()>>,
    release: Mutex<mpsc::Receiver<()>>,
}

impl Codec for GateCodec {
    fn render_chat(&self, turns: &[(&str, &str)]) -> Result<String, CodecError> {
        ByteCodec.render_chat(turns)
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError> {
        let _ = self.entered.lock().unwrap().send(());
        let _ = self
            .release
            .lock()
            .unwrap()
            .recv_timeout(Duration::from_secs(20));
        ByteCodec.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError> {
        ByteCodec.decode(tokens)
    }
    fn stop_token_ids(&self) -> Vec<u32> {
        ByteCodec.stop_token_ids()
    }
}

#[test]
fn tokenizing_does_not_block_the_runtime() {
    let (entered_tx, entered_rx) = mpsc::channel();
    let (release_tx, release_rx) = mpsc::channel();
    let codec = GateCodec {
        entered: Mutex::new(entered_tx),
        release: Mutex::new(release_rx),
    };
    let engine = StubEngine::new(b"ok".iter().map(|&b| u32::from(b)).collect());
    let addr = start_single_thread(Arc::new(engine), Arc::new(codec));

    let chat = std::thread::spawn(move || {
        request(
            addr,
            "POST",
            "/v1/chat/completions",
            PROMPT,
            Duration::from_secs(30),
        )
    });
    entered_rx
        .recv_timeout(Duration::from_secs(10))
        .expect("encode started");
    // The tokenizer is now blocked. With it on the runtime thread, /health cannot answer.
    let health = request(addr, "GET", "/health", "", Duration::from_secs(3));
    release_tx.send(()).expect("release encode");
    let health = health.expect("/health answers while a prompt is being tokenized");
    assert_eq!(health.status, 200, "{}", health.body);

    let chat = chat.join().unwrap().expect("chat reply");
    assert_eq!(chat.status, 200, "{}", chat.body);
    let v: Value = serde_json::from_str(&chat.body).expect("json");
    assert_eq!(v["choices"][0]["message"]["content"], "ok");
}

/// An engine whose chip is always held by another job.
struct BusyEngine;

impl Engine for BusyEngine {
    fn kind(&self) -> &str {
        "busy"
    }
    fn max_context_tokens(&self) -> usize {
        4096
    }
    fn generate(
        &self,
        _prompt: &[u32],
        _params: &GenerateParams,
        _on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        Err(EngineError::Busy(
            "GPU lock held by another job for 30 s".into(),
        ))
    }
}

#[test]
fn gpu_busy_is_an_openai_error_envelope() {
    let addr = start_single_thread(Arc::new(BusyEngine), Arc::new(ByteCodec));
    for stream in [false, true] {
        let body = PROMPT.replace(
            "\"max_tokens\":4",
            &format!("\"max_tokens\":4,\"stream\":{stream}"),
        );
        let r = request(
            addr,
            "POST",
            "/v1/chat/completions",
            &body,
            Duration::from_secs(30),
        )
        .expect("reply");
        assert_eq!(r.status, 503, "stream={stream}: {}", r.body);
        assert!(
            r.headers.to_ascii_lowercase().contains("retry-after: 5"),
            "stream={stream}: {}",
            r.headers
        );
        let v: Value = serde_json::from_str(&r.body)
            .unwrap_or_else(|e| panic!("stream={stream}: not JSON ({e}): {}", r.body));
        let err = &v["error"];
        assert_eq!(err["type"], "server_busy", "stream={stream}: {v}");
        assert_eq!(err["code"], "gpu_busy", "stream={stream}: {v}");
        let msg = err["message"].as_str().expect("message string");
        assert!(msg.contains("GPU busy"), "stream={stream}: {msg}");
        assert!(
            msg.contains("held by another job"),
            "stream={stream}: {msg}"
        );
        assert!(err.get("param").is_some(), "stream={stream}: {v}");
    }
}
