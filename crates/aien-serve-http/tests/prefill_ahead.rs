//! End-to-end tests of ahead-of-Enter prefill over real loopback HTTP with the stub engine
//! (host only: no chip). The stub keeps a fake KV with the Omega engine's prefix rule and
//! counts every prompt token it computes, so "the final call computes only the tail" is
//! measured, not assumed.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aien_serve_http::codec::ByteCodec;
use aien_serve_http::engine::StubEngine;
use aien_serve_http::server::{serve, ServerConfig};
use serde_json::{json, Value};

struct Server {
    addr: SocketAddr,
    engine: Arc<StubEngine>,
    _rt: tokio::runtime::Runtime,
}

fn start(engine: StubEngine, cfg: ServerConfig) -> Server {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let engine = Arc::new(engine);
    let cfg = ServerConfig { port: 0, model_id: "test-model".into(), ..cfg };
    let (addr, _h) = rt.block_on(serve(cfg, engine.clone(), Arc::new(ByteCodec))).expect("bind");
    Server { addr, engine, _rt: rt }
}

fn http(addr: SocketAddr, method: &str, path: &str, extra: &str, body: &str) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\n{extra}Content-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).unwrap();
    let split = raw.windows(4).position(|w| w == b"\r\n\r\n").unwrap();
    let head = String::from_utf8_lossy(&raw[..split]).to_string();
    let status: u16 = head.split_whitespace().nth(1).unwrap().parse().unwrap();
    let mut b = raw[split + 4..].to_vec();
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        let mut out = Vec::new();
        let mut i = 0;
        loop {
            let le = b[i..].windows(2).position(|w| w == b"\r\n").unwrap() + i;
            let n = usize::from_str_radix(std::str::from_utf8(&b[i..le]).unwrap().trim(), 16).unwrap();
            if n == 0 {
                break;
            }
            out.extend_from_slice(&b[le + 2..le + 2 + n]);
            i = le + 2 + n + 2;
        }
        b = out;
    }
    (status, serde_json::from_slice(&b).unwrap_or(Value::Null))
}

fn wait_idle(addr: SocketAddr) {
    let t0 = Instant::now();
    loop {
        let (_, v) = http(addr, "GET", "/v1/aien/sessions", "", "");
        if v["pending_session"].is_null() {
            return;
        }
        assert!(t0.elapsed() < Duration::from_secs(20), "prefill never finished: {v}");
        std::thread::sleep(Duration::from_millis(5));
    }
}

fn system_prompt() -> String {
    // ~2.3k tokens under the byte codec (one token per byte).
    "You are Odysseus. Follow the tool rules carefully. ".repeat(46)
}

fn msgs(user: &str) -> Value {
    json!([{"role":"system","content":system_prompt()},{"role":"user","content":user}])
}

fn cfg_chunk(chunk: usize) -> ServerConfig {
    let mut c = ServerConfig::default();
    c.prefill.chunk = chunk;
    c.prefill.hold_back = 0;
    c
}

fn chat(addr: SocketAddr, user: &str, extra: &str, session_field: Option<&str>) -> Value {
    let mut body = json!({"model":"test-model","messages":msgs(user),"max_tokens":4});
    if let Some(s) = session_field {
        body["session"] = json!(s);
    }
    let (st, v) = http(addr, "POST", "/v1/chat/completions", extra, &body.to_string());
    assert_eq!(st, 200, "{v}");
    v
}

#[test]
fn final_call_computes_only_the_tail_and_the_template_trap_is_respected() {
    let srv = start(StubEngine::new(b"ok".iter().map(|&b| u32::from(b)).collect()), cfg_chunk(256));
    let addr = srv.addr;
    // The human types; the client streams drafts. Each call returns quickly with cached_tokens.
    let (st, r1) = http(
        addr,
        "POST",
        "/v1/aien/prefill",
        "",
        &json!({"session":"s1","messages":msgs("what is the capital of fr")}).to_string(),
    );
    assert_eq!(st, 200, "{r1}");
    assert_eq!(r1["cached_tokens"], 0);
    assert_eq!(r1["status"], "queued");
    wait_idle(addr);
    let target1 = r1["target_tokens"].as_u64().unwrap();
    assert_eq!(srv.engine.resident_snapshot().len() as u64, target1);
    // The stable part excludes the closing of the user turn and the generation prompt.
    let resident = String::from_utf8(srv.engine.resident_snapshot().iter().map(|&t| t as u8).collect()).unwrap();
    assert!(resident.ends_with("what is the capital of fr"), "{resident:?}");
    assert!(!resident.contains("</user>") && !resident.ends_with("<assistant>"), "trailing template leaked");

    let (_, r2) = http(
        addr,
        "POST",
        "/v1/aien/prefill",
        "",
        &json!({"session":"s1","messages":msgs("what is the capital of france")}).to_string(),
    );
    assert_eq!(r2["cached_tokens"].as_u64().unwrap(), target1, "append-only draft reuses everything");
    assert_eq!(r2["queued_tokens"], 4);
    wait_idle(addr);

    // Enter: the final request. Only the tail ("?" plus closing and header) is computed.
    let before = srv.engine.computed_total();
    let v = chat(addr, "what is the capital of france?", "X-Aien-Session: s1\r\n", None);
    let computed = srv.engine.computed_total() - before;
    let prompt_tokens = v["usage"]["prompt_tokens"].as_u64().unwrap();
    let cached = v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap();
    assert_eq!(cached + computed, prompt_tokens);
    assert!(computed <= 20, "computed {computed} tokens for the final call, expected only the tail");
    assert!(prompt_tokens > 2000, "the prompt should be about 2.3k tokens, got {prompt_tokens}");
    eprintln!("final call: {prompt_tokens} prompt tokens, {cached} cached, {computed} computed");
}

#[test]
fn an_edit_rewinds_and_the_final_call_still_computes_only_the_tail() {
    let srv = start(StubEngine::new(vec![b'k' as u32]), cfg_chunk(256));
    let addr = srv.addr;
    http(addr, "POST", "/v1/aien/prefill", "", &json!({"session":"e","messages":msgs("tell me about parrots")}).to_string());
    wait_idle(addr);
    // The user deletes "parrots" and types "owls".
    let (_, r) = http(addr, "POST", "/v1/aien/prefill", "", &json!({"session":"e","messages":msgs("tell me about owls")}).to_string());
    wait_idle(addr);
    assert!(r["cached_tokens"].as_u64().unwrap() > 2000);
    let resident = String::from_utf8(srv.engine.resident_snapshot().iter().map(|&t| t as u8).collect()).unwrap();
    assert!(resident.ends_with("tell me about owls"), "KV was not rewound: ...{:?}", &resident[resident.len() - 30..]);
    let before = srv.engine.computed_total();
    let v = chat(addr, "tell me about owls", "", Some("e"));
    assert!(srv.engine.computed_total() - before <= 20);
    assert!(v["usage"]["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() > 2000);
}

#[test]
fn without_prefill_ahead_the_whole_prompt_is_computed_and_ttft_gap_is_visible() {
    // 100 microseconds per computed prompt token stands in for the chip's prefill cost.
    let mk = || start(StubEngine::new(vec![b'k' as u32]).with_prefill_delay(Duration::from_micros(100)), cfg_chunk(256));
    let cold = mk();
    let t0 = Instant::now();
    let v = chat(cold.addr, "what is the capital of france?", "", None);
    let cold_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(v["usage"]["prompt_tokens_details"]["cached_tokens"], 0);
    assert_eq!(cold.engine.computed_total(), v["usage"]["prompt_tokens"].as_u64().unwrap());

    let warm = mk();
    http(warm.addr, "POST", "/v1/aien/prefill", "", &json!({"session":"w","messages":msgs("what is the capital of france")}).to_string());
    wait_idle(warm.addr);
    let t1 = Instant::now();
    chat(warm.addr, "what is the capital of france?", "", Some("w"));
    let warm_ms = t1.elapsed().as_secs_f64() * 1000.0;
    eprintln!("host stub, 2.3k tokens at 0.1 ms/token: without prefill-ahead {cold_ms:.0} ms, with {warm_ms:.0} ms");
    assert!(warm_ms * 3.0 < cold_ms, "prefill-ahead should cut the final call a lot: {warm_ms} vs {cold_ms}");
}

#[test]
fn prefill_requests_are_validated() {
    let srv = start(StubEngine::new(vec![1]), cfg_chunk(64));
    let a = srv.addr;
    let (st, v) = http(a, "POST", "/v1/aien/prefill", "", &json!({"messages":msgs("x")}).to_string());
    assert_eq!(st, 400, "{v}");
    let (st, _) = http(a, "POST", "/v1/aien/prefill", "", &json!({"session":"s"}).to_string());
    assert_eq!(st, 400);
    let (st, _) = http(a, "POST", "/v1/aien/prefill", "", &json!({"session":"s","messages":msgs("x"),"prompt":"y"}).to_string());
    assert_eq!(st, 400);
    // Raw prompt path works.
    let (st, v) = http(a, "POST", "/v1/aien/prefill", "", &json!({"session":"s","prompt":"hello world"}).to_string());
    assert_eq!(st, 200, "{v}");
    assert_eq!(v["target_tokens"], 11);
}

#[test]
fn session_caps_and_lru_are_enforced_over_http() {
    let mut cfg = cfg_chunk(64);
    cfg.prefill.limits.max_sessions = 2;
    let srv = start(StubEngine::new(vec![1]), cfg);
    let a = srv.addr;
    for s in ["a", "b", "c"] {
        http(a, "POST", "/v1/aien/prefill", "", &json!({"session":s,"prompt":format!("session {s} prompt text")}).to_string());
        wait_idle(a);
    }
    let (_, v) = http(a, "GET", "/v1/aien/sessions", "", "");
    let names: Vec<&str> = v["sessions"].as_array().unwrap().iter().map(|s| s["session"].as_str().unwrap()).collect();
    assert_eq!(names, vec!["c", "b"], "least recently used 'a' is evicted: {v}");
}

#[test]
fn a_generation_in_flight_is_not_disturbed_by_prefill_requests() {
    let srv = start(
        StubEngine::new(b"abcdefghij".iter().map(|&b| u32::from(b)).collect()).with_prefill_delay(Duration::from_micros(200)),
        cfg_chunk(16),
    );
    let a = srv.addr;
    let gen = std::thread::spawn(move || chat(a, "hello", "", Some("g")));
    std::thread::sleep(Duration::from_millis(5));
    for i in 0..5 {
        http(a, "POST", "/v1/aien/prefill", "", &json!({"session":"other","prompt":format!("draft number {i} of the next message")}).to_string());
        std::thread::sleep(Duration::from_millis(2));
    }
    let v = gen.join().unwrap();
    assert_eq!(v["choices"][0]["message"]["content"], "abcd", "max_tokens 4 reply intact");
}
