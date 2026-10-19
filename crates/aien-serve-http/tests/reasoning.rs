//! Thinking-block split (sc #642 Qwen3.8 serving): a prompt that ends in `<think>\n` sends the
//! reply's text up to the closing token as `reasoning_content`, drops that token, trims the
//! answer's leading newlines, and `aien_return_token_ids` returns every generated id.
#![allow(dead_code)] // helpers shared in shape with tests/loopback.rs

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use aien_inference_abi::tokenizer::ChatOptions;
use aien_serve_http::codec::{ByteCodec, Codec, CodecError};
use aien_serve_http::engine::StubEngine;
use aien_serve_http::server::{serve, ServerConfig};
use serde_json::Value;

/// Byte codec whose generation prompt opens a thinking block unless `enable_thinking` is false;
/// token 300 closes it.
struct ThinkCodec;

const THINK_END: u32 = 300;

impl Codec for ThinkCodec {
    fn render_chat(&self, turns: &[(&str, &str)]) -> Result<String, CodecError> {
        self.render_chat_with(turns, &ChatOptions::default())
    }
    fn render_chat_with(
        &self,
        turns: &[(&str, &str)],
        o: &ChatOptions,
    ) -> Result<String, CodecError> {
        let mut t = ByteCodec.render_chat(turns)?;
        t.push_str(if o.enable_thinking == Some(false) {
            "<think>\n\n</think>\n\n"
        } else {
            "<think>\n"
        });
        Ok(t)
    }
    fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError> {
        ByteCodec.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> Result<String, CodecError> {
        ByteCodec.decode(tokens)
    }
    fn stop_token_ids(&self) -> Vec<u32> {
        ByteCodec.stop_token_ids()
    }
    fn reasoning_end_id(&self) -> Option<u32> {
        Some(THINK_END)
    }
}

fn reply_ids() -> Vec<u32> {
    let mut ids: Vec<u32> = "Plan it.\n".bytes().map(u32::from).collect();
    ids.push(THINK_END);
    ids.extend("\n\nThe answer.".bytes().map(u32::from));
    ids
}

fn start() -> (SocketAddr, tokio::runtime::Runtime) {
    start_with(None)
}

fn start_with(thinking: Option<bool>) -> (SocketAddr, tokio::runtime::Runtime) {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let cfg = ServerConfig {
        port: 0,
        model_id: "test-model".into(),
        default_enable_thinking: thinking,
        ..ServerConfig::default()
    };
    let (addr, _h) = rt
        .block_on(serve(
            cfg,
            Arc::new(StubEngine::new(reply_ids())),
            Arc::new(ThinkCodec),
        ))
        .expect("bind loopback");
    (addr, rt)
}

struct Reply {
    status: u16,
    headers: String,
    body: String,
}

fn dechunk(raw: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let mut i = 0;
    loop {
        let line_end = raw[i..]
            .windows(2)
            .position(|w| w == b"\r\n")
            .expect("chunk size line")
            + i;
        let size =
            usize::from_str_radix(std::str::from_utf8(&raw[i..line_end]).unwrap().trim(), 16)
                .expect("hex chunk size");
        if size == 0 {
            return out;
        }
        out.extend_from_slice(&raw[line_end + 2..line_end + 2 + size]);
        i = line_end + 2 + size + 2;
    }
}

fn request(addr: SocketAddr, method: &str, path: &str, body: Option<&str>) -> Reply {
    let mut s = TcpStream::connect(addr).expect("connect");
    s.set_read_timeout(Some(Duration::from_secs(30))).unwrap();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = Vec::new();
    s.read_to_end(&mut raw).expect("read reply");
    let split = raw
        .windows(4)
        .position(|w| w == b"\r\n\r\n")
        .expect("header end");
    let headers = String::from_utf8_lossy(&raw[..split]).to_string();
    let status: u16 = headers
        .split_whitespace()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .expect("status code");
    let body_raw = &raw[split + 4..];
    let body = if headers
        .to_ascii_lowercase()
        .contains("transfer-encoding: chunked")
    {
        String::from_utf8(dechunk(body_raw)).unwrap()
    } else {
        String::from_utf8(body_raw.to_vec()).unwrap()
    };
    Reply {
        status,
        headers,
        body,
    }
}

fn chat(addr: SocketAddr, body: &str) -> Reply {
    request(addr, "POST", "/v1/chat/completions", Some(body))
}

fn json(r: &Reply) -> Value {
    serde_json::from_str(&r.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", r.body))
}

/// `data:` payloads of an SSE body, in order; comment lines are skipped.
fn sse_data(body: &str) -> Vec<String> {
    body.lines()
        .filter_map(|l| l.strip_prefix("data: ").or_else(|| l.strip_prefix("data:")))
        .map(|s| s.trim().to_string())
        .collect()
}

#[test]
fn thinking_reply_splits_into_reasoning_and_answer() {
    let (addr, _rt) = start();
    let r = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"q"}],"aien_return_token_ids":true}"#),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v: Value = serde_json::from_str(&r.body).unwrap();
    let m = &v["choices"][0]["message"];
    assert_eq!(m["reasoning_content"], "Plan it.");
    assert_eq!(m["content"], "The answer.");
    let ids: Vec<u32> = v["usage"]["aien_token_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ids, reply_ids());
}

#[test]
fn thinking_off_keeps_everything_as_content_and_ids_stay_opt_in() {
    let (addr, _rt) = start();
    let r = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(
            r#"{"messages":[{"role":"user","content":"q"}],"chat_template_kwargs":{"enable_thinking":false}}"#,
        ),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v: Value = serde_json::from_str(&r.body).unwrap();
    let m = &v["choices"][0]["message"];
    assert!(m.get("reasoning_content").is_none());
    assert_eq!(m["content"], "Plan it.\n\n\nThe answer.");
    assert!(v["usage"].get("aien_token_ids").is_none());
    let bad = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(
            r#"{"messages":[{"role":"user","content":"q"}],"chat_template_kwargs":{"enable_thinking":"no"}}"#,
        ),
    );
    assert_eq!(bad.status, 400);
}

#[test]
fn streamed_thinking_reply_sends_reasoning_deltas_first() {
    let (addr, _rt) = start();
    let r = request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(r#"{"messages":[{"role":"user","content":"q"}],"stream":true}"#),
    );
    assert_eq!(r.status, 200);
    let (mut reasoning, mut content) = (String::new(), String::new());
    for d in sse_data(&r.body) {
        if d == "[DONE]" {
            break;
        }
        let v: Value = serde_json::from_str(&d).unwrap();
        let delta = &v["choices"][0]["delta"];
        if let Some(t) = delta["reasoning_content"].as_str() {
            assert!(content.is_empty(), "reasoning after the answer started");
            reasoning.push_str(t);
        }
        if let Some(t) = delta["content"].as_str() {
            content.push_str(t);
        }
    }
    assert_eq!(reasoning.trim(), "Plan it.");
    assert_eq!(content, "The answer.");
}

#[test]
fn server_default_thinking_off_applies_unless_the_request_asks() {
    let (addr, _rt) = start_with(Some(false));
    let body = r#"{"messages":[{"role":"user","content":"q"}]}"#;
    let v: Value =
        serde_json::from_str(&request(addr, "POST", "/v1/chat/completions", Some(body)).body)
            .unwrap();
    assert!(v["choices"][0]["message"]
        .get("reasoning_content")
        .is_none());
    let body = r#"{"messages":[{"role":"user","content":"q"}],"chat_template_kwargs":{"enable_thinking":true}}"#;
    let v: Value =
        serde_json::from_str(&request(addr, "POST", "/v1/chat/completions", Some(body)).body)
            .unwrap();
    assert_eq!(v["choices"][0]["message"]["reasoning_content"], "Plan it.");
}
