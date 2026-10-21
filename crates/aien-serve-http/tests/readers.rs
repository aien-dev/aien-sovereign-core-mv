//! The eight readers (sc L371) over a real TCP server: a stub engine, the byte codec (one token per
//! byte) and the loopback tests' hand-written HTTP/1.1 client.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::Duration;

use aien_serve_http::codec::ByteCodec;
use aien_serve_http::engine::StubEngine;
use aien_serve_http::server::{serve, ServerConfig};
use serde_json::Value;

struct Server {
    addr: SocketAddr,
    _rt: tokio::runtime::Runtime,
}

struct Reply {
    status: u16,
    #[allow(dead_code)]
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

use std::sync::Mutex;

use aien_serve_http::engine::{Engine, EngineError, GenerateOutcome, GenerateParams, GroupJob};
use aien_serve_http::readers::ReadersConfig;

/// A batching engine stand-in: says it runs `slots` at once and records every group it gets
/// (its prompts, as text) before running them in turn on the stub.
struct Spy {
    inner: StubEngine,
    slots: usize,
    groups: Arc<Mutex<Vec<Vec<String>>>>,
}

impl Engine for Spy {
    fn kind(&self) -> &str {
        "spy"
    }
    fn max_context_tokens(&self) -> usize {
        self.inner.max_context_tokens()
    }
    fn concurrency(&self) -> usize {
        self.slots
    }
    fn generate(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.inner.generate(prompt, params, on_token)
    }
    fn generate_group(
        &self,
        jobs: &[GroupJob],
        on_token: &mut dyn FnMut(usize, u32) -> bool,
    ) -> Vec<Result<GenerateOutcome, EngineError>> {
        let texts = jobs
            .iter()
            .map(|j| j.prompt.iter().map(|&t| t as u8 as char).collect())
            .collect();
        self.groups.lock().unwrap().push(texts);
        jobs.iter()
            .enumerate()
            .map(|(i, j)| {
                self.inner
                    .generate(&j.prompt, &j.params, &mut |t| on_token(i, t))
            })
            .collect()
    }
}

fn readers_cfg() -> ReadersConfig {
    ReadersConfig {
        threshold_tokens: 800,
        ..ReadersConfig::new("test-model")
    }
}

fn start_spy(slots: usize, reply: &str) -> (Server, Arc<Mutex<Vec<Vec<String>>>>) {
    let groups = Arc::new(Mutex::new(Vec::new()));
    let spy = Spy {
        inner: StubEngine::new(reply.bytes().map(u32::from).collect()).with_max_context(1 << 16),
        slots,
        groups: Arc::clone(&groups),
    };
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let cfg = ServerConfig {
        port: 0,
        model_id: "test-model".into(),
        readers: Some(readers_cfg()),
        ..ServerConfig::default()
    };
    let (addr, _handle) = rt
        .block_on(serve(cfg, Arc::new(spy), Arc::new(ByteCodec)))
        .expect("bind loopback");
    (Server { addr, _rt: rt }, groups)
}

/// 60 lines in 12 paragraphs, each line naming itself: about 4,000 bytes.
fn long_doc() -> String {
    let mut s = String::new();
    for l in 1..=60 {
        s.push_str(&format!(
            "line {l:02}: the quick brown fox jumps over the lazy dog {l:02}\n"
        ));
        if l % 5 == 0 {
            s.push('\n');
        }
    }
    s
}

fn body(model: &str, user: &str, extra: &str) -> String {
    let msgs = serde_json::json!([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": user},
    ]);
    format!(r#"{{"model":"{model}","messages":{msgs}{extra}}}"#)
}

#[test]
fn models_lists_the_readers_id_beside_the_plain_one() {
    let (srv, _) = start_spy(8, "x");
    let v = json(&request(srv.addr, "GET", "/v1/models", None));
    let ids: Vec<&str> = v["data"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap())
        .collect();
    assert_eq!(ids, ["test-model", "test-model-readers"]);
    let r = request(srv.addr, "GET", "/v1/models/test-model-readers", None);
    assert_eq!(r.status, 200);
    assert_eq!(json(&r)["id"], "test-model-readers");
}

#[test]
fn a_short_input_on_the_readers_id_takes_the_plain_path() {
    let (srv, groups) = start_spy(8, "plain answer");
    let v = json(&chat(srv.addr, &body("test-model-readers", "hi", "")));
    assert_eq!(v["choices"][0]["message"]["content"], "plain answer");
    assert_eq!(v["model"], "test-model-readers");
    assert!(v.get("aien_readers").is_none());
    assert!(groups.lock().unwrap().is_empty());
    // The plain id with a long input: the plain path too (the readers are opt-in by id).
    let v = json(&chat(srv.addr, &body("test-model", &long_doc(), "")));
    assert!(v.get("aien_readers").is_none());
    assert!(groups.lock().unwrap().is_empty());
}

#[test]
fn a_long_input_is_read_by_eight_readers_in_one_group_and_the_notes_come_back() {
    let (srv, groups) = start_spy(8, "NOTE");
    let user = format!("Which line mentions 42?\n\n{}", long_doc());
    let v = json(&chat(srv.addr, &body("test-model-readers", &user, "")));
    assert_eq!(v["model"], "test-model-readers");
    assert_eq!(v["choices"][0]["message"]["content"], "NOTE");
    let r = &v["aien_readers"];
    assert_eq!(r["readers"], 8);
    let parts = r["parts"].as_array().unwrap();
    assert_eq!(parts.len(), 8);
    for (k, p) in parts.iter().enumerate() {
        assert_eq!(p["part"], k + 1);
        assert_eq!(p["notes"], "NOTE");
        assert_eq!(p["note_tokens"], 4);
    }
    // One group of eight, in part order, each prompt holding its own part only.
    let g = groups.lock().unwrap();
    assert_eq!(g.len(), 1);
    assert_eq!(g[0].len(), 8);
    for (k, prompt) in g[0].iter().enumerate() {
        assert!(prompt.starts_with("<system>sys</system>"), "{prompt}");
        assert!(prompt.contains(&format!("You are reader {} of 8", k + 1)));
        assert!(prompt.contains("Which line mentions 42?"));
        let first = parts[k]["lines"][0].as_u64().unwrap() as usize;
        assert!(prompt.contains(long_doc().lines().nth(first - 1).unwrap()));
    }
    // (Every reader also sees the message's beginning and end, where a request usually sits.)
    assert!(!g[0][0].contains("line 30:"), "part 1 holds a middle line");
    // The usage counts the connect prompt (notes, not the input).
    let pt = v["usage"]["prompt_tokens"].as_u64().unwrap();
    assert_eq!(pt, r["connect_prompt_tokens"].as_u64().unwrap());
    assert!((pt as usize) < long_doc().len());
}

#[test]
fn show_notes_puts_a_collapsed_preface_before_the_answer() {
    let (srv, _) = start_spy(8, "NOTE");
    let v = json(&chat(
        srv.addr,
        &body(
            "test-model-readers",
            &long_doc(),
            r#","aien_show_notes":true"#,
        ),
    ));
    let c = v["choices"][0]["message"]["content"].as_str().unwrap();
    assert!(
        c.starts_with("<details>\n<summary>Notes from 8 readers"),
        "{c}"
    );
    assert!(c.ends_with("</details>\n\nNOTE"), "{c}");
    assert_eq!(c.matches("NOTE").count(), 9);
}

#[test]
fn streaming_sends_progress_then_the_notes_then_the_answer() {
    let (srv, _) = start_spy(8, "NOTE");
    let r = chat(
        srv.addr,
        &body("test-model-readers", &long_doc(), r#","stream":true"#),
    );
    assert_eq!(r.status, 200);
    let ev = sse_data(&r.body);
    assert_eq!(ev.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = ev[..ev.len() - 1]
        .iter()
        .map(|s| serde_json::from_str(s).unwrap())
        .collect();
    let progress: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["reasoning_content"].as_str())
        .collect();
    assert!(progress.starts_with("Reading "), "{progress}");
    assert!(progress.contains("Notes from 8 readers"));
    let at = chunks
        .iter()
        .position(|c| c.get("aien_readers").is_some())
        .expect("notes chunk");
    assert_eq!(
        chunks[at]["aien_readers"]["parts"]
            .as_array()
            .unwrap()
            .len(),
        8
    );
    let answer: String = chunks[at..]
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(answer, "NOTE");
    assert_eq!(
        chunks.last().unwrap()["choices"][0]["finish_reason"],
        "stop"
    );
}

#[test]
fn fewer_free_rows_mean_fewer_parts_and_one_engine_reads_in_turn() {
    // A one-at-a-time engine: all eight parts, read in turn under its one permit.
    let (srv, groups) = start_spy(1, "N");
    let v = json(&chat(
        srv.addr,
        &body("test-model-readers", &long_doc(), ""),
    ));
    assert_eq!(v["aien_readers"]["readers"], 8);
    assert_eq!(groups.lock().unwrap()[0].len(), 8);
    // A four-row engine: four parts.
    let (srv, groups) = start_spy(4, "N");
    let v = json(&chat(
        srv.addr,
        &body("test-model-readers", &long_doc(), ""),
    ));
    assert_eq!(v["aien_readers"]["readers"], 4);
    assert_eq!(groups.lock().unwrap()[0].len(), 4);
}

/// Tool calling through the readers id (serve/q36-tools): a long tool result is read by the
/// readers, the readers' prompts carry no tools, and the answer's tool call comes back parsed.
#[test]
fn a_long_tool_result_is_read_and_the_answer_can_call_a_tool() {
    let call = "<tool_call>\n<function=read_file>\n<parameter=path>\nsrc/main.rs\n</parameter>\n</function>\n</tool_call>";
    let (srv, groups) = start_spy(8, call);
    let msgs = serde_json::json!([
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "Find the crash, then open the file it names."},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "read_log", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "c1", "content": long_doc()},
    ]);
    let tools = r#"[{"type":"function","function":{"name":"read_file","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}]"#;
    let v = json(&chat(
        srv.addr,
        &format!(r#"{{"model":"test-model-readers","tools":{tools},"messages":{msgs}}}"#),
    ));
    assert_eq!(v["aien_readers"]["readers"], 8);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{v}");
    let tc = &v["choices"][0]["message"]["tool_calls"][0];
    assert_eq!(tc["function"]["name"], "read_file");
    assert!(tc["function"]["arguments"]
        .as_str()
        .unwrap()
        .contains("src/main.rs"));
    let g = groups.lock().unwrap();
    for prompt in &g[0] {
        assert!(!prompt.contains("<tools>"), "a reader got the tools");
        assert!(prompt.contains("Find the crash, then open the file it names."));
    }
}
