//! Loopback integration tests: a real TCP server with the stub engine and the
//! byte codec, driven by a hand-written HTTP/1.1 client (no extra crates).

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

fn start(engine: StubEngine, cfg: ServerConfig) -> Server {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let cfg = ServerConfig {
        port: 0,
        model_id: "test-model".into(),
        ..cfg
    };
    let (addr, _handle) = rt
        .block_on(serve(cfg, Arc::new(engine), Arc::new(ByteCodec)))
        .expect("bind loopback");
    Server { addr, _rt: rt }
}

fn stub(reply: &str) -> StubEngine {
    StubEngine::new(reply.bytes().map(u32::from).collect())
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

const PROMPT: &str = r#"{"model":"test-model","messages":[{"role":"system","content":"s"},{"role":"user","content":"hi"}]}"#;
// ByteCodec renders "<system>s</system>\n<user>hi</user>\n<assistant>" = 46 bytes.
const PROMPT_TOKENS: u64 = 46;

#[test]
fn models_lists_the_one_served_id() {
    let srv = start(stub("x"), ServerConfig::default());
    let r = request(srv.addr, "GET", "/v1/models", None);
    assert_eq!(r.status, 200);
    let v = json(&r);
    assert_eq!(v["object"], "list");
    assert_eq!(v["data"][0]["id"], "test-model");
    assert_eq!(v["data"][0]["owned_by"], "aien/stub");
    let r = request(srv.addr, "GET", "/v1/models/other", None);
    assert_eq!(r.status, 404);
    assert_eq!(json(&r)["error"]["type"], "not_found_error");
    let h = json(&request(srv.addr, "GET", "/health", None));
    assert_eq!(h["status"], "ok");
    assert_eq!(h["busy"], false);
}

#[test]
fn non_stream_completion_has_content_usage_and_finish() {
    let srv = start(stub("Hello, world!"), ServerConfig::default());
    let r = chat(srv.addr, PROMPT);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = json(&r);
    assert_eq!(v["object"], "chat.completion");
    assert_eq!(v["model"], "test-model");
    assert_eq!(v["choices"][0]["message"]["role"], "assistant");
    assert_eq!(v["choices"][0]["message"]["content"], "Hello, world!");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert_eq!(v["usage"]["prompt_tokens"], PROMPT_TOKENS);
    assert_eq!(v["usage"]["completion_tokens"], 13);
    assert_eq!(v["usage"]["total_tokens"], PROMPT_TOKENS + 13);
    assert!(v["id"].as_str().unwrap().starts_with("chatcmpl-"));
}

#[test]
fn stream_chunks_parse_and_end_with_done() {
    let srv = start(stub("héllo wörld"), ServerConfig::default());
    let body = r#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#;
    let r = chat(srv.addr, body);
    assert_eq!(r.status, 200, "{}", r.body);
    assert!(
        r.headers
            .to_ascii_lowercase()
            .contains("content-type: text/event-stream"),
        "{}",
        r.headers
    );
    let data = sse_data(&r.body);
    assert!(data.len() >= 3, "{data:?}");
    assert_eq!(data.last().unwrap(), "[DONE]");
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap_or_else(|e| panic!("chunk not JSON ({e}): {d}")))
        .collect();
    assert!(chunks
        .iter()
        .all(|c| c["object"] == "chat.completion.chunk"));
    assert_eq!(chunks[0]["choices"][0]["delta"]["role"], "assistant");
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "héllo wörld");
    assert!(
        chunks.iter().all(|c| !c.to_string().contains('\u{FFFD}')),
        "a chunk split a character"
    );
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "stop");
    assert_eq!(last["usage"]["completion_tokens"], 13);
    assert_eq!(last["usage"]["prompt_tokens"], 27); // "<user>hi</user>\n<assistant>"
}

#[test]
fn second_request_while_busy_is_rejected_then_accepted() {
    let reply: String = "x".repeat(60);
    let srv = start(
        stub(&reply).with_delay(Duration::from_millis(25)),
        ServerConfig::default(),
    );
    let addr = srv.addr;
    let first = std::thread::spawn(move || {
        chat(
            addr,
            r#"{"messages":[{"role":"user","content":"a"}],"stream":true}"#,
        )
    });
    std::thread::sleep(Duration::from_millis(300));
    let busy = chat(addr, PROMPT);
    assert_eq!(busy.status, 503, "{}", busy.body);
    assert_eq!(json(&busy)["error"]["type"], "server_busy");
    assert!(
        busy.headers.to_ascii_lowercase().contains("retry-after: 1"),
        "{}",
        busy.headers
    );
    assert_eq!(json(&request(addr, "GET", "/health", None))["busy"], true);
    let first = first.join().unwrap();
    assert_eq!(first.status, 200);
    assert_eq!(sse_data(&first.body).last().unwrap(), "[DONE]");
    let after = chat(addr, PROMPT);
    assert_eq!(after.status, 200, "{}", after.body);
    assert_eq!(json(&after)["choices"][0]["message"]["content"], reply);
}

#[test]
fn bad_requests_get_openai_error_json() {
    let srv = start(
        stub("x"),
        ServerConfig {
            max_body_bytes: 2048,
            ..ServerConfig::default()
        },
    );
    let cases: Vec<(&str, String, u16, &str)> = vec![
        ("not json", "{nope".into(), 400, "invalid JSON"),
        ("no messages", r#"{"messages":[]}"#.into(), 400, "non-empty"),
        (
            "temperature",
            r#"{"messages":[{"role":"user","content":"x"}],"temperature":5}"#.into(),
            400,
            "temperature",
        ),
        (
            "image part",
            r#"{"messages":[{"role":"user","content":[{"type":"image_url","image_url":{"url":"x"}}]}]}"#.into(),
            400,
            "text parts",
        ),
        (
            "wrong model",
            r#"{"model":"gpt-4","messages":[{"role":"user","content":"x"}]}"#.into(),
            404,
            "not found",
        ),
        (
            "too large",
            format!(r#"{{"messages":[{{"role":"user","content":"{}"}}]}}"#, "y".repeat(4000)),
            413,
            "exceeds",
        ),
    ];
    for (name, body, status, needle) in cases {
        let r = chat(srv.addr, &body);
        assert_eq!(r.status, status, "{name}: {}", r.body);
        let v = json(&r);
        let msg = v["error"]["message"].as_str().unwrap_or_default();
        assert!(msg.contains(needle), "{name}: {msg}");
    }
    // Tool calling is supported (Qwen3.6 tool format, coordinator 2026-10-21): tools, tool turns and
    // assistant tool_calls render instead of being refused.
    for body in [
        r#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
        r#"{"messages":[{"role":"user","content":"x"},{"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},{"role":"tool","tool_call_id":"c","content":"ok"}],"tools":[{"type":"function","function":{"name":"f","parameters":{}}}]}"#,
    ] {
        let r = chat(srv.addr, body);
        assert_eq!(r.status, 200, "tools request: {}", r.body);
    }
    let r = request(srv.addr, "GET", "/nope", None);
    assert_eq!(r.status, 404);
    assert_eq!(json(&r)["error"]["type"], "not_found_error");
}

#[test]
fn stop_sequences_and_max_tokens_apply() {
    let srv = start(stub("Hello, world!"), ServerConfig::default());
    let r = chat(
        srv.addr,
        r#"{"messages":[{"role":"user","content":"hi"}],"stop":["world"]}"#,
    );
    let v = json(&r);
    assert_eq!(v["choices"][0]["message"]["content"], "Hello, ");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");

    let r = chat(
        srv.addr,
        r#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":5,"stream":true}"#,
    );
    let data = sse_data(&r.body);
    let chunks: Vec<Value> = data[..data.len() - 1]
        .iter()
        .map(|d| serde_json::from_str(d).unwrap())
        .collect();
    let text: String = chunks
        .iter()
        .filter_map(|c| c["choices"][0]["delta"]["content"].as_str())
        .collect();
    assert_eq!(text, "Hello");
    let last = chunks.last().unwrap();
    assert_eq!(last["choices"][0]["finish_reason"], "length");
    assert_eq!(last["usage"]["completion_tokens"], 5);
}

#[test]
fn prompt_that_fills_the_context_is_refused() {
    let srv = start(stub("x").with_max_context(30), ServerConfig::default());
    let r = chat(srv.addr, PROMPT);
    assert_eq!(r.status, 400, "{}", r.body);
    assert_eq!(json(&r)["error"]["code"], "context_length_exceeded");
}

#[test]
fn non_loopback_bind_is_refused() {
    let rt = tokio::runtime::Runtime::new().unwrap();
    let cfg = ServerConfig {
        bind: "0.0.0.0".parse().unwrap(),
        port: 0,
        ..ServerConfig::default()
    };
    let err = rt
        .block_on(serve(cfg, Arc::new(stub("x")), Arc::new(ByteCodec)))
        .expect_err("must refuse");
    assert!(err.to_string().contains("loopback"), "{err}");
}

// ---------------------------------------------------------------------------------------------
// OWN-8B stage S: observed, seeded sampling through HTTP. The engine is the real Omega door over
// the tiny host-test Qwen3-MoE (`host-test-engine`: real logits, host matmuls, no chip), so these
// are gates (a) replay, (b) spread and (c) greedy of `scripts/sampling-gate.sh` on the host.

mod sampling_gate {
    use super::*;
    use aien_inference_abi::qwen3_moe_omega::{host_test_engine, WeightFormat};
    use aien_serve_http::codec::{Codec, CodecError};
    use aien_serve_http::engine::{Engine, EngineError, GenerateOutcome, GenerateParams};
    use aien_serve_http::omega_engine::OmegaDoor;
    use std::path::PathBuf;

    /// The tiny model has 64 tokens: bytes map to `b % 64` on the way in, token `t` decodes to
    /// the printable byte `'0' + t` on the way out. Nothing stops early (no stop ids).
    struct TinyCodec;

    impl Codec for TinyCodec {
        fn render_chat(&self, turns: &[(&str, &str)]) -> Result<String, CodecError> {
            ByteCodec.render_chat(turns)
        }
        fn encode(&self, text: &str) -> Result<Vec<u32>, CodecError> {
            Ok(text.bytes().map(|b| u32::from(b) % 64).collect())
        }
        fn decode(&self, tokens: &[u32]) -> Result<String, CodecError> {
            Ok(tokens
                .iter()
                .map(|&t| char::from(b'0' + (t % 64) as u8))
                .collect())
        }
        fn stop_token_ids(&self) -> Vec<u32> {
            Vec::new()
        }
    }

    /// Negative controls: the real door with the request's seed or temperature overridden,
    /// the way an engine that ignores the seed (or always takes the argmax) would behave.
    struct Broken {
        door: OmegaDoor,
        force_seed: Option<u64>,
        force_greedy: bool,
    }

    impl Engine for Broken {
        fn kind(&self) -> &str {
            "broken-sampler"
        }
        fn max_context_tokens(&self) -> usize {
            self.door.max_context_tokens()
        }
        fn generate(
            &self,
            prompt: &[u32],
            params: &GenerateParams,
            on_token: &mut dyn FnMut(u32) -> bool,
        ) -> Result<GenerateOutcome, EngineError> {
            let mut p = params.clone();
            if let Some(s) = self.force_seed {
                p.seed = s;
            }
            if self.force_greedy {
                p.temperature = 0.0;
            }
            self.door.generate(prompt, &p, on_token)
        }
    }

    fn lock_path(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!("stage-s-{tag}-{}.lock", std::process::id()))
    }

    fn door(tag: &str) -> OmegaDoor {
        let e = host_test_engine::engine(WeightFormat::Fp8All, 256).expect("tiny engine");
        OmegaDoor::from_parts(e, lock_path(tag), Duration::from_secs(20))
    }

    fn serve_engine(engine: Arc<dyn Engine>) -> Server {
        let rt = tokio::runtime::Runtime::new().expect("runtime");
        let cfg = ServerConfig {
            port: 0,
            model_id: "tiny".into(),
            ..ServerConfig::default()
        };
        let (addr, _h) = rt
            .block_on(serve(cfg, engine, Arc::new(TinyCodec)))
            .expect("bind loopback");
        Server { addr, _rt: rt }
    }

    const PROMPTS: [&str; 4] = [
        "In one sentence, what is a Grace Blackwell GB10?",
        "Count from one to ten in words, comma separated.",
        "Name three colours.",
        "Give me three numbered steps for checking a bolt is torqued correctly.",
    ];

    fn body(prompt: &str, temperature: f32, top_p: f32, top_k: u32, seed: Option<u64>) -> String {
        let mut v = serde_json::json!({
            "model": "tiny",
            "max_tokens": 48,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "messages": [{"role": "user", "content": prompt}],
        });
        if let Some(s) = seed {
            v["seed"] = s.into();
        }
        v.to_string()
    }

    /// (content, usage.aien_sampling) of one non-streaming request, sent the moment the previous
    /// reply arrived (no retry: the slot is free once a reply is out).
    fn ask(addr: SocketAddr, body: &str) -> (String, Value) {
        let r = chat(addr, body);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = json(&r);
        let content = v["choices"][0]["message"]["content"]
            .as_str()
            .expect("content")
            .to_string();
        (content, v["usage"]["aien_sampling"].clone())
    }

    /// Gate (a): seed 1 at temperature 1.0, top_p 1.0, top_k 0 twice: identical, per prompt.
    fn gate_a(addr: SocketAddr) -> usize {
        PROMPTS
            .iter()
            .filter(|p| {
                let b = body(p, 1.0, 1.0, 0, Some(1));
                ask(addr, &b).0 == ask(addr, &b).0
            })
            .count()
    }

    /// Gate (b): seeds 1, 2, 3 at the same settings: at least 2 distinct replies, per prompt.
    fn gate_b(addr: SocketAddr) -> usize {
        PROMPTS
            .iter()
            .filter(|p| {
                let mut r: Vec<String> = (1..=3)
                    .map(|s| ask(addr, &body(p, 1.0, 1.0, 0, Some(s))).0)
                    .collect();
                r.sort();
                r.dedup();
                r.len() >= 2
            })
            .count()
    }

    /// Gate (c): temperature 0 with seeds 1 and 2 equal each other and the engine's own greedy
    /// tokens (taken directly from a fresh door, outside HTTP), per prompt.
    fn gate_c(addr: SocketAddr) -> usize {
        let reference = door("ref");
        PROMPTS
            .iter()
            .filter(|p| {
                let (a, _) = ask(addr, &body(p, 0.0, 1.0, 0, Some(1)));
                let (b, _) = ask(addr, &body(p, 0.0, 1.0, 0, Some(2)));
                let rendered = TinyCodec
                    .render_chat(&[("user", p)])
                    .and_then(|t| TinyCodec.encode(&t))
                    .unwrap();
                let params = GenerateParams {
                    max_new_tokens: 48,
                    temperature: 0.0,
                    top_p: 1.0,
                    top_k: 0,
                    seed: 99,
                    stop_token_ids: vec![],
                };
                let mut toks = Vec::new();
                reference
                    .generate(&rendered, &params, &mut |t| {
                        toks.push(t);
                        true
                    })
                    .unwrap();
                let greedy = TinyCodec.decode(&toks).unwrap();
                a == b && a == greedy
            })
            .count()
    }

    #[test]
    fn honest_sampler_passes_a_b_c_and_records_every_draw() {
        let srv = serve_engine(Arc::new(door("honest")));
        assert_eq!(gate_a(srv.addr), 4, "(a) replay");
        assert_eq!(gate_b(srv.addr), 4, "(b) spread");
        assert_eq!(gate_c(srv.addr), 4, "(c) greedy");

        // The record: mode, the seed used, the settings, one kept-set size per sampled draw.
        let (_, rec) = ask(srv.addr, &body(PROMPTS[0], 0.7, 0.8, 5, Some(446001)));
        assert_eq!(rec["mode"], "sampled", "{rec}");
        assert_eq!(rec["seed"], 446001);
        assert_eq!(rec["top_k"], 5);
        assert_eq!(rec["top_k_applied"], true);
        assert_eq!(rec["sampled_tokens"], 48);
        let kept = rec["kept"].as_array().expect("kept sizes");
        assert_eq!(kept.len(), 48);
        assert!(
            kept.iter().all(|k| (1..=5).contains(&k.as_u64().unwrap())),
            "{rec}"
        );
        assert!(rec["mutant"].is_null());
        // Confidence: p_repeat = exp(reply_logprob), a probability in (0, 1].
        let (p, lp) = (
            rec["p_repeat"].as_f64().expect("p_repeat"),
            rec["reply_logprob"].as_f64().expect("reply_logprob"),
        );
        assert!(p > 0.0 && p <= 1.0 && (p - lp.exp()).abs() < 1e-9, "{rec}");
        let tp = rec["top_prob_min"].as_f64().expect("top_prob_min");
        assert!(
            tp > 0.0 && tp <= rec["top_prob_mean"].as_f64().unwrap(),
            "{rec}"
        );
        let (_, rec) = ask(srv.addr, &body(PROMPTS[0], 0.0, 1.0, 0, Some(1)));
        assert_eq!(rec["p_repeat"], 1.0, "{rec}");
        assert_eq!(rec["mode"], "greedy", "{rec}");
        assert_eq!(
            (
                rec["greedy_tokens"].as_u64(),
                rec["sampled_tokens"].as_u64()
            ),
            (Some(48), Some(0))
        );
    }

    #[test]
    fn a_server_chosen_seed_is_echoed_and_replays() {
        let srv = serve_engine(Arc::new(door("echo")));
        let (first, rec) = ask(srv.addr, &body(PROMPTS[2], 1.0, 1.0, 0, None));
        let seed = rec["seed"].as_u64().expect("echoed seed");
        let (again, rec2) = ask(srv.addr, &body(PROMPTS[2], 1.0, 1.0, 0, Some(seed)));
        assert_eq!(first, again);
        assert_eq!(rec["kept"], rec2["kept"]);
    }

    #[test]
    fn the_stream_carries_the_record_in_its_last_usage() {
        let srv = serve_engine(Arc::new(door("stream")));
        let mut v: Value = serde_json::from_str(&body(PROMPTS[1], 1.0, 1.0, 0, Some(3))).unwrap();
        v["stream"] = true.into();
        let r = chat(srv.addr, &v.to_string());
        assert_eq!(r.status, 200, "{}", r.body);
        let data = sse_data(&r.body);
        let last: Value = serde_json::from_str(&data[data.len() - 2]).unwrap();
        assert_eq!(last["usage"]["aien_sampling"]["seed"], 3, "{last}");
        assert_eq!(last["usage"]["aien_sampling"]["sampled_tokens"], 48);
    }

    #[test]
    fn an_engine_that_ignores_the_seed_fails_spread() {
        let broken = Broken {
            door: door("ignore-seed"),
            force_seed: Some(0),
            force_greedy: false,
        };
        let srv = serve_engine(Arc::new(broken));
        assert_eq!(gate_a(srv.addr), 4, "replay still holds");
        assert_eq!(gate_b(srv.addr), 0, "(b) must fail for a seed-blind engine");
    }

    #[test]
    fn an_engine_that_is_always_greedy_fails_spread_and_passes_greedy() {
        let broken = Broken {
            door: door("force-greedy"),
            force_seed: None,
            force_greedy: true,
        };
        let srv = serve_engine(Arc::new(broken));
        assert_eq!(
            gate_b(srv.addr),
            0,
            "(b) must fail for an always-greedy engine"
        );
        assert_eq!(gate_c(srv.addr), 4, "(c) still passes");
    }

    /// The seeds of the :18091 finding draw distinct per-step seeds, and a flat 4-way choice at
    /// temperature 1.0, top_p 1.0 takes at least 2 different tokens across seeds 1, 2, 3.
    #[test]
    fn repro_seeds_are_distinct_and_a_flat_choice_spreads() {
        use aien_inference_abi::transformer_backend::{decode_sampling_seed, sample_with_params};
        use aien_inference_abi::SamplingParams;
        let mut all: Vec<u64> = [446001u64, 447001, 448001]
            .iter()
            .flat_map(|&s| (0..8).map(move |step| decode_sampling_seed(s, step)))
            .collect();
        let n = all.len();
        all.sort_unstable();
        all.dedup();
        assert_eq!(all.len(), n, "per-step seeds collide");
        let sp = SamplingParams {
            temperature: 1.0,
            top_p: 1.0,
            ..Default::default()
        };
        let mut picks: Vec<u32> = (1..=3u64)
            .map(|s| sample_with_params(&[0.0; 4], &sp, decode_sampling_seed(s, 0)).0)
            .collect();
        picks.sort_unstable();
        picks.dedup();
        assert!(picks.len() >= 2, "{picks:?}");
    }
}
