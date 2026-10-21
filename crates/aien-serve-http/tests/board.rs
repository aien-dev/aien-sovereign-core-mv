//! Team board (L376) host gates: a real loopback server, a scripted engine that records every
//! prompt it is given, the byte codec (one token per byte), a hand-written HTTP/1.1 client.
//! Gates: the block lands once per conversation (own posts skipped), the prompt stays
//! append-only across turns, a block pasted by hand gives the same prompt tokens, a built-in
//! tool round continues the same prompt in place (prompt ++ reply ++ tail) on the engine's
//! continuation path, mixed rounds forward the client's calls, persistence survives a restart,
//! a retry after an engine error gets identical tokens, a rewritten turn forks the conversation,
//! two in-flight requests of one conversation share one patch, a block that would not fit is
//! left out with the cursors kept.
#![allow(dead_code)]

use std::collections::VecDeque;
use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aien_inference_abi::tokenizer::ChatOptions;
use aien_serve_http::board::{self, Board, Conversation, List, Settings};
use aien_serve_http::codec::{ByteCodec, Codec, CodecError};
use aien_serve_http::engine::{
    Engine, EngineError, FinishReason, GenerateOutcome, GenerateParams, TestSwitches,
};
use aien_serve_http::server::{serve_with_board, ServerConfig};
use aien_serve_http::tools;
use serde_json::{json, Value};

/// An engine that records every prompt (and whether it came on the continuation path) and
/// replies from a script, one entry per call; "ok" when the script runs out.
struct Script {
    seen: Mutex<Vec<(Vec<u32>, bool)>>,
    replies: Mutex<VecDeque<Result<String, EngineError>>>,
    max_context: usize,
    slots: usize,
    delay: Duration,
}

impl Script {
    fn new() -> Arc<Self> {
        Arc::new(Self {
            seen: Mutex::new(Vec::new()),
            replies: Mutex::new(VecDeque::new()),
            max_context: 4096,
            slots: 1,
            delay: Duration::ZERO,
        })
    }
    fn with(max_context: usize, slots: usize, delay: Duration) -> Arc<Self> {
        Arc::new(Self {
            seen: Mutex::new(Vec::new()),
            replies: Mutex::new(VecDeque::new()),
            max_context,
            slots,
            delay,
        })
    }
    fn say(&self, text: &str) {
        self.replies.lock().unwrap().push_back(Ok(text.to_string()));
    }
    fn fail(&self, msg: &str) {
        self.replies
            .lock()
            .unwrap()
            .push_back(Err(EngineError::Failed(msg.into())));
    }
    fn prompts(&self) -> Vec<Vec<u32>> {
        self.seen
            .lock()
            .unwrap()
            .iter()
            .map(|(p, _)| p.clone())
            .collect()
    }
    fn texts(&self) -> Vec<String> {
        self.prompts()
            .iter()
            .map(|p| ByteCodec.decode(p).unwrap())
            .collect()
    }
    fn continuations(&self) -> Vec<bool> {
        self.seen.lock().unwrap().iter().map(|(_, c)| *c).collect()
    }
    fn run(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
        continuation: bool,
    ) -> Result<GenerateOutcome, EngineError> {
        if prompt.len() >= self.max_context {
            return Err(EngineError::PromptTooLong {
                len: prompt.len(),
                max: self.max_context,
            });
        }
        self.seen
            .lock()
            .unwrap()
            .push((prompt.to_vec(), continuation));
        let reply = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Ok("ok".into()))?;
        let t0 = Instant::now();
        let mut emitted = 0usize;
        let mut finish = FinishReason::Stop;
        for tok in reply
            .bytes()
            .map(u32::from)
            .chain(THINK_MARK.iter().copied().take(0))
        {
            if emitted >= params.max_new_tokens || prompt.len() + emitted >= self.max_context {
                finish = FinishReason::Length;
                break;
            }
            if !self.delay.is_zero() {
                std::thread::sleep(self.delay);
            }
            emitted += 1;
            if !on_token(tok) {
                finish = FinishReason::Cancelled;
                break;
            }
        }
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish,
            prefill_ms: 0,
            decode_ms: t0.elapsed().as_millis() as u64,
            sampling: None,
        })
    }
}

const THINK_MARK: [u32; 0] = [];

impl Engine for Script {
    fn kind(&self) -> &str {
        "script"
    }
    fn max_context_tokens(&self) -> usize {
        self.max_context
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
        self.run(prompt, params, on_token, false)
    }
    fn generate_continuation(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        _test: &TestSwitches,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.run(prompt, params, on_token, true)
    }
}

/// An engine whose reply carries raw token ids (the thinking close is not a byte).
struct IdScript {
    inner: Arc<Script>,
    replies: Mutex<VecDeque<Vec<u32>>>,
}

impl Engine for IdScript {
    fn kind(&self) -> &str {
        "ids"
    }
    fn max_context_tokens(&self) -> usize {
        self.inner.max_context
    }
    fn generate(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.go(prompt, params, on_token, false)
    }
    fn generate_continuation(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        _test: &TestSwitches,
        on_token: &mut dyn FnMut(u32) -> bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.go(prompt, params, on_token, true)
    }
}

impl IdScript {
    fn go(
        &self,
        prompt: &[u32],
        params: &GenerateParams,
        on_token: &mut dyn FnMut(u32) -> bool,
        continuation: bool,
    ) -> Result<GenerateOutcome, EngineError> {
        self.inner
            .seen
            .lock()
            .unwrap()
            .push((prompt.to_vec(), continuation));
        let reply = self.replies.lock().unwrap().pop_front().unwrap_or_default();
        let mut emitted = 0;
        for tok in reply {
            if emitted >= params.max_new_tokens {
                break;
            }
            emitted += 1;
            if !on_token(tok) {
                break;
            }
        }
        Ok(GenerateOutcome {
            emitted_tokens: emitted,
            finish: FinishReason::Stop,
            prefill_ms: 0,
            decode_ms: 0,
            sampling: None,
        })
    }
}

/// Byte codec whose generation prompt opens a thinking block unless `enable_thinking` is false;
/// token 300 closes it. The tool-round tail follows the same switch (advisor item 7).
struct ThinkCodec;
const THINK_END: u32 = 300;

fn think_open(o: &ChatOptions) -> &'static str {
    if o.enable_thinking == Some(false) {
        "<think>\n\n</think>\n\n"
    } else {
        "<think>\n"
    }
}

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
        t.push_str(think_open(o));
        Ok(t)
    }
    fn tool_round_tail(&self, tool_turn: &str, o: &ChatOptions) -> Option<String> {
        Some(format!(
            "</assistant>\n<user>{tool_turn}</user>\n<assistant>{}",
            think_open(o)
        ))
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

struct Server {
    addr: SocketAddr,
    _rt: tokio::runtime::Runtime,
}

fn start(engine: Arc<dyn Engine>, codec: Arc<dyn Codec>, board: Option<Arc<Board>>) -> Server {
    start_cfg(engine, codec, board, ServerConfig::default())
}

fn start_cfg(
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    board: Option<Arc<Board>>,
    cfg: ServerConfig,
) -> Server {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let cfg = ServerConfig {
        port: 0,
        model_id: "test-model".into(),
        ..cfg
    };
    let (addr, _handle) = rt
        .block_on(serve_with_board(
            cfg,
            engine,
            codec,
            board,
            std::future::pending(),
        ))
        .expect("bind loopback");
    Server { addr, _rt: rt }
}

struct Reply {
    status: u16,
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
    Reply { status, body }
}

fn json(r: &Reply) -> Value {
    serde_json::from_str(&r.body).unwrap_or_else(|e| panic!("not JSON ({e}): {}", r.body))
}

fn post(addr: SocketAddr, list: &str, author: &str, text: &str) -> Reply {
    request(
        addr,
        "POST",
        &format!("/v1/aien/{list}"),
        Some(&json!({"author": author, "text": text}).to_string()),
    )
}

fn msgs(turns: &[(&str, &str)]) -> Vec<Value> {
    turns
        .iter()
        .map(|(r, c)| json!({"role": r, "content": c}))
        .collect()
}

fn chat_req(conv: &str, turns: &[(&str, &str)]) -> Value {
    json!({"model": "test-model", "messages": msgs(turns), "aien_conversation": conv})
}

fn chat(addr: SocketAddr, body: &Value) -> Reply {
    request(
        addr,
        "POST",
        "/v1/chat/completions",
        Some(&body.to_string()),
    )
}

fn content(r: &Reply) -> String {
    assert_eq!(r.status, 200, "{}", r.body);
    json(r)["choices"][0]["message"]["content"]
        .as_str()
        .unwrap_or("")
        .to_string()
}

fn key_of(conv: &str) -> String {
    format!("id-{}", board::digest(conv))
}

const READ_FILE: &str = r#"{"type":"function","function":{"name":"read_file","description":"read","parameters":{"type":"object","properties":{"path":{"type":"string"}}}}}"#;

fn with_tools(mut req: Value) -> Value {
    req["tools"] = json!([serde_json::from_str::<Value>(READ_FILE).unwrap()]);
    req
}

fn call(name: &str, param: &str, value: &str) -> String {
    format!(
        "<tool_call>\n<function={name}>\n<parameter={param}>\n{value}\n</parameter>\n</function>\n</tool_call>"
    )
}

fn count(hay: &str, needle: &str) -> usize {
    hay.matches(needle).count()
}

#[test]
fn block_lands_once_per_conversation_and_own_posts_are_skipped() {
    let eng = Script::new();
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    let r = post(s.addr, "board", "L371", "reader 3: part ok, sha abc");
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(json(&r)["seq"], 1);

    // Turn 1 of c1: the block is the end of the user turn.
    assert_eq!(
        content(&chat(s.addr, &chat_req("c1", &[("user", "hi")]))),
        "ok"
    );
    let p1 = eng.texts()[0].clone();
    assert!(p1.contains(board::BOARD_HEAD), "{p1}");
    assert!(p1.contains("L371: reader 3: part ok, sha abc"), "{p1}");
    assert!(
        p1.ends_with(&format!("{}</user>\n<assistant>", board::BLOCK_TAIL)),
        "{p1}"
    );
    let c1 = board.conversation(&key_of("c1"));
    assert_eq!(c1.board_seen, 1);
    assert_eq!(c1.patches.len(), 1);
    assert_eq!(c1.patches[0].turn, 0);

    // Turn 2: the same block at the same place, no second copy: the prompt is append-only.
    let hist = [("user", "hi"), ("assistant", "ok"), ("user", "more")];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &hist))), "ok");
    let p2 = eng.texts()[1].clone();
    let stem = p1.strip_suffix("<assistant>").unwrap();
    assert!(
        p2.starts_with(stem),
        "turn 2 does not extend turn 1:\n{p1}\n----\n{p2}"
    );
    assert_eq!(count(&p2, board::BOARD_HEAD), 1);

    // c2 has its own cursor: it gets the entry at its first turn.
    assert_eq!(
        content(&chat(s.addr, &chat_req("c2", &[("user", "yo")]))),
        "ok"
    );
    assert!(eng.texts()[2].contains("sha abc"));

    // c1's own post is not delivered back to c1, and reaches c2.
    board
        .post(List::Board, "c1-agent", "mine", Some(&key_of("c1")))
        .unwrap();
    let hist3 = [
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", "more"),
        ("assistant", "ok"),
        ("user", "again"),
    ];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &hist3))), "ok");
    let p3 = eng.texts()[3].clone();
    assert_eq!(count(&p3, board::BOARD_HEAD), 1, "{p3}");
    assert!(!p3.contains("mine"));
    assert_eq!(board.conversation(&key_of("c1")).board_seen, 2);
    let c2h = [("user", "yo"), ("assistant", "ok"), ("user", "next")];
    assert_eq!(content(&chat(s.addr, &chat_req("c2", &c2h))), "ok");
    let p4 = eng.texts()[4].clone();
    assert!(p4.contains("c1-agent: mine"), "{p4}");
    assert_eq!(count(&p4, board::BOARD_HEAD), 2, "{p4}");

    // Mutant: a cursor that did not advance delivers the entry again, so the once-only checks
    // above depend on the cursor and not on the block being empty.
    let mut rolled = board.conversation(&key_of("c1"));
    rolled.board_seen = 0;
    board.commit(&key_of("c1"), rolled).unwrap();
    let hist4 = [
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", "more"),
        ("assistant", "ok"),
        ("user", "again"),
        ("assistant", "ok"),
        ("user", "once more"),
    ];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &hist4))), "ok");
    let p5 = eng.texts()[5].clone();
    assert_eq!(count(&p5, "sha abc"), 2, "{p5}");
}

#[test]
fn a_block_pasted_by_hand_gives_the_same_prompt_tokens() {
    let board = Arc::new(Board::in_memory());
    board
        .post(List::Memory, "L354", "live door is 18091", None)
        .unwrap();
    board
        .post(List::Board, "L371", "readers: 8/8 parts ok", None)
        .unwrap();
    board
        .post(List::Board, "L375", "pool: prefix 17k shared", None)
        .unwrap();
    let block = board.unseen("x", &Conversation::default()).block().unwrap();

    let with = Script::new();
    let s = start(with.clone(), Arc::new(ByteCodec), Some(board.clone()));
    let turns = [
        ("system", "be brief"),
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", "status?"),
    ];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &turns))), "ok");

    // The same server, delivery off for the request (`aien_board: false`), the block pasted.
    let pasted = board::apply_block("status?", &block);
    let by_hand = [
        ("system", "be brief"),
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", pasted.as_str()),
    ];
    let mut req = chat_req("c9", &by_hand);
    req["aien_board"] = json!(false);
    assert_eq!(content(&chat(s.addr, &req)), "ok");
    let p = with.prompts();
    assert_eq!(p[0], p[1], "board prompt != pasted prompt");
    assert_eq!(
        board.conversation(&key_of("c9")),
        Conversation::default(),
        "no record"
    );
    // A server without a board has no routes for it.
    let plain = Script::new();
    let off = start(plain.clone(), Arc::new(ByteCodec), None);
    assert_eq!(request(off.addr, "GET", "/v1/aien/board", None).status, 404);
    assert_eq!(content(&chat(off.addr, &chat_req("c1", &by_hand))), "ok");
    assert_eq!(
        plain.prompts()[0],
        p[0],
        "no-board server: pasted prompt differs"
    );

    // A tool-response turn: the block rides in its own <tool_response> (the template's last_query
    // rule is untouched), and pasting it as one more tool message gives the same tokens.
    let with2 = Script::new();
    let s2 = start(with2.clone(), Arc::new(ByteCodec), Some(board.clone()));
    let mut req = json!({"model": "test-model", "aien_conversation": "c3", "messages": [
        {"role": "user", "content": "read it"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": "{\"path\": \"/x\"}"}}]},
        {"role": "tool", "tool_call_id": "call_1", "content": "contents"}
    ]});
    req = with_tools(req);
    assert_eq!(content(&chat(s2.addr, &req)), "ok");
    let t = with2.texts()[0].clone();
    let want = board::apply_block(&tools::tool_responses(&["contents".into()]), &block);
    assert!(t.contains(&format!("<user>{want}</user>")), "{t}");
    req["messages"]
        .as_array_mut()
        .unwrap()
        .push(json!({"role": "tool", "tool_call_id": "call_1", "content": block}));
    req["aien_board"] = json!(false);
    assert_eq!(content(&chat(s2.addr, &req)), "ok");
    let p = with2.prompts();
    assert_eq!(p[0], p[1], "tool turn: board prompt != pasted");
}

#[test]
fn a_builtin_tool_round_continues_the_same_prompt_in_place() {
    let eng = Script::new();
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    let r1 = call("board_post", "text", "sha 1234 pushed");
    eng.say(&r1);
    eng.say("done");
    let mut req = with_tools(chat_req("c1", &[("user", "push it and tell the team")]));
    req["aien_agent"] = json!("L376");
    let r = chat(s.addr, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = json(&r);
    assert_eq!(v["choices"][0]["message"]["content"], "done");
    assert_eq!(v["choices"][0]["finish_reason"], "stop");
    assert!(
        v["choices"][0]["message"]["tool_calls"].is_null(),
        "{}",
        r.body
    );
    assert!(!r.body.contains("tool_call"), "{}", r.body);

    let prompts = eng.prompts();
    assert_eq!(prompts.len(), 2);
    assert_eq!(eng.continuations(), vec![false, true]);
    assert_eq!(v["usage"]["prompt_tokens"], prompts[0].len());
    assert_eq!(
        v["usage"]["completion_tokens"],
        (r1.len() + "done".len()) as u64
    );
    let tail = ByteCodec
        .tool_round_tail(
            &tools::tool_responses(&["posted to the team board as entry 1".into()]),
            &ChatOptions::default(),
        )
        .unwrap();
    let mut want = prompts[0].clone();
    want.extend(ByteCodec.encode(&r1).unwrap());
    want.extend(ByteCodec.encode(&tail).unwrap());
    assert_eq!(
        prompts[1],
        want,
        "round 2 is not prompt ++ reply ++ tail:\n{}",
        eng.texts()[1]
    );
    let p0 = eng.texts()[0].clone();
    assert!(p0.contains("board_post") && p0.contains("board_read") && p0.contains("memory_note"));
    assert!(p0.contains("read_file"));

    let (entries, newest) = board.read(List::Board, 0, 10);
    assert_eq!(newest, 1);
    assert_eq!(entries[0].author, "L376");
    assert_eq!(entries[0].text, "sha 1234 pushed");
    assert_eq!(entries[0].conv.as_deref(), Some(key_of("c1").as_str()));

    // The other conversation reads it at its next turn; the poster does not get it back.
    assert_eq!(
        content(&chat(s.addr, &chat_req("c2", &[("user", "news?")]))),
        "ok"
    );
    assert!(eng.texts()[2].contains("L376: sha 1234 pushed"));
    let hist = [
        ("user", "push it and tell the team"),
        ("assistant", "done"),
        ("user", "thanks"),
    ];
    assert_eq!(
        content(&chat(s.addr, &with_tools(chat_req("c1", &hist)))),
        "ok"
    );
    assert!(
        !eng.texts()[3].contains(board::BOARD_HEAD),
        "{}",
        eng.texts()[3]
    );

    // Four built-in rounds at most: the fifth call is left in the text as plain reply.
    for _ in 0..5 {
        eng.say(&call("board_read", "since", "0"));
    }
    let r = chat(s.addr, &with_tools(chat_req("c4", &[("user", "loop")])));
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(
        eng.prompts().len() - 4,
        4,
        "four rounds for the looping conversation"
    );
}

#[test]
fn the_tail_follows_the_thinking_switch_each_round() {
    for thinking in [Some(false), None] {
        let inner = Script::new();
        let r1: Vec<u32> = if thinking == Some(false) {
            call("board_post", "text", "fact")
                .bytes()
                .map(u32::from)
                .collect()
        } else {
            let mut v: Vec<u32> = "plan".bytes().map(u32::from).collect();
            v.push(THINK_END);
            v.extend(
                format!("\n\n{}", call("board_post", "text", "fact"))
                    .bytes()
                    .map(u32::from),
            );
            v
        };
        let r2: Vec<u32> = if thinking == Some(false) {
            "done".bytes().map(u32::from).collect()
        } else {
            let mut v: Vec<u32> = "plan2".bytes().map(u32::from).collect();
            v.push(THINK_END);
            v.extend("\n\ndone".bytes().map(u32::from));
            v
        };
        let eng = Arc::new(IdScript {
            inner: inner.clone(),
            replies: Mutex::new(VecDeque::from(vec![r1.clone(), r2])),
        });
        let s = start(
            eng,
            Arc::new(ThinkCodec),
            Some(Arc::new(Board::in_memory())),
        );
        let mut req = with_tools(chat_req("c1", &[("user", "go")]));
        if let Some(t) = thinking {
            req["chat_template_kwargs"] = json!({"enable_thinking": t});
        }
        let r = chat(s.addr, &req);
        assert_eq!(r.status, 200, "{}", r.body);
        let v = json(&r);
        assert_eq!(
            v["choices"][0]["message"]["content"], "done",
            "{thinking:?}"
        );
        if thinking.is_none() {
            let reasoning = v["choices"][0]["message"]["reasoning_content"]
                .as_str()
                .unwrap();
            assert!(reasoning.contains("plan2"), "{reasoning}");
        }
        let prompts = inner.prompts();
        assert_eq!(prompts.len(), 2, "{thinking:?}");
        let opts = ChatOptions {
            enable_thinking: thinking,
            ..ChatOptions::default()
        };
        let tail = ThinkCodec
            .tool_round_tail(
                &tools::tool_responses(&["posted to the team board as entry 1".into()]),
                &opts,
            )
            .unwrap();
        let mut want = prompts[0].clone();
        want.extend(r1.iter().copied());
        want.extend(ByteCodec.encode(&tail).unwrap());
        assert_eq!(prompts[1], want, "{thinking:?}: round 2 prompt");
        let t1 = ByteCodec.decode(&prompts[1]).unwrap();
        assert!(t1.ends_with(think_open(&opts)), "{thinking:?}: {t1}");
    }
}

#[test]
fn a_mixed_round_answers_the_builtin_call_and_forwards_the_clients() {
    let eng = Script::new();
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    eng.say(&format!(
        "Let me note that.\n{}\n{}",
        call("memory_note", "text", "the live door is 18091"),
        call("read_file", "path", "/etc/hosts")
    ));
    let r = chat(s.addr, &with_tools(chat_req("c1", &[("user", "go")])));
    assert_eq!(r.status, 200, "{}", r.body);
    let v = json(&r);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls");
    let calls = v["choices"][0]["message"]["tool_calls"].as_array().unwrap();
    assert_eq!(calls.len(), 1, "{}", r.body);
    assert_eq!(calls[0]["function"]["name"], "read_file");
    assert_eq!(v["choices"][0]["message"]["content"], "Let me note that.");
    assert_eq!(
        eng.prompts().len(),
        1,
        "no second round: the client runs its call"
    );
    assert_eq!(board.counts(), (0, 1, 1));
    assert_eq!(
        content(&chat(s.addr, &chat_req("c2", &[("user", "hi")]))),
        "ok"
    );
    let p = eng.texts()[1].clone();
    assert!(
        p.contains(board::MEMORY_HEAD) && p.contains("the live door is 18091"),
        "{p}"
    );
    assert!(!p.contains(board::BOARD_HEAD), "{p}");
}

#[test]
fn builtin_tools_off_forwards_board_calls_to_the_client() {
    let eng = Script::new();
    let board = Arc::new(
        Board::open(&Settings {
            dir: None,
            tools: false,
        })
        .unwrap(),
    );
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    eng.say(&call("board_post", "text", "x"));
    let r = chat(s.addr, &with_tools(chat_req("c1", &[("user", "go")])));
    let v = json(&r);
    assert_eq!(v["choices"][0]["finish_reason"], "tool_calls", "{}", r.body);
    assert_eq!(
        v["choices"][0]["message"]["tool_calls"][0]["function"]["name"],
        "board_post"
    );
    assert!(!eng.texts()[0].contains("memory_note"));
    assert_eq!(board.counts().0, 0);
    // No tools in the request: no built-in tools either.
    let eng2 = Script::new();
    let s2 = start(
        eng2.clone(),
        Arc::new(ByteCodec),
        Some(Arc::new(Board::in_memory())),
    );
    assert_eq!(
        content(&chat(s2.addr, &chat_req("c1", &[("user", "go")]))),
        "ok"
    );
    assert!(!eng2.texts()[0].contains("board_post"));
}

#[test]
fn the_board_survives_a_restart() {
    let dir = std::env::temp_dir().join(format!("aien-board-gate-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let settings = Settings {
        dir: Some(dir.clone()),
        tools: true,
    };
    let eng = Script::new();
    {
        let board = Arc::new(Board::open(&settings).unwrap());
        let s = start(eng.clone(), Arc::new(ByteCodec), Some(board));
        assert_eq!(post(s.addr, "board", "L371", "part 1 ok").status, 200);
        assert_eq!(
            content(&chat(s.addr, &chat_req("c1", &[("user", "hi")]))),
            "ok"
        );
        assert!(eng.texts()[0].contains("part 1 ok"));
    }
    let board = Arc::new(Board::open(&settings).unwrap());
    assert_eq!(board.counts(), (1, 0, 1));
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board));
    let hist = [("user", "hi"), ("assistant", "ok"), ("user", "more")];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &hist))), "ok");
    let p = eng.texts()[1].clone();
    assert_eq!(count(&p, "part 1 ok"), 1, "{p}");
    assert!(p.starts_with(eng.texts()[0].strip_suffix("<assistant>").unwrap()));
    let v = json(&request(s.addr, "GET", "/v1/aien/board", None));
    assert_eq!(v["entries"].as_array().unwrap().len(), 1);
    let h = json(&request(s.addr, "GET", "/health", None));
    assert_eq!(h["board"]["entries"], 1);
    assert_eq!(h["board"]["conversations"], 1);
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn http_routes_list_post_and_refuse() {
    let s = start(
        Script::new(),
        Arc::new(ByteCodec),
        Some(Arc::new(Board::in_memory())),
    );
    for i in 1..=5 {
        let r = post(s.addr, "board", "L371", &format!("note {i}"));
        assert_eq!(r.status, 200, "{}", r.body);
        assert_eq!(json(&r)["seq"], i);
    }
    assert_eq!(post(s.addr, "memory", "L354", "keep this").status, 200);
    let v = json(&request(
        s.addr,
        "GET",
        "/v1/aien/board?since=2&limit=2",
        None,
    ));
    assert_eq!(v["object"], "list");
    assert_eq!(v["list"], "board");
    assert_eq!(v["newest"], 5);
    let e = v["entries"].as_array().unwrap();
    assert_eq!(e.len(), 2);
    assert_eq!(e[0]["seq"], 4);
    assert_eq!(e[1]["text"], "note 5");
    let m = json(&request(s.addr, "GET", "/v1/aien/memory", None));
    assert_eq!(m["entries"][0]["author"], "L354");
    assert_eq!(request(s.addr, "GET", "/v1/aien/nope", None).status, 404);
    assert_eq!(
        request(s.addr, "GET", "/v1/aien/board?since=x", None).status,
        400
    );
    let r = post(s.addr, "board", "a", "   ");
    assert_eq!(r.status, 400, "{}", r.body);
    let r = request(
        s.addr,
        "POST",
        "/v1/aien/board",
        Some("{\"text\": \"no author\"}"),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(json(&r)["author"], "http");
    // The same poster, the same text: one entry.
    assert_eq!(json(&post(s.addr, "board", "L371", "note 5"))["seq"], 5);
}

#[test]
fn a_retry_after_an_engine_error_gets_identical_tokens() {
    let eng = Script::new();
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    board.post(List::Board, "L371", "first", None).unwrap();
    eng.fail("chip hiccup");
    let r = chat(s.addr, &chat_req("c1", &[("user", "hi")]));
    assert_eq!(r.status, 500, "{}", r.body);
    board.post(List::Board, "L371", "second", None).unwrap();
    assert_eq!(
        content(&chat(s.addr, &chat_req("c1", &[("user", "hi")]))),
        "ok"
    );
    let p = eng.prompts();
    assert_eq!(p.len(), 2);
    assert_eq!(p[0], p[1], "the retry rendered a different prompt");
    assert!(!eng.texts()[1].contains("second"));
    let hist = [("user", "hi"), ("assistant", "ok"), ("user", "more")];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &hist))), "ok");
    assert!(eng.texts()[2].contains("L371: second"));
}

#[test]
fn a_rewritten_turn_forks_the_conversation() {
    let eng = Script::new();
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    board.post(List::Board, "L371", "fact one", None).unwrap();
    assert_eq!(
        content(&chat(s.addr, &chat_req("c1", &[("user", "hi")]))),
        "ok"
    );
    board.post(List::Board, "L371", "fact two", None).unwrap();
    let hist = [("user", "hi"), ("assistant", "ok"), ("user", "more")];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &hist))), "ok");
    let key = key_of("c1");
    assert_eq!(board.conversation(&key).patches.len(), 2);
    assert_eq!(board.conversation(&key).board_seen, 2);

    // The client edits turn 1 and resends: a fork with turn 1's patch dropped, the cursors
    // rolled back to before it, the old record untouched.
    let edited = [
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", "more, edited"),
    ];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &edited))), "ok");
    let fork = board::fork_key(&key, "more, edited");
    let f = board.conversation(&fork);
    assert_eq!(f.forked_from.as_deref(), Some(key.as_str()));
    assert_eq!(f.patches.len(), 2);
    assert_eq!(f.board_seen, 2);
    let p = eng.texts()[2].clone();
    assert_eq!(count(&p, "fact one"), 1, "{p}");
    assert_eq!(count(&p, "fact two"), 1, "{p}");
    assert!(p.contains("more, edited\n\n"), "{p}");
    assert_eq!(board.conversation(&key).patches.len(), 2, "old record kept");
    assert_eq!(board.counts().2, 2);

    // Sending the edited history again lands on the fork: the prompt is append-only from there.
    let edited2 = [
        ("user", "hi"),
        ("assistant", "ok"),
        ("user", "more, edited"),
        ("assistant", "ok"),
        ("user", "and on"),
    ];
    assert_eq!(content(&chat(s.addr, &chat_req("c1", &edited2))), "ok");
    let p2 = eng.texts()[3].clone();
    assert!(
        p2.starts_with(p.strip_suffix("<assistant>").unwrap()),
        "{p}\n----\n{p2}"
    );
    assert_eq!(board.counts().2, 2);
}

#[test]
fn concurrent_posts_and_one_conversation_in_flight_twice() {
    let eng = Script::with(4096, 2, Duration::from_millis(2));
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    let addr = s.addr;
    let posters: Vec<_> = (0..8)
        .map(|i| {
            std::thread::spawn(move || {
                for j in 0..5 {
                    let r = post(
                        addr,
                        "board",
                        &format!("agent-{i}"),
                        &format!("note {i}.{j}"),
                    );
                    assert_eq!(r.status, 200, "{}", r.body);
                }
            })
        })
        .collect();
    for p in posters {
        p.join().unwrap();
    }
    let (entries, newest) = board.read(List::Board, 0, 100);
    assert_eq!(newest, 40);
    assert!(entries.windows(2).all(|w| w[1].seq == w[0].seq + 1));

    // One conversation, two requests in flight at once (two engine slots): both render the
    // same block, one patch is recorded, nothing is delivered twice.
    eng.say("a reply of some length to hold the slot open");
    eng.say("a reply of some length to hold the slot open");
    let req = chat_req("c1", &[("user", "hi")]);
    let (ra, rb) = {
        let ra = {
            let req = req.clone();
            std::thread::spawn(move || chat(addr, &req))
        };
        let rb = {
            let req = req.clone();
            std::thread::spawn(move || chat(addr, &req))
        };
        (ra.join().unwrap(), rb.join().unwrap())
    };
    assert_eq!(
        (ra.status, rb.status),
        (200, 200),
        "{} / {}",
        ra.body,
        rb.body
    );
    let p = eng.prompts();
    assert_eq!(p.len(), 2);
    assert_eq!(p[0], p[1]);
    let c = board.conversation(&key_of("c1"));
    assert_eq!(c.patches.len(), 1);
    assert_eq!(c.board_seen, 40);
    assert_eq!(count(&eng.texts()[0], board::BOARD_HEAD), 1);
}

#[test]
fn a_block_that_would_not_fit_is_left_out_and_the_cursor_stays() {
    let eng = Script::with(160, 1, Duration::ZERO);
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    board
        .post(List::Board, "L371", &"x".repeat(120), None)
        .unwrap();
    assert_eq!(
        content(&chat(s.addr, &chat_req("c1", &[("user", "hi")]))),
        "ok"
    );
    let p = eng.texts()[0].clone();
    assert!(!p.contains(board::BOARD_HEAD), "{p}");
    let c = board.conversation(&key_of("c1"));
    assert_eq!(c.board_seen, 0);
    assert!(c.patches.is_empty());
    // A prompt that does not fit even bare is still refused.
    let long = "y".repeat(200);
    let r = chat(s.addr, &chat_req("c1", &[("user", long.as_str())]));
    assert_eq!(r.status, 400, "{}", r.body);
}

#[test]
fn a_reply_cut_by_the_length_cap_runs_no_builtin_call() {
    let eng = Script::new();
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    let full = call("board_post", "text", "half written");
    eng.say(&full);
    let mut req = with_tools(chat_req("c1", &[("user", "go")]));
    req["max_tokens"] = json!(full.len() - 10);
    let r = chat(s.addr, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    let v = json(&r);
    assert_eq!(v["choices"][0]["finish_reason"], "length", "{}", r.body);
    assert!(
        v["choices"][0]["message"]["tool_calls"].is_null(),
        "{}",
        r.body
    );
    assert_eq!(
        v["choices"][0]["message"]["content"].as_str().unwrap(),
        &full[..full.len() - 10]
    );
    assert_eq!(board.counts().0, 0, "a cut call must not be run");
    assert_eq!(eng.prompts().len(), 1);
}

#[test]
fn a_numeric_user_or_session_id_is_accepted_not_refused() {
    let eng = Script::new();
    let s = start(
        eng.clone(),
        Arc::new(ByteCodec),
        Some(Arc::new(Board::in_memory())),
    );
    eng.say("hello");
    // llama.cpp clients send integer ids; some SDKs send objects. 61cad330 ignored them: so do we.
    let req = json!({"model": "test-model", "messages": msgs(&[("user", "hi")]), "user": 42, "session_id": {"id": 7}});
    let r = chat(s.addr, &req);
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(content(&r), "hello");
}

#[test]
fn a_conversation_past_the_patch_budget_gets_no_new_block_and_keeps_its_cursor() {
    let eng = Script::with(4_000_000, 1, Duration::ZERO);
    let board = Arc::new(Board::in_memory());
    let s = start(eng.clone(), Arc::new(ByteCodec), Some(board.clone()));
    // A record whose blocks sum past the budget, every patch matching its turn.
    let n = board::PATCH_BUDGET / board::BLOCK_MAX_CHARS + 2;
    let mut turns: Vec<(String, String)> = Vec::new();
    let mut patches = Vec::new();
    for i in 0..n {
        let u = format!("turn {i}");
        patches.push(board::Patch {
            turn: 2 * i,
            hash: board::digest(&u),
            text: format!("[block {i}] {}", "y".repeat(board::BLOCK_MAX_CHARS - 16)),
            board_seen: 1,
            memory_seen: 0,
        });
        turns.push(("user".into(), u));
        turns.push(("assistant".into(), "ok".into()));
    }
    let carried: usize = patches.iter().map(|p| p.text.len()).sum();
    assert!(carried >= board::PATCH_BUDGET);
    board
        .commit(
            &key_of("c1"),
            Conversation {
                author: "c1".into(),
                board_seen: 1,
                memory_seen: 0,
                patches,
                last_used: 0,
                forked_from: None,
            },
        )
        .unwrap();
    let p = post(s.addr, "board", "other", "a new fact");
    assert_eq!(p.status, 200, "{}", p.body);
    turns.push(("user".into(), "and now?".into()));
    let refs: Vec<(&str, &str)> = turns
        .iter()
        .map(|(r, c)| (r.as_str(), c.as_str()))
        .collect();
    eng.say("fine");
    let r = chat(s.addr, &chat_req("c1", &refs));
    assert_eq!(r.status, 200, "{}", r.body);
    let c = board.conversation(&key_of("c1"));
    assert_eq!(c.patches.len(), n, "no new block past the budget");
    assert_eq!(
        c.board_seen, 1,
        "the cursor stays: the fact waits, it is not lost"
    );
    let prompt: String = eng
        .prompts()
        .last()
        .unwrap()
        .iter()
        .map(|&t| t as u8 as char)
        .collect();
    assert!(
        prompt.contains("[block 0]") && prompt.contains(&format!("[block {}]", n - 1)),
        "earlier blocks still re-applied"
    );
    assert!(!prompt.contains("a new fact"), "nothing new appended");
}
