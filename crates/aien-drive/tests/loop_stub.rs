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

//! One full tool-dispatch round trip against stub HTTP servers (issue #179
//! acceptance test), plus the cortex write payload and the failure paths.

use aien_drive::{run_directive, Config};
use serde_json::Value;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::TcpListener;
use std::sync::{Arc, Mutex};
use std::time::Duration;

/// Serve `replies.len()` requests; each gets the next canned (status, body).
/// Returns the bound URL base and the captured request bodies.
fn stub(replies: Vec<(u16, String)>) -> (String, Arc<Mutex<Vec<String>>>) {
    let l = TcpListener::bind("127.0.0.1:0").unwrap();
    let base = format!("http://{}", l.local_addr().unwrap());
    let seen = Arc::new(Mutex::new(vec![]));
    let s2 = seen.clone();
    std::thread::spawn(move || {
        for (status, body) in replies {
            let (mut sock, _) = l.accept().unwrap();
            let mut r = BufReader::new(sock.try_clone().unwrap());
            let mut len = 0usize;
            loop {
                let mut h = String::new();
                r.read_line(&mut h).unwrap();
                if h.trim().is_empty() {
                    break;
                }
                if let Some(v) = h.to_ascii_lowercase().strip_prefix("content-length:") {
                    len = v.trim().parse().unwrap();
                }
            }
            let mut b = vec![0u8; len];
            r.read_exact(&mut b).unwrap();
            s2.lock().unwrap().push(String::from_utf8(b).unwrap());
            let _ = write!(
                sock,
                "HTTP/1.0 {status} X\r\nContent-Type: text/event-stream\r\n\r\n{body}"
            );
        }
    });
    (base, seen)
}

fn sse(text: &str) -> String {
    let chunk = serde_json::json!({"choices": [{"delta": {"content": text}}]});
    format!("data: {chunk}\n\ndata: [DONE]\n\n")
}

fn cfg(endpoint: String, cortex: String) -> Config {
    Config {
        endpoint,
        model: "m".into(),
        cortex_endpoint: cortex,
        cortex_space: "aien-sources".into(),
        request_timeout: Duration::from_secs(10),
    }
}

#[test]
fn tool_round_trip_then_done() {
    let t1 = "<tool_call>\n{\"name\": \"run_command\", \"arguments\": {\"command\": \"echo ROUNDTRIP_OK\"}}\n</tool_call>";
    let t2 = "<tool_call>{\"name\": \"goal\", \"arguments\": {\"action\": \"milestone_done\"}}</tool_call>";
    let (base, seen) = stub(vec![(200, sse(t1)), (200, sse(t2))]);
    let ok = run_directive(
        &cfg(format!("{base}/v1/chat/completions"), String::new()),
        "do it",
        5,
    )
    .unwrap();
    assert!(ok);
    let reqs = seen.lock().unwrap();
    assert_eq!(reqs.len(), 2);
    let second: Value = serde_json::from_str(&reqs[1]).unwrap();
    let last = second["messages"].as_array().unwrap().last().unwrap()["content"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        last.contains("<tool_response name=\"run_command\">"),
        "{last}"
    );
    assert!(last.contains("ROUNDTRIP_OK"), "{last}");
    assert_eq!(second["stream"], true);
}

#[test]
fn idle_reply_is_not_done() {
    let (base, _) = stub(vec![(200, sse("just prose, no tools"))]);
    assert!(!run_directive(&cfg(format!("{base}/x"), String::new()), "d", 5).unwrap());
}

#[test]
fn max_turns_without_done_is_not_done() {
    let t = "<tool_call>{\"name\": \"list_dir\", \"arguments\": {\"path\": \".\"}}</tool_call>";
    let (base, seen) = stub(vec![(200, sse(t)), (200, sse(t))]);
    assert!(!run_directive(&cfg(format!("{base}/x"), String::new()), "d", 2).unwrap());
    assert_eq!(seen.lock().unwrap().len(), 2);
}

#[test]
fn http_error_and_dead_endpoint_are_errors() {
    let (base, _) = stub(vec![(500, "boom".into())]);
    assert!(run_directive(&cfg(format!("{base}/x"), String::new()), "d", 2).is_err());
    assert!(run_directive(&cfg("http://127.0.0.1:1/x".into(), String::new()), "d", 2).is_err());
}

#[test]
fn reasoning_only_tool_call_is_recovered_and_dashes_sanitized() {
    let chunk = serde_json::json!({"choices": [{"delta": {"reasoning": "a\u{2014}b <tool_call>{\"name\":\"goal\",\"arguments\":{\"action\":\"done\"}}</tool_call>"}}]});
    let (base, seen) = stub(vec![(200, format!("data: {chunk}\n\ndata: [DONE]\n"))]);
    assert!(run_directive(&cfg(format!("{base}/x"), String::new()), "d", 3).unwrap());
    assert_eq!(seen.lock().unwrap().len(), 1);
}

#[test]
fn cortex_tool_posts_payload() {
    let (cortex, seen) = stub(vec![(200, "{}".into())]);
    let a = serde_json::json!({"action":"write","name":"n1","content":"c1","kind":"procedure"});
    let r = aien_drive::dispatch_tool(&cfg(String::new(), cortex), "cortex", &a);
    assert_eq!(r["status"], "ok");
    assert_eq!(r["cortex_code"], 200);
    let p: Value = serde_json::from_str(&seen.lock().unwrap()[0]).unwrap();
    assert_eq!(p["kind"], "entity");
    assert_eq!(p["value"]["canonicalName"], "n1");
    assert_eq!(p["value"]["space"], "aien-sources");
    assert_eq!(p["value"]["metadata"]["kind"], "procedure");
}

#[test]
fn run_command_bad_cwd_is_an_error_value() {
    // An unknown cwd must come back as an error value, not a panic.
    let a = serde_json::json!({"command": "true", "cwd": "/definitely/not/here"});
    let r = aien_drive::dispatch_tool(&cfg(String::new(), String::new()), "run_command", &a);
    assert!(r.get("error").is_some(), "{r}");
}
