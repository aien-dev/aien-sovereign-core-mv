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

//! DRA-47: a slow client cannot hold a connection open past the header read timeout, and no more
//! than `max_connections` connections are served at once. Loopback, port 0, stub engine.

use std::io::{ErrorKind, Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::sync::Arc;
use std::time::{Duration, Instant};

use aien_serve_http::codec::ByteCodec;
use aien_serve_http::engine::StubEngine;
use aien_serve_http::server::{serve, ServerConfig};

const TOKEN: &str = "integration-test-private-credential";
const HEAD_TIMEOUT: Duration = Duration::from_secs(1);
/// Slack for scheduling on a loaded host; the timeout itself is one second.
const MARGIN: Duration = Duration::from_secs(2);

struct Server {
    addr: SocketAddr,
    _rt: tokio::runtime::Runtime,
}

fn start(cfg: ServerConfig) -> Server {
    let rt = tokio::runtime::Runtime::new().expect("runtime");
    let cfg = ServerConfig {
        port: 0,
        model_id: "test-model".into(),
        bearer_token: TOKEN.into(),
        ..cfg
    };
    let engine = StubEngine::new(b"ok".iter().map(|&b| u32::from(b)).collect());
    let (addr, _handle) = rt
        .block_on(serve(cfg, Arc::new(engine), Arc::new(ByteCodec)))
        .expect("bind loopback");
    Server { addr, _rt: rt }
}

const MODELS: &str = "GET /v1/models HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer integration-test-private-credential\r\n";

/// Reads until the peer closes the connection or `deadline` passes; returns when it closed, or
/// `None` while it is still open at the deadline. Bytes the server sends first (an error reply)
/// are read and dropped.
fn closed_by(s: &mut TcpStream, start: Instant, deadline: Duration) -> Option<Duration> {
    let mut buf = [0u8; 1024];
    loop {
        let left = deadline.saturating_sub(start.elapsed());
        if left.is_zero() {
            return None;
        }
        s.set_read_timeout(Some(left)).unwrap();
        match s.read(&mut buf) {
            Ok(0) => return Some(start.elapsed()),
            Ok(_) => continue,
            Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {
                return None
            }
            Err(_) => return Some(start.elapsed()),
        }
    }
}

/// Reads one `Content-Length` reply and returns its status.
fn read_reply(s: &mut TcpStream) -> u16 {
    s.set_read_timeout(Some(Duration::from_secs(10))).unwrap();
    let mut raw = Vec::new();
    let mut buf = [0u8; 4096];
    loop {
        if let Some(end) = raw.windows(4).position(|w| w == b"\r\n\r\n") {
            let head = String::from_utf8_lossy(&raw[..end]).to_ascii_lowercase();
            let len: usize = head
                .lines()
                .find_map(|l| l.strip_prefix("content-length:"))
                .map(|v| v.trim().parse().unwrap())
                .expect("content-length");
            if raw.len() >= end + 4 + len {
                return head.split_whitespace().nth(1).unwrap().parse().unwrap();
            }
        }
        let n = s.read(&mut buf).expect("read reply");
        assert!(n > 0, "connection closed mid-reply");
        raw.extend_from_slice(&buf[..n]);
    }
}

#[test]
fn slow_request_heads_are_disconnected_after_the_header_read_timeout() {
    let server = start(ServerConfig {
        header_read_timeout: HEAD_TIMEOUT,
        ..ServerConfig::default()
    });
    let t = Instant::now();
    let mut partial = TcpStream::connect(server.addr).unwrap();
    partial
        .write_all(b"GET /v1/models HTTP/1.1\r\nHost: localh")
        .unwrap();
    let mut silent = TcpStream::connect(server.addr).unwrap();
    // A kept-alive connection idle after a complete request is held to the same bound.
    let mut idle = TcpStream::connect(server.addr).unwrap();
    idle.write_all(format!("{MODELS}\r\n").as_bytes()).unwrap();
    assert_eq!(read_reply(&mut idle), 200);
    let idle_from = Instant::now();

    let bound = HEAD_TIMEOUT + MARGIN;
    let p = closed_by(&mut partial, t, bound);
    let s = closed_by(&mut silent, t, bound);
    let i = closed_by(&mut idle, idle_from, bound);
    eprintln!("closed after: partial head {p:?}, no bytes {s:?}, idle keep-alive {i:?}");
    assert!(
        p.is_some(),
        "partial request head still open after {bound:?}"
    );
    assert!(s.is_some(), "silent connection still open after {bound:?}");
    assert!(
        i.is_some(),
        "idle keep-alive connection still open after {bound:?}"
    );

    // A normal request is still served, bearer token and all.
    let mut ok = TcpStream::connect(server.addr).unwrap();
    ok.write_all(format!("{MODELS}Connection: close\r\n\r\n").as_bytes())
        .unwrap();
    assert_eq!(read_reply(&mut ok), 200);
}

#[test]
fn connections_past_the_limit_wait_for_a_free_slot() {
    let server = start(ServerConfig {
        max_connections: 1,
        ..ServerConfig::default()
    });
    let held = TcpStream::connect(server.addr).unwrap();
    // Let the server accept `held` before the next connection arrives.
    std::thread::sleep(Duration::from_millis(200));
    let mut waiting = TcpStream::connect(server.addr).unwrap();
    waiting
        .write_all(format!("{MODELS}Connection: close\r\n\r\n").as_bytes())
        .unwrap();
    waiting
        .set_read_timeout(Some(Duration::from_millis(500)))
        .unwrap();
    let mut b = [0u8; 1];
    match waiting.read(&mut b) {
        Err(e) if matches!(e.kind(), ErrorKind::WouldBlock | ErrorKind::TimedOut) => {}
        other => panic!("second connection was served past the limit of 1: {other:?}"),
    }
    drop(held);
    assert_eq!(read_reply(&mut waiting), 200);
}
