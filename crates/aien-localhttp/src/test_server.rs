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

//! A raw HTTP/1.1 test server for the crates that call local services through
//! this client. Enabled by the `test-server` feature; meant for dev-dependencies.
//!
//! It answers each accepted connection with the next scripted response, writes
//! the bytes exactly as given and closes, and hands back every request it saw
//! as raw text so tests can assert on the bytes on the wire.

use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

/// A `Content-Length` framed response with the given status line tail
/// (for example `"200 OK"`), content type and body.
pub fn response(status: &str, content_type: &str, body: &str) -> Vec<u8> {
    format!(
        "HTTP/1.1 {status}\r\nContent-Type: {content_type}\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
    .into_bytes()
}

/// Reads one request: the head up to the blank line, then a `Content-Length` body.
pub async fn read_request(stream: &mut TcpStream) -> String {
    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    while !raw.ends_with(b"\r\n\r\n") {
        match stream.read(&mut byte).await {
            Ok(0) | Err(_) => break,
            Ok(_) => raw.push(byte[0]),
        }
    }
    let head = String::from_utf8_lossy(&raw).to_string();
    let length = head
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(':')?;
            name.eq_ignore_ascii_case("content-length")
                .then(|| value.trim().parse::<usize>().ok())
                .flatten()
        })
        .unwrap_or(0);
    let mut body = vec![0u8; length];
    if stream.read_exact(&mut body).await.is_ok() {
        raw.extend_from_slice(&body);
    }
    String::from_utf8_lossy(&raw).into_owned()
}

/// Serves one scripted response per connection, in order, then stops.
/// Returns the base URL (`http://127.0.0.1:<port>`) and a handle that yields
/// the captured requests.
pub async fn serve(responses: Vec<Vec<u8>>) -> (String, JoinHandle<Vec<String>>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let base = format!("http://{}", listener.local_addr().expect("local addr"));
    let handle = tokio::spawn(async move {
        let mut seen = Vec::new();
        for reply in responses {
            let Ok((mut stream, _)) = listener.accept().await else {
                break;
            };
            seen.push(read_request(&mut stream).await);
            let _ = stream.write_all(&reply).await;
            let _ = stream.shutdown().await;
        }
        seen
    });
    (base, handle)
}

/// [`serve`] with a single response.
pub async fn serve_once(reply: Vec<u8>) -> (String, JoinHandle<String>) {
    let (base, handle) = serve(vec![reply]).await;
    let one = tokio::spawn(async move {
        handle
            .await
            .ok()
            .and_then(|mut v| v.pop())
            .unwrap_or_default()
    });
    (base, one)
}

/// Accepts one connection, reads the request and then never answers, so the
/// caller's timeout is what ends the exchange.
pub async fn serve_silent() -> (String, JoinHandle<String>) {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let base = format!("http://{}", listener.local_addr().expect("local addr"));
    let handle = tokio::spawn(async move {
        let Ok((mut stream, _)) = listener.accept().await else {
            return String::new();
        };
        let request = read_request(&mut stream).await;
        tokio::time::sleep(std::time::Duration::from_secs(30)).await;
        drop(stream);
        request
    });
    (base, handle)
}

/// A base URL on 127.0.0.1 where nothing listens (connection refused).
pub async fn closed_port() -> String {
    let listener = TcpListener::bind("127.0.0.1:0")
        .await
        .expect("bind 127.0.0.1:0");
    let base = format!("http://{}", listener.local_addr().expect("local addr"));
    drop(listener);
    base
}
