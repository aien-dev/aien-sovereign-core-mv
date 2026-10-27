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

//! Behavior tests against a local tokio TCP server that speaks raw HTTP/1.1,
//! so every byte on the wire is visible and every failure can be staged.

use aien_localhttp::{Client, Error, DEFAULT_MAX_BODY_BYTES, MAX_HEAD_BYTES};
use std::future::Future;
use std::time::{Duration, Instant};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::oneshot;
use tokio::task::JoinHandle;

/// Reads one request: head up to the blank line, then a Content-Length body.
async fn read_request(stream: &mut TcpStream) -> String {
    let mut raw = Vec::new();
    let mut byte = [0u8; 1];
    while !raw.ends_with(b"\r\n\r\n") {
        if stream.read(&mut byte).await.unwrap() == 0 {
            break;
        }
        raw.push(byte[0]);
    }
    let head = String::from_utf8_lossy(&raw).to_string();
    let length = head
        .lines()
        .find_map(|l| {
            let (n, v) = l.split_once(':')?;
            n.eq_ignore_ascii_case("content-length")
                .then(|| v.trim().parse::<usize>().unwrap())
        })
        .unwrap_or(0);
    let mut body = vec![0u8; length];
    stream.read_exact(&mut body).await.unwrap();
    raw.extend_from_slice(&body);
    String::from_utf8(raw).unwrap()
}

/// Accepts one connection, captures the request, then hands the stream to `script`.
async fn serve_once<F, Fut>(script: F) -> (String, JoinHandle<String>)
where
    F: FnOnce(TcpStream) -> Fut + Send + 'static,
    Fut: Future<Output = ()> + Send,
{
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let handle = tokio::spawn(async move {
        let (mut stream, _) = listener.accept().await.unwrap();
        let request = read_request(&mut stream).await;
        script(stream).await;
        request
    });
    (format!("http://{addr}"), handle)
}

/// Writes `bytes` and closes the connection.
async fn respond(bytes: &'static [u8]) -> (String, JoinHandle<String>) {
    serve_once(move |mut s| async move {
        s.write_all(bytes).await.unwrap();
        s.shutdown().await.ok();
    })
    .await
}

/// Writes the head, then holds the connection open without sending more.
async fn respond_then_stall(bytes: &'static [u8]) -> (String, JoinHandle<String>) {
    serve_once(move |mut s| async move {
        s.write_all(bytes).await.unwrap();
        tokio::time::sleep(Duration::from_secs(10)).await;
    })
    .await
}

async fn assert_no_connection(listener: &TcpListener) {
    let accepted = tokio::time::timeout(Duration::from_millis(150), listener.accept()).await;
    assert!(accepted.is_err(), "the client must not connect");
}

// ---------------------------------------------------------------- happy paths

#[tokio::test]
async fn get_writes_a_plain_request_and_reads_a_length_body() {
    let (base, server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nhello").await;
    let response = Client::new()
        .get(format!("{base}/health?x=1"))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 200);
    assert!(response.status().is_success());
    assert_eq!(response.content_length(), Some(5));
    assert_eq!(response.text().await.unwrap(), "hello");
    let request = server.await.unwrap();
    let host = base.trim_start_matches("http://");
    assert_eq!(
        request,
        format!(
            "GET /health?x=1 HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nAccept: */*\r\n\r\n"
        )
    );
}

#[tokio::test]
async fn post_json_sends_bearer_content_type_and_exact_length() {
    let (base, server) =
        respond(b"HTTP/1.1 201 Created\r\nContent-Length: 13\r\n\r\n{\"ok\":true}\r\n").await;
    let response = Client::new()
        .post(format!("{base}/api/cortex/write"))
        .bearer_auth("tok-123")
        .json(&serde_json::json!({"kind": "entity"}))
        .send()
        .await
        .unwrap();
    assert_eq!(response.status().as_u16(), 201);
    let value: serde_json::Value = response.json().await.unwrap();
    assert_eq!(value["ok"], true);
    let request = server.await.unwrap();
    assert!(request.starts_with("POST /api/cortex/write HTTP/1.1\r\n"));
    assert!(request.contains("\r\nAuthorization: Bearer tok-123\r\n"));
    assert!(request.contains("\r\nContent-Type: application/json\r\n"));
    assert!(request.contains("\r\nContent-Length: 17\r\n"));
    assert!(request.ends_with("\r\n\r\n{\"kind\":\"entity\"}"));
}

#[tokio::test]
async fn caller_content_type_is_kept_and_not_duplicated() {
    let (base, server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
    Client::new()
        .post(&base)
        .header("Content-Type", "application/json")
        .json(&serde_json::json!({}))
        .send()
        .await
        .unwrap();
    let request = server.await.unwrap();
    assert_eq!(request.matches("Content-Type").count(), 1, "{request}");
}

#[tokio::test]
async fn raw_body_and_empty_post_send_their_lengths() {
    let (base, server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
    Client::new()
        .post(&base)
        .header("content-type", "application/json")
        .body(b"{\"a\":1}".to_vec())
        .send()
        .await
        .unwrap();
    let request = server.await.unwrap();
    assert!(request.contains("\r\nContent-Length: 7\r\n"));
    assert!(request.ends_with("{\"a\":1}"));

    let (base, server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n").await;
    Client::new().post(&base).send().await.unwrap();
    assert!(server.await.unwrap().contains("\r\nContent-Length: 0\r\n"));
}

#[tokio::test]
async fn chunked_body_with_extensions_and_trailers_is_decoded() {
    let (base, _server) = respond(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
          5;ext=1\r\nhello\r\n1\r\n \r\n5\r\nworld\r\n0\r\nX-Trailer: t\r\n\r\n",
    )
    .await;
    let response = Client::new().get(&base).send().await.unwrap();
    assert_eq!(response.content_length(), None);
    assert_eq!(response.text().await.unwrap(), "hello world");
}

#[tokio::test]
async fn close_delimited_body_reads_to_eof() {
    let (base, _server) = respond(b"HTTP/1.0 200 OK\r\n\r\nuntil the end").await;
    let body = Client::new().get(&base).send().await.unwrap().bytes().await;
    assert_eq!(body.unwrap(), b"until the end");
}

#[tokio::test]
async fn interim_100_is_skipped_and_204_has_no_body() {
    let (base, _server) = respond(
        b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 204 No Content\r\nContent-Length: 9\r\n\r\n",
    )
    .await;
    let response = Client::new().get(&base).send().await.unwrap();
    assert_eq!(response.status().as_u16(), 204);
    assert_eq!(response.bytes().await.unwrap(), b"");
}

#[tokio::test]
async fn error_status_keeps_its_body_and_reqwest_style_display() {
    let (base, _server) =
        respond(b"HTTP/1.1 500 Internal Server Error\r\nContent-Length: 4\r\nX-Id: 7\r\n\r\nboom")
            .await;
    let response = Client::new().get(&base).send().await.unwrap();
    assert!(!response.status().is_success());
    assert_eq!(response.status().to_string(), "500 Internal Server Error");
    assert_eq!(response.header("x-id"), Some("7"));
    assert_eq!(response.text().await.unwrap(), "boom");
}

#[tokio::test]
async fn next_line_yields_each_sse_event_as_it_arrives() {
    let (release_tx, release_rx) = oneshot::channel::<()>();
    let (base, _server) = serve_once(move |mut s| async move {
        s.write_all(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n",
        )
        .await
        .unwrap();
        s.write_all(b"13\r\ndata: {\"t\":\"one\"}\n\n\r\n")
            .await
            .unwrap();
        // The second event is only sent after the client has seen the first.
        release_rx.await.unwrap();
        s.write_all(b"13\r\ndata: {\"t\":\"two\"}\n\n\r\ne\r\ndata: [DONE]\n\n\r\n0\r\n\r\n")
            .await
            .unwrap();
    })
    .await;
    let mut response = Client::new().post(&base).send().await.unwrap();
    assert_eq!(
        response.next_line().await.unwrap().as_deref(),
        Some("data: {\"t\":\"one\"}")
    );
    assert_eq!(response.next_line().await.unwrap().as_deref(), Some(""));
    release_tx.send(()).unwrap();
    let mut rest = Vec::new();
    while let Some(line) = response.next_line().await.unwrap() {
        rest.push(line);
    }
    assert_eq!(rest, ["data: {\"t\":\"two\"}", "", "data: [DONE]", ""]);
}

#[tokio::test]
async fn next_line_joins_a_line_split_across_chunks_and_strips_crlf() {
    let (base, _server) = respond(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n\
          4\r\ndata\r\n7\r\n: half\r\r\n5\r\n\nlast\r\n0\r\n\r\n",
    )
    .await;
    let mut response = Client::new().get(&base).send().await.unwrap();
    assert_eq!(
        response.next_line().await.unwrap().as_deref(),
        Some("data: half")
    );
    assert_eq!(response.next_line().await.unwrap().as_deref(), Some("last"));
    assert_eq!(response.next_line().await.unwrap(), None);
}

#[tokio::test]
async fn chunk_hands_out_body_pieces_in_order() {
    let (base, _server) = respond(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n2\r\nde\r\n0\r\n\r\n",
    )
    .await;
    let mut response = Client::new().get(&base).send().await.unwrap();
    let mut all = Vec::new();
    while let Some(piece) = response.chunk().await.unwrap() {
        all.extend(piece);
    }
    assert_eq!(all, b"abcde");
}

// ------------------------------------------------------------- refusals (no I/O)

#[tokio::test]
async fn https_is_refused_before_any_connection() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    let error = Client::new()
        .get(format!("https://127.0.0.1:{port}/health"))
        .send()
        .await
        .unwrap_err();
    assert!(matches!(error, Error::HttpsRefused { .. }), "{error:?}");
    assert!(error.to_string().contains("https is refused"));
    assert_no_connection(&listener).await;
}

#[tokio::test]
async fn other_schemes_and_bad_urls_are_refused() {
    let error = Client::new()
        .get("ftp://127.0.0.1/")
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::UnsupportedScheme { .. }),
        "{error:?}"
    );
    let error = Client::new().get("127.0.0.1:80").send().await.unwrap_err();
    assert!(matches!(error, Error::InvalidUrl { .. }), "{error:?}");
}

#[tokio::test]
async fn header_injection_and_framing_overrides_are_refused() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let base = format!("http://{}", listener.local_addr().unwrap());
    for (name, value) in [
        ("X-A", "a\r\nX-Injected: 1"),
        ("X-A", "a\nb"),
        ("Bad Name", "v"),
        ("", "v"),
        ("Content-Length", "1"),
        ("transfer-encoding", "chunked"),
        ("Host", "evil"),
        ("Connection", "keep-alive"),
    ] {
        let error = Client::new()
            .get(&base)
            .header(name, value)
            .send()
            .await
            .unwrap_err();
        assert!(
            matches!(error, Error::InvalidHeader { .. }),
            "{name}: {error:?}"
        );
    }
    let error = Client::new()
        .get(&base)
        .bearer_auth("tok\r\nX: y")
        .send()
        .await
        .unwrap_err();
    assert!(matches!(error, Error::InvalidHeader { .. }));
    assert_no_connection(&listener).await;
}

#[tokio::test]
async fn connect_refused_is_a_connect_error() {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let port = listener.local_addr().unwrap().port();
    drop(listener);
    let error = Client::builder()
        .connect_timeout(Duration::from_secs(2))
        .build()
        .get(format!("http://127.0.0.1:{port}/"))
        .send()
        .await
        .unwrap_err();
    assert!(error.is_connect(), "{error:?}");
    assert!(!error.is_timeout());
}

// ------------------------------------------------------------------- size caps

#[tokio::test]
async fn oversize_content_length_is_refused() {
    let (base, _server) =
        respond_then_stall(b"HTTP/1.1 200 OK\r\nContent-Length: 11\r\n\r\nhello").await;
    let client = Client::builder().max_body_bytes(10).build();
    let error = client.get(&base).send().await.unwrap().bytes().await;
    assert!(
        matches!(error, Err(Error::BodyTooLarge { limit: 10 })),
        "{error:?}"
    );
}

#[tokio::test]
async fn oversize_chunked_body_is_refused() {
    let (base, _server) = respond(
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\nabcdef\r\n6\r\nghijkl\r\n0\r\n\r\n",
    )
    .await;
    let client = Client::builder().max_body_bytes(10).build();
    let error = client.get(&base).send().await.unwrap().bytes().await;
    assert!(
        matches!(error, Err(Error::BodyTooLarge { limit: 10 })),
        "{error:?}"
    );
}

#[tokio::test]
async fn oversize_close_delimited_body_is_refused() {
    let (base, _server) = respond(b"HTTP/1.1 200 OK\r\n\r\n0123456789abc").await;
    let client = Client::builder().max_body_bytes(10).build();
    let error = client.get(&base).send().await.unwrap().text().await;
    assert!(
        matches!(error, Err(Error::BodyTooLarge { limit: 10 })),
        "{error:?}"
    );
}

#[tokio::test]
async fn body_exactly_at_the_cap_is_accepted() {
    let (base, _server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\n0123456789").await;
    let client = Client::builder().max_body_bytes(10).build();
    let body = client
        .get(&base)
        .send()
        .await
        .unwrap()
        .bytes()
        .await
        .unwrap();
    assert_eq!(body.len(), 10);
    assert_eq!(DEFAULT_MAX_BODY_BYTES, 16 * 1024 * 1024);
}

#[tokio::test]
async fn oversize_response_head_is_refused() {
    let (base, _server) = serve_once(|mut s| async move {
        let mut head = b"HTTP/1.1 200 OK\r\nX-Big: ".to_vec();
        head.extend(std::iter::repeat_n(b'a', MAX_HEAD_BYTES));
        head.extend_from_slice(b"\r\n\r\n");
        let _ = s.write_all(&head).await;
    })
    .await;
    let error = Client::new().get(&base).send().await.unwrap_err();
    assert!(matches!(error, Error::HeadTooLarge { .. }), "{error:?}");
}

// -------------------------------------------------------------------- timeouts

#[tokio::test]
async fn silent_server_hits_the_total_timeout() {
    let (base, _server) = serve_once(|s| async move {
        let _hold = s;
        tokio::time::sleep(Duration::from_secs(10)).await;
    })
    .await;
    let started = Instant::now();
    let error = Client::builder()
        .timeout(Duration::from_millis(200))
        .build()
        .get(&base)
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Timeout(d) if d == Duration::from_millis(200)),
        "{error:?}"
    );
    assert!(error.is_timeout());
    assert!(started.elapsed() < Duration::from_secs(3));
}

#[tokio::test]
async fn request_timeout_overrides_the_client_timeout() {
    let (base, _server) = serve_once(|s| async move {
        let _hold = s;
        tokio::time::sleep(Duration::from_secs(10)).await;
    })
    .await;
    let error = Client::builder()
        .timeout(Duration::from_secs(30))
        .build()
        .get(&base)
        .timeout(Duration::from_millis(150))
        .send()
        .await
        .unwrap_err();
    assert!(
        matches!(error, Error::Timeout(d) if d == Duration::from_millis(150)),
        "{error:?}"
    );
}

#[tokio::test]
async fn stalled_body_hits_the_total_timeout() {
    let (base, _server) =
        respond_then_stall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial").await;
    let response = Client::builder()
        .timeout(Duration::from_millis(250))
        .build()
        .get(&base)
        .send()
        .await
        .unwrap();
    let error = response.bytes().await.unwrap_err();
    assert!(matches!(error, Error::Timeout(_)), "{error:?}");
}

#[tokio::test]
async fn stalled_stream_line_hits_the_total_timeout() {
    let (base, _server) =
        respond_then_stall(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n6\r\ndata: \r\n")
            .await;
    let mut response = Client::builder()
        .timeout(Duration::from_millis(250))
        .build()
        .get(&base)
        .send()
        .await
        .unwrap();
    let error = response.next_line().await.unwrap_err();
    assert!(matches!(error, Error::Timeout(_)), "{error:?}");
}

// ----------------------------------------------------- malformed and early close

#[tokio::test]
async fn malformed_chunk_size_is_refused() {
    let (base, _server) =
        respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\nzz\r\nhello\r\n0\r\n\r\n")
            .await;
    let error = Client::new().get(&base).send().await.unwrap().bytes().await;
    assert!(matches!(error, Err(Error::MalformedChunk(_))), "{error:?}");
}

#[tokio::test]
async fn chunk_without_trailing_crlf_is_refused() {
    let (base, _server) =
        respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n5\r\nhelloXX0\r\n\r\n")
            .await;
    let error = Client::new().get(&base).send().await.unwrap().bytes().await;
    assert!(matches!(error, Err(Error::MalformedChunk(_))), "{error:?}");
}

#[tokio::test]
async fn oversized_chunk_size_number_is_refused() {
    let (base, _server) =
        respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n10000000000000000\r\nx\r\n")
            .await;
    let error = Client::new().get(&base).send().await.unwrap().bytes().await;
    assert!(matches!(error, Err(Error::MalformedChunk(_))), "{error:?}");
}

#[tokio::test]
async fn early_close_before_the_status_line() {
    let (base, _server) = serve_once(|s| async move { drop(s) }).await;
    let error = Client::new().get(&base).send().await.unwrap_err();
    assert!(matches!(error, Error::EarlyClose(_)), "{error:?}");
}

#[tokio::test]
async fn early_close_inside_the_head() {
    let (base, _server) = respond(b"HTTP/1.1 200 OK\r\nContent-Len").await;
    let error = Client::new().get(&base).send().await.unwrap_err();
    assert!(matches!(error, Error::EarlyClose(_)), "{error:?}");
}

#[tokio::test]
async fn early_close_inside_a_length_body() {
    let (base, _server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\n\r\nabcd").await;
    let error = Client::new().get(&base).send().await.unwrap().bytes().await;
    match error {
        Err(Error::EarlyClose(why)) => assert!(why.contains("4 of 10"), "{why}"),
        other => panic!("expected EarlyClose, got {other:?}"),
    }
}

#[tokio::test]
async fn early_close_inside_a_chunk() {
    let (base, _server) =
        respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\na\r\nabc").await;
    let error = Client::new().get(&base).send().await.unwrap().bytes().await;
    assert!(matches!(error, Err(Error::EarlyClose(_))), "{error:?}");
}

#[tokio::test]
async fn early_close_before_the_last_chunk() {
    let (base, _server) =
        respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n3\r\nabc\r\n").await;
    let mut response = Client::new().get(&base).send().await.unwrap();
    assert_eq!(response.chunk().await.unwrap().unwrap(), b"abc");
    let error = response.chunk().await;
    assert!(matches!(error, Err(Error::EarlyClose(_))), "{error:?}");
}

#[tokio::test]
async fn malformed_heads_are_refused() {
    for raw in [
        &b"HTTP/2 200 OK\r\n\r\n"[..],
        b"HTTP/1.1 2x0 OK\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nNo colon here\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nA: b\r\n folded\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\nContent-Length: 3\r\n\r\nabc",
        b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\nContent-Length: 4\r\n\r\nabc",
        b"HTTP/1.1 101 Switching Protocols\r\n\r\n",
    ] {
        let (base, _server) = respond(raw).await;
        let error = Client::new().get(&base).send().await.unwrap_err();
        assert!(
            matches!(error, Error::MalformedResponse(_)),
            "{:?}: {error:?}",
            String::from_utf8_lossy(raw)
        );
    }
}

#[tokio::test]
async fn invalid_json_body_is_a_decode_error() {
    let (base, _server) = respond(b"HTTP/1.1 200 OK\r\nContent-Length: 3\r\n\r\nnot").await;
    let error = Client::new()
        .get(&base)
        .send()
        .await
        .unwrap()
        .json::<serde_json::Value>()
        .await;
    assert!(matches!(error, Err(Error::Decode(_))), "{error:?}");
}

// -------------------------------------------------------------------- blocking

#[test]
fn send_blocking_works_from_a_plain_thread() {
    // The server runs on its own runtime; the caller thread has none.
    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (base, handle) =
        runtime.block_on(respond(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nok"));
    let response = Client::builder()
        .timeout(Duration::from_secs(5))
        .build()
        .post(format!("{base}/v1/probe"))
        .header("content-type", "application/json")
        .body(b"{}".to_vec())
        .send_blocking()
        .unwrap();
    assert!(response.status().is_success());
    assert_eq!(response.text(), "ok");
    let request = runtime.block_on(handle).unwrap();
    assert!(request.starts_with("POST /v1/probe HTTP/1.1\r\n"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn send_blocking_works_inside_a_runtime() {
    let (base, _server) = respond(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 2\r\n\r\nno").await;
    let response = Client::new().get(&base).send_blocking().unwrap();
    assert_eq!(response.status().as_u16(), 403);
    assert_eq!(response.bytes(), b"no");
}

#[test]
fn send_blocking_refuses_https_and_reports_timeouts() {
    let error = Client::new()
        .post("https://127.0.0.1:1/v1/probe")
        .send_blocking()
        .unwrap_err();
    assert!(matches!(error, Error::HttpsRefused { .. }), "{error:?}");

    let runtime = tokio::runtime::Runtime::new().unwrap();
    let (base, _server) = runtime.block_on(serve_once(|s| async move {
        let _hold = s;
        tokio::time::sleep(Duration::from_secs(10)).await;
    }));
    let error = Client::builder()
        .timeout(Duration::from_millis(200))
        .build()
        .get(&base)
        .send_blocking()
        .unwrap_err();
    assert!(matches!(error, Error::Timeout(_)), "{error:?}");
}

// ------------------------------------------------- review round 1 regressions

#[tokio::test]
async fn empty_transfer_encoding_with_content_length_is_refused() {
    let (base, _server) =
        respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: \r\nContent-Length: 3\r\n\r\nabc").await;
    let error = Client::new().get(&base).send().await.unwrap_err();
    assert!(
        matches!(&error, Error::MalformedResponse(m) if m.contains("both")),
        "{error:?}"
    );
}

#[tokio::test]
async fn empty_transfer_encoding_alone_is_refused() {
    let (base, _server) = respond(b"HTTP/1.1 200 OK\r\nTransfer-Encoding: \r\n\r\nabc").await;
    let error = Client::new().get(&base).send().await.unwrap_err();
    assert!(
        matches!(&error, Error::MalformedResponse(m) if m.contains("unsupported")),
        "{error:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn blocking_status_returns_before_a_stalled_body() {
    let (base, _server) =
        respond_then_stall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 100\r\n\r\npartial").await;
    let started = Instant::now();
    let status = tokio::task::spawn_blocking(move || {
        Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .get(&base)
            .send_blocking_status()
    })
    .await
    .unwrap()
    .unwrap();
    assert_eq!(status.as_u16(), 403);
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[tokio::test(flavor = "multi_thread")]
async fn blocking_full_read_of_a_stalled_body_times_out() {
    let (base, _server) =
        respond_then_stall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\npartial").await;
    let error = tokio::task::spawn_blocking(move || {
        Client::builder()
            .timeout(Duration::from_millis(250))
            .build()
            .get(&base)
            .send_blocking()
    })
    .await
    .unwrap()
    .unwrap_err();
    assert!(matches!(error, Error::Timeout(_)), "{error:?}");
}
