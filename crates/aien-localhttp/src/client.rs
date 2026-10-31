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

use crate::response::{self, BlockingResponse, Deadline, Response};
use crate::status::StatusCode;
use crate::{Error, Url};
use serde::Serialize;
use std::fmt;
use std::sync::Arc;
use std::time::Duration;
use tokio::io::{AsyncWriteExt, BufReader};
use tokio::net::TcpStream;

/// Default cap on a response body: 16 MiB. Applies to the cumulative bytes
/// read from one response, including streamed lines.
pub const DEFAULT_MAX_BODY_BYTES: usize = 16 * 1024 * 1024;

/// Header names the client writes itself. A caller value would change the
/// message framing, so these are refused.
const RESERVED_HEADERS: [&str; 4] = ["host", "content-length", "transfer-encoding", "connection"];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    Get,
    Post,
}

impl Method {
    pub fn as_str(self) -> &'static str {
        match self {
            Method::Get => "GET",
            Method::Post => "POST",
        }
    }
}

#[derive(Debug)]
struct Config {
    connect_timeout: Option<Duration>,
    timeout: Option<Duration>,
    max_body_bytes: usize,
}

/// A cheap to clone handle holding timeouts and caps. It keeps no connections.
#[derive(Clone, Debug)]
pub struct Client {
    config: Arc<Config>,
}

impl Default for Client {
    fn default() -> Self {
        Client::new()
    }
}

impl Client {
    /// No timeouts, default body cap.
    pub fn new() -> Client {
        ClientBuilder::new().build()
    }

    pub fn builder() -> ClientBuilder {
        ClientBuilder::new()
    }

    pub fn get(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::Get, url)
    }

    pub fn post(&self, url: impl AsRef<str>) -> RequestBuilder {
        self.request(Method::Post, url)
    }

    pub fn request(&self, method: Method, url: impl AsRef<str>) -> RequestBuilder {
        RequestBuilder {
            client: self.clone(),
            method,
            url: Url::parse(url.as_ref()),
            headers: Vec::new(),
            body: None,
            timeout: None,
            error: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct ClientBuilder {
    connect_timeout: Option<Duration>,
    timeout: Option<Duration>,
    max_body_bytes: usize,
}

impl Default for ClientBuilder {
    fn default() -> Self {
        ClientBuilder::new()
    }
}

impl ClientBuilder {
    pub fn new() -> ClientBuilder {
        ClientBuilder {
            connect_timeout: None,
            timeout: None,
            max_body_bytes: DEFAULT_MAX_BODY_BYTES,
        }
    }

    /// Bound on name resolution plus the TCP connect.
    pub fn connect_timeout(mut self, after: Duration) -> ClientBuilder {
        self.connect_timeout = Some(after);
        self
    }

    /// Bound on the whole exchange: connect, request write, response head and
    /// every body read. A request can override it with [`RequestBuilder::timeout`].
    pub fn timeout(mut self, after: Duration) -> ClientBuilder {
        self.timeout = Some(after);
        self
    }

    pub fn max_body_bytes(mut self, limit: usize) -> ClientBuilder {
        self.max_body_bytes = limit;
        self
    }

    /// Infallible: there is no TLS backend or system state to load.
    pub fn build(self) -> Client {
        Client {
            config: Arc::new(Config {
                connect_timeout: self.connect_timeout,
                timeout: self.timeout,
                max_body_bytes: self.max_body_bytes,
            }),
        }
    }
}

/// One request being assembled. Builder errors (bad URL, bad header, JSON
/// encode failure) are held and returned by `send`.
pub struct RequestBuilder {
    client: Client,
    method: Method,
    url: Result<Url, Error>,
    headers: Vec<(String, String)>,
    body: Option<Vec<u8>>,
    timeout: Option<Duration>,
    error: Option<Error>,
}

impl fmt::Debug for RequestBuilder {
    // Header values are left out: they may carry a Bearer token.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("RequestBuilder")
            .field("method", &self.method)
            .field("url", &self.url.as_ref().map(|u| u.to_string()))
            .field(
                "headers",
                &self.headers.iter().map(|(n, _)| n).collect::<Vec<_>>(),
            )
            .field("body_len", &self.body.as_ref().map(Vec::len))
            .finish()
    }
}

fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

impl RequestBuilder {
    pub fn header(mut self, name: impl AsRef<str>, value: impl AsRef<str>) -> RequestBuilder {
        if self.error.is_some() {
            return self;
        }
        let (name, value) = (name.as_ref(), value.as_ref());
        let reason = if name.is_empty() || !name.bytes().all(is_token_byte) {
            Some("name must be a non-empty HTTP token")
        } else if RESERVED_HEADERS.contains(&name.to_ascii_lowercase().as_str()) {
            Some("the client writes this framing header itself")
        } else if value.bytes().any(|b| matches!(b, b'\r' | b'\n' | 0)) {
            Some("value must not contain CR, LF or NUL")
        } else {
            None
        };
        match reason {
            Some(reason) => {
                self.error = Some(Error::InvalidHeader {
                    name: name.to_string(),
                    reason,
                })
            }
            None => self
                .headers
                .push((name.to_string(), value.trim().to_string())),
        }
        self
    }

    /// `Authorization: Bearer <token>`.
    pub fn bearer_auth(self, token: impl fmt::Display) -> RequestBuilder {
        self.header("Authorization", format!("Bearer {token}"))
    }

    /// Serializes `value` as the body and sets `Content-Type: application/json`
    /// unless the caller already set a Content-Type (reqwest behaves the same).
    pub fn json<T: Serialize + ?Sized>(mut self, value: &T) -> RequestBuilder {
        match serde_json::to_vec(value) {
            Ok(bytes) => {
                if !self.has_header("content-type") {
                    self = self.header("Content-Type", "application/json");
                }
                self.body = Some(bytes);
            }
            Err(e) => {
                if self.error.is_none() {
                    self.error = Some(Error::Encode(e.to_string()));
                }
            }
        }
        self
    }

    pub fn body(mut self, bytes: impl Into<Vec<u8>>) -> RequestBuilder {
        self.body = Some(bytes.into());
        self
    }

    /// Total timeout for this request only; overrides the client's.
    pub fn timeout(mut self, after: Duration) -> RequestBuilder {
        self.timeout = Some(after);
        self
    }

    fn has_header(&self, lower: &str) -> bool {
        contains_header(&self.headers, lower)
    }

    /// Sends the request and returns once the response head is read. The body
    /// is read by the [`Response`] methods, still under the total timeout.
    pub async fn send(self) -> Result<Response, Error> {
        let RequestBuilder {
            client,
            method,
            url,
            headers,
            body,
            timeout,
            error,
        } = self;
        if let Some(e) = error {
            return Err(e);
        }
        let url = url?;
        let config = client.config.clone();
        let deadline = timeout.or(config.timeout).map(Deadline::after);
        let message = encode_message(method, &url, &headers, body.as_deref());
        response::bounded(deadline, async move {
            let mut stream = connect(&url, config.connect_timeout).await?;
            stream.write_all(&message).await?;
            stream.flush().await?;
            let mut reader = BufReader::new(stream);
            let head = response::read_head(&mut reader).await?;
            Ok(Response::new(head, reader, deadline, config.max_body_bytes))
        })
        .await
    }

    /// Blocking form for sync callers. Runs the request on a fresh thread with
    /// its own current-thread runtime, so it is safe to call from inside or
    /// outside a tokio runtime. Reads the whole body under the body cap.
    pub fn send_blocking(self) -> Result<BlockingResponse, Error> {
        run_blocking(async move {
            let response = self.send().await?;
            let status = response.status();
            let headers = response.headers().to_vec();
            let body = response.bytes().await?;
            Ok(BlockingResponse::new(status, headers, body))
        })
    }

    /// Blocking form that returns once the status line and headers are read.
    /// The body is dropped unread, so a slow, stalled or truncated body cannot
    /// turn a decided status into a transport error.
    pub fn send_blocking_status(self) -> Result<StatusCode, Error> {
        run_blocking(async move { Ok(self.send().await?.status()) })
    }
}

/// Runs `request` on its own thread and current-thread runtime. The runtime is
/// shut down in the background, so blocking work still pending when the
/// request finishes or times out (a stalled name lookup) cannot hold the
/// caller past the request deadline.
fn run_blocking<T, F>(request: F) -> Result<T, Error>
where
    T: Send + 'static,
    F: std::future::Future<Output = Result<T, Error>> + Send + 'static,
{
    let worker = std::thread::Builder::new()
        .name("aien-localhttp-blocking".to_string())
        .spawn(move || {
            let runtime = tokio::runtime::Builder::new_current_thread()
                .enable_io()
                .enable_time()
                .build()
                .map_err(|e| Error::Blocking(format!("runtime: {e}")))?;
            let result = runtime.block_on(request);
            runtime.shutdown_background();
            result
        })
        .map_err(|e| Error::Blocking(format!("thread spawn: {e}")))?;
    worker
        .join()
        .map_err(|_| Error::Blocking("request thread panicked".to_string()))?
}

fn contains_header(headers: &[(String, String)], lower: &str) -> bool {
    headers.iter().any(|(n, _)| n.eq_ignore_ascii_case(lower))
}

/// Request line, headers and body as one buffer.
fn encode_message(
    method: Method,
    url: &Url,
    headers: &[(String, String)],
    body: Option<&[u8]>,
) -> Vec<u8> {
    let mut head = format!(
        "{} {} HTTP/1.1\r\nHost: {}\r\nConnection: close\r\n",
        method.as_str(),
        url.target(),
        url.authority()
    );
    if !contains_header(headers, "accept") {
        head.push_str("Accept: */*\r\n");
    }
    for (name, value) in headers {
        head.push_str(name);
        head.push_str(": ");
        head.push_str(value);
        head.push_str("\r\n");
    }
    match (body, method) {
        (Some(body), _) => head.push_str(&format!("Content-Length: {}\r\n", body.len())),
        (None, Method::Post) => head.push_str("Content-Length: 0\r\n"),
        (None, Method::Get) => {}
    }
    head.push_str("\r\n");
    let mut message = head.into_bytes();
    if let Some(body) = body {
        message.extend_from_slice(body);
    }
    message
}

async fn connect(url: &Url, connect_timeout: Option<Duration>) -> Result<TcpStream, Error> {
    let addr = if url.host().contains(':') {
        format!("[{}]:{}", url.host(), url.port())
    } else {
        format!("{}:{}", url.host(), url.port())
    };
    let attempt = async {
        let addrs = tokio::net::lookup_host((url.host(), url.port()))
            .await
            .map_err(|source| Error::Connect {
                addr: addr.clone(),
                source,
            })?;
        let mut last = None;
        for candidate in addrs {
            match TcpStream::connect(candidate).await {
                Ok(stream) => {
                    // Small request and response messages: send them without delay.
                    let _ = stream.set_nodelay(true);
                    return Ok(stream);
                }
                Err(e) => last = Some(e),
            }
        }
        Err(Error::Connect {
            addr: addr.clone(),
            source: last.unwrap_or_else(|| {
                std::io::Error::new(std::io::ErrorKind::NotFound, "no address resolved")
            }),
        })
    };
    match connect_timeout {
        None => attempt.await,
        Some(after) => match tokio::time::timeout(after, attempt).await {
            Ok(result) => result,
            Err(_) => Err(Error::ConnectTimeout { addr, after }),
        },
    }
}
