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

//! aien-localhttp: a small native HTTP/1.1 client for plain `http://` calls to
//! local services (Cortex, the model seat, AEGIS, health probes).
//!
//! What it does:
//! - GET and POST over a tokio `TcpStream`, one connection per request
//!   (`Connection: close`, no pooling, no proxies, no redirects).
//! - Request headers (Bearer, Content-Type), JSON bodies, `Content-Length`.
//! - Response bodies framed by `Content-Length`, `Transfer-Encoding: chunked`
//!   or connection close, with a body size cap and a response head size cap.
//! - A line stream over the body for server-sent events ([`Response::next_line`]).
//! - Connect timeout and a total timeout that covers the whole exchange,
//!   including reading the body.
//! - A blocking entry point for sync callers ([`RequestBuilder::send_blocking`]).
//!
//! What it refuses:
//! - `https://` URLs fail with [`Error::HttpsRefused`]. The client never
//!   downgrades https to http and carries no crypto. External https stays on
//!   the vetted TLS stack used elsewhere in the workspace.
//! - Any scheme other than `http`, credentials in the URL, header values with
//!   CR, LF or NUL, and caller-supplied framing headers (`Host`,
//!   `Content-Length`, `Transfer-Encoding`, `Connection`).

mod client;
mod error;
mod response;
mod status;
mod url;

pub use client::{Client, ClientBuilder, Method, RequestBuilder, DEFAULT_MAX_BODY_BYTES};
pub use error::Error;
pub use response::{BlockingResponse, Response, MAX_HEAD_BYTES};
pub use status::StatusCode;
pub use url::Url;

#[cfg(feature = "test-server")]
pub mod test_server;
