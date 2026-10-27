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

use std::time::Duration;

/// Every way a request can fail. Each variant names one cause so callers and
/// tests can tell them apart.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The URL uses `https://`. This client never speaks TLS and never
    /// downgrades, so the request is refused before any connection is made.
    #[error(
        "https is refused for {url}: aien-localhttp speaks plain http to local services only and never downgrades; use the vetted TLS client for https"
    )]
    HttpsRefused { url: String },

    /// The URL uses a scheme other than `http` or `https`.
    #[error("unsupported URL scheme in {url}: only http:// is accepted")]
    UnsupportedScheme { url: String },

    /// The URL could not be parsed.
    #[error("invalid URL {url:?}: {reason}")]
    InvalidUrl { url: String, reason: &'static str },

    /// A request header name or value was rejected.
    #[error("invalid request header {name:?}: {reason}")]
    InvalidHeader { name: String, reason: &'static str },

    /// The JSON request body could not be encoded.
    #[error("could not encode the JSON request body: {0}")]
    Encode(String),

    /// Name resolution or the TCP connect failed.
    #[error("connect to {addr} failed: {source}")]
    Connect {
        addr: String,
        #[source]
        source: std::io::Error,
    },

    /// The TCP connect did not finish within the connect timeout.
    #[error("connect to {addr} timed out after {after:?}")]
    ConnectTimeout { addr: String, after: Duration },

    /// The total timeout ran out (request, response head or body).
    #[error("request timed out after {0:?}")]
    Timeout(Duration),

    /// A read or write on the connection failed.
    #[error("i/o error: {0}")]
    Io(#[from] std::io::Error),

    /// The response status line or headers break HTTP/1.1 framing rules.
    #[error("malformed response: {0}")]
    MalformedResponse(String),

    /// A chunk size line or chunk terminator is invalid.
    #[error("malformed chunked body: {0}")]
    MalformedChunk(String),

    /// The server closed the connection before the response was complete.
    #[error("connection closed early: {0}")]
    EarlyClose(String),

    /// The status line plus headers exceed the head size cap.
    #[error("response head exceeds the {limit} byte cap")]
    HeadTooLarge { limit: usize },

    /// The body exceeds the body size cap.
    #[error("response body exceeds the {limit} byte cap")]
    BodyTooLarge { limit: usize },

    /// The response body is not valid JSON for the requested type.
    #[error("could not decode the JSON response body: {0}")]
    Decode(String),

    /// The blocking entry point could not run its private runtime.
    #[error("blocking request failed: {0}")]
    Blocking(String),
}

impl Error {
    /// True for the connect timeout and the total timeout.
    pub fn is_timeout(&self) -> bool {
        matches!(self, Error::Timeout(_) | Error::ConnectTimeout { .. })
    }

    /// True when no connection could be made (resolution, refusal, connect timeout).
    pub fn is_connect(&self) -> bool {
        matches!(self, Error::Connect { .. } | Error::ConnectTimeout { .. })
    }
}
