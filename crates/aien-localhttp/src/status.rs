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

use std::fmt;

/// A response status code (100 to 999).
///
/// `Display` prints `"<code> <reason>"`, the same text the `http` crate's
/// `StatusCode` prints (http-1.5.0 src/status.rs:219-227), so error messages
/// that format a status keep their wording after migrating from reqwest.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct StatusCode(u16);

impl StatusCode {
    /// Accepts 100 to 999, the range a three digit status line can carry.
    pub fn from_u16(code: u16) -> Option<StatusCode> {
        (100..=999).contains(&code).then_some(StatusCode(code))
    }

    pub fn as_u16(self) -> u16 {
        self.0
    }

    /// 200 to 299.
    pub fn is_success(self) -> bool {
        (200..300).contains(&self.0)
    }

    /// 100 to 199.
    pub fn is_informational(self) -> bool {
        (100..200).contains(&self.0)
    }

    /// The reason phrase table of the `http` crate (http-1.5.0 src/status.rs:329-524).
    pub fn canonical_reason(self) -> Option<&'static str> {
        Some(match self.0 {
            100 => "Continue",
            101 => "Switching Protocols",
            102 => "Processing",
            103 => "Early Hints",
            200 => "OK",
            201 => "Created",
            202 => "Accepted",
            203 => "Non Authoritative Information",
            204 => "No Content",
            205 => "Reset Content",
            206 => "Partial Content",
            207 => "Multi-Status",
            208 => "Already Reported",
            226 => "IM Used",
            300 => "Multiple Choices",
            301 => "Moved Permanently",
            302 => "Found",
            303 => "See Other",
            304 => "Not Modified",
            305 => "Use Proxy",
            307 => "Temporary Redirect",
            308 => "Permanent Redirect",
            400 => "Bad Request",
            401 => "Unauthorized",
            402 => "Payment Required",
            403 => "Forbidden",
            404 => "Not Found",
            405 => "Method Not Allowed",
            406 => "Not Acceptable",
            407 => "Proxy Authentication Required",
            408 => "Request Timeout",
            409 => "Conflict",
            410 => "Gone",
            411 => "Length Required",
            412 => "Precondition Failed",
            413 => "Payload Too Large",
            414 => "URI Too Long",
            415 => "Unsupported Media Type",
            416 => "Range Not Satisfiable",
            417 => "Expectation Failed",
            418 => "I'm a teapot",
            421 => "Misdirected Request",
            422 => "Unprocessable Entity",
            423 => "Locked",
            424 => "Failed Dependency",
            425 => "Too Early",
            426 => "Upgrade Required",
            428 => "Precondition Required",
            429 => "Too Many Requests",
            431 => "Request Header Fields Too Large",
            451 => "Unavailable For Legal Reasons",
            500 => "Internal Server Error",
            501 => "Not Implemented",
            502 => "Bad Gateway",
            503 => "Service Unavailable",
            504 => "Gateway Timeout",
            505 => "HTTP Version Not Supported",
            506 => "Variant Also Negotiates",
            507 => "Insufficient Storage",
            508 => "Loop Detected",
            510 => "Not Extended",
            511 => "Network Authentication Required",
            _ => return None,
        })
    }
}

impl fmt::Display for StatusCode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} {}",
            self.0,
            self.canonical_reason().unwrap_or("<unknown status code>")
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn display_matches_the_http_crate_wording() {
        assert_eq!(StatusCode(404).to_string(), "404 Not Found");
        assert_eq!(StatusCode(500).to_string(), "500 Internal Server Error");
        assert_eq!(StatusCode(299).to_string(), "299 <unknown status code>");
    }

    #[test]
    fn success_is_exactly_the_2xx_range() {
        assert!(!StatusCode(199).is_success());
        assert!(StatusCode(200).is_success());
        assert!(StatusCode(299).is_success());
        assert!(!StatusCode(300).is_success());
    }

    #[test]
    fn from_u16_rejects_codes_outside_three_digits() {
        assert!(StatusCode::from_u16(99).is_none());
        assert!(StatusCode::from_u16(1000).is_none());
        assert_eq!(StatusCode::from_u16(100).map(StatusCode::as_u16), Some(100));
    }
}
