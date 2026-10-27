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

use crate::Error;
use std::fmt;

/// A parsed `http://` URL: host, port and request target.
///
/// Only the `http` scheme parses. `https` fails with [`Error::HttpsRefused`],
/// every other scheme with [`Error::UnsupportedScheme`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Url {
    /// Host without IPv6 brackets, lowercased.
    host: String,
    port: u16,
    /// `host[:port]` as sent in the `Host` header (IPv6 keeps its brackets).
    authority: String,
    /// Path plus query, always starting with `/`. The fragment is dropped.
    target: String,
}

impl Url {
    pub fn parse(input: &str) -> Result<Url, Error> {
        let invalid = |reason: &'static str| Error::InvalidUrl {
            url: input.to_string(),
            reason,
        };
        let text = input.trim();
        let Some(sep) = text.find("://") else {
            return Err(invalid("missing scheme"));
        };
        match text[..sep].to_ascii_lowercase().as_str() {
            "http" => {}
            "https" => {
                return Err(Error::HttpsRefused {
                    url: input.to_string(),
                })
            }
            _ => {
                return Err(Error::UnsupportedScheme {
                    url: input.to_string(),
                })
            }
        }
        let rest = &text[sep + 3..];
        let rest = rest.split('#').next().unwrap_or("");
        let (authority, target) = match rest.find(['/', '?']) {
            Some(i) => (&rest[..i], &rest[i..]),
            None => (rest, ""),
        };
        if target.bytes().any(|b| b <= b' ' || b == 0x7f) {
            return Err(invalid("space or control character in path or query"));
        }
        let target = if target.is_empty() {
            "/".to_string()
        } else if target.starts_with('?') {
            format!("/{target}")
        } else {
            target.to_string()
        };
        if authority.is_empty() {
            return Err(invalid("missing host"));
        }
        if authority.contains('@') {
            return Err(invalid("credentials in the URL are not supported"));
        }
        let authority = authority.to_ascii_lowercase();
        let (host, port_text) = if let Some(after_bracket) = authority.strip_prefix('[') {
            let Some(end) = after_bracket.find(']') else {
                return Err(invalid("unclosed IPv6 bracket"));
            };
            let host = &after_bracket[..end];
            let tail = &after_bracket[end + 1..];
            if !host
                .bytes()
                .all(|b| b.is_ascii_hexdigit() || b == b':' || b == b'.')
            {
                return Err(invalid("invalid IPv6 host"));
            }
            match tail.strip_prefix(':') {
                Some(port) => (host.to_string(), Some(port)),
                None if tail.is_empty() => (host.to_string(), None),
                None => return Err(invalid("unexpected text after IPv6 host")),
            }
        } else {
            let (host, port) = match authority.rsplit_once(':') {
                Some((h, p)) => (h, Some(p)),
                None => (authority.as_str(), None),
            };
            if !host
                .bytes()
                .all(|b| b.is_ascii_alphanumeric() || matches!(b, b'-' | b'.' | b'_'))
            {
                return Err(invalid("invalid host"));
            }
            (host.to_string(), port)
        };
        if host.is_empty() {
            return Err(invalid("missing host"));
        }
        let port = match port_text {
            None => 80,
            Some(p) if !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()) => {
                match p.parse::<u16>() {
                    Ok(0) | Err(_) => return Err(invalid("port out of range")),
                    Ok(n) => n,
                }
            }
            Some(_) => return Err(invalid("invalid port")),
        };
        Ok(Url {
            host,
            port,
            authority,
            target,
        })
    }

    pub fn host(&self) -> &str {
        &self.host
    }

    pub fn port(&self) -> u16 {
        self.port
    }

    /// The `Host` header value.
    pub fn authority(&self) -> &str {
        &self.authority
    }

    /// Path plus query as written on the request line.
    pub fn target(&self) -> &str {
        &self.target
    }
}

impl fmt::Display for Url {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "http://{}{}", self.authority, self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_host_port_path_and_query() {
        let u = Url::parse("http://127.0.0.1:18080/api/cortex/recall?q=a&limit=5").unwrap();
        assert_eq!(u.host(), "127.0.0.1");
        assert_eq!(u.port(), 18080);
        assert_eq!(u.authority(), "127.0.0.1:18080");
        assert_eq!(u.target(), "/api/cortex/recall?q=a&limit=5");
    }

    #[test]
    fn defaults_port_80_and_root_path() {
        let u = Url::parse("http://localhost").unwrap();
        assert_eq!(u.port(), 80);
        assert_eq!(u.target(), "/");
        assert_eq!(u.authority(), "localhost");
        let q = Url::parse("http://localhost?x=1").unwrap();
        assert_eq!(q.target(), "/?x=1");
    }

    #[test]
    fn scheme_is_case_insensitive_and_fragment_dropped() {
        let u = Url::parse("HTTP://Example.LAN:8080/a#frag").unwrap();
        assert_eq!(u.host(), "example.lan");
        assert_eq!(u.target(), "/a");
    }

    #[test]
    fn ipv6_literal_keeps_brackets_in_authority() {
        let u = Url::parse("http://[::1]:9000/h").unwrap();
        assert_eq!(u.host(), "::1");
        assert_eq!(u.authority(), "[::1]:9000");
        assert_eq!(u.port(), 9000);
    }

    #[test]
    fn https_is_refused_in_any_case() {
        for s in [
            "https://127.0.0.1/",
            "HTTPS://localhost:443/x",
            " https://a/ ",
        ] {
            assert!(
                matches!(Url::parse(s), Err(Error::HttpsRefused { .. })),
                "{s}"
            );
        }
    }

    #[test]
    fn other_schemes_and_bad_urls_are_refused() {
        assert!(matches!(
            Url::parse("ftp://host/"),
            Err(Error::UnsupportedScheme { .. })
        ));
        for bad in [
            "127.0.0.1:80/x",
            "http://",
            "http:///path",
            "http://user:pw@host/",
            "http://host:0/",
            "http://host:65536/",
            "http://host:/",
            "http://host:abc/",
            "http://ho st/",
            "http://host/a b",
            "http://host/a\r\nX: y",
            "http://[::1/",
            "http://[::1]x/",
        ] {
            assert!(
                matches!(Url::parse(bad), Err(Error::InvalidUrl { .. })),
                "{bad} should be InvalidUrl, got {:?}",
                Url::parse(bad)
            );
        }
    }
}
