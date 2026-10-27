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

//! Minimal plain-HTTP client over `std::net`, loopback endpoints only: every
//! address the host resolves to must be loopback, and the connection goes to
//! those resolved addresses (no second lookup between the check and connect).
//! Sends HTTP/1.0 with `Connection: close` and decodes chunked bodies if the
//! server uses them anyway.

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

pub struct Response {
    pub status: u16,
    pub body: Box<dyn BufRead>,
}

fn split_url(url: &str) -> Result<(String, String), String> {
    let rest = url
        .strip_prefix("http://")
        .ok_or_else(|| format!("only http:// endpoints are supported: {url}"))?;
    let (host, path) = match rest.find('/') {
        Some(i) => (&rest[..i], &rest[i..]),
        None => (rest, "/"),
    };
    let host = if host.starts_with('[') && host.ends_with(']') {
        format!("{host}:80")
    } else if host.contains(':') {
        host.to_string()
    } else {
        format!("{host}:80")
    };
    Ok((host, path.to_string()))
}

/// Resolve `host:port` and refuse it unless every address is loopback.
fn loopback_addrs(host: &str) -> Result<Vec<SocketAddr>, String> {
    let addrs: Vec<SocketAddr> = host
        .to_socket_addrs()
        .map_err(|e| format!("resolve {host}: {e}"))?
        .collect();
    if addrs.is_empty() {
        return Err(format!("resolve {host}: no addresses"));
    }
    if let Some(a) = addrs.iter().find(|a| !a.ip().is_loopback()) {
        return Err(format!("refused: {host} resolves to non-loopback {a}"));
    }
    Ok(addrs)
}

struct Chunked<R: BufRead> {
    inner: R,
    left: usize,
    done: bool,
}

impl<R: BufRead> Read for Chunked<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.done {
            return Ok(0);
        }
        if self.left == 0 {
            let mut line = String::new();
            self.inner.read_line(&mut line)?;
            let size = usize::from_str_radix(line.trim().split(';').next().unwrap_or(""), 16)
                .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
            if size == 0 {
                self.done = true;
                return Ok(0);
            }
            self.left = size;
        }
        let n = buf.len().min(self.left);
        let got = self.inner.read(&mut buf[..n])?;
        self.left -= got;
        if self.left == 0 {
            let mut crlf = String::new();
            self.inner.read_line(&mut crlf)?;
        }
        Ok(got)
    }
}

/// POST a JSON body. `timeout` bounds each socket read and write.
pub fn post_json(url: &str, body: &str, timeout: Duration) -> Result<Response, String> {
    let (host, path) = split_url(url)?;
    let addrs = loopback_addrs(&host)?;
    let mut s = TcpStream::connect(&addrs[..]).map_err(|e| format!("connect {host}: {e}"))?;
    s.set_read_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    s.set_write_timeout(Some(timeout))
        .map_err(|e| e.to_string())?;
    let req = format!(
        "POST {path} HTTP/1.0\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes())
        .map_err(|e| format!("write: {e}"))?;
    let mut r = BufReader::new(s);
    let mut status_line = String::new();
    r.read_line(&mut status_line)
        .map_err(|e| format!("read: {e}"))?;
    let status: u16 = status_line
        .split_whitespace()
        .nth(1)
        .and_then(|c| c.parse().ok())
        .ok_or_else(|| format!("bad status line: {status_line:?}"))?;
    let mut chunked = false;
    loop {
        let mut h = String::new();
        let n = r.read_line(&mut h).map_err(|e| format!("read: {e}"))?;
        if n == 0 || h.trim().is_empty() {
            break;
        }
        let l = h.to_ascii_lowercase();
        if l.starts_with("transfer-encoding:") && l.contains("chunked") {
            chunked = true;
        }
    }
    let body: Box<dyn BufRead> = if chunked {
        Box::new(BufReader::new(Chunked {
            inner: r,
            left: 0,
            done: false,
        }))
    } else {
        Box::new(r)
    };
    Ok(Response { status, body })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loopback_hosts_are_accepted() {
        for h in ["127.0.0.1:18080", "127.8.9.10:1", "[::1]:1", "localhost:1"] {
            let a = loopback_addrs(h).unwrap_or_else(|e| panic!("{h}: {e}"));
            assert!(a.iter().all(|a| a.ip().is_loopback()), "{h}");
        }
        assert_eq!(split_url("http://[::1]/x").unwrap().0, "[::1]:80");
    }

    #[test]
    fn non_loopback_hosts_are_refused_before_connecting() {
        for url in [
            "http://10.0.0.1:18080/v1",
            "http://192.168.0.192/v1",
            "http://0.0.0.0:18080/v1",
            "http://[2001:db8::1]:80/v1",
            "http://203.0.113.7/v1",
        ] {
            let err = post_json(url, "{}", Duration::from_millis(50))
                .err()
                .unwrap_or_else(|| panic!("{url} was not refused"));
            assert!(err.starts_with("refused: "), "{url}: {err}");
        }
    }
}
