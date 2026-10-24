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

use crate::{Error, StatusCode};
use serde::de::DeserializeOwned;
use std::future::Future;
use std::time::Duration;
use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::net::TcpStream;

/// Cap on the status line plus all headers of one response: 64 KiB.
pub const MAX_HEAD_BYTES: usize = 64 * 1024;
/// Cap on the number of header lines (and on chunked trailer lines).
const MAX_HEADER_LINES: usize = 128;
/// Cap on one chunk size line, extensions included.
const MAX_CHUNK_LINE_BYTES: usize = 4096;
/// Largest piece handed out by one body read.
const READ_PIECE: usize = 64 * 1024;

/// The point in time the total timeout runs out, and its length for errors.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Deadline {
    at: tokio::time::Instant,
    total: Duration,
}

impl Deadline {
    pub(crate) fn after(total: Duration) -> Deadline {
        Deadline {
            at: tokio::time::Instant::now() + total,
            total,
        }
    }
}

/// Runs `work` under the deadline, if there is one.
pub(crate) async fn bounded<T, F>(deadline: Option<Deadline>, work: F) -> Result<T, Error>
where
    F: Future<Output = Result<T, Error>>,
{
    match deadline {
        None => work.await,
        Some(d) => match tokio::time::timeout_at(d.at, work).await {
            Ok(result) => result,
            Err(_) => Err(Error::Timeout(d.total)),
        },
    }
}

pub(crate) struct Head {
    status: StatusCode,
    headers: Vec<(String, String)>,
    framing: Framing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Framing {
    /// `Content-Length`: `remaining` of `total` bytes still to read.
    Length { remaining: u64, total: u64 },
    /// `Transfer-Encoding: chunked`.
    Chunked(ChunkState),
    /// No length: the body ends when the server closes the connection.
    UntilClose,
    /// No body left (fully read, or a status that has none).
    Done,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ChunkState {
    /// Next comes a chunk size line.
    Size,
    /// Inside a chunk with this many data bytes left.
    Data { remaining: u64 },
    /// The chunk data ended; the CRLF after it is next.
    DataEnd,
}

enum Line {
    Bytes(Vec<u8>),
    Eof,
    TooLong,
}

/// Reads one line ending in LF, at most `cap` bytes including the LF. The
/// returned bytes have the LF and an optional CR before it removed. EOF in the
/// middle of a line returns `Eof`.
async fn read_line_capped(
    reader: &mut BufReader<TcpStream>,
    cap: usize,
) -> Result<Line, std::io::Error> {
    let mut line = Vec::new();
    loop {
        let available = reader.fill_buf().await?;
        if available.is_empty() {
            return Ok(Line::Eof);
        }
        let (take, found) = match available.iter().position(|&b| b == b'\n') {
            Some(i) => (i + 1, true),
            None => (available.len(), false),
        };
        if line.len() + take > cap {
            return Ok(Line::TooLong);
        }
        line.extend_from_slice(&available[..take]);
        reader.consume(take);
        if found {
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            return Ok(Line::Bytes(line));
        }
    }
}

fn is_token_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&b)
}

fn parse_status_line(line: &[u8]) -> Result<StatusCode, Error> {
    let bad = || {
        Error::MalformedResponse(format!(
            "bad status line {:?}",
            String::from_utf8_lossy(&line[..line.len().min(80)])
        ))
    };
    let rest = line
        .strip_prefix(b"HTTP/1.1 ")
        .or_else(|| line.strip_prefix(b"HTTP/1.0 "))
        .ok_or_else(bad)?;
    if rest.len() < 3 || !rest[..3].iter().all(u8::is_ascii_digit) {
        return Err(bad());
    }
    if rest.len() > 3 && rest[3] != b' ' {
        return Err(bad());
    }
    let code = u16::from(rest[0] - b'0') * 100
        + u16::from(rest[1] - b'0') * 10
        + u16::from(rest[2] - b'0');
    StatusCode::from_u16(code).ok_or_else(bad)
}

/// One line of the response head, counted against [`MAX_HEAD_BYTES`].
async fn head_line(
    reader: &mut BufReader<TcpStream>,
    used: &mut usize,
    on_eof: &'static str,
) -> Result<Vec<u8>, Error> {
    let too_large = Error::HeadTooLarge {
        limit: MAX_HEAD_BYTES,
    };
    match read_line_capped(reader, MAX_HEAD_BYTES - *used).await? {
        Line::Bytes(b) => {
            // Count the line terminator too (at most CRLF).
            *used += b.len() + 2;
            if *used > MAX_HEAD_BYTES {
                return Err(too_large);
            }
            Ok(b)
        }
        Line::Eof => Err(Error::EarlyClose(on_eof.to_string())),
        Line::TooLong => Err(too_large),
    }
}

/// Reads the status line and headers, skipping interim 1xx responses, and
/// decides how the body is framed (RFC 9112 section 6.3, strict client rules).
pub(crate) async fn read_head(reader: &mut BufReader<TcpStream>) -> Result<Head, Error> {
    let mut used = 0usize;
    loop {
        let status_line = head_line(
            reader,
            &mut used,
            "connection closed before the status line",
        )
        .await?;
        let status = parse_status_line(&status_line)?;
        let mut headers = Vec::new();
        loop {
            let line = head_line(
                reader,
                &mut used,
                "connection closed inside the response head",
            )
            .await?;
            if line.is_empty() {
                break;
            }
            if headers.len() == MAX_HEADER_LINES {
                return Err(Error::MalformedResponse(format!(
                    "more than {MAX_HEADER_LINES} header lines"
                )));
            }
            if matches!(line[0], b' ' | b'\t') {
                return Err(Error::MalformedResponse(
                    "obsolete header line folding".to_string(),
                ));
            }
            let Some(colon) = line.iter().position(|&b| b == b':') else {
                return Err(Error::MalformedResponse(
                    "header line without a colon".to_string(),
                ));
            };
            let name = &line[..colon];
            if name.is_empty() || !name.iter().copied().all(is_token_byte) {
                return Err(Error::MalformedResponse("invalid header name".to_string()));
            }
            let value = String::from_utf8_lossy(&line[colon + 1..])
                .trim_matches([' ', '\t'])
                .to_string();
            headers.push((String::from_utf8_lossy(name).into_owned(), value));
        }
        if status.as_u16() == 101 {
            return Err(Error::MalformedResponse(
                "unexpected 101 Switching Protocols".to_string(),
            ));
        }
        if status.is_informational() {
            continue;
        }
        let framing = framing_for(status, &headers)?;
        return Ok(Head {
            status,
            headers,
            framing,
        });
    }
}

fn header_values<'a>(
    headers: &'a [(String, String)],
    name: &'a str,
) -> impl Iterator<Item = &'a str> + 'a {
    headers
        .iter()
        .filter(move |(n, _)| n.eq_ignore_ascii_case(name))
        .flat_map(|(_, v)| v.split(','))
        .map(|v| v.trim_matches([' ', '\t']))
}

fn framing_for(status: StatusCode, headers: &[(String, String)]) -> Result<Framing, Error> {
    let code = status.as_u16();
    if code == 204 || code == 304 {
        return Ok(Framing::Done);
    }
    // Presence decides, not content: an empty Transfer-Encoding still counts,
    // so it cannot hide behind a Content-Length (refused as ambiguous) or pass
    // as length framing (refused as an unsupported coding).
    let has_coding = headers
        .iter()
        .any(|(n, _)| n.eq_ignore_ascii_case("transfer-encoding"));
    let codings: Vec<String> = header_values(headers, "transfer-encoding")
        .map(str::to_ascii_lowercase)
        .collect();
    let lengths: Vec<&str> = header_values(headers, "content-length").collect();
    if has_coding {
        if !lengths.is_empty() {
            return Err(Error::MalformedResponse(
                "both Transfer-Encoding and Content-Length".to_string(),
            ));
        }
        if codings != ["chunked"] {
            return Err(Error::MalformedResponse(format!(
                "unsupported transfer coding {:?}",
                codings.join(", ")
            )));
        }
        return Ok(Framing::Chunked(ChunkState::Size));
    }
    if let Some(first) = lengths.first() {
        if lengths.iter().any(|v| v != first) {
            return Err(Error::MalformedResponse(
                "conflicting Content-Length values".to_string(),
            ));
        }
        if first.is_empty() || !first.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::MalformedResponse(format!(
                "invalid Content-Length {first:?}"
            )));
        }
        let total: u64 = first.parse().map_err(|_| {
            Error::MalformedResponse(format!("Content-Length {first:?} out of range"))
        })?;
        return Ok(Framing::Length {
            remaining: total,
            total,
        });
    }
    Ok(Framing::UntilClose)
}

fn parse_chunk_size(line: &[u8]) -> Result<u64, Error> {
    let size = line.split(|&b| b == b';').next().unwrap_or(&[]);
    let size = String::from_utf8_lossy(size);
    let size = size.trim_matches([' ', '\t']);
    if size.is_empty() || size.len() > 16 || !size.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err(Error::MalformedChunk(format!(
            "bad chunk size line {:?}",
            String::from_utf8_lossy(&line[..line.len().min(40)])
        )));
    }
    u64::from_str_radix(size, 16)
        .map_err(|_| Error::MalformedChunk(format!("chunk size {size:?} out of range")))
}

/// A response whose head has been read. The body is read through it, still
/// under the request's total timeout and the body size cap.
pub struct Response {
    status: StatusCode,
    headers: Vec<(String, String)>,
    reader: BufReader<TcpStream>,
    framing: Framing,
    deadline: Option<Deadline>,
    max_body: usize,
    delivered: usize,
    pending: Vec<u8>,
}

impl std::fmt::Debug for Response {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Response")
            .field("status", &self.status)
            .field("headers", &self.headers)
            .finish()
    }
}

impl Response {
    pub(crate) fn new(
        head: Head,
        reader: BufReader<TcpStream>,
        deadline: Option<Deadline>,
        max_body: usize,
    ) -> Response {
        Response {
            status: head.status,
            headers: head.headers,
            reader,
            framing: head.framing,
            deadline,
            max_body,
            delivered: 0,
            pending: Vec::new(),
        }
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    /// All response headers in arrival order.
    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    /// First value of a header, name compared without case.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    /// The declared `Content-Length`, when the body is length framed.
    pub fn content_length(&self) -> Option<u64> {
        match self.framing {
            Framing::Length { total, .. } => Some(total),
            _ => None,
        }
    }

    fn account(&mut self, n: usize) -> Result<(), Error> {
        self.delivered += n;
        if self.delivered > self.max_body {
            return Err(Error::BodyTooLarge {
                limit: self.max_body,
            });
        }
        Ok(())
    }

    async fn read_piece(&mut self, max: usize) -> Result<Vec<u8>, Error> {
        let available = self.reader.fill_buf().await?;
        let n = available.len().min(max);
        let piece = available[..n].to_vec();
        self.reader.consume(n);
        Ok(piece)
    }

    async fn next_piece(&mut self) -> Result<Option<Vec<u8>>, Error> {
        loop {
            match self.framing {
                Framing::Done => return Ok(None),
                Framing::Length { remaining: 0, .. } => {
                    self.framing = Framing::Done;
                    return Ok(None);
                }
                Framing::Length { remaining, total } => {
                    if total > self.max_body as u64 {
                        return Err(Error::BodyTooLarge {
                            limit: self.max_body,
                        });
                    }
                    let want = remaining.min(READ_PIECE as u64) as usize;
                    let piece = self.read_piece(want).await?;
                    if piece.is_empty() {
                        return Err(Error::EarlyClose(format!(
                            "body ended after {} of {} bytes",
                            total - remaining,
                            total
                        )));
                    }
                    self.framing = Framing::Length {
                        remaining: remaining - piece.len() as u64,
                        total,
                    };
                    self.account(piece.len())?;
                    return Ok(Some(piece));
                }
                Framing::UntilClose => {
                    let piece = self.read_piece(READ_PIECE).await?;
                    if piece.is_empty() {
                        self.framing = Framing::Done;
                        return Ok(None);
                    }
                    self.account(piece.len())?;
                    return Ok(Some(piece));
                }
                Framing::Chunked(ChunkState::Size) => {
                    let line =
                        match read_line_capped(&mut self.reader, MAX_CHUNK_LINE_BYTES).await? {
                            Line::Bytes(b) => b,
                            Line::Eof => {
                                return Err(Error::EarlyClose(
                                    "connection closed before a chunk size line".to_string(),
                                ))
                            }
                            Line::TooLong => {
                                return Err(Error::MalformedChunk(format!(
                                    "chunk size line longer than {MAX_CHUNK_LINE_BYTES} bytes"
                                )))
                            }
                        };
                    let size = parse_chunk_size(&line)?;
                    if size == 0 {
                        self.read_trailers().await?;
                        self.framing = Framing::Done;
                        return Ok(None);
                    }
                    if size > (self.max_body - self.delivered.min(self.max_body)) as u64 {
                        return Err(Error::BodyTooLarge {
                            limit: self.max_body,
                        });
                    }
                    self.framing = Framing::Chunked(ChunkState::Data { remaining: size });
                }
                Framing::Chunked(ChunkState::Data { remaining }) => {
                    let want = remaining.min(READ_PIECE as u64) as usize;
                    let piece = self.read_piece(want).await?;
                    if piece.is_empty() {
                        return Err(Error::EarlyClose(
                            "connection closed inside a chunk".to_string(),
                        ));
                    }
                    let left = remaining - piece.len() as u64;
                    self.framing = Framing::Chunked(if left == 0 {
                        ChunkState::DataEnd
                    } else {
                        ChunkState::Data { remaining: left }
                    });
                    self.account(piece.len())?;
                    return Ok(Some(piece));
                }
                Framing::Chunked(ChunkState::DataEnd) => {
                    let mut crlf = [0u8; 2];
                    match self.reader.read_exact(&mut crlf).await {
                        Ok(_) => {}
                        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
                            return Err(Error::EarlyClose(
                                "connection closed after chunk data".to_string(),
                            ))
                        }
                        Err(e) => return Err(Error::Io(e)),
                    }
                    if &crlf != b"\r\n" {
                        return Err(Error::MalformedChunk(
                            "chunk data not followed by CRLF".to_string(),
                        ));
                    }
                    self.framing = Framing::Chunked(ChunkState::Size);
                }
            }
        }
    }

    async fn read_trailers(&mut self) -> Result<(), Error> {
        for _ in 0..=MAX_HEADER_LINES {
            match read_line_capped(&mut self.reader, MAX_HEAD_BYTES).await? {
                Line::Bytes(b) if b.is_empty() => return Ok(()),
                Line::Bytes(_) => {}
                Line::Eof => {
                    return Err(Error::EarlyClose(
                        "connection closed inside the chunked trailer".to_string(),
                    ))
                }
                Line::TooLong => {
                    return Err(Error::MalformedChunk("trailer line too long".to_string()))
                }
            }
        }
        Err(Error::MalformedChunk(format!(
            "more than {MAX_HEADER_LINES} trailer lines"
        )))
    }

    /// The next piece of the decoded body, `None` at the end. Pieces follow
    /// network arrival, so a streaming server's events show up as they come.
    pub async fn chunk(&mut self) -> Result<Option<Vec<u8>>, Error> {
        if !self.pending.is_empty() {
            return Ok(Some(std::mem::take(&mut self.pending)));
        }
        let deadline = self.deadline;
        bounded(deadline, self.next_piece()).await
    }

    /// The next body line without its LF (and CR), `None` at the end. A last
    /// line without a terminator is still returned. Meant for server-sent
    /// event streams (`data: ...` lines). Invalid UTF-8 is replaced.
    pub async fn next_line(&mut self) -> Result<Option<String>, Error> {
        loop {
            if let Some(i) = self.pending.iter().position(|&b| b == b'\n') {
                let mut line: Vec<u8> = self.pending.drain(..=i).collect();
                line.pop();
                if line.last() == Some(&b'\r') {
                    line.pop();
                }
                return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
            }
            let deadline = self.deadline;
            match bounded(deadline, self.next_piece()).await? {
                Some(piece) => self.pending.extend_from_slice(&piece),
                None if self.pending.is_empty() => return Ok(None),
                None => {
                    let mut line = std::mem::take(&mut self.pending);
                    if line.last() == Some(&b'\r') {
                        line.pop();
                    }
                    return Ok(Some(String::from_utf8_lossy(&line).into_owned()));
                }
            }
        }
    }

    /// The whole body, under the body cap.
    pub async fn bytes(mut self) -> Result<Vec<u8>, Error> {
        let mut body = std::mem::take(&mut self.pending);
        while let Some(piece) = self.chunk().await? {
            body.extend_from_slice(&piece);
        }
        Ok(body)
    }

    /// The whole body as text. Invalid UTF-8 is replaced, as reqwest does.
    pub async fn text(self) -> Result<String, Error> {
        let body = self.bytes().await?;
        Ok(String::from_utf8_lossy(&body).into_owned())
    }

    /// The whole body decoded as JSON.
    pub async fn json<T: DeserializeOwned>(self) -> Result<T, Error> {
        let body = self.bytes().await?;
        serde_json::from_slice(&body).map_err(|e| Error::Decode(e.to_string()))
    }
}

/// A fully read response from [`crate::RequestBuilder::send_blocking`].
#[derive(Clone, Debug)]
pub struct BlockingResponse {
    status: StatusCode,
    headers: Vec<(String, String)>,
    body: Vec<u8>,
}

impl BlockingResponse {
    pub(crate) fn new(
        status: StatusCode,
        headers: Vec<(String, String)>,
        body: Vec<u8>,
    ) -> BlockingResponse {
        BlockingResponse {
            status,
            headers,
            body,
        }
    }

    pub fn status(&self) -> StatusCode {
        self.status
    }

    pub fn headers(&self) -> &[(String, String)] {
        &self.headers
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(n, _)| n.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn bytes(&self) -> &[u8] {
        &self.body
    }

    pub fn text(&self) -> String {
        String::from_utf8_lossy(&self.body).into_owned()
    }

    pub fn json<T: DeserializeOwned>(&self) -> Result<T, Error> {
        serde_json::from_slice(&self.body).map_err(|e| Error::Decode(e.to_string()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hv(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
        pairs
            .iter()
            .map(|(n, v)| (n.to_string(), v.to_string()))
            .collect()
    }

    fn ok() -> StatusCode {
        StatusCode::from_u16(200).unwrap()
    }

    #[test]
    fn status_line_parsing() {
        assert_eq!(parse_status_line(b"HTTP/1.1 200 OK").unwrap().as_u16(), 200);
        assert_eq!(parse_status_line(b"HTTP/1.0 404").unwrap().as_u16(), 404);
        assert_eq!(
            parse_status_line(b"HTTP/1.1 503 Service Unavailable")
                .unwrap()
                .as_u16(),
            503
        );
        for bad in [
            &b"HTTP/2 200 OK"[..],
            b"HTTP/1.1 20 OK",
            b"HTTP/1.1 2000 OK",
            b"HTTP/1.1 abc OK",
            b"HTTP/1.1 099 x",
            b"",
            b"garbage",
        ] {
            assert!(parse_status_line(bad).is_err(), "{bad:?}");
        }
    }

    #[test]
    fn framing_rules() {
        assert_eq!(
            framing_for(ok(), &hv(&[("Content-Length", "5")])).unwrap(),
            Framing::Length {
                remaining: 5,
                total: 5
            }
        );
        assert_eq!(
            framing_for(
                ok(),
                &hv(&[("content-length", "5"), ("Content-Length", "5")])
            )
            .unwrap(),
            Framing::Length {
                remaining: 5,
                total: 5
            }
        );
        assert_eq!(
            framing_for(ok(), &hv(&[("Transfer-Encoding", "Chunked")])).unwrap(),
            Framing::Chunked(ChunkState::Size)
        );
        assert_eq!(framing_for(ok(), &[]).unwrap(), Framing::UntilClose);
        assert_eq!(
            framing_for(
                StatusCode::from_u16(204).unwrap(),
                &hv(&[("Content-Length", "9")])
            )
            .unwrap(),
            Framing::Done
        );
        assert_eq!(
            framing_for(StatusCode::from_u16(304).unwrap(), &[]).unwrap(),
            Framing::Done
        );
    }

    #[test]
    fn framing_refusals() {
        for headers in [
            hv(&[("Content-Length", "5"), ("Content-Length", "6")]),
            hv(&[("Content-Length", "5, 6")]),
            hv(&[("Content-Length", "-1")]),
            hv(&[("Content-Length", "")]),
            hv(&[("Content-Length", "99999999999999999999999")]),
            hv(&[("Transfer-Encoding", "chunked"), ("Content-Length", "5")]),
            hv(&[("Transfer-Encoding", "gzip, chunked")]),
            hv(&[("Transfer-Encoding", "gzip")]),
        ] {
            assert!(
                matches!(
                    framing_for(ok(), &headers),
                    Err(Error::MalformedResponse(_))
                ),
                "{headers:?}"
            );
        }
    }

    #[test]
    fn chunk_size_parsing() {
        assert_eq!(parse_chunk_size(b"1a").unwrap(), 26);
        assert_eq!(parse_chunk_size(b"A;name=value").unwrap(), 10);
        assert_eq!(parse_chunk_size(b"0").unwrap(), 0);
        assert_eq!(parse_chunk_size(b"ffffffffffffffff").unwrap(), u64::MAX);
        for bad in [
            &b""[..],
            b"zz",
            b"-1",
            b"0x10",
            b"1 2",
            b"10000000000000000",
        ] {
            assert!(
                matches!(parse_chunk_size(bad), Err(Error::MalformedChunk(_))),
                "{bad:?}"
            );
        }
    }
}
