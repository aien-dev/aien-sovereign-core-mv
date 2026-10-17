//! Minimal IMAP4rev1 client (RFC 3501) for the Gandi mailbox reader.
//!
//! It speaks only the five commands the reader sends: LOGIN, EXAMINE,
//! UID SEARCH, UID FETCH and LOGOUT. The caller owns the stream (TLS stays on
//! native-tls). This module never logs, and no error it returns carries a
//! credential. Every server reply is bounded: line length, literal size,
//! untagged responses and bytes per command, and search results all have hard
//! caps, and overflow is a typed refusal, never a silent truncation.

use std::fmt;
use std::io::{ErrorKind, Read, Write};

/// Hard caps on what a server may send. Overflow is refused, never truncated.
#[derive(Clone, Copy, Debug)]
pub struct Limits {
    /// Longest response line, CRLF excluded (a whole `* SEARCH` reply is one line).
    pub line: usize,
    /// Largest single `{n}` literal.
    pub literal: usize,
    /// Most untagged responses accepted for one command.
    pub responses: usize,
    /// Most response bytes (lines plus literals) accepted for one command. Every line
    /// and literal also costs `ENTRY_COST` bytes, so a flood of empty literals is bounded
    /// by the memory it really takes, not only by its size on the wire.
    pub command_bytes: usize,
    /// Most UIDs accepted from one search.
    pub uids: usize,
}

/// Budget charged for each line and literal on top of its bytes: the allocation and
/// bookkeeping one entry costs whatever its length.
const ENTRY_COST: usize = 64;

impl Default for Limits {
    fn default() -> Self {
        Self {
            line: 1024 * 1024,
            literal: 32 * 1024 * 1024,
            responses: 10_000,
            command_bytes: 64 * 1024 * 1024,
            uids: 200_000,
        }
    }
}

/// Why an IMAP exchange was refused. No variant carries a credential.
#[derive(Debug, Eq, PartialEq)]
pub enum ImapError {
    Io(ErrorKind),
    /// The server closed the stream in the middle of a line or literal.
    Truncated,
    LineTooLong,
    LiteralTooLarge,
    TooManyResponses,
    CommandTooLarge,
    TooManyUids,
    Malformed(&'static str),
    /// A command argument holds CR, LF or NUL, which IMAP cannot carry.
    UnsafeArgument,
    /// Tagged NO, with the server's text reduced to printable ASCII.
    No(String),
    /// Tagged BAD, with the server's text reduced to printable ASCII.
    Bad(String),
    /// Untagged BYE where none was expected.
    Bye(String),
}

impl fmt::Display for ImapError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(kind) => write!(f, "IMAP stream error: {kind}"),
            Self::Truncated => f.write_str("IMAP server closed the stream mid-response"),
            Self::LineTooLong => f.write_str("IMAP response line exceeds the line limit"),
            Self::LiteralTooLarge => f.write_str("IMAP literal exceeds the literal limit"),
            Self::TooManyResponses => f.write_str("IMAP command drew too many responses"),
            Self::CommandTooLarge => f.write_str("IMAP command drew too many response bytes"),
            Self::TooManyUids => f.write_str("IMAP search returned too many UIDs"),
            Self::Malformed(what) => write!(f, "malformed IMAP response: {what}"),
            Self::UnsafeArgument => f.write_str("IMAP argument holds CR, LF or NUL"),
            Self::No(text) => write!(f, "IMAP server said NO: {text}"),
            Self::Bad(text) => write!(f, "IMAP server said BAD: {text}"),
            Self::Bye(text) => write!(f, "IMAP server said BYE: {text}"),
        }
    }
}

impl std::error::Error for ImapError {}

/// Which part of a message `uid_fetch` returns. Both use `BODY.PEEK`, so the
/// read never sets the `\Seen` flag.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Section {
    Header,
    Full,
}

pub struct Client<S> {
    stream: S,
    limits: Limits,
    /// Bytes read from the stream and not yet consumed.
    buf: Vec<u8>,
    next_tag: u32,
}

/// One server response: the text around each literal, then the literals.
/// `text.len() == literals.len() + 1`.
struct Response {
    text: Vec<Vec<u8>>,
    literals: Vec<Vec<u8>>,
}

/// One encoded command argument.
enum Arg<'a> {
    Atom(&'a str),
    Quoted(Vec<u8>),
    /// Sent as a synchronizing literal: `{n}`, wait for `+`, then the bytes.
    Literal(&'a [u8]),
}

impl<'a> Arg<'a> {
    /// An IMAP string: quoted when 7-bit, a literal when it holds 8-bit
    /// bytes, refused when it holds CR, LF or NUL (RFC 3501 section 4.3).
    fn string(value: &'a str) -> Result<Self, ImapError> {
        let bytes = value.as_bytes();
        if bytes.iter().any(|&b| matches!(b, b'\r' | b'\n' | 0)) {
            return Err(ImapError::UnsafeArgument);
        }
        if !bytes.is_ascii() {
            return Ok(Self::Literal(bytes));
        }
        let mut quoted = Vec::with_capacity(bytes.len() + 2);
        quoted.push(b'"');
        for &b in bytes {
            if b == b'"' || b == b'\\' {
                quoted.push(b'\\');
            }
            quoted.push(b);
        }
        quoted.push(b'"');
        Ok(Self::Quoted(quoted))
    }
}

impl<S: Read + Write> Client<S> {
    /// Read the server greeting with the default limits.
    pub fn connect(stream: S) -> Result<Self, ImapError> {
        Self::with_limits(stream, Limits::default())
    }

    pub fn with_limits(stream: S, limits: Limits) -> Result<Self, ImapError> {
        let mut client = Self {
            stream,
            limits,
            buf: Vec::new(),
            next_tag: 1,
        };
        // A greeting is one line; it gets one line's worth of budget.
        let mut budget = limits.line + 2 + ENTRY_COST;
        let greeting = client.read_response(&mut budget)?;
        if let Some(text) = untagged(&greeting.text[0], b"BYE") {
            return Err(ImapError::Bye(printable(text)));
        }
        if untagged(&greeting.text[0], b"OK").is_none() {
            return Err(ImapError::Malformed("server greeting is not * OK"));
        }
        Ok(client)
    }

    pub fn login(&mut self, user: &str, password: &str) -> Result<(), ImapError> {
        let args = [
            Arg::Atom("LOGIN"),
            Arg::string(user)?,
            Arg::string(password)?,
        ];
        self.run(&args, false).map(drop)
    }

    /// Open a mailbox read-only.
    pub fn examine(&mut self, mailbox: &str) -> Result<(), ImapError> {
        self.run(&[Arg::Atom("EXAMINE"), Arg::string(mailbox)?], false)
            .map(drop)
    }

    pub fn uid_search_all(&mut self) -> Result<Vec<u32>, ImapError> {
        let args = [Arg::Atom("UID"), Arg::Atom("SEARCH"), Arg::Atom("ALL")];
        let mut uids = Vec::new();
        for response in self.run(&args, false)? {
            let Some(rest) = untagged(&response.text[0], b"SEARCH") else {
                continue;
            };
            if !response.literals.is_empty() {
                return Err(ImapError::Malformed("search result carries a literal"));
            }
            for token in rest.split(|&b| b == b' ').filter(|t| !t.is_empty()) {
                if uids.len() == self.limits.uids {
                    return Err(ImapError::TooManyUids);
                }
                uids.push(
                    parse_uid(token).ok_or(ImapError::Malformed("search result is not a UID"))?,
                );
            }
        }
        Ok(uids)
    }

    /// The requested section of message `uid`, or `None` when the server
    /// returns no such message.
    pub fn uid_fetch(&mut self, uid: u32, section: Section) -> Result<Option<Vec<u8>>, ImapError> {
        let (request, reply) = match section {
            Section::Header => ("BODY.PEEK[HEADER]", &b"BODY[HEADER]"[..]),
            Section::Full => ("BODY.PEEK[]", &b"BODY[]"[..]),
        };
        let uid_text = uid.to_string();
        let args = [
            Arg::Atom("UID"),
            Arg::Atom("FETCH"),
            Arg::Atom(&uid_text),
            Arg::Atom(request),
        ];
        for response in self.run(&args, false)? {
            // Servers may interleave unsolicited FETCH data for other messages.
            if !is_fetch(&response.text[0]) || fetch_uid(&response) != Some(uid) {
                continue;
            }
            // A body sent as a quoted string would read as missing; refuse it. NIL (no
            // such section) stays `None`.
            if response.text.iter().any(|text| quotes_section(text, reply)) {
                return Err(ImapError::Malformed("section sent as a quoted string"));
            }
            for (text, literal) in response.text.iter().zip(response.literals) {
                if names_section(text, reply) {
                    return Ok(Some(literal));
                }
            }
        }
        Ok(None)
    }

    pub fn logout(&mut self) -> Result<(), ImapError> {
        self.run(&[Arg::Atom("LOGOUT")], true).map(drop)
    }

    /// Send one tagged command and collect its untagged responses up to the
    /// tagged OK. Arguments are encoded before anything is sent, so a refused
    /// argument leaves the stream untouched.
    fn run(&mut self, args: &[Arg<'_>], expect_bye: bool) -> Result<Vec<Response>, ImapError> {
        let tag = format!("A{:04}", self.next_tag);
        self.next_tag += 1;
        let mut budget = self.limits.command_bytes;
        let mut seen = 0;
        let mut line = tag.clone().into_bytes();
        for arg in args {
            line.push(b' ');
            match arg {
                Arg::Atom(atom) => line.extend_from_slice(atom.as_bytes()),
                Arg::Quoted(quoted) => line.extend_from_slice(quoted),
                Arg::Literal(bytes) => {
                    line.extend_from_slice(format!("{{{}}}\r\n", bytes.len()).as_bytes());
                    self.send(&line)?;
                    line.clear();
                    self.await_continuation(&tag, &mut budget, &mut seen)?;
                    line.extend_from_slice(bytes);
                }
            }
        }
        line.extend_from_slice(b"\r\n");
        self.send(&line)?;
        let mut responses = Vec::new();
        loop {
            let response = self.read_response(&mut budget)?;
            if let Some(rest) = tagged(&response.text[0], &tag) {
                return tagged_status(rest).map(|()| responses);
            }
            self.check_untagged(&response.text[0], expect_bye, &mut seen)?;
            responses.push(response);
        }
    }

    /// Wait for the `+` that lets a literal be sent. A tagged NO or BAD here
    /// means the server refused the command, and the literal is never sent.
    fn await_continuation(
        &mut self,
        tag: &str,
        budget: &mut usize,
        seen: &mut usize,
    ) -> Result<(), ImapError> {
        loop {
            let response = self.read_response(budget)?;
            let first = &response.text[0];
            if first.starts_with(b"+") {
                return Ok(());
            }
            if let Some(rest) = tagged(first, tag) {
                tagged_status(rest)?;
                return Err(ImapError::Malformed(
                    "tagged OK before the literal was sent",
                ));
            }
            self.check_untagged(first, false, seen)?;
        }
    }

    fn check_untagged(
        &self,
        first: &[u8],
        expect_bye: bool,
        seen: &mut usize,
    ) -> Result<(), ImapError> {
        if first.starts_with(b"+") {
            return Err(ImapError::Malformed("unexpected continuation request"));
        }
        if !first.starts_with(b"* ") {
            return Err(ImapError::Malformed("response with a foreign tag"));
        }
        if let Some(text) = untagged(first, b"BYE")
            && !expect_bye
        {
            return Err(ImapError::Bye(printable(text)));
        }
        if *seen == self.limits.responses {
            return Err(ImapError::TooManyResponses);
        }
        *seen += 1;
        Ok(())
    }

    fn send(&mut self, bytes: &[u8]) -> Result<(), ImapError> {
        self.stream
            .write_all(bytes)
            .map_err(|e| ImapError::Io(e.kind()))?;
        self.stream.flush().map_err(|e| ImapError::Io(e.kind()))
    }

    /// One response: a line, plus a literal and a further line for every
    /// line that ends in `{n}`. Only untagged responses carry literals: a tagged
    /// status or `+` line ending in `{n}` is just text.
    fn read_response(&mut self, budget: &mut usize) -> Result<Response, ImapError> {
        let mut text = Vec::new();
        let mut literals = Vec::new();
        loop {
            let line = self.read_line(budget)?;
            let literal = if text.is_empty() && !line.starts_with(b"* ") {
                None
            } else {
                literal_len(&line)?
            };
            text.push(line);
            let Some(len) = literal else {
                return Ok(Response { text, literals });
            };
            if len > self.limits.literal {
                return Err(ImapError::LiteralTooLarge);
            }
            charge(budget, len.saturating_add(ENTRY_COST))?;
            literals.push(self.read_exact(len)?);
        }
    }

    /// The next CRLF-terminated line, CRLF removed.
    fn read_line(&mut self, budget: &mut usize) -> Result<Vec<u8>, ImapError> {
        let mut scanned = 0;
        loop {
            if let Some(pos) = self.buf[scanned..].windows(2).position(|w| w == b"\r\n") {
                let pos = scanned + pos;
                if pos > self.limits.line {
                    return Err(ImapError::LineTooLong);
                }
                charge(budget, pos + 2 + ENTRY_COST)?;
                let mut line: Vec<u8> = self.buf.drain(..pos + 2).collect();
                line.truncate(pos);
                return Ok(line);
            }
            // The buffer may end in the CR of a line exactly at the limit.
            if self.buf.len() > self.limits.line + 1 {
                return Err(ImapError::LineTooLong);
            }
            scanned = self.buf.len().saturating_sub(1);
            if self.fill()? == 0 {
                return Err(ImapError::Truncated);
            }
        }
    }

    /// Exactly `len` bytes. Nothing is reserved up front, so a server that
    /// announces a literal and then stops costs only what it really sent.
    fn read_exact(&mut self, len: usize) -> Result<Vec<u8>, ImapError> {
        while self.buf.len() < len {
            if self.fill()? == 0 {
                return Err(ImapError::Truncated);
            }
        }
        Ok(self.buf.drain(..len).collect())
    }

    fn fill(&mut self) -> Result<usize, ImapError> {
        let mut chunk = [0u8; 16 * 1024];
        loop {
            match self.stream.read(&mut chunk) {
                Ok(n) => {
                    self.buf.extend_from_slice(&chunk[..n]);
                    return Ok(n);
                }
                Err(e) if e.kind() == ErrorKind::Interrupted => continue,
                Err(e) => return Err(ImapError::Io(e.kind())),
            }
        }
    }
}

fn charge(budget: &mut usize, bytes: usize) -> Result<(), ImapError> {
    *budget = budget
        .checked_sub(bytes)
        .ok_or(ImapError::CommandTooLarge)?;
    Ok(())
}

/// `Some(n)` when the line ends in a literal announcement `{n}`.
fn literal_len(line: &[u8]) -> Result<Option<usize>, ImapError> {
    let Some(body) = line.strip_suffix(b"}") else {
        return Ok(None);
    };
    let Some(open) = body.iter().rposition(|&b| b == b'{') else {
        return Ok(None);
    };
    let digits = &body[open + 1..];
    if digits.is_empty() || !digits.iter().all(u8::is_ascii_digit) {
        return Ok(None);
    }
    let mut len: usize = 0;
    for &d in digits {
        len = len
            .checked_mul(10)
            .and_then(|len| len.checked_add(usize::from(d - b'0')))
            .ok_or(ImapError::LiteralTooLarge)?;
    }
    Ok(Some(len))
}

fn split_word(s: &[u8]) -> (&[u8], &[u8]) {
    match s.iter().position(|&b| b == b' ') {
        Some(p) => (&s[..p], &s[p + 1..]),
        None => (s, &[]),
    }
}

/// The text after `* WORD`, when the line is that untagged response.
fn untagged<'a>(line: &'a [u8], word: &[u8]) -> Option<&'a [u8]> {
    let (first, text) = split_word(line.strip_prefix(b"* ")?);
    first.eq_ignore_ascii_case(word).then_some(text)
}

fn tagged<'a>(line: &'a [u8], tag: &str) -> Option<&'a [u8]> {
    line.strip_prefix(tag.as_bytes())?.strip_prefix(b" ")
}

fn tagged_status(rest: &[u8]) -> Result<(), ImapError> {
    let (word, text) = split_word(rest);
    if word.eq_ignore_ascii_case(b"OK") {
        Ok(())
    } else if word.eq_ignore_ascii_case(b"NO") {
        Err(ImapError::No(printable(text)))
    } else if word.eq_ignore_ascii_case(b"BAD") {
        Err(ImapError::Bad(printable(text)))
    } else {
        Err(ImapError::Malformed("tagged response is not OK, NO or BAD"))
    }
}

/// Server text reduced to printable ASCII and at most 200 characters, so it
/// can be shown without carrying terminal escapes.
fn printable(text: &[u8]) -> String {
    text.iter()
        .filter(|b| (0x20..=0x7e).contains(*b))
        .take(200)
        .map(|&b| char::from(b))
        .collect()
}

fn parse_uid(token: &[u8]) -> Option<u32> {
    if token.is_empty() || token.len() > 10 || !token.iter().all(u8::is_ascii_digit) {
        return None;
    }
    std::str::from_utf8(token)
        .ok()?
        .parse::<u32>()
        .ok()
        .filter(|&uid| uid != 0)
}

/// `* <n> FETCH ...`
fn is_fetch(line: &[u8]) -> bool {
    let Some(rest) = line.strip_prefix(b"* ") else {
        return false;
    };
    let (number, rest) = split_word(rest);
    !number.is_empty()
        && number.iter().all(u8::is_ascii_digit)
        && rest.len() >= 5
        && rest[..5].eq_ignore_ascii_case(b"FETCH")
}

/// The value of the `UID` item in a FETCH response.
fn fetch_uid(response: &Response) -> Option<u32> {
    response.text.iter().find_map(|text| {
        (1..text.len().saturating_sub(3)).find_map(|i| {
            if !matches!(text[i - 1], b'(' | b' ') || !text[i..i + 4].eq_ignore_ascii_case(b"UID ")
            {
                return None;
            }
            let digits = &text[i + 4..];
            let end = digits
                .iter()
                .position(|b| !b.is_ascii_digit())
                .unwrap_or(digits.len());
            parse_uid(&digits[..end])
        })
    })
}

/// Whether `text` carries `name` as a quoted string, as in `... BODY[] "..."`.
fn quotes_section(text: &[u8], name: &[u8]) -> bool {
    let mut quoted = name.to_vec();
    quoted.extend_from_slice(b" \"");
    text.windows(quoted.len()).enumerate().any(|(i, w)| {
        w.eq_ignore_ascii_case(&quoted) && i > 0 && matches!(text[i - 1], b'(' | b' ')
    })
}

/// Whether the text before a literal names `name` as the item it carries,
/// as in `... BODY[HEADER] {123}`.
fn names_section(text: &[u8], name: &[u8]) -> bool {
    let Some(open) = text.iter().rposition(|&b| b == b'{') else {
        return false;
    };
    let head = text[..open].trim_ascii_end();
    head.len() > name.len()
        && head[head.len() - name.len()..].eq_ignore_ascii_case(name)
        && matches!(head[head.len() - name.len() - 1], b'(' | b' ')
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A scripted server: replies are read from `input`, at most `chunk` bytes per
    /// read, and client writes are kept.
    struct Script {
        input: Cursor<Vec<u8>>,
        chunk: usize,
        written: Vec<u8>,
    }

    impl Script {
        fn new(replies: &[u8]) -> Self {
            Self {
                input: Cursor::new(replies.to_vec()),
                chunk: usize::MAX,
                written: Vec::new(),
            }
        }

        /// A server that hands over one byte per read.
        fn trickle(replies: &[u8]) -> Self {
            Self {
                chunk: 1,
                ..Self::new(replies)
            }
        }
    }

    impl Read for Script {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            let n = buf.len().min(self.chunk);
            self.input.read(&mut buf[..n])
        }
    }

    impl Write for Script {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            self.written.extend_from_slice(buf);
            Ok(buf.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    const GREETING: &[u8] = b"* OK [CAPABILITY IMAP4rev1] Dovecot ready.\r\n";

    fn client(replies: &[u8]) -> Client<Script> {
        let mut all = GREETING.to_vec();
        all.extend_from_slice(replies);
        Client::connect(Script::new(&all)).expect("greeting accepted")
    }

    fn small(replies: &[u8], limits: Limits) -> Result<Client<Script>, ImapError> {
        let mut all = GREETING.to_vec();
        all.extend_from_slice(replies);
        Client::with_limits(Script::new(&all), limits)
    }

    fn written(client: &Client<Script>) -> String {
        String::from_utf8_lossy(&client.stream.written).into_owned()
    }

    #[test]
    fn reads_headers_and_bodies_through_the_five_commands() {
        let header = b"From: a@example.com\r\nSubject: hi\r\n\r\n";
        let body = b"From: a@example.com\r\nSubject: hi\r\n\r\nhello\r\n";
        let mut replies = Vec::new();
        replies.extend_from_slice(b"A0001 OK Logged in\r\n");
        replies.extend_from_slice(
            b"* 3 EXISTS\r\n* FLAGS (\\Seen)\r\nA0002 OK [READ-ONLY] Examine completed\r\n",
        );
        replies.extend_from_slice(b"* SEARCH 4 9 12\r\nA0003 OK Search completed\r\n");
        replies.extend_from_slice(
            format!("* 2 FETCH (UID 9 BODY[HEADER] {{{}}}\r\n", header.len()).as_bytes(),
        );
        replies.extend_from_slice(header);
        replies.extend_from_slice(b")\r\nA0004 OK Fetch completed\r\n");
        replies.extend_from_slice(
            format!("* 3 FETCH (UID 12 BODY[] {{{}}}\r\n", body.len()).as_bytes(),
        );
        replies.extend_from_slice(body);
        replies.extend_from_slice(b")\r\nA0005 OK Fetch completed\r\n");
        replies.extend_from_slice(b"* BYE Logging out\r\nA0006 OK Logout completed\r\n");
        let mut c = client(&replies);
        c.login("aien@aienos.com", "pw").unwrap();
        c.examine("INBOX").unwrap();
        assert_eq!(c.uid_search_all().unwrap(), vec![4, 9, 12]);
        assert_eq!(
            c.uid_fetch(9, Section::Header).unwrap().as_deref(),
            Some(&header[..])
        );
        assert_eq!(
            c.uid_fetch(12, Section::Full).unwrap().as_deref(),
            Some(&body[..])
        );
        c.logout().unwrap();
        assert_eq!(
            written(&c),
            "A0001 LOGIN \"aien@aienos.com\" \"pw\"\r\n\
             A0002 EXAMINE \"INBOX\"\r\n\
             A0003 UID SEARCH ALL\r\n\
             A0004 UID FETCH 9 BODY.PEEK[HEADER]\r\n\
             A0005 UID FETCH 12 BODY.PEEK[]\r\n\
             A0006 LOGOUT\r\n"
        );
    }

    #[test]
    fn login_quotes_and_escapes_backslash_and_double_quote() {
        let mut c = client(b"A0001 OK Logged in\r\n");
        c.login("u", r#"a"b\c"#).unwrap();
        assert_eq!(written(&c), "A0001 LOGIN \"u\" \"a\\\"b\\\\c\"\r\n");
    }

    #[test]
    fn login_sends_eight_bit_password_as_a_literal_after_continuation() {
        let mut c = client(b"+ OK\r\nA0001 OK Logged in\r\n");
        c.login("u", "pässword").unwrap();
        let mut expected = b"A0001 LOGIN \"u\" {9}\r\n".to_vec();
        expected.extend_from_slice("pässword".as_bytes());
        expected.extend_from_slice(b"\r\n");
        assert_eq!(c.stream.written, expected);
    }

    #[test]
    fn login_refuses_cr_lf_nul_before_sending_anything() {
        for bad in ["pw\r\nA0002 DELETE INBOX", "pw\n", "pw\0"] {
            let mut c = client(b"");
            assert_eq!(c.login("u", bad), Err(ImapError::UnsafeArgument));
            assert!(
                c.stream.written.is_empty(),
                "nothing may be sent for {bad:?}"
            );
        }
    }

    #[test]
    fn errors_never_carry_the_password() {
        let secret = "s3cr3t-Pa55";
        let mut c = client(b"A0001 NO [AUTHENTICATIONFAILED] Authentication failed.\r\n");
        let err = c.login("u", secret).unwrap_err();
        assert!(matches!(err, ImapError::No(_)));
        assert!(!format!("{err} {err:?}").contains(secret));
        let mut c = client(b"");
        let err = c.login("u", &format!("{secret}\r\n")).unwrap_err();
        assert!(!format!("{err} {err:?}").contains(secret));
    }

    #[test]
    fn tagged_no_and_bad_are_typed_refusals() {
        let mut c = client(b"A0001 NO Authentication failed\r\n");
        assert_eq!(
            c.login("u", "p"),
            Err(ImapError::No("Authentication failed".into()))
        );
        let mut c = client(b"A0001 OK\r\nA0002 BAD Unknown mailbox syntax\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(
            c.examine("INBOX"),
            Err(ImapError::Bad("Unknown mailbox syntax".into()))
        );
    }

    #[test]
    fn server_text_is_reduced_to_printable_ascii() {
        let mut c = client(b"A0001 NO bad\x1b[31m\x07 news\r\n");
        assert_eq!(c.login("u", "p"), Err(ImapError::No("bad[31m news".into())));
    }

    #[test]
    fn oversized_literal_is_refused_without_reading_it() {
        let mut c = client(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {99999999999999999999}\r\nxx");
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_fetch(5, Section::Full),
            Err(ImapError::LiteralTooLarge)
        );
        let limits = Limits {
            literal: 8,
            ..Limits::default()
        };
        let mut c = small(
            b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {9}\r\n123456789)\r\nA0002 OK\r\n",
            limits,
        )
        .unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_fetch(5, Section::Full),
            Err(ImapError::LiteralTooLarge)
        );
    }

    #[test]
    fn truncated_line_and_truncated_literal_are_refused() {
        let mut c = client(b"A0001 OK Logged");
        assert_eq!(c.login("u", "p"), Err(ImapError::Truncated));
        let mut c = client(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {10}\r\nshort");
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_fetch(5, Section::Full), Err(ImapError::Truncated));
    }

    #[test]
    fn overlong_line_is_refused() {
        let limits = Limits {
            line: 48,
            ..Limits::default()
        };
        let mut replies = b"A0001 OK\r\n* SEARCH ".to_vec();
        replies.extend(std::iter::repeat_n(b'7', 64));
        let mut c = small(&replies, limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_search_all(), Err(ImapError::LineTooLong));
    }

    #[test]
    fn response_flood_and_byte_flood_are_refused() {
        let limits = Limits {
            responses: 3,
            ..Limits::default()
        };
        let mut c = small(
            b"A0001 OK\r\n* 1 EXISTS\r\n* 2 EXISTS\r\n* 3 EXISTS\r\n* 4 EXISTS\r\nA0002 OK\r\n",
            limits,
        )
        .unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.examine("INBOX"), Err(ImapError::TooManyResponses));
        let limits = Limits {
            // The login reply and the FETCH line fit; the 16-byte literal does not.
            command_bytes: 2 * ENTRY_COST + 40,
            ..Limits::default()
        };
        let mut c = small(
            b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {16}\r\n0123456789abcdef)\r\nA0002 OK\r\n",
            limits,
        )
        .unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_fetch(5, Section::Full),
            Err(ImapError::CommandTooLarge)
        );
    }

    #[test]
    fn search_refuses_too_many_or_non_uid_results() {
        let limits = Limits {
            uids: 2,
            ..Limits::default()
        };
        let mut c = small(b"A0001 OK\r\n* SEARCH 1 2 3\r\nA0002 OK\r\n", limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_search_all(), Err(ImapError::TooManyUids));
        for bad in [
            &b"* SEARCH 1 x\r\n"[..],
            b"* SEARCH 0\r\n",
            b"* SEARCH 4294967296\r\n",
        ] {
            let mut replies = b"A0001 OK\r\n".to_vec();
            replies.extend_from_slice(bad);
            replies.extend_from_slice(b"A0002 OK\r\n");
            let mut c = client(&replies);
            c.login("u", "p").unwrap();
            assert!(
                matches!(c.uid_search_all(), Err(ImapError::Malformed(_))),
                "{bad:?}"
            );
        }
        let mut c = client(b"A0001 OK\r\n* SEARCH\r\nA0002 OK\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_search_all().unwrap(), Vec::<u32>::new());
    }

    #[test]
    fn fetch_ignores_other_messages_and_unrequested_sections() {
        let mut c = client(
            b"A0001 OK\r\n\
              * 1 FETCH (FLAGS (\\Seen))\r\n\
              * 2 FETCH (UID 6 BODY[] {3}\r\nnot)\r\n\
              * 3 FETCH (UID 7 BODY[HEADER] {3}\r\nhdr)\r\n\
              * 3 FETCH (UID 7 BODY[] {4}\r\nmine)\r\n\
              A0002 OK\r\n",
        );
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_fetch(7, Section::Full).unwrap().as_deref(),
            Some(&b"mine"[..])
        );
    }

    #[test]
    fn fetch_of_a_missing_uid_is_none() {
        let mut c = client(b"A0001 OK\r\nA0002 OK Fetch completed\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_fetch(41, Section::Header).unwrap(), None);
    }

    #[test]
    fn unexpected_bye_foreign_tag_and_stray_continuation_are_refused() {
        let mut c = client(b"A0001 OK\r\n* BYE Server shutting down\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(
            c.examine("INBOX"),
            Err(ImapError::Bye("Server shutting down".into()))
        );
        let mut c = client(b"A0001 OK\r\nB0002 OK\r\n");
        c.login("u", "p").unwrap();
        assert!(matches!(c.examine("INBOX"), Err(ImapError::Malformed(_))));
        let mut c = client(b"A0001 OK\r\n+ go ahead\r\n");
        c.login("u", "p").unwrap();
        assert!(matches!(c.examine("INBOX"), Err(ImapError::Malformed(_))));
    }

    #[test]
    fn literal_login_refused_by_server_does_not_send_the_password() {
        let mut c = client(b"A0001 NO Literal refused\r\n");
        assert_eq!(
            c.login("u", "pässword"),
            Err(ImapError::No("Literal refused".into()))
        );
        assert_eq!(written(&c), "A0001 LOGIN \"u\" {9}\r\n");
    }

    #[test]
    fn greeting_must_be_ok() {
        let bye = Client::connect(Script::new(b"* BYE Too many connections\r\n"));
        assert_eq!(
            bye.err(),
            Some(ImapError::Bye("Too many connections".into()))
        );
        let preauth = Client::connect(Script::new(b"* PREAUTH ready\r\n"));
        assert!(matches!(preauth.err(), Some(ImapError::Malformed(_))));
        let junk = Client::connect(Script::new(b"HTTP/1.1 400 Bad Request\r\n"));
        assert!(matches!(junk.err(), Some(ImapError::Malformed(_))));
        let empty = Client::connect(Script::new(b""));
        assert_eq!(empty.err(), Some(ImapError::Truncated));
    }

    #[test]
    fn one_byte_per_read_gives_the_same_transcript() {
        // A 20 KiB literal crosses many fills; CRLFs split across reads resume the scan.
        let body: Vec<u8> = (0..20 * 1024).map(|i| b'a' + (i % 26) as u8).collect();
        let mut replies = GREETING.to_vec();
        replies.extend_from_slice(b"A0001 OK Logged in\r\n");
        replies.extend_from_slice(b"* 1 EXISTS\r\nA0002 OK [READ-ONLY] Examine completed\r\n");
        replies.extend_from_slice(b"* SEARCH 12\r\nA0003 OK Search completed\r\n");
        // The UID item may follow the literal.
        replies.extend_from_slice(format!("* 1 FETCH (BODY[] {{{}}}\r\n", body.len()).as_bytes());
        replies.extend_from_slice(&body);
        replies.extend_from_slice(b" UID 12)\r\nA0004 OK Fetch completed\r\n");
        replies.extend_from_slice(b"* BYE Logging out\r\nA0005 OK Logout completed\r\n");
        let mut c = Client::connect(Script::trickle(&replies)).unwrap();
        c.login("u", "p").unwrap();
        c.examine("INBOX").unwrap();
        assert_eq!(c.uid_search_all().unwrap(), vec![12]);
        assert_eq!(c.uid_fetch(12, Section::Full).unwrap(), Some(body));
        c.logout().unwrap();
    }

    #[test]
    fn line_exactly_at_the_limit_is_accepted_one_byte_per_read() {
        let limits = Limits {
            line: 48,
            ..Limits::default()
        };
        let mut at = b"* SEARCH ".to_vec();
        at.extend(std::iter::repeat_n(b'7', 48 - at.len()));
        let mut replies = GREETING.to_vec();
        replies.extend_from_slice(b"A0001 OK\r\n");
        replies.extend_from_slice(&at);
        replies.extend_from_slice(b"\r\nA0002 OK\r\n");
        let mut c = Client::with_limits(Script::trickle(&replies), limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_search_all(),
            Err(ImapError::Malformed("search result is not a UID"))
        );
        // One byte more is refused.
        let mut over = GREETING.to_vec();
        over.extend_from_slice(b"A0001 OK\r\n");
        over.extend_from_slice(&at);
        over.extend_from_slice(b"7\r\nA0002 OK\r\n");
        let mut c = Client::with_limits(Script::trickle(&over), limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_search_all(), Err(ImapError::LineTooLong));
    }

    #[test]
    fn empty_literal_flood_is_charged_against_the_command_budget() {
        // 2,000 `{0}` literals are 10 kB on the wire but 4,000 allocations; each line and
        // literal is charged a fixed overhead, so the flood hits the command budget.
        let limits = Limits {
            command_bytes: 64 * 1024,
            ..Limits::default()
        };
        let mut replies = b"A0001 OK\r\n* 1 FETCH (UID 5 X {0}\r\n".to_vec();
        for _ in 0..2_000 {
            replies.extend_from_slice(b"{0}\r\n");
        }
        replies.extend_from_slice(b")\r\nA0002 OK\r\n");
        let mut c = small(&replies, limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_fetch(5, Section::Full),
            Err(ImapError::CommandTooLarge)
        );
    }

    #[test]
    fn tagged_and_continuation_lines_never_announce_literals() {
        // A tagged status or `+` line ending in `{n}` is not followed by n bytes.
        let mut c = client(b"A0001 OK done {5}\r\nA0002 OK {9}\r\n");
        c.login("u", "p").unwrap();
        c.examine("INBOX").unwrap();
        let mut c = client(b"+ ready {4}\r\nA0001 OK\r\n");
        c.login("u", "pässword").unwrap();
    }

    #[test]
    fn tagged_no_after_a_fetch_literal_and_bye_before_continuation_are_refused() {
        let mut c =
            client(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {3}\r\nabc)\r\nA0002 NO denied\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(
            c.uid_fetch(5, Section::Full),
            Err(ImapError::No("denied".into()))
        );
        let mut c = client(b"* BYE going away\r\n");
        assert_eq!(
            c.login("u", "pässword"),
            Err(ImapError::Bye("going away".into()))
        );
    }

    #[test]
    fn section_sent_as_a_quoted_string_is_malformed_and_nil_is_none() {
        let mut c = client(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] \"hello\")\r\nA0002 OK\r\n");
        c.login("u", "p").unwrap();
        assert!(matches!(
            c.uid_fetch(5, Section::Full),
            Err(ImapError::Malformed(_))
        ));
        let mut c = client(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[HEADER] NIL)\r\nA0002 OK\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_fetch(5, Section::Header).unwrap(), None);
    }
}
