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
    /// Most response bytes (lines plus literals) accepted for one command.
    pub command_bytes: usize,
    /// Most UIDs accepted from one search.
    pub uids: usize,
}

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
}

impl<S: Read + Write> Client<S> {
    /// Read the server greeting with the default limits.
    pub fn connect(stream: S) -> Result<Self, ImapError> {
        Self::with_limits(stream, Limits::default())
    }

    pub fn with_limits(stream: S, limits: Limits) -> Result<Self, ImapError> {
        let _ = (stream, limits);
        Err(ImapError::Malformed("not implemented"))
    }

    pub fn login(&mut self, user: &str, password: &str) -> Result<(), ImapError> {
        let _ = (user, password, &mut self.stream, self.limits);
        Err(ImapError::Malformed("not implemented"))
    }

    /// Open a mailbox read-only.
    pub fn examine(&mut self, mailbox: &str) -> Result<(), ImapError> {
        let _ = mailbox;
        Err(ImapError::Malformed("not implemented"))
    }

    pub fn uid_search_all(&mut self) -> Result<Vec<u32>, ImapError> {
        Err(ImapError::Malformed("not implemented"))
    }

    /// The requested section of message `uid`, or `None` when the server
    /// returns no such message.
    pub fn uid_fetch(&mut self, uid: u32, section: Section) -> Result<Option<Vec<u8>>, ImapError> {
        let _ = (uid, section);
        Err(ImapError::Malformed("not implemented"))
    }

    pub fn logout(&mut self) -> Result<(), ImapError> {
        Err(ImapError::Malformed("not implemented"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// A scripted server: replies are read from `input`, client writes are kept.
    struct Script {
        input: Cursor<Vec<u8>>,
        written: Vec<u8>,
    }

    impl Script {
        fn new(replies: &[u8]) -> Self {
            Self {
                input: Cursor::new(replies.to_vec()),
                written: Vec::new(),
            }
        }
    }

    impl Read for Script {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.input.read(buf)
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
        replies.extend_from_slice(b"* 3 EXISTS\r\n* FLAGS (\\Seen)\r\nA0002 OK [READ-ONLY] Examine completed\r\n");
        replies.extend_from_slice(b"* SEARCH 4 9 12\r\nA0003 OK Search completed\r\n");
        replies.extend_from_slice(format!("* 2 FETCH (UID 9 BODY[HEADER] {{{}}}\r\n", header.len()).as_bytes());
        replies.extend_from_slice(header);
        replies.extend_from_slice(b")\r\nA0004 OK Fetch completed\r\n");
        replies.extend_from_slice(format!("* 3 FETCH (UID 12 BODY[] {{{}}}\r\n", body.len()).as_bytes());
        replies.extend_from_slice(body);
        replies.extend_from_slice(b")\r\nA0005 OK Fetch completed\r\n");
        replies.extend_from_slice(b"* BYE Logging out\r\nA0006 OK Logout completed\r\n");
        let mut c = client(&replies);
        c.login("aien@aienos.com", "pw").unwrap();
        c.examine("INBOX").unwrap();
        assert_eq!(c.uid_search_all().unwrap(), vec![4, 9, 12]);
        assert_eq!(c.uid_fetch(9, Section::Header).unwrap().as_deref(), Some(&header[..]));
        assert_eq!(c.uid_fetch(12, Section::Full).unwrap().as_deref(), Some(&body[..]));
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
            assert!(c.stream.written.is_empty(), "nothing may be sent for {bad:?}");
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
        assert_eq!(c.login("u", "p"), Err(ImapError::No("Authentication failed".into())));
        let mut c = client(b"A0001 OK\r\nA0002 BAD Unknown mailbox syntax\r\n");
        c.login("u", "p").unwrap();
        assert_eq!(c.examine("INBOX"), Err(ImapError::Bad("Unknown mailbox syntax".into())));
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
        assert_eq!(c.uid_fetch(5, Section::Full), Err(ImapError::LiteralTooLarge));
        let limits = Limits { literal: 8, ..Limits::default() };
        let mut c = small(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {9}\r\n123456789)\r\nA0002 OK\r\n", limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_fetch(5, Section::Full), Err(ImapError::LiteralTooLarge));
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
        let limits = Limits { line: 32, ..Limits::default() };
        let mut replies = b"A0001 OK\r\n* SEARCH ".to_vec();
        replies.extend(std::iter::repeat_n(b'7', 64));
        let mut c = small(&replies, limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_search_all(), Err(ImapError::LineTooLong));
    }

    #[test]
    fn response_flood_and_byte_flood_are_refused() {
        let limits = Limits { responses: 3, ..Limits::default() };
        let mut c = small(b"A0001 OK\r\n* 1 EXISTS\r\n* 2 EXISTS\r\n* 3 EXISTS\r\n* 4 EXISTS\r\nA0002 OK\r\n", limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.examine("INBOX"), Err(ImapError::TooManyResponses));
        let limits = Limits { command_bytes: 20, ..Limits::default() };
        let mut c = small(b"A0001 OK\r\n* 1 FETCH (UID 5 BODY[] {16}\r\n0123456789abcdef)\r\nA0002 OK\r\n", limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_fetch(5, Section::Full), Err(ImapError::CommandTooLarge));
    }

    #[test]
    fn search_refuses_too_many_or_non_uid_results() {
        let limits = Limits { uids: 2, ..Limits::default() };
        let mut c = small(b"A0001 OK\r\n* SEARCH 1 2 3\r\nA0002 OK\r\n", limits).unwrap();
        c.login("u", "p").unwrap();
        assert_eq!(c.uid_search_all(), Err(ImapError::TooManyUids));
        for bad in [&b"* SEARCH 1 x\r\n"[..], b"* SEARCH 0\r\n", b"* SEARCH 4294967296\r\n"] {
            let mut replies = b"A0001 OK\r\n".to_vec();
            replies.extend_from_slice(bad);
            replies.extend_from_slice(b"A0002 OK\r\n");
            let mut c = client(&replies);
            c.login("u", "p").unwrap();
            assert!(matches!(c.uid_search_all(), Err(ImapError::Malformed(_))), "{bad:?}");
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
        assert_eq!(c.uid_fetch(7, Section::Full).unwrap().as_deref(), Some(&b"mine"[..]));
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
        assert_eq!(c.examine("INBOX"), Err(ImapError::Bye("Server shutting down".into())));
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
        assert_eq!(c.login("u", "pässword"), Err(ImapError::No("Literal refused".into())));
        assert_eq!(written(&c), "A0001 LOGIN \"u\" {9}\r\n");
    }

    #[test]
    fn greeting_must_be_ok() {
        let bye = Client::connect(Script::new(b"* BYE Too many connections\r\n"));
        assert_eq!(bye.err(), Some(ImapError::Bye("Too many connections".into())));
        let preauth = Client::connect(Script::new(b"* PREAUTH ready\r\n"));
        assert!(matches!(preauth.err(), Some(ImapError::Malformed(_))));
        let junk = Client::connect(Script::new(b"HTTP/1.1 400 Bad Request\r\n"));
        assert!(matches!(junk.err(), Some(ImapError::Malformed(_))));
        let empty = Client::connect(Script::new(b""));
        assert_eq!(empty.err(), Some(ImapError::Truncated));
    }
}
