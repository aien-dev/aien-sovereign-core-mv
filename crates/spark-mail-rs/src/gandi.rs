//! Gandi mailbox transport. Credentials are resolved from the hardware vault
//! at call time and never written to the local mail store.

use imap::Session;
use lettre::transport::smtp::authentication::Credentials;
use lettre::{Message, SmtpTransport, Transport};
use mailparse::MailHeaderMap;
use native_tls::TlsConnector;
use serde::Serialize;
use std::io::{Error, ErrorKind};
use std::net::TcpStream;
use std::path::PathBuf;
use std::process::Command;

const HOST: &str = "mail.gandi.net";

#[derive(Clone, Copy, Debug, Eq, PartialEq, clap::ValueEnum)]
pub enum Account {
    Drake,
    Aien,
}

impl Account {
    pub fn address(self) -> &'static str {
        match self {
            Self::Drake => "drake@aienos.com",
            Self::Aien => "aien@aienos.com",
        }
    }

    fn vault_key(self) -> &'static str {
        match self {
            Self::Drake => "GANDI_DRAKE_MAIL_PASSWORD",
            Self::Aien => "GANDI_AIEN_MAIL_PASSWORD",
        }
    }
}

#[derive(Debug, Serialize)]
pub struct MailSummary {
    pub uid: u32,
    pub from: String,
    pub subject: String,
    pub date: String,
}

#[derive(Debug, Serialize)]
pub struct MailDetail {
    pub uid: u32,
    pub from: String,
    pub subject: String,
    pub date: String,
    pub body: String,
}

fn vault_password(account: Account) -> Result<String, Error> {
    let home = std::env::var_os("HOME")
        .map(PathBuf::from)
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "HOME is not set"))?;
    let local = home.join(".local/bin/atlas-vault");
    let binary = if local.is_file() {
        local
    } else {
        PathBuf::from("atlas-vault")
    };
    let output = Command::new(binary)
        .args(["get", account.vault_key()])
        .output()?;
    if !output.status.success() {
        return Err(Error::new(
            ErrorKind::PermissionDenied,
            format!(
                "mail credential {} is unavailable in atlas-vault",
                account.vault_key()
            ),
        ));
    }
    let password = String::from_utf8(output.stdout)
        .map_err(|_| Error::new(ErrorKind::InvalidData, "vault returned invalid UTF-8"))?;
    let password = password.trim_end_matches(['\r', '\n']);
    if password.is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidData,
            "vault returned an empty mail credential",
        ));
    }
    Ok(password.to_owned())
}

fn imap_session(account: Account) -> Result<Session<native_tls::TlsStream<TcpStream>>, Error> {
    let tls = TlsConnector::builder()
        .build()
        .map_err(|_| Error::other("could not initialize TLS"))?;
    let client = imap::connect((HOST, 993), HOST, &tls)
        .map_err(|_| Error::other("could not connect to Gandi IMAP over TLS"))?;
    client
        .login(account.address(), vault_password(account)?)
        .map_err(|_| Error::new(ErrorKind::PermissionDenied, "Gandi IMAP login failed"))
}

fn header_value(headers: &[mailparse::MailHeader<'_>], name: &str) -> String {
    headers.get_first_value(name).unwrap_or_default()
}

fn plain_body(part: &mailparse::ParsedMail<'_>) -> Option<String> {
    if part.ctype.mimetype == "text/plain" {
        return part.get_body().ok();
    }
    part.subparts.iter().find_map(plain_body)
}

pub fn list(account: Account, limit: usize) -> Result<Vec<MailSummary>, Error> {
    let mut session = imap_session(account)?;
    session
        .examine("INBOX")
        .map_err(|_| Error::other("cannot open inbox"))?;
    let mut uids: Vec<u32> = session
        .uid_search("ALL")
        .map_err(|_| Error::other("cannot search inbox"))?
        .into_iter()
        .collect();
    uids.sort_unstable_by(|a, b| b.cmp(a));
    let mut messages = Vec::new();
    for uid in uids.into_iter().take(limit.min(100)) {
        let fetched = session
            .uid_fetch(uid.to_string(), "BODY.PEEK[HEADER]")
            .map_err(|_| Error::other("cannot fetch mail headers"))?;
        if let Some(raw) = fetched.iter().next().and_then(|item| item.header()) {
            let (headers, _) = mailparse::parse_headers(raw)
                .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid mail headers"))?;
            messages.push(MailSummary {
                uid,
                from: header_value(&headers, "From"),
                subject: header_value(&headers, "Subject"),
                date: header_value(&headers, "Date"),
            });
        }
    }
    let _ = session.logout();
    Ok(messages)
}

pub fn read(account: Account, uid: u32) -> Result<MailDetail, Error> {
    let mut session = imap_session(account)?;
    session
        .examine("INBOX")
        .map_err(|_| Error::other("cannot open inbox"))?;
    let fetched = session
        .uid_fetch(uid.to_string(), "BODY.PEEK[]")
        .map_err(|_| Error::other("cannot fetch message"))?;
    let raw = fetched
        .iter()
        .next()
        .and_then(|item| item.body())
        .ok_or_else(|| Error::new(ErrorKind::NotFound, "message UID not found"))?;
    let parsed = mailparse::parse_mail(raw)
        .map_err(|_| Error::new(ErrorKind::InvalidData, "invalid mail message"))?;
    let body = plain_body(&parsed).unwrap_or_else(|| "[No plain-text body]".to_string());
    let detail = MailDetail {
        uid,
        from: header_value(&parsed.headers, "From"),
        subject: header_value(&parsed.headers, "Subject"),
        date: header_value(&parsed.headers, "Date"),
        body,
    };
    let _ = session.logout();
    Ok(detail)
}

pub fn send_as_aien(to: &str, subject: &str, body: &str) -> Result<(), Error> {
    if subject.trim().is_empty() || body.trim().is_empty() {
        return Err(Error::new(
            ErrorKind::InvalidInput,
            "subject and body are required",
        ));
    }
    let message = Message::builder()
        .from(
            Account::Aien
                .address()
                .parse()
                .map_err(|_| Error::other("invalid AIEN address"))?,
        )
        .to(to
            .parse()
            .map_err(|_| Error::new(ErrorKind::InvalidInput, "invalid recipient address"))?)
        .subject(subject)
        .body(body.to_owned())
        .map_err(|_| Error::new(ErrorKind::InvalidInput, "invalid mail content"))?;
    let transport = SmtpTransport::relay(HOST)
        .map_err(|_| Error::other("could not initialize Gandi SMTP over TLS"))?
        .credentials(Credentials::new(
            Account::Aien.address().to_owned(),
            vault_password(Account::Aien)?,
        ))
        .build();
    transport
        .send(&message)
        .map_err(|_| Error::other("Gandi SMTP did not confirm delivery"))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identities_have_separate_vault_keys() {
        assert_eq!(Account::Drake.address(), "drake@aienos.com");
        assert_eq!(Account::Aien.address(), "aien@aienos.com");
        assert_ne!(Account::Drake.vault_key(), Account::Aien.vault_key());
    }

    #[test]
    fn empty_mail_is_rejected_before_vault_access() {
        assert_eq!(
            send_as_aien("person@example.com", "", "body")
                .unwrap_err()
                .kind(),
            ErrorKind::InvalidInput
        );
    }
}
