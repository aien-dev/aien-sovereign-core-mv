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

use crate::cortex_sync::CortexSync;
use crate::models::{EmailMessage, SendEmailRequest};
use crate::store::MailStore;
use chrono::Utc;
use lettre::{Message, SmtpTransport, Transport};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tokio::net::TcpStream;
use uuid::Uuid;

#[derive(Clone, Debug)]
pub struct MailRelay {
    pub bridge_host: String,
    pub bridge_port: u16,
    pub store: Arc<MailStore>,
    pub cortex: Arc<CortexSync>,
    pub default_sender: String,
}

impl MailRelay {
    pub fn new(
        bridge_host: &str,
        bridge_port: u16,
        store: Arc<MailStore>,
        cortex: Arc<CortexSync>,
        default_sender: &str,
    ) -> Self {
        Self {
            bridge_host: bridge_host.to_string(),
            bridge_port,
            store,
            cortex,
            default_sender: default_sender.to_string(),
        }
    }

    pub async fn check_bridge(&self) -> bool {
        let addr = format!("{}:{}", self.bridge_host, self.bridge_port);
        tokio::time::timeout(Duration::from_millis(400), TcpStream::connect(&addr))
            .await
            .map(|r| r.is_ok())
            .unwrap_or(false)
    }

    pub async fn send_email(&self, req: SendEmailRequest) -> Result<EmailMessage, String> {
        reject_header_injection(req.from.as_deref(), &req.to, &req.subject)?;
        crate::effect::authorize_outbound(&req.to, &req.subject, &req.body)
            .map_err(|error| error.to_string())?;
        self.deliver(req).await
    }

    /// Submits to the bridge, then records. Nothing is stored as "sent" or
    /// indexed until the bridge's final 250 reply to end-of-DATA.
    async fn deliver(&self, req: SendEmailRequest) -> Result<EmailMessage, String> {
        reject_header_injection(req.from.as_deref(), &req.to, &req.subject)?;
        let sender = req
            .from
            .clone()
            .unwrap_or_else(|| self.default_sender.clone());
        let id = Uuid::new_v4().to_string();
        let now = Utc::now().to_rfc3339();

        let mut headers = HashMap::new();
        headers.insert("From".to_string(), sender.clone());
        headers.insert("To".to_string(), req.to.join(", "));
        headers.insert("Subject".to_string(), req.subject.clone());
        headers.insert("Date".to_string(), now.clone());
        headers.insert(
            "Message-ID".to_string(),
            format!("<{}@sovereign.spark>", id),
        );

        let mut msg = EmailMessage {
            id: id.clone(),
            from: sender.clone(),
            to: req.to.clone(),
            subject: req.subject.clone(),
            body: req.body.clone(),
            headers,
            received_at: now,
            folder: "sent".to_string(),
            cortex_indexed: false,
            untrusted: false,
        };

        let host = self.bridge_host.clone();
        let port = self.bridge_port;
        let transport_sender = sender.clone();
        let transport_id = id.clone();
        tokio::task::spawn_blocking(move || {
            submit_to_bridge(&host, port, &transport_sender, &transport_id, &req)
        })
        .await
        .map_err(|_| "SMTP transport worker failed; acceptance is unknown".to_string())??;
        msg.headers
            .insert("X-AIEN-Transport-Status".into(), "smtp_accepted".into());

        // Save sent message to local hardware storage
        if self.store.save_message(&msg).is_err() {
            // SMTP has already accepted it. Do not invite a duplicate by reporting a send failure.
            msg.headers
                .insert("X-AIEN-Storage-Status".into(), "persist_failed".into());
        }

        // Index in Cortex memory
        if self.cortex.commit_mail(&msg).await.is_ok() {
            msg.cortex_indexed = true;
            let _ = self.store.save_message(&msg);
        }

        Ok(msg)
    }
}

/// Refuses CR, LF and NUL in anything that becomes an SMTP command argument
/// or a message header, so a caller cannot smuggle extra commands or headers.
fn reject_header_injection(
    sender: Option<&str>,
    recipients: &[String],
    subject: &str,
) -> Result<(), String> {
    let unsafe_text = |s: &str| s.contains(['\r', '\n', '\0']);
    if sender.is_some_and(unsafe_text) {
        return Err("Sender contains a line break or NUL; refused".into());
    }
    if recipients.iter().any(|r| unsafe_text(r)) {
        return Err("Recipient contains a line break or NUL; refused".into());
    }
    if unsafe_text(subject) {
        return Err("Subject contains a line break or NUL; refused".into());
    }
    Ok(())
}

/// Success means the relay's final positive DATA reply, not recipient delivery.
fn submit_to_bridge(
    host: &str,
    port: u16,
    sender: &str,
    id: &str,
    request: &SendEmailRequest,
) -> Result<(), String> {
    reject_header_injection(Some(sender), &request.to, &request.subject)?;
    let mut builder = Message::builder()
        .from(sender.parse().map_err(|_| "Invalid sender address")?)
        .subject(&request.subject)
        .message_id(Some(format!("<{id}@sovereign.spark>")));
    if request.to.is_empty() {
        return Err("At least one recipient is required".into());
    }
    for recipient in &request.to {
        builder = builder.to(recipient.parse().map_err(|_| "Invalid recipient address")?);
    }
    let message = builder
        .body(request.body.clone())
        .map_err(|_| "Invalid mail headers or body")?;
    let transport = SmtpTransport::builder_dangerous(host)
        .port(port)
        .timeout(Some(Duration::from_secs(5)))
        .build();
    transport.send(&message).map_err(|_| {
        "SMTP relay refused or interrupted the send; acceptance is not confirmed".to_string()
    })?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::{BufRead, BufReader, Write};

    fn relay_case(reject_recipient: bool, reject_data: bool) -> Result<(), String> {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream.write_all(b"220 test relay\r\n").unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut data = false;
            let mut saw_body = false;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let reply: &[u8] = if data {
                    if line == ".\r\n" {
                        data = false;
                        if reject_data {
                            b"554 rejected\r\n"
                        } else {
                            b"250 accepted\r\n"
                        }
                    } else {
                        saw_body = true;
                        continue;
                    }
                } else if line.starts_with("EHLO") {
                    b"250-test\r\n250 8BITMIME\r\n"
                } else if line.starts_with("RCPT") && reject_recipient {
                    b"550 recipient refused\r\n"
                } else if line.starts_with("DATA") {
                    data = true;
                    b"354 send data\r\n"
                } else if line.starts_with("QUIT") {
                    let _ = stream.write_all(b"221 bye\r\n");
                    break;
                } else {
                    b"250 ok\r\n"
                };
                if stream.write_all(reply).is_err() {
                    break;
                }
            }
            saw_body
        });
        let req = SendEmailRequest {
            to: vec!["receiver@example.test".into()],
            subject: "test".into(),
            body: "hello\n.line\n".into(),
            from: None,
        };
        let result = submit_to_bridge("127.0.0.1", port, "sender@example.test", "test-id", &req);
        let saw_body = server.join().unwrap();
        if reject_recipient {
            assert!(!saw_body);
        } else {
            assert!(saw_body);
        }
        result
    }

    #[test]
    fn transport_requires_final_smtp_acceptance() {
        assert!(relay_case(false, false).is_ok());
        assert!(relay_case(true, false).is_err());
        assert!(relay_case(false, true).is_err());
    }

    #[test]
    fn unavailable_bridge_cannot_report_success() {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        let req = SendEmailRequest {
            to: vec!["receiver@example.test".into()],
            subject: "test".into(),
            body: "hello".into(),
            from: None,
        };
        assert!(
            submit_to_bridge("127.0.0.1", port, "sender@example.test", "test-id", &req).is_err()
        );
    }

    /// A port with nothing listening. Tests point Cortex here so they never
    /// write to the live Cortex on 127.0.0.1:18080.
    fn dead_port() -> u16 {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        drop(listener);
        assert_ne!(port, 18080);
        port
    }

    fn fake_relay(reject_data: bool) -> (u16, std::thread::JoinHandle<()>) {
        let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let port = listener.local_addr().unwrap().port();
        let server = std::thread::spawn(move || {
            let (mut stream, _) = listener.accept().unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(3)))
                .unwrap();
            stream.write_all(b"220 test relay\r\n").unwrap();
            let mut reader = BufReader::new(stream.try_clone().unwrap());
            let mut data = false;
            loop {
                let mut line = String::new();
                if reader.read_line(&mut line).unwrap_or(0) == 0 {
                    break;
                }
                let reply: &[u8] = if data {
                    if line != ".\r\n" {
                        continue;
                    }
                    data = false;
                    if reject_data {
                        b"554 rejected\r\n"
                    } else {
                        b"250 accepted\r\n"
                    }
                } else if line.starts_with("EHLO") {
                    b"250-test\r\n250 8BITMIME\r\n"
                } else if line.starts_with("DATA") {
                    data = true;
                    b"354 send data\r\n"
                } else if line.starts_with("QUIT") {
                    let _ = stream.write_all(b"221 bye\r\n");
                    break;
                } else {
                    b"250 ok\r\n"
                };
                if stream.write_all(reply).is_err() {
                    break;
                }
            }
        });
        (port, server)
    }

    fn test_relay(bridge_port: u16) -> (MailRelay, std::path::PathBuf) {
        let dir = std::env::temp_dir().join(format!("spark-mail-relay-{}", Uuid::new_v4()));
        let store = Arc::new(MailStore::new(Some(dir.clone())));
        let cortex = Arc::new(CortexSync::new(Some(&format!(
            "http://127.0.0.1:{}",
            dead_port()
        ))));
        (
            MailRelay::new(
                "127.0.0.1",
                bridge_port,
                store,
                cortex,
                "sender@example.test",
            ),
            dir,
        )
    }

    fn request(subject: &str) -> SendEmailRequest {
        SendEmailRequest {
            to: vec!["receiver@example.test".into()],
            subject: subject.into(),
            body: "hello".into(),
            from: None,
        }
    }

    #[tokio::test]
    async fn sent_is_recorded_only_after_final_acceptance() {
        let (port, server) = fake_relay(false);
        let (relay, dir) = test_relay(port);
        let msg = relay.deliver(request("accepted")).await.unwrap();
        server.join().unwrap();
        assert_eq!(
            msg.headers
                .get("X-AIEN-Transport-Status")
                .map(String::as_str),
            Some("smtp_accepted")
        );
        assert!(!msg.cortex_indexed, "dead Cortex port must not index");
        assert_eq!(relay.store.count_folder("sent"), 1);
        std::fs::remove_dir_all(dir).unwrap();

        let (port, server) = fake_relay(true);
        let (relay, dir) = test_relay(port);
        assert!(relay.deliver(request("rejected")).await.is_err());
        server.join().unwrap();
        assert_eq!(relay.store.count_folder("sent"), 0);
        std::fs::remove_dir_all(dir).unwrap();

        let (relay, dir) = test_relay(dead_port());
        assert!(relay.deliver(request("no bridge")).await.is_err());
        assert_eq!(relay.store.count_folder("sent"), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[tokio::test]
    async fn line_breaks_in_sender_recipient_or_subject_are_refused() {
        let (relay, dir) = test_relay(dead_port());
        let mut subject = request("hi\r\nBcc: victim@example.test");
        assert!(
            relay
                .deliver(subject.clone())
                .await
                .unwrap_err()
                .contains("Subject")
        );
        subject.subject = "hi\nRCPT TO:<x@example.test>".into();
        assert!(
            relay
                .send_email(subject)
                .await
                .unwrap_err()
                .contains("Subject")
        );
        let mut recipient = request("ok");
        recipient.to = vec!["a@example.test>\r\nRCPT TO:<b@example.test".into()];
        assert!(
            relay
                .deliver(recipient)
                .await
                .unwrap_err()
                .contains("Recipient")
        );
        let mut sender = request("ok");
        sender.from = Some("a@example.test\r\nX-Injected: 1".into());
        assert!(relay.deliver(sender).await.unwrap_err().contains("Sender"));
        assert_eq!(relay.store.count_folder("sent"), 0);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
