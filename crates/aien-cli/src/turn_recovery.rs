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

//! sovereign-core #665: failure classes and one-shot recovery for the `aien` chat loop.
//!
//! Mechanism after Hermes Agent (Nous Research, MIT, commit 5480918),
//! re-implemented; no code copied.
//!
//! A failed model call is put in one class, and each class has at most one
//! recovery per turn. When a class fails a second time the turn stops with a
//! named reason. A recovery can therefore never loop: the number of retries in a
//! turn is bounded by the number of flags. Pure: no I/O, no clock.

use std::time::Duration;

/// The output cap the loop starts each turn with (tokens).
pub const BASE_OUTPUT_CAP: usize = 4096;
/// The raised cap used once when a tool call was cut off by the cap.
pub const RAISED_OUTPUT_CAP: usize = 8192;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FailureClass {
    /// Network trouble, a dropped stream, or a 5xx from the model seat.
    Transient,
    /// HTTP 429.
    RateLimited,
    /// The request was larger than the model's window.
    ContextOverflow,
    /// The reply ended inside a tool call because it hit the output cap.
    OutputTruncated,
    /// HTTP 401 or 403: retrying cannot help.
    Auth,
    /// Anything else.
    Fatal,
}

/// Puts an error from `ChatClient` into a class. The client reports errors as
/// text (`"Model HTTP 429 ..."`, `"Connection failed ..."`, `"Stream error ..."`).
pub fn classify(err: &str) -> FailureClass {
    let e = err.to_ascii_lowercase();
    let http = e
        .split("model http ")
        .nth(1)
        .and_then(|rest| rest.get(..3))
        .and_then(|code| code.parse::<u16>().ok());
    let overflow = e.contains("context length")
        || e.contains("context_length")
        || e.contains("maximum context")
        || e.contains("too many tokens")
        || e.contains("prompt is too long");
    match http {
        _ if overflow => FailureClass::ContextOverflow,
        Some(413) => FailureClass::ContextOverflow,
        Some(429) => FailureClass::RateLimited,
        Some(401) | Some(403) => FailureClass::Auth,
        Some(c) if (500..600).contains(&c) => FailureClass::Transient,
        Some(_) => FailureClass::Fatal,
        None if e.contains("connection failed")
            || e.contains("stream error")
            || e.contains("timed out")
            || e.contains("timeout") =>
        {
            FailureClass::Transient
        }
        None => FailureClass::Fatal,
    }
}

/// What the loop does next.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Recovery {
    /// Ask again after waiting.
    Retry { wait: Duration },
    /// Compact the conversation, then ask again.
    CompactThenRetry,
    /// Ask again with a larger output cap. The cut-off reply is not executed.
    RaiseCapThenRetry { max_tokens: usize },
    /// Stop the turn; the reason is shown to the operator.
    Stop { reason: &'static str },
}

/// One flag per recovery; each is spent at most once per turn.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct RecoveryFlags {
    pub transient_retried: bool,
    pub rate_limit_waited: bool,
    pub compacted: bool,
    pub cap_raised: bool,
}

impl RecoveryFlags {
    /// Picks the recovery for a failure and spends its flag.
    pub fn next(&mut self, class: FailureClass) -> Recovery {
        fn once(flag: &mut bool, then: Recovery, reason: &'static str) -> Recovery {
            if *flag {
                Recovery::Stop { reason }
            } else {
                *flag = true;
                then
            }
        }
        match class {
            FailureClass::Transient => once(
                &mut self.transient_retried,
                Recovery::Retry {
                    wait: Duration::from_secs(2),
                },
                "model-unreachable: the model seat failed twice in this turn",
            ),
            FailureClass::RateLimited => once(
                &mut self.rate_limit_waited,
                Recovery::Retry {
                    wait: Duration::from_secs(20),
                },
                "rate-limited: still refused after one wait",
            ),
            FailureClass::ContextOverflow => once(
                &mut self.compacted,
                Recovery::CompactThenRetry,
                "context-overflow: still too long after compacting",
            ),
            FailureClass::OutputTruncated => once(
                &mut self.cap_raised,
                Recovery::RaiseCapThenRetry {
                    max_tokens: RAISED_OUTPUT_CAP,
                },
                "tool-call-truncated: the tool call was cut off even at the raised output limit; it was not run",
            ),
            FailureClass::Auth => Recovery::Stop {
                reason: "model-auth: the model seat refused the credentials",
            },
            FailureClass::Fatal => Recovery::Stop {
                reason: "model-error: an error with no safe recovery",
            },
        }
    }
}

/// True when the reply ends inside an unfinished `<tool_call>` block, which is
/// what a reply cut off by the output cap looks like. Such a call must never
/// run: its arguments are incomplete, and repairing them would be a guess.
pub fn tool_call_truncated(text: &str) -> bool {
    match text.rfind("<tool_call>") {
        Some(open) => !text[open..].contains("</tool_call>"),
        None => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifies_the_clients_error_texts() {
        assert_eq!(
            classify("Model HTTP 429 Too Many Requests: slow down"),
            FailureClass::RateLimited
        );
        assert_eq!(
            classify("Model HTTP 503 Service Unavailable: busy"),
            FailureClass::Transient
        );
        assert_eq!(
            classify("Model HTTP 401 Unauthorized: no"),
            FailureClass::Auth
        );
        assert_eq!(
            classify("Model HTTP 413 Payload Too Large: x"),
            FailureClass::ContextOverflow
        );
        assert_eq!(
            classify(
                "Model HTTP 400 Bad Request: This model's maximum context length is 32768 tokens"
            ),
            FailureClass::ContextOverflow
        );
        assert_eq!(
            classify("Model HTTP 400 Bad Request: bad json"),
            FailureClass::Fatal
        );
        assert_eq!(
            classify("Connection failed to model seat: refused"),
            FailureClass::Transient
        );
        assert_eq!(classify("Stream error: reset"), FailureClass::Transient);
        assert_eq!(classify("something odd"), FailureClass::Fatal);
    }

    #[test]
    fn a_forced_failure_loop_stops_after_one_recovery() {
        for class in [
            FailureClass::Transient,
            FailureClass::RateLimited,
            FailureClass::ContextOverflow,
            FailureClass::OutputTruncated,
        ] {
            let mut flags = RecoveryFlags::default();
            assert!(
                !matches!(flags.next(class), Recovery::Stop { .. }),
                "{class:?}"
            );
            assert!(
                matches!(flags.next(class), Recovery::Stop { .. }),
                "{class:?}"
            );
        }
    }

    #[test]
    fn a_classifier_that_always_fails_terminates() {
        // The same failure forever: one recovery, then a stop.
        let mut flags = RecoveryFlags::default();
        let mut n = 0;
        while !matches!(flags.next(FailureClass::Transient), Recovery::Stop { .. }) {
            n += 1;
            assert!(n <= 1, "recovery looped");
        }
        assert_eq!(n, 1);

        // Every class in turn, forever: at most one recovery per flag, then a stop.
        let classes = [
            FailureClass::Transient,
            FailureClass::RateLimited,
            FailureClass::ContextOverflow,
            FailureClass::OutputTruncated,
        ];
        let mut flags = RecoveryFlags::default();
        let mut recoveries = 0;
        for i in 0.. {
            match flags.next(classes[i % classes.len()]) {
                Recovery::Stop { .. } => break,
                _ => recoveries += 1,
            }
            assert!(recoveries <= classes.len(), "recovery looped");
        }
        assert_eq!(recoveries, classes.len());
    }

    #[test]
    fn auth_and_fatal_never_retry() {
        let mut flags = RecoveryFlags::default();
        assert!(matches!(
            flags.next(FailureClass::Auth),
            Recovery::Stop { .. }
        ));
        assert!(matches!(
            flags.next(FailureClass::Fatal),
            Recovery::Stop { .. }
        ));
    }

    #[test]
    fn detects_a_tool_call_cut_off_mid_arguments() {
        assert!(tool_call_truncated("ok <tool_call>{\"name\":\"write_to_file\",\"arguments\":{\"path\":\"a\",\"content\":\"hal"));
        assert!(!tool_call_truncated(
            "<tool_call>{\"name\":\"view_file\",\"arguments\":{}}</tool_call>"
        ));
        assert!(!tool_call_truncated("no tools here"));
        // A finished call followed by a cut-off one is still cut off.
        assert!(tool_call_truncated(
            "<tool_call>{\"name\":\"a\"}</tool_call> then <tool_call>{\"name\":\"b\",\"argu"
        ));
    }
}
