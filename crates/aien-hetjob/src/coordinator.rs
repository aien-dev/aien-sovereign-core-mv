//! The coordinator: sends one job, reads the worker's replies as hints,
//! verifies the receipt and checks the output against its own recompute.
//!
//! The coordinator owns the job: its deadline, the open task and the
//! connection. When the deadline passes it sends `cancel_request`, waits a
//! short grace, and otherwise reports `Uncertain`; the caller then closes the
//! connection (for SSH, kills the child). A drop before a final answer is
//! `Uncertain`, never `Failed` or `Cancelled` (PST1 section 8).

use crate::receipt::{Receipt, ReceiptError, MAX_RECEIPT_BYTES};
use crate::task::TaskSpec;
use crate::wire::{self, Ids, Inbound};
use ed25519_dalek::VerifyingKey;
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

/// One job to send.
#[derive(Debug, Clone, Copy)]
pub struct Job {
    pub spec: TaskSpec,
    pub ids: Ids,
}

/// How long to wait, and the coordinator's own answer if it has one.
#[derive(Debug, Clone, Copy)]
pub struct Policy {
    pub deadline: Duration,
    pub cancel_grace: Duration,
    /// The coordinator's recompute of the output digest (`task::oracle`).
    /// `None` accepts a verified receipt without an output check and says so.
    pub oracle: Option<[u8; 32]>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OracleCheck {
    Matched,
    NotChecked,
}

#[derive(Debug, Clone)]
pub struct Verified {
    pub receipt: Receipt,
    pub worker_key: VerifyingKey,
    pub output: [u8; 32],
    pub oracle: OracleCheck,
}

#[derive(Debug, Clone)]
pub enum Outcome {
    Verified(Box<Verified>),
    /// The worker reported `failed` with this reason.
    Failed(String),
    Cancelled,
    /// No final answer: disconnect, deadline, or the worker said uncertain.
    Uncertain(String),
    /// The coordinator refused what the worker sent.
    Rejected(String),
}

/// Counts that do not change the outcome.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Stats {
    /// Messages for a different task or idempotence key, ignored.
    pub stale_ignored: u32,
    pub frames: u32,
    pub round_trip: Duration,
}

fn reject(why: impl Into<String>) -> Option<Outcome> {
    Some(Outcome::Rejected(why.into()))
}

struct Progress<'a> {
    job: &'a Job,
    roster: &'a [VerifyingKey],
    oracle: Option<[u8; 32]>,
    input_digest: [u8; 32],
    expect: u32,
    receipt_bytes: Vec<u8>,
    receipt: Option<(Receipt, VerifyingKey)>,
}

impl Progress<'_> {
    /// Some(outcome) ends the job.
    fn step(&mut self, m: wire::Msg, stats: &mut Stats) -> Option<Outcome> {
        let h = &m.header;
        let ids = &self.job.ids;
        if h.task_id != ids.task_id || h.idempotence_key != ids.idempotence_key {
            stats.stale_ignored += 1;
            return None;
        }
        if wire::requester_type(h.msg_type) {
            return reject(format!("WRONG_SIDE:type {}", h.msg_type));
        }
        if h.cause_id != ids.cause_id || h.objective_id != ids.objective_id {
            return reject("HEADER_MISMATCH");
        }
        if h.ordinal != self.expect {
            return reject(format!(
                "OUT_OF_ORDER:expected {} got {}",
                self.expect, h.ordinal
            ));
        }
        self.expect += 1;
        match h.msg_type {
            wire::T_ACCEPTED | wire::T_RUNNING | 3 => None,
            wire::T_TEXT_CHUNK => {
                if self.receipt.is_some() {
                    return reject("TEXT_AFTER_RECEIPT");
                }
                self.receipt_bytes.extend_from_slice(&m.payload);
                if self.receipt_bytes.len() > MAX_RECEIPT_BYTES {
                    return reject(ReceiptError::TooLarge(self.receipt_bytes.len()).code());
                }
                None
            }
            wire::T_RECEIPT_REFERENCE => {
                let digest: [u8; 32] = Sha256::digest(&self.receipt_bytes).into();
                if m.payload != digest {
                    return reject("RECEIPT_REFERENCE_MISMATCH");
                }
                let (r, key) = match Receipt::verify(&self.receipt_bytes, self.roster) {
                    Ok(v) => v,
                    Err(e) => return reject(e.code()),
                };
                let binds = [
                    ("task_id", r.task_id == ids.task_id),
                    ("idempotence_key", r.idempotence_key == ids.idempotence_key),
                    ("cause_id", r.cause_id == ids.cause_id),
                    ("input_digest", r.input_digest == self.input_digest),
                    ("kind", r.kind == self.job.spec.kind_name()),
                ];
                if let Some((field, _)) = binds.iter().find(|(_, ok)| !ok) {
                    return reject(format!("RECEIPT_MISMATCH:{field}"));
                }
                self.receipt = Some((r, key));
                None
            }
            wire::T_COMPLETED => {
                let Some((r, key)) = self.receipt.take() else {
                    return reject("COMPLETED_WITHOUT_RECEIPT");
                };
                if m.payload != r.output_digest {
                    return reject("RECEIPT_MISMATCH:output_digest");
                }
                let oracle = match self.oracle {
                    Some(o) if o == r.output_digest => OracleCheck::Matched,
                    Some(_) => return reject("ORACLE_MISMATCH"),
                    None => OracleCheck::NotChecked,
                };
                let output = r.output_digest;
                Some(Outcome::Verified(Box::new(Verified {
                    receipt: r,
                    worker_key: key,
                    output,
                    oracle,
                })))
            }
            wire::T_FAILED => Some(Outcome::Failed(
                String::from_utf8_lossy(&m.payload).into_owned(),
            )),
            wire::T_CANCELLED => Some(Outcome::Cancelled),
            10 => Some(Outcome::Uncertain("WORKER_UNCERTAIN".to_string())),
            other => reject(format!("UNEXPECTED_TYPE:{other}")),
        }
    }
}

/// Send `job` on `w`, read replies from `r`, return the outcome.
pub fn dispatch<R: Read + Send + 'static, W: Write>(
    r: R,
    w: &mut W,
    job: &Job,
    roster: &[VerifyingKey],
    policy: &Policy,
) -> io::Result<(Outcome, Stats)> {
    let payload = job.spec.encode();
    let (tx, rx) = mpsc::channel::<Inbound>();
    std::thread::spawn(move || wire::pump(r, tx, |i| i));
    let t0 = Instant::now();
    wire::write_frame(w, &wire::build(&job.ids, wire::T_REQUEST, 0, &payload))?;
    let mut p = Progress {
        job,
        roster,
        oracle: policy.oracle,
        input_digest: Sha256::digest(&payload).into(),
        expect: 0,
        receipt_bytes: Vec::new(),
        receipt: None,
    };
    let mut stats = Stats::default();
    let mut deadline = t0 + policy.deadline;
    let mut cancel_sent = false;
    let outcome = loop {
        let wait = deadline.saturating_duration_since(Instant::now());
        match rx.recv_timeout(wait) {
            Ok(Inbound::Msg(m)) => {
                stats.frames += 1;
                if let Some(o) = p.step(*m, &mut stats) {
                    break o;
                }
            }
            Ok(Inbound::Refused(code)) => break Outcome::Rejected(format!("PST1:{}", code.code())),
            Ok(Inbound::Eof) => break Outcome::Uncertain("DISCONNECTED".to_string()),
            Ok(Inbound::Broken(e)) => break Outcome::Uncertain(format!("DISCONNECTED:{e}")),
            Err(RecvTimeoutError::Disconnected) => {
                break Outcome::Uncertain("DISCONNECTED".to_string())
            }
            Err(RecvTimeoutError::Timeout) if !cancel_sent => {
                cancel_sent = true;
                deadline = Instant::now() + policy.cancel_grace;
                let cancel = wire::build(&job.ids, wire::T_CANCEL_REQUEST, 1, &[]);
                if wire::write_frame(w, &cancel).is_err() {
                    break Outcome::Uncertain("DEADLINE:cancel not delivered".to_string());
                }
            }
            Err(RecvTimeoutError::Timeout) => {
                break Outcome::Uncertain("DEADLINE:no answer after cancel".to_string())
            }
        }
    };
    stats.round_trip = t0.elapsed();
    Ok((outcome, stats))
}
