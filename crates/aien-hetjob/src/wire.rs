//! PST1 messages on a byte stream.
//!
//! The stream (SSH stdio, or a socket pair in tests) carries one PST1 message
//! per frame: a u16 big-endian length, then exactly that many bytes, at most
//! `pst::MAX_MESSAGE` (1024). That is the M6-A rule of one PST1 message per
//! transport message; nothing is reassembled across frames.
//!
//! Roles: the coordinator sends PST1's requester types (`request`,
//! `cancel_request`); the worker sends the executor types. PST1 Draft v0.1
//! names those sides "Pi" and "Spark"; here the Spark is the requester. Every
//! status the worker sends is a hint to the coordinator, which owns the open
//! job, the deadline and the connection.

use aien_pi_spark_task as pst;
use std::io::{self, Read, Write};

pub const T_REQUEST: u8 = 1;
pub const T_ACCEPTED: u8 = 2;
pub const T_RUNNING: u8 = 4;
pub const T_TEXT_CHUNK: u8 = 5;
pub const T_COMPLETED: u8 = 7;
pub const T_FAILED: u8 = 8;
pub const T_CANCELLED: u8 = 9;
pub const T_RECEIPT_REFERENCE: u8 = 12;
pub const T_CANCEL_REQUEST: u8 = 13;
pub const T_RESUME_REQUEST: u8 = 14;

/// PST1 `effect_class` COMPUTE: the only class a worker admits.
pub const EFFECT_COMPUTE: u8 = 1;

/// True for the types only the requester (coordinator) may send.
pub fn requester_type(t: u8) -> bool {
    matches!(t, T_REQUEST | T_CANCEL_REQUEST | T_RESUME_REQUEST)
}

/// The identifiers one job carries in every PST1 header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Ids {
    pub cause_id: [u8; 32],
    pub origin_run_id: [u8; 32],
    pub objective_id: [u8; 32],
    pub subject_id: [u8; 32],
    pub task_id: [u8; 16],
    pub idempotence_key: [u8; 16],
}

impl Ids {
    pub fn of(h: &pst::Header) -> Ids {
        Ids {
            cause_id: h.cause_id,
            origin_run_id: h.origin_run_id,
            objective_id: h.objective_id,
            subject_id: h.subject_id,
            task_id: h.task_id,
            idempotence_key: h.idempotence_key,
        }
    }
}

/// A decoded message that owns its payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Msg {
    pub header: pst::Header,
    pub payload: Vec<u8>,
}

/// Build one PST1 message.
pub fn build(ids: &Ids, msg_type: u8, ordinal: u32, payload: &[u8]) -> Vec<u8> {
    assert!(
        payload.len() <= pst::MAX_PAYLOAD,
        "PST1 payload above 800 bytes"
    );
    let h = pst::Header {
        version: pst::VERSION,
        msg_type,
        cause_id: ids.cause_id,
        origin_run_id: ids.origin_run_id,
        objective_id: ids.objective_id,
        subject_id: ids.subject_id,
        parent_task_id: [0; 16],
        task_id: ids.task_id,
        idempotence_key: ids.idempotence_key,
        ordinal,
        effect_class: EFFECT_COMPUTE,
        payload_len: payload.len() as u16,
        payload_digest: pst::payload_digest(payload),
    };
    let mut out = vec![0u8; pst::HEADER_LEN + payload.len()];
    let n = pst::encode(&h, payload, &mut out).expect("PST1 encode of a bounded payload");
    out.truncate(n);
    out
}

/// Write one frame and flush.
pub fn write_frame(w: &mut impl Write, msg: &[u8]) -> io::Result<()> {
    if msg.len() > pst::MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "frame above 1024 bytes",
        ));
    }
    w.write_all(&(msg.len() as u16).to_be_bytes())?;
    w.write_all(msg)?;
    w.flush()
}

/// Read one frame. `Ok(None)` is a clean end of stream at a frame boundary.
/// A declared length above 1024 is an error before any body byte is read.
pub fn read_frame(r: &mut impl Read) -> io::Result<Option<Vec<u8>>> {
    let mut len = [0u8; 2];
    match r.read(&mut len[..1])? {
        0 => return Ok(None),
        _ => r.read_exact(&mut len[1..])?,
    }
    let n = usize::from(u16::from_be_bytes(len));
    if n > pst::MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("frame length {n} above 1024"),
        ));
    }
    let mut body = vec![0u8; n];
    r.read_exact(&mut body)?;
    Ok(Some(body))
}

/// Decode a frame with the PST1 reader; the refusal code is the spec's.
pub fn decode(frame: &[u8]) -> Result<Msg, pst::Refusal> {
    let m = pst::decode(frame)?;
    Ok(Msg {
        header: m.header,
        payload: m.payload.to_vec(),
    })
}

/// What a reader thread hands to its owner.
#[derive(Debug)]
pub enum Inbound {
    Msg(Box<Msg>),
    Refused(pst::Refusal),
    Eof,
    Broken(String),
}

/// Read frames until end of stream, decoding each, and send them on `tx`.
/// Stops after the first end, I/O error or closed channel.
pub fn pump<T>(mut r: impl Read, tx: std::sync::mpsc::Sender<T>, wrap: impl Fn(Inbound) -> T) {
    loop {
        let item = match read_frame(&mut r) {
            Ok(Some(f)) => match decode(&f) {
                Ok(m) => Inbound::Msg(Box::new(m)),
                Err(code) => Inbound::Refused(code),
            },
            Ok(None) => Inbound::Eof,
            Err(e) => Inbound::Broken(e.to_string()),
        };
        let last = matches!(item, Inbound::Eof | Inbound::Broken(_));
        if tx.send(wrap(item)).is_err() || last {
            return;
        }
    }
}
