//! The worker: admits allowlisted jobs one at a time and answers in PST1.
//!
//! Reply sequence for an admitted job (worker ordinals, from 0):
//! `accepted`, `running`, then on success the receipt as consecutive
//! `text_chunk` messages, `receipt_reference` (SHA-256 of the receipt bytes)
//! and `completed` (the 32-byte output digest). The receipt is stored durably
//! before `completed` is sent. A cancel ends in `cancelled`; the deadline in
//! `failed` with `TIMEOUT` (or `INVALID_INPUT:<why>` for an input file the job
//! cannot use). A request that is not admitted gets one `failed`
//! with ordinal 0 and creates no task.

use crate::receipt::{Platform, Receipt};
use crate::store::{self, Lookup, Store};
use crate::task::{self, Abort, Control, Inputs, Limits, TaskSpec};
use crate::wire::{self, Ids, Inbound, Msg};
use aien_pi_spark_task as pst;
use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};
use std::io::{self, Read, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

pub struct Worker {
    pub key: SigningKey,
    pub limits: Limits,
    pub store: Store,
    pub platform: Platform,
}

/// Counts for one served connection.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Summary {
    pub completed: u32,
    pub replayed: u32,
    pub denied: u32,
    pub cancelled: u32,
    pub failed: u32,
    /// Frames the PST1 reader refused (no reply is possible without ids).
    pub refused_frames: u32,
}

enum Event {
    In(Inbound),
    Done(Finished),
}

struct Finished {
    task_id: [u8; 16],
    result: Result<[u8; 32], Abort>,
    started_unix_ms: u64,
    compute_ns: u64,
}

struct Running {
    ids: Ids,
    spec: TaskSpec,
    input_digest: [u8; 32],
    cancel: Arc<AtomicBool>,
    handle: std::thread::JoinHandle<()>,
    next: u32,
}

fn send(w: &mut impl Write, ids: &Ids, t: u8, ordinal: u32, payload: &[u8]) -> io::Result<()> {
    wire::write_frame(w, &wire::build(ids, t, ordinal, payload))
}

/// Receipt chunks, reference and `completed`, starting at ordinal `next`.
fn deliver(
    w: &mut impl Write,
    ids: &Ids,
    mut next: u32,
    receipt: &[u8],
    output: &[u8; 32],
) -> io::Result<()> {
    for chunk in receipt.chunks(pst::MAX_PAYLOAD) {
        send(w, ids, wire::T_TEXT_CHUNK, next, chunk)?;
        next += 1;
    }
    let reference: [u8; 32] = Sha256::digest(receipt).into();
    send(w, ids, wire::T_RECEIPT_REFERENCE, next, &reference)?;
    send(w, ids, wire::T_COMPLETED, next + 1, output)
}

fn peer_gone(e: &io::Error) -> bool {
    matches!(
        e.kind(),
        io::ErrorKind::BrokenPipe
            | io::ErrorKind::ConnectionReset
            | io::ErrorKind::ConnectionAborted
    )
}

fn unix_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl Worker {
    /// Serve one connection until the peer closes it. Returns the counts.
    pub fn serve<R: Read + Send + 'static, W: Write>(&self, r: R, mut w: W) -> io::Result<Summary> {
        let (tx, rx) = mpsc::channel::<Event>();
        let reader_tx = tx.clone();
        std::thread::spawn(move || wire::pump(r, reader_tx, Event::In));
        let mut running: Option<Running> = None;
        let mut sum = Summary::default();
        let res = self.serve_loop(&mut w, &tx, &rx, &mut running, &mut sum);
        if let Some(job) = running.take() {
            job.cancel.store(true, Ordering::Relaxed);
            let _ = job.handle.join();
        }
        match res {
            // The peer may hang up while a reply is being written (it refused
            // this worker, or gave up): that ends the session like end of input.
            Err(e) if !peer_gone(&e) => Err(e),
            _ => Ok(sum),
        }
    }

    fn serve_loop<W: Write>(
        &self,
        mut w: W,
        tx: &mpsc::Sender<Event>,
        rx: &mpsc::Receiver<Event>,
        running: &mut Option<Running>,
        sum: &mut Summary,
    ) -> io::Result<()> {
        loop {
            let ev = rx.recv().expect("worker holds a sender");
            match ev {
                Event::In(Inbound::Msg(m)) => self.on_msg(&mut w, tx, running, sum, *m)?,
                Event::In(Inbound::Refused(_)) => sum.refused_frames += 1,
                Event::In(Inbound::Eof) | Event::In(Inbound::Broken(_)) => return Ok(()),
                Event::Done(fin) => {
                    let Some(job) = running.take() else { continue };
                    if job.ids.task_id != fin.task_id {
                        *running = Some(job);
                        continue;
                    }
                    let _ = job.handle.join();
                    match fin.result {
                        Ok(output) => {
                            let receipt = Receipt {
                                task_id: job.ids.task_id,
                                idempotence_key: job.ids.idempotence_key,
                                cause_id: job.ids.cause_id,
                                kind: job.spec.kind_name().to_string(),
                                input_digest: job.input_digest,
                                output_digest: output,
                                started_unix_ms: fin.started_unix_ms,
                                compute_ns: fin.compute_ns,
                                threads: self.limits.max_threads as u64,
                                platform: self.platform.clone(),
                            }
                            .sign(&self.key);
                            if let Err(e) = self.store.put(&job.ids.idempotence_key, &receipt) {
                                sum.failed += 1;
                                send(
                                    &mut w,
                                    &job.ids,
                                    wire::T_FAILED,
                                    job.next,
                                    format!("STORE_ERROR:{e}").as_bytes(),
                                )?;
                            } else {
                                sum.completed += 1;
                                deliver(&mut w, &job.ids, job.next, &receipt, &output)?;
                            }
                        }
                        Err(Abort::Cancelled) => {
                            sum.cancelled += 1;
                            send(&mut w, &job.ids, wire::T_CANCELLED, job.next, &[])?;
                        }
                        Err(Abort::Timeout) => {
                            sum.failed += 1;
                            send(&mut w, &job.ids, wire::T_FAILED, job.next, b"TIMEOUT")?;
                        }
                        Err(Abort::Invalid(why)) => {
                            sum.failed += 1;
                            let reason = format!("INVALID_INPUT:{why}");
                            send(
                                &mut w,
                                &job.ids,
                                wire::T_FAILED,
                                job.next,
                                reason.as_bytes(),
                            )?;
                        }
                    }
                }
            }
        }
    }

    fn on_msg<W2: Write>(
        &self,
        w: &mut W2,
        tx: &mpsc::Sender<Event>,
        running: &mut Option<Running>,
        sum: &mut Summary,
        m: Msg,
    ) -> io::Result<()> {
        let ids = Ids::of(&m.header);
        let t = m.header.msg_type;
        if !wire::requester_type(t) {
            return deny(
                w,
                sum,
                &ids,
                format!("REFUSED:{}", pst::Refusal::WrongSide.code()),
            );
        }
        match t {
            wire::T_CANCEL_REQUEST => {
                if let Some(job) = running.as_ref().filter(|j| j.ids.task_id == ids.task_id) {
                    job.cancel.store(true, Ordering::Relaxed);
                }
                Ok(())
            }
            wire::T_RESUME_REQUEST => deny(w, sum, &ids, "UNSUPPORTED:resume_request"),
            _ => {
                if let Some(job) = running.as_ref() {
                    if job.ids.task_id == ids.task_id
                        && job.ids.idempotence_key == ids.idempotence_key
                    {
                        return Ok(()); // the same request again while it runs
                    }
                    return deny(w, sum, &ids, "BUSY");
                }
                if m.header.effect_class != wire::EFFECT_COMPUTE {
                    return deny(w, sum, &ids, "DENIED_EFFECT");
                }
                let spec = match TaskSpec::decode(&m.payload) {
                    Ok(s) => s,
                    Err(e) => return deny(w, sum, &ids, format!("DENIED_TASK:{e:?}")),
                };
                if let Err(d) = spec.admit(&self.limits) {
                    return deny(w, sum, &ids, d.code());
                }
                let input_digest = m.header.payload_digest;
                match self.store.lookup(
                    &ids.idempotence_key,
                    &ids.task_id,
                    &input_digest,
                    &self.key.verifying_key(),
                ) {
                    Ok(Lookup::Missing) => {}
                    Ok(Lookup::Same(receipt)) => {
                        let (r, _) = Receipt::verify(&receipt, &[self.key.verifying_key()])
                            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e.code()))?;
                        sum.replayed += 1;
                        send(w, &ids, wire::T_ACCEPTED, 0, &[])?;
                        return deliver(w, &ids, 1, &receipt, &r.output_digest);
                    }
                    Ok(Lookup::Conflict) => return deny(w, sum, &ids, "DUPLICATE_ID"),
                    Err(e) => return deny(w, sum, &ids, format!("STORE_REFUSED:{e}")),
                }
                let inputs = match self.load_inputs(&spec) {
                    Ok(i) => i,
                    Err(reason) => return deny(w, sum, &ids, reason),
                };
                send(w, &ids, wire::T_ACCEPTED, 0, &[])?;
                send(w, &ids, wire::T_RUNNING, 1, &[])?;
                let cancel = Arc::new(AtomicBool::new(false));
                let (flag, done, limits, task_id) =
                    (cancel.clone(), tx.clone(), self.limits, ids.task_id);
                let handle = std::thread::spawn(move || {
                    let started_unix_ms = unix_ms();
                    let t0 = Instant::now();
                    let ctl = Control {
                        cancel: &flag,
                        deadline: t0 + limits.max_wall,
                        threads: limits.max_threads,
                    };
                    let result = task::run(&spec, &inputs, &ctl);
                    let compute_ns = t0.elapsed().as_nanos() as u64;
                    let _ = done.send(Event::Done(Finished {
                        task_id,
                        result,
                        started_unix_ms,
                        compute_ns,
                    }));
                });
                *running = Some(Running {
                    ids,
                    spec,
                    input_digest,
                    cancel,
                    handle,
                    next: 2,
                });
                Ok(())
            }
        }
    }

    /// The job's input files: sizes admitted first, then each read and
    /// checked against its digest. Err is the denial reason.
    fn load_inputs(&self, spec: &TaskSpec) -> Result<Inputs, String> {
        let files = spec.input_files();
        if files.is_empty() {
            return Ok(Inputs::default());
        }
        let dir = self.store.dir();
        let sizes = files
            .iter()
            .map(|d| store::blob_size(dir, d).map_err(|e| e.code(d)))
            .collect::<Result<Vec<u64>, String>>()?;
        spec.admit_sized(&sizes, &self.limits)
            .map_err(|d| d.code())?;
        let mut loaded = files
            .iter()
            .map(|d| store::load_blob(dir, d).map_err(|e| e.code(d)))
            .collect::<Result<Vec<Vec<u8>>, String>>()?;
        let corpus = loaded.pop().unwrap_or_default();
        let tokenizer = loaded.pop().unwrap_or_default();
        Ok(Inputs { tokenizer, corpus })
    }
}

/// Refuse one request: a single `failed` with ordinal 0, no task created.
fn deny<W: Write>(
    w: &mut W,
    sum: &mut Summary,
    ids: &Ids,
    reason: impl AsRef<str>,
) -> io::Result<()> {
    sum.denied += 1;
    send(w, ids, wire::T_FAILED, 0, reason.as_ref().as_bytes())
}
