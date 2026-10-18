//! The allowlist: the only jobs a worker runs, their request payloads, the
//! admission arithmetic, the worker's computation and the coordinator's
//! independent recompute.
//!
//! Every kind is integer-exact (wrapping u32 arithmetic, SHA-256), so an
//! x86_64 and an aarch64 machine must produce the same digest bit for bit.
//! There is no floating point anywhere in this file.

use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Fixed-shape matrix product C = A x B over wrapping u32, inputs generated
/// from a seed. Payload: kind, m, k, n (u16 BE each), seed (u64 BE).
pub const KIND_MATMUL_U32: u8 = 1;
/// SHA-256 applied `rounds` times to a 32-byte seed. Payload: kind,
/// rounds (u32 BE), seed (32 bytes).
pub const KIND_DIGEST_CHAIN: u8 = 2;

const MATMUL_LEN: usize = 1 + 2 + 2 + 2 + 8;
const CHAIN_LEN: usize = 1 + 4 + 32;
const MATMUL_DOMAIN: &[u8] = b"AIEN-HETJOB-MATMUL-U32-V1";

/// How often (in inner work items) the worker looks at the cancel flag and
/// the deadline.
const CHECK_EVERY: u32 = 4096;

/// One allowlisted job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskSpec {
    MatmulU32 { m: u16, k: u16, n: u16, seed: u64 },
    DigestChain { rounds: u32, seed: [u8; 32] },
}

/// Why a request payload is not an allowlisted job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpecError {
    /// Empty payload.
    Empty,
    /// A kind byte that is not on the allowlist.
    UnknownKind(u8),
    /// The payload length does not match the kind.
    BadLength,
}

/// Why admission refused a well-formed job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Denial {
    /// A matrix dimension or the round count is zero.
    Empty,
    /// Admission arithmetic says the job needs more memory than allowed.
    OverMemory { need: u64, limit: u64 },
    /// More multiply-adds or hash rounds than allowed.
    OverWork { need: u64, limit: u64 },
}

impl Denial {
    pub fn code(&self) -> String {
        match self {
            Denial::Empty => "OVER_BUDGET:EMPTY".to_string(),
            Denial::OverMemory { need, limit } => {
                format!("OVER_BUDGET:MEMORY need={need} limit={limit}")
            }
            Denial::OverWork { need, limit } => {
                format!("OVER_BUDGET:WORK need={need} limit={limit}")
            }
        }
    }
}

/// Worker ceilings. `max_memory_bytes` and `max_work_units` are admission
/// arithmetic checked before anything is allocated; they are not a kernel
/// enforced ceiling (see the crate README for what the OS enforces).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Limits {
    pub max_memory_bytes: u64,
    pub max_work_units: u64,
    pub max_wall: Duration,
    pub max_threads: usize,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_memory_bytes: 512 << 20,
            max_work_units: 1 << 34,
            max_wall: Duration::from_secs(120),
            max_threads: 4,
        }
    }
}

/// Why the worker stopped before finishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Abort {
    Cancelled,
    Timeout,
}

/// What the worker's computation may look at while it runs.
pub struct Control<'a> {
    pub cancel: &'a AtomicBool,
    pub deadline: Instant,
    pub threads: usize,
}

impl Control<'_> {
    fn stop(&self) -> Option<Abort> {
        if self.cancel.load(Ordering::Relaxed) {
            Some(Abort::Cancelled)
        } else if Instant::now() >= self.deadline {
            Some(Abort::Timeout)
        } else {
            None
        }
    }
}

impl TaskSpec {
    /// The request payload (well under PST1's 800 bytes).
    pub fn encode(&self) -> Vec<u8> {
        match *self {
            TaskSpec::MatmulU32 { m, k, n, seed } => {
                let mut v = Vec::with_capacity(MATMUL_LEN);
                v.push(KIND_MATMUL_U32);
                v.extend_from_slice(&m.to_be_bytes());
                v.extend_from_slice(&k.to_be_bytes());
                v.extend_from_slice(&n.to_be_bytes());
                v.extend_from_slice(&seed.to_be_bytes());
                v
            }
            TaskSpec::DigestChain { rounds, seed } => {
                let mut v = Vec::with_capacity(CHAIN_LEN);
                v.push(KIND_DIGEST_CHAIN);
                v.extend_from_slice(&rounds.to_be_bytes());
                v.extend_from_slice(&seed);
                v
            }
        }
    }

    pub fn decode(b: &[u8]) -> Result<TaskSpec, SpecError> {
        let kind = *b.first().ok_or(SpecError::Empty)?;
        match kind {
            KIND_MATMUL_U32 => {
                if b.len() != MATMUL_LEN {
                    return Err(SpecError::BadLength);
                }
                let u16_at = |i: usize| u16::from_be_bytes([b[i], b[i + 1]]);
                let mut seed = [0u8; 8];
                seed.copy_from_slice(&b[7..15]);
                Ok(TaskSpec::MatmulU32 {
                    m: u16_at(1),
                    k: u16_at(3),
                    n: u16_at(5),
                    seed: u64::from_be_bytes(seed),
                })
            }
            KIND_DIGEST_CHAIN => {
                if b.len() != CHAIN_LEN {
                    return Err(SpecError::BadLength);
                }
                let mut seed = [0u8; 32];
                seed.copy_from_slice(&b[5..37]);
                Ok(TaskSpec::DigestChain {
                    rounds: u32::from_be_bytes([b[1], b[2], b[3], b[4]]),
                    seed,
                })
            }
            other => Err(SpecError::UnknownKind(other)),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            TaskSpec::MatmulU32 { .. } => "matmul_u32",
            TaskSpec::DigestChain { .. } => "digest_chain",
        }
    }

    /// Bytes the worker allocates for this job: A, B and C for the matrix
    /// product, nothing that scales for the digest chain.
    pub fn memory_bytes(&self) -> u64 {
        match *self {
            TaskSpec::MatmulU32 { m, k, n, .. } => {
                let (m, k, n) = (u64::from(m), u64::from(k), u64::from(n));
                4 * (m * k + k * n + m * n)
            }
            TaskSpec::DigestChain { .. } => 0,
        }
    }

    /// Multiply-adds for the matrix product, hash rounds for the chain.
    pub fn work_units(&self) -> u64 {
        match *self {
            TaskSpec::MatmulU32 { m, k, n, .. } => u64::from(m) * u64::from(k) * u64::from(n),
            TaskSpec::DigestChain { rounds, .. } => u64::from(rounds),
        }
    }

    /// Admission: checked before any allocation.
    pub fn admit(&self, limits: &Limits) -> Result<(), Denial> {
        if self.work_units() == 0 {
            return Err(Denial::Empty);
        }
        let need = self.memory_bytes();
        if need > limits.max_memory_bytes {
            return Err(Denial::OverMemory {
                need,
                limit: limits.max_memory_bytes,
            });
        }
        let need = self.work_units();
        if need > limits.max_work_units {
            return Err(Denial::OverWork {
                need,
                limit: limits.max_work_units,
            });
        }
        Ok(())
    }
}

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// A (m x k, row major) then B (k x n, row major) from one splitmix64 stream.
pub fn matmul_inputs(m: usize, k: usize, n: usize, seed: u64) -> (Vec<u32>, Vec<u32>) {
    let mut s = seed;
    let a = (0..m * k).map(|_| splitmix64(&mut s) as u32).collect();
    let b = (0..k * n).map(|_| splitmix64(&mut s) as u32).collect();
    (a, b)
}

fn matmul_digest(m: u16, k: u16, n: u16, seed: u64, c: &[u32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(MATMUL_DOMAIN);
    h.update(m.to_be_bytes());
    h.update(k.to_be_bytes());
    h.update(n.to_be_bytes());
    h.update(seed.to_be_bytes());
    for x in c {
        h.update(x.to_be_bytes());
    }
    h.finalize().into()
}

/// The worker's computation: row bands across threads, i-k-j order inside a
/// band, cancel flag and deadline checked between rows.
pub fn run(spec: &TaskSpec, ctl: &Control) -> Result<[u8; 32], Abort> {
    match *spec {
        TaskSpec::MatmulU32 { m, k, n, seed } => {
            let (mu, ku, nu) = (usize::from(m), usize::from(k), usize::from(n));
            let (a, b) = matmul_inputs(mu, ku, nu, seed);
            let mut c = vec![0u32; mu * nu];
            let threads = ctl.threads.clamp(1, mu.max(1));
            let band = mu.div_ceil(threads);
            let abort = std::thread::scope(|s| {
                let mut handles = Vec::new();
                for (t, rows) in c.chunks_mut(band * nu).enumerate() {
                    let (a, b) = (&a, &b);
                    handles.push(s.spawn(move || {
                        let i0 = t * band;
                        for (di, crow) in rows.chunks_mut(nu).enumerate() {
                            if let Some(why) = ctl.stop() {
                                return Some(why);
                            }
                            let arow = &a[(i0 + di) * ku..(i0 + di + 1) * ku];
                            for (p, &aip) in arow.iter().enumerate() {
                                let brow = &b[p * nu..(p + 1) * nu];
                                for (cij, &bpj) in crow.iter_mut().zip(brow) {
                                    *cij = cij.wrapping_add(aip.wrapping_mul(bpj));
                                }
                            }
                        }
                        None
                    }));
                }
                handles
                    .into_iter()
                    .filter_map(|h| h.join().expect("matmul band panicked"))
                    .next()
            });
            match abort {
                Some(why) => Err(why),
                None => Ok(matmul_digest(m, k, n, seed, &c)),
            }
        }
        TaskSpec::DigestChain { rounds, seed } => {
            let mut h = seed;
            for i in 0..rounds {
                if i % CHECK_EVERY == 0 {
                    if let Some(why) = ctl.stop() {
                        return Err(why);
                    }
                }
                h = Sha256::digest(h).into();
            }
            Ok(h)
        }
    }
}

/// The coordinator's recompute. The matrix product uses a different loop
/// order (i-j-k dot products, one thread) from the worker, so a shared bug in
/// the blocked loop does not hide. The digest chain has only one algorithm;
/// its independence is a different machine and a different build, nothing
/// more.
pub fn oracle(spec: &TaskSpec) -> [u8; 32] {
    match *spec {
        TaskSpec::MatmulU32 { m, k, n, seed } => {
            let (mu, ku, nu) = (usize::from(m), usize::from(k), usize::from(n));
            let (a, b) = matmul_inputs(mu, ku, nu, seed);
            let mut c = vec![0u32; mu * nu];
            for i in 0..mu {
                for j in 0..nu {
                    let mut acc = 0u32;
                    for p in 0..ku {
                        acc = acc.wrapping_add(a[i * ku + p].wrapping_mul(b[p * nu + j]));
                    }
                    c[i * nu + j] = acc;
                }
            }
            matmul_digest(m, k, n, seed, &c)
        }
        TaskSpec::DigestChain { rounds, seed } => {
            let mut h = seed;
            for _ in 0..rounds {
                h = Sha256::digest(h).into();
            }
            h
        }
    }
}
