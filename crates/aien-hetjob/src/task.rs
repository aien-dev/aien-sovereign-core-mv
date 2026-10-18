//! The allowlist: the only jobs a worker runs, their request payloads, the
//! admission arithmetic, the worker's computation and the coordinator's
//! independent recompute.
//!
//! Every kind is integer-exact (wrapping u32 arithmetic, SHA-256, BPE token
//! ids), so an x86_64 and an aarch64 machine must produce the same digest bit
//! for bit. There is no floating point anywhere in this file.

use aien_bitnet_tokenizer::BitNetTokenizer;
use sha2::{Digest, Sha256};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Fixed-shape matrix product C = A x B over wrapping u32, inputs generated
/// from a seed. Payload: kind, m, k, n (u16 BE each), seed (u64 BE).
pub const KIND_MATMUL_U32: u8 = 1;
/// SHA-256 applied `rounds` times to a 32-byte seed. Payload: kind,
/// rounds (u32 BE), seed (32 bytes).
pub const KIND_DIGEST_CHAIN: u8 = 2;
/// Tokenize every line of a corpus with our own BitNet BPE. Payload: kind,
/// tokenizer.json sha256 (32), corpus sha256 (32), flags (bit 0: prepend
/// BOS). Both files are read by digest from the worker's blob folder.
pub const KIND_TOKENIZE_BATCH: u8 = 3;

const MATMUL_LEN: usize = 1 + 2 + 2 + 2 + 8;
const CHAIN_LEN: usize = 1 + 4 + 32;
const TOKENIZE_LEN: usize = 1 + 32 + 32 + 1;
const MATMUL_DOMAIN: &[u8] = b"AIEN-HETJOB-MATMUL-U32-V1";
const TOKENIZE_DOMAIN: &[u8] = b"AIEN-HETJOB-TOKENIZE-BPE-V1";

/// Admission estimate for tokenize_batch, in bytes per input byte: the parsed
/// vocabulary and merge table per tokenizer.json byte, and the token ids and
/// line index per corpus byte. Calibrated from peak RSS (crate README).
const TOKENIZER_BYTES_PER_BYTE: u64 = 16;
const CORPUS_BYTES_PER_BYTE: u64 = 8;

/// How often (in inner work items) the worker looks at the cancel flag and
/// the deadline.
const CHECK_EVERY: u32 = 4096;

/// One allowlisted job.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskSpec {
    MatmulU32 {
        m: u16,
        k: u16,
        n: u16,
        seed: u64,
    },
    DigestChain {
        rounds: u32,
        seed: [u8; 32],
    },
    TokenizeBatch {
        tokenizer: [u8; 32],
        corpus: [u8; 32],
        bos: bool,
    },
}

/// The files a job reads, already checked against the digests in its
/// request (empty for the kinds that generate their own inputs).
#[derive(Debug, Clone, Default)]
pub struct Inputs {
    pub tokenizer: Vec<u8>,
    pub corpus: Vec<u8>,
}

/// What a tokenize_batch run produced, from the coordinator's recompute.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizeSummary {
    pub lines: u64,
    pub tokens: u64,
    /// Lines whose ids decode back to exactly the line (a second code path).
    pub round_trips: u64,
    pub digest: [u8; 32],
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
    /// A flags byte with bits this version does not define.
    BadFlags,
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
    /// Largest single input file a job may name.
    pub max_blob_bytes: u64,
}

impl Default for Limits {
    fn default() -> Self {
        Limits {
            max_memory_bytes: 512 << 20,
            max_work_units: 1 << 34,
            max_wall: Duration::from_secs(120),
            max_threads: 4,
            max_blob_bytes: 64 << 20,
        }
    }
}

/// Why the worker stopped before finishing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Abort {
    Cancelled,
    Timeout,
    /// An input file passed its digest check but is not usable (for example a
    /// tokenizer.json our BPE refuses, or a corpus that is not UTF-8).
    Invalid(&'static str),
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
            TaskSpec::TokenizeBatch {
                tokenizer,
                corpus,
                bos,
            } => {
                let mut v = Vec::with_capacity(TOKENIZE_LEN);
                v.push(KIND_TOKENIZE_BATCH);
                v.extend_from_slice(&tokenizer);
                v.extend_from_slice(&corpus);
                v.push(u8::from(bos));
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
            KIND_TOKENIZE_BATCH => {
                if b.len() != TOKENIZE_LEN {
                    return Err(SpecError::BadLength);
                }
                let flags = b[65];
                if flags > 1 {
                    return Err(SpecError::BadFlags);
                }
                let mut tokenizer = [0u8; 32];
                let mut corpus = [0u8; 32];
                tokenizer.copy_from_slice(&b[1..33]);
                corpus.copy_from_slice(&b[33..65]);
                Ok(TaskSpec::TokenizeBatch {
                    tokenizer,
                    corpus,
                    bos: flags == 1,
                })
            }
            other => Err(SpecError::UnknownKind(other)),
        }
    }

    pub fn kind_name(&self) -> &'static str {
        match self {
            TaskSpec::MatmulU32 { .. } => "matmul_u32",
            TaskSpec::DigestChain { .. } => "digest_chain",
            TaskSpec::TokenizeBatch { .. } => "tokenize_batch",
        }
    }

    /// Digests of the files this job reads, in `Inputs` order.
    pub fn input_files(&self) -> Vec<[u8; 32]> {
        match *self {
            TaskSpec::TokenizeBatch {
                tokenizer, corpus, ..
            } => vec![tokenizer, corpus],
            _ => Vec::new(),
        }
    }

    /// Bytes the worker allocates for this job, given the sizes of its input
    /// files (`input_files` order): A, B and C for the matrix product,
    /// nothing that scales for the digest chain, the estimate above for
    /// tokenize_batch.
    pub fn memory_bytes(&self, sizes: &[u64]) -> u64 {
        match *self {
            TaskSpec::MatmulU32 { m, k, n, .. } => {
                let (m, k, n) = (u64::from(m), u64::from(k), u64::from(n));
                4 * (m * k + k * n + m * n)
            }
            TaskSpec::DigestChain { .. } => 0,
            TaskSpec::TokenizeBatch { .. } => {
                let (t, c) = (size(sizes, 0), size(sizes, 1));
                t.saturating_mul(TOKENIZER_BYTES_PER_BYTE)
                    .saturating_add(c.saturating_mul(CORPUS_BYTES_PER_BYTE))
            }
        }
    }

    /// Multiply-adds for the matrix product, hash rounds for the chain,
    /// corpus bytes for tokenize_batch.
    pub fn work_units(&self, sizes: &[u64]) -> u64 {
        match *self {
            TaskSpec::MatmulU32 { m, k, n, .. } => u64::from(m) * u64::from(k) * u64::from(n),
            TaskSpec::DigestChain { rounds, .. } => u64::from(rounds),
            TaskSpec::TokenizeBatch { .. } => size(sizes, 1),
        }
    }

    /// Admission for the request alone, before any file is opened. A kind
    /// with input files is admitted again with their sizes (`admit_sized`).
    pub fn admit(&self, limits: &Limits) -> Result<(), Denial> {
        if self.input_files().is_empty() {
            self.admit_sized(&[], limits)
        } else {
            Ok(())
        }
    }

    /// Admission with the input file sizes: checked before any allocation
    /// and before any input file is read.
    pub fn admit_sized(&self, sizes: &[u64], limits: &Limits) -> Result<(), Denial> {
        if let Some(&big) = sizes.iter().find(|&&s| s > limits.max_blob_bytes) {
            return Err(Denial::OverMemory {
                need: big,
                limit: limits.max_blob_bytes,
            });
        }
        if self.work_units(sizes) == 0 {
            return Err(Denial::Empty);
        }
        let need = self.memory_bytes(sizes);
        if need > limits.max_memory_bytes {
            return Err(Denial::OverMemory {
                need,
                limit: limits.max_memory_bytes,
            });
        }
        let need = self.work_units(sizes);
        if need > limits.max_work_units {
            return Err(Denial::OverWork {
                need,
                limit: limits.max_work_units,
            });
        }
        Ok(())
    }
}

fn size(sizes: &[u64], i: usize) -> u64 {
    sizes.get(i).copied().unwrap_or(0)
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

fn parse_tokenizer(inputs: &Inputs) -> Result<BitNetTokenizer, Abort> {
    let text =
        std::str::from_utf8(&inputs.tokenizer).map_err(|_| Abort::Invalid("TOKENIZER_UTF8"))?;
    BitNetTokenizer::from_json(text).map_err(|_| Abort::Invalid("TOKENIZER_REFUSED"))
}

/// The corpus lines: split at `\n`; a final `\n` does not start an empty line.
fn corpus_lines(inputs: &Inputs) -> Result<Vec<&str>, Abort> {
    let text = std::str::from_utf8(&inputs.corpus).map_err(|_| Abort::Invalid("CORPUS_UTF8"))?;
    Ok(text.split_terminator('\n').collect())
}

/// sha256 over the domain, both file digests, the BOS flag, the line count,
/// then each line's id count and ids, all big-endian u32.
fn tokenize_digest(
    tokenizer: &[u8; 32],
    corpus: &[u8; 32],
    bos: bool,
    ids: &[Vec<u32>],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(TOKENIZE_DOMAIN);
    h.update(tokenizer);
    h.update(corpus);
    h.update([u8::from(bos)]);
    h.update((ids.len() as u32).to_be_bytes());
    for line in ids {
        h.update((line.len() as u32).to_be_bytes());
        for id in line {
            h.update(id.to_be_bytes());
        }
    }
    h.finalize().into()
}

/// The worker's computation. Matrix product: row bands across threads, i-k-j
/// order inside a band, cancel flag and deadline checked between rows.
/// Tokenize: line bands across threads, checked between lines.
pub fn run(spec: &TaskSpec, inputs: &Inputs, ctl: &Control) -> Result<[u8; 32], Abort> {
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
        TaskSpec::TokenizeBatch {
            tokenizer,
            corpus,
            bos,
        } => {
            let tok = parse_tokenizer(inputs)?;
            let lines = corpus_lines(inputs)?;
            if let Some(why) = ctl.stop() {
                return Err(why);
            }
            let mut ids: Vec<Vec<u32>> = vec![Vec::new(); lines.len()];
            let threads = ctl.threads.clamp(1, lines.len().max(1));
            let band = lines.len().div_ceil(threads).max(1);
            let abort = std::thread::scope(|s| {
                let mut handles = Vec::new();
                for (out, text) in ids.chunks_mut(band).zip(lines.chunks(band)) {
                    let tok = &tok;
                    handles.push(s.spawn(move || {
                        for (slot, line) in out.iter_mut().zip(text) {
                            if let Some(why) = ctl.stop() {
                                return Some(why);
                            }
                            *slot = tok.encode(line, bos);
                        }
                        None
                    }));
                }
                handles
                    .into_iter()
                    .filter_map(|h| h.join().expect("tokenize band panicked"))
                    .next()
            });
            match abort {
                Some(why) => Err(why),
                None => Ok(tokenize_digest(&tokenizer, &corpus, bos, &ids)),
            }
        }
    }
}

/// The coordinator's recompute for tokenize_batch: one thread, line by line,
/// plus a decode of every line's ids back to text. The encoder is the same
/// code as the worker's; its independence is a different machine and build.
pub fn tokenize_summary(spec: &TaskSpec, inputs: &Inputs) -> Result<TokenizeSummary, String> {
    let TaskSpec::TokenizeBatch {
        tokenizer,
        corpus,
        bos,
    } = *spec
    else {
        return Err("not a tokenize_batch job".to_string());
    };
    let why = |a: Abort| format!("{a:?}");
    let tok = parse_tokenizer(inputs).map_err(why)?;
    let lines = corpus_lines(inputs).map_err(why)?;
    let mut ids = Vec::with_capacity(lines.len());
    let (mut tokens, mut round_trips) = (0u64, 0u64);
    for line in &lines {
        let v = tok.encode(line, bos);
        tokens += v.len() as u64;
        let body = if bos { &v[1..] } else { &v[..] };
        if tok.decode(body).is_ok_and(|t| t == *line) {
            round_trips += 1;
        }
        ids.push(v);
    }
    Ok(TokenizeSummary {
        lines: lines.len() as u64,
        tokens,
        round_trips,
        digest: tokenize_digest(&tokenizer, &corpus, bos, &ids),
    })
}

/// The coordinator's recompute. The matrix product uses a different loop
/// order (i-j-k dot products, one thread) from the worker, so a shared bug in
/// the blocked loop does not hide. The digest chain has only one algorithm;
/// its independence is a different machine and a different build, nothing
/// more; tokenize_batch is `tokenize_summary`.
pub fn oracle(spec: &TaskSpec, inputs: &Inputs) -> Result<[u8; 32], String> {
    if let TaskSpec::TokenizeBatch { .. } = spec {
        return tokenize_summary(spec, inputs).map(|s| s.digest);
    }
    Ok(generated_oracle(spec))
}

fn generated_oracle(spec: &TaskSpec) -> [u8; 32] {
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
        TaskSpec::TokenizeBatch { .. } => unreachable!("tokenize_batch goes to tokenize_summary"),
    }
}
