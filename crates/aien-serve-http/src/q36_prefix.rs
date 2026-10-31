//! The 35B door's shared prompt cache (sc L375, Drake 2026-10-21: "I cannot be waiting 6.5
//! minutes to talk to Hermes").
//!
//! Every new Hermes chat starts with the same ~11k tokens of tool descriptions and ~6k of system
//! text. The batch door's per-row reuse only continues the same conversation on the same row;
//! this pool is shared by every row. While a row reads a prompt, the door saves the row's whole
//! state at checkpoints (every [`EVERY`] tokens, at the end of the first turn and at the last
//! turn header): the DeltaNet tails and states, the position, and the attention K/V written since
//! the previous checkpoint of the same prefix. The checkpoints form a tree: a node holds the K/V
//! of positions `kv_from .. pos` and its parent holds the rest, so a shared prefix's K/V is held
//! once. A joining prompt restores the deepest node whose tokens it starts with (compared token
//! for token, never by hash alone) and reads only the tail.
//!
//! Exact by construction: the restored row holds the bytes the reading row held at that
//! position, and every later token runs the same kernels in the same order.
//!
//! Memory: LRU by bytes over the leaves (a parent outlives its children), budget
//! `AIEN_Q36_PREFIX_POOL_GB` (default 16; 0 turns the pool off). The pool runs only with
//! `AIEN_Q36_PREFIX=1` (DRA-86: default off).
//!
//! Disk (coordinator 12:07Z, Drake: "11,000 words can stay where they need to stay"): with
//! `AIEN_Q36_PREFIX_DIR` set (a private directory, `AIEN_KV_STORE_KEY` required) every node is
//! also written as one sealed file (AES-256-GCM, aien-kv-store's `SealKey`; the header binds the
//! namespace, the token digest, the parent and the lengths) on a writer thread, and every stored
//! node is loaded back at start, before the door takes requests. The namespace binds the
//! checkpoint digest, the linked `libomega_gpu.a` sha256, the state shape, [`NUMERICS_EPOCH`]
//! and every `AIEN_Q36_*` / `AIEN_OMEGA_*` switch except the pool's own: a file made under
//! anything else is never opened. Any problem with a file ends in discarding it (a cold read).
use std::collections::HashMap;
use std::io::Write;
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::sync::{mpsc, Arc};
use std::time::Instant;

use aien_kv_store::record::{HEADER_REGION_LEN, SEAL_SLOT_OFFSET};
use aien_kv_store::seal::SealKey;
use sha2::{Digest, Sha256};

/// Tokens between two checkpoints of one prompt (`AIEN_Q36_PREFIX_EVERY`).
pub const EVERY: usize = 2048;
/// Default pool budget in GiB (`AIEN_Q36_PREFIX_POOL_GB`; DRA-86 G4: 16 GiB).
pub const POOL_GB: f64 = 16.0;
/// The switch: the shared prompt cache runs only with `AIEN_Q36_PREFIX=1` (default off).
pub const SWITCH_ENV: &str = "AIEN_Q36_PREFIX";
/// Bump when an sc change moves a state byte for the same lib and weights (stored files made
/// before it are then never opened).
pub const NUMERICS_EPOCH: u32 = 1;
/// The position mutant (`AIEN_Q36_PREFIX_MUTANT=1`): a restored row continues one position
/// early. The gate must go red with it.
pub const MUTANT_ENV: &str = "AIEN_Q36_PREFIX_MUTANT";

const MAGIC: &[u8; 16] = b"AIEN-Q36-PREFIX1";
const FILE_EXT: &str = "q36p";

/// What a row's state copy must offer the pool.
pub trait PoolSnap: Send + Sync + 'static {
    fn host_bytes(&self) -> usize;
    fn to_bytes(&self) -> Vec<u8>;
    /// `to_bytes().len()` without making the bytes.
    fn bytes_len(&self) -> usize {
        self.to_bytes().len()
    }
    fn from_bytes(b: &[u8]) -> Result<Self, String>
    where
        Self: Sized;
    /// The position the state continues at, when the state records it (a restore refuses a
    /// state whose position is not the node's token count).
    fn pos(&self) -> Option<usize> {
        None
    }
}

impl PoolSnap for aien_inference_abi::qwen38_omega::Qwen38Snapshot {
    fn host_bytes(&self) -> usize {
        self.bytes()
    }
    fn to_bytes(&self) -> Vec<u8> {
        Self::to_bytes(self)
    }
    fn bytes_len(&self) -> usize {
        24 + self.bytes()
    }
    fn from_bytes(b: &[u8]) -> Result<Self, String> {
        Self::from_bytes(b)
    }
    fn pos(&self) -> Option<usize> {
        Some(self.pos)
    }
}

/// The pool's settings from the environment.
#[derive(Debug, Clone)]
pub struct PoolConfig {
    pub budget: usize,
    pub every: usize,
    pub dir: Option<PathBuf>,
    pub mutant: bool,
}

impl PoolConfig {
    pub fn from_env() -> Result<Self, String> {
        Self::from_lookup(|k| std::env::var(k).ok())
    }

    pub fn from_lookup(get: impl Fn(&str) -> Option<String>) -> Result<Self, String> {
        let on = match get(SWITCH_ENV).as_deref() {
            None | Some("") | Some("0") => false,
            Some("1") => true,
            Some(v) => return Err(format!("{SWITCH_ENV}={v}: 0 or 1")),
        };
        let gb = match get("AIEN_Q36_PREFIX_POOL_GB") {
            Some(v) => v
                .parse::<f64>()
                .ok()
                .filter(|g| g.is_finite() && *g >= 0.0)
                .ok_or(format!("AIEN_Q36_PREFIX_POOL_GB={v}: a number of GiB >= 0"))?,
            None => POOL_GB,
        };
        let every = match get("AIEN_Q36_PREFIX_EVERY") {
            Some(v) => v
                .parse::<usize>()
                .ok()
                .filter(|&n| n >= 1)
                .ok_or(format!("AIEN_Q36_PREFIX_EVERY={v}: a whole number >= 1"))?,
            None => EVERY,
        };
        Ok(Self {
            budget: if on {
                (gb * (1u64 << 30) as f64) as usize
            } else {
                0
            },
            every,
            dir: get("AIEN_Q36_PREFIX_DIR")
                .filter(|s| !s.is_empty())
                .map(PathBuf::from),
            mutant: get(MUTANT_ENV).as_deref() == Some("1"),
        })
    }
}

/// One checkpoint: the row's state after `tokens`.
pub struct Node<S> {
    parent: Option<u64>,
    tokens: Vec<u32>,
    /// K/V of positions `kv_from .. tokens.len()` (the parent holds the ones before).
    kv_from: usize,
    kv: Arc<Vec<u16>>,
    snap: Arc<S>,
    digest: [u8; 32],
    /// The pool's tag (model, chip library, state shape, numeric switches) when it was saved.
    tag: [u8; 32],
    children: usize,
    last_used: u64,
}

impl<S: PoolSnap> Node<S> {
    fn bytes(&self) -> usize {
        self.kv.len() * 2 + self.snap.host_bytes() + self.tokens.len() * 4
    }
}

/// What a restore copies into a row: the K/V pieces root first, then the deepest state.
pub struct Restore<'a, S> {
    pub pos: usize,
    pub kv: Vec<(usize, &'a [u16])>,
    pub snap: &'a S,
    /// The bytes this conversation's chain holds resident in the pool (every node root first).
    pub resident: usize,
}

/// Why a restore was refused (the row then reads the prompt cold; nothing was written to it).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Refused {
    /// The node was evicted.
    Gone,
    /// A node of the chain was saved under another model, chip library or switch tag.
    Stale { pos: usize },
    /// The node's tokens (or their digest) are not the prompt's.
    TokenMismatch { pos: usize },
    /// A K/V piece does not start where the one before ended.
    KvGap { kv_from: usize, expected: usize },
    /// A K/V piece holds another number of values than its span.
    KvLength { pos: usize, values: usize, expected: usize },
    /// The state continues at another position than the node's token count.
    StateLength { state: usize, pos: usize },
}

impl std::fmt::Display for Refused {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gone => write!(f, "node gone"),
            Self::Stale { pos } => write!(f, "stale node at {pos} (another model/lib tag)"),
            Self::TokenMismatch { pos } => write!(f, "node at {pos} holds other tokens"),
            Self::KvGap { kv_from, expected } => {
                write!(f, "K/V gap: piece starts at {kv_from}, not {expected}")
            }
            Self::KvLength {
                pos,
                values,
                expected,
            } => write!(f, "K/V of node at {pos}: {values} values, not {expected}"),
            Self::StateLength { state, pos } => {
                write!(f, "state continues at {state}, node holds {pos} tokens")
            }
        }
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct PoolCounters {
    pub nodes: usize,
    pub bytes: usize,
    pub hits: u64,
    pub misses: u64,
    pub saved: u64,
    pub evicted: u64,
    pub loaded: u64,
    pub discarded: u64,
    pub refused: u64,
}

/// The shared pool (owned by the scheduler thread).
pub struct PrefixPool<S> {
    nodes: HashMap<u64, Node<S>>,
    next_id: u64,
    budget: usize,
    bytes: usize,
    clock: u64,
    every: usize,
    mutant: bool,
    store: Option<Store>,
    counters: PoolCounters,
    tag: [u8; 32],
    /// Test-only: restore without the refusal checks (the unchecked base, as a negative control).
    unchecked: bool,
}

fn digest_of(tokens: &[u32]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIEN-Q36-PREFIX-TOKENS-V1\0");
    for t in tokens {
        h.update(t.to_le_bytes());
    }
    h.finalize().into()
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

impl<S: PoolSnap> PrefixPool<S> {
    pub fn new(cfg: &PoolConfig) -> Self {
        Self {
            nodes: HashMap::new(),
            next_id: 1,
            budget: cfg.budget,
            bytes: 0,
            clock: 0,
            every: cfg.every,
            mutant: cfg.mutant,
            store: None,
            counters: PoolCounters::default(),
            tag: [0; 32],
            unchecked: false,
        }
    }

    /// Binds the pool to `tag` (the model, chip library, state shape and numeric switches): a
    /// node saved under another tag is refused at restore and dropped.
    pub fn bind_tag(&mut self, tag: [u8; 32]) {
        self.tag = tag;
    }

    #[cfg(test)]
    pub(crate) fn set_unchecked(&mut self, on: bool) {
        self.unchecked = on;
    }

    /// The bytes the chain ending at the node for `tokens` holds resident (nodes, bytes): one
    /// conversation's share of the pool.
    pub fn resident(&self, tokens: &[u32]) -> Option<(usize, usize)> {
        let (&id, _) = self.nodes.iter().find(|(_, n)| n.tokens[..] == *tokens)?;
        let chain = self.chain(id)?;
        Some((
            chain.len(),
            chain.iter().map(|i| self.nodes[i].bytes()).sum(),
        ))
    }

    /// Node `id` and its ancestors, deepest first.
    fn chain(&self, id: u64) -> Option<Vec<u64>> {
        let mut chain = Vec::new();
        let mut at = Some(id);
        while let Some(i) = at {
            chain.push(i);
            at = self.nodes.get(&i)?.parent;
        }
        Some(chain)
    }

    /// Why node `id` must not be restored as the state after `tokens` with `kv_values` K/V
    /// values per position, if it must not.
    fn check(&self, chain: &[u64], tokens: &[u32], kv_values: usize) -> Result<(), Refused> {
        let deepest = &self.nodes[&chain[0]];
        if deepest.tokens[..] != *tokens || deepest.digest != digest_of(tokens) {
            return Err(Refused::TokenMismatch {
                pos: deepest.tokens.len(),
            });
        }
        let mut expected = 0;
        for i in chain.iter().rev() {
            let n = &self.nodes[i];
            let pos = n.tokens.len();
            if n.tag != self.tag {
                return Err(Refused::Stale { pos });
            }
            if !tokens.starts_with(&n.tokens) {
                return Err(Refused::TokenMismatch { pos });
            }
            if n.kv_from != expected || n.kv_from > pos {
                return Err(Refused::KvGap {
                    kv_from: n.kv_from,
                    expected,
                });
            }
            let want = (pos - n.kv_from) * kv_values;
            if n.kv.len() != want {
                return Err(Refused::KvLength {
                    pos,
                    values: n.kv.len(),
                    expected: want,
                });
            }
            expected = pos;
        }
        match deepest.snap.pos() {
            Some(state) if state != tokens.len() => Err(Refused::StateLength {
                state,
                pos: tokens.len(),
            }),
            _ => Ok(()),
        }
    }

    pub fn budget(&self) -> usize {
        self.budget
    }

    pub fn on(&self) -> bool {
        self.budget > 0
    }

    pub fn mutant(&self) -> bool {
        self.mutant
    }

    pub fn every(&self) -> usize {
        self.every
    }

    pub fn counters(&self) -> PoolCounters {
        PoolCounters {
            nodes: self.nodes.len(),
            bytes: self.bytes,
            ..self.counters
        }
    }

    /// The deepest node whose tokens `prompt` starts with, holding at most `max` tokens.
    pub fn longest(&self, prompt: &[u32], max: usize) -> Option<(u64, usize)> {
        self.nodes
            .iter()
            .filter(|(_, n)| n.tokens.len() <= max && prompt.starts_with(&n.tokens))
            .max_by_key(|(_, n)| n.tokens.len())
            .map(|(&id, n)| (id, n.tokens.len()))
    }

    /// Where `prompt` leaves the pool: the longest run of leading tokens it shares with a node
    /// that it does not wholly continue (`None` when it shares none, or continues every node it
    /// touches).
    pub fn divergence(&self, prompt: &[u32]) -> Option<usize> {
        self.nodes
            .values()
            .map(|n| {
                n.tokens
                    .iter()
                    .zip(prompt)
                    .take_while(|(a, b)| a == b)
                    .count()
            })
            .zip(self.nodes.values())
            .filter(|&(k, n)| k > 0 && k < n.tokens.len())
            .map(|(k, _)| k)
            .max()
    }

    /// Counts a join's lookup (`hit`: the pool gave the longest reuse).
    pub fn note_lookup(&mut self, hit: bool) {
        if hit {
            self.counters.hits += 1;
        } else {
            self.counters.misses += 1;
        }
    }

    /// The copies a restore of node `id` as the state after `tokens` (with `kv_values` K/V values
    /// per position) writes, and marks the chain used. Refused, nothing is to be written: a
    /// stale chain is dropped with every other node of another tag, a broken leaf is dropped.
    pub fn restore(
        &mut self,
        id: u64,
        tokens: &[u32],
        kv_values: usize,
    ) -> Result<Restore<'_, S>, Refused> {
        let chain = self.chain(id).ok_or(Refused::Gone)?;
        if !self.unchecked {
            if let Err(e) = self.check(&chain, tokens, kv_values) {
                self.counters.refused += 1;
                if matches!(e, Refused::Stale { .. }) {
                    let tag = self.tag;
                    while let Some(v) = self
                        .nodes
                        .iter()
                        .find(|(_, n)| n.tag != tag && n.children == 0)
                        .map(|(&i, _)| i)
                    {
                        self.remove(v);
                    }
                } else if self.nodes[&id].children == 0 {
                    self.remove(id);
                }
                return Err(e);
            }
        }
        self.clock += 1;
        for i in &chain {
            self.nodes.get_mut(i).expect("chain node").last_used = self.clock;
        }
        let deepest = &self.nodes[&id];
        Ok(Restore {
            pos: deepest.tokens.len(),
            kv: chain
                .iter()
                .rev()
                .map(|i| {
                    let n = &self.nodes[i];
                    (n.kv_from, &n.kv[..])
                })
                .collect(),
            snap: &deepest.snap,
            resident: chain.iter().map(|i| self.nodes[i].bytes()).sum(),
        })
    }

    /// Whether `tokens` is already a node (and marks it used).
    pub fn touch(&mut self, tokens: &[u32]) -> bool {
        self.clock += 1;
        let c = self.clock;
        match self.nodes.values_mut().find(|n| n.tokens[..] == *tokens) {
            Some(n) => {
                n.last_used = c;
                true
            }
            None => false,
        }
    }

    /// Where a new node for `tokens` starts its K/V: its parent (the deepest proper prefix).
    pub fn parent_for(&self, tokens: &[u32]) -> Option<(u64, usize)> {
        self.longest(tokens, tokens.len().saturating_sub(1))
    }

    /// Adds the state after `tokens` (K/V from `kv_from`, the parent's length), then evicts to
    /// the budget. Returns false when the node alone does not fit.
    pub fn insert(&mut self, tokens: Vec<u32>, parent: Option<u64>, kv: Vec<u16>, snap: S) -> bool {
        let kv_from = parent.map_or(0, |p| self.nodes[&p].tokens.len());
        let id = self.add(tokens, parent, kv_from, Arc::new(kv), Arc::new(snap));
        if let Some(st) = &self.store {
            let n = &self.nodes[&id];
            let parent_digest = parent.map(|p| self.nodes[&p].digest);
            st.write(n, parent_digest);
        }
        self.counters.saved += 1;
        self.evict(id);
        self.nodes.contains_key(&id)
    }

    fn add(
        &mut self,
        tokens: Vec<u32>,
        parent: Option<u64>,
        kv_from: usize,
        kv: Arc<Vec<u16>>,
        snap: Arc<S>,
    ) -> u64 {
        self.clock += 1;
        let id = self.next_id;
        self.next_id += 1;
        if let Some(p) = parent {
            self.nodes.get_mut(&p).expect("parent node").children += 1;
        }
        let n = Node {
            parent,
            digest: digest_of(&tokens),
            tokens,
            kv_from,
            kv,
            snap,
            children: 0,
            last_used: self.clock,
            tag: self.tag,
        };
        self.bytes += n.bytes();
        self.nodes.insert(id, n);
        id
    }

    /// Drops least recently used leaves until the pool fits; `fresh` goes last.
    fn evict(&mut self, fresh: u64) {
        while self.bytes > self.budget {
            let victim = self
                .nodes
                .iter()
                .filter(|(&i, n)| n.children == 0 && i != fresh)
                .min_by_key(|(_, n)| n.last_used)
                .map(|(&i, _)| i)
                .or_else(|| self.nodes.contains_key(&fresh).then_some(fresh));
            let Some(v) = victim else { break };
            self.remove(v);
            self.counters.evicted += 1;
        }
    }

    /// Drops leaf `v` (and its stored file).
    fn remove(&mut self, v: u64) {
        let n = self.nodes.remove(&v).expect("victim");
        self.bytes -= n.bytes();
        if let Some(p) = n.parent {
            if let Some(pn) = self.nodes.get_mut(&p) {
                pn.children -= 1;
            }
        }
        if let Some(st) = &self.store {
            st.delete(&n.digest, n.tokens.len());
        }
    }

    /// Opens the sealed store at `dir` and loads every node it holds (parents first) that fits
    /// the budget; later nodes are written there too. `shape` checks a loaded state against
    /// this model (bytes per K/V position, state bytes).
    pub fn open_store(
        &mut self,
        dir: &Path,
        key: SealKey,
        namespace: [u8; 32],
        kv_values: usize,
        snap_ok: impl Fn(&S) -> bool,
    ) -> Result<(), String> {
        open_private_dir(dir)?;
        // The seal budget's per-store usage counts (aien-kv-store `SealKey::seal_file`).
        open_private_dir(&dir.join(aien_kv_store::seal::USAGE_DIR_NAME))?;
        let t0 = Instant::now();
        let mut found: Vec<Loaded> = Vec::new();
        let rd = std::fs::read_dir(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for ent in rd.flatten() {
            let p = ent.path();
            if p.extension().and_then(|e| e.to_str()) != Some(FILE_EXT) {
                continue;
            }
            match read_node(&p, &key, &namespace, kv_values) {
                Ok(l) => found.push(l),
                Err(e) => {
                    eprintln!(
                        "aien-serve-http: q36 prefix: discarding {}: {e}",
                        p.display()
                    );
                    let _ = std::fs::remove_file(&p);
                    self.counters.discarded += 1;
                }
            }
        }
        found.sort_by_key(|l| l.tokens.len());
        let mut by_digest: HashMap<[u8; 32], u64> = HashMap::new();
        for l in found {
            let parent = match l.parent {
                None if l.kv_from == 0 => Ok(None),
                None => Err("no parent but K/V starts past 0".to_string()),
                Some(pd) => match by_digest.get(&pd) {
                    Some(&p)
                        if self.nodes[&p].tokens.len() == l.kv_from
                            && l.tokens.starts_with(&self.nodes[&p].tokens) =>
                    {
                        Ok(Some(p))
                    }
                    Some(_) => Err("parent does not match".into()),
                    None => Err("parent not loaded".into()),
                },
            };
            let snap = parent.and_then(|p| {
                S::from_bytes(&l.snap)
                    .and_then(|s| {
                        if snap_ok(&s) {
                            Ok(s)
                        } else {
                            Err("state of another shape".into())
                        }
                    })
                    .map(|s| (p, s))
            });
            let fits =
                self.bytes + l.tokens.len() * 4 + l.kv.len() * 2 + l.snap.len() <= self.budget;
            match snap {
                Ok((parent, snap)) if fits => {
                    let d = digest_of(&l.tokens);
                    let id = self.add(l.tokens, parent, l.kv_from, Arc::new(l.kv), Arc::new(snap));
                    by_digest.insert(d, id);
                    self.counters.loaded += 1;
                }
                r => {
                    let why = match r {
                        Err(e) => e,
                        Ok(_) => "over the pool budget".into(),
                    };
                    eprintln!(
                        "aien-serve-http: q36 prefix: discarding {}: {why}",
                        l.path.display()
                    );
                    let _ = std::fs::remove_file(&l.path);
                    self.counters.discarded += 1;
                }
            }
        }
        eprintln!(
            "aien-serve-http: q36 prefix: store {} open: {} nodes loaded ({:.1} MiB), {} discarded, in {:.1} s",
            dir.display(),
            self.counters.loaded,
            self.bytes as f64 / (1u64 << 20) as f64,
            self.counters.discarded,
            t0.elapsed().as_secs_f64()
        );
        self.store = Some(Store::start(dir.to_path_buf(), Arc::new(key), namespace));
        Ok(())
    }

    /// Waits for every queued write (tests and shutdown).
    pub fn flush(&self) {
        if let Some(st) = &self.store {
            st.flush();
        }
    }
}

/// The namespace a stored node must carry: the weights, the linked chip library, the state shape,
/// [`NUMERICS_EPOCH`] and the run switches that can move a state byte.
pub fn namespace(
    checkpoint: &[u8; 32],
    lib_sha256: &str,
    kv_values: usize,
    snap_bytes: usize,
    env: &[(String, String)],
) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(b"AIEN-Q36-PREFIX-NS-V1\0");
    h.update(checkpoint);
    h.update(lib_sha256.as_bytes());
    h.update(NUMERICS_EPOCH.to_le_bytes());
    h.update((kv_values as u64).to_le_bytes());
    h.update((snap_bytes as u64).to_le_bytes());
    let mut env = env.to_vec();
    env.sort();
    for (k, v) in env {
        h.update(format!("\n{k}={v}").as_bytes());
    }
    h.finalize().into()
}

/// The switches the namespace binds: every `AIEN_Q36_*` and `AIEN_OMEGA_*` variable except the
/// pool's own settings and the beat's prompt share (neither moves a state byte); the mutant is
/// bound, so a mutant run never feeds the store of a real one.
pub fn numeric_env(vars: impl Iterator<Item = (String, String)>) -> Vec<(String, String)> {
    vars.filter(|(k, _)| {
        (k.starts_with("AIEN_Q36_") || k.starts_with("AIEN_OMEGA_"))
            && (!k.starts_with("AIEN_Q36_PREFIX_") || k == MUTANT_ENV)
            && k != "AIEN_Q36_PROMPT_PER_BEAT"
            && k != SWITCH_ENV
            && k != "AIEN_OMEGA_GPU_LIB"
    })
    .collect()
}

/// The directory rules of aien-kv-store's host store, kept short: an existing directory must be
/// ours and private (0700); a missing one is made so.
fn open_private_dir(dir: &Path) -> Result<(), String> {
    match std::fs::symlink_metadata(dir) {
        Ok(m) => {
            if !m.is_dir() {
                return Err(format!("{}: not a directory", dir.display()));
            }
            // SAFETY: plain libc call without arguments.
            let uid = unsafe { libc::geteuid() };
            if m.uid() != uid {
                return Err(format!("{}: not owned by this user", dir.display()));
            }
            if m.permissions().mode() & 0o077 != 0 {
                return Err(format!(
                    "{}: mode {:o}, must be private (0700)",
                    dir.display(),
                    m.permissions().mode() & 0o777
                ));
            }
            Ok(())
        }
        Err(_) => std::fs::DirBuilder::new()
            .mode(0o700)
            .create(dir)
            .map_err(|e| format!("{}: {e}", dir.display())),
    }
}

fn file_name(digest: &[u8; 32], pos: usize) -> String {
    format!("{}-{pos}.{FILE_EXT}", hex(&digest[..16]))
}

struct Loaded {
    path: PathBuf,
    tokens: Vec<u32>,
    parent: Option<[u8; 32]>,
    kv_from: usize,
    kv: Vec<u16>,
    snap: Vec<u8>,
}

fn u64_at(b: &[u8], o: usize) -> usize {
    u64::from_le_bytes(b[o..o + 8].try_into().expect("8 bytes")) as usize
}

/// Header region layout (bytes `0 .. SEAL_SLOT_OFFSET` are associated data of the seal):
/// magic 16 | namespace 32 | token digest 32 | parent digest 32 (zeros: none) | has parent 8 |
/// tokens 8 | kv_from 8 | kv values 8 | state bytes 8.
fn header<S: PoolSnap>(
    n: &Node<S>,
    parent: Option<[u8; 32]>,
    ns: &[u8; 32],
    snap_len: usize,
) -> Vec<u8> {
    let mut r = vec![0u8; HEADER_REGION_LEN];
    r[0..16].copy_from_slice(MAGIC);
    r[16..48].copy_from_slice(ns);
    r[48..80].copy_from_slice(&n.digest);
    r[80..112].copy_from_slice(&parent.unwrap_or([0; 32]));
    for (i, v) in [
        usize::from(parent.is_some()),
        n.tokens.len(),
        n.kv_from,
        n.kv.len(),
        snap_len,
    ]
    .into_iter()
    .enumerate()
    {
        r[112 + i * 8..120 + i * 8].copy_from_slice(&(v as u64).to_le_bytes());
    }
    const { assert!(152 <= SEAL_SLOT_OFFSET) };
    r
}

fn read_node(p: &Path, key: &SealKey, ns: &[u8; 32], kv_values: usize) -> Result<Loaded, String> {
    let mut file = std::fs::read(p).map_err(|e| e.to_string())?;
    if file.len() < HEADER_REGION_LEN {
        return Err("short file".into());
    }
    let (region, payload) = file.split_at_mut(HEADER_REGION_LEN);
    if &region[0..16] != MAGIC {
        return Err("not a prompt cache file".into());
    }
    if region[16..48] != ns[..] {
        return Err("made under another model, build or switch set".into());
    }
    let (has_parent, nt, kv_from, nkv, nsnap) = (
        u64_at(region, 112),
        u64_at(region, 120),
        u64_at(region, 128),
        u64_at(region, 136),
        u64_at(region, 144),
    );
    let want = nt
        .checked_mul(4)
        .and_then(|a| nkv.checked_mul(2).and_then(|b| a.checked_add(b)))
        .and_then(|a| a.checked_add(nsnap));
    if want != Some(payload.len()) {
        return Err("lengths do not match the file".into());
    }
    if kv_from > nt || nkv != (nt - kv_from) * kv_values {
        return Err("K/V length does not match the positions".into());
    }
    key.open_payload(region, payload)
        .map_err(|e| format!("seal refused ({e:?})"))?;
    let tokens: Vec<u32> = payload[..nt * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    if digest_of(&tokens)[..] != region[48..80] {
        return Err("token digest does not match".into());
    }
    let kv: Vec<u16> = payload[nt * 4..nt * 4 + nkv * 2]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| u16::from_le_bytes(*c))
        .collect();
    let mut pd = [0u8; 32];
    pd.copy_from_slice(&region[80..112]);
    Ok(Loaded {
        path: p.to_path_buf(),
        tokens,
        parent: (has_parent == 1).then_some(pd),
        kv_from,
        kv,
        snap: payload[nt * 4 + nkv * 2..].to_vec(),
    })
}

enum Msg<S> {
    Write {
        name: String,
        region: Vec<u8>,
        tokens: Vec<u32>,
        kv: Arc<Vec<u16>>,
        snap: Arc<S>,
    },
    Delete(String),
    Flush(mpsc::SyncSender<()>),
}

/// One write or delete, run on the writer thread with the store dir and key.
type StoreJob = Box<dyn FnOnce(&Path, &SealKey) + Send>;

/// The writer thread: seals and writes nodes, deletes evicted ones.
struct Store {
    tx: mpsc::Sender<StoreJob>,
    ns: [u8; 32],
}

impl Store {
    fn start(dir: PathBuf, key: Arc<SealKey>, ns: [u8; 32]) -> Self {
        let (tx, rx) = mpsc::channel::<Box<dyn FnOnce(&Path, &SealKey) + Send>>();
        std::thread::Builder::new()
            .name("q36-prefix-store".into())
            .spawn(move || {
                while let Ok(job) = rx.recv() {
                    job(&dir, &key);
                }
            })
            .expect("q36 prefix store thread");
        Self { tx, ns }
    }

    fn send<S: PoolSnap>(&self, m: Msg<S>) {
        let job: StoreJob = Box::new(move |dir, key| match m {
            Msg::Write {
                name,
                region,
                tokens,
                kv,
                snap,
            } => {
                if let Err(e) = write_file(dir, key, &name, region, &tokens, &kv, &*snap) {
                    eprintln!("aien-serve-http: q36 prefix: writing {name} failed: {e}");
                }
            }
            Msg::Delete(name) => {
                let _ = std::fs::remove_file(dir.join(name));
            }
            Msg::Flush(done) => {
                let _ = done.send(());
            }
        });
        let _ = self.tx.send(job);
    }

    fn write<S: PoolSnap>(&self, n: &Node<S>, parent: Option<[u8; 32]>) {
        // The state's byte length goes in the header; the writer thread checks its bytes against it.
        let snap_len = n.snap.bytes_len();
        self.send(Msg::Write {
            name: file_name(&n.digest, n.tokens.len()),
            region: header(n, parent, &self.ns, snap_len),
            tokens: n.tokens.clone(),
            kv: Arc::clone(&n.kv),
            snap: Arc::clone(&n.snap),
        });
    }

    fn delete(&self, digest: &[u8; 32], pos: usize) {
        self.send::<NoSnap>(Msg::Delete(file_name(digest, pos)));
    }

    fn flush(&self) {
        let (tx, rx) = mpsc::sync_channel(1);
        self.send::<NoSnap>(Msg::Flush(tx));
        let _ = rx.recv();
    }
}

/// A stand-in state type for messages that carry none.
struct NoSnap;
impl PoolSnap for NoSnap {
    fn host_bytes(&self) -> usize {
        0
    }
    fn to_bytes(&self) -> Vec<u8> {
        Vec::new()
    }
    fn from_bytes(_: &[u8]) -> Result<Self, String> {
        Ok(Self)
    }
}

fn write_file<S: PoolSnap>(
    dir: &Path,
    key: &SealKey,
    name: &str,
    region: Vec<u8>,
    tokens: &[u32],
    kv: &[u16],
    snap: &S,
) -> Result<(), String> {
    let sb = snap.to_bytes();
    if u64_at(&region, 144) != sb.len() {
        return Err("state length changed".into());
    }
    let mut file = region;
    file.reserve(tokens.len() * 4 + kv.len() * 2 + sb.len());
    for t in tokens {
        file.extend_from_slice(&t.to_le_bytes());
    }
    for v in kv {
        file.extend_from_slice(&v.to_le_bytes());
    }
    file.extend_from_slice(&sb);
    key.seal_file(&dir.join(aien_kv_store::seal::USAGE_DIR_NAME), &mut file)
        .map_err(|e| format!("seal: {e:?}"))?;
    let tmp = dir.join(format!(".{name}.tmp"));
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(&tmp)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.write_all(&file).map_err(|e| e.to_string())?;
    f.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, dir.join(name)).map_err(|e| e.to_string())
}

#[cfg(test)]
mod tests;
