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

//! Deterministic nonce-source and restore tests for sealed host objects (#708).
//!
//! The synthetic key is the fixed pattern `0xA5` repeated. It is not a vault
//! secret and it is not drawn from the OS.
//!
//! Reproduction:
//! `cargo test -p aien-kv-store --lib seal_nonce_tests -- --test-threads=8`
//!
//! `with_nonce_source` exists only in this crate's unit-test build. A
//! production build of `seal_file` calls `SystemRandom` on every seal.

use crate::host_fs::{HostFsConfig, HostFsStore, WorkerMode};
use crate::identity::{
    chunk_chain, BackendFormat, ComputeConfig, HiddenAct, KvDtype, LayoutStrides, NamespaceFields,
};
use crate::pin::{HolderGuard, PinnedDest, PinnedSource};
use crate::record::{RecordHeader, HEADER_LEN, HEADER_REGION_LEN, SEAL_SLOT_OFFSET};
use crate::seal::{
    NonceSource, SealError, SealKey, AAD_LEN, NIST_SP800_38D_RANDOM_NONCE_LIMIT, NONCE_LEN,
    RECOMMENDED_SEALS_PER_KEY,
};
use crate::{ChunkHeader, ChunkKey, Incarnation, JobId, JobStage, KvStore, Terminal};
use aien_platform::PlatformError;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier};
use std::time::{SystemTime, UNIX_EPOCH};

const BS: usize = 8192;
const NONCE_AT: usize = SEAL_SLOT_OFFSET + 4;
/// Fixed test key. Not `AIEN_KV_STORE_KEY` and not OS output.
const SYNTHETIC_KEY: [u8; 32] = [0xA5; 32];
const OTHER_KEY: [u8; 32] = [0x5C; 32];

/// How a test result is allowed to be described.
///
/// `UnsafeControl` is misuse the sealer accepts. A test that observes it must
/// keep this label. Relabeling it `Protected` is a failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum OutcomeClass {
    Protected,
    UnsafeControl,
}

fn nonce_for(n: u64) -> [u8; NONCE_LEN] {
    let mut out = [0u8; NONCE_LEN];
    out[..8].copy_from_slice(&n.to_le_bytes());
    out
}

struct FailNonce;

impl NonceSource for FailNonce {
    fn fill_nonce(&self, _out: &mut [u8; NONCE_LEN]) -> Result<(), SealError> {
        Err(SealError::Random)
    }
}

struct RepeatNonce([u8; NONCE_LEN]);

impl NonceSource for RepeatNonce {
    fn fill_nonce(&self, out: &mut [u8; NONCE_LEN]) -> Result<(), SealError> {
        *out = self.0;
        Ok(())
    }
}

struct SeqNonce {
    next: AtomicU64,
}

impl SeqNonce {
    fn starting_at(n: u64) -> Self {
        Self {
            next: AtomicU64::new(n),
        }
    }
}

impl NonceSource for SeqNonce {
    fn fill_nonce(&self, out: &mut [u8; NONCE_LEN]) -> Result<(), SealError> {
        let n = self.next.fetch_add(1, Ordering::SeqCst);
        *out = nonce_for(n);
        Ok(())
    }
}

struct SeqThenFail {
    next: AtomicU64,
    fail_at: u64,
}

impl NonceSource for SeqThenFail {
    fn fill_nonce(&self, out: &mut [u8; NONCE_LEN]) -> Result<(), SealError> {
        let n = self.next.fetch_add(1, Ordering::SeqCst);
        if n == self.fail_at {
            return Err(SealError::Random);
        }
        *out = nonce_for(n);
        Ok(())
    }
}

fn key_with(source: Arc<dyn NonceSource + Send + Sync>) -> Arc<SealKey> {
    Arc::new(SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(source))
}

fn pattern(n: u32, len: usize) -> Vec<u8> {
    (0..len)
        .map(|i| ((i as u32).wrapping_mul(31).wrapping_add(n.wrapping_mul(17)) % 251 + 1) as u8)
        .collect()
}

fn xor(a: &[u8], b: &[u8]) -> Vec<u8> {
    a.iter().zip(b.iter()).map(|(x, y)| x ^ y).collect()
}

fn plain(payload: &[u8]) -> Vec<u8> {
    let mut file = vec![0u8; HEADER_REGION_LEN + payload.len()];
    for (i, b) in file[..256].iter_mut().enumerate() {
        *b = (i as u8).wrapping_add(1);
    }
    file[300] = 0x5A;
    file[400] = 0xA5;
    file[HEADER_REGION_LEN..].copy_from_slice(payload);
    file
}

fn open_buf(key: &SealKey, file: &[u8]) -> (Result<(), SealError>, Vec<u8>) {
    let (region, body) = file.split_at(HEADER_REGION_LEN);
    let mut body = body.to_vec();
    let got = key.open_payload(region, &mut body);
    (got, body)
}

fn file_nonce(path: &Path) -> [u8; NONCE_LEN] {
    let bytes = std::fs::read(path).unwrap();
    bytes[NONCE_AT..NONCE_AT + NONCE_LEN].try_into().unwrap()
}

struct Tmp(PathBuf);

impl Tmp {
    fn new(tag: &str) -> Self {
        static N: AtomicU64 = AtomicU64::new(0);
        let n = N.fetch_add(1, Ordering::Relaxed);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p = std::env::temp_dir().join(format!(
            "aien-kv-nonce-{tag}-{}-{n}-{nanos}",
            std::process::id()
        ));
        std::os::unix::fs::DirBuilderExt::mode(std::fs::DirBuilder::new().recursive(true), 0o700)
            .create(&p)
            .unwrap();
        Tmp(p)
    }
}

impl Drop for Tmp {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

struct EnvSet(&'static str);

impl EnvSet {
    fn set(name: &'static str, value: &str) -> Self {
        std::env::set_var(name, value);
        EnvSet(name)
    }
}

impl Drop for EnvSet {
    fn drop(&mut self) {
        std::env::remove_var(self.0);
    }
}

fn test_fields() -> NamespaceFields<'static> {
    let compute = ComputeConfig {
        num_heads: 32,
        hidden_dim: 2048,
        intermediate_dim: 8192,
        vocab_size: 128_256,
        rms_norm_eps: 1e-5,
        tie_word_embeddings: true,
        hidden_act: HiddenAct::Silu,
        arith_id: "ReferenceCpuBackend",
        arith_epoch: 1,
    };
    NamespaceFields {
        model_sha256: [0xAB; 32],
        model_digest_kind: "sha256-file",
        model_id: "llama-3.2-1b-instruct",
        num_layers: 2,
        num_kv_heads: 4,
        head_dim: 64,
        block_size: 4,
        rope_theta: 500000.0,
        rope_scaling: None,
        qk_norm: false,
        dtype: KvDtype::Bf16,
        layout: LayoutStrides {
            head_stride: 128,
            token_stride: 512,
            kv_plane_stride: 2048,
            layer_stride: 4096,
            block_stride: BS as u64,
            element_bytes: 2,
        },
        backend_format: BackendFormat::HostPagedPoolV1,
        chunk_tokens: 8,
        sharing_namespace: "default",
        compute_config_digest: compute.digest().unwrap(),
    }
}

fn open_at(root: &Path, key: Arc<SealKey>) -> HostFsStore {
    let mut cfg = HostFsConfig::from_fields(root.to_path_buf(), &test_fields()).unwrap();
    cfg.seal_key = Some(key);
    cfg.worker = WorkerMode::Manual;
    let s = HostFsStore::open(cfg).expect("open");
    let mut steps = 0;
    while !s.recover_step(50).done {
        steps += 1;
        assert!(steps < 10_000, "recovery did not finish");
    }
    s
}

fn terminal_for(s: &HostFsStore, job: JobId) -> Terminal {
    let mut seen = 0;
    loop {
        let batch = s.poll(32);
        assert!(!batch.is_empty(), "no terminal for {job:?} after {seen}");
        for c in batch {
            seen += 1;
            if c.job == job {
                if let Some(t) = c.terminal {
                    return t;
                }
            }
        }
    }
}

struct Published {
    key: ChunkKey,
    header: ChunkHeader,
    path: PathBuf,
    payload: Vec<u8>,
}

fn publish(s: &HostFsStore, fields: &NamespaceFields<'_>, n: u32) -> Published {
    let ns = fields.namespace_id().unwrap();
    let toks: Vec<u32> = (0..4).map(|i| n * 100 + i).collect();
    let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
    let header = fields.chunk_header(&ns, key, None, 0, 4, 1).unwrap();
    let payload = pattern(n, BS);
    let mut buf = payload.clone().into_boxed_slice();
    // SAFETY: `buf` stays alive through `run_until_idle` and `poll`, which is
    // the whole time the store holds this source.
    let src =
        unsafe { PinnedSource::new(buf.as_mut_ptr(), BS, vec![0], 4, HolderGuard::detached()) };
    let job = s
        .submit_store(header, src, Incarnation::seq(n as u64))
        .unwrap();
    s.run_until_idle();
    assert_eq!(
        terminal_for(s, job),
        Terminal::Done(JobStage::Durable),
        "publish {n}"
    );
    let path = s.object_path(&key);
    drop(buf);
    Published {
        key,
        header,
        path,
        payload,
    }
}

fn load_bytes(s: &HostFsStore, key: ChunkKey, header: &ChunkHeader) -> (Terminal, Vec<u8>) {
    let mut buf = vec![0xEEu8; BS].into_boxed_slice();
    // SAFETY: `buf` stays alive through `run_until_idle` and `poll`.
    let dst = unsafe { PinnedDest::new(buf.as_mut_ptr(), BS, vec![0], HolderGuard::detached()) };
    let job = s
        .submit_load(key, header, dst, Incarnation::seq(9_000))
        .unwrap();
    s.run_until_idle();
    let term = terminal_for(s, job);
    (term, buf.to_vec())
}

fn kvc_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let Ok(shards) = std::fs::read_dir(root.join("objects")) else {
        return out;
    };
    for shard in shards.flatten() {
        let Ok(files) = std::fs::read_dir(shard.path()) else {
            continue;
        };
        for f in files.flatten() {
            if f.path().extension().is_some_and(|e| e == "kvc") {
                out.push(f.path());
            }
        }
    }
    out.sort();
    out
}

fn tmp_count(root: &Path) -> usize {
    std::fs::read_dir(root.join("tmp"))
        .map(|rd| rd.count())
        .unwrap_or(0)
}

fn assert_pattern_absent(root: &Path, needle: &[u8]) {
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(rd) = std::fs::read_dir(&dir) else {
            continue;
        };
        for ent in rd.flatten() {
            let p = ent.path();
            if p.is_dir() {
                stack.push(p);
                continue;
            }
            let Ok(bytes) = std::fs::read(&p) else {
                continue;
            };
            assert!(
                needle.is_empty() || bytes.windows(needle.len()).all(|w| w != needle),
                "plaintext prefix in {}",
                p.display()
            );
        }
    }
}

fn write_0600(path: &Path, bytes: &[u8]) {
    use std::io::Write;
    use std::os::unix::fs::OpenOptionsExt;
    let mut f = std::fs::OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .mode(0o600)
        .open(path)
        .unwrap();
    f.write_all(bytes).unwrap();
}

fn assert_discarded(s: &HostFsStore, pubd: &Published) {
    assert_eq!(
        s.lookup(&[pubd.key]),
        vec![Some(BS as u64)],
        "recovery still indexes a structurally valid object"
    );
    let (term, bytes) = load_bytes(s, pubd.key, &pubd.header);
    assert_eq!(term, Terminal::Failed(PlatformError::InvalidLayout));
    assert!(
        bytes.iter().all(|b| *b == 0xEE),
        "failed authentication wrote into the destination"
    );
    assert_ne!(bytes, pubd.payload);
    assert!(!pubd.path.exists(), "refused object was kept");
    assert_eq!(s.lookup(&[pubd.key]), vec![None]);
}

#[test]
fn entropy_source_error_returns_random_and_leaves_the_buffer_unchanged() {
    let key = SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(FailNonce));
    let mut file = plain(&[0x11; 32]);
    let before = file.clone();
    // Expected: Err(Random), buffer byte-identical to the input.
    // A mutant that drops the `?` on `fill_nonce` seals (Ok) or writes the slot.
    assert_eq!(key.seal_file(&mut file), Err(SealError::Random));
    assert_eq!(file, before);
}

#[test]
fn entropy_source_error_publishes_no_partial_or_plaintext_object() {
    let root = Tmp::new("entropy");
    let fields = test_fields();
    let s = open_at(&root.0, key_with(Arc::new(FailNonce)));
    let ns = fields.namespace_id().unwrap();
    let toks: Vec<u32> = (0..4).collect();
    let key = chunk_chain(&ns, 8, &toks).unwrap()[0];
    let header = fields.chunk_header(&ns, key, None, 0, 4, 1).unwrap();
    let payload = pattern(1, BS);
    let mut buf = payload.clone().into_boxed_slice();
    // SAFETY: `buf` outlives the job, which fails before any read past submit.
    let src =
        unsafe { PinnedSource::new(buf.as_mut_ptr(), BS, vec![0], 4, HolderGuard::detached()) };
    let job = s.submit_store(header, src, Incarnation::seq(1)).unwrap();
    s.run_until_idle();
    // Expected: the job fails, no `*.kvc`, no temp, the plaintext prefix is
    // absent under the root. Ignoring `seal_file`'s error would publish.
    assert_eq!(
        terminal_for(&s, job),
        Terminal::Failed(PlatformError::HardwareFault("seal failed".into()))
    );
    assert!(kvc_files(&root.0).is_empty());
    assert_eq!(tmp_count(&root.0), 0);
    assert_eq!(s.lookup(&[key]), vec![None]);
    assert_eq!(s.capacity().objects, 0);
    assert_pattern_absent(&root.0, &payload[..64]);
    drop(buf);
    drop(s);
}

#[test]
fn second_seal_entropy_error_leaves_the_first_object_and_publishes_nothing_else() {
    let root = Tmp::new("second-fail");
    let fields = test_fields();
    let source = Arc::new(SeqThenFail {
        next: AtomicU64::new(0),
        fail_at: 1,
    });
    let s = open_at(&root.0, key_with(source));
    let first = publish(&s, &fields, 1);
    assert_eq!(file_nonce(&first.path), nonce_for(0));

    let ns = fields.namespace_id().unwrap();
    let toks: Vec<u32> = (0..4).map(|i| 200 + i).collect();
    let key2 = chunk_chain(&ns, 8, &toks).unwrap()[0];
    let header2 = fields.chunk_header(&ns, key2, None, 0, 4, 1).unwrap();
    let payload2 = pattern(2, BS);
    let mut buf = payload2.clone().into_boxed_slice();
    // SAFETY: `buf` outlives the refused job.
    let src =
        unsafe { PinnedSource::new(buf.as_mut_ptr(), BS, vec![0], 4, HolderGuard::detached()) };
    let job = s.submit_store(header2, src, Incarnation::seq(2)).unwrap();
    s.run_until_idle();
    assert_eq!(
        terminal_for(&s, job),
        Terminal::Failed(PlatformError::HardwareFault("seal failed".into()))
    );
    assert!(!s.object_path(&key2).exists());
    assert_eq!(tmp_count(&root.0), 0);
    assert_eq!(kvc_files(&root.0), vec![first.path.clone()]);
    assert_pattern_absent(&root.0, &payload2[..64]);
    let (term, bytes) = load_bytes(&s, first.key, &first.header);
    assert_eq!(term, Terminal::Done(JobStage::Completed));
    assert_eq!(bytes, first.payload);
    drop(buf);
    drop(s);
}

#[test]
fn scripted_sequence_assigns_the_expected_nonce_to_each_file() {
    let key =
        SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(SeqNonce::starting_at(0)));
    for i in 0..3u64 {
        let pt = vec![0x30 + i as u8; 24];
        let mut file = plain(&pt);
        key.seal_file(&mut file).unwrap();
        assert_eq!(&file[NONCE_AT..NONCE_AT + NONCE_LEN], &nonce_for(i));
        let (got, body) = open_buf(&key, &file);
        assert_eq!(got, Ok(()));
        assert_eq!(body, pt);
    }
}

#[test]
fn restart_opens_restored_files_without_restoring_the_nonce_counter() {
    let root = Tmp::new("restart");
    let fields = test_fields();
    let s = open_at(&root.0, key_with(Arc::new(SeqNonce::starting_at(5))));
    let pubd = publish(&s, &fields, 1);
    assert_eq!(file_nonce(&pubd.path), nonce_for(5));
    let path = pubd.path.clone();
    drop(s);

    // The new source starts at 0. The file still carries nonce 5. A clone of
    // the previous process would have issued 6 next. That counter is not in
    // the store directory, so this reopen does not continue it.
    let s = open_at(&root.0, key_with(Arc::new(SeqNonce::starting_at(0))));
    assert_eq!(s.lookup(&[pubd.key]), vec![Some(BS as u64)]);
    let (term, bytes) = load_bytes(&s, pubd.key, &pubd.header);
    assert_eq!(term, Terminal::Done(JobStage::Completed));
    assert_eq!(bytes, pubd.payload);
    assert_eq!(file_nonce(&path), nonce_for(5));

    let next = publish(&s, &fields, 2);
    assert_eq!(file_nonce(&next.path), nonce_for(0));
    assert_ne!(file_nonce(&next.path), nonce_for(6));
    drop(s);
}

#[test]
fn disk_restore_opens_under_the_same_key_and_does_not_draw_a_nonce() {
    let src_root = Tmp::new("restore-src");
    let fields = test_fields();
    let s = open_at(&src_root.0, key_with(Arc::new(SeqNonce::starting_at(3))));
    let pubd = publish(&s, &fields, 4);
    let sealed = std::fs::read(&pubd.path).unwrap();
    assert_eq!(file_nonce(&pubd.path), nonce_for(3));
    drop(s);

    let dst_root = Tmp::new("restore-dst");
    let s = open_at(&dst_root.0, key_with(Arc::new(FailNonce)));
    let path = s.object_path(&pubd.key);
    drop(s);
    write_0600(&path, &sealed);

    // FailNonce errors if anything seals. Recovery and load of the copied
    // file only read the nonce stored in the seal slot.
    let s = open_at(&dst_root.0, key_with(Arc::new(FailNonce)));
    assert_eq!(s.lookup(&[pubd.key]), vec![Some(BS as u64)]);
    let (term, bytes) = load_bytes(&s, pubd.key, &pubd.header);
    assert_eq!(term, Terminal::Done(JobStage::Completed));
    assert_eq!(bytes, pubd.payload);

    let ns = fields.namespace_id().unwrap();
    let toks: Vec<u32> = (0..4).map(|i| 80 + i).collect();
    let key2 = chunk_chain(&ns, 8, &toks).unwrap()[0];
    let header2 = fields.chunk_header(&ns, key2, None, 0, 4, 1).unwrap();
    let payload2 = pattern(9, BS);
    let mut buf = payload2.clone().into_boxed_slice();
    // SAFETY: `buf` outlives the refused job.
    let src =
        unsafe { PinnedSource::new(buf.as_mut_ptr(), BS, vec![0], 4, HolderGuard::detached()) };
    let job = s.submit_store(header2, src, Incarnation::seq(9)).unwrap();
    s.run_until_idle();
    assert_eq!(
        terminal_for(&s, job),
        Terminal::Failed(PlatformError::HardwareFault("seal failed".into()))
    );
    assert!(!s.object_path(&key2).exists());
    assert!(
        path.exists(),
        "restored object stays after a later seal failure"
    );
    assert_eq!(kvc_files(&dst_root.0), vec![path]);
    assert_pattern_absent(&dst_root.0, &payload2[..64]);
    drop(buf);
    drop(s);
}

#[test]
fn concurrent_seal_calls_take_distinct_nonces() {
    let key = key_with(Arc::new(SeqNonce::starting_at(0)));
    let barrier = Arc::new(Barrier::new(2));
    let mut handles = Vec::new();
    for i in 0..2u8 {
        let key = Arc::clone(&key);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            barrier.wait();
            let pt = vec![0x10 + i; 40];
            let mut file = plain(&pt);
            key.seal_file(&mut file).unwrap();
            let mut nonce = [0u8; NONCE_LEN];
            nonce.copy_from_slice(&file[NONCE_AT..NONCE_AT + NONCE_LEN]);
            let (got, body) = open_buf(&key, &file);
            assert_eq!(got, Ok(()));
            assert_eq!(body, pt);
            nonce
        }));
    }
    let mut nonces: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    nonces.sort();
    // Expected set, either scheduling order: nonce 0 and nonce 1.
    assert_eq!(nonces, vec![nonce_for(0), nonce_for(1)]);
}

#[test]
fn concurrent_host_writers_publish_distinct_nonces() {
    let key = key_with(Arc::new(SeqNonce::starting_at(10)));
    let barrier = Arc::new(Barrier::new(2));
    let fields = test_fields();
    let mut handles = Vec::new();
    for n in 1..=2u32 {
        let key = Arc::clone(&key);
        let barrier = Arc::clone(&barrier);
        handles.push(std::thread::spawn(move || {
            let root = Tmp::new("conc");
            let s = open_at(&root.0, key);
            barrier.wait();
            let pubd = publish(&s, &fields, n);
            let nonce = file_nonce(&pubd.path);
            let (term, bytes) = load_bytes(&s, pubd.key, &pubd.header);
            assert_eq!(term, Terminal::Done(JobStage::Completed));
            assert_eq!(bytes, pubd.payload);
            drop(s);
            nonce
        }));
    }
    let mut nonces: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
    nonces.sort();
    assert_eq!(nonces, vec![nonce_for(10), nonce_for(11)]);
}

#[test]
fn aad_binding_rejects_identity_digest_and_padding_bytes() {
    let key =
        SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(RepeatNonce(nonce_for(1))));
    let pt = vec![0x11u8; 48];
    let mut file = plain(&pt);
    key.seal_file(&mut file).unwrap();
    let (got, body) = open_buf(&key, &file);
    assert_eq!(got, Ok(()));
    assert_eq!(body, pt);

    // These offsets sit in `region[..AAD_LEN]` and are not the seal magic or
    // version (those fail `check_marker` before AEAD). Expected: Rejected,
    // and the payload buffer is all zeros. If `seal_inner` passes a shorter
    // AAD, or an empty AAD, the matching `open` returns Ok and this fails.
    for at in [0usize, 255, 300, 400] {
        assert!(at < AAD_LEN, "{at}");
        let mut bad = file.clone();
        bad[at] ^= 0x01;
        let (got, body) = open_buf(&key, &bad);
        assert_eq!(got, Err(SealError::Rejected), "AAD byte {at}");
        assert!(
            body.iter().all(|b| *b == 0),
            "AAD failure left a usable payload at {at}"
        );
        assert_ne!(body, pt);
    }
}

#[test]
fn failed_open_zeroes_payload_for_wrong_key_tamper_and_plaintext() {
    let key =
        SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(RepeatNonce(nonce_for(4))));
    let pt = vec![0x11u8; 48];
    let mut file = plain(&pt);
    key.seal_file(&mut file).unwrap();

    let other = SealKey::from_bytes(OTHER_KEY);
    let (got, body) = open_buf(&other, &file);
    assert_eq!(got, Err(SealError::Rejected));
    assert!(body.iter().all(|b| *b == 0));
    assert_ne!(body, pt);

    let mut tampered = file.clone();
    tampered[HEADER_REGION_LEN] ^= 0x01;
    let (got, body) = open_buf(&key, &tampered);
    assert_eq!(got, Err(SealError::Rejected));
    assert!(body.iter().all(|b| *b == 0));

    let raw = plain(&pt);
    let (got, body) = open_buf(&key, &raw);
    assert_eq!(got, Err(SealError::NotSealed));
    assert!(body.iter().all(|b| *b == 0));
    assert_ne!(body, pt);
}

fn publish_then_reopen(
    reopen: Arc<SealKey>,
    mutate: impl FnOnce(&mut Vec<u8>),
    check: impl FnOnce(&HostFsStore, &Published),
) {
    let root = Tmp::new("reopen");
    let fields = test_fields();
    let s = open_at(&root.0, key_with(Arc::new(SeqNonce::starting_at(0))));
    let pubd = publish(&s, &fields, 1);
    drop(s);
    let mut bytes = std::fs::read(&pubd.path).unwrap();
    mutate(&mut bytes);
    std::fs::write(&pubd.path, bytes).unwrap();
    let s = open_at(&root.0, reopen);
    check(&s, &pubd);
    drop(s);
}

#[test]
fn old_key_on_restore_discards_and_keeps_the_destination_poisoned() {
    publish_then_reopen(
        Arc::new(SealKey::from_bytes(OTHER_KEY)),
        |_| {},
        assert_discarded,
    );
}

#[test]
fn tampered_ciphertext_on_restore_discards_and_keeps_the_destination_poisoned() {
    publish_then_reopen(
        key_with(Arc::new(SeqNonce::starting_at(0))),
        |b| b[HEADER_REGION_LEN + 10] ^= 1,
        assert_discarded,
    );
}

#[test]
fn rewritten_identity_header_is_discarded_because_aad_binds_it() {
    // `written_unix_ns` is inside the identity header and inside the AAD.
    // `encode` recomputes the header checksum, so recovery accepts the file.
    // Expected: load fails `InvalidLayout`, destination stays `0xEE`.
    // An empty or shorter AAD lets this load return the plaintext.
    publish_then_reopen(
        key_with(Arc::new(SeqNonce::starting_at(0))),
        |b| {
            let mut rec = RecordHeader::decode(b).unwrap();
            rec.chunk.written_unix_ns ^= 0xFFFF;
            b[..HEADER_LEN].copy_from_slice(&rec.encode());
        },
        assert_discarded,
    );
}

#[test]
fn usage_budget_constants_match_the_policy_and_seal_file_does_not_enforce_them() {
    assert_eq!(RECOMMENDED_SEALS_PER_KEY, 1 << 20);
    assert_eq!(NIST_SP800_38D_RANDOM_NONCE_LIMIT, 1 << 32);
    let key =
        SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(SeqNonce::starting_at(0)));
    // Three successful seals. There is no counter that would refuse the third.
    // This does not stand in for 2^20 seals. The gap is the missing counter.
    for i in 0..3u8 {
        let mut file = plain(&[i; 16]);
        assert_eq!(key.seal_file(&mut file), Ok(()));
        assert_eq!(&file[NONCE_AT..NONCE_AT + NONCE_LEN], &nonce_for(i as u64));
    }
}

#[test]
fn scripted_source_ignores_a_nonce_environment_variable() {
    let _env = EnvSet::set("AIEN_KV_STORE_NONCE", "00112233445566778899aabb");
    let want = nonce_for(9);
    let key = SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(RepeatNonce(want)));
    let mut file = plain(&[0x03; 24]);
    key.seal_file(&mut file).unwrap();
    assert_eq!(&file[NONCE_AT..NONCE_AT + NONCE_LEN], &want);
}

#[test]
fn unsafe_control_repeated_nonce_is_not_detected_and_leaks_plaintext_xor() {
    let class = OutcomeClass::UnsafeControl;
    assert_eq!(class, OutcomeClass::UnsafeControl);
    assert_ne!(
        class,
        OutcomeClass::Protected,
        "do not relabel this control as protected"
    );

    let nonce = [0x42u8; NONCE_LEN];
    let key = SealKey::from_bytes(SYNTHETIC_KEY).with_nonce_source(Arc::new(RepeatNonce(nonce)));
    let pt1 = vec![0x11u8; 64];
    let pt2 = vec![0x22u8; 64];
    let mut a = plain(&pt1);
    let mut b = plain(&pt2);
    // Expected: both Ok. `Nonce::assume_unique_for_key` does not reject a repeat.
    assert_eq!(key.seal_file(&mut a), Ok(()));
    assert_eq!(key.seal_file(&mut b), Ok(()));
    assert_eq!(&a[NONCE_AT..NONCE_AT + NONCE_LEN], &nonce);
    assert_eq!(&b[NONCE_AT..NONCE_AT + NONCE_LEN], &nonce);
    let ct1 = &a[HEADER_REGION_LEN..];
    let ct2 = &b[HEADER_REGION_LEN..];
    assert_eq!(xor(ct1, ct2), xor(&pt1, &pt2));

    let mut same = plain(&pt1);
    key.seal_file(&mut same).unwrap();
    assert_eq!(&same[HEADER_REGION_LEN..], ct1);
}

#[test]
fn unsafe_control_host_fs_publishes_a_repeated_nonce() {
    let class = OutcomeClass::UnsafeControl;
    assert_eq!(class, OutcomeClass::UnsafeControl);
    assert_ne!(
        class,
        OutcomeClass::Protected,
        "do not relabel this control as protected"
    );

    let nonce = nonce_for(7);
    let root = Tmp::new("reuse");
    let fields = test_fields();
    let s = open_at(&root.0, key_with(Arc::new(RepeatNonce(nonce))));
    let a = publish(&s, &fields, 1);
    let b = publish(&s, &fields, 2);
    // Expected: both objects exist, both nonces equal `nonce`, each load
    // returns its own plaintext, and the payload XOR matches. The store
    // accepted the repeat. That acceptance is the unsafe control.
    assert_eq!(file_nonce(&a.path), nonce);
    assert_eq!(file_nonce(&b.path), nonce);
    let ca = std::fs::read(&a.path).unwrap();
    let cb = std::fs::read(&b.path).unwrap();
    assert_eq!(
        xor(&ca[HEADER_REGION_LEN..], &cb[HEADER_REGION_LEN..]),
        xor(&a.payload, &b.payload)
    );
    let (ta, ba) = load_bytes(&s, a.key, &a.header);
    let (tb, bb) = load_bytes(&s, b.key, &b.header);
    assert_eq!(ta, Terminal::Done(JobStage::Completed));
    assert_eq!(tb, Terminal::Done(JobStage::Completed));
    assert_eq!(ba, a.payload);
    assert_eq!(bb, b.payload);
    drop(s);
}
