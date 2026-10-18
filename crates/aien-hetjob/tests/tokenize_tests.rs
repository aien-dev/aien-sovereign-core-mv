//! tokenize_batch over a socket pair: a toy tokenizer.json in our BPE's
//! format, input files by digest from the worker's blob store, and every way
//! an input can be wrong (missing, tampered, too big, unusable).

use aien_bitnet_tokenizer::LLAMA3_SPLIT_PATTERN;
use aien_hetjob::coordinator::{dispatch, Job, OracleCheck, Outcome, Policy};
use aien_hetjob::receipt::Platform;
use aien_hetjob::store::{self, Store};
use aien_hetjob::task::{self, Control, Inputs, Limits, SpecError, TaskSpec};
use aien_hetjob::wire::Ids;
use aien_hetjob::worker::{Summary, Worker};
use ed25519_dalek::SigningKey;
use sha2::{Digest, Sha256};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;
use std::time::{Duration, Instant};

fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("aien-hetjob-tok-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn key() -> SigningKey {
    SigningKey::from_bytes(&[1; 32])
}

fn worker(dir: &Path, limits: Limits) -> Worker {
    Worker {
        key: key(),
        limits,
        store: Store::open(dir).unwrap(),
        platform: Platform {
            os: "testos".into(),
            arch: "testarch".into(),
            cpus: 2,
            memory_bytes: 1 << 30,
            program_digest: [7; 32],
            version: "0.1.0".into(),
        },
    }
}

fn ids(n: u8) -> Ids {
    Ids {
        cause_id: [0xC1; 32],
        origin_run_id: [0; 32],
        objective_id: [0x0B; 32],
        subject_id: [0; 32],
        task_id: [n; 16],
        idempotence_key: [n; 16],
    }
}

/// GPT-2's byte alphabet: printable bytes map to themselves, the rest to
/// U+0100 upwards in byte order.
fn byte_alphabet() -> Vec<char> {
    let keep =
        |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || (174..=255).contains(&b);
    let mut next = 256;
    (0..256u32)
        .map(|b| {
            if keep(b) {
                char::from_u32(b).unwrap()
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            }
        })
        .collect()
}

/// A toy tokenizer.json: the 256 bytes plus "ab", "abc", "Ġab", "bc".
fn toy_tokenizer() -> Vec<u8> {
    let mut vocab = serde_json::Map::new();
    for (b, c) in byte_alphabet().iter().enumerate() {
        vocab.insert(c.to_string(), (b as u64).into());
    }
    vocab.insert("ab".into(), 256.into());
    vocab.insert("abc".into(), 257.into());
    vocab.insert("Ġab".into(), 258.into());
    vocab.insert("bc".into(), 259.into());
    let special = |id: u32, s: &str| {
        serde_json::json!({"id": id, "content": s, "special": true, "normalized": false,
            "lstrip": false, "rstrip": false, "single_word": false})
    };
    serde_json::json!({
        "normalizer": null,
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": LLAMA3_SPLIT_PATTERN}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": false}]},
        "post_processor": {"type": "Sequence", "processors": [{"type": "TemplateProcessing",
            "special_tokens": {"<|begin_of_text|>": {"ids": [260]}}}]},
        "added_tokens": [special(260, "<|begin_of_text|>"), special(261, "<|eot_id|>")],
        "model": {"type": "BPE", "ignore_merges": true, "byte_fallback": false, "dropout": null,
            "continuing_subword_prefix": null, "end_of_word_suffix": null,
            "vocab": vocab, "merges": ["b c", "a b", "ab c", "Ġ ab"]}
    })
    .to_string()
    .into_bytes()
}

const CORPUS: &str =
    "abc ab abab\n\nüber 3½ ab<|eot_id|>abc\n12345 it's  ok \t\nlast line without newline";

fn inputs(corpus: &str) -> Inputs {
    Inputs {
        tokenizer: toy_tokenizer(),
        corpus: corpus.as_bytes().to_vec(),
    }
}

fn spec_of(i: &Inputs, bos: bool) -> TaskSpec {
    TaskSpec::TokenizeBatch {
        tokenizer: Sha256::digest(&i.tokenizer).into(),
        corpus: Sha256::digest(&i.corpus).into(),
        bos,
    }
}

/// Put both input files into the worker's blob store.
fn provision(dir: &Path, i: &Inputs) {
    store::put_blob(dir, &i.tokenizer).unwrap();
    store::put_blob(dir, &i.corpus).unwrap();
}

fn run_job(w: &Worker, job: &Job, oracle: Option<[u8; 32]>) -> (Outcome, Summary) {
    let p = Policy {
        deadline: Duration::from_secs(20),
        cancel_grace: Duration::from_secs(5),
        oracle,
    };
    let roster = [key().verifying_key()];
    let (c, s) = UnixStream::pair().unwrap();
    std::thread::scope(|sc| {
        let h = sc.spawn(|| {
            let s2 = s.try_clone().unwrap();
            w.serve(s2, s).unwrap()
        });
        let r = c.try_clone().unwrap();
        let mut wr = c;
        let (o, _) = dispatch(r, &mut wr, job, &roster, &p).unwrap();
        wr.shutdown(std::net::Shutdown::Both).unwrap();
        (o, h.join().unwrap())
    })
}

fn failed(o: &Outcome) -> String {
    match o {
        Outcome::Failed(s) => s.clone(),
        other => panic!("expected Failed, got {other:?}"),
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn verified_tokenize_matches_the_recompute() {
    let dir = tmpdir("happy");
    let i = inputs(CORPUS);
    provision(&dir, &i);
    let w = worker(&dir, Limits::default());
    for (n, bos) in [(1u8, false), (2, true)] {
        let spec = spec_of(&i, bos);
        let s = task::tokenize_summary(&spec, &i).unwrap();
        assert_eq!(s.lines, 5);
        assert_eq!(s.round_trips, 5, "every line decodes back to itself");
        let job = Job { spec, ids: ids(n) };
        let (o, sum) = run_job(&w, &job, Some(s.digest));
        match o {
            Outcome::Verified(v) => {
                assert_eq!(v.oracle, OracleCheck::Matched);
                assert_eq!(v.output, s.digest);
                assert_eq!(v.receipt.kind, "tokenize_batch");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(sum.completed, 1);
    }
}

/// Frozen: any change to the encoder, the line split or the digest layout
/// shows up here (the worker and coordinator both use these bytes).
#[test]
fn golden_tokenize_digests() {
    let i = inputs(CORPUS);
    let plain = task::tokenize_summary(&spec_of(&i, false), &i).unwrap();
    let bos = task::tokenize_summary(&spec_of(&i, true), &i).unwrap();
    assert_eq!(
        (plain.tokens, hex(&plain.digest)),
        (
            57,
            String::from("0c06c2e7ee01712604c5271f30cf3e37c061e21d8f97b23a74d78a25f0c686ea")
        )
    );
    assert_eq!(
        (bos.tokens, hex(&bos.digest)),
        (
            62,
            String::from("81abf962cd4d3c69d4199a1e67dc7c19c9bb01fc27807116b7c32faa68188679")
        )
    );
}

#[test]
fn every_thread_count_gives_the_same_digest() {
    let i = inputs(&CORPUS.repeat(9));
    let spec = spec_of(&i, true);
    let want = task::tokenize_summary(&spec, &i).unwrap().digest;
    let cancel = AtomicBool::new(false);
    for threads in [1, 2, 3, 7, 64] {
        let ctl = Control {
            cancel: &cancel,
            deadline: Instant::now() + Duration::from_secs(30),
            threads,
        };
        assert_eq!(
            task::run(&spec, &i, &ctl).unwrap(),
            want,
            "threads={threads}"
        );
    }
}

#[test]
fn missing_and_tampered_inputs_create_no_task() {
    let dir = tmpdir("missing");
    let i = inputs(CORPUS);
    let spec = spec_of(&i, false);
    store::put_blob(&dir, &i.tokenizer).unwrap();
    let w = worker(&dir, Limits::default());
    let (o, sum) = run_job(&w, &Job { spec, ids: ids(1) }, None);
    assert_eq!(
        failed(&o),
        format!("MISSING_INPUT:{}", hex(&Sha256::digest(&i.corpus)))
    );
    assert_eq!((sum.denied, sum.completed), (1, 0));

    // The corpus file exists under its digest, but its bytes were changed.
    let corpus_digest: [u8; 32] = Sha256::digest(&i.corpus).into();
    std::fs::write(store::blob_path(&dir, &corpus_digest), b"abc tampered").unwrap();
    let (o, sum) = run_job(&w, &Job { spec, ids: ids(2) }, None);
    assert_eq!(
        failed(&o),
        format!("INPUT_DIGEST_MISMATCH:{}", hex(&corpus_digest))
    );
    assert_eq!(sum.denied, 1);
}

#[test]
fn oversize_inputs_are_refused_before_reading() {
    let dir = tmpdir("big");
    let i = inputs(CORPUS);
    provision(&dir, &i);
    let small_blobs = Limits {
        max_blob_bytes: 1024,
        ..Limits::default()
    };
    let w = worker(&dir, small_blobs);
    let (o, _) = run_job(
        &w,
        &Job {
            spec: spec_of(&i, false),
            ids: ids(1),
        },
        None,
    );
    assert!(failed(&o).starts_with("OVER_BUDGET:MEMORY need="), "{o:?}");

    let little_memory = Limits {
        max_memory_bytes: 4096,
        ..Limits::default()
    };
    let dir2 = tmpdir("mem");
    provision(&dir2, &i);
    let w = worker(&dir2, little_memory);
    let (o, _) = run_job(
        &w,
        &Job {
            spec: spec_of(&i, false),
            ids: ids(1),
        },
        None,
    );
    assert!(failed(&o).starts_with("OVER_BUDGET:MEMORY need="), "{o:?}");
}

#[test]
fn unusable_inputs_fail_with_the_reason() {
    let dir = tmpdir("invalid");
    let w = worker(&dir, Limits::default());
    // Valid JSON, but not a tokenizer our BPE implements.
    let wrong = Inputs {
        tokenizer: br#"{"normalizer": null, "model": {"type": "WordPiece"}}"#.to_vec(),
        corpus: b"abc\n".to_vec(),
    };
    provision(&dir, &wrong);
    let (o, sum) = run_job(
        &w,
        &Job {
            spec: spec_of(&wrong, false),
            ids: ids(1),
        },
        None,
    );
    assert_eq!(failed(&o), "INVALID_INPUT:TOKENIZER_REFUSED");
    assert_eq!((sum.failed, sum.denied), (1, 0));

    let not_utf8 = Inputs {
        tokenizer: toy_tokenizer(),
        corpus: vec![b'a', 0xFF, b'\n'],
    };
    provision(&dir, &not_utf8);
    let (o, _) = run_job(
        &w,
        &Job {
            spec: spec_of(&not_utf8, false),
            ids: ids(2),
        },
        None,
    );
    assert_eq!(failed(&o), "INVALID_INPUT:CORPUS_UTF8");
}

#[test]
fn request_payload_round_trips_and_refuses_unknown_flags() {
    let i = inputs(CORPUS);
    for bos in [false, true] {
        let spec = spec_of(&i, bos);
        let p = spec.encode();
        assert_eq!(p.len(), 66);
        assert_eq!(TaskSpec::decode(&p), Ok(spec));
        let mut bad = p.clone();
        bad[65] = 2;
        assert_eq!(TaskSpec::decode(&bad), Err(SpecError::BadFlags));
        assert_eq!(TaskSpec::decode(&p[..65]), Err(SpecError::BadLength));
    }
}
