//! sovereign-core #408, recovery audit row 8: receipt continuity across a
//! restart. The record mark names the newest record's digest and the record
//! count; each Cortex record digest covers only its own record. So a record
//! before the newest one, rewritten with a freshly computed digest, opened as
//! if intact. The mark now also carries the running hash over every record
//! digest (the same chain omega keeps in memory, `rx_cortex.c` chain_step),
//! and the open recomputes it: a forged, reordered, removed or spliced record
//! anywhere before the mark is refused by name (`E_MARK_CHAIN`).
//!
//! Each case rewrites `cortex.cx` the way someone with the file (and no key)
//! can: every tampered record gets a valid digest, so omega's own open check
//! passes and only the continuity check can catch it. Needs librx_compose.a;
//! in a stub build the cases return early.
#[path = "support/home_guard.rs"]
mod home_guard;

use aien_runtime::compose::{ComposeBridge, ComposeProposer, Generation};
use aien_runtime::control::ControlResponse;
use aien_runtime::cortex_mark;
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::sync::Arc;

// Journal layout (omega rx_cortex.c): a 4-word header, then per record a
// 13-word fixed part (marker, id, cls<<32|kind, subject, t, generation,
// branch<<32|protect, tag, links x4, n), n payload words, 4 digest words.
const HDR_BYTES: usize = 32;
const FIXED: usize = 13;
const W_ID: usize = 1;
const W_T: usize = 4;
const W_N: usize = 12;
const DOMAIN: &[u8] = b"AIEN.CORTEX.OBJECT.V1";

#[derive(Clone)]
struct Rec {
    w: [u64; FIXED],
    payload: Vec<u64>,
    digest: [u8; 32],
}

fn le(b: &[u8], i: usize) -> u64 {
    u64::from_le_bytes(b[i..i + 8].try_into().unwrap())
}

/// omega `cx_digest`: sha256 over the domain, fixed words 1..13, payload.
fn digest_of(w: &[u64; FIXED], payload: &[u64]) -> [u8; 32] {
    let mut h = Sha256::new();
    h.update(DOMAIN);
    for x in &w[1..] {
        h.update(x.to_le_bytes());
    }
    for x in payload {
        h.update(x.to_le_bytes());
    }
    h.finalize().into()
}

fn read_journal(p: &Path) -> (Vec<u8>, Vec<Rec>) {
    let b = std::fs::read(p).unwrap();
    let hdr = b[..HDR_BYTES].to_vec();
    let mut recs = Vec::new();
    let mut at = HDR_BYTES;
    while at < b.len() {
        let mut w = [0u64; FIXED];
        for (i, x) in w.iter_mut().enumerate() {
            *x = le(&b, at + 8 * i);
        }
        let n = w[W_N] as usize;
        let payload = (0..n).map(|i| le(&b, at + 8 * (FIXED + i))).collect();
        let d0 = at + 8 * (FIXED + n);
        let digest: [u8; 32] = b[d0..d0 + 32].try_into().unwrap();
        recs.push(Rec { w, payload, digest });
        at = d0 + 32;
    }
    assert_eq!(at, b.len(), "journal parse ends on a record boundary");
    // Control: this parser and digest agree with every record omega wrote.
    for (i, r) in recs.iter().enumerate() {
        assert_eq!(r.w[W_ID], i as u64 + 1);
        assert_eq!(digest_of(&r.w, &r.payload), r.digest, "record #{}", i + 1);
    }
    (hdr, recs)
}

/// Renumber every record to its position and give it a valid digest, as a
/// forger without a key can.
fn write_journal(p: &Path, hdr: &[u8], recs: &mut [Rec]) {
    let mut b = hdr.to_vec();
    for (i, r) in recs.iter_mut().enumerate() {
        r.w[W_ID] = i as u64 + 1;
        r.w[W_N] = r.payload.len() as u64;
        r.digest = digest_of(&r.w, &r.payload);
        for x in r.w.iter().chain(&r.payload) {
            b.extend_from_slice(&x.to_le_bytes());
        }
        b.extend_from_slice(&r.digest);
    }
    std::fs::write(p, b).unwrap();
}

/// Rewrite the text of a host note payload (`[len, sha256 x4, bytes...]`)
/// to `to` (same length), with a matching payload sha256.
fn forge_note(r: &mut Rec, from: &str, to: &str) {
    assert_eq!(from.len(), to.len());
    let len = r.payload[0] as usize;
    let mut bytes: Vec<u8> = r.payload[5..]
        .iter()
        .flat_map(|w| w.to_le_bytes())
        .collect();
    let text = String::from_utf8(bytes[..len].to_vec()).unwrap();
    assert!(text.contains(from), "{text}");
    let forged = text.replacen(from, to, 1);
    bytes[..len].copy_from_slice(forged.as_bytes());
    for (i, chunk) in bytes.chunks(8).enumerate() {
        r.payload[5 + i] = u64::from_le_bytes(chunk.try_into().unwrap());
    }
    let sha: [u8; 32] = Sha256::digest(forged.as_bytes()).into();
    for i in 0..4 {
        r.payload[1 + i] = u64::from_le_bytes(sha[8 * i..8 * i + 8].try_into().unwrap());
    }
}

fn proposer() -> ComposeProposer {
    Arc::new(|_prompt: &str, _limit: std::time::Duration| {
        Ok(Generation {
            text: "filename: NOTES.md\nnote\n".into(),
            tokens: 4,
            finish_reason: Some("eos".into()),
            ..Default::default()
        })
    })
}

fn total(b: &ComposeBridge) -> u64 {
    match b.recall(&[], None) {
        ControlResponse::ComposeRecalled(r) => r.records_total,
        other => panic!("recall: {other:?}"),
    }
}

/// What the restarted home answers: the refusal text, or why it opened.
fn reopen_refusal(home: &Path) -> String {
    let b = ComposeBridge::new(home.to_path_buf(), proposer(), "test");
    match b.recall(&[], None) {
        ControlResponse::Error(e) => e,
        ControlResponse::ComposeRecalled(r) => panic!(
            "tampered journal accepted after the restart: opened with {} records",
            r.records_total
        ),
        other => panic!("unexpected answer {other:?}"),
    }
}

/// A home with five consecutive constraint receipts and one more after them,
/// closed (the restart). Returns (tempdir, home, receipt ids, mark records).
fn setup() -> Option<(tempfile::TempDir, PathBuf, Vec<u64>, u64)> {
    let tmp = tempfile::tempdir().unwrap();
    let home = tmp.path().join("compose");
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let r = b.note("constraint", "keep changes inside the workspace", &[]);
    if !aien_omega_compose::LINKED {
        assert!(matches!(r, ControlResponse::Error(_)));
        return None;
    }
    let mut ids = Vec::new();
    for n in 0..6 {
        match b.note(
            "constraint",
            &json!({"approver": "drake", "n": n}).to_string(),
            &[],
        ) {
            ControlResponse::ComposeNoted(x) => ids.push(x.id),
            other => panic!("note: {other:?}"),
        }
    }
    assert!(ids.windows(2).all(|w| w[1] == w[0] + 1), "{ids:?}");
    let n = total(&b);
    drop(b);
    let m = cortex_mark::read(&cortex_mark::mark_path(&home))
        .unwrap()
        .unwrap();
    assert_eq!(m.records, n);
    // The receipts tampered below all sit before the newest marked record.
    assert!(ids[4] < m.records, "{ids:?} vs mark {}", m.records);
    // Negative control: untouched, the home reopens.
    let b = ComposeBridge::new(home.clone(), proposer(), "test");
    let after = total(&b);
    drop(b);
    let m = cortex_mark::read(&cortex_mark::mark_path(&home))
        .unwrap()
        .unwrap();
    assert_eq!(m.records, after);
    Some((tmp, home, ids, m.records))
}

fn idx(id: u64) -> usize {
    id as usize - 1
}

#[test]
fn forged_middle_receipt_is_refused() {
    let _home = home_guard::home_slot();
    let Some((_tmp, home, ids, _)) = setup() else {
        return;
    };
    let cx = home.join("cortex.cx");
    let (hdr, mut recs) = read_journal(&cx);
    forge_note(&mut recs[idx(ids[2])], "drake", "decoy");
    write_journal(&cx, &hdr, &mut recs);
    let e = reopen_refusal(&home);
    assert!(e.contains("E_MARK_CHAIN"), "{e}");
}

#[test]
fn reordered_receipts_are_refused() {
    let _home = home_guard::home_slot();
    let Some((_tmp, home, ids, _)) = setup() else {
        return;
    };
    let cx = home.join("cortex.cx");
    let (hdr, mut recs) = read_journal(&cx);
    // Swap two receipts; each keeps the logical time of its slot (omega
    // refuses time going backwards on a subject).
    let (a, b) = (idx(ids[1]), idx(ids[3]));
    let (ta, tb) = (recs[a].w[W_T], recs[b].w[W_T]);
    recs.swap(a, b);
    recs[a].w[W_T] = ta;
    recs[b].w[W_T] = tb;
    write_journal(&cx, &hdr, &mut recs);
    let e = reopen_refusal(&home);
    assert!(e.contains("E_MARK_CHAIN"), "{e}");
}

#[test]
fn gap_with_the_count_restored_is_refused() {
    let _home = home_guard::home_slot();
    let Some((_tmp, home, ids, _)) = setup() else {
        return;
    };
    let cx = home.join("cortex.cx");
    let (hdr, mut recs) = read_journal(&cx);
    // Drop receipt 1, repeat receipt 3 after itself: same count, same
    // newest record, one receipt gone.
    let dup = recs[idx(ids[3])].clone();
    recs.insert(idx(ids[3]) + 1, dup);
    recs.remove(idx(ids[1]));
    write_journal(&cx, &hdr, &mut recs);
    let e = reopen_refusal(&home);
    assert!(e.contains("E_MARK_CHAIN"), "{e}");
}

#[test]
fn spliced_foreign_receipt_is_refused() {
    let _home = home_guard::home_slot();
    let Some((_tmp, home, ids, _)) = setup() else {
        return;
    };
    // A receipt from another home, written by the same code.
    let Some((_tmp2, other, other_ids, _)) = setup() else {
        return;
    };
    let (_, foreign) = read_journal(&other.join("cortex.cx"));
    let mut f = foreign[idx(other_ids[2])].clone();
    forge_note(&mut f, "drake", "other");
    let cx = home.join("cortex.cx");
    let (hdr, mut recs) = read_journal(&cx);
    let at = idx(ids[2]);
    f.w[W_T] = recs[at].w[W_T];
    recs[at] = f;
    write_journal(&cx, &hdr, &mut recs);
    let e = reopen_refusal(&home);
    assert!(e.contains("E_MARK_CHAIN"), "{e}");
}

#[test]
fn truncated_tail_is_refused() {
    let _home = home_guard::home_slot();
    let Some((_tmp, home, ids, _)) = setup() else {
        return;
    };
    let cx = home.join("cortex.cx");
    let (hdr, mut recs) = read_journal(&cx);
    recs.truncate(idx(ids[3]));
    write_journal(&cx, &hdr, &mut recs);
    let e = reopen_refusal(&home);
    assert!(e.contains("E_MARK_TRUNCATED"), "{e}");

    // The same cut with the count padded back by repeating the last kept
    // record: refused by name (omega's J-Space anchor or the mark).
    let (hdr, mut recs) = read_journal(&cx);
    let mark_records = cortex_mark::read(&cortex_mark::mark_path(&home))
        .unwrap()
        .unwrap()
        .records;
    let last = recs.last().unwrap().clone();
    while (recs.len() as u64) < mark_records {
        recs.push(last.clone());
    }
    write_journal(&cx, &hdr, &mut recs);
    let e = reopen_refusal(&home);
    assert!(
        e.contains("E_MARK_DIGEST") || e.contains("E_MARK_CHAIN") || e.contains("E_REPLAY"),
        "{e}"
    );
}
