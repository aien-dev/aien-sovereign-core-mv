//! Artifact and index codecs, and the crash-safe on-disk store.

mod common;

use aien_expert_store::format::{decode_index, encode_index, hex, sha256};
use aien_expert_store::store::INDEX_FILE;
use aien_expert_store::{
    ArtifactHeader, ExpertStore, Projection, ScaleDtype, StoreError, StoreLimits, ALIGN,
    HEADER_REGION,
};
use common::*;

#[test]
fn header_layout_is_aligned_and_round_trips() {
    let h = ArtifactHeader::new(key(1, 2, Projection::Down), 200, 300, ScaleDtype::F32).unwrap();
    assert_eq!(h.weight_offset, HEADER_REGION as u64);
    assert_eq!(h.weight_len, 60_000);
    assert_eq!((h.scale_rows, h.scale_cols), (2, 3));
    assert_eq!(h.scale_len, 24);
    assert!(h.scale_offset.is_multiple_of(ALIGN));
    assert!(h.object_len.is_multiple_of(ALIGN));
    assert_eq!(ArtifactHeader::decode(&h.encode()).unwrap(), h);
}

#[test]
fn header_decode_refuses_any_tampered_field() {
    let h = ArtifactHeader::new(key(0, 0, Projection::Gate), 128, 256, ScaleDtype::Bf16).unwrap();
    let good = h.encode();
    for (at, want) in [
        (0usize, StoreError::BadMagic),
        (8, StoreError::UnsupportedVersion),
        (12, StoreError::BadHeader("header_len")),
        (56, StoreError::BadHeader("projection")),
        (57, StoreError::BadHeader("quant")),
        (58, StoreError::BadHeader("weight dtype")),
        (59, StoreError::BadHeader("scale dtype")),
        (60, StoreError::BadHeader("block")),
        (104, StoreError::BadHeader("derived fields disagree")),
        (4000, StoreError::BadHeader("nonzero reserved bytes")),
    ] {
        let mut b = good.clone();
        b[at] ^= 0x40;
        assert_eq!(ArtifactHeader::decode(&b), Err(want), "byte {at}");
    }
    assert!(ArtifactHeader::decode(&good[..100]).is_err());
}

#[test]
fn artifact_build_refuses_wrong_payload_lengths_and_size_bound() {
    let k = key(0, 0, Projection::Up);
    let w = vec![0u8; 128 * 128];
    let s = vec![0u8; 2];
    let build = aien_expert_store::format::build_artifact;
    assert!(build(k, 128, 128, ScaleDtype::Bf16, &w, &s, u64::MAX).is_ok());
    assert!(matches!(
        build(k, 128, 128, ScaleDtype::Bf16, &w[1..], &s, u64::MAX),
        Err(StoreError::PayloadLength { .. })
    ));
    assert!(matches!(
        build(k, 128, 128, ScaleDtype::F32, &w, &s, u64::MAX),
        Err(StoreError::PayloadLength { .. })
    ));
    assert!(matches!(
        build(k, 128, 128, ScaleDtype::Bf16, &w, &s, 4096),
        Err(StoreError::ObjectTooLarge { .. })
    ));
    assert_eq!(
        build(k, u64::MAX, 2, ScaleDtype::Bf16, &w, &s, u64::MAX).unwrap_err(),
        StoreError::Overflow
    );
}

#[test]
fn index_round_trips_and_refuses_duplicates_tamper_and_oversize() {
    let a = artifact(key(0, 0, Projection::Gate), 128, 128, 1).0;
    let b = artifact(key(0, 1, Projection::Gate), 128, 128, 2).0;
    let bytes = encode_index(&[a, b]).unwrap();
    assert_eq!(decode_index(&bytes, 2).unwrap(), vec![a, b]);
    assert_eq!(decode_index(&bytes, 1), Err(StoreError::TooManyEntries));
    assert!(encode_index(&[a, a]).is_err());
    assert!(encode_index(&[b, a]).is_err());

    let mut t = bytes.clone();
    t[30] ^= 1;
    assert_eq!(decode_index(&t, 2), Err(StoreError::IndexChecksum));
    assert!(decode_index(&bytes[..bytes.len() - 1], 2).is_err());

    // A duplicate key with a valid trailer is still refused.
    let mut dup = bytes[..bytes.len() - 32].to_vec();
    let rec = dup[24..24 + 84].to_vec();
    dup[24 + 84..24 + 168].copy_from_slice(&rec);
    let tr = sha256(&dup);
    dup.extend_from_slice(&tr);
    assert_eq!(
        decode_index(&dup, 2),
        Err(StoreError::BadIndex("entries not strictly ascending"))
    );
}

#[test]
fn store_put_commit_reopen_and_idempotence() {
    let d = TempDir::new("store");
    let mut s = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    let k = key(3, 4, Projection::Down);
    let (want, bytes, w, sc) = artifact(k, 256, 192, 9);
    let e = s.put(k, 256, 192, ScaleDtype::Bf16, &w, &sc).unwrap();
    assert_eq!(e, want);
    assert_eq!(s.put(k, 256, 192, ScaleDtype::Bf16, &w, &sc).unwrap(), want);
    let on_disk = std::fs::read(d.path().join("objects").join(hex(&e.data_digest))).unwrap();
    assert_eq!(on_disk, bytes);
    // Not visible to a reopen until committed.
    assert!(ExpertStore::open(d.path(), StoreLimits::default())
        .unwrap()
        .is_empty());
    s.commit().unwrap();
    let r = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    assert_eq!(r.get(&k), Some(want));
    assert_eq!(r.len(), 1);
}

#[test]
fn store_refuses_rebinding_a_key_to_other_content() {
    let d = TempDir::new("dup");
    let mut s = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    let k = key(0, 0, Projection::Gate);
    let (_, _, w, sc) = artifact(k, 128, 128, 1);
    s.put(k, 128, 128, ScaleDtype::Bf16, &w, &sc).unwrap();
    let mut w2 = w.clone();
    w2[0] ^= 1;
    assert_eq!(
        s.put(k, 128, 128, ScaleDtype::Bf16, &w2, &sc),
        Err(StoreError::DuplicateKey(k))
    );
}

#[test]
fn store_bounds_entries_and_object_size() {
    let d = TempDir::new("bounds");
    let lim = StoreLimits {
        max_entries: 1,
        max_object_bytes: 1 << 20,
    };
    let mut s = ExpertStore::open(d.path(), lim).unwrap();
    let (_, _, w, sc) = artifact(key(0, 0, Projection::Gate), 128, 128, 1);
    s.put(
        key(0, 0, Projection::Gate),
        128,
        128,
        ScaleDtype::Bf16,
        &w,
        &sc,
    )
    .unwrap();
    assert_eq!(
        s.put(
            key(0, 1, Projection::Gate),
            128,
            128,
            ScaleDtype::Bf16,
            &w,
            &sc
        ),
        Err(StoreError::TooManyEntries)
    );
    let big = vec![0u8; 1024 * 1024];
    let bs = vec![0u8; 8 * 8 * 2];
    let mut s2 = ExpertStore::open(TempDir::new("b2").path(), lim).unwrap();
    assert!(matches!(
        s2.put(
            key(0, 0, Projection::Up),
            1024,
            1024,
            ScaleDtype::Bf16,
            &big,
            &bs
        ),
        Err(StoreError::ObjectTooLarge { .. })
    ));
    // A committed index with more entries than the limit is refused on open.
    let mut s3 = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    s3.put(
        key(0, 1, Projection::Gate),
        128,
        128,
        ScaleDtype::Bf16,
        &w,
        &sc,
    )
    .unwrap();
    s3.put(
        key(0, 0, Projection::Gate),
        128,
        128,
        ScaleDtype::Bf16,
        &w,
        &sc,
    )
    .unwrap();
    s3.commit().unwrap();
    assert!(matches!(
        ExpertStore::open(d.path(), lim),
        Err(StoreError::TooManyEntries)
    ));
}

#[test]
fn crash_leftovers_are_cleaned_and_a_torn_index_is_refused() {
    let d = TempDir::new("crash");
    let mut s = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    let k = key(0, 0, Projection::Gate);
    let (_, _, w, sc) = artifact(k, 128, 128, 1);
    s.put(k, 128, 128, ScaleDtype::Bf16, &w, &sc).unwrap();
    s.commit().unwrap();
    // A temporary left by a crash mid-write.
    std::fs::write(d.path().join("tmp").join("index-v1.1.1"), b"partial").unwrap();
    let r = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    assert_eq!(r.len(), 1);
    assert_eq!(std::fs::read_dir(d.path().join("tmp")).unwrap().count(), 0);
    // A torn index (not produced by our rename protocol) fails closed.
    let p = d.path().join(INDEX_FILE);
    let full = std::fs::read(&p).unwrap();
    std::fs::write(&p, &full[..full.len() - 7]).unwrap();
    assert!(ExpertStore::open(d.path(), StoreLimits::default()).is_err());
}

#[test]
fn put_replaces_a_damaged_existing_object() {
    let d = TempDir::new("heal");
    let mut s = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    let k = key(0, 0, Projection::Gate);
    let (want, bytes, w, sc) = artifact(k, 128, 128, 1);
    s.put(k, 128, 128, ScaleDtype::Bf16, &w, &sc).unwrap();
    let p = d.path().join("objects").join(hex(&want.data_digest));
    std::fs::write(&p, b"garbage").unwrap();
    let mut s2 = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    s2.put(k, 128, 128, ScaleDtype::Bf16, &w, &sc).unwrap();
    assert_eq!(std::fs::read(&p).unwrap(), bytes);
}

#[test]
fn objects_and_index_are_owner_only() {
    use std::os::unix::fs::PermissionsExt;
    let d = TempDir::new("perm");
    let mut s = ExpertStore::open(d.path(), StoreLimits::default()).unwrap();
    let k = key(0, 0, Projection::Gate);
    let (want, _, w, sc) = artifact(k, 128, 128, 1);
    s.put(k, 128, 128, ScaleDtype::Bf16, &w, &sc).unwrap();
    s.commit().unwrap();
    for p in [
        d.path().join("objects").join(hex(&want.data_digest)),
        d.path().join(INDEX_FILE),
    ] {
        let mode = std::fs::metadata(p).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode, 0o600);
    }
}
