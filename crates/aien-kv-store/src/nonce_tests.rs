use super::*;
use std::sync::Arc;
include!("nonce_process_tests.rs");

fn plain(byte: u8) -> Vec<u8> {
    let mut v = vec![byte; HEADER_REGION_LEN + 32];
    v[..HEADER_REGION_LEN].fill(0);
    v
}
#[test]
fn entropy_error_is_transactional() {
    let key = SealKey::from_bytes([7; 32]).test_source(Arc::new(|_| Err(SealError::Random)));
    let mut file = plain(3);
    let before = file.clone();
    assert_eq!(
        key.seal_file(&mut file),
        Err(SealError::Random),
        "entropy failure must propagate"
    );
    assert_eq!(
        file, before,
        "entropy failure must leave complete caller buffer unchanged"
    );
}
#[test]
fn repeated_nonce_is_unsafe_control_not_protection() {
    let key = SealKey::from_bytes([7; 32]).test_source(Arc::new(|out| {
        out.fill(5);
        Ok(())
    }));
    let (mut a, mut b) = (plain(3), plain(9));
    key.seal_file(&mut a).unwrap();
    key.seal_file(&mut b).unwrap();
    assert_eq!(
        &a[O_NONCE..O_TAG],
        &[5; 12],
        "injected source must be consumed"
    );
    assert_eq!(&a[O_NONCE..O_TAG], &b[O_NONCE..O_TAG]);
    assert!(
        a[HEADER_REGION_LEN..]
            .iter()
            .zip(&b[HEADER_REGION_LEN..])
            .all(|(a, b)| a ^ b == 3 ^ 9),
        "repeated GCM nonce discloses payload XOR, excluding tags"
    );
    let (r, p) = a.split_at_mut(HEADER_REGION_LEN);
    key.open_payload(r, p).unwrap();
    assert_eq!(p, &[3; 32]);
}
#[test]
fn aad_every_byte_and_failed_open_zeroes_payload() {
    let key = SealKey::from_bytes([7; 32]);
    let mut file = plain(3);
    key.seal_file(&mut file).unwrap();
    for at in 0..AAD_LEN {
        let mut corrupt = file.clone();
        corrupt[at] ^= 1;
        let (r, p) = corrupt.split_at_mut(HEADER_REGION_LEN);
        assert!(
            key.open_payload(r, p).is_err(),
            "AAD byte {at} must be authenticated"
        );
        assert!(
            p.iter().all(|b| *b == 0),
            "failed open must zero usable payload"
        );
    }
}
#[test]
fn independent_instances_threads_and_restored_files() {
    let counter = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let source: TestNonceSource = {
        let c = counter.clone();
        Arc::new(move |out| {
            out.fill(0);
            out[..8].copy_from_slice(&c.fetch_add(1, Ordering::SeqCst).to_le_bytes());
            Ok(())
        })
    };
    let jobs: Vec<_> = (0..8)
        .map(|_| {
            let source = source.clone();
            std::thread::spawn(move || {
                let key = SealKey::from_bytes([7; 32]).test_source(source);
                let mut f = plain(3);
                key.seal_file(&mut f).unwrap();
                f
            })
        })
        .collect();
    let files: Vec<_> = jobs.into_iter().map(|j| j.join().unwrap()).collect();
    let mut seen = std::collections::HashSet::new();
    for f in &files {
        assert!(
            seen.insert(f[O_NONCE..O_TAG].to_vec()),
            "shared synthetic source sequence must survive recreated keys/threads"
        );
    }
    assert_eq!(
        counter.load(Ordering::SeqCst),
        8,
        "each concurrent seal must consume exactly one injected nonce"
    );
    // Disk restoration does not restore source state. Recreate a key as after restart.
    let key = SealKey::from_bytes([7; 32]).test_source(source);
    let mut next = plain(3);
    key.seal_file(&mut next).unwrap();
    assert!(seen.insert(next[O_NONCE..O_TAG].to_vec()));
    let mut restored = files[0].clone();
    let (r, p) = restored.split_at_mut(HEADER_REGION_LEN);
    key.open_payload(r, p).unwrap();
    assert_eq!(
        p, &[3; 32],
        "valid old file authenticates but does not prove freshness"
    );
}
