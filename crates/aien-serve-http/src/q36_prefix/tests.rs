//! Host tests of the shared prompt cache: longest match, leaf-first LRU, the sealed store
//! (round trip, tamper, another namespace), the switch, and the restore refusals (DRA-86).
use super::*;

#[derive(Debug, Clone, PartialEq, Default)]
struct S(Vec<u8>);

impl PoolSnap for S {
    fn host_bytes(&self) -> usize {
        self.0.len()
    }
    fn to_bytes(&self) -> Vec<u8> {
        self.0.clone()
    }
    fn from_bytes(b: &[u8]) -> Result<Self, String> {
        Ok(Self(b.to_vec()))
    }
    /// `put` fills the state with its token count.
    fn pos(&self) -> Option<usize> {
        self.0.first().map(|&b| b as usize)
    }
}

fn cfg(budget: usize) -> PoolConfig {
    PoolConfig {
        budget,
        every: 4,
        dir: None,
        mutant: false,
    }
}

fn toks(n: usize, salt: u32) -> Vec<u32> {
    (0..n as u32)
        .map(|i| if i < 6 { i } else { i * 31 + salt })
        .collect()
}

/// Inserts the node for `t` under its deepest prefix, K/V one value per position.
fn put(p: &mut PrefixPool<S>, t: &[u32]) -> bool {
    let (parent, from) = p.parent_for(t).map_or((None, 0), |(i, n)| (Some(i), n));
    let kv: Vec<u16> = t[from..].iter().map(|&x| x as u16).collect();
    p.insert(t.to_vec(), parent, kv, S(vec![t.len() as u8; 8]))
}

#[test]
fn the_longest_node_the_prompt_continues_wins_and_its_kv_comes_root_first() {
    let mut p = PrefixPool::new(&cfg(1 << 20));
    let a = toks(20, 0);
    for n in [4, 8, 12] {
        put(&mut p, &a[..n]);
    }
    let mut b = a.clone();
    b[10] = 9999;
    let (id, n) = p.longest(&b, b.len() - 1).unwrap();
    assert_eq!(n, 8);
    let r = p.restore(id, &b[..8], 1).unwrap();
    assert_eq!(r.pos, 8);
    let kv: Vec<u16> = r.kv.iter().flat_map(|(_, k)| k.iter().copied()).collect();
    assert_eq!(kv, a[..8].iter().map(|&x| x as u16).collect::<Vec<_>>());
    assert_eq!(r.kv.iter().map(|&(f, _)| f).collect::<Vec<_>>(), [0, 4]);
    assert_eq!(r.snap, &S(vec![8; 8]));
    // At most `max` tokens: the last prompt token is always read.
    assert_eq!(p.longest(&a[..12], 11).map(|x| x.1), Some(8));
    assert_eq!(p.divergence(&b), Some(10));
    assert_eq!(p.divergence(&a), None);
}

#[test]
fn eviction_drops_least_recently_used_leaves_and_keeps_parents_of_live_children() {
    let a = toks(16, 0);
    let b = toks(16, 7);
    // Node bytes: whole prefix 4 B a token + its K/V delta 2 B a value + 8 B state: a4 32, a8 48,
    // a12 64, b8 48, b12 64 (256 in all), a16 80.
    let mut p = PrefixPool::new(&cfg(300));
    for n in [4, 8, 12] {
        put(&mut p, &a[..n]);
    }
    put(&mut p, &b[..8]); // parent: a[..4] (b shares 6 tokens with a); 4 kv values
    put(&mut p, &b[..12]);
    assert_eq!(p.counters().nodes, 5);
    let (id, n) = p.longest(&a, 15).unwrap();
    assert!(p.restore(id, &a[..n], 1).is_ok()); // a's chain is now the newest
    put(&mut p, &a[..16]); // over budget: b[..12] (the oldest leaf) goes, then nothing else
    let c = p.counters();
    assert_eq!((c.nodes, c.evicted), (5, 1));
    assert_eq!(p.longest(&b, 15).map(|x| x.1), Some(8));
    assert_eq!(p.longest(&a, 16).map(|x| x.1), Some(16));
}

fn tmp_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("l375-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    d
}

fn key() -> SealKey {
    SealKey::from_bytes([7; 32])
}

fn files(d: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = std::fs::read_dir(d)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|e| e.to_str()) == Some(FILE_EXT))
        .collect();
    v.sort();
    v
}

#[test]
fn the_store_brings_every_node_back_after_a_restart() {
    let d = tmp_dir("rt");
    let ns = [3u8; 32];
    let a = toks(12, 0);
    {
        let mut p = PrefixPool::<S>::new(&cfg(1 << 20));
        p.open_store(&d, key(), ns, 1, |_| true).unwrap();
        for n in [4, 8, 12] {
            put(&mut p, &a[..n]);
        }
        p.flush();
    }
    assert_eq!(files(&d).len(), 3);
    let mode = std::fs::metadata(&d).unwrap().permissions().mode() & 0o777;
    assert_eq!(mode, 0o700);
    let mut q = PrefixPool::<S>::new(&cfg(1 << 20));
    q.open_store(&d, key(), ns, 1, |_| true).unwrap();
    assert_eq!(q.counters().loaded, 3);
    let (id, n) = q.longest(&a, 12).unwrap();
    assert_eq!(n, 12);
    let r = q.restore(id, &a, 1).unwrap();
    let kv: Vec<u16> = r.kv.iter().flat_map(|(_, k)| k.iter().copied()).collect();
    assert_eq!(kv, a.iter().map(|&x| x as u16).collect::<Vec<_>>());
    assert_eq!(r.snap, &S(vec![12; 8]));
    // Eviction deletes the file too.
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_tampered_file_another_key_or_another_namespace_is_discarded() {
    let d = tmp_dir("tamper");
    let ns = [3u8; 32];
    let a = toks(8, 0);
    {
        let mut p = PrefixPool::<S>::new(&cfg(1 << 20));
        p.open_store(&d, key(), ns, 1, |_| true).unwrap();
        put(&mut p, &a[..4]);
        put(&mut p, &a[..8]);
        p.flush();
    }
    // Another namespace: nothing opens (and the files go).
    let mut q = PrefixPool::<S>::new(&cfg(1 << 20));
    q.open_store(&d, key(), [4u8; 32], 1, |_| true).unwrap();
    assert_eq!((q.counters().loaded, q.counters().discarded), (0, 2));
    drop(q);
    {
        let mut p = PrefixPool::<S>::new(&cfg(1 << 20));
        p.open_store(&d, key(), ns, 1, |_| true).unwrap();
        put(&mut p, &a[..4]);
        put(&mut p, &a[..8]);
        p.flush();
    }
    // One flipped payload byte in the root: it and its child (no parent) are discarded.
    let f = files(&d)
        .into_iter()
        .find(|p| p.to_string_lossy().ends_with("-4.q36p"))
        .unwrap();
    let mut b = std::fs::read(&f).unwrap();
    let last = b.len() - 1;
    b[last] ^= 1;
    std::fs::write(&f, b).unwrap();
    let mut q = PrefixPool::<S>::new(&cfg(1 << 20));
    q.open_store(&d, key(), ns, 1, |_| true).unwrap();
    assert_eq!((q.counters().loaded, q.counters().discarded), (0, 2));
    drop(q);
    // Another key.
    {
        let mut p = PrefixPool::<S>::new(&cfg(1 << 20));
        p.open_store(&d, key(), ns, 1, |_| true).unwrap();
        put(&mut p, &a[..4]);
        p.flush();
    }
    let mut q = PrefixPool::<S>::new(&cfg(1 << 20));
    q.open_store(&d, SealKey::from_bytes([8; 32]), ns, 1, |_| true)
        .unwrap();
    assert_eq!(q.counters().loaded, 0);
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn a_shared_directory_is_refused() {
    let d = tmp_dir("mode");
    std::fs::create_dir(&d).unwrap();
    std::fs::set_permissions(&d, std::fs::Permissions::from_mode(0o755)).unwrap();
    let mut p = PrefixPool::<S>::new(&cfg(1 << 20));
    assert!(p.open_store(&d, key(), [0; 32], 1, |_| true).is_err());
    std::fs::remove_dir_all(&d).unwrap();
}

#[test]
fn the_namespace_binds_switches_but_not_the_pools_own() {
    let e = |v: &[(&str, &str)]| numeric_env(v.iter().map(|(k, v)| (k.to_string(), v.to_string())));
    let base = e(&[
        ("AIEN_Q36_STROKE", "0"),
        ("AIEN_Q36_PREFIX_POOL_GB", "8"),
        ("HOME", "/x"),
    ]);
    assert_eq!(base, [("AIEN_Q36_STROKE".to_string(), "0".to_string())]);
    let m = e(&[("AIEN_Q36_PREFIX_MUTANT", "1"), ("AIEN_Q36_PREFIX", "1")]);
    assert_eq!(m.len(), 1, "the switch moves no state byte; the mutant is bound");
    let ck = [1u8; 32];
    assert_ne!(
        namespace(&ck, "aa", 1, 2, &base),
        namespace(&ck, "aa", 1, 2, &e(&[("AIEN_Q36_STROKE", "1")]))
    );
    assert_ne!(
        namespace(&ck, "aa", 1, 2, &base),
        namespace(&ck, "ab", 1, 2, &base)
    );
}

// ---- DRA-86: the switch, the budget default and the restore refusals ----

#[test]
fn the_cache_is_off_without_the_switch_and_16_gib_with_it() {
    let off = PoolConfig::from_lookup(|_| None).unwrap();
    assert_eq!(off.budget, 0);
    assert!(!PrefixPool::<S>::new(&off).on());
    let gb_only =
        PoolConfig::from_lookup(|k| (k == "AIEN_Q36_PREFIX_POOL_GB").then(|| "4".into())).unwrap();
    assert_eq!(gb_only.budget, 0, "the budget alone must not turn the cache on");
    let on = PoolConfig::from_lookup(|k| (k == SWITCH_ENV).then(|| "1".into())).unwrap();
    assert_eq!(on.budget, 16 << 30);
    let half = PoolConfig::from_lookup(|k| match k {
        SWITCH_ENV => Some("1".into()),
        "AIEN_Q36_PREFIX_POOL_GB" => Some("0.5".into()),
        _ => None,
    })
    .unwrap();
    assert_eq!(half.budget, 1 << 29);
    assert!(PoolConfig::from_lookup(|k| (k == SWITCH_ENV).then(|| "yes".into())).is_err());
}

/// A pool holding a[..4], a[..8], a[..12] (a = toks(16, 0)); the id of a[..12].
fn chain() -> (PrefixPool<S>, Vec<u32>, u64) {
    let mut p = PrefixPool::new(&cfg(1 << 20));
    let a = toks(16, 0);
    for n in [4, 8, 12] {
        assert!(put(&mut p, &a[..n]));
    }
    let (id, n) = p.longest(&a, 12).unwrap();
    assert_eq!(n, 12);
    (p, a, id)
}

fn node<'a>(p: &'a mut PrefixPool<S>, t: &[u32]) -> &'a mut Node<S> {
    p.nodes.values_mut().find(|n| n.tokens[..] == *t).unwrap()
}

#[test]
fn a_restore_gives_back_the_saved_bytes_and_its_conversation_resident_size() {
    let (mut p, a, id) = chain();
    let r = p.restore(id, &a[..12], 1).unwrap();
    let kv: Vec<u16> = r.kv.iter().flat_map(|(_, k)| k.iter().copied()).collect();
    assert_eq!(kv, a[..12].iter().map(|&x| x as u16).collect::<Vec<_>>());
    assert_eq!(r.kv.iter().map(|&(f, _)| f).collect::<Vec<_>>(), [0, 4, 8]);
    assert_eq!(r.snap, &S(vec![12; 8]));
    // Node bytes 4 B a token + 2 B a K/V value + 8 B state: 32 + 48 + 64.
    assert_eq!(r.resident, 144);
    assert_eq!(p.resident(&a[..12]), Some((3, 144)));
    assert_eq!(p.resident(&a[..8]), Some((2, 80)));
    assert_eq!(p.counters().refused, 0);
}

#[test]
fn a_node_of_another_model_or_lib_tag_is_refused_and_dropped() {
    let (mut p, a, id) = chain();
    p.bind_tag([1; 32]); // the chip library or the weights moved
    assert_eq!(
        p.restore(id, &a[..12], 1).err(),
        Some(Refused::Stale { pos: 4 })
    );
    let c = p.counters();
    assert_eq!((c.nodes, c.bytes, c.refused), (0, 0, 1));
    // Nodes saved under the new tag restore.
    assert!(put(&mut p, &a[..4]));
    let (id, _) = p.longest(&a, 15).unwrap();
    assert!(p.restore(id, &a[..4], 1).is_ok());
}

#[test]
fn a_node_restored_as_other_tokens_or_one_turn_off_is_refused() {
    let (mut p, a, id) = chain();
    let mut b = a[..12].to_vec();
    b[11] ^= 1;
    assert_eq!(
        p.restore(id, &b, 1).err(),
        Some(Refused::TokenMismatch { pos: 12 })
    );
    // The node after 12 tokens offered as the state after 11 or 13: off by one, refused.
    let (mut p, a, id) = chain();
    assert!(matches!(
        p.restore(id, &a[..11], 1),
        Err(Refused::TokenMismatch { .. })
    ));
    let (mut p, a, id) = chain();
    assert!(matches!(
        p.restore(id, &a[..13], 1),
        Err(Refused::TokenMismatch { .. })
    ));
    // A stored digest that is not the tokens' (a corrupted node) is refused too.
    let (mut p, a, id) = chain();
    node(&mut p, &a[..12]).digest[0] ^= 1;
    assert_eq!(
        p.restore(id, &a[..12], 1).err(),
        Some(Refused::TokenMismatch { pos: 12 })
    );
}

#[test]
fn a_kv_gap_or_a_kv_or_state_length_mismatch_is_refused() {
    let (mut p, a, id) = chain();
    node(&mut p, &a[..12]).kv_from = 9;
    assert_eq!(
        p.restore(id, &a[..12], 1).err(),
        Some(Refused::KvGap {
            kv_from: 9,
            expected: 8
        })
    );
    let (mut p, a, id) = chain();
    Arc::make_mut(&mut node(&mut p, &a[..8]).kv).pop();
    assert_eq!(
        p.restore(id, &a[..12], 1).err(),
        Some(Refused::KvLength {
            pos: 8,
            values: 3,
            expected: 4
        })
    );
    // Two K/V values a position where the pool holds one.
    let (mut p, a, id) = chain();
    assert!(matches!(
        p.restore(id, &a[..12], 2),
        Err(Refused::KvLength { .. })
    ));
    // The state continues at 11 for a node of 12 tokens.
    let (mut p, a, id) = chain();
    node(&mut p, &a[..12]).snap = Arc::new(S(vec![11; 8]));
    assert_eq!(
        p.restore(id, &a[..12], 1).err(),
        Some(Refused::StateLength { state: 11, pos: 12 })
    );
    // A refused leaf is dropped; its parent stays.
    assert_eq!(p.longest(&a, 15).map(|x| x.1), Some(8));
}

#[test]
fn the_budget_evicts_the_least_recently_used_leaf_and_drops_a_node_alone_over_it() {
    let a = toks(16, 0);
    let b = toks(16, 7);
    // a4 32 B, a8 48, then b8 under a4 (4 K/V values): 32 + 8 + 8 = 48 B; budget 120.
    let mut p = PrefixPool::new(&cfg(120));
    put(&mut p, &a[..4]);
    put(&mut p, &a[..8]);
    put(&mut p, &b[..8]);
    let c = p.counters();
    assert_eq!(
        (c.nodes, c.bytes, c.evicted),
        (2, 80, 1),
        "a8 (the oldest leaf) goes"
    );
    assert!(c.bytes <= p.budget());
    assert_eq!(p.longest(&a, 15).map(|x| x.1), Some(4));
    assert_eq!(p.longest(&b, 15).map(|x| x.1), Some(8));
    // A node bigger than the whole budget is dropped at once.
    let mut q = PrefixPool::new(&cfg(40));
    assert!(!put(&mut q, &a[..8]));
    assert_eq!(q.counters().nodes, 0);
}

type Spoil = Box<dyn Fn(&mut PrefixPool<S>, &[u32]) -> Vec<u32>>;

/// The negative control: the base's restore (no checks) hands every bad node above to the row.
#[test]
fn the_unchecked_base_restores_every_bad_node() {
    let bad: Vec<Spoil> = vec![
        Box::new(|p, a| {
            p.bind_tag([1; 32]);
            a[..12].to_vec()
        }),
        Box::new(|_, a| a[..13].to_vec()),
        Box::new(|p, a| {
            node(p, &a[..12]).kv_from = 9;
            a[..12].to_vec()
        }),
        Box::new(|p, a| {
            Arc::make_mut(&mut node(p, &a[..8]).kv).pop();
            a[..12].to_vec()
        }),
        Box::new(|p, a| {
            node(p, &a[..12]).snap = Arc::new(S(vec![11; 8]));
            a[..12].to_vec()
        }),
    ];
    for (k, f) in bad.iter().enumerate() {
        let (mut checked, a, id) = chain();
        let t = f(&mut checked, &a);
        assert!(checked.restore(id, &t, 1).is_err(), "case {k} not refused");
        let (mut base, a, id) = chain();
        base.set_unchecked(true);
        let t = f(&mut base, &a);
        assert!(
            base.restore(id, &t, 1).is_ok(),
            "case {k}: the base refuses it too"
        );
    }
}
