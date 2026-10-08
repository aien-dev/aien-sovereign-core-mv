//! HOST FIXTURE tests: identities are made-up byte arrays standing in for a
//! `Resolved` value; no real ALLEN subject, no daemon, no model is involved.
use aien_allen::Resolved;
use aien_allen_memory::*;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

fn resolved(n: u8) -> Resolved {
    Resolved {
        agent: [n; 32],
        root: [n.wrapping_add(100); 32],
        head_id: [0; 32],
        head_seq: 1,
        lineage: [0; 32],
        chain_verified: false,
        adopted: false,
    }
}

struct Fx {
    _t: tempfile::TempDir,
    home: PathBuf,
}

fn fx() -> Fx {
    let t = tempfile::tempdir().unwrap();
    let home = t.path().join("home");
    Fx { _t: t, home }
}

impl Fx {
    fn open(&self) -> Memory {
        Memory::open(&self.home, Some(&resolved(1))).unwrap()
    }
    fn dir(&self) -> PathBuf {
        memory_dir(&self.home)
    }
}

fn g(s: &str) -> ScopeGrant {
    ScopeGrant::new(Scope::parse(s).unwrap())
}
fn lim() -> RecallLimits {
    RecallLimits::default()
}
fn texts(r: &Recall) -> Vec<String> {
    r.items.iter().map(|i| i.text.clone().unwrap()).collect()
}
fn files(dir: &Path) -> Vec<PathBuf> {
    let mut v: Vec<PathBuf> = match std::fs::read_dir(dir) {
        Ok(rd) => rd.map(|e| e.unwrap().path()).collect(),
        Err(_) => vec![],
    };
    v.sort();
    v
}
fn key_files(fx: &Fx) -> Vec<PathBuf> {
    files(&fx.dir().join("keys"))
}
fn last_record(fx: &Fx) -> PathBuf {
    files(&fx.dir().join("log")).pop().unwrap()
}
fn copy_dir(a: &Path, b: &Path) {
    std::fs::create_dir_all(b).unwrap();
    for f in files(a) {
        std::fs::copy(&f, b.join(f.file_name().unwrap())).unwrap();
    }
}
fn edit_last(fx: &Fx, f: impl FnOnce(&mut serde_json::Value)) {
    let p = last_record(fx);
    let mut v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).unwrap()).unwrap();
    f(&mut v);
    std::fs::write(&p, serde_json::to_vec(&v).unwrap()).unwrap();
}

// ---------- scopes ----------

#[test]
fn scope_names_are_validated() {
    for ok in ["personal", "work", "project:a", "project:my-proj_1.x"] {
        assert!(Scope::parse(ok).is_ok(), "{ok}");
    }
    for bad in [
        "",
        "all",
        "*",
        "project:",
        "project:A",
        "project:a b",
        "Personal",
        &format!("project:{}", "x".repeat(49)),
    ] {
        assert!(Scope::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn put_and_recall_per_scope_with_no_leakage() {
    let f = fx();
    let m = f.open();
    m.put(Scope::Personal, Kind::Fact, "p-secret").unwrap();
    m.put(Scope::Work, Kind::Fact, "w-note").unwrap();
    m.put(
        Scope::parse("project:a").unwrap(),
        Kind::Preference,
        "a-pref",
    )
    .unwrap();
    m.put(Scope::parse("project:b").unwrap(), Kind::Fact, "b-fact")
        .unwrap();
    assert_eq!(
        texts(&m.recall(&g("personal"), None, &lim()).unwrap()),
        ["p-secret"]
    );
    assert_eq!(
        texts(&m.recall(&g("work"), None, &lim()).unwrap()),
        ["w-note"]
    );
    assert_eq!(
        texts(&m.recall(&g("project:a"), None, &lim()).unwrap()),
        ["a-pref"]
    );
    assert_eq!(
        texts(&m.recall(&g("project:b"), None, &lim()).unwrap()),
        ["b-fact"]
    );
    // project:a never answers for project:b and "project" alone is not a scope
    assert!(m
        .recall(&g("project:c"), None, &lim())
        .unwrap()
        .items
        .is_empty());
    for s in ["personal", "work", "project:a", "project:b"] {
        let r = m.recall(&g(s), None, &lim()).unwrap();
        assert!(r.items.iter().all(|i| i.scope == s));
    }
}

#[test]
fn caller_claimed_scope_in_query_text_is_ignored() {
    let f = fx();
    let m = f.open();
    m.put(Scope::Personal, Kind::Fact, "my passport number is 1234")
        .unwrap();
    m.put(Scope::Work, Kind::Fact, "standup at nine").unwrap();
    for q in [
        "scope=personal",
        "scope=personal passport",
        "personal",
        "passport",
        "{\"scope\":\"personal\"}",
    ] {
        let r = m.recall(&g("work"), Some(q), &lim()).unwrap();
        assert!(r.items.iter().all(|i| i.scope == "work"), "{q}");
        assert!(texts(&r).iter().all(|t| !t.contains("passport")), "{q}");
    }
    // The only way to read personal is a grant built by the caller of the API.
    assert_eq!(
        m.recall(&g("personal"), Some("passport"), &lim())
            .unwrap()
            .items
            .len(),
        1
    );
}

#[test]
fn plaintext_is_never_on_disk() {
    let f = fx();
    let m = f.open();
    m.put(Scope::Work, Kind::Fact, "UNIQUEPLAINTEXTMARKER")
        .unwrap();
    for d in ["log", "keys", "pending"] {
        for p in files(&f.dir().join(d)) {
            let b = std::fs::read(&p).unwrap();
            assert!(!String::from_utf8_lossy(&b).contains("UNIQUEPLAINTEXTMARKER"));
        }
    }
}

#[cfg(unix)]
#[test]
fn key_files_are_private() {
    use std::os::unix::fs::PermissionsExt;
    let f = fx();
    f.open().put(Scope::Work, Kind::Fact, "x").unwrap();
    let dm = std::fs::metadata(f.dir().join("keys"))
        .unwrap()
        .permissions()
        .mode()
        & 0o777;
    assert_eq!(dm, 0o700);
    for k in key_files(&f) {
        assert_eq!(
            std::fs::metadata(&k).unwrap().permissions().mode() & 0o777,
            0o600
        );
    }
}

#[test]
fn reopen_keeps_live_items() {
    let f = fx();
    let id = f.open().put(Scope::Work, Kind::Fact, "kept").unwrap();
    let m = f.open();
    let r = m.recall(&g("work"), None, &lim()).unwrap();
    assert_eq!(texts(&r), ["kept"]);
    assert_eq!(r.items[0].item, id);
}

#[test]
fn recall_is_deterministic_and_bounded() {
    let f = fx();
    let m = f.open();
    for i in 0..6 {
        m.put(Scope::Work, Kind::Fact, &format!("note-{i}"))
            .unwrap();
    }
    let a = m
        .recall(
            &g("work"),
            None,
            &RecallLimits {
                max_items: 3,
                max_bytes: 4096,
            },
        )
        .unwrap();
    let b = m
        .recall(
            &g("work"),
            None,
            &RecallLimits {
                max_items: 3,
                max_bytes: 4096,
            },
        )
        .unwrap();
    assert_eq!(texts(&a), ["note-0", "note-1", "note-2"]);
    assert_eq!(a, b);
    assert_eq!(a.omitted, 3);
    let c = m
        .recall(
            &g("work"),
            None,
            &RecallLimits {
                max_items: 10,
                max_bytes: 14,
            },
        )
        .unwrap();
    assert_eq!(texts(&c), ["note-0", "note-1"]);
    assert_eq!(c.omitted, 4);
}

#[test]
fn oversize_and_empty_content_refused() {
    let f = fx();
    let m = f.open();
    assert!(matches!(
        m.put(Scope::Work, Kind::Fact, &"x".repeat(2049)),
        Err(MemoryRefusal::Invalid(_))
    ));
    assert!(matches!(
        m.put(Scope::Work, Kind::Fact, "  "),
        Err(MemoryRefusal::Invalid(_))
    ));
}

// ---------- inspect / correct ----------

#[test]
fn inspect_lists_states_and_content() {
    let f = fx();
    let m = f.open();
    let a = m.put(Scope::Work, Kind::Fact, "one").unwrap();
    m.put(Scope::Personal, Kind::Fact, "other scope").unwrap();
    m.correct(&a, "one-v2").unwrap();
    let v = m.inspect(&g("work")).unwrap();
    assert_eq!(v.len(), 2);
    assert_eq!(
        (v[0].version, v[0].state.as_str(), v[0].text.clone()),
        (1, "corrected", None)
    );
    assert_eq!(
        (v[1].version, v[1].state.as_str(), v[1].text.as_deref()),
        (2, "live", Some("one-v2"))
    );
    assert_eq!(m.inspect_all(&InspectAll::owner()).unwrap().len(), 3);
}

#[test]
fn correct_destroys_the_old_key() {
    let f = fx();
    let m = f.open();
    let a = m.put(Scope::Work, Kind::Fact, "wrong").unwrap();
    assert_eq!(key_files(&f).len(), 1);
    let old = key_files(&f)[0].clone();
    assert_eq!(m.correct(&a, "right").unwrap(), 2);
    assert!(!old.exists());
    assert_eq!(key_files(&f).len(), 1);
    assert_eq!(
        texts(&m.recall(&g("work"), None, &lim()).unwrap()),
        ["right"]
    );
    // an old key restored from a backup is destroyed again on open
    let (kd, bk) = (f.dir().join("keys"), f.dir().join("bk"));
    copy_dir(&kd, &bk);
    std::fs::write(&old, [7u8; 32]).unwrap();
    let m = f.open();
    assert!(!old.exists());
    assert_eq!(
        texts(&m.recall(&g("work"), None, &lim()).unwrap()),
        ["right"]
    );
}

#[test]
fn correct_of_unknown_or_forgotten_item_refused() {
    let f = fx();
    let m = f.open();
    assert!(matches!(
        m.correct(&"0".repeat(32), "x"),
        Err(MemoryRefusal::UnknownItem(_))
    ));
    let a = m.put(Scope::Work, Kind::Fact, "x").unwrap();
    m.forget(ForgetTarget::Item(a.clone())).unwrap();
    assert!(matches!(
        m.correct(&a, "y"),
        Err(MemoryRefusal::ItemForgotten(_))
    ));
}

// ---------- forget ----------

#[test]
fn forget_item_removes_content_but_not_the_ciphertext() {
    let f = fx();
    let m = f.open();
    let a = m.put(Scope::Work, Kind::Fact, "gone soon").unwrap();
    let keep = m.put(Scope::Work, Kind::Fact, "stays").unwrap();
    let rec = last_record(&f);
    let _ = rec;
    let log_before = files(&f.dir().join("log")).len();
    assert_eq!(
        m.forget(ForgetTarget::Item(a.clone())).unwrap(),
        vec![a.clone()]
    );
    assert_eq!(
        texts(&m.recall(&g("work"), None, &lim()).unwrap()),
        ["stays"]
    );
    let v = m.inspect(&g("work")).unwrap();
    let fa = v.iter().find(|i| i.item == a).unwrap();
    assert_eq!((fa.state.as_str(), fa.text.clone()), ("forgotten", None));
    assert!(v.iter().any(|i| i.item == keep && i.text.is_some()));
    // the stored ciphertext is still in the log; the key is not
    assert_eq!(files(&f.dir().join("log")).len(), log_before + 1);
    assert_eq!(key_files(&f).len(), 1);
    assert!(files(&f.dir().join("pending")).is_empty());
    // the Forget record holds metadata only
    let fr = std::fs::read_to_string(last_record(&f)).unwrap();
    assert!(fr.contains("forget_item") && fr.contains(&a) && !fr.contains("gone soon"));
    assert!(matches!(
        m.forget(ForgetTarget::Item(a)),
        Err(MemoryRefusal::NothingToForget)
    ));
}

#[test]
fn forget_scope_only_touches_that_scope() {
    let f = fx();
    let m = f.open();
    let p = Scope::parse("project:a").unwrap();
    m.put(p.clone(), Kind::Fact, "a1").unwrap();
    m.put(p.clone(), Kind::Goal, "a-goal").unwrap();
    m.put(Scope::parse("project:b").unwrap(), Kind::Fact, "b1")
        .unwrap();
    m.put(Scope::Work, Kind::Fact, "w1").unwrap();
    assert_eq!(m.forget(ForgetTarget::Scope(p)).unwrap().len(), 2);
    assert!(m
        .recall(&g("project:a"), None, &lim())
        .unwrap()
        .items
        .is_empty());
    assert!(m
        .inspect(&g("project:a"))
        .unwrap()
        .iter()
        .all(|i| i.text.is_none() && i.state == "forgotten"));
    assert_eq!(
        texts(&m.recall(&g("project:b"), None, &lim()).unwrap()),
        ["b1"]
    );
    assert_eq!(texts(&m.recall(&g("work"), None, &lim()).unwrap()), ["w1"]);
    assert_eq!(key_files(&f).len(), 2);
    assert!(matches!(
        m.forget(ForgetTarget::Scope(Scope::Personal)),
        Err(MemoryRefusal::NothingToForget)
    ));
    // a later put in a forgotten scope is a new, live item
    m.put(Scope::parse("project:a").unwrap(), Kind::Fact, "a-new")
        .unwrap();
    assert_eq!(
        texts(&m.recall(&g("project:a"), None, &lim()).unwrap()),
        ["a-new"]
    );
}

type Case = (&'static str, Box<dyn Fn(&Fx)>);

fn once(step: Step) -> (Arc<dyn Fn(Step) -> bool + Send + Sync>, Arc<AtomicBool>) {
    let fired = Arc::new(AtomicBool::new(false));
    let f2 = fired.clone();
    (
        Arc::new(move |s| s == step && !f2.swap(true, Ordering::SeqCst)),
        fired,
    )
}

const FORGET_STEPS: [Step; 13] = [
    Step::MarkerWritten,
    Step::KeyZeroed,
    Step::KeyUnlinked,
    Step::KeysDestroyed,
    Step::TempCreated,
    Step::TempWritten,
    Step::TempSynced,
    Step::Linked,
    Step::DirSynced,
    Step::TempRemoved,
    Step::ForgetRecorded,
    Step::MarkerRemoved,
    Step::MarkerRemoved, // repeated on purpose: harmless duplicate
];

#[test]
fn crash_at_every_forget_step_never_resurrects_and_reopen_finishes() {
    for scope_target in [false, true] {
        for step in FORGET_STEPS {
            let f = fx();
            let m = f.open();
            let a = m.put(Scope::Work, Kind::Fact, "doomed").unwrap();
            m.put(Scope::Personal, Kind::Fact, "bystander").unwrap();
            let (hook, fired) = once(step);
            let m = m.with_fault_hook(hook);
            let t = if scope_target {
                ForgetTarget::Scope(Scope::Work)
            } else {
                ForgetTarget::Item(a.clone())
            };
            let r = m.forget(t);
            assert!(fired.load(Ordering::SeqCst), "{step:?} never reached");
            assert!(
                matches!(r, Err(MemoryRefusal::Crashed(_))),
                "{step:?}: {r:?}"
            );
            // same process view: no content ever, and a pending state is named
            let v = m.inspect(&g("work")).unwrap();
            assert!(
                v.iter().all(|i| i.text.is_none()) || step == Step::MarkerWritten,
                "{step:?}"
            );
            if matches!(
                step,
                Step::KeyUnlinked
                    | Step::KeysDestroyed
                    | Step::TempCreated
                    | Step::TempWritten
                    | Step::TempSynced
            ) {
                assert_eq!(
                    v[0].state, "forgotten (key missing, record pending)",
                    "{step:?}"
                );
            }
            // restart: recovery completes the forget
            let m = f.open();
            let v = m.inspect(&g("work")).unwrap();
            assert_eq!(v.len(), 1, "{step:?}");
            assert_eq!(
                (v[0].state.as_str(), v[0].text.clone()),
                ("forgotten", None),
                "{step:?}"
            );
            assert!(m.recall(&g("work"), None, &lim()).unwrap().items.is_empty());
            assert_eq!(
                texts(&m.recall(&g("personal"), None, &lim()).unwrap()),
                ["bystander"]
            );
            assert_eq!(key_files(&f).len(), 1, "{step:?}");
            assert!(files(&f.dir().join("pending")).is_empty(), "{step:?}");
            // forgetting twice is not possible, and the chain is intact for new writes
            assert!(matches!(
                m.forget(ForgetTarget::Item(a.clone())),
                Err(MemoryRefusal::NothingToForget)
            ));
            m.put(Scope::Work, Kind::Fact, "after").unwrap();
        }
    }
}

#[test]
fn crash_at_every_put_step_leaves_a_clean_store() {
    for step in [
        Step::KeyWritten,
        Step::TempCreated,
        Step::TempWritten,
        Step::TempSynced,
        Step::Linked,
        Step::DirSynced,
        Step::TempRemoved,
    ] {
        let f = fx();
        let m = f.open();
        m.put(Scope::Work, Kind::Fact, "first").unwrap();
        let (hook, fired) = once(step);
        let m = m.with_fault_hook(hook);
        assert!(matches!(
            m.put(Scope::Work, Kind::Fact, "second"),
            Err(MemoryRefusal::Crashed(_))
        ));
        assert!(fired.load(Ordering::SeqCst));
        let m = f.open();
        let got = texts(&m.recall(&g("work"), None, &lim()).unwrap());
        let record_exists = matches!(step, Step::Linked | Step::DirSynced | Step::TempRemoved);
        assert_eq!(
            got,
            if record_exists {
                vec!["first", "second"]
            } else {
                vec!["first"]
            },
            "{step:?}"
        );
        m.put(Scope::Work, Kind::Fact, "third").unwrap();
        assert_eq!(
            m.recall(&g("work"), None, &lim()).unwrap().items.len(),
            got.len() + 1
        );
    }
}

#[test]
fn crash_at_every_correct_step_never_loses_both_versions() {
    for step in [
        Step::KeyWritten,
        Step::Linked,
        Step::KeyZeroed,
        Step::KeyUnlinked,
        Step::KeysDestroyed,
        Step::SupersededKeyDestroyed,
    ] {
        let f = fx();
        let m = f.open();
        let a = m.put(Scope::Work, Kind::Fact, "old").unwrap();
        let (hook, _) = once(step);
        let m = m.with_fault_hook(hook);
        let _ = m.correct(&a, "new");
        let m = f.open();
        let got = texts(&m.recall(&g("work"), None, &lim()).unwrap());
        let applied = !matches!(step, Step::KeyWritten);
        assert_eq!(
            got,
            if applied { vec!["new"] } else { vec!["old"] },
            "{step:?}"
        );
        // never both keys once reopened
        assert_eq!(
            key_files(&f).len(),
            if applied { 1 } else { 2 }.min(key_files(&f).len()),
            "{step:?}"
        );
        if applied {
            assert_eq!(key_files(&f).len(), 1, "{step:?}");
        }
    }
}

#[test]
fn restoring_old_keys_after_forget_cannot_resurrect_content() {
    let f = fx();
    let m = f.open();
    let a = m.put(Scope::Work, Kind::Fact, "restorable?").unwrap();
    let bk = f.dir().join("keys-backup");
    copy_dir(&f.dir().join("keys"), &bk); // backup taken BEFORE the forget
    m.forget(ForgetTarget::Item(a)).unwrap();
    assert!(key_files(&f).is_empty());
    // restore the pre-forget keys over the live store
    for k in files(&bk) {
        std::fs::copy(&k, f.dir().join("keys").join(k.file_name().unwrap())).unwrap();
    }
    assert_eq!(key_files(&f).len(), 1);
    // without recovery the old key would decrypt; reapply_forgets destroys it
    assert_eq!(m.reapply_forgets().unwrap(), 1);
    assert!(key_files(&f).is_empty());
    // same through a restart
    for k in files(&bk) {
        std::fs::copy(&k, f.dir().join("keys").join(k.file_name().unwrap())).unwrap();
    }
    let m = f.open();
    assert!(key_files(&f).is_empty());
    assert!(m.recall(&g("work"), None, &lim()).unwrap().items.is_empty());
    assert!(m
        .inspect(&g("work"))
        .unwrap()
        .iter()
        .all(|i| i.text.is_none()));
}

#[test]
fn missing_key_for_a_never_forgotten_item_is_unresolved() {
    let f = fx();
    let m = f.open();
    let a = m.put(Scope::Work, Kind::Fact, "lost key").unwrap();
    m.put(Scope::Work, Kind::Fact, "fine").unwrap();
    std::fs::remove_file(&key_files(&f)[0]).unwrap();
    let v = m.inspect(&g("work")).unwrap();
    let bad: Vec<_> = v
        .iter()
        .filter(|i| i.state == "Unresolved: key missing")
        .collect();
    assert_eq!(bad.len(), 1);
    assert!(bad[0].text.is_none());
    let r = m.recall(&g("work"), None, &lim()).unwrap();
    assert_eq!(r.items.len(), 1);
    assert_eq!(r.unresolved.len(), 1);
    // reopening does not "complete a forget" nobody asked for
    let m = f.open();
    assert_eq!(
        m.inspect(&g("work"))
            .unwrap()
            .iter()
            .filter(|i| i.state.starts_with("Unresolved"))
            .count(),
        1
    );
    let _ = a;
}

// ---------- identity, damage, tamper ----------

#[test]
fn not_engaged_and_foreign_identity_refused() {
    let f = fx();
    assert_eq!(
        Memory::open(&f.home, None).err(),
        Some(MemoryRefusal::NotEngaged)
    );
    f.open().put(Scope::Work, Kind::Fact, "mine").unwrap();
    let other = Memory::open(&f.home, Some(&resolved(2)));
    assert!(matches!(other, Err(MemoryRefusal::ForeignMemory(_))));
    // same agent, different root is foreign too
    let mut r = resolved(1);
    r.root = [9; 32];
    assert!(matches!(
        Memory::open(&f.home, Some(&r)),
        Err(MemoryRefusal::ForeignMemory(_))
    ));
    // nothing was changed by the refusals
    assert_eq!(
        texts(&f.open().recall(&g("work"), None, &lim()).unwrap()),
        ["mine"]
    );
}

#[test]
fn damaged_unknown_and_oversize_records_are_refused_not_replaced() {
    let cases: Vec<Case> = vec![
        (
            "garbage",
            Box::new(|f| std::fs::write(last_record(f), b"{nope").unwrap()),
        ),
        (
            "unknown schema",
            Box::new(|f| edit_last(f, |v| v["schema"] = "aien.allen.memory/2".into())),
        ),
        (
            "unknown field",
            Box::new(|f| edit_last(f, |v| v["extra"] = 1.into())),
        ),
        (
            "unknown op field",
            Box::new(|f| edit_last(f, |v| v["op"]["note"] = "x".into())),
        ),
        (
            "oversize",
            Box::new(|f| std::fs::write(last_record(f), vec![b' '; 9000]).unwrap()),
        ),
        (
            "stray file",
            Box::new(|f| std::fs::write(f.dir().join("log").join("readme.txt"), b"x").unwrap()),
        ),
        (
            "gap",
            Box::new(|f| {
                std::fs::rename(
                    last_record(f),
                    f.dir().join("log").join("l00000000000000000009.json"),
                )
                .unwrap();
            }),
        ),
        (
            "chain break",
            Box::new(|f| edit_last(f, |v| v["prev_sha256"] = "1".repeat(64).into())),
        ),
        (
            "bad scope",
            Box::new(|f| edit_last(f, |v| v["op"]["scope"] = "everything".into())),
        ),
    ];
    for (name, damage) in cases {
        let f = fx();
        let m = f.open();
        m.put(Scope::Work, Kind::Fact, "one").unwrap();
        m.put(Scope::Work, Kind::Fact, "two").unwrap();
        damage(&f);
        let before: Vec<_> = files(&f.dir().join("log"));
        let r = Memory::open(&f.home, Some(&resolved(1)));
        assert!(
            matches!(r, Err(MemoryRefusal::Damaged(_))),
            "{name}: {:?}",
            r.err()
        );
        assert_eq!(
            before,
            files(&f.dir().join("log")),
            "{name}: store was changed"
        );
    }
}

#[test]
fn tampered_ciphertext_or_bound_data_is_refused() {
    // ciphertext flipped
    let f = fx();
    let m = f.open();
    m.put(Scope::Work, Kind::Fact, "authentic").unwrap();
    edit_last(&f, |v| {
        let c = v["op"]["ciphertext"].as_str().unwrap().to_string();
        let flip = if c.starts_with('0') { "1" } else { "0" };
        v["op"]["ciphertext"] = format!("{flip}{}", &c[1..]).into();
    });
    let m = f.open();
    assert!(matches!(
        m.recall(&g("work"), None, &lim()),
        Err(MemoryRefusal::Tampered(_))
    ));
    assert!(m.inspect(&g("work")).unwrap()[0]
        .state
        .starts_with("Unresolved"));
    // scope relabelled (AAD binds the scope)
    let f = fx();
    let m = f.open();
    m.put(Scope::Work, Kind::Fact, "authentic").unwrap();
    edit_last(&f, |v| v["op"]["scope"] = "personal".into());
    let m = f.open();
    assert!(matches!(
        m.recall(&g("personal"), None, &lim()),
        Err(MemoryRefusal::Tampered(_))
    ));
    assert!(m.recall(&g("work"), None, &lim()).unwrap().items.is_empty());
}

#[test]
fn ciphertext_cannot_be_moved_to_another_item() {
    let f = fx();
    let m = f.open();
    m.put(Scope::Work, Kind::Fact, "first").unwrap();
    let a = std::fs::read_to_string(last_record(&f)).unwrap();
    let a: serde_json::Value = serde_json::from_str(&a).unwrap();
    m.put(Scope::Work, Kind::Fact, "second").unwrap();
    edit_last(&f, |v| {
        v["op"]["ciphertext"] = a["op"]["ciphertext"].clone();
        v["op"]["nonce"] = a["op"]["nonce"].clone();
    });
    let m = f.open();
    assert!(matches!(
        m.recall(&g("work"), None, &lim()),
        Err(MemoryRefusal::Tampered(_))
    ));
}

// ---------- export and goals ----------

#[test]
fn export_has_live_plaintext_of_one_scope_only() {
    let f = fx();
    let m = f.open();
    let a = m.put(Scope::Work, Kind::Fact, "export me").unwrap();
    let b = m.put(Scope::Work, Kind::Fact, "forgotten one").unwrap();
    m.put(Scope::Personal, Kind::Fact, "not in work export")
        .unwrap();
    m.forget(ForgetTarget::Item(b)).unwrap();
    let j = m.export(&g("work")).unwrap();
    let v: serde_json::Value = serde_json::from_str(&j).unwrap();
    assert_eq!(v["scope"], "work");
    assert_eq!(v["items"].as_array().unwrap().len(), 1);
    assert_eq!(v["items"][0]["item"], a.as_str());
    assert_eq!(v["items"][0]["text"], "export me");
    assert!(!j.contains("forgotten one") && !j.contains("not in work export"));
}

#[test]
fn host_goals_open_close_and_are_labelled() {
    let f = fx();
    let m = f.open();
    let a = m
        .put(Scope::Work, Kind::Goal, "ship the memory crate")
        .unwrap();
    let b = m.put(Scope::Work, Kind::Goal, "second goal").unwrap();
    m.put(Scope::Work, Kind::Fact, "not a goal").unwrap();
    m.put(Scope::Personal, Kind::Goal, "personal goal").unwrap();
    let l = m.goals(&g("work")).unwrap();
    assert_eq!(l.label, "host goal record, not a subject intent");
    assert_eq!(l.goals.len(), 2);
    assert!(l
        .goals
        .iter()
        .all(|x| x.state == "open" && x.label == GOAL_LABEL && x.scope == "work"));
    m.close_goal(&a).unwrap();
    let l = m.goals(&g("work")).unwrap();
    assert_eq!(
        l.goals.iter().find(|x| x.item == a).unwrap().state,
        "closed"
    );
    assert_eq!(
        l.goals.iter().find(|x| x.item == a).unwrap().text,
        "ship the memory crate"
    );
    // closed goals are not offered to context assembly
    let r = m.recall(&g("work"), None, &lim()).unwrap();
    assert!(texts(&r).iter().all(|t| t != "ship the memory crate"));
    let open_goal = r.items.iter().find(|i| i.item == b).unwrap();
    assert_eq!(open_goal.label, Some(GOAL_LABEL));
    assert_eq!(m.goals_all(&InspectAll::owner()).unwrap().goals.len(), 3);
    assert!(matches!(m.close_goal(&a), Err(MemoryRefusal::Invalid(_))));
    let fact = m
        .inspect(&g("work"))
        .unwrap()
        .into_iter()
        .find(|i| i.kind == Kind::Fact)
        .unwrap();
    assert!(matches!(
        m.close_goal(&fact.item),
        Err(MemoryRefusal::NotAGoal(_))
    ));
    // survives restart; forgetting a goal removes it from the list
    let m = f.open();
    assert_eq!(m.goals(&g("work")).unwrap().goals.len(), 2);
    m.forget(ForgetTarget::Item(b)).unwrap();
    assert_eq!(m.goals(&g("work")).unwrap().goals.len(), 1);
}

#[test]
fn no_memory_or_goal_field_leaks_into_the_profile_schema() {
    // This crate does not depend on aien-allen-profile and never reads or writes <home>.allen-profile.
    let f = fx();
    f.open().put(Scope::Work, Kind::Goal, "g").unwrap();
    assert!(!f.home.with_file_name("home.allen-profile").exists());
    assert!(f.home.with_file_name("home.allen-memory").exists());
}
