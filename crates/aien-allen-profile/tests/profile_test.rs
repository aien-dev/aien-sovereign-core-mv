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

//! Profile store tests. FIXTURE-ONLY: identities here are fixed byte arrays,
//! not subjects resolved from AIENOS. No model is involved.
use aien_allen_profile::{
    Changes, Identity, PersonaContext, PersonaState, PrefChange, ProfileRefusal as R, Scope, Store,
    Tone, Verbosity, WriteStep, CONTEXT_MAX_BYTES,
};
use std::sync::{Arc, Barrier};

fn id(tag: u8) -> Identity {
    Identity {
        agent: [tag; 32],
        root: [tag.wrapping_add(100); 32],
    }
}

fn name(n: &str) -> Changes {
    Changes {
        name: Some(n.into()),
        ..Default::default()
    }
}

fn pref(k: &str, v: &str) -> PrefChange {
    PrefChange {
        key: k.into(),
        value: v.into(),
        scope: Scope::All,
    }
}

fn dir(t: &tempfile::TempDir) -> std::path::PathBuf {
    t.path().join("compose.allen-profile")
}

#[test]
fn rename_tone_verbosity_and_pref_survive_reopen() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    assert!(s.head().unwrap().is_none());
    let ch = Changes {
        name: Some("  Nova ".into()),
        tone: Some(Tone::Warm),
        verbosity: Some(Verbosity::Brief),
        plain_language: Some(true),
        set_prefs: vec![pref("summary.style", "bullet points")],
        ..Default::default()
    };
    let p = s.set(0, &ch).unwrap();
    assert_eq!(p.revision, 1);
    drop(s);
    // "Restart": a new Store from the same folder.
    let s2 = Store::at(dir(&t), id(1));
    let h = s2.head().unwrap().unwrap();
    assert_eq!(h.persona.display_name, "Nova");
    assert_eq!(h.persona.tone, Tone::Warm);
    assert_eq!(h.persona.verbosity, Verbosity::Brief);
    assert!(h.persona.plain_language);
    assert_eq!(h.working_preferences[0].key, "summary.style");
    assert_eq!(h.working_preferences[0].provenance, "user_explicit");
}

#[test]
fn foreign_agent_and_wrong_root_are_refused() {
    let t = tempfile::tempdir().unwrap();
    Store::at(dir(&t), id(1)).set(0, &name("Nova")).unwrap();
    let other_agent = Store::at(dir(&t), id(2));
    assert!(matches!(other_agent.head(), Err(R::ForeignProfile(_))));
    assert!(matches!(
        other_agent.set(1, &name("X")),
        Err(R::ForeignProfile(_))
    ));
    // Same agent, different root.
    let wrong_root = Store::at(
        dir(&t),
        Identity {
            agent: id(1).agent,
            root: [9; 32],
        },
    );
    assert!(matches!(wrong_root.head(), Err(R::ForeignProfile(_))));
    // The original identity still reads its own profile, untouched.
    assert_eq!(
        Store::at(dir(&t), id(1)).head().unwrap().unwrap().revision,
        1
    );
}

#[test]
fn stale_expect_changes_nothing() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    s.set(0, &name("A")).unwrap();
    match s.set(0, &name("B")) {
        Err(R::StaleUpdate { expected, current }) => assert_eq!((expected, current), (0, 1)),
        o => panic!("{o:?}"),
    }
    assert!(matches!(s.set(5, &name("B")), Err(R::StaleUpdate { .. })));
    assert_eq!(s.head().unwrap().unwrap().persona.display_name, "A");
    assert_eq!(s.history().unwrap().len(), 1);
}

#[test]
fn concurrent_writers_exactly_one_wins() {
    let t = tempfile::tempdir().unwrap();
    let d = Arc::new(dir(&t));
    let barrier = Arc::new(Barrier::new(8));
    let results: Vec<_> = (0..8)
        .map(|i| {
            let d = d.clone();
            let barrier = barrier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                Store::at((*d).clone(), id(1)).set(0, &name(&format!("Name{i}")))
            })
        })
        .collect::<Vec<_>>()
        .into_iter()
        .map(|h| h.join().unwrap())
        .collect();
    let wins = results.iter().filter(|r| r.is_ok()).count();
    assert_eq!(wins, 1, "{results:?}");
    assert!(results
        .iter()
        .filter_map(|r| r.as_ref().err())
        .all(|e| matches!(e, R::StaleUpdate { .. })));
    let s = Store::at((*d).clone(), id(1));
    assert_eq!(s.history().unwrap().len(), 1);
    // No temp files linger after clean runs.
    let leftovers = std::fs::read_dir(&*d)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp-")
        })
        .count();
    assert_eq!(leftovers, 0);
}

fn first_rev(t: &tempfile::TempDir) -> std::path::PathBuf {
    dir(t).join(format!("r{:020}.json", 1))
}

fn good(t: &tempfile::TempDir) -> Vec<u8> {
    Store::at(dir(t), id(1)).set(0, &name("Nova")).unwrap();
    std::fs::read(first_rev(t)).unwrap()
}

#[test]
fn damaged_files_are_refused_and_never_replaced() {
    type Mutate = Box<dyn Fn(&str) -> Vec<u8>>;
    let cases: Vec<(&str, Mutate)> = vec![
        ("malformed json", Box::new(|_| b"{not json".to_vec())),
        (
            "unknown schema",
            Box::new(|g| {
                g.replace("aien.allen.profile/1", "aien.allen.profile/2")
                    .into_bytes()
            }),
        ),
        (
            "unknown field",
            Box::new(|g| g.replacen('{', "{\"avatar\":\"x\",", 1).into_bytes()),
        ),
        (
            "oversize",
            Box::new(|g| {
                let mut v = g.as_bytes().to_vec();
                v.extend(std::iter::repeat_n(b' ', 20_000));
                v
            }),
        ),
        (
            "bad tone",
            Box::new(|g| g.replace("\"neutral\"", "\"sarcastic\"").into_bytes()),
        ),
    ];
    for (label, mutate) in cases {
        let t = tempfile::tempdir().unwrap();
        let g = String::from_utf8(good(&t)).unwrap();
        let bad = mutate(&g);
        std::fs::write(first_rev(&t), &bad).unwrap();
        let s = Store::at(dir(&t), id(1));
        assert!(matches!(s.head(), Err(R::Damaged(_))), "{label}");
        assert!(
            matches!(s.set(1, &name("X")), Err(R::Damaged(_))),
            "{label}"
        );
        assert!(matches!(s.reset(1), Err(R::Damaged(_))), "{label}");
        assert_eq!(
            std::fs::read(first_rev(&t)).unwrap(),
            bad,
            "{label}: untouched"
        );
        // The context falls back to defaults and says why.
        let c = PersonaContext::from_store(&s);
        assert_eq!(c.state, PersonaState::Refused, "{label}");
        assert_eq!(c.persona.display_name, "ALLEN");
        assert!(c.reason.is_some());
    }
}

#[test]
fn broken_chain_gap_and_foreign_file_are_refused() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    s.set(0, &name("A")).unwrap();
    s.set(1, &name("B")).unwrap();
    // Editing revision 1 in place breaks revision 2's chain link.
    let p1 = first_rev(&t);
    let edited = String::from_utf8(std::fs::read(&p1).unwrap())
        .unwrap()
        .replace("\"A\"", "\"Z\"");
    std::fs::write(&p1, edited).unwrap();
    assert!(matches!(s.head(), Err(R::Damaged(_))));
    // A gap.
    let t2 = tempfile::tempdir().unwrap();
    let s2 = Store::at(dir(&t2), id(1));
    s2.set(0, &name("A")).unwrap();
    s2.set(1, &name("B")).unwrap();
    std::fs::remove_file(first_rev(&t2)).unwrap();
    assert!(matches!(s2.head(), Err(R::Damaged(_))));
    // A foreign file.
    let t3 = tempfile::tempdir().unwrap();
    let s3 = Store::at(dir(&t3), id(1));
    s3.set(0, &name("A")).unwrap();
    std::fs::write(dir(&t3).join("notes.txt"), "hi").unwrap();
    assert!(matches!(s3.head(), Err(R::Damaged(_))));
}

#[test]
fn permission_like_keys_are_rejected() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    for k in [
        "grant_write",
        "permission.fs",
        "auto_approve",
        "capability-x",
        "allow_all",
        "authorized_paths",
        "skip_approval",
        "auto-approval",
        "bypass.checks",
        "sudo",
    ] {
        let ch = Changes {
            set_prefs: vec![pref(k, "yes")],
            ..Default::default()
        };
        assert!(matches!(s.set(0, &ch), Err(R::PermissionKey(_))), "{k}");
    }
    assert!(s.head().unwrap().is_none());
    // Other invalid input.
    for ch in [
        Changes {
            set_prefs: vec![pref("Bad Key", "v")],
            ..Default::default()
        },
        Changes {
            set_prefs: vec![pref("ok", "line\nbreak")],
            ..Default::default()
        },
        Changes {
            set_prefs: vec![pref("ok", &"x".repeat(201))],
            ..Default::default()
        },
        name(""),
        name(&"n".repeat(65)),
        name("bad\u{7}name"),
        Changes {
            unset_prefs: vec!["missing".into()],
            ..Default::default()
        },
    ] {
        assert!(matches!(s.set(0, &ch), Err(R::Invalid(_))), "{ch:?}");
    }
    let many = Changes {
        set_prefs: (0..33).map(|i| pref(&format!("k{i:02}"), "v")).collect(),
        ..Default::default()
    };
    assert!(matches!(s.set(0, &many), Err(R::Invalid(_))));
    assert!(matches!(
        s.set(0, &Changes::default()),
        Err(R::NothingToChange)
    ));
}

#[test]
fn crash_at_every_write_step_leaves_an_intact_head() {
    for step in WriteStep::ALL {
        let t = tempfile::tempdir().unwrap();
        // Revision 1 written cleanly; the crash hits revision 2.
        let clean = Store::at(dir(&t), id(1));
        clean.set(0, &name("Old")).unwrap();
        let crashing = Store::at(dir(&t), id(1)).with_fault_hook(Arc::new(move |s| s == step));
        assert!(
            matches!(crashing.set(1, &name("New")), Err(R::Crashed(_))),
            "{step:?}"
        );
        let head = Store::at(dir(&t), id(1)).head().unwrap().unwrap();
        let landed = matches!(
            step,
            WriteStep::Linked | WriteStep::DirSynced | WriteStep::TempRemoved
        );
        let (rev, nm) = if landed { (2, "New") } else { (1, "Old") };
        assert_eq!(
            (head.revision, head.persona.display_name.as_str()),
            (rev, nm),
            "{step:?}"
        );
        // Recovery: the next honest write works from whatever the head is.
        let again = Store::at(dir(&t), id(1));
        again.set(rev, &name("After")).unwrap();
        assert_eq!(again.revision().unwrap(), rev + 1);
    }
}

#[test]
fn crash_on_the_very_first_write_leaves_no_profile_or_a_whole_one() {
    for step in WriteStep::ALL {
        let t = tempfile::tempdir().unwrap();
        let s = Store::at(dir(&t), id(1)).with_fault_hook(Arc::new(move |s| s == step));
        assert!(s.set(0, &name("Nova")).is_err());
        let h = Store::at(dir(&t), id(1)).head().unwrap();
        match step {
            WriteStep::Linked | WriteStep::DirSynced | WriteStep::TempRemoved => {
                assert_eq!(h.unwrap().persona.display_name, "Nova")
            }
            _ => assert!(h.is_none(), "{step:?}"),
        }
    }
}

#[test]
fn history_revert_and_reset_write_new_revisions() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    s.set(0, &name("Nova")).unwrap();
    s.set(
        1,
        &Changes {
            tone: Some(Tone::Direct),
            note: Some("make it terse".into()),
            ..Default::default()
        },
    )
    .unwrap();
    let r3 = s.revert(2, 1).unwrap();
    assert_eq!(r3.revision, 3);
    assert_eq!(r3.persona.tone, Tone::Neutral);
    assert_eq!(r3.persona.display_name, "Nova");
    assert_eq!(s.get(2).unwrap().persona.tone, Tone::Direct, "history kept");
    assert!(matches!(s.revert(3, 9), Err(R::UnknownRevision(9))));
    assert!(matches!(s.revert(3, 3), Err(R::NothingToChange)));
    assert!(matches!(s.revert(2, 1), Err(R::StaleUpdate { .. })));
    let r4 = s.reset(3).unwrap();
    assert_eq!(r4.persona.display_name, "ALLEN");
    assert!(matches!(s.reset(4), Err(R::NothingToChange)));
    let h = s.history().unwrap();
    assert_eq!(h.len(), 4);
    assert_eq!(h[1].note, "make it terse");
    assert_eq!(h[3].author.clone(), aien_allen_profile::Author::Reset);
    // Reset on an empty store.
    let t2 = tempfile::tempdir().unwrap();
    assert!(matches!(
        Store::at(dir(&t2), id(1)).reset(0),
        Err(R::NoProfile)
    ));
}

#[test]
fn context_is_deterministic_and_bounded() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    // Default context, no profile.
    let d = PersonaContext::from_store(&s);
    assert_eq!(d.state, PersonaState::Default);
    assert!(d.render().contains("name: \"ALLEN\""));
    // Insert in scrambled order; output is sorted by key and identical on every call.
    let ch = Changes {
        name: Some("Nova".into()),
        set_prefs: vec![pref("zeta", "z"), pref("alpha", "a"), pref("mid", "m")],
        ..Default::default()
    };
    s.set(0, &ch).unwrap();
    let a = PersonaContext::from_store(&s).render();
    let b = PersonaContext::from_store(&Store::at(dir(&t), id(1))).render();
    assert_eq!(a, b);
    let (ia, im, iz) = (
        a.find("alpha").unwrap(),
        a.find("mid =").unwrap(),
        a.find("zeta").unwrap(),
    );
    assert!(ia < im && im < iz);
    assert!(a.contains("grant no permission"));
    // Worst case: 32 preferences, each 200 multi-byte characters, 64-char name.
    let t2 = tempfile::tempdir().unwrap();
    let s2 = Store::at(dir(&t2), id(1));
    let big = Changes {
        name: Some("N".repeat(64)),
        set_prefs: (0..32)
            .map(|i| pref(&format!("pref.{i:02}"), &"é".repeat(150)))
            .collect(),
        ..Default::default()
    };
    s2.set(0, &big).unwrap();
    let c = PersonaContext::from_store(&s2);
    let r = c.render();
    assert!(r.len() <= CONTEXT_MAX_BYTES, "{} bytes", r.len());
    assert!(c.dropped > 0 && c.shown.len() + c.dropped == 32);
    assert!(r.contains(&format!("({} more preferences not shown)", c.dropped)));
    assert_eq!(r, PersonaContext::from_store(&s2).render());
}

#[test]
fn control_and_bidi_characters_are_refused() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    for c in [
        '\u{2028}', '\u{2029}', '\u{202A}', '\u{202B}', '\u{202C}', '\u{202D}', '\u{202E}',
        '\u{2066}', '\u{2067}', '\u{2068}', '\u{2069}', '\n', '\u{0}',
    ] {
        assert!(
            matches!(s.set(0, &name(&format!("Ab{c}cd"))), Err(R::Invalid(_))),
            "name {c:?}"
        );
        let ch = Changes {
            set_prefs: vec![pref("reply_language", &format!("en{c}x"))],
            ..Default::default()
        };
        assert!(matches!(s.set(0, &ch), Err(R::Invalid(_))), "value {c:?}");
    }
    assert!(s.head().unwrap().is_none());
}

#[test]
fn normal_key_is_accepted() {
    let t = tempfile::tempdir().unwrap();
    let s = Store::at(dir(&t), id(1));
    let ch = Changes {
        set_prefs: vec![pref("reply_language", "english")],
        ..Default::default()
    };
    assert!(s.set(0, &ch).is_ok());
}

/// Both writers pass the stale check; the first finishes its whole write
/// between the second's temp-file sync and its hard link, so the second hits
/// the AlreadyExists path and must lose cleanly.
#[test]
fn link_race_loser_gets_stale_update_and_winner_is_intact() {
    let t = tempfile::tempdir().unwrap();
    let d = dir(&t);
    let winner_dir = d.clone();
    let loser = Store::at(d.clone(), id(1)).with_fault_hook(Arc::new(move |step| {
        if step == WriteStep::TempSynced {
            Store::at(winner_dir.clone(), id(1))
                .set(0, &name("Winner"))
                .unwrap();
        }
        false
    }));
    match loser.set(0, &name("Loser")) {
        Err(R::StaleUpdate { expected, current }) => assert_eq!((expected, current), (0, 1)),
        o => panic!("{o:?}"),
    }
    let s = Store::at(d.clone(), id(1));
    let head = s.head().unwrap().unwrap();
    assert_eq!(
        (head.revision, head.persona.display_name.as_str()),
        (1, "Winner")
    );
    assert_eq!(s.history().unwrap().len(), 1);
    let leftovers = std::fs::read_dir(&d)
        .unwrap()
        .filter(|e| {
            e.as_ref()
                .unwrap()
                .file_name()
                .to_string_lossy()
                .starts_with(".tmp-")
        })
        .count();
    assert_eq!(leftovers, 0);
}
