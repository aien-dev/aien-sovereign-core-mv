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

//! Restart, corruption, rollback, machine change, negative control.
mod support;
use aien_allen::binding::{self, pin_path};
use aien_allen::hex;
use aien_allen::resolve::{resolve, Context, Refusal, SubjectSource};
use aien_allen::subject_v0::object_id;
use support::*;

const M1: [u8; 32] = [1; 32];
const M2: [u8; 32] = [2; 32];

fn ctx(f: &Fx, m: [u8; 32]) -> Context {
    Context {
        machine_id: m,
        lineage: Some(f.cortex),
    }
}

struct Rig {
    _t: tempfile::TempDir,
    home: std::path::PathBuf,
    subj: std::path::PathBuf,
    f: Fx,
}

fn rig_head(n: u64) -> Rig {
    let t = tempfile::tempdir().unwrap();
    let f = fx("a");
    let c = chain(&f, n);
    let subj = write_head(&t.path().join("subject.bin"), c.last().unwrap());
    Rig {
        home: t.path().join("compose"),
        subj,
        f,
        _t: t,
    }
}

fn adopt(r: &Rig) -> aien_allen::Resolved {
    resolve(
        &SubjectSource::Head(r.subj.clone()),
        &pin_path(&r.home),
        &ctx(&r.f, M1),
        Some(&hex(&r.f.agent)),
    )
    .unwrap()
}

fn again(r: &Rig, m: [u8; 32]) -> Result<aien_allen::Resolved, Refusal> {
    resolve(
        &SubjectSource::Head(r.subj.clone()),
        &pin_path(&r.home),
        &ctx(&r.f, m),
        None,
    )
}

#[test]
fn restart_preserves_identity() {
    let r = rig_head(2);
    let first = adopt(&r);
    assert!(first.adopted);
    let pin_bytes = std::fs::read(pin_path(&r.home)).unwrap();
    // "Restart": nothing in memory, everything from files.
    let second = again(&r, M1).unwrap();
    assert!(!second.adopted);
    assert_eq!(first.agent, second.agent);
    assert_eq!(first.head_id, second.head_id);
    assert_eq!(first.root, second.root);
    assert_eq!(pin_bytes, std::fs::read(pin_path(&r.home)).unwrap());
}

#[test]
fn negative_control_nothing_is_ever_minted() {
    let r = rig_head(1);
    // No subject file at all, even with a correct adopt value.
    let missing = r._t.path().join("nope.bin");
    let e = resolve(
        &SubjectSource::Head(missing),
        &pin_path(&r.home),
        &ctx(&r.f, M1),
        Some(&hex(&r.f.agent)),
    )
    .unwrap_err();
    assert!(matches!(e, Refusal::Absent(_)), "{e}");
    assert!(!pin_path(&r.home).exists(), "no pin written");
    // An empty subject file.
    std::fs::write(&r.subj, b"").unwrap();
    assert!(matches!(again(&r, M1), Err(Refusal::Absent(_))));
    // An empty directory.
    let d = r._t.path().join("d");
    std::fs::create_dir(&d).unwrap();
    let e = resolve(
        &SubjectSource::Dir(d.clone()),
        &pin_path(&r.home),
        &ctx(&r.f, M1),
        None,
    )
    .unwrap_err();
    assert!(matches!(e, Refusal::Absent(_)));
    assert_eq!(
        std::fs::read_dir(&d).unwrap().count(),
        0,
        "directory untouched"
    );
    assert!(!pin_path(&r.home).exists());
}

#[test]
fn no_pin_means_refusal_not_adoption() {
    let r = rig_head(1);
    assert_eq!(again(&r, M1).unwrap_err(), Refusal::PinAbsent);
    assert!(!pin_path(&r.home).exists());
}

#[test]
fn adopt_rules() {
    let r = rig_head(1);
    let src = SubjectSource::Head(r.subj.clone());
    let p = pin_path(&r.home);
    let c = ctx(&r.f, M1);
    // wrong agent value
    assert_eq!(
        resolve(&src, &p, &c, Some(&hex(&h32("other")))).unwrap_err(),
        Refusal::AdoptMismatch
    );
    assert_eq!(
        resolve(&src, &p, &c, Some("zz")).unwrap_err(),
        Refusal::AdoptBadValue
    );
    assert!(!p.exists());
    adopt(&r);
    // adopt variable refused once a pin exists
    assert_eq!(
        resolve(&src, &p, &c, Some(&hex(&r.f.agent))).unwrap_err(),
        Refusal::AdoptRefusedPinExists
    );
}

#[test]
fn corrupted_pin_is_refused_and_untouched() {
    let r = rig_head(1);
    adopt(&r);
    let p = pin_path(&r.home);
    let good = std::fs::read(&p).unwrap();
    for i in [0usize, 9, 20, 100, 115, 160, 230] {
        let mut bad = good.clone();
        bad[i] ^= 1;
        std::fs::write(&p, &bad).unwrap();
        assert!(
            matches!(again(&r, M1), Err(Refusal::PinDamaged(_))),
            "byte {i}"
        );
        assert_eq!(std::fs::read(&p).unwrap(), bad, "pin not repaired");
    }
    // truncated and extended
    std::fs::write(&p, &good[..100]).unwrap();
    assert!(matches!(again(&r, M1), Err(Refusal::PinDamaged(_))));
    let mut long = good.clone();
    long.push(0);
    std::fs::write(&p, &long).unwrap();
    assert!(matches!(again(&r, M1), Err(Refusal::PinDamaged(_))));
    assert!(binding::read(&p).is_err());
}

#[test]
fn corrupted_or_truncated_subject_is_refused() {
    let r = rig_head(1);
    adopt(&r);
    let good = std::fs::read(&r.subj).unwrap();
    for i in [0usize, 9, 20, 60, 130, 200, good.len() - 1] {
        let mut bad = good.clone();
        bad[i] ^= 1;
        std::fs::write(&r.subj, &bad).unwrap();
        assert!(again(&r, M1).is_err(), "flip at {i} must be refused");
    }
    std::fs::write(&r.subj, &good[..good.len() - 8]).unwrap();
    assert!(matches!(again(&r, M1), Err(Refusal::Corrupt(_))));
    let mut long = good.clone();
    long.push(0);
    std::fs::write(&r.subj, &long).unwrap();
    assert!(matches!(again(&r, M1), Err(Refusal::Corrupt(_))));
}

#[test]
fn rolled_back_head_is_refused_and_same_sequence_other_object_too() {
    let r = rig_head(3);
    adopt(&r);
    let c = chain(&r.f, 3);
    std::fs::write(&r.subj, &c[1]).unwrap(); // sequence 2 < pinned 3
    assert_eq!(
        again(&r, M1).unwrap_err(),
        Refusal::RolledBack {
            pinned: 3,
            found: 2
        }
    );
    // Same sequence, different bytes (different payload).
    let other = encode(&r.f, 3, object_id(&c[1]), [7, 2000]);
    std::fs::write(&r.subj, &other).unwrap();
    assert_eq!(again(&r, M1).unwrap_err(), Refusal::SameSequenceDifferentId);
}

#[test]
fn legitimate_successor_advances_the_pin() {
    let r = rig_head(2);
    adopt(&r);
    let c = chain(&r.f, 3);
    std::fs::write(&r.subj, &c[2]).unwrap();
    let res = again(&r, M1).unwrap();
    assert_eq!(res.head_seq, 3);
    assert_eq!(
        binding::read(&pin_path(&r.home)).unwrap().unwrap().head_seq,
        3
    );
}

#[test]
fn machine_change_is_refused_never_rebound() {
    let r = rig_head(1);
    adopt(&r);
    let before = std::fs::read(pin_path(&r.home)).unwrap();
    assert_eq!(again(&r, M2).unwrap_err(), Refusal::MachineChanged);
    assert_eq!(
        before,
        std::fs::read(pin_path(&r.home)).unwrap(),
        "pin not rebound"
    );
    // identity (the subject file) is untouched by the machine id
    assert!(again(&r, M1).is_ok());
}

#[test]
fn lineage_rules() {
    let r = rig_head(1);
    let src = SubjectSource::Head(r.subj.clone());
    let a = Some(hex(&r.f.agent));
    let mut c = ctx(&r.f, M1);
    c.lineage = Some(h32("another journal"));
    assert_eq!(
        resolve(&src, &pin_path(&r.home), &c, a.as_deref()).unwrap_err(),
        Refusal::LineageMismatch
    );
    c.lineage = None;
    assert_eq!(
        resolve(&src, &pin_path(&r.home), &c, a.as_deref()).unwrap_err(),
        Refusal::LineageMissing
    );
    assert!(!pin_path(&r.home).exists());
}

#[test]
fn foreign_subject_against_pin_is_refused() {
    let r = rig_head(1);
    adopt(&r);
    let g = fx("b");
    let o = chain(&g, 1);
    std::fs::write(&r.subj, &o[0]).unwrap();
    let c = Context {
        machine_id: M1,
        lineage: Some(g.cortex),
    };
    let e = resolve(
        &SubjectSource::Head(r.subj.clone()),
        &pin_path(&r.home),
        &c,
        None,
    )
    .unwrap_err();
    assert!(matches!(e, Refusal::ForeignIdentity(_)), "{e}");
}

#[test]
fn directory_mode_chain_rows() {
    let t = tempfile::tempdir().unwrap();
    let f = fx("a");
    let c = chain(&f, 3);
    let home = t.path().join("compose");
    let d = t.path().join("subj");
    write_dir(&d, &c);
    let src = SubjectSource::Dir(d.clone());
    let cx = ctx(&f, M1);
    let ok = resolve(&src, &pin_path(&home), &cx, Some(&hex(&f.agent))).unwrap();
    assert!(ok.chain_verified);
    assert_eq!(ok.head_seq, 3);
    // forked: a second object at sequence 3
    let fork = encode(&f, 3, object_id(&c[1]), [7, 9]);
    write_dir(&d, std::slice::from_ref(&fork));
    assert_eq!(
        resolve(&src, &pin_path(&home), &cx, None).unwrap_err(),
        Refusal::Forked(3)
    );
    std::fs::remove_file(d.join(format!("{}.bin", hex(&object_id(&fork))))).unwrap();
    // gap: remove sequence 2
    std::fs::remove_file(d.join(format!("{}.bin", hex(&object_id(&c[1]))))).unwrap();
    assert_eq!(
        resolve(&src, &pin_path(&home), &cx, None).unwrap_err(),
        Refusal::Gap(2)
    );
    write_dir(&d, std::slice::from_ref(&c[1]));
    // changed bytes under an id name
    let name = d.join(format!("{}.bin", hex(&object_id(&c[0]))));
    let mut bad = c[0].clone();
    bad[100] ^= 1;
    std::fs::write(&name, &bad).unwrap();
    assert!(matches!(
        resolve(&src, &pin_path(&home), &cx, None),
        Err(Refusal::Corrupt(_))
    ));
    std::fs::write(&name, &c[0]).unwrap();
    // link broken: sequence 2 naming a wrong previous
    let wrong = encode(&f, 2, h32("nonsense"), [7, 1000]);
    let d2 = t.path().join("subj2");
    write_dir(&d2, &[c[0].clone(), wrong, c[2].clone()]);
    assert!(matches!(
        resolve(&SubjectSource::Dir(d2), &pin_path(&home), &cx, None),
        Err(Refusal::Corrupt(_))
    ));
    // foreign object mixed into the chain
    let g = fx("b");
    let alien = encode(&g, 2, object_id(&c[0]), [7, 1000]);
    let d3 = t.path().join("subj3");
    write_dir(&d3, &[c[0].clone(), alien]);
    assert!(matches!(
        resolve(&SubjectSource::Dir(d3), &pin_path(&home), &cx, None),
        Err(Refusal::ForeignIdentity(_))
    ));
}

#[test]
fn directory_mode_rejects_a_chain_that_forks_from_the_pin() {
    let t = tempfile::tempdir().unwrap();
    let f = fx("a");
    let c = chain(&f, 2);
    let home = t.path().join("compose");
    let d = t.path().join("subj");
    write_dir(&d, &c);
    let cx = ctx(&f, M1);
    resolve(
        &SubjectSource::Dir(d.clone()),
        &pin_path(&home),
        &cx,
        Some(&hex(&f.agent)),
    )
    .unwrap();
    // A different history with the same agent: seq 2 differs, seq 3 added.
    let alt2 = encode(&f, 2, object_id(&c[0]), [7, 5555]);
    let alt3 = encode(&f, 3, object_id(&alt2), [7, 1000]);
    let d2 = t.path().join("alt");
    write_dir(&d2, &[c[0].clone(), alt2, alt3]);
    assert_eq!(
        resolve(&SubjectSource::Dir(d2), &pin_path(&home), &cx, None).unwrap_err(),
        Refusal::ForkedFromPin
    );
}

#[test]
fn known_object_ids_are_stable() {
    // Same bytes, same id, on any machine, under any model (ADR 0018 s2.2).
    let f = fx("a");
    let o = encode(&f, 1, [0; 32], [7, 1000]);
    assert_eq!(object_id(&o), object_id(&o.clone()));
    let mut o2 = o.clone();
    o2[100] ^= 1;
    assert_ne!(object_id(&o), object_id(&o2));
}
