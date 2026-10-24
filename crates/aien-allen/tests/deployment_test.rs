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

//! Model swap / rollback / KV discard / deployment record tamper rows.
//!
//! PLACEHOLDER rows use synthetic digests and do NOT establish real
//! model-swap continuity. REAL_DIGESTS rows hash two real, checked-in weight
//! files (crates/aien-inference-abi/fixtures) with the host hasher; no model
//! is run, so they prove the record binds real artifact bytes, not behaviour.
mod support;
use aien_allen::binding::{pin_path, Pin};
use aien_allen::deployment::{
    append, check_artifact, deployments_path, sha256_file_hex, verify, DeployRefusal, Input,
};
use aien_allen::hex;
use aien_allen::resolve::{resolve, Context, SubjectSource};
use std::path::PathBuf;
use support::*;

const M1: [u8; 32] = [1; 32];

fn input(model: &str, placeholder: bool) -> Input {
    Input {
        model_sha256: model.to_string(),
        config_sha256: hex(&h32("config")),
        candidate_id: "cand-test".into(),
        executable_sha256: hex(&h32("exe")),
        revision: "rev-test".into(),
        placeholder,
    }
}

struct Rig {
    _t: tempfile::TempDir,
    home: PathBuf,
    subj: PathBuf,
    kv: PathBuf,
    f: Fx,
}

fn rig() -> Rig {
    let t = tempfile::tempdir().unwrap();
    let f = fx("a");
    let c = chain(&f, 2);
    let subj = write_head(&t.path().join("subject.bin"), c.last().unwrap());
    let kv = t.path().join("kv-standin");
    std::fs::create_dir_all(&kv).unwrap();
    std::fs::write(kv.join("block0"), b"reconstructible kv bytes").unwrap();
    let r = Rig {
        home: t.path().join("compose"),
        subj,
        kv,
        f,
        _t: t,
    };
    resolve(
        &SubjectSource::Head(r.subj.clone()),
        &pin_path(&r.home),
        &Context {
            machine_id: M1,
            lineage: Some(r.f.cortex),
        },
        Some(&hex(&r.f.agent)),
    )
    .unwrap();
    r
}

fn state(r: &Rig) -> (Vec<u8>, Vec<u8>) {
    (
        std::fs::read(&r.subj).unwrap(),
        std::fs::read(pin_path(&r.home)).unwrap(),
    )
}

#[test]
fn placeholder_model_swap_and_rollback_leave_identity_untouched() {
    let r = rig();
    let before = state(&r);
    let dep = deployments_path(&r.home);
    let (a, b) = (hex(&h32("PLACEHOLDER-A")), hex(&h32("PLACEHOLDER-B")));
    append(&dep, &r.f.agent, input(&a, true)).unwrap();
    append(&dep, &r.f.agent, input(&b, true)).unwrap();
    append(&dep, &r.f.agent, input(&a, true)).unwrap(); // rollback to A
                                                        // Subject object and pin: byte-identical (empty diff).
    assert_eq!(before, state(&r));
    let h = verify(&dep, &r.f.agent).unwrap();
    assert_eq!(h.len(), 3, "history retained, never rewound");
    assert_eq!(h[0].model_sha256, h[2].model_sha256);
    assert_ne!(h[0].model_sha256, h[1].model_sha256);
    assert!(h.iter().all(|e| e.placeholder));
    // The pin has no model field: it is the same bytes, so it cannot name one.
    let p = std::fs::read(pin_path(&r.home)).unwrap();
    assert!(!p.windows(32).any(|w| w == h32("PLACEHOLDER-A")));
    // Identity still resolves to the same logical agent.
    let res = resolve(
        &SubjectSource::Head(r.subj.clone()),
        &pin_path(&r.home),
        &Context {
            machine_id: M1,
            lineage: Some(r.f.cortex),
        },
        None,
    )
    .unwrap();
    assert_eq!(res.agent, r.f.agent);
}

#[test]
fn real_digests_of_two_checked_in_weight_files() {
    let r = rig();
    let fx_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../aien-inference-abi/fixtures");
    let wa = fx_dir.join("openwaldo-byte/model.safetensors");
    let wb = fx_dir.join("llama32_1b_oracle.safetensors");
    let (da, db) = (sha256_file_hex(&wa).unwrap(), sha256_file_hex(&wb).unwrap());
    assert_ne!(da, db);
    let before = state(&r);
    let dep = deployments_path(&r.home);
    append(&dep, &r.f.agent, input(&da, false)).unwrap();
    append(&dep, &r.f.agent, input(&db, false)).unwrap();
    let h = verify(&dep, &r.f.agent).unwrap();
    check_artifact(&h[0], &wa).unwrap();
    check_artifact(&h[1], &wb).unwrap();
    // Cross-check: model A's record does not match model B's bytes.
    assert!(matches!(
        check_artifact(&h[0], &wb),
        Err(DeployRefusal::ArtifactChanged { .. })
    ));
    assert_eq!(
        before,
        state(&r),
        "subject and pin unchanged across the swap"
    );
}

#[test]
fn changed_artifact_bytes_are_refused() {
    let r = rig();
    let art = r._t.path().join("model.bin");
    std::fs::write(&art, b"weights v1").unwrap();
    let dep = deployments_path(&r.home);
    append(
        &dep,
        &r.f.agent,
        input(&sha256_file_hex(&art).unwrap(), true),
    )
    .unwrap();
    let h = verify(&dep, &r.f.agent).unwrap();
    check_artifact(&h[0], &art).unwrap();
    std::fs::write(&art, b"weights v2").unwrap();
    assert!(matches!(
        check_artifact(&h[0], &art),
        Err(DeployRefusal::ArtifactChanged { seq: 1, .. })
    ));
}

#[test]
fn kv_discard_does_not_change_identity() {
    let r = rig();
    let before = state(&r);
    std::fs::remove_dir_all(&r.kv).unwrap();
    assert_eq!(before, state(&r));
    let res = resolve(
        &SubjectSource::Head(r.subj.clone()),
        &pin_path(&r.home),
        &Context {
            machine_id: M1,
            lineage: Some(r.f.cortex),
        },
        None,
    )
    .unwrap();
    assert_eq!(res.agent, r.f.agent);
    // The pin type has no field in which KV could be named.
    let p = Pin::decode(&before.1).unwrap();
    assert_eq!(p.agent, r.f.agent);
}

#[test]
fn tampered_deployment_records_are_refused() {
    let r = rig();
    let dep = deployments_path(&r.home);
    for n in 0..3 {
        append(&dep, &r.f.agent, input(&hex(&h32(&format!("m{n}"))), true)).unwrap();
    }
    let good = std::fs::read_to_string(&dep).unwrap();
    let lines: Vec<&str> = good.lines().collect();
    // Edited content (hash no longer matches).
    std::fs::write(&dep, good.replace("rev-test", "rev-evil")).unwrap();
    assert!(matches!(
        verify(&dep, &r.f.agent),
        Err(DeployRefusal::ChainBroken(1))
    ));
    // Deleted middle line.
    std::fs::write(&dep, format!("{}\n{}\n", lines[0], lines[2])).unwrap();
    assert!(matches!(
        verify(&dep, &r.f.agent),
        Err(DeployRefusal::ChainBroken(2))
    ));
    // Reordered lines.
    std::fs::write(&dep, format!("{}\n{}\n{}\n", lines[1], lines[0], lines[2])).unwrap();
    assert!(verify(&dep, &r.f.agent).is_err());
    // Truncated final line.
    std::fs::write(&dep, &good[..good.len() - 5]).unwrap();
    assert!(verify(&dep, &r.f.agent).is_err());
    // Appending onto a damaged record is refused, not repaired.
    assert!(append(&dep, &r.f.agent, input("x", true)).is_err());
    // Forged identity: a record for another agent is refused for this one.
    std::fs::write(&dep, &good).unwrap();
    let other = h32("other agent");
    assert!(matches!(
        verify(&dep, &other),
        Err(DeployRefusal::ForgedIdentity(1))
    ));
    // Restored original verifies again.
    assert_eq!(verify(&dep, &r.f.agent).unwrap().len(), 3);
}

#[test]
fn model_supplied_text_cannot_become_identity() {
    // The record's agent field is the host-resolved agent; an attempt to pass
    // a model-claimed identity has no API: append takes the agent from the
    // resolved subject. A hand-forged line with another agent is refused above.
    let r = rig();
    let dep = deployments_path(&r.home);
    let e = append(&dep, &r.f.agent, input(&hex(&h32("m")), true)).unwrap();
    assert_eq!(e.agent, hex(&r.f.agent));
}
