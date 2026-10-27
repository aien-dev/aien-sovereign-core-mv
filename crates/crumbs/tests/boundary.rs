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

//! BREADCRUMB-1 and security gates: CRUMB_V1_FORMAT_PASS, SEALED_BOUNDARY_PASS,
//! CRUMB_DETERMINISM_PASS (golden), and the leak tests over serialization,
//! Debug/Display, errors, protocol transcripts, traces and training samples.

use crumbs::audit::{find_labels, find_needles, sealed_needles};
use crumbs::digest::Digest;
use crumbs::gen::{self, registry};
use crumbs::protocol::{LearnerProcess, SearchConfig};
use crumbs::verify::{evaluate, Verdict};
use crumbs::visible::{Encoding, VisibleCrumb, VisibleError};

const PROBE: &str = env!("CARGO_BIN_EXE_crumbs-probe-learner");

fn tmp(name: &str) -> String {
    let d = std::env::temp_dir().join(format!("crumbs-test-{}-{name}", std::process::id()));
    let _ = std::fs::remove_file(&d);
    d.to_string_lossy().into_owned()
}

#[test]
fn crumb_v1_format_pass() {
    let pairs = vec![
        (vec![1u64], vec![3u64]),
        (vec![2], vec![5]),
        (vec![10], vec![21]),
    ];
    let c = VisibleCrumb::from_values(Encoding::DecimalUtf8, 0, 0, &pairs, None).unwrap();
    let b = c.to_bytes();
    assert_eq!(&b[..4], b"CRB1");
    assert_eq!(VisibleCrumb::from_bytes(&b).unwrap(), c);
    // The digest is BLAKE3 over exactly these bytes and nothing else.
    assert_eq!(
        c.digest(),
        crumbs::digest::digest(crumbs::digest::Domain::CrumbDigest, &b)
    );
    // Any byte change changes identity; a sealed label cannot influence it.
    let c2 = VisibleCrumb::from_values(
        Encoding::DecimalUtf8,
        0,
        0,
        &[(vec![1], vec![3]), (vec![2], vec![5]), (vec![10], vec![22])],
        None,
    )
    .unwrap();
    assert_ne!(c.digest(), c2.digest());

    // Strict decoding.
    let mut bad = b.clone();
    bad[0] = b'X';
    assert_eq!(VisibleCrumb::from_bytes(&bad), Err(VisibleError::Magic));
    let mut bad = b.clone();
    bad[4] = 2;
    assert_eq!(VisibleCrumb::from_bytes(&bad), Err(VisibleError::Version));
    let mut bad = b.clone();
    bad.push(0);
    assert!(
        VisibleCrumb::from_bytes(&bad).is_err(),
        "trailing byte accepted"
    );
    let mut bad = b.clone();
    bad[7] = 0x80;
    assert!(
        VisibleCrumb::from_bytes(&bad).is_err(),
        "unknown flag accepted"
    );
    // Non-canonical decimal ("03" instead of "3") is rejected.
    let text = String::from_utf8_lossy(&b).to_string();
    assert!(text.contains('3'));
    let mut e = crumbs::canon::Enc::new();
    e.bytes(b"CRB1")
        .u16(1)
        .u8(1)
        .u8(0)
        .u8(1)
        .u8(1)
        .u8(0)
        .u8(0)
        .u32(1)
        .blob(b"1")
        .blob(b"03");
    assert_eq!(
        VisibleCrumb::from_bytes(&e.finish()),
        Err(VisibleError::NonCanonical)
    );

    // Raw little-endian encoding round-trips and masks to the lane width.
    let r = VisibleCrumb::from_values(
        Encoding::RawLe,
        2,
        1,
        &[(vec![0x1_0005], vec![0x1FF])],
        None,
    )
    .unwrap();
    assert_eq!(r.values(), vec![(vec![5], vec![0xFF])]);
    assert_eq!(VisibleCrumb::from_bytes(&r.to_bytes()).unwrap(), r);
}

/// Golden identities: any behavioural change to a generator or to the
/// canonical encodings without a version bump fails here.
#[test]
fn crumb_determinism_golden_pass() {
    let reg = registry();
    let mut acc = blake3::Hasher::new();
    for f in &reg.families {
        for seed in [0u64, 1, 99] {
            let g = gen::generate(f, seed);
            acc.update(&g.visible.digest().0);
            acc.update(&g.sealed.digest().0);
            acc.update(&g.sealed.generator_instance_digest.0);
        }
    }
    let got = Digest(*acc.finalize().as_bytes()).hex();
    let golden = include_str!("golden_registry_digest.txt").trim();
    assert_eq!(
        got, golden,
        "generator output drifted; bump GENERATOR_VERSION and re-pin"
    );
}

#[test]
fn sealed_boundary_pass_visible_bytes() {
    for f in &registry().families {
        for seed in [0u64, 5, 11] {
            let g = gen::generate(f, seed);
            let bytes = g.visible.to_bytes();
            let hits = find_labels(&bytes);
            assert!(
                hits.is_empty(),
                "{}: labels in visible bytes: {hits:?}",
                f.key
            );
            let needles = sealed_needles(&g.sealed, &g.visible.values(), 1 << 32);
            let hits: Vec<String> = needles
                .iter()
                .filter(|n| crumbs::audit::contains(&bytes, n))
                .map(|n| String::from_utf8(n.clone()).unwrap_or_else(|_| format!("{n:?}")))
                .collect();
            assert!(
                hits.is_empty(),
                "{} seed {seed}: sealed value in visible bytes: {hits:?} in {:?}",
                f.key,
                String::from_utf8_lossy(&bytes)
            );
            // Sealed digests appear nowhere in the visible crumb.
            for d in [
                g.sealed.digest(),
                g.sealed.generator_instance_digest,
                g.sealed.heldouts.digest(),
            ] {
                assert!(!crumbs::audit::contains(&bytes, &d.0));
            }
        }
    }
}

#[test]
fn sealed_boundary_pass_debug_display_errors() {
    let f = registry().by_key("decoy-simple-rule-w12").unwrap();
    let g = gen::generate(f, 3);
    for s in [
        format!("{:?}", g.sealed),
        format!("{:?}", g.sealed.heldouts),
        format!("{:?}", g.sealed.heldouts.adversarial.first().unwrap()),
    ] {
        assert!(
            find_labels(s.as_bytes()).is_empty(),
            "Debug leaked labels: {s}"
        );
        for ex in g.sealed.heldouts.adversarial.iter() {
            assert!(
                !s.contains(&ex.output[0].to_string()) || ex.output[0] < 1000,
                "Debug leaked a held-out value: {s}"
            );
        }
        assert!(s.contains("redacted"));
    }
    // Error types carry no content.
    for e in [
        VisibleError::Magic,
        VisibleError::Version,
        VisibleError::Shape,
        VisibleError::Lane,
        VisibleError::Length,
        VisibleError::NonCanonical,
    ] {
        let s = e.to_string();
        assert!(find_labels(s.as_bytes()).is_empty());
        assert!(s.len() < 64);
    }
    let pe = crumbs::protocol::ProtocolError::Violation.to_string();
    assert!(find_labels(pe.as_bytes()).is_empty());
    let de = crumbs::canon::DecodeError.to_string();
    assert!(find_labels(de.as_bytes()).is_empty());
}

type ProbeRun = (
    Result<Vec<(u8, Verdict)>, String>,
    Vec<u8>,
    Vec<u8>,
    Vec<Vec<u8>>,
);

fn run_probe(mode: &str, fam: &str, seed: u64, max_q: u32) -> ProbeRun {
    let loot = tmp(&format!("{mode}-{fam}-{seed}"));
    let mut lp = LearnerProcess::spawn(PROBE, &[mode, &loot], true).unwrap();
    let cfg = SearchConfig {
        max_oracle_queries: max_q,
        ..SearchConfig::default()
    };
    lp.hello(&cfg, false).unwrap();
    let f = registry().by_key(fam).unwrap();
    let g = gen::generate(f, seed);
    let mut verdicts = Vec::new();
    let mut n = 0u32;
    let res = lp.run_crumb(&g.visible, &mut |sub| {
        n += 1;
        if n > max_q {
            return Verdict::BudgetExhausted;
        }
        let ev = evaluate(&g.visible, &g.sealed, &sub.program, sub.hypotheses, false);
        verdicts.push((sub.hypotheses, ev.verdict));
        ev.verdict
    });
    let transcript = lp.transcript.clone().unwrap();
    let out = match res {
        Ok(_) => {
            let _ = lp.shutdown();
            Ok(verdicts)
        }
        Err(e) => {
            lp.kill();
            Err(e.to_string())
        }
    };
    let needles = sealed_needles(&g.sealed, &g.visible.values(), 0);
    let loot_bytes = std::fs::read(&loot).unwrap_or_default();
    (out, transcript, loot_bytes, needles)
}

/// Everything sent to a learner is sterile: no labels, no digests, no held-out values.
#[test]
fn sealed_boundary_pass_protocol_transcript() {
    for (fam, seed) in [
        ("chain-basic-l2-w16", 1u64),
        ("decoy-simple-rule-w12", 2),
        ("insufficient-evidence-n2", 3),
        ("capability-03", 4),
    ] {
        let (res, transcript, loot, needles) = run_probe("honest", fam, seed, 4);
        res.unwrap();
        // The learner received exactly the transcript, then the shutdown frame.
        assert_eq!(loot, [transcript.clone(), vec![0x05, 0, 0, 0, 0]].concat());
        assert!(
            find_labels(&transcript).is_empty(),
            "{fam}: labels in transcript"
        );
        // Held-out values that are not also visible never cross the boundary.
        let vis = gen::generate(registry().by_key(fam).unwrap(), seed)
            .visible
            .to_bytes();
        let leaked: Vec<_> = needles
            .iter()
            .filter(|n| {
                n.len() >= 8
                    && crumbs::audit::contains(&transcript, n)
                    && !crumbs::audit::contains(&vis, n)
            })
            .collect();
        assert!(leaked.is_empty(), "{fam}: sealed bytes in transcript");
    }
}

/// Negative test: a learner that asks for held-out data through the protocol
/// gets nothing. The session aborts on the forged frame and no reply is sent.
#[test]
fn no_hidden_access_hostile_learner() {
    let (res, transcript, loot, needles) = run_probe("hostile", "decoy-simple-rule-w12", 9, 4);
    assert!(res.is_err(), "a forged frame must abort the session");
    // The learner received HELLO and CRUMB only.
    let mut kinds = Vec::new();
    let mut i = 0;
    while i + 5 <= loot.len() {
        kinds.push(loot[i]);
        let n = u32::from_le_bytes(loot[i + 1..i + 5].try_into().unwrap()) as usize;
        i += 5 + n;
    }
    assert_eq!(
        kinds,
        vec![0x01, 0x02],
        "learner received more than HELLO and CRUMB"
    );
    assert_eq!(transcript, loot);
    assert_eq!(
        find_needles(
            &loot,
            &needles
                .into_iter()
                .filter(|n| n.len() >= 32)
                .collect::<Vec<_>>()
        ),
        0
    );
}

/// Negative test: the oracle cannot be mined. After the query budget every
/// answer is "budget exhausted", and every answer is a single byte.
#[test]
fn no_hidden_access_oracle_budget() {
    let (res, _transcript, loot, _) = run_probe("flood", "chain-basic-l3-w16", 12, 3);
    let verdicts = res.unwrap();
    assert!(
        verdicts.len() <= 3,
        "the sealed evaluator ran more than the budget"
    );
    let mut i = 0;
    let mut vbytes = Vec::new();
    while i + 5 <= loot.len() {
        let t = loot[i];
        let n = u32::from_le_bytes(loot[i + 1..i + 5].try_into().unwrap()) as usize;
        if t == 0x03 {
            assert_eq!(n, 1, "a verdict frame carried more than one byte");
            vbytes.push(loot[i + 5]);
        }
        i += 5 + n;
    }
    assert!(vbytes.iter().all(|b| (1..=3).contains(b)));
    assert!(
        vbytes.iter().skip(3).all(|b| *b == 3),
        "answers after the budget must be 'exhausted'"
    );
}
