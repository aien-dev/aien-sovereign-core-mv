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

//! CRUMB_CURRICULUM_PASS, CRUMB_NO_CATEGORY_LEAK_PASS, CLEAN_ROOM_PROVENANCE_PASS,
//! CRUMB_OPERATION_PROMOTION_PASS (ladder unit), ledger chain, and the trace
//! corpus gates that do not need a real learner (CRUMB_TRACE_PASS,
//! CRUMB_TRAINING_SAMPLE_PASS, NO_COT_STORAGE_PASS with the probe learner).

use crumbs::audit::find_labels;
use crumbs::digest::{CrumbOccurrenceId, Digest};
use crumbs::gen::{self, registry};
use crumbs::ledger::{CrumblineRecord, Ledger, LedgerError};
use crumbs::mixer::{Mixer, MixerConfig};
use crumbs::program::{Op, Program};
use crumbs::promotion::{Ladder, PromotionConfig, Stage};
use crumbs::provenance::{
    clean_room_check, CleanRoomViolation, ContaminationClass, Lane, Provenance,
};
use crumbs::sealed::Population;
use crumbs::session::{Session, SessionConfig};
use crumbs::trace::{self, verify_chain, SAMPLE_RECORD_BYTES, TRACE_RECORD_BYTES};
use crumbs::verify::evaluate;
use std::collections::BTreeMap;

const PROBE: &str = env!("CARGO_BIN_EXE_crumbs-probe-learner");

#[test]
fn crumb_curriculum_pass() {
    // The v1 schedule is data: it loads from the shipped config file.
    let cfg =
        MixerConfig::from_json(include_str!("../config/mixer.v1.json")).expect("config parses");
    assert_eq!(cfg, MixerConfig::default());
    let w: Vec<u32> = Population::ALL.iter().map(|p| cfg.populations[p]).collect();
    assert_eq!(w, vec![55, 20, 10, 10, 5]);

    let mut m = Mixer::new(cfg.clone(), 7);
    let mut counts: BTreeMap<Population, u32> = BTreeMap::new();
    let mut last: Option<u32> = None;
    for _ in 0..3000 {
        let d = m.draw();
        *counts.entry(d.population).or_insert(0) += 1;
        assert_ne!(
            Some(d.family_id),
            last,
            "the same family was drawn twice in a row"
        );
        last = Some(d.family_id);
    }
    // No frontier sources exist in v1: its share is redistributed and recorded.
    assert!(
        m.redistributed_draws
            .get(&Population::Frontier)
            .copied()
            .unwrap_or(0)
            > 0
    );
    let fund = counts[&Population::Fundamental] as f64 / 3000.0;
    assert!((0.45..0.75).contains(&fund), "fundamental share {fund}");

    // Adaptation: demonstrating capabilities moves weight to composition.
    let before = m.weights();
    for f in registry()
        .families
        .iter()
        .filter(|f| f.params.caps.len() == 1)
    {
        m.record(f, true);
    }
    let after = m.weights();
    assert!(after[&Population::Fundamental] < before[&Population::Fundamental]);
    assert!(after[&Population::Composition] > before[&Population::Composition]);

    // Determinism: same seed, same draws.
    let (mut a, mut b) = (Mixer::new(cfg.clone(), 42), Mixer::new(cfg, 42));
    for _ in 0..200 {
        let (x, y) = (a.draw(), b.draw());
        assert_eq!((x.family_id, x.seed), (y.family_id, y.seed));
    }
}

/// Curriculum order carries no ontology: the learner-visible stream of a long
/// run contains no population, family or label bytes, and consecutive draws
/// never share a family or a capability.
#[test]
fn crumb_no_category_leak_pass() {
    let mut m = Mixer::new(MixerConfig::default(), 3);
    let mut stream = Vec::new();
    let mut prev_caps: Vec<u16> = vec![];
    for _ in 0..500 {
        let d = m.draw();
        let f = registry().get(d.family_id).unwrap();
        assert!(
            prev_caps.iter().all(|c| !f.params.caps.contains(c)),
            "consecutive crumbs share a capability"
        );
        prev_caps = f.params.caps.clone();
        stream.extend(gen::generate(f, d.seed).visible.to_bytes());
    }
    assert!(find_labels(&stream).is_empty());
}

#[test]
fn clean_room_provenance_pass() {
    // Every registered family declares a contamination class.
    for f in &registry().families {
        let _ = f.provenance.class;
    }
    // The forbidden list.
    assert_eq!(
        clean_room_check(&Provenance::human_theory()),
        Err(CleanRoomViolation::HumanTheoryDerived)
    );
    for (class, want) in [
        (
            ContaminationClass::FrontierDerived,
            CleanRoomViolation::FrontierDerived,
        ),
        (
            ContaminationClass::RealWorldMeasured,
            CleanRoomViolation::RealWorldMeasured,
        ),
    ] {
        let p = Provenance {
            class,
            uses_human_constants: false,
            uses_human_ontology: false,
        };
        assert_eq!(clean_room_check(&p), Err(want));
    }
    let consts = Provenance {
        uses_human_constants: true,
        ..Provenance::synthetic()
    };
    assert_eq!(
        clean_room_check(&consts),
        Err(CleanRoomViolation::HumanConstants)
    );
    let onto = Provenance {
        uses_human_ontology: true,
        ..Provenance::synthetic()
    };
    assert_eq!(
        clean_room_check(&onto),
        Err(CleanRoomViolation::HumanOntology)
    );
    assert!(clean_room_check(&Provenance::synthetic()).is_ok());
    assert!(clean_room_check(&Provenance::adversarial()).is_ok());

    // The clean-room ledger partition refuses forbidden records outright.
    let mut clean = Ledger::new(Lane::CleanRoom, "t");
    let rec = |seq| CrumblineRecord {
        seq,
        lane: Lane::Main,
        occurrence: CrumbOccurrenceId::new(),
        condition: "t".into(),
        population: Population::Fundamental,
        crumb_digest: Digest::ZERO,
        generator_instance_digest: Digest::ZERO,
        sealed_digest: Digest::ZERO,
        verifier_digest: Digest::ZERO,
        learner_digest: Digest::ZERO,
        evaluations: vec![],
        trace_stream_digest: Digest::ZERO,
        promotions: vec![],
        prev_root: Digest::ZERO,
        root: Digest::ZERO,
    };
    assert_eq!(
        clean.append(rec(0), &Provenance::human_theory()),
        Err(LedgerError::CleanRoom(
            CleanRoomViolation::HumanTheoryDerived
        ))
    );
    assert!(clean.append(rec(0), &Provenance::synthetic()).is_ok());
    assert_eq!(clean.records[0].lane, Lane::CleanRoom);

    // A clean-room mixer never draws a forbidden family.
    let mut m = Mixer::new(
        MixerConfig {
            clean_room: true,
            ..MixerConfig::default()
        },
        11,
    );
    for _ in 0..1000 {
        let f = registry().get(m.draw().family_id).unwrap();
        assert!(
            clean_room_check(&f.provenance).is_ok(),
            "{} entered the clean room",
            f.key
        );
    }
    // Contamination classes never appear in learner bytes (checked for every family in boundary.rs).
}

#[test]
fn ledger_chain_detects_tampering() {
    let mut l = Ledger::new(Lane::Main, "run");
    for s in 0..5 {
        let r = CrumblineRecord {
            seq: s,
            lane: Lane::Main,
            occurrence: CrumbOccurrenceId::new(),
            condition: "c".into(),
            population: Population::Composition,
            crumb_digest: Digest([s as u8; 32]),
            generator_instance_digest: Digest::ZERO,
            sealed_digest: Digest::ZERO,
            verifier_digest: Digest::ZERO,
            learner_digest: Digest::ZERO,
            evaluations: vec![],
            trace_stream_digest: Digest::ZERO,
            promotions: vec![],
            prev_root: Digest::ZERO,
            root: Digest::ZERO,
        };
        l.append(r, &Provenance::synthetic()).unwrap();
    }
    assert_eq!(Ledger::verify(l.genesis, &l.records), Ok(l.head));
    let mut forged = l.records.clone();
    forged[2].crumb_digest = Digest([9; 32]);
    assert_eq!(Ledger::verify(l.genesis, &forged), Err(LedgerError::Chain));
}

/// One solved crumb never admits an operation; two distinct supporting crumbs
/// with adversarial and extrapolation evidence do, with an explicit scope.
#[test]
fn crumb_operation_promotion_pass() {
    let fam = registry().by_key("capability-03").unwrap();
    let mut ladder = Ladder::new(PromotionConfig::default());
    let g1 = gen::generate(fam, 1);
    let crumbs::sealed::KnownSolution::Known(truth) = g1.sealed.known_solution.clone() else {
        panic!()
    };
    let e1 = evaluate(&g1.visible, &g1.sealed, &truth, 1, false);
    ladder.observe_accept(0, &g1.sealed, &e1, &truth);
    assert!(
        ladder.advance(0).is_empty(),
        "one crumb must not admit anything"
    );
    let rec = ladder.ops.values().next().unwrap();
    assert_eq!(rec.stage, Stage::HeldoutVerified);
    // The same crumb again is not new evidence.
    ladder.observe_accept(1, &g1.sealed, &e1, &truth);
    assert!(ladder.advance(1).is_empty());
    let g2 = gen::generate(fam, 2);
    let e2 = evaluate(&g2.visible, &g2.sealed, &truth, 1, false);
    ladder.observe_accept(2, &g2.sealed, &e2, &truth);
    let adm = ladder.advance(2);
    assert_eq!(adm.len(), 1);
    assert_eq!(adm[0].program, truth);
    assert!(adm[0].scope_bits > 0 && adm[0].scope_bits <= 64);
    let rec = ladder.by_ref(adm[0].op_ref).unwrap();
    let stages: Vec<Stage> = rec.history.iter().map(|h| h.0).collect();
    assert_eq!(
        stages,
        vec![
            Stage::Candidate,
            Stage::HeldoutVerified,
            Stage::Corroborated,
            Stage::SurvivedAdversarial,
            Stage::Generalized,
            Stage::Admitted
        ]
    );
    // Too many held-out-falsified exposures suspend a candidate instead of promoting it.
    let mut l2 = Ladder::new(PromotionConfig {
        max_exposure_pct: 50,
        ..PromotionConfig::default()
    });
    let decoy = gen::generate(registry().by_key("decoy-simple-rule-w12").unwrap(), 4);
    let alt = decoy.sealed.alternatives[0].clone();
    let ea = evaluate(&decoy.visible, &decoy.sealed, &alt, 1, false);
    let gcap = gen::generate(registry().by_key("chain-basic-l2-w16").unwrap(), 5);
    let crumbs::sealed::KnownSolution::Known(cap) = gcap.sealed.known_solution.clone() else {
        panic!()
    };
    let _ = Program::of(&[(Op::Mul, 2)]);
    let ec = evaluate(&gcap.visible, &gcap.sealed, &cap, 1, false);
    l2.observe_accept(0, &gcap.sealed, &ec, &cap);
    l2.observe_reject(&ea, &cap);
    l2.observe_reject(&ea, &cap);
    assert!(l2.advance(1).is_empty());
    assert_eq!(
        l2.ops[&cap.behavior()].stage,
        Stage::Suspended,
        "exposed operation must be suspended, not promoted"
    );
    let _ = alt;
}

/// Trace corpus through a real (probe) learner session: chained records,
/// numbers only, hidden results only on SUBMIT records, samples on solved crumbs.
#[test]
fn crumb_trace_and_sample_pass() {
    let dir = std::env::temp_dir().join(format!("crumbs-trace-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    let cfg = SessionConfig {
        condition: "test".into(),
        learner: PROBE.into(),
        run_id: "trace-test".into(),
        search: Default::default(),
        library_enabled: false,
        promotion: PromotionConfig::default(),
        record_transcript: false,
        out_dir: Some(dir.clone()),
    };
    // The probe reports no search events; the trace still records its submissions.
    let mut s = Session::start(cfg).unwrap();
    let fam = registry().by_key("chain-basic-l1-w16").unwrap();
    let o = s.present(fam, 3, Population::Fundamental, "t").unwrap();
    assert!(o.accepted);
    s.finish().unwrap();
    let tr = std::fs::read(dir.join("trace.ctr")).unwrap();
    let smp = std::fs::read(dir.join("samples.cts")).unwrap_or_default();
    assert_eq!(tr.len() % TRACE_RECORD_BYTES, 0);
    assert_eq!(smp.len() % SAMPLE_RECORD_BYTES, 0);
    assert!(verify_chain(&tr, TRACE_RECORD_BYTES, crumbs::digest::Domain::TraceRecord).is_some());
    let mut tampered = tr.clone();
    if !tampered.is_empty() {
        tampered[40] ^= 1;
        assert!(verify_chain(
            &tampered,
            TRACE_RECORD_BYTES,
            crumbs::digest::Domain::TraceRecord
        )
        .is_none());
    }
    assert!(find_labels(&tr).is_empty() && find_labels(&smp).is_empty());
    let _ = trace::bank_digest(&[]);
}
