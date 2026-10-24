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

//! BREADCRUMB-2 gates: CRUMB_GENERATORS_PASS, CRUMB_ADVERSARIAL_PASS, CRUMB_AMBIGUITY_PASS.

use crumbs::gen::{self, registry, Mechanism};
use crumbs::sealed::{AdversarialClass, DecoyStatus, HeldoutTier, KnownSolution, Rung};
use crumbs::verify::{evaluate, EvalClass};
use std::collections::HashSet;

const SEEDS: [u64; 6] = [0, 1, 2, 3, 17, 1_000_003];

#[test]
fn crumb_generators_pass() {
    let reg = registry();
    assert!(
        reg.families.len() >= 100,
        "registered families: {}",
        reg.families.len()
    );
    let mechs: HashSet<_> = reg.families.iter().map(|f| f.mechanism).collect();
    assert!(mechs.len() >= 24, "distinct mechanisms: {}", mechs.len());
    assert_eq!(
        mechs.len(),
        Mechanism::ALL.len(),
        "every mechanism is registered"
    );
    let rungs: HashSet<_> = reg.families.iter().map(|f| f.rung).collect();
    for r in Rung::ALL {
        assert!(rungs.contains(&r), "rung {r:?} has no family");
    }
    let keys: HashSet<_> = reg.families.iter().map(|f| f.key.clone()).collect();
    assert_eq!(keys.len(), reg.families.len(), "family keys unique");

    for fam in &reg.families {
        for seed in SEEDS {
            let g = gen::generate(fam, seed);
            let bytes = g.visible.to_bytes();
            let back =
                crumbs::visible::VisibleCrumb::from_bytes(&bytes).expect("visible round-trips");
            assert_eq!(back, g.visible);
            assert_eq!(g.sealed.crumb_digest, g.visible.digest());
            assert!(
                g.visible.examples.len() >= 2,
                "{} has too few visible examples",
                fam.key
            );
            assert!(
                !g.sealed.heldouts.is_empty(),
                "{} has no held-outs",
                fam.key
            );

            // Held-out inputs are disjoint from visible inputs and from each other.
            let vis: HashSet<Vec<u64>> = g.visible.values().into_iter().map(|(i, _)| i).collect();
            let mut seen = HashSet::new();
            for t in HeldoutTier::ALL {
                for ex in g.sealed.heldouts.tier(t) {
                    assert!(
                        !vis.contains(&ex.input),
                        "{} held-out repeats a visible input",
                        fam.key
                    );
                    assert!(
                        seen.insert(ex.input.clone()),
                        "{} held-out tiers overlap",
                        fam.key
                    );
                }
            }

            // A known CPG1 truth must pass the verifier.
            if let KnownSolution::Known(p) = &g.sealed.known_solution {
                if g.visible.out_arity == 1 {
                    let ev = evaluate(&g.visible, &g.sealed, p, 1, false);
                    assert!(
                        matches!(
                            ev.class,
                            EvalClass::Verified | EvalClass::AmbiguousUnderdetermined
                        ),
                        "{} seed {seed}: truth evaluated as {:?}",
                        fam.key,
                        ev.class
                    );
                }
            }
        }
    }
}

#[test]
fn crumb_determinism_pass() {
    for fam in &registry().families {
        let a = gen::generate(fam, 42);
        let b = gen::generate(fam, 42);
        assert_eq!(a.visible.to_bytes(), b.visible.to_bytes(), "{}", fam.key);
        assert_eq!(a.sealed.to_bytes(), b.sealed.to_bytes(), "{}", fam.key);
        assert_eq!(a.sealed.digest(), b.sealed.digest());
        let c = gen::generate(fam, 43);
        assert_ne!(
            a.sealed.generator_instance_digest,
            c.sealed.generator_instance_digest
        );
    }
}

/// Every rabbit hole: the alternative fits all visible examples and fails a held-out.
#[test]
fn crumb_adversarial_pass() {
    let mut classes = HashSet::new();
    for fam in &registry().families {
        let DecoyStatus::RabbitHole(class) = fam.decoy else {
            continue;
        };
        for seed in SEEDS {
            let g = gen::generate(fam, seed);
            classes.insert(class);
            if class == AdversarialClass::Noisy {
                assert!(!g.sealed.noisy_visible.is_empty());
                let KnownSolution::Known(truth) = &g.sealed.known_solution else {
                    panic!()
                };
                let vals = g.visible.values();
                let wrong = vals.iter().filter(|(i, o)| truth.run(i[0]) != o[0]).count();
                assert_eq!(
                    wrong,
                    g.sealed.noisy_visible.len(),
                    "{}: noise count",
                    fam.key
                );
                assert!(wrong < vals.len() / 2, "truth still explains the majority");
                continue;
            }
            assert!(
                !g.sealed.alternatives.is_empty(),
                "{} seed {seed}: no alternative recorded",
                fam.key
            );
            for alt in &g.sealed.alternatives {
                let ev = evaluate(&g.visible, &g.sealed, alt, 1, false);
                assert_ne!(
                    ev.class,
                    EvalClass::FalsifiedVisible,
                    "{} seed {seed}: alternative must fit visible",
                    fam.key
                );
                assert!(
                    matches!(ev.class, EvalClass::FalsifiedHidden),
                    "{} seed {seed}: held-outs must break the alternative, got {:?}",
                    fam.key,
                    ev.class
                );
            }
        }
    }
    for c in [
        AdversarialClass::SimpleWrongRule,
        AdversarialClass::MultiFit,
        AdversarialClass::SpuriousDimension,
        AdversarialClass::PrefixDiverges,
        AdversarialClass::ElegantArbitrary,
        AdversarialClass::InsufficientEvidence,
        AdversarialClass::Noisy,
    ] {
        assert!(classes.contains(&c), "class {c:?} not generated");
    }
}

/// Insufficient-evidence crumbs carry >= 2 rival explanations, and the
/// evaluation records whether the learner recognized the ambiguity.
#[test]
fn crumb_ambiguity_pass() {
    let reg = registry();
    let fams: Vec<_> = reg
        .families
        .iter()
        .filter(|f| f.decoy == DecoyStatus::RabbitHole(AdversarialClass::InsufficientEvidence))
        .collect();
    assert!(!fams.is_empty());
    for fam in fams {
        for seed in SEEDS {
            let g = gen::generate(fam, seed);
            assert!(g.sealed.alternatives.len() >= 2, "{} seed {seed}", fam.key);
            let KnownSolution::Known(truth) = &g.sealed.known_solution else {
                panic!()
            };
            let lone = evaluate(&g.visible, &g.sealed, truth, 1, false);
            assert_eq!(lone.class, EvalClass::AmbiguousUnderdetermined);
            assert_eq!(lone.ambiguity_recognized, Some(false));
            let aware = evaluate(&g.visible, &g.sealed, truth, 3, false);
            assert_eq!(aware.ambiguity_recognized, Some(true));
        }
    }
    let ns = reg.by_key("no-structure").unwrap();
    let g = gen::generate(ns, 5);
    assert_eq!(g.sealed.decoy_status, DecoyStatus::NoStructure);
    assert_eq!(g.sealed.known_solution, KnownSolution::NoCompactStructure);
}
