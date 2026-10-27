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

//! BREADCRUMB-8: the acquisition and reuse experiment.
//!
//! Pre-registered design (fixed in code before any run):
//!
//! * Capabilities 0..=7 are TAUGHT (phase 1 may draw them); 8..=11 are never
//!   presented in phase 1.
//! * Phase 1: `phase1` mixer draws over the whole curriculum, restricted to
//!   taught capabilities and to the TRAIN composition pairs below.
//! * Phase 2: a fixed list of unseen test crumbs, identical in both
//!   conditions, presented with the library frozen:
//!   - `heldout-pair`    compositions of two taught capabilities never shown as a pair
//!   - `triple`          compositions of three taught capabilities (never shown)
//!   - `untaught`        compositions involving an untaught capability
//!   - `novel-primitive` the untaught capabilities themselves
//!   - `general`         ordinary chains (harm check: does a library slow search?)
//! * Conditions: CONTROL (admissions never sent; the learner cannot reuse
//!   discovered operations) and LEARNING (verified admissions are sent).
//!   Same learner binary, same search budget, same test seeds.
//!
//! The report states whatever happened. Nothing here is tuned to a result.

use crate::gen::{registry, rng::Rng, FamilySpec};
use crate::mixer::{Mixer, MixerConfig};
use crate::promotion::{PromotionConfig, Stage};
use crate::protocol::SearchConfig;
use crate::sealed::{AdversarialClass, DecoyStatus, Population};
use crate::session::{CrumbOutcome, Session, SessionConfig};
use crate::verify::{verifier_digest, EvalClass};
use serde::Serialize;
use std::collections::BTreeMap;
use std::path::PathBuf;

pub const TAUGHT: [u16; 8] = [0, 1, 2, 3, 4, 5, 6, 7];
pub const TRAIN_PAIRS: [&str; 7] = [
    "compose-00-01",
    "compose-02-03",
    "compose-04-05",
    "compose-06-07",
    "compose-01-03",
    "compose-03-05",
    "compose-05-07",
];
pub const HELDOUT_PAIRS: [&str; 6] = [
    "compose-01-02",
    "compose-03-04",
    "compose-05-06",
    "compose-00-02",
    "compose-02-04",
    "compose-04-06",
];
pub const TRIPLES: [&str; 4] = [
    "compose-00-01-02",
    "compose-03-04-05",
    "compose-00-03-06",
    "compose-01-04-07",
];
pub const UNTAUGHT_PAIRS: [&str; 11] = [
    "compose-07-08",
    "compose-08-09",
    "compose-09-10",
    "compose-10-11",
    "compose-11-00",
    "compose-06-08",
    "compose-07-09",
    "compose-08-10",
    "compose-09-11",
    "compose-10-00",
    "compose-11-01",
];
pub const NOVEL_PRIMITIVES: [&str; 4] = [
    "capability-08",
    "capability-09",
    "capability-10",
    "capability-11",
];
pub const GENERAL: [&str; 4] = [
    "chain-basic-l2-w16",
    "chain-basic-l3-w16",
    "latent-offset-w16",
    "mod-affine-w12",
];
pub const GROUPS: [&str; 5] = [
    "heldout-pair",
    "triple",
    "untaught",
    "novel-primitive",
    "general",
];

#[derive(Clone, Debug, Serialize)]
pub struct TestItem {
    pub group: &'static str,
    pub family: String,
    pub seed: u64,
}

pub fn test_plan(run_seed: u64) -> Vec<TestItem> {
    let mut out = Vec::new();
    let groups: [(&'static str, &[&str], u64); 5] = [
        ("heldout-pair", &HELDOUT_PAIRS, 3),
        ("triple", &TRIPLES, 3),
        ("untaught", &UNTAUGHT_PAIRS, 2),
        ("novel-primitive", &NOVEL_PRIMITIVES, 2),
        ("general", &GENERAL, 2),
    ];
    for (gi, (g, fams, reps)) in groups.into_iter().enumerate() {
        for (i, f) in fams.iter().enumerate() {
            for r in 0..reps {
                let seed = Rng::named("experiment-test-seed", &[run_seed, gi as u64, i as u64, r])
                    .next_u64();
                out.push(TestItem {
                    group: g,
                    family: f.to_string(),
                    seed,
                });
            }
        }
    }
    out
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GroupMetrics {
    pub crumbs: u32,
    pub supported: u32,
    pub solved: u32,
    pub verified: u32,
    pub solve_rate: f64,
    pub interpolation_pass_rate: f64,
    pub extrapolation_pass_rate: f64,
    pub adversarial_pass_rate: f64,
    pub submissions: u32,
    pub falsified_hidden: u32,
    pub false_generalization_rate: f64,
    pub mean_candidates_generated: f64,
    pub mean_candidates_evaluated: f64,
    pub candidates_per_solution: f64,
    pub mean_exec_count: f64,
    pub mean_wall_ms: f64,
    pub reuse_rate: f64,
}

fn metrics(outs: &[&CrumbOutcome]) -> GroupMetrics {
    let mut m = GroupMetrics {
        crumbs: outs.len() as u32,
        ..Default::default()
    };
    if outs.is_empty() {
        return m;
    }
    let (mut ip, mut it, mut ep, mut et, mut ap, mut at) = (0u64, 0u64, 0u64, 0u64, 0u64, 0u64);
    let (mut gen_sum, mut eval_sum, mut exec_sum, mut wall, mut gen_to_solution) =
        (0u64, 0u64, 0u64, 0u64, 0u64);
    let mut reuse = 0u32;
    for o in outs {
        m.supported += o.supported as u32;
        m.solved += o.accepted as u32;
        m.verified += o.verified as u32;
        m.submissions += o.evaluations.len() as u32;
        m.falsified_hidden += o.falsified_hidden() as u32;
        gen_sum += o.stats.generated as u64;
        eval_sum += o.stats.evaluated as u64;
        exec_sum += o.stats.exec_count;
        wall += o.stats.wall_ns;
        reuse += o.reused_library() as u32;
        if o.accepted {
            gen_to_solution += o.stats.candidates_to_solution as u64;
        }
        // Held-out pass rates of the learner's final committed answer.
        if let Some(e) = o.evaluations.last() {
            ip += e.interpolation.passed as u64;
            it += e.interpolation.total as u64;
            ep += e.extrapolation.passed as u64;
            et += e.extrapolation.total as u64;
            ap += e.adversarial.passed as u64;
            at += e.adversarial.total as u64;
        }
    }
    let n = outs.len() as f64;
    let r = |a: u64, b: u64| if b == 0 { 0.0 } else { a as f64 / b as f64 };
    m.solve_rate = m.solved as f64 / n;
    m.interpolation_pass_rate = r(ip, it);
    m.extrapolation_pass_rate = r(ep, et);
    m.adversarial_pass_rate = r(ap, at);
    m.false_generalization_rate = r(m.falsified_hidden as u64, m.submissions as u64);
    m.mean_candidates_generated = gen_sum as f64 / n;
    m.mean_candidates_evaluated = eval_sum as f64 / n;
    m.candidates_per_solution = if m.solved == 0 {
        f64::NAN
    } else {
        gen_to_solution as f64 / m.solved as f64
    };
    m.mean_exec_count = exec_sum as f64 / n;
    m.mean_wall_ms = wall as f64 / n / 1e6;
    m.reuse_rate = if m.solved == 0 {
        0.0
    } else {
        reuse as f64 / m.solved as f64
    };
    m
}

/// Curriculum-wide epistemic metrics (phase 1).
#[derive(Clone, Debug, Default, Serialize)]
pub struct CurriculumMetrics {
    pub crumbs: u32,
    pub by_population: BTreeMap<String, u32>,
    pub redistributed_draws: BTreeMap<String, u32>,
    pub visible_fit_rate: f64,
    pub heldout_verified_rate: f64,
    pub rabbit_hole_crumbs: u32,
    /// Rabbit holes (classes A-E, G) whose first submitted answer was not held-out-falsified.
    pub adversarial_survival_rate: f64,
    pub false_generalization_rate: f64,
    pub insufficient_evidence_crumbs: u32,
    pub ambiguity_recognition_rate: f64,
    pub false_ambiguity_rate_on_clean: f64,
    pub noisy_crumbs: u32,
    pub noisy_verified: u32,
    pub operations_tracked: u32,
    pub operations_admitted: u32,
}

#[derive(Serialize)]
pub struct ConditionReport {
    pub condition: String,
    pub curriculum: CurriculumMetrics,
    pub test: BTreeMap<String, GroupMetrics>,
    pub test_all: GroupMetrics,
    pub ledger_head: String,
    pub ledger_records: usize,
    pub ledger_verified: bool,
}

#[derive(Serialize)]
pub struct PairedComparison {
    pub group: String,
    pub crumbs: u32,
    pub solved_control: u32,
    pub solved_learning: u32,
    pub solved_only_learning: u32,
    pub solved_only_control: u32,
    pub both_solved: u32,
    pub fewer_candidates_learning: u32,
    pub fewer_candidates_control: u32,
    pub equal_candidates: u32,
    /// Geometric mean of learning/control candidates-to-solution over crumbs both solved.
    pub geomean_candidate_ratio_both_solved: f64,
    pub mean_generated_control: f64,
    pub mean_generated_learning: f64,
    pub mean_exec_control: f64,
    pub mean_exec_learning: f64,
    pub mean_wall_ms_control: f64,
    pub mean_wall_ms_learning: f64,
    pub heldout_correct_control: u32,
    pub heldout_correct_learning: u32,
}

#[derive(Serialize)]
pub struct ExperimentReport {
    pub design: &'static str,
    pub run_seed: u64,
    pub phase1_draws: u64,
    pub search: SearchConfig,
    pub promotion: PromotionConfig,
    pub mixer: MixerConfig,
    pub test_plan: Vec<TestItem>,
    pub verifier_digest: String,
    pub learner_digest: String,
    pub registry_version: &'static str,
    pub generator_version: &'static str,
    pub control: ConditionReport,
    pub learning: ConditionReport,
    pub paired: Vec<PairedComparison>,
    pub aien_next_action_guidance: &'static str,
    pub wall_seconds: f64,
}

fn curriculum_metrics(
    outs: &[CrumbOutcome],
    ladder: &crate::promotion::Ladder,
    mixer: &Mixer,
) -> CurriculumMetrics {
    let ph: Vec<&CrumbOutcome> = outs.iter().filter(|o| o.phase == "phase1").collect();
    let mut c = CurriculumMetrics {
        crumbs: ph.len() as u32,
        ..Default::default()
    };
    for o in &ph {
        *c.by_population
            .entry(format!("{:?}", o.population))
            .or_insert(0) += 1;
    }
    for (p, n) in &mixer.redistributed_draws {
        c.redistributed_draws.insert(format!("{p:?}"), *n);
    }
    let n = ph.len().max(1) as f64;
    c.visible_fit_rate = ph.iter().filter(|o| o.stats.visible_fits > 0).count() as f64 / n;
    c.heldout_verified_rate = ph.iter().filter(|o| o.verified).count() as f64 / n;
    let rh: Vec<_> = ph
        .iter()
        .filter(|o| matches!(o.decoy, DecoyStatus::RabbitHole(k) if k != AdversarialClass::InsufficientEvidence))
        .collect();
    c.rabbit_hole_crumbs = rh.len() as u32;
    let survived = rh
        .iter()
        .filter(|o| o.first_class() != Some(EvalClass::FalsifiedHidden))
        .count();
    c.adversarial_survival_rate = if rh.is_empty() {
        f64::NAN
    } else {
        survived as f64 / rh.len() as f64
    };
    let subs: usize = ph.iter().map(|o| o.evaluations.len()).sum();
    let fh: usize = ph.iter().map(|o| o.falsified_hidden()).sum();
    c.false_generalization_rate = if subs == 0 {
        0.0
    } else {
        fh as f64 / subs as f64
    };
    let ie: Vec<_> = ph
        .iter()
        .filter(|o| {
            o.decoy == DecoyStatus::RabbitHole(AdversarialClass::InsufficientEvidence)
                && !o.evaluations.is_empty()
        })
        .collect();
    c.insufficient_evidence_crumbs = ie.len() as u32;
    let recog = ie
        .iter()
        .filter(|o| o.stats.hypotheses_at_first_submit >= 2)
        .count();
    c.ambiguity_recognition_rate = if ie.is_empty() {
        f64::NAN
    } else {
        recog as f64 / ie.len() as f64
    };
    let clean: Vec<_> = ph
        .iter()
        .filter(|o| o.decoy == DecoyStatus::Clean && !o.evaluations.is_empty())
        .collect();
    let false_amb = clean
        .iter()
        .filter(|o| o.stats.hypotheses_at_first_submit >= 2)
        .count();
    c.false_ambiguity_rate_on_clean = if clean.is_empty() {
        f64::NAN
    } else {
        false_amb as f64 / clean.len() as f64
    };
    let noisy: Vec<_> = ph
        .iter()
        .filter(|o| o.decoy == DecoyStatus::RabbitHole(AdversarialClass::Noisy))
        .collect();
    c.noisy_crumbs = noisy.len() as u32;
    c.noisy_verified = noisy.iter().filter(|o| o.verified).count() as u32;
    c.operations_tracked = ladder.ops.len() as u32;
    c.operations_admitted = ladder.count(Stage::Admitted) as u32;
    c
}

pub struct ConditionRun {
    pub outcomes: Vec<CrumbOutcome>,
    pub report: ConditionReport,
    pub learner_digest: String,
}

#[allow(clippy::too_many_arguments)]
pub fn run_condition(
    learner: &str,
    condition: &str,
    library: bool,
    run_seed: u64,
    phase1: u64,
    search: SearchConfig,
    mixer_cfg: &MixerConfig,
    out_dir: Option<PathBuf>,
) -> Result<ConditionRun, crate::protocol::ProtocolError> {
    let cfg = SessionConfig {
        condition: condition.to_string(),
        learner: learner.to_string(),
        run_id: format!("crumbline-experiment-{run_seed}-{condition}"),
        search,
        library_enabled: library,
        promotion: PromotionConfig::default(),
        record_transcript: false,
        out_dir,
    };
    let mut s = Session::start(cfg)?;
    let learner_digest = s.learner_digest.hex();
    let mut mixer = Mixer::new(mixer_cfg.clone(), run_seed);
    for _ in 0..phase1 {
        let d = mixer.draw();
        let fam = registry().get(d.family_id).expect("registered");
        let o = s.present(fam, d.seed, d.population, "phase1")?;
        mixer.record(fam, o.verified);
    }
    s.freeze_admissions();
    for item in test_plan(run_seed) {
        let fam: &FamilySpec = registry()
            .by_key(&item.family)
            .expect("test family registered");
        s.present(fam, item.seed, Population::Composition, item.group)?;
    }
    let genesis = s.ledger.genesis;
    let (ledger, outcomes, ladder) = s.finish()?;
    let verified = crate::ledger::Ledger::verify(genesis, &ledger.records).is_ok();
    let curriculum = curriculum_metrics(&outcomes, &ladder, &mixer);
    let mut test = BTreeMap::new();
    for g in GROUPS {
        let v: Vec<&CrumbOutcome> = outcomes.iter().filter(|o| o.phase == g).collect();
        test.insert(g.to_string(), metrics(&v));
    }
    let all: Vec<&CrumbOutcome> = outcomes.iter().filter(|o| o.phase != "phase1").collect();
    let report = ConditionReport {
        condition: condition.to_string(),
        curriculum,
        test,
        test_all: metrics(&all),
        ledger_head: ledger.head.hex(),
        ledger_records: ledger.records.len(),
        ledger_verified: verified,
    };
    Ok(ConditionRun {
        outcomes,
        report,
        learner_digest,
    })
}

pub fn paired(
    control: &[CrumbOutcome],
    learning: &[CrumbOutcome],
    group: &str,
) -> PairedComparison {
    let pick = |o: &&CrumbOutcome| {
        if group == "all" {
            o.phase != "phase1"
        } else {
            o.phase == group
        }
    };
    let c: Vec<&CrumbOutcome> = control.iter().filter(pick).collect();
    let l: Vec<&CrumbOutcome> = learning.iter().filter(pick).collect();
    let mut p = PairedComparison {
        group: group.to_string(),
        crumbs: c.len() as u32,
        solved_control: 0,
        solved_learning: 0,
        solved_only_learning: 0,
        solved_only_control: 0,
        both_solved: 0,
        fewer_candidates_learning: 0,
        fewer_candidates_control: 0,
        equal_candidates: 0,
        geomean_candidate_ratio_both_solved: f64::NAN,
        mean_generated_control: 0.0,
        mean_generated_learning: 0.0,
        mean_exec_control: 0.0,
        mean_exec_learning: 0.0,
        mean_wall_ms_control: 0.0,
        mean_wall_ms_learning: 0.0,
        heldout_correct_control: 0,
        heldout_correct_learning: 0,
    };
    let mut log_sum = 0.0;
    for (a, b) in c.iter().zip(l.iter()) {
        assert_eq!(
            a.crumb_digest, b.crumb_digest,
            "paired crumbs must be identical"
        );
        p.solved_control += a.accepted as u32;
        p.solved_learning += b.accepted as u32;
        p.heldout_correct_control += a.verified as u32;
        p.heldout_correct_learning += b.verified as u32;
        match (a.accepted, b.accepted) {
            (true, true) => {
                p.both_solved += 1;
                let x = a.stats.candidates_to_solution.max(1) as f64;
                let y = b.stats.candidates_to_solution.max(1) as f64;
                log_sum += (y / x).ln();
            }
            (false, true) => p.solved_only_learning += 1,
            (true, false) => p.solved_only_control += 1,
            _ => {}
        }
        let (ga, gb) = (a.stats.generated, b.stats.generated);
        if gb < ga {
            p.fewer_candidates_learning += 1;
        } else if ga < gb {
            p.fewer_candidates_control += 1;
        } else {
            p.equal_candidates += 1;
        }
        p.mean_generated_control += ga as f64;
        p.mean_generated_learning += gb as f64;
        p.mean_exec_control += a.stats.exec_count as f64;
        p.mean_exec_learning += b.stats.exec_count as f64;
        p.mean_wall_ms_control += a.stats.wall_ns as f64 / 1e6;
        p.mean_wall_ms_learning += b.stats.wall_ns as f64 / 1e6;
    }
    let n = c.len().max(1) as f64;
    p.mean_generated_control /= n;
    p.mean_generated_learning /= n;
    p.mean_exec_control /= n;
    p.mean_exec_learning /= n;
    p.mean_wall_ms_control /= n;
    p.mean_wall_ms_learning /= n;
    if p.both_solved > 0 {
        p.geomean_candidate_ratio_both_solved = (log_sum / p.both_solved as f64).exp();
    }
    p
}

pub fn run(
    learner: &str,
    run_seed: u64,
    phase1: u64,
    out: Option<PathBuf>,
) -> Result<ExperimentReport, crate::protocol::ProtocolError> {
    let t0 = std::time::Instant::now();
    let search = SearchConfig::default();
    let mixer_cfg = MixerConfig {
        capability_subset: Some(TAUGHT.to_vec()),
        composition_keys: Some(TRAIN_PAIRS.iter().map(|s| s.to_string()).collect()),
        ..MixerConfig::default()
    };
    let sub = |name: &str| out.as_ref().map(|d| d.join(name));
    let control = run_condition(
        learner,
        "control",
        false,
        run_seed,
        phase1,
        search,
        &mixer_cfg,
        sub("control"),
    )?;
    let learning = run_condition(
        learner,
        "learning",
        true,
        run_seed,
        phase1,
        search,
        &mixer_cfg,
        sub("learning"),
    )?;
    let mut paired_v = Vec::new();
    for g in GROUPS.iter().copied().chain(["all"]) {
        paired_v.push(paired(&control.outcomes, &learning.outcomes, g));
    }
    let report = ExperimentReport {
        design: "pre-registered: taught caps 0-7, train pairs fixed, 68 unseen test crumbs, library frozen before test, identical budgets",
        run_seed,
        phase1_draws: phase1,
        search,
        promotion: PromotionConfig::default(),
        mixer: mixer_cfg,
        test_plan: test_plan(run_seed),
        verifier_digest: verifier_digest().hex(),
        learner_digest: control.learner_digest.clone(),
        registry_version: crate::gen::registry::REGISTRY_VERSION,
        generator_version: crate::gen::GENERATOR_VERSION,
        control: control.report,
        learning: learning.report,
        paired: paired_v,
        aien_next_action_guidance: "not available: no trained AIEN_0 policy exists yet; the trace corpus produced here is its training input",
        wall_seconds: t0.elapsed().as_secs_f64(),
    };
    if let Some(d) = &out {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(
            d.join("report.json"),
            serde_json::to_string_pretty(&report).unwrap(),
        );
        for (name, outs) in [
            ("control", &control.outcomes),
            ("learning", &learning.outcomes),
        ] {
            let _ = std::fs::write(
                d.join(format!("{name}-outcomes.json")),
                serde_json::to_string(outs).unwrap(),
            );
        }
    }
    Ok(report)
}

/// Pooled comparison over independent replicates (different curriculum draws
/// and different test seeds per replicate; same design).
#[derive(Serialize)]
pub struct ReplicateReport {
    pub run_seeds: Vec<u64>,
    pub phase1_draws: u64,
    pub pooled: Vec<PairedComparison>,
    pub per_seed_all: Vec<PairedComparison>,
    pub control_curriculum: Vec<CurriculumMetrics>,
    pub learning_curriculum: Vec<CurriculumMetrics>,
    pub wall_seconds: f64,
}

pub fn replicate(
    learner: &str,
    first_seed: u64,
    k: u64,
    phase1: u64,
    out: Option<PathBuf>,
) -> Result<ReplicateReport, crate::protocol::ProtocolError> {
    let t0 = std::time::Instant::now();
    let search = SearchConfig::default();
    let mixer_cfg = MixerConfig {
        capability_subset: Some(TAUGHT.to_vec()),
        composition_keys: Some(TRAIN_PAIRS.iter().map(|s| s.to_string()).collect()),
        ..MixerConfig::default()
    };
    let (mut all_c, mut all_l) = (Vec::new(), Vec::new());
    let mut per_seed = Vec::new();
    let (mut cc, mut lc) = (Vec::new(), Vec::new());
    let seeds: Vec<u64> = (first_seed..first_seed + k).collect();
    for &seed in &seeds {
        let sub = |name: &str| {
            out.as_ref()
                .map(|d| d.join(format!("seed-{seed}")).join(name))
        };
        let c = run_condition(
            learner,
            "control",
            false,
            seed,
            phase1,
            search,
            &mixer_cfg,
            sub("control"),
        )?;
        let l = run_condition(
            learner,
            "learning",
            true,
            seed,
            phase1,
            search,
            &mixer_cfg,
            sub("learning"),
        )?;
        per_seed.push(paired(&c.outcomes, &l.outcomes, "all"));
        cc.push(c.report.curriculum.clone());
        lc.push(l.report.curriculum.clone());
        all_c.extend(c.outcomes);
        all_l.extend(l.outcomes);
    }
    let pooled = GROUPS
        .iter()
        .copied()
        .chain(["all"])
        .map(|g| paired(&all_c, &all_l, g))
        .collect();
    let rep = ReplicateReport {
        run_seeds: seeds,
        phase1_draws: phase1,
        pooled,
        per_seed_all: per_seed,
        control_curriculum: cc,
        learning_curriculum: lc,
        wall_seconds: t0.elapsed().as_secs_f64(),
    };
    if let Some(d) = &out {
        let _ = std::fs::create_dir_all(d);
        let _ = std::fs::write(
            d.join("replicates.json"),
            serde_json::to_string_pretty(&rep).unwrap(),
        );
    }
    Ok(rep)
}

pub fn main(args: &[String]) -> i32 {
    let get = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let learner = get("--learner").expect("--learner PATH");
    let seed = get("--seed").map_or(20260927, |v| v.parse().unwrap());
    let phase1 = get("--phase1").map_or(120, |v| v.parse().unwrap());
    let out = get("--out").map(PathBuf::from);
    if let Some(k) = get("--replicates") {
        return match replicate(&learner, seed, k.parse().unwrap(), phase1, out) {
            Ok(r) => {
                println!("{}", serde_json::to_string_pretty(&r.pooled).unwrap());
                println!("wall {:.1}s", r.wall_seconds);
                0
            }
            Err(e) => {
                eprintln!("{e}");
                1
            }
        };
    }
    match run(&learner, seed, phase1, out) {
        Ok(r) => {
            println!("{}", serde_json::to_string_pretty(&r.paired).unwrap());
            println!("wall {:.1}s", r.wall_seconds);
            0
        }
        Err(e) => {
            eprintln!("{e}");
            1
        }
    }
}
