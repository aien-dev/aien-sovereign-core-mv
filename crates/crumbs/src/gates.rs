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

//! Learner-dependent qualification gates, run against a real learner binary
//! (for v1: Omega's `crumbline-learner`).
//!
//! crumbs gates --learner PATH [--out DIR]
//!
//! Prints one `GATE_NAME PASS|FAIL detail` line per gate and writes gates.json
//! plus an aien-proof assertion file (`aien-proof receipt import --assert ...`).

use crate::audit::{find_labels, read_process_writable_memory, sealed_specific_labels};
use crate::digest::Domain;
use crate::gen::{self, registry, rng::Rng};
use crate::program::{Op, Program};
use crate::promotion::{PromotionConfig, Stage};
use crate::protocol::{LearnerProcess, SearchConfig, Submission};
use crate::sealed::{HeldoutTier, KnownSolution, LaneExample, Population};
use crate::session::{CrumbOutcome, Session, SessionConfig};
use crate::trace::{verify_chain, ResultClass, SAMPLE_RECORD_BYTES, TRACE_RECORD_BYTES};
use crate::verify::{EvalClass, Verdict};
use crate::visible::{Encoding, VisibleCrumb};
use serde::Serialize;
use std::path::{Path, PathBuf};

#[derive(Serialize, Clone)]
pub struct GateResult {
    pub id: String,
    pub pass: bool,
    pub observed: String,
}

fn gate(id: &str, pass: bool, observed: String) -> GateResult {
    println!("{id} {} {observed}", if pass { "PASS" } else { "FAIL" });
    GateResult {
        id: id.to_string(),
        pass,
        observed,
    }
}

fn session(learner: &str, cond: &str, library: bool, dir: &Path) -> Session {
    Session::start(SessionConfig {
        condition: cond.into(),
        learner: learner.into(),
        run_id: format!("gates-{cond}"),
        search: SearchConfig::default(),
        library_enabled: library,
        promotion: PromotionConfig::default(),
        record_transcript: true,
        out_dir: Some(dir.join(cond)),
    })
    .expect("learner starts")
}

fn present(s: &mut Session, key: &str, seed: u64) -> CrumbOutcome {
    let f = registry()
        .by_key(key)
        .unwrap_or_else(|| panic!("family {key}"));
    s.present(f, seed, Population::Fundamental, "gates")
        .expect("session")
}

pub fn run_gates(learner: &str, out: &Path) -> Vec<GateResult> {
    let _ = std::fs::remove_dir_all(out);
    std::fs::create_dir_all(out).unwrap();
    let mut gates = Vec::new();

    // CRUMB_OMEGA_VISIBLE_PASS: the learner synthesizes programs that explain
    // the visible examples, and never submits one that does not.
    let mut s = session(learner, "visible", false, out);
    let (mut fits, mut bad_visible, mut n) = (0u32, 0usize, 0u32);
    let (mut expressible, mut expressible_found) = (0u32, 0u32);
    for key in [
        "chain-basic-l1-w16",
        "chain-basic-l2-w16",
        "chain-basic-l3-w16",
        "latent-offset-w16",
        "mod-affine-w12",
    ] {
        for seed in 1..=8 {
            let f = registry().by_key(key).unwrap();
            let g = gen::generate(f, seed);
            let o = s
                .present(f, seed, Population::Fundamental, "gates")
                .expect("session");
            n += 1;
            let found = o.stats.visible_fits > 0;
            fits += found as u32;
            bad_visible += o
                .evaluations
                .iter()
                .filter(|e| e.class == EvalClass::FalsifiedVisible && !e.robust_submission)
                .count();
            // Truth expressible as <= 3 single-step operations the learner itself
            // declared: an exhaustive depth-3 search must find a visible fit.
            if let KnownSolution::Known(t) = &g.sealed.known_solution {
                let in_bank = t.steps.iter().all(|st| {
                    o.declared_bank
                        .iter()
                        .any(|b| b.steps.len() == 1 && b.steps[0] == *st)
                });
                if in_bank && t.len() <= 3 {
                    expressible += 1;
                    expressible_found += found as u32;
                }
            }
        }
    }
    gates.push(gate(
        "CRUMB_OMEGA_VISIBLE_PASS",
        expressible > 0 && expressible_found == expressible && bad_visible == 0,
        format!("visible fit found on {expressible_found}/{expressible} crumbs whose truth is <= 3 of the learner's declared operations ({fits}/{n} overall); {bad_visible} exact submissions that failed the visible examples"),
    ));
    let (_, visible_outs, _) = s.finish().unwrap();

    // CRUMB_HIDDEN_VERIFY_PASS: visible fits that fail held-outs are recorded
    // as useful failures and search continues; acceptance needs held-outs.
    let mut s = session(learner, "hidden", false, out);
    let (mut useful_failures, mut continued, mut accepted_not_verified) = (0usize, 0u32, 0usize);
    for key in [
        "decoy-simple-rule-w12",
        "decoy-simple-rule-w16",
        "multi-fit-w12",
        "multi-fit-w16",
        "insufficient-evidence-n3",
        "prefix-diverges-k8",
    ] {
        for seed in 1..=4 {
            let o = present(&mut s, key, seed);
            let fh = o.falsified_hidden();
            useful_failures += fh;
            if fh > 0
                && (o.evaluations.len() > 1 || o.stats.generated > o.stats.candidates_to_solution)
            {
                continued += 1;
            }
            accepted_not_verified += o
                .evaluations
                .iter()
                .filter(|e| {
                    e.verdict == Verdict::Accept
                        && !matches!(
                            e.class,
                            EvalClass::Verified | EvalClass::AmbiguousUnderdetermined
                        )
                })
                .count();
        }
    }
    let (_, hidden_outs, _) = s.finish().unwrap();
    let rejected_records = count_records(
        &out.join("hidden").join("trace.ctr"),
        ResultClass::RejectedHidden,
    );
    gates.push(gate(
        "CRUMB_HIDDEN_VERIFY_PASS",
        useful_failures > 0 && continued > 0 && accepted_not_verified == 0 && rejected_records == useful_failures,
        format!("{useful_failures} visible-fit candidates falsified by held-outs, all recorded ({rejected_records} REJECTED_HIDDEN trace records); search continued after a rejection in {continued} crumbs; {accepted_not_verified} acceptances without held-out verification"),
    ));

    gates.push(no_hidden_access(learner, out));

    // Trace corpus gates over the real Omega sessions above.
    let (mut chains_ok, mut total_records, mut total_samples, mut label_hits) =
        (true, 0usize, 0usize, 0usize);
    for d in ["visible", "hidden"] {
        let tr = std::fs::read(out.join(d).join("trace.ctr")).unwrap_or_default();
        let smp = std::fs::read(out.join(d).join("samples.cts")).unwrap_or_default();
        chains_ok &= verify_chain(&tr, TRACE_RECORD_BYTES, Domain::TraceRecord).is_some();
        chains_ok &= verify_chain(&smp, SAMPLE_RECORD_BYTES, Domain::TrainingSample).is_some();
        total_records += tr.len() / TRACE_RECORD_BYTES;
        total_samples += smp.len() / SAMPLE_RECORD_BYTES;
        label_hits += find_labels(&tr).len() + find_labels(&smp).len();
    }
    let all: Vec<&CrumbOutcome> = visible_outs.iter().chain(hidden_outs.iter()).collect();
    let integrity: u32 = all.iter().map(|o| o.trace.integrity_violations).sum();
    let samples_for_solved = all
        .iter()
        .filter(|o| o.accepted && o.stats.generated > 0)
        .all(|o| o.trace.samples > 0);
    gates.push(gate(
        "CRUMB_TRACE_PASS",
        chains_ok && total_records > 0 && integrity == 0,
        format!("{total_records} hash-chained CTR1 records re-verified from disk; {integrity} submitted-vs-reconstructed program mismatches"),
    ));
    let consistent = check_samples(&out.join("visible").join("samples.cts"))
        && check_samples(&out.join("hidden").join("samples.cts"));
    gates.push(gate(
        "CRUMB_TRAINING_SAMPLE_PASS",
        total_samples > 0 && samples_for_solved && consistent,
        format!("{total_samples} CTS1 samples; every solved crumb produced samples; every sample's positives are a non-empty subset of the operations tried"),
    ));
    gates.push(gate(
        "NO_COT_STORAGE_PASS",
        label_hits == 0 && chains_ok,
        format!("fixed-size numeric records ({TRACE_RECORD_BYTES} B / {SAMPLE_RECORD_BYTES} B) with no text fields; {label_hits} label strings in the corpus"),
    ));

    gates.extend(promotion_and_reuse(learner, out));
    gates.push(scope_enforcement(learner));

    let _ = std::fs::write(
        out.join("gates.json"),
        serde_json::to_string_pretty(&gates).unwrap(),
    );
    let assertions: Vec<serde_json::Value> = gates
        .iter()
        .map(|g| {
            serde_json::json!({"id": g.id, "expected": "PASS", "observed": g.observed,
                               "result": if g.pass {"pass"} else {"fail"}, "source": "crumbs gates", "note": ""})
        })
        .collect();
    let _ = std::fs::write(
        out.join("aien-proof-assertions.json"),
        serde_json::to_string_pretty(&serde_json::json!({ "assertions": assertions })).unwrap(),
    );
    gates
}

fn count_records(path: &Path, class: ResultClass) -> usize {
    let tr = std::fs::read(path).unwrap_or_default();
    // byte 7 of a CTR1 record is its result class
    tr.chunks(TRACE_RECORD_BYTES)
        .filter(|r| r.len() == TRACE_RECORD_BYTES && r[7] == class as u8)
        .count()
}

/// Positive bits must be a non-empty subset of tried bits in every sample.
fn check_samples(path: &Path) -> bool {
    let b = std::fs::read(path).unwrap_or_default();
    b.chunks(SAMPLE_RECORD_BYTES).all(|r| {
        // body: magic 4, version 2, n_ops 2, digests 64, 4 bytes, values 64, targets 64 => 204
        let pos = &r[204..220];
        let tried = &r[220..236];
        pos.iter().zip(tried).all(|(p, t)| p & !t == 0) && pos.iter().any(|p| *p != 0)
    })
}

fn no_hidden_access(learner: &str, out: &Path) -> GateResult {
    // Canary crumbs: the visible examples come from a known chain, and the
    // held-outs are random 64-bit inputs the learner never sees. Neither those
    // inputs nor their outputs can appear in the learner's memory unless the
    // boundary leaks.
    let mut s = session(learner, "canary", false, out);
    let mut needles: Vec<Vec<u8>> = Vec::new();
    let mut rng = Rng::named("canary", &[1]);
    let mut accepted = 0u32;
    for seed in 1..=4u64 {
        let f = registry().by_key("chain-basic-l2-w16").unwrap();
        let mut g = gen::generate(f, seed);
        let KnownSolution::Known(p) = g.sealed.known_solution.clone() else {
            unreachable!()
        };
        let mut mk = || -> Vec<LaneExample> {
            (0..16)
                .map(|_| {
                    let x = rng.next_u64() | (1 << 63);
                    LaneExample {
                        input: vec![x],
                        output: vec![p.run(x)],
                    }
                })
                .collect()
        };
        g.sealed.heldouts.interpolation = mk();
        g.sealed.heldouts.extrapolation = mk();
        g.sealed.heldouts.adversarial = mk();
        for t in HeldoutTier::ALL {
            for ex in g.sealed.heldouts.tier(t) {
                for v in [ex.input[0], ex.output[0]] {
                    if v.to_le_bytes().iter().filter(|b| **b != 0).count() >= 6 {
                        needles.push(v.to_le_bytes().to_vec());
                        needles.push(v.to_string().into_bytes());
                    }
                }
            }
        }
        needles.push(g.sealed.digest().0.to_vec());
        needles.push(g.sealed.generator_instance_digest.0.to_vec());
        if let Ok(o) = s.present_with(&g.visible, &g.sealed) {
            accepted += o.accepted as u32;
        }
    }
    let pid = s.learner_pid().unwrap();
    let mem = read_process_writable_memory(pid).unwrap_or_default();
    let mem_hits = needles
        .iter()
        .filter(|n| crate::audit::contains(&mem, n))
        .count();
    let label_hits_mem = find_labels(&mem);
    let transcript = s.transcript().map(|t| t.to_vec()).unwrap_or_default();
    let tr_hits = needles
        .iter()
        .filter(|n| crate::audit::contains(&transcript, n))
        .count();
    s.finish().unwrap();
    let bin = std::fs::read(learner).unwrap_or_default();
    let bin_hits: Vec<String> = sealed_specific_labels()
        .into_iter()
        .filter(|l| crate::audit::contains(&bin, l.as_bytes()))
        .collect();
    gate(
        "CRUMB_NO_HIDDEN_ACCESS_PASS",
        !mem.is_empty() && mem_hits == 0 && label_hits_mem.is_empty() && tr_hits == 0 && bin_hits.is_empty() && accepted > 0,
        format!(
            "read {} KiB of the live learner's writable memory: {mem_hits} of {} canary held-out values/digests found; labels in memory {:?}; canary bytes in transcript {tr_hits}; sealed labels in learner binary {:?}; canary crumbs verified {accepted}/4",
            mem.len() >> 10,
            needles.len(),
            label_hits_mem,
            bin_hits
        ),
    )
}

fn promotion_and_reuse(learner: &str, out: &Path) -> Vec<GateResult> {
    // A short trail: two capabilities, each shown twice, then their composition.
    let script: [(&str, u64); 5] = [
        ("capability-03", 1),
        ("capability-05", 1),
        ("capability-03", 2),
        ("capability-05", 2),
        ("compose-03-05", 9),
    ];
    let mut runs = Vec::new();
    for (cond, lib) in [("control-trail", false), ("learning-trail", true)] {
        let mut s = session(learner, cond, lib, out);
        let outs: Vec<CrumbOutcome> = script
            .iter()
            .map(|(k, seed)| present(&mut s, k, *seed))
            .collect();
        let (_, _, ladder) = s.finish().unwrap();
        runs.push((outs, ladder));
    }
    let (learn_outs, ladder) = &runs[1];
    let admitted: Vec<_> = ladder
        .ops
        .values()
        .filter(|r| r.stage == Stage::Admitted)
        .collect();
    let evidence_ok = admitted.iter().all(|r| {
        r.distinct_crumbs() >= 2
            && r.supports
                .iter()
                .all(|s| s.adversarial_total > 0 && s.extrapolation_total > 0)
            && r.history
                .windows(2)
                .all(|w| w[0].0 < w[1].0 && w[0].1 <= w[1].1)
            && r.scope_bits > 0
    });
    let mut res = vec![gate(
        "CRUMB_OPERATION_PROMOTION_PASS",
        !admitted.is_empty() && evidence_ok,
        format!(
            "{} operations admitted, each with >= 2 distinct held-out-verified crumbs, adversarial and extrapolation evidence, ordered ladder history and a recorded scope; {} tracked candidates not admitted",
            admitted.len(),
            ladder.ops.len() - admitted.len()
        ),
    )];
    let (cl, cc) = (&learn_outs[4], &runs[0].0[4]);
    res.push(gate(
        "CRUMB_LIBRARY_REUSE_PASS",
        cl.accepted && cl.verified && cl.reused_library(),
        format!(
            "composition crumb: learning verified using admitted ops {:?} after {} candidates; control verified={} after {} candidates",
            cl.trace.library_refs_in_solution, cl.stats.candidates_to_solution, cc.verified, cc.stats.candidates_to_solution
        ),
    ));
    res
}

fn scope_enforcement(learner: &str) -> GateResult {
    let mut lp = LearnerProcess::spawn(learner, &[], false).expect("learner");
    lp.hello(
        &SearchConfig {
            max_candidates: 2000,
            ..SearchConfig::default()
        },
        true,
    )
    .unwrap();
    let op = Program::of(&[(Op::Mul, 3), (Op::Add, 2)]);
    lp.admit(1, 4, &op).unwrap(); // verified only for inputs below 2^4
    let small: Vec<(Vec<u64>, Vec<u64>)> =
        (1..9u64).map(|x| (vec![x], vec![op.run(x) * 2])).collect();
    let wide: Vec<(Vec<u64>, Vec<u64>)> = [3u64, 700, 40_000, 9, 12_345, 60_001]
        .iter()
        .map(|&x| (vec![x], vec![op.run(x) * 2]))
        .collect();
    let mut reject = |_: &Submission| Verdict::Reject;
    let vs = VisibleCrumb::from_values(Encoding::DecimalUtf8, 0, 0, &small, None).unwrap();
    let vw = VisibleCrumb::from_values(Encoding::DecimalUtf8, 0, 0, &wide, None).unwrap();
    let in_scope = lp.run_crumb(&vs, &mut reject).unwrap();
    let out_scope = lp.run_crumb(&vw, &mut reject).unwrap();
    let _ = lp.shutdown();
    let lib_in = in_scope.bank.iter().filter(|b| b.origin == 1).count();
    let lib_out = out_scope.bank.iter().filter(|b| b.origin == 1).count();
    gate(
        "CRUMB_SCOPE_ENFORCEMENT_PASS",
        lib_in == 1 && lib_out == 0,
        format!("operation admitted with a 4-bit scope: offered on a crumb with inputs below 2^4 ({lib_in}), withheld on a crumb with 16-bit inputs ({lib_out})"),
    )
}

pub fn main(args: &[String]) -> i32 {
    let get = |n: &str| {
        args.iter()
            .position(|a| a == n)
            .and_then(|i| args.get(i + 1).cloned())
    };
    let learner = get("--learner").expect("--learner PATH");
    let out = get("--out")
        .map(PathBuf::from)
        .unwrap_or_else(|| std::env::temp_dir().join("crumbs-gates"));
    let g = run_gates(&learner, &out);
    let passed = g.iter().filter(|x| x.pass).count();
    println!(
        "GATES {passed}/{} PASS  (evidence: {})",
        g.len(),
        out.display()
    );
    i32::from(passed != g.len())
}
