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

//! The guardrail end to end on the host fake `EchoChainModel` (ADR 0004 section 4) with the
//! 64-token synthetic table: exact walks, refusal through the rollback, the no-constraint path
//! byte for byte, and (with `OMEGESE_BIN`) random walks against the real oracle.

mod common;

use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use aien_inference_abi::generation::echo_chain::EchoChainModel;
use aien_inference_abi::generation::{
    Constraint, ForwardPass, GenerationController, SampleNext, Sampler, Want,
};
use aien_inference_abi::qwen3_moe_omega::{
    FinishReason, GenerateOutcome, GenerateParams, StepTiming,
};
use aien_omegese_constraint::{
    Answer, Framing, ItemOracle, OmegeseConstraint, Shape, SubprocessOracle, TokenTable,
};
use common::*;

type Ctl = GenerationController<EchoChainModel>;

fn ctl(ctx: usize, graph: bool) -> Ctl {
    let mut m = EchoChainModel::new(ctx);
    m.graph_mode = graph;
    GenerationController::new(m)
}

fn prompt(n: usize, seed: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 13 + 1) % 63) as u32)
        .collect()
}

fn greedy(n: usize) -> GenerateParams {
    GenerateParams::greedy(n, vec![STOP])
}

fn sampled(n: usize, seed: u64) -> GenerateParams {
    GenerateParams {
        max_new_tokens: n,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed,
        stop_token_ids: vec![STOP],
    }
}

fn run(
    c: &mut Ctl,
    p: &[u32],
    params: &GenerateParams,
    sampler: &mut dyn Sampler,
    constraint: Option<&mut dyn Constraint>,
) -> Result<(Vec<u32>, GenerateOutcome), String> {
    let mut out = Vec::new();
    let o = c.generate(p, params, sampler, constraint, &mut |t| {
        out.push(t);
        true
    })?;
    Ok((out, o))
}

fn run_sn(
    c: &mut Ctl,
    p: &[u32],
    params: &GenerateParams,
    constraint: Option<&mut dyn Constraint>,
) -> Result<(Vec<u32>, GenerateOutcome), String> {
    run(c, p, params, &mut SampleNext(params.clone()), constraint)
}

fn counted(o: &GenerateOutcome) -> GenerateOutcome {
    GenerateOutcome {
        prefill_ms: 0,
        decode_ms: 0,
        ..o.clone()
    }
}

fn full() -> Framing {
    Framing {
        open_fence: Some("```\n".into()),
        header: Some(sig_f()),
        mask_body: true,
        close_fence: "```".into(),
        stop_token: STOP,
    }
}

const TARGET: &str = "fn f a uint b uint to uint\nret add a b\nend\n";

/// Every walk under the exact one-program oracle writes the fence, the program (up to blank space)
/// and the closing fence, whatever tokens the model prefers, and stops on the stop token.
#[test]
fn exact_walks_write_the_one_program() {
    let table = Arc::new(synthetic_table());
    let mut multi_item_picked = 0;
    let mut multi_refused = 0;
    for seed in 0..12 {
        for params in [greedy(64), sampled(64, seed as u64 + 1)] {
            let mut c = ctl(256, seed % 2 == 0);
            let mut k =
                OmegeseConstraint::new(table.clone(), ExactText::new(TARGET), full()).unwrap();
            let (toks, o) = run_sn(&mut c, &prompt(9, seed), &params, Some(&mut k)).unwrap();
            assert_eq!(o.finish, FinishReason::Stop, "seed {seed}");
            let text = table.decode(&toks).unwrap();
            let program = k.program().to_vec();
            assert_eq!(normalize(&program), TARGET.as_bytes(), "seed {seed}");
            let mut want = b"```\n".to_vec();
            want.extend(&program);
            if !program.ends_with(b"\n") {
                want.push(b'\n');
            }
            want.extend(b"```");
            assert_eq!(text, want, "seed {seed}");
            assert!(k.finished());
            multi_item_picked += toks
                .iter()
                .filter(|&&t| {
                    matches!(table.shape(t), Shape::OneBoundary { .. })
                        && table.text(t).unwrap().len() > 1
                })
                .count();
            multi_refused += k.stats().multi_boundary_refused;
        }
    }
    assert!(multi_item_picked > 0, "some walk used a multi-item token");
    assert!(multi_refused > 0, "` a ` is refused where ` a` is legal");
}

/// The masked arm with no fence: the header is forced, the body masked, and eof is the stop token
/// itself, so the reply is the program and the run ends on stop.
#[test]
fn the_unfenced_arm_ends_on_the_stop_token() {
    let table = Arc::new(synthetic_table());
    for seed in 0..6 {
        for params in [greedy(64), sampled(64, seed as u64 + 31)] {
            let arm = Framing {
                open_fence: None,
                ..full()
            };
            let mut k = OmegeseConstraint::new(table.clone(), ExactText::new(TARGET), arm).unwrap();
            let (toks, o) = run_sn(
                &mut ctl(256, seed % 2 == 1),
                &prompt(9, seed),
                &params,
                Some(&mut k),
            )
            .unwrap();
            assert_eq!(o.finish, FinishReason::Stop, "seed {seed}");
            let program = k.program().to_vec();
            assert_eq!(normalize(&program), TARGET.as_bytes(), "seed {seed}");
            assert_eq!(
                table.decode(&toks).unwrap(),
                program,
                "the reply is the program"
            );
            // Completion here is `FinishReason::Stop` (asserted above): the stop was allowed among
            // blank tokens, not forced, so `finished()` cannot know it was taken.
            assert!(!k.finished());
        }
    }
}

/// The fake oracle answers REFUSED (or breaks its contract) from its `n`th call on.
struct FailAt<O> {
    inner: O,
    n: usize,
    calls: usize,
    broken: bool,
}

impl<O: ItemOracle> ItemOracle for FailAt<O> {
    fn next(&mut self, p: &[u8]) -> Result<Answer, String> {
        self.calls += 1;
        if self.calls >= self.n {
            return if self.broken {
                Err("oracle exit 0 with no item".into())
            } else {
                Ok(Answer::Refused("REFUSED omegese: line 2: injected".into()))
            };
        }
        self.inner.next(p)
    }
}

/// A REFUSED answer ends the generation with `Err` after the controller's rollback: the ledger
/// keeps the tokens emitted before, every layer holds exactly the ledger, no head is pending, and
/// a warm run then equals a cold one.
#[test]
fn an_oracle_refusal_goes_through_the_rollback() {
    let table = Arc::new(synthetic_table());
    let p = prompt(10, 3);
    for broken in [false, true] {
        // Call 1 is the header check when the constraint is built; call 2 is the first step.
        for n in [2, 5] {
            let mut c = ctl(256, true);
            let oracle = FailAt {
                inner: ExactText::new(TARGET),
                n,
                calls: 0,
                broken,
            };
            let mut k = OmegeseConstraint::new(table.clone(), oracle, full()).unwrap();
            let mut emitted = Vec::new();
            let r = c.generate(
                &p,
                &greedy(64),
                &mut SampleNext(greedy(64)),
                Some(&mut k),
                &mut |t| {
                    emitted.push(t);
                    true
                },
            );
            let e = r.expect_err("the refusal is an error");
            assert!(
                e.contains(if broken { "no item" } else { "injected" }),
                "{e}"
            );
            assert!(
                emitted.len() >= 2,
                "the fence was forced before the oracle ran"
            );
            assert_eq!(&c.history()[p.len()..], &emitted[..], "{n} {broken}");
            let pos = c.position();
            assert_eq!(
                c.model().layer_lens(),
                vec![pos; 2],
                "every layer at the ledger"
            );
            assert!(
                c.model_mut()
                    .next(0, Want::Logits, &mut StepTiming::default())
                    .is_err(),
                "the rollback forgets the pending head"
            );
            let mut q = c.history().to_vec();
            q.push(1);
            let (warm, _) = run_sn(&mut c, &q, &greedy(12), None).unwrap();
            let (cold, _) = run_sn(&mut ctl(256, true), &q, &greedy(12), None).unwrap();
            assert_eq!(warm, cold);
        }
    }
}

/// With every switch off the constraint answers `Any` on every step and the run is byte for byte
/// the run with no constraint (tokens, outcome, call log, head reads), with no oracle call.
#[test]
fn every_switch_off_is_byte_identical_to_no_constraint() {
    let table = Arc::new(synthetic_table());
    let p = prompt(12, 19);
    for graph in [false, true] {
        for params in [
            GenerateParams::greedy(10, vec![]),
            GenerateParams {
                stop_token_ids: vec![],
                ..sampled(10, 21)
            },
        ] {
            let mut a = ctl(128, graph);
            let (ta, oa) = run_sn(&mut a, &p, &params, None).unwrap();
            let mut b = ctl(128, graph);
            let off = Framing {
                open_fence: None,
                header: None,
                mask_body: false,
                close_fence: "```".into(),
                stop_token: STOP,
            };
            let mut k = OmegeseConstraint::new(table.clone(), ExactText::new(TARGET), off).unwrap();
            let (tb, ob) = run_sn(&mut b, &p, &params, Some(&mut k)).unwrap();
            assert_eq!(ta, tb, "same tokens");
            assert_eq!(counted(&oa), counted(&ob), "same outcome and counters");
            assert_eq!(a.model().calls(), b.model().calls(), "same call log");
            let (ma, mb) = (a.model(), b.model());
            assert_eq!(
                (ma.chip_reads, ma.lm_heads, ma.rows_reads),
                (mb.chip_reads, mb.lm_heads, mb.rows_reads)
            );
            assert_eq!(mb.rows_reads, 0, "next_rows is never called");
            assert_eq!(k.oracle().calls, 0, "no oracle call");
            assert_eq!(k.stats().forced_steps, 0);
        }
    }
}

/// The unmasked arm: only the fence is forced; after it the model runs free, exactly as a run
/// whose prompt ended with the fence, and the oracle is never asked.
#[test]
fn the_fence_only_arm_forces_the_fence_and_nothing_else() {
    let table = Arc::new(synthetic_table());
    let fence = table.encode("```\n").unwrap();
    let p = prompt(12, 5);
    for params in [
        GenerateParams::greedy(12, vec![]),
        GenerateParams {
            stop_token_ids: vec![],
            ..sampled(12, 9)
        },
    ] {
        let arm = Framing {
            open_fence: Some("```\n".into()),
            header: None,
            mask_body: false,
            close_fence: "```".into(),
            stop_token: STOP,
        };
        let mut k = OmegeseConstraint::new(table.clone(), ExactText::new(TARGET), arm).unwrap();
        let (t, _) = run_sn(&mut ctl(128, true), &p, &params, Some(&mut k)).unwrap();
        assert_eq!(&t[..fence.len()], &fence[..]);
        assert_eq!(k.oracle().calls, 0);
        let mut q = p.clone();
        q.extend(&fence);
        if params.temperature == 0.0 {
            let (free, _) = run_sn(
                &mut ctl(128, true),
                &q,
                &GenerateParams::greedy(12 - fence.len(), vec![]),
                None,
            )
            .unwrap();
            assert_eq!(&t[fence.len()..], &free[..], "the rest is the free run");
        }
    }
}

/// A seeded random walk among the allowed tokens, weighted by the fake's logits. From step
/// `steer_after` on it steers home: it takes the longest allowed token that starts what
/// `omegese complete` would write after the reply so far (shared through `reply`), or the closing
/// fence when the program is already whole. The constraint alone decides what is allowed.
struct Walk {
    seed: u64,
    table: Arc<TokenTable>,
    reply: Rc<RefCell<Vec<u32>>>,
    bin: PathBuf,
    steer_after: Option<usize>,
    plan: Vec<u8>,
}

fn mix(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    x = (x ^ (x >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

impl Walk {
    fn steer(&mut self, allowed: &[u32]) -> Option<u32> {
        let text = self.table.decode(&self.reply.borrow()).ok()?;
        let program = text.strip_prefix(b"```\n".as_slice())?;
        // Keep one plan (the prefix plus `omegese complete`'s text) while the walk follows it:
        // asking again after every token can loop (a name's first letters are a keyword).
        if !self.plan.starts_with(program) {
            self.plan = program.to_vec();
            self.plan.extend(complete(&self.bin, program).ok()?);
        }
        let rest = &self.plan[program.len()..];
        allowed
            .iter()
            .copied()
            .filter(|&id| {
                let t = self.table.text(id).unwrap_or_default();
                if rest.is_empty() {
                    t.contains(&b'`')
                } else {
                    !t.is_empty() && rest.starts_with(t)
                }
            })
            .max_by_key(|&id| self.table.text(id).map_or(0, <[u8]>::len))
    }
}

impl Sampler for Walk {
    fn wants_logits(&self) -> bool {
        true
    }
    fn pick(&mut self, logits: &[f32], step: usize) -> u32 {
        let allowed: Vec<u32> = (0..logits.len() as u32)
            .filter(|&i| logits[i as usize].is_finite())
            .collect();
        if self.steer_after.is_some_and(|s| step >= s) {
            if let Some(t) = self.steer(&allowed) {
                return t;
            }
        }
        let w: Vec<f32> = allowed
            .iter()
            .map(|&i| (3.0 * logits[i as usize]).exp())
            .collect();
        let total: f32 = w.iter().sum();
        let r = (mix(self.seed ^ mix(step as u64)) >> 11) as f32 / (1u64 << 53) as f32 * total;
        let mut acc = 0.0;
        for (&i, &x) in allowed.iter().zip(&w) {
            acc += x;
            if acc >= r {
                return i;
            }
        }
        *allowed.last().unwrap_or(&0)
    }
}

/// With the real oracle (`OMEGESE_BIN`): every guardrail-on walk is sound. A walk that ended on
/// `eof` wrote a program `omegese emit` accepts, inside the fences; a walk cut by length wrote a
/// prefix `omegese complete` finishes into one `emit` accepts.
#[test]
#[ignore = "needs OMEGESE_BIN (the omegese binary built from omega main)"]
fn real_oracle_walks_write_programs_emit_accepts() {
    let bin = omegese_bin();
    let table = Arc::new(synthetic_table());
    let (mut ended, mut cut, mut steps, mut calls, mut multi) = (0, 0, 0u64, 0u64, 0u64);
    let mut times = Vec::new();
    for seed in 0..40u64 {
        let mut c = ctl(512, seed % 2 == 0);
        let oracle = SubprocessOracle::new(&bin).unwrap();
        let mut k = OmegeseConstraint::new(table.clone(), oracle, full()).unwrap();
        let reply = Rc::new(RefCell::new(Vec::new()));
        // Every eighth walk never steers, so some walks are cut by length.
        let steer_after = (seed % 8 != 7).then_some(20 + (seed as usize % 5) * 30);
        let mut s = Walk {
            seed,
            table: table.clone(),
            reply: reply.clone(),
            bin: bin.clone(),
            steer_after,
            plan: Vec::new(),
        };
        let params = sampled(300, seed + 1);
        let p = prompt(8, seed as usize);
        let o = c
            .generate(&p, &params, &mut s, Some(&mut k), &mut |t| {
                reply.borrow_mut().push(t);
                true
            })
            .unwrap_or_else(|e| panic!("walk {seed}: {e}"));
        let toks = reply.borrow().clone();
        let text = table.decode(&toks).unwrap();
        let program = k.program().to_vec();
        match o.finish {
            FinishReason::Stop => {
                ended += 1;
                emit(&bin, &program).unwrap_or_else(|e| {
                    panic!(
                        "walk {seed}: emit refused {:?}: {e}",
                        String::from_utf8_lossy(&program)
                    )
                });
                let mut want = b"```\n".to_vec();
                want.extend(&program);
                if !program.ends_with(b"\n") {
                    want.push(b'\n');
                }
                want.extend(b"```");
                assert_eq!(text, want, "walk {seed}: the fences frame the program");
            }
            FinishReason::Length => {
                cut += 1;
                let rest = complete(&bin, &program).unwrap_or_else(|e| {
                    panic!(
                        "walk {seed}: no completion for {:?}: {e}",
                        String::from_utf8_lossy(&program)
                    )
                });
                let mut whole = program.clone();
                whole.extend(&rest);
                emit(&bin, &whole)
                    .unwrap_or_else(|e| panic!("walk {seed}: completed, refused: {e}"));
            }
            f => panic!("walk {seed}: finish {f:?}"),
        }
        let st = k.stats();
        steps += st.masked_steps;
        calls += st.oracle_calls;
        multi += st.multi_boundary_refused;
        times.extend(st.step_times.iter().copied());
    }
    times.sort();
    let us = |d: std::time::Duration| d.as_micros();
    println!(
        "walks: {ended} ended on eof (emit accepted), {cut} cut by length (completed and accepted); \
         {steps} masked steps, {calls} oracle calls ({:.2} per step), {multi} two-boundary refusals; \
         step time median {} us, p95 {} us, max {} us",
        calls as f64 / steps as f64,
        us(times[times.len() / 2]),
        us(times[times.len() * 95 / 100]),
        us(times[times.len() - 1]),
    );
    assert!(
        ended >= 20 && cut >= 3,
        "walks of both kinds ({ended} ended, {cut} cut)"
    );
}

/// The mask reads every blank run in its canonical form (`canonical_seps`): one space, or one
/// newline when the run holds one. With the real oracle: at every cut of three programs, the
/// text after a blank run and after its canonical form gets the same items (or both are
/// refused), with brackets around and between the blanks too.
#[test]
#[ignore = "needs OMEGESE_BIN (the omegese binary built from omega main)"]
fn real_oracle_blank_runs_are_interchangeable() {
    let mut oracle = SubprocessOracle::new(omegese_bin()).unwrap();
    let programs = [
        TARGET,
        "fn gcd a uint b uint to uint\nwhile ne b 0 bound 100\nt = rem a b\na = b\nb = t\nend\nret a\nend\n",
        "fn g v view uint 4 to uint\ns = uint 0\nfor i uint 4\ns = add s (get v i)\nend\nret s\nend\n",
    ];
    let runs = [
        "  ", "\t", " \t ", "\n\n", " \n", "\n ", "\n\t\n  ", "  (\t ", "(  ", ")\n\n", ") \n ",
        "\t)\t", "  ((", "(\n\n", "()  ",
    ];
    let (mut compared, mut items_both) = (0, 0);
    for prog in programs {
        for cut in 0..=prog.len() {
            let p = &prog.as_bytes()[..cut];
            for run in runs {
                let canon = aien_omegese_constraint::constraint::canonical_seps(run.as_bytes());
                assert_ne!(canon, run.as_bytes(), "{run:?} is not already canonical");
                let ask = |o: &mut SubprocessOracle, s: &[u8]| {
                    let mut q = p.to_vec();
                    q.extend_from_slice(s);
                    o.next(&q).unwrap()
                };
                let (x, y) = (ask(&mut oracle, run.as_bytes()), ask(&mut oracle, &canon));
                match (&x, &y) {
                    (Answer::Items(a), Answer::Items(b)) => {
                        assert_eq!(a, b, "after {:?} + {run:?}", String::from_utf8_lossy(p));
                        items_both += 1;
                    }
                    (Answer::Refused(_), Answer::Refused(_)) => {}
                    _ => panic!(
                        "after {:?}: {run:?} gives {x:?}, {:?} gives {y:?}",
                        String::from_utf8_lossy(p),
                        String::from_utf8_lossy(&canon)
                    ),
                }
                compared += 1;
            }
        }
    }
    println!("blank runs: {compared} pairs compared, {items_both} with items on both sides");
    assert!(items_both > 100);
}
