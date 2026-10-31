//! The 30B adapter on the tiny MoE (ADR 0004 step 3): the forward-pass contract, the moved
//! controller tests, the dual run against the old loop (steps 3 and 4 only; step 5 deletes it),
//! the mirror resync and the traces rule.

use super::super::test_support::*;
use super::super::*;
use super::UNSEEN;
use crate::generation::tests::{
    cold_on, forward_pass_contract, gen, gen_with, greedy, moved_failed_restore,
    moved_prefix_reuse, moved_restored_prefix, moved_stops_caps_and_cancels, prompt, sampled,
    LayerFill,
};
use crate::generation::{ForwardPass, GenerationController, Sampler};
use crate::tensor::sample_argmax;

const CTX: usize = 128;

type Ctl = GenerationController<Qwen3MoeOmegaEngine>;

/// The tiny MoE on the host pool (the cache that exports and imports), FP8 so batched prefill
/// runs.
fn tiny(cap: usize) -> Qwen3MoeOmegaEngine {
    let c = tiny_config();
    let mut src = MemorySource::for_config(&c);
    let up = RecordingUploader::default();
    let (w, _) =
        load_qwen3_moe_omega_from_with(&c, &mut src, &up, WeightFormat::Fp8All, &mut |_| {})
            .unwrap();
    Qwen3MoeOmegaEngine::new(
        w,
        Box::new(HostPoolKv::new(&KvGeometry::from(&c), cap).unwrap()),
    )
}

fn tiny_chunk3() -> Qwen3MoeOmegaEngine {
    let mut e = tiny(CTX);
    e.prefill_chunk = 3;
    e
}

impl LayerFill for Qwen3MoeOmegaEngine {
    /// The `KvStore` view shows the smallest per-layer fill, which is what I1 checks.
    fn layer_fill(&self) -> Vec<usize> {
        vec![self.kv.len()]
    }
}

#[test]
fn the_30b_adapter_keeps_the_forward_pass_contract() {
    forward_pass_contract(&|| tiny(CTX));
    forward_pass_contract(&tiny_chunk3);
}

#[test]
fn moved_tests_run_on_the_tiny_moe() {
    let o = moved_prefix_reuse(&|| tiny(CTX));
    // The warm turn prefilled only its 3 new tokens: as many launches as a cold 3-token prefill.
    let mut c = GenerationController::new(tiny(CTX));
    let (_, cold3) = gen(&mut c, &[5, 6, 7], &greedy(1), None).unwrap();
    assert_eq!(o.prefill_timing.launches, cold3.prefill_timing.launches);
    moved_stops_caps_and_cancels(&|| tiny(CTX));
    moved_restored_prefix(&|| tiny(CTX), &tiny_chunk3);
    moved_failed_restore(&|| tiny(CTX));
}

// ---------------------------------------------------------------------------------------------
// Dual run: the old loop (`generate`, `generate_with` on the engine) against the controller
// driving the same engine through the adapter.
// ---------------------------------------------------------------------------------------------

/// The outcome without wall-clock fields: milliseconds and every nanosecond counter zeroed; the
/// counts (`launches`, `graph_tokens`, `graph_launches`, `graph_submits`) kept.
fn counts(o: &GenerateOutcome) -> GenerateOutcome {
    let t = |s: &StepTiming| StepTiming {
        launches: s.launches,
        graph_tokens: s.graph_tokens,
        graph_launches: s.graph_launches,
        graph_submits: s.graph_submits,
        ..StepTiming::default()
    };
    GenerateOutcome {
        prefill_ms: 0,
        decode_ms: 0,
        decode_timing: t(&o.decode_timing),
        prefill_timing: t(&o.prefill_timing),
        ..o.clone()
    }
}

/// The door's way of sampling: always the full logits (a logits-wanting `sample_next`).
struct DoorSampler(GenerateParams);

impl Sampler for DoorSampler {
    fn wants_logits(&self) -> bool {
        true
    }
    fn pick(&mut self, logits: &[f32], step: usize) -> u32 {
        sample_next(logits, &self.0, step)
    }
}

#[derive(Clone, Copy, Debug)]
enum Path {
    /// `generate` / the controller with `SampleNext` (the argmax shortcut when greedy).
    Convenience,
    /// `generate_with` / the controller with a logits-wanting sampler (the door).
    Door,
}

fn old_turn(
    e: &mut Qwen3MoeOmegaEngine,
    p: &[u32],
    params: &GenerateParams,
    path: Path,
) -> (Vec<u32>, GenerateOutcome) {
    let mut out = Vec::new();
    let mut emit = |t| {
        out.push(t);
        true
    };
    let o = match path {
        Path::Convenience => e.generate(p, params, &mut emit),
        Path::Door => {
            let q = params.clone();
            e.generate_with(p, params, &mut |l, s| sample_next(l, &q, s), &mut emit)
        }
    }
    .unwrap();
    (out, o)
}

fn new_turn(
    c: &mut Ctl,
    p: &[u32],
    params: &GenerateParams,
    path: Path,
) -> (Vec<u32>, GenerateOutcome) {
    match path {
        Path::Convenience => gen(c, p, params, None).unwrap(),
        Path::Door => gen_with(c, p, params, &mut DoorSampler(params.clone()), None).unwrap(),
    }
}

/// Same tokens, same outcome counts, same ledger, the mirror equal to the ledger and the same
/// K/V bytes.
fn assert_same(
    what: &str,
    old: &mut Qwen3MoeOmegaEngine,
    new: &mut Ctl,
    a: (Vec<u32>, GenerateOutcome),
    b: (Vec<u32>, GenerateOutcome),
) {
    assert!(
        a.0.len() >= 8 || a.1.finish != FinishReason::Length,
        "{what}: a run of 8"
    );
    assert_eq!(a.0, b.0, "{what}: tokens");
    assert_eq!(counts(&a.1), counts(&b.1), "{what}: outcome");
    assert_eq!(old.history(), new.history(), "{what}: ledger");
    assert_eq!(
        new.model().history(),
        new.history(),
        "{what}: mirror history"
    );
    assert_eq!(
        new.model().position(),
        new.position(),
        "{what}: mirror position"
    );
    let n = new.position();
    assert_eq!(
        old.export_kv(0, n).unwrap(),
        new.export_kv(0, n).unwrap(),
        "{what}: K/V bytes"
    );
}

fn dual_run(make: &dyn Fn() -> Qwen3MoeOmegaEngine, label: &str) {
    for (pname, params) in [("greedy", greedy(8)), ("seeded", sampled(8, 5))] {
        for path in [Path::Convenience, Path::Door] {
            let what = format!("{label} {pname} {path:?}");
            let mut old = make();
            let mut new = GenerationController::new(make());
            // Cold.
            let p1 = prompt(12, 1);
            let a = old_turn(&mut old, &p1, &params, path);
            let b = new_turn(&mut new, &p1, &params, path);
            assert_eq!(b.1.reused_prompt_tokens, 0);
            assert_same(&format!("{what} cold"), &mut old, &mut new, a, b);
            // Warm: the thread resent with new tokens (prefix reuse).
            let mut p2 = new.history().to_vec();
            p2.extend([5, 6, 7]);
            let a = old_turn(&mut old, &p2, &params, path);
            let b = new_turn(&mut new, &p2, &params, path);
            assert_eq!(b.1.reused_prompt_tokens, p2.len() - 3);
            assert_same(&format!("{what} warm"), &mut old, &mut new, a, b);
            // A shorter prompt that diverges inside the history (a rewind).
            let mut p3 = new.history()[..9].to_vec();
            p3.push(2);
            let a = old_turn(&mut old, &p3, &params, path);
            let b = new_turn(&mut new, &p3, &params, path);
            assert_eq!(b.1.reused_prompt_tokens, 9);
            assert_same(&format!("{what} rewind"), &mut old, &mut new, a, b);
            // A stop id taken from the last reply: consumed, never emitted, in both loops.
            let stop = new.history()[new.position() - 2];
            let mut ps = params.clone();
            ps.stop_token_ids = vec![stop];
            let mut p5 = new.history().to_vec();
            p5.push(1);
            let a = old_turn(&mut old, &p5, &ps, path);
            let b = new_turn(&mut new, &p5, &ps, path);
            assert_same(&format!("{what} stop"), &mut old, &mut new, a, b);
            // Restored: the parked prefix into fresh engines, the old one through its own
            // restore_kv, the new one through the controller (the mirror resync case of I1).
            let h = new.history().to_vec();
            let bytes = new.export_kv(0, h.len()).unwrap();
            let mut old_r = make();
            old_r.restore_kv(&h, &bytes).unwrap();
            let mut new_r = GenerationController::new(make());
            new_r.restore_kv(&h, &bytes).unwrap();
            let mut p4 = h.clone();
            p4.extend([4, 2]);
            let a = old_turn(&mut old_r, &p4, &params, path);
            let b = new_turn(&mut new_r, &p4, &params, path);
            assert_eq!(b.1.reused_prompt_tokens, h.len());
            assert_same(&format!("{what} restored"), &mut old_r, &mut new_r, a, b);
        }
    }
}

#[test]
fn the_old_loop_and_the_controller_on_the_adapter_agree() {
    dual_run(&|| tiny(CTX), "chunk 8");
    dual_run(&tiny_chunk3, "chunk 3");
}

// ---------------------------------------------------------------------------------------------
// The mirror (I1) and the traces rule.
// ---------------------------------------------------------------------------------------------

#[test]
fn the_mirror_resyncs_after_a_restore_and_refuses_a_position_the_cache_does_not_hold() {
    let mut a: Ctl = GenerationController::new(tiny(CTX));
    gen(&mut a, &prompt(20, 3), &greedy(4), None).unwrap();
    let h = a.history().to_vec();
    let bytes = a.export_kv(0, h.len()).unwrap();

    // Through the controller: the restore empties the mirror (begin_sequence is the engine's
    // reset), the next forward call shows it the ledger and it takes the restored tokens.
    let mut b: Ctl = GenerationController::new(tiny(CTX));
    gen(&mut b, &[9, 8, 7], &greedy(2), None).unwrap();
    b.restore_kv(&h, &bytes).unwrap();
    assert_eq!((b.model().position(), b.model().history().len()), (0, 0));
    let mut p = h.clone();
    p.push(1);
    gen(&mut b, &p, &greedy(3), None).unwrap();
    assert_eq!(b.model().history(), b.history());
    assert_eq!(b.model().position(), b.position());
    assert_eq!(&b.history()[..h.len()], &h[..]);

    // Driven directly after an import it never saw, the mirror records unseen tokens, which the
    // old loop's prefix rule never reuses.
    let mut e = tiny(CTX);
    e.kv().import_blocks(0, h.len(), &bytes).unwrap().unwrap();
    let mut t = StepTiming::default();
    e.decode(h.len(), &[1], &mut t).unwrap();
    assert_eq!(e.position(), h.len() + 1);
    assert!(e.history()[..h.len()].iter().all(|&x| x == UNSEEN));
    assert_eq!(e.history()[h.len()], 1);
    assert_eq!(e.reusable_prefix(&p), 0);

    // A position the cache does not hold is refused, whatever the mirror says (I1).
    let mut f = tiny(CTX);
    f.prefill(0, &prompt(10, 4), &mut t).unwrap();
    f.kv().truncate(6);
    assert!(f.decode(10, &[1], &mut t).is_err(), "the cache holds 6");
    assert!(f.decode(7, &[1], &mut t).is_err(), "a gap");
    f.decode(6, &[1], &mut t).unwrap();
    assert_eq!((f.position(), f.history().len()), (7, 7));

    // A different sequence of the same length imported over the mirror's own: without a
    // ledger the mirror no longer claims its old tokens; with one it takes the ledger's.
    let other = prompt(h.len(), 9);
    let mut g = tiny(CTX);
    g.prefill(0, &other, &mut t).unwrap();
    g.kv().truncate(0);
    g.kv().import_blocks(0, h.len(), &bytes).unwrap().unwrap();
    g.decode(h.len(), &[1], &mut t).unwrap();
    assert!(g.history()[..h.len()].iter().all(|&x| x == UNSEEN));
    let mut g = tiny(CTX);
    g.prefill(0, &other, &mut t).unwrap();
    g.kv().truncate(0);
    g.kv().import_blocks(0, h.len(), &bytes).unwrap().unwrap();
    g.mirror_ledger(&h);
    g.decode(h.len(), &[1], &mut t).unwrap();
    assert_eq!(&g.history()[..h.len()], &h[..]);
}

#[test]
fn reset_clears_traces_and_rewind_keeps_them() {
    let traced = || {
        let mut e = tiny(CTX);
        e.trace_layers = vec![0, 1];
        e
    };
    let mut old = traced();
    let mut new: Ctl = GenerationController::new(traced());
    let p1 = prompt(10, 6);
    old_turn(&mut old, &p1, &greedy(3), Path::Convenience);
    gen(&mut new, &p1, &greedy(3), None).unwrap();
    let t1 = new.model().traces.clone();
    assert!(!t1.is_empty());
    assert_eq!(old.traces, t1);
    // Prefix reuse rewinds: the traces of turn 1 stay, turn 2's are added after them.
    let mut p2 = new.history().to_vec();
    p2.push(4);
    old_turn(&mut old, &p2, &greedy(3), Path::Convenience);
    gen(&mut new, &p2, &greedy(3), None).unwrap();
    let t2 = &new.model().traces;
    assert!(t2.len() > t1.len());
    assert_eq!(&t2[..t1.len()], &t1[..], "a rewind keeps traces");
    assert_eq!(&old.traces, t2);
    // A cold turn resets: only its own traces remain.
    old_turn(&mut old, &[1, 2], &greedy(2), Path::Convenience);
    gen(&mut new, &[1, 2], &greedy(2), None).unwrap();
    assert_eq!(old.traces, new.model().traces);
    assert!(new.model().traces.iter().all(|t| t.pos < 3));
    new.reset();
    assert!(new.model().traces.is_empty(), "reset clears traces");
}

#[test]
fn kv_identity_names_every_switch_and_mutants_and_tracing_refuse() {
    let base = tiny(32).kv_identity().unwrap();
    assert_eq!(base.kind, "qwen3-moe-omega");
    let names: Vec<_> = base.entries.iter().map(|(k, _)| *k).collect();
    let mut unique = names.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), names.len(), "entry names are unique");
    let changed = |f: &dyn Fn(&mut Qwen3MoeOmegaEngine)| {
        let mut e = tiny(32);
        f(&mut e);
        let id = e.kv_identity().unwrap();
        assert_ne!(id, base);
        id.entries
            .iter()
            .zip(&base.entries)
            .filter(|(a, b)| a != b)
            .map(|(a, _)| a.0)
            .collect::<Vec<_>>()
    };
    assert_eq!(changed(&|e| e.prefill_chunk = 3), ["prefill_chunk"]);
    assert_eq!(changed(&|e| e.prefill_gemm ^= true), ["prefill_gemm"]);
    assert_eq!(
        changed(&|e| e.prefill_device_routing ^= true),
        ["prefill_device_routing"]
    );
    assert_eq!(changed(&|e| e.device_routing ^= true), ["device_routing"]);
    assert_eq!(changed(&|e| e.decode_graph ^= true), ["decode_graph"]);
    let c = tiny_config();
    assert_eq!(
        changed(&|e| e.kv = Box::new(HostKv::new(&KvGeometry::from(&c), 32))),
        ["attention_path"]
    );
    type ConfigFlip = dyn Fn(&mut Qwen3MoeConfig);
    // Every config field that reaches a K/V byte moves its own entry; the stop ids do not.
    let cfg: [(&str, &ConfigFlip); 13] = [
        ("layers", &|c| c.layers += 1),
        ("hidden", &|c| c.hidden += 1),
        ("q_heads", &|c| c.q_heads += 1),
        ("kv_heads", &|c| c.kv_heads += 1),
        ("head_dim", &|c| c.head_dim += 1),
        ("experts", &|c| c.experts += 1),
        ("top_k", &|c| c.top_k += 1),
        ("moe_intermediate", &|c| c.moe_intermediate += 1),
        ("vocab", &|c| c.vocab += 1),
        ("rms_eps", &|c| c.rms_eps *= 2.0),
        ("rope_theta", &|c| c.rope_theta *= 2.0),
        ("norm_topk_prob", &|c| c.norm_topk_prob ^= true),
        ("tie_word_embeddings", &|c| c.tie_word_embeddings ^= true),
    ];
    for (name, f) in cfg {
        assert_eq!(changed(&|e| f(&mut e.weights.config)), [name]);
    }
    let mut s = tiny(32);
    s.weights.config.eos_token_ids.push(7);
    assert_eq!(
        s.kv_identity().unwrap(),
        base,
        "stop ids change no K/V byte"
    );
    let mut m = tiny(32);
    m.routing_mutant = true;
    assert!(m.kv_identity().is_err(), "the routing mutant refuses");
    let mut t = tiny(32);
    t.trace_layers = vec![0];
    assert!(t.kv_identity().is_err(), "tracing refuses");
}

/// The greedy argmax the controller takes on the host head is the old loop's `sample_next`.
#[test]
fn greedy_on_the_host_head_is_the_argmax_of_its_logits() {
    let mut e = tiny(CTX);
    let mut t = StepTiming::default();
    let p = prompt(12, 2);
    e.prefill(0, &p, &mut t).unwrap();
    let mut t = StepTiming::default();
    let l = match e.next(0, crate::generation::Want::Greedy, &mut t).unwrap() {
        crate::generation::Next::Logits(l) => l.to_vec(),
        crate::generation::Next::Token(_) => panic!("no graph on the host"),
    };
    assert_eq!(sample_argmax(&l).0, sample_next(&l, &greedy(1), 0));
    assert_eq!(t.launches, 1, "the host lm_head counts one launch");
    let cold = cold_on(&|| tiny(CTX), &p, &greedy(8));
    assert_eq!(cold[0], sample_argmax(&l).0);
}
