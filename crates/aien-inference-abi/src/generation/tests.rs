//! ADR 0004 section 5: the forward-pass contract suite and the generation controller's tests,
//! run on the host fake `EchoChainModel`. Every sharpness test compares a run of at least 8
//! tokens (a chance agreement of 8 tokens is about 1 in 64^8).

use super::echo_chain::{echo_geometry, Call, CorruptRowKv, EchoChainModel, EchoKv, ECHO_VOCAB};
use super::*;
use crate::qwen3_moe_omega::{HostPoolKv, KvAttention, KvGeometry};

const CTX: usize = 128;

type Ctl = GenerationController<EchoChainModel>;

fn ctl(graph: bool) -> Ctl {
    let mut m = EchoChainModel::new(CTX);
    m.graph_mode = graph;
    GenerationController::new(m)
}

pub(crate) fn prompt(n: usize, seed: usize) -> Vec<u32> {
    (0..n)
        .map(|i| ((i * 7 + seed * 13 + 1) % 64) as u32)
        .collect()
}

pub(crate) fn greedy(n: usize) -> GenerateParams {
    GenerateParams::greedy(n, Vec::new())
}

pub(crate) fn sampled(n: usize, seed: u64) -> GenerateParams {
    GenerateParams {
        max_new_tokens: n,
        temperature: 1.0,
        top_p: 1.0,
        top_k: 0,
        seed,
        stop_token_ids: Vec::new(),
    }
}

pub(crate) fn gen_with<M: ForwardPass>(
    c: &mut GenerationController<M>,
    prompt: &[u32],
    params: &GenerateParams,
    sampler: &mut dyn Sampler,
    constraint: Option<&mut dyn Constraint>,
) -> Result<(Vec<u32>, GenerateOutcome), String> {
    let mut out = Vec::new();
    let o = c.generate(prompt, params, sampler, constraint, &mut |t| {
        out.push(t);
        true
    })?;
    Ok((out, o))
}

pub(crate) fn gen<M: ForwardPass>(
    c: &mut GenerationController<M>,
    prompt: &[u32],
    params: &GenerateParams,
    constraint: Option<&mut dyn Constraint>,
) -> Result<(Vec<u32>, GenerateOutcome), String> {
    let mut s = SampleNext(params.clone());
    gen_with(c, prompt, params, &mut s, constraint)
}

/// A cold run on a fresh fake.
fn cold(prompt: &[u32], params: &GenerateParams) -> Vec<u32> {
    gen(&mut ctl(false), prompt, params, None).unwrap().0
}

/// What the moved tests read beyond the seam: how many positions each layer of the cache holds
/// (as many layers as the model can show).
pub(crate) trait LayerFill {
    fn layer_fill(&self) -> Vec<usize>;
}

impl LayerFill for EchoChainModel {
    fn layer_fill(&self) -> Vec<usize> {
        self.layer_lens()
    }
}

/// Every layer holds exactly the ledger's positions.
pub(crate) fn assert_at_ledger<M: ForwardPass + LayerFill>(c: &GenerationController<M>) {
    let n = c.position();
    let fill = c.model().layer_fill();
    assert!(!fill.is_empty());
    assert_eq!(
        fill,
        vec![n; fill.len()],
        "every layer must hold exactly position() = {n} rows"
    );
}

/// The outcome without its wall-clock fields.
pub(crate) fn counted(o: &GenerateOutcome) -> GenerateOutcome {
    GenerateOutcome {
        prefill_ms: 0,
        decode_ms: 0,
        ..o.clone()
    }
}

fn decodes(calls: &[Call]) -> usize {
    calls
        .iter()
        .filter(|c| matches!(c, Call::Decode { .. }))
        .count()
}

/// Rules in order, then `then` forever. Records what it was shown.
struct Script {
    rules: Vec<Result<StepRule, String>>,
    then: StepRule,
    seen: Vec<Vec<u32>>,
}

impl Script {
    fn new(rules: Vec<Result<StepRule, String>>, then: StepRule) -> Self {
        Self {
            rules,
            then,
            seen: Vec::new(),
        }
    }
    fn always(rule: StepRule) -> Self {
        Self::new(Vec::new(), rule)
    }
}

impl Constraint for Script {
    fn rule(&mut self, _prompt: &[u32], generated: &[u32]) -> Result<StepRule, String> {
        let i = self.seen.len();
        self.seen.push(generated.to_vec());
        self.rules
            .get(i)
            .cloned()
            .unwrap_or_else(|| Ok(self.then.clone()))
    }
}

/// A logits-wanting argmax that records the step counter it is given.
#[derive(Default)]
struct StepRecorder(Vec<usize>);

impl Sampler for StepRecorder {
    fn wants_logits(&self) -> bool {
        true
    }
    fn pick(&mut self, logits: &[f32], step: usize) -> u32 {
        self.0.push(step);
        sample_argmax(logits).0
    }
}

// ---------------------------------------------------------------------------------------------
// The contract suite (I1 to I5) for any forward pass.
// ---------------------------------------------------------------------------------------------

fn read_logits<M: ForwardPass>(m: &mut M, head: usize) -> Vec<f32> {
    match m.next(head, Want::Logits, &mut StepTiming::default()) {
        Ok(Next::Logits(l)) => l.to_vec(),
        other => panic!("Want::Logits must give logits, got {other:?}"),
    }
}

/// From a model whose head 0 predicts the token at `pos`: decodes `n` greedy tokens starting
/// with `first`, returning every step's logits.
fn greedy_chain<M: ForwardPass>(m: &mut M, pos: usize, first: u32, n: usize) -> Vec<Vec<f32>> {
    let mut t = StepTiming::default();
    let mut tok = first;
    let mut out = Vec::new();
    for i in 0..n {
        m.decode(pos + i, &[tok], &mut t).unwrap();
        let l = read_logits(m, 0);
        tok = sample_argmax(&l).0;
        out.push(l);
    }
    out
}

pub(crate) fn forward_pass_contract<M: ForwardPass>(make: &dyn Fn() -> M) {
    let p = prompt(20, 1); // more than one KV block
    let n = p.len();
    let mut t = StepTiming::default();

    // prefill of n tokens leaves exactly one head; a head is read once (I2).
    let mut a = make();
    a.prefill(0, &p, &mut t).unwrap();
    assert!(
        a.next(1, Want::Logits, &mut t).is_err(),
        "prefill leaves head 0 only"
    );
    let la = read_logits(&mut a, 0);
    assert_eq!(la.len(), a.vocab());
    assert!(
        a.next(0, Want::Logits, &mut t).is_err(),
        "a head is read once (I2)"
    );

    // Want::Greedy gives Token only as the argmax of that head's logits (I4): on the prefill
    // head and on a decode head (the graph's held token).
    let first = sample_argmax(&la).0;
    let mut b = make();
    b.prefill(0, &p, &mut t).unwrap();
    match b.next(0, Want::Greedy, &mut t).unwrap() {
        Next::Token(x) => assert_eq!(x, first),
        Next::Logits(l) => assert_eq!(l, &la[..]),
    }
    a.decode(n, &[first], &mut t).unwrap();
    b.decode(n, &[first], &mut t).unwrap();
    let la2 = read_logits(&mut a, 0);
    match b.next(0, Want::Greedy, &mut t).unwrap() {
        Next::Token(x) => assert_eq!(x, sample_argmax(&la2).0),
        Next::Logits(l) => assert_eq!(l, &la2[..]),
    }

    // A head read after a truncate that drops a row errors (I2).
    let mut c = make();
    c.prefill(0, &p, &mut t).unwrap();
    c.kv().truncate(n - 1);
    assert!(
        c.next(0, Want::Logits, &mut t).is_err(),
        "no head survives a truncate"
    );

    // A head read after a new forward call gives the new call's head (I2): an unread decode
    // head (on a graph, a held token) is replaced by the next prefill's head.
    let mut g = make();
    g.prefill(0, &p, &mut t).unwrap();
    let _ = read_logits(&mut g, 0);
    g.decode(n, &[first], &mut t).unwrap();
    let _ = read_logits(&mut g, 0);
    g.prefill(n + 1, &[7, 9], &mut t).unwrap();
    let lg = read_logits(&mut g, 0);
    let mut f = make();
    f.prefill(0, &p, &mut t).unwrap();
    let _ = read_logits(&mut f, 0);
    f.decode(n, &[first], &mut t).unwrap();
    f.prefill(n + 1, &[7, 9], &mut t).unwrap();
    match f.next(0, Want::Greedy, &mut t).unwrap() {
        Next::Token(x) => assert_eq!(x, sample_argmax(&lg).0),
        Next::Logits(l) => assert_eq!(l, &lg[..]),
    }
    assert!(f.next(0, Want::Logits, &mut t).is_err());

    // A position the cache does not hold is refused (I1).
    let mut d = make();
    assert!(
        d.prefill(1, &p, &mut t).is_err(),
        "prefill past an empty cache"
    );
    let mut d = make();
    d.prefill(0, &p, &mut t).unwrap();
    assert!(d.decode(n + 1, &[first], &mut t).is_err(), "a gap");
    assert!(d.decode(n - 1, &[first], &mut t).is_err(), "an overwrite");

    // next_rows equals the gather of next(Logits).
    let mut e = make();
    e.prefill(0, &p, &mut t).unwrap();
    let rows = [0u32, 5, 9, 31, 63];
    let mut out = [0f32; 5];
    e.next_rows(0, &rows, &mut out, &mut t).unwrap();
    let want: Vec<f32> = rows.iter().map(|&r| la[r as usize]).collect();
    assert_eq!(&out[..], &want[..]);

    // Export, then import into a fresh model, then decode equals the uninterrupted run.
    let mut u = make();
    u.prefill(0, &p, &mut t).unwrap();
    u.clear_pending();
    let reference = greedy_chain(&mut u, n, first, 8);
    let mut s = make();
    s.prefill(0, &p, &mut t).unwrap();
    let bytes = s.kv().export_blocks(0, n).unwrap().unwrap();
    let mut r = make();
    r.kv().import_blocks(0, n, &bytes).unwrap().unwrap();
    assert_eq!(r.kv().len(), n);
    assert_eq!(greedy_chain(&mut r, n, first, 8), reference);

    // A multi-token decode leaves one head per token, each as a single decode would.
    let mut m = make();
    if m.max_decode_batch() >= 3 {
        m.prefill(0, &p, &mut t).unwrap();
        m.clear_pending();
        let toks = [first, 7, 11];
        m.decode(n, &toks, &mut t).unwrap();
        let mut one = make();
        one.prefill(0, &p, &mut t).unwrap();
        for (j, &tok) in toks.iter().enumerate() {
            one.decode(n + j, &[tok], &mut t).unwrap();
            assert_eq!(read_logits(&mut m, j), read_logits(&mut one, 0), "head {j}");
        }
        assert!(m.next(3, Want::Logits, &mut t).is_err());
        // A new forward call replaces every unread head (I2).
        m.decode(n + 3, &[1, 2, 3], &mut t).unwrap();
        m.decode(n + 6, &[4], &mut t).unwrap();
        assert!(
            m.next(1, Want::Logits, &mut t).is_err(),
            "head 1 of the old call"
        );
        assert!(
            m.next(2, Want::Logits, &mut t).is_err(),
            "head 2 of the old call"
        );
        assert!(m.next(0, Want::Logits, &mut t).is_ok());
    }
}

#[test]
fn echo_chain_keeps_the_forward_pass_contract() {
    forward_pass_contract(&|| EchoChainModel::new(CTX));
    forward_pass_contract(&|| {
        let mut m = EchoChainModel::new(CTX);
        m.graph_mode = true;
        m
    });
    forward_pass_contract(&|| {
        let mut m = EchoChainModel::new(CTX);
        m.graph_mode = true;
        m.max_decode_batch = 3;
        m
    });
}

// ---------------------------------------------------------------------------------------------
// The controller (moved tests, then the new ones of section 5).
// ---------------------------------------------------------------------------------------------

// The moved tests (ADR 0004 section 5), generic over the forward pass: each runs here on the
// fake and on the tiny MoE through the 30B adapter (`qwen3_moe_omega::forward_pass` tests).

/// A cold run on a fresh model.
pub(crate) fn cold_on<M: ForwardPass>(
    make: &dyn Fn() -> M,
    prompt: &[u32],
    params: &GenerateParams,
) -> Vec<u32> {
    gen(&mut GenerationController::new(make()), prompt, params, None)
        .unwrap()
        .0
}

/// Returns the warm turn's outcome, so the caller can check its own prefill counters.
pub(crate) fn moved_prefix_reuse<M: ForwardPass + LayerFill>(
    make: &dyn Fn() -> M,
) -> GenerateOutcome {
    let mut c = GenerationController::new(make());
    let p1 = prompt(10, 2);
    let (first, o) = gen(&mut c, &p1, &greedy(4), None).unwrap();
    assert_eq!((o.reused_prompt_tokens, o.prompt_tokens), (0, 10));
    let mut p2 = c.history().to_vec();
    assert_eq!(p2.len(), 13, "prompt plus the three decoded tokens");
    assert_eq!(&p2[10..], &first[..3]);
    p2.extend([5, 6, 7]);
    let (warm, warm_outcome) = gen(&mut c, &p2, &greedy(8), None).unwrap();
    assert_eq!(warm_outcome.reused_prompt_tokens, 13);
    assert_eq!(warm, cold_on(make, &p2, &greedy(8)));
    assert_at_ledger(&c);
    // A prompt equal to the whole history still prefills its last token; an unrelated prompt
    // starts cold.
    let h = c.history().to_vec();
    let (_, o) = gen(&mut c, &h, &greedy(2), None).unwrap();
    assert_eq!(o.reused_prompt_tokens, h.len() - 1);
    let (_, o) = gen(&mut c, &[1, 2, 3], &greedy(2), None).unwrap();
    assert_eq!(o.reused_prompt_tokens, 0);
    assert_at_ledger(&c);
    warm_outcome
}

#[test]
fn prefix_reuse_prefills_only_the_new_tokens_and_matches_a_cold_run() {
    let o = moved_prefix_reuse(&|| EchoChainModel::new(CTX));
    assert_eq!(
        o.prefill_timing.launches, 3,
        "only the new tokens are prefilled"
    );
}

#[test]
fn the_unreused_suffix_reaches_prefill_in_one_call() {
    let mut c = ctl(false);
    let p1 = prompt(10, 3);
    let (_, _) = gen(&mut c, &p1, &greedy(4), None).unwrap();
    let mut p2 = c.history()[..12].to_vec();
    p2.extend([1, 2, 3, 4, 5]);
    let before = c.model().calls().len();
    let (out, o) = gen(&mut c, &p2, &greedy(5), None).unwrap();
    assert_eq!(o.reused_prompt_tokens, 12);
    let calls = c.model().calls()[before..].to_vec();
    let mut want = vec![Call::Prefill { pos: 12, len: 5 }];
    for (i, &t) in out[..out.len() - 1].iter().enumerate() {
        want.push(Call::Decode {
            pos: 17 + i,
            tokens: vec![t],
        });
    }
    assert_eq!(
        calls, want,
        "one prefill, one decode per emitted token but the last"
    );
}

#[test]
fn greedy_with_chip_argmax_reads_no_logits() {
    let p = prompt(12, 4);
    let mut graph = ctl(true);
    let (g, _) = gen(&mut graph, &p, &greedy(8), None).unwrap();
    let m = graph.model();
    // Every decode head here is a graph head (7 of them): asking it for Want::Logits would
    // count 7 chip reads, so 0 shows the controller asked Want::Greedy and took the held token.
    assert_eq!(m.chip_reads, 0, "greedy on the graph reads no logits back");
    assert_eq!(m.lm_heads, 1, "only the prefill head is a host output head");
    let mut host = ctl(false);
    let (h, _) = gen(&mut host, &p, &greedy(8), None).unwrap();
    assert_eq!((host.model().lm_heads, host.model().chip_reads), (8, 0));
    assert_eq!(g, h, "graph and host greedy give the same tokens");

    let mut graph = ctl(true);
    let (gs, o) = gen(&mut graph, &p, &sampled(8, 5), None).unwrap();
    assert_eq!(o.emitted_tokens, 8);
    assert_eq!(
        graph.model().chip_reads,
        7,
        "one read-back per decoded token"
    );
    assert_eq!(graph.model().lm_heads, 1);
    let mut host = ctl(false);
    let (hs, _) = gen(&mut host, &p, &sampled(8, 5), None).unwrap();
    assert_eq!(gs, hs, "graph and host sampling give the same tokens");
}

/// R1, red before green (ADR 0004 section 5): written against the loop without rollback, where
/// layer 0 keeps the half-written row; green once the controller cuts the cache to the ledger.
#[test]
fn a_forward_error_leaves_every_layer_at_the_ledger() {
    // Mid-prefill on a warm turn.
    let mut c = ctl(false);
    let p1 = prompt(10, 5);
    gen(&mut c, &p1, &greedy(4), None).unwrap();
    let h = c.history().to_vec();
    let mut p2 = h.clone();
    p2.extend([3, 4, 5, 6, 7, 8]);
    c.model_mut().fail_at = Some(h.len() + 3);
    let err = gen(&mut c, &p2, &greedy(8), None).unwrap_err();
    assert!(err.contains("injected failure"), "{err}");
    assert_eq!(c.history(), &h[..], "the ledger is unchanged");
    eprintln!(
        "R1 mid-prefill: position {} layer lengths {:?}",
        c.position(),
        c.model().layer_lens()
    );
    assert_at_ledger(&c);
    c.model_mut().fail_at = None;
    let (warm, _) = gen(&mut c, &p2, &greedy(8), None).unwrap();
    assert_eq!(
        warm,
        cold(&p2, &greedy(8)),
        "the next turn equals a cold run"
    );

    // Mid-decode.
    let mut c = ctl(true);
    let p = prompt(10, 6);
    c.model_mut().fail_at = Some(p.len() + 2);
    let mut out = Vec::new();
    let err = c
        .generate(&p, &greedy(8), &mut SampleNext(greedy(8)), None, &mut |t| {
            out.push(t);
            true
        })
        .unwrap_err();
    assert!(err.contains("injected failure"), "{err}");
    assert_eq!(c.position(), p.len() + 2);
    assert_eq!(&c.history()[p.len()..], &out[..2]);
    eprintln!(
        "R1 mid-decode: position {} layer lengths {:?}",
        c.position(),
        c.model().layer_lens()
    );
    assert_at_ledger(&c);
    c.model_mut().fail_at = None;
    let mut p3 = c.history().to_vec();
    p3.push(9);
    let (warm, _) = gen(&mut c, &p3, &greedy(8), None).unwrap();
    assert_eq!(warm, cold(&p3, &greedy(8)));
}

pub(crate) fn moved_stops_caps_and_cancels<M: ForwardPass>(make: &dyn Fn() -> M) {
    let p = prompt(10, 7);
    let chain = cold_on(make, &p, &greedy(8));
    assert_eq!(chain.len(), 8);
    let stop = chain[4];
    let i = chain.iter().position(|&t| t == stop).unwrap();
    let mut c = GenerationController::new(make());
    let (out, o) = gen(&mut c, &p, &GenerateParams::greedy(8, vec![stop]), None).unwrap();
    assert_eq!(
        (&out[..], o.finish, o.emitted_tokens),
        (&chain[..i], FinishReason::Stop, i)
    );

    let mut c = GenerationController::new(make());
    let (out, o) = gen(&mut c, &p, &greedy(5), None).unwrap();
    assert_eq!((&out[..], o.finish), (&chain[..5], FinishReason::Length));

    let mut c = GenerationController::new(make());
    let mut out = Vec::new();
    let o = c
        .generate(&p, &greedy(8), &mut SampleNext(greedy(8)), None, &mut |t| {
            out.push(t);
            out.len() < 3
        })
        .unwrap();
    assert_eq!(
        (&out[..], o.finish, o.emitted_tokens),
        (&chain[..3], FinishReason::Cancelled, 3)
    );
}

#[test]
fn generate_stops_on_stop_ids_caps_length_and_cancels() {
    moved_stops_caps_and_cancels(&|| EchoChainModel::new(CTX));
}

#[test]
fn stop_and_cancel_tokens_are_never_stepped() {
    let p = prompt(10, 8);
    let chain = cold(&p, &greedy(8));
    let stop = chain[3];
    let i = chain.iter().position(|&t| t == stop).unwrap();
    let mut c = ctl(false);
    gen(&mut c, &p, &GenerateParams::greedy(8, vec![stop]), None).unwrap();
    assert_eq!(
        &c.history()[p.len()..],
        &chain[..i],
        "the stop token is not stored"
    );
    assert_eq!(decodes(&c.model().calls()), i);
    assert_at_ledger(&c);

    let mut c = ctl(false);
    let mut n = 0;
    c.generate(
        &p,
        &greedy(8),
        &mut SampleNext(greedy(8)),
        None,
        &mut |_| {
            n += 1;
            n < 3
        },
    )
    .unwrap();
    assert_eq!(
        &c.history()[p.len()..],
        &chain[..2],
        "the cancelled token is not stored"
    );
    assert_eq!(decodes(&c.model().calls()), 2);
    assert_at_ledger(&c);
}

#[test]
fn each_on_token_fires_before_the_next_decode() {
    let p = prompt(9, 9);
    let mut c = ctl(false);
    let log = c.model().log_handle();
    let mut out = Vec::new();
    c.generate(&p, &greedy(5), &mut SampleNext(greedy(5)), None, &mut |t| {
        log.lock().unwrap().push(Call::Emitted(t));
        out.push(t);
        true
    })
    .unwrap();
    let mut want = vec![Call::Prefill { pos: 0, len: 9 }];
    for (i, &t) in out.iter().enumerate() {
        want.push(Call::Emitted(t));
        if i + 1 < out.len() {
            want.push(Call::Decode {
                pos: 9 + i,
                tokens: vec![t],
            });
        }
    }
    assert_eq!(c.model().calls(), want);
}

#[test]
fn a_full_context_ends_with_length_and_no_error() {
    let mut c = GenerationController::new(EchoChainModel::new(32));
    assert_eq!(c.max_context_tokens(), 32);
    let p = prompt(30, 10);
    let (out, o) = gen(&mut c, &p, &greedy(10), None).unwrap();
    assert_eq!((out.len(), o.finish), (3, FinishReason::Length));
    assert_eq!(c.position(), 32);
    assert_at_ledger(&c);
    assert!(
        gen(&mut c, &prompt(32, 10), &greedy(1), None).is_err(),
        "a prompt that fills the context"
    );
}

/// `make_a` runs the uninterrupted engine, `make_b` the one the prefix is restored into.
pub(crate) fn moved_restored_prefix<M: ForwardPass + LayerFill>(
    make_a: &dyn Fn() -> M,
    make_b: &dyn Fn() -> M,
) {
    let mut a = GenerationController::new(make_a());
    gen(&mut a, &prompt(20, 11), &greedy(8), None).unwrap();
    let h = a.history().to_vec();
    let bytes = a.export_kv(0, h.len()).unwrap();
    let mut next = h.clone();
    next.extend([4, 2]);
    let (want, _) = gen(&mut a, &next, &greedy(8), None).unwrap();

    let mut b = GenerationController::new(make_b());
    b.restore_kv(&h, &bytes).unwrap();
    assert_eq!(b.history(), &h[..]);
    assert_at_ledger(&b);
    assert_eq!(
        b.export_kv(0, h.len()).unwrap(),
        bytes,
        "import then export is byte-identical"
    );
    let (got, o) = gen(&mut b, &next, &greedy(8), None).unwrap();
    assert_eq!(o.reused_prompt_tokens, h.len());
    assert_eq!(got, want);
    assert_eq!(got, cold_on(make_b, &next, &greedy(8)));
}

#[test]
fn restored_prefix_continues_exactly_like_the_uninterrupted_engine() {
    moved_restored_prefix(&|| EchoChainModel::new(CTX), &|| {
        let mut m = EchoChainModel::new(CTX);
        m.graph_mode = true;
        m
    });
}

pub(crate) fn moved_failed_restore<M: ForwardPass + LayerFill>(make: &dyn Fn() -> M) {
    let p = prompt(20, 12);
    let mut a = GenerationController::new(make());
    gen(&mut a, &p, &greedy(4), None).unwrap();
    let h = a.history().to_vec();
    let mut snap = a.export_kv(0, h.len()).unwrap();
    assert!(a.export_kv(0, h.len() + 1).is_err(), "past the ledger");

    let mut b = GenerationController::new(make());
    gen(&mut b, &[1, 2, 3, 4], &greedy(2), None).unwrap();
    assert!(b.restore_kv(&h, &snap[..snap.len() - 2]).is_err());
    assert_eq!(b.position(), 0);
    assert_at_ledger(&b);
    let last = snap.len() - 1;
    snap[last] ^= 1; // a padding byte set
    assert!(b.restore_kv(&h, &snap).is_err());
    assert!(b.restore_kv(&[], &[]).is_err());
    let ctx = b.max_context_tokens();
    assert!(b.restore_kv(&vec![1u32; ctx], &snap).is_err());
    assert_eq!(b.position(), 0);
    assert_at_ledger(&b);
    let (got, o) = gen(&mut b, &p, &greedy(8), None).unwrap();
    assert_eq!(o.reused_prompt_tokens, 0);
    assert_eq!(
        got,
        cold_on(make, &p, &greedy(8)),
        "plain prefill after a refused restore"
    );
}

#[test]
fn a_failed_restore_resets_the_ledger() {
    moved_failed_restore(&|| EchoChainModel::new(CTX));
}

#[test]
fn max_new_tokens_zero_reads_no_head() {
    let mut c = ctl(false);
    let mut rule = Script::always(StepRule::Any);
    let (out, o) = gen(&mut c, &prompt(8, 13), &greedy(0), Some(&mut rule)).unwrap();
    assert_eq!((out.len(), o.finish), (0, FinishReason::Length));
    let m = c.model();
    assert_eq!((m.lm_heads, m.chip_reads, m.rows_reads), (0, 0, 0));
    assert_eq!(m.calls(), vec![Call::Prefill { pos: 0, len: 8 }]);
    assert!(rule.seen.is_empty(), "no step, no rule");
}

#[test]
fn timing_buckets_sum_the_calls_of_each_phase() {
    let p = prompt(10, 14);
    let (_, o) = gen(&mut ctl(false), &p, &greedy(4), None).unwrap();
    assert_eq!(o.prefill_timing.launches, 10);
    assert_eq!(
        o.decode_timing.launches,
        3 + 4,
        "three decodes and four host output heads"
    );
    let (_, o) = gen(&mut ctl(true), &p, &greedy(4), None).unwrap();
    assert_eq!(
        o.decode_timing.launches,
        3 + 1,
        "the prefill head is the only host head"
    );
    assert_eq!(
        (o.decode_timing.graph_tokens, o.decode_timing.graph_submits),
        (3, 3)
    );
}

#[test]
fn kv_identity_names_the_salt_and_a_mutant_refuses() {
    let a = ctl(false).kv_identity().unwrap();
    assert_eq!(a.kind, "echo-chain");
    let mut m = EchoChainModel::new(CTX);
    m.salt = 1;
    let b = GenerationController::new(m).kv_identity().unwrap();
    assert_ne!(
        a, b,
        "the salt changes K/V bytes, so it changes the identity"
    );
    let mut g = EchoChainModel::new(CTX);
    g.graph_mode = true;
    assert_eq!(GenerationController::new(g).kv_identity().unwrap(), a);
    let bad = EchoChainModel::with_kv(Box::new(CorruptRowKv::new(CTX, 1, 0).unwrap()));
    assert!(GenerationController::new(bad).kv_identity().is_err());
}

#[test]
fn locked_hands_out_the_controller_and_is_send_and_sync() {
    fn send_sync<T: Send + Sync>() {}
    send_sync::<Locked<EchoChainModel>>();
    let p = prompt(10, 15);
    let l = Locked::new(ctl(false));
    let (out, _) = l
        .with_controller(|c| gen(c, &p, &greedy(8), None))
        .unwrap()
        .unwrap();
    assert_eq!(out, cold(&p, &greedy(8)));
    assert_eq!(l.with_controller(|c| c.position()).unwrap(), 17);
}

// ---------------------------------------------------------------------------------------------
// Constraints (section 2.3), end to end on the fake.
// ---------------------------------------------------------------------------------------------

#[test]
fn an_allowed_set_picks_the_best_token_inside_it() {
    let p = prompt(12, 16);
    let u0 = cold(&p, &greedy(1))[0];
    // logits[i] = ((h + 37 i) % 64) / 64 with the maximum at u0, so the score of i is
    // (63 + 37 (i - u0)) mod 64.
    let score = |i: u32| (63 + 37 * ((i + 64 - u0) % 64)) % 64;
    let allowed: Vec<u32> = {
        let mut v: Vec<u32> = (1..=8).map(|k| (u0 + k) % 64).collect();
        v.sort_unstable();
        v
    };
    let best = *allowed.iter().max_by_key(|&&i| score(i)).unwrap();
    assert_ne!(best, u0);
    let mut c = ctl(true);
    let mut rule = Script::always(StepRule::Allow(allowed.clone()));
    let (out, _) = gen(&mut c, &p, &greedy(1), Some(&mut rule)).unwrap();
    assert_eq!(out, vec![best]);
    let m = c.model();
    assert_eq!((m.rows_reads, m.chip_reads), (1, 0));

    // Over many steps, greedy and sampled, every token is inside the set.
    for params in [greedy(10), sampled(10, 3)] {
        let mut c = ctl(true);
        let mut rule = Script::always(StepRule::Allow(allowed.clone()));
        let (out, _) = gen(&mut c, &p, &params, Some(&mut rule)).unwrap();
        assert_eq!(out.len(), 10);
        assert!(out.iter().all(|t| allowed.contains(t)), "{out:?}");
        assert_eq!(c.model().rows_reads, 10);
    }
}

#[test]
fn a_forced_token_is_emitted_decoded_and_counted() {
    let p = prompt(12, 17);
    let u0 = cold(&p, &greedy(1))[0];
    let forced = (u0 + 1) % 64;
    let mut c = ctl(true);
    let mut rule = Script::new(vec![Ok(StepRule::Force(forced))], StepRule::Any);
    let mut steps = StepRecorder::default();
    let (out, _) = gen_with(&mut c, &p, &greedy(9), &mut steps, Some(&mut rule)).unwrap();
    assert_eq!(out[0], forced, "in the stream");
    assert_eq!(c.history()[p.len()], forced, "in the history");
    assert_eq!(
        c.model().calls()[1],
        Call::Decode {
            pos: p.len(),
            tokens: vec![forced]
        }
    );
    assert_eq!(
        steps.0,
        (1..9).collect::<Vec<_>>(),
        "the forced step advanced the counter"
    );
    assert_eq!(
        rule.seen[1],
        vec![forced],
        "the constraint sees the forced token"
    );
    // The forced token's K/V bytes equal a prompt that ended with it, and so does the rest.
    let mut q = p.clone();
    q.push(forced);
    let mut d = ctl(false);
    let (rest, _) = gen(&mut d, &q, &greedy(8), None).unwrap();
    assert_eq!(&out[1..], &rest[..]);
    assert_eq!(
        c.export_kv(0, q.len()).unwrap(),
        d.export_kv(0, q.len()).unwrap()
    );
    // No head is read for a forced step.
    let mut c = ctl(true);
    let mut rule = Script::always(StepRule::Force(forced));
    gen(&mut c, &p, &greedy(1), Some(&mut rule)).unwrap();
    let m = c.model();
    assert_eq!((m.lm_heads, m.chip_reads, m.rows_reads), (0, 0, 0));
}

#[test]
fn finish_ends_with_finish_reason_constraint() {
    let p = prompt(12, 18);
    let chain = cold(&p, &greedy(8));
    let mut c = ctl(false);
    let mut rule = Script::new(vec![Ok(StepRule::Any); 3], StepRule::Finish);
    let (out, o) = gen(&mut c, &p, &greedy(8), Some(&mut rule)).unwrap();
    assert_eq!(
        (&out[..], o.finish, o.emitted_tokens),
        (&chain[..3], FinishReason::Constraint, 3)
    );
    assert_eq!(&c.history()[p.len()..], &chain[..3]);
    assert_at_ledger(&c);
    assert_eq!(c.model().lm_heads, 3, "the finishing step reads no head");
}

#[test]
fn any_is_byte_identical_to_no_constraint() {
    let p = prompt(12, 19);
    for graph in [false, true] {
        for params in [greedy(10), sampled(10, 21)] {
            let mut a = ctl(graph);
            let (ta, oa) = gen(&mut a, &p, &params, None).unwrap();
            let mut b = ctl(graph);
            let mut rule = Script::always(StepRule::Any);
            let (tb, ob) = gen(&mut b, &p, &params, Some(&mut rule)).unwrap();
            assert_eq!(ta, tb, "same tokens");
            assert_eq!(counted(&oa), counted(&ob), "same outcome and counters");
            assert_eq!(a.model().calls(), b.model().calls(), "same call log");
            let (ma, mb) = (a.model(), b.model());
            assert_eq!(
                (ma.chip_reads, ma.lm_heads, ma.rows_reads),
                (mb.chip_reads, mb.lm_heads, mb.rows_reads)
            );
            assert_eq!(
                (ma.rows_reads, mb.rows_reads),
                (0, 0),
                "next_rows is never called"
            );
            assert_eq!(rule.seen.len(), 10, "asked once per step");
        }
    }
}

#[test]
fn a_constraint_error_rolls_back_like_a_forward_error() {
    let p = prompt(12, 20);
    let chain = cold(&p, &greedy(8));
    let bad_rules = [
        Err("the automaton broke".to_string()),
        Ok(StepRule::Allow(Vec::new())),
        Ok(StepRule::Allow(vec![5, 3])),
        Ok(StepRule::Allow(vec![3, ECHO_VOCAB as u32])),
        Ok(StepRule::Force(ECHO_VOCAB as u32)),
    ];
    for bad in bad_rules {
        let mut c = ctl(true);
        let mut rule = Script::new(
            vec![Ok(StepRule::Any), Ok(StepRule::Any), bad.clone()],
            StepRule::Any,
        );
        assert!(
            gen(&mut c, &p, &greedy(8), Some(&mut rule)).is_err(),
            "{bad:?}"
        );
        assert_eq!(&c.history()[p.len()..], &chain[..2], "{bad:?}");
        assert_at_ledger(&c);
        assert!(
            c.model_mut()
                .next(0, Want::Logits, &mut StepTiming::default())
                .is_err(),
            "the rollback forgets the pending head ({bad:?})"
        );
        let mut q = c.history().to_vec();
        q.push(1);
        let (warm, _) = gen(&mut c, &q, &greedy(8), None).unwrap();
        assert_eq!(warm, cold(&q, &greedy(8)), "{bad:?}");
    }
}

#[test]
fn seeded_sampling_replays_under_a_constraint() {
    let p = prompt(12, 21);
    let allowed: Vec<u32> = (0..64).step_by(4).collect();
    let run = |seed: u64| {
        let mut rule = Script::always(StepRule::Allow(allowed.clone()));
        gen(&mut ctl(false), &p, &sampled(16, seed), Some(&mut rule))
            .unwrap()
            .0
    };
    let a = run(7);
    assert_eq!(a, run(7), "same seed, same tokens");
    assert!(a.iter().all(|t| allowed.contains(t)));
    assert_ne!(a, run(8), "another seed, other tokens");
}

// ---------------------------------------------------------------------------------------------
// The two negative controls of section 4: the fake must see a damaged or foreign K/V row.
// ---------------------------------------------------------------------------------------------

#[test]
fn negative_control_a_corrupted_surviving_row_changes_the_warm_run() {
    let p1 = prompt(20, 22);
    let corrupt = CorruptRowKv::new(CTX, 5, 1).unwrap();
    let mut c = GenerationController::new(EchoChainModel::with_kv(Box::new(corrupt)));
    gen(&mut c, &p1, &greedy(4), None).unwrap();
    let mut p2 = c.history().to_vec();
    p2.extend([9, 10]);
    let (damaged, o) = gen(&mut c, &p2, &greedy(8), None).unwrap();
    assert_eq!(
        o.reused_prompt_tokens, 23,
        "the warm run reuses the corrupted row"
    );

    // Baseline: the same warm run on a clean cache equals the cold run.
    let mut clean = ctl(false);
    gen(&mut clean, &p1, &greedy(4), None).unwrap();
    let (clean_warm, _) = gen(&mut clean, &p2, &greedy(8), None).unwrap();
    let cold_run = cold(&p2, &greedy(8));
    assert_eq!(clean_warm, cold_run);

    let differ = damaged
        .iter()
        .zip(&cold_run)
        .filter(|(a, b)| a != b)
        .count();
    eprintln!("negative control 1 (corrupt row at position 5, layer 1):\n  cold    {cold_run:?}\n  damaged {damaged:?}\n  {differ} of 8 tokens differ");
    assert_ne!(
        damaged, cold_run,
        "a corrupted surviving row must change the warm run"
    );
}

#[test]
fn negative_control_restoring_a_under_b_continues_differently() {
    let mut a = ctl(false);
    gen(&mut a, &prompt(20, 23), &greedy(4), None).unwrap();
    let ha = a.history().to_vec();
    let bytes = a.export_kv(0, ha.len()).unwrap();
    let hb: Vec<u32> = ha.iter().map(|t| (t + 1) % 64).collect();

    // Baseline: A's payload under A's tokens continues like a cold run on A.
    let mut next_a = ha.clone();
    next_a.push(3);
    let mut r = ctl(false);
    r.restore_kv(&ha, &bytes).unwrap();
    assert_eq!(
        gen(&mut r, &next_a, &greedy(8), None).unwrap().0,
        cold(&next_a, &greedy(8))
    );

    let mut next_b = hb.clone();
    next_b.push(3);
    let mut r = ctl(false);
    r.restore_kv(&hb, &bytes).unwrap();
    let (foreign, o) = gen(&mut r, &next_b, &greedy(8), None).unwrap();
    assert_eq!(
        o.reused_prompt_tokens,
        hb.len(),
        "the foreign payload is reused"
    );
    let cold_b = cold(&next_b, &greedy(8));
    let differ = foreign.iter().zip(&cold_b).filter(|(a, b)| a != b).count();
    eprintln!("negative control 2 (A's payload restored under B's tokens):\n  cold B  {cold_b:?}\n  A as B  {foreign:?}\n  {differ} of 8 tokens differ");
    assert_ne!(
        foreign, cold_b,
        "a foreign payload must continue differently"
    );
}

// ---------------------------------------------------------------------------------------------
// Review additions: the controller's own clear_pending calls, a sampler that leaves the allowed
// set, and a cache that cannot export or import.
// ---------------------------------------------------------------------------------------------

/// Whether head 0 is still pending (reading it consumes it).
fn head_pending(c: &mut Ctl) -> bool {
    c.model_mut()
        .next(0, Want::Logits, &mut StepTiming::default())
        .is_ok()
}

/// A generation that a constraint finishes leaves its last head unread.
fn finished_with_a_pending_head(graph: bool) -> Ctl {
    let mut c = ctl(graph);
    let mut rule = Script::new(vec![Ok(StepRule::Any); 2], StepRule::Finish);
    gen(&mut c, &prompt(12, 24), &greedy(8), Some(&mut rule)).unwrap();
    c
}

#[test]
fn rewind_a_new_generation_and_a_forced_token_forget_the_pending_head() {
    for graph in [false, true] {
        assert!(
            head_pending(&mut finished_with_a_pending_head(graph)),
            "precondition"
        );

        // rewind, even one that keeps every row.
        let mut c = finished_with_a_pending_head(graph);
        let n = c.position();
        c.rewind(n);
        assert!(!head_pending(&mut c), "rewind clears pending heads");

        // The start of every generation, even one refused before it touches the cache.
        let mut c = finished_with_a_pending_head(graph);
        assert!(gen(&mut c, &[], &greedy(4), None).is_err());
        assert!(!head_pending(&mut c), "generate clears pending heads first");

        // A forced token discards the head it did not read.
        let mut c = ctl(graph);
        let mut rule = Script::always(StepRule::Force(3));
        let (out, _) = gen(&mut c, &prompt(12, 25), &greedy(1), Some(&mut rule)).unwrap();
        assert_eq!(out, vec![3]);
        assert!(
            !head_pending(&mut c),
            "a forced step clears the unread head"
        );
    }
}

/// A sampler that always picks one token, whatever the logits say.
struct Picks(u32);

impl Sampler for Picks {
    fn wants_logits(&self) -> bool {
        true
    }
    fn pick(&mut self, _logits: &[f32], _step: usize) -> u32 {
        self.0
    }
}

#[test]
fn a_pick_outside_the_allowed_set_is_an_error() {
    let p = prompt(12, 26);
    let mut c = ctl(false);
    let mut rule = Script::new(
        vec![Ok(StepRule::Any), Ok(StepRule::Allow(vec![1, 2, 3]))],
        StepRule::Any,
    );
    let mut out = Vec::new();
    let r = c.generate(&p, &greedy(8), &mut Picks(0), Some(&mut rule), &mut |t| {
        out.push(t);
        true
    });
    assert!(
        r.is_err(),
        "token 0 is outside the allowed set: {r:?} {out:?}"
    );
    assert_eq!(out, vec![0], "only the unconstrained step emitted");
    assert_eq!(c.position(), p.len() + 1);
    assert_at_ledger(&c);
    assert!(!head_pending(&mut c));
}

/// The fake's pool with persistence switched off (export and import answer `None`).
struct NoPersistKv(HostPoolKv);

impl KvStore for NoPersistKv {
    fn geometry(&self) -> KvGeometry {
        self.0.geometry()
    }
    fn capacity(&self) -> usize {
        self.0.capacity()
    }
    fn len(&self) -> usize {
        self.0.len()
    }
    fn truncate(&mut self, len: usize) {
        self.0.truncate(len)
    }
}

impl KvAttention for NoPersistKv {
    fn append(&mut self, layer: usize, pos: usize, k: &[f32], v: &[f32]) -> Result<(), String> {
        self.0.append(layer, pos, k, v)
    }
    fn attend(
        &mut self,
        layer: usize,
        q: &[f32],
        context_len: usize,
        out: &mut [f32],
    ) -> Result<u64, String> {
        self.0.attend(layer, q, context_len, out)
    }
    fn resident(&self) -> bool {
        false
    }
}

impl EchoKv for NoPersistKv {
    fn layer_len(&self, layer: usize) -> usize {
        self.0.layer_len(layer)
    }
}

#[test]
fn a_cache_that_cannot_persist_refuses_export_and_restore() {
    let pool = HostPoolKv::new(&echo_geometry(), CTX).unwrap();
    let mut c = GenerationController::new(EchoChainModel::with_kv(Box::new(NoPersistKv(pool))));
    let p = prompt(20, 27);
    gen(&mut c, &p, &greedy(4), None).unwrap();
    let err = c.export_kv(0, 16).unwrap_err();
    assert!(err.contains("cannot export"), "{err}");
    let err = c.restore_kv(&p, &[0u8; 8]).unwrap_err();
    assert!(err.contains("cannot import"), "{err}");
    assert_eq!((c.position(), c.model().layer_lens()), (0, vec![0, 0]));
    let (got, _) = gen(&mut c, &p, &greedy(8), None).unwrap();
    assert_eq!(got, cold(&p, &greedy(8)));
}
