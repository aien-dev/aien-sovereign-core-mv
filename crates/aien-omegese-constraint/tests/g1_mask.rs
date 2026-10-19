//! Gate G1 of the Omegese mask (ledger L112, evidence/OMEGESE-MASK): the constraint with the reader
//! linked as a library, the real vocabulary of the model that will run it, driven through the
//! generation controller with no chip.
//!
//! 1. Reproduction: for each worked example of the exp445 `omegese` arm (`many.list`), a teacher
//!    model puts its highest logit on an illegal token (a decoy holding `{`) and its second on the
//!    token the tokenizer writes next for the example. With the greedy sampler, the controller
//!    picks the highest-probability legal token at every masked step, so the reply is the example,
//!    fenced, ending on the stop token, only if every token of the example is legal and the decoy
//!    never is.
//! 2. Random walks: `G1_WALKS` (default 2000) seeded walks of a wandering model (noise logits, a
//!    lift on Omegese's fixed words, newlines and the closing fence so walks end). Every walk's
//!    program must be accepted by `omegese emit` (the command line, built separately from the
//!    library: the judge is not the code under test): a walk that ended is judged as written; a
//!    walk cut at `G1_MAX_TOKENS` is judged after `omegese complete` writes its ending. A refusal
//!    is a bug in the mask or in `next`.
//!
//! Needs: `OMEGESE_TOKENIZER` (the model's tokenizer.json), `OMEGESE_EXAMPLES`
//! (tools/exp445/arms/examples), `OMEGESE_BIN` (the omegese command), a build that linked the
//! library (`AIEN_OMEGESE_DIR` or `AIEN_OMEGESE_LIB`). Optional: `G1_WALKS`, `G1_MAX_TOKENS`,
//! `G1_THREADS`, `G1_OUT` (receipt file). Run with `--ignored --nocapture`.

mod common;

use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use aien_inference_abi::generation::{
    ForwardPass, GenerationController, KvIdentityInputs, Next, SampleNext, Want,
};
use aien_inference_abi::qwen3_moe_omega::{
    FinishReason, GenerateParams, KvGeometry, KvStore, StepTiming,
};
use aien_omegese_constraint::{linked, Framing, LinkedOracle, OmegeseConstraint, TokenTable};
use common::{complete, emit, omegese_bin};

/// A position counter: the controller's ledger needs a cache; these models hold no K/V.
struct CountKv {
    len: usize,
}

impl KvStore for CountKv {
    fn geometry(&self) -> KvGeometry {
        KvGeometry {
            layers: 1,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 1,
            rope_theta: 1.0,
            block_size: 16,
        }
    }
    fn capacity(&self) -> usize {
        1 << 20
    }
    fn len(&self) -> usize {
        self.len
    }
    fn truncate(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

/// How a model scores the next token, given the reply text so far.
trait Scorer: Send {
    fn score(&mut self, written: &[u8], logits: &mut [f32]);
}

/// A model over the real vocabulary: the reply text so far (the bytes of the tokens decoded
/// after the prompt) in, a scorer's logits out.
struct TextModel<S: Scorer> {
    table: Arc<TokenTable>,
    kv: CountKv,
    written: Vec<u8>,
    logits: Vec<f32>,
    scorer: S,
}

impl<S: Scorer> TextModel<S> {
    fn new(table: Arc<TokenTable>, scorer: S) -> Self {
        let n = table.len();
        Self {
            table,
            kv: CountKv { len: 0 },
            written: Vec::new(),
            logits: vec![0.0; n],
            scorer,
        }
    }
}

impl<S: Scorer> ForwardPass for TextModel<S> {
    fn kind(&self) -> &'static str {
        "g1-text-model"
    }
    fn vocab(&self) -> usize {
        self.table.len()
    }
    fn kv(&mut self) -> &mut dyn KvStore {
        &mut self.kv
    }
    fn prefill(&mut self, pos: usize, tokens: &[u32], _t: &mut StepTiming) -> Result<(), String> {
        self.written.clear();
        self.kv.len = pos + tokens.len();
        Ok(())
    }
    fn decode(&mut self, pos: usize, tokens: &[u32], _t: &mut StepTiming) -> Result<(), String> {
        for &t in tokens {
            if let Some(x) = self.table.text(t) {
                self.written.extend_from_slice(x);
            }
        }
        self.kv.len = pos + tokens.len();
        Ok(())
    }
    fn next(&mut self, _head: usize, _want: Want, _t: &mut StepTiming) -> Result<Next<'_>, String> {
        self.scorer.score(&self.written, &mut self.logits);
        Ok(Next::Logits(&self.logits))
    }
    fn clear_pending(&mut self) {}
    fn kv_identity(&self) -> Result<KvIdentityInputs, String> {
        Err("the G1 models store no K/V".into())
    }
}

struct XorShift(u64);
impl XorShift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / (1u64 << 24) as f32
    }
}

/// Highest: an illegal decoy; second: the tokenizer's next token for the target; the rest noise
/// below both.
struct Teacher {
    table: Arc<TokenTable>,
    target: Vec<u8>,
    stop: u32,
    decoy: u32,
    rng: XorShift,
    /// The tokenizer's next token at the last step, and how many picks differed from it (the mask
    /// refused it, e.g. a token across two word boundaries, and a shorter piece was legal).
    last_canonical: Option<(usize, u32)>,
    non_canonical: u64,
}

impl Scorer for Teacher {
    fn score(&mut self, written: &[u8], logits: &mut [f32]) {
        for l in logits.iter_mut() {
            *l = self.rng.unit();
        }
        logits[self.decoy as usize] = 20.0;
        if let Some((at, c)) = self.last_canonical.take() {
            let took = &written[at.min(written.len())..];
            if self.table.text(c) != Some(took) {
                self.non_canonical += 1;
            }
        }
        // Every token that continues the target ranks above the noise (longer first), the
        // tokenizer's own next token above them all.
        if let Some(rest) = self.target.strip_prefix(written) {
            for (i, l) in logits.iter_mut().enumerate() {
                if let Some(t) = self.table.text(i as u32) {
                    if !t.is_empty() && rest.starts_with(t) {
                        *l = 5.0 + 0.01 * t.len() as f32;
                    }
                }
            }
        }
        let want = if written == self.target.as_slice() {
            Some(self.stop)
        } else if self.target.starts_with(written) {
            let rest = std::str::from_utf8(&self.target[written.len()..]).unwrap();
            self.table.encode(rest).ok().and_then(|v| v.first().copied())
        } else {
            None
        };
        if let Some(t) = want {
            logits[t as usize] = 10.0;
            if t != self.stop {
                self.last_canonical = Some((written.len(), t));
            }
        }
    }
}

/// Noise, lifted on structure so walks end: the closing fence most, then newlines and Omegese's
/// fixed words.
struct Wanderer {
    lift: Arc<Vec<f32>>,
    rng: XorShift,
}

impl Scorer for Wanderer {
    fn score(&mut self, _written: &[u8], logits: &mut [f32]) {
        for (l, &b) in logits.iter_mut().zip(self.lift.iter()) {
            *l = self.rng.unit() + b;
        }
    }
}

fn env_path(k: &str) -> PathBuf {
    let p = PathBuf::from(std::env::var_os(k).unwrap_or_else(|| panic!("set {k}")));
    assert!(p.exists(), "{k} {} does not exist", p.display());
    p
}

fn env_num(k: &str, d: usize) -> usize {
    std::env::var(k).ok().map_or(d, |v| v.parse().unwrap())
}

fn framing(stop: u32) -> Framing {
    Framing {
        open_fence: Some("```\n".into()),
        header: None,
        mask_body: true,
        close_fence: "```".into(),
        stop_token: stop,
    }
}

fn params(max: usize, stop: u32) -> GenerateParams {
    GenerateParams {
        max_new_tokens: max,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        seed: 0,
        stop_token_ids: vec![stop],
    }
}

fn load() -> (Arc<TokenTable>, u32, u32) {
    assert!(
        linked::is_linked(),
        "this build did not link libomegese ({})",
        linked::LIB_SOURCE
    );
    let table = TokenTable::from_tokenizer_json(&env_path("OMEGESE_TOKENIZER")).unwrap();
    let stop = table.special_id("<|im_end|>").expect("<|im_end|>");
    let decoy = (0..table.len() as u32)
        .find(|&i| table.text(i) == Some(b"{".as_slice()))
        .expect("a `{` token");
    (Arc::new(table), stop, decoy)
}

fn receipt(line: &str) {
    println!("{line}");
    if let Ok(p) = std::env::var("G1_OUT") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&p)
            .unwrap();
        writeln!(f, "{line}").unwrap();
    }
}

fn pct(v: &mut [Duration], q: f64) -> Duration {
    if v.is_empty() {
        return Duration::ZERO;
    }
    v.sort_unstable();
    v[((v.len() - 1) as f64 * q).round() as usize]
}

#[test]
#[ignore = "needs OMEGESE_TOKENIZER, OMEGESE_EXAMPLES and the linked library"]
fn g1_reproduces_every_worked_example_under_a_decoy() {
    let (table, stop, decoy) = load();
    let dir = env_path("OMEGESE_EXAMPLES");
    let list = std::fs::read_to_string(dir.join("many.list")).unwrap();
    let names: Vec<&str> = list.split_whitespace().collect();
    assert_eq!(names.len(), 14, "many.list names the 14 worked examples");
    let mut failures = Vec::new();
    for (i, name) in names.iter().enumerate() {
        let prog = std::fs::read_to_string(dir.join(name).join("program.omegese")).unwrap();
        let mut target = b"```\n".to_vec();
        target.extend_from_slice(prog.as_bytes());
        if !target.ends_with(b"\n") {
            target.push(b'\n');
        }
        target.extend_from_slice(b"```");
        let teacher = Teacher {
            table: table.clone(),
            target: target.clone(),
            stop,
            decoy,
            rng: XorShift(0x9E37_79B9_7F4A_7C15 ^ i as u64),
            last_canonical: None,
            non_canonical: 0,
        };
        let mut ctl = GenerationController::new(TextModel::new(table.clone(), teacher));
        let mut k = OmegeseConstraint::new(table.clone(), LinkedOracle::new().unwrap(), framing(stop))
            .unwrap();
        let mut out = Vec::new();
        let o = ctl
            .generate(
                &[1, 2, 3],
                &params(4096, stop),
                &mut SampleNext(params(4096, stop)),
                Some(&mut k),
                &mut |t| {
                    out.push(t);
                    true
                },
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let text = table.decode(&out).unwrap();
        let same = text == target && o.finish == FinishReason::Stop;
        let s = k.stats();
        let nc = ctl.model().scorer.non_canonical;
        receipt(&format!(
            "G1-REPRO {name}: {} tokens, {} masked steps (the decoy ranked above the pick at each), {} forced, {} oracle calls, {nc} picks other than the tokenizer's own token, reply == fenced example: {}",
            out.len(),
            s.masked_steps,
            s.forced_steps,
            s.oracle_calls,
            if same { "yes" } else { "NO" }
        ));
        if !same {
            failures.push(format!(
                "{name}: finish {:?}, wrote {:?}",
                o.finish,
                String::from_utf8_lossy(&text)
            ));
        }
    }
    receipt(&format!(
        "G1-REPRO {}: {} of {} worked examples reproduced exactly",
        if failures.is_empty() { "PASS" } else { "FAIL" },
        names.len() - failures.len(),
        names.len()
    ));
    assert!(failures.is_empty(), "{failures:#?}");
}

/// The lift of every token: the closing fence most, then newlines and the fixed words.
fn lifts(table: &TokenTable, words: &[String]) -> Vec<f32> {
    (0..table.len() as u32)
        .map(|i| match table.text(i) {
            Some(t) if t.contains(&b'`') => 8.0,
            Some(t) if t.contains(&b'\n') => 1.5,
            Some(t) => {
                let w = std::str::from_utf8(t).unwrap_or("").trim();
                if words.iter().any(|x| x == w) {
                    1.0
                } else {
                    0.0
                }
            }
            None => 0.0,
        })
        .collect()
}

#[test]
#[ignore = "needs OMEGESE_TOKENIZER, OMEGESE_BIN and the linked library"]
fn g1_random_walks_are_all_accepted_by_emit() {
    let (table, stop, _) = load();
    let bin = omegese_bin();
    let words: Vec<String> = String::from_utf8(
        std::process::Command::new(&bin)
            .arg("words")
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap()
    .lines()
    .map(str::to_string)
    .collect();
    assert!(words.len() > 20, "omegese words printed {}", words.len());
    let lift = Arc::new(lifts(&table, &words));
    let walks = env_num("G1_WALKS", 2000);
    let max = env_num("G1_MAX_TOKENS", 160);
    let threads = env_num("G1_THREADS", 8).max(1);
    let next = AtomicU64::new(0);
    let tally = Mutex::new((0u64, 0u64, 0u64, Vec::<String>::new())); // ended, cut, accepted, refusals
    let steps = Mutex::new((0u64, 0u64, 0u64, Vec::<Duration>::new())); // masked, forced, calls, step times
    let t0 = Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let w = next.fetch_add(1, Ordering::Relaxed);
                if w >= walks as u64 {
                    break;
                }
                let model = TextModel::new(
                    table.clone(),
                    Wanderer {
                        lift: lift.clone(),
                        rng: XorShift(0xD1B5_4A32_D192_ED03 ^ (w + 1).wrapping_mul(0x9E37_79B9)),
                    },
                );
                let mut ctl = GenerationController::new(model);
                let mut k =
                    OmegeseConstraint::new(table.clone(), LinkedOracle::new().unwrap(), framing(stop))
                        .unwrap();
                let o = ctl.generate(
                    &[1, 2, 3],
                    &params(max, stop),
                    &mut SampleNext(params(max, stop)),
                    Some(&mut k),
                    &mut |_| true,
                );
                let prog = k.program().to_vec();
                let (ended, verdict) = match o {
                    Err(e) => (false, Err(format!("walk {w}: the constraint failed: {e}"))),
                    Ok(o) if o.finish == FinishReason::Stop && k.finished() => {
                        (true, emit(&bin, &prog).map_err(|e| format!("walk {w} (ended): {e}")))
                    }
                    Ok(_) => (
                        false,
                        complete(&bin, &prog)
                            .and_then(|tail| {
                                let mut p = prog.clone();
                                p.extend_from_slice(&tail);
                                emit(&bin, &p)
                            })
                            .map_err(|e| format!("walk {w} (cut at {max}): {e}")),
                    ),
                };
                {
                    let st = k.stats();
                    let mut g = steps.lock().unwrap();
                    g.0 += st.masked_steps;
                    g.1 += st.forced_steps;
                    g.2 += st.oracle_calls;
                    g.3.extend_from_slice(&st.step_times);
                }
                let mut g = tally.lock().unwrap();
                if ended {
                    g.0 += 1;
                } else {
                    g.1 += 1;
                }
                match verdict {
                    Ok(()) => g.2 += 1,
                    Err(e) => g.3.push(format!(
                        "{e}\nprogram: {:?}",
                        String::from_utf8_lossy(&prog)
                    )),
                }
            });
        }
    });
    let (ended, cut, accepted, refusals) = tally.into_inner().unwrap();
    let (masked, forced, calls, mut times) = steps.into_inner().unwrap();
    receipt(&format!(
        "G1-WALKS {}: {walks} walks ({threads} threads, cap {max} tokens) in {:.1} s: {ended} ended on the closing fence and stop, {cut} cut and completed; {accepted} accepted by omegese emit, {} refused",
        if refusals.is_empty() && accepted == walks as u64 { "PASS" } else { "FAIL" },
        t0.elapsed().as_secs_f64(),
        refusals.len()
    ));
    receipt(&format!(
        "G1-WALKS steps: {masked} masked, {forced} forced, {calls} oracle calls ({:.2} per masked step); masked step wall time median {:?}, p95 {:?}, max {:?} ({threads} walks in parallel, the reader serialised)",
        calls as f64 / masked.max(1) as f64,
        pct(&mut times, 0.5),
        pct(&mut times, 0.95),
        pct(&mut times, 1.0)
    ));
    for r in refusals.iter().take(5) {
        receipt(&format!("G1-WALKS refusal: {r}"));
    }
    assert!(refusals.is_empty(), "{} refusals", refusals.len());
    assert_eq!(accepted, walks as u64);
}
