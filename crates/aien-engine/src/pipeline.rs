//! Two pipelines (L258, Drake 2026-10-21: "Is there no way to make two pipelines? ... Make it happen."). One machine has
//! one memory pipe; a second machine brings a second. The 35B is cut into stages by layer: stage `i` fires layers
//! `first..=last` on its own pipe and hands the residual (`[rows][2048]` f32, 8 KB per decode token) to stage `i + 1`
//! over the wire; the last stage turns it into a token and sends that back to stage 0, which feeds it to the next step
//! of the same conversation. Each stage keeps the DeltaNet and KV state of its own layers; only the residual moves.
//!
//! Many conversations in flight: pass `p` of every stage is step `p / convs` of conversation `p % convs` (round robin,
//! [`Stage::pass`]), so while stage 1 works on conversation 0, stage 0 already works on conversation 1. The order is
//! static (part of the schedule); a message out of order fails the stage closed.
use crate::link::{self, Header, Link};
use crate::plan::{Head, PlacementPlan, RouteTrace, Tier};
use crate::schedule::{Schedule, Stage};
use aien_heartbeat::manifest::Manifest;
use aien_inference_abi::qwen38::{argmax, Logits, Qwen38Reference, StageInput, StageOutput};
use aien_inference_abi::qwen3_moe_omega::TensorSource;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::net::TcpListener;
use std::path::Path;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

pub fn epoch_ns() -> i64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_nanos() as i64).unwrap_or(0)
}

/// Head per layer = the union of the experts every conversation's golden routed to in that layer (the replayed tokens
/// then never misfire), for the stage's layers only.
pub fn stage_plan(model: &str, dir: &Path, traces: &[RouteTrace], first: u16, last: u16, buffers: usize) -> PlacementPlan {
    let layers: Vec<u16> = (first..=last).collect();
    let heads = layers
        .iter()
        .map(|&l| {
            let set: BTreeSet<u16> = traces.iter().flat_map(|t| t.routes.iter().flat_map(move |r| r[l as usize].iter().copied())).collect();
            Head { layer: l, experts: set.into_iter().collect() }
        })
        .collect();
    PlacementPlan {
        schema: "aien-engine-plan/1".into(),
        model: model.into(),
        model_dir: dir.display().to_string(),
        synthetic: true,
        source: format!("L258 stage {first}..={last}: union of routed experts per layer over {} goldens", traces.len()),
        layers,
        heads,
        head_tier: Tier::Nvme,
        buffers_in_flight: buffers,
    }
}

/// The stage's table: `convs * steps` passes over its layers, the boundary recorded in the `stage` field and checked.
pub fn stage_schedule(plan: &PlacementPlan, m: &Manifest, stage: &Stage, steps: u32) -> Result<Schedule, String> {
    let mut s = Schedule::build(plan, m, stage.convs * steps)?;
    s.stage = Some(stage.clone());
    let r = s.check();
    if !r.pass() {
        return Err(format!("stage schedule check FAIL: {:?}", r.problems));
    }
    Ok(s)
}

/// One pass of one stage, wall clock (ns since the epoch, comparable between processes on one machine; between
/// machines shifted by the clock offset, see the report).
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct PassEvent {
    pub pass: u32,
    pub conv: u32,
    pub step: u32,
    pub rows: u32,
    /// Input arrived (stage 0: the token arrived, or the start for step 0).
    pub recv_ns: i64,
    pub start_ns: i64,
    pub done_ns: i64,
    /// Output written to the wire.
    pub sent_ns: i64,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct StageReport {
    pub schema: String,
    pub stage: Option<Stage>,
    pub host: String,
    pub source: String,
    pub boundary_check: bool,
    pub events: Vec<PassEvent>,
    /// Stage 0 (or a single stage): the token each conversation produced at each step.
    pub tokens: Vec<Vec<u32>>,
    /// Steps whose routes over this stage's layers equal the golden's, per conversation.
    pub routes_identical: Vec<u32>,
    pub failure: Option<String>,
    pub open_seconds: f64,
    pub wall_seconds: f64,
}

/// The wire of one stage: receive from the previous stage, send to the next. Stage 0's previous is the last stage
/// (tokens come back to it). Hello carries the sender's range; the receiver refuses a neighbour whose range does not
/// meet its own unless the boundary check is off (the wrong-boundary mutant, which must then fail the token gate).
pub struct Wire {
    pub prev: Link,
    pub next: Link,
}

/// How one end of a hop is opened. The Spark's firewall admits only outbound connections from the LAN side, so a
/// Spark stage dials both its neighbours and the remote machine listens on both; on one machine either works.
#[derive(Debug, Clone)]
pub enum End {
    Listen(String),
    Dial(String),
}

impl End {
    /// `listen:ADDR` or `dial:ADDR` (a bare ADDR means `default`).
    pub fn parse(s: &str, default_listen: bool) -> Self {
        match s.split_once(':').filter(|(k, _)| *k == "listen" || *k == "dial") {
            Some(("listen", a)) => Self::Listen(a.into()),
            Some((_, a)) => Self::Dial(a.into()),
            None if default_listen => Self::Listen(s.into()),
            None => Self::Dial(s.into()),
        }
    }

    fn open(self) -> std::thread::JoinHandle<Result<Link, String>> {
        std::thread::spawn(move || match self {
            End::Listen(a) => {
                let lis = TcpListener::bind(&a).map_err(|e| format!("bind {a}: {e}"))?;
                let (s, _) = lis.accept().map_err(|e| format!("accept on {a}: {e}"))?;
                Link::new(s)
            }
            End::Dial(a) => {
                let t = Instant::now();
                loop {
                    match Link::connect(&a) {
                        Ok(l) => return Ok(l),
                        Err(e) if t.elapsed() > Duration::from_secs(900) => return Err(e),
                        Err(_) => std::thread::sleep(Duration::from_millis(200)),
                    }
                }
            }
        })
    }
}

impl Wire {
    pub fn open(st: &Stage, prev: End, next: End, check: bool) -> Result<Self, String> {
        let (p, n) = (prev.open(), next.open());
        let mut prev = p.join().map_err(|_| "prev end panicked")?.map_err(|e| format!("stage {} prev: {e}", st.index))?;
        let mut next = n.join().map_err(|_| "next end panicked")?.map_err(|e| format!("stage {} next: {e}", st.index))?;
        let hello: Vec<u8> = [st.first as u32, st.last as u32, st.of, st.index].iter().flat_map(|v| v.to_le_bytes()).collect();
        next.send(&Header { kind: link::KIND_HELLO, ..Default::default() }, &hello)?;
        let mut p = Vec::new();
        let h = prev.recv(&mut p)?;
        if h.kind != link::KIND_HELLO || p.len() != 16 {
            return Err(format!("stage {}: expected hello, got kind {}", st.index, h.kind));
        }
        let w = |i: usize| u32::from_le_bytes(p[i * 4..i * 4 + 4].try_into().unwrap());
        let (pfirst, plast, pof, pindex) = (w(0), w(1), w(2), w(3));
        let meets = if st.index == 0 { plast + 1 == st.model_layers as u32 && pindex + 1 == st.of } else { plast + 1 == st.first as u32 && pindex + 1 == st.index };
        if check && (!meets || pof != st.of) {
            return Err(format!(
                "stage {} ({}..={}): previous stage {pindex} of {pof} holds {pfirst}..={plast}, which does not meet this boundary (fail closed)",
                st.index, st.first, st.last
            ));
        }
        Ok(Self { prev, next })
    }
}

/// Runs one stage to the end. `before_pass(p)` runs before each pass (the ring source learns which schedule pass the
/// next expert reads belong to). `wire` is `None` only for a single stage holding the whole model (Spark alone).
pub fn run_stage<S: TensorSource>(
    model: &mut Qwen38Reference<S>,
    st: &Stage,
    goldens: &[RouteTrace],
    steps: u32,
    mut wire: Option<Wire>,
    check: bool,
    before_pass: &mut dyn FnMut(u32),
) -> StageReport {
    let mut rep = StageReport { schema: "aien-engine-stage/1".into(), stage: Some(st.clone()), boundary_check: check, routes_identical: vec![0; st.convs as usize], tokens: vec![Vec::new(); st.convs as usize], ..Default::default() };
    let t0 = Instant::now();
    if let Err(e) = stage_loop(model, st, goldens, steps, wire.as_mut(), check, before_pass, &mut rep) {
        rep.failure = Some(e);
    }
    // the stages behind this one stop too (a failed stage tells its next; a finished stage 0 tells everyone)
    if let Some(w) = wire.as_mut() {
        let _ = w.next.send(&Header { kind: link::KIND_DONE, ..Default::default() }, &[]);
    }
    rep.wall_seconds = t0.elapsed().as_secs_f64();
    rep
}

fn routes_match(model: &Qwen38Reference<impl TensorSource>, st: &Stage, golden: &RouteTrace, step: usize) -> bool {
    let Some(want) = golden.routes.get(step) else { return false };
    (st.first..=st.last).all(|l| {
        let mut got = model.routes.get(l as usize).cloned().unwrap_or_default();
        got.sort_unstable();
        got.dedup();
        want.get(l as usize).map(|w| *w == got).unwrap_or(false)
    })
}

#[allow(clippy::too_many_arguments)]
fn stage_loop<S: TensorSource>(
    model: &mut Qwen38Reference<S>,
    st: &Stage,
    goldens: &[RouteTrace],
    steps: u32,
    mut wire: Option<&mut Wire>,
    check: bool,
    before_pass: &mut dyn FnMut(u32),
    rep: &mut StageReport,
) -> Result<(), String> {
    let c = st.convs as usize;
    if goldens.len() < c {
        return Err(format!("{} conversations but {} goldens", c, goldens.len()));
    }
    let h = model.config.hidden;
    let (first, last) = (st.first as usize, st.last as usize);
    let is_last = st.index + 1 == st.of;
    let mut states: Vec<_> = (0..c).map(|_| model.new_state()).collect();
    let total = st.convs * steps;
    let mut payload = Vec::new();
    // A pass: run the stage's layers on `input`, record the event, hand the output on (or keep the token locally).
    let mut fire = |model: &mut Qwen38Reference<S>, wire: &mut Option<&mut Wire>, rep: &mut StageReport, p: u32, input: StageInput, recv_ns: i64| -> Result<Option<u32>, String> {
        let (conv, step) = (p % st.convs, p / st.convs);
        before_pass(p);
        let start = epoch_ns();
        let out = model.forward_stage(&mut states[conv as usize], input, first, last, Logits::Last).map_err(|e| format!("pass {p} (conv {conv} step {step}): {e}"))?;
        let done = epoch_ns();
        if routes_match(model, st, &goldens[conv as usize], step as usize) {
            rep.routes_identical[conv as usize] += 1;
        }
        let (rows, token) = match out {
            StageOutput::Residual(x) => {
                let w = wire.as_mut().ok_or("a middle stage needs a wire")?;
                w.next.send(&Header { kind: link::KIND_RESIDUAL, conv, step, layer: (last + 1) as u32, rows: (x.len() / h) as u32, ..Default::default() }, &link::f32_bytes(&x))?;
                ((x.len() / h) as u32, None)
            }
            StageOutput::Logits(l) => {
                let tok = argmax(&l) as u32;
                if let Some(w) = wire.as_mut() {
                    w.next.send(&Header { kind: link::KIND_TOKEN, conv, step, rows: 1, ..Default::default() }, &tok.to_le_bytes())?;
                }
                (1, Some(tok))
            }
        };
        rep.events.push(PassEvent { pass: p, conv, step, rows, recv_ns, start_ns: start, done_ns: done, sent_ns: epoch_ns() });
        Ok(token)
    };
    let expect = |hd: &Header, p: u32| -> Result<(), String> {
        if (hd.conv, hd.step) != (p % st.convs, p / st.convs) {
            return Err(format!("out of order: got conv {} step {}, the schedule's pass {p} is conv {} step {}", hd.conv, hd.step, p % st.convs, p / st.convs));
        }
        Ok(())
    };
    if st.index == 0 {
        let prompt = |conv: usize| goldens[conv].prompt_tokens.clone();
        if wire.is_none() {
            // Spark alone: one stage holds the whole model, same pass order, no wire
            for p in 0..total {
                let (conv, step) = ((p % st.convs) as usize, p / st.convs);
                let ids = if step == 0 { prompt(conv) } else { vec![*rep.tokens[conv].last().unwrap()] };
                let tok = fire(model, &mut wire, rep, p, StageInput::Tokens(&ids), epoch_ns())?.ok_or("single stage gave no token")?;
                rep.tokens[conv].push(tok);
            }
            return Ok(());
        }
        for conv in 0..c {
            let ids = prompt(conv);
            fire(model, &mut wire, rep, conv as u32, StageInput::Tokens(&ids), epoch_ns())?;
        }
        for p in 0..total {
            let w = wire.as_mut().unwrap();
            let hd = w.prev.recv(&mut payload)?;
            let recv = epoch_ns();
            if hd.kind != link::KIND_TOKEN {
                return Err(format!("stage 0: expected a token for pass {p}, got kind {} (the last stage stopped)", hd.kind));
            }
            expect(&hd, p)?;
            let tok = u32::from_le_bytes(payload[..4].try_into().map_err(|_| "short token")?);
            let conv = hd.conv as usize;
            rep.tokens[conv].push(tok);
            let next = p + st.convs;
            if next < total {
                fire(model, &mut wire, rep, next, StageInput::Tokens(&[tok]), recv)?;
            }
        }
        return Ok(());
    }
    for p in 0..total {
        let w = wire.as_mut().ok_or("stage > 0 needs a wire")?;
        let hd = w.prev.recv(&mut payload)?;
        let recv = epoch_ns();
        if hd.kind == link::KIND_DONE {
            return Err(format!("previous stage stopped before pass {p}"));
        }
        if hd.kind != link::KIND_RESIDUAL {
            return Err(format!("expected a residual for pass {p}, got kind {}", hd.kind));
        }
        expect(&hd, p)?;
        if check && hd.layer != st.first as u32 {
            return Err(format!("pass {p}: the residual enters layer {}, this stage starts at {} (fail closed)", hd.layer, st.first));
        }
        let x = link::bytes_f32(&payload)?;
        let tok = fire(model, &mut wire, rep, p, StageInput::Residual(&x), recv)?;
        if is_last {
            let (conv, _) = ((p % st.convs) as usize, p / st.convs);
            rep.tokens[conv].push(tok.unwrap_or(u32::MAX));
        }
    }
    Ok(())
}

/// Text timing diagram over several stages' reports (one machine's clock): one row per stage and time bucket column,
/// the digit of the conversation the stage was computing, `.` idle. The picture of "stage 0 works on conversation 2
/// while stage 1 works on conversation 1".
pub fn pipe_diagram(reports: &[StageReport], width: usize) -> String {
    let t0 = reports.iter().flat_map(|r| r.events.iter().map(|e| e.start_ns)).min().unwrap_or(0);
    let t1 = reports.iter().flat_map(|r| r.events.iter().map(|e| e.done_ns)).max().unwrap_or(t0 + 1);
    let span = (t1 - t0).max(1) as f64;
    let mut out = format!("time 0 .. {:.3} s across {} columns ({:.1} ms each); digit = conversation computing, . = idle\n", span / 1e9, width, span / 1e6 / width as f64);
    for r in reports {
        let mut row = vec!['.'; width];
        for e in &r.events {
            let a = (((e.start_ns - t0) as f64 / span) * width as f64) as usize;
            let b = ((((e.done_ns - t0) as f64 / span) * width as f64).ceil() as usize).min(width);
            for cell in row.iter_mut().take(b.max(a + 1).min(width)).skip(a.min(width - 1)) {
                *cell = char::from_digit(e.conv % 10, 10).unwrap();
            }
        }
        let st = r.stage.as_ref();
        out.push_str(&format!("stage {} L{:>2}..{:>2} {:<8} |{}|\n", st.map(|s| s.index).unwrap_or(0), st.map(|s| s.first).unwrap_or(0), st.map(|s| s.last).unwrap_or(0), r.host.chars().take(8).collect::<String>(), row.into_iter().collect::<String>()));
    }
    out
}

/// Busy fraction of each stage and the overlap: time two or more stages computed at once / wall.
pub fn overlap(reports: &[StageReport]) -> (Vec<f64>, f64) {
    let t0 = reports.iter().flat_map(|r| r.events.iter().map(|e| e.start_ns)).min().unwrap_or(0);
    let t1 = reports.iter().flat_map(|r| r.events.iter().map(|e| e.done_ns)).max().unwrap_or(t0 + 1);
    let span = (t1 - t0).max(1) as f64;
    let busy: Vec<f64> = reports.iter().map(|r| r.events.iter().map(|e| (e.done_ns - e.start_ns) as f64).sum::<f64>() / span).collect();
    let mut edges: Vec<(i64, i32)> = reports.iter().flat_map(|r| r.events.iter().flat_map(|e| [(e.start_ns, 1), (e.done_ns, -1)])).collect();
    edges.sort();
    let (mut depth, mut last, mut both) = (0i32, t0, 0i64);
    for (t, d) in edges {
        if depth >= 2 {
            both += t - last;
        }
        depth += d;
        last = t;
    }
    (busy, both as f64 / span)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn st(index: u32, of: u32, first: u16, last: u16) -> Stage {
        Stage { index, of, first, last, model_layers: 40, convs: 2 }
    }

    #[test]
    fn stage_boundaries_are_checked() {
        assert!(st(0, 2, 0, 19).problems(&(0..=19).collect::<Vec<_>>()).is_empty());
        assert!(st(1, 2, 20, 39).problems(&(20..=39).collect::<Vec<_>>()).is_empty());
        assert!(!st(1, 2, 19, 39).problems(&(20..=39).collect::<Vec<_>>()).is_empty(), "table layers disagree with the field");
        assert!(!st(1, 2, 20, 38).problems(&(20..=38).collect::<Vec<_>>()).is_empty(), "last stage must end at layer 39");
        assert!(!st(0, 2, 1, 19).problems(&(1..=19).collect::<Vec<_>>()).is_empty(), "first stage must start at 0");
        assert_eq!(st(0, 2, 0, 19).pass(1, 3), 7);
    }

    #[test]
    fn overlap_counts_time_two_stages_compute_at_once() {
        let ev = |s: i64, d: i64| PassEvent { start_ns: s, done_ns: d, ..Default::default() };
        let a = StageReport { events: vec![ev(0, 10), ev(20, 30)], ..Default::default() };
        let b = StageReport { events: vec![ev(5, 25)], ..Default::default() };
        let (busy, both) = overlap(&[a.clone(), b.clone()]);
        assert!((busy[0] - 20.0 / 30.0).abs() < 1e-9 && (busy[1] - 20.0 / 30.0).abs() < 1e-9);
        assert!((both - 10.0 / 30.0).abs() < 1e-9, "{both}");
        assert!(pipe_diagram(&[a, b], 30).lines().count() == 3);
    }
}
