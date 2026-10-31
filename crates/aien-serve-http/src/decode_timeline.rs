// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Drake Stapleton (AIEN / aien-dev)

//! The 35B one-row decode timeline: where one single-row decode beat of the batch door spends its
//! time, on the host and on the chip.
//!
//! `AIEN_Q36_DECODE_TIMELINE=<file.jsonl>` turns it on. When unset the door records nothing,
//! reads no extra clock and leaves the chip stamps off. When set, every row's one-row graph runs
//! with its per-launch chip stamps, and each beat that steps exactly one decoding row (no prompt
//! chunk, no other row) appends one JSON line, schema [`SCHEMA`]:
//!
//! - `host`: [`HOST_PHASES`] in order, contiguous, on [`timeline_now_ns`]. Each beat runs from its
//!   join to the next beat's join, or to the idle point when nobody is left.
//!   `graph_record` and `graph_submit_wait` come from the graph's own `record_ns` / `total_ns`
//!   (the list build, then submit + chip + completion wait + readback).
//! - `dev`: one span per launch, contiguous, on the chip's PTIMER. That is a different clock, so
//!   only durations are compared with host spans and never instants.
//!
//! [`check`] holds a file to that shape, and `aien-serve-http --check-decode-timeline` runs it and prints a
//! Wolfram-ready summary. Each stamp sits after a launch's wait-for-idle and adds a timestamp
//! report, so the stamps cost chip time: a timeline run is never a speed receipt.
//!
//! DRA-66, the per-token receipt: `AIEN_Q36_DECODE_TIMELINE_KIND=token` (default `launch`, the
//! above) writes one [`TokenRecord`] (schema [`TOKEN_SCHEMA`]) per beat that fed at least one
//! decoding row, at any number of conversations, with the chip stamps OFF (a speed receipt):
//! the beat's period on the host clock (`total_ns`, join to the next join or idle), the chip time
//! of every step in it (`chip_ns`, prefill chunks included) and the host's share
//! `host_overhead_ns = total_ns - chip_ns`. `prompt_rows > 0` marks a beat a joining prompt rode
//! in. [`check_tokens`] holds such a file to its shape; the same `--check-decode-timeline` CLI
//! dispatches on the schema.
//!
//! `chip_ns` is the graph's host-clock span from the first submit to the last completion marker
//! read back, so it also holds host work: the queue submit and the marker poll. The per-token
//! `host_overhead_ns` is therefore a LOWER bound of the host time. The DRA-66 gate
//! ([`GATE_WL`]) adds the submit + poll share `chip_ns - dev_span`, measured on a stamped
//! one-conversation launch timeline of the same window, to that lower bound.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use aien_inference_abi::qwen38_omega::RowTimeline;
use aien_omega_q38::GraphStats;
use serde::{Deserialize, Serialize};

pub use aien_inference_abi::qwen38_omega::timeline_now_ns;

/// The switch: a path to append the timeline to. Unset: off.
pub const ENV: &str = "AIEN_Q36_DECODE_TIMELINE";
/// What the switched-on door records: `launch` (default, [`Record`]) or `token` ([`TokenRecord`]).
pub const KIND_ENV: &str = "AIEN_Q36_DECODE_TIMELINE_KIND";
/// The record schema.
pub const SCHEMA: &str = "aien.q36-decode-timeline/1";
/// The per-token receipt's schema.
pub const TOKEN_SCHEMA: &str = "aien.q36-token-receipt/1";

/// Which records a switched-on door writes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Kind {
    /// One stamped single-row decode beat per [`Record`] (chip stamps on: not a speed run).
    Launch,
    /// One [`TokenRecord`] per decoding beat, stamps off.
    Token,
}

impl Kind {
    /// [`KIND_ENV`]: unset or `launch` is [`Kind::Launch`], `token` is [`Kind::Token`].
    pub fn from_env() -> Result<Self, String> {
        match std::env::var(KIND_ENV).as_deref() {
            Err(_) | Ok("") | Ok("launch") => Ok(Self::Launch),
            Ok("token") => Ok(Self::Token),
            Ok(v) => Err(format!("{KIND_ENV}={v}: launch | token")),
        }
    }
}

/// One decoding beat's sums, handed to [`Recorder::beat`] at the end of the beat.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BeatSums {
    pub beat: u64,
    /// Decoding rows fed (tokens this beat produced for conversations past their prompt).
    pub rows: u64,
    /// Rows that fed prompt tokens this beat (chunks or single tokens).
    pub prompt_rows: u64,
    /// `step_rows` and `prefill_row` calls.
    pub steps: u64,
    /// The `step_rows` calls' graph stats, summed over every row's out.
    pub launches: u64,
    pub submits: u64,
    pub graph_ns: u64,
    pub record_ns: u64,
    pub step_chip_ns: u64,
    /// Host wall of every step call (prefill chunks included).
    pub step_ns: u64,
    /// Chip time of every step (prefill chunks included).
    pub chip_ns: u64,
}

/// One decoding beat of the per-token receipt (`total_ns`, `step_ns`, `graph_ns`, `t0`, `t1` on
/// the host clock; `chip_ns` the graphs' submit-to-marker spans, also measured on the host).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct TokenRecord {
    pub schema: String,
    pub seq: u64,
    pub beat: u64,
    pub rows: u64,
    pub prompt_rows: u64,
    pub steps: u64,
    pub launches: u64,
    pub submits: u64,
    pub t0: u64,
    pub t1: u64,
    pub total_ns: u64,
    pub step_ns: u64,
    pub graph_ns: u64,
    pub record_ns: u64,
    pub step_chip_ns: u64,
    pub chip_ns: u64,
    pub host_overhead_ns: u64,
}

/// The host phases of one beat, in order.
pub const HOST_PHASES: [&str; 11] = [
    "sched_join",
    "sample",
    "sched_feed",
    "step_enter",
    "embed",
    "graph_record",
    "graph_submit_wait",
    "graph_exit",
    "logits_read",
    "step_return",
    "sched_post",
];
/// The device categories [`categories`] assigns, in summary order. They are the groups of
/// docs/speed/Q36_CEILING.md section 10, with the exact head split into its screen and its
/// recheck and the greedy pick taken out of the glue.
pub const DEV_CATEGORIES: [&str; 10] = [
    "experts",
    "gdn",
    "attention",
    "router",
    "head_screen",
    "head_recheck",
    "head_plain",
    "token_pick",
    "glue",
    "unknown",
];

/// One span: `name` from `t0` to `t1` (ns).
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Span {
    pub name: String,
    pub t0: u64,
    pub t1: u64,
}

/// One single-row decode beat.
#[derive(Serialize, Deserialize, Debug, Clone, PartialEq, Eq)]
pub struct Record {
    pub schema: String,
    /// Records written by this door before this one: a missing or repeated record breaks the run.
    pub seq: u64,
    /// The door's beat number.
    pub beat: u64,
    /// The row stepped.
    pub row: usize,
    /// That conversation's decode steps before this one.
    pub step: u64,
    /// The graph's stats for the token.
    pub launches: u32,
    /// Submissions of the token's list (1 normally; 0 in a file written before this field).
    #[serde(default)]
    pub submits: u32,
    /// Host clock, the first submit to the last completion marker read back: the chip's work plus
    /// the submit and the marker poll around it.
    pub chip_ns: u64,
    pub record_ns: u64,
    pub total_ns: u64,
    pub host: Vec<Span>,
    pub dev: Vec<Span>,
}

fn span(name: &str, t0: u64, t1: u64) -> Span {
    Span {
        name: name.into(),
        t0,
        t1,
    }
}

/// The open part of a beat's record: every host span up to `step_return`. `sched_post` closes it
/// at the next beat's join or at idle.
fn open_record(
    marks: [u64; 3],
    beat: u64,
    row: usize,
    step: u64,
    call: (u64, u64),
    stats: &GraphStats,
    tl: &RowTimeline,
) -> Record {
    let [join, sample, feed] = marks;
    let [h0, h1, h2, h3] = tl.host;
    let rec_end = h1 + stats.record_ns;
    let wait_end = h1 + stats.total_ns;
    let host = vec![
        span("sched_join", join, sample),
        span("sample", sample, feed),
        span("sched_feed", feed, call.0),
        span("step_enter", call.0, h0),
        span("embed", h0, h1),
        span("graph_record", h1, rec_end),
        span("graph_submit_wait", rec_end, wait_end),
        span("graph_exit", wait_end, h2),
        span("logits_read", h2, h3),
        span("step_return", h3, call.1),
    ];
    let dev = tl
        .kinds
        .iter()
        .enumerate()
        .map(|(i, k)| span(k, tl.stamps[i], tl.stamps[i + 1]))
        .collect();
    Record {
        schema: SCHEMA.into(),
        seq: 0,
        beat,
        row,
        step,
        launches: stats.launches,
        submits: stats.submits,
        chip_ns: stats.chip_ns,
        record_ns: stats.record_ns,
        total_ns: stats.total_ns,
        host,
        dev,
    }
}

/// The batch door's timeline writer (one per door, held only while [`ENV`] is set).
pub struct Recorder {
    path: PathBuf,
    kind: Kind,
    out: Option<BufWriter<File>>,
    /// join, sample, feed marks of the current beat.
    marks: [u64; 3],
    pending: Option<Record>,
    pending_beat: Option<BeatSums>,
    written: u64,
    failed: u64,
    reported: (u64, u64),
}

impl Recorder {
    /// Appends `kind` records to `path` (created if missing).
    pub fn open(path: &Path, kind: Kind) -> Result<Self, String> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("decode timeline: {}: {e}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            kind,
            out: Some(BufWriter::new(f)),
            marks: [0; 3],
            pending: None,
            pending_beat: None,
            written: 0,
            failed: 0,
            reported: (0, 0),
        })
    }

    /// True when the door must turn the chip stamps on ([`Kind::Launch`]).
    pub fn stamps(&self) -> bool {
        self.kind == Kind::Launch
    }

    /// A decoding beat ended ([`Kind::Token`] only; others ignore it). The record closes at
    /// the next beat's join or at idle.
    pub fn beat(&mut self, sums: BeatSums) {
        if self.kind == Kind::Token && sums.rows > 0 {
            self.pending_beat = Some(sums);
        }
    }

    /// The path from [`ENV`], when set and not empty.
    pub fn path_from_env() -> Option<PathBuf> {
        std::env::var_os(ENV)
            .filter(|v| !v.is_empty())
            .map(PathBuf::from)
    }

    /// A beat's join starts: it closes the previous beat's record.
    pub fn join(&mut self, now: u64) {
        self.finish(now);
        self.marks = [now; 3];
    }

    /// The beat's sampling starts.
    pub fn sample(&mut self, now: u64) {
        self.marks[1] = now;
    }

    /// The beat's feed is being built.
    pub fn feed(&mut self, now: u64) {
        self.marks[2] = now;
    }

    /// The beat stepped one decoding row: `call` is the step call's start and end, `tl` the
    /// row's timeline for the token.
    pub fn stepped(
        &mut self,
        beat: u64,
        row: usize,
        step: u64,
        call: (u64, u64),
        stats: &GraphStats,
        tl: Result<RowTimeline, String>,
    ) {
        match tl {
            Ok(tl) if tl.stamps.len() == tl.kinds.len() + 1 => {
                self.pending = Some(open_record(self.marks, beat, row, step, call, stats, &tl));
            }
            Ok(tl) => {
                self.failed += 1;
                eprintln!(
                    "aien-serve-http: decode timeline: beat {beat}: {} stamps for {} launches",
                    tl.stamps.len(),
                    tl.kinds.len()
                );
            }
            Err(e) => {
                self.failed += 1;
                eprintln!("aien-serve-http: decode timeline: beat {beat}: {e}");
            }
        }
    }

    /// The door went idle: closes the last record and reports the counts.
    pub fn idle(&mut self, now: u64) {
        self.finish(now);
        if (self.written, self.failed) != self.reported {
            self.reported = (self.written, self.failed);
            eprintln!(
                "aien-serve-http: decode timeline: {} beats written to {}, {} not recorded",
                self.written,
                self.path.display(),
                self.failed
            );
        }
    }

    fn finish(&mut self, end: u64) {
        if let Some(s) = self.pending_beat.take() {
            let t0 = self.marks[0];
            let total_ns = end.saturating_sub(t0);
            let r = TokenRecord {
                schema: TOKEN_SCHEMA.into(),
                seq: self.written,
                beat: s.beat,
                rows: s.rows,
                prompt_rows: s.prompt_rows,
                steps: s.steps,
                launches: s.launches,
                submits: s.submits,
                t0,
                t1: end,
                total_ns,
                step_ns: s.step_ns,
                graph_ns: s.graph_ns,
                record_ns: s.record_ns,
                step_chip_ns: s.step_chip_ns,
                chip_ns: s.chip_ns,
                host_overhead_ns: total_ns.saturating_sub(s.chip_ns),
            };
            self.write(serde_json::to_string(&r).expect("a record serializes"));
        }
        let Some(mut r) = self.pending.take() else {
            return;
        };
        let post0 = r.host.last().map_or(end, |s| s.t1);
        r.host.push(span("sched_post", post0, end));
        r.seq = self.written;
        self.write(serde_json::to_string(&r).expect("a record serializes"));
    }

    fn write(&mut self, line: String) {
        let Some(out) = self.out.as_mut() else {
            return;
        };
        let res = writeln!(out, "{line}").and_then(|()| out.flush());
        match res {
            Ok(()) => self.written += 1,
            Err(e) => {
                eprintln!(
                    "aien-serve-http: decode timeline: write to {} failed: {e}; recording stops",
                    self.path.display()
                );
                self.out = None;
            }
        }
    }
}

/// Device categories, one per launch. Before the last `down` launch: section 10's groups. After
/// it (the head): with the exact head an `lm_head` launch follows an `argmax`, so the `lm_head`
/// launches before the head's first `argmax` are the FP8 screen and those after it the BF16
/// recheck; without one every `lm_head` is `head_plain`. The last two launches, when both are
/// `argmax`, are the greedy pick.
pub fn categories(kinds: &[String]) -> Vec<&'static str> {
    let tail = kinds.iter().rposition(|k| k == "down").map_or(0, |i| i + 1);
    let first_am = kinds[tail..]
        .iter()
        .position(|k| k == "argmax")
        .map(|i| tail + i);
    let exact = first_am.is_some_and(|a| kinds[a..].iter().any(|k| k == "lm_head"));
    let n = kinds.len();
    let pick = n >= 2 && kinds[n - 2] == "argmax" && kinds[n - 1] == "argmax";
    kinds
        .iter()
        .enumerate()
        .map(|(i, k)| match k.as_str() {
            _ if pick && i + 2 >= n => "token_pick",
            "lm_head" if !exact => "head_plain",
            "lm_head" if first_am.is_some_and(|a| i < a) => "head_screen",
            "lm_head" => "head_recheck",
            "gate_up" | "down" => "experts",
            "mix" | "gdn" | "ab" => "gdn",
            "o_proj" | "attn" => "attention",
            "router" => "router",
            "res" | "swiglu" | "argmax" | "gather" | "gate" | "norm" | "perm" | "group" => "glue",
            _ => "unknown",
        })
        .collect()
}

fn dur(s: &Span) -> u64 {
    s.t1.saturating_sub(s.t0)
}

/// Holds a timeline to its shape. Empty: green. For every record: the schema; the host phases
/// complete, in order, each `t1 >= t0`, each starting where the last ended; the device spans
/// one per launch, contiguous; the device span at most the graph's chip time, and that at most
/// its submit-and-wait span (durations only: two clocks); every record the same launch-kind
/// census as the first; no unknown kind. Across records: no overlap, and consecutive beats
/// touch.
pub fn check(records: &[Record]) -> Vec<String> {
    let mut bad = Vec::new();
    if records.is_empty() {
        bad.push("no records".into());
    }
    fn census(r: &Record) -> Vec<&str> {
        let mut c: Vec<&str> = r.dev.iter().map(|s| s.name.as_str()).collect();
        c.sort_unstable();
        c
    }
    let first_census = records.first().map(census);
    for (i, r) in records.iter().enumerate() {
        let at = format!("record {i} (beat {})", r.beat);
        if r.seq != i as u64 {
            bad.push(format!(
                "{at}: seq {} where {i} belongs (a record is missing or repeated)",
                r.seq
            ));
        }
        if r.schema != SCHEMA {
            bad.push(format!("{at}: schema {:?}", r.schema));
        }
        let names: Vec<&str> = r.host.iter().map(|s| s.name.as_str()).collect();
        if names != HOST_PHASES {
            bad.push(format!("{at}: host phases {names:?}"));
        }
        for (what, spans) in [("host", &r.host), ("dev", &r.dev)] {
            for (j, s) in spans.iter().enumerate() {
                if s.t1 < s.t0 {
                    bad.push(format!("{at}: {what} {j} {} ends before it starts", s.name));
                }
            }
            for (j, w) in spans.windows(2).enumerate() {
                if w[0].t1 != w[1].t0 {
                    bad.push(format!(
                        "{at}: {what} {j} {} ends at {}, {} starts at {}",
                        w[0].name, w[0].t1, w[1].name, w[1].t0
                    ));
                }
            }
        }
        if r.dev.len() != r.launches as usize {
            bad.push(format!(
                "{at}: {} device spans for {} launches",
                r.dev.len(),
                r.launches
            ));
        }
        if let (Some(a), Some(b)) = (r.dev.first(), r.dev.last()) {
            let dev_span = b.t1.saturating_sub(a.t0);
            if dev_span > r.chip_ns {
                bad.push(format!(
                    "{at}: device span {dev_span} ns > chip_ns {}",
                    r.chip_ns
                ));
            }
        }
        if let Some(w) = r.host.iter().find(|s| s.name == "graph_submit_wait") {
            if r.chip_ns > dur(w) {
                bad.push(format!(
                    "{at}: chip_ns {} > graph_submit_wait {}",
                    r.chip_ns,
                    dur(w)
                ));
            }
        }
        if let Some(c) = &first_census {
            if &census(r) != c {
                bad.push(format!("{at}: launch-kind census differs from record 0"));
            }
        }
        let kinds: Vec<String> = r.dev.iter().map(|s| s.name.clone()).collect();
        if categories(&kinds).contains(&"unknown") {
            bad.push(format!("{at}: unknown launch kind"));
        }
        if i > 0 {
            let p = &records[i - 1];
            if let (Some(pe), Some(s)) = (p.host.last(), r.host.first()) {
                if s.t0 < pe.t1 {
                    bad.push(format!(
                        "{at}: starts {} before record {} ends",
                        pe.t1 - s.t0,
                        i - 1
                    ));
                }
                if r.beat == p.beat + 1 && s.t0 != pe.t1 {
                    bad.push(format!("{at}: beat follows beat {} with a gap", p.beat));
                }
            }
        }
    }
    bad
}

/// The DRA-66 0.5 ms gate, a sum of two medians over two arms of the same window: `hostLB`, the
/// median `host_overhead_ns` of the clean one-conversation beats of the per-token receipt's
/// `receipt-1conv` slice ([`summary_tokens`]), plus `submitPoll`, the median submit + marker
/// poll share `chip_ns - dev_span` of the stamped one-conversation timeline ([`summary`]). Both
/// summaries must be evaluated first, in the same Wolfram session. It returns
/// `{host ms, submit + poll ms, sum ms, verdict}`.
pub const GATE_WL: &str = "(* DRA-66 gate, sum of medians (hostLB: token receipt, receipt-1conv slice, clean 1-row beats; submitPoll: stamped timeline, single-submit beats): {host lower bound ms, submit + poll ms, sum ms, sum <= 0.5 ms} *) {N[hostLB / 10^6], N[submitPoll / 10^6], N[(hostLB + submitPoll) / 10^6], hostLB + submitPoll <= 500000}\n";

/// A Wolfram-ready summary: one association of per-beat integer nanosecond lists (`period`, every
/// host phase, every device category, `dev_span`, `sync` = `graph_submit_wait - dev_span`,
/// `submits`, `chip_ns`, `submit_poll` = `chip_ns - dev_span`), plus the queries that reduce it,
/// ending with [`GATE_WL`]. The arithmetic is left to Wolfram.
///
/// `submit_poll` is the host time [`Record::chip_ns`] (the same graph stat a [`TokenRecord`]
/// sums into its `chip_ns`) holds besides the chip's work: the queue submit, the chip's start,
/// the completion marker's poll (the served 2000 us spin, then 50 us sleeps) and its read back.
/// `chip_ns` and `dev_span` both carry the stamps' chip cost, so the
/// difference does not. With more than one submit the device span also covers the host's
/// emit between the chunks, which `chip_ns` does not, so only single-submit beats count.
pub fn summary(records: &[Record]) -> String {
    let mut keys: Vec<String> = vec!["period".into()];
    keys.extend(HOST_PHASES.iter().map(|p| format!("host:{p}")));
    keys.extend(DEV_CATEGORIES.iter().map(|c| format!("dev:{c}")));
    keys.extend(["dev_span", "sync", "submits", "chip_ns", "submit_poll"].map(String::from));
    let mut cols: Vec<Vec<u64>> = vec![Vec::new(); keys.len()];
    for r in records {
        let mut row = Vec::with_capacity(keys.len());
        let period = match (r.host.first(), r.host.last()) {
            (Some(a), Some(b)) => b.t1.saturating_sub(a.t0),
            _ => 0,
        };
        row.push(period);
        for p in HOST_PHASES {
            row.push(r.host.iter().filter(|s| s.name == p).map(dur).sum());
        }
        let kinds: Vec<String> = r.dev.iter().map(|s| s.name.clone()).collect();
        let cats = categories(&kinds);
        for c in DEV_CATEGORIES {
            row.push(
                r.dev
                    .iter()
                    .zip(&cats)
                    .filter(|(_, k)| **k == c)
                    .map(|(s, _)| dur(s))
                    .sum(),
            );
        }
        let dev_span = match (r.dev.first(), r.dev.last()) {
            (Some(a), Some(b)) => b.t1.saturating_sub(a.t0),
            _ => 0,
        };
        let wait = r
            .host
            .iter()
            .find(|s| s.name == "graph_submit_wait")
            .map_or(0, dur);
        row.push(dev_span);
        row.push(wait.saturating_sub(dev_span));
        row.push(u64::from(r.submits));
        row.push(r.chip_ns);
        row.push(r.chip_ns.saturating_sub(dev_span));
        for (c, v) in cols.iter_mut().zip(row) {
            c.push(v);
        }
    }
    let mut s = format!(
        "(* {SCHEMA}: {} beats; integer ns per beat; host clock for host:*, period, sync, chip_ns, submit_poll; chip PTIMER for dev:*, dev_span *)\n",
        records.len()
    );
    s.push_str("timeline = <|\n");
    for (i, (k, c)) in keys.iter().zip(&cols).enumerate() {
        let vals: Vec<String> = c.iter().map(u64::to_string).collect();
        let comma = if i + 1 < keys.len() { "," } else { "" };
        s.push_str(&format!("  \"{k}\" -> {{{}}}{comma}\n", vals.join(", ")));
    }
    s.push_str("|>;\n");
    s.push_str("(* ms per beat, medians: *) N[Median /@ timeline / 10^6]\n");
    s.push_str(
        "(* outside the chip + chip + sync vs the period: *) N[{Median[timeline[\"period\"] - timeline[\"host:graph_submit_wait\"]], Median[timeline[\"dev_span\"]], Median[timeline[\"sync\"]], Median[timeline[\"period\"]]} / 10^6]\n",
    );
    s.push_str("(* residual against the recorded 20.7 ms step: *) N[207/10 - Median[timeline[\"period\"]] / 10^6]\n");
    s.push_str("(* DRA-66 submit + poll share per token, chip_ns - dev_span over single-submit beats: {beats, median ms} *) one = Flatten[Position[timeline[\"submits\"], 1]]; submitPoll = Median[timeline[\"submit_poll\"][[one]]]; {Length[one], N[submitPoll / 10^6]}\n");
    s.push_str(GATE_WL);
    s
}

/// Holds a per-token receipt to its shape. Empty: red. For every record: the schema; `seq` its
/// index (a dropped or repeated record breaks the run); at least one decoding row and one step;
/// `total_ns = t1 - t0`; `record_ns <= graph_ns`, `step_chip_ns <= graph_ns`,
/// `step_chip_ns <= chip_ns`, `graph_ns <= step_ns`, `chip_ns <= step_ns <= total_ns`;
/// `host_overhead_ns = total_ns - chip_ns`; a beat with no prompt and one step submitted at
/// least once and at most once per row. Across records: no overlap, and consecutive beats touch.
pub fn check_tokens(records: &[TokenRecord]) -> Vec<String> {
    let mut bad = Vec::new();
    if records.is_empty() {
        bad.push("no records".into());
    }
    for (i, r) in records.iter().enumerate() {
        let at = format!("record {i} (beat {})", r.beat);
        if r.schema != TOKEN_SCHEMA {
            bad.push(format!("{at}: schema {}", r.schema));
        }
        if r.seq != i as u64 {
            bad.push(format!(
                "{at}: seq {} (a record is missing or repeated)",
                r.seq
            ));
        }
        if r.rows == 0 || r.steps == 0 {
            bad.push(format!("{at}: {} rows, {} steps", r.rows, r.steps));
        }
        if r.t1 < r.t0 || r.total_ns != r.t1 - r.t0 {
            bad.push(format!(
                "{at}: total {} for t0 {} t1 {}",
                r.total_ns, r.t0, r.t1
            ));
        }
        let order = [
            ("record_ns <= graph_ns", r.record_ns <= r.graph_ns),
            ("step_chip_ns <= graph_ns", r.step_chip_ns <= r.graph_ns),
            ("step_chip_ns <= chip_ns", r.step_chip_ns <= r.chip_ns),
            ("graph_ns <= step_ns", r.graph_ns <= r.step_ns),
            ("chip_ns <= step_ns", r.chip_ns <= r.step_ns),
            ("step_ns <= total_ns", r.step_ns <= r.total_ns),
        ];
        for (what, ok) in order {
            if !ok {
                bad.push(format!("{at}: not {what}"));
            }
        }
        if r.host_overhead_ns != r.total_ns.saturating_sub(r.chip_ns) || r.chip_ns > r.total_ns {
            bad.push(format!(
                "{at}: host_overhead {} for total {} chip {}",
                r.host_overhead_ns, r.total_ns, r.chip_ns
            ));
        }
        if r.prompt_rows == 0 && r.steps == 1 && (r.submits == 0 || r.submits > r.rows) {
            bad.push(format!("{at}: {} submits for {} rows", r.submits, r.rows));
        }
        if i > 0 {
            let p = &records[i - 1];
            if r.t0 < p.t1 {
                bad.push(format!(
                    "{at}: starts {} before record {} ends",
                    p.t1 - r.t0,
                    i - 1
                ));
            }
            if r.beat == p.beat + 1 && r.t0 != p.t1 {
                bad.push(format!("{at}: beat follows beat {} with a gap", p.beat));
            }
        }
    }
    bad
}

/// The per-token receipt's Wolfram-ready summary: per-beat integer lists, then the queries. A
/// beat yields one token to each of its `rows` conversations, so a beat's host overhead is each
/// of those tokens' host time. `hostLB` is the median host overhead of the clean
/// one-conversation beats (`rows` 1, `prompt_rows` 0): a LOWER bound of the host time per token,
/// because `chip_ns` also holds the submit and the completion marker's poll. The DRA-66 gate
/// adds the stamped timeline's `submitPoll` to it ([`GATE_WL`]); `hostLB` alone is never the
/// verdict.
pub fn summary_tokens(records: &[TokenRecord]) -> String {
    type Col = fn(&TokenRecord) -> u64;
    let cols: [(&str, Col); 11] = [
        ("rows", |r| r.rows),
        ("prompt_rows", |r| r.prompt_rows),
        ("steps", |r| r.steps),
        ("submits", |r| r.submits),
        ("total_ns", |r| r.total_ns),
        ("step_ns", |r| r.step_ns),
        ("graph_ns", |r| r.graph_ns),
        ("record_ns", |r| r.record_ns),
        ("step_chip_ns", |r| r.step_chip_ns),
        ("chip_ns", |r| r.chip_ns),
        ("host_overhead_ns", |r| r.host_overhead_ns),
    ];
    let mut s = format!(
        "(* {TOKEN_SCHEMA}: {} decoding beats; integer ns, host clock *)\nreceipt = <|\n",
        records.len()
    );
    for (i, (k, f)) in cols.iter().enumerate() {
        let vals: Vec<String> = records.iter().map(|r| f(r).to_string()).collect();
        let comma = if i + 1 < cols.len() { "," } else { "" };
        s.push_str(&format!("  \"{k}\" -> {{{}}}{comma}\n", vals.join(", ")));
    }
    s.push_str("|>;\n");
    s.push_str("clean[k_] := Pick[Range[Length[receipt[\"rows\"]]], MapThread[#1 == k && #2 == 0 &, {receipt[\"rows\"], receipt[\"prompt_rows\"]}]];\n");
    s.push_str("(* DRA-66 host lower bound per token at 1 conversation (wall - chip_ns; chip_ns holds the submit and the marker poll, see submitPoll): {beats, median ms} *) hostLB = Median[receipt[\"host_overhead_ns\"][[clean[1]]]]; {Length[clean[1]], N[hostLB / 10^6]}\n");
    s.push_str("(* per conversations k, clean beats: {beats, median total, chip, host overhead ms} *) Table[{k, Length[clean[k]], N[Median[receipt[\"total_ns\"][[clean[k]]]] / 10^6], N[Median[receipt[\"chip_ns\"][[clean[k]]]] / 10^6], N[Median[receipt[\"host_overhead_ns\"][[clean[k]]]] / 10^6]}, {k, Select[Union[receipt[\"rows\"]], Length[clean[#]] > 0 &]}]\n");
    s.push_str("(* host overhead at 1 conversation by where it sits (ms, medians): outside the step, step outside the graph, graph outside record and chip, record *) With[{c = clean[1], r = receipt}, N[{Median[(r[\"total_ns\"] - r[\"step_ns\"])[[c]]], Median[(r[\"step_ns\"] - r[\"graph_ns\"])[[c]]], Median[(r[\"graph_ns\"] - r[\"record_ns\"] - r[\"step_chip_ns\"])[[c]]], Median[r[\"record_ns\"][[c]]]} / 10^6]]\n");
    s
}

/// The token checker's mutants and slice ([`run_check`]).
#[derive(Default)]
struct TokenOpts<'a> {
    drop: Option<&'a str>,
    bend: Option<&'a str>,
    slice: Option<&'a str>,
}

fn run_check_tokens(path: &str, o: &TokenOpts, text: &str) -> u8 {
    let usage = "token receipt: --drop REC:record:0 (REC below the last record), --bend REC:chip | host:NS, --slice A:B (0 <= A < B <= records)";
    let mut records: Vec<TokenRecord> = match read_jsonl_as(text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 2;
        }
    };
    if let Some(d) = o.drop {
        let rec = match d.split(':').collect::<Vec<_>>().as_slice() {
            [r, "record", _] => r.parse::<usize>().ok().filter(|&r| r + 1 < records.len()),
            _ => None,
        };
        let Some(r) = rec else {
            eprintln!("--drop {d}: {usage}");
            return 2;
        };
        records.remove(r);
        eprintln!("negative control: dropped {d}");
    }
    if let Some(b) = o.bend {
        match b.split(':').collect::<Vec<_>>().as_slice() {
            // The accounting mutant: seq stays dense, one beat's chip time moves by 1 ns while
            // its host share stays put. It must come out red.
            [r, "chip"] if r.parse::<usize>().is_ok_and(|r| r < records.len()) => {
                records[r.parse::<usize>().expect("checked")].chip_ns += 1;
                eprintln!("negative control: bent {b} (chip_ns + 1, host_overhead_ns kept)");
            }
            // The synthetic receipt: every beat NS ns longer on the host, consistently (t0, t1,
            // total and host share), so the shape stays green and the 0.5 ms gate must turn.
            ["host", ns] if ns.parse::<u64>().is_ok() => {
                let ns = ns.parse::<u64>().expect("checked");
                for (i, r) in records.iter_mut().enumerate() {
                    let i = i as u64;
                    r.t0 += i * ns;
                    r.t1 += (i + 1) * ns;
                    r.total_ns += ns;
                    r.host_overhead_ns += ns;
                }
                eprintln!("synthetic receipt: bent {b} (every beat {ns} ns more host time)");
            }
            _ => {
                eprintln!("--bend {b}: {usage}");
                return 2;
            }
        }
    }
    let range = match o.slice {
        None => 0..records.len(),
        Some(s) => {
            let ab = s
                .split_once(':')
                .and_then(|(a, b)| a.parse::<usize>().ok().zip(b.parse::<usize>().ok()))
                .filter(|&(a, b)| a < b && b <= records.len());
            let Some((a, b)) = ab else {
                eprintln!("--slice {s}: {usage}");
                return 2;
            };
            a..b
        }
    };
    let bad = check_tokens(&records);
    if bad.is_empty() {
        eprintln!(
            "token receipt GREEN: {} records, {path}; summary over records {}..{}",
            records.len(),
            range.start,
            range.end
        );
        if o.slice.is_some() {
            println!(
                "(* records {}..{} (seq) of {path}; the whole file checked green *)",
                range.start, range.end
            );
        }
        print!("{}", summary_tokens(&records[range]));
        0
    } else {
        for b in bad.iter().take(20) {
            eprintln!("RED: {b}");
        }
        eprintln!(
            "token receipt RED: {} findings over {} records, {path}",
            bad.len(),
            records.len()
        );
        1
    }
}

/// `aien-serve-http --check-decode-timeline FILE [--drop SPEC] [--bend SPEC] [--slice A:B]`:
/// [`check`] and, when green, [`summary`] on stdout. `--drop` deletes one span (or, as
/// `REC:record:0`, the whole record) before checking: the negative control, which must come out
/// red. A file whose first record is a [`TOKEN_SCHEMA`] receipt goes to [`check_tokens`] /
/// [`summary_tokens`]: `--drop` then takes `REC:record:0` only; `--bend REC:chip` adds 1 ns to
/// one beat's `chip_ns` (a red control: seq stays dense); `--bend host:NS` adds NS ns of host
/// time to every beat consistently (green shape, the synthetic red arm of the 0.5 ms gate);
/// `--slice A:B` checks the whole file and summarizes records `A..B` only (one gate's beats).
/// Exit 0 green, 1 red, 2 usage or an unreadable file.
pub fn run_check(args: &[String]) -> u8 {
    let usage = "usage: aien-serve-http --check-decode-timeline FILE [--drop REC:host|dev|record:IDX] [--bend REC:chip|host:NS] [--slice A:B] (--bend and --slice: token receipts)";
    let Some((path, rest)) = args.split_first() else {
        eprintln!("{usage}");
        return 2;
    };
    let mut o = TokenOpts::default();
    for pair in rest.chunks(2) {
        let slot = match pair {
            [f, _] if f == "--drop" => &mut o.drop,
            [f, _] if f == "--bend" => &mut o.bend,
            [f, _] if f == "--slice" => &mut o.slice,
            _ => {
                eprintln!("{usage}");
                return 2;
            }
        };
        if slot.replace(pair[1].as_str()).is_some() {
            eprintln!("{} given twice; {usage}", pair[0]);
            return 2;
        }
    }
    let drop = o.drop;
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 2;
        }
    };
    let first = text.lines().find(|l| !l.trim().is_empty());
    if first.is_some_and(|l| l.contains(TOKEN_SCHEMA)) {
        return run_check_tokens(path, &o, &text);
    }
    if o.bend.is_some() || o.slice.is_some() {
        eprintln!("{path}: a launch timeline takes --drop only; {usage}");
        return 2;
    }
    let mut records = match read_jsonl(&text) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 2;
        }
    };
    if let Some(d) = drop {
        let parts: Vec<&str> = d.split(':').collect();
        let parsed = match parts.as_slice() {
            [r, w, i] => r
                .parse::<usize>()
                .ok()
                .zip(i.parse::<usize>().ok())
                .map(|(r, i)| (r, *w, i)),
            _ => None,
        };
        let ok = match parsed {
            Some((r, "record", _)) if r < records.len() => {
                records.remove(r);
                true
            }
            Some((r, "host", i)) if r < records.len() && i < records[r].host.len() => {
                records[r].host.remove(i);
                true
            }
            Some((r, "dev", i)) if r < records.len() && i < records[r].dev.len() => {
                records[r].dev.remove(i);
                true
            }
            _ => false,
        };
        if !ok {
            eprintln!("--drop {d}: no such span; {usage}");
            return 2;
        }
        eprintln!("negative control: dropped {d}");
    }
    let bad = check(&records);
    if bad.is_empty() {
        eprintln!("decode timeline GREEN: {} records, {path}", records.len());
        print!("{}", summary(&records));
        0
    } else {
        for b in bad.iter().take(20) {
            eprintln!("RED: {b}");
        }
        eprintln!(
            "decode timeline RED: {} findings over {} records, {path}",
            bad.len(),
            records.len()
        );
        1
    }
}

/// Reads a JSONL timeline.
pub fn read_jsonl(text: &str) -> Result<Vec<Record>, String> {
    read_jsonl_as(text)
}

/// Reads JSONL records of any one type.
pub fn read_jsonl_as<T: serde::de::DeserializeOwned>(text: &str) -> Result<Vec<T>, String> {
    text.lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| serde_json::from_str(l).map_err(|e| format!("line {}: {e}", i + 1)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A 40-layer exact-head kind sequence shaped like the graph's (3 GDN : 1 attention).
    pub(crate) fn kinds40() -> Vec<String> {
        let mut k = Vec::new();
        for l in 0..40 {
            k.push("norm");
            if l % 4 == 3 {
                k.extend(["mix", "attn", "gate", "o_proj"]);
            } else {
                k.extend(["mix", "ab", "gdn", "o_proj"]);
            }
            k.extend([
                "res", "norm", "router", "gate_up", "swiglu", "down", "gather", "res",
            ]);
        }
        k.extend([
            "norm", "lm_head", "lm_head", "lm_head", "argmax", "argmax", "lm_head", "argmax",
            "argmax",
        ]);
        k.into_iter().map(String::from).collect()
    }

    fn timeline(kinds: Vec<String>, base: u64) -> RowTimeline {
        let stamps = (0..=kinds.len() as u64).map(|i| base + i * 1000).collect();
        RowTimeline {
            host: [110, 120, 120 + 900_000, 120 + 900_010],
            stamps,
            kinds,
        }
    }

    fn records(n: usize) -> Vec<Record> {
        let mut rec = Recorder {
            path: PathBuf::from("/dev/null"),
            kind: Kind::Launch,
            out: None,
            marks: [0; 3],
            pending: None,
            pending_beat: None,
            written: 0,
            failed: 0,
            reported: (0, 0),
        };
        let mut out = Vec::new();
        let mut t = 0u64;
        for b in 0..n as u64 {
            rec.join(t);
            if let Some(r) = rec.pending.take() {
                out.push(r);
            }
            rec.marks = [t; 3];
            rec.sample(t + 10);
            rec.feed(t + 50);
            let k = kinds40();
            let launches = k.len() as u32;
            let mut tl = timeline(k, 5_000_000 * b);
            tl.host = tl.host.map(|h| h + t);
            let stats = GraphStats {
                launches,
                submits: 1,
                chip_ns: launches as u64 * 1000 + 40_000,
                record_ns: 300_000,
                total_ns: 899_000,
                ..Default::default()
            };
            rec.stepped(b, 0, b, (t + 100, t + 900_200), &stats, Ok(tl));
            let mut r = rec.pending.take().expect("a record");
            r.host.push(span("sched_post", t + 900_200, t + 1_000_000));
            r.seq = b;
            out.push(r);
            t += 1_000_000;
        }
        out
    }

    #[test]
    fn complete_beats_are_green() {
        let r = records(5);
        assert_eq!(check(&r), Vec::<String>::new());
        assert_eq!(
            read_jsonl(
                &r.iter()
                    .map(|x| serde_json::to_string(x).unwrap() + "\n")
                    .collect::<String>()
            )
            .unwrap(),
            r
        );
    }

    #[test]
    fn dropping_one_host_span_is_red() {
        let mut r = records(3);
        r[1].host.remove(6);
        let bad = check(&r);
        assert!(bad.iter().any(|b| b.contains("host phases")), "{bad:?}");
    }

    #[test]
    fn dropping_one_device_span_is_red() {
        let mut r = records(3);
        r[2].dev.remove(100);
        let bad = check(&r);
        assert!(
            bad.iter().any(|b| b.contains("device spans for")),
            "{bad:?}"
        );
        assert!(bad.iter().any(|b| b.contains("dev 99")), "{bad:?}");
    }

    #[test]
    fn dropping_one_record_is_red() {
        let mut r = records(4);
        r.remove(2);
        let bad = check(&r);
        assert!(
            bad.iter().any(|b| b.contains("missing or repeated")),
            "{bad:?}"
        );
    }

    #[test]
    fn overlap_and_gap_are_red() {
        let mut r = records(2);
        r[1].host[0].t0 -= 5;
        assert!(check(&r).iter().any(|b| b.contains("before record 0 ends")));
        let mut r = records(2);
        r[1].host[0].t0 += 5;
        assert!(check(&r).iter().any(|b| b.contains("with a gap")));
    }

    #[test]
    fn exact_head_is_split() {
        let k = kinds40();
        let c = categories(&k);
        let n = k.len();
        assert_eq!(
            &c[n - 9..],
            [
                "glue",
                "head_screen",
                "head_screen",
                "head_screen",
                "glue",
                "glue",
                "head_recheck",
                "token_pick",
                "token_pick"
            ]
        );
        assert_eq!(c.iter().filter(|x| **x == "experts").count(), 80);
        assert_eq!(c.iter().filter(|x| **x == "router").count(), 40);
        let plain: Vec<String> = ["down", "norm", "lm_head", "argmax", "argmax"]
            .map(String::from)
            .into();
        assert_eq!(
            categories(&plain),
            ["experts", "glue", "head_plain", "token_pick", "token_pick"]
        );
    }

    #[test]
    fn summary_lists_every_beat() {
        let r = records(4);
        let s = summary(&r);
        assert!(
            s.contains("\"period\" -> {1000000, 1000000, 1000000, 1000000}"),
            "{s}"
        );
        assert!(
            s.contains("\"dev:head_recheck\" -> {1000, 1000, 1000, 1000}"),
            "{s}"
        );
        assert!(s.contains("\"submits\" -> {1, 1, 1, 1}"), "{s}");
        assert!(
            s.contains("\"submit_poll\" -> {40000, 40000, 40000, 40000}"),
            "{s}"
        );
        assert!(s.ends_with(GATE_WL), "{s}");
    }

    /// A launch record written before `submits` existed (#787's host timeline) still reads, as 0.
    #[test]
    fn record_without_submits_reads_as_zero() {
        let r = records(1);
        let line = serde_json::to_string(&r[0]).unwrap();
        let old = line.replace("\"submits\":1,", "");
        assert_ne!(old, line);
        let back = read_jsonl(&old).unwrap();
        assert_eq!(back[0].submits, 0);
        assert_eq!(back[0].chip_ns, r[0].chip_ns);
    }

    /// `n` consistent single-row token records, 1 ms apart: 0.6 ms chip, 0.4 ms host each.
    fn tokens(n: u64) -> Vec<TokenRecord> {
        (0..n)
            .map(|i| TokenRecord {
                schema: TOKEN_SCHEMA.into(),
                seq: i,
                beat: i + 1,
                rows: 1,
                prompt_rows: 0,
                steps: 1,
                launches: 40,
                submits: 1,
                t0: 1_000_000 * i,
                t1: 1_000_000 * (i + 1),
                total_ns: 1_000_000,
                step_ns: 900_000,
                graph_ns: 800_000,
                record_ns: 100_000,
                step_chip_ns: 600_000,
                chip_ns: 600_000,
                host_overhead_ns: 400_000,
            })
            .collect()
    }

    fn token_file(name: &str, recs: &[TokenRecord]) -> String {
        let p = std::env::temp_dir().join(format!("dra66-{}-{name}.jsonl", std::process::id()));
        let text: String = recs
            .iter()
            .map(|r| serde_json::to_string(r).unwrap() + "\n")
            .collect();
        std::fs::write(&p, text).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn cli(file: &str, extra: &[&str]) -> u8 {
        let mut a = vec![file.to_string()];
        a.extend(extra.iter().map(|s| s.to_string()));
        run_check(&a)
    }

    #[test]
    fn token_bends_and_slice() {
        let recs = tokens(6);
        assert_eq!(check_tokens(&recs), Vec::<String>::new());
        let f = token_file("bend", &recs);
        assert_eq!(cli(&f, &[]), 0);
        // The accounting mutant: one beat's chip moves, its host share does not. Red.
        assert_eq!(cli(&f, &["--bend", "3:chip"]), 1);
        // The synthetic receipt: 0.6 ms more host time on every beat keeps the shape green.
        assert_eq!(cli(&f, &["--bend", "host:600000"]), 0);
        let mut bent = recs.clone();
        for (i, r) in bent.iter_mut().enumerate() {
            let i = i as u64;
            r.t0 += i * 600_000;
            r.t1 += (i + 1) * 600_000;
            r.total_ns += 600_000;
            r.host_overhead_ns += 600_000;
        }
        assert_eq!(check_tokens(&bent), Vec::<String>::new());
        assert!(summary_tokens(&bent).contains("\"host_overhead_ns\" -> {1000000, "));
        // Usage: unknown bend, bad slice, a flag twice, a dangling flag.
        assert_eq!(cli(&f, &["--bend", "6:chip"]), 2);
        assert_eq!(cli(&f, &["--bend", "host:x"]), 2);
        assert_eq!(cli(&f, &["--slice", "4:4"]), 2);
        assert_eq!(cli(&f, &["--slice", "2:7"]), 2);
        assert_eq!(cli(&f, &["--bend", "1:chip", "--bend", "2:chip"]), 2);
        assert_eq!(cli(&f, &["--slice"]), 2);
        assert_eq!(cli(&f, &["--slice", "2:5"]), 0);
        // A dropped record stays red whatever the slice.
        assert_eq!(cli(&f, &["--drop", "1:record:0", "--slice", "2:4"]), 1);
        let _ = std::fs::remove_file(&f);
    }

    #[test]
    fn token_summary_covers_the_slice_only() {
        let mut recs = tokens(6);
        // Beats 2..5 carry a distinct host share; the summary over them must show only those.
        for r in &mut recs[2..5] {
            r.chip_ns -= 1_000;
            r.step_chip_ns -= 1_000;
            r.host_overhead_ns += 1_000;
        }
        assert_eq!(check_tokens(&recs), Vec::<String>::new());
        let s = summary_tokens(&recs[2..5]);
        assert!(
            s.contains("\"host_overhead_ns\" -> {401000, 401000, 401000}\n"),
            "{s}"
        );
        assert!(s.contains("hostLB = Median["), "{s}");
        assert!(
            !s.contains("<= 500000"),
            "the lower bound alone carries no verdict: {s}"
        );
    }

    #[test]
    fn launch_timeline_takes_drop_only() {
        let r = records(3);
        let p = std::env::temp_dir().join(format!("dra66-{}-launch.jsonl", std::process::id()));
        std::fs::write(
            &p,
            r.iter()
                .map(|x| serde_json::to_string(x).unwrap() + "\n")
                .collect::<String>(),
        )
        .unwrap();
        let f = p.to_string_lossy().into_owned();
        assert_eq!(cli(&f, &[]), 0);
        assert_eq!(cli(&f, &["--drop", "1:record:0"]), 1);
        assert_eq!(cli(&f, &["--bend", "1:chip"]), 2);
        assert_eq!(cli(&f, &["--slice", "0:2"]), 2);
        let _ = std::fs::remove_file(&p);
    }
}
