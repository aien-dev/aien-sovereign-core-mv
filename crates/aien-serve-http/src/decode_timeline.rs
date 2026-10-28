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
//! [`check`] holds a file to that shape, and `q36-decode-timeline-check` runs it and prints a
//! Wolfram-ready summary. Each stamp sits after a launch's wait-for-idle and adds a timestamp
//! report, so the stamps cost chip time: a timeline run is never a speed receipt.
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use aien_inference_abi::qwen38_omega::RowTimeline;
use aien_omega_q38::GraphStats;
use serde::{Deserialize, Serialize};

pub use aien_inference_abi::qwen38_omega::timeline_now_ns;

/// The switch: a path to append the timeline to. Unset: off.
pub const ENV: &str = "AIEN_Q36_DECODE_TIMELINE";
/// The record schema.
pub const SCHEMA: &str = "aien.q36-decode-timeline/1";
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
    out: Option<BufWriter<File>>,
    /// join, sample, feed marks of the current beat.
    marks: [u64; 3],
    pending: Option<Record>,
    written: u64,
    failed: u64,
    reported: (u64, u64),
}

impl Recorder {
    /// Appends to `path` (created if missing).
    pub fn open(path: &Path) -> Result<Self, String> {
        let f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)
            .map_err(|e| format!("decode timeline: {}: {e}", path.display()))?;
        Ok(Self {
            path: path.to_path_buf(),
            out: Some(BufWriter::new(f)),
            marks: [0; 3],
            pending: None,
            written: 0,
            failed: 0,
            reported: (0, 0),
        })
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
        let Some(mut r) = self.pending.take() else {
            return;
        };
        let post0 = r.host.last().map_or(end, |s| s.t1);
        r.host.push(span("sched_post", post0, end));
        r.seq = self.written;
        let Some(out) = self.out.as_mut() else {
            return;
        };
        let line = serde_json::to_string(&r).expect("a record serializes");
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
            bad.push(format!("{at}: {} device spans for {} launches", r.dev.len(), r.launches));
        }
        if let (Some(a), Some(b)) = (r.dev.first(), r.dev.last()) {
            let dev_span = b.t1.saturating_sub(a.t0);
            if dev_span > r.chip_ns {
                bad.push(format!("{at}: device span {dev_span} ns > chip_ns {}", r.chip_ns));
            }
        }
        if let Some(w) = r.host.iter().find(|s| s.name == "graph_submit_wait") {
            if r.chip_ns > dur(w) {
                bad.push(format!("{at}: chip_ns {} > graph_submit_wait {}", r.chip_ns, dur(w)));
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
                    bad.push(format!("{at}: starts {} before record {} ends", pe.t1 - s.t0, i - 1));
                }
                if r.beat == p.beat + 1 && s.t0 != pe.t1 {
                    bad.push(format!("{at}: beat follows beat {} with a gap", p.beat));
                }
            }
        }
    }
    bad
}

/// A Wolfram-ready summary: one association of per-beat integer nanosecond lists (`period`, every
/// host phase, every device category, `dev_span`, `sync` = `graph_submit_wait - dev_span`), plus
/// the queries that reduce it. The arithmetic is left to Wolfram.
pub fn summary(records: &[Record]) -> String {
    let mut keys: Vec<String> = vec!["period".into()];
    keys.extend(HOST_PHASES.iter().map(|p| format!("host:{p}")));
    keys.extend(DEV_CATEGORIES.iter().map(|c| format!("dev:{c}")));
    keys.push("dev_span".into());
    keys.push("sync".into());
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
        for (c, v) in cols.iter_mut().zip(row) {
            c.push(v);
        }
    }
    let mut s = format!(
        "(* {SCHEMA}: {} beats; integer ns per beat; host clock for host:*, period, sync; chip PTIMER for dev:*, dev_span *)\n",
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
    s
}

/// `aien-serve-http --check-decode-timeline FILE [--drop REC:host|dev:IDX]`: [`check`] and, when
/// green, [`summary`] on stdout. `--drop` deletes one span (or, as `REC:record:0`, the whole
/// record) before checking: the negative control, which must come out red. Exit 0 green,
/// 1 red, 2 usage or an unreadable file.
pub fn run_check(args: &[String]) -> u8 {
    let usage = "usage: aien-serve-http --check-decode-timeline FILE [--drop REC:host|dev|record:IDX]";
    let (path, drop) = match args {
        [p] => (p, None),
        [p, flag, d] if flag == "--drop" => (p, Some(d.as_str())),
        _ => {
            eprintln!("{usage}");
            return 2;
        }
    };
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{path}: {e}");
            return 2;
        }
    };
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
            [r, w, i] => r.parse::<usize>().ok().zip(i.parse::<usize>().ok()).map(|(r, i)| (r, *w, i)),
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
        eprintln!("decode timeline RED: {} findings over {} records, {path}", bad.len(), records.len());
        1
    }
}

/// Reads a JSONL timeline.
pub fn read_jsonl(text: &str) -> Result<Vec<Record>, String> {
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
            k.extend(["res", "norm", "router", "gate_up", "swiglu", "down", "gather", "res"]);
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
            out: None,
            marks: [0; 3],
            pending: None,
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
                chip_ns: launches as u64 * 1000,
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
        assert_eq!(read_jsonl(&r.iter().map(|x| serde_json::to_string(x).unwrap() + "\n").collect::<String>()).unwrap(), r);
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
        assert!(bad.iter().any(|b| b.contains("device spans for")), "{bad:?}");
        assert!(bad.iter().any(|b| b.contains("dev 99")), "{bad:?}");
    }

    #[test]
    fn dropping_one_record_is_red() {
        let mut r = records(4);
        r.remove(2);
        let bad = check(&r);
        assert!(bad.iter().any(|b| b.contains("missing or repeated")), "{bad:?}");
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
        assert_eq!(&c[n - 9..], ["glue", "head_screen", "head_screen", "head_screen", "glue", "glue", "head_recheck", "token_pick", "token_pick"]);
        assert_eq!(c.iter().filter(|x| **x == "experts").count(), 80);
        assert_eq!(c.iter().filter(|x| **x == "router").count(), 40);
        let plain: Vec<String> = ["down", "norm", "lm_head", "argmax", "argmax"].map(String::from).into();
        assert_eq!(categories(&plain), ["experts", "glue", "head_plain", "token_pick", "token_pick"]);
    }

    #[test]
    fn summary_lists_every_beat() {
        let r = records(4);
        let s = summary(&r);
        assert!(s.contains("\"period\" -> {1000000, 1000000, 1000000, 1000000}"), "{s}");
        assert!(s.contains("\"dev:head_recheck\" -> {1000, 1000, 1000, 1000}"), "{s}");
    }
}
