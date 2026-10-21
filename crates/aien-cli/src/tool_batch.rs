//! sovereign-core #666: run one reply's tool calls as a planned batch.
//!
//! Mechanism after Hermes Agent (Nous Research, MIT, commit 5480918),
//! re-implemented; no code copied.
//!
//! For each segment of the plan (`tool_plan`), in order:
//! 1. every request of the segment is written to the session journal first
//!    (persist-before-act, #665), so a crash mid-segment leaves each one
//!    recorded as in doubt and none half-known;
//! 2. the calls run: one alone, or up to eight together on scoped threads;
//! 3. oversized results spill to files (`tool_spill`), file before journal;
//! 4. the results are journaled and added to the conversation in the order
//!    the model asked for them, whatever order they finished in.

use crate::tool_plan::{plan, Access, MAX_PARALLEL};
use crate::tool_spill::{self, SpillBudget};
use crate::turn_journal::{tool_response_message, Journal, TurnStop};
use serde_json::{json, Value};
use std::path::PathBuf;

pub struct BatchConfig {
    pub max_parallel: usize,
    pub budget: SpillBudget,
    /// Folder for spilled results; `None` keeps an inline preview only.
    pub spill_dir: Option<PathBuf>,
}

impl BatchConfig {
    pub fn new(window_tokens: usize, spill_dir: Option<PathBuf>) -> Self {
        Self {
            max_parallel: MAX_PARALLEL,
            budget: SpillBudget::for_window(window_tokens),
            spill_dir,
        }
    }
}

/// What a batch did, for the operator's screen.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct BatchReport {
    pub segments: usize,
    /// Calls that ran together with at least one other call.
    pub ran_together: usize,
    pub spilled: Vec<PathBuf>,
}

fn persist(e: crate::turn_journal::JournalError) -> TurnStop {
    TurnStop::PersistFailed(e.to_string())
}

/// Runs `calls` under the plan. `access` says what each call touches;
/// `dispatch` runs one call and must be safe to call from several threads.
pub fn run_batch<A, D>(
    journal: &mut Journal,
    messages: &mut Vec<Value>,
    calls: Vec<(String, Value)>,
    cfg: &BatchConfig,
    access: A,
    dispatch: D,
) -> Result<BatchReport, TurnStop>
where
    A: Fn(&str, &Value) -> Access,
    D: Fn(&str, &Value) -> Value + Sync,
{
    let accesses: Vec<Access> = calls.iter().map(|(n, a)| access(n, a)).collect();
    let segments = plan(&accesses, cfg.max_parallel);
    let mut report = BatchReport {
        segments: segments.len(),
        ..BatchReport::default()
    };
    let mut turn_used = 0usize;
    for seg in &segments {
        // 1. Every request of the segment is on disk before any of them runs.
        let mut intents = Vec::with_capacity(seg.calls.len());
        for &i in &seg.calls {
            intents.push(journal.intent(&calls[i].0, &calls[i].1).map_err(persist)?);
        }
        // 2. Run.
        let results: Vec<Value> = if seg.calls.len() == 1 {
            let (n, a) = &calls[seg.calls[0]];
            vec![dispatch(n, a)]
        } else {
            report.ran_together += seg.calls.len();
            let d = &dispatch;
            std::thread::scope(|s| {
                let handles: Vec<_> = seg
                    .calls
                    .iter()
                    .map(|&i| {
                        let (n, a) = &calls[i];
                        s.spawn(move || d(n, a))
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| {
                        h.join().unwrap_or_else(
                            |_| json!({"error": "the tool stopped with an internal fault (panic)"}),
                        )
                    })
                    .collect()
            })
        };
        // 3. Spill what is too large, file before journal.
        let texts: Vec<String> = results
            .iter()
            .map(|r| serde_json::to_string(r).unwrap_or_default())
            .collect();
        let sizes: Vec<usize> = texts.iter().map(|t| t.chars().count()).collect();
        let exempt: Vec<bool> = seg
            .calls
            .iter()
            .map(|&i| tool_spill::exempt(&calls[i].0))
            .collect();
        let room = cfg.budget.per_turn.saturating_sub(turn_used);
        let spill = tool_spill::choose(&sizes, &exempt, &cfg.budget, room);
        // 4. Results in call order.
        for (k, &i) in seg.calls.iter().enumerate() {
            let name = &calls[i].0;
            let shown = if spill[k] {
                let full = tool_spill::render_full(name, &intents[k].call_id, &results[k]);
                let path = cfg
                    .spill_dir
                    .as_ref()
                    .and_then(|d| tool_spill::write_spill(d, &full).ok());
                if let Some(p) = &path {
                    report.spilled.push(p.clone());
                }
                tool_spill::stub(&results[k], sizes[k], path.as_deref(), &cfg.budget)
            } else {
                results[k].clone()
            };
            let message = tool_response_message(name, &shown);
            turn_used += message["content"].as_str().map_or(0, |c| c.chars().count());
            journal.outcome(messages, &intents[k], message)?;
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tool_plan::access_of;
    use crate::turn_journal::replay;
    use std::path::Path;
    use std::sync::{Condvar, Mutex};
    use std::time::Duration;

    fn journal() -> (tempfile::TempDir, Journal, PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("s.journal.jsonl");
        let j = Journal::create(&p).unwrap();
        (d, j, p)
    }

    fn cfg(spill: Option<PathBuf>) -> BatchConfig {
        BatchConfig::new(24_000, spill)
    }

    fn acc(n: &str, a: &Value) -> Access {
        access_of(n, a, Path::new("/w"))
    }

    fn body(message: &Value) -> Value {
        let c = message["content"].as_str().unwrap();
        let start = c.find('\n').unwrap() + 1;
        let end = c.rfind("\n</tool_response>").unwrap();
        serde_json::from_str(&c[start..end]).unwrap()
    }

    /// Counts how many calls run at the same moment.
    ///
    /// A sleep is not proof of overlap: under load the second thread can start
    /// after the first has finished. So a call that is expected to run beside
    /// `expect - 1` others waits (up to `DEADLINE`) until all `expect` calls
    /// have entered, which makes `max == expect` certain if the batch really
    /// runs them together, and the wait times out only if it serialises them.
    /// With `expect == 1` a call holds for `HOLD`, or less if another call
    /// enters meanwhile, so the "never together" tests still catch overlap.
    struct Overlap {
        expect: usize,
        state: Mutex<(usize, usize, usize)>, // (arrivals, running now, max running)
        entered: Condvar,
    }
    impl Overlap {
        const DEADLINE: Duration = Duration::from_secs(10);
        const HOLD: Duration = Duration::from_millis(150);

        fn alone() -> Self {
            Self::together(1)
        }
        fn together(expect: usize) -> Self {
            Self {
                expect,
                state: Mutex::new((0, 0, 0)),
                entered: Condvar::new(),
            }
        }
        fn max(&self) -> usize {
            self.state.lock().unwrap().2
        }
        fn run(&self) -> Value {
            let mut s = self.state.lock().unwrap();
            s.0 += 1;
            s.1 += 1;
            s.2 = s.2.max(s.1);
            self.entered.notify_all();
            s = if self.expect > 1 {
                let expect = self.expect;
                self.entered
                    .wait_timeout_while(s, Self::DEADLINE, |s| s.0 < expect)
                    .unwrap()
                    .0
            } else {
                self.entered
                    .wait_timeout_while(s, Self::HOLD, |s| s.1 < 2)
                    .unwrap()
                    .0
            };
            s.1 -= 1;
            json!({"ok": true})
        }
    }

    #[test]
    fn two_readers_of_one_path_run_together() {
        let (_d, mut j, _p) = journal();
        let mut msgs = Vec::new();
        let o = Overlap::together(2);
        let calls = vec![
            ("view_file".to_string(), json!({"path": "/w/a"})),
            ("view_file".to_string(), json!({"path": "/w/a"})),
        ];
        let r = run_batch(&mut j, &mut msgs, calls, &cfg(None), acc, |_, _| o.run()).unwrap();
        assert_eq!(o.max(), 2);
        assert_eq!((r.segments, r.ran_together), (1, 2));
        assert_eq!(msgs.len(), 2);
    }

    #[test]
    fn two_writers_on_one_path_never_run_together() {
        let (_d, mut j, _p) = journal();
        let mut msgs = Vec::new();
        let o = Overlap::alone();
        let calls = vec![
            (
                "write_to_file".to_string(),
                json!({"path": "/w/a", "content": "1"}),
            ),
            (
                "write_to_file".to_string(),
                json!({"path": "/w/a", "content": "2"}),
            ),
            ("view_file".to_string(), json!({"path": "/w/a"})),
        ];
        let r = run_batch(&mut j, &mut msgs, calls, &cfg(None), acc, |_, _| o.run()).unwrap();
        assert_eq!(o.max(), 1);
        assert_eq!((r.segments, r.ran_together), (3, 0));
    }

    #[test]
    fn a_shell_command_never_runs_beside_anything() {
        let (_d, mut j, _p) = journal();
        let mut msgs = Vec::new();
        let o = Overlap::alone();
        let calls = vec![
            ("view_file".to_string(), json!({"path": "/w/a"})),
            ("run_command".to_string(), json!({"command": "ls"})),
            ("view_file".to_string(), json!({"path": "/w/b"})),
        ];
        run_batch(&mut j, &mut msgs, calls, &cfg(None), acc, |_, _| o.run()).unwrap();
        assert_eq!(o.max(), 1);
    }

    #[test]
    fn every_request_of_a_segment_is_on_disk_before_any_runs() {
        let (_d, mut j, p) = journal();
        let mut msgs = Vec::new();
        let seen = Mutex::new(Vec::new());
        let calls = vec![
            ("view_file".to_string(), json!({"path": "/w/a"})),
            ("view_file".to_string(), json!({"path": "/w/b"})),
            ("view_file".to_string(), json!({"path": "/w/c"})),
        ];
        run_batch(&mut j, &mut msgs, calls, &cfg(None), acc, |_, _| {
            let rep = replay(&std::fs::read_to_string(&p).unwrap()).unwrap();
            seen.lock().unwrap().push(rep.in_doubt.len());
            json!({"ok": true})
        })
        .unwrap();
        assert_eq!(*seen.lock().unwrap(), vec![3, 3, 3]);
        let rep = replay(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert!(rep.in_doubt.is_empty());
        assert_eq!(rep.messages, msgs);
    }

    #[test]
    fn results_come_back_in_call_order_whatever_finishes_first() {
        let (_d, mut j, _p) = journal();
        let mut msgs = Vec::new();
        let calls: Vec<(String, Value)> = (0..4)
            .map(|i| ("view_file".to_string(), json!({"path": format!("/w/{i}")})))
            .collect();
        run_batch(&mut j, &mut msgs, calls, &cfg(None), acc, |_, a| {
            let i: u64 = a["path"].as_str().unwrap()[3..].parse().unwrap();
            std::thread::sleep(Duration::from_millis(40 * (4 - i)));
            json!({"i": i})
        })
        .unwrap();
        let order: Vec<u64> = msgs
            .iter()
            .map(|m| body(m)["i"].as_u64().unwrap())
            .collect();
        assert_eq!(order, vec![0, 1, 2, 3]);
    }

    #[test]
    fn a_one_megabyte_result_spills_and_the_file_is_addressable() {
        let (d, mut j, _p) = journal();
        let spill = d.path().join("spill");
        let mut msgs = Vec::new();
        let big: String = (0..45_000)
            .map(|i| format!("line {i:05} of the output\n"))
            .collect();
        assert!(big.len() >= 1_000_000);
        let c = cfg(Some(spill.clone()));
        let r = run_batch(
            &mut j,
            &mut msgs,
            vec![("run_command".to_string(), json!({"command": "big"}))],
            &c,
            acc,
            |_, _| json!({"stdout": big, "exit_code": 0}),
        )
        .unwrap();
        let content = msgs[0]["content"].as_str().unwrap();
        assert!(
            content.chars().count() < c.budget.per_result,
            "{} chars",
            content.len()
        );
        let b = body(&msgs[0]);
        assert_eq!(b["spilled"], json!(true));
        let path = PathBuf::from(b["path"].as_str().unwrap());
        assert_eq!(r.spilled, vec![path.clone()]);
        assert!(path.starts_with(&spill));
        let file = std::fs::read_to_string(&path).unwrap();
        assert!(
            file.contains(&format!("## stdout\n{big}")),
            "the spill file holds the whole output"
        );
        assert!(b["preview"].as_str().unwrap().contains("TRUNCATED"));
    }

    #[test]
    fn a_large_file_read_is_never_spilled() {
        let (d, mut j, _p) = journal();
        let mut msgs = Vec::new();
        let big = "x".repeat(200_000);
        run_batch(
            &mut j,
            &mut msgs,
            vec![("view_file".to_string(), json!({"path": "/w/a"}))],
            &cfg(Some(d.path().join("spill"))),
            acc,
            |_, _| json!({"content": big}),
        )
        .unwrap();
        assert_eq!(body(&msgs[0])["content"].as_str().unwrap().len(), 200_000);
        assert!(!d.path().join("spill").exists());
    }

    #[test]
    fn when_the_spill_file_cannot_be_written_a_preview_is_still_kept() {
        let (d, mut j, _p) = journal();
        let blocker = d.path().join("not-a-dir");
        std::fs::write(&blocker, b"x").unwrap();
        let mut msgs = Vec::new();
        run_batch(
            &mut j,
            &mut msgs,
            vec![("run_command".to_string(), json!({"command": "big"}))],
            &cfg(Some(blocker)),
            acc,
            |_, _| json!({"stdout": "y".repeat(50_000), "status": "ok"}),
        )
        .unwrap();
        let b = body(&msgs[0]);
        assert_eq!(b["spilled"], json!(false));
        assert!(b["note"].as_str().unwrap().contains("could not be saved"));
        assert_eq!(b["status"], json!("ok"), "the status is carried over");
    }

    #[test]
    fn a_panicking_tool_in_a_parallel_segment_does_not_lose_the_others() {
        let (_d, mut j, _p) = journal();
        let mut msgs = Vec::new();
        let calls = vec![
            ("view_file".to_string(), json!({"path": "/w/a"})),
            ("view_file".to_string(), json!({"path": "/w/boom"})),
        ];
        run_batch(&mut j, &mut msgs, calls, &cfg(None), acc, |_, a| {
            if a["path"] == "/w/boom" {
                panic!("tool fault");
            }
            json!({"ok": true})
        })
        .unwrap();
        assert_eq!(body(&msgs[0]), json!({"ok": true}));
        assert!(body(&msgs[1])["error"].as_str().unwrap().contains("panic"));
    }
}
