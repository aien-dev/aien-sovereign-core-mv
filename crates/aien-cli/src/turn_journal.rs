//! sovereign-core #665: persist-before-act for the `aien` chat loop.
//!
//! Mechanism after Hermes Agent (Nous Research, MIT, commit 5480918),
//! re-implemented; no code copied. Same rule as the AIENOS continuity commit
//! (aienos ADR 0016): advance the record, then act.
//!
//! Every message the loop adds, and every tool request, is appended to an
//! append-only, hash-chained journal and flushed to disk *before* the loop acts
//! on it. A tool runs only after its request is on disk. If the process dies
//! between the request and the result, a resumed session finds the request with
//! no result: read-only tools are run again, and anything that may have changed
//! the world is reported to the model as "not known whether it happened" and is
//! never run a second time. So nothing is lost and nothing is done twice.
//!
//! Shape (ADR 0024): the pure part ([`replay`], [`resume_steps`],
//! [`effect_of`], [`tool_response_message`]) has no I/O and works on
//! content-addressed records, so it can move to Omega later. [`Journal`] is the
//! only edge that touches the disk.
//!
//! File format, one JSON object per line:
//! `{"v":1,"prev":<digest of previous line or "genesis">,"digest":<hex>,"record":{...}}`
//! where `digest = sha256("AIEN_TURN_JOURNAL_V1\n" + prev + "\n" + record bytes)`.
//! A torn last line (the process died mid-write) is dropped on resume; damage
//! anywhere else is refused, never guessed around.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};

const DOMAIN: &str = "AIEN_TURN_JOURNAL_V1";
const GENESIS: &str = "genesis";

/// Whether running a tool again could change the world a second time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Effect {
    /// Only reads; safe to run again after a crash.
    ReadOnly,
    /// May write, run commands, spend, or talk to the outside. Never re-run blindly.
    Effectful,
}

/// The tools the loop knows to be pure reads. Everything else, including any
/// tool this list has never heard of, counts as effectful.
pub fn effect_of(tool_name: &str) -> Effect {
    match tool_name {
        "view_file" | "list_dir" | "grep_search" | "search_directory" | "find_by_name"
        | "find_file" => Effect::ReadOnly,
        _ => Effect::Effectful,
    }
}

/// A tool request as it was recorded before it ran.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolIntent {
    pub call_id: String,
    pub name: String,
    pub args: Value,
    pub effect: Effect,
}

/// One line of the journal.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    /// A message appended to the conversation, stored exactly as it is sent.
    Message { message: Value },
    /// The whole conversation as it stands at the end of a turn (after any
    /// compaction). Replay starts from the newest snapshot.
    Snapshot { messages: Vec<Value> },
    /// A tool request, written and flushed before the tool runs.
    ToolIntent(ToolIntent),
    /// The result of a tool request, as the message the model will see.
    ToolOutcome { call_id: String, message: Value },
    /// The turn ended early, with the reason the operator was shown.
    TurnStop { reason: String },
}

/// Why a turn ended without finishing. Each one has a fixed name the operator sees.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TurnStop {
    /// A message or tool request could not be written to disk, so nothing after it ran.
    PersistFailed(String),
    /// The recovery rules ran out (see `turn_recovery`).
    Recovery(&'static str),
}

impl TurnStop {
    pub fn describe(&self) -> String {
        match self {
            TurnStop::PersistFailed(e) => format!(
                "turn stopped (persist-failed): could not record the next step on disk ({e}); nothing after it was run"
            ),
            TurnStop::Recovery(reason) => format!("turn stopped ({reason})"),
        }
    }
}

/// What a journal says, read back.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Replay {
    /// The conversation to continue from.
    pub messages: Vec<Value>,
    /// Tool requests that were recorded but have no recorded result.
    pub in_doubt: Vec<ToolIntent>,
    /// Reasons of turns that stopped early, oldest first.
    pub stops: Vec<String>,
    /// Digest of the last good line (`"genesis"` for an empty journal).
    pub last_digest: String,
    /// The next unused tool call number.
    pub next_call: u64,
    /// Bytes of the file that are whole, chained lines.
    pub valid_len: usize,
    /// True if a partly written last line was dropped.
    pub torn_tail: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum JournalError {
    /// A line other than the last does not parse, or the chain is broken.
    Corrupt {
        line: usize,
        why: String,
    },
    Io(String),
}

impl std::fmt::Display for JournalError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            JournalError::Corrupt { line, why } => {
                write!(f, "journal damaged at line {line}: {why}")
            }
            JournalError::Io(e) => write!(f, "journal I/O: {e}"),
        }
    }
}

fn digest_of(prev: &str, record: &Value) -> String {
    let body = serde_json::to_vec(record).unwrap_or_default();
    let mut h = Sha256::new();
    h.update(DOMAIN.as_bytes());
    h.update(b"\n");
    h.update(prev.as_bytes());
    h.update(b"\n");
    h.update(&body);
    hex::encode(h.finalize())
}

fn call_number(call_id: &str) -> Option<u64> {
    call_id.strip_prefix("call-")?.parse().ok()
}

/// Reads a journal back. Pure: no I/O.
pub fn replay(text: &str) -> Result<Replay, JournalError> {
    let mut out = Replay {
        last_digest: GENESIS.to_string(),
        ..Replay::default()
    };
    let mut offset = 0usize;
    let mut lines = text.split_inclusive('\n').enumerate().peekable();
    while let Some((idx, raw)) = lines.next() {
        let is_last = lines.peek().is_none();
        let whole = raw.ends_with('\n');
        let line = raw.trim_end_matches('\n');
        if line.trim().is_empty() {
            offset += raw.len();
            continue;
        }
        let parsed: Option<Value> = serde_json::from_str(line).ok();
        let Some(parsed) = parsed.filter(|_| whole) else {
            if is_last {
                // Died mid-write: the line never finished, so nothing acted on it.
                out.torn_tail = true;
                break;
            }
            return Err(JournalError::Corrupt {
                line: idx + 1,
                why: "not a whole JSON line".into(),
            });
        };
        let prev = parsed.get("prev").and_then(Value::as_str).unwrap_or("");
        let digest = parsed.get("digest").and_then(Value::as_str).unwrap_or("");
        let record_v = parsed.get("record").cloned().unwrap_or(Value::Null);
        if prev != out.last_digest {
            return Err(JournalError::Corrupt {
                line: idx + 1,
                why: "chain broken (prev does not match the line before)".into(),
            });
        }
        if digest != digest_of(prev, &record_v) {
            return Err(JournalError::Corrupt {
                line: idx + 1,
                why: "digest does not match the record".into(),
            });
        }
        let record: Record =
            serde_json::from_value(record_v).map_err(|e| JournalError::Corrupt {
                line: idx + 1,
                why: format!("unknown record: {e}"),
            })?;
        match record {
            Record::Message { message } => out.messages.push(message),
            Record::Snapshot { messages } => out.messages = messages,
            Record::ToolIntent(intent) => {
                if let Some(n) = call_number(&intent.call_id) {
                    out.next_call = out.next_call.max(n + 1);
                }
                out.in_doubt.push(intent);
            }
            Record::ToolOutcome { call_id, message } => {
                out.in_doubt.retain(|i| i.call_id != call_id);
                out.messages.push(message);
            }
            Record::TurnStop { reason } => out.stops.push(reason),
        }
        out.last_digest = digest.to_string();
        offset += raw.len();
        out.valid_len = offset;
    }
    Ok(out)
}

/// The message the model sees for a tool result. Kept byte-for-byte the same
/// as the loop has always sent, so prompts do not change.
pub fn tool_response_message(name: &str, result: &Value) -> Value {
    let body = serde_json::to_string(result).unwrap_or_default();
    json!({
        "role": "user",
        "content": format!("<tool_response name=\"{}\">\n{}\n</tool_response>", name, body)
    })
}

/// What a resumed session does with one in-doubt request.
#[derive(Debug, Clone, PartialEq)]
pub enum ResumeStep {
    /// A pure read: run it again.
    Rerun(ToolIntent),
    /// May have changed the world: tell the model, do not run it again.
    ReportUnknown { intent: ToolIntent, message: Value },
}

/// Decides how to settle in-doubt requests. Pure.
pub fn resume_steps(in_doubt: &[ToolIntent]) -> Vec<ResumeStep> {
    in_doubt
        .iter()
        .map(|i| match i.effect {
            Effect::ReadOnly => ResumeStep::Rerun(i.clone()),
            Effect::Effectful => ResumeStep::ReportUnknown {
                intent: i.clone(),
                message: tool_response_message(
                    &i.name,
                    &json!({
                        "status": "outcome_unknown",
                        "call_id": i.call_id,
                        "detail": "The session stopped after this request was recorded and before its result was. It was not run again, so it is not known whether it happened. Check its effect before asking for it again."
                    }),
                ),
            },
        })
        .collect()
}

/// The on-disk journal. The only part of this module that does I/O.
pub struct Journal {
    path: PathBuf,
    file: File,
    last: String,
    next_call: u64,
}

impl Journal {
    /// Starts a new journal. Refuses to overwrite an existing one.
    pub fn create(path: &Path) -> Result<Self, JournalError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).map_err(|e| JournalError::Io(e.to_string()))?;
        }
        let file = OpenOptions::new()
            .create_new(true)
            .append(true)
            .open(path)
            .map_err(|e| JournalError::Io(format!("{}: {e}", path.display())))?;
        if let Some(dir) = path.parent() {
            // Make the new file's name durable too (best effort; not every OS allows it).
            if let Ok(d) = File::open(dir) {
                let _ = d.sync_all();
            }
        }
        Ok(Self {
            path: path.to_path_buf(),
            file,
            last: GENESIS.to_string(),
            next_call: 0,
        })
    }

    /// Opens an existing journal to continue it. Drops a torn last line first,
    /// so the chain continues from the last whole line.
    pub fn resume(path: &Path) -> Result<(Self, Replay), JournalError> {
        let text = std::fs::read_to_string(path)
            .map_err(|e| JournalError::Io(format!("{}: {e}", path.display())))?;
        let rep = replay(&text)?;
        let file = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|e| JournalError::Io(e.to_string()))?;
        if rep.valid_len < text.len() {
            file.set_len(rep.valid_len as u64)
                .map_err(|e| JournalError::Io(e.to_string()))?;
            file.sync_data()
                .map_err(|e| JournalError::Io(e.to_string()))?;
        }
        Ok((
            Self {
                path: path.to_path_buf(),
                file,
                last: rep.last_digest.clone(),
                next_call: rep.next_call,
            },
            rep,
        ))
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one record and flushes it to disk before returning.
    pub fn append(&mut self, record: &Record) -> Result<(), JournalError> {
        let record_v = serde_json::to_value(record).map_err(|e| JournalError::Io(e.to_string()))?;
        let digest = digest_of(&self.last, &record_v);
        let line = json!({"v": 1, "prev": self.last, "digest": digest, "record": record_v});
        let mut bytes = serde_json::to_vec(&line).map_err(|e| JournalError::Io(e.to_string()))?;
        bytes.push(b'\n');
        self.file
            .write_all(&bytes)
            .and_then(|_| self.file.sync_data())
            .map_err(|e| JournalError::Io(e.to_string()))?;
        self.last = digest;
        Ok(())
    }

    /// Records a message, then adds it to the conversation. The conversation
    /// only grows after the disk has it.
    pub fn push(&mut self, messages: &mut Vec<Value>, message: Value) -> Result<(), TurnStop> {
        self.append(&Record::Message {
            message: message.clone(),
        })
        .map_err(|e| TurnStop::PersistFailed(e.to_string()))?;
        messages.push(message);
        Ok(())
    }

    /// Records the whole conversation at the end of a turn.
    pub fn snapshot(&mut self, messages: &[Value]) -> Result<(), JournalError> {
        self.append(&Record::Snapshot {
            messages: messages.to_vec(),
        })
    }

    /// Records a tool request. The tool must not run unless this returns Ok.
    pub fn intent(&mut self, name: &str, args: &Value) -> Result<ToolIntent, JournalError> {
        let intent = ToolIntent {
            call_id: format!("call-{}", self.next_call),
            name: name.to_string(),
            args: args.clone(),
            effect: effect_of(name),
        };
        self.append(&Record::ToolIntent(intent.clone()))?;
        self.next_call += 1;
        Ok(intent)
    }

    /// Records a tool result and adds it to the conversation.
    pub fn outcome(
        &mut self,
        messages: &mut Vec<Value>,
        intent: &ToolIntent,
        message: Value,
    ) -> Result<(), TurnStop> {
        self.append(&Record::ToolOutcome {
            call_id: intent.call_id.clone(),
            message: message.clone(),
        })
        .map_err(|e| TurnStop::PersistFailed(e.to_string()))?;
        messages.push(message);
        Ok(())
    }

    /// Records that a turn ended early (best effort: the reason is also printed).
    pub fn stop(&mut self, stop: &TurnStop) {
        let _ = self.append(&Record::TurnStop {
            reason: stop.describe(),
        });
    }
}

/// Runs a batch of tool calls under persist-before-act: for each call the
/// request is on disk before the tool runs, and the result is on disk before
/// the model sees it.
pub fn run_tool_calls<F>(
    journal: &mut Journal,
    messages: &mut Vec<Value>,
    calls: Vec<(String, Value)>,
    mut dispatch: F,
) -> Result<(), TurnStop>
where
    F: FnMut(&str, &Value) -> Value,
{
    for (name, args) in calls {
        let intent = journal
            .intent(&name, &args)
            .map_err(|e| TurnStop::PersistFailed(e.to_string()))?;
        let result = dispatch(&name, &args);
        journal.outcome(messages, &intent, tool_response_message(&name, &result))?;
    }
    Ok(())
}

/// Settles the in-doubt requests of a resumed session: re-runs reads, reports
/// the rest as unknown. Every settlement is journaled like a normal result.
pub fn settle_in_doubt<F>(
    journal: &mut Journal,
    messages: &mut Vec<Value>,
    in_doubt: &[ToolIntent],
    mut dispatch: F,
) -> Result<Vec<ResumeStep>, TurnStop>
where
    F: FnMut(&str, &Value) -> Value,
{
    let steps = resume_steps(in_doubt);
    for step in &steps {
        match step {
            ResumeStep::Rerun(i) => {
                let result = dispatch(&i.name, &i.args);
                journal.outcome(messages, i, tool_response_message(&i.name, &result))?;
            }
            ResumeStep::ReportUnknown { intent, message } => {
                journal.outcome(messages, intent, message.clone())?;
            }
        }
    }
    Ok(steps)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp() -> (tempfile::TempDir, PathBuf) {
        let d = tempfile::tempdir().unwrap();
        let p = d.path().join("s.journal.jsonl");
        (d, p)
    }

    fn user(t: &str) -> Value {
        json!({"role": "user", "content": t})
    }

    #[test]
    fn effect_defaults_to_effectful_for_unknown_tools() {
        assert_eq!(effect_of("view_file"), Effect::ReadOnly);
        assert_eq!(effect_of("grep_search"), Effect::ReadOnly);
        assert_eq!(effect_of("run_command"), Effect::Effectful);
        assert_eq!(effect_of("write_to_file"), Effect::Effectful);
        assert_eq!(effect_of("some_new_tool"), Effect::Effectful);
    }

    #[test]
    fn replay_rebuilds_the_conversation_byte_identical() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        let mut msgs = Vec::new();
        j.push(&mut msgs, json!({"role": "system", "content": "sys"}))
            .unwrap();
        j.push(&mut msgs, user("hello \"Allen\" \u{1F30A}"))
            .unwrap();
        j.push(
            &mut msgs,
            json!({"role": "assistant", "content": "<tool_call>{\"name\":\"view_file\",\"arguments\":{\"path\":\"a\"}}</tool_call>"}),
        )
        .unwrap();
        run_tool_calls(
            &mut j,
            &mut msgs,
            vec![("view_file".into(), json!({"path": "a"}))],
            |_, _| json!({"ok": true, "lines": ["x"]}),
        )
        .unwrap();
        drop(j);
        let rep = replay(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(
            serde_json::to_vec(&rep.messages).unwrap(),
            serde_json::to_vec(&msgs).unwrap()
        );
        assert!(rep.in_doubt.is_empty());
        assert!(!rep.torn_tail);
    }

    #[test]
    fn snapshot_replaces_history_so_compaction_survives_resume() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        let mut msgs = Vec::new();
        for i in 0..5 {
            j.push(&mut msgs, user(&format!("m{i}"))).unwrap();
        }
        let compacted = vec![user("summary of m0..m3"), user("m4")];
        j.snapshot(&compacted).unwrap();
        let mut after = compacted.clone();
        j.push(&mut after, user("m5")).unwrap();
        drop(j);
        let rep = replay(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert_eq!(rep.messages, after);
    }

    #[test]
    fn a_tool_runs_only_after_its_request_is_on_disk() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        let mut msgs = Vec::new();
        let path = p.clone();
        run_tool_calls(
            &mut j,
            &mut msgs,
            vec![("run_command".into(), json!({"command": "true"}))],
            |name, _| {
                // At the moment the tool runs, the disk already holds the request.
                let rep = replay(&std::fs::read_to_string(&path).unwrap()).unwrap();
                assert_eq!(rep.in_doubt.len(), 1);
                assert_eq!(rep.in_doubt[0].name, name);
                json!({"exit": 0})
            },
        )
        .unwrap();
    }

    #[test]
    fn crash_between_request_and_run_reports_effectful_as_unknown_and_reruns_reads() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        let mut msgs = Vec::new();
        j.push(&mut msgs, user("go")).unwrap();
        // Two requests recorded, then the process "dies" before either runs.
        j.intent("write_to_file", &json!({"path": "out.txt", "content": "x"}))
            .unwrap();
        j.intent("view_file", &json!({"path": "in.txt"})).unwrap();
        drop(j);

        let (mut j, rep) = Journal::resume(&p).unwrap();
        assert_eq!(rep.in_doubt.len(), 2);
        let mut msgs = rep.messages.clone();
        let mut ran = Vec::new();
        let steps = settle_in_doubt(&mut j, &mut msgs, &rep.in_doubt, |name, _| {
            ran.push(name.to_string());
            json!({"ok": true})
        })
        .unwrap();
        // The write is not done again; the read is.
        assert_eq!(ran, vec!["view_file".to_string()]);
        assert!(matches!(steps[0], ResumeStep::ReportUnknown { .. }));
        assert!(msgs[1]["content"]
            .as_str()
            .unwrap()
            .contains("outcome_unknown"));
        drop(j);
        // Settled: a second resume has nothing in doubt and the same conversation.
        let (_j, rep2) = Journal::resume(&p).unwrap();
        assert!(rep2.in_doubt.is_empty());
        assert_eq!(rep2.messages, msgs);
        assert_eq!(rep2.next_call, 2);
    }

    #[test]
    fn a_torn_last_line_is_dropped_and_the_chain_continues() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        let mut msgs = Vec::new();
        j.push(&mut msgs, user("one")).unwrap();
        drop(j);
        // The process died halfway through writing the next line.
        let mut f = OpenOptions::new().append(true).open(&p).unwrap();
        f.write_all(b"{\"v\":1,\"prev\":\"abc\",\"dig").unwrap();
        drop(f);
        let (mut j, rep) = Journal::resume(&p).unwrap();
        assert!(rep.torn_tail);
        assert_eq!(rep.messages, msgs);
        j.push(&mut msgs, user("two")).unwrap();
        drop(j);
        let rep = replay(&std::fs::read_to_string(&p).unwrap()).unwrap();
        assert!(!rep.torn_tail);
        assert_eq!(rep.messages, msgs);
    }

    #[test]
    fn damage_before_the_last_line_is_refused() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        let mut msgs = Vec::new();
        j.push(&mut msgs, user("one")).unwrap();
        j.push(&mut msgs, user("two")).unwrap();
        drop(j);
        let text = std::fs::read_to_string(&p)
            .unwrap()
            .replacen("one", "0ne", 1);
        assert!(matches!(
            replay(&text),
            Err(JournalError::Corrupt { line: 1, .. })
        ));
    }

    #[test]
    fn a_failed_write_stops_the_turn_before_the_tool_runs() {
        let (_d, p) = tmp();
        let mut j = Journal::create(&p).unwrap();
        // Swap the file for a read-only handle so every append fails.
        j.file = File::open(&p).unwrap();
        let mut msgs = Vec::new();
        let mut ran = false;
        let r = run_tool_calls(
            &mut j,
            &mut msgs,
            vec![("run_command".into(), json!({"command": "rm x"}))],
            |_, _| {
                ran = true;
                json!({})
            },
        );
        assert!(matches!(r, Err(TurnStop::PersistFailed(_))));
        assert!(!ran);
        assert!(msgs.is_empty());
    }

    #[test]
    fn create_refuses_to_overwrite_a_journal() {
        let (_d, p) = tmp();
        let _j = Journal::create(&p).unwrap();
        assert!(Journal::create(&p).is_err());
    }

    /// Child half of `killed_between_record_and_effect_*`. A no-op unless the
    /// parent test starts this binary with `AIEN_JOURNAL_KILL_DIR` set.
    #[test]
    fn kill_child() {
        let Ok(dir) = std::env::var("AIEN_JOURNAL_KILL_DIR") else {
            return;
        };
        let dir = PathBuf::from(dir);
        let mut j = Journal::create(&dir.join("s.journal.jsonl")).unwrap();
        let mut msgs = Vec::new();
        j.push(&mut msgs, user("write the report")).unwrap();
        let _ = run_tool_calls(
            &mut j,
            &mut msgs,
            vec![(
                "write_to_file".into(),
                json!({"path": "report.md", "content": "x"}),
            )],
            |_, _| {
                // The request is on disk now. Tell the parent, then take long
                // enough that the parent kills us before the effect happens.
                std::fs::write(dir.join("running"), b"").unwrap();
                std::thread::sleep(std::time::Duration::from_secs(30));
                std::fs::write(dir.join("effect"), b"done").unwrap();
                json!({"ok": true})
            },
        );
    }

    /// sc#665 acceptance: kill the process between persist and tool run; the
    /// resumed session has the request recorded, and the effect is neither lost
    /// (the model is told it is unknown) nor doubled (it is not run again).
    #[test]
    fn killed_between_record_and_effect_resumes_without_loss_or_double() {
        let d = tempfile::tempdir().unwrap();
        let mut child = std::process::Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "turn_journal::tests::kill_child",
                "--nocapture",
                "--test-threads=1",
            ])
            .env("AIEN_JOURNAL_KILL_DIR", d.path())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap();
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
        while !d.path().join("running").exists() {
            assert!(
                std::time::Instant::now() < deadline,
                "child never reached the tool"
            );
            std::thread::sleep(std::time::Duration::from_millis(20));
        }
        child.kill().unwrap(); // SIGKILL on Unix: no cleanup runs in the child.
        let _ = child.wait();
        assert!(!d.path().join("effect").exists());

        let p = d.path().join("s.journal.jsonl");
        let (mut j, rep) = Journal::resume(&p).unwrap();
        assert_eq!(rep.messages, vec![user("write the report")]);
        assert_eq!(
            rep.in_doubt.len(),
            1,
            "the request was recorded before the tool ran"
        );
        assert_eq!(rep.in_doubt[0].name, "write_to_file");

        let mut msgs = rep.messages.clone();
        let mut reran = 0;
        settle_in_doubt(&mut j, &mut msgs, &rep.in_doubt, |_, _| {
            reran += 1;
            json!({})
        })
        .unwrap();
        assert_eq!(reran, 0, "an effectful request is never run a second time");
        assert!(msgs[1]["content"]
            .as_str()
            .unwrap()
            .contains("outcome_unknown"));
        drop(j);
        let (_j, again) = Journal::resume(&p).unwrap();
        assert!(again.in_doubt.is_empty(), "settled once, settled for good");
    }
}
