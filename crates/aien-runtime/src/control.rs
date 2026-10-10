//! Typed Operator Control Protocol and RuntimeController
//! Connects external operator CLI tools to the in-process runtime over local domain sockets.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LaunchSwarmReq {
    pub model_handle: u64,
    pub branch_count: usize,
    pub max_active_sequences: usize,
    pub max_tokens_per_branch: usize,
    pub root_world_id: u64,
    pub priority: u8,
    pub prompt_tokens: Vec<u32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
}

/// Role `tool` is refused on this path (issue #310): untrusted tool output must enter through
/// `aien_inference_abi::tool_boundary::ToolConversation`, which refuses control sequences and
/// checks call/result pairing. Otherwise a `ChatTurn` with role `tool` would reach the
/// byte-exact Qwen3 renderer unguarded.
fn refuse_tool_turns(messages: &[ChatTurn]) -> Result<(), aien_inference_abi::TokenizerError> {
    if messages
        .iter()
        .any(|m| m.role.trim().eq_ignore_ascii_case("tool"))
    {
        return Err(aien_inference_abi::TokenizerError::ToolTurnRefused(
            "role tool refused on the ChatTurn path: use tool_boundary::ToolConversation".into(),
        ));
    }
    Ok(())
}

/// Renders the chat with a model's template. Generation continues from an assistant
/// header unless the last turn is a non-empty assistant turn. Role `tool` and a plain model
/// (no chat template) are refused with an error, never a panic.
pub fn format_chat(
    template: aien_inference_abi::ChatTemplate,
    messages: &[ChatTurn],
) -> Result<String, aien_inference_abi::TokenizerError> {
    try_format_chat(template, messages)
}

/// Same as [`format_chat`]; kept as the explicit fallible name.
pub fn try_format_chat(
    template: aien_inference_abi::ChatTemplate,
    messages: &[ChatTurn],
) -> Result<String, aien_inference_abi::TokenizerError> {
    refuse_tool_turns(messages)?;
    let turns: Vec<(&str, &str)> = messages
        .iter()
        .map(|m| (m.role.as_str(), m.content.as_str()))
        .collect();
    template.try_render(&turns)
}

/// TinyLlama chat template. Generation always continues from an assistant header.
pub fn format_tinyllama_chat(
    messages: &[ChatTurn],
) -> Result<String, aien_inference_abi::TokenizerError> {
    format_chat(aien_inference_abi::ChatTemplate::Zephyr, messages)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlCommand {
    LaunchSwarm(LaunchSwarmReq),
    InspectSwarm(u64),
    CancelSwarm(u64),
    GetRuntimeStatus,
    Shutdown,
    /// One operator turn on the in-process PR #68 spine. The server streams
    /// `TurnDelta` lines and finishes with `TurnFinished`.
    StreamTurn {
        messages: Vec<ChatTurn>,
        max_tokens: usize,
        temperature: f32,
    },
    /// NEXT-PHASE-1 cut 1b: one goal through omega COMPOSITION-2 (J-Space
    /// alternatives, AEGIS verification, World commit, Cortex record) with a
    /// "model" Skill that proposes one file change for `workspace`. Handled on
    /// the socket connection (it runs inference). Nothing is written to
    /// `workspace` in this cut: the committed result is the proposal record.
    RunComposeTask {
        goal: String,
        workspace: String,
        /// ALLEN memory context for this task (`personal`, `work`, `project:<name>`),
        /// supplied by the operator only. `None`: no memory is included.
        #[serde(default)]
        context: Option<String>,
    },
    /// NEXT-PHASE-1 cut 2: append one operator record (`kind` = "constraint",
    /// "authorization" or "effect") to the composition's Cortex journal through
    /// its own writer. `links` name related Cortex ids (each must exist).
    ComposeNote {
        kind: String,
        text: String,
        links: Vec<u64>,
    },
    /// NEXT-PHASE-1 cut 2: recall every host record (constraints, authorizations,
    /// effects, repair records) plus the records named in `ids`, each digest
    /// re-checked; `prefix` (records 1..=prefix) gets a digest over their digests
    /// so a restart can be compared byte for byte.
    ComposeRecall {
        ids: Vec<u64>,
        prefix: Option<u64>,
    },
    /// NEXT-PHASE-1 cut 2: operator repair of a compose home that refuses to open
    /// (torn journal tail, or a journal behind its J-Space anchor). Closes this
    /// process's handle first; the cut is recorded in the journal.
    RecoverComposeHome,
    /// ALLEN persona profile (arch#159): identity, profile, model and unsupported
    /// facilities in one answer. Handled on the socket connection.
    AllenStatus,
    /// The saved profile (or the defaults when none).
    AllenProfileShow,
    /// Change the profile; `expected_revision` is the compare-and-swap guard.
    AllenProfileSet {
        expected_revision: u64,
        changes: aien_allen_profile::Changes,
    },
    /// Every saved revision, oldest first.
    AllenProfileHistory,
    /// Write a NEW revision that copies revision `to`.
    AllenProfileRevert {
        expected_revision: u64,
        to: u64,
    },
    /// Write a NEW revision with the defaults. The identity is untouched.
    AllenProfileReset {
        expected_revision: u64,
    },
    /// ALLEN scoped memory (arch#159). `context` is parsed by the daemon from this
    /// operator command only; no model text or file content ever reaches it.
    AllenMemoryPut {
        context: String,
        /// `fact` or `preference`.
        kind: String,
        text: String,
    },
    AllenMemoryRecall {
        context: String,
        query: Option<String>,
    },
    /// One context, or `owner` = true for every context (exactly one of the two).
    AllenMemoryInspect {
        context: Option<String>,
        owner: bool,
    },
    AllenMemoryCorrect {
        context: String,
        item: String,
        text: String,
    },
    /// Forget one `item`, or everything in `context` (`all_in_context`); exactly one.
    AllenMemoryForget {
        context: String,
        item: Option<String>,
        all_in_context: bool,
    },
    AllenMemoryExport {
        context: Option<String>,
        owner: bool,
    },
    AllenGoalsList {
        context: Option<String>,
        owner: bool,
    },
    AllenGoalAdd {
        context: String,
        text: String,
    },
    AllenGoalClose {
        context: String,
        item: String,
    },
    /// NEXT-PHASE-2: the effect-boundary checks and the durable intent, in one
    /// step (ACCEPTANCE-v2 2.1, 2.3). Answered with `ComposeNoted` (the intent).
    ComposeEffectIntent {
        authorization: u64,
        proposal_sha256: String,
        path: String,
        target: String,
        content_sha256: String,
        executor_pid: u32,
        executor_start: u64,
    },
    /// NEXT-PHASE-2: the executor's report after the write; the daemon records
    /// the state it reads from the world. Answered with `ComposeNoted`.
    ComposeEffectAck {
        intent: u64,
        reported: serde_json::Value,
    },
    /// NEXT-PHASE-2: reconcile unsettled effects, or record one operator
    /// declaration (ACCEPTANCE-v2 2.4).
    ComposeReconcile {
        declare: Option<ReconcileDeclare>,
    },
    /// sovereign-core #249: one approved proposal through the proposer hook
    /// (crate::approved): desk-key MAC authentication, a durable replay claim,
    /// then the production compose run (J-Space, AEGIS verify, World commit,
    /// Cortex records) with `workspace` as the task's workspace. Answered with
    /// `ComposeApprovedResult` or `ComposeApprovedRefused`. Writes nothing to
    /// `workspace`; the effect still needs an authorization, an intent and an ack.
    ComposeApprovedProposal {
        proposal: crate::approved::ApprovedProposal,
        workspace: String,
    },
    /// sovereign-core #261: the operator approves ONE proposal this daemon
    /// committed (`RunComposeTask`), and the daemon mints the grant itself.
    /// The caller names only the promotion, the proposal digest, the workspace
    /// it expects and who approves; path, content and target come from the
    /// daemon's own commit record. Answered with `ComposeNoted` (the grant).
    /// This replaces a client-written `authorization` note, which is refused.
    ComposeAuthorize {
        cx_promotion: u64,
        proposal_sha256: String,
        workspace: String,
        approver: String,
        constraints: Vec<u64>,
        /// sovereign-core #297: the approval desk's MAC over this authorize.
        /// Optional on the wire (old clients omit it; `None` is not written).
        /// Required only when the daemon runs with the desk switch ON.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        desk_proof: Option<DeskProof>,
    },
    /// NEXT-PHASE-2: operator `stop`, `resume`, `revoke` (ACCEPTANCE-v2 2.5, 2.6).
    ComposeControl {
        action: String,
        approver: String,
        authorization: Option<u64>,
    },
}

/// The desk key's proof for one `ComposeAuthorize` (sovereign-core #297): a
/// fresh nonce and the HMAC over the authorize binding
/// (`approved_auth::AuthorizeBinding`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeskProof {
    pub nonce: String,
    pub mac: String,
}

/// An operator's decision for one unsettled effect.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileDeclare {
    pub intent: u64,
    /// "done" or "not_done".
    pub state: String,
    pub approver: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ControlEnvelope {
    pub protocol_version: u16,
    pub request_id: u64,
    pub operation_id: u128,
    pub operator_session: u64,
    pub command: ControlCommand,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuntimeStatusReport {
    pub active_sequences: usize,
    pub active_swarms: usize,
    pub active_worlds: usize,
    pub free_kv_blocks: usize,
    pub total_kv_blocks: usize,
    pub shared_kv_pages: usize,
    pub cow_faults: usize,
    pub gpu_utilization_pct: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ControlResponse {
    SwarmAccepted {
        swarm_id: u64,
        operation_id: u128,
    },
    SwarmCancelled {
        swarm_id: u64,
    },
    Status(RuntimeStatusReport),
    ShutdownAck,
    TurnDelta {
        text: String,
    },
    TurnFinished {
        text: String,
        total_tokens: usize,
        /// Compose ledger id of the daemon's generation record for this turn
        /// (docs/DAEMON_GENERATION_RECORD.md). `None` = no record, no claim:
        /// no ledger, no model identity, or the append failed.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        generation_record: Option<u64>,
    },
    /// Result record of `RunComposeTask`.
    ComposeTaskResult(Box<ComposeTaskReport>),
    /// Result of `AllenStatus`.
    AllenStatusReport(Box<AllenStatusReport>),
    /// Result of `AllenProfileShow`, `AllenProfileSet`, `AllenProfileRevert`, `AllenProfileReset`.
    AllenProfile(Box<AllenProfileReport>),
    /// Result of `AllenProfileHistory`.
    AllenHistory(Box<AllenHistoryReport>),
    /// A profile command was refused; nothing was changed.
    AllenRefused(Box<AllenRefusalReport>),
    /// Result of an `AllenMemory*` or `AllenGoal*` command.
    AllenMemoryResult(Box<AllenMemoryReport>),
    /// Result of `ComposeNote`.
    ComposeNoted(ComposeNoteReport),
    /// Result of `ComposeRecall`.
    ComposeRecalled(Box<ComposeRecallReport>),
    /// Result of `RecoverComposeHome`.
    ComposeRecovered(Box<ComposeRecoverReport>),
    /// Result of `ComposeReconcile`.
    ComposeReconciled(Box<ComposeReconcileReport>),
    /// Result of `ComposeControl`.
    ComposeControlled(Box<ComposeControlReport>),
    /// `ComposeApprovedProposal` committed (state COMMITTED or ALREADY_COMMITTED).
    ComposeApprovedResult(Box<crate::approved::ApprovedComposeReport>),
    /// `ComposeApprovedProposal` refused (named; nothing ran unless the
    /// refusal says the run happened).
    ComposeApprovedRefused(Box<crate::approved::ApprovedRefusal>),
    Error(String),
}

/// Actor managing operator sessions and enforcing idempotent command execution.
/// Processed operation IDs persist to disk so idempotency survives restarts.
///
/// The state file path is fixed when the controller is built (#275): a later
/// change of `AIEN_RUNTIME_STATE_DIR` never redirects its writes. Every save
/// is a read-merge-write under an exclusive `flock` on `<file>.lock`, through
/// a temp file unique to that write, fsynced, then renamed (#299). Two
/// controllers sharing one file (threads or processes) therefore never tear
/// it and never drop each other's ids.
pub struct RuntimeController {
    processed_operations: HashSet<u128>,
    state_path: std::path::PathBuf,
}

/// The state file: `$AIEN_RUNTIME_STATE_DIR/processed_operations.json`, else
/// the machine-wide `/tmp/aien-runtime-processed-ops.json`. Unit tests of this
/// crate fall back to a per-process directory instead, so no lib test reads
/// state another program wrote.
fn operations_state_path() -> std::path::PathBuf {
    if let Ok(dir) = std::env::var("AIEN_RUNTIME_STATE_DIR") {
        if !dir.trim().is_empty() {
            return std::path::PathBuf::from(dir.trim()).join("processed_operations.json");
        }
    }
    #[cfg(test)]
    {
        let dir = std::env::temp_dir().join(format!("aien-runtime-unit-{}", std::process::id()));
        let _ = std::fs::create_dir_all(&dir);
        dir.join("processed_operations.json")
    }
    #[cfg(not(test))]
    std::path::PathBuf::from("/tmp/aien-runtime-processed-ops.json")
}

/// Reads the ids in `path`. Absent = empty; unreadable or unparsable = error.
fn read_operations(path: &std::path::Path) -> Result<Vec<u128>, String> {
    match std::fs::read(path) {
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(e) => Err(format!("idempotency state {}: {e}", path.display())),
        Ok(bytes) => serde_json::from_slice::<Vec<u128>>(&bytes).map_err(|e| {
            format!(
                "idempotency state {} is damaged ({e}); refusing to start rather than forget processed operations",
                path.display()
            )
        }),
    }
}

/// Exclusive advisory lock on `<state file>.lock`, released on drop.
struct StateLock(std::fs::File);

impl StateLock {
    fn acquire(path: &std::path::Path) -> Result<Self, String> {
        use std::os::fd::AsRawFd;
        let lock_path = path.with_extension("json.lock");
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(&lock_path)
            .map_err(|e| format!("idempotency lock {}: {e}", lock_path.display()))?;
        loop {
            // SAFETY: flock on a file descriptor this function owns.
            if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } == 0 {
                return Ok(Self(file));
            }
            let e = std::io::Error::last_os_error();
            if e.kind() != std::io::ErrorKind::Interrupted {
                return Err(format!("idempotency lock {}: {e}", lock_path.display()));
            }
        }
    }
}

impl Drop for StateLock {
    fn drop(&mut self) {
        use std::os::fd::AsRawFd;
        // SAFETY: unlock the descriptor locked in acquire (close would too).
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

static TMP_SEQ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

impl Default for RuntimeController {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeController {
    /// Panics when the state file exists but is damaged (see [`Self::load`]);
    /// the daemon calls `load` first and refuses to start instead.
    pub fn new() -> Self {
        Self::load().unwrap_or_else(|e| panic!("{e}"))
    }

    /// Loads the persisted operation ids from the default state file. An
    /// absent file is an empty set; a file that cannot be read or parsed is
    /// an error (NEXT-PHASE-2, ACCEPTANCE-v2 2.8): idempotency state is never
    /// reset silently.
    pub fn load() -> Result<Self, String> {
        Self::load_from(operations_state_path())
    }

    /// [`Self::load`] for an explicit state file, independent of the
    /// environment.
    pub fn load_from(state_path: std::path::PathBuf) -> Result<Self, String> {
        let ids = read_operations(&state_path)?;
        Ok(Self {
            processed_operations: ids.into_iter().collect(),
            state_path,
        })
    }

    /// The file this controller reads and writes.
    pub fn state_path(&self) -> &std::path::Path {
        &self.state_path
    }

    pub fn is_operation_processed(&self, op_id: u128) -> bool {
        self.processed_operations.contains(&op_id)
    }

    /// Records `op_id` in memory and on disk. The id stays recorded in memory
    /// even when the disk write fails; the error says the record is not
    /// durable (a restart would forget it). A damaged file on disk is an
    /// error and is never overwritten.
    pub fn mark_operation_processed(&mut self, op_id: u128) -> Result<(), String> {
        self.processed_operations.insert(op_id);
        let _lock = StateLock::acquire(&self.state_path)?;
        // Merge with what other controllers on this file recorded meanwhile.
        let on_disk = read_operations(&self.state_path)?;
        self.processed_operations.extend(on_disk);
        let ids: Vec<u128> = self.processed_operations.iter().copied().collect();
        let bytes = serde_json::to_vec(&ids).map_err(|e| format!("idempotency state: {e}"))?;
        // Write a temp file unique to this write, sync, then rename: a crash
        // or a concurrent writer never leaves a truncated state file.
        let tmp = self.state_path.with_extension(format!(
            "json.tmp.{}.{}",
            std::process::id(),
            TMP_SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        ));
        let write = || -> std::io::Result<()> {
            use std::io::Write;
            let mut f = std::fs::File::create(&tmp)?;
            f.write_all(&bytes)?;
            f.sync_all()?;
            std::fs::rename(&tmp, &self.state_path)
        };
        write().map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            format!(
                "idempotency state {}: write failed ({e}); operation {op_id:#x} is not durable",
                self.state_path.display()
            )
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn state_file(dir: &tempfile::TempDir) -> std::path::PathBuf {
        dir.path().join("processed_operations.json")
    }

    #[test]
    fn idempotency_survives_restart() {
        // Explicit path: this test never sets the process-wide
        // AIEN_RUNTIME_STATE_DIR other tests in this binary read (#275).
        let dir = tempfile::tempdir().unwrap();
        let path = state_file(&dir);
        let op_id = 0xC0FFEEu128;

        {
            let mut first = RuntimeController::load_from(path.clone()).unwrap();
            assert!(!first.is_operation_processed(op_id));
            first.mark_operation_processed(op_id).unwrap();
            assert!(first.is_operation_processed(op_id));
        }

        // Simulate a process restart: a fresh controller reloads from disk.
        {
            let second = RuntimeController::load_from(path.clone()).unwrap();
            assert!(
                second.is_operation_processed(op_id),
                "replayed operation ID must be rejected after restart"
            );
        }

        // NEXT-PHASE-2: a damaged state file is refused, never reset.
        std::fs::write(&path, b"{").unwrap();
        let err = RuntimeController::load_from(path.clone())
            .err()
            .expect("damaged state must be refused");
        assert!(err.contains("is damaged"), "{err}");
        std::fs::write(&path, b"[1, 2").unwrap();
        assert!(RuntimeController::load_from(path.clone()).is_err());
        // An empty file (what the #299 race left behind) is damaged too.
        std::fs::write(&path, b"").unwrap();
        assert!(RuntimeController::load_from(path).is_err());
    }

    #[test]
    fn a_save_never_overwrites_damaged_state() {
        let dir = tempfile::tempdir().unwrap();
        let path = state_file(&dir);
        let mut c = RuntimeController::load_from(path.clone()).unwrap();
        std::fs::write(&path, b"[1, 2").unwrap();
        let err = c.mark_operation_processed(7).unwrap_err();
        assert!(err.contains("is damaged"), "{err}");
        assert_eq!(
            std::fs::read(&path).unwrap(),
            b"[1, 2",
            "damaged file kept as found"
        );
        assert!(c.is_operation_processed(7), "still recorded in memory");
    }

    #[test]
    fn concurrent_saves_never_damage_or_drop_state() {
        // #299 root cause: every save used the temp name `json.tmp.<pid>`, so
        // two controllers in one process truncated each other's temp file and
        // renamed an empty file into place. Red on 12c1a5f: "is damaged (EOF
        // while parsing a value at line 1 column 0)".
        let dir = tempfile::tempdir().unwrap();
        let path = state_file(&dir);
        let threads: Vec<_> = (0..8u128)
            .map(|t| {
                let path = path.clone();
                std::thread::spawn(move || {
                    let mut c = RuntimeController::load_from(path).unwrap();
                    for i in 0..200u128 {
                        c.mark_operation_processed(t * 1000 + i).unwrap();
                    }
                })
            })
            .collect();
        for t in threads {
            t.join().unwrap();
        }
        let after = RuntimeController::load_from(path).expect("state loads after concurrent saves");
        for t in 0..8u128 {
            for i in 0..200u128 {
                assert!(
                    after.is_operation_processed(t * 1000 + i),
                    "id {t}/{i} forgotten"
                );
            }
        }
    }

    #[test]
    fn interrupted_save_leaves_the_last_good_state() {
        // A process killed mid-save leaves its temp file (and the lock file)
        // behind; the state file itself is the last complete write.
        let dir = tempfile::tempdir().unwrap();
        let path = state_file(&dir);
        let mut c = RuntimeController::load_from(path.clone()).unwrap();
        c.mark_operation_processed(1).unwrap();
        std::fs::write(path.with_extension("json.tmp.999999.0"), b"[1, 2, 3").unwrap();
        let mut restarted = RuntimeController::load_from(path.clone()).unwrap();
        assert!(restarted.is_operation_processed(1));
        assert!(
            !restarted.is_operation_processed(3),
            "half-written temp file is never read"
        );
        restarted.mark_operation_processed(2).unwrap();
        let again = RuntimeController::load_from(path).unwrap();
        assert!(again.is_operation_processed(1) && again.is_operation_processed(2));
    }

    #[test]
    fn path_is_fixed_at_construction() {
        // #275: a controller keeps writing where it loaded from, whatever a
        // parallel test does to AIEN_RUNTIME_STATE_DIR afterwards.
        let dir = tempfile::tempdir().unwrap();
        let path = state_file(&dir);
        let mut c = RuntimeController::load_from(path.clone()).unwrap();
        assert_eq!(c.state_path(), path.as_path());
        c.mark_operation_processed(5).unwrap();
        assert!(RuntimeController::load_from(path)
            .unwrap()
            .is_operation_processed(5));
    }

    #[test]
    fn chat_template_opens_an_assistant_turn() {
        let prompt = format_tinyllama_chat(&[
            ChatTurn {
                role: "system".into(),
                content: "You are AIEN.".into(),
            },
            ChatTurn {
                role: "user".into(),
                content: "Status?".into(),
            },
        ])
        .expect("chat");
        assert!(prompt.starts_with("<|system|>\nYou are AIEN.</s>\n"));
        assert!(prompt.contains("<|user|>\nStatus?</s>\n"));
        assert!(prompt.ends_with("<|assistant|>\n"));
    }
}

/// The record of one `RunComposeTask` (omega rxc_host result, NEXT-PHASE-1).
/// Cortex ids name records in `<compose dir>/cortex.cx`; digests are hex.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeTaskReport {
    /// The daemon's compose-commit record for this run (#261), when it
    /// committed a one-file proposal on the ordinary (non-approved) path.
    #[serde(default)]
    pub compose_commit: Option<u64>,
    /// Ledger id of the generation record of the attempt the committed
    /// proposal came from (arch#162). Evidence only; absent when no record
    /// was written (no model identity, approved proposal, failed append).
    #[serde(default)]
    pub generation_record: Option<u64>,
    /// Composition home (holds machine.id, cortex.cx, jspace).
    pub compose_dir: String,
    /// AienMachineId (32 bytes, hex) the home is bound to.
    pub machine_id: String,
    pub task: u64,
    /// RXC_OUT_*: 1 committed, 2 no winner, 3 not committed, 4 not durable, 5 record failed.
    pub outcome: i32,
    pub committed: bool,
    /// Staged J-Space branches forked (one per routed Skill alternative).
    pub branch_count: u32,
    pub branches_reclaimed: u32,
    /// AEGIS verdict: index of the winning alternative (None = no winner)
    /// and the pass mask (bit k = alternative k met the contract).
    pub winner: Option<u32>,
    pub aegis_pass_mask: u32,
    pub cx_goal: u64,
    pub cx_candidates: Vec<u64>,
    pub cx_evidence: u64,
    pub cx_promotion: u64,
    pub cx_admissions: Vec<u64>,
    pub winner_digest: String,
    pub record_digest: String,
    /// The committed proposal (the model Skill's output) and its sha256.
    pub proposal: Option<String>,
    pub proposal_sha256: Option<String>,
    /// The proposal parsed as one file change (AEGIS contract): relative path
    /// and the sha256 of the new content.
    pub proposal_path: Option<String>,
    pub proposal_content_sha256: Option<String>,
    /// When nothing committed: the model text AEGIS rejected (it did not
    /// parse as one file change), or why the model Skill returned nothing.
    #[serde(default)]
    pub uncommitted_proposal: Option<String>,
    #[serde(default)]
    pub proposer_error: Option<String>,
    /// Every proposal the model Skill made for this task (ACCEPTANCE-v2 3b).
    #[serde(default)]
    pub proposal_attempts: Vec<ProposalAttempt>,
    /// Requirements recognized in the goal (crate::requirements); the only
    /// ones checked. Empty = none recognized, nothing checked.
    #[serde(default)]
    pub requirements_recognized: Vec<String>,
    /// Goal text that looks like a measurable requirement but could not be
    /// interpreted reliably (crate::requirements::analyze). Non-empty = the
    /// task was refused before any model call.
    #[serde(default)]
    pub requirements_uncertain: Vec<String>,
    /// "model" when the Skill ran inference, or the stub label.
    pub proposer: String,
    /// ALLEN persona used for this task (arch#159). Old reports have none.
    #[serde(default)]
    pub persona: Option<PersonaReport>,
    /// ALLEN memory used for this task (arch#159). Old reports have none.
    #[serde(default)]
    pub memory: Option<MemoryReport>,
    /// Workspace files whose text the proposal prompt carried (sc#382): name
    /// relative to the workspace, sha256 and size of the bytes shown. Reading
    /// inside the workspace is not an effect; this list is its record.
    #[serde(default)]
    pub inputs: Vec<InputRef>,
    /// Input candidates that were not shown to the model, with the reason.
    #[serde(default)]
    pub inputs_omitted: Vec<InputOmitted>,
}

/// One workspace file shown to the model as input (sc#382).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputRef {
    pub name: String,
    pub sha256: String,
    pub bytes: usize,
}

/// An input candidate that was not shown, and why.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InputOmitted {
    pub name: String,
    pub reason: String,
}

/// One proposal the compose "model" Skill made (ACCEPTANCE-v2 3b).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProposalAttempt {
    /// 1-based.
    pub attempt: u32,
    /// Wall time of the model call.
    pub ms: u64,
    /// Generated tokens (0 when the call failed).
    pub tokens: usize,
    /// "parsed" | "refused" (template parser) | "timeout" | "error".
    pub outcome: String,
    /// Why the attempt was refused or failed.
    pub reason: Option<String>,
    pub text_sha256: Option<String>,
    pub text: Option<String>,
    /// For the attempt handed to AEGIS: "pass" or "fail".
    pub aegis: Option<String>,
    /// How generation stopped: "eos" | "max_tokens" | "aborted" | "preempted"
    /// (ACCEPTANCE-v5 Q3). None when the call failed or the proposer did not say.
    #[serde(default)]
    pub finish_reason: Option<String>,
    /// Generated token ids in order (NEXT-PHASE-1 v6 R1); None when the call
    /// failed or the proposer does not expose them.
    #[serde(default)]
    pub token_ids: Option<Vec<u32>>,
    /// How many prompt token ids were submitted (NEXT-PHASE-1 v6 R1).
    #[serde(default)]
    pub prompt_tokens: Option<usize>,
    /// sha256 of the prompt ids, each 4 little-endian bytes (NEXT-PHASE-1 v6 R1).
    #[serde(default)]
    pub prompt_ids_sha256: Option<String>,
    /// How the backend chose this attempt's tokens (sc#294). None when the
    /// call failed or nothing observed it; omitted from the JSON then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub decoding: Option<aien_abi_core::DecodeObservation>,
    /// The tensor backend and its op counters when this attempt's call
    /// finished (sc#337). None when the call failed or the backend does not
    /// account its ops; omitted from the JSON then.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ops: Option<aien_abi_core::OpEvidence>,
    /// Requirements of the goal this attempt's complete document failed
    /// (crate::requirements); empty when none failed or none were recognized.
    #[serde(default)]
    pub unmet_requirements: Vec<String>,
}

/// The record `ComposeNote` appended.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeNoteReport {
    pub machine_id: String,
    pub id: u64,
    pub kind: String,
    /// Cortex record digest (hex), re-read after the append.
    pub digest: String,
    /// sha256 (hex) of the note text.
    pub text_sha256: String,
    pub links: Vec<u64>,
}

/// One Cortex record as recalled (digest re-checked by omega).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeRecordView {
    pub id: u64,
    pub cls: u32,
    pub kind: u32,
    pub subject: u64,
    pub tag: u64,
    pub links: Vec<u64>,
    pub digest: String,
    pub verified: bool,
    /// "constraint", "authorization", "effect", "repair_tail" for host records.
    pub note: Option<String>,
    /// Host note text (sha256-checked), when the record is a note.
    pub text: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeRecallReport {
    pub compose_dir: String,
    pub machine_id: String,
    pub records_total: u64,
    /// Every host record (subject 0), oldest first.
    pub host: Vec<ComposeRecordView>,
    /// The records asked for by id (absent ids are listed in `missing`).
    pub cited: Vec<ComposeRecordView>,
    pub missing: Vec<u64>,
    /// sha256 over the digests of records 1..=prefix (hex), when asked.
    pub prefix: Option<u64>,
    pub prefix_digest: Option<String>,
    /// True when the daemon was built with the native Omega composition library
    /// (not the stub). Absent in older replies (reads as false).
    #[serde(default)]
    pub compose_native: bool,
    /// The omega commit the linked library was pinned to; empty for the stub.
    #[serde(default)]
    pub omega_sha: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeRecoverReport {
    pub compose_dir: String,
    pub repaired: bool,
    pub tail_torn: bool,
    /// CX_ERR_TORN (-8), RX_ERR_REPLAY (-21), 0 = nothing to repair.
    pub cause: i32,
    pub cut_lo: u64,
    pub cut_hi: u64,
    pub records_kept: u64,
    pub dropped_records: u64,
    pub anchor_records: u64,
    /// Cortex id of the repair record (0 = none).
    pub repair_record: u64,
    pub cut_bytes_kept: u64,
    pub cut_sha256: String,
    /// The composition opened after the repair (trial open).
    pub opens: bool,
    pub open_rc: i32,
    pub rolled_back: u32,
    pub recovered_completed: u32,
    /// ACCEPTANCE-v3 2.3: records the set-aside record mark said were lost
    /// (0 = the mark was kept or only damaged), where it was kept, and the
    /// host constraint record that names the repair (0 = none).
    #[serde(default)]
    pub mark_lost: u64,
    #[serde(default)]
    pub mark_kept_as: Option<String>,
    #[serde(default)]
    pub mark_repair_record: u64,
}

/// One unsettled effect as `ComposeReconcile` left it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconcileOutcome {
    pub intent: u64,
    pub authorization: u64,
    /// OPEN, DONE, NOT_DONE or UNRESOLVED.
    pub state: String,
    pub disk_sha256: Option<String>,
    /// The reconcile record appended (None: nothing recorded this time).
    pub record: Option<u64>,
    /// Who decided, or why nothing was decided.
    pub note: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeReconcileReport {
    pub compose_dir: String,
    pub machine_id: String,
    /// Unsettled intents looked at.
    pub checked: u64,
    pub outcomes: Vec<ReconcileOutcome>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ComposeControlReport {
    pub action: String,
    /// The control record appended (None: nothing recorded).
    pub recorded: Option<ComposeNoteReport>,
    /// For `revoke`: false when the grant was already spent or revoked.
    pub revoked: Option<bool>,
}

/// Which persona a task ran with (arch#159). `state`: `not_engaged` (ALLEN is
/// off, no persona text was added), `default` (no profile saved), `applied`
/// (profile `revision` in use) or `refused` (profile damaged or foreign:
/// defaults used, `reason` says why).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PersonaReport {
    pub state: String,
    pub display_name: String,
    pub revision: u64,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Latest deployment-record line, when the record exists and verifies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DeploymentSummary {
    pub seq: u64,
    pub candidate_id: String,
    pub placeholder: bool,
}

/// `AllenStatus`: identity, persona, model and what is not supported yet,
/// reported separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllenStatusReport {
    /// `not_engaged` or `engaged` (an engaged identity that cannot resolve stops the daemon).
    pub identity: String,
    /// First 8 hex characters of the agent id (engaged only).
    pub fingerprint: Option<String>,
    pub head_sequence: Option<u64>,
    pub chain_verified: Option<bool>,
    pub persona: PersonaReport,
    /// The proposer label the daemon was started with.
    pub model: String,
    /// `local_model` or `stub`.
    pub execution_mode: String,
    pub deployment: Option<DeploymentSummary>,
    /// Facilities that exist in the plan but not in v1; their controls do not exist.
    pub unsupported: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllenProfileReport {
    /// 0 = no profile saved (defaults in effect).
    pub revision: u64,
    pub state: String,
    /// The saved revision; `None` when none is saved.
    pub profile: Option<aien_allen_profile::Profile>,
    /// The values in effect when no profile is saved.
    pub defaults: Option<aien_allen_profile::Persona>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllenHistoryReport {
    pub entries: Vec<aien_allen_profile::HistoryEntry>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllenRefusalReport {
    pub code: String,
    /// Plain-language reason.
    pub message: String,
    pub current_revision: Option<u64>,
}

/// Which memory a task saw (arch#159). `state`: `not_requested` (no context
/// was given, nothing included), `not_engaged`, `included` (`items_included`
/// notes for exactly `context`) or `refused` (store damaged or foreign: the
/// task ran with no memory, `reason` says why). `items_omitted` notes were left
/// out by the count/byte bound; `items_unresolved` notes in the context could
/// not be read (key missing) and were not shown to the model.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryReport {
    pub context: Option<String>,
    pub items_included: usize,
    #[serde(default)]
    pub items_omitted: usize,
    #[serde(default)]
    pub items_unresolved: usize,
    pub state: String,
    #[serde(default)]
    pub reason: Option<String>,
}

/// Answer to a memory or goal command: what was done, in which context, and
/// the JSON result (items, ids, export).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AllenMemoryReport {
    pub action: String,
    /// The context the call named; `None` for an owner-wide call.
    pub context: Option<String>,
    pub result: serde_json::Value,
}
