//! Task execution states for the provisional scheduler (Headstart M2,
//! issue #535).
//!
//! ```text
//!   Blocked --dispatch--> Provisional --work done--> AwaitingFinalProof --finalize--> Committable
//!      ^                      |                           |                               |
//!      +------ revoke --------+---------------------------+-------------------------------+
//!   any of the first four --cancel / failure / lease expiry--> Aborted --retry--> Blocked
//!
//!   Committable --begin commit--> Committing --outcome--> Committed | CommitUncertain | Aborted
//! ```
//!
//! `Committable` means "every dependency is final-verified for its current
//! generation". Committing itself is done by the existing effect/World
//! machinery in aien-runtime, which this crate does not touch. The optional
//! `Committing` state (Headstart M3, issue #537) lets a caller record that
//! it is about to run such a commit: a journaled `BeginCommit` freezes the
//! task's generation and the producer completions it was checked against,
//! and a journaled `CommitOutcome` ends it. While a task is `Committing`,
//! every event that would revoke, cancel, fail or re-generate it (directly
//! or through an upstream producer) is refused. A `Committing` task found
//! after a restart has no recorded outcome and becomes `CommitUncertain`;
//! it is never run again. `Committed` and `CommitUncertain` are terminal.
//!
//! Revocation returns a task to `Blocked` rather than `Aborted`. The task did
//! nothing wrong: an upstream producer failed, changed its interface or
//! started a new generation. Its provisional result is discarded and it waits
//! to be dispatched again against the new upstream state, without a manual
//! retry. A task fails into `Aborted` only for its own reasons (its own
//! failure, cancellation, lost lease).
//!
//! Only `Provisional` holds resource permits. Leaving `Provisional` releases
//! them, so every other state holds none.

use serde::{Deserialize, Serialize};
use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum TaskState {
    Blocked,
    Provisional,
    AwaitingFinalProof,
    Committable,
    Aborted,
    /// A commit of this task's result is running (see the module docs).
    Committing,
    /// The commit reported success. Terminal.
    Committed,
    /// The commit's outcome is unknown (it may have reached the World).
    /// Terminal: never run again; an operator resolves it.
    CommitUncertain,
}

impl TaskState {
    pub fn as_str(self) -> &'static str {
        match self {
            TaskState::Blocked => "blocked",
            TaskState::Provisional => "provisional",
            TaskState::AwaitingFinalProof => "awaiting_final_proof",
            TaskState::Committable => "committable",
            TaskState::Aborted => "aborted",
            TaskState::Committing => "committing",
            TaskState::Committed => "committed",
            TaskState::CommitUncertain => "commit_uncertain",
        }
    }

    /// `Committed` or `CommitUncertain`: the task is done and no upstream
    /// change reaches it or, through it, its consumers.
    pub fn is_commit_terminal(self) -> bool {
        matches!(self, TaskState::Committed | TaskState::CommitUncertain)
    }

    /// A completion that consumers may treat as final.
    pub fn is_final(self) -> bool {
        matches!(
            self,
            TaskState::Committable | TaskState::Committing | TaskState::Committed
        )
    }

    /// States whose work or proof is built on upstream artifacts and must be
    /// revoked when upstream changes.
    pub fn is_active(self) -> bool {
        matches!(
            self,
            TaskState::Provisional | TaskState::AwaitingFinalProof | TaskState::Committable
        )
    }

    /// Only a running task holds resource permits.
    pub fn holds_permits(self) -> bool {
        self == TaskState::Provisional
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IllegalTransition {
    pub from: TaskState,
    pub to: TaskState,
}

impl fmt::Display for IllegalTransition {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "illegal task transition {} -> {}",
            self.from.as_str(),
            self.to.as_str()
        )
    }
}

impl std::error::Error for IllegalTransition {}

/// The complete transition table. Anything not listed is refused.
pub fn check_transition(from: TaskState, to: TaskState) -> Result<(), IllegalTransition> {
    use TaskState::*;
    let ok = matches!(
        (from, to),
        (Blocked, Provisional)
            | (Provisional, AwaitingFinalProof)
            | (AwaitingFinalProof, Committable)
            | (Provisional | AwaitingFinalProof | Committable, Blocked)
            | (
                Blocked | Provisional | AwaitingFinalProof | Committable,
                Aborted
            )
            | (Aborted, Blocked)
            | (Committable, Committing)
            | (Committing, Committed | CommitUncertain | Aborted)
    );
    if ok {
        Ok(())
    } else {
        Err(IllegalTransition { from, to })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use TaskState::*;

    #[test]
    fn table_refuses_shortcuts() {
        for (from, to) in [
            (Blocked, AwaitingFinalProof),
            (Blocked, Committable),
            (Provisional, Committable),
            (Aborted, Provisional),
            (Aborted, Committable),
            (Committable, Provisional),
            (Blocked, Blocked),
            (AwaitingFinalProof, Committing),
            (Committing, Blocked),
            (Committing, Committable),
            (Committed, Blocked),
            (Committed, Aborted),
            (Committed, Committing),
            (CommitUncertain, Blocked),
            (CommitUncertain, Aborted),
            (CommitUncertain, Committing),
        ] {
            assert!(check_transition(from, to).is_err(), "{from:?} -> {to:?}");
        }
        for (from, to) in [
            (Blocked, Provisional),
            (Provisional, AwaitingFinalProof),
            (AwaitingFinalProof, Committable),
            (Committable, Blocked),
            (Aborted, Blocked),
            (Committable, Committing),
            (Committing, Committed),
            (Committing, CommitUncertain),
            (Committing, Aborted),
        ] {
            assert!(check_transition(from, to).is_ok(), "{from:?} -> {to:?}");
        }
    }
}
