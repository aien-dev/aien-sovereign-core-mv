//! Every way the provisional scheduler refuses an event or a gate query.

use crate::completion::CompletionError;
use crate::dependency::GraphError;
use crate::execution::{IllegalTransition, TaskState};
use crate::readiness::{OperationClass, ReadinessLevel, UnknownOperation};
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SchedError {
    InvalidEvent(String),
    UnknownTask(String),
    Graph(GraphError),
    /// The task can never run: its request exceeds the whole budget.
    ExceedsCapacity(String),
    /// The event refers to a generation older than the task's current one.
    StaleGeneration {
        task: String,
        submitted: u64,
        current: u64,
    },
    /// The event refers to a generation the scheduler has not reached.
    FutureGeneration {
        task: String,
        submitted: u64,
        current: u64,
    },
    /// The task is in a state where this event or query is not allowed.
    WrongState {
        task: String,
        state: TaskState,
        action: &'static str,
    },
    Transition(IllegalTransition),
    LeaseExpired(String),
    /// A heartbeat whose beat number is not above the last accepted one
    /// for the current lease (a redelivery or a reordered beat).
    StaleHeartbeat {
        task: String,
        beat: u64,
        last: u64,
    },
    /// The same event id was delivered with a different payload.
    EventIdConflict(String),
    UnknownOperation(UnknownOperation),
    /// The producer's verified readiness does not allow this operation.
    NotPermitted {
        task: String,
        op: OperationClass,
        producer: String,
        level: ReadinessLevel,
    },
    /// The consumer has no binding to the producer (it was not dispatched
    /// against it), or the binding is for an older generation.
    StaleBinding {
        task: String,
        producer: String,
        detail: String,
    },
    /// A producer is not final-verified for its current generation.
    ProducerNotFinal {
        producer: String,
        detail: String,
    },
    /// Re-verification from the harness store failed.
    Proof {
        task: String,
        error: CompletionError,
    },
    Journal(String),
    /// The event would revoke, cancel, fail or re-generate a `Committing`
    /// task (`committing`), directly or through `task` upstream of it, or
    /// would start a second commit in the same dependency chain. Deliver it
    /// again after the commit outcome is recorded.
    CommitInProgress {
        task: String,
        committing: String,
    },
    /// A new request key would exceed the task's share of the
    /// bound-request table.
    BindingsFull {
        task: String,
        capacity: usize,
    },
}

impl fmt::Display for SchedError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SchedError::InvalidEvent(m) => write!(f, "invalid event: {m}"),
            SchedError::UnknownTask(t) => write!(f, "unknown task '{t}'"),
            SchedError::Graph(e) => write!(f, "dependency graph: {e}"),
            SchedError::ExceedsCapacity(t) => {
                write!(f, "task '{t}' requests more than the whole resource budget")
            }
            SchedError::StaleGeneration {
                task,
                submitted,
                current,
            } => write!(
                f,
                "stale generation for '{task}': event {submitted}, current {current}"
            ),
            SchedError::FutureGeneration {
                task,
                submitted,
                current,
            } => write!(
                f,
                "unknown future generation for '{task}': event {submitted}, current {current}"
            ),
            SchedError::WrongState {
                task,
                state,
                action,
            } => write!(f, "task '{task}' is {}; cannot {action}", state.as_str()),
            SchedError::Transition(e) => write!(f, "{e}"),
            SchedError::LeaseExpired(t) => write!(f, "lease of task '{t}' expired"),
            SchedError::StaleHeartbeat { task, beat, last } => write!(
                f,
                "heartbeat {beat} for task '{task}' is not above the last accepted beat {last}"
            ),
            SchedError::EventIdConflict(id) => {
                write!(f, "event id '{id}' was already used for a different event")
            }
            SchedError::UnknownOperation(e) => write!(f, "{e}"),
            SchedError::NotPermitted {
                task,
                op,
                producer,
                level,
            } => write!(
                f,
                "task '{task}' may not run '{}': producer '{producer}' is only {level:?}",
                op.as_str()
            ),
            SchedError::StaleBinding {
                task,
                producer,
                detail,
            } => write!(f, "task '{task}' binding to '{producer}': {detail}"),
            SchedError::ProducerNotFinal { producer, detail } => {
                write!(f, "producer '{producer}' is not final-verified: {detail}")
            }
            SchedError::Proof { task, error } => {
                write!(f, "proof for '{task}' failed re-verification: {error}")
            }
            SchedError::Journal(m) => write!(f, "journal: {m}"),
            SchedError::CommitInProgress { task, committing } => write!(
                f,
                "task '{committing}' is committing; '{task}' cannot change it until its outcome is recorded"
            ),
            SchedError::BindingsFull { task, capacity } => write!(
                f,
                "task '{task}' holds its maximum of {capacity} bound request keys; no new key is bound"
            ),
        }
    }
}

impl std::error::Error for SchedError {}

impl From<GraphError> for SchedError {
    fn from(e: GraphError) -> Self {
        SchedError::Graph(e)
    }
}

impl From<IllegalTransition> for SchedError {
    fn from(e: IllegalTransition) -> Self {
        SchedError::Transition(e)
    }
}
