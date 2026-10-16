//! Store-backed gates: `may_start`, `may_finalize`, event proof checks and
//! the post-replay audit.
//!
//! Every gate re-reads receipts from the harness store and compares them
//! with the producers' current generations at call time. Scheduler state
//! (which records what was accepted earlier) only says which receipt to
//! re-check; it is never the authority on its own.

use super::core::{Binding, BindingLevel, CommitResult, EventKind, Revocation, TaskEntry};
use super::error::SchedError;
use super::{Event, Outcome, Scheduler, RESERVED_ID_PREFIX};
use crate::completion::CompletionError;
use crate::execution::TaskState;
use crate::readiness::{verify_final, verify_interface, ArtifactReadiness, OperationClass};

fn proof(task: &str) -> impl Fn(CompletionError) -> SchedError + '_ {
    move |error| SchedError::Proof {
        task: task.to_string(),
        error,
    }
}

impl Scheduler {
    /// Checks that need the store, run before an event is journaled.
    pub(crate) fn verify_proofs(&self, kind: &EventKind) -> Result<(), SchedError> {
        match kind {
            EventKind::InterfacePublished {
                task,
                generation,
                claim,
            } => {
                self.core.require_generation(task, *generation)?;
                verify_interface(&self.store, task, *generation, claim).map_err(proof(task))?;
            }
            EventKind::CompletionVerified {
                task,
                generation,
                claim,
            } => {
                self.core.require_generation(task, *generation)?;
                verify_final(&self.store, task, *generation, claim).map_err(proof(task))?;
            }
            EventKind::Finalize { task, generation } => {
                self.core.require_generation(task, *generation)?;
                self.may_finalize(task)?;
            }
            EventKind::BeginCommit {
                task,
                generation,
                completion,
                ..
            } => {
                // Every producer re-verified (recursively), then the task's
                // own completion; `apply` checks the claims equal the state.
                self.core.require_generation(task, *generation)?;
                self.may_finalize(task)?;
                verify_final(&self.store, task, *generation, completion).map_err(proof(task))?;
            }
            _ => {}
        }
        Ok(())
    }

    /// May `task` run `op` right now? Requires a `Provisional` task with a
    /// live lease, a binding to every producer at the producer's current
    /// generation, and a store-verified readiness for each producer whose
    /// level permits `op`. Irreversible classes therefore need every
    /// producer final-verified. A grant only says the dependencies allow
    /// the operation; authorization and World commit stay with AEGIS and
    /// aien-runtime.
    pub fn may_start(
        &self,
        task: &str,
        op: OperationClass,
    ) -> Result<Vec<ArtifactReadiness>, SchedError> {
        let t = self.core.task(task)?;
        if t.state != TaskState::Provisional {
            return Err(SchedError::WrongState {
                task: task.to_string(),
                state: t.state,
                action: "start an operation",
            });
        }
        let now = self.now();
        if !t.lease_expires_at.is_some_and(|exp| now < exp) {
            return Err(SchedError::LeaseExpired(task.to_string()));
        }
        let mut out = Vec::new();
        for p in self.core.graph.producers_of(task) {
            let r = self.bound_readiness(task, t, p)?;
            if !r.permits(op) {
                return Err(SchedError::NotPermitted {
                    task: task.to_string(),
                    op,
                    producer: p.clone(),
                    level: r.level(),
                });
            }
            out.push(r);
        }
        Ok(out)
    }

    /// [`Scheduler::may_start`] for an operation name from outside (MCP,
    /// config, CLI). Any name that is not an [`OperationClass`] is refused.
    pub fn may_start_named(
        &self,
        task: &str,
        op: &str,
    ) -> Result<Vec<ArtifactReadiness>, SchedError> {
        let op = OperationClass::parse(op).map_err(SchedError::UnknownOperation)?;
        self.may_start(task, op)
    }

    /// May `task` become (or stay) `Committable`? Requires the task to be
    /// `AwaitingFinalProof` or `Committable`, every binding at the producer's
    /// current generation, an interface binding whose receipt still verifies
    /// (no interface change), and every producer final-verified: M1
    /// completion verifies for its current generation, the producer is
    /// `Committable`, and the same check passes for its own producers.
    pub fn may_finalize(&self, task: &str) -> Result<Vec<ArtifactReadiness>, SchedError> {
        self.finalize_check(task, 0)
    }

    fn finalize_check(
        &self,
        task: &str,
        depth: usize,
    ) -> Result<Vec<ArtifactReadiness>, SchedError> {
        if depth > self.core.tasks.len() {
            return Err(SchedError::InvalidEvent(
                "dependency recursion deeper than the task count".to_string(),
            ));
        }
        let t = self.core.task(task)?;
        // A producer (depth > 0) may also be frozen in `Committing`.
        if !matches!(
            t.state,
            TaskState::AwaitingFinalProof | TaskState::Committable
        ) && !(depth > 0 && t.state == TaskState::Committing)
        {
            return Err(SchedError::WrongState {
                task: task.to_string(),
                state: t.state,
                action: "finalize",
            });
        }
        let mut out = Vec::new();
        for p in self.core.graph.producers_of(task) {
            let b = self.binding(task, t, p)?;
            if b.level == BindingLevel::Interface {
                verify_interface(&self.store, p, b.generation, &b.claim).map_err(proof(p))?;
            }
            out.push(self.producer_final(p, depth + 1)?);
        }
        Ok(out)
    }

    fn producer_final(&self, p: &str, depth: usize) -> Result<ArtifactReadiness, SchedError> {
        let pt = self.core.task(p)?;
        let claim = match (&pt.completion, pt.final_ready()) {
            (Some(c), true) => c,
            _ => {
                return Err(SchedError::ProducerNotFinal {
                    producer: p.to_string(),
                    detail: format!(
                        "state {}, completion accepted: {}",
                        pt.state.as_str(),
                        pt.completion.is_some()
                    ),
                })
            }
        };
        let r = verify_final(&self.store, p, pt.generation, claim).map_err(proof(p))?;
        // A committed producer is done: its own completion is the proof,
        // and its upstream may have moved on since (see `Core::affected`).
        if pt.state != TaskState::Committed {
            self.finalize_check(p, depth)?;
        }
        Ok(r)
    }

    fn binding<'a>(
        &self,
        task: &str,
        t: &'a TaskEntry,
        p: &str,
    ) -> Result<&'a Binding, SchedError> {
        let stale = |detail: String| SchedError::StaleBinding {
            task: task.to_string(),
            producer: p.to_string(),
            detail,
        };
        let current = self.core.task(p)?.generation;
        let b = t
            .bindings
            .get(p)
            .ok_or_else(|| stale("no binding".to_string()))?;
        if b.generation != current {
            return Err(stale(format!(
                "bound to generation {}, producer is at {current}",
                b.generation
            )));
        }
        Ok(b)
    }

    /// Re-verified readiness of producer `p` as `task` is bound to it. An
    /// interface binding is upgraded to final when the producer is final and
    /// that proof verifies; if the upgrade fails the interface level (the
    /// more restrictive one) is returned.
    fn bound_readiness(
        &self,
        task: &str,
        t: &TaskEntry,
        p: &str,
    ) -> Result<ArtifactReadiness, SchedError> {
        let b = self.binding(task, t, p)?;
        match b.level {
            BindingLevel::Interface => {
                let r =
                    verify_interface(&self.store, p, b.generation, &b.claim).map_err(proof(p))?;
                if self.core.task(p)?.final_ready() {
                    if let Ok(f) = self.producer_final(p, 1) {
                        return Ok(f);
                    }
                }
                Ok(r)
            }
            BindingLevel::Final => self.producer_final(p, 1),
        }
    }

    /// Active tasks whose own accepted interface or completion claim no
    /// longer verifies.
    fn own_claim_failures(&self) -> Vec<Revocation> {
        let mut out = Vec::new();
        for (id, t) in self.core.tasks.iter().filter(|(_, t)| t.state.is_active()) {
            let g = t.generation;
            let i = t
                .interface
                .as_ref()
                .map(|c| verify_interface(&self.store, id, g, c));
            let f = t
                .completion
                .as_ref()
                .map(|c| verify_final(&self.store, id, g, c));
            if let Some(e) = [i, f].into_iter().flatten().find_map(Result::err) {
                out.push(revocation(id, proof(id)(e)));
            }
        }
        out
    }

    /// Active tasks whose bindings or finalize check no longer verify.
    fn binding_failures(&self) -> Vec<Revocation> {
        let mut out = Vec::new();
        for (id, t) in &self.core.tasks {
            let check = match t.state {
                TaskState::Provisional | TaskState::AwaitingFinalProof => self
                    .core
                    .graph
                    .producers_of(id)
                    .try_for_each(|p| self.bound_readiness(id, t, p).map(|_| ())),
                TaskState::Committable => self.may_finalize(id).map(|_| ()),
                // Frozen or done: not re-checked (see `audit`).
                TaskState::Blocked
                | TaskState::Aborted
                | TaskState::Committing
                | TaskState::Committed
                | TaskState::CommitUncertain => Ok(()),
            };
            if let Err(e) = check {
                out.push(revocation(id, e));
            }
        }
        out
    }

    /// Re-verify every active task against the store and revoke the ones
    /// that no longer verify. Run whenever the store may have changed; a
    /// replayed journal must pass through it (see [`super::Unaudited`]).
    /// Returns the revoked tasks.
    ///
    /// Each round is one journaled `AuditRevoke`: first every task whose own
    /// claim fails (which cascades to its consumers), and only when there
    /// are none, every task whose bindings fail. One batch dispatches once,
    /// after all of its revocations, so no consumer is dispatched against a
    /// claim the same round is about to revoke. Each round revokes at least
    /// one active task; the round count is capped as a backstop.
    ///
    /// A revocation that would reach a `Committing` task is deferred (left
    /// out of the batch): the commit was checked against the store when it
    /// began, and a later audit catches the change once the outcome is in.
    pub fn audit(&mut self) -> Result<Vec<String>, SchedError> {
        let mut revoked = Vec::new();
        let ready = |s: &Self, mut b: Vec<Revocation>| {
            b.retain(|r| s.core.guard_commit(&r.task).is_ok());
            b
        };
        for _ in 0..=2 * self.core.tasks.len() {
            let mut batch = ready(self, self.own_claim_failures());
            if batch.is_empty() {
                batch = ready(self, self.binding_failures());
            }
            if batch.is_empty() {
                return Ok(revoked);
            }
            let event = Event {
                id: format!("{RESERVED_ID_PREFIX}audit-{}", self.core.seq + 1),
                kind: EventKind::AuditRevoke { revocations: batch },
            };
            if let Outcome::Applied(fx) = self.deliver_unchecked(event)? {
                revoked.extend(fx.revoked);
            }
        }
        Err(SchedError::InvalidEvent(
            "audit did not converge; refusing to continue".to_string(),
        ))
    }

    /// After a replay: every task still `Committing` lost its outcome with
    /// the old process. Its commit may or may not have reached the World, so
    /// it becomes `CommitUncertain` through a journaled `CommitOutcome` and
    /// is never run again (as `approved_replay` treats an in-flight claim).
    /// Returns those tasks.
    pub(crate) fn settle_interrupted_commits(&mut self) -> Result<Vec<String>, SchedError> {
        let open: Vec<(String, u64)> = self
            .core
            .tasks
            .iter()
            .filter(|(_, t)| t.state == TaskState::Committing)
            .map(|(id, t)| (id.clone(), t.generation))
            .collect();
        for (task, generation) in &open {
            let event = Event {
                id: format!("{RESERVED_ID_PREFIX}commit-uncertain-{}", self.core.seq + 1),
                kind: EventKind::CommitOutcome {
                    task: task.clone(),
                    generation: *generation,
                    outcome: CommitResult::Uncertain {
                        reason: "restarted while committing; no outcome was recorded".to_string(),
                    },
                },
            };
            self.deliver_unchecked(event)?;
        }
        Ok(open.into_iter().map(|(t, _)| t).collect())
    }
}

fn revocation(task: &str, e: SchedError) -> Revocation {
    Revocation {
        task: task.to_string(),
        reason: format!("audit: {e}"),
    }
}
