# Headstart provisional preparation: rollout and rollback

Status: **off everywhere**. Issue: aien-spark/aien-sovereign-core#538 (M6).
This page says how provisional preparation (Headstart M2 scheduler plus the
M3 adapter, #537) is switched on and off, what fails closed, what a restart
does, and what must be true before anyone activates it. Line references are
to this repository at the commit that added this page.

## 1. Two switches, both off by default

1. **Cargo feature `provisional-prep`** on `aien-runtime`
   (`crates/aien-runtime/Cargo.toml`, the `[features]` table:
   `provisional-prep = ["dep:spark-harness", "dep:aien-proof"]`). It is not in
   `default`, and no other crate in the workspace enables it. Without it the
   `provisional` module, `AienRuntimeServer::with_prep_adapter` and every
   fence branch in `server.rs` are not compiled (`#[cfg(feature =
   "provisional-prep")]`), and the control ABI is identical in both builds.
2. **Runtime flag `ProvisionalConfig::enabled`**
   (`crates/spark-harness/src/scheduler/mod.rs`, `impl Default for
   ProvisionalConfig`: `enabled: false`). With the flag off, M2 dispatches a
   consumer only when every producer is final (committable with a verified
   M1 completion), which is plain sequential behavior.

Enable, in order: build with `--features provisional-prep`, construct the
adapter with `PrepAdapter::open(ProvisionalConfig { enabled: true, .. }, store,
journal, clock)`, and hand it to the daemon with
`AienRuntimeServer::with_prep_adapter`. Disable in reverse: drop the adapter
(no `with_prep_adapter` call, so nothing is fenced and every compose request
runs the original call unchanged), then build without the feature.

**The daemon installs no adapter today.** No code outside tests calls
`with_prep_adapter`, so even a feature-on build fences nothing and prepares
nothing. Activation in production needs a separate, reviewed change.

**Manual off switch.** There is no environment variable or config file for
this today. The off switch is the code path above: start the daemon without
an adapter (the default), or with `enabled: false`. Both are deterministic
startup defaults. A rebuild without the feature removes the code entirely.

## 2. Default-off proof

| Claim | Check |
| --- | --- |
| The default build links neither `spark-harness` nor `aien-proof` | CI job "Headstart feature-on tests (provisional-prep)", step "Feature-off invariant": `cargo tree -e normal -p aien-runtime` must name neither crate (with the feature on, both appear: the negative control) |
| The default build's tests are unchanged | the required job "Workspace Verification & Invariant Audit" runs `cargo test --workspace` with default features |
| The flag defaults to off and is sequential | `disabled_flag_is_the_default_and_sequential` (aien-runtime `tests/provisional_prep_test.rs`), `disabled_is_the_default` (spark-harness `tests/provisional_scheduler.rs`) |
| With the flag off nothing prepares on an interface | `headstart_ab_smoke` asserts `preparations_provisional == 0` on every off arm |

## 3. What may start early, and what still needs the final fence

Safe to start early (flag on, producer only interface-verified):

- exactly one operation, `PrepOperation::ComposePlanPreview`
  (`OperationClass::ReadOnlyAnalysis`): it reads the workspace to build the
  compose prompt and destination. It opens no compose home, runs no model,
  calls no tool, writes nothing and uses no network. Its result is held in
  memory only and is never journaled.

Needs the final fence (task `Committable`, every producer final-verified
recursively through `may_finalize`, and the task's own M1 completion receipt
verifying through `verify_final`):

- `ComposeApprovedProposal` (`PrepAdapter::submit_approved`, the World commit);
- `ComposeAuthorize` (`PrepAdapter::authorize`, the grant).

The fence is a prerequisite only. Desk MAC, AEGIS verify, the replay ledger,
the effect ledger and the World commit stay inside the unchanged calls and
still decide.

## 4. Fail-closed states

| State or error | Cause | What happens | Who clears it |
| --- | --- | --- | --- |
| `SchedError::BindingsFull` | a task already holds `MAX_APPROVAL_BINDINGS_PER_TASK` (64, `crates/spark-harness/src/scheduler/core.rs`) approval keys, all at its current generation | the new binding is refused; nothing is fenced or run for it. A full task whose keys include bindings at a superseded generation (left by retries) is not refused: the new key reclaims the lowest superseded one (lowest generation, then lowest key), deterministically, so retries cannot exhaust the task (`a_full_task_reclaims_its_superseded_bindings_oldest_first`) | the caller (fewer keys, or a new generation) |
| `DeliverError::DeferredFull` | a commit is running and `DEFERRED_CAPACITY` (64, `crates/aien-runtime/src/provisional.rs`) events already wait | the event is refused and nothing is queued | the caller redelivers after the commit ends; a queued event refused as `StaleGeneration` on redelivery is reported (`Redelivered::dropped`) and dropped, since it can never apply; one refused for any other reason holds a slot until `PrepAdapter::discard_deferred` |
| `SchedError::CommitInProgress` | an event would change a task (or an ancestor) that is `Committing`, or a second commit in the same chain | through `deliver` the event is queued and redelivered after the outcome | automatic |
| `CommitUncertain` | the commit call panicked, or a restart found a `Committing` task with no recorded outcome | terminal: never run again, consumers revoked, bindings kept for good (a bound request delivered again is refused by the fence, `AlreadySubmitted`, instead of running unfenced; only a request under a new key, which nobody bound, runs as the original call, and the durable ledgers decide; `an_uncertain_commit_keeps_the_binding_so_the_request_stays_fenced`) | **an operator**: check the World and the approved replay ledger to learn whether the effect landed; M2 has no retry from this state (`Retry` needs `Aborted`) |
| `FenceError::NotCommittable`, `NoCompletion`, `CompletionInvalid`, `StaleGeneration`, `AlreadySubmitted` | the fence checks failed | the call does not run | the workflow (finish the task, or bind again at the new generation) |

## 5. Restart semantics

- The M2 journal replays to the same state, and `open` then audits every
  active task against the M1 store and revokes the ones whose receipts no
  longer verify (`Unaudited::audit`, `scheduler/mod.rs`).
- A replayed `Committing` task becomes `CommitUncertain` and never runs again.
- Preparation results are never journaled: after a restart a `Provisional`
  task has no result and prepares again.
- An unfinished task is never treated as committed. `Committable` is reached
  only through a journaled `Finalize`, which M2 journals only after its
  producers were final. A completion receipt minted in M1 but never
  delivered to M2 is not a completion after replay; the fence refuses
  (`NoCompletion`) until it is delivered. Tests:
  `a_restart_with_an_unfinished_task_is_never_treated_as_committed`,
  `restart_with_a_provisional_task_drops_the_result_and_prepares_again`,
  `a_restart_during_a_commit_is_uncertain_and_never_runs_it_again`
  (aien-runtime), `replay_never_turns_provisional_work_committable`
  (spark-harness `tests/provisional_journal.rs`).
- Journaled deferred events are delivered again on every open.
- **Untested:** reopening a journal written with `enabled: true` under
  `enabled: false`. Until a test covers it, switch the flag off only with a
  fresh journal or no adapter.

## 6. Activation checklist (#538 task 5)

Every item must hold, with evidence linked from #538, before any production
change installs an adapter. Today none of the timing items is met.

- [ ] Zero unauthorized or double commits: the fence tests above, the A/B
      sample gate (`check_sample` in `crates/aien-runtime/tests/headstart_ab.rs`:
      no task committed twice, every commit record final-verified) over the
      held samples.
- [ ] Zero unhandled state divergence: `check_invariants` after every sample,
      and equal final states and commit counts between the off and on arms
      (compared by the smoke run only; the held-sample script does not
      compare arms).
- [ ] Exact baseline behavior with the flag off: section 2.
- [ ] No new OOM or deadlock: every sample reports `deadlock: false` and
      `max_rss_kb_process`; compare arms.
- [ ] Resource budgets enforced: `BindingsFull`, `DeferredFull` and M2 permits
      (`the_binding_table_is_bounded_per_task_and_fails_closed`,
      `the_deferred_queue_is_bounded_and_fails_closed`,
      `concurrency_is_bounded_by_the_m2_permits`,
      `resource_budget_bounds_queue_and_permits`).
- [ ] At least 10% repeatable wall-time improvement on one representative
      workflow: held LINKED A/B samples (`scripts/bench/headstart-m6-ab.sh`),
      with the A/A spread smaller than the A/B difference.
      **NOT MET** on 2026-10-16 (section 7).
- [ ] The daemon wiring change itself reviewed, with this page updated.

## 7. Measured, hypothesis, untested

- **Measured (gate NOT MET):** held LINKED samples on the Spark, 2026-10-16
  (`benchmarks/data/headstart/m6-2026-10-16/RESULTS.md`, n=5 per arm). The
  on arm is 1.06% slower on chain and 1.90% slower on parallel, inside the
  A/A spread (1.31% and 2.70%). No measurable gain; the feature stays off.
  Smoke runs are single unheld runs labelled SMOKE and are not results.
- **Hypothesis:** on the chain shape the preparable share of a sample is
  small next to the producer work and the World commit (the per-sample
  `ceiling_estimate` field, prep or producer time over wall time). The held samples agree (estimate at most about 2% to 6%): the
  gate needs a workload where read-only preparation is a larger share.
- **Platforms:** results from the CI runner (`CI-RUNNER`, x86_64, hosted
  Ubuntu) are correctness only; timing comes from held samples on the Spark
  (aarch64). Neither says anything about the other architecture. The CI job
  builds the STUB composition library; feature-on tests against the real
  linked library run only locally (the linked CI job does not enable
  `provisional-prep`).
- **M5** experimental compiler results are independent of these runtime
  readiness results and are reported separately.
