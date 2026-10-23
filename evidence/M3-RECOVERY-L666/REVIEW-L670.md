# L670 independent fix-pass review

## Verdict

No Important/Critical defects found in the frozen six-file correction. The prior L668 fail-open path is addressed in source: reclaimed active bindings move to `retired_approvals`; `Scheduler::approval` checks active and retired keys; retired history is independently capped per task; full history rejects a new key; same-key rebind is processed transactionally through the cloned core; and committed outcomes clear both tables while uncertain outcomes retain them. Journal replay deterministically rebuilds retired records from the unchanged `BindApproval` event stream. The runtime route test added in source exercises an evicted key before and after restart and checks that workspace/compose state is unchanged.

Deferred stale-generation redelivery removal is also narrowly scoped: only `StaleGeneration` refusals leave the queue, with a reported `dropped` result; other refusals remain queued. The monotonic-generation rationale and sidecar rewrite path are consistent in the reviewed source.

## Minor documentation drift

`PrepAdapter::bind_approval`'s method comment in `crates/aien-runtime/src/provisional.rs` still describes a new key past the per-task cap as always refused and says bindings are dropped when a commit ends. The new behavior can reclaim a superseded active binding while retired capacity remains, and only a known committed result releases bindings. The module-level and scheduler docs describe the new behavior correctly. This is non-blocking documentation cleanup, not a routing defect.

## Scope and verification

- Read the complete L670 instructions, L668 finding, and six-file source diff against `754c07b118f41ad899607b60329bd851b1ccb13e`.
- Verified HEAD is `fc62b5f2cf576a24ac7200ed1cb660aa9ee1e625` and all six hashes in `L670-inputs.json` match both before and after review.
- Worktree still has the six pinned modified files and pre-existing untracked `evidence/`; no source edits were made.
- No builds or tests were run, per instruction. This is a source review; parent owns runtime/harness execution and verification of the added tests.
