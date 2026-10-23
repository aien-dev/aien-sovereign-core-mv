# L668 independent source review

**Candidate:** `/Users/drakestapleton/.hermes/cache/scratch/loop19-m3-559`, `fc62b5f2cf576a24ac7200ed1cb660aa9ee1e625` against `754c07b118f41ad899607b60329bd851b1ccb13e`.

## Verdict

One important fail-open defect in superseded-binding reclamation. The deferred-slot reclamation and uncertain-commit fence changes look correct in source. Deterministic journal replay is addressed by the implementation and its tests. No source edits or builds/tests were performed; parent owns test execution.

## Finding: reclaiming a stale binding makes its request silently unbound

**Severity: Important.** In `crates/spark-harness/src/scheduler/core.rs:947-957`, when a task at its binding cap gets a new key, the scheduler removes the lowest-key binding at a superseded generation. That key is not retained as a tombstone or otherwise remembered as a request that must stay fenced.

This breaks the invariant the surrounding code depends on: a request remains fenced until deliberately rebound, and stale requests are refused rather than run unfenced. `crates/aien-runtime/src/provisional.rs:1204-1206` / `1226-1228` resolve the request key with `binding(...)` and return `None` if no binding exists. The actual callers in `crates/aien-runtime/src/server.rs:1111-1120` and `1043-1057` then execute the original approved-submit / authorize call when the route returns `None`.

Concrete sequence: bind request key K to task T at generation 1; cancel/retry T so it is generation 2 while K remains bound to generation 1; fill T's per-task table with 64 bindings; bind a distinct key at generation 2. The new code reclaims K (if it is the selected lowest superseded key). A later delivery of K now misses the fence lookup and runs the original call, instead of producing the stale-generation refusal it would have produced while bound. This is the precise path the new test currently permits by asserting the old binding is `None` (`crates/spark-harness/tests/provisional_commit.rs:792-800`), but it never redelivers that key through the real runtime routing path.

Reclamation should not erase the fact that the external key was fenced. A durable retired/revoked-key state (or another lookup result that routes the request to an explicit refusal) is needed if old keys are to be evicted from the bounded active-binding table. Otherwise eviction must not be used for these keys.

## Acceptance checks

- **Free deferred slots on terminal stale-generation redelivery:** Source does this in `provisional.rs` `Inner::redeliver` (around lines 525-558): stale-generation refusals are returned with `dropped: true` and removed; other refusals remain queued. The scheduler checks generations monotonically (`scheduler/core.rs` generation validation), so the stale event cannot apply later. Tests cover queue and sidecar removal, including restart (`provisional_prep_test.rs` around 2356-2455). No issue found in this change.
- **Keep uncertain commits fenced, including panic before writes:** `provisional.rs:989-1007` catches panic, records `CommitResult::Uncertain`, then resumes it; `scheduler/core.rs:981-985` retains bindings for `CommitUncertain`. Runtime routes still find the retained key and the fence rejects the terminal task, rather than falling back to the original call. The panic test explicitly panics before writes and checks refusal and restart retention (`provisional_prep_test.rs:2458-2503`). No issue found in this change.
- **Reclaim superseded per-task bindings deterministically without weakening fail-closed behavior:** selection is deterministic (minimum generation, then BTreeMap key) and tests exercise replay consistency (`provisional_commit.rs:780-831`). However, eviction silently unbinds an external request and fails closed behavior, as detailed above.
- **Default behavior:** Candidate changes do not edit the feature declaration; `aien-runtime/Cargo.toml:49` keeps `provisional-prep` outside `default`, and module/wiring remain feature-gated. No default-path change was identified in this diff.

## Scope and validation

Reviewed the complete five-file diff, changed tests/docs, relevant scheduler state transitions, fence implementation, and both daemon routing callers. `git diff --check` reported no whitespace errors. No cargo commands, builds, test suites, service calls, or writes to source were run. The worktree had pre-existing untracked `evidence/`; it was not touched.
