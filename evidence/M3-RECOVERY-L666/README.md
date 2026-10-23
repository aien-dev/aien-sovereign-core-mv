# Issue559 / PR609 recovered M3 source qualification

## Executed evidence

Final functional source: `834b55004acfd2e1c937e484e132635a83688ccc`, based on current main `754c07b118f41ad899607b60329bd851b1ccb13e`. Original implementation `1f4c41bcacc3f8253c6bd5015260d29d42b7798b` is preserved by author-attributed cherry-pick `fc62b5f2cf576a24ac7200ed1cb660aa9ee1e625`. Existing PR609 remains the sole integration target, last observed at `ad975aec00e15b6d7fca8e5dd382c81863b2824e`.

Final locked Linux commands, complete raw logs in `final/`:

| Command | Actual result |
| --- | --- |
| `cargo test --locked -p spark-harness` | 118 passed, 0 failed, 1 existing performance test ignored |
| `cargo test --locked -p aien-runtime --features provisional-prep` | 467 passed, 0 failed, 169 existing tests ignored |
| `cargo check --locked -p aien-runtime --no-default-features` | exit0 |
| `cargo clippy --locked -p spark-harness -p aien-runtime --features aien-runtime/provisional-prep --tests -- -D warnings` | exit0 |
| `cargo fmt --all -- --check` | exit0 |

Total executed passing tests: **585**, with **170 ignored**, not an all-tests or whole-workspace PASS. `counts.json` preserves parsed totals and runtime ignore reasons. The opt-in M3 integration binary itself executed all49 tests without ignores, including reclaimed-request routing before and after restart. Native compose/GPU archives were not linked in this host run; some existing tests explicitly assert stub refusal, and160 linked-compose tests are among the ignored tests. Real-checkpoint, model/GPU, A/B/performance and child-mode cases retain their original scope/ignore behavior. No native compute, production service, physical-device or release acceptance is claimed.

The final runner reused its private target directory named for the recovery checkpoint. Git HEAD during qualification was `fc62b5f`, with six explicit source-input SHA256s in `final/state.json`. Those bytes were verified identical to the subsequently committed functional source above. No executable Rust source changed after that run. Runner source is `run-final-linux.py`; reproduction uses a private 0700 TMPDIR parent and `quiet-build`, never a production store.

## Acceptance and review disposition

1. Terminal stale-generation deferred events are reported and removed from queue/sidecar; other refusals remain queued. Existing new original-branch regression and restart checks retained.
2. A panic before writes leaves an uncertain commit's requests bound and refused. It does not silently restore unfenced routing. Original panic/restart regression retained.
3. Superseded active bindings can be reclaimed deterministically without forgetting old request fences. The correction retains the old key/generation in a separate retired table used by normal runtime lookup and replay. Active and retired tables each have the existing per-task cap64. If retired history fills, a new key fails closed; explicit existing-key rebind can reuse a retired slot. A new generation alone is not permission to forget request history. Only a known committed outcome releases both tables; uncertain retains both. This is bounded reclamation, not infinite fresh-key capacity. Replaying an older unsafe history beyond the new retired budget may refuse open rather than silently unfence.

`REVIEW-L668.md` preserves the independent Important finding: the original reclaim removed the key and the real runtime route returned None, allowing the daemon's original-call fallthrough. Parent reproduced both an actual scheduler RED101 and actual Linux `route_approved` RED101. `review-fix/red.log` and `review-fix/route-red.log` preserve the failures. The old None assertions were strengthened to retained binding assertions. The active-only rebinding loop excludes the three known retired records; no safety assertion was removed. Added bounded-history/transactional-rebind/replay/isolation/commit-release coverage and a real routing regression that checks unchanged compose/workspace state before and after restart.

`REVIEW-L670.md` is the separate independent author-fix review: no Important/Critical defects found, all six frozen source hashes checked before and after, no builds/tests by reviewer. Its minor stale method comment was corrected after review. That final delta was mechanically checked to contain only Rust method-doc changes, followed by the final full run above. `reviewed-inputs.json` keeps the original reviewed hashes; `final/state.json` pins the later qualified bytes. No reviewer test replay is claimed.

## Preserved qualification failures

- Original recovered source: Mac full harness117/0/1, then runtime compilation exit101 because inference ABI callers reference Linux-only `external_default_buffer`. `host/` retains the failure; Linux alternative executed unchanged code. No platform guards or assertions were weakened.
- Original recovered Linux scoped feature/default/Clippy passed (`initial-linux/`), but those tests missed the fail-open defect. This is not acceptance of the original reclaim.
- The first corrected full runtime run found9 `kv_restart_test` AccessDenied failures (`full-suite-failure/`). The owned test TMPDIR parent was mode0775, uid1000, group-writable and nonsticky; the unchanged KV store correctly refused that parent. Only that agent-owned test parent was corrected to0700. No KV guard or test source was changed. `private-parent-green/` and then `final/` reran the complete commands successfully. Failures remain archived, not retried unchanged or relabeled PASS.

The base-pinned crumb tool `94fffbcf88492c5a1c54211b5126c8a6c370a98d` reported one stale root after source edits. The source was advanced to the committed functional revision, input hashes rechecked, and the exact pinned compiler refreshed that note. Verification passed16 notes (`pinned-notes.log`). Exact-head verification after adding the public evidence then correctly reported the root stale again (`published-note-red.log`): the first compile had not included the final evidence tree. The final metadata step refreshes the root only after all evidence/inventory changes, with exact published-head verification required before handoff. No executable source or gate was changed to clear either note failure. Local note checks are not signed Spark gates.

## Publication and remaining acceptance

Candidate and current main have identical workflow files. `workflow-inspection.txt` preserves all four: thin verification triggers PR/main, releases require a v-tag or manual dispatch, and release dry-run/publish are manual. No feature-push automerge/deployment path was found. No workflow, trust-surface, ABI, dependency pin or default feature was changed.

L667 alone advances **existing PR609**, with fresh expected-head/main checks, normal signed exact-head gates, hosted verification, diff/CLEAN checks and final integrated acceptance. No duplicate PR, resumed paused lane, merge, issue closure or release is authorized by this source/evidence packet. This remains host source qualification, not release acceptance.
