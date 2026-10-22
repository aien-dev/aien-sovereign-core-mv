# L609 independent source review — frozen 3aae7e9ce51a2e8a30720c9b482374145a2e0e19

Base: `132e709b535a821079e62566c4cd5b3a0fa9059a`; repo: `~/workspace/hermes-desk/renda-pr-fastpath-L593`.

## Verdict

**BLOCK installation pending correction.** The 92-test pass does not cover a cache-integrity bypass: a cache file with exact source chunks and identity but fabricated `decision: true` values is treated as a complete approval with zero model calls. This violates the explicit requirement that untrusted cache content never confer approval. No source was modified, installed, or reloaded; no model, GPU, network PR, or judge was invoked.

## Finding

**P1 — `review_chunks.py:_valid_cache` trusts unauthenticated cached decisions.** It verifies identity, diff digest, every chunk's exact text/offset/digest, and that decisions are booleans, but the decision values are not authenticated or bound to a trusted model response. Cache identity is a public SHA-256 over public inputs; changing `null` to `true` (or writing a cache from scratch) needs no secret or model call. The public `review_diff(..., budget=0)` then reports `complete=true, decision=true`. This is a direct approval bypass, including for large-diff `delivery.assessment`, which revalidates the same cache through `review_diff`.

### Executable proof

On Spark, ran a synthetic probe importing the frozen `review_chunks.py`; made a 25,014-character diff (3 chunks), generated the expected public identity/chunk rows, set all decisions to `true` and notes to `forged`, and called `review_diff` with `budget=0` and an `ask` callback that raises if invoked. Actual result:

```json
{"calls":0,"chunks":3,"decision":true,"complete":true,"notes":["forged","forged","forged"]}
```

### Needed before approval

Do not let cache-sourced booleans satisfy approval without an integrity mechanism rooted outside the writable cache (or re-review all cached decisions). Add a regression that mutates only cached decisions while preserving the otherwise valid identity/text and proves the result stays incomplete/non-approving. Then rerun integration and delivery tests.

## Verified items / coverage

- Read `REVIEW-INTEGRATION-L603.md`, exact base-to-head diff, and scoped queue/chunker/delivery/policy/install/startup code.
- Existing test executions on Spark passed: `test_review_chunks.py` 8, `test_review_integration.py` 2, `test_delivery.py` 24, and `test_local_queue.py` 26.
- Source semantics inspected: chunk ranges span the whole diff; source/repository/base/head/model/policy and text are included in the chunk identity; stale/malformed caches are replaced; explicit false is terminal; queued chunk callbacks have a per-pass remaining budget; partial host plans are reused by exact plan identity; publish still demands eligible receipt, exact fresh remote refs, ownership/pause checks, and PR readback.
- Those controls do not mitigate the forged-decision cache issue above. No claim made about live publication or model correctness.

Files changed: only this review report (`~/handoffs/pr-failure-triage-20261022/L609-review.md`). No source changes.
