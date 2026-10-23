# Read-only review: adaptive weight-flow candidate

Reviewed Spark worktree `/home/drakestapleton/workspace/hermes-desk/sc-weight-flow-L689` against base `836a53446d3674e86cbfc435d814775a0b8503a8`. No source edits, tests, builds, or launches were performed.

## Finding

- **`native/weight-flow/policy.om:76` — zero compute duration can divide by zero.** `validate` (lines 39–58) permits `compute` (field 5) to be zero, and the host bridge's `NativePolicy::decide` (weight_policy.rs:47–51) passes arbitrary `u64` fields through without checking this. A valid request such as fields `[1, 2, 4, 2, 1, 0, 0, 0, 0, 0]` reaches `gt wait div compute 8` with `wait=1, compute=0`. Depending on Omega's integer-division semantics this can trap or produce a nonsensical decision; at minimum the policy accepts an input for which its comparison is undefined rather than rejecting/handling it. Runtime behavior was not launched/verified due to the read-only constraint. (The adaptive caller normally measures nonzero compute time, but the bridge is public and no invariant enforces it.)

## Areas checked

- **Deadlocks / joins:** flow's scoped worker queue is bounded by the active depth/capacity; error exits join remaining workers, and scoped threads also join on unwind. I found no concrete deadlock path in the inspected flow loop.
- **Pin lifetimes:** each successful prepare returns a `Lease` whose `Drop` releases the pin; this covers normal compute completion, returned errors, and unwinding. Failed outstanding prepares are joined and successful leases dropped.
- **Failed-stage output transactionality:** flow computes into temporary `gu`/`act`/`down` vectors, obtains the trace lock before committing, and outer `ffn` validates caller buffer lengths first. I found no partial caller-output write on ordinary stage/policy errors in this path.
- **Bounds:** flow validates settings `1..=32`, requested experts `<=32`, capacity, unique IDs, and manifest membership. Bridge bounds returned depth to `1..=cap` and calm to `<=3`. The zero-divisor case above remains the concrete exception in policy input validation.
- **Policy bridge:** framing loops handle partial reads/writes, use nonblocking pipes plus deadline, latch errors, and kill/reap the child. No other concrete bridge failure found by source inspection.
- **Benchmark equality/metrics:** sweep's `FLOW_EQUAL` checks exact generated-token vectors (`chip`) against legacy, and requires non-legacy flow trace attachment; it does not claim logits equality. Flow trace represents the last recorded flow call, not a per-prompt aggregate, but it is named `last_layer_flow` in output. No concrete false equality claim found. Timing comparisons may still be affected by shared pager residency across modes; interpret as sequential same-process measurements, not independent cold-start runs.

## Limits

This was static source review only. I did not run tests/builds or execute the native Omega policy, so could not confirm actual divide-by-zero behavior or end-to-end inference/benchmark results. No GPU or production/live-model claims assessed.

## Reviewed source SHA-256

```
b27d608273751ba9ef21705b3a8fd12e721c018783521d653630ca644cb351b8  crates/aien-heartbeat/src/weight_flow.rs
808593000f2b2351d9263f60dbdcc4da409f77a9622d730ab9fbd72ee94c6539  crates/aien-heartbeat/src/weight_policy.rs
7ab29323127ba5cec5d2d80a4c13ec0a0406ed6caa95cce372a06a998155113f  crates/aien-heartbeat/src/lib.rs
ca55cd6d5f274c31744d470d227d795ae34af605c929630fb995c92082bb98ed  crates/aien-heartbeat/tests/ribbon_store.rs
814e218ecfa9d322817628bb2dac7d0389ebfb13f6959b395405f32a71390115  crates/aien-heartbeat/tests/weight_policy.rs
94133c9c9b525a0cf4a22652d6a02a2ce43229fde90836e6cdb511325fcd0b8a  crates/aien-inference-abi/src/qwen3_moe_omega/ribbon.rs
2a005d90670713f53ef84882b7e0ef7b2721717423e249d5f8e004502fe2be7a  crates/aien-inference-abi/src/bin/own30b_check.rs
2904c0e869b7ad801aab319c5d16455191160a83383eebc8d229fb538400e2aa  native/weight-flow/policy.om
68385ab2ece5e9d376c5543e9a2efdcebfa210d2aaa63f9bfc6b947fbd392975  native/weight-flow/build.py
9715cb7b7ddd7a6a2043c03e143635c16bc637c4b69e52d3f6c888ce9df37843  native/weight-flow/mutants.py
947398cd4f720d1fe462da904fc1909f8951983f4d5fb95639e8c6ad169ac6dc  native/weight-flow/test_policy.py
0c18bd4b75270f2b1c5f5c627e4ab2d5ee3fd2b4e1c9540717641a10ed5b2b07  docs/superpowers/plans/2026-10-23-weight-flow.md
```
