# ADR 0004 step 3 gate receipts (2026-10-15)

CALC-EXEMPT: every number is copied from a log or `run.json` in this folder or the step 0 baseline;
the floors are the step 0 README's, with their calc ids there.

Branch `adr0004-step3` (PR #557). The 30B adapter `impl ForwardPass for Qwen3MoeOmegaEngine` sits beside
`generate_inner`; no caller switched, so the chip run below runs the old loop (the running code until step 4).

| Gate | Receipt | Result |
|---|---|---|
| Host tests, both crates (`cargo test -p aien-inference-abi -p aien-serve-http --features aien-inference-abi/host-test-engine -- --test-threads=1`, stub build), at 1cde6150 less one test-only type alias added for clippy | `host-tests.txt` | 394 passed, 0 failed, 17 ignored (as step 2); the `aien-inference-abi` lib 231 passed (step 2: 224). Includes the dual run, the contract suite on the tiny MoE (default chunk and chunk 3), the moved tests on both adapters, the resync-after-restore case and the traces test |
| Red before green for each adapter fix | `red-run-naive-adapter.txt` | the naive adapter red on 4 tests (time counted twice, mirror unseen without the hook, dual-run decode launches, a head surviving a truncate); the review fix (cache lent through `kv()`) red without the flag |
| `clippy -D warnings` (`aien-inference-abi` with and without `host-test-engine`, `aien-serve-http`, all targets) | run locally, and CI | clean |
| G30-tokens, against `../adr0004-step0-baseline-20261015/check-full-3` | `check-full/run.json`, `check-full.raw` | `SUMMARY parity=PASS tokens=PASS`; `run.json` pass true; `first_diff` null on all 4 prompts; sha256 of `jq -c '[.prompts[] | .chip]'` begins `bb1a0401175ab69d` in both: IDENTICAL |
| G30-speed, decode tok/s against the step 0 floors | `check-full/run.json` `.prompts[].tok_s` | p1 38.74 (floor 37.927), p2 35.90 (35.017), p3 36.70 (36.375), p4 36.36 (35.502): all above |

## Chip run
- `step3-runs.sh` with `SKIP_DOOR=1` (step 3's gate names G30-tokens and G30-speed only; the door does not
  change until step 4). One `quietlock hold --owner STEP3 --minutes 20`, 02:06:53Z to 02:08:33Z. The exact
  command line is in `check-full.start`; it is step 0's check-full with the same arguments and `--force-memory`
  (the idle 30B server on :18091, about 36 GB, was left running).
- sc `f2b9e017` (this branch merged with main), no tracked change outside this folder; omega `4799f9eb`,
  `libomega_gpu.a` sha256 `0629ad9c...`, the same locked archive as step 0. The binary was built at `1cde6150`;
  the two later merges of main changed only crumbs, docs, scripts and benchmark data, none of it in the binary.
- 1-minute load average: 1.72 at start, 3.47 at end (`check-full.start`, `check-full.end`). The gate waited
  for load at or under 4 and for the ternary T0 and trainer lanes to be idle for 10 minutes.

`SHA256SUMS` lists the receipts.
