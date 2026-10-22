# L599 core historical qualification

Date: 2026-10-22. Host: Spark, CPU-only. Source: `a844edd6955a2958d1395aa79536c70b6b81f35b` (parent `a2a41b10251b3fe6e0385765827d676457202051`). Isolated detached worktree: `~/workspace/hermes-desk/core-qualification-L599`; private target: `~/workspace/targets/core-qualification-L599`; `CARGO_BUILD_JOBS=4`. Worktree remained clean. No runtime/model code modified.

## Results

- **Deployment fixture test: PASS, all 6 tests.** Ran the full `aien-allen` `deployment_test` target. `real_digests_of_two_checked_in_weight_files` passed; both expected safetensors files exist in the exact source tree. This disproves the old missing-fixture diagnosis on the historical source. Exact fixture sizes: `openwaldo-byte/model.safetensors` 36,632 bytes; `llama32_1b_oracle.safetensors` 50,834,760 bytes. SHA-256: `926db1cbbcea200d1acd12fa4b8a942a6b29f2339d2d35098b52231142dcec40`, `bf1d83edc1b802ad25cfbbce7a043658d3d4797bd67955447ce8630b51f515ed` respectively.
- **PREFILL-E2E-2: compile-only PASS, not behavior-qualified.** `cargo test --no-run` compiled the exact integration test successfully, exit 0, under the private target. I did not execute its ignored test: source at line 2117 calls `strict_load(&paths)` and loads the 2.2 GB TinyLlama checkpoint; the brief prohibits any model load. The source preserves the original assertions (KV parity, logits/tokens, old blocks, recomputation, branch isolation, COW, pool exhaustion, reclaim, fence). No assertions were changed or skipped.
- **Historical PREFILL failure is not classified as fixed or reproduced.** The supplied historical log reports the test binary under `/var/lib/aien-run/target/aien-sovereign-core/pr/release/deps/`, and contains post-decode KV parity failures. The parent correction identifies this old receipt as using shared target paths and says the installed helper now uses per-head targets. Current private-target execution of the behavior test is disallowed here; therefore runtime status under current isolation remains unqualified. No source defect claim or repair is made.
- No GPU, serving, judge, inference, benchmark/timing, cache mutation, or other owner's tree was used.

## Exact commands / exits

1. `cd /home/drakestapleton/workspace/hermes-desk/core-qualification-L599 && timeout 900 ~/.local/bin/quiet-build env CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=/home/drakestapleton/workspace/targets/core-qualification-L599 cargo test -p aien-allen --test deployment_test -- --nocapture` — exit **0**; 6 passed, 0 failed, 0 ignored.
2. `cd /home/drakestapleton/workspace/hermes-desk/core-qualification-L599 && timeout 900 ~/.local/bin/quiet-build env CARGO_BUILD_JOBS=4 CARGO_TARGET_DIR=/home/drakestapleton/workspace/targets/core-qualification-L599 cargo test -p aien-runtime --test prefill_e2e2 --no-run` — exit **0**; test executable built, not run.

## Evidence files on Spark

Directory: `~/handoffs/pr-failure-triage-20261022/`

- `L599-qualification.md` — this report
- `L599-qualification.json` — machine-readable outcome and command/exit summary
- `L599-deployment-test.raw.log`, `L599-deployment-test.exit`
- `L599-prefill-compile.raw.log`, `L599-prefill-compile.exit`
- Historical source log referenced: `aien-sovereign-core-a844edd6-PREFILL-E2E-2-5898b278511e4a54abddec562ce4d92119bbd0c356bfba4cb73f40d0499b2669.log`

No repair, PR, ledger closure, or runtime claim was made. Parent owns closure and any next decision.
