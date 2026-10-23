# WHOLE-SYSTEM-E2E, harness v1.3: rows for RUN-2-GB10 (milestone R6, GB10 native Omega backend)

Tracking: aien-dev/aien-sovereign-core#402 (owning issue), aien-dev/aien-architecture#190 milestone R6. Contract:
`ACCEPTANCE_E2E.md` FROZEN at aien-architecture b2400fe, sha256
`9e3d5f4e23d65ac17be8f2f4ca48355f0aaa486f67ff5a8a8044dcdd646d34e8` (written into `VERDICT_ROWS.txt`; the verifier takes
it on the command line). This file is committed before the harness change that implements it. `ACCEPTANCE-v1.md`,
`-v1.1.md` and `-v1.2.md` stay as written; RUN-1 and its write-ups stay as recorded.

## What changes, and what does not

Every row, objective text, identity (`objective_id`, `run_id`), control, threshold, budget and PASS rule of
`ACCEPTANCE-v1.2.md` (and through it v1 and v1.1) applies unchanged. The same fixture, the same CAND-4 model (Llama 3.2
1B Instruct, internal-test only; weights and tokenizer digests compared with `release/candidate.toml`), the same
`E4_MAX_TOKENS` (400), the same ALLEN fixture subject, the same desk, the same kill points. The default harness mode
(CPU) is unchanged. Only the backend rule changes, in a new mode `WSE2E_BACKEND=gb10`:

| rule | v1.2 (CPU, RUN-1) | v1.3 gb10 mode (RUN-2-GB10) |
|---|---|---|
| G1 build | release build with the prebuilt compose archive, stub GPU engine | `scripts/pins.sh --gpu` materializes omega, physics and aienos at `omega.lock` and omega's `physics.lock` and `aienos.lock`; its `env.sh` is sourced (it sets `AIEN_OMEGA_DIR` and `AIEN_REQUIRE_NATIVE_LIBS=1`, so a stub engine is a build error); `cargo build --release -p aien-cli`, the same commands as the sc#94 release golden path (sc#92, sc#375). A build log naming the stub is refused. `scripts/zero-cuda-gate.sh` runs on the built `aien-cli`; its output and exit code are kept; a non-zero exit refuses the run. The ALLEN fixture subject is built with the same environment |
| G2 backend line | the daemon's `  Backend:` line must contain `CPU-reference`, else the start fails | every daemon start (CTRL-E3a's no-key start included) carries `AIEN_GPU_BACKEND=omega` and `AIEN_REQUIRE_BLACKWELL=1`, so a missing engine is fatal at start and never a CPU fallback; the `  Backend:` line must contain `OmegaGb10Backend (native Omega engine, no CUDA`, else the start fails exactly as a wrong line fails in CPU mode; the daemon's `STRICT` start line must read `strict=true dev_fallback_build=false`; `AIEN_DEV_FALLBACK` is unset |
| G3 counters | none | every receipt carries `gpu_executions`, `gpu_executions_process_total`, `fallback_count`, `fallback_detected` (sources below) |
| G4 fallback | none | strict: a fallback anywhere marks that row FAIL (whatever its other facts; a row that measured nothing stays NOT_RUN), and `fallback_detected` becomes true for the rest of the run; the run summary `GB10_SUMMARY.json` (inside the manifest) carries the final value |
| G5 model calls reach the chip | none | a row whose propose report holds model-call evidence (`report.proposal_attempts[].ops`) FAILs unless that evidence names the Omega GB10 backend and `gpu_executions` grew during the row (at least 1); a row with no model call carries `gpu_executions: null` and is judged on its v1.2 rule plus G4 only |
| G6 identity of the stack | commit, binary, model digests | as v1.2, plus in every receipt: `pins` (`omega`, `physics`, `aienos` HEAD commits of the pinned checkouts, each checked equal to its lock file), `nvidia_driver` (first line of `/proc/driver/nvidia/version`), `zero_cuda_gate` (exit code), `backend_mode` `gb10`, and `host_load` (`/proc/loadavg`, `MemFree`, `MemAvailable` at the receipt) because RUN-1 showed CPU contention can move a budgeted row and this machine is shared |

### Counter sources (G3), all from what the daemon already emits after sc#405

- `gpu_executions_process_total`: the largest `ops.gpu_executions` among the row's `report.proposal_attempts[]` in
  `artifacts/<row>-propose.json` (sc#405: the Omega engine's `chip_calls`, a process total since the daemon built the
  backend). Null when the row made no model call.
- `gpu_executions`: that total minus the last total seen from the same daemon process (the daemon number `DN`); the
  first observation after a start counts from the start (warm-up included), recorded as
  `gpu_executions_since: "daemon start"`. Null when the row made no model call.
- `fallback_count`: the largest `ops.native_fallbacks` among the row's attempts (0 when none) plus the number of
  `STRICT_REAL_MODEL_VIOLATION` and `OMEGA_BACKEND chip error` lines in every daemon log of the run so far (the
  production build panics on the first claimed-native fallback, `crates/aien-inference-abi/src/strict.rs`; every chip
  error is a counted fallback, `omega_backend.rs` `fail`). Counted over all logs so far, so a fallback during any row
  taints every later receipt of the run: strict by design.
- `fallback_detected`: true once any receipt of the run had `fallback_count > 0`, or any daemon log of the run (CTRL-E3a's
  included) holds one of those lines at the end of the run.

The independent verifier (`tools/aien-verify`, VERIFIER-v1.md) is not changed by this file and does not read the G3
fields. It re-derives rows from the v1.2 facts. A harness FAIL caused only by G4 or G5 would therefore appear as a
`status_disagreement` on that row in `VERIFIER_VERDICT.txt`; that is expected, is recorded in the result note, and is
not a weakening: the harness status and the G3 fields stay in the chain. (In a production build a fallback panics the
call, so the v1.2 facts of that row fail as well and no disagreement is expected.)

## Predictions (written before the run; PASS or FAIL as measured)

| row | prediction | why |
|---|---|---|
| PRE | PASS | sc#94 (fefce80) showed the release build on this machine reaching the GB10 engine with 0 fallbacks. Risk: omega#327, GPU system-memory allocations refused (`NV_ERR_NO_MEMORY`) when `MemFree` is low although reclaimable cache is large; the machine is shared tonight. A refusal at start is a PRE FAIL, recorded, never a CPU fallback |
| CTRL-E3a | PASS | the desk check does not depend on the backend |
| E1 | NOT_RUN | as v1.2 (RUN-3, R8) |
| E2, CTRL-E2 | PASS | no model call; backend independent |
| E3 | PASS or FAIL as measured | UNVERIFIED: sc#94 used the micro model `aien-micro-gb10-v1`, not CAND-4 through `compose propose`; the GB10 arithmetic may change the greedy text, so the 200-word and file-name checks are measured, not assumed |
| E3m | PASS or FAIL as measured | as E3 |
| E4 | PASS or FAIL as measured | as E3; memory recall itself is backend independent |
| E6, CTRL-E3b, CTRL-E6, CTRL-E4 | PASS | effect and refusal paths are backend independent; each makes at most one short model call |
| NET | PASS | as v1.2 |
| CTRL-E4b | RECORDED | not judged (v1.2); on the chip the propose should finish inside its budget |
| E5, CTRL-E1, CTRL-E5 | NOT_RUN | as v1.2 (R7 second machine, R8 release artifact), unless a merged harness change on main supplies one |
| G4, G5 over the run | `fallback_detected: false`, every model-calling row `gpu_executions >= 1` | sc#94: `gpu_executions` 156, `fallback_count` 0 |

## Rules of this run

One declared attempt (contract rule 7), from the merged main commit, run folder `runs/RUN-2-GB10-<utc>/`, started
detached, never killed. If it aborts for a harness defect, the folder stays and the next attempt is a new folder with a
note. Nothing under `runs/` is edited after the run. `quietlock check` before the start. No cache drops, no other GPU
lease touched. `git status --ignored` on the run folder and `sha256sum -c chain/SHA256SUMS` from the committed tree are
checked before the evidence PR. A PASS of any row here says nothing about an installed artifact, a public model,
native AIENOS, a second machine or a hostile operator.
