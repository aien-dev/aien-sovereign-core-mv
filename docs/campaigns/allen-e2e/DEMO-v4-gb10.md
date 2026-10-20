# ALLEN end-to-end demo v4-gb10 (predeclared)

Status: PREDECLARED. To be committed and pushed before any v4-gb10 run. v1 to v3 and their results stay as recorded.
Refs aien-architecture#159, #162.

Question: does the v3 ALLEN chain (identity, memory, inference, capability, AEGIS, approved effect, World commit, Cortex,
evidence) hold when inference runs on the qualified GB10 candidate instead of the CPU reference?

## Candidate
sovereign-core main 6bbe2ec269768c7c9b94b9484c757ca45f55f564 (the sc#277 attempt-3 build: #311 + #313), omega 6c6180cf,
PHYSICS 6d7cf0d4, the same environment as ~/workspace/hive/Q277/DECLARATION.md. The GB10 daemon is the SEALED binary
`~/workspace/hive/Q277/bin/aien-cli-main`, sha256 `152c0aecce662f618bf683c8854d6de56a7075e0461c2433570f4c15b68571a5`
(Q277 `SHA256SUMS`, `IDENTITY.txt`). The driver checks the file against that sha256 (written in the script and in `SHA256SUMS`)
and refuses on any mismatch, and refuses unless the tree is 6bbe2ec2 plus demo-only files. All client commands and the M-B
daemon use the v3-style CPU build of the same tree. Environment label: Linux-hosted GB10. Not native Aienux.

GB10 daemon environment: `AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1 AIEN_REQUIRE_BLACKWELL=1 AIEN_REQUIRE_CHECKPOINT=1
AIEN_KV_CONTEXT_TOKENS=4096`, `AIEN_MODEL_PATH=<model>/model.safetensors.index.json`; `AIEN_OMEGA_*`, `AIEN_FORCE_CPU_STUB`,
`AIEN_OMEGA_SPIN_US`, `AIEN_OMEGA_CTA_BUDGET`, `AIEN_COMPOSE_BUDGET_MS` unset (the driver refuses if the last three are set).

## Steps
DEMO-v2.md unchanged (S0-S8, S3-red, checker C1-C5, request text with the stated format rules), with these substitutions:
- M-A = Qwen3-4B-Instruct-2507 (`~/models/qwen3-4b-instruct-2507-cdbee75`) on the GB10. M-B = SmolLM2-1.7B on the CPU reference.
- The driver is run as `DEMO_VERSION=v4-gb10 scripts/allen_e2e_demo.sh run` INSIDE a quietlock hold given by the coordinator
  (the driver refuses to start when the quiet flag is not held, and never takes or clears it).
- GB10 daemon lives: S0 start, S1 restart with the identity subject, S5 restart: three lives. (The draft said two; v3's flow
  starts the daemon at S0 and again at S1 once the identity fixture exists, so the driver's S0 and S1 are two lives.)
  Each life is one daemon on the chip. Estimate 5-8 min each, about 15-25 min in total.

## Chip safety (hard rule)
No signal of any kind is sent to a process that holds a GB10 channel (gpu-chip-test-no-kill; omega 03d2820: destroying a
channel under a spinning seat makes the driver's stop request time out). The driver stops a GB10 daemon only through
`aien-cli compose shutdown` (the control Shutdown op, `crates/aien-cli/src/compose.rs` "shutdown", answered
`ShutdownAck`, `crates/aien-runtime/src/spine.rs` Shutdown), then waits with NO timeout for the daemon to exit. This applies at
S1, S5, the S7 switch-over, to every error path, and to the exit trap. `aien-cli stop` is not used (it also runs `pkill`).
If a GB10 daemon never takes the request or never exits, the driver waits and a human decides; it does not kill.
A stop is "clean" only if: a shutdown request was sent at least once, the daemon exits 0, its socket file is removed, and the
daemon log has no `seat kill` line. After an unclean stop the driver starts no further GB10 daemon.
No "session closed" line exists: the daemon prints none (checked in the Q277 logs and the sources). The four clean-stop signs used are (1) ShutdownAck received, (2) exit code 0, (3) the socket file removed, (4) no `seat kill` line in the log; the
kernel-log NVRM delta below is recorded as well, and a delta above 0 blocks any further GB10 life (Q277 judged a close by exit 0 and no new NVRM lines).

## Evidence recorded per GB10 life (receipt `GB10-lives`)
The `Backend:` line, the `GB10_SERVING_RESERVATION reserved bytes=` line, count of lines matching `fallback`, count of NVRM
lines in the daemon log, count of allocation failures (`failures=` or `failures_since_warmup=` non-zero, or `refused`),
count of RM failures (`NV_ERR_NO_MEMORY`, `status 0x51`, `RM_ALLOC ... fail`), kernel-log NVRM line count before and after the
life, the daemon exit code, the stop result.

## PASS
All 10 steps PASS as v3 did, and for every GB10 life: Backend line is the GB10 one (`OmegaGb10Backend`), reservation line
present, 0 allocation and RM failures, 0 fallback lines, 0 NVRM lines (daemon log and kernel-log delta), clean graceful stop
as defined above. S7 must show both the `Model:` line and the `Backend:` line changed (GB10 to CPU reference) and the model
digest changed. If the kernel log cannot be read, the `GB10-lives` receipt is UNKNOWN, not PASS. If M-A does not show the GB10
Backend line and the reservation line, the run stops there (after a graceful stop). Any FAIL is kept, no rerun.

## Labels and outputs
S0-S6 real-GB10 (Linux-hosted). S7-S8 real-CPU. Stack label as v3 computes it. native-AIENOS: NOT_RUN.
Outputs: `receipts-v4-gb10.jsonl`, `artifacts-v4-gb10/`, `RESULT-v4-gb10.md`. RESULT-v4-gb10.md must state:
"hard-crash recovery on GB10: NOT TESTED (chip safety rule)"; the S5 receipt reads "graceful stop (exit 0, session closed)
then restart on M-A (GB10)". Kill -9 recovery evidence comes from the CPU legs (v3 S5/S7, recovery matrix sc#307,
concurrency test sc#325).

## Limits
One request per GB10 life; no endurance; no cache fill, no drop_caches, no setting change. v3's SmolLM2/Qwen3 order is swapped,
so S3 and S8 texts are not comparable to v3. Whether the CLI client built without the GPU library can drive the GB10 daemon
over the socket is expected (same commit, same wire protocol) but untested here. The daemon runs with the driver's isolated
HOME, which the Q277 run did not use: UNKNOWN whether the GB10 path reads HOME.

## Signals to the driver
The GB10 daemon is started in its own session (`setsid`, with HUP and INT ignored, `$!` still its pid; the driver checks that it
is its own session leader). A Ctrl-C, a terminal hangup or a signal to the driver's process group therefore cannot reach it.
The driver traps INT, TERM, HUP, QUIT, USR1, USR2, PIPE and ALRM: it stops a live GB10 daemon gracefully (unbounded wait, a repeated signal only
queues the exit) and only then exits (130); it never exits while a GB10 daemon is alive. `selftest-gb` proves this with a
fake daemon, including a control showing an ordinary child in the same group does receive the signal.
A SIGKILL of the driver itself, or a power loss, cannot be handled and is outside this rule.
Signals that arrive while the daemon is being spawned are deferred until its pid is recorded, then handled by the graceful
stop. The pid of a live GB10 daemon is held in `GB_LIVE_PID` until the stop's wait returns; the only function that sends a
kill (`safe_kill9`, CPU daemons only) refuses that pid. The one unprotected instant is the few microseconds between the
fork and `setsid`, when the child is still a plain shell that has not opened the chip.

### What the driver cannot survive
SIGKILL of the driver, or a tree-kill by a harness (kill of the whole process tree or container), bypasses every handler. The
GB10 daemon then survives orphaned in its own session and no Shutdown has been sent. A human must stop it gracefully,
without any signal. At every GB10 spawn the driver writes the exact command, with absolute paths, to
`$CLAUDE_JOB_DIR/tmp/demo/run/logs/STOP-BY-HAND.txt` (a human shell may not have `CLAUDE_JOB_DIR`; for this campaign's job
that is `/home/drakestapleton/.claude/jobs/294ea9b6/tmp/demo/run/logs/STOP-BY-HAND.txt`). Its form is:

    AIEN_RUNTIME_SOCK=<job>/tmp/demo/run/s <job>/tmp/demo/target/release/aien-cli compose shutdown

(expect `{"step": "S7", "shutdown": true}`), then wait until `pgrep -f 'aien-cli-main daemon'` shows nothing and check the
kernel log for new NVRM lines. Never `kill` it.
