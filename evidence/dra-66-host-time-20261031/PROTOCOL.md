# DRA-66 host time per token: protocol

Refs aien-dev/aien-sovereign-core-mv#857 (Linear DRA-66). Written before any chip run; the
window has not run. Every chip line below is NOT_RUN until Main runs `window.sh` inside a
spark-window.

## Binary

- `/home/drakestapleton/work/dra-66/bin-head/aien-serve-http`, sha256
  `3bd287beb5c74fc5c241307765f3a858c1a7e33e0d1d5ae2d766f2dae13e9c69`.
- Built from sovereign-core `84a28c0923b75dde5450321307bfda8c525549dc` (the binary reports
  `sovereign-core commit: 84a28c09… (dirty 0)`) with omega `a647b6a9` (`omega.lock`), physics
  `048e5f5e`, aienos `b84c0a67`:
  `scripts/pins.sh --gpu ~/work/dra-66/pins && . ~/work/dra-66/pins/env.sh &&
  nice -n 19 taskset -c 0-9 cargo build --release -p aien-serve-http --bin aien-serve-http
  --target-dir ~/work/dra-66/target-gpu`. Native `libomega_gpu.a` linked (no CPU stub).
- The commits after 84a28c09 on the branch touch only evidence files and the PR text.

## Server

The live 18091 flags on loopback port 18190: `--engine omega --model-dir
/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0 --port 18190 --bind 127.0.0.1
--max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8`, env
`AIEN_Q36_STROKE=1`, `AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-18190.lock`.

## Run

Main stops 18091 and Flint (18300/18310) first, then:

```
spark-window open dra66 --gb10 none --quiet exclusive --lease 20 --wait
spark-window run bash evidence/dra-66-host-time-20261031/window.sh
spark-window close
```

Why `--gb10 none`: with `AIEN_GB10_LOCK_HELD=1` the server accepts an inherited chip
reservation only when the flock holder of `/tmp/aien-gb10.lock` is one of its ancestors
(`crates/aien-serve-http/src/chip_reservation.rs`, `detect`; refusal message from `main.rs`
"none an ancestor of this server", exit 1). spark-window's holder (`setsid spark-window _hold
ID`) is never an ancestor of the `run` command, so in a window that holds the chip the server
cannot start, and without the marker it would wait for the lock until the window closes.
spark-window's own rule for a job that takes the lock itself is a window with `--gb10 none`
(approved by Main, 2026-10-31; no new trust path). `window.sh` then takes the lock through
`flock -w 90 /tmp/aien-gb10.lock`, which is the ancestor of both servers, and starts them with
`AIEN_GB10_LOCK_HELD=1` after verifying that holder in `/proc/locks`. The window's quiet lock
(`--quiet exclusive`) is what keeps the timings quiet.

`window.sh` prints `NOT_RUN: …` and exits 3 before any chip contact unless: `SPARK_WINDOW` and
`SPARK_WINDOW_DIR` are set and name the open, held window (`spark-window id`); the window is
`gb10=none` and `quiet=exclusive`; `AIEN_GB10_LOCK_HELD` is not already set; the chip lock
comes free within 90 s and its exclusive holder (the `/proc/locks` FLOCK WRITE line whose
device and inode both match the lock file) is an ancestor of the script; nothing answers
on 18091 and nothing listens on 18091/18300/18310; no compute process is on the chip; the
binary has the pinned sha256 and a native omega gpu engine. Host check: run outside a window,
it printed `NOT_RUN: not inside a spark-window (…)`, exit 3.

Receipts land in `/home/drakestapleton/handoffs/dra-66-host-time/window-<UTC stamp>/`. Each
gate writes `gate-<name>.json` (schema `aien.dra66-window-gate/1`): `window_id`, `window_dir`,
`bin`, `bin_sha256`, `sc_commit` (from `--version`), `gb10_lock_holder_pid`, `start`, `end`,
`elapsed_s`, `lease_left_s`, `status` (PASS / FAIL / NOT_RUN), `detail`, and the sha256 of every
file the gate produced. At exit `receipt.json` (`aien.dra66-window/1`) collects all gates with
the window id, binary sha256, total elapsed time and the overall `status`; the board gets
GPU-START / GPU-END lines.

## Gates, in order, and the 20-minute lease

A gate starts only if the lease (`held` mtime + `lease` minutes of the window record) has at
least the gate's cap left; otherwise that gate is written NOT_RUN and the script stops (exit 3).
Each cap is enforced inside the gate: server load ≤ 150 s to health (time-bounded loop),
requests by `curl -m`, server stop ≤ 30 s then KILL. Server loads measured 39 s, 57 s and 78 s
in the GoldenP04 window (`~/handoffs/GoldenP04/evidence/exec-20261027T110114Z/window.log`,
2026-10-27, same model and flags). Decode estimates use the pinned DRA-64 model
(t(1) = 23.494 ms, t(8) = 83.7676 ms per beat).

| Gate | Content | Expected | Estimate | Cap | Receipt |
|------|---------|----------|----------|-----|---------|
| (outer) | window checks, `flock -w 90` | lock free at once | ~0 s | 120 s | `preconditions.txt`, `gate-preconditions.json` |
| `mutant4-red` | mutant server (`AIEN_Q36_TEST_MUTANT=4`), the gate2 request (24 greedy tokens of "Write a C function that returns the larger of two ints."), stop | `DIFFER` from `evidence/q36-host-20261020/gen-chat-max-24.json` `generated_ids`; otherwise FAIL and the window stops (exit 4) | load ≤ 78 s + 0.56 s + stop ~2 s | 240 s | `golden-mutant4.json`, `server-mutant4.log` |
| `golden-identity` | honest server, per-token receipt on (`AIEN_Q36_DECODE_TIMELINE_KIND=token`), the gate2 request, 2 s idle; records so far = A | `IDENTICAL` | load ≤ 78 s + 0.56 s + 2 s | 210 s | `golden-honest.json` |
| `receipt-1conv` | 3 sequential greedy replies of 256 tokens, 2 s idle; records so far = B (B > A + 1) | all 3 answer | 18.04 s + 2 s | 92 s | `solo-*.json`, `tokens.jsonl` |
| `receipt-8conv` | 8 concurrent greedy replies of 256 tokens | all 8 answer | 21.44 s + 2 s | 62 s | `eight-*.json`, `tokens.jsonl` |
| `seed-t07` | seed 1, seed 2, seed 1 again at T=0.7 through the HTTP door, 64 tokens | seed 1 repeats, seed 2 differs | 4.51 s | 60 s | `seed-*.json` |
| `stamps-1conv` | stop the token server; honest server with the stamped launch timeline (`AIEN_Q36_DECODE_TIMELINE=stamps.jsonl`, default kind); 1 greedy reply of 128 tokens, 2 s idle; stop | the reply answers, ≥ 1 record | load ≤ 78 s + ≥ 3.01 s (stamps slow the chip) + 2 s + stops | 242 s | `stamped.json`, `stamps.jsonl`, `server-stamps.log` |
| `checker` | both servers stopped. Token receipt: check (exit 0), `--slice A:B` (exit 0), `--drop 0:record:0` and `--drop n/2:record:0` (exit 1), `--bend A+1:chip` (exit 1), `--bend host:600000 --slice A:B` (exit 0, the synthetic red arm). Stamped timeline: check (exit 0), `--drop 0:record:0` (exit 1). Writes `gate.wl` = `summary-1conv.wl` + `summary-stamps.wl` and `gate-red.wl` = `summary-red-host.wl` + `summary-stamps.wl` | exits `0 0 1 1 1 0 0 1` | ~5 s | 45 s | `gate.wl`, `gate-red.wl`, `summary-*.wl`, `check*.txt`, `red-*.txt`, `spells.txt` |
| (Main) | host time ≤ 0.5 ms per token at one conversation | see "The 0.5 ms gate" below: `gate-red.wl` must evaluate to `False` first, then `gate.wl` to `True` | — | — | both files + the two Wolfram replies quoted in the report |

`receipt.json` carries an overall `status`: FAIL if any gate failed, else NOT_RUN if any gate did
not run, else PASS; the script then exits 4 on FAIL. That status covers the receipts' mechanics
only; the 0.5 ms verdict is Main's Wolfram evaluation.

Wolfram (MCP evaluator):
- `t1 = 23.494; t8 = 83.7676; {24 t1/1000, 3*256 t1/1000, 256 t8/1000, 3*64 t1/1000, 128 t1/1000}`
  → `{0.563856, 18.043392, 21.4445056, 4.510848, 3.007232}` s.
- `caps = {120, 150 + 60 + 30, 150 + 60, 3*30 + 2, 60 + 2, 3*20, 30 + 150 + 30 + 2 + 30, 45};
  {caps, Total[caps], 20*60 - Total[caps]}` → `{{120, 240, 210, 92, 62, 60, 242, 45}, 1071, 129}`
  s of the 1200 s lease. The outer row counts 120 s: 90 s flock plus 30 s of checks.
- Expected run: `t1 = 23.494; t8 = 83.7676; N[3*78 + 2*24 t1/1000 + 3*256 t1/1000 + 256 t8/1000 +
  3*64 t1/1000 + 128 t1/1000 + 4*2 + 3*2 + 5]` → `301.1336896` s (5.02 min) with the slowest
  measured load three times; the stamped reply counts at the unstamped t(1), its cap is 30 s.
  The run fits in one window, so the script is not split.

## The 0.5 ms gate

`chip_ns` is the graph's own host-clock span from the first queue submit to the last
completion marker read back (omega `cmdlist.c` a647b6a9, `elapsed` from L390 to L399, summed
per chunk). It therefore holds host work besides the chip's: the submit, the marker poll (the
served 2000 µs spin, then 50 µs sleeps) and the marker read back. So the per-token receipt's
`host_overhead_ns = total_ns − chip_ns` is a LOWER bound of the host time per token, and is
labelled `hostLB`; it never carries a verdict alone.

The stamped launch timeline measures the missing share. At one conversation a lone row takes
its own one-row graph (`rows.rs` `step_rows`, the lone-row path), the same graph as the token
arm, so per record `submit_poll = chip_ns − dev_span`, with `dev_span` the chip PTIMER span from
the first launch stamp to the last. The stamps' chip cost sits in both `chip_ns` and `dev_span`,
so it cancels to first order; the stamp buffer's own syncs fall outside `chip_ns` (cmdlist L372,
L404). Only `submits == 1` records count: with more chunks the host's emit between them lands in
`dev_span` but not in `chip_ns`.

The gate is a sum of medians over two arms of the same window, `GATE_WL` in
`decode_timeline.rs`:
`{N[hostLB/10^6], N[submitPoll/10^6], N[(hostLB + submitPoll)/10^6], hostLB + submitPoll <= 500000}`
with `hostLB = Median[host_overhead_ns]` over the clean (rows 1, no prompt joining) beats of
records A..B and `submitPoll = Median[submit_poll]` over the single-submit stamped records.
`summary-stamps.wl` ends with it, so `gate.wl` evaluates to the verdict as its last result.

Main's procedure, in one Wolfram session per file:
1. Evaluate `gate-red.wl`: the same slice with every beat 0.6 ms more host time
   (`--bend host:600000`, consistent t0/t1/total/host share, shape green). The last result must
   be `False`. If it is `True`, the gate is broken and `gate.wl` does not count.
2. Evaluate `gate.wl`. The last result's verdict is the gate: `True` PASS, `False` FAIL.

Accounting notes (reviewer MINOR 7):
- The chip lock is taken once per spell before the first beat; its wait sits in TTFT and never
  in a token record.
- The recorder writes and flushes one line per record at the next beat's start; that write
  lands in the next beat's `total_ns`, so it counts as host time (conservative).

The timing number is a quiet-window number only because the window holds the quiet lock
exclusively with 18091 and Flint down; `preconditions.txt` records the load average. The receipt's
`total_ns` is beat close to beat close on the host monotonic clock; `chip_ns` is the graphs'
reported submit-to-marker time for every step of the beat (prefill chunks included);
`host_overhead_ns = total_ns − chip_ns`. Beats where a prompt joined (`prompt_rows > 0`) are kept
in the receipt and excluded from the gate.

## Host evidence (not chip qualification)

- `host-token-receipt.jsonl`: the batch door over a timed fake stepper (`ChipFake`), 1 then 8
  conversations, 49 records; `--check-decode-timeline` exit 0, `--drop 3:record:0` exit 1
  (`host-token-check.err`, `host-token-drop.err`). `host-token-summary.wl` evaluated with the
  Wolfram MCP gave `{19, 0.115041, True}` for the first version's gate query, which judged the
  lower bound alone; the review (MAJOR 1) replaced it with `GATE_WL` below. Records 0..19 are the
  one-conversation reply. The fake's sleep, not a chip number.
- `seed-red-green.log`: `seeds_spread_at_t07_through_the_batch_door` in a debug build is red
  under `AIEN_SAMPLING_MUTANT=ignore-seed` and `force-greedy` (honest leg, tests.rs:1267), green
  with it unset.
- `host-tests-2.log`: `nice -n 19 taskset -c 0-9 cargo test --release -p aien-serve-http`, all
  green (before the review fixes).
- `host-tests-3.log`: the same command after the review fixes and the #787 fold: lib 92 passed,
  1 ignored; every other target green.
- `host-stamps.jsonl`: the stamped launch timeline of
  `decode_timeline_records_every_single_row_beat` (`StampFake`: 200 µs host sleep before the
  step, `submits` 1, `chip_ns` = dev span + 20000 ns), kept with
  `AIEN_Q36_DECODE_TIMELINE_KEEP`; 19 records.
- `host-rev863.log`: every new CLI path, red first, with the commands and exits: `--bend 3:chip`
  exit 1 then plain exit 0; `--slice 0:19` and `--bend host:600000 --slice 0:19` exit 0; stamped
  `--drop 0:record:0` exit 1 then plain exit 0; usage errors exit 2 (a launch timeline with
  `--bend`, an empty slice, a flag given twice); the lock-holder parse with the right device
  (prints the flock pid) and a wrong one (prints nothing).
- Host gate files, each evaluated whole with the Wolfram MCP (last result = `GATE_WL`):
  - `host-gate-red.wl` (slice 0..19 bent host:600000 + `host-summary-stamps.wl`) →
    `{0.715041, 0.02, 0.735041, False}`;
  - `host-gate-absorb.wl` (slice 0..19 + `host-stamps-absorb.jsonl`, where `chip_ns` takes the
    whole `graph_submit_wait`, the shape of the review's "chip = step wall" mutant that stayed
    green before; the checker still says GREEN) → `{0.115041, 0.628145, 0.743186, False}`;
  - `host-gate.wl` (slice 0..19 + `host-summary-stamps.wl`) → `{0.115041, 0.02, 0.135041, True}`.
  These exercise the queries on fakes; the numbers are the fakes' sleeps, not chip numbers.
