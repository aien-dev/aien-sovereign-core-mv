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
comes free within 90 s and its exclusive holder is an ancestor of the script; nothing answers
on 18091 and nothing listens on 18091/18300/18310; no compute process is on the chip; the
binary has the pinned sha256 and a native omega gpu engine. Host check: run outside a window,
it printed `NOT_RUN: not inside a spark-window (…)`, exit 3.

Receipts land in `/home/drakestapleton/handoffs/dra-66-host-time/window-<UTC stamp>/`. Each
gate writes `gate-<name>.json` (schema `aien.dra66-window-gate/1`): `window_id`, `window_dir`,
`bin`, `bin_sha256`, `sc_commit` (from `--version`), `gb10_lock_holder_pid`, `start`, `end`,
`elapsed_s`, `lease_left_s`, `status` (PASS / FAIL / NOT_RUN), `detail`, and the sha256 of every
file the gate produced. At exit `receipt.json` (`aien.dra66-window/1`) collects all gates with
the window id, binary sha256 and total elapsed time; the board gets GPU-START / GPU-END lines.

## Gates, in order, and the 20-minute lease

A gate starts only if the lease (`held` mtime + `lease` minutes of the window record) has at
least the gate's cap left; otherwise that gate is written NOT_RUN and the script stops (exit 3).
Each cap is enforced inside the gate: server load ≤ 180 s to health (time-bounded loop),
requests by `curl -m`, server stop ≤ 30 s then KILL. Server loads measured 39 s, 57 s and 78 s
in the GoldenP04 window (`~/handoffs/GoldenP04/evidence/exec-20261027T110114Z/window.log`,
2026-10-27, same model and flags). Decode estimates use the pinned DRA-64 model
(t(1) = 23.494 ms, t(8) = 83.7676 ms per beat).

| Gate | Content | Expected | Estimate | Cap | Receipt |
|------|---------|----------|----------|-----|---------|
| (outer) | window checks, `flock -w 90` | lock free at once | ~0 s | 90 s | `preconditions.txt`, `gate-preconditions.json` |
| `mutant4-red` | mutant server (`AIEN_Q36_TEST_MUTANT=4`), the gate2 request (24 greedy tokens of "Write a C function that returns the larger of two ints."), stop | `DIFFER` from `evidence/q36-host-20261020/gen-chat-max-24.json` `generated_ids`; otherwise FAIL and the window stops (exit 4) | load ≤ 78 s + 0.56 s + stop ~2 s | 270 s | `golden-mutant4.json`, `server-mutant4.log` |
| `golden-identity` | honest server, per-token receipt on (`AIEN_Q36_DECODE_TIMELINE_KIND=token`), the gate2 request | `IDENTICAL` | load ≤ 78 s + 0.56 s | 240 s | `golden-honest.json` |
| `receipt-1conv` | 3 sequential greedy replies of 256 tokens | all 3 answer | 18.04 s + 2 s idle | 180 s | `solo-*.json`, `tokens.jsonl` |
| `receipt-8conv` | 8 concurrent greedy replies of 256 tokens | all 8 answer | 21.44 s + 2 s idle | 120 s | `eight-*.json`, `tokens.jsonl` |
| `seed-t07` | seed 1, seed 2, seed 1 again at T=0.7 through the HTTP door, 64 tokens | seed 1 repeats, seed 2 differs | 4.51 s | 90 s | `seed-*.json` |
| `checker` | stop the server, `--check-decode-timeline` + two `--drop` red controls | exit 0, then exit 1 and exit 1 | ~5 s | 35 s | `summary.wl`, `check.txt`, `red-drop-*.txt`, `spells.txt` |
| (Main) | host overhead ≤ 0.5 ms per token at one conversation | Main evaluates `summary.wl` with the Wolfram MCP; the first query returns `{beats, median host_overhead ms, verdict}` over clean 1-row beats (rows 1, no prompt joining); PASS needs verdict `True` | — | — | `summary.wl` + the Wolfram reply quoted in the report |

Wolfram (MCP evaluator):
- `t1 = 23.494; t8 = 83.7676; {24 t1/1000, 3*256 t1/1000, 256 t8/1000, 3*64 t1/1000}` →
  `{0.563856, 18.043392, 21.4445056, 4.510848}` s.
- `caps = {120, 30 + 180 + 60, 180 + 60, 3*60, 120, 3*30, 30 + 5}; {Total[caps], 20*60 - Total[caps]}`
  → `{1055, 145}` s of the 1200 s lease. The outer row counts 120 s: 90 s flock plus 30 s of
  checks.
- Expected run: `2*78 + 2*0.563856 + 18.043392 + 21.4445056 + 4.510848 + 3*2 + 2*2 + 5 + 5` →
  `221.13` s (3.69 min) with the slowest measured load twice. The run fits in one window, so the
  script is not split.

The timing number is a quiet-window number only because the window holds the quiet lock
exclusively with 18091 and Flint down; `preconditions.txt` records the load average. The receipt's
`total_ns` is beat close to beat close on the host monotonic clock; `chip_ns` is the graphs'
reported chip time for every step of the beat (prefill chunks included);
`host_overhead_ns = total_ns − chip_ns`. Beats where a prompt joined (`prompt_rows > 0`) are kept
in the receipt and excluded from the gate.

## Host evidence (not chip qualification)

- `host-token-receipt.jsonl`: the batch door over a timed fake stepper (`ChipFake`), 1 then 8
  conversations, 49 records; `--check-decode-timeline` exit 0, `--drop 3:record:0` exit 1
  (`host-token-check.err`, `host-token-drop.err`). `host-token-summary.wl` evaluated with the
  Wolfram MCP gives `{19, 0.115041, True}` for the gate query: this exercises the query, it is
  the fake's sleep, not a chip number.
- `seed-red-green.log`: `seeds_spread_at_t07_through_the_batch_door` in a debug build is red
  under `AIEN_SAMPLING_MUTANT=ignore-seed` and `force-greedy` (honest leg, tests.rs:1267), green
  with it unset.
- `host-tests-2.log`: `nice -n 19 taskset -c 0-9 cargo test --release -p aien-serve-http`, all
  green.
