# DRA-66 host time per token: protocol

Refs aien-dev/aien-sovereign-core-mv#857 (Linear DRA-66). Written before any chip run; the
window has not run. Every chip line below is NOT_RUN until Main runs `window.sh` in an
exclusive window.

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

`flock -w 900 /tmp/aien-gb10.lock bash evidence/dra-66-host-time-20261031/window.sh`, after Main
stopped 18091 and Flint. Without the lock held by an ancestor, with 18091/18300/18310
listening, with a compute process on the chip, or with a different binary sha, it prints
`NOT_RUN: …` and exits 3 before any chip contact (host check: exit 3 on the unheld lock).
Receipts land in `/home/drakestapleton/handoffs/dra-66-host-time/window-<UTC stamp>/`.

## Gates, in order

| # | Gate | Expected | Receipt |
|---|------|----------|---------|
| 1 | Negative control: mutant server (`AIEN_Q36_TEST_MUTANT=4`), the gate2 request (24 greedy tokens of "Write a C function that returns the larger of two ints.") | `DIFFER` from `evidence/q36-host-20261020/gen-chat-max-24.json` `generated_ids`; otherwise the window stops (exit 4) | `golden-mutant4.json`, `window.log` |
| 2 | Golden identity, honest server with the token receipt on | `IDENTICAL` | `golden-honest.json` |
| 3 | Per-token receipt, 1 conversation (3 sequential greedy replies of 256 tokens) and 8 conversations (8 concurrent greedy replies of 256 tokens) | checker exit 0 | `tokens.jsonl`, `check.txt`, `spells.txt` |
| 4 | Checker negative controls | `--drop 0:record:0` exit 1, `--drop <n/2>:record:0` exit 1 | `red-drop-first.txt`, `red-drop-middle.txt` |
| 5 | Seeds at T=0.7 through the HTTP door (seed 1, seed 2, seed 1 again, 64 tokens) | seed 1 repeats, seed 2 differs | `seed-*.json`, `window.log` |
| 6 | Host overhead ≤ 0.5 ms per token at one conversation | Main evaluates `summary.wl` with the Wolfram MCP; the first query returns `{beats, median host_overhead ms, verdict}` over clean 1-row beats (rows 1, no prompt joining); PASS needs verdict `True` | `summary.wl` + the Wolfram reply quoted in the report |

Gate 6 is a quiet-window number only if the window is quiet (no other chip work, 18091 down);
`preconditions.txt` records the load. The receipt's `total_ns` is beat close to beat close on
the host monotonic clock; `chip_ns` is the graphs' reported chip time for every step of the
beat (prefill chunks included); `host_overhead_ns = total_ns − chip_ns`. Beats where a prompt
joined (`prompt_rows > 0`) are kept in the receipt and excluded from the gate.

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
