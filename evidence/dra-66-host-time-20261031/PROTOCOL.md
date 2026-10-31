# DRA-66 host time per token: protocol

Refs aien-dev/aien-sovereign-core-mv#857 (Linear DRA-66). Written before any chip run. Every chip
line below is NOT_RUN until the P-DRA-4 chip queue (`~/handoffs/pdra4-hive/chip/run-queue.sh`) has
run the two DRA-66 slices in `chip/`.

## Binary

- `/home/drakestapleton/work/dra-66/bin-head/aien-serve-http`, sha256
  `a33cc2abf545927b293316cca018ff81d492bacfdd8fb996dcc5dccdb02449e4`.
- Built from sovereign-core `9d70dca4eb41facf906092998c0033dcedbf84dd` (the binary reports
  `sovereign-core commit: 9d70dca4… (dirty 0)`) with omega `a647b6a9` (`omega.lock`), physics
  `048e5f5e`, aienos `b84c0a67`:
  `scripts/pins.sh --gpu ~/work/dra-66/pins && . ~/work/dra-66/pins/env.sh &&
  nice -n 19 taskset -c 0-9 cargo build --release -p aien-serve-http --bin aien-serve-http
  --target-dir ~/work/dra-66/target-gpu`. Native `libomega_gpu.a` linked (no CPU stub).
- The commits after 9d70dca4 on the branch touch only evidence files and the PR text.

## Server

The live 18091 flags on loopback port 18190: `--engine omega --model-dir
/home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0 --port 18190 --bind 127.0.0.1
--max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8`, env
`AIEN_GB10_LOCK_HELD=1 AIEN_Q36_STROKE=1 AIEN_STROKE_IN_WINDOW=1 AIEN_OMEGA_FP8_GROUPED_WAVE=0`
(the wave setting is the one 18091 actually serves; evidence in `chip/PLAN.md`).

## Run: two P-DRA-4 chip-queue slices

`chip/` holds the slices, byte-identical (`chip/SHA256SUMS`) to `~/handoffs/pdra4-hive/chip/DRA-66/`,
listed in `chip/QUEUE.txt` under `# Main (DRA-66)`:

- `chip/s01-tokens.sh`: `mutant4-red`, `golden-identity`, `receipt-1conv`, `receipt-8conv`,
  `seed-t07`, `check-tokens` (cap 889 s);
- `chip/s02-stamps.sh`: `stamps-1conv`, `check-stamps` (cap 377 s; refuses until s01 has a PASS
  VERDICT);
- `chip/dra66-lib.sh`: the body both source; `chip/PLAN.md`: gates, caps, the Wolfram wall model,
  the lock route and Main's steps after the window; `chip/dry-host.log`: the host dry path and its
  negative controls.

Main runs them with the rest of the queue (GO, preflight, 18091 and Flint down); the runner opens a
`--gb10 none --quiet exclusive --lease 20` window per slice (each slice's `# pdra4-gb10: none`
header) and caps each at 1140 s. Receipts: `~/handoffs/pdra4-hive/receipts/DRA-66/<slice>/`, one-line
`VERDICT` each.

Why the slice takes the chip lock: with `AIEN_GB10_LOCK_HELD=1` the server accepts an inherited
chip reservation only when the flock holder of `/tmp/aien-gb10.lock` is one of its ancestors
(`crates/aien-serve-http/src/chip_reservation.rs`, `detect`; refusal "none an ancestor of this
server", exit 1). spark-window's holder (`setsid spark-window _hold ID`) is never an ancestor, so in
a window that holds the chip the server cannot start. The slice takes the lock through `flock -w 90
-E 75`, which is the ancestor of every server; it verifies that holder in `/proc/locks` (device and
inode) and in the server's own "chip reservation inherited" line. No new trust path; the window's
quiet lock keeps the timings quiet. This replaces the manual single-window `window.sh` (removed).

## Gates and budget

See `chip/PLAN.md` for the table. Each cap is enforced inside its gate: server load ≤ 180 s to
health, requests by `curl -m`, server stop ≤ 30 s then KILL. A gate starts only if its cap fits
before both the lease end and 1140 − 15 s after the slice started; otherwise it and the later gates
are NOT_RUN. Wolfram (MCP evaluator), quoted in full in `chip/PLAN.md`: caps 889 s and 377 s; expected
walls 222.1264576 s and 93.007232 s with 78 s loads and 5 s stops (476.1264576 s and 220.007232 s at
the 180 s load cap and 30 s stops), from the pinned DRA-64 model t(1) = 23.494 ms, t(8) = 83.7676 ms.

`receipt.json` carries an overall `status`: FAIL if any gate failed, else NOT_RUN if any gate did
not run, else PASS (exit 4 / 3 / 0). That status covers the receipts' mechanics only; the 0.5 ms
verdict is Main's Wolfram evaluation of `s02-stamps/gate-red.wl` then `s02-stamps/gate.wl`.

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
  The CLI runs used the host (CPU stub) build of 3cd0410d; 9d70dca4 differs from it only by
  rustfmt and the regenerated `host-tests-3.log`.
- `chip/dry-host.log`: `DRY_RUN=1` runs of both slices on the host (no window, lock, GPU or
  server): static checks, plans with caps, and the slices' own checker gates on
  `host-token-receipt.jsonl` (slice 0..19, exits `0 1 1 1 0 0`) and `host-stamps.jsonl` (exits
  `1 0`); the resulting `gate.wl` / `gate-red.wl` equal `host-gate.wl` / `host-gate-red.wl` from
  line 2 on, and the Wolfram MCP gave `{0.715041, 0.02, 0.735041, False}` and
  `{0.115041, 0.02, 0.135041, True}` for them. Negative controls: outside a window NOT_RUN exit 3
  with no VERDICT; a fake window NOT_RUN exit 3; the inner part with no ancestor holding the chip
  lock NOT_RUN exit 3 before any chip query; a corrupted binary sha exit 3.
  `queue-dry-run.log`: `run-queue.sh --dry-run` listing both slices `(gb10 none)`.
