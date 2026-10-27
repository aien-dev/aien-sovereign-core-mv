# q36 decode timeline: host results (#785)

Pins: sc `6691c931` (timeline), arm A `f30eb963` for the host A/B, omega `a647b6a9`. Protocol: `PROTOCOL.md` (sha256 `babce814...7d00` at `472eba15`), with Amendments 1 to 3 written before any number they govern existed.

## EXERCISED on host

| Check | Result | File |
|---|---|---|
| `cargo test --release --no-fail-fast -p aien-serve-http -p aien-inference-abi -p aien-omega-q38` | all green except `tinyllama_parity` (4 of 6): the loader refuses the symlinked Hugging Face cache file (`symlink or nonregular checkpoint file`); this branch does not touch the loader or that test | `/home/drakestapleton/handoffs/DecodeTimeline/test-head.log` |
| Door timeline over a stamping fake, 20 tokens | 19 records, checker GREEN | `host-timeline/host-timeline.jsonl`, `host-check.txt` |
| `--check-decode-timeline --drop 5:dev:3` | exit 1, RED (3 findings) | `host-timeline/host-red-5-dev-3.txt` |
| `--drop 7:record:0` | exit 1, RED (11 findings) | `host-timeline/host-red-7-record-0.txt` |
| `--drop 3:host:4` | exit 1, RED (2 findings) | `host-timeline/host-red-3-host-4.txt` |
| `window.sh` precondition, lock held by another lane | `NOT_RUN`, exit 3, no server started, no BOARD line | `window-negcontrol.log` |

## Host zero-cost A/B (switch unset)

20 rounds after 2 warm-ups, seeded order (785), `taskset -c 12 nice -n 19`, 20,000 tokens per run, fake step 100 us. Same digest `565d66b3abf21a8a` and 20,000 tokens in all 44 runs. Load1 6.2 to 7.0 (shared box).

WolframLanguageEvaluator on `host-ab/ab.wl`:

| | A `f30eb963` | B `6691c931` |
|---|---|---|
| median ns per token | 107440.73 | 107504.26 |
| min..max ns per token | 107356.21..107579.22 | 107404.34..107742.43 |

- `Median[nsB/nsA]` -> 1.00048088 (min..max 0.99959..1.00209). Gate `<= 1.02`: **PASS**.
- B was slower in 18 of 20 rounds, `SignedRankTest[nsB - nsA]` -> p 0.00042: a real host cost of `Median` difference 63.5 ns per token, which is `63.5*^-9/20.7*^-3` -> 3.07*10^-6 of the recorded 20.7 ms step.
- Gate sensitivity: `0.02 * 107440.73` -> 2148.8 ns per token.

## NOT_RUN

- Chip timeline (200 tokens on the 35B) and the chip A/B: they need a stopped 18091 and an exclusive window. `window.sh` runs both; the procedure is in `/home/drakestapleton/handoffs/DecodeTimeline/RESUME.md`.
- The ABI host marks and `row_timeline` on the real `Qwen38Omega`: compiled and covered by the door tests only through the fake; the chip run exercises them.

## Addendum: missing-logits fix (`f551b1e2`)

- `missing_logits_fail_the_request` (host, fake stepper that never returns logits, both chip-greedy settings): RED before the fix, "the door hung on a step without the logits it asked for: Timeout" after 20 s (`host-timeline/nologits-before.log`); GREEN after (`host-timeline/nologits-after.log`).
- Package tests at `f551b1e2`: same result as above, all green except `tinyllama_parity` 4 of 6 (`/home/drakestapleton/handoffs/DecodeTimeline/test-head2.log`).
- The host A/B above measured `6691c931`, before this fix. `window.sh` now pins the head binary at `f551b1e2` (sha256 `6eca898f...9c10`), so the chip A/B measures the final code.
