# L313 results: DeltaNet state precision and one state touch per k tokens (Qwen3.6-35B-A3B)

sc branch hum/state-precision (base q36/serve aa272174). Host reference only (no chip). Model rev
95a723d0. Tool `q36_state_drift` (crates/aien-inference-abi/src/bin), queues `run.sh` and `run2.sh`,
tables `table.sh` -> DRIFT-TABLE.md. Every number below is from the JSON receipts in this folder.

## For Drake, in plain words

The model keeps a running memory for each conversation: 63 MB per conversation in full precision,
read and rewritten for every word it writes. Past about 8 conversations at once, that memory moves
more bytes than the model's own weights. We tested two ways to shrink that traffic.

1. **Store the memory smaller.** Half size (bf16) halves the memory traffic. On five test prompts,
   512 words each, the half-size memory wrote exactly the same words as full size on four, and on
   the fifth it differed once, at word 265, on a word where the model's top two choices were
   0.005 apart (a coin toss). That holds when the experts each word is sent to are held the same.
   When the experts are free to choose, a word or two in a few hundred differs earlier: the tiny
   memory difference tips experts that were tied, the same thing that made gate 1 fail on the chip.
   Quality on the two scoring texts did not change (within 0.01 of the full-size score).
   **Verdict: accept, with measured drift (not identical).**
   Quarter size (fp8) quarters the traffic but drifts more: first differing word as early as word 4,
   97-99% of words still the same, scoring texts within about 0.03 nats (a nat is the unit of
   surprise per word; a known wrong-expert swap moves it by 0.07). **Usable, but clearly lossier; second
   choice after half size.**

2. **Touch the memory once for several words.** When the chip checks k guessed words in one go, it
   can read and write the memory once instead of k times: 6.7x less memory traffic per word at 8
   words, 11.6x at 16 (calculator receipts in KTOKEN-STATE.md). The block form of the maths gave
   exactly the same 511 words as word by word on all five prompts, at block sizes 2, 4, 8 and 16.
   The chip already has the routine that holds the memory on chip across a run of words (prompt
   reading); the change it needs is written as a spec. **No chip run yet; that comes in its turn.**

## Method

- States run in lockstep on the same ids: f32 (reference), bf16, fp8 e4m3 with one f32 scale per
  (value head, k row), and (second queue) fp8 with the scale per (value head, v column). Compute is
  f32; the state is rounded to the format after every token's update and output (what a chip that
  stores the format reads back next token).
- "+forced": the expert ids are taken from the f32 run at the same layer and row (weights from the
  state's own logits). Forced = state drift alone; free = state drift plus router flips.
- f32 continues each golden prompt greedily for 512 tokens (f32 ids == the L306 goldens over their
  32 / 24 tokens on all five); the others are teacher-forced on the f32 ids, so the first step whose
  argmax differs is exactly the first divergence of their own greedy run. Agreement after that is
  top-1 agreement on the f32 text, not a measure of their own text.
- A divergence where the f32 top-1 minus top-2 logit is under 0.1 (L306 gate-1 margin floor) is
  tagged near-tie.
- REJECT rule (set before the runs finished, from the advisor review): on the AGPL passage, the
  forced variant's mean NLL rising by more than 0.06936 nats over f32, the measured effect of
  swapping every row's 8th expert for the 9th (ORACLE.md mutant 1: 0.16450 - 0.09514,
  calc:54abc4c6022bfce48ed40e6b3c0ec6dd35431f8bd95143d6906bfdc730a7f171). A state format costing
  more than that is worse than router jitter the chip already lives with. The wrong-expert mutant
  2 gap (4.43 nats) would pass anything, so it is not used. README passage (ppl 17.5) is
  report-only, added because AGPL (ppl 1.18) is near-saturated.

## Verdicts

| format | state bytes per token (read + write) | greedy, forced routes | greedy, free routes | NLL rise AGPL forced / README forced | verdict |
|---|---|---|---|---|---|
| f32 | 125,829,120 | reference | reference | 0 / 0 | reference |
| bf16 | 62,914,560 | identical 512/512 on p1, p2, p3, chat; p4 first diff at 265 (near-tie 0.005) | first diff 4 to 99 (near-ties) on p1-p4; chat 82 (margin 1.43) | -0.0023 / +0.0003 | ACCEPT, PARTLY (measured drift; not EXACT-EQUIVALENT) |
| fp8 e4m3, scale per k row | 32,440,320 | first diff 4 to 144; 496-511 of 512 agree | first diff 4 to 58; 497-510 of 512 | -0.0104 / +0.0143 | PARTLY: under the REJECT line, clearly lossier than bf16 |
| fp8 e4m3, scale per v column | 32,440,320 | COLSUMMARY | COLFREE | +0.0318 / +0.0034 | COLVERDICT |

EXACT-EQUIVALENT (greedy identical to 512 on every golden) is met by no reduced format. Bytes:
2 x the state (KTOKEN-STATE.md, calc:18861a5d... and calc:9eacec99...). Column-scaled fp8 has a
per-column scale instead of per-row; same byte count at 128 x 128.

Chunked (WY) state update, f32: first divergence none, 511/511 on all five prompts at chunk 2, 4,
8 and 16, free and forced. On the chat prompt, chunk 2 and 4 flipped 804 expert rows (logits cosine
min 0.919, state rel err 0.019) and still gave the same 511 tokens; elsewhere state rel err about
1e-6.

## Limits and caveats

- Rounding every token is the decode case (one token per stroke). A k-token stroke or a prefill
  call stores the state once per call, so for those the per-token numbers are an upper bound.
  Step-0 numbers include the prompt being rounded per token, which the chip prefill would not do.
- The KV cache of the 10 attention layers stays f32 in every variant.
- `reference_vs_decode_agree` in chunk-*.json (511/511 everywhere) compares the host fed 16 rows
  per call with the drift run's one row per call: it checks host matmul batching, not the chunk form.
- Worst-layer state error is high in places (bf16 forced up to 0.32 on the chat prompt): some heads
  hold long memories where rounding accumulates. It did not show in the tokens over 512 steps; longer
  contexts are not measured.
- Host only. The chip receipt for the k-token scheme follows the HUM run order (L313 is 4th) after
  the 18091 swap.

## Files

DRIFT-TABLE.md (all tables), drift-*.json / drift-col-*.json (per step: agree, cosine, NLL of the
f32 pick, state rel err, worst layer, flips; f32 margins), score-agpl.json, score-agpl-col.json,
score-readme.json, chunk-*.json, KTOKEN-STATE.md (bytes + chip spec), run.sh, run2.sh, table.sh,
*.log.
