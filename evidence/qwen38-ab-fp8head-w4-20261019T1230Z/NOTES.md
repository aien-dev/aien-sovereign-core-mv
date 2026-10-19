# A/B: FP8 final word-scoring step, and real 4-bit weights, for Qwen3.8-27B on Omega (2026-10-19, WORKING NOTES)

Brief: ~/handoffs/2026-10-19-ab-fp8head-4bit-brief.txt. Final plain-English report goes at the end of this file.

## Ground (12:30Z)
- omega qwen38/decode-speed 0010f7d3 (worktree omega-q38-speed, not mine). sc qwen38/decode-speed b6c61c54 (sc-q38-speed, not mine).
- Live 27B server pid 38628 (RSS ~39.7 GB). MemAvailable ~70 GiB at start.
- Whole-model tool: sc bin qwen38_omega_full <ckpt> 64 golden.json report.json (teacher-forced golden, 47 positions,
  chip ms per step, margin + cosine vs host ref). Runner pattern ~/workspace/omega-q38-speed-work/bin/wm.sh.
- Engine API: Qwen38Omega::open(dir, layers, max_pos), step(token) -> (argmax, GraphStats), logits().
- FP8 GEMV generator: omega src/omega_blackwell_codegen_moe.c (decode e4m3 via bits, 4 int ops + FFMA per weight).
- f7 MXFP4 rule (sc-f7 w4.rs): X = 2^(floor(log2 amax) - 2) per 32 weights along k, e2m1 RNE saturating at 6.

## Plan
1. Worktrees: omega ~/workspace/omega-w4 branch qwen38/w4-gemv off 0010f7d3; sc ~/workspace/sc-w4 branch
   qwen38/w4-decode off b6c61c54. Own target dirs. Fresh omega archive per build (PHYSICS_DIR=~/workspace/physics-6d7cf0d4).
2. Task A tool: new sc bin qwen38_ab: `dump <ckpt> <tokens.json> <out-dir>` teacher-forced through the decode graph,
   writes per-position f32 logits + chip ms; `gen` greedy free-run after a prompt; `compare a b` = argmax agreement,
   top-two margin at flips, cosine min/mean, KL, NLL/perplexity of the fixed text. Head A/B: same tokens, two runs
   (FP8 head on/off), teacher forcing makes the hidden states identical, so only the head differs.
3. Task A speed: qwen38_omega_full (golden) interleaved FP8-head on/off, 4+ rounds each, flock per run.
4. Task B omega: new generator omega_blackwell_codegen_w4.c (MXFP4: nibbles e2m1 -> f32 bits ((n&7)<<22)|((n&8)<<28)
   = value*2^-126; E8M0 scale per 32 weights, one lane step = one 32-weight group; x' = x*2^64; S' bits (E+62)<<23),
   API omega_gpu_w4_api.c (upload, gemv, record, host converter from FP8+bf16 scale = f7 rule, host oracle),
   graph hook set_layer_w4. Tests: host (converter, oracle), sim, chip parity at every 27B shape, mutants caught.
5. Task B sc: AIEN_Q38_W4=1 loads FP8, converts to MXFP4, uploads W4 only. Whole model speed interleaved vs FP8,
   quality via qwen38_ab (top-pick agreement vs FP8 dump, NLL on license text, story + code free-run).
6. Decision table with receipts; handoff final section.

## Log
- 12:45Z advisor (Fable) review: math OK. Changes adopted: (1) realistic 4-bit figure = weight stream 56.7 ms + today's
  24.5 ms non-weight work = ~81 ms / 12.3 tok/s (advisor calc:257ea276, calc:59e255a4), report beside 17.62 paper;
  (2) synthetic 4-bit GEMV speed at k 5120/17408 vs FP8 FIRST, sched-1 style pipeline likely needed; (3) rows 8 vs 4;
  (4) parity data with distinct exponents per adjacent group, mutants: wrong scale byte, swapped nibbles, scale not
  advancing; nibble order in one shared header; subnormal nibbles 1 and 9 in a one-hot chip test, no .FTZ guard;
  (5) head A/B: KL + NLL over a few thousand teacher-forced tokens is the evidence, free-run is a sanity read;
  (6) 4-bit quality three-way (host ref, FP8, 4-bit).
- Worktrees: omega ~/workspace/omega-w4 (qwen38/w4-gemv @0010f7d3), sc ~/workspace/sc-q38w4 (qwen38/w4-decode @b6c61c54;
  ~/workspace/sc-w4 is someone else's old worktree, untouched). Work dir ~/workspace/ab-w4-work, targets ~/workspace/targets/ab-w4.
- sc tool qwen38_ab (text | run [--gen N] | compare) written for the drift study.

## Log 13:2xZ: Task A speed done (4 interleaved rounds each, golden 47 positions, whole model, chip ms median per round)
- bf16 head (h0): 124.05, 123.99, 124.45, 134.41 (r4 outlier, chip shared). Median of rounds 124.25 ms = 8.048 tok/s (calc:e8524c48)
- FP8 head (h1): 119.00, 117.24, 117.62, 122.62. Median of rounds 118.31 ms = 8.452 tok/s (calc:1c19cda2)
- FP8 head faster in every paired round; saves 5.94 ms/token (calc:54c49f10)
- Both: 24/24 golden tokens identical. Full-model gate: bf16 PASS (cos 0.999968, p95_rel 0.0907); FP8 head FAIL on the p95_rel limit (cos 0.999375, p95_rel 0.4696): tokens same, logits drift larger than the gate allows.
## Log: omega qwen38/w4-gemv 3c185ee1 pushed: graph set_layer_w4 + set_lm_head_w4; graph chip test --w4 34 checks PASS, FP8 control caught (runs/g4)
- In-graph 2-layer timeline: W4 mix 437 us vs FP8 777; gate_up 963 vs 1682; down 537 vs 941 (GB/s 191/197/177 vs 202/212/189)
- One-shot sweep timings quantised at ~102 us (host wait granularity): do not rank shapes with them; use graph timeline.
## Log 14:1xZ: Task A quality (teacher-forced, same inputs, bf16 head = a, FP8 head = b; runs/cmpA/*.json)
- top-pick agreement: golden 47/47, license 255/255, prose 286/289, codelong 302/305, code 1000/1023, arch 2018/2047; total 3908/3966 = 98.54% (calc:c42edc26)
- every flip is a near tie: largest bf16 top-two margin at a flip 0.186 logits (code p241); cosine min 0.998498 (arch), mean >= 0.99964 everywhere; KL max 6.9e-3
- mean NLL (nats/token) a vs b: golden 2.0294/2.0215, license 0.0871/0.0874, prose 0.98948/0.98954, codelong 0.9639/0.9654, code 2.1176/2.1189, arch 2.1036/2.1047 (perplexity-style score moves by <= 0.0016 nats/token)
## Log: W4 shape sweep (graph timeline 2 layers, runs/g4s): best 16,2,8 / 16,4,4 / default+coal0 within noise (~3-5%); repeat runs in runs/g4r
## Log: Task A free runs (256 tokens, same prompt): prose identical through position 219 (prompt 34 + 185 generated), code identical through position 119 (prompt 50 + 69 generated); each split at one near-tie, both texts coherent (runs/text/genA-*.txt)
## Log 14:4xZ: W4 default shape 16,2,8 (omega 7ed31249, g4r medians 1975.1 vs 2006.6 us); lib-b sha ce1deb65; sc 47da2ea9; suite + chain-b launched (runs/suite, runs/chain-b.out)
## Log 15:0xZ: first whole-model W4 runs FAILED at load (runs/failed-b1): layer 56 out_proj "dma_alloc REFUSED_LIMIT" = the 1022 buffer-slot cap (FP8 ~608 slots + W4 2 per tensor 512). One WW run also got "create_channel NO_MEMORY (0x51)" right after the failed W run exited (transient: WW-r2 got past it). Fix omega ae900541 (one buffer per W4 tensor, ~900 slots). lib-b sha fb9153fb; suite2 then chain-b relaunched.
## Log 16:0xZ: Task B whole-model speed (4 rounds each, order rotated, golden 47 positions, lib-b fb9153fb, omega ae900541, sc 47da2ea9)
- F  (FP8 weights + FP8 head): 117.39 118.81 122.55 129.11 -> median 120.68 ms = 8.286 tok/s (calc:9c639a0c), 0.779 of its 10.64 tok/s weight ceiling (calc:00b3665f, ceiling calc:905bdd97)
- W  (MXFP4 weights + FP8 head): 74.00 75.16 75.49 76.71 -> median 75.325 ms = 13.28 tok/s (calc:c5eab84e), 0.693 of 19.17 (calc:a3d7bb22, ceiling calc:b8d0bffb)
- WW (MXFP4 weights + MXFP4 head): 70.40 71.91 72.82 73.25 -> median 72.365 ms = 13.82 tok/s (calc:da33c438), 0.691 of 20.01 (calc:58f08c18, ceiling calc:07f2175b)
- golden: F 24/24, W 24/24, WW 23/24 (step 38 golden 318, WW picks 264, its margin 0.744). Residual cosine min W 0.782 / WW 0.777 (FP8 0.99996). Load: W 122-161 s, WW 93-145 s (host conversion) vs F 36-61 s.
## Log 17:3xZ: converter scale-rule survey (host only, all 64 layers, 24.33 G weights, runs/w4err): OCP rel RMS 0.116722 with 629,015,175 clipped; FIT 0.118093 with 0 clipped (worse overall); BEST 0.112674 with 348,131,158 clipped (3.47% lower error, calc:b50e06c5). Chip quality dumps with AIEN_Q38_W4_RULE=best queued (bin/chain-d.sh, sc-t binaries, lib-c ede0cfe2) after chain-c retries.
## Log 18:2xZ: Task B quality table (teacher-forced vs h0 = FP8 weights + bf16 head; runs/cmpB): agree / positions, cosine min, mean NLL h0 -> 4-bit, KL mean, largest h0 margin at a flip
- W  golden 47/47 0.782 2.0294->1.8573 KL 0.028 | license 255/255 0.891 0.0871->0.0928 | codelong 285/305 0.403 0.9639->0.9329 max 2.21 | code 901/1023 0.536 2.1176->2.1649 max 1.73 | arch 1811/2047 0.103 2.1036->2.1435 max 9.46
- WW golden 45/47 0.777 ->1.9012 (flips p18 h0 margin 6.62, p38 0.74) | license 254/255 0.886 ->0.0887 | prose 254/289 0.830 0.9895->1.0200 max 1.67 | codelong 287/305 0.402 ->0.9494 max 2.21 | code 880/1023 0.538 ->2.1754 max 1.78
- Not measured yet (memory guard refused, chain-c retrying): prose W, arch WW, story free run W/WW, code free run W. Code free run WW decoded: coherent C (runs/text/genB-codelong-WW.txt).
- Fable review 18:1xZ: W vs WW per-round ranges do not overlap (slowest WW 73.25 < fastest W 74.00), ranking real; a second 27B server shared the chip in every round, so the ranking holds, absolute figures are under load. Asked: say "vs FP8 model with bf16 head", list gate FAIL for all quantised configs, compute missing compares (done for codelong-WW).

---

# FINAL REPORT (plain English)

## What was asked
Make Qwen3.8-27B write words as fast as our own engine allows on the Spark, by testing two choices instead of assuming:
(A) a smaller number format (8-bit FP8) for the very last step that scores every possible next word, against the
current 16-bit (bf16); (B) storing the model's weights in 4 bits (MXFP4) instead of 8 bits, with a 4-bit reading routine
written by us in our own GPU engine.

## The decision
**Fastest end to end: 4-bit weights everywhere, including the last scoring step (called WW below): 13.82 words
(tokens) per second, against 8.29 for today's 8-bit setup. That is 1.668 times as fast (calc:cfc16fae).**
For Task A alone: the 8-bit last step beats the 16-bit one, 8.45 against 8.05 tokens per second, so where 8-bit
weights stay in use, the 8-bit last step is the pick.

The quality cost of WW is real and is listed below. It is the fastest setup, not yet a quality-approved one.

## Speed table (whole model, median of 4 rounds, order rotated, every round under the GPU lock)
| Setup | ms per token | tokens per second | share of its memory-speed ceiling | receipt |
| --- | --- | --- | --- | --- |
| 8-bit weights, 16-bit last step (old default) | 124.25 | 8.048 | | calc:e8524c48 |
| 8-bit weights, 8-bit last step | 118.31 to 120.68 | 8.29 to 8.45 | 0.779 of 10.64 | calc:1c19cda2, calc:9c639a0c, calc:00b3665f |
| 4-bit weights, 8-bit last step (W) | 75.33 | 13.28 | 0.693 of 19.17 | calc:c5eab84e, calc:a3d7bb22 |
| 4-bit weights, 4-bit last step (WW) | 72.37 | 13.82 | 0.691 of 20.01 | calc:da33c438, calc:58f08c18 |

- The rankings are not luck: in every pair of rounds the faster setup won, and the slowest WW round (73.25 ms) was
  quicker than the fastest W round (74.00 ms). Task A's 8-bit last step saved 5.94 ms per token (calc:54c49f10).
- Two runs measured the 8-bit setup at different times (Task A 118.31, Task B 120.68); both were under the same load.
- Another program's second copy of the 27B model shared the GPU during every round, so the absolute numbers are
  "under load"; the order between setups is what the rotation protects.
- The paper figure 17.62 in the brief assumed a 16-bit last step and a perfect memory stream. Measured: 13.82.

## Quality table
Reference = the same model with 8-bit weights and the 16-bit last step (the old default), fed the exact same text.
"Same pick" = how often both choose the same next word.
| Text | 8-bit last step (Task A) | 4-bit weights W | 4-bit everywhere WW |
| --- | --- | --- | --- |
| Reference sentence, 47 words | 47/47 | 47/47 | 45/47 |
| Licence text, 255 words | 255/255 | 255/255 | 254/255 |
| Story text, 289 | 286/289 | NOT MEASURED (memory) | 254/289 = 87.9% (calc:4e474c3b) |
| Long code, 305 | 302/305 | 285/305 | 287/305 = 94.1% (calc:9c04794b) |
| C source file, 1023 | 1000/1023 | 901/1023 = 88.1% (calc:d0d0741b) | 880/1023 = 86.0% (calc:767b387a) |
| Architecture document, 2047 | 2018/2047 | 1811/2047 = 88.5% (calc:abfbb73e) | NOT MEASURED (memory) |

- Task A total: 3908 of 3966 = 98.54% same pick (calc:c42edc26). Every disagreement was a near tie (the two best
  words within 0.19 of each other), and the perplexity-style score moved by at most 0.0016 per word. Effectively free.
- 4-bit: about 1 word in 8 differs on code and technical text, and some differences are confident ones (the reference
  was sure of its word by a margin up to 9.5 on the architecture text, 6.6 on the reference sentence), so it is not
  only near ties. Perplexity-style score (lower is better, nats per word), reference to 4-bit: licence 0.087 to 0.093
  (W) / 0.089 (WW); C source 2.118 to 2.165 (W) / 2.175 (WW); architecture 2.104 to 2.144 (W).
- Free-running writing: the 4-bit code sample reads as correct, sensible C (runs/text/genB-codelong-WW.txt). The 4-bit
  story samples and the W code sample were blocked by the memory guard: NOT MEASURED.
- Our existing full-model accuracy check, written for 8-bit agreement, FAILS for every reduced setup (8-bit last step
  included), although all of them produce the same 24 reference words when writing freely, except WW, which makes
  1 different choice out of 24 (a close call). A check fit for 4-bit is owed.

## Where the time goes now (GPU timeline, WW 71.68 ms per token)
Weight reading is about 67 ms of it; other work about 5 ms (calc:b9134d23). Our 4-bit reading routine averages 209 GB/s
(calc:832d43a7), at best 221 GB/s (calc:e5d5ccc9), against 239 GB/s for the 8-bit one (calc:106355c5). If it reached the full 273 GB/s,
WW would be 54.82 ms per token (calc:b49da385), 18.24 tokens per second (calc:b175589a). That gap is the next work.

## Branches (all pushed, no pull requests opened)
- omega `qwen38/w4-gemv` at dd5e89ff (base qwen38/decode-speed 0010f7d3): the 4-bit reading routine, its tests, the
  graph hookup, the converter rounding rules.
- sovereign-core `qwen38/w4-decode` at c5ae3958 (base qwen38/decode-speed b6c61c54): the switches AIEN_Q38_W4=1,
  AIEN_Q38_W4_HEAD=1, AIEN_Q38_W4_RULE, the A/B tool qwen38_ab, and all evidence in
  evidence/qwen38-ab-fp8head-w4-20261019T1230Z.
- Tests on the final code all pass: chip tests of the 4-bit routine (11 checks, deliberately broken versions all
  caught), graph tests (37 with 4-bit, 35 with 8-bit), 297 computer-side checks, 17 simulator checks.

## Owed
1. Faster 4-bit reading routine: start the next block's memory read while the current one is being added up (needs a
   second "wait slot" in our GPU code generator). Target: the 54.82 ms floor.
2. Convert to 4-bit once, ahead of time, and store it. Today the 4-bit copy is made at every start (93 to 161 s
   instead of 36 to 61 s), and the 8-bit copy is kept too for prompt reading, so the model holds about 44 GB.
3. A 4-bit routine for prompt reading (prefill), so the 8-bit copy can go.
4. An accuracy check made for 4-bit (today's check is for 8-bit agreement and fails all reduced setups).
5. Better rounding when converting to 4-bit: on the computer side, the "BEST" rule cut the rounding error by 3.47%
   (calc:b50e06c5); its GPU quality runs are queued (see below).
6. The runs the memory guard refused: story text with W, architecture text with WW, story free runs W/WW, code free
   run W. They are queued to run by themselves when about 58 GB of memory is free.

## Watchdog warnings seen during this work (none caused by this work, none changed)
GitHub calls going around the guard (about 33 a minute, source unknown), two project-file checks out of date
(benchmarks, interplane), two new scheduled jobs not on the approved list (a notes scanner, a sensor logger),
and 9 leftover hidden browser pieces from a stopped chat.
