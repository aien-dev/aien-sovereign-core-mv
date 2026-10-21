# HUM lever L314: exact two-pass lm_head on Qwen3.6-35B-A3B

## Verdict for Drake (plain English)

**What changed.** Every word the 35B writes ends with one big step that scores all 248,320 words in the
vocabulary and picks the best one. That step reads 1.017 GB from memory every word, more than a quarter of all
the reading. Now it runs in two passes. First, a rough copy of the scoring table at half the size (8-bit numbers)
scores every word quickly. With each rough score comes a proven worst-case error, so the rough pass can rule
out almost every word for certain. Second, the exact original table rescores only the few words still in the
running (31 on average, 761 at most, out of 248,320).

**The number.** The head step drops from 1.017 GB to 0.520 GB read per word (calc:e42a986f, calc:4694a43d).
That raises the 35B's memory-speed ceiling from 58.5 to 67.0 words per second to 68.1 to 78.0
(calc:c66bca93, calc:c52efc2d, calc:9016f1ef; measured memory speed 207 to 237 GB/s).

**Identical words, by construction.** The rough pass may only rule out a word when even its best possible
true score is below the winner's worst possible score, so the word the exact table would pick is always
recomputed exactly and still wins. Checked on 1,010 real positions of the 35B: 0 differences in every variant.
A deliberately broken version (error bounds shrunk) does change words, so the check can fail.

**What did not work.** A 4-bit rough copy (a quarter of the size) is too rough: the proven bound then leaves
20,000 to 218,000 words to recheck. The worst-case bound is about 50 times wider than the typical error, and
at 4 bits that swamps the gap between the best words.

**Served (2026-10-21, Drake 02:25Z "just turn it on").** The served 35B uses it for every greedy request. The words are
identical to the golden and to the lever switched off (5 of 5 extra requests), and a broken build fails. One request at
a time, the chip spent 21.8 ms per word instead of 24.0 (`serve-combined/README.md`). Sampled requests still use the
full head. See "Serving" below.

Chip receipt: see "Chip" below (pending its turn on the GPU when this page was written).

## Method

`logit_i = w_i . h`. Screen `s_i = w~_i . h` with `w~` = e4m3 codes times one bf16 scale per 128 x 128 block (the
layout omega's FP8 GEMV reads, so the chip reuses that kernel). Bound (Row): `|L_i - s_i| <= |w_i - w~_i| |h| +
gamma(|w_i| + |w~_i|) |h|`, gamma covering any f32 summation order (Higham 3.1; 2048 on the host, 2100 on the chip
for the block partial sums). Candidates `C = { i : s_i + B_i >= tau }`, `tau = max_j (s_j - B_j)`. The full argmax
`i*` (first on ties) has `L_i* >= s_j - B_j` for every j, so `s_i* + B_i* >= L_i* >= tau`: `i*` is in C. The recheck
computes the same `L_i` bits as today's head and takes the first maximum, so it returns `i*`. Proof and rounding
margins: `crates/aien-inference-abi/src/exact_head.rs` header; chip margins: omega `XBOUND` in
`src/omega_blackwell_codegen_dec.h`.

## Files

| file | what |
|---|---|
| `hidden.sh`, `hidden-resume.sh`, `hidden.txt` | 1010 final hidden states from the 35B host reference: the 5 goldens replayed token by token (full argmax == golden at every step) + 400 teacher-forced positions each of two passages |
| `analyze.sh`, `analyze-v1-w4-fp8row.json`, `analyze-v1.txt` | W4 and per-row-scale FP8 screens, 4 bounds each |
| `analyze.json`, `analyze-v2.txt` | v1 plus the fp8b screen the chip runs (per-record |C|, bytes/token, mutant sweep, top-20 sampling check) |
| `chain.sh`, `gate-fp8b-row-x*.json/.txt` | identity gate: greedy decode with the two-pass head choosing every token on the 5 goldens, at the proven bound (x1) and the mutants x0.5, x0.02, x0 |
| `chip.sh`, `chip-receipt.json`, `chip.txt`, `chip-mutant1.*` | omega `gpu_exact_head_test --chip`: full head vs two-pass head on the chip per record |

## Host results (1010 records, |C| = rows rechecked exactly)

| screen | bound | |C| mean | |C| max | exact-tau mean | wrong | largest mutant scale that loses a token |
|---|---|---|---|---|---|---|
| W4 (128-col groups) | Group | 107013 | 248320 | 20227 | 0 | 0.02 |
| FP8 per-row 128-col scales | Row | 27.1 | 725 | 4.2 | 0 | 0.02 |
| FP8 per-row 128-col scales | Group | 24.6 | 652 | 4.0 | 0 | 0.02 |
| **fp8b (128x128 blocks, chip layout)** | **Row** | **31.0** | **761** | 4.4 | 0 | 0.05 |
| fp8b | Group | 28.1 | 686 | 4.2 | 0 | 0.05 |

The halved bound (x0.5) loses no token: the proven bound has 20x to 50x slack on these states. The mutant
that does fail is x0.05 / x0.02, and x0 (no bound) always fails; that is the red control.

## Sampling

The 35B's generation config is temperature 1.0, top_k 20, top_p 0.95, and the door masks to the top 20 before
softmax and top-p. Candidates that certify the exact top 20 (`candidates_topk`: tau = 20th largest `s - B`) make
the draw a function of exact logits only: bit-identical draws, no fallback needed. Checked with the Group bound
(`Min` rows of `analyze.json`, which equal Group here): 8080 draws (1010 records x 8 seeds), 0 differ. Not yet
checked with the Row bound the chip uses. Certifying 20 rows costs more rechecks than greedy: fp8b Row |C| mean
712.8, max 13452 rows (about 2.92 MB mean calc:a9bb7f70, 55.1 MB worst calc:55b06ae3, still far below the 1.017 GB head); fp8b Group mean 667.0,
max 12617. The chip pipeline here is greedy only; the top-20 chip path is the next step. A request with top_k = 0
and temperature > 0 needs every logit: the full head runs, always.

## Bytes per token (chip pipeline, mean)

screen codes 508,559,360 + scales 62,080 + 9 passes of n f32 (s, coef, lo, hi, y) 8,939,520 + two padded
ARGMAX reads 1,998,848 + 31 bf16 rows 126,976 = 519,686,784 B (calc:e42a986f) vs 1,017,118,720 B (calc:4694a43d).

## Chip

Pending at the time of writing; filled in from `chip-receipt.json` when the window ran.

## Serving

- `serve/`: bin-e1f02b29, the exact head alone. `serve-combined/`: bin-409c8ffa, with the L356 levers (the swap request).
- Engine: `qwen38_omega/moe.rs` builds `exact_head::chip_screen` at load (7 to 9 s) and `qwen38_engine.rs` turns it on per
  request when temperature <= 0.001.
- Graph: omega `omega_gpu_q36_graph_set_exact_head`, 5 extra launches per token: XBOUND SCALE, FP8 screen, XBOUND,
  ARGMAX x2, then the bf16 MASK GEMV in place of the plain one.
- The omega lib must be built against physics at `physics.lock`. A build with `PHYSICS_LOCK_CHECK=0` against a stale
  checkout cannot load on the GB10 (`serve/attempt1-stale-physics/`).
