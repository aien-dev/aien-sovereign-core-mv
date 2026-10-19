# Skipped ideas lab: verdicts (Qwen3.8-27B, 2026-10-19)

Branch research/skipped-ideas. Every run (pass, fail, aborted) is under runs/, each with command, ground (load, free memory, binary hash), log, summary and receipt. Tool: crates/aien-inference-abi/src/qwen38/lab.rs and bin/qwen38_lab.rs (f32 CPU reference of our own model, one change per variant, all variants measured against the unchanged model on the same tokens).

How to read the numbers. Text = the model own greedy answer (story: 34 prompt + 256 tokens, code: 50 + 256), fed back token by token; 256 scored positions each. "agree" = the changed model picks the same next token as the unchanged model, out of 256. KL = how far the whole word distribution moved (0 = identical). "GB/token" is a byte count of weights that must be read per token (a model, not a stopwatch). Only the "several answers per pass" row is a chip timing; it ran under the quiet lock on 8 of 64 layers and the whole-model figure is scaled from that, so it is an estimate.

| Idea | Verdict | Receipts |
|---|---|---|
| 1. Layer skipping | RULED OUT by measurement as a speed-up that keeps the answers | skip-story-20261019T1751Z, skip-code-20261019T1846Z, base-*: early exit |
| 2. TEAL activation sparsity (alone) | PARTLY: quality holds at 20 to 30 percent, saving is unproven in time | teal-story-20261019T1824Z, teal-code-20261019T1905Z |
| 2b. TEAL with a shared mask across verify rows | RULED OUT by measurement | same two runs |
| 3. 3-bit weights | PARTLY: 3 bit as plain rounding costs too much; 4 bit is the usable point | quant-story-20261019T1730Z, quant-code-20261019T1835Z |
| 4. Online drafter learning | RULED OUT by measurement for the updaters tried | mtp-code-20261019T1730Z, mtp-arch-20261019T1742Z |
| 5. Several answers per pass | PROVEN for 2 to 4 rows (about 2.25x tokens per second at 4 rows, estimate) | rows-L8-20261019T1730Z |

## 1. Layer skipping
- Early exit (stop after N of 64 layers): agrees on 93/256 (story) and 97/256 (code) at 60 layers, 198/256 and 221/256 at 63 layers. The last layers matter.
- Skipping layers in the middle, static (same layers always): skipping 1 attention layer keeps 248/256 (story) and 251/256 (code) while reading 98.6 percent of the bytes; skipping 8 layers keeps 183 to 187/256 (story) and 231 to 235/256 (code) while still reading 88.6 percent. Best saving seen is 11.4 percent of bytes for a 29 percent token loss on story and 8 to 10 percent loss on code.
- Dynamic skipping (decide per token, 4 or 8 DeltaNet or attention layers, with or without state update): best is 247/256 (code) and 238/256 (story) at 97 percent of the bytes, i.e. 3 percent saving.
- Reading: the model has no layers cheap to drop; the byte saving per skipped layer (1.4 percent) is small next to the damage. Not a free speed-up, ruled out for lossless serving. A lossy mode at 4 skipped layers (5.5 percent fewer bytes, 6 to 13 percent token changes) is possible but not attractive next to 4-bit weights.

## 2. TEAL (zero the small activations, skip the matching weight columns)
- Per-row (each token picks its own columns, the decode case): 20 percent sparsity keeps 249/256 on both texts and reads 81.9 percent of the bytes; 30 percent: 241 (story) and 249 (code) of 256 at 72.8 percent bytes; 50 percent: 212 and 235 of 256 at 54.7 percent. So decode quality survives 20 to 30 percent. Byte saving is real on paper only: a column-gather kernel on the chip has not been built or timed here, so wall-time gain is unproven.
- 2b. With speculative verify (several rows ride one weight read): the union of the columns the rows need grows fast. At 50 percent per row, the union over 2/4/8/16 rows keeps 71/89/98/99.5 percent of columns (both texts, same values within 1 point), so the weight read is only 11 percent smaller at 4 rows and nothing at 8. Forcing one shared mask instead keeps the read small but breaks quality: shared over 4 rows at 30 percent keeps 204/256 (story) and 237/256 (code); at 50 percent 172 and 223; shared over 8 rows at 50 percent 126 and 217. Verdict: no bandwidth win when verifying several words at once.

## 3. 3-bit weights (plain rounding of the real FP8 weights, no calibration)
- 4-bit: MXFP4 keeps 223/256 (story) and 243/256 (code) at 57.6 percent of the bytes; int4 with group 32 keeps 222 and 245 at 60.4 percent.
- 3-bit: MX int3 173 and 223, MX fp3 170 and 226, int3 group 32 184 and 227 (of 256) at 46 to 49 percent of the bytes; KL 5 to 8 times worse than 4-bit on story.
- Reading: with plain rounding, 3 bit loses 28 to 34 percent of story tokens. The 3 bit bytes (46 percent) buy about 20 percent fewer bytes than 4 bit (58 percent) at a large quality cost. This is the floor without calibration or retraining; GPTQ-style calibration or training could close part of the gap and was not tried here, so the idea is not closed, only the naive version.

## 4. Online drafter learning (the model own draft head adapting during the session, k = 1)
- Replay of the real draft-and-verify loop over a recorded MTP trace, configs: vocabulary bias, hidden bias, low-rank adapter (LoRA) at several rates.
- Code (255 positions): no learning 120/136 hits (88.2 percent); best learner equal (hbias 0.001), others 80 to 85 percent; LoRA at a high rate collapses to 12 percent.
- Long text (arch, 2046 positions): no learning 73.1 percent; best learner 72.8 percent (hbias 0.0003), vbias 63 to 69 percent, LoRA 0.001 52 percent.
- Reading: no tried updater beats the frozen head; most make it worse. The head already agrees with the main model on 74 to 90 percent of rows. Ruled out for these updaters at these rates and session lengths; longer sessions or a trained-offline improvement of the head are not excluded.

## 5. Several answers per pass (multi-row exact path on the chip)
- 8-layer slice, GB10, quiet lock: 1 row 31.9 ms, 2 rows 33.3 ms, 4 rows 52.9 ms, 8 rows 116 ms (the vocabulary projection grows linearly: 13, 26, 53, 105 ms).
- Whole-model estimate: 268 ms per step at 1 row; 146.6 ms per answer token at 2 rows (1.83x), 119.2 ms at 4 rows (2.25x), 129.2 ms at 8 rows (2.07x). Peak at 4 rows; beyond that the matmul turns compute bound.
- Reading: several answers per pass work and pay on our own kernel up to about 4 rows. That is the verify width a draft head can feed. The 8-layer scaling to 64 layers is an estimate, not a full-model chip timing.

## Caveats
- Agreement is teacher-forced on the model own greedy text; free-running generation can drift differently.
- Two texts only. Quality numbers on story are consistently worse than on code (code is more predictable).
- Early runs with a slower kernel were stopped for the shared-machine quiet rule; they are kept under runs/ with STOPPED.txt or ABORTED.txt, and no numbers from them are used.
