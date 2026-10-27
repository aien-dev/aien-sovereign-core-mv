# Qwen3.8-27B ceilings on the DGX Spark

How fast can the 27B model write tokens (pieces of words) on this machine at most? There are two
limits, and which one applies depends on how many tokens the chip works on per pass over the
weights.

1. Memory ceiling, one token at a time: writing one token reads every weight the model uses
   once, so memory speed limits one user writing one token after another.
2. Compute ceiling, many tokens per pass: once the weights are read, the arithmetic on them limits
   how many tokens in total the chip can work on per second.

Every number below shows its arithmetic, so a calculator is enough to redo it. The `calc:` ids
name the receipt for each step in [calc-receipts/](calc-receipts/) (file `<id>...json`); each
receipt stores the expression and its exact result (see the last section).

## 1. Memory ceiling: one token at a time

Weight bytes read per generated token, FP8 weights (1 byte each) with the bf16 final layer
(lm_head, 2 bytes each), from the layer shapes (calc:ca9900d8):

| Part | Arithmetic | Bytes |
|---|---|---|
| MLP, 64 layers, 3 matrices of 5120 x 17408 each | 64 x 3 x 5120 x 17408 | 17,112,760,320 |
| 48 linear-attention layers, each 5120 x 10240 + 5120 x 6144 + 2 x 5120 x 48 + 6144 x 5120 = 115,834,880 | 48 x 115,834,880 | 5,560,074,240 |
| 16 full-attention layers, each 5120 x 12288 + 2 x 5120 x 1024 + 6144 x 5120 = 104,857,600 | 16 x 104,857,600 | 1,677,721,600 |
| lm_head, 248320 x 5120 weights, 2 bytes each | 248320 x 5120 x 2 | 2,542,796,800 |
| Total | sum of the four lines | 26,893,352,960 |

- Nominal memory speed of the DGX Spark (GB10, LPDDR5x): 273 GB/s (NVIDIA's specification,
  [nvidia.com/en-us/products/workstations/dgx-spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)).
- Memory ceiling: 273,000,000,000 / 26,893,352,960 = 10.15 tokens per second (calc:9822c2ba).
- Measured with our engine: the best weight-streaming speed so far is about 219 GB/s. Three
  whole-model decode runs took 122.18, 123.33 and 123.62 ms per token; their mean is
  (122.18 + 123.33 + 123.62) / 3 = 123.04 ms, and 26,893,352,960 bytes / 0.12304 s = 218.57 GB/s
  (calc:e726ff77). At 219 GB/s the same division gives 219,000,000,000 / 26,893,352,960 = 8.14
  tokens per second (calc:c3bbcc06). That is the practical memory ceiling until the engine
  streams weights faster.

The live server's 7.63 tokens per second is 1000 / 131.05 ms. The 131.05 ms is the median chip
time of 4 runs of 256 greedy tokens (129.0, 129.9, 132.2 and 134.9 ms; the median of four is the
mean of the middle two, (129.9 + 132.2) / 2 = 131.05; `evidence/qwen38-serve-fast-20261019/after/runs.tsv`).
Against the ceiling: (1000 / 131.05) / 10.15 = 7.63 / 10.15 = 0.752 (calc:64f74f90).

## 2. Compute ceiling: many tokens per pass over the weights

Each weight takes part in one multiply and one add for every token the chip works on, so the
arithmetic is 2 operations per weight per token.

- Weights: the bytes above less the second byte of each lm_head weight:
  26,893,352,960 - 248320 x 5120 = 26,893,352,960 - 1,271,398,400 = 25,621,954,560 weights
  (calc:19d172c0).
- Operations per token: 2 x 25,621,954,560 = 51,243,909,120, about 51.2 billion (calc:c7acd006).
  This counts the weights only; attention over the earlier tokens of a conversation adds a little
  more as it grows.

Chip speed for that arithmetic, and which figures are measured and which are specification:

| Chip speed | Measured or specification | Tokens per second in total |
|---|---|---|
| 207.7 TFLOPS, FP8 dense | Measured by a third party, not by us: [StorageReview's DGX Spark review](https://www.storagereview.com/review/nvidia-dgx-spark-review-the-ai-appliance-bringing-datacenter-capabilities-to-desktops), FP8 (E4M3) matrix multiply. It is the FP8 figure our prefill floor uses (`evidence/qwen38-prefill-20261018T1836Z/README.md`). | 207.7e12 / 51,243,909,120 = 4,053 (calc:4a61446a) |
| 250 TFLOPS, FP8 dense | Specification, derived: NVIDIA's 1 PFLOP is FP4 with sparsity (footnote on the page above); without sparsity it is half, 500 TFLOPS FP4 dense; FP8 runs at half the FP4 rate, 250 | 1e15 / 2 / 2 / 51,243,909,120 = 4,879 (calc:72c1312d) |
| 415.4 TFLOPS, FP4 dense | Twice the measured FP8 figure (2 x 207.7); not measured | 415.4e12 / 51,243,909,120 = 8,106 (calc:92ff5f95) |
| 500 TFLOPS, FP4 dense | Specification, derived as above | 1e15 / 2 / 51,243,909,120 = 9,757 (calc:e020bb81) |
| 115.36 TFLOPS, bf16 | Measured by us: Omega's tensor-core micro-test (HMMA.16816.F32.BF16), omega commit 4f5ecdd1, [peak.log in omega-mv](https://github.com/aien-dev/omega-mv/blob/4f5ecdd15d1bda719b523d50359b148b6ec61ea6/evidence/OWN30B-T3-TC-PEAK-BW-5faa0f70-20261013T2236Z/peak.log) (best line: 115.36 TFLOPS) | 115.36e12 / 51,243,909,120 = 2,251 (calc:c749c639), bf16 arithmetic, not FP8 |

So with the FP8 weights shipped today the compute ceiling is about 4,050 to 4,880 tokens per
second in total, and about twice that, 8,100 to 9,760, with FP4 weights (not shipped). We have not
measured an FP8 peak ourselves; 115 TFLOPS bf16 is the only chip peak we measured.

## 3. Why the gap between the two ceilings matters

One user writing one token at a time is held to the memory ceiling: each token needs a full read
of the weights and uses only one token's worth of arithmetic per read. At 10.15 tokens per second
the chip does about 1/400 of the arithmetic it could do (4,053 / 10.15 = 399, calc:28e59303).

A single user gets more than 10.15 tokens per second only by checking several tokens per read of
the weights: guess the next few tokens cheaply, then check all the guesses in one pass over the
weights. Each checked guess costs arithmetic (about 51.2 billion operations) but no extra weight
reading, so a few hundred guesses fit in one pass before arithmetic becomes the limit. Every guess
the model agrees with is a token written without its own read of the weights. The gap between the
memory ceiling (10.15) and the compute ceiling (about 4,050 at FP8) is the room the cascade work
(guessing and checking several tokens per pass) is meant to use. How much of it a real guesser
uses depends on how often its guesses are right; that is not measured yet.

## Sources and how to check

- The byte count and the 10.15 figure were first written down in the operator handoff
  2026-10-18-qwen38-27b-on-omega (line 26), and the three 122 to 124 ms runs in the decode speed
  handoff of 2026-10-19; both live only on the project's Spark. This file is the public copy.
- Redo any line above by hand or with any calculator; the arithmetic is written out in full. The
  receipts in [calc-receipts/](calc-receipts/) hold the same expressions (field `inputs.expr`)
  and their exact results (field `result.value`, a fraction, and `result.decimal`). They were made
  with `aien-calc`, our exact-arithmetic tool, which is not on this branch: it lives on the
  unmerged branch `calc/aien-calc`. Once that is merged, `aien-calc verify calc-receipts/<file>.json`
  recomputes a receipt from its own inputs.

What this does not cover: time to the first token (reading the prompt), which is limited by
compute rather than memory speed; and power.
