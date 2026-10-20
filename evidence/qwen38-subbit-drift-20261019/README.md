# Below 4 bits: CPU drift study of a 3-bit code on Qwen3.8-27B (ledger L134, SPEED-10)

Question: if the 27B's weights were stored in about 3 bits instead of 4, how far would its next-word choices move?
Answered before any GPU kernel exists, on the CPU, on the same six texts and the same reference as the 4-bit study
(`evidence/qwen38-ab-fp8head-w4-20261019T1230Z`).

## What was run
- Reference: the FP8 model with the 16-bit scoring head, computed in 32-bit floats on the CPU (`qwen38.rs`, one pass over the
  checkpoint for all six texts). Its average log-loss matches the GPU run of the same model to 3 to 4 digits
  (code 2.11761 vs 2.1176, arch 2.10388 vs 2.1036, license 0.08707 vs 0.0871).
- Every coded variant is fed the same token ids (teacher forced), so only the weights differ. Metrics per text: same top
  pick as the reference, KL(reference || coded), average log-loss (NLL) of the text's real next word, the reference's
  margin at each disagreement, and the same in windows of 128 positions.
- Codes (all in `qwen38_subbit.rs`, tool `qwen38_subbit`):
  - `mxfp4`: the 4-bit control the GPU runs (4.25 bits per weight).
  - `rht-scalar`: random sign flips plus a 128-point Hadamard rotation along the input dimension, a bf16 scale per 128
    weights, then the best 3-bit scalar levels (3.125 bits).
  - `trellis:K:L:G`: same rotation and scale, then a bit-shift trellis code (QTIP / EXL3 family): an L-bit state, K new
    bits per weight, the weight is a table lookup of the state (2^L entries trained on Gaussian data), exact Viterbi
    encoder. K=3, L=10, G=128 costs 3.152 bits per weight including the scale and the first state of each 256 weights.
  - `--act`: columns scaled by (mean square of that input channel / average)^exponent before the code and back after,
    from a calibration pass over 5000 tokens of held-out text (PROJECT.md, CONSTITUTION.md, qwen3_moe.rs,
    AIEN_RUNTIME_ARCHITECTURE.md; none of them is one of the six test texts). This is the diagonal case of what GPTQ and
    EXL3 do with a full Hessian.
- Checks on the harness: 5 unit tests (Viterbi equals brute force on a tiny code, pack/unpack round trip, Hadamard
  orthonormal, e2m1 ties, training lowers error). The MXFP4 control on the CPU reproduces the GPU 4-bit run: same pick code
  902/1024 (GPU 901/1023), codelong 286/306 (285/305), arch 1819/2048 (1811/2047); NLL code 2.1645 (2.1649).
- Codebook quality on unit Gaussian data (rate-distortion limit at 3 bits is 0.01562): L=8 0.02328, L=10 0.02036, L=12 0.01867.
  Encoding on the loaded CPU cost 46 / 148 / 703 ns per weight, so the whole model takes about 20 min / 63 min / 5 h; L=10 was
  used. L=12 or 16 (what QTIP uses) is expected to be better, not measured.

## Results (same pick % of the reference / NLL change in nats per word; positive NLL change = worse)
| Variant (bits per weight of the body) | golden 48 | license 256 | prose 290 | codelong 306 | code 1024 | arch 2048 |
|---|---|---|---|---|---|---|
| MXFP4 (4.25) | 100.0 / -0.166 | 100.0 / +0.005 | 88.6 / -0.010 | 93.5 / -0.033 | 88.1 / +0.047 | 88.8 / +0.040 |
| rotation + 3-bit scalar (3.125) | 91.7 / -0.435 | 98.4 / +0.074 | 76.9 / +0.100 | 87.3 / -0.042 | 83.3 / +0.117 | 83.4 / +0.047 |
| trellis K3 L10, no calibration (3.152) | 97.9 / -0.151 | 99.2 / +0.017 | 86.2 / +0.052 | 93.5 / -0.017 | 88.1 / +0.071 | 81.8 / +0.572 |
| trellis K3 L10 + calibration exp 0.25 (3.152) | 100.0 / -0.084 | 99.6 / +0.017 | 88.6 / +0.021 | 95.8 / +0.029 | 91.8 / +0.031 | 89.6 / -0.008 |
| same, exp 0.125 | 100.0 / -0.146 | 99.2 / +0.009 | 88.6 / +0.027 | 95.4 / +0.002 | 90.9 / +0.024 | 89.9 / +0.025 |
| same, exp 0.5 | 100.0 / -0.191 | 95.7 / +0.191 | 83.8 / +0.064 | 92.5 / +0.073 | 86.3 / +0.046 | 89.5 / -0.007 |
| trellis K4 L10 + calibration 0.25 (4.152) | 100.0 / +0.029 | 100.0 / +0.007 | 93.1 / +0.012 | 97.4 / +0.020 | 94.2 / +0.008 | 94.3 / +0.045 |
| trellis K3 + calibration 0.25 + MXFP4 head | 95.8 / +0.003 | 99.2 / +0.013 | 87.9 / +0.073 | 93.8 / +0.058 | 90.3 / +0.043 | 87.1 / -0.009 |

For comparison, the GPU run with MXFP4 body and MXFP4 head (today's fastest setup, 13.82 tok/s) had same pick 95.7 / 99.6 / 87.9 / 94.1 / 86.0 on
golden / license / prose / codelong / code (arch not measured). Weight error energy (error over signal): MXFP4 0.0137, trellis
without calibration 0.0201 (the calibrated runs trade weight error for output error, so the number is not comparable).

## Reading the numbers
- Without calibration the 3-bit trellis is not good enough: same as MXFP4 on code and codelong, worse on prose, and the
  architecture text collapses (NLL +0.57). The collapse sits in a few places (the calibrated run shows one 128-position window
  with KL 1.32, rest 0.02 to 0.07), not everywhere.
- With a small held-out calibration the same 3.15-bit code is better than 4.25-bit MXFP4 on 3 of 6 texts by same pick
  (code +3.7 points, codelong +2.3, arch +0.8), equal on 2 (golden, prose) and 0.4 points lower on license. NLL is mixed: better on code and arch,
  worse by 0.02 to 0.06 on prose and codelong. The exponent is flat between 0.125 and 0.25 and worse at 0.5.
- With the 4-bit head added (the moonshot row 6 configuration, 10.26 GB per word) it is within 0.4 points of the GPU MXFP4+MXFP4
  setup on golden, license, prose and codelong (0.4 lower on license and 0.3 lower on codelong, equal on prose, +0.1 on golden)
  and 4.3 points better on code, while reading 25 percent fewer bytes (10.26 vs 13.65 GB per word).
- The calibrated trellis is also a better 4-bit code: at 4.15 bits it is 5 to 6 points closer to FP8 than MXFP4 on prose, code and arch.

## Limits (say them out loud)
- One run per variant, six texts, 4,000 positions: differences of a point or two are not settled. One calibration set, one
  exponent grid. L=10 is a smaller code than QTIP's L=16; the real number should be at least this good, unmeasured.
- Same pick is teacher forced; free-running text was not generated for these variants.
- Speed is NOT measured: no 3-bit GPU decoder exists. On paper: 24.33 G body weights x 3.152 bits + 1.2714 G head weights x
  4.25 bits = 10.26 GB per word (calc:35dffe17), ceiling 273 GB/s -> 26.6 tok/s (calc:d6d1cc2c); at the 4-bit engine's measured 0.691 of its
  ceiling that is 18.4 tok/s (calc:9cf923b4), against 13.82 measured for 4 bits. That assumes a decoder that keeps the stream at the same
  efficiency; the table lookup per weight on sm_121 has no receipt. J per word: not measured (CPU study).
- The activation columns of q/k/v and gate/up share rotation signs by construction, but the runtime cost of rotating the activation
  and the 3-bit decode are not built.

## Go / no-go by our own numbers
GO for the next step (a small GPU microbenchmark of the trellis decode on sm_121, then a kernel), with calibration required: the
data-free version is a NO-GO. Gate for the kernel step: decode must hold the weight stream at 0.69 of the 273 GB/s ceiling or better,
otherwise 3 bits does not beat the 4-bit engine's 13.82 tok/s. Files: `out-*/summary.json`, per text JSON with windows (runs from
the 3rd on), `chain*.sh` (exact commands), `ref/` (reference metrics).
