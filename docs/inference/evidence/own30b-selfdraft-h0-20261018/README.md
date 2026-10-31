# OWN-30B self-draft acceptance (aienix#4, Heartbeat H0), 2026-10-18

The 30B drafts with only a resident subset of R experts per layer (host routing, top-8 among the allowed,
same renormalize); the exact full-router 30B verifies greedily. Calibration trace: 8 prompts x 256 decode
tokens, distinct from the 20 eval prompts. Eval: 20 prompts, up to 256 greedy tokens each (stop at EOS),
4886 tokens total. Tokens per block = accepted draft prefix + 1, averaged over all blocks.

| R (of 128) | resident experts (R x 48 x 4718592 B) | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|
| 16 | 3.62 GB (calc:e86ec86f) | 1.865 | 1.905 | 1.906 | 1.906 |
| 32 | 7.25 GB (calc:287f0b89) | 2.577 | 2.883 | 2.988 | 2.999 |
| 64 | 14.50 GB (calc:692306fd) | 3.912 | 5.559 | 6.930 | 7.460 |

Receipts per cell: run.txt ACCEPT lines; ratios calc:58f6157b, 44cb0d94, c5a7a76f (R16 k16 and k32),
02e4d2d6, b9012804, bc7cfa66, 3ff52f62, 8f875fbb, eeb24d92, 3bc738f1, 36bbc76b.
Trace coverage of the resident set on the calibration trace: R16 0.382, R32 0.605, R64 0.865.

Cross-check: literal draft k=8, rewind, full-model step+logits verify, accept prefix+1, on 3 prompts x 3 R:
same block counts as the simulation and the same tokens as the full model (9/9 OK).

Files: `full/run.txt`, `full/run.json` (resident sets, run lengths per position), `full.start`, `smoke/`.
Limits: greedy only; decode tokens only in the calibration trace (prefill not counted); one model and
quantization (FP8 Qwen3-30B-A3B-Instruct-2507); no speed measured (draft steps run host-routed);
the chip lock was held by a stuck pipeline worker, so the run went without flock on an idle chip.
