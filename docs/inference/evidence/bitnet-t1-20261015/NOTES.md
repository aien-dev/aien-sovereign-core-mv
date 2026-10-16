# BitNet T1: ternary loader, host only, 2026-10-15

Design: `~/handoffs/2026-10-15-ternary-engine-design.md` section 5 row T1. Code:
`crates/aien-inference-abi/src/bitnet_omega.rs`. No chip run, nothing of Microsoft's run.
Master: `BitNet-b1.58-2B-4T-bf16/model.safetensors`, sha256 `529637ff...5ae1` (pinned in code).
Layout id: `omega-trit2-rm16-hfrows-v1` (16 trits per little-endian u32, omega's code
0b00 = 0, 0b01 = +1, 0b10 = -1, 0b11 refused; `[n][k/16]`; checkpoint row order).

## Gates

| gate | result | receipt |
|---|---|---|
| config refusals by name (family, relu2, tied, bf16, quantisation, unknown keys) | pass | unit tests `refuses_*` |
| trit grid validated, 0b11 refused with row and column | pass | `trit_grid_is_validated_no_0b11` |
| q,k,v and gate,up stacked: trits and one scale per segment | pass | byte test, `byte_test_passes_and_stacks_*` |
| RSS bound, measured in a child process (three full-width layers, 16k vocab): embedding + larger loader phase + 24 MB | growth 203,575,296 B, bound 224,067,584 B (calc:05935794) | `rss-gate.txt` |
| real checkpoint: resident bytes | 849,387,060 B, about 0.85 GB (calc:f379a7e8) | `real-load.txt` |
| real checkpoint: tensors, uploads, peak RSS | exactly the 332 expected tensors; 121 uploads; peak 986,660,864 B over start, bound 1,052,155,024 B (calc:33d3fb7b) | `real-load.txt` |
| real checkpoint: our scales equal the reference int2 scales (spec 6, layers 0, 15, 29) | all 21 equal | `real-load.txt` |
| gguf trit agreement (informational, parsed here) | 2,026,762,053 of 2,084,044,800 agree, 97.2514 % (calc:736bea70) | `gguf-agreement.txt` |

## Negative controls

- A k,v row swap inside the stacked qkv fails the byte test (`stored row 256`).
- A plus/minus plane swap fails the byte test.
- A k,v scale swap fails the byte test (`segment K ... scale`).
- One word-table byte moved by one code step fails the byte test.
- Layer 1's input and post-attention norms swapped on the host fails the byte test.
- An unexpected tensor (a q_proj bias, a stored down_proj scale) or a missing sub-norm is
  refused by name before any weight is read.
- Two uploaders break the RSS bound: one keeping an f32 image of every matrix and the table
  (growth 1,143,644,160 B) and one keeping an f32 copy of layer 0's qkv only
  (243,077,120 B), both against 224,067,584 B.

## What the gguf count says

The test asserts all three of these. Block size 128 decodes (agreement 6,473,881 of 6,553,600 on layer 0 q; block 64 gives
3,789,499). q and k agree in checkpoint row order, not in the interleaved-RoPE order, so the
gguf keeps the HF rows. The 2.7 % that differ follow from the scale: the gguf quantised with
its own f32 mean (layer 0 q 1.2188548), we with PyTorch's bf16 steps (1.21875, the reference
int2 checkpoint's value), so weights near the half-scale boundary fall either side.

CALC-EXEMPT: the figures without a calc id are measured values copied from the receipt files named beside them.
