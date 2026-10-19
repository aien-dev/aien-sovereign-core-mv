# Flash-Next M1 (part 1): host reference, fetched tensors, expert format study (2026-10-19T16:49Z)

Branch flashnext/port of aien-sovereign-core, base 2a33909e (qwen38/serve-fast). Host only, no GPU.
Machine: the Spark (GB10), loadavg at start and end in formats.log. Model checkpoints NOT downloaded (disk blocker,
master log ~/handoffs/2026-10-19-flashnext-on-omega.md); every tensor below was fetched by HTTP range
(tools/flashnext/hf-range-fetch.sh). No tensor bytes are in this folder: sources/*.SOURCE.tsv list repo, revision,
shard, byte range and sha256 of every fetched tensor, so the bytes can be fetched again and checked.

## What is here
- `crates/aien-inference-abi/src/flashnext.rs`: the reference (config, n-gram hash, NVFP4 / MXFP4 / FP8 / bf16
  projections, 4-lane gated residual, Gated DeltaNet with the sigmoid output gate, QSA indexer + indexed attention,
  MoE top-10 of 512 + shared expert, PLE n-gram block, the model loop). Spec: transformers `modular_qwen4_exp.py`
  read as the specification (nothing copied), config.json of Qwen/Qwen3.8-Flash-Next @ de4b8e4d.
- `m1-unit-tests.log`: 13 host tests PASS. They include: the derived n-gram multipliers, head primes and offsets
  EQUAL the values the checkpoint ships (fixtures/flashnext/ple-constants.json, fetched from the FP8 repo);
  one call == token by token through every carried state (DeltaNet, KV + indexer keys, PLE history and dilated conv),
  with the indexer budget shrunk so the mask bites; negative control: dropping the PLE conv state is caught; gated
  residual and NVFP4 decode against scalar formulas; indexer selects everything under the budget and exactly
  budget + tail past it, and picks the aligned block.
- Tokenizer: Flash-Next tokenizer.json sha256 0997f410... equals the Qwen3.8-27B one (tokenizer-sha256.txt), so the
  existing tokenizer path is reused unchanged.

## Expert format study (MEASURED, formats.json, command below)
6 real experts (layers 0, 24, 47; experts 0, 311), each fetched three times: bf16 (Qwen/Qwen3.8-Flash-Next
@ de4b8e4d, sliced from the fused tensors), FP8 128x128 (Qwen/Qwen3.8-Flash-Next-FP8 @ 236dfdf2), NVFP4
(nvidia/Qwen3.8-Flash-Next-NVFP4 @ fc694b54). Error against the bf16 checkpoint, relative RMS over the three
matrices; output cosine of the expert MLP on 8 pseudo-random unit-RMS rows (not real activations).

| form | bytes per expert | weight rel RMS (6 experts) | output cosine (6 experts) |
|---|---|---|---|
| FP8 block 128 (Qwen) | 4,915,800 | 0.0265 to 0.0266 | 0.9989 to 0.9992 |
| NVFP4 (NVIDIA, e4m3 per 16 + f32) | 2,764,812 | 0.0812 to 0.0816 | 0.9900 to 0.9922 |
| MXFP4 MSE exponent, from bf16 | 2,611,200 | 0.1127 to 0.1154 | 0.9807 to 0.9853 |
| MXFP4 Omega rule, from bf16 | 2,611,200 | 0.1177 to 0.1193 | 0.9786 to 0.9841 |
| MXFP4 Omega rule, from FP8 | 2,611,200 | 0.1211 to 0.1224 | 0.9775 to 0.9831 |
| MXFP4 Omega rule, from NVFP4 | 2,611,200 | 0.1392 to 0.1440 | 0.9685 to 0.9779 |

Finding: NVIDIA's NVFP4 experts carry about 30% less weight error than any MXFP4 we can make (0.081 against
0.113 at best), for 5.9% more bytes. Re-quantising NVFP4 to MXFP4 is the worst option. Decision for M2: the Omega
grouped expert GEMV reads NVFP4 natively (same nibble layout as docs/w4-tile-format.md, scale part e4m3 per 16
plus one f32 per row), so the 132.7 GB NVFP4 checkpoint is used as shipped, with no conversion. Not yet measured:
the effect on model output (needs the full model).

## Commands
```
tools/flashnext/mkspec.sh spec-fmt
tools/flashnext/hf-range-fetch.sh Qwen/Qwen3.8-Flash-Next de4b8e4d43b917e7706784d8bb445c9af86a3540 meta/bf16/headers tensors/fmt-bf16 spec-fmt/bf16.spec
tools/flashnext/hf-range-fetch.sh Qwen/Qwen3.8-Flash-Next-FP8 236dfdf285828023ca3bcd3f37366c58a3469b13 meta/fp8/headers tensors/fmt-fp8 spec-fmt/fp8.spec
tools/flashnext/hf-range-fetch.sh nvidia/Qwen3.8-Flash-Next-NVFP4 fc694b54fb0174e0913e6adf86691ef85a4ead47 meta/nvfp4/headers tensors/fmt-nvfp4 spec-fmt/nvfp4.spec
flock -s /tmp/aien-quiet.lock nice -n 10 flashnext_check formats tensors/fmt-bf16 tensors/fmt-fp8 tensors/fmt-nvfp4 runs/formats.json 0:0 0:311 24:0 24:311 47:0 47:311
flock -s /tmp/aien-quiet.lock nice -n 10 cargo test --release -p aien-inference-abi --lib flashnext
```
Bytes fetched so far: fetched-bytes.txt (about 103 MB).
