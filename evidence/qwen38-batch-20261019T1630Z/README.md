# Continuous batching on the Qwen3.8-27B door: measurements

Branch cascade/continuous-batching. Binary built from 6a440c6d linked to omega cascade/batch-gemm-k32768
fd3191a2 (libomega_gpu.a sha256 6db2e952...), binary sha256 a21197fb... (~/workspace/targets/q38-batch/bin-6a440c6d).
Test port 18191 only. Every run, including failed ones, lands in this folder as it happens.

- bench.sh: N concurrent greedy requests, token ids returned, wall per request and aggregate.
- run.sh single: one-at-a-time door, each prompt alone (single-stream greedy references).
- run.sh batch: --batch-slots 8, each prompt alone, then 2, 4, 8 at once.

## Log
