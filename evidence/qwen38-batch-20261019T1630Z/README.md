# Continuous batching on the Qwen3.8-27B door: measurements

Branch cascade/continuous-batching. Binary built from 6a440c6d linked to omega cascade/batch-gemm-k32768
fd3191a2 (libomega_gpu.a sha256 6db2e952...), binary sha256 a21197fb... (~/workspace/targets/q38-batch/bin-6a440c6d).
Test port 18191 only. Every run, including failed ones, lands in this folder as it happens.

- bench.sh: N concurrent greedy requests, token ids returned, wall per request and aggregate.
- run.sh single: one-at-a-time door, each prompt alone (single-stream greedy references).
- run.sh batch: --batch-slots 8, each prompt alone, then 2, 4, 8 at once.

## Log
- 2026-10-19T16:44Z host tests (CPU stub, aien-serve-http + aien-inference-abi + aien-omega-q38) at 63b7fc4c under flock -s quiet lock: 448 passed, 0 failed, 22 ignored (host-tests-63b7fc4c.log, host-tests.sh). A first run started 16:2xZ without the quiet lock was stopped by me when the rule arrived.
- 2026-10-19T16:37:51Z single-door reference session started (session.sh); server pid 3519763 on 18191 queued for the GPU lock behind other agents.
- 2026-10-19T17:25Z single-20261019T1635Z ABORTED by me, no measurement: the server loaded (70.0 s, rss 30.08 GB, need 33245 MiB at context 4096) but the first timed call waited 33 min for the exclusive quiet lock (flock has no writer priority; shared holders kept overlapping). Server stopped cleanly. Changed: correctness-only passes (single-door references, batch c-* pass) run without the quiet lock (coordinator rule: correctness-only chip runs do not need it); only timed passes take it.
- 2026-10-19T17:50Z single-20261019T1720Z ABORTED by me: the server loaded (56.2 s) but request single-p0 got "GPU busy: GPU lock held by other jobs for 300 s" (the door waits 300 s per request for /tmp/aien-gb10.lock; 12 jobs were queued on it). run.sh now repeats a bench call (labels -aK, every attempt kept) until no request comes back empty; good.txt names the attempt used. Speeds are taken from the server log (decode time, lock wait excluded), not client wall.
