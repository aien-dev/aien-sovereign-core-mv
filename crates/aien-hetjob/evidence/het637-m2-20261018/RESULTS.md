# het637-m2-20261018: Spark coordinator to M2 worker, first physical run

Issue: aien-sovereign-core#637 (parent aienix#6). Branch `het/637-job-adapter`. Date 2026-10-18, about 18:00Z.

## Setup

| Item | Value |
|---|---|
| Coordinator | DGX Spark (GB10, Linux aarch64), `aien-hetjob` release build from this branch |
| Worker | MacBook Air M2 (macOS 26.6.2, 8 CPUs, 8 GiB), `aien-hetjob` release build from the same source tree, `~/aien-hetjob/bin/aien-hetjob` |
| Worker program digest (claimed, sha256 of the file) | `a57f7c08b47c1f37510cd981e8b30e8c045f2365687e39033c484643f3523077` |
| Worker receipt key | `8d5fe96b...7be2a`, fingerprint `310482aa...af309` ([roster.txt](roster.txt)) |
| Link | SSH over the tailnet; host key pinned (`StrictHostKeyChecking yes`, fingerprint in roster.txt); dedicated key with `command="aien-hetjob serve ...",restrict` in the M2 `authorized_keys`, so the key starts only the worker (asking it for `ls /` started the worker, which exited with no job) |
| Worker limits | defaults: 512 MiB admission memory, 2^34 work units, 120 s wall, 4 threads, RLIMIT_CPU = 120 x 4 + 30 s |
| iMac | NOT_RUN: it runs the AIENOS lab image and has no hosted OS; SSH times out. Its disk was not touched. |

Tests: `cargo test -p aien-hetjob` 16/16 on the Spark (debug) and 16/16 on the M2 (release), including the frozen golden digests, so both architectures give the same bits for the same job.

## Runs (files in this folder)

| File | Job | Outcome | Worker compute (claimed) | Spark, same algorithm | Round trip |
|---|---|---|---|---|---|
| 01 | matmul_u32 512x512x512, seed 637 | VERIFIED, oracle matched | 9.034 ms | 8.538 ms | 144.814 ms |
| 02 / 02b | matmul_u32 1024x1024x1024, seed 2026 | VERIFIED, oracle matched | 87.748 ms | 54.984 ms | 105.025 ms |
| 03 / 03b | digest_chain 10,000,000 rounds | VERIFIED, oracle matched | 1741.902 ms | 1178.227 ms | 1756.295 ms |
| 04 / 04b | matmul 512^3 seed 4242, then the exact retry (same idempotence key, task id, cause id) | VERIFIED both; the retry replayed the stored receipt (same compute time 8.815 ms, nothing recomputed) | 8.815 ms | | 105.778 / 138.666 ms |
| 05 | roster holding a different real key | REJECTED `WRONG_PEER` (the M2's genuine receipt is refused) | | | |
| 06 | matmul 16384x16384x1 | FAILED `OVER_BUDGET:MEMORY need=1073872896 limit=536870912`, refused before allocation | | | |
| 07 | new job body under the idempotence key of 01 | FAILED `DUPLICATE_ID` | | | |
| 08 | digest_chain 2^32-1 rounds, coordinator deadline 2 s | CANCELLED: the coordinator sent `cancel_request` at its deadline, the worker stopped and answered `cancelled` | | | 2004.720 ms |

Round trip is measured from the request write to the final message. SSH session setup overlaps it and is not separately measured.

## What this shows and what it does not

- Shown: one frozen job dispatched from the Spark to a different physical machine, answered in PST1 frames, with a receipt signed by a pinned worker key, and the output digest matching the Spark's own recompute (a different loop order for the matrix product). Wrong peer, over budget, duplicate id, deadline cancel and exact retry behave as specified on the real link.
- Self-attested: the platform, program digest and timings in the receipt are the M2 worker's own claims. The signature proves which pinned key made them, not that they are true.
- Not offload: the Spark recomputes every output to check it, so these runs verify remote execution; they do not save the Spark any work. A check cheaper than recompute (for example Freivalds on a returned matrix) is future work.
- No "useful workload" yet: matmul and the digest chain are deterministic test tasks. The issue's real workload (batch tokenization, model evaluation or a small forward pass) is cut 2.
- Energy: not measured.
