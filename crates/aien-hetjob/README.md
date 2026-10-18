# aien-hetjob

Runs allowlisted CPU jobs on another machine and checks the answer. The Spark (coordinator) sends a job to a hosted worker (today: an M2 MacBook over SSH); the worker runs it, signs a receipt with its own pinned key and stores it; the Spark verifies the receipt and compares the result with its own recompute. Issue: aien-sovereign-core#637, parent aienix#6.

Status 2026-10-18: cut 1 and cut 2 shown on real hardware from the Spark to the M2. Cut 1 (test jobs, every refusal): [evidence/het637-m2-20261018/RESULTS.md](evidence/het637-m2-20261018/RESULTS.md). Cut 2 (a useful job: our BitNet tokenizer over our own docs, 40,137 lines, 818,046 tokens, VERIFIED): [evidence/het637-m2-tokenize-20261018/RESULTS.md](evidence/het637-m2-tokenize-20261018/RESULTS.md). The iMac is NOT_RUN (no hosted OS).

## What a worker will run

Only these kinds, integer-exact so every architecture gives the same bits (frozen digests in `tests/protocol_tests.rs` and `tests/tokenize_tests.rs`):

| Kind | Job | Admission |
|---|---|---|
| `matmul_u32` | C = A x B over wrapping u32, A and B generated from a seed; result = sha256 of C | memory 4(mk + kn + mn) bytes and m*k*n multiply-adds against the limits |
| `digest_chain` | sha256 applied N times to a 32-byte seed | N against the work limit |
| `tokenize_batch` | every line of a corpus through our own BitNet BPE (`aien-bitnet-tokenizer`); result = sha256 over each line's ids. Both files are named by sha256 and read from the worker's blob store | each file against 64 MiB, then 16 bytes per tokenizer.json byte plus 8 per corpus byte against memory (peak RSS measured at about half of that, see the cut 2 evidence), corpus bytes against work |

Anything else is refused (`DENIED_TASK`). There is no remote shell and no command line from the peer. Input files travel by digest only: the request carries the sha256 of each file, a file the worker does not hold is `MISSING_INPUT`, one whose bytes do not match its name is `INPUT_DIGEST_MISMATCH` (both refused before a task starts), and a file that matches but cannot be used (a tokenizer.json our BPE refuses, a corpus that is not UTF-8) ends the task as `INVALID_INPUT`.

## Wire

PST1 (aien-protocols `specs/pi-spark-task`, crate `aien-pi-spark-task`), one message per frame, each frame a u16 length plus at most 1024 bytes. The Spark sends `request` and `cancel_request`; the worker answers `accepted`, `running`, the receipt as `text_chunk` messages, `receipt_reference` (sha256 of the receipt), then `completed` (the output digest), or `failed` / `cancelled`. PST1 Draft v0.1 calls the requester side "Pi"; here the Spark is the requester, and every worker status is a hint the Spark checks.

## Trust and limits, stated plainly

- Receipt: self-attested. The signature says which pinned key made the claim; the platform, program digest and timings are the worker's claims. The Spark trusts its own recompute.
- Transport: SSH with a pinned host key, and a dedicated key whose `authorized_keys` line is `command="aien-hetjob serve --state ...",restrict`, so the key can only start the worker.
- Ceilings: memory and work are admission arithmetic checked before allocation, not a kernel limit. The OS enforces only CPU seconds (`RLIMIT_CPU`) and no core dumps. On macOS there is no seccomp, no namespaces and no reliable address-space limit.
- Durable results: a finished receipt is written, synced and renamed before `completed` is sent. A retry with the same idempotence key, task id and cause id gets the stored receipt; a different request under the same key is refused (`DUPLICATE_ID`). One worker per state folder (`flock`).
- The Spark owns the deadline: it sends `cancel_request`, waits a grace period, then reports `UNCERTAIN` and closes the connection. A dropped link is `UNCERTAIN`, never `FAILED`.

## Use

```
# on the worker
aien-hetjob keygen --state ~/aien-hetjob/state      # prints the public key for the Spark's roster
aien-hetjob caps --state ~/aien-hetjob/state        # claimed platform and a local baseline
aien-hetjob put-blob --state ~/aien-hetjob/state tokenizer.json   # store an input file by its sha256 (works while serving)
# on the Spark
aien-hetjob dispatch --roster roster.txt --ssh macbook-hetjob --matmul 512 512 512 637
aien-hetjob local --matmul 512 512 512 637          # same job here, worker algorithm and oracle timed
aien-hetjob dispatch --roster roster.txt --ssh macbook-hetjob --tokenize tokenizer.json corpus.txt --bos
```

Exit 0 means verified; anything else prints `OUTCOME FAILED|CANCELLED|UNCERTAIN|REJECTED <reason>` and exits 1.
