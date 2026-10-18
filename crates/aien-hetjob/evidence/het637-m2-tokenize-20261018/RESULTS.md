# het637-m2-tokenize-20261018: a useful job, Spark coordinator to M2 worker

Issue: aien-sovereign-core#637 cut 2 (parent aienix#6). Branch `feat/637-het-tokenize`, commit `b5ab184f` (the M2 worker was built from `git archive` of that commit). Date 2026-10-18, about 18:40Z.

## The job

`tokenize_batch`: tokenize every line of a corpus with our own BitNet b1.58 2B-4T BPE (`crates/aien-bitnet-tokenizer`, no tokenizer library). The request names both input files by sha256; the worker reads them from its blob store and checks each digest before use.

| Input | sha256 | Size |
|---|---|---|
| `tokenizer.json` (BitNet b1.58 2B-4T, the digest pinned as `BITNET_2B4T_TOKENIZER_SHA256`) | `e134af98b985517b4f068e3755ae90d4e9cd2d45d328325dc503f1c6b2d06cc7` | 9,085,698 bytes |
| corpus: every `*.md` in aien-architecture at `6142a352d3e6fe324ed757a44519e9b0142aa48d` (196 files, C-locale sorted paths, blobs concatenated) | `683ca95b94d1b5b8e214e22e50ecb9ceb0a889b3be2a28f5d8b794c90d663273` | 3,065,803 bytes, 40,137 lines |

Rebuild the corpus: `git ls-tree -r --name-only 6142a352 | grep '\.md$' | LC_ALL=C sort`, then `git cat-file blob 6142a352:<path>` for each, appended in that order.

Worker setup, roster and link: as in [../het637-m2-20261018/RESULTS.md](../het637-m2-20261018/RESULTS.md) (same pinned worker key, fingerprint `310482aa...af309`, same forced-command SSH key). Worker program digest (claimed): `a7406da86c6f107714d282564348e36912c791efaade0d59621ca238347404ce`.

Tests: `cargo test -p aien-hetjob` 23/23 on the Spark (debug) and 23/23 on the M2 (release), including the frozen tokenize digests, so both machines produce the same token ids.

## Runs (files in this folder)

| File | Job | Outcome | Worker compute (claimed) | Round trip |
|---|---|---|---|---|
| 09 | tokenize, BOS on | VERIFIED, oracle matched; 40,137 lines, 818,046 tokens, every line decodes back to itself | 215.370 ms | 330.033 ms |
| 10 | exact retry of 09 (same idempotence key, task id, cause id) | VERIFIED; the stored receipt was replayed (same compute time, nothing recomputed) | 215.370 ms | |
| 11 | a corpus the worker does not hold (09's corpus plus one line) | FAILED `MISSING_INPUT:0b443873...`, no task created | | |
| 12 | tokenize, BOS off | VERIFIED, oracle matched; 777,909 tokens (40,137 fewer: one BOS per line) | 232.432 ms | 339.001 ms |
| 13 | a 100-line corpus whose stored file was changed after `put-blob` | FAILED `INPUT_DIGEST_MISMATCH:3f6bdaa2...`, no task created (the tampered file was removed afterwards) | | |

The Spark running the same worker code locally (`aien-hetjob local --tokenize ... --bos --threads 4`, three runs, load average 18 from pipeline runs): 380.627, 289.224, 286.643 ms. The M2's load average was about 2.9. Neither number is a clean benchmark; both include parsing the 9 MB tokenizer.json, which is single threaded.

Peak memory (Spark, `/usr/bin/time -v`, one process running the worker algorithm and the recompute): 90,964 KiB with 4 threads, 90,336 KiB with 1. The admission estimate for the same inputs is 169,897,592 bytes (16 bytes per tokenizer.json byte plus 8 per corpus byte), so admission is conservative by about 1.9x here.

## What this shows and what it does not

- Shown: a real workload (our tokenizer on our own documentation) dispatched from the Spark to a different physical machine, answered with a receipt signed by the pinned worker key, and matching the Spark's recompute bit for bit. Inputs travel by digest only: the 66-byte request carries no data, and a missing or altered file is refused before any task starts.
- Self-attested: platform, program digest and timings are the M2's claims.
- Not offload yet: the Spark recomputes every output to check it. The recompute also decodes every line back to its text, a second code path; the encoder itself is the same code on both sides, so its independence is a different machine and a different build only.
- Provisioning: the input files reached the M2 by `scp` with the operator's normal SSH key, then `put-blob`. The forced-command job key cannot copy files.
- Energy: not measured. iMac: NOT_RUN (no hosted OS).
