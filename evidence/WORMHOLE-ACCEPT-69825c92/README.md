# WORMHOLE-ACCEPT-69825c92: acceptance receipt (quality only, not timing)

Tool: `wormhole_accept` from sc `wormhole/27b-flashnext@69825c92` (binary.sha256). Host CPU references only
(Flash-Next: the L020 reference on the real NVFP4 checkpoint; 27B: the qwen38 host reference). No chip,
no timing. Wall times in the json files are CPU on a shared box and are not speed claims.

## Method

- Verifier text: the 27B's own greedy output on the chip (exact path, k = 0) for the cascade golden
  prompts, taken from sc-dflash `evidence/qwen38-spec-20261019T0323Z/gen-{code-golden,prose,codelong}.json`
  (commit 891d33c6). `inputs/q38-*.json` = prompt ids + that continuation.
- Drafter: Flash-Next teacher-forced over the same ids (`fn-score`): its greedy choice after every position
  plus the 10 routed experts of each of its 48 layers at every position (the expert trace).
- Acceptance (`accept`): exact greedy block acceptance. While the drafts match, the drafter's context IS the
  verifier's prefix, so its next draft equals its teacher-forced choice there; the round simulation is exact
  for every block size from one pass (unit test `rounds_exact_on_a_hand_case`).
- Hit rate (`hitrate`): LRU expert cache of 2, 4, 8, 16, 32 GB (2,764,824 bytes per expert), prompt
  (prefill) warms the cache, hits counted on the continuation only; 480 lookups per word.

## Results so far

Flash-Next drafting, 27B verifying, code-golden (24 prompt words, 30 continuation words):

| k | rounds | drafted | accepted | accept rate | words per round |
|---|---|---|---|---|---|
| 1 | 15 | 15 | 15 | 1.000 | 2.00 |
| 2 | 11 | 22 | 20 | 0.909 | 2.73 |
| 4 | 7 | 28 | 24 | 0.857 | 4.29 |
| 8 | 4 | 29 | 27 | 0.931 | 7.50 |
| 16 | 2 | 30 | 29 | 0.967 | 15.00 |

Flash-Next's own top-1 equals the 27B's word at 29 of 30 positions.

Expert cache hit rate on that trace (14,400 lookups, cold start, short trace):

| cache | experts | share of 24,576 | LRU hit |
|---|---|---|---|
| 2 GB | 723 | 2.9% | 0.375 |
| 4 GB | 1,446 | 5.9% | 0.597 |
| 8 GB | 2,893 | 11.8% | 0.721 |
| 16 GB | 5,786 | 23.5% | 0.802 |
| 32 GB | 11,573 | 47.1% | 0.805 |

The 16 and 32 GB rows are the same because 30 words are too few: the misses left are first uses.

Pending (queued behind GPU measurements in quiet-build since 17:05 CDT): prose (256 words) and codelong
(256 words) for this direction; the reversed pair (Flash-Next greedy, then 27B teacher-forced on it); the
27B host reference on the chip text (to check the host reference agrees with the chip). DFlash drafting for
the 27B: no acceptance data exists yet (L019's chip runs were refused for memory); not run here, by
agreement with L019.

Weight bytes per class: `bytes-fn.txt`, `bytes-27b.txt` (from `bytes.py`, safetensors headers only).
