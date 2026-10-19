# Document memory for the 27B door: measurements 2026-10-19

Branch cascade/document-memory. Binary: bin-e07a3b4c (sha256 prefix c1bf428c, see chip-1/run.log). Every file here is a raw
result, failures included. Nothing was rewritten after the fact.

## Gate: resumed answer equals recomputed answer

chip-1/gate.tsv: 10 checks, 10 MATCH (first 32 answer tokens identical, greedy decoding).
Sizes in bytes of document text (token counts in timings.tsv): 16000 (3.5k tokens), 74000 (16k), 148000 (32k).
Each check resumes the saved state either from RAM or from NVMe after the server was made to forget it
(a "displace" request in between), and compares with a cold run that recomputes the whole document.
The two join-experiment rows (join30000-q0/q1-store-exact) are the same gate for the two separately read documents.

## Time to answer a second question (seconds, wall clock, from chip-1/timings.tsv)

| document (tokens) | from RAM | from NVMe | recompute from scratch |
|---|---|---|---|
| 3.5k  | 4.76 | 7.77 | 23.5 / 24.8 |
| 16k   | 5.77 | 6.86 | 105.2 / 87.1 |
| 32k   | 17.6 (first, see note), 6.25 | 7.68, 9.51 | 182.6, 188.5, 293.9, 266.9 |

Note: all times were taken while other measurement jobs shared the machine (load average 5 to 90 at the time, listed per row
in timings.tsv), and the 32k recompute times vary by 1.6x for that reason. The 17.6 s RAM row is the first 32k question after
the server was started and is slower than the three later ones; cause not investigated. These are not quiet-lock
measurements: the chip run was held up by memory for hours and then ran as one chain. Treat as measured under load,
not as the best the hardware can do.

## Disk round trips without the model (bench/)

NVMe O_DIRECT write and read back of the saved state, 0 mismatched words each:
4096 tokens 0.425 GB, 16384 tokens 1.70 GB, 32768 tokens 3.40 GB.
Load 0.87, 0.64, 1.95 GB/s; spill 0.69, 0.38, 0.67 GB/s. Receipt wall time is mostly queue wait for the quiet lock.
The GPU copy benchmark (KV rows device to host and back) FAILED at both 4096 and 32768 tokens: attn256 channel alloc status 0x51 (no memory; GPU memory is shared with other servers). Raw output in bench/kv.out and the two kv json files (exit 1). No GPU-copy number exists yet; the RAM resume times above include it end to end.

## Approximate join of two separately read documents (chip-1/join-*)

Document A is stored exactly and document B has its keys and values moved to follow A on the host. The DeltaNet state
is taken from B (last), from A (first) or the sum of both. Compared with cold recompute of A+B, first 32 answer tokens:

| question | variant | leading tokens equal | tokens equal at same position | of |
|---|---|---|---|---|
| q0 | last  | 13 | 13 | 32 |
| q0 | first | 5  | 12 | 32 |
| q0 | sum   | 2  | 2  | 32 |
| q1 | last / first / sum | 32 | 32 | 32 |

Reading: on q1 all three variants gave the identical answer to the recompute. On q0 none did, the best (last) held for
13 of 32 tokens. Two questions on one document pair is a small sample; I did not check why q1 is easier (it may not
depend on the second document). The join is approximate and is not used unless the caller asks for it.
Times in these rows range 15 to 122 s and are dominated by load from other jobs, not by the join.
