# SPEED-2 speculative decode measurements (ledger L126), 2026-10-20

Tool: qwen38_spec (sc speed2/specdecode) against omega speed2/specdecode; exact binary and library hashes in command-*.txt.
Model: Qwen3.8-27B FP8 (~/models/qwen3.8-27b-fp8-017b9c7a). All runs inside a spark-window (receipt-*.json carry the window id and wall joules).

- plan-graph.txt (env AIEN_Q38_FP8_HEAD=1): `--verify graph` = the whole verify pass as one decode-graph command list
  (omega rows mode); `--verify tree` = the earlier host-driven pass with the tv GEMM for the dense projections. Each
  kernel's k = 0 is its plain greedy reference (one row per pass, same kernels); PASS = every k > 0 run has the same
  tokens as its k = 0. Mutant runs must FAIL (negative controls). `--graph 1` adds the one-row decode graph with the FP8 head.
- plan-plain.txt (no env, NOT RUN: dropped after hours of window waits; the server check in serve/ measures the served
  plain decode instead): the served configuration (FP8 GEMV, bf16 head) one token at a time, `--graph 1`: the plain
  tok/s these numbers are compared with.

## Results: plan-graph (window 20261020T043024Z-0b576d, 2026-10-20 04:30 to 04:56Z, box load 48 to 53)

Binary qwen38_spec at sc e6bd69c9 against omega 4b20a9c5 (hashes in command-graph.txt). Full table: summary-graph.csv
(`./summary.sh 93.88 <files>`). The wall: one pass reads the resident weights once; with the FP8 head that is
93.88 ms at 273 GB/s (calc:b44f97c63277), so the wall at concurrency 1 is words_per_pass / 93.88 ms.

Every real run gives the same tokens as its own k = 0 (identical_to_k0 true); the three mutant runs fail as they
must (ExtraAccept, NoRollback, RowShift). The suite's overall FAIL line counts those three planted faults.

| prompt | path | k | words per pass | ms per token | tok/s | wall tok/s at that words per pass | fraction of wall |
|---|---|---|---|---|---|---|---|
| codelong | one-row decode graph (plain greedy, FP8 head) | - | 1 | 123.47 | 8.10 | 10.65 | 0.760 |
| codelong | Graph | 0 | 1 | 163.72 | 6.11 | 10.65 | 0.573 |
| codelong | Graph | 4 | 3.148 | 104.78 | 9.54 | 33.53 | 0.285 |
| codelong | Graph | 8 | 3.657 | 133.66 | 7.48 | 38.96 | 0.192 |
| prose | one-row decode graph (plain greedy, FP8 head) | - | 1 | 119.47 | 8.37 | 10.65 | 0.786 |
| prose | Graph | 4 | 2.14 | 163.95 | 6.10 | 22.80 | 0.268 |
| prose | Graph | 8 | 2.185 | 232.29 | 4.30 | 23.27 | 0.185 |
| codelong | TreeGemm (host-driven) | 4 | 3.338 | 324.96 | 3.08 | 35.55 | 0.087 |
| codelong | TreeGemm (host-driven) | 8 | 4.015 | 406.30 | 2.46 | 42.77 | 0.058 |
| prose | TreeGemm (host-driven) | 8 | 2.133 | 831.80 | 1.20 | 22.72 | 0.053 |

The gate's own command (`--verify tree`, codelong and prose, n 256, k 0/4/8): tokens identical to k 0 on both prompts;
words per pass 3.34 / 4.02 (codelong k4 / k8) and 2.12 / 2.13 (prose); wall tok/s 3.08 / 2.46 and 1.77 / 1.20 against
plain 8.10 / 8.37. The host-driven tree path is slower than plain. The rows-mode path (`--verify graph`) is the one
that wins: codelong k4 is 1.18 times plain (calc:912590acfd01; 9.54 against 8.10 tok/s), prose does not win (2.1 words per pass).

Where a k4 codelong round goes (graph-codelong.json, 81 rounds): verify pass on the chip 204 ms for 5 rows, MTP drafts
53 ms (each draft's FP8 head call about 11 ms: kernel plus host round trip), rollback 4 ms. A one-row pass in rows mode
costs 163.5 ms of chip time against 123.2 ms for the one-row decode graph: rows mode spends about 40 ms per pass more
than the fused one-row graph (separate per-row glue launches, the GDN output copies, chunked submits under the
2048-launch cap). Next levers, in order: (1) rows mode glue as fused as the one-row graph, (2) the MTP draft chain on
the chip in the same command list as the pass (no host round trip per draft), (3) a smaller draft head.

Energy: the receipt's whole-box wall energy is 391,693 J over 1,454.9 s (mean 269.2 W, calc:5617bb671aed) with other jobs running
(load 48 to 53, idle reading before the window 310 W), so the joules of these runs alone are UNMEASURED. Box wall
power times time per token gives 28.2 J/token (calc:b1017f1709cb) for codelong k4 against 33.2 J/token (calc:c0e8818937c0) for the one-row graph, both
including the other jobs' share.

Note on "plain": the one-row decode graph (FP8 GEMV) and the rows mode (tv GEMM) are different kernels, so their
greedy texts can part at a near tie: prose agreed for the first 209 of 256 tokens, codelong for all 256. The
exactness claim is per kernel: every k equals k = 0 of the same kernels bit for bit.

## Results: server check (serve/, window 20261020T104028Z-12a88e, 10:40 to 10:56Z, load 33 to 42)

aien-serve-http at sc e6bd69c9 against omega 4b20a9c5 (sha in serve/command.txt), 127.0.0.1:18126, context 4096. The
window was opened with `--gb10 none` because this server takes the chip lock itself and does not read
AIEN_GB10_LOCK_HELD (that fix is on another branch, L068); so other jobs' chip work could interleave between requests.
Each configuration: one server start, the same greedy chat request (the codelong prompt, max_tokens 256,
temperature 0) sent twice; the second request reuses the snapshot (85 of 90 prompt tokens).

All eight replies have the same text (serve/content-sha.txt: one sha 01ba4719c80b6fe9 for plain FP8 head, spec k0,
k4, k8, both requests each): the speculative server reply is the plain greedy reply, word for word.

| configuration | words per pass | wall ms/token (request 1, 2) | fraction of wall |
|---|---|---|---|
| plain, FP8 head (AIEN_Q38_FP8_HEAD=1) | 1 | 124.7, 121.9 | 0.753 (calc:5e7a15a49ac0) |
| spec k0, FP8 head | 1 | 165.9, 167.0 | |
| spec k4, FP8 head | 2.52 | 111.4, 113.2 | 0.334 (calc:f3b17ae942b3) |
| spec k8, FP8 head | 2.63 | 157.5, 152.9 | |
| plain, bf16 head (served default) | | NOT A RESULT: the upload was refused by the driver (channel create status 0x51, box crowded), serve/serve-plain.refused.log | |

Spec k4 is 1.10 times plain on this request (calc:c87ba9ee9c54). It accepts fewer drafts than the tool's codelong
run (2.52 against 3.15 words per pass): the server feeds the prompt through the prefill kernels and renders its own
chat prompt, so the greedy text differs from the tool's. J/token: UNMEASURED for the server alone (other jobs ran;
receipts in serve/receipt-*.json give whole-box wall power 261 to 274 W).
