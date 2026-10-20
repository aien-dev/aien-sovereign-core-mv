# SPEED-5 batching (L129) evidence

What this is: the Qwen3.8-27B FP8 server answering up to 32 requests at once, sharing one read of the weights per
step, measured at N = 1, 2, 4, 8, 16, 32 streams in one spark-window (20261020T045957Z-8afb15), with the energy
receipt of every call.

Code: sovereign-core branch speed5/batching. Omega branch speed5/handle-slots (c79e2fe1 handle slots 1024 -> 4096,
66e84cf6 tree GEMM tile table rewrite) built as lib-tvtab (sha256 59ff304a...), server bin-946f6fed-tvtab
(sha256 3de4eba2...). Scripts (shell only): window.sh, session.sh, bench.sh, summarize.sh, compare.sh.

Fixed settings for every result: 32 batch slots of 1024 positions (C: 8 slots), 32 prompts in prompts.txt, at most
256 new tokens per stream, greedy, timing taken over the whole answer (positions from prompt end to 256).
Box state during the window: load average 27 to 48, the 27B server on :18091 resident, other agents running.

## Results (session B: tree GEMM, FP8 head, one weight read for all rows)

Wall fraction = (weight bytes 25.62 GB + 0.30 GB DeltaNet state per row) / (ms per step x 273 GB/s).
J/token = whole-UPS wall joules (wall_J in each <call>-a1.receipt.json) of the call / tokens generated. The UPS measures the whole box (27B server and the
other agents included), so J/token is box-wide: CONTAMINATED as a Spark-alone number. incremental_J is not usable
here (idle baseline taken under a moving load; several came out negative).

| N | agg tok/s | per-stream tok/s | ms/step | memory-wall fraction | J/token (box) |
|---|---|---|---|---|---|
| 1 | 3.598 | 3.750 | 265.4 | 0.358 (calc:420e895b) | 75.02 |
| 2 | 5.413 | 3.397 | 270.9 | 0.355 (calc:42c813e1) | 55.42 |
| 4 | 9.150 | 3.180 | 290.8 | 0.338 (calc:15eb4bd5) | 30.23 |
| 8 | 13.509 | 1.993 | 467.2 | 0.220 (calc:90f76c3f) | 19.41 |
| 16 | 18.747 | 1.346 | 669.4 | 0.167 (calc:d6b2e78e) | 13.97 |
| 32 | 21.243 | 0.723 | 1252.4 | 0.103 (calc:9c29f413) | 12.17 |
| 1 again (drift control) | 4.481 | 4.717 | 210.8 | UNMEASURED fraction | 60.75 |

- 32 streams give 5.90x the aggregate tokens of 1 stream in the same server (calc:5517488d), and 2.87x the one-at-a-time
  door (calc:9bc9ce0f, D below). Energy per token falls from 37.27 J (one-at-a-time door, calc:35d0b157) to 12.17 J (calc:5c7d71ee) at 32 streams,
  3.06x less (calc:7cbf80b3), box-wide.
- Arithmetic wall at 32 rows: 0.0052 of 207.7 TFLOPS FP8 (calc:20965ff0). Memory 128 GB: 33,050 MiB model + slots.
- The drift control (N=1 first and last) differs by 21 percent (265 vs 211 ms/step, calc:726a6a5b) under the same settings: box load
  moves the host-driven batch path that much. Read every row above with that spread.

Budget per step at N = 32 (ms): GEMM wall 659.8, of which the GPU side is 265.1; the other 395 ms (calc:e11d68dc) are the host wrapper
of each GEMM call (permute, upload, download). DeltaNet step 173.9, attention 87.5, a|b GEMV 49.0, head 28.5,
host glue 231.2. The weight read is no longer the binding term; per-call host work is.

## Other sessions

- C, row-pair GEMM (the first batch cut, each extra row costs about another weight read), bf16 head:
  N=1 2.476 tok/s (393.4 ms/step, 107.81 J/token); N=8 4.069 tok/s aggregate (1613 ms/step, 66.05 J/token).
  The tree GEMM (B) is 3.32x C at N=8 (calc:967e7235).
- D, one-at-a-time door (decode graph, one command list per token): 7.392 and 7.565 tok/s, 136 and 133 ms/token,
  37.27 and 48.09 J/token. It is faster than the batch door at N=1 because the batch door issues about a thousand
  host calls per step; the decode graph issues one.
- E, as B with the SiLU on 8 host threads: N=8 4.591 tok/s, a loss (threads fight the loaded box). n32 stopped by me
  to leave window time for A: not a result.
- A, tree GEMM with the bf16 head: the correctness pass, see "Correctness".

## Correctness

Token ids of every answer, compared with compare.sh (one line per stream in each <session>/compare*.txt). The
reference for prompts 0 to 3 is the prompt answered alone (n1-p0..p2 from partial-A-0533Z-nice10, same binary and
settings; n1-p3 from A); for prompts 4 to 31 it is that prompt's stream in A n32. Negative control: every stream is
also compared with the next prompt's reference and must differ; it differed every time.

- A (tree GEMM, bf16 head), the gate: 79 of 79 streams SAME (A-tree-bf16head/compare.txt). Of these, 28 are A n32
  against itself (prompts 4 to 31), so 51 are real checks: prompts 0 to 3 at N = 2, 4, 8, 16, 32 against the stream
  alone, prompts 4 to 15 at N = 8 and 16, and prompts 16 to 31 run as a second group of 16 (n16s), all against N = 32.
  A row's answer does not change with how many other rows share the step.
- B (FP8 head) against A: 46 of 64 streams differ; 24 of the 32 prompts change at some position (first difference at
  generated position 1 to 251). This is the FP8 head rounding near-tied scores differently, not batching: B against
  itself (B n1 for prompt 0, B n32 for the rest) is 64 of 64 SAME (B-tree-fp8head/compare-self.txt). The FP8 head is
  a speed option with a word-drift cost, not the reference.
- E (8 host threads) against B: 8 of 8 SAME. Host threads do not change bits.
- C (row-pair GEMM, the other GEMM kernel) against A: 3 of 9 differ (prompts 3, 5, 7). D (one-at-a-time door) against
  A: prompt 0 SAME, prompt 3 differs at position 43, the same position C differs at. Different GEMM kernels round
  differently; reported as counts, not as a gate.

A timing (A-tree-bf16head/*.summary) is not comparable with B: box load was 45 to 74 during A against 30 to 37 during
B, and the host-driven path slows with load (L159). One thing A does show: the bf16 head costs 491 ms per step at
N = 32 against 28.5 ms for the FP8 head, the largest single term after the GEMMs. A n16s ran 56 s past the lease end
(07:30:01Z): its timing is NOT A RESULT, its bits are. A n1-last was not run (lease over).

## Not results

Every not-a-result-* and partial-* folder has a WHY.txt. Causes found and fixed on the way: the server waited on the
GPU lock its own window held (AIEN_GB10_LOCK_HELD, 946f6fed); handle table full at 32 slots (omega c79e2fe1);
32 x 4096 positions did not fit next to the resident servers (CTX 1024); "tv gemm: tile table full" at 8 rows and up
(omega 66e84cf6); one load refused by the driver at channel create (NV_ERR_NO_MEMORY, transient with large servers
resident).

## Next levers (in order of size, from the budget above)

1. Record the whole multi-row step into one command list, as the one-at-a-time door does: removes most of the
   395 ms GEMM host wrapper and the 231 ms host glue at N=32.
2. Keep activations on the device between layers (no permute, upload, download per GEMM call).
3. Per-row DeltaNet and attention as one launch for all rows (174 + 88 ms at N=32).

## Side findings (handed on)

- L159 (fixer): the tree GEMM runs about 3.5x slower inside the server than alone, and the gap tracks box load; the
  decode graph is not affected. Probes: probe-tv-nice, probe-tvtab. L159 found the cause on the host: each GEMM call
  waited for the chip by spinning a CPU core, and under box load that thread was taken off the CPU. Fix pushed on
  sovereign-core fix/tv-inserver 3208be4b (sleep-poll wait); chip check pending in its own window. The B numbers above
  are before that fix.
- gpu_tcgemm_test xf32 form errors at N 17 to 32 on q38o and q38down shapes (pre-existing, not on the tv_f32 path).

## Log
- 2026-10-20T03:20Z host tests (CPU stub; aien-serve-http, aien-inference-abi, aien-omega-fp8) at 1636f455: 448 passed,
  0 failed, 22 ignored (host-tests-1636f455.txt). 7c3b0381 adds par_rows + its test (passed alone).
- 2026-10-20T05:00Z to 07:30Z window 20261020T045957Z-8afb15: B, C, D, E in window.sh; A rerun after a refused load.
- 2026-10-20T06:47Z to 07:31Z session A rerun in the same window (after-window.sh). The lease ended 07:30:01Z, 56 s
  before A finished. Then after-window.sh ran `spark-window close` without the window id and closed the next window,
  L159's (opened 07:30:47Z, its chip run cut short). L159's script then closed L125's queued window the same way.
  Told L159 and L125 on the ledger; the tool hazard (close with no id closes whoever's window is current) is on L113,
  already reported there by L125. My scripts pass the id from now on.
- 2026-10-20T07:40Z compare.sh over A, B, C, D, E (compare*.txt in each folder).
