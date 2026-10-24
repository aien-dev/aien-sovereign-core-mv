# Q36 decode ceiling: the 35B against the one bus

Lane L310 ("35B decode to the memory ceiling"), 2026-10-24. Model `qwen3.6-35b-a3b-fp8` as served on the Spark
(live build `bin-19173f9e` = sovereign-core `ship/motor-l387` f0230a59 + omega `ship/motor-l387` 6de999da).
No chip run was needed for this document: it is a byte count, the published bus speed, and the live server's own log.

Every number below went through the calculator (`calc:` receipt, full ids in section 9) and, as the independent check,
through Wolfram (`evidence/q36-ceiling-20261024/wolfram/INDEX.md`, raw request and reply per row). The two agree on
every row. Rule: Drake 2026-10-24, "Mathematics goes through Wolfram and the calculator".

## 1. What the ceiling is

The Spark has one memory pool, 128 GB LPDDR5X at 273 GB/s (GB10 whitepaper Table 1; `nvidia-smi` on this box).
The memory controllers sit on the CPU die, so every byte the GPU reads crosses the chip-to-chip link; that link
(345 GB/s per direction) is faster than the memory, so the memory is the ceiling (spark-hardware-manual section 1).

Decoding one token reads every active weight byte once. Nothing in the chip can finish a token sooner than
`active bytes / 273 GB/s`. That number is the ceiling. The 24 MiB GPU L2 cannot hold the 2.9 GB a token touches,
so there is no cache trick that lifts it; only fewer bytes per token (W4, drafts, shared reads across rows) or a
faster bus (not ours) move it.

## 2. Bytes one token must read (1 row, live build)

From the safetensors headers of the checkpoint (layer total calc `d4f864ca`), the head from the exact two-pass head
that is ON in the live build (L314). Per layer: GDN block 33,886,592 B (30 layers), full attention 27,267,328 B
(10 layers), one routed expert 3,146,112 B (8 active of 256 per layer, 40 layers), shared expert 3,150,208 B (40 layers).

| part | bytes | receipt |
|---|---:|---|
| 30 GDN blocks | 1,016,597,760 | calc `8d19dfc2` (as 3.724 ms), Wolfram `floor_gdn_ms` |
| 10 attention blocks | 272,673,280 | calc `857a5be3`, Wolfram `floor_attn_ms` |
| 40 x 8 routed experts | 1,006,755,840 | calc `ccf635f1`, Wolfram `floor_exp_ms` (with shared) |
| 40 shared experts | 126,008,320 | same |
| routers, norms, embeddings (remainder of the layer total) | 41,585,920 | calc `a85d621f`, Wolfram `routers_rem` |
| **layers total** | **2,463,621,120** | calc `d4f864ca` |
| head: fp8b screen 248,320 x 2,048 | 508,559,360 | calc `4655f814` (as 1.863 ms), Wolfram `floor_head_ms` |
| head: BF16 recheck of ~31 candidate rows x 2,048 x 2 | 126,976 | calc `79b72206` (head total 508,686,336), Wolfram `bytes1` |
| **one token, 1 row** | **2,972,307,456** | calc `aa09ce73` (= `7d82935b`), Wolfram `bytes1` |

The 31 candidate rows is the measured p50 from the L314 receipt (hand-off 2026-10-20-hum-exact-head); the recheck is
0.004 % of the token and does not move any row below. Earlier figures (3.48 GB in the 10-24 survey part 1, 3.537 GB in
the coordinator's 10-20 estimate) counted a full BF16 head (1.02 GB); the live build does not read one.

## 3. Ceiling table

Bus 273 GB/s. "Measured" is the live server's own `chip N ms/step` line (section 4). Fraction = ceiling / measured.

| rows per step | bytes per step | ceiling | ceiling tok/s | measured chip | measured tok/s | fraction of bus | receipts (calc / Wolfram) |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 2,972,307,456 | 10.888 ms | 91.85 | 20.7 ms | 48.31 | **52.6 %** | `882844dd` `5d519513` `51752d62` `f6e3c4ed` / `ceil1_ms` `ceil1_tps` `meas1_tps` `frac1` |
| 8 (expected expert overlap) | 9,192,535,039 | 33.672 ms | 237.6 | 59.2 ms | 135.1 | **56.9 %** | `e3398026` `99d9b471` `726e7afa` `c1524bc6` `fbfcba7f` / `bytes8` `ceil8_ms` `ceil8_tps` `meas8_chip_tps` `frac8_chip` |
| 8 (no overlap, worst case) | 10,020,487,168 | 36.705 ms | 218.0 | 59.2 ms | 135.1 | 62.0 % | `119cf0b1` `3af4c46a` / `bytes8_worst` `ceil8_worst_ms` |

Reading the 1-row line: the bus could deliver 91.8 words a second; the chip delivers 48.3; the live step moves data at
143.6 GB/s (calc `9e25148c`, Wolfram `meas1_gbps`), 52.6 % of the bus.

8-row model. Weights that every row shares (GDN, attention, shared experts, routers, the head screen) are read once per
step: 1,456,865,280 + 508,559,360 B (calc `b886b3e7`, Wolfram `base8`). Routed experts are read once per distinct
expert the eight rows pick. With independent uniform picks the expected distinct count is
256 x (1 - (31/32)^8) = 57.42 of 64 slots per layer (calc `ffe2bed5`, Wolfram `uniq8`). Real routing is correlated
(prompts in one spell share topics), so the true byte count sits between the "expected" and "worst" lines; where in
between is UNKNOWN until counted from the router output. The 8-row fraction with the earlier round 60 ms figure is
56.1 % (calc `3d15ca8e`, Wolfram `frac8`).

## 4. Measured (live server log, quoted)

`~/.local/state/aien-q36/serve.log`, live pid 3489680 (environment `AIEN_Q36_STROKE=1` only):

- line 4048: `qwen36 omega: exact two-pass head ON (fp8b screen + Row bound, built in 7.8 s)`
- line 11597 (1 row): `spell over: 97 beats, 101 steps, 122 row tokens (95 decoded), 20.8 ms/step wall, chip 20.7 ms/step`
- line 11596 (same spell): `decode 96 tok in 1890 ms over 95 beats, mean rows 1.00, wall 19.9 ms/token`
- line 11558 (8 rows): `spell over: 64 beats, 70 steps, 560 row tokens (504 decoded), 59.4 ms/step wall, chip 59.2 ms/step`
- line 11552 (same spell): `decode 64 tok in 3762 ms over 63 beats, mean rows 8.00, wall 59.7 ms/token`

Wall and chip differ by 0.1 to 0.2 ms per step: the host side is not where the time goes (L356 found the same, 0.2 %).

## 5. Where the other 9.8 ms goes: UNKNOWN

At 1 row the step takes 20.7 ms and the bytes need 10.888 ms. The gap is 9.812 ms (calc `6209daa6`, Wolfram `gap1_ms`).
What is known about its composition:

- The per-launch floor is small. L312's stroke stamps (hand-off 2026-10-20-engine, `stroke8l2-stamps`) give 30 us per GDN
  launch and 66 us per attention launch, launch plus drain. A token at 1 row runs ~414 launches (L356); the
  GDN + attention heads alone are 30 x 30 + 10 x 66 us = 1.56 ms (calc `190a4565`, Wolfram `launch_floor_1row_us`);
  the 8-row stroke's 240 + 80 head launches are 12.48 ms (calc `0cc43bec`, Wolfram `launch_floor_8row_us`).
  Stamps measure launch + wait-for-idle, so they rank launches; they do not add up to wall time (L356 advisor note).
- Compute is not the limit: the roofline ridge on this box is 96 TFLOPS / 273 GB/s = 351.6 FLOP per byte
  (calc `a7b847e5`, Wolfram `ridge`); decode at 1 row does about 2 FLOP per weight byte.
- The expert reads are 8 small matrices per layer (3.1 MB each) chosen at run time: 320 dependent gathers per token.
  Whether they stream at bus speed or at a fraction of it is UNKNOWN; this is the first thing the timeline must show.
- Cached interior (GPU-L2-cached chip-only intermediates) is already ON (section 6), so it is not a pending lever.

Composition of the 9.8 ms: UNKNOWN. Next measurement (section 8): a 1-row per-launch timeline on the live build.

## 6. Two confirmations (code and live process, no chip run)

1. **Cached interior is ON.** omega 6de999da `src/omega_gpu_q36_graph.c` lines 4-5 and 36-40: `buf` (host-touched
   boundary: x in, token and logits out, tables) is allocated GPU-uncached; `dbuf` (chip-only intermediates) is
   allocated GPU-cached via `omega_gpu_session_alloc_cached` unless `AIEN_OMEGA_Q36_BUF_CACHED=0` (line 208; the prefill
   buffers follow the same rule at line 908). The live process environment carries no such switch. Measured on
   2026-10-19 (`omega-q38-speed-work/runs/probe-buf`): cached, down 2100 to 1735 us and mix 1596 to 1458 us over 4
   layers, `gpu_q38_graph_test: 38 checks, 0 failed: PASS` both ways. This matches the manual: host-polled words
   uncached (section 4 rule 1), bulk chip-only data may stay cached (rule 3). Verified independently by L803 (ledger L310).
2. **Exact two-pass head is ON** (serve.log line 4048 above; L314 receipt: head 4774.1 to 2340.5 us p50 at 213-222 GB/s,
   1010/1010 tokens identical). The "FP8 lm_head" rung of this lane is therefore taken: the screen already reads the
   head in fp8b, and the dense-27B `AIEN_Q38_FP8_HEAD` path is a different graph.

## 7. Levers ledger (bytes or time per token)

| lever | moves | owner / state |
|---|---|---|
| exact two-pass head (fp8b screen) | head 1.02 GB to 0.51 GB per token | L314, ON in live |
| W4 routed experts | experts 1.01 GB to ~0.57 GB per row | L355: 10-21 attempt FAILED identity; store lever, redo behind the accuracy rule |
| CN rows (one glue launch per layer) | launches | L356, ON in live |
| strokes v2 (rows on one graph) | shares the 1.97 GB of common weights across 8 rows | L312 lever 2 ON; levers 3-7 (prefetch, GDN/attention heads form, head once per 8) gate not green |
| glue fusion / persistent tile | dependent launches | omega #603, #604 (open) |
| multi-token draft (MTP) | words per step | survey estimate ~6 % at 8 rows (calc `521952a7`, survey-era, not Wolfram-checked); L355 |
| cached interior whole-step A/B | time, not bytes | this lane: merge-train arm `interior-uncached` (section 8) |
| Spark-shaped model (2.3 % active, trained-in 4-bit, hierarchical head) | bytes per token 2.97 GB to 2.09 GB | aien-train #21, design only |

## 8. Method and next steps (this lane)

1. This document (no chip).
2. Merge-train arm `interior-uncached`: `server_env AIEN_OMEGA_Q36_BUF_CACHED=0` with a `log_proof` line, so the
   whole-step cached-vs-uncached A/B rides the next train cycle on identical cases (branch `l310/train-arm-interior-uncached`).
3. One-row per-launch timeline on the live build: existing stamps camera (`omega_gpu_cmdlist_stamps`, `q36_levers`,
   `AIEN_STROKE_STAMPS=1`), baseline only, N >= 5 tokens on the host goldens, one shared chip window of at most 20 min
   requested on the team board, not opened by this lane while CHIP-RESERVED lanes hold the chip. Camera stays outside
   the measured path: no prints or waits added.
4. The lever the timeline names, one variable per run, greedy tokens identical on the goldens or the exact divergence stated.
5. Shape sweep for aien-train #21 only in gaps.

Rules kept: no number from a model's head (calculator + Wolfram), no estimate without a receipt, one variable per run,
never open a chip window over a CHIP-RESERVED lane, read the manual before any chip edit.

## 9. Receipts (full calculator ids)

| short | full id | value |
|---|---|---|
| d4f864ca | calc:d4f864ca44105aa4bf29af1d178d9f169e2ca21ca634db21bb8e4f54d2c97124 | layers 2,463,621,120 B |
| 79b72206 | calc:79b722061b0b14def42d8cb84b93eb2e09b55d63afaf2e202cee6083985b6329 | head 508,686,336 B |
| aa09ce73 | calc:aa09ce73195cbf8082e306ba59d673bbabb88e5e024c569fa3fdf287b5c799eb | one token 2,972,307,456 B |
| 7d82935b | calc:7d82935b07aabecb27d9ab5ae3128389637628543fb5230514f2dc75574a8a0a | same (survey part 3) |
| a85d621f | calc:a85d621fe28ab50e3d41ccff5e44007f23a17c8327079c06ef35a25db7fee56e | remainder 41,585,920 B |
| 882844dd | calc:882844dd4d6f4112e7ddfc1ccf09e5993fc8fd557895deee49bb5f67ea3aa0e8 | ceiling 1 row 10.888 ms |
| 222e9847 | calc:222e9847852227642662d439d433435cde245e34189bc4b6da576278ed036943 | same (survey part 3) |
| 5d519513 | calc:5d51951381f98fdeeb0394215d0068302d0be99c996951f52ce22b3d1cecd14c | ceiling 1 row 91.85 tok/s |
| 9e25148c | calc:9e25148ce2a5953fcd6635e2d08c2be00572aa0b2057a8598c3d712b38bd2882 | measured 143.6 GB/s |
| f6e3c4ed | calc:f6e3c4edd044ff5281afb6e0d2608f14c976fc19c9a7a2116f68d0f596b739f1 | 52.6 % |
| 51752d62 | calc:51752d627c6c240fc40be50c6a714ea589df892eb3c4a4e68195a9f722372770 | measured 48.31 tok/s |
| 6209daa6 | calc:6209daa6afa0d9a632a61f235f97104c19226f38ab5b21611bc6d640a38f6fbf | gap 9.812 ms |
| b886b3e7 | calc:b886b3e78e21db85497b4dcb0be4547611512a2217bb5a7b5fb1049a9910efa1 | shared-once layer bytes 1,456,865,280 B |
| ffe2bed5 | calc:ffe2bed515ea8fd57e0b0d1cd6b92625b566d2617bb71e37d7e98540120da393 | expected distinct experts 57.42 |
| e3398026 | calc:e3398026b50e4cb92dab921d5ce973a32c96ffcb53bda502c33238f3c68c0c55 | 8 rows 9,192,535,039 B |
| 99d9b471 | calc:99d9b4710f232916eee7a017cf580a197dc53ca679383d33114296d35fad5279 | ceiling 8 rows 33.672 ms |
| cd160229 | calc:cd1602296364f890b6d4a1accdd5b64dd195458fbd605a74ebd4b416f6de9b1e | same (survey part 3) |
| 726e7afa | calc:726e7afaf1c7bebd361aa424a090c26c5e9419f65f315a1259760cca7ad805ab | ceiling 8 rows 237.6 tok/s |
| fbfcba7f | calc:fbfcba7f8dd1ac2823b6c652cc3fbf7c3e731d3ce7243a845c450af6a5f377a0 | 56.9 % at 59.2 ms |
| 3d15ca8e | calc:3d15ca8e164d399a2ec89413ab7e736106d74fa0e672d5b8b2caf6a3f52aae2a | 56.1 % at 60 ms |
| 6cb7c2fd | calc:6cb7c2fda4c72a361a9f0b477b33fe34424c3814743ca487014196bed85178af | same (survey part 3) |
| c1524bc6 | calc:c1524bc6204a4e2389105c42eddd58a2210f9ae3706cdd2cda225881d12a4b55 | measured 8 rows 135.1 tok/s |
| 6a3e5637 | calc:6a3e56377cacddb696e578238b57ebced83010dd219f6547c6353282f43d8957 | 133.3 tok/s at 60 ms |
| 119cf0b1 | calc:119cf0b1af83302317a64adfa0482796887925866b97b565a0e19ceb543294e4 | 8 rows worst 10,020,487,168 B |
| 3af4c46a | calc:3af4c46ab9b942128b8590c7a3c54ac2bbf07968d0d058753c5cf35bce76c33e | ceiling 8 rows worst 36.705 ms |
| 8d19dfc2 | calc:8d19dfc2b417bd7c370958b891a87e0aa87546e3d05320bccf17f0c38b61ad52 | GDN floor 3.724 ms |
| 857a5be3 | calc:857a5be3b47ea2b8528598091ac90d90033a77e8db018a3444fe620222ff6e04 | attention floor 0.999 ms |
| ccf635f1 | calc:ccf635f1746d2370bb900e6782c2867f3c0166221f755eade518c339841c7be4 | experts + shared floor 4.149 ms |
| 4655f814 | calc:4655f8142b77a94350648d395bfee675b23e453afbf3cfa68935c6fd9f8b70c7 | head screen floor 1.863 ms |
| 190a4565 | calc:190a456582e819d14dfd129ddcc9d59c5954d479ac225d47fd721c631d59c0d6 | launch floor 1 row 1,560 us |
| 0cc43bec | calc:0cc43becba4ca21dab19b1708f4e60499feee9fd846e6720cb800dbe7421cfb6 | launch floor 8-row stroke 12,480 us |
| a7b847e5 | calc:a7b847e5765dc0f45fec204e00eedcb14fff9222bdfc9600742f0ef093f32442 | ridge 351.6 FLOP/B |
| 521952a7 | calc:521952a7339c2650c5484bd91867db2525a7c2eae0becdfd4f5cedfebb6873b7 | MTP ~6 % at 8 rows (survey) |

Wolfram: `evidence/q36-ceiling-20261024/wolfram/INDEX.md` (24 rows, request + raw reply + sha256 each). (plus 31 rows for the measured timeline, section 10)

## 10. Measured: where the 1-row step goes (2026-10-24, L310)

Camera: the graph's own per-launch stamps, read by `q36_levers` built from the live pins, 15 stamped tokens, chip alone, clean repeat run. Full report, receipts and raw replies: `evidence/q36-ceiling-20261024/timeline/REPORT.md`.

| group | ms per token | share | floor at 273 GB/s | gap | effective GB/s |
|---|---|---|---|---|---|
| experts (gate_up + down) | 7.092 | 0.344 | 4.148 | 2.944 | 159.7 |
| GDN (mix + gdn + ab) | 5.559 | 0.270 | 3.947 | 1.612 | 193.8 |
| attention (o_proj + attn) | 3.089 | 0.150 | 1.263 | 1.827 | 111.6 |
| exact head (4 launches) | 2.352 | 0.114 | 1.863 | 0.489 | 216.2 |
| router (40 launches) | 1.687 | 0.082 | 0.154 | 1.532 | 25.0 |
| no-byte kinds (res, swiglu, argmax, gather, gate, norm) | 0.712 | 0.035 | 0 | 0.712 | - |
| step (chip, mean of 15) | 20.594 (calc:18fd24e064fae9b9) | | 11.375 (harness bytes, head corrected) | 9.117 (calc:d3863e9dca4455c5) | |

Receipts for every cell are in the report (aien-calc ids, Wolfram raw replies, 31 rows, all agree). The harness reproduces the live server: 0.995 of the 20.7 ms in serve.log:11597 (calc:71911245a16960d0). Negative control: `AIEN_Q36_EXACT_HEAD=0` moved only the head, +2.322 ms (calc:701f962c929e3d3a), so the L314 lever is worth 2.32 ms per token on this build. Byte-model note: the harness counts the head as BF16 and its GDN/attention groups 133 MB above section 2; UNKNOWN which is exact (follow-up). Identity: gen-p1..p3 identical; gen-p4 differs at token index 4 from its 2026-10-20 host-oracle golden in every run, head mode independent, and the train carries no p4 case: UNKNOWN which side is right.

Next lever by size: the expert kernels at 159.7 GB/s (58.5 % of the bus) with a 2.94 ms gap; then the router's 1.53 ms of pure launch latency.
