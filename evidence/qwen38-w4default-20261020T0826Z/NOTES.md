# Qwen3.8-27B served in 4 bits from a stored file (ledger L125, SPEED-1), 2026-10-20

One measurement window, `20261020T082430Z-0d26f5` (gb10 exclusive, 08:26Z to 09:57Z). Builds: omega `speed1/w4-default`
d2d74041 (`libomega_gpu.a` sha 317f4980...), sc `speed1/w4-default` d964ec7c. Each run folder holds the command, the
ground snapshot (binary hashes, free memory) and its run log; `receipts/` holds the spark-receipt wall-power files.

Arms: **S** = 4-bit only from the store (the new default), **F** = 8-bit weights + 8-bit head, **H0** = 8-bit weights +
16-bit head (the accuracy reference), **L** = 4-bit converted at load beside the 8-bit copy (the old way).

## Speed (4 rotated rounds F S / S F / F S / S F, 47 steps each, golden prompt, `qwen38_omega_full`)

| arm | median ms per word (188 steps) | words per second | ceiling | fraction of ceiling |
|---|---|---|---|---|
| S | 69.957 | 14.29 (calc:f0c1b118) | 20.01 (calc:07f2175b) | 0.714 (calc:babb1a77) |
| F | 117.614 | 8.50 (calc:69c20e5a) | 10.64 (calc:905bdd97) | 0.799 (calc:cf5b6ca5) |

S is 1.68 times F (calc:40e77198) and 1.034 times the 10-19 4-bit result of 13.82 words per second (calc:d48c71f1).
Round medians: S 69.55 / 71.19 / 69.72 / 69.79 ms, F 117.21 / 117.34 / 129.16 / 116.29 ms. The box ran at load average
about 33 beside the live server, so these are shared-box numbers. Every round gave the same words as the others.

`qwen38_omega_full` prints FAIL for both arms: that is the strict exactness gate (logits cosine 0.9999), which every
reduced setup fails. F: 24/24 golden words, cosine min 0.9994. S: 23/24, residual cosine min 0.7755.

## Memory and load

| arm | weights on the chip | open time |
|---|---|---|
| S (store) | 13.65 GB | 37.4 s (gen run), uploads 15.8 to 47.6 s in the speed rounds |
| F | 28.19 GB | 114.6 s (gen run), uploads 50.3 to 85.0 s |
| L (old 4-bit way) | 40.52 GB | 334.6 s |

The store removes 26.88 GB of weights against the old 4-bit way (calc:70171787). Peak process memory: UNMEASURED.

## Energy (256-word free run on the prose prompt, 289 steps, whole-box wall power from the UPS, coarse)

S 23.5 J per word (calc:f92829fe), F 31.8 J per word (calc:b1828c30). The wall figure includes the live server and every
other job on the box; read it as a comparison inside this window only.

## 4-bit fit gate (`qwen38_ab gate4`, limits frozen 02:35Z before the window in `docs/inference/qwen38/w4-fit-gate.json`)

**QWEN38_W4_FIT PASS**, `runs/summary/gate4.json`.

| line | same next word (limit) | KL mean (limit) | NLL change, nats per word (limit) | confident flips, worst margin |
|---|---|---|---|---|
| exact: store vs convert-at-load, golden | 47/47, max diff 0 | | | |
| full: residual cosine min | 0.7755 (0.70) | | | |
| golden 47 | 45 (44) | 0.0338 (0.05), max 0.844 (1.3) | -0.128 | 1, 6.62 |
| license 255 | 254 (250) | 0.0026 (0.004) | +0.0016 (0.035) | 0, 0.64 |
| prose 289 | 254, 266 with near ties (245) | 0.0772 (0.12) | +0.0305 (0.06) | 8, 1.67 |
| codelong 305 | 287, 292 with near ties (275) | 0.0588 (0.09) | -0.0145 (0.02) | 6, 2.21 |
| code 1023 | 880, 943 with near ties (850) | 0.0758 (0.11) | +0.0578 (0.09) | 9, 1.78 |
| arch 2047 (first measurement, not a pass line) | 1785, 1901 with near ties | 0.1033, max 13.9 | +0.0411 | 19, 9.46 (limit 10) |

Pooled over the five scored texts: 0.938 with near ties (limit 0.87), 0.896 without (calc:6bc81f7b).

## What went wrong on the way (kept, not hidden)

- `runs/aborted-1`: first start. S runs refused by a Rust check that still demanded a 16-bit head (fixed in sc d964ec7c);
  the chip test hung on a lock my own window already held (one.sh fixed to skip the lock inside a window, the run was
  stopped, rc 143); F-r1 refused at channel setup; the L round took 775.8 s to load on the busy box, so L speed rounds were
  dropped and one L golden dump kept for the exactness line.
- `runs/aborted-2`: five drift dumps (golden, license, prose, arch H0; prose S) refused at channel setup in session 2
  (RM_ALLOC 0xcec0 status 0x51: the driver's main context buffer needs one contiguous block, Cortex lesson 45a41b22).
  None reached a measurement. Each was rerun once in the same window (`bin/session3.sh`, free large blocks logged in
  `session.log`), all five completed.
- An earlier window, 07:31Z, was closed one second after opening by another agent's close request (reported on L113).

## Not done here

With no 8-bit copy, prompts are read one token at a time through the decode path (exact, slow for long prompts). A
4-bit prompt routine is needed before the live server on port 18091, which runs another engine path, can switch.
