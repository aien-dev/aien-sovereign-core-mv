# Expert Ribbon: OWN-30B decode with paged FP8 experts (2026-10-18)

sc ribbon/p1-store 79e46d77 + omega 1549caae (omega 4799f9eb + the slot pool, omega#533; slot chip gate PASS 3/3
on this base). Model Qwen3-30B-A3B-Instruct-2507-FP8, 6,144 experts, model digest 57e416ad49...
Flags as the resident gate in own30b-omega4799-20261015, tokens only: `--weights fp8 --attn fused --router device
--lm-head gemv --prefill auto --new 16`, plus `--paged-experts SLOTS` (`run.sh`). The live aien-serve-http and CI jobs
shared the chip and memory during both runs. Prefill runs token by token in paged mode (batched prefill and the
decode graph need resident experts and refuse a paged load).

## Tokens: identical to the CPU golden at both capacities

| Run | Slots (of 6,144) | Prompts | Tokens vs golden | Weights resident | Peak RSS |
|---|---|---|---|---|---|
| s768-p12 | 768 (1/8) | p1, p2 | IDENTICAL x2 | 1.53 GB + 3.62 GB slots | 6.94 GB |
| s3072-p1234 | 3072 (1/2) | p1, p2, p3, p4 | IDENTICAL x4 | 1.53 GB + 14.5 GB slots | 17.82 GB |

The resident run of the same flags matched the same golden (own30b-omega4799-20261015, tokens IDENTICAL x4).
Host tests prove the paged path gives the resident logits bit for bit on the tiny model
(`cargo test -p aien-inference-abi --lib ribbon`).

## Speed (decode, 16 tokens)

| Run | Prompt | Hit rate (whole prompt) | Decode tok/s | Page time per decode step | Receipt |
|---|---|---|---|---|---|
| 768 slots | p1 (cold) | 0.60 | 2.396 | 388.5 ms | calc:eac8a69d |
| 3072 slots | p1 (cold) | 0.68 | 2.738 | 336.5 ms | calc:440646a6 |
| 3072 slots | p4 (warm, after 657 prompt tokens) | 0.96 | 7.07 | 112.2 ms | calc:59123200 |

- Ceiling: every expert of every token missed, read at the measured random verified rate 10.60 GB/s
  (calc:8d297d5b): 5.849 tok/s (calc:1fc0715c). Cold 768-slot decode is 0.41 of it (calc:3bf87234); warm 3072-slot
  decode is 1.21 times it (calc:d185ac62), because hits skip the SSD.
- The SSD is not the limit yet: during misses the device delivered 1.715 GB/s (768 slots, p1, calc:ff9762d7) and
  0.729 GB/s (3072 slots, p4, calc:2f952a9d) of the 10.60 GB/s it gives a full queue. Misses are read layer by
  layer (a layer's misses in parallel, the next layer only after the router picks), so paging waits on read latency
  (about 3.5 ms per expert at queue depth 1, read-bench.jsonl). Overlap and prefetch (sc#631) is the next step.
- Resident decode speed under the same load was NOT MEASURED in this session: a resident load needs about 31.5 GB
  and own30b_check's 70 GB gate refused to share the box with the live server; it was not overridden.

## Read adapter (ribbon_read_bench, random whole experts, O_DIRECT, BLAKE3-verified, 3,000 reads)

1 worker depth 1: 1.32 GB/s (calc:b43572b8); 8/16: 8.69 GB/s (calc:84b90d97); 16/32: 10.60 GB/s (calc:8d297d5b),
0.673 of the PCIe Gen5 x4 link (calc:cf3a961c). Raw JSON in read-bench.jsonl.
