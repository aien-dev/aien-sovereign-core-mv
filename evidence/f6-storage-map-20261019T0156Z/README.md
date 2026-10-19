# Storage map receipts (sc #678, FORTRAN 1956 idea 6)

Spark GB10, branch fortran/6-storage-map, omega archive `libomega_gpu.a` sha256 373d122c... (omega qwen38/prefill-graph
6e24f7fd, fresh build), model Qwen3.8-27B FP8 @ 017b9c7a. Every run under `flock /tmp/aien-gb10.lock`, lock released
between runs; the live 27B server was up for all of them. Exact commands: `command-*.txt`.

Each run writes `<out>.storage-map-load.{txt,json}` (after the upload) and `<out>.storage-map-end.{txt,json}` (end of run):
one line per named block (space, bytes, lifetime, owner, writers), totals by lifetime, held vs named per space and the
UNACCOUNTED difference, per allocating call planned vs measured, and the storage check verdict.

Truth used: `/proc/self/smaps`. Driver = summed Size of the `/dev/nvidia*` mappings (omega's allocations). Heap =
Anonymous of every other mapping. omega's own counters are printed beside it and fall short: the full-model golden run
shows 27,174,064,128 counted vs 27,176,820,736 mapped, because `omega_gpu_pf_alloc_dev` does not count.

## Results (final builds: storage mode r4 7f0301ce, golden/ttft r5)

| run | layers | named driver | held driver | unaccounted driver | planned peak | held peak (VmHWM) | verdict |
|---|---|---|---|---|---|---|---|
| storage-l4-c256 (512 tok, chunk 256) | 4 | 4,205,166,592 | 4,205,166,592 | 0 | 7340.3 MiB | 7277.8 MiB | PASS |
| storage-l4-c128 (300 tok, chunk 128) | 4 | 4,148,383,744 | 4,148,387,840 | 4,096 (one code page) | 7340.3 MiB | 7277.3 MiB | PASS |
| storage-l4-t4096-c4096 | 4 | 5,887,287,296 | 5,887,287,296 | 0 | 8873.3 MiB | 8766.3 MiB | PASS |
| golden-full-c8 | 64 | 27,176,824,832 | 27,176,820,736 | -4,096 | 28733.3 MiB | 28419.0 MiB | storage PASS |
| ttft-full-8192-c4096 | 64 | 29,476,577,280 | 29,476,515,840 | -61,440 (first-decode code, never run in ttft) | 31442.3 MiB | 31377.1 MiB | PASS |
| **storage-l4-c256-LEAK** (negative control) | 4 | 4,205,166,592 | 4,473,602,048 | **268,435,456** | 7340.3 MiB | 7277.9 MiB | **FAIL (red)** |

- Negative control: `AIEN_STORAGE_MAP_LEAK_BYTES=268435456` makes one omega host buffer and one heap vector of that size,
  every byte written, never named. Unaccounted driver = 268,435,456 exactly (calc:a6f2908221929e2775e469fdfc3b17b51013ec20ce76bcd92e0503f0b0618dfc),
  unaccounted heap 256,126,976; both past their bars (64 MiB / 128 MiB): FAIL.
- Old vs new estimate, full model, golden: old `need_bytes` said 25829 MiB (prior evidence qwen38-prefill-20261018T1836Z);
  the process peaked at 28419.0 MiB, so the old figure was 2589.99 MiB low (calc:4b9bab3ef3e8e7d5d3f7951a5a09ecae4dd5de4751e7d2ad417a8cf4ccd16334).
  New plan 28733.3 MiB covers the peak by 314.31 MiB (calc:caacd3c05172a20ea06113f3064f48a10b5b4ae11e597c887927668195aab574).
- Chunk 4096 (the configuration of the 2026-10-18 19:20Z near miss), 8192 tokens: plan covers the peak by 65.20 MiB
  (calc:58c8052257a34357e4d7113de8a4ef07249ade51e3a444a91d1869c5efea7d73). Peak = 0.2394 of the 128 GiB box
  (calc:e0e436912f60129384877ece194d4d027938e0485601a297b48847306679194c).
- 128k, chunk 4096: NOT MEASURED (would hold the GPU lock about 16 minutes against the live server). Plan peak 39183 MiB
  (host test `qwen38_omega::storage::incident`), old formula 35509 MiB: the plan is 3674.46 MiB higher
  (calc:7a6382a1d38d8f98a040fae714d34ab975d5d47138546378f74e3f9a82ce48b1). Predicted held peak from the measured 8192 run plus
  the plan's difference (KV planes, RoPE rows; the attention-cache call matched its formula exactly at every cache size run
  here): 39188.4 MiB (calc:bc999dc2e478c70b10a45939a1bc488ab0588a5ad7ba428ecf78c81ac0d9a098). The incident's "RSS 39 GiB"
  was a rounded figure, not a receipt.
- Output unchanged: golden 24/24 tokens equal to the host reference, first-token cosine 0.9928993630379012, the same value as
  the prior evidence (that mode's overall FAIL is its pre-existing 0.999 cosine bar, not this change). ttft first token 65
  in both runs. TTFT times here are not speed results (probes and a shared box).

## What the first run found (storage-l4.*, build r1)

25 of 30 allocating calls matched the plan to the byte on the first try. The rest: session open 1,323,008 bytes, kernel code
pages, and a 64 MiB omega host-entry stage the path never makes (removed from the plan). The peak check failed because the
2.5 GB lm_head host copy coexists with its upload; the plan is now a running peak in load order. `ttft-...-run1-peak-short`
missed the peak by 6 MiB: the tokenizer's heap before the model opens, now measured and named in every receipt.

Allowances (measured, exact, in `qwen38_omega/storage.rs`): session 1,323,008; graph code 69,632; prefill code 126,976;
head code 8,192; first decode 61,440; process heap 64 MiB. Code pages move between first-use calls with the order of a mode
(golden decodes before it prefills): allowance calls may drift 64 KiB, derived calls must match exactly.
