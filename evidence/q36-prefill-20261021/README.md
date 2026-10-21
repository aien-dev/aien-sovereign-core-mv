# L369 chip prefill: evidence (2026-10-21)

Prompt tokens of one conversation read in chunks of 128 through the chip (GEMMs for the dense trunk and the experts,
one multi-token launch for each DeltaNet layer, one prep + attend pair for each attention layer, COMBINE|NORM and the
attention gate in rows form). Every step is bit-identical to feeding the prompt one token at a time through the decode graph.

Build v3: omega serve/q36-prefill-v2 60ba3051, sc serve/q36-prefill-v2 6127db50 (qwen36_prefill_gate, sha in gate-binary.sha256).

## Timing (full 35B, inside spark-window 20261021T215012Z-134c82, chip + quiet exclusive): full-time-v3.{json,log}
| prompt | token by token | chunked | speedup |
| 256  | 49.7 tok/s | 275.5 tok/s (calc:d6ee1b8d) | 5.54x (calc:6519689a) |
| 2048 | 48.6 tok/s | 249.95 tok/s (calc:3c01f9f6) | 5.145x (calc:01436d26) |
| 8192 | 45.3 tok/s | 196.77 tok/s (calc:84250c57) | 4.344x (calc:1bb46114) |
Bit for bit at every length: logits of the last prompt token, every DeltaNet state, and 16 decode tokens after.
Ceiling: fp8 weight bytes streamed 504.5 MB per prompt token (calc:6e17d89c; experts re-read once per 4 rows, dense
once per 8) -> 436 tok/s at 220 GB/s (calc:57f49733); at 256 the run streams at 0.63 of 220 GB/s (calc:1455fcbe).
The row tile of the fp8 GEMM is the lever past 436.

## Exactness on a 4-layer slice (prefix4v3.result, prefix4v3-*.log): lengths 7/64/129/300
real PASS; per-token controls PASS (gdn0 = AIEN_OMEGA_Q36_PF_GDN=0, attn0 = AIEN_OMEGA_Q36_PF_ATTN=0, cn0 =
AIEN_OMEGA_Q36_CN_ROWS=0); mutants FAIL as they must: 1 (GDN steps of the last two tokens swapped), 2 (one expert row
dropped), 3 (GDN output copy deferred), 5 (DeltaNet gates row stride 0), 6 (attention prep launch dropped).
Chip unit tests (omega): gpu_gdn_multi_test 30/30, gpu_dattn256_multi_test 10/10 (up to 8k context, controls caught).
v1 full 35B value run (no window): full-value-real.{json,log}, PASS at 256/2048/8192.
