# WORMHOLE-30B evidence (ledger L190)

27B = ~/models/qwen3.8-27b-fp8-017b9c7a, 30B = ~/models/qwen3-30b-a3b-instruct-2507-fp8.

## 01 Shared weights (Drake 2026-10-20T17:20Z: load common weights once for both?)
Command: `wormhole30 tensors DIR_27B DIR_30B 01-shared-tensors.json tensors.tsv` (host only, under quiet-build).
- 27B: 1,606 tensors, 30.87 GB, hashed in 44.7 s. 30B: 37,491 tensors, 31.17 GB, hashed in 58.7 s.
- Identical tensors (sha256 of the bytes): 0. Bytes shared: 0. GB saved if loaded once: 0.
- 64 KiB byte ranges (aligned to each tensor start) of the 27B found anywhere in the 30B: 0 of 471,996
  (0 non-zero, 0 all-zero).
- Shapes in common: one, `[128]` BF16 norm vectors (27B 48 x linear_attn.norm, 30B 96 x q_norm/k_norm), bytes differ.
- Config: hidden 5120 vs 2048, vocab rows 248,320 vs 151,936, layers 64 vs 48, head_dim 256 vs 128.
Verdict: nothing to load once for both. Limit: byte ranges shared at an offset that is not a multiple of
64 KiB from a tensor start are not detected; with different hidden sizes, dtypes per role and vocabularies
no such sharing is expected.
Per-tensor list (model, name, dtype, shape, bytes, sha256): 01-tensors.tsv.gz.
