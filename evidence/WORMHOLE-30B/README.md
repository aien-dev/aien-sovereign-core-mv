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

## 00 Prediction
Pre-registered before any acceptance run: 00-prediction.md.

## 03 Vocabulary (W1)
Command: `wormhole30 vocab TOK_27B TOK_30B 03-vocab.json <the 3 L080 inputs>` (host only).
- 27B 248,077 ids (with added), 30B 151,669. Normal 27B tokens whose bytes are also a 30B token: 131,586;
  of those with the same id: 281. Different vocabularies: the text bridge (align.rs) is required.
- 6 texts (3 prompts, 3 27B continuations), 2,570 bytes: 613 tokens in the 27B, 578 in the 30B. Every
  30B token boundary is also a 27B boundary (share 1.000); 94.3% of 27B boundaries are 30B boundaries.
  Round trips exact both sides; the 27B's chip ids equal a fresh cut of its text on all 3 continuations.
- Bridge cost per block, k=4: 3.4 to 6.6 microseconds on the host (box load average 64 at the run).
