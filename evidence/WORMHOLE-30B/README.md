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

## 06 Heartbeat ring for the 30B (W2, host side): exactness and memory
Command: `wormhole30 ringcheck DIR_30B 06-ringcheck.json 64 512 q38-codelong.json` (host only, 20:19Z).
- The 30B with only its dense part resident (6.14 GB: attention f32, embed + lm_head f32, routers) and the 8
  routed experts of every layer read through the Heartbeat ring (512 slots, 2.43 GB, BLAKE3 per expert,
  O_DIRECT), against the resident CPU reference (about 35 GB): 64/64 tokens with logits bit for bit equal.
- Fail-closed control: every layer-0 digest corrupted and the ring emptied; the next token refused with
  "digest mismatch for expert (0, 75)".
- Memory: MemAvailable 75.2 GB before, 65.7 GB with the ring drafter open (dense + slots). Manifest build
  10.3 s (page cache warm), ring open 6.7 s.
- Ring traffic: 64.66 GB read from the SSD = exactly 13,701 whole experts (calc:607ab5d8) for 24,576 lookups
  (64 words x 48 layers x 8): hit rate 0.4425 at 2.43 GB from a cold start (calc:aed4654a). The counters in the
  JSON say 0 misses: that was a counting bug (fixed in 4463f607, see below), the byte count is the truth.
- Speed: 4,413 ms per 30B word on the host (CPU arithmetic plus SSD fills, 82.3 s of the 282 s were fills),
  33.7x one 27B word at its live 7.63 tok/s (calc:cf20fd80). As a drafter on the host it cannot pay for itself.
- Bug found and fixed here: aien-wormhole cache counted a lookup's hit or miss on the manifest's tensors[0];
  the 30B drafter reads gate_proj first, so every fill counted as a hit. Now one miss per demand fill.

## Not run (with the reason)
- 02 one-read load timing (Drake's two-birds path when nothing is shared): tool built and its digests path
  unit-tested through the real loaders; the run refused twice for memory (needs 45 GB for the 35 GB reference;
  30.6 GB free at the 20:12Z window), then the 30B server on 18091 (59 GB, from 20:28Z) left about 31 GB.
- 04 forward acceptance sweep k=1,2,4,8 (CPU reference drafting, the 27B's recorded output verifying): started,
  stopped at 20:19Z by the chat's 2 h background limit before its first text finished; not rerun (memory).
- 05 endpoint receipt (27B alone vs the pair on 18190, byte-identical gate): binary built (native, omega lib
  sha bd6802a5 = the lib the 27B server ran); the 27B-alone server started and was cleaned up with its parent
  job; Drake 20:30Z dropped the 27B (MoE only), so it was not restarted. Identity holds by construction (the
  27B door's own loop writes every token; the drafter only watches) but has no run receipt.
- Chip: 30B experts in a chip-side ring (needs an omega API to write expert bytes into slot memory), the 27B
  exact-rows verify port (cascade/mtp-chain f21b8e88), W3 on chip and the W4 timed race: not built, not run.
