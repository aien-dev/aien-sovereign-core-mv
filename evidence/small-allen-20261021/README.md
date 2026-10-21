# Small Allen student on the host reference (L389, 2026-10-21)

Student: Qwen/Qwen3.5-0.8B @ 2fc06364715b967f1860aea9cf38778875588b17 (Apache-2.0), local copy
`~/models/qwen3.5-0.8b-2fc06364`, LFS sha256 verified (`~/models/qwen3.5-0.8b-2fc06364.sha256`).
Tokenizer: `tokenizer.json` sha256 5f9e4d4901a92b997e463c1f46055088b6cca5ca61a6522d1b9f64c4bb81cb42, identical to the
teacher Qwen3.6-35B-A3B-FP8 @ 95a723d0 (vocab 248320), so the 35B can check the student token by token.

Reference changes (crates/aien-inference-abi/src/qwen38.rs): tied embeddings read the embedding table as the lm_head;
F32 vectors (A_log, linear_attn.norm.weight in the 0.8B) read exactly.

Receipts (host only, no chip, `target/release/qwen38_check`, Spark CPU):
| file | what | result |
|---|---|---|
| host-score-license.json | Apache LICENSE first 2500 bytes, 256 tokens | perplexity 1.319, next-token acc 0.933 |
| host-score-license-shuffled.json | negative control: same words shuffled | perplexity 442.634, acc 0.125 |
| host-score-readme.json | model README bytes 1000..3000, 256 tokens | perplexity 21.108, acc 0.439 |
| host-gen-hey.json | greedy, "hey allen how are you doing tonight", 24 tokens | "Hey everyone! I'm doing well, thanks for asking. How about you?" |
