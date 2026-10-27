# Qwen3.8-27B host reference, first results (2026-10-18)

Checkpoint: Qwen/Qwen3.8-27B-FP8 at 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a; all 67 LFS files match their published sha256.
Code: `crates/aien-inference-abi/src/qwen38.rs`, tool `src/bin/qwen38_check.rs` (release build). Host only, no chip.
Machine: DGX Spark, CPU only, weights streamed from disk one layer at a time (the 30B server held most memory).

- `score-license-256.json`: the first 256 tokens of the checkpoint's own Apache-2.0 `LICENSE` file, one pass.
  Perplexity 1.091, next-token accuracy 0.988 (252 of 255), pass 82.3 s.
  `qwen38_check score <dir> LICENSE 256 score-license-256.json`
- `gen-chat-max-24.json`: greedy continuation of `prompt-chat-max.txt` (thinking off), 24 tokens with each step's top five
  logits: "```c\nint max(int a, int b) {\n    return (a > b) ? a : b". These ids are the golden every Omega stage matches.
  `qwen38_check generate <dir> prompt-chat-max.txt 24 gen-chat-max-24.json`

Unit tests (`cargo test -p aien-inference-abi --lib qwen38`): config reader, FP8 matmul vs the scalar formula, partial RoPE,
one call vs token by token (state carry), refusal of an FP8 weight without scales. Negative control: dropping the conv
tail carry makes the state-carry test fail (max logit difference 2.23).
