# aien-bitnet-tokenizer

Our own tokenizer for BitNet b1.58 2B-4T: the LLaMA-3 byte-level BPE read from the checkpoint's `tokenizer.json`, with no tokenizer library in the path. Special tokens are cut out first, a hand-written scanner follows the LLaMA-3 split pattern exactly (Unicode tables from `regex-syntax`, no regex engine runs), then BPE merges by rank. Any `tokenizer.json` setting it does not implement is refused by name.

It lives in its own crate so a CPU-only machine can tokenize without building the GPU stack: `aien-hetjob` runs it on remote workers (the `tokenize_batch` job, aien-sovereign-core#637). `aien-inference-abi` re-exports it as `aien_inference_abi::bitnet_tokenizer`, so existing paths are unchanged.

Tests: `cargo test -p aien-bitnet-tokenizer` (split pattern, byte alphabet, toy merges, refusals). Against the real checkpoint: `BITNET_DIR=~/models/BitNet-b1.58-2B-4T-bf16 cargo test -p aien-inference-abi --test bitnet_tokenizer_real -- --ignored` (acceptance id lists and round trips). On real data: 40,137 lines of our architecture docs, every line decodes back to itself, same ids on the Spark (aarch64 Linux) and an M2 (aarch64 macOS), see `crates/aien-hetjob/evidence/het637-m2-tokenize-20261018/`.
