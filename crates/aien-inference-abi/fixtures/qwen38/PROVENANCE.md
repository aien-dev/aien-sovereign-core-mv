# Qwen3.8-27B host reference golden

`gen-chat-max-24.json` is the greedy 24-token answer of the host reference forward
(`crates/aien-inference-abi/src/qwen38.rs`, run with the `qwen38_check` binary in `generate` mode)
for one chat prompt. It is the expected output that `tools/qwen38_serve_gate.sh` compares the live
server against, and `tests/qwen38_chat_template.rs` checks that the chat template renders the same
prompt ids.

- Model: Hugging Face `Qwen/Qwen3.8-27B-FP8`, revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Prompt (thinking off), as rendered by the model's chat template:

  ```
  <|im_start|>user
  Write a C function that returns the larger of two ints.<|im_end|>
  <|im_start|>assistant
  <think>

  </think>
  ```

- Command: `qwen38_check generate <checkpoint-dir> prompt-chat-max.txt 24 gen-chat-max-24.json`
  on the DGX Spark CPU, 2026-10-18.
- Answer text: ```` ```c\nint max(int a, int b) {\n    return (a > b) ? a : b ````
- Fields: `prompt_ids`, `generated_ids` (the golden), `generated_text`, `top5` logits per step,
  and the wall time of each step. Only the ids are compared.

Copied unchanged from the operator's reference folder so that anyone can run the gate; do not
edit it. Regenerate it only with the same command and model revision.
