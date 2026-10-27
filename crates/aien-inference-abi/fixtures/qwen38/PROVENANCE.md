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

The same bytes are kept as evidence in
`docs/inference/evidence/qwen38-ref-20261018/gen-chat-max-24.json` (sha256 b5ec5494...); this copy
is the test fixture and the gate default.

# Speed test references

`decode-lighthouses-256.json`: the request `tools/qwen38_serve_speed.sh` sends for its three
decode runs (greedy, 256 tokens, thinking off) and the 256 token ids the live server returned
for it on 2026-10-19 (binary sha256 df936011..., sovereign-core 46134247, omega 0010f7d3). Taken
unchanged from `evidence/qwen38-serve-fast-20261019/after/new-1.json`; all four runs there gave
the same ids. The previous binary also gave the same ids. The speed tool reports MATCH when a
server returns exactly these ids.

`speed-corpus.txt`: the text the speed tool cuts its long prompts from (the first 7,100 and
100,000 bytes). It is plain license texts, freely redistributable verbatim, concatenated in this
order: the GNU AGPL v3 (this repository's LICENSE), and from Ubuntu's
/usr/share/common-licenses: GPL-3, Apache-2.0, MPL-2.0, GFDL-1.3. The corpus used for the
first-token figures in older evidence folders was not recorded, so first-token times measured
with this file are not comparable with those.
