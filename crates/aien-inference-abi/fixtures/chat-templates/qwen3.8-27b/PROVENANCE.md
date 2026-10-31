# Qwen3.8-27B chat template (pinned)

- Source: Hugging Face `Qwen/Qwen3.8-27B-FP8` revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, `chat_template.jinja`,
  copied byte for byte from the size-checked download `~/models/qwen3.8-27b-fp8-017b9c7a` (identical to the
  `chat_template` field of that revision's `tokenizer_config.json`).
- Licence: Apache-2.0, upstream `LICENSE` copied unchanged. Not relicensed.
- sha256 chat_template.jinja: c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041
- sha256 upstream tokenizer_config.json: b11349aafa7cdc6a320767cf7ceb29ed82f7eda5d65e8e0819e76f0ce947bf27
- `render_oracle.json` (sha256 06ffc4504ebf3871276c00bf8b76a82ad41510646cd98a257cb74f07a4043f80): the template
  rendered by the reference engine, jinja2 3.1.2 `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)`
  (the Hugging Face transformers configuration), `tools=None`, `add_generation_prompt=True`, 8 conversations x 6
  option sets (thinking undefined / true / false, reasoning effort low / medium, thinking off with effort low).
  Produced once with a scratch script outside this repository (no Python in the repository); the Rust test compares the
  engine's rendering to it.
- The engine renders text-only system / user / assistant turns (`ChatTemplate::Qwen38`). Earlier assistant turns carry
  no `reasoning_content` (the template's empty think block). Tools, images and video are refused.
