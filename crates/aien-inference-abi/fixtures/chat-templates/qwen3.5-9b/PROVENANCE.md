# Qwen3.5-9B chat template (pinned)

- Source: Hugging Face `Qwen/Qwen3.5-9B` revision c202236235762e1c871ad0ccb60c8ee5ba337b9a (the download
  `~/models/qwen3.5-9b-c2022362`, sha256 list `~/models/qwen3.5-9b-c2022362.sha256`, 2026-10-27),
  `chat_template.jinja` copied byte for byte. The `chat_template` field of that revision's `tokenizer_config.json`
  is the same text; the loader trims before matching, so either copy selects this layout.
- Licence: Apache-2.0, upstream `LICENSE` copied unchanged. Not relicensed.
- sha256 chat_template.jinja: a4aee8afcf2e0711 (first 16 hex digits; full hash in `sha256sum` of this directory)
- sha256 upstream tokenizer_config.json: 316230d6a809701f (first 16 hex digits)
- Differences from the Qwen3.6-35B-A3B template (`../qwen3.6-35b-a3b`): no `preserve_thinking` switch (an
  assistant turn carries its think block only when it follows the last user query), and tool-call arguments are
  rendered with `tojson` for mappings and sequences, `string` otherwise. The engine refuses tools, so only the first
  difference reaches a rendering; `ChatTemplate::Qwen35` is the Qwen3.6 renderer with `preserve_thinking` ignored.
- `render_oracle.json`: the template rendered by the reference engine, jinja2 3.1.2
  `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` with `raise_exception` and `tojson` as
  transformers provides them, `tools=None`, `add_generation_prompt=True`, for the same 48 conversation x option
  cases and 5 refusals as the Qwen3.6 oracle. 45 cases render identically to the Qwen3.6 oracle; the 3 that set
  `preserve_thinking: true` with an earlier assistant turn differ. Produced once with a scratch script outside this
  repository (`~/handoffs/H2H9B/ours-evidence/template/make_oracle.py`; no Python in the repository); the Rust test
  `tests/qwen36_chat_template.rs` (`qwen35_*`) compares the engine's rendering to it.
