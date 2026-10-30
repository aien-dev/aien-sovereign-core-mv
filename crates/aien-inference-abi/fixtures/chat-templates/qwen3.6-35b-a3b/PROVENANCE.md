# Qwen3.6-35B-A3B chat template (pinned)

- Source: Hugging Face `Qwen/Qwen3.6-35B-A3B-FP8` revision 95a723d0 (the size-checked download
  `~/models/qwen3.6-35b-a3b-fp8-95a723d0`, `download.log` 2026-10-20; the full revision hash was not kept in the
  download log), `chat_template.jinja` copied byte for byte. The `chat_template` field of that revision's
  `tokenizer_config.json` is the same text without the final newline; the loader trims before matching, so either
  copy selects this layout.
- Licence: Apache-2.0, upstream `LICENSE` copied unchanged. Not relicensed.
- sha256 chat_template.jinja: e84f32a23fdda276 (first 16 hex digits; full hash in `sha256sum` of this directory)
- sha256 upstream tokenizer_config.json: 5186f0defcd7f232 (first 16 hex digits)
- `render_oracle.json`: the template rendered by the reference engine, jinja2 3.1.2
  `ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)` with `raise_exception` and `tojson` as
  transformers provides them, `tools=None`, `add_generation_prompt=True`, 8 conversations x 6 option sets
  (thinking undefined / true / false, preserve_thinking true / false, reasoning_effort low which this template
  ignores), plus 5 refusals with the template's own exception text. Produced once with a scratch script outside
  this repository (no Python in the repository); the Rust test `tests/qwen36_chat_template.rs` compares the
  engine's rendering to it.
- The engine renders text-only system / user / assistant turns (`ChatTemplate::Qwen36`). Differences from the
  27B template that the renderer carries: no reasoning-effort system text; a first system turn renders even when
  empty; earlier assistant turns are plain unless `preserve_thinking` is true or the turn follows the last user
  query, in which case the think block carries the reasoning split out of a `</think>`-bearing content. Tools,
  images and video are refused.
