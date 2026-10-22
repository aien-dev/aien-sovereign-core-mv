# DEV_MODEL_0 Model Discovery: Gemma 4 12B instruction-tuned

Fetch date: 2026-10-01. Method: plain HTTPS GET, no credentials, no weights downloaded.
Every fact is tagged VERIFIED (fetched from the named official URL) or UNVERIFIED.
Base URL B = https://huggingface.co/google/gemma-4-12B-it

## 1. Identity

- VERIFIED: A model with exactly this name exists. Repo id `google/gemma-4-12B-it`, host Hugging Face, author `google` (HF API https://huggingface.co/api/models/google/gemma-4-12B-it). Last modified 2026-07-20, repo commit sha 707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7.
- VERIFIED: Pre-trained sibling `google/gemma-4-12B`. Also `google/gemma-4-12B-it-qat-q4_0-gguf` (4-bit GGUF) exists (HF API listing, author=google, search=gemma-4).
- VERIFIED: It is dense (`enable_moe_block: false`). The card calls it "Gemma 4 12B Unified": one decoder with no separate vision or audio encoder (raw image patches and audio go through lightweight linear layers). Architecture class `Gemma4UnifiedForConditionalGeneration`, model_type `gemma4_unified`. It accepts text, image, audio, video input and produces text.
- VERIFIED: Card says 11.95B total parameters, 48 layers, 256K context (B/raw/main/README.md).
- MoE sibling, VERIFIED: `google/gemma-4-26B-A4B-it` (25.2B total, 3.8B active, 30 layers per README). Other family members: E2B, E4B, 31B dense.

## 2. Architecture (from B/raw/main/config.json, text_config unless noted)

| Field | Value (VERIFIED) |
|---|---|
| hidden_size | 3840 |
| num_hidden_layers | 48 |
| num_attention_heads | 16 |
| num_key_value_heads (sliding layers) | 8 |
| num_global_key_value_heads (full layers) | 1 |
| head_dim (sliding) | 256 |
| global_head_dim (full) | 512 |
| attention_k_eq_v | true (the card says global layers use unified Keys and Values) |
| intermediate_size | 15360 |
| hidden_activation | gelu_pytorch_tanh |
| rms_norm_eps | 1e-06 (norm type is RMSNorm by the field name; layer layout not in config) |
| vocab_size | 262144 |
| max_position_embeddings | 262144 |
| sliding_window | 1024 |
| tie_word_embeddings | true |
| final_logit_softcapping | 30.0 |
| attention logit soft-capping | No field in config. Treat as absent (VERIFIED absent from config; whether code ignores it is UNVERIFIED) |
| attention_bias / dropout | false / 0.0 |
| use_bidirectional_attention | "vision" (bidirectional only among image tokens) |
| hidden_size_per_layer_input | 0 (no per-layer embeddings, unlike E2B/E4B) |
| num_kv_shared_layers | 0 |
| use_double_wide_mlp | false |
| dtype | bfloat16 |
| bos / eos / pad id (text_config) | 2 / 1 / 0 |

Layer schedule, VERIFIED from `layer_types`: 48 entries, pattern is 5 sliding_attention then 1 full_attention, repeated 8 times. Full attention sits at layer indexes 5, 11, 17, 23, 29, 35, 41, 47 (zero based). The last layer is global.

RoPE, VERIFIED from `rope_parameters`:
- full_attention: rope_type `proportional` (p-RoPE), rope_theta 1000000.0, partial_rotary_factor 0.25.
- sliding_attention: rope_type `default`, rope_theta 10000.0.

Not in config, so UNVERIFIED (read the Hugging Face transformers `gemma4_unified` modeling code before implementing): embedding scaling (Gemma models historically multiply embeddings by sqrt(hidden_size); not confirmed here), exact norm placement (pre/post norms around attention and MLP), QK-norm, query pre-attention scalar, how global layers with 1 KV head and head_dim 512 split the 16 query heads, and how p-RoPE rotates 25 percent of the dims.

Vision and audio blocks, VERIFIED present in config: vision patch_size 16, model_patch_size 48, pooling_kernel_size 3, num_soft_tokens 280, mm_embed_dim 3840; audio hidden_size 640, audio_samples_per_token 640. Not needed for a text-only first slice.

## 3. Files, size, tokenizer, chat template

File list, VERIFIED (HF tree API, https://huggingface.co/api/models/google/gemma-4-12B-it/tree/main):
- `model.safetensors`: ONE file, 23,919,549,408 bytes (about 22.3 GiB), sha256 5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d (LFS oid). Content-length confirmed by HEAD request.
- There is no `model.safetensors.index.json` (fetch returned "Entry not found"). So 1 shard, not sharded.
- `tokenizer.json` 32,169,626 bytes (sha256 cc8d3a0c...760bf; full value in LFS pointer), `tokenizer_config.json` 3,089, `chat_template.jinja` 18,683, `config.json` 4,423, `generation_config.json` 260, `processor_config.json` 1,382, `README.md`, `.gitattributes`. No LICENSE file and no `tokenizer.model` in the repo.
- Size sanity check: 11.95B parameters at 2 bytes each is about 23.9 GB, which matches bf16.

Tokenizer, VERIFIED: `tokenizer.json` is HF tokenizers format, model type BPE, byte_fallback true. Processor class `Gemma4UnifiedProcessor`, padding_side left.

Special tokens, VERIFIED from tokenizer_config.json and config.json:
- `<bos>` id 2, `<eos>` id 1, `<pad>` id 0, `<unk>`, `<mask>`.
- eos_token_id in config.json: [1, 106]. generation_config.json eos_token_id: [1, 106, 50]. Token 106 is the turn-end marker `<turn|>` (name from tokenizer_config `eot_token`; the id-to-string mapping 106 = `<turn|>` is UNVERIFIED, check tokenizer.json vocab). Meaning of id 50 is UNVERIFIED.
- Turn and channel markers: `<|turn>` (start), `<turn|>` (end), `<|channel>` / `<channel|>` (thinking channel), `<|think|>`, tool markers `<|tool_call>`/`<tool_call|>`, `<|tool>`/`<tool|>`, `<|tool_response>`/`<tool_response|>`, string quote `<|"|>`. Media: `<|image>`/`<image|>`, `<|audio>`/`<audio|>`, `<|image|>`, `<|audio|>`, `<|video|>`. Media ids: image 258880, audio 258881, video 258884, boi 255999, boa 256000, eoi 258882, eoa 258883.

generation_config.json, VERIFIED: do_sample true, temperature 1.0, top_k 64, top_p 0.95, suppress_tokens [258883, 258882].

Chat template, VERIFIED (B/raw/main/chat_template.jinja, "Published: 2026-07-09"):
- Output starts with `<bos>`. Roles: `user`, `model` (assistant is renamed to model), `system`/`developer` (native system role).
- Each turn: `<|turn>ROLE\n` + content + `<turn|>\n`.
- Optional system turn first: `<|turn>system\n` then, if thinking is enabled, `<|think|>\n`, then system text, then `<turn|>\n`.
- Generation prompt: `<|turn>model\n`. When thinking is NOT enabled (the default, `enable_thinking` defaults to false) the template then appends an empty thought block `<|channel>thought\n<channel|>`.
- Model replies with thinking use `<|channel>thought\n...<channel|>` before the answer. Tool calls use the `<|tool_call>call:NAME{...}<tool_call|>` form.
- Minimal prompt for one user message "Hi", no thinking: `<bos><|turn>user\nHi<turn|>\n<|turn>model\n<|channel>thought\n<channel|>` (assembled from the template logic above; the exact user-turn rendering was read from the template, but I did not run the renderer, so treat this exact string as UNVERIFIED until tested).

## 4. License

- VERIFIED: Hugging Face metadata and README front matter say `license: apache-2.0`, `license_link: https://ai.google.dev/gemma/docs/gemma_4_license`. README header line: "License: Apache 2.0".
- VERIFIED: That link resolves to a Google page titled "Apache License 2.0" (final URL https://ai.google.dev/gemma/apache_2) whose body is the standard text "Apache License, Version 2.0, January 2004". I confirmed the header and the standard section structure; I did not diff the whole text line by line against the canonical Apache text (UNVERIFIED that it is unmodified).
- Terms (standard Apache 2.0, from general knowledge of that license, UNVERIFIED against this page beyond the header): free use, modification, and redistribution including commercial; patent grant; you must give recipients a copy of the license, keep copyright, patent and attribution notices, and mark files you changed; keep any NOTICE file contents if one exists. No NOTICE file exists in the repo (VERIFIED by file list). Warranty disclaimer applies.
- Important difference from earlier Gemma versions: Gemma 1 to 3 used a custom "Gemma Terms of Use". Gemma 4 here is Apache 2.0 with no click-through terms.
- Prohibited use: a separate page exists, https://ai.google.dev/gemma/prohibited_use_policy (HTTP 200), but it is dated "Last modified: February 21, 2024" and speaks of "Gemma or Model Derivatives", wording tied to the old Gemma Terms of Use. The Gemma 4 model card does not link or reference it, and Apache 2.0 itself has no use restrictions. Whether Google intends it to bind Gemma 4 users is UNVERIFIED. Recommendation: treat it as a courtesy norm and have Drake's counsel or Drake decide; it does not block engineering work.
- Attribution to carry in my repo: model name, source repo URL, license name and URL, and a statement of any changes (for example conversion to my own format).

## 5. Access and download

- VERIFIED: Not gated. HF API reports `"gated": false, "private": false`. Every file URL above returned 200 (or a 302 to the file store) with no token.
- So Drake does NOT need an account, license acceptance, or token. No steps are required from him.
- Download without Python, no credentials, resumable (it is a 23.9 GB file; check free disk first):
  `curl -L -C - -o model.safetensors https://huggingface.co/google/gemma-4-12B-it/resolve/main/model.safetensors`
- This model is an open-weight input I use to build with. The weights are kept on local disk, and nothing is fetched at runtime; AIEN must work fully offline, so the download above is a one-time build-time step.
  Repeat for config.json, generation_config.json, tokenizer.json, tokenizer_config.json, chat_template.jinja, processor_config.json (same URL pattern). Afterwards verify with `sha256sum model.safetensors` against 5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d. git-lfs also works but is unnecessary.
- If Hugging Face ever gates it, Drake would create a free HF account, open the model page, accept the terms, create a read-only token under Settings, then I would add `-H "Authorization: Bearer <token>"` to curl. Not needed today.

## 6. Open items before implementation

1. Read the transformers `gemma4_unified` modeling source (UNVERIFIED items in section 2: embedding scaling, norm layout, QK-norm, global-layer head handling, p-RoPE details).
2. Confirm token id 106 and 50 strings in tokenizer.json.
3. Safetensors tensor names and shapes: not yet read (header of the 23.9 GB file would be needed; can be fetched with an HTTP range request of the first few MB).
4. Verify the chat template by rendering it.
