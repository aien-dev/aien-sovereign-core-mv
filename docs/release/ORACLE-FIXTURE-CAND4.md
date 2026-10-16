# CAND-4 oracle fixture: provenance

The release gate (`scripts/check-release-candidate.sh`) requires two oracle-fixture digests in the `[model]` table of
`release/candidate.toml`. CAND-3's fixture (`crates/aien-inference-abi/fixtures/tinyllama_oracle.*`) belongs to
TinyLlama and does not apply to CAND-4's model (unsloth/Llama-3.2-1B-Instruct, snapshot 5a8abab). This file records
how the CAND-4 fixture was made, with the same procedure as CAND-3's.

## What it is

An independent reference: Hugging Face Transformers runs the model on the CPU in FP32 (eager attention) on one fixed
chat prompt and records 132 activation tensors (embedding, per layer post_rmsnorm, post_rope_q, post_rope_k,
post_attention, post_attn_residual, post_attn_norm, post_swiglu, post_residual for 16 layers, final norm, logits,
last-token logits), the top-10 next-token logits and 16 greedy tokens. It does not use any AIEN code.

The oracle was made once as scratch outside every repository. AIEN does not depend on it at
build, test or run time: no Python and no Hugging Face code is committed, and nothing in the build, the tests or the
runtime runs it. Only its two output files and their digests are kept, as a recorded reference. The architecture
record is the additive amendment `qualification/candidates/CAND-4.amendment-1.toml` (aien-architecture#152); the
frozen CAND-4 manifest itself is unchanged. The release gate (`scripts/check-release-candidate.sh`) reads
only `release/candidate.toml`: it requires the two digests to be present there as sha256 values, and it never reads
the architecture manifest, so it needs no repointing to the amendment file.

| file (crates/aien-inference-abi/fixtures/) | bytes | sha256 |
|---|---|---|
| llama32_1b_oracle.safetensors | 50834760 | bf1d83edc1b802ad25cfbbce7a043658d3d4797bd67955447ce8630b51f515ed |
| llama32_1b_oracle_manifest.json | 23139 | 6df1b39eb351db7f8626734e8de882b572c8678bab924b932640621b21235e82 |

Stored in plain git like CAND-3's fixture (no LFS). The tokenizer and config copies the generator also writes are
not committed (their names would overwrite CAND-3's files); they are the model's own files, pinned by digest in the
`[model]` table: tokenizer.json 6b9e4e7f..., config.json dfb67fd8..., tokenizer_config.json 9ddd255c....

## Inputs

- Model: /home/drakestapleton/.cache/huggingface/hub/models--unsloth--Llama-3.2-1B-Instruct/snapshots/5a8abab4a5d6f164389b1079fb721cfab8d7126c,
  model.safetensors 1ff795ff6a07e6a68085d206fb84417da2f083f68391c2843cd2b8ac6df8538f (the manifest records it as `model_sha256`).
- Prompt: the CAND-3 system and user text ("You are a sovereign AI assistant." / "Explain the role of an operating
  system in one sentence.") in the Llama 3 layout that AIEN renders (`ChatTemplate::Llama3::render`,
  crates/aien-inference-abi/src/tokenizer.rs), BOS added by the tokenizer: 34 tokens
  `128000 128006 9125 128007 271 2675 527 264 46384 15592 18328 13 128009 128006 882 128007 271 849 21435 279 3560 315
  459 10565 1887 304 832 11914 13 128009 128006 78191 128007 271`.

## Generator

- Origin: `scripts/generate_tinyllama_oracle.py` as of `d2cacfa^` (the commit before "Remove the remaining Python
  scripts", d2cacfa; first added in 8c34468), file sha256 6247aa9d4bded40fa8756be278eec3c043ec6e65c506ccbc1f579807ae006b0d.
- It was run as scratch tooling outside every repository (Drake's rule: no Python in any AIEN repo, build, CI or
  tooling; agents may use Python as scratch). The script is not committed. The adapted copy has sha256
  7f7fa648c1aed28560a3d7547d63a788ad8625df0b425f2aed404946a30e96ee.
- Changes are constants only: snapshot path, model id, output file names, prompt text, and the expected prompt tokens,
  first greedy token, 16 greedy tokens and decoded text (each taken from the run's own failure message, as the
  original constants were). The procedure, hooks, dtype, tolerances and file format are unchanged. Diff:

```diff
--- generate_tinyllama_oracle.orig.py
+++ generate_llama32_1b_oracle.py
@@ -22,32 +22,31 @@
 
 PINNED_SNAPSHOT = (
     "/home/drakestapleton/.cache/huggingface/hub/"
-    "models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/"
-    "fe8a4ea1ffedaf415f4da2f062534de366a451e6"
+    "models--unsloth--Llama-3.2-1B-Instruct/snapshots/"
+    "5a8abab4a5d6f164389b1079fb721cfab8d7126c"
 )
 
 CANONICAL_PROMPT = (
-    "<|system|>\n"
-    "You are a sovereign AI assistant.</s>\n"
-    "<|user|>\n"
-    "Explain the role of an operating system in one sentence.</s>\n"
-    "<|assistant|>\n"
+    "<|start_header_id|>system<|end_header_id|>\n\n"
+    "You are a sovereign AI assistant.<|eot_id|>"
+    "<|start_header_id|>user<|end_header_id|>\n\n"
+    "Explain the role of an operating system in one sentence.<|eot_id|>"
+    "<|start_header_id|>assistant<|end_header_id|>\n\n"
 )
 
 EXPECTED_PROMPT_TOKENS = [
-    1, 529, 29989, 5205, 29989, 29958, 13, 3492, 526, 263, 577, 369, 7577, 319,
-    29902, 20255, 29889, 2, 13, 29966, 29989, 1792, 29989, 29958, 13, 9544,
-    7420, 278, 6297, 310, 385, 13598, 1788, 297, 697, 10541, 29889, 2, 13,
-    29966, 29989, 465, 22137, 29989, 29958, 13
+    128000, 128006, 9125, 128007, 271, 2675, 527, 264, 46384, 15592, 18328, 13,
+    128009, 128006, 882, 128007, 271, 849, 21435, 279, 3560, 315, 459, 10565,
+    1887, 304, 832, 11914, 13, 128009, 128006, 78191, 128007, 271
 ]
 
 EXPECTED_GREEDY_16 = [
-    2744, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179,
-    278, 12837, 322, 7047
+    2127, 10565, 1887, 320, 3204, 8, 374, 264, 3241, 6324, 430, 29972,
+    6500, 12035, 5070, 11
 ]
 
 EXPECTED_DECODED_TEXT = (
-    "An operating system (OS) is a software that manages the hardware and software"
+    "An operating system (OS) is a software layer that manages computer hardware resources,"
 )
 
 
@@ -208,8 +207,8 @@
     greedy_next_token_text = tokenizer.decode([greedy_next_token_id])
 
     print(f"Greedy next token: ID={greedy_next_token_id} ({repr(greedy_next_token_text)}), logit={greedy_next_token_logit:.4f}")
-    if greedy_next_token_id != 2744:
-        raise ValueError(f"Greedy next token mismatch: expected 2744, got {greedy_next_token_id}")
+    if greedy_next_token_id != 2127:
+        raise ValueError(f"Greedy next token mismatch: expected 2127, got {greedy_next_token_id}")
 
     top_tokens_meta = []
     for idx, logit in zip(top10_token_ids, top10_logit_values):
@@ -256,7 +255,7 @@
         for name, tensor in activations.items()
     }
 
-    safetensors_path = os.path.join(output_dir, "tinyllama_oracle.safetensors")
+    safetensors_path = os.path.join(output_dir, "llama32_1b_oracle.safetensors")
     print(f"Saving {len(contiguous_activations)} oracle tensors to {safetensors_path}...")
     safetensors.torch.save_file(contiguous_activations, safetensors_path)
     safetensors_sha256 = compute_file_sha256(safetensors_path)
@@ -290,10 +289,10 @@
         }
 
     manifest = {
-        "model_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
+        "model_id": "unsloth/Llama-3.2-1B-Instruct",
         "model_sha256": model_sha256,
         "tokenizer_sha256": tokenizer_sha256,
-        "safetensors_file": "tinyllama_oracle.safetensors",
+        "safetensors_file": "llama32_1b_oracle.safetensors",
         "safetensors_sha256": safetensors_sha256,
         "safetensors_bytes": safetensors_bytes,
         "prompt_text": CANONICAL_PROMPT,
@@ -319,7 +318,7 @@
         "tensors": tensor_catalog,
     }
 
-    manifest_path = os.path.join(output_dir, "tinyllama_oracle_manifest.json")
+    manifest_path = os.path.join(output_dir, "llama32_1b_oracle_manifest.json")
     with open(manifest_path, "w", encoding="utf-8") as f:
         json.dump(manifest, f, indent=2)
     manifest_sha256 = compute_file_sha256(manifest_path)
```

## Run

- Environment: ~/.venv, torch 2.14.0+cu130, transformers 5.17.0, safetensors 0.8.0 (CPU only), on the Spark
  (Linux 7.0.0-1019-nvidia aarch64), 2026-10-06.
- Command: `~/.venv/bin/python3 -I generate_llama32_1b_oracle.py --output-dir out` (about 13 s wall).
- Ran twice into separate directories: both runs gave byte-identical files (the two sha256 above).
- Result: next token 2127 "An" (logit 30.5234; then 791 "The" 28.4298, 32 "A" 26.6412). 16 greedy tokens
  `2127 10565 1887 320 3204 8 374 264 3241 6324 430 29972 6500 12035 5070 11`, text
  "An operating system (OS) is a software layer that manages computer hardware resources,".

## Independence check against the CAND-4 aien-cli

Result: AGREE (decoded text level). On 2026-10-06 20:41Z the CAND-4 aien-cli (sha256
ad6b7eb5330ea19d31d02e68271624d5c964bcbc24983742caae845f62c4c324, the frozen REAL3 build) ran as a daemon on the GB10
(`Backend: NativeTransformerBackend/OmegaGb10Backend`, model and tokenizer digests as above, `AIEN_REQUIRE_BLACKWELL=1`,
private socket and state dirs, under a quietlock hold). One `StreamTurn` control request with the same system and
user messages, `max_tokens` 16, `temperature` 0.0 returned 16 tokens with the text
"An operating system (OS) is a software layer that manages computer hardware resources,", identical to the
oracle's 16-token greedy text.

Limits: aien-cli exposes decoded text, not token ids or logits, so this check compares the 16-token text and count,
not the activations or logits in the fixture. The prompt rendering on the aien-cli side is AIEN's own
(`ChatTemplate::Llama3`); its token ids were not printed. Logit-level parity against this fixture (as the TinyLlama
strict test does for CAND-3) is not run here.
