# DEV-MODEL-0 Phase A: Live Code Audit

Repo: aien-spark/aien-sovereign-core. Audited commit: `7011f2be58a1a08e4774d79ae28db27ad5d17624` (main HEAD on 2026-10-01, unchanged from the brief). Read-only: no builds, no tests were run. All line numbers refer to that SHA. Paths are relative to the repo root; `abi` = `crates/aien-inference-abi/src`, `rt` = `crates/aien-inference-runtime/src`.
External review: Gemini (agy, plan mode); Codex was out of usage. Its citation corrections (weight cache, MaxServingBackend stub, extra template copies, line shifts) are folded in.
Classes: GENERIC (reuse as is), GENERALIZABLE (right idea, hard-coded values to lift), MODEL-SPECIFIC (belongs to one family), LEGACY (remove or quarantine).

## 1. Verdict on the brief (sections 1 and 2)

| Claim | Result | Evidence |
|---|---|---|
| Files in section 1 exist | CONFIRMED (all 11 plus runtime model.rs) | `abi/{capsule,checkpoint,tensor_abi,backend,blackwell_backend,transformer_backend,tokenizer,qwen3_coder,qwen3_moe,qwen3_serve}.rs` |
| ModelCapsule, CapsuleTensor, WeightDType, QuantizationDescriptor exist | CONFIRMED | `abi/capsule.rs:13,57,68,79` |
| "Strict Safetensors ingest" | PARTLY. Strict for TinyLlama only: BF16-only, exact catalog, one file. Extra tensors in the file are silently ignored (loop iterates the catalog, not the header). | `abi/checkpoint.rs:233-384`, dtype gate `:298` |
| "Resident Qwen3 GPU-weight serving" | CONFIRMED, but it is a Mojo shared library for ONE Qwen3-Coder-30B-A3B shape with compile-time constants | `abi/qwen3_serve.rs:1-5,43-60`, `abi/qwen3_coder.rs:23-32` |
| 2.1 `model.rs` is TinyLlama-oriented | CONFIRMED | `rt/model.rs:3,22,74,134,155,207-237` |
| 2.2 `tokenizer.rs` embeds TinyLlama ids, context, template, stops | CONFIRMED | `abi/tokenizer.rs:43-49,70-88,106,143` |
| 2.3 ModelCapsule is the format; no competing one | CONFIRMED, with a caveat: the capsule is single-file (`bytes: Arc<[u8]>` is the whole file). Gemma 4 12B ships as ONE 23.9 GB file (see MODEL_DISCOVERY), so single file works, but whole-file `std::fs::read` into RAM is the current loader (`abi/checkpoint.rs:392-403`). Qwen uses a different, sharded, positional reader (`Shard`, `abi/qwen3_moe.rs:274-370`). | |
| 2.4 Qwen3 is the closest precedent | CONFIRMED, and it is the only model with a sharded reader, resident device weights, QK-norm and per-family constants. It hard-codes every shape as `const`. | `abi/qwen3_coder.rs:23-32` |
| Brief section 1 "Blackwell paths" | REFINED: two separate GPU stacks exist (see 4). The TinyLlama one is CUDA/cuBLAS built by nvcc; the Qwen one is Mojo. | `abi/build.rs`, `abi/cuda/`, `abi/mojo/` |

## 2. Generic infrastructure inventory

| Component | Where | Class | Notes |
|---|---|---|---|
| ModelCapsule, CapsuleTensor, digests | `abi/capsule.rs:13-229` | GENERIC | dtype enum already covers F32/BF16/FP16/FP8/INT8/INT4; `QuantizationDescriptor` uses `String` scheme/packing (not an enum), so identity of quantized variants must be added by caller. |
| Tensor ABI (host/device views, residency) | `abi/tensor_abi.rs:7-241`, C header `abi/cuda/tensor_abi.h` | GENERIC | `MAX_TENSOR_RANK=4`; `DeviceTensor::upload` real only under `cfg(has_blackwell_cuda)` (`:144-161`). |
| Safetensors ingest (catalog validated) | `abi/checkpoint.rs:233-403` | GENERALIZABLE | Validator logic is generic; `tinyllama_catalog()` (`:192-231`) and BF16-only (`:298`) are not. Needs: catalog supplied by family, dtype set, extra-tensor rejection, overlap check, multi-file. |
| Sharded safetensors reader | `abi/qwen3_moe.rs:274-370` (`Shard`, `pub(crate)`), index handling `:396-420` | GENERALIZABLE | Positional reads, header cap, range checks, index path-escape check. Error type is `Qwen3MoeError`; make it family-neutral. |
| TensorBackend trait (CPU oracle) | `abi/backend.rs:82-278`, `ReferenceCpuBackend` `:280-410` | GENERALIZABLE | Fixed op set: rmsnorm, rope, matmul, swiglu, gqa/paged attention, logits. No GELU, no QK-norm, no sliding window, no softcap, no partial rotary, no per-layer head dim. |
| CPU math kernels (f64 accumulate) | `abi/tensor.rs:10-300` | GENERALIZABLE | Good oracle numerics (f64 sums). `rmsnorm` is `x*scale*w` (`:78-94`), not Gemma `(1+w)`; RoPE is full-dim rotate_half with one theta (`:101-154`). |
| Native forward loop (single token, paged KV, batch) | `abi/transformer_backend.rs:496-668,919-1256` | GENERALIZABLE | One uniform Llama block per layer; see section 3. |
| Paged KV cache with COW branches | `crates/aien-kv-cache/src/lib.rs` (1633 lines), `KvPoolConfig` `:46-53` | GENERALIZABLE | One `num_kv_heads` and one `head_dim` for the whole pool. Gemma 4 12B needs per-layer-type shapes (8 x 256 sliding, 1 x 512 global) and a window-bounded sliding cache. Has dtype enum incl. BF16/FP8/FP4 (`:12-26`). |
| Scheduler | `crates/aien-scheduler/src/{lib,sequence}.rs` | GENERIC | No model strings (grep clean). Builds `ScheduledBatch`. `SamplingParams.top_p` exists (`abi/lib.rs:170`) but is never read by any sampler. |
| Backend trait `AienInferenceBackend` | `abi/lib.rs:234-247` | GENERIC | `load_model(&ModelConfig)`, `execute_step`, `manages_kv_cache`. |
| Streaming | `abi/transformer_backend.rs:313-394` (`generate_tokens_streaming`), `rt/model.rs:138-173` | GENERIC | Callback per token, stop list passed in. |
| Branch/context API (create/fork/decode_branch_step) | `abi/transformer_backend.rs:184-311` | GENERIC | Feeds branch-native runtime. |
| Sampling | `abi/tensor.rs:262-307` | GENERALIZABLE | Only argmax and temperature with a toy LCG. No top-k, top-p, penalties, no logit softcap. Gemma defaults are temp 1.0, top_k 64, top_p 0.95. |
| Runtime model owner `EmbeddedModel` | `rt/model.rs:21-204` | MODEL-SPECIFIC (TinyLlama) | Concrete tokenizer and backend types, no abstraction. |
| Inference service wrapper | `rt/service.rs:1-113`, `crates/aien-inference-service/src/*` (small) | GENERALIZABLE | Wraps `EmbeddedModel`. |
| Runtime server and chat format | `crates/aien-runtime/src/server.rs:6,29,216,236`, `control.rs:25-45` | MODEL-SPECIFIC | Calls `format_tinyllama_chat`, stop `[TinyLlamaTokenizer::EOS_TOKEN_ID]`. |
| Model identity | `ModelConfig.model_id: String` (`abi/lib.rs:85`), capsule digests | GENERALIZABLE | No family tag, no tokenizer/template/config digest, no license reference. |

## 3. Hard-coded model assumptions (every one found)

TinyLlama (all LEGACY-or-MODEL-SPECIFIC until a family boundary exists):
- Config: `ModelConfig::tinyllama_1_1b()` 22 layers, 32 heads, 4 KV heads, head_dim 64, hidden 2048, inter 5632, vocab 32000, eps 1e-5, theta 1e4, ctx 2048 (`abi/lib.rs:106-121`). Used at `rt/model.rs:74,134`, `rt/service.rs:24`, `crates/aien-cli/src/commands.rs:1066`, `abi/transformer_backend.rs:433`.
- Catalog: 201 tensors, fixed shapes, `abi/checkpoint.rs:192-231`; `load_safetensors_from_bytes` always uses it (`:386-390`).
- Tokenizer constants BOS 1 / EOS 2 / UNK 0 / ctx 2048 (`abi/tokenizer.rs:43-49`), template `<|system|>..</s>` (`:70-88`), stop = EOS or UNK (`:143`), context error at 2048 (`:106`). More copies of the template: `crates/aien-runtime/src/control.rs:25-45` and twice in `crates/aien-inference-runtime/src/service.rs:48-55,73-80`.
- Stop tokens hard-coded in runtime `[EOS, UNK]` (`rt/model.rs:154-157,187-190`); default `SamplingParams.stop_token_ids = [0,1,2]` (`abi/lib.rs:175-185`).
- Paged KV pool size 2048 blocks hard-coded (`rt/model.rs:100`); KV dtype hard-coded FP32 (`abi/transformer_backend.rs:99,123`).
- Local absolute paths to a TinyLlama HF snapshot under /home/drakestapleton (`rt/model.rs:214-215,237`), env `TINYLLAMA_MODEL_PATH`.
- `ModelConfig::default()` is Nemotron-3.5 30B shaped, not neutral (`abi/lib.rs:148-165`); serde defaults assume Qwen (`vocab 151936`, `kv_heads 8`) (`:68-83`).
- KV helper `for_tinyllama` / `for_qwen2_5_7b` (`aien-kv-cache/src/lib.rs:56-77`).

Llama-block assumptions baked into the forward path (`abi/transformer_backend.rs:507-649`, `abi/weights.rs:174-275`), none of which Gemma 4 satisfies:
- Block order: input_norm, QKV, rope, attn, o_proj, residual, post_attention_norm used as the PRE-MLP norm, SwiGLU, residual. Gemma-style layers add post-attention and post-MLP norms (verify against the real tensor list in Phase E; do not assume).
- Norm: `x * rsqrt(mean(x^2)+eps) * w` (`abi/tensor.rs:78-94`).
- Activation: SiLU gate times up (`abi/tensor.rs:236-241`, `backend.rs:127`). Gemma 4 config says `gelu_pytorch_tanh` (MODEL_DISCOVERY).
- Attention: one scale `1/sqrt(head_dim)` (`tensor.rs:176`), full causal over the whole cache, no window, no softcap, no per-layer topology; GQA ratio `num_heads/num_kv_heads` computed once globally (`:175`).
- Heads: single `num_heads`, `num_kv_heads`, `head_dim` for all layers (`weights.rs:176-186`); Gemma 4 12B has two head shapes.
- RoPE: one theta, full head dim, rotate_half pairing i with i+half (`tensor.rs:101-154`); `f64` `powf` per element per token (slow, but a fine oracle). No partial rotary, no proportional RoPE, no per-layer-type theta.
- Embedding: raw lookup, `token_id % vocab_size` (silently wraps out-of-range ids, `weights.rs:192`, `transformer_backend.rs:526`). No embedding scale.
- LM head: separate required tensor `lm_head.weight` (`weights.rs:683,736`). No weight tying path; Gemma 4 ties embeddings.
- Logits: raw matmul, no softcap (`weights.rs:278-291`); Gemma 4 has final soft-cap 30.0.
- dtype/layout: every weight decoded to host `Vec<f32>` at load (`weights.rs:22-41,692-745`; doc says "not stored twice" but the f32 copy is the working set). 11.95B params x 4 bytes is about 48 GB of host RAM before KV. Layout is PyTorch row-major `[out,in]` (`tensor.rs:4-9`).
- Weight names: HF Llama names `model.layers.N.self_attn.q_proj.weight` etc. (`weights.rs:643-684,705-736`); hard-coded format strings, no name map.
- Tokenizer: wraps the HF Rust `tokenizers` crate with `onig` (`abi/Cargo.toml`); `add_special_tokens=true` decides BOS (`tokenizer.rs:92`), no template engine.
- Context: `max_sequence_length` in config is never enforced in the forward or scheduler (grep: only the tokenizer constant enforces 2048).

Qwen3-Coder (MODEL-SPECIFIC, all `const`): 48 layers, hidden 2048, 32 Q heads, 4 KV, head_dim 128, vocab 151936, eps 1e-6, theta 1e7 (`abi/qwen3_coder.rs:23-32`); FP8 block quant `FP8_BLOCK`; QK-norm per head (`qwen3_coder.rs:61-71`, `rmsnorm_per_head :~290`); MoE shapes `QWEN3_A3B_*` (`abi/qwen3_moe.rs:29-31`); fixed exported C symbols `qwencoder_*` (`abi/qwen3_serve.rs:63-100`). Reusable PATTERN, not code.

Absent everywhere (grep for sliding, softcap, gelu, tie, query_pre, rope_local in abi, cuda, mojo: zero hits): sliding-window attention, soft-capping, GELU, tied embeddings, embedding scaling, per-layer head shapes, partial/proportional RoPE, multiple stop-token sets, text-only guard for a multimodal checkpoint.

## 4. Device execution boundaries

Path 1, TinyLlama via `TensorBackend` (`abi/blackwell_backend.rs`, `abi/cuda/blackwell_gemm.cu`):
- Weights are host f32 `Vec`s. The CUDA side keeps a cache keyed by the HOST pointer (`crates/aien-inference-abi/cuda/blackwell_gemm.cu:82-102`, `internal_get_cached_weight`), so each weight matrix is uploaded once as FP32 and then reused; only activations cross per call. That makes it effectively resident but FP32 (12B params would need about 48 GB of device memory, plus the same on host), and the cache is keyed by pointer address, which is fragile. rmsnorm and rope always run on the CPU backend (`blackwell_backend.rs:263-279`); matmul (`:281`) and logits (`:470-494`, `blackwell_gemv_f32`) use the GPU when available. Swiglu is CPU.
- `paged_attention` (BF16 pool kernel, `cuda/paged_attention_bf16.cu`, supports head_dim up to 256 per its pair loop `:118-119`, so head_dim 512 global layers are out of range) is real; falls back to CPU if unavailable. A `fallback_count` is kept (`blackwell_backend.rs:186`). Hardware init failure prints and falls back silently (`:135-168`).
- Built only if `/usr/local/cuda/bin/nvcc` exists (present on this machine) and `AIEN_FORCE_CPU_STUB` is unset; arch `sm_121`, links cuBLAS and cudart (`abi/build.rs:1-103`). This directly conflicts with the standing rule "no CUDA / libcuda dependence". Flagged for the orchestrator (decision D1 below).

Path 2, Blackwell batch executor (`abi/blackwell_batch_executor.rs`, `crates/aien-inference-abi/cuda/blackwell_layer.cu` 1118 lines): resident BF16 weights with fused QKV and gate/up, persistent workspace, KV scatter into the canonical pool, GPU argmax, one fence per step (`:1-11`). Fully Llama-shaped (`blackwell_batch_executor.rs:281-325`: single `head_dim`, `kv_dim`, `intermediate_dim`). Also CUDA. It is the closest thing to what Phase I wants, but it cannot express Gemma 4 without new kernels.

Path 3, Qwen3-Coder resident serve (`abi/qwen3_serve.rs`, `abi/mojo/qwencoder_serve/*.mojo`, 617+136 lines): one-time upload of all FP8/BF16 weights, then only activation rows cross; CPU does rmsnorm, QK-norm, RoPE, residuals, router; device does fused QKV, o_proj, MoE, KV append, attention, logits (`qwen3_serve.rs:1-5,369-499`). Device and CPU KV are separate caches with a loud guard against mixing (`:424-436`). Uses Mojo `max.gpu.DeviceContext` (`qwencoder_serve.mojo:16`), loaded with `libloading` from `libqwencoder_serve.so` (`qwen3_serve.rs:61-70`). This is the pattern to copy for Gemma (resident weights, parity-checked device calls, always-available CPU fallback).

Other: `MojoGb10Backend` (`abi/mojo_backend.rs`) loads `libaien_kernels.so` from `abi/mojo/lib.mojo` (TinyLlama-era generic kernels). `MaxServingBackend` (`abi/lib.rs:308-375`) holds a `reqwest` client for a Modular MAX endpoint, but its `execute_step` (`:337-373`) only returns synthetic token ids (`151643 + step_id % 100`), no network I/O (LEGACY stub; keep out of the Gemma path).

Sampling happens on host after full logits are read back (`transformer_backend.rs:331-360`, vocab 262144 x f32 = 1 MB per token for Gemma). GPU argmax exists only in the batch executor.

## 5. Python, Mojo, build, CLI inventory

Python files in repo: 11 (confirmed). None is imported or shelled out to from the inference path.
- `scripts/generate_tinyllama_oracle.py`: moved out of this repo to a separate repository, archived on 2026-10-15.
- `scripts/aien_drive.py`, `scripts/browser_mirror_test.py`: removed (no Python in this repo; see the Rust replacement issues). `crates/aien-cli/src/sandbox.rs` and `platform.rs` still shell out to a script outside the repo.
- `modular/nemotron_h_kvexp` (a MAX architecture plugin for Nemotron-H, not referenced by any Rust crate): MOVED 2026-10-04 out of this repo to a separate repository (archived on 2026-10-15), history preserved.
- Python also appears inside shell scripts (`scripts/golden_path.sh`, `mac-platform-proxy.sh`, `sync-standalone-repos.sh:127`). Out of scope for DM0 but they are Python-in-shell, relevant to the standing rule.

Mojo kernels present: `abi/mojo/lib.mojo` (299), `abi/mojo/qwen3_moe/{kernels,lib}.mojo`, `abi/mojo/qwencoder_serve/{qwencoder_kernels,qwencoder_serve,test_kernels}.mojo`, `crates/spark-max-cabi/mojo/spark_max_bridge.mojo`, `crates/spark-aegis/mojo/simd_matcher.mojo`, `imprints/en2-trinity/mojo/en2_kv_adapter.mojo`. Built by `abi/mojo/build.sh` with `mojo build --emit shared-lib`.

Build: Cargo workspace of 34 members (`Cargo.toml:1-37`), resolver 2, edition 2021, release profile LTO + `panic=abort` (`:48-52`). NO `rust-toolchain` file; machine has rustc/cargo 1.98.1 aarch64. `Cargo.lock` is committed (527 packages). No `vendor/` directory; `~/.cargo/registry/cache` holds 979 crates including `tokenizers-0.21.4`, `onig-6.5.3`, `rayon`, `libloading`, `reqwest`. Offline `cargo check -p aien-inference-abi -p aien-inference-runtime --offline` with `AIEN_FORCE_CPU_STUB=1` (skips nvcc, `build.rs:14-18`) looks feasible; whole-workspace check also needs the sibling `../aien-protocols` path dependency (`crates/spark-mail-rs/Cargo.toml:26`, `spark-harvester/Cargo.toml:25-26`), which is present next to this worktree. NOT run in Phase A. CI runs fmt, check, clippy -D warnings and test with `--test-threads=1` (`.github/workflows/ci.yml:39-51`).

CLI entry points: `aien` binary (`crates/aien-cli/src/main.rs`) only starts a daemon (`commands::run_daemon_server`, `main.rs:44`). There is NO `aien model import` command. Model discovery is `resolve_daemon_manifest()` scanning `$AIEN_MODEL_DIR`, `./models`, `~/models` for any `.safetensors` (`commands.rs:970-1027`), then `load_production_weights` assumes TinyLlama config (`:1066`) and on ANY load error silently substitutes synthetic reference test weights (`:1086-1098`; the same silent fallback exists at `rt/model.rs:125-136` and in `weights.rs:641` for the bytes loader). That silent substitution must go before any new model is wired. Examples for Qwen: `abi/examples/qwen3_coder_chat.rs`, `qwen3_serve_check.rs`, `qwen3_parity_check.rs`, `qwen3_noise_floor.rs`.

Tests that pin TinyLlama behavior and must keep passing through Phase B: `abi/tests/{tinyllama_parity,transformer_parity,paged_transformer_parity,branch_client_harness,blackwell_batch_executor_parity,paged_attention_parity,bf16_paged_attention,qwen3_moe_gpu}.rs`, `rt/tests/runtime_tests.rs`, `crates/aien-runtime/tests/runtime_end_to_end_tests.rs`. Oracle fixtures: `abi/fixtures/tinyllama_oracle.safetensors` + manifest.

## 6. Ten most important findings

1. No model-family boundary exists; TinyLlama types are concrete in the runtime (`rt/model.rs:3,22`), runtime server (`server.rs:6,216`) and CLI (`commands.rs:1055-1070`).
2. Tokenizer is a TinyLlama-only struct with associated constants; a second copy of the chat template lives in `aien-runtime/src/control.rs`.
3. The forward pass has no way to express Gemma 4: no sliding window, no two head shapes, no GELU, no softcap, no tied head, no embedding scale, no partial/proportional RoPE.
4. The paged KV pool and the BF16 attention kernel assume one head shape for all layers and head_dim <= 256; Gemma 4 12B global layers are 1 KV head x head_dim 512.
5. Whole-model FP32 host expansion (about 48 GB for 12B) is the current load path; BF16-resident is needed and only exists on the Qwen path.
6. The only GPU path for non-MoE models is CUDA/cuBLAS via nvcc (conflicts with the no-CUDA rule); the resident Mojo path exists only for Qwen3-Coder.
7. Silent synthetic-weights fallback on load failure in three places violates "fail loudly".
8. Checkpoint ingest is TinyLlama-catalog, BF16-only, whole-file, and ignores extra tensors; the sharded reader is private to Qwen code.
9. `top_p`, `max_sequence_length`, `block_size` configuration fields are declared but not enforced or used by samplers/forward.
10. No Python is in the inference path; Python exists as offline oracle (`scripts/generate_tinyllama_oracle.py`), harnesses, and a MAX plugin. No `aien model import` CLI exists.

## 7. Decisions needed before Phase B/F (for the orchestrator)

- D1 GPU path for Gemma (Phase F/I): (a) extend the existing CUDA/cuBLAS stack (fast to start, conflicts with the standing no-CUDA rule and with the sovereignty direction), (b) follow the Qwen precedent, a Mojo resident-weights library (no CUDA toolkit, matches rule; Mojo is available). Recommendation: (b). Needs a one-line orchestrator confirmation because brief section 15 says "reuse BlackwellGb10Backend".
- D2 Oracle without Python: parity (Phase E/H) needs an independent reference. Options: generate artifacts outside the repo with an external tool and commit only digests/artifacts (brief section 4), or hand-derive a second C implementation. Recommendation: external offline artifacts, clearly recorded.
  - Tie to ADR 0024 ("Rust is scaffolding, Omega is destination, C where hardware-justified", aien-spark/aien-architecture ADR 0024): a second C implementation is acceptable only where hardware justifies C; otherwise the oracle stays outside the repo as external offline artifacts.
- D3 Multimodal: the Gemma 4 12B checkpoint is `Gemma4UnifiedForConditionalGeneration` with audio and vision config (MODEL_DISCOVERY). Recommendation: text-only import that FAILS LOUDLY on unsupported modality tensors/features, never ignores them silently.

## 8. Phase B-I plan (file ownership, order, first PRs)

Ownership (no file in two lanes; new files named; shared files get exactly one owner, others request changes via board):

| Lane | Owns |
|---|---|
| A model family / runtime | `abi/lib.rs` (ModelConfig region only), new `abi/model_family.rs`, `abi/gemma4.rs` (contract, Phase D contract struct), `rt/model.rs`, `rt/service.rs`, `aien-runtime/src/{server,control}.rs` (remove template copy by calling B's trait) |
| B tokenizer / template | `abi/tokenizer.rs` plus new `abi/model_tokenizer.rs`, `abi/chat_template.rs`, `abi/tests/tokenizer_*.rs`, fixtures `abi/fixtures/gemma4_tokenizer/` (artifact digests only) |
| C importer / catalog | `abi/checkpoint.rs`, `abi/capsule.rs`, new `abi/gemma4_catalog.rs`, new `abi/safetensors_shard.rs` (lifted from qwen3_moe `Shard`; Qwen file keeps a thin re-export, touched only via C's PR), new `aien-cli/src/model_import.rs` + the single `main.rs` subcommand registration, `abi/tests/import_*.rs` |
| D CPU reference | new `abi/gemma4_cpu.rs`, new `abi/gemma_ops.rs` (GELU-tanh, (1+w) norm, sliding attention, partial RoPE, softcap), `abi/weights.rs` (additive struct only), `abi/tensor.rs` untouched (additions go in gemma_ops) |
| E oracle / parity | `abi/tests/gemma4_parity_*.rs`, `abi/fixtures/gemma4/`, `docs/dev-model-0/DEV_MODEL_0_PARITY_PROTOCOL.md`; no Python files |
| F Blackwell | `abi/blackwell_*.rs`, `abi/cuda/`, `abi/mojo/` (new `mojo/gemma4_serve/`), new `abi/gemma4_serve.rs`, `crates/aien-kv-cache/src/lib.rs` (heterogeneous layers), `abi/build.rs` |
| G DevelopmentModel / RSI | new crate `crates/aien-dev-model` (single removable crate), `aien-runtime/src/rsi.rs`, workspace `Cargo.toml` member line |
| H benchmark quiescence | new `scripts/dm0-quiesce.sh`, `docs/dev-model-0/DEV_MODEL_0_BENCHMARK_PROTOCOL.md`, hooks into the existing quiet-flag protocol only |
| I off-ramp | `docs/dev-model-0/DEV_MODEL_0_OFF_RAMP.md`, new `scripts/dm0-removal-qualify.sh`, new `crates/aien-dev-model/tests/removal_*.rs` coordinated with G |
| J review / licensing | `NOTICE`, `ATTRIBUTION.md`, `LICENSES/`, `docs/dev-model-0/DEV_MODEL_0_LICENSE_AUDIT.md`; read-only elsewhere |

Shared hot spot: `abi/lib.rs` (A) re-exports every module; every lane adds a `pub mod` line there. Rule: new modules are announced on the board, A applies the one-line `pub mod` edits, or each lane's PR touches only its own `pub mod` line and merges serially.

Order and parallelism:
1. Start now, independent: B (trait + TinyLlama impl), C (catalog + shard reader), J, H (design), I (design), G (interface only).
2. A starts with B's trait merged (needs `ModelTokenizer` type), runs in parallel with C.
3. D starts when A's contract struct and C's catalog types are merged.
4. E starts when D has single-token output and the external oracle artifacts exist (D2).
5. F starts after DM4 (CPU forward) passes; kernels can be prototyped in parallel against E's intermediates.
6. G integrates after DM8 interface can run on CPU; H and I proceed in parallel with F using a stub model.

First three PR-sized steps:
1. B: introduce `trait ModelTokenizer` (encode, decode, format_chat, is_stop_token, stop_token_ids, bos/eos, vocab_size, max_context, digests) and make `TinyLlamaTokenizer` implement it with identical behavior; delete the duplicate template in `aien-runtime/control.rs` by routing through the trait. Evidence: existing tokenizer tests plus `runtime_end_to_end_tests` unchanged.
2. A: add `ModelFamily` enum and `ModelDescriptor` (config, tensor catalog fn, tokenizer ctor, stop tokens, context rules); make `EmbeddedModel` hold `Box<dyn ModelTokenizer>` and take a descriptor; remove `tinyllama_1_1b()` calls and the hard-coded paths from `rt/model.rs`; replace the three silent synthetic-weight fallbacks with explicit errors (keep a clearly named test-only constructor). Evidence: `tinyllama_parity`, `runtime_tests` pass unchanged.
3. C: lift `Shard` into a neutral `safetensors_shard.rs`, add a catalog-from-config generator (pure function from a Gemma4 config.json to the expected tensor list) with negative tests (extra tensor, wrong dtype, overflow offsets, missing tensor), no weights needed. Evidence: unit tests on synthetic safetensors.

No production code was changed in Phase A. Gate DM0_AUDIT_PASS is claimed only when this document is merged.
