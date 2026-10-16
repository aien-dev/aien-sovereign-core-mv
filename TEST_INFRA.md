# Test Infrastructure Specification: AIEN Sovereign Core Real-Model Execution

## 1. Architectural Overview and Verification Principles

This document defines the opaque-box test infrastructure and four-tier test methodology for the AIEN Sovereign Core TinyLlama real-model execution system. The test suite exercises the system through public interfaces, binary CLIs, and boundary contracts without accessing internal private states or facade mocks.

### Core Testing Invariants
1. Opaque-Box Requirement-Driven Verification: All test assertions derive directly from PROJECT.md, ORIGINAL_REQUEST.md, Hugging Face reference architectures, and hardware specifications. Tests inspect observable outputs (return types, stdout, stderr, exit codes, tensor buffers, generated tokens).
2. Authoritative Ground Truth: Expected values originate from the Hugging Face Transformers CPU FP32 reference execution of `TinyLlama/TinyLlama-1.1B-Chat-v1.0`, official tokenizer manifests, and mathematical properties of transformers.
3. Loud Failure Contract: System components must fail loudly on invalid inputs with descriptive error variants (`MissingTensor`, `ShapeMismatch`, `DtypeMismatch`, `OffsetOutOfBounds`, `InvalidHeader`). Silent fallbacks to synthetic or random weights are strictly forbidden.
4. Zero Disk Secrets: Tests verify that zero plaintext secret files (`.env`, `.env.*`) exist in working directories. Secrets resolve exclusively from the hardware TPM vault (`atlas-vault`).
5. Zero Unhandled Panics: Negative tests assert that malformed inputs produce graceful error results or non-zero exit codes with clean error diagnostics, never segmentation faults or unexpected panics.

---

## 2. Four-Tier Testing Methodology

The test matrix covers all 18 features across four structured tiers:

```
+-----------------------------------------------------------------------------------+
| Tier 1: Feature Coverage (>= 5 per feature, 90 test cases)                        |
| Direct validation of functional requirements, happy paths, and primary behaviors. |
+-----------------------------------------------------------------------------------+
| Tier 2: Boundary & Corner Cases (>= 5 per feature, 90 test cases)                 |
| Extreme values, empty inputs, malformed headers, off-by-one indices, type drift.  |
+-----------------------------------------------------------------------------------+
| Tier 3: Cross-Feature Interactions (>= 18 pairwise combinations)                  |
| End-to-end data pipeline combinations spanning checkpoint, tokenizer, and compute.|
+-----------------------------------------------------------------------------------+
| Tier 4: Real-World Scenarios (>= 9 operational workflows)                         |
| Complete production journeys: single prompt, multi-turn, benchmark, doctor, PR.   |
+-----------------------------------------------------------------------------------+
```

---

## 3. Feature Matrix and Test Inventory

### Feature Inventory Mapping (N=18)
- F1: Strict Safetensors Loader
- F2: Loud Validation & Catalog
- F3: Dual Weight Storage
- F4: Pure Rust Tokenizer
- F5: Chat Template & Special Tokens
- F6: Encode & Decode Engine
- F7: Reference Oracle Script
- F8: Deterministic Oracle Fixtures
- F9: Algorithmic Core Alignment
- F10: Multi-Stage Parity Test
- F11: Decoupled TensorBackend Trait
- F12: ReferenceCpuBackend
- F13: Mojo GB10 C-ABI Kernels (removed 2026-10-15)
- F14: MojoGb10Backend Implementation (removed 2026-10-15)
- F15: Neutral Benchmark Driver
- F16: Concurrency Sweeps & Telemetry
- F17: Hardware & Secrets Compliance
- F18: PR Lifecycle & Cortex Receipt

---

### Tier 1: Feature Coverage (5 Test Cases per Feature = 90 Tests)

#### Feature F1: Strict Safetensors Loader
- TC-F1-01: Load valid safetensors header with valid JSON metadata and verify tensor count.
- TC-F1-02: Verify rejection of empty file (0 bytes) with descriptive error.
- TC-F1-03: Verify rejection of truncated safetensors header (bytes < 8).
- TC-F1-04: Verify rejection of invalid JSON within safetensors header boundary.
- TC-F1-05: Verify header length prefix larger than actual file size fails with InvalidHeader.

#### Feature F2: Loud Validation & Catalog
- TC-F2-01: Verify detection of missing tensor `model.embed_tokens.weight` returning `CheckpointError::MissingTensor`.
- TC-F2-02: Verify detection of missing layer projection `model.layers.0.self_attn.q_proj.weight`.
- TC-F2-03: Verify shape mismatch on `lm_head.weight` (expected [32000, 2048], actual [32000, 1024]) returns `ShapeMismatch`.
- TC-F2-04: Verify dtype mismatch (tensor stored as F32 instead of BF16) returns `DtypeMismatch`.
- TC-F2-05: Verify out-of-bounds byte range (data_offsets exceed file length) returns `OffsetOutOfBounds`.

#### Feature F3: Dual Weight Storage
- TC-F3-01: Verify loaded checkpoint contains populated FP32 weight map for reference execution.
- TC-F3-02: Verify loaded checkpoint retains raw BF16 byte vectors for GPU acceleration.
- TC-F3-03: Verify FP32 decoded values correctly represent BF16 bit patterns (bit shift `u16 << 16` to `f32`).
- TC-F3-04: Verify tensor shapes in checkpoint match both FP32 and BF16 allocations.
- TC-F3-05: Verify memory isolation between FP32 and BF16 buffers (modifying one does not mutate the other).

#### Feature F4: Pure Rust Tokenizer
- TC-F4-01: Instantiate `TinyLlamaTokenizer` from valid `tokenizer.json` file.
- TC-F4-02: Verify tokenizer loads without invoking Python runtime or child processes.
- TC-F4-03: Verify vocabulary size equals 32,000.
- TC-F4-04: Verify rejection of missing `tokenizer.json` path with descriptive file error.
- TC-F4-05: Verify rejection of corrupt JSON in `tokenizer.json` with descriptive parse error.

#### Feature F5: Chat Template & Special Tokens
- TC-F5-01: Verify special token ID BOS equals 1 (`<s>`).
- TC-F5-02: Verify special token ID EOS equals 2 (`</s>`).
- TC-F5-03: Verify special token ID UNK equals 0 (`<unk>`).
- TC-F5-04: Format chat prompt with system message and user message, asserting exact template structure.
- TC-F5-05: Format chat prompt with None system message, asserting default user-only template structure.

#### Feature F6: Encode & Decode Engine
- TC-F6-01: Encode canonical test string `"Hello world"` to token IDs.
- TC-F6-02: Decode token IDs back to original text `"Hello world"`.
- TC-F6-03: Verify EOS token (ID 2) terminates autoregressive generation sequence.
- TC-F6-04: Verify round-trip encoding and decoding across multilingual and ASCII inputs.
- TC-F6-05: Verify decode of empty token slice returns empty string without error.

#### Feature F7: Reference Oracle Script
- TC-F7-01: (retired) the oracle generator is no longer kept in AIEN; only its recorded outputs remain, as test fixtures. This repo holds no Python.
- TC-F7-02: Verify script CLI options `--model-dir` and `--output-dir`.
- TC-F7-03: Verify script specifies CPU device and FP32 torch precision.
- TC-F7-04: Verify script registers hooks on all 22 layers (RMSNorm, RoPE, Attention, SwiGLU, final norm).
- TC-F7-05: Verify script exports JSON manifest with SHA-256 digests.

#### Feature F8: Deterministic Oracle Fixtures
- TC-F8-01: Verify `tinyllama_oracle_manifest.json` schema and field presence.
- TC-F8-02: Verify manifest references 46 prompt token IDs matching canonical golden sequence.
- TC-F8-03: Verify greedy next token ID in manifest equals 2744 (`"An"`).
- TC-F8-04: Verify top-5 tokens in manifest match rank order `[2744, 1576, 6716, 7094, 797]`.
- TC-F8-05: Verify 16-step greedy decode text in manifest matches golden operating system definition.

#### Feature F9: Algorithmic Core Alignment
- TC-F9-01: Verify RoPE implementation pairs coordinate `i` with `i + half_dim` (canonical `rotate_half`).
- TC-F9-02: Verify RoPE does NOT pair adjacent coordinates `2*i` and `2*i + 1`.
- TC-F9-03: Verify matrix multiplication computes dot products against row slices of shape `[out_dim, in_dim]`.
- TC-F9-04: Verify decode step sequence position matches current token index (not offset by +1).
- TC-F9-05: Verify RMSNorm epsilon defaults to `1e-5` for TinyLlama.

#### Feature F10: Multi-Stage Parity Test
- TC-F10-01: Verify Stage 1 tokenizer parity asserts 100% token ID match against oracle.
- TC-F10-02: Verify Stage 2 shape parity asserts exact dimensions across all 22 layers.
- TC-F10-03: Verify Stage 3 activation parity verifies absolute tolerance <= 1e-4.
- TC-F10-04: Verify Stage 3 activation parity verifies cosine similarity > 0.9999.
- TC-F10-05: Verify Stage 4 greedy parity asserts top-1 token ID equals oracle top-1.

#### Feature F11: Decoupled TensorBackend Trait
- TC-F11-01: Verify `TensorBackend` trait exposes `rmsnorm` method with explicit buffer slices.
- TC-F11-02: Verify `TensorBackend` trait exposes `apply_rope` with head and position arguments.
- TC-F11-03: Verify `TensorBackend` trait exposes `matmul_vec` with `[out_dim, in_dim]` contract.
- TC-F11-04: Verify `TensorBackend` trait exposes `swiglu` activation method.
- TC-F11-05: Verify `TensorBackend` trait exposes `gqa_attention` supporting 8:1 query-to-KV ratio.

#### Feature F12: ReferenceCpuBackend
- TC-F12-01: Instantiate `ReferenceCpuBackend` and verify backend name reporting.
- TC-F12-02: Execute RMSNorm on unit vector and verify mathematical correctness.
- TC-F12-03: Execute SwiGLU forward pass with known inputs and verify output values.
- TC-F12-04: Execute GQA attention with single query token against cached KV vectors.
- TC-F12-05: Execute `compute_logits` and verify output dimension matches vocabulary size (32000).

#### Feature F13: Mojo GB10 C-ABI Kernels
- Removed on 2026-10-15 with Modular MAX and Mojo; its tests were deleted.

#### Feature F14: MojoGb10Backend Implementation
- Removed on 2026-10-15 with Modular MAX and Mojo; its tests were deleted.

#### Feature F15: Neutral Benchmark Driver
- TC-F15-01: Verify benchmark crate `benchmarks/crates/bench_apples_to_apples` exists in workspace.
- TC-F15-02: Verify benchmark driver accepts `--engines` argument (AIEN only since 2026-10-15).
- TC-F15-03: Verify benchmark driver accepts `--concurrency` sweep argument.
- TC-F15-04: Verify benchmark driver executes engines sequentially with cool-down pauses.
- TC-F15-05: Verify benchmark driver verifies output token correctness against reference oracle.

#### Feature F16: Concurrency Sweeps & Telemetry
- TC-F16-01: Verify concurrency levels sweep through C = 1, 2, 4, 8, 16, 32, 64.
- TC-F16-02: Verify prompt length is fixed at 128 input tokens and generation at 128 output tokens.
- TC-F16-03: Verify TTFT metric collection computes p50, p95, and p99 percentiles.
- TC-F16-04: Verify ITL metric collection computes p50, p95, and p99 percentiles.
- TC-F16-05: Verify telemetry records resident memory (RSS GB), GPU power (W), and energy (J/token).

#### Feature F17: Hardware & Secrets Compliance
- TC-F17-01: Verify absence of plaintext `.env` or `.env.*` files across the workspace.
- TC-F17-02: Verify secrets resolution directs to `atlas-vault` in-memory lookup.
- TC-F17-03: Verify codebase contains zero em dashes (`\u2014`) and zero en dashes (`\u2013`).
- TC-F17-04: Verify absence of banned AI buzzwords (`delve`, `tapestry`, `testament`, `beacon`, `pivotal`).
- TC-F17-05: Verify execution user check requires `drakestapleton` on DGX Spark.

#### Feature F18: PR Lifecycle & Cortex Receipt
- TC-F18-01: Verify active git branch is `feat/real-model-execution-tinyllama`.
- TC-F18-02: Verify PR automation uses GitHub CLI (`gh pr create`, `gh pr merge`).
- TC-F18-03: Verify squash merge strategy is enforced (`--squash --delete-branch`).
- TC-F18-04: Verify Cortex memory integration target is `http://127.0.0.1:18080` in space `atlas-memory`.
- TC-F18-05: Verify post-merge receipt schema includes commit SHA, timestamp, and benchmark summary.

---

### Tier 2: Boundary & Corner Cases (5 Test Cases per Feature = 90 Tests)

#### Feature F1 (Safetensors Loader Boundaries)
- TC-F1-B01: Safetensors file with 0-byte header length field (8 zero bytes).
- TC-F1-B02: Header JSON containing extra unknown metadata fields (must ignore gracefully).
- TC-F1-B03: Header length field specifying `u64::MAX` bytes (must reject without memory allocation panic).
- TC-F1-B04: Safetensors header with trailing junk bytes after valid JSON.
- TC-F1-B05: File path pointing to directory instead of regular file.

#### Feature F2 (Validation Boundaries)
- TC-F2-B01: Tensor with shape containing zero dimension (e.g. `[0, 2048]`).
- TC-F2-B02: Layer index out of bounds (e.g. `model.layers.99.input_layernorm.weight`).
- TC-F2-B03: Byte range offset where `begin == end` (0 elements for non-empty shape).
- TC-F2-B04: Byte range offset where `begin > end`.
- TC-F2-B05: Header specifying 200 out of 201 required tensors (exact catalog boundary).

#### Feature F3 (Dual Storage Boundaries)
- TC-F3-B01: Conversion of BF16 subnormal floating point values to FP32.
- TC-F3-B02: Conversion of BF16 positive and negative infinity to FP32.
- TC-F3-B03: Conversion of BF16 NaN values preserving NaN semantics.
- TC-F3-B04: Alignment verification: raw BF16 buffer starts at 2-byte aligned address.
- TC-F3-B05: Allocation of maximum single tensor buffer (`lm_head.weight`, 32000 * 2048 * 2 bytes = 131 MB).

#### Feature F4 (Tokenizer Boundaries)
- TC-F4-B01: Tokenizing empty string `""` returns empty token array.
- TC-F4-B02: Tokenizing string of 100,000 continuous whitespace characters.
- TC-F4-B03: Tokenizing string with invalid UTF-8 byte sequences gracefully handled or replaced.
- TC-F4-B04: Tokenizing string containing only emojis and zero-width joiners.
- TC-F4-B05: Tokenizing text exceeding max sequence length (2048 tokens).

#### Feature F5 (Chat Template Boundaries)
- TC-F5-B01: System prompt with empty string `""`.
- TC-F5-B02: User prompt with empty string `""`.
- TC-F5-B03: User prompt containing raw template tokens (`<|system|>`, `<|user|>`, `</s>`).
- TC-F5-B04: Deep multi-turn conversation with 50 alternating turns.
- TC-F5-B05: Chat prompt containing newlines and escape characters (`\r\n`, `\t`, `\0`).

#### Feature F6 (Encode/Decode Boundaries)
- TC-F6-B01: Decoding token ID sequence containing unknown ID (e.g. `u32::MAX`).
- TC-F6-B02: Decoding single token ID corresponding to partial multi-byte UTF-8 character.
- TC-F6-B03: Decoding consecutive EOS tokens (`[2, 2, 2]`).
- TC-F6-B04: Encoding string with single character repeated 2048 times.
- TC-F6-B05: Stopping check on sequence where EOS appears at position 0.

#### Feature F7 (Oracle Script Boundaries)
- TC-F7-B01: Oracle generation invoked with non-existent model path fails with exit code 1.
- TC-F7-B02: Oracle generation with read-only output directory reports permission error.
- TC-F7-B03: Oracle generation with batch size > 1 asserts unsupported configuration.
- TC-F7-B04: Oracle generation with float16 precision instead of float32 asserts validation error.
- TC-F7-B05: Oracle generation interrupt (SIGINT) cleans up partial temporary files.

#### Feature F8 (Oracle Fixture Boundaries)
- TC-F8-B01: Manifest with modified SHA-256 hash fails integrity check.
- TC-F8-B02: Oracle safetensors file truncated by 1 byte fails hash verification.
- TC-F8-B03: Oracle logits tensor contains zero NaN or Infinite values.
- TC-F8-B04: Oracle top-10 logits strictly descending in value.
- TC-F8-B05: Oracle intermediate activations have zero NaN values across all 22 layers.

#### Feature F9 (Algorithmic Alignment Boundaries)
- TC-F9-B01: RoPE execution at sequence position 0 (angles evaluate to 0, cos=1, sin=0).
- TC-F9-B02: RoPE execution at maximum sequence position 2047.
- TC-F9-B03: Matrix multiplication with all-zero input vector produces all-zero output.
- TC-F9-B04: Matrix multiplication with identity-like weights preserves inputs.
- TC-F9-B05: RMSNorm with all-zero input vector handles epsilon without division by zero.

#### Feature F10 (Parity Test Boundaries)
- TC-F10-B01: Parity check with absolute difference at exactly 1e-4 passes.
- TC-F10-B02: Parity check with absolute difference at 1.0001e-4 fails with descriptive message.
- TC-F10-B03: Parity check with cosine similarity at exactly 0.9999 passes.
- TC-F10-B04: Parity check with cosine similarity at 0.99989 fails.
- TC-F10-B05: Early divergence detection: inject deliberate delta at Layer 3, verify failure logs Layer 3.

#### Feature F11 (TensorBackend Trait Boundaries)
- TC-F11-B01: `rmsnorm` with slice length 0 handled without panic.
- TC-F11-B02: `matmul_vec` with dimensions `[1, 1]`.
- TC-F11-B03: `gqa_attention` with single key-value token (seq_len = 1).
- TC-F11-B04: `swiglu` with negative input values verifying SiLU non-linearity.
- TC-F11-B05: `compute_logits` with hidden dimension 2048 and vocab 32000.

#### Feature F12 (ReferenceCpuBackend Boundaries)
- TC-F12-B01: Extreme float values in input vector (`1e30`, `-1e30`) checking overflow safety.
- TC-F12-B02: Attention with identical query and key producing uniform attention weights.
- TC-F12-B03: Attention with large negative score masking out non-attended tokens.
- TC-F12-B04: KV cache with capacity exactly at max sequence length (2048).
- TC-F12-B05: Argmax sampling with uniform logits selecting lowest index deterministically.

#### Feature F13 (Mojo GB10 Kernels Boundaries)
- Removed on 2026-10-15 with Modular MAX and Mojo; its tests were deleted.

#### Feature F14 (MojoGb10Backend Boundaries)
- Removed on 2026-10-15 with Modular MAX and Mojo; its tests were deleted.

#### Feature F15 (Neutral Benchmark Driver Boundaries)
- TC-F15-B01: Target engine endpoint unreachable reports connection error and moves to next.
- TC-F15-B02: Target engine returns HTTP 500 reports server error without driver crash.
- TC-F15-B03: Concurrency sweep with C=0 rejected by CLI argument parser.
- TC-F15-B04: Output generation truncated early by server recorded as partial sample.
- TC-F15-B05: Cool-down timer interrupted by signal terminates gracefully.

#### Feature F16 (Concurrency Sweeps Boundaries)
- TC-F16-B01: Concurrency C=64 under thread pool exhaustion maintains queue ordering.
- TC-F16-B02: Request latency exceeding 60 seconds triggers timeout telemetry event.
- TC-F16-B03: Power telemetry query failure during benchmark recorded as null, not zero.
- TC-F16-B04: GPU temperature above 80 C triggers thermal warning log.
- TC-F16-B05: Telemetry calculations with 1 single request sample calculate percentiles accurately.

#### Feature F17 (Compliance Boundaries)
- TC-F17-B01: Hidden `.env` files in nested subdirectories detected and flagged.
- TC-F17-B02: Vault lookup for non-existent key returns clean None without panic.
- TC-F17-B03: String containing em dash Unicode character rejected by style checker.
- TC-F17-B04: String containing en dash Unicode character rejected by style checker.
- TC-F17-B05: Execution under unauthorized username terminates with exit code 1.

#### Feature F18 (PR Lifecycle Boundaries)
- TC-F18-B01: Attempting direct push to `main` branch rejected by repository policy.
- TC-F18-B02: PR creation with unstaged git modifications prevented.
- TC-F18-B03: Cortex write to unreachable IP address logs warning and continues.
- TC-F18-B04: Cortex write with invalid bearer token logs authentication error.
- TC-F18-B05: Git squash merge on branch with failing CI prevented.

---

### Tier 3: Cross-Feature Interactions (>= 18 Pairwise Combinations)

1. Pair P1 (F1 + F2): Safetensors loading of a file missing 1 of 201 tensors triggers loud `CheckpointError::MissingTensor` without entering execution.
2. Pair P2 (F1 + F3): Safetensors loading with valid BF16 weights populates both decoded FP32 weights and raw BF16 weights in `LoadedCheckpoint`.
3. Pair P3 (F2 + F9): Validating tensor catalog shapes enforces `[out_dim, in_dim]` layout used directly by aligned `matmul_vec`.
4. Pair P4 (F4 + F5): Pure Rust tokenizer loading `tokenizer.json` formats chat template and pins BOS=1, EOS=2, UNK=0.
5. Pair P5 (F4 + F6): Pure Rust tokenizer encodes prompt string, then decode reverses the token sequence back to original text.
6. Pair P6 (F5 + F6): Chat formatted prompt encoded to tokens terminates on EOS token ID 2 during autoregressive decode.
7. Pair P7 (F7 + F8): Reference Python oracle script execution produces `tinyllama_oracle.safetensors` matching `tinyllama_oracle_manifest.json` hashes.
8. Pair P8 (F8 + F10): Multi-stage parity test loads oracle safetensors fixture and validates Stage 1 through Stage 5 sequentially.
9. Pair P9 (F9 + F10): Aligned RoPE `rotate_half` and row-major matmul pass Stage 3 activation parity (cosine > 0.9999) without layer drift.
10. Pair P10 (F9 + F12): Reference CPU backend executing aligned math passes all 5 parity stages against Hugging Face oracle.
11. Pair P11 (F3 + F13): removed 2026-10-15 with Mojo.
12. Pair P12 (F11 + F12): `ReferenceCpuBackend` implements `TensorBackend` trait and processes full forward pass token evaluation.
13. Pair P13 (F11 + F14): removed 2026-10-15 with Mojo.
14. Pair P14 (F13 + F14): removed 2026-10-15 with Mojo.
15. Pair P15 (F10 + F14): removed 2026-10-15 with Mojo.
16. Pair P16 (F15 + F16): Neutral benchmark driver executes concurrency sweep C=1..64 on AIEN collecting TTFT and ITL.
17. Pair P17 (F16 + F17): Telemetry collection records GPU power and memory while maintaining zero disk secrets in memory.
18. Pair P18 (F15 + F18): Benchmark completion writes structured results and triggers autonomous PR squash merge with Cortex memory receipt.

---

### Tier 4: Real-World Scenarios (>= 9 Workflows)

1. Scenario SC-01 (Golden Prompt Single-Turn Inference):
   End user submits golden operating system prompt. System loads checkpoint, tokenizes chat prompt, executes forward pass on `ReferenceCpuBackend`, and emits greedy next token 2744 (`"An"`), matching Hugging Face oracle 100%.
2. Scenario SC-02 (16-Step Autoregressive Generation):
   End user initiates 16-step text generation. System autoregressively decodes tokens `[2744, 13598, 1788, 313, 3267, 29897, 338, 263, 7047, 393, 767, 1179, 278, 12837, 322, 7047]`, outputting text `"An operating system (OS) is a software that manages the hardware and software"`.
3. Scenario SC-03 (Corrupted Checkpoint Recovery & Rejection):
   End user supplies a safetensors file with missing layer weights. System catches `CheckpointError::MissingTensor` at initialization, logs descriptive error, and aborts startup cleanly without panicking.
4. Scenario SC-04 (CLI Doctor Inspection):
   Operator executes `aien-cli --doctor`. System validates platform context, execution surface detection, filesystem structure, and reports operational health.
5. Scenario SC-05 (CLI Status & Version Check):
   Operator executes `aien-cli --version` and `aien-cli --status`. System detects host surface, confirms release tag, and prints formatted status without launching background daemons.
6. Scenario SC-06 (Automated Parity Gate):
   CI or operator runs `cargo test -p aien-inference-abi --test tinyllama_parity --verbose`. System validates Stages 1 to 5 against oracle fixture and reports zero regressions.
7. Scenario SC-07 (Apples-to-Apples Multi-Engine Benchmark):
   Operator launches neutral benchmark orchestrator. System pauses for thermal cool-down, tests AIEN, and outputs TTFT/ITL telemetry.
8. Scenario SC-08 (Zero-Disk-Secrets Invariant Audit):
   Automated auditor scans workspace. Verifies zero `.env` files, tests in-memory TPM vault resolution, and verifies compliance.
9. Scenario SC-09 (Sovereign Voice & Git PR Lifecycle):
   System verifies branch naming, validates zero em/en dashes and zero banned words across code and docs, opens public PR, and records merge receipt in Cortex memory.

---

## 4. Test Execution Architecture & Single Command Runner

The complete E2E test suite resides in `e2e_tests/` and is executable via a single top-level runner command:

```bash
bash e2e_tests/run_e2e_tests.sh
```

Or via direct Cargo invocation:

```bash
cargo test --manifest-path e2e_tests/Cargo.toml -- --nocapture
```

The runner executes tests in progressive phases:
1. Environment & Invariant Checks (Zero Secrets, Sovereign Voice, User identity).
2. Checkpoint Loader & Validation Tests (Positive, Negative, Boundaries).
3. Tokenizer & Chat Template Tests (Encode, Decode, Special Tokens).
4. Tensor Math & Backend Integration Tests (RoPE, MatMul, RMSNorm, SwiGLU).
5. CLI Entry Point & Negative Argument Tests (`aien-cli` commands, zero panics).
6. Parity Verification & Oracle Contract Tests.
