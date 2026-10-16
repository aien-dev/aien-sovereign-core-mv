# Project: AIEN Sovereign Core TinyLlama Real-Model Execution

## Architecture
The AIEN Sovereign Core inference architecture executes real-model weights on NVIDIA DGX Spark (GB10 Grace Blackwell unified memory architecture) with mathematical parity against Hugging Face reference models.

```
+-----------------------------------------------------------------------------------+
| crates/aien-inference-abi                                                         |
|                                                                                   |
|  +---------------------------+       +-----------------------------------------+  |
|  |       tokenizer.rs        |       |              checkpoint.rs              |  |
|  | - Pure Rust tokenizers    |       | - Strict safetensors header parser      |  |
|  | - TinyLlama chat template |       | - Shape, dtype, range loud validation   |  |
|  | - Special token pinning   |       | - 201 TinyLlama tensor catalog          |  |
|  | - UTF-8 encode / decode   |       | - Dual FP32 (oracle) & BF16 (GPU)       |  |
|  +-------------+-------------+       +--------------------+--------------------+  |
|                |                                          |                       |
|                v                                          v                       |
|  +-----------------------------------------------------------------------------+  |
|  |                        transformer_backend.rs                               |  |
|  | - High-level autoregressive execution and KV cache orchestration            |  |
|  | - Dispatch via decoupled TensorBackend trait                                |  |
|  +-------------------------------------+---------------------------------------+  |
|                                        |                                          |
|                +-----------------------+-----------------------+                  |
|                |                                               |                  |
|                v                                               v                  |
|  +---------------------------+               +---------------------------------+  |
|  |    ReferenceCpuBackend    |               |        OmegaGb10Backend         |  |
|  | - Pure Rust FP32 math     |               | - Unified memory zero-copy      |  |
|  | - rotate_half RoPE        |               | - Omega native kernels, no CUDA |  |
|  | - [out_dim, in_dim] GEMV  |               | - AIEN-owned GPU driver path    |  |
|  | - Correctness oracle      |               | - GB10 sm_121a hardware speed   |  |
|  +---------------------------+               +---------------------------------+  |
+-----------------------------------------------------------------------------------+
```

## Feature Inventory
Every feature identified during survey is mapped directly to a milestone.

| # | Feature | Description | Milestone | Source |
|---|---|---|---|---|
| F1 | Strict Safetensors Loader | Read safetensors, parse JSON metadata without synthetic fallback | M1 | R1, Survey 1 |
| F2 | Loud Validation & Catalog | Validate all 201 tensors, shapes, BF16 dtype, byte ranges | M1 | R1, Survey 1 |
| F3 | Dual Weight Storage | Maintain decoded FP32 for oracle and raw BF16 for accelerated kernels | M1 | R1, Survey 1 |
| F4 | Pure Rust Tokenizer | Load tokenizer.json via tokenizers crate without Python runtime | M2 | R2, Survey 1 |
| F5 | Chat Template & Special Tokens | Format TinyLlama chat template, pin IDs (<s>=1, </s>=2, <unk>=0) | M2 | R2, Survey 1 |
| F6 | Encode & Decode Engine | String to token ID encoding, EOS stopping, token ID to UTF-8 text | M2 | R2, Survey 1 |
| F7 | Reference Oracle Script | Python script on Spark capturing exact intermediate activations | M3 | R3, Survey 2 |
| F8 | Deterministic Oracle Fixtures | Immutable tinyllama_oracle.safetensors and SHA-256 manifest | M3 | R3, Survey 2 |
| F9 | Algorithmic Core Alignment | Correct RoPE to rotate_half, matmul to [out_dim, in_dim], decode index | M4 | R4, Survey 1, 2 |
| F10 | Multi-Stage Parity Test | 5-stage test asserting tokenizer, shape, activations, logits, decode | M4 | R4, Survey 2 |
| F11 | Decoupled TensorBackend Trait | Narrow abstraction for RMSNorm, RoPE, GQA, SwiGLU, matmul, logits | M5 | R5, Survey 3 |
| F12 | ReferenceCpuBackend | Complete FP32 reference implementation satisfying TensorBackend | M5 | R5, Survey 3 |
| F13 | Mojo GB10 C-ABI Kernels | Removed 2026-10-15 with Modular MAX and Mojo; GPU path is OmegaGb10Backend | M5 | R5, Survey 3 |
| F14 | MojoGb10Backend Implementation | Removed 2026-10-15 with Modular MAX and Mojo | M5 | R5, Survey 3 |
| F15 | Neutral Benchmark Driver | Standalone AIEN benchmark driver on DGX Spark (the MAX engine was removed 2026-10-15; speed is measured against the machine's physical limit, and correctness against our own reference) | M6 | R6, Survey 3 |
| F16 | Concurrency Sweeps & Telemetry | C=1..64 sweeps, TTFT, ITL, throughput, RSS, power, energy metrics | M6 | R6, Survey 3 |
| F17 | Hardware & Secrets Compliance | User drakestapleton on Spark, zero disk secrets (TPM atlas-vault) | M7 | R7, Survey 3 |
| F18 | PR Lifecycle & Cortex Receipt | Branch feat/real-model-execution-tinyllama, squash merge, Cortex write | M7 | R7, Survey 3 |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|---|---|---|---|
| M1 | Strict Safetensors Checkpoint Loader | Implement checkpoint.rs with loud validation, MissingTensor/ShapeMismatch errors, 201-tensor catalog, dual FP32/BF16 storage | none | DONE |
| M2 | Pure Rust Tokenizer and Chat Template | Integrate tokenizers crate in aien-inference-abi, implement TinyLlama chat template, special token pinning, encode/decode | none | DONE |
| M3 | Reference Oracle Fixture Generation | Author scripts/generate_tinyllama_oracle.py on Spark, generate tinyllama_oracle.safetensors, export JSON manifest with SHA-256 | none | DONE |
| M4 | Algorithmic Alignment and Parity Test Harness | Align RoPE split-half and matmul layout in tensor.rs, implement crates/aien-inference-abi/tests/tinyllama_parity.rs passing 5 stages | M1, M2, M3 | DONE |
| M5 | Decoupled TensorBackend and Mojo GB10 Acceleration | Define TensorBackend trait, implement ReferenceCpuBackend and MojoGb10Backend (libaien_kernels.so) on Spark GB10 | M4 | DONE (Mojo backend removed 2026-10-15) |
| M6 | Neutral Apples-to-Apples Benchmark Driver | Build benchmarks/crates/bench_apples_to_apples, run AIEN and MAX on DGX Spark across C=1..64, record telemetry | M5 | DONE (MAX engine removed 2026-10-15) |
| M7 | Invariant Verification, Autonomous PR Lifecycle, and Cortex Receipt | Verify zero disk secrets, sovereign voice, execute PR squash merge, commit receipt to Cortex memory | M6 | DONE |

## Interface Contracts

### checkpoint.rs <-> weights.rs / backend.rs
```rust
pub enum CheckpointError {
    MissingTensor(String),
    ShapeMismatch { tensor: String, expected: Vec<usize>, actual: Vec<usize> },
    DtypeMismatch { tensor: String, expected: String, actual: String },
    OffsetOutOfBounds { tensor: String, offset: usize, buffer_len: usize },
    InvalidHeader(String),
}

pub struct LoadedCheckpoint {
    pub fp32_weights: std::collections::HashMap<String, Vec<f32>>,
    pub raw_bf16_weights: std::collections::HashMap<String, Vec<u8>>,
    pub shapes: std::collections::HashMap<String, Vec<usize>>,
}

pub fn load_safetensors_checkpoint<P: AsRef<std::path::Path>>(path: P) -> Result<LoadedCheckpoint, CheckpointError>;
```

### tokenizer.rs <-> transformer_backend.rs
```rust
pub struct TinyLlamaTokenizer {
    // wraps tokenizers::Tokenizer
}

impl TinyLlamaTokenizer {
    pub const BOS_TOKEN_ID: u32 = 1;
    pub const EOS_TOKEN_ID: u32 = 2;
    pub const UNK_TOKEN_ID: u32 = 0;
    pub const MAX_CONTEXT_LEN: usize = 2048;

    pub fn from_file<P: AsRef<std::path::Path>>(path: P) -> Result<Self, TokenizerError>;
    pub fn format_chat_prompt(&self, system: Option<&str>, user: &str) -> String;
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError>;
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError>;
}
```

### TensorBackend Trait
```rust
pub trait TensorBackend: Send + Sync {
    fn rmsnorm(&self, out: &mut [f32], x: &[f32], weight: &[f32], eps: f32);
    fn apply_rope(&self, q: &mut [f32], k: &mut [f32], pos: usize, head_dim: usize, num_q_heads: usize, num_kv_heads: usize, theta: f32);
    fn matmul_vec(&self, out: &mut [f32], x: &[f32], weight: &[f32], out_dim: usize, in_dim: usize);
    fn swiglu(&self, out: &mut [f32], gate: &[f32], up: &[f32]);
    fn gqa_attention(&self, out: &mut [f32], q: &[f32], k_cache: &[f32], v_cache: &[f32], seq_len: usize, num_q_heads: usize, num_kv_heads: usize, head_dim: usize);
    fn compute_logits(&self, logits: &mut [f32], hidden: &[f32], embed_weight: &[f32], vocab_size: usize, hidden_dim: usize);
}
```

### Numerical Parity Tolerances (tinyllama_parity.rs)
```rust
pub const ABSOLUTE_TOLERANCE: f32 = 1e-4;
pub const RELATIVE_TOLERANCE: f32 = 1e-4;
pub const MIN_COSINE_SIMILARITY: f32 = 0.9999;
```

## Code Layout
```
crates/aien-inference-abi/
├── Cargo.toml                              (owns: M1, M2 dependency wiring)
├── src/
│   ├── lib.rs                              (exports checkpoint, tokenizer, backend)
│   ├── checkpoint.rs                       (owns: M1 strict safetensors parsing and validation)
│   ├── tokenizer.rs                        (owns: M2 pure Rust tokenizers and chat templates)
│   ├── tensor.rs                           (owns: M4 rotate_half RoPE and row-major matmul)
│   ├── backend.rs                          (owns: M5 TensorBackend trait and ReferenceCpuBackend)
│   ├── omega_backend.rs                    (owns: OmegaGb10Backend, the GB10 GPU path)
│   ├── transformer_backend.rs              (owns: M4 decode position index, M5 backend dispatch)
│   └── weights.rs                          (owns: legacy compatibility delegating to checkpoint.rs)
├── tests/
│   ├── tinyllama_parity.rs                 (owns: M4 5-stage numerical parity test harness)
│   └── fixtures/                           (owns: M3 compact oracle and tokenizer fixtures)
scripts/
└── (no Python; the TinyLlama oracle generator was removed, only its recorded outputs are kept as test fixtures)
benchmarks/
├── Cargo.toml                              (owns: M6 benchmark workspace)
└── crates/
    └── bench_apples_to_apples/
        ├── Cargo.toml                      (owns: M6 benchmark crate dependencies)
        └── src/
            └── main.rs                     (owns: M6 AIEN driver and telemetry)
```

## Execution Constraints
- Strict Spark-Only Execution: All test runs, compilations, model evaluations, and benchmarks must execute strictly on the NVIDIA DGX Spark workstation (`ssh drakestapleton@spark`). Zero test execution or heavy compilation is permitted on the local MacBook host.

## Acceptance Criteria Cross-Check
The M5 Mojo and M6 MAX items below are the record of what shipped then; that code was removed on 2026-10-15.
- [x] load_safetensors_checkpoint rejects missing keys with MissingTensor and shape mismatches with ShapeMismatch (M1).
- [x] cargo test -p aien-inference-abi --test tinyllama_parity passes all 5 parity stages (M4).
- [x] Multi-step greedy generation produces identical token IDs and recognizable English text (M4).
- [x] TensorBackend trait cleanly separates Rust transformer semantics from compute backends (M5).
- [x] libaien_kernels.so compiles on Spark GB10 and executes without per-token host-device memory copying (M5).
- [x] Neutral benchmark orchestrator executes AIEN and MAX sequentially with thermal cool-downs (M6).
- [x] Concurrency sweep (C=1 to C=64) captures TTFT, ITL, throughput, memory, and power telemetry (M6).
- [x] Benchmark results and receipt committed to benchmarks/data/ and documented in walkthrough.md (M6).
- [x] Clean PR opened on aien-spark/aien-sovereign-core, CI green, linear squash merge, Cortex receipt in atlas-memory (M7).
