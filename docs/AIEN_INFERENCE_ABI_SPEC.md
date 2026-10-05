# AIEN Inference ABI & Native Runtime Architecture Specification

## 1. Research Thesis
Modern inference engines expend substantial compute and memory overhead mitigating runtime friction introduced by Python, dynamic tracing guards, and interpreter-level scheduling. 

This research branch evaluates the thesis:
> How much of modern LLM inference latency, memory pressure, and energy consumption belongs to forward tensor mathematics, and how much is an artifact of the software orchestration stack?

By systematically replacing each layer of the inference engine with native compiled Rust and Mojo, we measure the empirical delta across each removal.

---

## 2. Silicon Execution Architecture

AIEN serves as the native compiled inference runtime. High-performance tensor execution is decoupled into silicon backends conforming to the AIEN Inference ABI:

```text
                         AIEN / ENOS
                             │
                     aien-inference-abi
                             │
             ┌───────────────┼────────────────┐
             │               │                │
             ▼               ▼                ▼
       Blackwell Native     Mojo/MAX       Native CPU
       GB10 fast path       kernels         fallback
             │               │                │
             └───────────────┼────────────────┘
                             │
                     AIEN Scheduler
                             │
                       aien-kv-cache
                             │
                         Model State
```

| Subsystem | Fast Path (Blackwell) | Accelerated Path (Mojo) | Fallback Path (Native CPU) | Test Oracle |
| :--- | :--- | :--- | :--- | :--- |
| **Scheduler** | AIEN Native Continuous Batching Scheduler | AIEN Native Continuous Batching Scheduler | AIEN Native Continuous Batching Scheduler | Deterministic Step Scheduler |
| **KV Management** | AIEN Physical Unified CoW Pool | AIEN Physical Unified CoW Pool | In-process Paged Pool | Virtual Page Table |
| **Execution Engine** | GB10 sm_121 Omega native kernels (`OmegaGb10Backend`, no CUDA) | Mojo C-ABI (`libaien_kernels.so`) | Rayon Multi-Core NEON/AVX | ReferenceCpuBackend (FP32 Oracle) |
| **Inference Backend** | `BlackwellInferenceBackend` | `MojoInferenceBackend` | `NativeCpuInferenceBackend` | `MockInferenceBackend` |
| **Memory Substrate** | DGX Spark 128 GB Unified Coherent Memory | DGX Spark 128 GB Unified Coherent Memory | Standard Host DRAM | Heap Slices |

### Monitored Telemetry per Stage
1. **TTFT (Time To First Token)**: Prefill latency in milliseconds.
2. **ITL (Inter-Token Latency)**: Decode step time in milliseconds (p50, p95, p99).
3. **Throughput**: Sustained output tokens per second under concurrency sweeps (C = 1, 4, 16, 64).
4. **Memory Footprint**: Host RSS (MB), GPU Allocated VRAM (MB), GPU Reserved VRAM (MB).
5. **KV Efficiency**: Physical block fragmentation percentage, prefix reuse hit rate.
6. **Host Resource Draw**: CPU Utilization (%), GPU Tensor Core Utilization (%), NVLink-C2C bus saturation (GB/s).
7. **Thermodynamic Efficiency**: Measured Joules / token via power sampling.

---

## 3. Core Architecture: AIEN Inference ABI

The inference interface belongs to AIEN. Silicon backends conform to this trait:

```rust
use async_trait::async_trait;
use std::collections::HashMap;

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub model_id: String,
    pub max_sequence_length: usize,
    pub block_size: usize,
    pub num_layers: usize,
    pub num_heads: usize,
    pub head_dim: usize,
}

#[derive(Debug, Clone)]
pub struct SequenceRequest {
    pub request_id: u64,
    pub prompt_tokens: Vec<u32>,
    pub sampling_params: SamplingParams,
}

#[derive(Debug, Clone)]
pub struct SamplingParams {
    pub temperature: f32,
    pub top_p: f32,
    pub max_tokens: usize,
    pub stop_token_ids: Vec<u32>,
}

#[derive(Debug, Clone)]
pub struct ScheduledBatch {
    pub prefill_requests: Vec<SequenceRequest>,
    pub decode_requests: Vec<u64>,
    pub block_tables: HashMap<u64, Vec<usize>>,
}

#[derive(Debug, Clone)]
pub enum DecodeOutput {
    Token { token_id: u32, logprob: Option<f32> },
    Finished { reason: FinishReason },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FinishReason {
    StopToken,
    LengthLimit,
    Aborted,
}

#[async_trait]
pub trait AienInferenceBackend: Send + Sync {
    async fn load_model(&mut self, config: &ModelConfig) -> Result<(), String>;
    async fn execute_step(
        &mut self,
        batch: &ScheduledBatch,
    ) -> Result<Vec<DecodeOutput>, String>;
}
```

---

## 4. Native Scheduling & Paged KV-Cache Management

### KV Block Table Specification
Blocks are fixed-token chunks (e.g. 16 or 32 tokens) allocated from a pre-mapped pool in unified host/device memory.

```rust
pub struct KvBlock {
    pub block_id: usize,
    pub ref_count: usize,
    pub is_shared: bool,
}

pub struct BlockTable {
    pub blocks: Vec<usize>,
}

pub struct AienKvManager {
    total_blocks: usize,
    free_blocks: Vec<usize>,
    allocated_tables: HashMap<u64, BlockTable>,
}

impl AienKvManager {
    pub fn allocate_prompt_blocks(&mut self, token_count: usize) -> Result<BlockTable, String>;
    pub fn fork_sequence(&mut self, parent_req_id: u64, child_req_id: u64) -> Result<(), String>;
    pub fn free_sequence(&mut self, req_id: u64);
}
```

### Prefix Sharing & Zero-Copy Subagent Branching
When a subagent forks from a parent agent:
1. The parent's `BlockTable` entries have their `ref_count` incremented.
2. `is_shared` is marked true.
3. The child agent inherits the exact block indices without copying KV memory.
4. During decode, if a child writes to a shared block, copy-on-write allocates a new private block.

---

## 5. Empirical Benchmark Telemetry & Live Pressure Verification

> [!NOTE]
> **Control Plane Scope Clarification**:
> The throughput and latency figures in Sections 5.1 through 5.4 measure native control plane execution:
> block-table indexing, sequence queue transitions, reference-counting mutations, and batch assembly.
> Block indices represent coordinate pointers in pre-allocated unified memory.
> Physical tensor manipulation across the Blackwell GPU substrate begins in Stages 2 through 4.


Execution Substrate: NVIDIA DGX Spark (NVIDIA Grace Blackwell GB10, aarch64, 121 GB Unified LPDDR5X Memory, Linux 7.0.0-1019-nvidia).

### 5.1 Paged KV Cache Block-Table Allocator Throughput (Control Plane)
Benchmark binary: `target/release/bench_inference_stack` (built only with `--features max-yardstick`; a comparison yardstick, not part of a default build)
- Workload: 10,000 sequence allocations (160,000 block-table indices managed, block size = 16 tokens).

| Metric | Result | Sub-unit Latency |
| :--- | :--- | :--- |
| Allocation Throughput | 134,338,182.94 blocks/sec | 119.10 ns/sequence (7.44 ns/block) |
| Deallocation Throughput | 238,709,061.40 blocks/sec | 67.03 ns/sequence (4.19 ns/block) |
| Memory Management Status | Zero-allocation pooling verified | Deterministic constant time O(1) |

### 5.2 Subagent Zero-Copy Sequence Fork vs Naive Memory Copy
Parent sequence context: 4,096 tokens (256 KV blocks).

| Subagents Forked | Zero-Copy Fork Time | Naive Copy Est. | Speedup Ratio | Projected Tensor Memory Saved |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 1.58 µs | 1.92 ms | 1,212.1x | 0.38 GB |
| 10 | 0.46 µs | 19.20 ms | 4,152.2x | 3.75 GB |
| 50 | 0.47 µs | 96.00 ms | 4,067.8x | 18.75 GB |
| 100 | 0.39 µs | 192.00 ms | 4,977.2x | 37.50 GB |
| 500 | 0.45 µs | 960.00 ms | 4,236.1x | 187.50 GB |

### 5.3 Copy-on-Write (CoW) Mutation Latency
- Iterations: 1,000 token generations on shared parent blocks.
- Latency per CoW Token Append: 8.96 nanoseconds.
- Divergence Overhead: Negligible. Subagents branch without stalling parent streams.

### 5.4 Continuous Batching Scheduler Step Overhead
- Scheduler Config: Max batch tokens = 8,192, max prefill tokens = 4,096, chunked prefill enabled.

| Active Sequences | Batch Build Time (µs) | Scheduler Overhead Ratio (in 10ms step) |
| :--- | :--- | :--- |
| 1 | 1.01 µs | 0.0101% |
| 9 | 2.02 µs | 0.0202% |
| 40 | 5.20 µs | 0.0520% |
| 96 | 9.66 µs | 0.0966% |
| 192 | 17.68 µs | 0.1768% |

### 5.5 Live Microservice Call Stress & Memory Pressure
Benchmark harness: `target/release/stress_aien_live` executing against live production endpoints on `spark`.

#### Cortex-rs Vector Memory Call Stress (Port 18080, `/api/cortex/search`)
| Concurrency | Total Calls | Throughput (req/s) | p50 Latency (ms) | p95 Latency (ms) | p99 Latency (ms) | Success Rate |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 10 | 200 | 1,637.21 req/s | 6.03 ms | 7.10 ms | 11.35 ms | 100.0% |
| 25 | 200 | 1,867.17 req/s | 10.97 ms | 29.68 ms | 41.53 ms | 100.0% |
| 50 | 200 | 1,960.52 req/s | 11.82 ms | 57.90 ms | 78.56 ms | 100.0% |
| 100 | 200 | 2,103.73 req/s | 23.02 ms | 68.83 ms | 88.18 ms | 100.0% |

#### Cortex Transformer Encoder Batch & Density Stress (Port 18081, `/embed`)
- Model: `BAAI/bge-base-en-v1.5` (ONNX INT8, CPU).

| Batch Size | Total Texts | Duration (ms) | Throughput (texts/sec) | Per-Text Latency (ms) |
| :--- | :--- | :--- | :--- | :--- |
| 1 | 20 | 100.21 ms | 199.59 texts/s | 5.01 ms |
| 4 | 80 | 451.71 ms | 177.11 texts/s | 5.65 ms |
| 8 | 160 | 851.81 ms | 187.84 texts/s | 5.32 ms |
| 16 | 320 | 1547.26 ms | 206.82 texts/s | 4.84 ms |
| 32 | 640 | 3165.60 ms | 202.17 texts/s | 4.95 ms |

#### MAX LLM Streaming & Latency Pressure (Port 18082, `unsloth/Llama-3.2-1B-Instruct`)
| Concurrent Streams | TTFT p50 (ms) | ITL p50 (ms) | ITL p95 (ms) | Aggregate Tok/s | E2E Latency p50 (ms) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 130.59 ms | 90.62 ms | 97.77 ms | 11.58 tok/s | 2,763.97 ms |
| 4 | 5,652.78 ms | 86.60 ms | 103.46 ms | 11.61 tok/s | 8,299.38 ms |
| 8 | 11,198.92 ms | 87.25 ms | 101.76 ms | 11.59 tok/s | 13,795.19 ms |
| 16 | 22,234.08 ms | 84.76 ms | 100.95 ms | 11.66 tok/s | 24,854.70 ms |

#### Resident Set Size (RSS) Memory Stability Under Pressure
| Service | Baseline RSS | Peak Concurrency RSS | Post-Stress Delta | Memory Stability |
| :--- | :--- | :--- | :--- | :--- |
| cortex-rs | 15.97 MB | 18.57 MB | +2.60 MB | Confirmed leak-free |
| cortex-encoder-rs | 780.02 MB | 780.39 MB | +0.36 MB | Confirmed leak-free |
| max inference engine | 9,011.61 MB | 9,013.99 MB | +2.38 MB | Confirmed leak-free |
| aegis daemon | 4.80 MB | 4.80 MB | +0.00 MB | Zero drift |
