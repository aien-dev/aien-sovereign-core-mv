# State of AIEN: Systems Maturity Audit

Date: 2026-09-19  
Platform: NVIDIA DGX Spark (Grace Blackwell GB10, 121 GB LPDDR5X Unified Memory)  
Standard: Brutally Factual Systems Audit. Zero Marketing Language.

## Maturity Classification Scale

1. Concept: Architectural design documented; no functioning implementation.
2. Prototype: Proof-of-concept implementation; incomplete error handling or mock dependencies.
3. Working Locally: Fully implemented and validated on development hardware with unit tests.
4. Benchmarked: Empirically measured under reproducible load with published data.
5. Integrated: Multi-crate or multi-service automated interaction verified end-to-end.
6. Cross-platform Verified: Tested across multiple architectures (aarch64 and x86_64).
7. Production-ready: Hardened for continuous multi-user deployment with zero unhandled panics.

## Component Maturity Matrix

| Component | Status | Maturity Level | Verification Evidence | Known Limitations |
| :--- | :--- | :--- | :--- | :--- |
| **aegis-runtime** | Active | **Integrated** | Sub-millisecond heartbeat, 4.79 MB RSS, Axum HTTP routes active | Single-node gateway; cluster gossip specified in `docs/MULTI_STATION_ARCHITECTURE.md`. |
| **cortex-rs** | Active | **Benchmarked** | 21,262 req/s, 0.45 ms p50 latency, 10.67 MB RSS, SQLite WAL persistence tests | Schema migrations currently manual. |
| **spark-cockpit-rs** | Active | **Benchmarked** | 4,421 req/s, 1.74 ms p50 latency, 10.05 MB RSS | Terminal UI requires ANSI 256-color support. |
| **cortex-encoder-rs** | Active | **Working Locally** | 37,214 health req/s, ONNX BGE-M3 runtime loads on CUDA | Heavy cold-start initialization latency (~1.2s). |
| **atlas-vault** | Active | **Working Locally** | TPM-bound secret resolution, systemd-creds integration | Requires sudo credentials or TPM access during test runs. |
| **aien-kv-cache** | Active | **Benchmarked** | 1.44M blocks/sec allocation, 0.69 µs alloc latency, 8.88 ns CoW append, physical unified pool mmap | Radix tree eviction across memory pressure under heavy fragmentation needs long-duration soak test. |
| **aien-scheduler** | Active | **Benchmarked** | Sub-microsecond batch build overhead (2.2 µs at B=1 to 49.1 µs at B=256), chunked prefill validated | Priority queues currently hold 5 static priority tiers. |
| **spark-max-cabi** | Active | **Working Locally** | C-ABI dynamic library libspark_max.so compiled via Mojo 1.0.0, FFI tests passing | Opt-in `mojo-bridge` feature, comparison yardstick only: a default build never runs `mojo`. Shared object path must be present in LD_LIBRARY_PATH or binary build dir. |
| **spark-max-rs** | Active | **Working Locally** | Rust bindings to Mojo C-ABI with hardware synchronization and zero-overhead native fallback | Full tensor dispatch requires running MAX graph sessions. |
| **aien-inference-abi** | Active | **Cross-platform Verified** | Trait contracts, BlackwellInferenceBackend, MojoInferenceBackend, NativeCpuInferenceBackend, MockInferenceBackend (and transitional MaxServingBackend); tested on Apple Silicon M2, Linux x86_64, and Grace Blackwell | Live streaming SSE parsing is raw text; structured function calling parsing is pending. |
| **Continuous Batching Scheduler** | Active | **Benchmarked** | Swept C=1 to 256 with 8-11 µs step latency, 12.3k steps/sec dispatch throughput, live power telemetry (11.5W-12.0W), 1,000 child zero-copy forks | Multi-stream continuous prefill pipeline verified against live MAX endpoints. |

## Empirical Baseline Measurements

Withdrawn pending regeneration under the publication rule in the top-level
README (every headline number must resolve to a reproducible command and
artifact). The table below is retained for provenance of what was previously
stated and is not authoritative.

<!-- AIEN:BENCHMARKS:START -->
<!-- Sourced automatically from benchmarks/data/benchmarks_latest.json (Measurement Suite v0.2.0) -->
| Workload / Service | Architecture | Measurement ID | Resident Memory (RSS) | p50 Latency | Throughput |
| :--- | :--- | :--- | :---: | :---: | :---: |
| Python Microservice Baseline | Python 3.12 + FastAPI + Uvicorn | `BENCH-FASTAPI-RSS-001` | 44.76 MB | 1.28 ms | 7252 req/s |
| Sovereign Gateway & Heartbeat | Native Rust (Grace Blackwell) | `BENCH-OPENCLAW-RSS-001` | 4.56 MB (-89.8%) | N/A | N/A |
| Canonical Memory Engine | Native Rust + SQLite WAL | `BENCH-CORTEX-RSS-001` | 18.66 MB (-58.3%) | 0.50 ms | 19367 req/s |
| Real-time Telemetry Cockpit | Native Rust + Axum | `BENCH-COCKPIT-RSS-001` | 12.90 MB (-71.2%) | 1.63 ms | 5840 req/s |
| Neural Embedding Microservice | Rust + ONNX Runtime (BGE-M3) | `BENCH-ENCODER-RSS-001` | 777.43 MB | 0.26 ms | 34685 req/s |

*Hardware Reference: NVIDIA DGX Spark (NVIDIA Grace Blackwell (GB10, aarch64), 121 GB unified memory). All metrics measured under concurrency C=10 over 500 requests per endpoint.*
<!-- AIEN:BENCHMARKS:END -->

Canonical dataset and commit anchors: [aien-dev/benchmarks](https://github.com/aien-dev/benchmarks).

## Deficit Inventory

1. **Versioned Release Packaging**: Resolved. Tagged `v0.1.0` release workflow (`release.yml`) with automated SHA-256 checksums and multi-OS binary packaging (`sovereign-*.tar.gz`).
2. **API Stability Policy**: Resolved. Formal stability tiers, SemVer guarantees, and deprecation policies documented in `docs/API_STABILITY.md`.
3. **Cluster Multi-Node**: Addressed. Multi-station mesh topology, Radicle anchor synchronization, and WireGuard transport documented in `docs/MULTI_STATION_ARCHITECTURE.md`.
4. **Weight Formats**: Native Mojo tensor loaders support unquantized GGUF, FP16, and BF16; NVFP4 weights require MAX graph compilation or ModelOpt runtime.
