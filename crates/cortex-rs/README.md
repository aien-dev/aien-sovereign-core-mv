<p align="center">
  <img src="assets/avatar.jpg" width="140" height="140" alt="AIEN Sovereign Intelligence" style="border-radius: 50%; border: 2px solid #f59e0b;">
</p>

# cortex-rs

[![CI](https://github.com/aien-dev/cortex-rs/actions/workflows/ci.yml/badge.svg)](https://github.com/aien-dev/cortex-rs/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)
[![Security](https://img.shields.io/badge/tpm--vault-zero--disk--secrets-green.svg)](SECURITY.md)
[![Standard](https://img.shields.io/badge/standard-unslop-black.svg)](CONTRIBUTING.md)
[![Mission](https://img.shields.io/badge/mission-sovereign--defense-amber.svg)](https://drakestapleton.com)

Native Rust memory engine with SQLite FTS5 lexical recall, bi-encoder vector similarity, and hardware-secured loopback isolation.

## Overview

`cortex-rs` is the Linux-hosted memory service (native Rust, Axum) that Atlas and the AIEN agent tools call for lexical and semantic recall. It is not the canonical memory owner: under [ADR 0022](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0022-canonical-cortex-owner.md) the canonical Cortex owner at the Omega layer is omega `rx_cortex`, and under [ADR 0036](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0036-release-v1-memory-owner.md) the memory the v1 release tests is `aien-allen-memory` attached to an engaged ALLEN identity.

It provides hybrid search combining BM25 full-text indexing with ONNX vector embeddings, enabling autonomous agents to recall procedures, lessons, and architectural discoveries across thousands of turns without context collapse.

## Features

- **Sub-Millisecond Retrieval**: Pure compiled Rust using SQLite WAL mode and FTS5 inverted indexes with typical latency <1ms.
- **Hybrid Semantic Recall**: Direct integration with local bi-encoder vector servers for cosine similarity and dense embedding scoring.
- **Hardware Vault Security**: Zero plaintext secrets. All authentication uses bearer token validation bound to loopback interfaces (`127.0.0.1`).
- **Graph & Provenance Traversal**: Bidirectional relationship edges (`cortex_claims`, `cortex_entities`) for deep contextual graph traversal.
- **Unslop Standard**: Enforces clean technical facts without AI filler or speculative abstractions.

## Turnkey Reproduction & Benchmarks

Run the engine and observe live latency and throughput metrics directly with a single command.

### 1. One-Line Docker Benchmark

Execute the compiled native benchmark inside a container to evaluate SQLite WAL ingestion, FTS5 lexical retrieval, and graph claim traversal:

```bash
docker run --rm ghcr.io/aien-dev/cortex-rs:latest --bench
```

Or build and run locally with Docker:

```bash
docker build -t cortex-rs https://github.com/aien-dev/cortex-rs.git
docker run --rm cortex-rs --bench
```

Verified benchmark output on Grace Blackwell GB10:

```text
================================================================================
CORTEX-RS FLAGSHIP REPRODUCTION BENCHMARK
Target: SQLite WAL + FTS5 Inverted Index + Graph Claim Traversal
Records to ingest: 500
================================================================================

[1/3] Benchmarking Entity Ingestion (FTS5 + SQLite WAL)...
[2/3] Benchmarking FTS5 Lexical Search across 500 records...
[3/3] Benchmarking Bidirectional Graph Claim Traversal...

================================================================================
BENCHMARK RESULTS SUMMARY (Reproducible Single-Command Output)
================================================================================
Corpus Size:                      500 entities
Total Search Queries Run:         500
--------------------------------------------------------------------------------
METRIC                          p50 (µs)    p95 (µs)    p99 (µs)      Rate / QPS
--------------------------------------------------------------------------------
Entity Ingestion (WAL+FTS5)         76.9       145.6      8121.9       4375.6 writes/s
FTS5 Lexical Search                121.8       452.9       461.3       6341.2 qps
Graph Claim Traversal                7.3         7.5        10.2     133247.2 qps
--------------------------------------------------------------------------------
Search p50 in milliseconds:      0.122 ms
Search p99 in milliseconds:      0.461 ms
================================================================================
```

### 2. Standalone Native Reproduction (Zero Docker)

To run natively on Linux or macOS without Docker:

```bash
# Clone repository and execute the automated verification runner
git clone https://github.com/aien-dev/cortex-rs.git
cd cortex-rs
./scripts/reproduce.sh
```

Or run the built-in benchmark directly with Cargo:

```bash
cargo run --release -- --bench --bench-records 1000
```

### 3. Running the Production Server

#### Local Process

```bash
# Compile release binary
cargo build --release

# Run Cortex-RS on local loopback
./target/release/cortex-rs --port 18080 --db-path ~/.config/cortex/cortex.db
```

#### Docker Container

```bash
docker run -d \
  --name cortex-rs \
  -p 18080:18080 \
  -e CORTEX_TOKEN=your-vault-token \
  ghcr.io/aien-dev/cortex-rs:latest
```

### 4. API Endpoints

#### Health Check (Unauthenticated)

```bash
curl http://127.0.0.1:18080/health
# {"runtime":"native-arm64-rust","service":"cortex-rs","space":"atlas-memory","status":"ok","version":"0.1.0"}
```

#### Entity Ingestion

```bash
curl -X POST http://127.0.0.1:18080/api/cortex/write \
  -H "Authorization: Bearer <TOKEN>" \
  -H "Content-Type: application/json" \
  -d '{
    "kind": "entity",
    "value": {
      "space": "atlas-memory",
      "canonicalName": "system_module_1",
      "entityType": "discovery",
      "content": "Grace Blackwell GB10 unified memory architecture verification."
    }
  }'
```

#### Lexical Search

```bash
curl -G http://127.0.0.1:18080/api/cortex/search \
  --data-urlencode "q=unified memory" \
  -H "Authorization: Bearer <TOKEN>"
```

## License

This repository is licensed under the **Apache License, Version 2.0 (with LLVM Exception)**.

See [LICENSE](LICENSE) for the full legal text.
