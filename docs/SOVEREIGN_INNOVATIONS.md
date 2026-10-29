# Sovereign Core Architectural Innovations

### Monorepo Blueprint of Compiled Native Crates, Invariant Gates, and Local Hardware Acceleration

`aien-sovereign-core` is the foundational monorepo of the Sovereign AI Commons. Engineered in pure native compiled Rust with Omega GPU kernels, it provides the operating system substrate for autonomous agents, local GPU inference, and global open stewardship on NVIDIA DGX Spark workstations and Grace Blackwell GB10 hardware.

---

## 1. Monorepo Crate Directory

The workspace comprises native compiled crates, each dedicated to an isolated system responsibility:

### Governance and Verification
- **`crates/spark-inquisitor`**: Autonomous GitHub gatekeeper, diff auditor, and issue triage engine. Enforces constitutional invariants on pull requests and issues, audits diffs for tracking patterns, verifies unslop compliance, and evaluates contributor oaths.
- **`crates/spark-harness`**: Autonomous evaluation and benchmark test harness providing JSON Schema validation, regression tracking, and execution reliability verification for agent actions.
- **`crates/spark-debugger`**: Hardware-level crash diagnostics and GPU fault isolation engine, inspecting execution traces and PTX/CUDA faults.

### Memory and Epistemics
- **`crates/cortex-rs`**: Persistent epistemic memory engine backed by SQLite WAL with FTS5 lexical full-text indexing, vector similarity, and entity provenance tracking.
- **`crates/cortex-encoder-rs`**: Native Rust embedding encoder providing fast local text vectorization for semantic retrieval.
- **`crates/spark-dream`**: Dynamic dream cycle engine that activates during GPU idle cycles to consolidate episodic session logs into verified semantic knowledge entities.
- **`crates/spark-crumbs`**: Cryptographic, append-only operational event ledger implementing the Crumb protocol with session trees and stigmergic directory anchors.

### Process Lifecycle and Orchestration
- **`crates/spark-supervisor`**: High-reliability process supervisor, watchdog, and crash recovery daemon managing agent processes with deterministic restart policies.
- **`crates/spark-hive`**: Honeycomb hive engine with 2D axial hexagonal geometry for routing agent tasks and coordinating consumer model adapters.
- **`crates/spark-cockpit-rs`**: High-performance Axum HTTP and Server-Sent Events (SSE) gateway with embedded live monitoring dashboard, memory graph explorer, and telemetry streaming.
- **`crates/spark-discord-hub`**: Sovereign bridge daemon facilitating agent observability and external event broadcasting over Discord websockets.
- **`crates/aien-cli`**: Sovereign command-line interface and system orchestrator.

### Hardware Acceleration and Model Serving
- **`crates/spark-adapters`**: Consumer model adapter pipeline enabling quantization, weight formatting, and hardware adaptation.
- **`crates/spark-harvester`**: High-throughput reasoning trace extractor and dataset distillation engine.
- **`crates/spark-mail-rs`**: Sovereign peer-to-peer messaging and notification subsystem.

---

## 2. Constitutional Invariant Gates

Every crate and workflow within `aien-sovereign-core` adheres to four architectural axioms:

1. **Zero Telemetry and Anti-Surveillance**:
   All crates are prohibited from calling external telemetry, surveillance, or analytics endpoints. Network activity is restricted strictly to user-configured endpoints and peer relays.
2. **Pure Compiled Native Systems**:
   All daemons, memory engines, gateways, and CLI tools must compile to native machine binaries. Interpreted runtimes are banned from core services.
3. **Hardware Secret Security (Zero Plaintext Disk Secrets)**:
   Credentials, API keys, and signing tokens never touch disk in plaintext. All secrets resolve dynamically in memory via the `SecretProvider` abstraction (hardware TPM vault `atlas-vault` on DGX Spark, or memory providers on commodity platforms) with automated stream redaction (`[REDACTED_BY_ATLAS_VAULT]`).
4. **Sovereign Voice and Anti-Slop**:
   All code comments, CLI outputs, and documentation strictly ban em dashes, en dashes, and formulaic AI marketing buzzwords.

---

## 3. Hardware Targets and Benchmark Proof

Verified on NVIDIA Grace Blackwell GB10 (NVIDIA DGX Spark, Linux aarch64):
- **Core Binary Build Time**: < 1.0s release compilation for `spark-inquisitor`.
- **Memory Consumption**: Core background daemons operate under 25 MB resident memory each.
- **Inference Concurrency**: Sustained local generation hitting 94% to 96% GPU utilization at 20.7 tokens per second on local Modular MAX endpoints (2026-09 measurement; Modular MAX was removed on 2026-10-15).

---

## 4. License and Commons Covenant

Licensed under the **Apache License, Version 2.0**.
Copyright (c) 2026 AIEN Sovereign Systems & Ecosystem Contributors.
See [LICENSE](../LICENSE) and [NOTICE](../NOTICE) for terms.
Voluntary bilateral research covenants and artifact access are governed by [OPEN_COOPERATION_COVENANT.md](../OPEN_COOPERATION_COVENANT.md) and [ARTIFACT_LICENSING.md](../ARTIFACT_LICENSING.md).
For sovereign ecosystem coordination: `aien@aienos.com`.
