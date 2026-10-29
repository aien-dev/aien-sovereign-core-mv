# AIEN Artifact Licensing Matrix

This document defines the licensing structure governing software, specifications, benchmark data, and model artifacts across the AIEN ecosystem.

AIEN software is licensed independently from AIEN model artifacts. Possession, compilation, execution, or deployment of AIEN software creates zero claims, obligations, or covenants concerning artificial intelligence models trained with independent datasets, weights, services, or infrastructure, unless a specific model or dataset expressly carries separate terms.

---

## Artifact Classification Matrix

| Layer | Artifact Description | Canonical License | Identifier |
| :--- | :--- | :--- | :--- |
| **Core Software Runtime** | Rust crates (`aien-scheduler`, `aien-kv-cache`, `cortex-rs`, `spark-rsi`, `aien-harness`, CLI, web gateways) | Apache License 2.0 | `Apache-2.0` |
| **Protocols & Specifications** | Inference ABI specifications, Crumb protocol definitions, schemas, and formats | Apache License 2.0 for schemas and reference code; Community Specification License 1.0 for published specifications (aien-protocols-mv) | `Apache-2.0` / `CSL-1.0` |
| **Benchmark Harness Code** | Measurement driver, CLI, test generators, and harness binaries in `benchmarks/` | Apache License 2.0 | `Apache-2.0` |
| **Benchmark Datasets & Metrics** | Raw telemetry JSON, hardware timing records, power logs, and rendered SVG comparison charts | Creative Commons Zero 1.0 Universal | `CC0-1.0` |
| **AI Foundation Model Weights** | Neural network checkpoints, tensors, and tokenizer parameters released by AIEN | Specific Model License | Refer to accompanying `MODEL_LICENSE` |
| **Training Code & Distillation** | Model training recipes, data processing pipelines, and distillation scripts | Apache License 2.0 | `Apache-2.0` |
| **Training Datasets** | Curated datasets, imprints, or knowledge corpora | Dedicated Data License | Refer to accompanying `DATA_LICENSE` |
| **Constitutional Governance** | Mission, principles, engineering standards, and developer oaths | Upstream Project Charter | Non-binding on downstream software users |
| **Voluntary Reciprocity** | Shared research freedoms, open weight releases, and reciprocal distillation program | Bilateral Research Covenant | Refer to `OPEN_COOPERATION_COVENANT.md` |

---

## Model Release Terminology

To maintain technical and legal precision:

1. **Open Source AI**: This designation is reserved exclusively for artificial intelligence releases that fully satisfy the Open Source AI Definition (OSAID 1.0) established by the Open Source Initiative (OSI), providing model parameters, training and inference code, and comprehensive training data documentation.
2. **Open Weights**: Artificial intelligence models published with public parameter weights, inference code, and architecture specifications, but which do not bundle the exhaustive training corpus necessary to satisfy OSAID 1.0, are precisely designated as Open Weights.

---

## Downstream Implementation Autonomy

AIEN software is licensed under the Apache License 2.0 (`LICENSE`, `NOTICE`). Its terms, not this document, decide what a downstream user may do: they cover copying, modifying, linking, distributing and offering the software over a network, and they include an express patent grant. Model weights are separate works: a third-party model keeps its publisher's license (for example SmolLM2-1.7B-Instruct, Apache-2.0, see `NOTICE` and `ATTRIBUTION.md`), and running a model on AIEN does not relicense it.
