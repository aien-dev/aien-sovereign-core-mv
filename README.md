# aien-sovereign-core

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)

The earlier **Linux-hosted** AIEN runtime, written in Rust: agent CLI, persistent memory (Cortex), a unified-memory KV-cache with copy-on-write branching, a continuous-batching scheduler, and an inference ABI whose GPU path runs AIEN's own Omega kernels. It runs on top of Linux on the NVIDIA DGX Spark (Grace Blackwell GB10) and is the reference for what the native stack must beat.

**This repository is legacy.** AIEN's own stack (the [aienos](https://github.com/aien-spark/aienos) kernel, the [omega](https://github.com/aien-spark/omega) runtime, [physics](https://github.com/aien-spark/physics) FORGE) is replacing it. Under [ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md), working Rust here is kept as temporary scaffolding, not rewritten into C: it stays behind language-neutral boundaries and Omega replaces components one at a time after passing the same conformance tests. New Rust is allowed where it is not hardware-bound (for example the DEV-MODEL-0 program, a local development model on AIEN-owned inference). Component ownership (for example the Cortex now owned by omega, per ADR 0022) is recorded in [aien-architecture](https://github.com/aien-spark/aien-architecture); here `crates/cortex-rs` is non-authoritative and to be retired, not ported.

No Python in this repo (a CI guard, `scripts/check-no-python.sh`, fails the build if any Python file is added), and no comparison with other people's systems: speed is measured against the machine's physical limit, and correctness against our own reference.

> SUPERSEDED 2026-10-01 by [ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md). Previous text, kept as history: **This repository is legacy.** The project's target language is C, and AIEN's own stack (the [aienos](https://github.com/aien-spark/aienos) kernel, the [omega](https://github.com/aien-spark/omega) runtime, [physics](https://github.com/aien-spark/physics) FORGE) is replacing it. No new Rust is added here, with one declared exception: the DEV-MODEL-0 program (a local development model on AIEN-owned inference) may temporarily extend this Rust substrate while the C replacements are built. Component ownership (for example the Cortex now owned by omega, per ADR 0022) is recorded in [aien-architecture](https://github.com/aien-spark/aien-architecture); here `crates/cortex-rs` is non-authoritative and to be retired, not ported.

## Current state

Research-grade and pre-alpha. Crates build and have unit tests on the DGX Spark; execution is CPU plus GPU through AIEN's own Omega kernels. Modular MAX and Mojo were removed from the project on 2026-10-15. Nothing here is qualified on hardware as a finished system. Live status and sequencing: [CURRENT_EXECUTION_PLAN.md](https://github.com/aien-spark/aien-architecture/blob/main/CURRENT_EXECUTION_PLAN.md) and the [implementation status snapshot](https://github.com/aien-spark/aien-architecture/blob/main/docs/02-implementation-status.md). Component detail and platform status: [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md), [docs/AIEN_RUNTIME_ARCHITECTURE.md](docs/AIEN_RUNTIME_ARCHITECTURE.md).

Known limits: the GPU fallback counter covers only instrumented operations, so execution is not exclusively accelerated; Qwen full-runtime qualification is not complete; the C1 evidence campaign is in progress.

## Measured results

Every headline performance number must resolve to a reproducible command and an artifact bundle (commit identity, hardware and environment record, exact command, raw samples, SHA-256 digests, measurement definition, reproducibility steps). Earlier figures (branch fork latency, COW page mutation latency, memory sharing ratios, service comparison table) are withdrawn until regenerated to that standard. The evidence bundles live in [aien-spark/benchmarks](https://github.com/aien-spark/benchmarks).

## Standing rules

Language rule: Rust is scaffolding, Omega is the destination, and C or assembly stay only where hardware, boot, ABI or measurement justifies them ([ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md), which supersedes the old "C is the target, no new Rust" rule). No Python anywhere in repos, build, CI or tooling. No CUDA toolkit and no new CUDA dependence. No systemd in AIENOS. No outside dependencies in the trusted base. No Modular MAX or Mojo anywhere in the tree (removed 2026-10-15); kernels are Omega or Rust, closest to the metal, fastest wins, build what is missing.

> SUPERSEDED 2026-10-01 by [ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md) (language part only; the Python, CUDA, systemd and dependency rules above are unchanged). Previous text, kept as history: C is the target language, with assembly only where measured; no new Rust, and existing Rust is legacy. No new Python. The few existing `.py` files here (helper scripts and Modular experiment code) are legacy and are to be rewritten in C or shell. No CUDA toolkit and no new CUDA dependence. No systemd in AIENOS. No outside dependencies in the trusted base. Mojo is the default for kernels, not dogma: closest to the metal, fastest wins, beat it if you can, build what is missing.

## Build and test

Needs Rust 1.98.1, the version pinned in `rust-toolchain.toml` (rustup installs it on first use), on the DGX Spark or another Linux machine.

```bash
git clone https://github.com/aien-spark/aien-sovereign-core
cd aien-sovereign-core
cargo test --workspace -- --test-threads=1   # one test thread, as CI runs it
scripts/repro-build.sh -p aien-cli   # release build with machine-local paths removed; build aien-cli on its own, because its digest changes with the other packages in the same cargo command (docs/release/F1-binary-digest-difference.md)
```

Installing a release (state on 2026-10-11, from `install.sh` and the dry-run artifact of Actions run 38105672607, nothing else):

- `install.sh` with no arguments builds from source and verifies no signature. With `AIEN_RELEASE_TAG=vX.Y.Z` it installs a published release instead: it downloads the archive, `SHA256SUMS.txt` and `SHA256SUMS.txt.sig`, checks the signature with `ssh-keygen`, then the archive checksum, and writes `install-receipt.json` with a `signature_check` of `release`, `internal-test-key` or `dry-run-key` (see [docs/release/RELEASE_SIGNING.md](docs/release/RELEASE_SIGNING.md)).
- No public release exists. The public release key is not pinned yet (`AIEN_RELEASE_SIGNER` is empty until the key ceremony in [docs/release/KEY_CEREMONY.md](docs/release/KEY_CEREMONY.md)), so nothing can install as `release`. v0.1.0 and v0.1.1 are internal test releases. The release gate (`scripts/check-release-candidate.sh`) fails on current main (omega.lock differs from the CAND-4 manifest), see [#414](https://github.com/aien-spark/aien-sovereign-core/issues/414).
- After an install, follow [docs/FIRST_RUN.md](docs/FIRST_RUN.md): create the desk key (`aien compose desk-key --create 1`; `aien start` refuses without it), provision the model, start, record an objective. Objective records default to a durable per-user folder (`~/.local/state/aien-runtime/provenance`), not `/tmp`. The installer does not provision a model. `aien harness` is not shipped in this release and exits with a refusal.

## Contributing

Read [CONSTITUTION.md](CONSTITUTION.md) and [AGENTS.md](AGENTS.md) before opening a pull request. Changes ship on a branch with verification proof: the commands you ran and their output. Prefer work that moves capability into the C repositories over new Rust here. Contact: aien@aienos.com.

## License and values

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). See [LICENSE](LICENSE) and [NOTICE](NOTICE). Values live in the nonbinding [COVENANT.md](COVENANT.md): keep foundational advances open. It grants and restricts no legal rights. Built for NVIDIA Blackwell hardware; full notices in [ATTRIBUTION.md](ATTRIBUTION.md).
