# aien-sovereign-core

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)

The Rust workspace of the AIEN host runtime and its verification tooling, running on Linux on the NVIDIA DGX Spark (Grace Blackwell GB10). It is where AIEN serves models today, where every change to the project is judged, and where releases are built and installed.

What lives here:

- **Host runtime.** The `aien` command line and the single-process runtime daemon (`crates/aien-cli`, `crates/aien-runtime`): model loading, the inference ABI (`crates/aien-inference-abi`), a unified-memory paged KV cache with copy-on-write branching (`crates/aien-kv-cache`, persistent form in `crates/aien-kv-store`), a continuous-batching scheduler with chunked prefill (`crates/aien-scheduler`), and an OpenAI-compatible HTTP door (`crates/aien-serve-http`).
- **GPU through our own kernels.** The `crates/aien-omega-*` crates bind the Omega GPU engine from the [omega](https://github.com/aien-spark/omega) repository, pinned by `omega.lock`: matmul, bf16 and FP8 GEMV, MoE routing, prefill and decode attention, and the one-command-list decode step for Qwen3-30B-A3B. No CUDA toolkit anywhere.
- **The judge.** `tools/aien-test` and `tools/aien-local-ci` are the AIEN verification pipeline ([ADR 0033](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0033-aien-verification-pipeline.md)): gate manifests, verdicts, immutable evidence receipts and signed bundles. Every pull request in every AIEN repository is judged on the Spark by this code before it merges; GitHub-hosted runners only check the signed bundle.
- **Releases.** `install.sh`, the release scripts and `docs/release/` build, sign, verify and install AIEN; `crates/aien-verify` and `crates/aien-campaign-verifier` check runs and evidence campaigns without linking the code under test.
- **Identity and agents.** ALLEN, the persistent cognitive entity boundary ([ADR 0035](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0035-persistent-cognitive-entity-boundary-allen.md)) in `crates/aien-allen*`; the agent harness, MCP, capability and switchboard crates.
- **Supporting services and scaffolding.** Cortex memory (`crates/cortex-rs`, non-authoritative since omega owns Cortex under ADR 0022), the `spark-*` services, Crumb v1 (`crates/crumbs`) and the shared test board (`crates/aien-proof`).

**Where it sits in AIEN.** [ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md) sets the language rule: Rust is the scaffolding, Omega is the destination, and C or assembly stay only where hardware, boot, ABI or measurement justify them. Working Rust here is kept and extended, never rewritten into C as a migration step. Each subsystem sits behind a deliberately boring, language-neutral boundary so that Omega can replace it one piece at a time after passing the same conformance tests. Component ownership across repositories is recorded in [aien-architecture](https://github.com/aien-spark/aien-architecture). Earlier wording that called this repository legacy is withdrawn; its history is in the git log and in ADR 0024.

## Current state

Research-grade and pre-alpha. Crates build and have unit tests on the DGX Spark; execution is CPU plus GPU through AIEN's own Omega kernels. Modular MAX and Mojo were removed from the project on 2026-10-15. Nothing here is qualified on hardware as a finished system, and no public release exists yet. Live status and sequencing: [CURRENT_EXECUTION_PLAN.md](https://github.com/aien-spark/aien-architecture/blob/main/CURRENT_EXECUTION_PLAN.md) and the [implementation status snapshot](https://github.com/aien-spark/aien-architecture/blob/main/docs/02-implementation-status.md). Component detail and platform status: [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md), [docs/AIEN_RUNTIME_ARCHITECTURE.md](docs/AIEN_RUNTIME_ARCHITECTURE.md). Repository vocabulary and ownership notes: [CONTEXT.md](CONTEXT.md).

Known limits: the GPU fallback counter covers only instrumented operations, so execution is not exclusively accelerated; Qwen full-runtime qualification is not complete; the C1 evidence campaign is in progress.

## Measured results

Every headline performance number must resolve to a reproducible command and an artifact bundle (commit identity, hardware and environment record, exact command, raw samples, SHA-256 digests, measurement definition, reproducibility steps). Speed is reported against the machine's physical limit and against our own previous run, never against other people's systems; correctness against our own reference. Earlier figures (branch fork latency, COW page mutation latency, memory sharing ratios, service comparison table) are withdrawn until regenerated to that standard. The evidence bundles live in [aien-spark/benchmarks](https://github.com/aien-spark/benchmarks).

## Standing rules

Rust is scaffolding, Omega is the destination, C or assembly only where hardware, boot, ABI or measurement justify them ([ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md)). No Python anywhere in repos, build, CI or tooling (`scripts/check-no-python.sh` fails the build if a Python file is added). No CUDA toolkit and no new CUDA dependence. No systemd in AIENOS. No outside dependencies in the trusted base. No Modular MAX or Mojo anywhere in the tree. No comparison with other people's systems. Kernels are Omega or Rust: closest to the metal, fastest wins, build what is missing.

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
