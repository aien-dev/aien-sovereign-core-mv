# aien-sovereign-core

[![License: AGPL-3.0-or-later](https://img.shields.io/badge/License-AGPL--3.0--or--later-blue.svg)](LICENSE)

This repository is the Rust workspace of the AIEN host runtime and its verification tooling. It runs on Linux on the NVIDIA DGX Spark (Grace Blackwell GB10). It holds the `aien` command line and runtime daemon, the model serving path that runs on AIEN's own Omega GPU kernels, the test judge that checks every AIEN pull request, and the scripts that build, sign and install releases.

## Where it fits in AIEN

[Aienix](https://github.com/aien-dev/aienix) is the core tree of AIEN and the best place to start. The sibling repositories this one works with:

- [aien-architecture-mv](https://github.com/aien-dev/aien-architecture-mv): decision records (ADRs), the execution plan, and which repository owns which component.
- [omega-mv](https://github.com/aien-dev/omega-mv): the Omega language and GPU engine. This repository links Omega at the commit pinned in `omega.lock`.
- [physics-mv](https://github.com/aien-dev/physics-mv) and [aienos-mv](https://github.com/aien-dev/aienos-mv): pinned by omega's own lock files and checked out next to this repository by some tests.
- [aien-protocols-mv](https://github.com/aien-dev/aien-protocols-mv): specifications and test vector corpora used by the SC-TRN1 and SC-CRB1 gates.
- [benchmarks-mv](https://github.com/aien-dev/benchmarks-mv): measured evidence bundles.
- [aien-receipts](https://github.com/aien-dev/aien-receipts): signed verification bundles the judge publishes for each pull request head.

The language rule is [ADR 0024](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md): Rust is scaffolding, Omega is the destination, and C or assembly stay only where hardware, boot, ABI or measurement justify them. Working Rust here is kept, not rewritten into C. Each part sits behind a plain, language-neutral boundary so Omega can replace it one piece at a time after passing the same conformance tests. Earlier wording that called this repository "legacy" is withdrawn by ADR 0024.

## Status (2026-10-19)

Research-grade and pre-alpha. Nothing here is qualified on hardware as a finished system.

- **Releases.** No public release exists. v0.1.0 (2026-09-20) and v0.1.1 (2026-10-04) are internal test pre-releases (`gh release list`). The public release key is not pinned yet: `AIEN_RELEASE_SIGNER` is empty in `install.sh` until the key ceremony in [docs/release/KEY_CEREMONY.md](docs/release/KEY_CEREMONY.md). The release gate (`scripts/check-release-candidate.sh`) fails on main because `omega.lock` differs from the CAND-4 manifest; tracked in [#414](https://github.com/aien-dev/aien-sovereign-core-mv/issues/414), open on 2026-10-19.
- **Testing.** Since 2026-10-18 the Spark is the only test judge ([#671](https://github.com/aien-dev/aien-sovereign-core-mv/pull/671) removed the hosted test workflows). The only workflows left are `aien-verification.yml` (checks the signed bundle, builds no repository code) and the three `release*.yml` workflows.
- **Inference.** The GPU path runs on Omega kernels; no CUDA toolkit is used. Modular MAX and Mojo were removed on 2026-10-15 ([#529](https://github.com/aien-dev/aien-sovereign-core-mv/pull/529)). Known limits: the GPU fallback counter covers only instrumented operations, so not every operation is proven to run on the GPU, and full-runtime qualification of the Qwen models is not complete.
- **Live plan.** [CURRENT_EXECUTION_PLAN.md](https://github.com/aien-dev/aien-architecture-mv/blob/main/CURRENT_EXECUTION_PLAN.md) and the [implementation status snapshot](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/02-implementation-status.md) in aien-architecture-mv. Component and platform detail: [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md), [docs/AIEN_RUNTIME_ARCHITECTURE.md](docs/AIEN_RUNTIME_ARCHITECTURE.md). Words used in this repository: [CONTEXT.md](CONTEXT.md).

## What is in the tree

The workspace has 57 packages (`cargo metadata --no-deps`): 56 under `crates/` plus `tools/aien-test`.

- **Host runtime.** `crates/aien-cli` (the `aien` command), `crates/aien-runtime` (the single-process runtime daemon), `crates/aien-inference-abi` and `crates/aien-inference-runtime` (backend interface and model execution), `crates/aien-kv-cache` (paged KV cache in unified memory, with copy-on-write branching and prefix caching), `crates/aien-kv-store` (persistent KV cache), `crates/aien-scheduler` (continuous batching with chunked prefill), and `crates/aien-serve-http` (an HTTP server with `/v1/models`, `/v1/chat/completions` and `/health` routes that speak the common chat-completions request format). The KV cache keeps attention state between tokens; continuous batching means new requests join a running batch.
- **GPU through Omega.** The `crates/aien-omega-*` crates call Omega's GPU engine through its C interface: matmul, bf16 and FP8 matrix-vector products, the MoE router, prefill and decode attention, and the one-command-list decode step for Qwen3-30B-A3B (the OWN-30B engine). `crates/aien-abi-conformance` checks the Rust copies of Omega's C headers against the pinned headers. Without an Omega checkout these crates build a stub and print a warning (see [crates/aien-omega-compose/README.md](crates/aien-omega-compose/README.md)).
- **The judge.** `tools/aien-test` and `tools/aien-local-ci` are the AIEN verification pipeline ([ADR 0033](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0033-aien-verification-pipeline.md)): gate manifests, verdicts, evidence receipts and signed bundles. The gates for this repository are the `.gate` files in `tests/gates/` (25 on 2026-10-19), all run through `tests/gates/run.sh`. How the Spark side works: [tools/aien-local-ci/README.md](tools/aien-local-ci/README.md).
- **Release and independent checks.** `install.sh`, `scripts/` and [docs/release/](docs/release/). `tools/aien-verify` (builds alone with `rustc`, no cargo) and `crates/aien-campaign-verifier` check runs and evidence without linking the code under test. `crates/aien-closure` checks that every internal dependency has verified evidence.
- **Identity and agents.** ALLEN, the persistent identity boundary of [ADR 0035](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0035-persistent-cognitive-entity-boundary-allen.md), in `crates/aien-allen`, `crates/aien-allen-memory` and `crates/aien-allen-profile`. Agent tooling in `crates/spark-harness`, `crates/aien-mcp`, `crates/aien-capability`, `crates/aien-switchboard` and `crates/aien-drive`.
- **Supporting services.** `crates/cortex-rs` (a memory service; not the canonical memory owner, see [ADR 0022](https://github.com/aien-dev/aien-architecture-mv/blob/main/docs/adr/0022-canonical-cortex-owner.md)), the other `crates/spark-*` services, `crates/crumbs` and `crates/aien-proof` (shared test board).

## Build and test

Needs Rust 1.98.1, the version pinned in `rust-toolchain.toml` (rustup installs it on first use), on the DGX Spark or another Linux machine.

```bash
git clone https://github.com/aien-dev/aien-sovereign-core-mv
cd aien-sovereign-core-mv
CI=true AIEN_DEV_FALLBACK=1 cargo test --workspace --verbose -- --test-threads=1
scripts/repro-build.sh -p aien-cli
```

The `cargo test` line is the command the SC-TEST gate runs (`tests/gates/run.sh test`). `CI=true` makes the tests that need live services skip themselves. `AIEN_DEV_FALLBACK=1` allows CPU fallback for development runs; a production build refuses it. `scripts/repro-build.sh` makes a release build with machine-local paths removed. Build `aien-cli` on its own, because its digest changes when other packages are built in the same cargo command ([docs/release/F1-binary-digest-difference.md](docs/release/F1-binary-digest-difference.md)).

To see which gates a change selects, run this inside a checkout on the Spark:

```bash
aien-local-ci plan --base origin/main
```

Some gates need sibling checkouts next to this one (`../omega`, `../physics`, `../aienos`, `../aien-protocols`); see the header of `tests/gates/run.sh`.

## Installing

State on 2026-10-11, from `install.sh` and the dry-run artifact of Actions run 38105672607:

- `install.sh` with no arguments builds from source and verifies no signature. With `AIEN_RELEASE_TAG=vX.Y.Z` it installs a published release instead: it downloads the archive, `SHA256SUMS.txt` and `SHA256SUMS.txt.sig`, checks the signature with `ssh-keygen`, then the archive checksum, and writes `install-receipt.json` with a `signature_check` of `release`, `internal-test-key` or `dry-run-key` (see [docs/release/RELEASE_SIGNING.md](docs/release/RELEASE_SIGNING.md)). Nothing can install as `release` until the public key is pinned.
- After an install, follow [docs/FIRST_RUN.md](docs/FIRST_RUN.md): create the desk key (`aien compose desk-key --create 1`; `aien start` refuses without it), provision the model, start, record an objective. Objective records default to a per-user folder (`~/.local/state/aien-runtime/provenance`), not `/tmp`. The installer does not provision a model. `aien harness` is not shipped in this release and exits with a refusal.

## Evidence

Every headline performance number must resolve to a reproducible command and an artifact bundle (commit identity, hardware and environment record, exact command, raw samples, SHA-256 digests, measurement definition, reproducibility steps). Speed is reported against the machine's physical limit and our own earlier runs; correctness against our own reference. Earlier figures (branch fork latency, copy-on-write page mutation latency, memory sharing ratios, service comparison table) are withdrawn until regenerated to that standard.

- [docs/inference/evidence/](docs/inference/evidence/): dated inference runs, one folder per run.
- [benchmarks/](benchmarks/) and `benchmarks/data/`: build and KV measurements, for example [benchmarks/headstart/README.md](benchmarks/headstart/README.md).
- [benchmarks-mv](https://github.com/aien-dev/benchmarks-mv): evidence bundles.
- [aien-receipts](https://github.com/aien-dev/aien-receipts): the signed judge bundle for each verified pull request head.

## Standing rules

- Rust is scaffolding, Omega is the destination, C or assembly only where hardware, boot, ABI or measurement justify them (ADR 0024).
- No Python anywhere in repos, build, CI or tooling. `scripts/check-no-python.sh` fails the build if a Python file is added.
- No CUDA toolkit and no new CUDA dependence. `scripts/zero-cuda-gate.sh` (gate SC-ZERO-CUDA) fails an `aien-cli` build that needs a CUDA library.
- No systemd in AIENOS. No outside dependencies in the trusted base. No Modular MAX or Mojo anywhere in the tree.
- No comparison with other people's systems.

## Contributing

Read [CONSTITUTION.md](CONSTITUTION.md), [AGENTS.md](AGENTS.md) and [CONTRIBUTING.md](CONTRIBUTING.md) before opening a pull request. Changes ship on a branch with verification proof: the commands you ran and their output. A pull request merges only after the Spark judge passes its exact head commit and it carries the `ready-to-merge` label; the verdict of record is the commit status "AIEN Verification (Spark)". Keep new Rust behind language-neutral boundaries so Omega can take each piece over later (ADR 0024). Contact: aien@aienos.com.

## License and values

Licensed under the GNU Affero General Public License v3.0 or later (AGPL-3.0-or-later). All 57 workspace packages declare it (`cargo metadata --no-deps`). See [LICENSE](LICENSE) and [NOTICE](NOTICE); other license texts (Apache-2.0, CC0-1.0, LLVM-exception) are in [LICENSES/](LICENSES/), and [ARTIFACT_LICENSING.md](ARTIFACT_LICENSING.md) covers artifacts. Values live in the nonbinding [COVENANT.md](COVENANT.md): keep foundational advances open. It grants and restricts no legal rights. Built for NVIDIA Blackwell hardware; full notices in [ATTRIBUTION.md](ATTRIBUTION.md).
