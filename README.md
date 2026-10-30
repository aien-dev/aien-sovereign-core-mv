# aien-sovereign-core

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](LICENSE)

The Rust workspace of the AIEN host runtime and its verification tooling, running on Linux on the NVIDIA DGX Spark (Grace Blackwell GB10). It is where AIEN serves models today, where every change to the project is judged, and where releases are built and installed.

What lives here:

- **Host runtime.** The `aien` command line and the single-process runtime daemon (`crates/aien-cli`, `crates/aien-runtime`): model loading, the inference ABI (`crates/aien-inference-abi`), a unified-memory paged KV cache with copy-on-write branching (`crates/aien-kv-cache`, persistent form in `crates/aien-kv-store`), a continuous-batching scheduler with chunked prefill (`crates/aien-scheduler`), and an OpenAI-compatible HTTP door (`crates/aien-serve-http`).
- **GPU through our own kernels.** The `crates/aien-omega-*` crates bind the Omega GPU engine from the [omega](https://github.com/aien-spark/omega) repository, pinned by `omega.lock`: matmul, bf16 and FP8 GEMV, MoE routing, prefill and decode attention, and the one-command-list decode step for Qwen3-30B-A3B. No CUDA toolkit anywhere.
- **The judge.** `tools/aien-test` and `tools/aien-local-ci` are the AIEN verification pipeline ([ADR 0033](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0033-aien-verification-pipeline.md)): gate manifests, verdicts, immutable evidence receipts and signed bundles. Every pull request in every AIEN repository is judged on the Spark by this code before it merges; GitHub-hosted runners only check the signed bundle.
- **Releases.** `install.sh`, the release scripts and `docs/release/` build, sign, verify and install AIEN; `crates/aien-verify` and `crates/aien-campaign-verifier` check runs and evidence campaigns without linking the code under test.
- **Identity and agents.** ALLEN, the persistent cognitive entity boundary ([ADR 0035](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0035-persistent-cognitive-entity-boundary-allen.md)) in `crates/aien-allen*`; the agent harness, MCP, capability and switchboard crates.
- **Supporting services and scaffolding.** Cortex memory (`crates/cortex-rs`, non-authoritative since omega owns Cortex under ADR 0022), the `spark-*` services, Crumb v1 (`crates/crumbs`) and the shared test board (`crates/aien-proof`).

**Where it sits in AIEN.** [ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md) sets the language rule: Rust is the scaffolding, Omega is the destination, and C or assembly stay only where hardware, boot, ABI or measurement justify them. Working Rust here is kept and extended, never rewritten into C as a migration step. Each subsystem sits behind a deliberately boring, language-neutral boundary so that Omega can replace it one piece at a time after passing the same conformance tests. The first piece in Omega is the effect ledger's restart world check ([om/](om/README.md)): byte-identical to the Rust on its parity fixtures, not yet wired into the daemon. Component ownership across repositories is recorded in [aien-architecture](https://github.com/aien-spark/aien-architecture). Earlier wording that called this repository legacy is withdrawn; its history is in the git log and in ADR 0024.

## Current state

Research-grade, and in alpha testing for one path: the AienOS alpha bundle (distributed from aien-dev/aienos-releases) carries the inference engine built from this repository's main. That GPU path is qualified only on the DGX Spark (GB10, sm_121), and the three-run physical qualification is NOT_RUN. The rest of this repository (the `aien` command line tools, Cortex, the cockpit, the daemon) is not in that bundle and stays pre-alpha. Crates build and have unit tests on the DGX Spark; execution is CPU plus GPU through AIEN's own Omega kernels. Modular MAX and Mojo were removed from the project on 2026-10-15. Nothing here is qualified on hardware as a finished system, and no public release of this repository exists yet. Live status and sequencing: [CURRENT_EXECUTION_PLAN.md](https://github.com/aien-spark/aien-architecture/blob/main/CURRENT_EXECUTION_PLAN.md) and the [implementation status snapshot](https://github.com/aien-spark/aien-architecture/blob/main/docs/02-implementation-status.md). Component detail and platform status: [docs/PLATFORM_MATRIX.md](docs/PLATFORM_MATRIX.md), [docs/AIEN_RUNTIME_ARCHITECTURE.md](docs/AIEN_RUNTIME_ARCHITECTURE.md). Repository vocabulary and ownership notes: [CONTEXT.md](CONTEXT.md).

Qwen3.8-27B (the hybrid DeltaNet + attention model, FP8) has a host reference in `crates/aien-inference-abi/src/qwen38.rs`: on 256 tokens of the Apache license it predicts the next token 98.8% of the time (perplexity 1.091, [evidence](docs/inference/evidence/qwen38-ref-20261018/NOTES.md)). `aien-serve-http --engine omega` serves it on the Spark GPU through AIEN's own Omega engine: about 7.6 tokens per second while writing an answer, and greedy output equal to the host reference on the gate prompt ([evidence](evidence/qwen38-serve-fast-20261019/README.md)). Text chat only, one request at a time, no tool calls.

The OWN-30B server (`aien-serve-http --engine omega`, Qwen3-30B-A3B FP8) holds up to 128k context with `--max-context 131072` (default 32768) and refuses to start when its KV cache cannot hold what was asked. On the Spark GPU it recalled a code word from the first line of a 118,186-token prompt ([evidence](docs/inference/evidence/own30b-128k-20261018/NOTES.md)).

Known limits: the GPU fallback counter covers only instrumented operations, so execution is not exclusively accelerated; Qwen full-runtime qualification is not complete; the C1 evidence campaign is in progress.

## Measured results

Every headline performance number must resolve to a reproducible command and an artifact bundle (commit identity, hardware and environment record, exact command, raw samples, SHA-256 digests, measurement definition, reproducibility steps). Speed is reported against the machine's physical limit and against our own previous run, never against other people's systems; correctness against our own reference. Earlier figures (branch fork latency, COW page mutation latency, memory sharing ratios, service comparison table) are withdrawn until regenerated to that standard. The evidence bundles live in [aien-spark/benchmarks](https://github.com/aien-spark/benchmarks).

- Expert Heartbeat H1 ring streamer (aienix#4): verified expert rotation from NVMe at 10.36 GB/s payload, 2.80 s per full 6,144-expert beat; method, receipts and limits in [crates/aien-heartbeat](crates/aien-heartbeat/README.md). Measured with aien-calc receipts; not yet packaged as the artifact bundle this section requires.

The one deliberate exception is bar P0 of the GPU-stack plan: a predeclared, same-Spark, end-to-end head-to-head of this repo's `aien-serve-http` against vLLM and llama.cpp on the same Qwen3.6-35B-A3B checkpoint. It lives in its own repo, [aien-dev/gpu-head-to-head](https://github.com/aien-dev/gpu-head-to-head), which pins the measured binary's source commit `f0230a59` and publishes the result even where this stack loses.

## Standing rules

Rust is scaffolding, Omega is the destination, C or assembly only where hardware, boot, ABI or measurement justify them ([ADR 0024](https://github.com/aien-spark/aien-architecture/blob/main/docs/adr/0024-rust-scaffolding-omega-destination.md)). No Python anywhere in repos, build, CI or tooling (`scripts/check-no-python.sh` fails the build if a Python file is added). No CUDA toolkit and no new CUDA dependence. No systemd in AIENOS. No outside dependencies in the trusted base. No Modular MAX or Mojo anywhere in the tree. No comparison with other people's systems. Kernels are Omega or Rust: closest to the metal, fastest wins, build what is missing.

## Build and test

Needs Rust 1.98.1, the version pinned in `rust-toolchain.toml` (rustup installs it on first use), on the DGX Spark or another Linux machine.

```bash
git clone https://github.com/aien-dev/aien-sovereign-core-mv
cd aien-sovereign-core-mv
cargo test --workspace -- --test-threads=1   # one test thread, as CI runs it
scripts/repro-build.sh -p aien-cli   # release build with machine-local paths removed; build aien-cli on its own, because its digest changes with the other packages in the same cargo command (docs/release/F1-binary-digest-difference.md)
```

### Model server on a DGX Spark (Qwen3.8-27B, source build)

Only on an NVIDIA DGX Spark (aarch64 Linux, GB10, NVIDIA open driver 580.173.02 tested). No CUDA toolkit is needed: Omega talks to the driver directly. The installer checks the machine first and stops with a plain message when something is missing; it never installs a server that has only the CPU stub.

```bash
git clone https://github.com/aien-dev/aien-sovereign-core-mv && cd aien-sovereign-core-mv
./install.sh --with-server     # CLI tools as usual, plus aien-serve-http and the start script aien-serve-27b
# weights: Hugging Face Qwen/Qwen3.8-27B-FP8, revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a,
# into ~/models/qwen3.8-27b-fp8-017b9c7a (about 31 GB)
aien-serve-27b                 # http://127.0.0.1:18091/v1, prints how to stop it
aien-serve-http --version      # shows whether the Omega engine is linked (native) or a stub
```

`tools/qwen38_serve_gate.sh http://127.0.0.1:18091 "" <out-dir>` checks a running server against the shipped host reference (`crates/aien-inference-abi/fixtures/qwen38/`), and `tools/qwen38_serve_speed.sh http://127.0.0.1:18091 "" <out-dir>` measures it; both write a `receipt.json` (binary sha256, build, model folder, machine state, figures, MATCH or MISMATCH). Model checksums, the memory speed ceiling and how to check: [docs/qwen38/](docs/qwen38/README.md). The loader needs 49,497 MiB (48.3 GiB) of free memory when it loads (41,305 MiB plus 8,192 MiB kept free) and refuses with less; once loaded, the server peaked at about 36.7 GiB on the current build (evidence/qwen38-serve-memory-20261019).

The Core-main HTTP server now requires `AIEN_SERVE_HTTP_BEARER_TOKEN` at startup and
`Authorization: Bearer <token>` on inference requests. Merging this source change
does **not** update the existing Spark service on port 18091, which runs a binary
from the separate `ship/motor-l387` line. A future, separately authorized
cutover must provision the server credential and update all known clients in
the same window: omp's spark provider, `allen-courier` (installed under
`~/.local/state/aienos/allen/bin/allen-courier-*`), aien-translate,
drake-voice, and `allen-chat`, the public www.aienos.com/allen/ gateway (system
unit `allen-chat.service`, user `allen-chat`; it reads the bearer from
`ALLEN_UPSTREAM_TOKEN` in `/etc/allen-chat/env`, empty until the cutover, and
sends it on chat and `/health` requests). Check those clients before replacing
the live binary; otherwise they will receive 401 responses. No client
credentials are stored in this repo.

Step by step for a Spark with nothing of AIEN on it (requirements, install, model download and checksum, start, first request, the checks, known limits, how to report): [docs/qwen38/TRY_IT_ON_YOUR_SPARK.md](docs/qwen38/TRY_IT_ON_YOUR_SPARK.md). What "alpha" covers and what stays pre-alpha: [docs/qwen38/ALPHA_STATUS.md](docs/qwen38/ALPHA_STATUS.md).

Installing a release (state on 2026-10-11, from `install.sh` and the dry-run artifact of Actions run 38105672607, nothing else):

- `install.sh` with no arguments builds from source and verifies no signature. With `AIEN_RELEASE_TAG=vX.Y.Z` it installs a published release instead: it downloads the archive, `SHA256SUMS.txt` and `SHA256SUMS.txt.sig`, checks the signature with `ssh-keygen`, then the archive checksum, and writes `install-receipt.json` with a `signature_check` of `release`, `internal-test-key` or `dry-run-key` (see [docs/release/RELEASE_SIGNING.md](docs/release/RELEASE_SIGNING.md)).
- No public release exists. The public release key is not pinned yet (`AIEN_RELEASE_SIGNER` is empty until the key ceremony in [docs/release/KEY_CEREMONY.md](docs/release/KEY_CEREMONY.md)), so nothing can install as `release`. v0.1.0 and v0.1.1 are internal test releases. The release gate (`scripts/check-release-candidate.sh`) fails on current main (omega.lock differs from the CAND-4 manifest), see [#414](https://github.com/aien-dev/aien-sovereign-core-mv/issues/414).
- After an install, follow [docs/FIRST_RUN.md](docs/FIRST_RUN.md): create the desk key (`aien compose desk-key --create 1`; `aien start` refuses without it), provision the model, start, record an objective. Objective records default to a durable per-user folder (`~/.local/state/aien-runtime/provenance`), not `/tmp`. The installer does not provision a model. `aien harness` is not shipped in this release and exits with a refusal.

## Agent CLI: approval floors and rollback

`aien-cli` checks every tool request in one fixed order (issue #667). Four floors come first and hold in every approval mode, including `yolo` and `off`: a list of catastrophic commands (wiping a system root, formatting or writing a raw disk, a fork bomb, killing every process), deleting or rewriting the runtime's own files (its binary, the approval policy, the checkpoint store, session journals), `sudo` fed a password on stdin, and the operator's own deny patterns. Only then come the existing policy denials, the mode (`AIEN_APPROVAL_MODE`: `manual`, `smart`, `yolo`, `off`), the operator's allowlist, whether a person is present (unattended runs deny anything that needs a decision), the dangerous-command patterns, and the decision itself (a guardian model in `smart` mode, behind a breaker that opens after 3 failures, else the person). Deny and allow patterns live in `~/basecamp/approval-policy.json` (`{"mode": "...", "deny": [regex...], "allow": [regex...]}`); a pattern that does not compile, or a file that does not parse, denies everything until fixed. `/approval` shows the active settings.

Every file the agent writes in a chat session is recorded: one snapshot of the directory per turn in a shared content-addressed store (`~/basecamp/checkpoints`), plus a hash-chained ledger of the agent's own writes. `/rollback [turn] [session]` puts back only the files the agent wrote, and only where they still hold what the agent left; files the operator changed are kept and listed. `/checkpoints` lists the turns. Shown on the host by unit and dispatcher tests; not yet run against a live model.

Mechanism after Hermes Agent (Nous Research, MIT licence, commit 5480918), re-implemented from reading; no code copied.

## Contributing

Read [CONSTITUTION.md](CONSTITUTION.md) and [AGENTS.md](AGENTS.md) before opening a pull request. Changes ship on a branch with verification proof: the commands you ran and their output. Prefer work that moves capability into the C repositories over new Rust here. Contact: aien@aienos.com.

## License and values

Licensed under the Apache License, Version 2.0 (Apache-2.0). See [LICENSE](LICENSE) and [NOTICE](NOTICE). Values live in the nonbinding [COVENANT.md](COVENANT.md): keep foundational advances open. It grants and restricts no legal rights. Built for NVIDIA Blackwell hardware; full notices in [ATTRIBUTION.md](ATTRIBUTION.md).
