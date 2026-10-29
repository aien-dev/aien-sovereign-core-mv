# Audit scope

Written for an outside, independent security audit. Revision described:
`f30eb963918d71dd2998b1cb936c251429417de1` on `main`, 2026-10-26. Everything below is taken from
the repository's own files at that revision; items marked NOT_RUN were not executed while writing
this document.

## What this repository is

The Rust workspace of the AIEN host runtime and its verification tooling, running on Linux on the
NVIDIA DGX Spark (GB10). It contains the `aien` command line and runtime daemon, model loading, a
paged KV cache, a batching scheduler, an OpenAI-compatible HTTP server, Rust bindings to the Omega
GPU engine (`crates/aien-omega-*`, pinned by `omega.lock`), the AIEN verification pipeline
(`tools/aien-test`, `tools/aien-local-ci`), release and install scripts, ALLEN identity crates, and
supporting `spark-*` services. Status (README): research-grade; the GPU inference path is in alpha
testing on the DGX Spark only and its three-run physical qualification is NOT_RUN; the rest is
pre-alpha; no public release exists.

## What to review, in priority order

1. Verification pipeline: `tools/aien-test` (gate manifests, verdicts, evidence receipts, signed
   bundles, `verify-bundle`), `tools/aien-local-ci` (worker, `pipeline.sh`, `aien-run-exec`,
   `forgejo-webhook.sh`, `siblings`), `tools/aien-verify`, `crates/aien-verify`,
   `crates/aien-campaign-verifier`. Every pull request in every AIEN repository is judged by this
   code; check sandbox escape, verdict forgery, bundle signature checks, and the
   `BLOCKED_OPERATOR` rule for pipeline-config changes (`tools/aien-test/docs/LOCAL_VERIFICATION.md`).
2. `crates/aien-serve-http`: bearer-token check (`AIEN_SERVE_HTTP_BEARER_TOKEN`, required at
   startup), request parsing, context limits (`--max-context`), refusal when the KV cache cannot
   hold the request.
3. `crates/aien-cli`: approval floors (catastrophic commands, runtime self-files, `sudo` with a
   password on stdin, operator deny patterns), approval modes (`AIEN_APPROVAL_MODE`), unattended
   denial, and the per-turn snapshot store and hash-chained write ledger behind `/rollback`.
4. Release and install: `install.sh`, `install.ps1`, `install-inquisitor.sh`,
   `scripts/release-publish-gate.sh`, `scripts/check-release-candidate.sh`,
   `docs/release/RELEASE_SIGNING.md`, `docs/release/allowed_signers`,
   `docs/release/internal_test_signers`, `release/candidate.toml`.
5. Capability and agent boundary crates: `aien-capability`, `aien-mcp`, `aien-switchboard`,
   `aien-security`, `spark-aegis`, `aien-allen*`.
6. Native linking: `crates/aien-abi-conformance` build script checking omega archive digests
   against `native-artifacts.lock`; `crates/aien-omega-*` FFI.
7. Workflows: `.github/workflows/aien-verification.yml` (advisory checks and Alpha bundle
   verification, `contents: read`), `release.yml` (tag `v*` or manual; publish job has
   `contents: write`), `release-dry-run.yml` (manual, read only, no secrets), `release-publish.yml`
   (manual, workflow default token has no permissions, one job with `contents: write`), and
   `.forgejo/workflows/ci.yml` (push and pull request on a self-hosted runner labelled `spark`,
   uses `actions/checkout@v4` by tag rather than sha).
8. `tests/gates/run.sh` and the gate manifests.

## Trust boundaries

- Untrusted: HTTP clients of `aien-serve-http`; model output and tool requests reaching the agent
  CLI; candidate code in any pull request the pipeline tests; downloaded release archives before
  signature verification.
- Trusted: the Spark signing key (public half in `.aien/verify.pub` on `main`); release signers in
  `docs/release/allowed_signers`; pinned omega archives (`native-artifacts.lock`); the pinned
  toolchain (`rust-toolchain.toml`, Rust 1.98.1); the operator present at the desk.
- The public release key is not pinned yet (`AIEN_RELEASE_SIGNER` is empty until the key
  ceremony), so nothing can install with `signature_check` `release`.

## Threat model summary

- Assets: integrity of verdicts and signed bundles for all AIEN repositories; release signing;
  files on the operator's machine reachable by the agent; inference service availability.
- Attackers considered: a malicious pull request (to any AIEN repository) trying to reach a PASS
  or escape the sandbox; an unauthenticated network client of the HTTP server; a prompt-injected
  model issuing destructive tool requests; a tampered release archive.
- Defended: trust anchors read from the base ref, not the head; gate changes run only by the
  operator; bearer token on inference requests; approval floors that hold in every mode; signed
  release checksums verified with `ssh-keygen`; recorded native archive digests.
- Not defended / explicitly out of scope: no public release and no pinned release key; the release
  gate fails on current main (README, issue #414); the GPU fallback counter covers only
  instrumented operations; the agent rollback path has not run against a live model (README).

## Build from source at a pin

Rust 1.98.1 from `rust-toolchain.toml`. Native pins: `omega.lock`
(`a647b6a9821fffcf52ba9ce813f75afa4b84be62`), `native-artifacts.lock` (sha256 per omega archive,
target and compiler), `crumb.lock`.

```sh
git clone https://github.com/aien-dev/aien-sovereign-core-mv
cd aien-sovereign-core-mv
cargo test --workspace -- --test-threads=1
scripts/repro-build.sh -p aien-cli
```

The GPU server build (`./install.sh --with-server`) needs a DGX Spark with the NVIDIA open
driver; NOT_RUN on a plain host.

## Run every test

```sh
cargo fmt --all -- --check
cargo check --workspace --all-targets
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=1
tests/gates/run.sh <step>
```

Gate steps in `tests/gates/run.sh` include `crumb-verify`, `fmt`, `check`, `clippy`, `test`,
`strict-mode`, `zero-cuda`, `zero-disk-secrets`, `anti-slop`, `inquisitor`, `release-gate-negatives`,
`release-dry-run`, `aien-verify-tests`, `pins-selftest`, `native-identity`, `install-test`,
`scoring-v5`, `preflight-guard`, `agent-preflight`, `linked-compose`, `gb10-serve-chat`,
`gb10-model-dir`, `kv-props`, `miri`, `trn1`, `crb1` and `verify-closure`. `.aien/always` runs
`CRUMB-VERIFY`, `SC-ZERO-CUDA`, `SC-ZERO-DISK-SECRETS`, `SC-ANTI-SLOP` and `SC-INQUISITOR` on every
change.

The DGX Spark is the test judge: it runs these gates and signs a bundle. GitHub checks are
advisory (workflow header: nothing on GitHub gates a merge). `SC-GB10-SERVE-CHAT` requires the GB10
and a downloaded model; NOT_RUN on a plain host. All commands above: NOT_RUN while writing this
document.

## Known limitations and NOT_RUN

- GPU path three-run physical qualification: NOT_RUN.
- Qwen full-runtime qualification incomplete; C1 evidence campaign in progress.
- `.forgejo/workflows/ci.yml` runs pull request code on a self-hosted runner and pins
  `actions/checkout` by tag; whether this runner is active is unknown from the repository.
- Some `evidence/` directories are named `INVALID` (for example
  `qwen38-omega-full-20261018T1641Z-INVALID-oom`); they are kept as failed records.

## Dependencies and provenance

- `Cargo.lock` for the workspace; crates.io dependencies.
- Native archives from omega, digests in `native-artifacts.lock`; mismatches fail `build.rs`.
- Licences: Apache-2.0; third-party notices in `THIRD_PARTY.md`, `ATTRIBUTION.md`, `LICENSES/`.
- No Python (`scripts/check-no-python.sh`); no CUDA toolkit (`zero-cuda` gate).
- Model weights are downloaded separately from Hugging Face at a pinned revision named in the
  README; they are not in the repository.

## Evidence

- `evidence/` and `docs/inference/evidence/` hold run records; `docs/release/receipts/` holds
  release receipts.
- Signed Spark bundles: `aien-dev/aien-receipts` at `aien-sovereign-core/<sha>/bundle.json`
  (workflow path). Receipts are append-only and never edited.
