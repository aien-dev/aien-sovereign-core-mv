# Forgejo Actions CI Standard Operating Practice

This document defines the standard practice for automated continuous integration across the AIEN ecosystem.

## Core Policy

All continuous integration pipelines, automated invariant verifications, test suites, and lint checks execute natively on the NVIDIA DGX Spark workstation via Forgejo Actions.

Public GitHub repositories maintain GitHub Actions workflow definitions configured strictly for manual execution (`workflow_dispatch`). Automated execution on GitHub is disabled to eliminate billable cloud runner minutes and avoid recurring compute fees.

## Architectural Advantages

1. **Zero Cloud Compute Fees**: CI tasks run on owned, dedicated local hardware with zero marginal cost.
2. **Native Grace Blackwell GB10 Access**: Tests and benchmarks execute directly on Grace Neoverse V2 ARM64 cores and Blackwell silicon, providing realistic production runtime telemetry rather than generic x86 virtual machines.
3. **High-Speed Execution**: Local NVMe storage, shared compiler artifact caches, and 121 GB unified memory reduce workspace build and test times from minutes down to seconds.
4. **Hardware TPM Integration**: Verification suites test against the physical TPM 2.0 interface (`/dev/tpmrm0`) and hardware vaulting mechanisms directly.

## Runner Infrastructure

The execution environment runs as a dedicated systemd user service on the DGX Spark workstation:

- **Runner Binary**: `~/.local/bin/forgejo-runner`
- **Service Name**: `forgejo-runner.service` (`systemctl --user status forgejo-runner`)
- **Configuration**: `~/atlas-forgejo/runner/config.yaml`
- **Working Directory**: `~/atlas-forgejo/runner/workspace`

### Configured Runner Labels

The runner maps standard workflow labels directly to host execution:

| Label | Execution Backend | Platform |
| :--- | :--- | :--- |
| `spark` | Host process (`:host`) | Linux aarch64 (Grace Blackwell) |
| `spark:host` | Host process (`:host`) | Linux aarch64 (Grace Blackwell) |
| `ubuntu-latest` | Host process (`:host`) | Linux aarch64 (Grace Blackwell) |
| `linux-arm64` | Host process (`:host`) | Linux aarch64 (Grace Blackwell) |
| `self-hosted` | Host process (`:host`) | Linux aarch64 (Grace Blackwell) |

## Repository Workflow Conventions

Every repository in the AIEN ecosystem adheres to the following dual-workflow layout:

1. **Forgejo Workflows (`.forgejo/workflows/`)**:
   - Filename: `.forgejo/workflows/ci.yml`
   - Trigger: `push` on `main` and `workflow_dispatch`, never `pull_request` (DRA-45: a self-hosted runner never runs PR code; every runner label here maps to the Spark host, so there is no hosted event-guard job)
   - Runner Target: `runs-on: spark`
   - Purpose: Automatic test execution, code style verification, zero-plaintext-secret auditing, and terminology enforcement.

2. **GitHub Workflows (`.github/workflows/`)**:
   - Filename: `.github/workflows/ci.yml`
   - Trigger: `workflow_dispatch` (manual only)
   - Runner Target: `runs-on: ubuntu-latest`
   - Purpose: Optional manual validation for outside contributors without consuming automated runner minutes.

## Invariant Verification Gates

The Forgejo CI pipeline enforces four non-negotiable invariant gates before any code merges to `main`:

1. **Zero Plaintext Secrets**:
   Scans the workspace for `.env` or credential files. All secret material must resolve dynamically through an approved `SecretProvider` (`MemorySecretProvider` or hardware TPM `Tpm2SecretProvider`).

2. **Sovereign Voice & Anti-Slop**:
   Audits repository documentation and code for em dashes (`—`), en dashes (`–`), and formulaic marketing tropes.

3. **Normative Terminology**:
   Audits documentation to ensure compliance with the normative specifications in `docs/SECURITY_MODEL.md`, `docs/NETWORK_POLICY.md`, and `docs/PLATFORM_MATRIX.md`.

4. **Code Quality and Test Coverage**:
   Executes `cargo fmt --all -- --check`, `cargo check --workspace --all-targets`, and `cargo test --workspace --verbose`.
