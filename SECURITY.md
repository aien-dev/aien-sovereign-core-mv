# Security policy

## Reporting a vulnerability

Report privately. Do not open a public issue, pull request or discussion about it.

1. GitHub: open this repository's Security tab and choose "Report a vulnerability". This
   creates a private advisory that only the maintainers can see. If that option is not shown,
   private vulnerability reporting has not been enabled for this repository yet; use email.
2. Email: drake@aienos.com with the subject `[aien-sovereign-core-mv] security report`.

Include the commit or release you tested, the component (path), steps to reproduce, and what an
attacker gains. If the report involves a secret, say where it was exposed and do not send the
secret itself. Encrypted mail is not required; do not post proof-of-concept payloads publicly.

## Response targets

This project has one maintainer; the targets below are best effort and stated honestly.

- Acknowledgement: within 3 working days.
- Triage and severity: within 7 days of acknowledgement.
- Fix or mitigation for confirmed issues: aim for 30 days for high and critical, 90 days
  otherwise, with a status update at least every 14 days until closed.
- Disclosure: coordinated. An advisory is published with credit (unless you ask otherwise)
  once the fix is on the default branch. Default embargo 90 days, negotiable.

## Scope

In scope:

- The verification pipeline in `tools/aien-test`, `tools/aien-local-ci` and `tools/aien-verify`:
  a way to get a PASS verdict, a signed bundle or a merge for code that did not pass its gates,
  or to run candidate code outside the sandbox.
- The OpenAI-compatible HTTP server (`crates/aien-serve-http`): bypassing the bearer token
  required by `AIEN_SERVE_HTTP_BEARER_TOKEN`, request handling, memory exhaustion.
- The agent CLI approval floors (`crates/aien-cli`): a tool request that escapes the four floors
  or the rollback ledger in any approval mode.
- Installer and release signing (`install.sh`, `install.ps1`, `scripts/release-*`,
  `docs/release/`, `.github/workflows/release*.yml`): installing an unsigned or wrongly signed
  archive as `release`.
- Capability, MCP, switchboard and security crates (`crates/aien-capability`, `crates/aien-mcp`,
  `crates/aien-switchboard`, `crates/aien-security`).
- `.github/workflows/` and `.forgejo/workflows/`.

Out of scope: the NVIDIA driver; the Omega GPU engine itself (report to its upstream repository,
pinned in `omega.lock`); model weights and model output; services that run a binary from another
release line.

## Supported versions

Only the default branch `main`. No public release of this repository exists; v0.1.0 and v0.1.1
are internal test releases (README). Fixes land on the default branch only.

## Status

Research-grade; one GPU inference path is in alpha testing on the DGX Spark, and its three-run
physical qualification is NOT_RUN. There is no certification and no bug bounty. See
AUDIT_SCOPE.md for what to review and how to build and test it.
