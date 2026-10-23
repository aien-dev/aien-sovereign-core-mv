# L653: existing PR577 bot-text qualification

Existing issue: aien-sovereign-core-mv#477. Continue **existing PR577**, not a competing PR.

Original PR head: `3ccf32ed42c645d33d4c44fe08eca5d21a1858b9`.
PR base: `754c07b118f41ad899607b60329bd851b1ccb13e`.
Qualified functional successor: `9c11d42971fb24b63867e5ff4a53d902d66ec1f5`.

## What actually ran

- On the original PR head, the complete `cargo test --locked -p spark-inquisitor` suite passed89 tests, no failed or ignored tests. All-target component Clippy with warnings denied and the complete workspace formatting check passed.
- Built and ran the old doctor from the byte-exact base `main.rs`: native exit0 but it emitted `Contributor Oath evaluator`. The actual `qualify.py --check-output old-doctor.stdout` checker exited1. This is a behavioral text regression, not a compile-error RED.
- Restored all original source bytes before testing the PR candidate.
- Ran ten real local CLI cases: help, doctor, interview, clean/telemetry review, clean/telemetry audit, issue triage, accepted legacy testimony and refused empty testimony. All original exit/refusal checks and retired-term checks passed, but the combined output lacked the named governing doctrine and values document. `cli-run.log` retains that failed positive-coverage check; `cli/` is the original incomplete qualification, not a whole PASS.
- Added explicit assertions to the existing emitted-interview test; it failed at the absent `Stapleton Doctrine`, exit101 (`governance-red.log`).
- Corrected only the emitted interview's text: the Stapleton Doctrine governs project standards, COVENANT.md states values, LICENSE supplies AGPL-3.0-or-later terms. All checks, evaluation logic, legacy certification compatibility, labels, exit rules and original tests remain unchanged.
- On the functional successor: complete component suite **89 passed, 0 failed, 0 ignored**, workspace formatting check0, all-target component Clippy0 and native component build0. Ten actual CLI cases then passed with the named standards and no emitted SparkOS, oath or CONSTITUTION.md references. `post-fix-cli/manifest.json` is complete and binds the native executable hash and source SHA.

The fixtures and testimony are explicitly synthetic; the executable, file I/O, stdout/stderr and exit statuses are real. No bot comment was posted by the test, no model or production data was used, and no production service changed. The test-only Python runner is not a production dependency.

## Reproduce

At the exact functional successor, with the repository's Rust1.98.1 toolchain:

```sh
cargo test --locked -p spark-inquisitor
cargo fmt --all -- --check
cargo clippy --locked -p spark-inquisitor --all-targets -- -D warnings
cargo build --locked -p spark-inquisitor
python3.12 evidence/INQUISITOR-477-L653/qualify.py \
  --source 9c11d42971fb24b63867e5ff4a53d902d66ec1f5 \
  --output /your/owned/nonexistent/output-directory
```

The script lives on the evidence successor branch; its selected source SHA must exactly match the component sources, manifests, lock and toolchain file. Output directories must not already exist. Original failed receipts are never overwritten.

## Publication boundary

The original PR's hosted AIEN Verification failure was inspected directly: run39780382420 exited2 because the exact-head signed Spark bundle was404, explicitly **NOT_RUN**. It was not a reported inquisitor test failure. This qualification does not create a signed bundle, replace the operator preflight, satisfy all selected whole-repository gates or authorize merging. The base still requires CRUMB-VERIFY, so required pinned metadata must be refreshed and verified before publication. Independent source review and exact-head authoritative Renda integration remain separate steps. Do not attribute this successor's extra text correction to the old PR head.
