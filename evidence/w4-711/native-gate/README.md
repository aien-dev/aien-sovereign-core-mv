# Native W4 automatic qualification

The existing gate entry `tests/gates/run.sh native-w4` compiled and ran all
15 native groups, exit0. Exact stdout is retained in entry.log. It uses the
immutable Omega compiler commit and source/executable digests recorded in
native/w4/compiler.lock and compiler.sha256. This is an explicit proposed
compiler dependency, not merged default-branch capability.

Executed under quiet-build on the AArch64 host, with a private target:

    CARGO_TARGET_DIR=/home/drakestapleton/workspace/targets/L793-w4-gate tests/gates/run.sh native-w4

No GPU, model service, hardware judge, or performance measurement ran.
The repository's existing shell pin/build dispatch and native Om supervisor
are the test execution path. No new Rust, C or Python implementation was added.

Independent reproduction compiled the same gate bytes and passed15groups.
All six controls refused: reused scratch91; wrong child exit, signal, actual
30-second deadline, altered numeric test stdout and compiler failure99.
Raw summarized results and review scope are in controls.txt and review.txt.
The review's copied manifest predates the supported inline-observation syntax
correction; the actual runner below parses and executes the corrected manifest.
A separate private wrapper copy with an incorrect compiler digest exits3 before
building/running any test. Its diagnostic is retained in digest-refusal.log.

The actual existing aien-test runner passed SC-NATIVE-W4 at committed source
1d24d97e7cac46f0d7dbd1b588261b5b173cc745. runner.json records exit0,
w4_groups15, no cache reuse, and a clean tree before and after. The immutable
unmerged compiler dependency was explicitly materialized by the existing pin
mechanism; this is host component evidence, not signed release qualification.
The runner itself is the retained Rust aien-test tool, not new implementation.

The actual planner at the same source and canonical base27b0d670 reports no
unmapped paths. It still returns BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED.
Configuration review, compiler dependency integration and the other selected
repository gates remain required. No source exemption or waiver was added.
