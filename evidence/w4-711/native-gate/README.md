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
