# Alpha launch readiness fixture receipt

- Base: `origin/alpha/ceiling-and-model` at `56d42869b90a1b297a7faafa27b4502c1a104425`.
- Candidate branch: `fix/alpha-launch-readiness-20261022`.
- Fixture port: `18164` only. Fixture state, model metadata, PID, log, and process are created under a temporary directory and removed by the test.
- Tests: `scripts/test-serve-27b.sh` and `scripts/test-install.sh`; raw stdout/stderr and exit statuses are in the neighboring `*.log` and `*.rc` files.
- Fixture suite: 6 passing cases plus the explicit RED reproduction against the base helper. The six cases cover early exit status 23 and log, exit 0 without API readiness, timeout and process cleanup, advertised model identity and TERM forwarding, unrelated PID non-signaling, and foreground attachment.
- Installer suite: 9 existing installer cases pass and the fixture suite passes when invoked by the harness.
- Shell syntax checks and `git diff --check` passed.
- These are fixture results. No real model start, model load, GPU operation, live server action, or request to port `18091` was performed.
- The existing 40 GiB warning in `tools/serve-27b.sh` was left unchanged. The current branch source has a 45 GiB load policy and the checked-in memory receipt establishes peak RSS only; this checkout does not contain current branch evidence for the precise 48.3 GiB floor recorded by the older L035 finding. L035 remains for its parent to reconcile.
