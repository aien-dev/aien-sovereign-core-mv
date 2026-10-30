# Automatic native state-reuse qualification

The existing dispatcher tests/gates/run.sh native-state-reuse passes all7
native suites using the explicit, digest-verified unmerged Omega PR676
compiler pin. The original legacy omega.lock is unchanged. See author.json
and entry.log for the dispatcher/control runs. The independent report records
fresh reproduction and rejection of corrupted output and invalid source.

The gate checks exact five count/failure outputs and two empty-output cases.
A reused scratch directory returns91. Wrong exits, signals, the actual
30-second deadline, mismatched output and failed input/compilation refuse99.
This is host-component qualification, not complete-state restore, GPU execution,
model speed, signed Spark qualification or evidence of merged dependencies.

The actual existing aien-test runner passed SC-NATIVE-STATE-REUSE at
5e03586a8c4953e702c0b18be4a6c98f16d0b848: exit0, observation reuse_groups7,
no cache reuse, clean tree before and after. runner.json is the exact receipt.
The retained runner is Rust infrastructure; no new C/Rust/Python test or runtime
implementation was added. The unmerged compiler is explicitly pinned offline.

The actual planner reports no unmapped paths and returns
BLOCKED_OPERATOR PIPELINE_CONFIG_CHANGED. Configuration review, prerequisite
integration and all other selected checks remain required. No waiver or
allow-list exemption was added. Raw plan retained in plan.txt.
