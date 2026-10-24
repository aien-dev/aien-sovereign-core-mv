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
