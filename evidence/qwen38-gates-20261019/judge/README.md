# Judge run of the two new gates (2026-10-19, ledger L068)

`aien-local-ci once aien-sovereign-core c4316dca17b86a7506fd4042fa99be1b15977490 --operator`, run on the
Spark by the judge. `bundle.json` is the judge's signed verification bundle for that head: it lists
QWEN38-SERVE and QWEN38-SPEED among its checks (backend gb10, required, EXACT), so the judge selects both
gates from the manifests under tests/gates. The two receipts are the evidence files the bundle points at.

Result in this run: both gates NOT_RUN with reason `gpu_lock`. The judge's wait for the chip lock
`/tmp/aien-gb10.lock` ran out (about 18 minutes) because other agents' timed measurements held it. NOT_RUN is
the judge's honest answer for "could not get the chip", it is not a pass. No tok/s figure comes from this
bundle; the real measurement is `evidence/qwen38-gates-20261019-run2/` (or the notes in the handoff if memory
never freed up).

The other failing checks in the same bundle (formatting, clippy, tests, ...) belong to the base this branch
sits on (alpha/ceiling-and-model, ahead of main), not to the two gate files.
