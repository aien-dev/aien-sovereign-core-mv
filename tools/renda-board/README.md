# Renda local board

Run `renda-board` on Spark for a read-only Markdown view; `renda-board --json` retains full item IDs and provenance. Default input is `~/.local/state/aien-local-ci/renda-queue/*.json` using L522 schema `renda-local-queue/1`. `--state PATH` inspects a captured state without changing it. No third-party dependencies, network calls, queue writes, service restarts or authorization changes.

This is a projection, NOT another scheduler. Actionable means local assessment is next, not merge-ready. Existing schema does not record per-item execution leases, owners or dependency completion, so these fields remain unrecorded; the board will not invent Running or Done. A valid plan with approved review and qualified receipt is Awaiting publication, never merge-authorized. Missing/stale/corrupt inputs are explicit. Historical READY prose is not implicitly imported.

Verification: `python3 tools/renda-board/test_renda_board.py`:10 tests pass on Mac and Spark. Coverage: cold READY, qualification refusal, stale and mismatched source, corrupt queue, partial SHA, read-only behavior, duplicate warnings, table escaping. Parsed actual L522 capture-v2 queue:63 unique rows,0 warnings. Installed executable exercised against live default path: queue absent, reports warning and0 items. This is honest deployment state, not evidence that the new runtime queue is deployed.

Parent independently ran L522 candidate test_local_queue.py:20 tests pass at inspection; candidate is still being edited by its sole worker, so this does not certify a final revision. Runtime upgrade and real cold-start enqueue acknowledgement remain L522's acceptance gate. Separate review still required before promoting this board source through the desk.
