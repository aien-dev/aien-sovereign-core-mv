# evidence/merge-train

The merge train (ledger L384): one candidate build, one 35B load, one shared suite, one verdict per lever.
The standard is `docs/MERGE_TRAIN.md`. The suite is `suite.json`. The command is `merge-train`.
Each cycle writes `runs/<UTC stamp>-<candidate sha8>/` with `verdict.md` (the lever x case table), `cells.jsonl`,
`receipt.json` (loads and wall time) and one folder per arm and case holding the raw replies.
