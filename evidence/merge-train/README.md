# evidence/merge-train

The merge train (ledger L384): one candidate build, one 35B load, one shared suite, one verdict per lever.
The standard is `docs/MERGE_TRAIN.md`. The suite is `suite.json`. The command is `merge-train`.
Each cycle writes `runs/<UTC stamp>-<candidate sha8>/` with `verdict.md` (the lever x case table), `cells.jsonl`,
`receipt.json` (loads and wall time) and one folder per arm and case holding the raw replies.

Arm `interior-uncached` (L310, off until its proof line is in a candidate): the same suite on a second server with
`AIEN_OMEGA_Q36_BUF_CACHED=0`; identity must hold, the lever figure is chip ms/step in `server-interior-uncached.log` vs `server-plain.log`.

Arm `grouped-wave` (L310, off until its proof line `CTAs per group (wave rule)` is in a candidate): the same suite on a second
server with `AIEN_OMEGA_FP8_GROUPED_WAVE=1`, which makes omega size the grouped expert launch to one wave of the 48 SMs
(min(6, 48 / groups) = 5 CTAs per group on the 35B's 9 groups instead of the compiled 6 that was measured on 8 groups);
identity must hold, the lever figure is chip ms/step in `server-grouped-wave.log` vs `server-plain.log` (measured alone:
step x0.9614, `~/workspace/investigations/2026-10-24-l310-grouped-ctas`).
