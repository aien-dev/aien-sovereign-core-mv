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

Arm `single-wave` (L310 lever 2, off until its proof line `single-projection one-wave rule` is in a candidate): the same
suite on a second server with `AIEN_OMEGA_FP8_SINGLE_WAVE=1`, which makes omega give the k 4096 single projection (o_proj,
40 launches per token) 2 rows per warp: 128 CTAs at 72 registers, 3 per SM, one wave of the 48 SMs, instead of 256 CTAs at
64 registers in two waves; identity must hold, the lever figure is chip ms/step in `server-single-wave.log` vs
`server-plain.log` (measured alone: o_proj 54.66 -> 53.04 us per launch against the 30.73 us byte floor, -64.9 us per
1-row step at 15/15 positions, output bit-identical; the whole step moved less than the harness drift, which is why this
arm decides it; `~/workspace/investigations/2026-10-24-l310-single-proj-waves`, REPORT.md window 2).
