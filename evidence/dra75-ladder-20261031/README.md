# DRA-75 (#869): per-kernel ladder, HISTORICAL host prep

Status: parser and gate EXERCISED on the host; every number below is HISTORICAL (lever-2 shared-chip runs,
`in_window false`, one last-stroke timeline at N = 8) or MODELED. Chip stamps on current main: NOT_RUN.

## Files
- `historical/`: inputs copied unchanged (`stroke8l7-stamps.{log,json}`, `stroke8l6-stamps.log` from sc-motor
  `evidence/motor-chip-20261021`, `motor-stroke8l2-gate.log`); hashes in `SOURCES.sha256`.
- `per-class/stroke8{l2,l1,v2,b,l7}-stamps.tsv`: per-class launches and ms, produced by `om/timeline/timeline.om`
  and byte-equal to the same table derived from each run's json `timeline_last_stroke`.
- `fit.wl` / `fit-output.txt`: Wolfram evaluation (quoted below).

## Claim checks (Wolfram, `fit.wl`)
- Per conversation per stroke at N = 8: gdn 7.237/8 -> 0.904625 ms, attn 5.301/8 -> 0.662625, gate 0.221/8 -> 0.027625.
  There is no PERM_O kind (CUDApp 4bbccf08 `omega_gpu_q36_graph.c` L947-950, `rec_perm` L657-661): PERM_XQ/O/XO/ACT all
  stamp as `perm`. 1.500*240/370/8 -> 0.1216 is the PERM_O launch share; the measured class is 1.500/8 -> 0.1875.
- Launches: main L(N) = 531 + 82N + Ceiling[N/4] -> {614, 696, 860, 1189}; lever-2 (per-head COMBINE|NORM)
  491 + 122N + Ceiling[N/4] -> {614, 736, 980, 1469}; class sums of the l2 log -> 1469 launches, 77.739 ms.
- GDN state: 2*32*128*128*4*30*8 -> 1,006,632,960 B; /7.237 ms -> 139.095 GB/s.
- Attention: 5.301/80 -> 66.2625 us per launch; KV read at pos 70, 10*2*256*2*2*71 -> 1,454,080 B per conversation,
  /0.662625 ms -> 2.194 GB/s.

## Bytes vs latency (per conversation, N = 8, HISTORICAL)
- Bytes: GDN 0.905 ms moves 125.8 MB (f32 state read + write) at 139 GB/s, about half the 231-237 GB/s the
  expert classes reach. [INFERENCE] byte-bound but not at the pool ceiling; the 139 GB/s is derived from the class
  time itself, so it cannot independently "match" it.
- Latency: attention 0.663 ms moves ~1.45 MB (2.2 GB/s): launch/latency bound on 4-CTA grids. gate 0.028,
  argmax 0.017 and the per-head PERM_O share 0.122 are small launches. Per-head total 1.7337 ms.
- The rest of b: the lumped fit attributes 3.29 ms per head; per-head classes explain 1.73 ms; on lever-2 the
  per-head COMBINE|NORM share of `res` adds up to 2.833*320/360/8 -> about 0.31 ms. The remainder is
  inter-launch gaps and N-growth in "rows" classes (mix, o_proj, router), which N = 8-only data cannot separate.

## Ladder fits (lever-2 medians 25.320/32.465/44.351/85.428 ms at N = 1/2/4/8)
- a + bN: a 14.581, b 8.616, residuals {2.124, 0.652, -4.694, 1.919}, RMS 2.768 ms.
- a + bN + h Ceiling[N/4]: a 3.617, b 6.286, h 15.761, RMS 0.321 ms (3 parameters, 4 points: near interpolation).
- DRA-64 published form: predicted {23.494, 31.865, 48.467, 83.768}, residuals {1.826, 0.600, -4.117, 1.660}.
- c check: (22.840 + 15.858)/55.6 -> 0.696007 ms per expert-layer-union unit.

## MODELED forecast for main (CN rows), to be tested by the chip queue
mainPred (fit.wl) at U = 9/16.3/30.7/55.6 -> {26.156, 32.970, 46.460, 75.536} ms per stroke; duplicate-golden arms
(U = 9 at every N) -> N=2/4/8 {27.890, 31.357, 43.102}; distinct minus duplicate at N = 8 -> 0.696*(55.6 - 9) = 32.434 ms.
Rows classes are held at their N = 8 time, so the forecast over-predicts at small N if they grow with N.
