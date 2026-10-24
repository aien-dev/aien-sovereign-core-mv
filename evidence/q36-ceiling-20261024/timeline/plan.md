# Plan: one-row per-launch timeline of the live 35B (L310, step 3 of the confirmed order)
Stated before acting, 2026-10-24. Folder is the record; the candidate (live build pins) stays untouched.

DOCS READ: spark-hardware-manual s.1 (one 273 GB/s bus; C2C 345 GB/s per direction is not the ceiling) and s.4
(nvos.h:1114-1118 "For system memory this will not be coherent with direct CPU mappings"; rule 1 host-polled words
uncached, rule 3 bulk chip-only data may stay cached). Recorded with docs-read (repo aien-sovereign-core).
Camera doc: omega_gpu_cmdlist.h:65,104-106 (`opts.stamps`, `omega_gpu_cmdlist_stamps`: PTIMER ns before launch 0 and
after each launch's WAIT_FOR_IDLE). L356 advisor note: stamps are launch + drain, they RANK launches; they do not sum
to the wall step. q36_levers.rs header (sc f0230a59): identity vs goldens (ids + FNV logits hashes), timing band to
Q36_LEVERS_POS (default 1100), per-kind timeline at positions Q36_LEVERS_STAMPS (default 40,300,1000), bytes per kind
at BW 228 GB/s, KEEP_FREE 12 GiB, exit 0 only when every golden is identical.

SYMPTOM (one line): a 1-row decode step on the live build takes 20.7 ms (serve.log:11597) while its bytes need
10.888 ms at 273 GB/s (docs/speed/Q36_CEILING.md); 9.812 ms per token are unexplained (composition UNKNOWN).

PINS (candidate, untouched): sc ship/motor-l387 f0230a59 + omega ship/motor-l387 6de999da (lib sha256 110295ef...),
the pins of the live bin-19173f9e. The q36_levers binary is built from exactly these pins in a detached worktree with
its own target dir BEFORE the window, sha256 recorded in PINS; the window is run-only.

SMALLEST REPRO (must show the symptom first): q36_levers on the live pins, 1 row, one golden, Q36_LEVERS_POS=60,
Q36_LEVERS_STAMPS=40: the timing band must read about 20.7 ms per step (same number as the live server) BEFORE any
timeline is read. If the band is not within the live server's range, stop: the repro is not the live path.

CAMERA (what the run prints, UNKNOWN fields named): per launch, PTIMER stamp deltas (launch + drain) grouped by kind
(gdn, attn, expert, shared, router, head screen, head recheck, glue/copy), sum per kind, bytes per kind, implied GB/s per
kind at the 228 GB/s stream rate, and the ratio (sum of stamps) / (wall step) printed as a number, never assumed 1.
Any launch whose kind the harness cannot name prints `kind=UNKNOWN reason=unnamed`. The camera is the existing
stamps option; nothing is added inside the measured path (no prints, no waits, no flushes).

NEGATIVE CONTROL (the instrument must be able to say no): a run with stamps OFF must produce identical ids and logits
hashes to the stamps-ON run (stamps do not change results), and a run with a KNOWN byte change must move the right kind
by the predicted amount: AIEN_Q36_EXACT_HEAD=0 (plain one-pass BF16 head, +508,559,360 B per token: 1.02 GB head read
instead of the 0.51 GB fp8b screen) must add about 1.86 ms at 273 GB/s / 2.23 ms at 228 GB/s to the head kind and
nothing to the others. If the camera does not see that, it is blind and nothing else from it is read.

BASELINE then VARIANTS (one variable each, N repeats): baseline = live pins, every switch at its live default (exact
head ON, CN rows ON, stroke path 1 row), positions 40,41,42,43,44,300,301,302,303,304,1000,1001,1002,1003,1004
(N = 5 tokens per band), goldens evidence/q36-host-20261020/gen-p1..p4.json; then variant head-plain
(AIEN_Q36_EXACT_HEAD=0) once, same positions. No second variant in this window. The lever is named only from the
baseline timeline, after the control passed.

RULES: candidate untouched; recorders act after the real path finishes; no chip kill; one exclusive timing window
<= 20 min, gb10 exclusive, machine quiet (quietlock check, <= 3 quiet-lock jobs, load <= 10); MemAvailable >= 46 GB
beside the live 8-slot server before the load; the window is requested on BOARD and opened by the coordinator, never
by this lane while CHIP-RESERVED lanes hold the chip; seal the folder after the run; report in the required shape.

## Window request (posted on BOARD)
ONE exclusive timing window <= 20 min: load (own 35B copy), baseline run, control run, stop. Run-only; binary and
golden paths sealed beforehand. Receipt files land in runs/.

## Deviations
(none yet)
## Result (CLAIMS tagged, LIMITS, WHY EASY TO GET WRONG, STATE AFTER, NEXT QUESTION)
(not run)

## Deviations / notes before the run
- 2026-10-24 19:1xZ: binary built and pinned (PINS). Harness facts from q36_levers.rs (f0230a59): identity pass runs with stamps OFF; the timing pass turns stamps on only at each listed position (set_stamps(true) -> timeline -> set_stamps(false)); the harness refuses when MemAvailable < model + 12 GiB. MemAvailable at writing 60.4 GiB beside the live server (RSS 45.5 GB): fits, with little margin; a second test server on the box would block the load.
- run.sh guards: binary sha, quiet flag, control skipped if the baseline used more than 11 min of the 20 min window. Load time of a 35B copy: UNKNOWN (not measured by this lane).
