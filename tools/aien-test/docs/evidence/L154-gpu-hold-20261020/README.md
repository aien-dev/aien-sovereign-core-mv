# L154: how long the judge holds the chip lock, and what `prepare:` moves out of it

Receipts, 2026-10-20, Spark judge evidence store `~/.local/state/aien-local-ci/evidence/`.

## Before (measured)

`held-before.tsv`: every gb10 receipt of 2026-10-19/20 whose GPU lock was acquired:
repo/commit, gate, start, held seconds (`duration_ms - gpu_wait.waited_ms`, an upper bound:
it still counts the pool slot and target-dir lock), verdict. Per omega PR run:

| omega head | chip-lock held | COMPOSITION + M17 + M19 | other 8 chip gates |
|---|---|---|---|
| 2676fd67 | 1,733 s | 1,553 s | 180 s |
| 3bd56731 | 4,041 s | 3,629 s | 412 s |
| 39493026 | 5,060 s | 4,911 s | 149 s |
| 47699e9d | 5,312 s | 5,012 s | 300 s |

The three big gates swing from 35 s (M19, quiet box, 23:58Z) to 2,233 s (M17, load near 30):
that time is the chip program itself, not its build.

## Build share (measured on a scratch clone of omega fix/gb10-gate-prepare)

`prepare-all.sh` runs each gate's `prepare:` step, then dry-runs (`MAKEFLAGS=n`) its run step
and counts compiler commands still left for the locked part.

- `prepare-run1-cold.tsv`: cold build, nice 19 on the A725 cores: 13 to 61 s per gate, 339.6 s
  for all 11. Before the stamp fix 9 gates still relinked under the lock (a phony prerequisite,
  aienos-authority-capability, makes the program always out of date).
- `prepare-run2-stamped.tsv`: with `chip_make` (make -o for what prepare built at this commit):
  0 compiler commands left under the lock in all 11 gates.

## After

Not measured yet: the first judge runs on the new binary record it directly
(`volatile.gpu_held_ms` and `volatile.prepare.duration_ms` in every gb10 receipt). Expected:
the build part (13 to 61 s per gate at nice 19; less on the judge's cores) leaves the lock;
the big three gates' hold is dominated by their programs and shrinks only by their build.
