# OWN-30B: measuring in an uncontended GPU window

Issue #625. OWN-30B decode ran at about 36 to 39 tok/s on an idle GB10 and about 2.4 to 2.6 tok/s while a heavyweight model shared the GPU (evidence: `docs/inference/evidence/own30b-omega4799-20261015/NOTES.md`). That gap is hardware contention, not a kernel change, so a speed number means nothing unless the run says whether the GPU was shared.

## The runner

`scripts/bench/own30b-quiet-run.sh` wraps one benchmark or HTTP session:

```
scripts/bench/own30b-quiet-run.sh --owner <you> --out <new dir> \
  --server-pid <aien-serve-http pid> --server-bin <path to aien-serve-http> \
  --model <model dir> --omega-pin <omega commit> --minutes 10 \
  -- <benchmark client command>
```

1. **Preflight.** Records GPU compute processes, GPU utilization, AIEN listeners, repo HEAD, server binary SHA-256, model path and size, Omega pin and `OWN30B_DECODE_GRAPH`.
2. **Refusal.** Any GPU compute process other than `--server-pid`, or an `nvidia-smi` that cannot answer, stops the run with exit 3 before anything starts. `--allow-contended "<why>"` runs it anyway, records the reason and marks the run not comparable.
3. **Quiet hold.** The command runs under `quietlock hold` (1 to 20 minutes, owner and reason recorded). Ctrl-C or SIGTERM stops the command through quietlock and the flag is released. A quiet flag already held by someone else stops the run with exit 75.
4. **Postflight.** The same GPU query again. A foreign process seen then also marks the run not comparable.
5. **Manifest.** `<out>/manifest.json` (schema `own30b-quiet-run/v1`) holds everything above, the exact command, the exit status, `comparable` and every reason it is false.

Only a manifest with `"comparable": true` supports an uncontended speed claim.

## What it does not do

- It never takes `/tmp/aien-gb10.lock`. `aien-serve-http` takes that lock itself for each generation; holding it from outside would deadlock the server.
- File locks coordinate cooperating workloads only. A GPU process that ignores the quiet flag is detected, not excluded, and a process that starts and stops between the two samples is not seen.
- It never kills, pauses or renices anything.

## Asking for a window

1. Check `quietlock check`. If someone holds the flag, wait or ask the holder; do not remove a live flag.
2. Check `nvidia-smi`. If another model is loaded on the GPU, ask its owner to stop it for the window. Do not stop it yourself.
3. Run the runner. If it refuses as contended, the manifest names the process.
4. Self-test (no GPU, fakes only): `bash scripts/bench/own30b-quiet-run-selftest.sh`. With `QUIETLOCK_REAL=$(command -v quietlock)` it repeats the hold and interrupt cases against the real binary in a private state directory.
