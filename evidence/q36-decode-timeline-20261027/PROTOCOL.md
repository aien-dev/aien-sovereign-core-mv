# q36 decode timeline: zero-cost A/B protocol (frozen before any number, #785)

Question: with `AIEN_Q36_DECODE_TIMELINE` unset (the default), does the instrumented door cost anything against the door without the instrumentation?

## Arms

- A (base): `aien-sovereign-core-mv` main `f30eb963`, which is the PR's base.
- B (head): branch `perf/q36-decode-timeline` at the commit that carries this file, with the switch unset.
- Both arms: omega `a647b6a9` through the same `pins/env.sh`, the same `rust-toolchain.toml`, `--release`, built with `nice -n 19 taskset -c 10-14 cargo ... -j5`.

## Host A/B (runs now, host only)

- Driver: `bench.patch.rs` in this folder. It is THROWAWAY and appended unchanged to `crates/aien-serve-http/src/qwen36_batch/tests.rs` in a scratch worktree of each arm, then built as each arm's test binary. It sends one request of 20,000 greedy tokens through `Qwen36BatchDoor::with_stepper` over a fake model that does no work, so it times only the door's beat loop, which is the code the switched-off path touches.
- Correctness gate: both arms must print the same token count (20,000) and the same token digest in every sample. Otherwise the run is void.
- Design: 2 discarded warm-up rounds, then 20 rounds. Within each round the two arms run in a seeded random order (Python `random.Random(785)`, recorded in the output), one process per arm per round, both on `taskset -c 12` under `nice -n 19`. Load1 is recorded before each round.
- Metric: ns per token (wall over the whole request / 20,000), one sample per arm per round. Report the median plus min..max for each arm, and the median of the per-round ratios B/A.
- Verdict: PASS if the median paired ratio B/A <= 1.02. Otherwise REGRESSION. Medians and ratios are computed in Wolfram.

## Chip A/B (prepared in window.sh, NOT_RUN until an exclusive window)

- Server per block: `--engine omega --model-dir /home/drakestapleton/models/qwen3.6-35b-a3b-fp8-95a723d0 --port 18190 --bind 127.0.0.1 --max-context 49152 --max-tokens-cap 16384 --thinking off --batch-slots 8`, env `AIEN_Q36_STROKE=1`, `AIEN_SERVE_GPU_LOCK=/tmp/aien-serve-18190.lock`, timeline switch unset. These are the live 18091 flags.
- Design: 4 blocks, with arm order drawn by `random.Random(785)` among the 6 orders that hold two A and two B. Each block is one server start, 1 discarded warm-up request, then 10 requests: the fixed prompt in window.sh, `temperature 0`, `max_tokens 200`, one at a time.
- Metric: tokens/s = `usage.completion_tokens` / request wall. Before each sample, `curl 127.0.0.1:18091/health` is recorded; the window has 18091 stopped, so any sample where 18091 answers with `in_flight > 0` is discarded and counted. The `spell over ... chip N ms/step` lines are kept as a second view.
- Correctness gate: every request in both arms returns the same text as the first A request. Otherwise void.
- Verdict: PASS if median tokens/s of B >= 0.99 x median of A (pooled over the 20 kept samples per arm). Otherwise REGRESSION. Medians are computed in Wolfram.

## No re-run selection

Each A/B runs once. A re-run happens only for a documented machine-load fault, and both runs are then reported.

## The timeline run is not a speed receipt

With the switch on, the chip writes one stamp per launch, which costs chip time. The 200-token timeline run says where the time goes, not how fast the step is.
