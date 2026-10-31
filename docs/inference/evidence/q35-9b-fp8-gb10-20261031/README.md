# q35-9b FP8 serving on GB10, 2026-10-31 quiet window (CoreChip)

Label: EXERCISED on an unqualified development build. Not QUALIFIED.

## What ran

- Server: `aien-serve-http` built from this branch at ab77b970ed5c132a2d62b5bedab277a8a1627ec4,
  linked against omega-mv#905 `libomega_gpu.a` through `AIEN_OMEGA_GPU_LIB` (a development
  override; the exact omega sha and library sha256 are in each `*/session.start`).
- Model: `~/models/qwen3.5-9b-c2022362-fp8b128`, `--max-context 8192 --max-tokens-cap 1024 --thinking off`, port 18295.
- Client: HeadToHead `bench/bench.py` at e430e23f with one change, an
  `Authorization: Bearer` header (the server refuses to start without
  `AIEN_SERVE_HTTP_BEARER_TOKEN`; a fresh random token per run, not recorded).
  16 prompts `prompts/decode.jsonl`, greedy, 128 tokens, concurrency 1.
- Chip route: spark-window `--gb10 none --quiet exclusive --lease 20`, the runner holds
  `flock -x /tmp/aien-gb10.lock` and sets `AIEN_GB10_LOCK_HELD=1` inside it. The server verifies
  an exclusive ancestor holder (`chip_reservation.rs`) and reports `Inherited`. Flint was stopped
  by Main for the quiet window; no other model on the chip.
- Negative control `negctl`: the same checkpoint with 3072 bytes of shard 2
  (`model.safetensors-00002-of-00004.safetensors`, first difference at byte 50345306) altered;
  sha256 of the altered shard 0165e59632a08893ed1976ff502e43e44807b40b016cde872e35a7dc2866f327.

## Results

| Run | window | rc | ready / load s | HTTP | outputs |
|---|---|---|---|---|---|
| clean | 20261031T081315Z-809e71 | 0 | 1 / 17 | 16 x 200 | 16 of 16 texts byte-identical to `~/handoffs/H2H9B/ours-evidence/main/agree.jsonl` (same prompts, 128 tokens) |
| negctl | 20261031T081515Z-f54001 | 0 | 1 / 12 | 16 x 200 | 0 of 16 texts equal to clean; first differing character at 0 or 4 |

Server log (clean): decode about 22.6 tok/s per request at 128 tokens, graph chip 44.5 ms/token.
These are shared-machine figures, not a timing gate.

## Earlier attempts in this window (not committed, kept under /tmp/corechip/e-attempt1, e-attempt2)

1. 08:09Z: the server exited before listening: `AIEN_SERVE_HTTP_BEARER_TOKEN must be set`.
2. 08:12Z: shared window route: the server refused the chip: `AIEN_GB10_LOCK_HELD=1 but nobody
   holds /tmp/aien-gb10.lock` (a shared flock does not count as a reservation). Correct refusal.

## NOT_RUN

- A qualified build (omega-mv#905 merged and pinned, no `AIEN_OMEGA_GPU_LIB` override).
- Timing gates and CI-equivalent value gates on the chip.

`SHA256SUMS` covers every file in this directory except itself.
