# OWN-30B 128k context on the GB10 (2026-10-18)

Binary: `aien-serve-http` from this branch (sc d046142c = main 12e71aad + omega.lock 7c2ae49e), sha256
d0237909b09c826f2e88a6dd6db66b954b81383be83c7079e9c8d28a56e3236a, run with `OWN30B_DECODE_GRAPH=1 --max-context 131072`.
Model: Qwen3-30B-A3B-Instruct-2507-FP8 (max_position_embeddings 262144, no rope scaling).

## Red (before)
Pinned main binary afe59ae1 started with `--max-context 131072`: `kv fused cache 6144 MB` (65536 positions; round_max_pos
clamped to omega's OMEGA_GPU_DATTN_MAX_POS 65536) and no message about it: `server-old-binary-clamped.log`.

## Green (after)
- `server-128k.log`: `kv fused cache 12288 MB`, ready line `context 131072 tokens`.
- Recall at 118,186 prompt tokens (`needle.txt`, request built with `jq -n --rawfile t needle.txt`, temperature 0,
  max_tokens 24): reply `BLUE-HERON-4471`, the code word on the first line (`needle.out`, usage prompt_tokens 118186).
  Whole request 188.4 s wall (curl), dominated by the prompt read.
- A 209,536-token prompt is refused with `context_length_exceeded` (no crash).
- Omega chip proof of the attention kernel at contexts 65536 / 100000 / 131072: omega `evidence/dattn-maxpos-128k-20261018/`.

## Speed, same minute, same machine state (`swap.sh`, `speed.sh`; decode = (576 - 64 tokens) / wall difference, run 2)
| binary | --max-context | 64 tok wall s | 576 tok wall s | decode tok/s |
|---|---|---|---|---|
| main afe59ae1 (pin) | 32768 | 1.659 | 15.250 | 37.67 |
| this branch | 32768 | 1.695 | 15.537 | 37.0 |
| this branch | 131072 | 1.691 | 15.481 | 37.13 |
The 576-token reply is byte-identical to the main binary's. A Spark judge `cargo test --workspace` ran on the CPU during
all three rows; an earlier quieter measurement of the pin with the decode graph gave 40.75 tok/s.
