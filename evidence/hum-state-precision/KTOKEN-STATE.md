# L313 part 2: one DeltaNet state touch for k tokens (Qwen3.6-35B-A3B)

## Bytes (aien-calc receipts)

One head (one sequence) carries 30 DeltaNet layers x 32 value heads x 128 x 128 state values.

| item | bytes | receipt |
|---|---|---|
| state, f32 | 62,914,560 | calc:18861a5dc8e65a6d8c97ec3aa3e507799b9f9c817bde1dfe7250cbead7bc95f2 |
| state, bf16 | 31,457,280 | half of the f32 figure |
| state, fp8 e4m3 + one f32 scale per (value head, k row) | 16,220,160 | calc:9eacec999b509ee6179e3d5a38e0cc598c44e5b4357aa5ae7d3c46e8db90829a |
| one token's replay record (q, k, v 128 each + g, beta, f32, all heads and layers) | 1,482,240 | calc:54dbca8e8b8fbf1deebfcd91a303a5b42ef61bda2abb6f8a0eddbdbcab186656 |

Today (decode, one token per stroke) the state is read and written every token: 2 x 62,914,560 =
125,829,120 B per token (f32).

A k-token verify stroke under the scheme below moves the state once (read + write) plus a replay
record per candidate, written once and read back once: `(2 S + 2 k R) / k` bytes per token, all k
accepted.

| k | f32 state, B per token | vs today | receipt |
|---|---|---|---|
| 1 | 125,829,120 | 1 | (today) |
| 2 | 65,879,040 | | calc:13271840e7ca5889a0a37e5708b636ffeae462fbaeef23102d6dadd911fbbc81 |
| 4 | 34,421,760 | | calc:bb53be2829ea9e717acf86fccb24b5c92ed57c29770691f7c3b3133290f29c7e |
| 8 | 18,693,120 | 6.731x less | calc:995dde337d55e2a24a7c5b87a1beb1cd42ec77516e5cdaed5a3da765d5934060, calc:d25e446a02a846151d0ac516ef0ac998bddada2db78fc87f7584f0b0800c2fb2 |
| 16 | 10,828,800 | 11.62x less | calc:2c8539591aad02043b931b7251719478ad3ac7105617c68041e0fa126aa63fbe, calc:e970bd27e1b892e3904a467f3c8d5c0aee4aa86d88c16479b10b6165ddf90c86 |

bf16 state at k = 8: 10,828,800 B per token (calc:bf7c736a2517d9a7ec48eecde8446629ea4600a99d74e868e2139840a67c277b).
With fewer accepted than verified (m of k), divide the stroke's bytes by m instead of k. The
records are an upper bound: replay does not need q (the outputs of replayed tokens are already
known), so 2/3 of R would do.

## What the chip already has (omega origin/q36/serve, read 2026-10-21)

`src/omega_blackwell_codegen_gdn.h` prefill (omega#543): `prep` (conv, norms, per-token records),
`scan` (one CTA per value head; each thread holds its slice of S in registers "for the whole
prompt", walking the tokens in order, the state written once at the end), `post` (gated norm).
So the scan already reads and writes the state once per call of T tokens, and it computes the
token-by-token recurrence, not the chunked (WY) form. Today it needs T >= 3.

## The chip change (spec, not built; chip receipt only in the HUM run order after the swap)

The catch with verify is partial acceptance: the stroke learns how many of its k candidates are
accepted (m) only after the head, but the state to keep is the state after m tokens, not after k.
Writing all k intermediate states would cost k writes again. The scheme:

1. Keep the committed state `S_c` in memory (state after the last accepted token) and, beside it,
   the replay records of the previous stroke's accepted tokens (from `prep`, m rows) plus the
   conv tail after them.
2. Next stroke, per layer: the scan loads `S_c` once into registers, first replays the m
   accepted records (state update only, no output, no gate norm), then runs the k new candidates
   (update + output). It writes back one state: the state after the replayed m tokens (the new
   `S_c`), stored as soon as the replay rows are done and before the candidates touch the
   registers (no second register copy), and the k new records.
3. After the head decides m', the next stroke replays those m' records. The conv tail likewise
   advances by the accepted rows only (the last three accepted qkv rows).

Exactness: the replay runs the same recurrence on the same inputs in the same order as a
token-by-token pass, so the stored state is the one the scan would have produced had it run only
the accepted tokens. Equal to today's decode only as far as the scan and the decode kernel agree
bit for bit (they share the update formula; L306's prefill-vs-decode gate is the receipt for
that).

Chip work: allow T = 1 and 2 in the scan (k = 2 verify, and replay of 0..k rows), a "replay
count" argument that skips output for the first m rows, the state store moved from the end of the
scan to the end of the replay rows,
and records kept across strokes (k x 1.48 MB per head at f32).

## Chunked (WY) form: measured on the host, not needed for the chip at k <= 16

`delta_chunk` in sc `qwen38.rs` computes the same block update in the chunked (WY) form, the form
that suits tensor-core matmuls at large k. In exact arithmetic it equals the recurrence; the f32
rounding differs. `chunk-*.json` here measure that difference on the goldens (chunk 2/4/8/16 vs
token by token at the same matmul shapes). The scheme above keeps the recurrence, so it does not
pay that rounding.
