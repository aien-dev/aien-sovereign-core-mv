# f4: one pinned resident layout for every Qwen3.8-27B tool (sc #676, FORTRAN idea 4)

Branch fortran/4-pinned-resident-layout (base qwen38/prefill-ttft 5a82ea60). Spark, 2026-10-19 02:25Z to 03:05Z,
live 27B server up the whole time (about 38.5 GB, server-state.txt), every chip run under `flock /tmp/aien-gb10.lock`, MemAvailable
checked before each (12 GiB kept free). Omega archive `fresh-pg-6e24f7fd/libomega_gpu.a` (sha in binaries.sha256).

## What is pinned

`qwen38_layout build` staged the checkpoint `~/models/qwen3.8-27b-fp8-017b9c7a` into
`~/models/qwen3.8-27b-fp8-017b9c7a.layouts/a6edf911...27a0.q38layout`: 963 images, 29,520,480,256 bytes, built in
117.7 s (build.log). Layout id `a6edf91182a0bd38827b94121f5e11cc8318a5f0aeb420e76e98217ff84d27a0` = sha256 of
layout.manifest.json, pinned in code as `layout::QWEN38_27B_FP8_017B9C7A`.

Identity across doors and tools:
- The checkpoint door (reads safetensors, applies the transforms) recorded checkpoint-door-recorded.manifest.json;
  it is byte-equal to the pack's manifest (`cmp`), so both doors uploaded the same 963 images.
- qwen38_omega_full and qwen38_prefill print the same id a6edf911... from either door (before-t*.log, after-t*.log).
- Full pack verify (every image hashed against the manifest): 2.22 s, 13.29 GB/s (calc:7d86f5a7) (verify-hwsha.log);
  9.56 s with the software SHA-256 first (verify.log).

## Load time, whole model (64 layers), qwen38_layout load (loadbench.log, r*.json)

| run | door | page cache before | load s | read | hash | upload |
|---|---|---|---|---|---|---|
| r1-ckpt-warm | checkpoint | ckpt 29.47 of 30.87 GB | 44.643 | 26.088 | 1.642 | 14.607 |
| r2-ckpt-cold | checkpoint | ckpt 0.06 GB (evicted) | 35.781 | 18.971 | 1.355 | 13.238 |
| r1-pack-warm | pack, serial hash | pack 3.45 GB | 19.061 | 0 | 6.206 | 12.763 |
| r2-pack-warm | pack, serial hash | pack 22.9 GB | 21.111 | 0 | 6.775 | 14.242 |
| r2-pack-cold | pack, serial hash | pack 0 (evicted) | 26.421 | 0 | 14.674 | 11.679 |
| r3-pack-cold | pack, read-ahead + hash beside upload | pack 0 (evicted) | 14.472 | 0 | 6.154 (overlapped) | 11.554 |
| r3-pack-warm | pack, read-ahead + hash beside upload | pack 29.5 of 29.5 GB | 12.001 | 0 | 1.298 (overlapped) | 11.426 |
| r4-pack-warm | pack, final build (checked dispatch) | pack 9.73 of 29.5 GB | 15.072 | 0 | 5.527 (overlapped) | 12.478 |

Checkpoint warm 44.643 s to pack warm 12.001 s: 3.72x (calc:9ee22852); cold 35.781 to 14.472 s: 2.47x (calc:742db239).
The two checkpoint rows ran under different box load (the cold one was faster); both are reported as measured.
Binaries: r1/r2 rows ran builds from before the read-ahead change (commit a1f28ccf / c59d800f work in progress, not
recorded); r3 rows, both pairs and the controls ran the build in binaries.sha256; r4 and check16-pack-final ran the
final build in binaries-final.sha256 (dispatch checks each held image by name; still PASS / every image matched).

Against the ceiling: reading 29.52 GB from the NVMe at 10.4 GB/s takes 2.84 s (calc:1fd7120f), so the cold pack load
uses 0.196 of the disk ceiling (calc:a7d32641). What is left is the copy into the driver's buffers inside omega's
upload calls: 26.98 GB of device images in 11.426 s = 2.36 GB/s (calc:dc8a58f1); a copy at the 273 GB/s memory ceiling
(read + write) would take 0.198 s (calc:bcf3cec7). That copy is omega code (memcpy into the driver mapping), not
changed here: the next lever.

## Two tools back to back (the issue's test; command-pairs.txt)

Tool 1 qwen38_omega_full golden (one 64-layer window), then tool 2 qwen38_prefill golden (chunk 8).

| | tool 1 load s | tool 2 load s | heap after load (tool 1) | pack pages mapped (tool 1) | driver (each) |
|---|---|---|---|---|---|
| before (checkpoint) | 29.29 | 42.50 | 2,708,729,856 B | 0 | 27,170,836,480 B |
| after (pack) | 14.41 | 14.39 | 166,449,152 B | 2,540,097,536 B (embedding, shared page cache) | 27,170,836,480 B |

Both loads: 71.79 s before, 28.80 s after: 2.49x (calc:afb65392). Heap per tool 2.54 GB smaller (calc:9e3db391): the
embedding is read from the shared pack mapping instead of a private copy. Chip memory per tool is unchanged: each tool
still holds its own 27.17 GB copy (true one-copy sharing is not built; see the handoff).

Outputs are bit-identical between doors (outputs-compared.txt holds the filters and digests): tool 1's tokens,
cosine, p95 and every per-step number hash to the same value (ad9919fa...) before and after; tool 2's tokens, cosine
0.9928993630379012 and max_abs equal (99f2ce4b...), and equal to the earlier receipt evidence/qwen38-prefill-20261018T1836Z/golden-chunk8.json. Tool 1 PASS 24/24. Tool 2 says FAIL
both times for the same, already known reason (its cosine bar 0.999 vs 0.9929 with all 24 tokens equal; not moved).

## Negative controls (all went red)

- Corrupted image: `AIEN_Q38_LAYOUT_FLIP=layers.17.down.w` (one byte changed in memory, file untouched): refused,
  `LAYOUT_REFUSED ... entry layers.17.down.w hashes to 5cea661b..., its manifest says e7e92a90...`, rc 1 (neg-flip.log).
- Wrong transform: a 16-layer pack built with layer 0's input norm NOT folded to 1 + w (neg-mutant-build.log), id
  c9a6582f...: internally consistent, so only the pinned id can catch it. qwen38_omega_check 16 layers with it:
  refused by the pinned id, rc 2 (check16-mutant-pinned.log). Forced past the id (`AIEN_Q38_LAYOUT_EXPECT=c9a6582f...`):
  the parity check FAILS, cosine(min) 0.958943, greedy differs (check16-mutant-expected.log). Same tool with the real
  pack: PASS, cosine(min) 1.000000 (check16-pack-good.log). The golden checks have teeth; integrity is not correctness.
- Unit tests 7/7 green (green.log); with `AIEN_Q38_LAYOUT_MUTANT_SKIP_VERIFY=1` the tamper test fails (red-skip-verify.log).
