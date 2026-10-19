# READ DRUM evidence (aien-sovereign-core-mv#677, FORTRAN idea 5)

Dense Qwen3.8-27B FP8 (26.9 GB of weights) run under a pretend 16 GiB budget: the first layer groups stay
resident, the rest stream from the NVMe through a drum file while the current group computes. Branch
`fortran/5-dense-nvme-subdivision` on `origin/qwen38/prefill-ttft` 5a82ea60. Needs the omega archive from
`aien-spark/omega` branch `fortran/5-fp8-refill` (06d5c876 refill door + eight-byte NaN scan, cdbd1fb3 handle
slots 4096) on `qwen38/prefill-graph` 6e24f7fd.

## Design

- `qwen38_omega/drum.rs`: the drum is one file of chip-ready per-layer images, written once from the checkpoint
  through the same fuse code as `Qwen38Omega::load_at` (q|k|v, qkv|z, ab rows, gate|up). Every entry is 4 KiB
  aligned. A JSON manifest carries per-layer sha256 and per-entry offsets. Entry names match f4's pack entries
  (`layers.N.mix.w/.s`, `ab`, `o`, `gate_up`, `down`), so the index can later be built from f4's pack manifest
  instead of this file.
- `DrumModel`: plan(budget) pins whole groups while fixed + pinned + slots x slot fits; the rest stream. Each
  group has its own Q38Graph and state per conversation; non-last groups get a 2048-row stand-in head. The
  residual crosses groups through the host as f32 (exact).
- Upload path (first build): O_DIRECT reader into host buffers, then `Fp8Tensor::upload`. Refill path (final):
  two device slot sets allocated once; the reader O_DIRECTs straight into the tensor's device mapping (zero
  copy), then `commit` (NaN scan + scales + sync). `ab` stays resident.
- Batch: B conversations step in lockstep and share each streamed group.

## Proof of correctness (every PASS run)

Teacher-forced on the golden record `~/handoffs/qwen38-ref/gen-chat-max-24.json` (24 generated tokens):
tokens equal golden, logits bit-identical to the production whole-model run (`ref.json`, `ref.logits.sha256`),
VmHWM at or under the budget, and the reference itself matches golden.

## Negative controls (must go red, did)

| control | what it does | result |
|---|---|---|
| flip | flips one byte in a streamed weight after the read | FAIL: tokens still equal golden, logits 0/24 bit-identical (max diff 0.00375). Only the bit-identity check catches it. |
| stale | skips the reload of one slot after pass 0 | FAIL: tokens differ, logits 0/24 (max diff 28.14) |

## Results (16 GiB budget, groups of 4 layers, 4 reader threads)

| run | resident | streamed per pass | median pass | per conversation | total | receipt | VmHWM | verdict |
|---|---|---|---|---|---|---|---|---|
| reference: whole model resident (`ref.json`) | all 64 layers | 0 | 155.39 ms | 6.435 tok/s | 6.435 tok/s | calc:12ea11dd | 27.75 GiB | 24/24 golden |
| upload path, B=1 (`run-b1`) | 4 groups | 17.03 GiB | 8525.6 ms (upload 7825.5) | 0.1173 tok/s | 0.1173 | calc:6742a55a | 14.99 GiB | PASS |
| refill path, B=1 (`run-refill-b1`) | 5 groups | 15.61 GiB | 1648.5 ms (reader 1553.7 at 10.77 GB/s, compute 163.9) | 0.6066 tok/s | 0.6066 | calc:603d188a | 15.02 GiB | PASS |
| refill, B=1, slots-4096 build (`run2-refill-b1`) | 5 groups | 15.61 GiB | 1649.0 ms | 0.606 tok/s | 0.606 | (same as above) | 15.02 GiB | PASS |
| refill, B=4 (`run2-refill-b4`) | 4 groups | 17.03 GiB | 1814.0 ms (compute 676.9, reader 1699.2) | 0.551 tok/s | 2.2051 tok/s | calc:d054943e | 14.38 GiB | PASS |
| refill, B=8 (`run2-refill-b8`) | 4 groups | 17.03 GiB | 2078.4 ms (compute 1348.8, reader 1712.5) | 0.481 tok/s | 3.8491 tok/s | calc:4c90ee02 | 15.43 GiB | PASS |
| refill, B=12 and B=16 | | | | | | | | refused at open: physics nvrm live table full (4096) |

- Refill vs upload, B=1: 5.172x (calc:d0c59d6b). The upload path was bound by omega's byte-loop NaN scan
  (3.33 GB/s) and per-group allocation, not by the NVMe (`upload-bench*.json`).
- Against the ceiling: NVMe floor for 15.61 GiB at 10.4 GB/s is 1.6083 s (calc:a5a8e4c1). The measured B=1
  pass reaches 0.9756 of that floor (calc:dbd93b32). The read bench reached 10.03 to 12.81 GB/s on a fresh file.
- B=8 total is 6.3472x B=1 (calc:728c5eb0) and 0.598 of the resident single-conversation speed (calc:8b5ba014).
- Compute per conversation at B=8 is 168.6 ms (calc:3050c9b8). The compute-bound ceiling for the total is
  1 / 0.166 s = 6.0241 tok/s (calc:b52ac54d). Breakeven, where compute covers the read, is about 9.48
  conversations (calc:1421132d).
- B=4 and B=8 pin one group less than B=1 because each conversation's state counts against the budget.

### Why B=12 and B=16 were refused

Each conversation holds, per 4-layer group, one Q38Graph (1 activation buffer + 11 kernel code buffers + 5
command-list buffers), one GDN state object (4 buffers) and one attention cache (7 buffers): about 30 live
allocations per group, about 480 per conversation. omega's handle table was 1024 slots (batch 4 refused,
`run-refill-b4-slots1024-refused.log`); omega cdbd1fb3 raises it to 4096, equal to physics `NVRM_MAX_LIVE`,
which then refuses cleanly at B=12 and B=16. Nothing ran and no output was produced. Fix path (not done): share
the kernel code between graphs of the same shape, or let one state object serve several conversations; either
cuts the per-conversation count by more than half. Raising `NVRM_MAX_LIVE` is a physics change.

## The honest ceiling (issue question)

At 10.4 GB/s a dense model past RAM is a batch machine. Per-conversation speed is capped by streamed bytes /
NVMe rate, and batching only lowers it (0.606, 0.551, 0.481 tok/s at B=1, 4, 8). Batching raises the total
up to the compute rate. **No batch size recovers conversation speed.** The only lever that raises one
conversation's speed is more accepted tokens per pass: 10.61 per pass would match the resident 6.435 tok/s
(calc:1d53e97f; arithmetic only, speculative decoding is f7's).

Past 128 GiB (arithmetic only, 100 GiB resident, rest streamed at 10.4 GB/s):

| dense model | streaming per pass | per conversation | compute per conversation (173.2 GB/s effective) |
|---|---|---|---|
| 256 GiB | 16.11 s (calc:24081a2b) | 0.06207 tok/s (calc:16a99d3d) | 1.587 s (calc:07260dda) |
| 512 GiB | 42.54 s (calc:6ae131d5) | | 3.174 s (calc:81d4dce0) |

Prompt reading (prefill) breaks even for a 256 GiB model at about 6,086 tokens per pass at 207.7 TFLOPS FP8
(calc:9f59233c).

## UNVERIFIED / not run

- The read bench ran on a freshly written file; the drive's cache may have helped. The streamed runs, read
  later, held 10.66 to 10.77 GB/s.
- Controls ran on the refill path only (the upload path's flip and stale were not run).
- Batched conversations are copies of the same prompt (same golden), so they do not test diverging histories.
- B above 8 not measured (refused, see above). Everything past 128 GiB and the speculative-decode figure are
  arithmetic only.

## Files

`command-*.txt` hold the exact command per run (each under `flock /tmp/aien-gb10.lock`). `binary.sha256`:
first line for `run-*` (archive omega-f5-out, 06d5c876), second for `run2-*` (archive a1b0b11d, cdbd1fb3).
`drum-manifest.json` is the drum index; the 24 GB drum itself stays in scratch and is not committed.
