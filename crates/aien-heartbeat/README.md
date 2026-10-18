# aien-heartbeat

Expert Heartbeat milestone H1 ([aienix#4](https://github.com/aien-dev/aienix/issues/4), design in [this comment](https://github.com/aien-dev/aienix/issues/4#issuecomment-6246231509)): a fixed-slot ring that streams MoE expert weights from the internal NVMe into versioned memory slots at a steady rhythm. Standalone; no model math yet.

## What it does

- **Manifest** (`manifest.rs`): reads `model.safetensors.index.json` and every shard header, groups the six tensors of each `(layer, expert)` (gate/up/down `weight` + `weight_scale_inv`) into contiguous segments, and content-addresses each expert once with BLAKE3. safetensors orders tensors by dtype, so a Qwen3 FP8 expert is two segments (F8 weights, then the BF16 scales elsewhere in the file), three when it straddles a shard boundary.
- **Ring** (`ring.rs`): N aligned slots. A slot becomes visible as `(layer, expert, version)` only after the whole expert is read and its digest matches. A writer waits for every lease on a slot to drop before refilling it. Any I/O error, short read, digest mismatch or bad range fails the whole ring closed: later fills and leases return `RingFailed`.
- **Streamer** (`stream.rs`): a fixed pool of workers doing `O_DIRECT` `pread` into the slots in layer order. Schedule item `i` lands in slot `i % slots`, owned by worker `slot % workers`, so each slot has one writer. Hot-resident experts are excluded from the schedule.
- **Bench** (`heartbeat_bench`): full rotations over the real checkpoint; reports beat time, GB/s, worker read time and digest time separately. `--raw` reads and discards without digest or publish to show the device cost alone.

```
cargo build --release -p aien-heartbeat
quietlock check
target/release/heartbeat_bench --model ~/models/qwen3-30b-a3b-instruct-2507-fp8 \
  --manifest /tmp/hb/manifest.json --slots 64 --workers 16 --rotations 3 [--raw] [--hot-per-layer K]
```

## Measured (2026-10-18, DGX Spark, Samsung MZALC4T0HBL1 on PCIe Gen5 x4)

One rotation = 6,144 experts, 28,994,568,192 payload bytes. 3 rotations per row, O_DIRECT, 64 slots (80 for 20 workers).

| Mode | Workers | Beat (mean) | Payload GB/s | Receipt |
|---|---|---|---|---|
| raw read, no digest | 16 | 2.574 s | 11.26 | calc:d9a6b540 |
| verify + publish | 16 | 2.799 s | 10.36 | calc:5bb845e8 |
| verify + publish | 8 | 3.673 s | 7.89 | bench JSON (not receipted) |
| verify + publish, 16 hot/layer excluded | 16 | 2.437 s (5,376 experts) | 10.41 | bench JSON (not receipted) |

- Ceilings: PCIe Gen5 x4 link 15.75 GB/s (calc:94b7fdc7). The verified beat reaches 0.658 of it (calc:bb0b89f0), raw reads 0.715 (calc:ea65fc82). The earlier SSD figure of 10.03 GB/s (calc:14359036) is below what raw O_DIRECT reads measured here, so it is not the device ceiling.
- Digest cost: BLAKE3 runs at 1.50 GB/s per worker core (calc:ec3efd8e), about 58 worker-seconds per 3 rotations. With 16 workers the verified beat is 8.7% longer than the raw beat (calc:d216067c).
- io_uring was not built or measured. The `pread` pool reaches 11.26 GB/s raw; whether a raw io_uring path closes the remaining gap to the link ceiling is NOT MEASURED and is the next comparison to run.

## Expert Ribbon: on-demand paging (sc#629, sc#630, sc#632)

The ring above streams every expert in a fixed rhythm. The Ribbon path reads only the experts the router picks:

- **Layout** (`layout.rs`): checks each expert's six tensors (names, dtypes, shapes, lengths) against the model shape and splits a verified payload into what a GPU slot takes: gate|up `[2i][h]` e4m3 + bf16 scales, down `[h][i]` + scales.
- **Read adapter** (`reader.rs`): a bounded queue of `O_DIRECT` reads into a fixed pool of aligned buffers, BLAKE3-checked before a caller sees a byte. Full queue, I/O error, short read, digest mismatch, deadline, cancel and shutdown are distinct errors; nothing waits forever. `FakeShards` injects faults for tests.
- **Residency** (`residency.rs`): which expert sits in which slot, pins, LRU eviction of unpinned slots only, and a memory budget check.
- **Pager** (`pager.rs`): `ensure(layer, experts)` pins exactly the router's experts in router order; misses are read and loaded in parallel; any failure releases everything and fails the call. Nothing is substituted.
- The GPU side is omega's versioned slot pool (`omega_gpu_fp8_slots`, omega#533) and the engine path is `ExpertWeights::Paged` in `aien-inference-abi` (`qwen3_moe_omega/ribbon.rs`).

Random verified expert reads through the adapter (`ribbon_read_bench`, 3,000 reads, 4,719,168 payload bytes each, real checkpoint):

| Workers | Queue depth | Payload GB/s | Receipt |
|---|---|---|---|
| 1 | 1 | 1.32 | calc:b43572b8 |
| 8 | 16 | 8.69 | calc:84b90d97 |
| 16 | 32 | 10.60 | calc:8d297d5b |

At depth 32 random single-expert reads reach 0.673 of the PCIe Gen5 x4 link (calc:cf3a961c), about the same as the sequential verified beat.

OWN-30B on the GB10 with paged experts (evidence `docs/inference/evidence/ribbon-paged-20261018T1719Z`): greedy tokens IDENTICAL to the CPU golden with 768 slots (1/8 of the experts, prompts 1-2) and 3072 slots (1/2, prompts 1-4), 1.53 GB of other weights resident. Decode 2.396 tok/s cold at 768 slots (calc:eac8a69d) and 7.07 tok/s warm at 3072 slots (calc:59123200); the all-miss ceiling at the measured read rate is 5.849 tok/s (calc:1fc0715c). Paging is bound by per-layer read latency, not SSD bandwidth (1.715 GB/s delivered during misses, calc:ff9762d7); overlap and prefetch (sc#631) are next.

```
target/release/ribbon_read_bench --model ~/models/qwen3-30b-a3b-instruct-2507-fp8 --reads 3000 --workers 16 --depth 32
own30b_check chip <ckpt> <golden.json> <dir> --tokens-only --weights fp8 --router device --paged-experts 768
```

## Limits

- The ring (H1) has no GPU fence of its own; the Ribbon path loads into omega's slot pool, which publishes a version only after the platform sync and refuses to reload a slot a launch has pinned.
- Linux-hosted only; native AIENOS NVMe is a separate qualification.
- Digest runs on the worker after its read, not overlapped inside a worker.
- Bench uses no concurrent readers; lease waits are exercised by the tests.
