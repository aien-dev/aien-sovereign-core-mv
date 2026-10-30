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

## Limits

- No GPU visibility fence. On the GB10 unified memory a slot still needs an explicit fence before a kernel may read it; that is H2 work (with station queues and the batched expert GEMM).
- Linux-hosted only; native AIENOS NVMe is a separate qualification.
- Digest runs on the worker after its read, not overlapped inside a worker.
- Bench uses no concurrent readers; lease waits are exercised by the tests.
