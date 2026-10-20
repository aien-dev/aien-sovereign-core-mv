# KV memory ownership and concurrency: the contract as the code has it

This page maps who owns every KV buffer and staging region, how they change state, who may write
when, what makes a write visible to the GPU, and what bounds, cancellation, reuse and restart
recovery actually exist. It is a map of current code and receipts, not a design. The design
authority for persistent KV is
[ADR 0003](../../docs/adr/0003-persistent-kv-identity-storage-contract-and-ownership.md); this
page links into it instead of repeating it. Tracker: aien-spark/aien-architecture#233 (P1 memory
contracts lane). Owning issues: sc#485 (KV offload concurrency), sc#494 (Wayfinder: persistent KV
on the Spark), aienos#367 (native persistent KV), aienos#327 (native GB10 model).

Why it lives here: `aien-kv-cache` owns the pool and the block tables every other layer writes
through, and the issue asks for documentation next to the source rather than another top-level
document. `docs/persistent-kv/` covers persistence only; this page also covers the in-memory pool
and GPU visibility.

## How to read the claims

- Line links point at sovereign-core main
  [`deff6f7`](https://github.com/aien-spark/aien-sovereign-core/tree/deff6f7a56e707871cefca9a35ee0fa17c8b4d40)
  (base `SC`) and omega at the `omega.lock` pin
  [`4799f9e`](https://github.com/aien-spark/omega/tree/4799f9eba563e9c69579b47c477e7d9c2b0f15cb)
  (base `OM`). Function names are the stable anchor when lines move.
- Every claim carries one tag:
  **[S]** source-observed (the quoted line is in the linked source),
  **[H]** host-tested (named test, CPU only, passes at `deff6f7`),
  **[C]** chip-tested (named receipt; hosted Linux, GB10 through the NVIDIA Linux driver),
  **[N]** native-hardware-tested (named receipt; Aienux on real hardware, no Linux),
  **[I]** inferred (reasoning from the cited lines, not run),
  **UNVERIFIED** / **NOT_RUN** (nothing proves it).
- An asynchronous API is not an overlap proof. No receipt below shows storage I/O overlapping chip
  compute; every overlap row is NOT_RUN.

[SC]: https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40
[OM]: https://github.com/aien-spark/omega/blob/4799f9eba563e9c69579b47c477e7d9c2b0f15cb

## 1. Platforms are separate claims

| Platform | KV code that runs there | Strongest evidence |
|---|---|---|
| Linux-hosted Spark, CPU reference backend | `AienKvManager` + `UnifiedKvTensorPool` on a host-mapped `LinuxUnifiedBuffer`; persistent KV through `aien-kv-store` | [H] the tests named below |
| Linux-hosted Spark, GB10 through the NVIDIA Linux driver (`OmegaGb10Backend`, 8B daemon path) | same manager, pool memory from omega's resident allocation | [C] `docs/inference/evidence/cta1024-20261012T0315Z/` (section 4) |
| Linux-hosted Spark, OWN-30B door (`aien-serve-http`) | the engine's own single-sequence KV (`OmegaPagedKv`, `OmegaFusedKv`), not `AienKvManager` block tables; park/restore through `kv_persist.rs` | [H] host engine tests; chip check `crates/aien-serve-http/tests/kv_persist_chip.rs` is `#[ignore]`, NOT_RUN (sc#507) |
| Native Aienux, Pi 5 CPU | `aienos-infer` `DecodeState`: one dense f32 cache, no paging, no sharing, no persistence | [N] aienos `docs/native/PI5-6-INFER-RUNS.md` (8 tokens equal llama.cpp, "KV capacity 64 in this probe") |
| Native Aienux, GB10 GPU | none: no native GPU platform implementation exists | NOT_RUN (aienos#327; omega `docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md` section 2 rows NOT_RUN) |
| Emulator / QEMU only | native persistent-KV slab is PLANNED in QEMU (ADR 0003 section 11, cuts N2/N3/N5); no code exists | NOT_RUN (aienos#367) |

A hosted chip result never stands in for native GB10, and a Pi 5 result never stands in for the Spark.

## 2. Owners

| Buffer | Owner and lifetime | Evidence |
|---|---|---|
| Block allocator, block metadata, block tables, prefix cache | `AienKvManager` ([SC]/crates/aien-kv-cache/src/lib.rs#L830), shared as `SharedKvManager = Arc<RwLock<AienKvManager<B>>>` ([#L1890](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1890)). One manager for spine and backend: "creates ONE manager with a physical tensor pool ... and hands the same `Arc` to the spine ... and to the backend" ([shared_kv.rs#L5-L10](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/shared_kv.rs#L5-L10)) | [S] |
| Pool bytes | `UnifiedKvTensorPool` owns its buffer `B` ([lib.rs#L399-L403](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L399-L403)); freed when the pool drops | [S] |
| Pool backing, case A: host-mapped | `LinuxUnifiedBuffer::allocate(..., ResidencyPolicy::FaultIn)` ([lib.rs#L643-L664](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L643-L664)); `HeapUnifiedBuffer` off Linux and in tests ([#L347-L397](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L347-L397)) | [S] |
| Pool backing, case B: omega resident pool | omega allocates, the pool borrows: `external_default_buffer` "`release` runs exactly once, when the buffer is dropped" ([lib.rs#L322-L344](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L322-L344)); requested by `OmegaGb10Backend::alloc_kv_pool` ([omega_backend.rs#L1125-L1150](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/omega_backend.rs#L1125-L1150)) and by the 30B `OmegaPagedKv::new` ([qwen3_moe_omega.rs#L2968](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/qwen3_moe_omega.rs#L2968)). "One per process" ([aien-omega-gpu/src/lib.rs#L1170-L1175](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-omega-gpu/src/lib.rs#L1170-L1175)); a second request is refused "attention pool: already allocated" ([OM]/src/omega_gpu_attention_api.c#L775) and the caller falls back to case A | [S] |
| Dense per-sequence copy | `NativeTransformerBackend.sequences: HashMap<u64, SequenceState>` ([transformer_backend.rs#L163](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L163)). With a pool attached it is scratch: "Drop the dense copy only when the physical pool holds the K/V" ([#L1694-L1698](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L1694-L1698)) | [S] |
| Chip staging (q, table, out, staged pool) | omega statics `g_q, g_pool, g_tab, g_out` "staging, grown to the high-water mark, reused per call" ([OM]/src/omega_gpu_attention_api.c#L744), all session allocations GPU-uncached ([OM]/src/omega_gpu_session.c#L228) | [S] |
| 30B engine KV | `OmegaPagedKv` (one sequence, blocks in order, resident pool when granted, [qwen3_moe_omega.rs#L2943-L2950](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/qwen3_moe_omega.rs#L2943-L2950)); `OmegaFusedKv` wraps `aien_omega_dattn::DecodeAttnCache`, "GPU-resident KV, one sequence", the chip does "the bf16 K/V append" ([qwen3_moe_omega_dattn.rs#L1-L18](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/qwen3_moe_omega_dattn.rs#L1-L18)) | [S] |
| Storage references (park and restore) | a holder is a real block table under a reserved id, `HOLDER_SLOT << 32 \| job` ([lib.rs#L11-L29](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L11-L29)); the storage worker never holds a raw block id (ADR 0003 section 5.2) | [S], [H] `holder_ownership_tests.rs::holder_id_never_collides_with_arena_ids` |
| Worker in-flight buffers | the worker keeps every source and destination until that job's terminal completion (ADR 0003 section 7.5 (a)) | [H] `worker_tests.rs::worker_shutdown_keeps_buffers_until_each_job_terminal` |
| Persisted objects | host store files under `AIEN_KV_STORE_DIR` (`aien-kv-store/src/host_fs.rs`); opt-in, off when unset ([shared_kv.rs#L590-L604](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/shared_kv.rs#L590-L604)) | [S], [H] `kv_restart_test.rs::persistence_off_when_store_dir_unset_changes_nothing` |
| Native Pi KV | `DecodeState.k`/`.v`, "`[layer][pos][kv_dim]`", allocated once for `capacity` (aienos `crates/aienos-infer/src/decode.rs` L21, L56-L62 at aienos `b7ebf7c`) | [S], [N] runs above |
| Native NVMe bounce buffer | the driver's own: "Every request goes through the bounce buffer ... One command is outstanding at a time" (aienos `native/disk/disk_nvme.h` L17-L19) | [S]; no KV code uses it yet |

## 3. State transitions

**Block** (`KvBlock { ref_count, num_tokens, is_shared }`, [lib.rs#L675-L681](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L675-L681)):

```text
free --allocate_block (ref 1, zeroed)--> private --fork_sequence / fork_range / assemble_table (ref +1, is_shared)--> shared
shared --append on a partial shared tail: copy-on-write into a new private block, old ref -1--> (old stays shared or private)
any --free_sequence / truncate_sequence: ref -1; at ref 0 back on the free list--> free
```

- Allocation zeroes the block: "`pool.zero_block(blk_id)`" in `allocate_block` ([lib.rs#L1065-L1078](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1065-L1078)). [S], [H] `kv_lifecycle_qualification.rs::freed_block_reuse_never_exposes_previous_sequence_kv`.
- Copy-on-write lives only in `append_token_with_slot` (host `copy_block`, [#L1440-L1462](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1440-L1462)) and `append_token_with_slot_device` ([#L1488](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1488)). [S], [H] `cow_branching_tests.rs::test_tail_divergence_cow`, `kv_lifecycle_qualification.rs::sibling_appends_do_not_alter_parent_or_siblings`.
- A failed device copy gives the new block back and counts no fault ([#L1514-L1525](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1514-L1525)). [H] `kv_lifecycle_qualification.rs::device_cow_copy_failure_releases_the_new_block`.

**Block table readiness** (`PrefillState`, [lib.rs#L683-L700](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L683-L700)): `Allocated -> PrefillPending -> PrefillReady -> SharedFrozen`; `complete_prefill` is "the ONLY setter of `PrefillReady`" ([#L1155-L1176](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1155-L1176)), called by the scheduler after the forward returned Ok ([aien-scheduler/src/lib.rs#L1257](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-scheduler/src/lib.rs#L1257)). [S], [H] `lib.rs` tests `test_prefill_gate_*`.

**Sequence residency** (persistent KV, [kv_residency.rs#L26-L39](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/kv_residency.rs#L26-L39), design in ADR 0003 section 5.1):

```text
Resident --finish: fork_range one holder per chunk (S1), sequence freed--> (parked = objects in the store; no runtime state)
submit: chain hit --S2--> Restoring --every job Completed + S3--> Prefilling --suffix prefilled--> Resident
Restoring --any job fails or is cancelled--> Draining --every sibling terminal--> Failed --plain prefill--> Resident
```

"`Parked` is a property of the store (objects exist), not of the runtime" (ADR 0003 section 5.1). [S].
Host tests: `kv_park_restore_test.rs::finished_turn_publishes_full_chunks_and_partial_tail`,
`awaiting_restore_is_not_admitted_until_all_chunks_completed`,
`restore_txn_adopts_assembled_table_only_after_all_jobs_terminal`,
`multi_chunk_restore_failure_does_not_free_destination_in_use`. [H]

**30B door** (separate path, no residency map): park exports after a generation under the GPU lock, restore imports all or nothing; `restore_kv` "On any error the engine is reset (empty history)" ([qwen3_moe_omega.rs#L3959-L3964](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/qwen3_moe_omega.rs#L3959-L3964); [kv_persist.rs#L11-L22](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-serve-http/src/kv_persist.rs#L11-L22)). [S], [H] host engine tests (sc#514); chip NOT_RUN.

**Native Pi**: no states beyond `pos`; `reset` "Forget the sequence (buffers are kept)" sets `pos = 0` (aienos `decode.rs` L93-L96). [S]

## 4. Who may write when

- One step owner. The daemon's worker loop takes the spine `tokio::Mutex` for each `step` ([server.rs#L73](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/server.rs#L73), [#L259-L268](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/server.rs#L259-L268)); all block-table changes take the manager `RwLock` write guard. Lock order spine -> KV -> store index (ADR 0003 section 8). [S]
- The writable slot comes only from an append. Decode reserves one slot per token before any layer runs ([transformer_backend.rs#L1156-L1185](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L1156-L1185)) and writes each layer through `write_explicit_token_kv` ([#L1211-L1222](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L1211-L1222)); prefill resolves every position before touching state ([#L1420-L1462](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L1420-L1462)). [S]
- The manager's `write_token_kv` / `write_explicit_token_kv` do **not** check `is_shared` or `ref_count` ([lib.rs#L1810-L1857](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1810-L1857)). Sharing safety is the caller's contract: write only a slot an append returned (always private) or a position of a table nobody shares. [S]; a write into a shared block would change every sharer [I].
- Shared prefixes are immutable: a fork of a ready table freezes both sides `SharedFrozen` ([lib.rs#L1214-L1225](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1214-L1225)); `truncate_sequence` refuses a shared block ([#L1639-L1646](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1639-L1646)). [S], [H] `truncate_sequence_tests.rs::truncate_refuses_shared_blocks`, `holder_ownership_tests.rs::owner_append_on_shared_partial_block_copies_not_holder`.
- Storage holders are written only by restore loads into holder-owned blocks that "are invisible to the scheduler until assembled" (ADR 0003 section 5.4); commit (S3) runs on the spine thread, never on the worker ([kv_residency.rs#L17-L21](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/kv_residency.rs#L17-L21)). [S], [H] `kv_park_restore_test.rs::park_never_blocks_step`.
- Value rounding: a bf16 pool rounds on write, `f32_to_bf16_bits` round-to-nearest-even ([lib.rs#L561-L568](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L561-L568)); the shared pool is bf16 since sc#384, "K/V are rounded to bf16 on write" ([shared_kv.rs#L109-L116](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/shared_kv.rs#L109-L116)). Within one prefill chunk attention reads the unrounded dense f32 copy, while a later chunk rebuilds the dense copy from the bf16 pool ([transformer_backend.rs#L1488-L1508](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L1488-L1508)), so chunked and one-shot prefill can differ [S for the code, I for the effect]. In flight, PR pending: branch `fix/prefill-e2e2-kv-round-at-birth` rounds K/V once at birth (`SequenceState.kv_dtype`, `round_to_dtype_in_place`). Not on main; nothing here depends on it.

## 5. GPU and DMA visibility

Hosted GB10, resident pool (case B):

1. Memory: every omega session allocation is `.gpu_cache = OMEGA_GPU_CACHE_GPU_UNCACHED` ([OM]/src/omega_gpu_session.c#L228); the binding states "one GPU-uncached, host-visible allocation ... whose chip address equals its host address" ([aien-omega-gpu/src/lib.rs#L1170-L1175](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-omega-gpu/src/lib.rs#L1170-L1175)). [S]
2. Host write: plain CPU stores into the pool under the KV write guard (section 4). [S]
3. Publish to the chip: each attention launch calls `omega_gpu_session_sync_to_gpu` on the resident pool ([OM]/src/omega_gpu_attention_api.c#L911-L914); the Linux platform's sync is a full barrier, "uncached system memory: a full barrier is the whole sync" (`dsb sy` on AArch64, [OM]/src/omega_gpu_platform_linux.c#L239-L258). A GPU-cached buffer would instead queue an L2 invalidate on the next submit (same lines, omega#457). [S]
4. Completion: the CPU polls an uncached marker written after `L2_FLUSH_DIRTY` ([OM]/src/omega_gpu_session.c#L501-L511); the rule "anything the CPU polls for a device write is HOST_POLLED and must be GPU_UNCACHED" is enforced by the platform check (omega `docs/numeric/OMEGA_GPU_NATIVE_BOUNDARY.md` section 2). [S]
5. Synchronous: `chip_attention` holds the backend's single chip mutex across the call ("One lock serializes every chip call", [omega_backend.rs#L81-L83](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/omega_backend.rs#L81-L83), [#L543-L570](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/omega_backend.rs#L543-L570)), so no host write to the pool races a running launch from this backend [S for the lock, I for the conclusion].
6. Receipt: [C] `docs/inference/evidence/cta1024-20261012T0315Z/FINDINGS.md` "Decode attention reads a resident KV pool" and `smoke-ctx-resident.json`: `kv_resident_calls_total=12312`, fallback 0, output ids equal to the staged run. Built at omega `809c0aa6`, not the current pin `4799f9e`; it shows correct replies, not the barrier in isolation.

Other paths:

- Host-mapped pool (case A) or a block id past the kernel's 32-bit offset limit: omega copies the referenced data into its staging pool on every call ([OM]/src/omega_gpu_attention_api.c#L902-L905, `resident_fits` #L760-L766), counted in `kv_staged_bytes_total`. [S], [C] bf16 pool receipt `smoke-bf16kv-cta1024.json` (staged prompt-read rows).
- `append_token_with_slot_device` / `copy_block_device` return a `Fence`, but on Linux `MemoryDevice::copy` is a CPU `ptr::copy` and `synchronize` returns `Ok(())` ([aien-platform-linux/src/device.rs#L63-L82](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-platform-linux/src/device.rs#L63-L82)); `allocate_block` zeroes the new block on the host first. No production caller uses the device variant (only `kv_lifecycle_qualification.rs`). It is not a DMA path today. [S]
- 30B fused path: the chip appends K/V itself inside the decode launch, and export reads whole bf16 rows back through omega's cache helpers (sc#514 commit message; [qwen3_moe_omega_dattn.rs#L1-L9](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/qwen3_moe_omega_dattn.rs#L1-L9)). Export runs "under the GPU lock" ([kv_persist.rs#L1099](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-serve-http/src/kv_persist.rs#L1099)). [S]; GPU-to-host ordering of that read UNVERIFIED here.
- Hazard: when the omega session closes, omega zeroes its record of the resident pool, "the caller's mapping is dead" ([OM]/src/omega_gpu_attention_api.c#L808-L811), while an `AienKvManager` may still hold the pointer. Nothing in sovereign-core orders session close after the pool drops. [S for both lines, I for the risk]
- Native GB10: no visibility code exists; the platform contract names `sync_*` and the DMA service, device NOT_RUN (omega boundary doc section 2 "Visibility / cache maintenance ... NOT_RUN"). Native NVMe: DMA visibility is the driver's, through its bounce buffer (ADR 0003 section 9 item 1). [S], NOT_RUN for KV.

## 6. Bounded queueing and backpressure

| Bound | Where | Evidence |
|---|---|---|
| Storage intake per class: 48 restore ops, 64 park ops, 256 MiB in flight, 16 waiters per job, 1024 waiters total, dispatch depth 4 | `WorkerConfig::default` ([aien-kv-store/src/worker.rs#L58-L71](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-store/src/worker.rs#L58-L71)); refusal is `PlatformError::QueueFull` ([#L417](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-store/src/worker.rs#L417), `credit_check` #L500) | [S], [H] `worker_tests.rs::worker_bounds_ops_and_bytes_in_flight`, `coalesced_waiters_are_bounded` |
| Dispatch fairness 3 restores : 1 park | [worker.rs#L712-L726](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-store/src/worker.rs#L712-L726) | [S]; the 10 000-op saturation test is PLANNED in ADR 0003 section 7.2, UNVERIFIED that it exists |
| Refused park = "not parked this turn" | ADR 0003 section 7.1 | [H] `kv_concurrency_test.rs::queue_full_and_disk_full_under_load_never_lose_resident_progress` |
| Restore refused below the scheduler watermark | S2, [kv_residency.rs#L14-L16](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/kv_residency.rs#L14-L16) | [H] `kv_park_restore_test.rs::restore_refused_below_watermark_falls_back` |
| KV admission: blocks needed (`incremental_blocks_needed`) plus watermark (default 4 blocks); preempt below watermark; `max_batch_size` 64 per step | [aien-scheduler/src/lib.rs#L44-L55](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-scheduler/src/lib.rs#L44-L55), [#L718-L752](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-scheduler/src/lib.rs#L718-L752), [#L953-L968](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-scheduler/src/lib.rs#L953-L968) | [S], [H] `incremental_blocks_tests.rs::prediction_matches_real_appends_across_shapes`, `kv_concurrency_test.rs::memory_pressure_restore_and_admission_make_progress_without_deadlock` |
| Pool exhaustion is a loud error, checked before mutation | [transformer_backend.rs#L1156-L1158](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-inference-abi/src/transformer_backend.rs#L1156-L1158) | [S], [H] `kv_lifecycle_qualification.rs::exhaustion_refuses_without_changing_state` |
| Scheduler waiting queue | `VecDeque` with no length limit ([aien-scheduler/src/lib.rs#L114](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-scheduler/src/lib.rs#L114)); the arena "capacity" is only the initial size: `allocate_slot` pushes a new slot when none is free ([sequence.rs#L418-L444](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-scheduler/src/sequence.rs#L418-L444)) | [S]: **no bound observed**; whether the socket/HTTP intake bounds requests is UNVERIFIED |
| Native NVMe | one outstanding command; ADR plans a step-budgeted `kv_slab_pump` (ADR 0003 section 9 item 3) | [S] driver; pump NOT_RUN, no code |

Overlap. [H] `kv_concurrency_test.rs::resident_sequence_emits_tokens_before_unrelated_restore_completes` shows a resident sequence emitting tokens while a slow fake restore is pending, on the CPU reference backend. That is progress ordering, not measured I/O-compute overlap. Chip and native overlap: NOT_RUN (sc#508, aienos#367 "prove overlap or explicitly report the limitation"; ADR 0003 section 9 item 8 reports it as a limitation).

## 7. Cancellation

- A restore with a failed or cancelled chunk moves to `Draining` and keeps every destination until each sibling job is terminal ([kv_residency.rs#L1025](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/kv_residency.rs#L1025), ADR 0003 section 5.5). [H] `kv_park_restore_test.rs::multi_chunk_restore_failure_does_not_free_destination_in_use`.
- Cancel removes only the caller's waiter; shared loads keep running ([worker.rs#L1451](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-store/src/worker.rs#L1451)). [H] `worker_tests.rs::worker_cancel_removes_only_own_waiter`, `cancel_waiter_removes_exactly_that_incarnation`, `worker_never_touches_dest_after_cancel_ack`.
- Fork then cancel of the parent during a park keeps the child valid. [H] `kv_concurrency_test.rs::fork_then_cancel_parent_during_park_keeps_child_valid`.
- A failed reservation rolls back by snapshot (`begin_transaction` / `rollback`, [lib.rs#L1567-L1616](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-cache/src/lib.rs#L1567-L1616)). [H] `lib.rs::test_kv_transaction_commit_and_rollback`.
- Chip calls cannot be cancelled mid-launch from sovereign-core; the call returns when the marker arrives or the poll times out (section 5) [S]. Native: NOT_RUN.

## 8. Reuse

- Blocks: zeroed on allocation (section 3). [H] `freed_block_reuse_never_exposes_previous_sequence_kv`.
- Sequence ids: a reused arena slot gets a new generation, and a completion commits only if `(seq_id, epoch)` still matches ([kv_residency.rs#L36-L39](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/kv_residency.rs#L36-L39)). [H] `kv_park_restore_test.rs::stale_completion_cannot_touch_reused_sequence_id`, `worker_tests.rs::stale_completion_with_reused_seq_id_but_new_epoch_is_rejected`.
- Holders: a holder table is never replaced, and its blocks return to the free list only at ref 0. [H] `holder_ownership_tests.rs::holder_free_returns_blocks_only_at_zero_refcount`, `fork_range_forks_only_the_named_blocks_and_does_not_replace_a_live_child`.
- Chip staging: reused per call, grown to the high-water mark (section 2). [S]
- Native Pi: buffers kept across prompts, `pos` reset (section 3), so a native park must copy out before the next prompt (ADR 0003 section 9 item 7). [S], NOT_RUN.

## 9. Restart recovery

- The in-memory pool does not survive a process exit: it is an allocation owned by the process (section 2). Without `AIEN_KV_STORE_DIR` a restart recomputes everything. [S], [H] `kv_restart_test.rs::persistence_off_when_store_dir_unset_changes_nothing`.
- With persistence on, recovery runs in slices on the worker thread and every lookup misses until it is done, so a cold start costs only plain prefills ([shared_kv.rs#L582-L589](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-runtime/src/shared_kv.rs#L582-L589), `recover_step_on` [host_fs.rs#L1704](https://github.com/aien-spark/aien-sovereign-core/blob/deff6f7a56e707871cefca9a35ee0fa17c8b4d40/crates/aien-kv-store/src/host_fs.rs#L1704)). [H] `host_fs_recovery_tests.rs::recovery_deletes_tmp_and_corrupt_headers_only`, `kv_restart_test.rs::process_restart_rediscovers_and_restores_identical_bytes`, `greedy_continuation_after_restart_matches_uninterrupted_baseline` (real child processes, CPU reference backend).
- A changed model, engine build, weight format or rounding class is never served. [H] `kv_restart_test.rs::*_is_never_served_after_restart`.
- Shutdown drain is bounded for intake (default 2000 ms), not for a stalled `fsync` (ADR 0003 section 7.5). [H] `worker_tests.rs::shutdown_with_stalled_fsync_returns_by_deadline_and_reports_unfinished`.
- Clean reboot, power loss, hardware cold boot: NOT CLAIMED on the host, PLANNED in QEMU and on Pi 5, Spark-native NOT_RUN (ADR 0003 section 11, aienos#336).
- Chip: the 30B restart-and-restore acceptance check (`crates/aien-serve-http/tests/kv_persist_chip.rs`, `#[ignore]`) is NOT_RUN (sc#507); the real-NVMe benchmark is NOT_RUN (sc#508).

## 10. Open items this page does not close

- sc#485 acceptance boxes and sc#494 items 4 and 5 (concurrency and performance on the real Spark): NOT_RUN on chip.
- aienos#367: no native KV persistence code exists; every native row above is design (ADR 0003 section 9) or NOT_RUN.
- aienos#327: no native GB10 KV path exists.
- Two source-observed gaps worth an owner's look, unchanged by this documentation: the scheduler waiting queue has no bound (section 6), and omega session close can invalidate a pool sovereign-core still points at (section 5).
