# ADR 0003: Persistent KV cache: versioned chunk identity, storage contract, ownership and recovery

- **Status**: Proposed (2026-10-13), revised 2026-10-13 after the independent review (section 17). Binding design for sovereign-core #484 and #485 and aienos #367. Implementation cuts: `docs/persistent-kv/CUTS.md`. Metrics and settings: `docs/persistent-kv/METRICS_AND_SETTINGS.md`.
- **Date**: 2026-10-13
- **Author**: Drake Stapleton <aien@aienos.com> (design drafted by the persistent-KV design owner, Fable 5.1)
- **Source read at**: sovereign-core `origin/main` f560be4; aienos `origin/main` 5429602; LMCache `4d69cdc4` (Apache-2.0).
- **Evidence labels**: VERIFIED (read in source at the cited line, or run and the output seen), INFERRED (follows from source, not run), UNKNOWN (not established), PLANNED (a test or measurement that is designed here and has no receipt yet). Nothing in this ADR was run on hardware, and no proposed test has a receipt: every test name below is PLANNED until its cut merges with a green run.

## In plain words

Today the runtime computes the model's "working memory" (the KV cache) for every turn from scratch and throws it away when the turn ends (spine.rs:198-230 frees the sequence on `Finished`). This design saves that working memory to disk in fixed-size, self-describing pieces (chunks), each named by a fingerprint of exactly what it contains (which model, which tokens, at which positions, in which byte layout). When the same conversation continues, even after the runtime process restarts, the saved pieces are checked and loaded back instead of being recomputed. Saving and loading happen on a separate worker so the model keeps answering other conversations meanwhile. Nothing in this design changes which answers the model gives: a restored continuation must produce the same greedy tokens as an uninterrupted one, and anything that cannot be proven compatible is recomputed from the conversation history, which stays the authority.

## Context (what exists, verified)

### Host runtime (sovereign-core)

1. KV memory is one pool of fixed blocks. `KvPoolConfig { num_blocks, block_size, num_layers, num_kv_heads, head_dim, dtype }` (aien-kv-cache/src/lib.rs:46-53, VERIFIED). The pool is one allocation (`UnifiedKvTensorPool::allocate`, lib.rs:611-631, VERIFIED); freeing a block returns it to `free_blocks` and does not release memory (`free_sequence`, lib.rs:1413-1431, VERIFIED).
2. Block layout is fixed by `KvLayout::from_config` (lib.rs:169-218, VERIFIED): `head_stride = head_dim * element_bytes`, `token_stride = num_kv_heads * head_stride`, `kv_plane_stride = block_size * token_stride`, `layer_stride = 2 * kv_plane_stride` (K plane then V plane), `block_stride = num_layers * layer_stride`. Element offsets: `block*block_stride + layer*layer_stride + is_value*kv_plane_stride + token*token_stride + head*head_stride + dim*element_bytes` (lib.rs:221-242, VERIFIED). The daemon pool is bf16 (`SHARED_KV_DTYPE = KvDType::Bf16`, aien-runtime/src/shared_kv.rs:116, VERIFIED; K/V are rounded on write, lib.rs:530-537, VERIFIED).
3. Blocks carry `ref_count`, `num_tokens`, `is_shared` (`KvBlock`, lib.rs:643-648, VERIFIED). `fork_sequence` bumps every parent block's refcount and marks it shared (lib.rs:1165-1210, VERIFIED). `append_token_with_slot` copies on write only when the last block is shared and not full (lib.rs:1234-1255, VERIFIED): sharing a full block costs nothing. `free_sequence` decrements and frees at zero (lib.rs:1413-1431, VERIFIED).
4. `BlockTable { sequence_id, block_ids, total_tokens, prefill_state }` with `PrefillState::{Allocated, PrefillPending, PrefillReady, SharedFrozen}` (lib.rs:658-668, 714-722, VERIFIED). Decode requires a ready table (aien-scheduler/src/lib.rs:26-34, VERIFIED). `prefix_cache` on the manager is dead code (`#[allow(dead_code)]`, lib.rs:804-805, VERIFIED).
5. The manager is shared as `Arc<parking_lot::RwLock<AienKvManager>>` (lib.rs:1685, VERIFIED) between the spine (spine.rs:23) and the backend (`NativeTransformerBackend.kv_manager`, transformer_backend.rs:130).
6. The daemon steps one spine under a `tokio::sync::Mutex` (server.rs:27, 196-230, VERIFIED): one worker, `step` per iteration, 5 ms sleep when idle. `AienScheduler::step` calls `backend.execute_step` with no KV lock held (aien-scheduler/src/lib.rs:1005-1015, VERIFIED) and takes the KV write lock only for short commits (`complete_prefill` at 1027-1032, `append_token` at 1070, `free_sequence` at 1097-1106, VERIFIED).
7. Sequence ids are `slot: u32 << 32 | generation: u32`, generation 0 invalid (aien-scheduler/src/sequence.rs:7-50, VERIFIED). The arena reuses slots with a bumped generation (sequence.rs:329-370, VERIFIED). `SequenceRecord` has `prompt_tokens_prefilled` and `is_prefilled` (sequence.rs:221-222, VERIFIED), which the step uses to drive chunked prefill (lib.rs:1023-1033, VERIFIED).
8. Backend per-sequence state is `SequenceState { tokens, layers }` (aien-inference-abi/src/weights.rs:20-23, VERIFIED); in paged mode `layers` is empty and `tokens.len()` equals the number of cached positions while prefilling (transformer_backend.rs:136-143, VERIFIED). `pending_prefill_token` holds the token sampled after the last prefill chunk (same lines). Sampling is seeded by `decode_sampling_seed(request_id, position)` (transformer_backend.rs:43-45, VERIFIED). The runtime calls the **trait** method `release_sequence` (transformer_backend.rs:2307, VERIFIED), which drops backend state only and never touches the KV manager (its doc comment says the scheduler and swarm manager free the block tables, VERIFIED at 2300-2306); the inherent `NativeTransformerBackend::release_sequence` at 613-622, which also calls `release_branch`, is the standalone-generation path and is not used by the runtime. Two more facts that the park path depends on, both VERIFIED: a decode step appends the sampled token to `SequenceState.tokens` after computing the previous token's K/V (transformer_backend.rs:1776-1782 reads `pos = tokens.len() - 1`, 2158 pushes the sample), so the last token of a finished sequence has no K/V; and `fork_sequence` copies the parent's whole table and replaces a live child id (aien-kv-cache/src/lib.rs:1165-1210).
9. `ModelConfig` carries `model_id, max_sequence_length, block_size, num_layers, num_heads, head_dim, num_kv_heads, rope_theta, rope_scaling: Option<Llama3RopeScaling { factor, low_freq_factor, high_freq_factor, original_max_position_embeddings }>, qk_norm, ...` (aien-abi-core/src/lib.rs:123-160, 45-50, VERIFIED). The daemon records `ModelIdentity { model_sha256, model_digest_kind, model_path, tokenizer_sha256, tokenizer_path }` (aien-runtime/src/generation.rs:24-30, VERIFIED).
10. Memory planning from #239 (PRs #246 and #270): `KvPoolPlan` (shared_kv.rs:130-145), `plan_checked_model_kv` (shared_kv.rs:368-388), operator cap `AIEN_KV_CONTEXT_TOKENS` (shared_kv.rs:306), `AIEN_ALLOW_UNCHECKED_KV_MEMORY` (shared_kv.rs:295), all VERIFIED. This design reuses the plan and adds no second sizing path.
11. Platform traits: `ObjectStore { read(ObjectId) -> ObjectRef, commit(ObjectWrite) -> ObjectId }` (aien-platform/src/objects.rs:17-20), `UnifiedBuffer` (compute.rs:18-28), `PlatformError` (lib.rs:31-43), `WorkQueue` (queue.rs:47-55), all VERIFIED, all `no_std`. `ObjectStore` is synchronous and copies through a borrowed slice; it has no submit/complete split, no capacity, no delete. It is reused as the *identifier and error vocabulary* (`ObjectId`, `PlatformError`), not as the KV store trait (section 3 says why).
12. `RuntimeStatusReport` is built from `kv.metrics()`, `free_block_count`, `allocated_block_count` (spine.rs:472-497, VERIFIED). `KvMetrics { physical_pages, logical_pages, shared_pages, private_pages, cow_faults, bytes_saved_vs_full_copy, used_blocks }` (lib.rs:788-796, VERIFIED).
13. `sha2` and `blake3` are already in `Cargo.lock` (VERIFIED by grep); `crc32c`/`crc32fast` are not.

### Native (aienos)

14. `disk_dev`: `read`/`write` return only after the transfer completed or failed; `flush` returns only after every acknowledged write is persistent; requests over `max_blocks_per_io` are split (native/disk/disk.h:17-24, VERIFIED). `disk_queue`: at most `DISK_QUEUE_DEPTH` 32 requests, drained in order, FLUSH is a barrier, fail-stop until reset (disk.h:60-83, VERIFIED).
15. NVMe driver: every request goes through the bounce buffer, one command outstanding at a time, the driver waits for completion before returning (native/disk/disk_nvme.h:17-19, VERIFIED). Transfer per command `min(bounce bytes, 128 KiB, 4 KiB << MDTS)` (disk_nvme.h:53-56, VERIFIED). `nvme_write_is_power_fail_atomic` exists (disk_nvme.h:192, VERIFIED).
16. Store v1 is append-only with a rewritten catalog per transaction; objects up to 64 MiB, 256 per transaction, 4096 catalog entries (native/store/store_v1.h:16-22, VERIFIED). No delete, free or reclaim entry point exists in `store_v1.h` or `store_engine.h` (VERIFIED by grep). `st_transact` requires payload buffers alive and unchanged until it returns (store_engine.h:146-149, VERIFIED).
17. The sealed Store caps application objects at `SS_MAX_PLAINTEXT` 16384 bytes and 8 per transaction (native/store/store_sealed.h:57-58, VERIFIED) and refuses plaintext kinds (`SS_E_FOREIGN_KIND`, store_sealed.h:77, VERIFIED). The M5 envelope itself allows 64 MiB plaintext in chunks up to 64 KiB with a 16-byte tag per chunk (native/m5/m5.h:160-171, VERIFIED). Keys: `m5_hkdf_expand`, `m5_derive_object_key(domain_key, binding, ...)` (m5.h:36, 73, VERIFIED). AGENTS.md section 4 scratchpad rule applies to every decrypt.
18. `torn_slot`: dual-slot torn-write-safe 4096-byte record, 4 units (native/store/torn_slot.h:53-55, VERIFIED).
19. Native inference: `ck_infer_prompt(text, len, cancel_poll, reply)` steps to the reply end calling `cancel_poll()` between steps (native/kernel/include/ck.h:240-262, VERIFIED). The session FFI is one step per call: `aienos_infer_session_open/prompt/step/cancel/tokens/close` (native/kernel/core/infer.c:76-93, VERIFIED). The probe session has `kv_capacity` 64 positions, `max_new` 16 (infer.c:268-274, VERIFIED). The native KV is f32, `k`/`v` each `n_layer * capacity * kv_dim` floats (crates/aienos-infer/src/decode.rs:50-62, VERIFIED), `#![no_std]` (lib.rs:1, VERIFIED). 4096 positions of Llama-3.2-1B is about 256 MiB of f32 KV (aienos-infer-kernel/src/session.rs:74, VERIFIED comment).
20. The kernel scheduler (`core/sched.h`) is run-queue bookkeeping, not preemptive threads with blocking I/O (VERIFIED by reading sched.h:1-60). There is no asynchronous disk completion path in native.

### LMCache (reference, Apache-2.0)

21. `TokenHasher`: rolling prefix hash, `blake3(prefix_hash || tokens as big-endian u32)`, `none_hash` for the first chunk, fixed `chunk_size` (lmcache/v1/multiprocess/token_hasher.py:30-52, 60-70, 183-190, VERIFIED).
22. Dynamic adapter: write to `<final>.tmp.<uuid>` then atomic rename; lookup on miss checks disk and lazily populates the index; capacity counted in `_total_bytes`; `_inflight_stores` dedups same-key stores; delete skips pinned objects (nixl_store_dynamic_l2_adapter.py:4-22, 139-144, 370-399, 420-470, 486-520, VERIFIED). Its weakness, named in #484: capacity counts only objects already discovered (`_execute_lookup_in_the_loop` adds `_total_bytes` on lazy hit, line 503-511, VERIFIED), so an unseen file is free until found.
23. FS adapter: reversible filename from the key, `.tmp` suffix + `os.replace`, unlink of the temp on failure; no `fsync` anywhere in the file (fs_l2_adapter.py:99-135, 518-532, 640-662, VERIFIED; absence of `fsync` VERIFIED by grep).
24. Tests adapted (test_nixl_store_dynamic_l2_adapter.py): `test_salted_objects_recover_independently_after_restart` (745), `test_store_rejected_when_capacity_exceeded` (985), `test_batched_store_reports_failure_when_capacity_exceeded` (1034), `test_secondary_lookup_finds_key` (1113), `test_secondary_lookup_and_load_data` (1138), `test_load_partial_failure_marks_only_successful` (561), `test_delete_skips_pinned_key` (887). Each is re-expressed as a Rust test; none is copied. Attribution: an Apache-2.0 header on `crates/aien-kv-store/src/identity.rs` (the prefix-chain idea) and the test file that adapts the cases, plus a THIRD_PARTY.md row ("LMCache, Apache-2.0, ideas adapted: deterministic rolling prefix keys, completed-object publication by temp+rename, lazy disk recovery; no code copied"). The LMCache clone has a LICENSE and no NOTICE file (VERIFIED by `ls`).

### Prior art in this repository (not depended on)

25. OWN-30B serve branch `own30b/serve` (sc-own30b, unmerged, pinned at commit 69459db): in-memory prefix reuse for one conversation, `reused = common.min(prompt.len() - 1)` so at least the last prompt token is always recomputed to produce logits (crates/aien-inference-abi/src/qwen3_moe_omega.rs:2163 at 69459db, VERIFIED), test `prefix_reuse_prefills_only_the_new_tokens_and_matches_a_cold_run` (fn at 2421 at 69459db, VERIFIED). Line numbers move on that branch; the pin is the commit. This ADR adopts the same "never restore the last prompt token" rule (section 6.5) and the same warm-equals-cold test shape. Nothing else overlaps: that branch has no disk, no identity, no sharing across requests.

## Decision drivers

1. Correctness before speed: no byte is reused unless its identity proves compatibility, and validation happens before any allocation or copy.
2. The runtime mutex and the KV lock are never held across storage I/O.
3. Everything is bounded: queue depth, bytes in flight, recovery memory, disk quota including objects never looked up.
4. One record header for host and native, so tooling and tests read both; payloads differ by backend format and are never cross-loaded.
5. The native side keeps the single-command, bounce-buffered NVMe contract and the freestanding no_std rule.
6. Conversation history stays authoritative; the KV store is a cache that may be dropped at any time.

## 1. Versioned chunk identity

### 1.1 Units

- A **chunk** is `chunk_tokens = chunk_blocks * block_size` consecutive positions of one sequence, aligned to a multiple of `chunk_tokens`. `chunk_blocks` is a setting fixed into the namespace (default 16 blocks; with the 16-token blocks of `model_dir.rs` that is 256 tokens (calc:ceaa1589), LMCache's default). A chunk is **full** when it holds `chunk_tokens` positions. The last chunk of a parked sequence may be **partial** (`token_count < chunk_tokens`); partial chunks are keyed with their exact token count and therefore only ever match an identical prefix of identical length. Only full chunks are prefix-share candidates.
- Hash everywhere is SHA-256 (host: `sha2`, already in the lockfile; native: argus `sha256.h`, used by `store_v1.h`). blake3 is not used because native has no blake3.

### 1.2 Namespace id (binds everything that must be equal for bytes to be reusable)

`namespace_id = SHA-256( "AIEN-PKV-NS-V2\0" || fields )` (V1 until sc#504; the domain moved when field 14 started to bind the engine, section 1.2.2, so no V1 namespace can match) where `fields`, in this order, little-endian:

| # | field | type | source |
|---|---|---|---|
| 1 | format_version | u16 = 1 | this ADR |
| 2 | model_sha256 | [u8;32] | `ModelIdentity.model_sha256` decoded from hex (generation.rs:25) |
| 3 | model_digest_kind | u8 len + bytes | `ModelIdentity.model_digest_kind` (generation.rs:26) |
| 4 | model_id | u16 len + bytes | `ModelConfig.model_id` |
| 5 | num_layers, num_kv_heads, head_dim, block_size | 4 x u32 | `ModelConfig` |
| 6 | rope_theta | u32 (f32 bits) | `ModelConfig.rope_theta` |
| 7 | rope_scaling | u8 tag (0 none, 1 llama3) then, if 1: factor f64 bits, low_freq_factor f64 bits, high_freq_factor f64 bits, original_max_position_embeddings u64 | `Llama3RopeScaling` (abi-core lib.rs:45-50) |
| 8 | qk_norm | u8 | `ModelConfig.qk_norm` |
| 9 | dtype | u8 (1 Fp4, 2 Fp8, 3 Bf16, 4 Fp16, 5 Fp32) | `KvDType` |
| 10 | layout | 5 x u64 strides (head, token, kv_plane, layer, block) + u64 element_bytes | `KvLayout` (lib.rs:143-156) |
| 11 | backend_format | u8 (1 = host paged pool v1; 2 = native f32 decode state v1) | section 2.3 |
| 12 | chunk_tokens | u32 | setting |
| 13 | sharing_namespace | u16 len + bytes | setting `AIEN_KV_STORE_NAMESPACE` (host); sealed Store uuid + identity class + subject id (native, section 9) |
| 14 | compute_config_digest | [u8;32] | `ComputeConfig::digest_with_engine(engine)`, sections 1.2.1 and 1.2.2 (added by the review, C1 delta in section 12; binds the engine since sc#504) |

Rationale: fields 2-8 cover "which weights, which geometry, which RoPE"; 9-11 cover "which bytes"; 13 is the authorization scope; 14 covers every other number that changes what the forward pass computes. Two runtimes with different `block_size` or `chunk_tokens` cannot share objects, by design: positions and chunk boundaries are part of identity.

#### 1.2.1 Compute configuration digest (review finding 1)

The review showed (VERIFIED by the orchestrator) that two model directories with the same weights digest and the same fields 1-13 but a different `rms_norm_eps` in `config.json` produce different K/V: the backend reads `weights.config.rms_norm_eps` for every RMSNorm (transformer_backend.rs:738, 1111, 1391, 1760) and the value comes from `config.json` (model_dir.rs:112, 239). The weights digest covers the safetensors bytes only, not `config.json`.

`ComputeConfig::digest() = SHA-256("AIEN-PKV-CFG-V1\0" || ComputeConfig)` (namespace field 14 until sc#504; field 14 is now the V2 digest of section 1.2.2, which hashes the same nine fields first) where `ComputeConfig` is, little-endian and in this order:

| # | field | type | source |
|---|---|---|---|
| 1 | num_heads | u32 | `ModelConfig.num_heads` |
| 2 | hidden_dim | u64 | `ModelConfig.hidden_dim` |
| 3 | intermediate_dim | u64 | `ModelConfig.intermediate_dim` |
| 4 | vocab_size | u64 | `ModelConfig.vocab_size` |
| 5 | rms_norm_eps | u32 (f32 bits) | `ModelConfig.rms_norm_eps` |
| 6 | tie_word_embeddings | u8 | `ModelConfig.tie_word_embeddings` |
| 7 | hidden_act | u8 (1 = silu) | `config.json` `hidden_act`; the loader refuses everything but silu (model_dir.rs:198-199, VERIFIED), so the byte is a constant today and exists so a future activation cannot alias |
| 8 | arith_id | u8 len + bytes | `TensorBackend::name()` (backend.rs:131, VERIFIED trait method; implementations return `"ReferenceCpuBackend"` at backend.rs:509-511, `"NativeCpuBackend (Rayon Multi-Core)"` at mojo_backend.rs:242-244, and an availability-dependent string at omega_backend.rs:815-820, all VERIFIED) |
| 9 | arith_epoch | u32 | a constant `KV_ARITH_EPOCH` in `aien-inference-abi`, bumped by any change that alters numeric results of the forward pass |

Fields not listed and why (each is a decision, not an omission): `model_id` (field 4 of the namespace), `num_layers`, `num_kv_heads`, `head_dim`, `block_size`, `rope_theta`, `rope_scaling`, `qk_norm` (namespace fields 5-8), `max_sequence_length` (caps context, does not change a K/V value; the cap is applied to the restorable prefix, section 6.3), `eos_token_ids` (decides when to stop, not what is computed), `attention_bias` and `mlp_bias` (the loader refuses true, model_dir.rs:201-203, so they cannot vary).

**Why the resolved `ModelConfig` and not a digest of the raw `config.json`** (INFERRED, decision): (a) the raw file carries fields that cannot change a K/V value (`torch_dtype`, `transformers_version`, `use_cache`, `initializer_range`; model_dir.rs:22-52 lists them for Qwen3), so a raw digest would orphan every cached object on a harmless edit; (b) JSON has no canonical form (key order, `1e-05` versus `0.00001`, duplicate keys), and a hand-written canonicalizer is an attack and bug surface; (c) serde defaults (`default_rms_norm_eps`, `default_rope_theta`) mean the effective value is the resolved one, not the one in the file. The omission risk of an explicit list is closed mechanically: the conversion `ComputeConfig::from(&ModelConfig)` lives in `aien-runtime` and destructures `ModelConfig` **exhaustively, with no `..`**, so adding a field to `ModelConfig` is a compile error until its author decides "bound" or "excluded with a reason". A test (C6, PLANNED) additionally mutates every field of `ModelConfig` one at a time and asserts the namespace id changes for every field not in the written exclusion list.

`arith_epoch` is guarded by a pinned-bytes test (C6, PLANNED): a fixed tiny model and prompt run on the reference backend; the SHA-256 of the resulting K/V pool bytes is pinned in the repository. A numeric change fails the test until the author bumps `KV_ARITH_EPOCH` and re-pins. `arith_id` is a coarse label (VERIFIED that the Omega name changes with availability, omega_backend.rs:815-820); mixing backends is therefore refused by identity, and the greedy guarantee of section 6.5 is stated per backend.

#### 1.2.2 Engine identity (sc#504)

The same weights digest and the same `ComputeConfig` do not fix the K/V bytes when one checkpoint directory can be run by different engines and builds (research/pkv-omega-engine.md section 4, VERIFIED there): the omega engine loads one directory as bf16, fp8 experts or fp8 everywhere (same file hash), its kernels come from a separate build, `KV_ARITH_EPOCH` is pinned only on the CPU reference, and whole-prompt prefill changes activation rounding. So field 14 binds the engine too:

`compute_config_digest = SHA-256("AIEN-PKV-CFG-V2\0" || ComputeConfig fields 1-9 (section 1.2.1, same encoding) || weight_format u8 || engine_build_id u8 len + bytes || rounding_class u8 len + bytes || engine_config_digest [u8;32])`

`engine_config_digest = SHA-256("AIEN-PKV-ENG-V1\0" || num_experts u32 || experts_per_token u32 || expert_intermediate_dim u64 || whole_prompt_prefill u8 || device_router u8 || matmul_path u8 len + bytes)` (`EngineConfig::digest()`; an engine may supply its own 32-byte digest over its MoE shape and run switches instead).

| field | type | what the caller must supply |
|---|---|---|
| weight_format | u8 (1 as-file, 2 bf16, 3 fp8-experts, 4 fp8-all) | the format the weights are computed in after load, not the file format |
| engine_build_id | 1-255 bytes | engine plus kernel build, for example omega commit + linked kernel archive digest |
| rounding_class | 1-255 bytes | numeric rounding class of the arithmetic |
| engine_config_digest | [u8;32] | MoE shape and numeric run switches (whole-prompt prefill, device router, GEMM or GEMV path) |

Rules: `EngineIdentity` has no default; `ModelBinding::namespace_fields` and the daemon's `ModelKvBinding` require it, so every caller that enables persistence states each value. Empty `engine_build_id`, `rounding_class` or `matmul_path` is refused (`KvError::FieldEmpty`). `ComputeConfig::digest()` stays as the compute-only V1 digest and is no longer field 14. The host `TensorBackend` path fills it with `HostEngineIdentity::reference(arith_id)` (aien-runtime `compute_config.rs`): weight format `as-file`, build `aien-runtime/<version>+git:<commit>` (commit from `build.rs`, `AIEN_BUILD_COMMIT` override, `unknown` without git), rounding class `<arith_id>@kv-arith-epoch-<KV_ARITH_EPOCH>`, engine config dense with every switch off and matmul path `host-tensor-backend`. The omega engine builds its own identity (sc#505). Tests: `each_engine_field_alone_changes_the_namespace_and_refuses_old_records`, `v1_namespace_domain_no_longer_matches`, `field_14_binds_compute_config_then_engine_identity` (kv-store) and the four `*_change_is_never_served_after_restart` restart tests (aien-runtime).

### 1.3 Chunk key (prefix chaining, adapted from LMCache TokenHasher)

```
chunk_key_0 = SHA-256("AIEN-PKV-CHUNK-V1\0" || namespace_id || [0u8;32]     || first_position u32 || token_count u32 || tokens[0..token_count] as u32le)
chunk_key_i = SHA-256("AIEN-PKV-CHUNK-V1\0" || namespace_id || chunk_key_{i-1} || first_position u32 || token_count u32 || tokens[...] as u32le)
```

`first_position = i * chunk_tokens`. The parent key makes a chunk depend on every earlier token, so a chunk can only be found by walking the chain from position 0; the explicit `first_position` makes a position mismatch impossible even if a caller computes the chain wrongly. The RoPE dependence of K is covered because position, theta and scaling are all in the key.

### 1.4 Validation order (before any allocation or copy)

On lookup: compute keys from tokens (pure, no I/O), consult the in-memory index. On load, after the header region is read (4096 bytes, one aligned read; section 2.1) and before a destination block is allocated: magic, format_version, header_sha256, `namespace_id == ours`, `chunk_key == requested`, `parent_key == expected`, `first_position`, `token_count` in `1..=chunk_tokens`, geometry fields equal to the server's own, `payload_len == expected_blocks * block_stride_bytes`, the header region's padding is all zero, file size equals `4096 + payload_len`. Only then are destination blocks allocated and the payload read; each segment's SHA-256 is checked against the segment digest table as the segment lands (section 15.4), and the root over the table must equal `payload_sha256`; on the host this happens over the destination blocks before the table is published (section 5.4). A failing segment or root hash deletes the object and reports a miss. A header that fails validation is counted and the object is deleted (it is a cache). Nothing is ever partially published.

## 2. On-disk record format (shared header; payload per backend)

### 2.1 Header (256 bytes, little-endian, identical on host and native)

| offset | size | field |
|---|---|---|
| 0 | 8 | magic `AIENKVC1` |
| 8 | 2 | format_version = 1 |
| 10 | 2 | record_kind = 1 (chunk) |
| 12 | 4 | header_len = 256 |
| 16 | 32 | namespace_id |
| 48 | 32 | chunk_key |
| 80 | 32 | parent_key (zero for the first chunk) |
| 112 | 32 | model_sha256 |
| 144 | 4 | first_position |
| 148 | 4 | token_count |
| 152 | 4 | chunk_tokens |
| 156 | 4 | block_size |
| 160 | 4 | num_layers |
| 164 | 4 | num_kv_heads |
| 168 | 4 | head_dim |
| 172 | 1 | dtype |
| 173 | 1 | backend_format |
| 174 | 2 | flags (bit 0 partial_tail; others zero) |
| 176 | 8 | payload_len |
| 184 | 32 | payload_sha256 |
| 216 | 8 | written_unix_ns (informational, not identity) |
| 224 | 32 | header_sha256 = SHA-256(bytes 0..224) |

**Header bytes unchanged by the review, record layout changed (C1 delta, section 12).** Offsets 0 to 255 are all assigned, so there is no room for a new field and none is needed for finding 1: `namespace_id` at offset 16 already commits to `compute_config_digest` (section 1.2.1). Two meanings change and one rule is added:

1. **Header region.** A record file is `header region (4096 bytes) || payload (payload_len bytes)`. Bytes 0 to 255 are the table above. Bytes 256 to `256 + 32 * n_seg - 1` are the **segment digest table**: `SHA-256` of each payload segment in order (section 15.4). Every remaining byte up to 4095 is zero and a nonzero byte is a validation failure (`BadPadding`). The payload starts at file offset 4096 (`PAYLOAD_OFFSET`; native slots do not use the file layout), so a pool block can be read straight from the file with `O_DIRECT` (section 15.3). File size is `4096 + payload_len`, not `256 + payload_len`.
2. **`payload_sha256` (offset 184)** is the root `SHA-256("AIEN-PKV-PAYLOAD-V1\0" || seg_digest_0 || ... || seg_digest_{n-1})`, not a flat hash of the payload bytes. Integrity is not weakened: every payload byte is covered by exactly one segment digest, and the root covers every digest (section 15.4 argues this).
3. **Segment size** is derived, not stored: host = `block_stride_bytes` (one pool block); native = the whole payload (one segment, so no stored table; the AEAD tags already authenticate every 4080-byte chunk, section 2.5). On the host `n_seg = ceil(payload_len / block_stride_bytes)` must be at most 120, so `chunk_blocks` is capped at 120 on the host (calc:12f9efaa: `(4096 - 256) / 32 = 120`).
4. `header_len` (offset 12) stays 256: it is the length of the structure, not of the header region.

### 2.2 Host payload (backend_format 1)

`ceil(token_count / block_size)` blocks, each exactly `block_stride_bytes`, in the pool's own layout (context item 2): `[layer][K plane | V plane][token][head][dim]` bf16. Positions beyond `token_count` in the last block are zero **in the object**, and the writer enforces it: the pool does not (a block recycled from a finished sequence keeps its old rows, and the last position of a finished decode sequence has no K/V at all, context item 8). Full blocks are written straight from the pinned pool blocks; the one partial last block goes through a one-block scratch copy in which, for every layer and for K and V, the token rows at index `token_count mod block_size` and above are zeroed. Without that, a restore could expose rows another conversation left in a recycled block. The reader writes straight into freshly allocated pool blocks. No reshaping, so a load is a sequence of `block_stride_bytes` transfers.

### 2.3 Native payload (backend_format 2)

For each layer `l` in `0..num_layers`: K rows for positions `first..first+token_count` (`token_count * kv_dim` f32), then V rows likewise (decode.rs:56-62 holds `k` and `v` as `n_layer * capacity * kv_dim` floats, so the source stride per layer is `capacity * kv_dim` and the chunk is a strided gather). Native and host payloads differ (f32 strided-by-layer versus bf16 paged blocks) and `backend_format` is in the namespace, so one side can never load the other's object. Byte compatibility across host and native is **not** a goal of this milestone: the native model path is a different model file (GGUF Q4_K_M, infer.c) and different arithmetic; its K/V bytes are not the host's. What is shared is the header, the key derivation, the validation order and the test vectors for both.

### 2.4 Host object layout on the filesystem

```
<root>/                      AIEN_KV_STORE_DIR
  objects/<aa>/<64-hex>.kvc  published chunks, sharded by the first byte of chunk_key
  tmp/<64-hex>.<pid>.<job>.part   in-progress writes (never read by anyone)
  STORE_ID                   16 random bytes written once at first open (recovery sanity)
```

**Directory creation (review finding 10).** All 256 shard directories are created at first open, not lazily: `mkdir` each `objects/<aa>`, then `fsync(objects)`, write `STORE_ID` to a temp name, `fsync` it, `rename`, `fsync(root)`. After that a publication never creates a directory, so no parent-directory durability is owed on the hot path. A store that finds a missing shard directory at open recreates it with the same `mkdir`, `fsync(parent)` pair; one that cannot opens `Failed`.

**Publication** (adapted from LMCache dynamic adapter, with the fsync it lacks), with the five fault boundaries named because each is a PLANNED fault-injection test (C4):

| step | boundary | if it fails | what exists afterwards | outcome reported |
|---|---|---|---|---|
| P1 | write header region + payload to `tmp/...part` | write error / ENOSPC | partial temp, unlinked by the worker | job `Err`, nothing published |
| P2 | `fsync(file)` | fsync error | temp, unlinked | job `Err`, nothing published; adapter enters `Degraded` (Linux may drop dirty pages after a failed fsync, so the same fd is never retried) |
| P3 | `rename` to `objects/aa/<key>.kvc` | rename error | temp, unlinked | job `Err`, nothing published |
| P4 | `fsync(objects/aa)` | fsync error **after a successful rename** | a complete, valid object under its final name; durability across power loss unknown | **`Published` with `PublishOutcome::Indeterminate`**: the object is indexed and served, the job result is `Err(HardwareFault)` for the durability claim only, the adapter enters `Degraded` (no new stores until reopen) |
| P5 | index insert | cannot fail (capacity was reserved at submit, section 7.1) | n/a | n/a |

The job reports **Published** only after P4 returned success, in which case Published and Durable coincide on the host (section 3.3). In the P4 failure case the caller is told the truth: the object exists, is complete and verified, and may not survive power loss. Recovery after a restart finds the object and indexes it like any other (it passed P1 to P3, so it is whole). This replaces the earlier claim that a failed job always leaves nothing visible.

### 2.5 Native slot layout (slab region)

The KV region is a dedicated contiguous range of 4096-byte units, outside the sealed Store region and its anchor, bound through a `kv_disk_bind(disk, base_lba, units, store_range, anchor_range)` that refuses any overlap with the Store or anchor ranges (same refusal style as `st_disk_bind`, store_engine.h:90-92). On the Pi the range is inside the AIENOS partition view (section 16.2), because the kernel writes nothing outside that view (dev/disk_part.h:1-27, VERIFIED). It holds `N` fixed slots of `slot_units` each. `slot_units` is fixed at format time and never depends on the record being written (review 2, new finding 1): `slot_units = 1 + max_sealed_units`, with `max_sealed_units` the `sealed_units` of the largest payload the namespace can produce. Each record carries its own `sealed_units = ceil(sealed_len / 4096)`, `sealed_len = payload_len + 16 * ceil(payload_len / 4080)` (the M5 envelope without its 64-byte header), and is refused unless `sealed_units <= slot_units - 1` (implemented in aienos `native/kv/kv_slab.h:87` and the decode check of `native/kv/README.md:86`, VERIFIED at aienos-pkv 8e34b63). Slot `s`, complete wire layout (review finding 12), little-endian:

```
unit 0 (slot header unit, 4096 bytes)
  0    256  record header (section 2.1)
  256   64  M5 envelope header, exactly as m5.h:160-169 lays it out (magic "AIENENV1", version 1, suite 1, flags 0, chunk_size = 4080,
            total_plaintext_len = payload_len, envelope_id[16], nonce_prefix[8], key_epoch u64, reserved 0)
  320    8  slot_incarnation u64
  328    4  sealed_units u32
  332    4  envelope_len u32  (= 64 + sealed_len, must equal m5_envelope_len(payload_len, 4080), m5.h:186)
  336    4  slot_index u32    (must equal the slot's position in the slab)
  340 3724  zero (a nonzero byte makes the slot empty)
  4064  32  unit_sha256 = SHA-256(bytes 0..4064)
units 1..sealed_units
  the M5 envelope chunks, each (plaintext_n + 16 tag) bytes, back to back, chunk_size 4080 so every chunk but the last is exactly 4096 bytes = one unit;
  the last chunk is (n_last + 16) bytes followed by zero padding to the unit boundary (nonzero padding makes the slot empty)
```

Exact bounds: `payload_len` in `1 ..= 16711680` (`M5_ENV_MAX_CHUNKS` 4096 times 4080, m5.h:164, VERIFIED comment "C-only cap"; calc:76bfb969). Native has one segment (the whole payload), so there is no stored digest table on native: `payload_sha256` is `SHA-256("AIEN-PKV-PAYLOAD-V1\0" || SHA-256(plaintext payload))`, recomputed after the tag-verified decrypt and compared with the header.

**Authenticated mapping from the KV identity into M5 inputs.** `m5_object_binding` (m5.h:61-68) has no namespace or chunk-key fields, so the mapping goes through the envelope id, which is in every chunk's AAD (m5.h:164-168) and in the per-object key derivation (m5.h:70-73), the same way the Store does it (store_sealed.c:443-455, VERIFIED):

- `k_kv = HKDF-Expand(k_domain, "AIENOS-PKV-SLOT-KEY-V1\0", 32)` and `k_envid_kv = HKDF-Expand(k_domain, "AIENOS-PKV-ENVID-KEY-V1\0", 32)` (`m5_hkdf_expand`, m5.h:36).
- `envelope_id[16] || nonce_prefix[8]` = first 24 bytes of `HMAC-SHA256(k_envid_kv, "AIENOS-PKV-ENVID-V1\0" || store_uuid || namespace_id || chunk_key || slot_index u32 || slot_incarnation u64 || key_generation u64 || identity_class u8 || payload_len u64 || payload_sha256)`.
- `m5_object_binding`: `store_uuid` = the sealed Store uuid, `object_kind` = `0x0520` (a grep of the aienos tree finds no other use of `0x052x`, VERIFIED; the Store uses 0x0510 and 0x0511, store_sealed.h:43-44), `object_version` = 1, `envelope_id` as above, `key_generation` = the Store's current key generation, `identity_class` = the Store's class.
- Open recomputes the binding from the header fields it validated; a slot copied under another header, another slot index, another namespace or another chunk key fails `M5_ERR_BINDING` or `M5_ERR_AUTH` before any byte reaches the session. Because the plaintext hash is inside the envelope id, a rewritten slot with different bytes never repeats a (key, nonce) pair, so `slot_incarnation` is not load-bearing for nonce safety (INFERRED from the Store's own argument at store_sealed.c:450-455); it identifies the write instance for stale-completion checks and is recovered as `1 + max(valid incarnations seen)`.

Write order per slot (section 3.4 gives the states): zero the header unit, flush (the slot is now provably empty), write payload units, flush, write the header unit, flush. A header whose `unit_sha256` or record header fails is an empty slot. Recovery reads `N` header units and nothing else (section 4.2). No catalog, no journal, no generation counter: the slab is a cache and claims nothing about history. Why not Store v1: it is append-only with no reclaim (context 16), so a cache in it could only grow or be reformatted; and the sealed Store caps objects at 16 KiB (context 17) while one native chunk of 64 positions of Llama-3.2-1B is 65536 bytes per position (16 layers x 8 kv heads x 64 dims x K and V x 4 bytes, calc:e2ace192) x 64 = 4 MiB (calc:3df1c283). The slab reuses `disk_dev`, `disk_queue`, the torn-write discipline of writing a self-checking header last, and M5 chunk sealing. It is not a competing identity or journal: it stores nothing authoritative.

## 3. Storage contract (`KvStore` trait)

### 3.1 Why not `aien_platform::ObjectStore`

`ObjectStore::commit` is synchronous and takes a borrowed slice (objects.rs:17-20). A KV store must (a) submit without blocking the caller, (b) read and write pool memory in place without a copy, (c) report four distinct completion stages, (d) enforce capacity and (e) support cancellation. The new trait lives in a new crate `aien-kv-store` (`no_std` + `alloc` for the identity, codec and validation modules; the host adapter behind a `std` feature) and reuses `aien_platform::{ObjectId, PlatformError}`: `ObjectId(pub [u8;32])` carries a `chunk_key`, and every error is a `PlatformError` variant (`QueueFull`, `OutOfMemory` for quota, `ObjectNotFound`, `HardwareFault(String)` for I/O, `InvalidLayout` for validation).

### 3.2 Trait sketch (Rust, binding shape; names final, bodies not)

```rust
pub struct ChunkKey(pub aien_platform::ObjectId);
pub struct NamespaceId(pub [u8; 32]);
pub struct JobId(pub u64);                // never reused within a store lifetime
pub struct Incarnation(pub u64);          // caller's token, echoed in every completion

/// A read-only view of pinned pool blocks; the guard keeps the storage reference
/// (section 5.2) alive until dropped. Dropping it before completion is a bug the
/// worker detects (the job fails with `PlatformError::InvalidLayout`).
pub struct PinnedSource { pub base: *const u8, pub block_bytes: usize, pub blocks: Vec<usize>, guard: HolderGuard }
pub struct PinnedDest   { pub base: *mut u8,   pub block_bytes: usize, pub blocks: Vec<usize>, guard: HolderGuard }

pub struct ChunkHeader { /* exactly the fields of section 2.1 minus the two hashes */ }

pub enum JobStage { Submitted, Completed, Published, Durable }
/// How a store job ended with respect to durability (section 2.4, P4).
pub enum PublishOutcome { NotApplicable, Durable, Indeterminate }
pub enum StoreState { Recovering, Open, Degraded, Failed }
/// Exactly one terminal completion is produced for every accepted job (done, failed or cancelled).
pub enum Terminal { Done(JobStage), Failed(aien_platform::PlatformError), Cancelled }
pub struct Completion { pub job: JobId, pub incarnation: Incarnation, pub key: ChunkKey, pub stage: JobStage,
                        pub terminal: Option<Terminal>, pub publish: PublishOutcome,
                        pub bytes: u64, pub elapsed_us: u64 }

pub struct Capacity { pub state: StoreState,
                      pub quota_bytes: u64, pub used_bytes: u64, pub reserved_bytes: u64, pub evicting_bytes: u64,
                      pub unindexed_bytes: u64, pub unindexed_objects: u64,
                      pub objects: u64, pub max_objects: u64,
                      pub restore_ops: u32, pub max_restore_ops: u32, pub park_ops: u32, pub max_park_ops: u32,
                      pub inflight_bytes: u64, pub max_inflight_bytes: u64, pub max_waiters_per_job: u32 }

pub struct RecoveryProgress { pub done: bool, pub indexed: u64, pub bytes: u64, pub unindexed_objects: u64, pub unindexed_bytes: u64,
                              pub foreign_namespace: u64, pub corrupt_deleted: u64, pub tmp_deleted: u64, pub elapsed_us: u64 }

pub struct DrainReport { pub published: u64, pub cancelled: u64, pub unfinished: Vec<UnfinishedJob> }

pub trait KvStore: Send + Sync {
    fn namespace(&self) -> NamespaceId;
    fn capacity(&self) -> Capacity;
    /// Index only, no I/O, O(keys). `Some(len)` means a published object exists. While `state` is
    /// `Recovering` every key is a miss.
    fn lookup(&self, keys: &[ChunkKey]) -> Vec<Option<u64>>;
    /// Bounded and non-blocking, never evicts and never touches the filesystem. Reserves
    /// `payload + 4096` bytes of quota (section 7.4); refuses with QueueFull when the class's ops or the bytes
    /// would exceed their limits, OutOfMemory when used + reserved + request exceeds quota plus the
    /// bytes that are evictable (unpinned, not in flight). Reclamation happens on the worker (section 7.4).
    fn submit_store(&self, header: ChunkHeader, src: PinnedSource, inc: Incarnation) -> Result<JobId, PlatformError>;
    fn submit_load(&self, key: ChunkKey, expect: &ChunkHeader, dst: PinnedDest, inc: Incarnation) -> Result<JobId, PlatformError>;
    /// Durability barrier for everything Published before the call.
    fn submit_flush(&self, inc: Incarnation) -> Result<JobId, PlatformError>;
    /// Coalescing: register one more waiter on an in-flight load; QueueFull past `max_waiters_per_job`.
    fn attach_waiter(&self, job: JobId) -> Result<(), PlatformError>;
    /// Non-blocking: drains at most `max` completions. Completion capacity is reserved at submit
    /// (three slots per job), so the worker never blocks or drops on a full queue.
    fn poll(&self, max: usize) -> Vec<Completion>;
    /// Cancelled: the job never touched the destination/published nothing. TooLate: it will complete; ignore by incarnation.
    /// Either way the job later yields its one terminal completion; the destination is safe to free
    /// only after that completion has been polled (section 5.5).
    fn cancel(&self, job: JobId) -> CancelOutcome;
    /// Pins against eviction while a restore chain is being assembled (bounded by max_restore_ops).
    fn pin(&self, keys: &[ChunkKey]) -> Result<PinToken, PlatformError>;
    fn unpin(&self, token: PinToken);
    /// Marks keys for deletion; skips pinned and in-flight keys; the unlink runs on the worker.
    fn delete(&self, keys: &[ChunkKey]) -> u64;
    /// Resumable recovery (section 4): call until `done`, each call bounded by `slice_ms`. Admission of
    /// stores stays closed until `done`.
    fn recover_step(&self, slice_ms: u64) -> RecoveryProgress;
    /// Stops intake and drains or cancels until `intake_deadline_ms`. Never frees a buffer an I/O may
    /// still use; jobs still running at the deadline are listed in `unfinished` (section 7.5).
    fn shutdown(&self, intake_deadline_ms: u64) -> DrainReport;
}
```

A **deterministic fake adapter** implements the same trait in memory with scripted latencies and faults (slow I/O, queue-full, disk-full, flush failure, torn write, stale completion, rename-ok-dir-fsync-fail, stalled fsync, stalled eviction) so #485 develops against it before the host adapter lands (cuts C3, C5 in CUTS.md).

### 3.3 Stage semantics

| stage | host adapter | native adapter | what the caller may do |
|---|---|---|---|
| Submitted | job accepted; `inflight_ops += 1`, `inflight_bytes += payload_len`; source blocks are pinned | same; bounce buffer not yet touched | nothing; may cancel |
| Completed | bytes written to `tmp/...part` and `fsync(file)` returned | payload units written and `flush` returned | still not visible to lookup; may cancel (temp is unlinked / header stays zero) |
| Published | `rename` returned and `fsync(dir)` returned (if `fsync(dir)` fails after the rename: Published with `PublishOutcome::Indeterminate`, section 2.4 P4); index updated; visible to lookup and to recovery after restart | header unit written; index updated; visible to lookup; **not yet durable** | release the source pins; too late to cancel |
| Durable | same instant as Published (reported as one completion carrying `Durable`) | `flush` after the header returned | claim persistence |

A load reports `Completed` (terminal for the store) when every segment landed and verified. The caller's table commit under the KV lock (section 5.4) is the caller's step, not a store stage; `Durable` is not a load stage. Every accepted job, store or load, yields exactly one terminal completion (done, failed or cancelled); a caller that cancels or abandons a job must still poll that terminal completion before it frees the job's memory.

### 3.4 Failure semantics

- Any failure before the host rename (P1 to P3 of section 2.4) or before the native header write leaves the store exactly as it was: no index change, no visible object, temp unlinked, slot header still zero. The last usable published object is never touched by a failing write.
- Disk full (`ENOSPC` on host, no free slot on native) is reported as `PlatformError::OutOfMemory` on submit when the quota says so, or as a Completion with `Err` when the filesystem refuses despite the quota (host). Either way existing objects stay.
- A flush failure (native `flush` error, host `fsync` error) fails the job; on native it also fail-stops the queue (disk.h:69-72) and the adapter enters `Failed` until re-opened; metrics say so.
- Host `fsync(dir)` failing after a successful rename is **indeterminate**, not failed-and-absent: the object is whole and indexed, only its survival across power loss is unknown, and the caller is told so through `PublishOutcome::Indeterminate`. This keeps the rule "a failed job published nothing" true for every case where it can be true, and names the one case where it cannot.

## 4. Recovery algorithm

### 4.1 Host

Recovery is **resumable and complete** (review finding 8). The earlier design stopped at an object or time limit and left the remainder uncounted, so deleting one indexed object could reopen admission while an arbitrarily large unscanned remainder still occupied disk. Now: the store is in state `Recovering` from open until a full pass has accounted for every file; in that state `lookup` returns misses, `submit_store` and `submit_load` refuse with `QueueFull`, and the runtime runs without persistence (plain prefill, no park). Nothing about recovery blocks daemon start: the worker calls `recover_step(slice_ms)` repeatedly (default slice 50 ms, so a step never holds the index mutex for long). The state becomes `Open` only when `done`.

1. Read or create `STORE_ID` (section 2.4).
2. Stream `tmp/`: unlink every `*.part`, count `tmp_deleted`. No list is built.
3. Stream `objects/<aa>/` for all 256 shards, a directory cursor persisted across steps. For each entry: the name must be 64 hex + `.kvc` (else `foreign_file`, left alone, not counted: AIEN did not write it). `stat` for size and allocated blocks. **Physical record bytes** = `max(st_size, st_blocks * 512)`, which includes the 4096-byte header region, not the payload alone.
   - While fewer than `max_objects` are indexed: read the 4096-byte header region (one aligned read; the 256-byte struct is validated, padding checked), validate magic, version, `header_sha256`, name == `chunk_key`, size == 4096 + `payload_len`. Invalid: unlink, `corrupt_deleted`. Valid: insert `(chunk_key, physical_bytes, mtime, namespace_id)` into the index (56 bytes per entry; `max_objects` default 65536 caps it at 3670016 bytes, about 3.5 MiB, calc:77ff91f3) and add `physical_bytes` to `used_bytes` whether or not the namespace matches the server's own. The payload is not read at recovery; it is verified segment by segment on first load, and a failing load deletes the object.
   - Past `max_objects`: the file is **counted but not indexed**: `unindexed_objects += 1`, `unindexed_bytes += physical_bytes`, `used_bytes += physical_bytes`. Memory stays constant; accounting stays complete. `Capacity.used_bytes` always includes `unindexed_bytes`.
4. When the pass reaches `done`: state `Open`. If `used_bytes + reserved_bytes` is over quota, or `unindexed_objects > 0`, the worker runs a **purge pass** (on the worker, never under the spine or KV lock): it walks shards again and unlinks unindexed files first (they cannot be served anyway), then evicts per step 6, until `used_bytes` is back under 90% of quota and the index has room. Admission of stores is allowed during the purge only against the headroom already free.
5. `delete` and eviction subtract bytes only when the `unlink` has returned on the worker. Removing an index entry never reopens admission by itself.
6. Eviction policy at runtime: LRU by `mtime` then by load time, never pinned, never in-flight, foreign namespaces first.
7. Temp files of in-flight writes are covered by `reserved_bytes` (section 7.4) from submit, so quota cannot be overcommitted by writes that have not yet become objects.

Memory bound: one 4096-byte header buffer, the index (capped), one directory cursor, and counters. Time: one `stat` per object plus one 4 KiB read per indexed object; a partly scanned store is never treated as fully accounted.

### 4.2 Native

Read the `N` slot header units in order through the bounce buffer (one command each, `N * 4 KiB`), validate each (`unit_sha256`, record header, namespace), build the index of at most `N` entries in a caller-provided workspace (static storage, like `st_workspace`). A header that fails is an empty slot. Nothing else is read. `used_bytes` counts `slot_units * 4096` physical bytes for every valid slot regardless of namespace, and the pass is complete by construction because `N` is fixed at format time (no unscanned remainder can exist, so finding 8 does not arise natively). There is no temp state to clean: a torn write leaves either a zero header (write order, section 2.5) or a header whose hash fails.

## 5. Ownership model

### 5.1 States (per sequence, in a new `KvResidency` map owned by the spine)

```
Resident --(scheduler finish transaction forks holders, S1)--> Parking{jobs, holders, inc}
Parking  --every job terminal--> holders freed (S4); published chunks live in the store
Lookup hit --restore()--> Restoring{txn: RestoreTxn, inc} --all jobs Completed + commit (S3)--> Prefilling (suffix only) --> Resident
Restoring --any job fails / cancel / stale--> Draining{txn} --every sibling job terminal--> Failed{reason} --> plain prefill --> Resident
```

`Parked` is a property of the store (objects exist), not of the runtime: this milestone frees the sequence after publication (section 6), so there is no resident-but-parked state to track. `Failed` is sticky per request until the request's recompute path has started, then it is cleared; it exists so a stale completion cannot flip a request back. `Draining` is new (review finding 5): a restore that has a failed chunk still owns its destinations until every sibling job has produced its terminal completion.

### 5.2 Storage references reuse the fork refcount

The storage worker never holds a raw block id. A **holder** is a real block table in `AienKvManager` whose id lives in the reserved slot `u32::MAX` (`SequenceId { slot: u32::MAX, generation: job_id as u32 }`, generation nonzero). Arena slot indices start at 0 and grow by one per new slot (`allocate_slot` pushes when `free_slots` is empty, sequence.rs:382-399, VERIFIED), so slot `u32::MAX` is unreachable in practice but not refused today; C3 adds the refusal in `allocate_slot` (slot index must stay below `u32::MAX`) so the reservation is a rule, not an assumption. **One holder per chunk**, both directions:

- Park: a new primitive `fork_range(seq, holder, block_range)` takes one reference on exactly the blocks of that chunk (C3). `fork_sequence(seq, holder)` is not used: it copies the parent's whole table (lib.rs:1165-1210, VERIFIED), so a per-chunk holder built with it would pin every block of the sequence.
- Restore: `allocate_sequence(holder, tokens)` allocates zeroed blocks owned by that chunk's holder.
- Assembly: `assemble_table(seq, holders)` (C3) builds the request's own table from the holders' block lists, adding one reference per block and marking the table `PrefillPending`. It refuses a `seq` that already has a table (unlike `fork_sequence`, which frees a live child id's blocks, lib.rs:1181-1184, VERIFIED) and refuses any holder that is not complete.
- Release: `free_sequence(holder)`.

Consequences, all from existing code paths:

- A block reaches `free_blocks` only when its refcount hits zero (lib.rs:1419-1425), so a slot is reused only after the sequence tables and the storage holders are all done. No new free-list logic.
- Shared-prefix owners stay valid when another owner parks or cancels: the holder is one more sharer; the existing COW tests already cover the invariants (`crates/aien-kv-cache/tests/cow_branching_tests.rs`).
- A parked sequence that resumes while its park is in flight copies on write only on a partial last block (lib.rs:1234-1255): the worker keeps reading the frozen original.
- `fork_range` leaves the parent's state alone; `fork_sequence` sets the parent to `SharedFrozen` (lib.rs:1192-1197), which is the intended meaning for a fully ready parent.

### 5.3 Incarnation checks

Every job carries `Incarnation = (seq_id u64 as issued by the arena, residency_epoch u32)`; the spine bumps `residency_epoch` on every state transition of that sequence. A completion is committed only if the sequence still exists with the same `seq_id` (its generation makes a reused slot a different id, sequence.rs:7-50) **and** the same epoch. Anything else is counted `kv_store_stale_completions` and its holder is freed after the terminal completion. A reused sequence id can therefore never be mutated by an old job.

### 5.4 Short critical sections (the only places the KV lock is taken for storage)

| name | lock | work | bound |
|---|---|---|---|
| S1 finish-fork | KV write, **inside the scheduler's finish transaction, before `free_sequence`** (section 6.1) | `fork_range(seq, holder_i, range_i)` per chunk; push a `ParkIntent` to a lock-free outbox | O(blocks) |
| S2 restore-alloc | KV write | `allocate_sequence(holder_i, tokens_i)` per chunk; refuse if `free < watermark + needed` | O(blocks) |
| S3 restore-commit | KV write, **on the spine thread** while it handles polled completions, never on the storage worker (review 2, new finding 2) | verify incarnation and that **every** job of the txn is `Completed`; `assemble_table(seq, holders)`; scheduler `admit_restored_prefix` (section 6.3) | O(blocks) |
| S4 holder-release | KV write | `free_sequence(holder_i)`, only when the txn has no non-terminal job | O(blocks) |
| S5 queue | store index mutex | push/pop/complete | O(1) |

The segment hashes on load are verified over the destination blocks *before* S3 (they are holder-owned and invisible to the scheduler until assembled), so no copy to a staging buffer is needed on the host.

### 5.5 Cancellation and restore transactions

- A **restore transaction** (`RestoreTxn`) owns all of one request's chunk holders and the list of its submitted job ids. It is the unit of ownership: no holder of a txn is freed, reused or handed to prefill while any job of the txn is non-terminal (review finding 5). When chunk A fails while chunk B is mid-transfer: the txn moves to `Draining`, `cancel(job)` is issued for every non-terminal sibling, and the txn waits for each sibling's terminal completion (`Done`, `Failed` or `Cancelled`); only then S4 frees the holders and the request goes to plain prefill. `TooLate` siblings simply finish, their bytes are discarded.
- Cancel own waiter only: a request that is cancelled (`CancelSwarm`, `Aborted`, client gone) cancels its own txn the same way. Shared work (a same-key load another request is waiting on) is never cancelled: the coalescing table keeps the job alive while any waiter remains; cancelling removes the waiter only.
- Mid-transfer cancel of a store: the worker checks the cancel flag between segment writes (bounded granularity = one block); the temp is unlinked; nothing was published.
- The worker never writes into a freed holder: the free happens in S4 only after the terminal completion of every job that targets it, so there is no use-after-free of pool blocks, including on the multi-chunk failure path.

## 6. Park and restore in the runtime

### 6.1 Boundary (review finding 2)

The earlier text parked in `AienRuntimeSpine::step` when it processed `DecodeOutput::Finished` (spine.rs:198-230). That is too late (VERIFIED by the review and by reading the scheduler): `AienScheduler::step` frees the sequence's block table inside its own finish handling, `free_sequence(request_id)` at aien-scheduler/src/lib.rs:1106 for a normal finish and at 1152 for a backend-originated `Finished`, before it returns the `Finished` output (lib.rs:1111, 1172). By the time the spine sees `Finished`, the blocks are gone; what remains there is `backend.release_sequence` (spine.rs:225), which only drops backend state.

Design: ownership transfer happens **inside the scheduler's finish transaction**. The scheduler gets an optional `FinishHook` (a trait object set by the spine; absent means today's behaviour exactly). At each of the two finish sites, for any finish reason except `Preempted`, in this order and inside one KV write guard:

1. `prefix = backend.computed_prefix(request_id)` (section 6.2), read while the backend still holds the sequence (the backend call happens before the guard is taken, the backend is not touched under the guard).
2. `hook.on_finish(&mut kv, request_id, prefix)`: `fork_range` into per-chunk holders (S1), a `ParkIntent { holders, keys, headers, incarnation }` pushed to an outbox (a `Mutex<Vec>` with O(1) push; no I/O, no store call).
3. The existing `free_sequence(request_id)`.

The spine drains the outbox right after `step` returns and calls `submit_store` for each intent; a refused submit frees that holder (S4) and counts `kv_park_failed_total`. Because the fork happens before the free, the blocks have refcount at least 1 at every instant; the window the review found does not exist. Active decode KV stays resident: nothing parks a running sequence (milestone rule). A `Preempted` finish is not parked (its K/V is incomplete by definition).

### 6.2 Park

1. **Computed prefix, not the token list** (review finding 3). The backend exposes `computed_prefix(seq_id) -> Option<Vec<u32>>`, the tokens whose K/V is actually in the pool. VERIFIED facts it rests on: decode reads `pos = tokens.len() - 1` and computes K/V for the last token (transformer_backend.rs:1776-1782), then appends the newly sampled token (2158), so after any decode step the last token in `tokens` has no K/V; after prefill, `tokens.len()` equals the cached positions and the sample is held apart in `pending_prefill_token` (transformer_backend.rs:136-143), also with no K/V. So `computed = tokens.len()` when a pending prefill token exists and no decode has run, otherwise `tokens.len() - 1`. The earlier rule "tokens plus the pending token" would have labelled an unwritten position as valid and let a longer later prompt restore it. Nothing beyond `computed` is ever hashed, keyed or written. The mapping is asserted by a red test that reads the pool and shows position `computed` is unwritten.
2. Compute the key chain for full chunks of `computed` and the partial tail (`token_count = computed mod chunk_tokens`). Skip chunks whose keys `lookup` already finds or that are in flight (same-key publication dedup).
3. S1 inside the finish transaction (section 6.1), one holder per chunk. The writer applies the tail-row zeroing of section 2.2 to the last partial block. The stores are low priority (section 7.2).
4. The sequence's own references are dropped as today; the holders keep the blocks until each job is terminal, then S4.

### 6.3 Restore

1. On `submit_request` in the spine, compute the key chain of the prompt, walk it until the first miss, cap the restorable prefix at `prompt.len() - 1` (context 25: the last prompt token is always recomputed so the logits come from this prompt). Also cap at the operator context (`AIEN_KV_CONTEXT_TOKENS`). A restorable prefix that ends inside a chunk uses only whole stored chunks; a partial-tail chunk is used only if its exact token count is the cap.
2. `pin` the keys; S2 allocates one holder per chunk; submit loads (high priority) for all chunks under one `RestoreTxn`; the request's residency becomes `Restoring`. The request stays in the scheduler's waiting queue with a new `SequencePhase::AwaitingRestore` that `build_scheduled_batch` skips: ready resident sequences keep decoding (the #485 slow-I/O proof).
3. When every job of the txn is `Completed` (and none failed): S3 runs `assemble_table(seq, holders)` and the scheduler's new `admit_restored_prefix(seq_id, restored)` (review finding 4). The existing admission cannot be reused as it was designed: for a waiting sequence whose table is ready it sets `prompt_tokens_prefilled = prompt_len` and enters decode (aien-scheduler/src/lib.rs:779-784, VERIFIED), skipping the unprocessed suffix; for a table that is not ready it frees the table and prefills from zero (lib.rs:802-806, VERIFIED). So `admit_restored_prefix` bypasses the waiting queue: it leaves the table `PrefillPending` (never `complete_prefill` at restore), sets `prompt_tokens_prefilled = restored`, `is_prefilled = false`, `phase = Prefill`, and pushes the sequence into `running_sequences`, where the existing "continuing chunked prefill" branch schedules `prompt[restored..]` (lib.rs:572-606, VERIFIED) and the existing completion fence marks the table ready only after the last chunk (lib.rs:1023-1033, VERIFIED). Because `restored <= prompt.len() - 1`, at least one token is always left, so the fence always runs and decode can never start on a restored-only table. The backend hook `restore_sequence(seq_id, tokens)` sets `SequenceState.tokens = prompt[..restored]` so the next chunk's positions line up (checks as `fork_sequence`).
4. Any failure or stale completion: txn `Draining` (section 5.5), then `Failed`, holders freed, phase `Waiting` with `prompt_tokens_prefilled = 0`: a plain prefill. No request is ever refused because of the cache; refusal exists only when history itself is unavailable, which is the generation record's job, not this design's.

### 6.4 Non-tensor state

Positions: reconstructed from chunk order. Pending prefill token: not needed, the restore ends inside the prompt and the suffix prefill produces a fresh pending token. Sampling state: `SamplingParams` come with every request (abi-core lib.rs:455-460); the sampling seed is `decode_sampling_seed(request_id, position)` with a fresh `request_id` after a restart, so **stochastic continuation is not reproducible across restart and is not claimed**; greedy continuation is. A test for stochastic continuation, if ever wanted, must pin the seed explicitly; that is outside this milestone.

### 6.5 Guarantee: greedy continuation equals the uninterrupted baseline

Holds when the restored bf16 block bytes are identical, the remaining prefill runs on the same backend with the same weights and the same `compute_config_digest` (identity guarantees the last two, section 1.2.1). The test runs a prompt cold, records the greedy tokens, parks, restarts the process, restores the prefix and compares tokens. The backend must prefill the remainder starting at the restored position with the same chunking as the cold run would have used from that position (chunked prefill is position-based, lib.rs:572-606).

## 7. Queues, bounds, fairness, drain

### 7.1 Bounds and per-class reservation (review finding 9)

Two bounded intake classes (restore, park) and one completion queue, all behind the store's index mutex. Limits are **per class**, not shared: `max_inflight_ops` (default 64) is split 48 restore and 64 park (`max_restore_ops`, `max_park_ops`; park raised from 16 after the C9 bench, section 18.4), and `max_inflight_bytes` (default 256 MiB) is split the same 3:1. A class can never consume the other's reservation, so continuous restores cannot make every park fail before the dispatcher runs. Bytes count the sum of `payload_len + 4096` of every job between Submitted and its terminal completion. `submit_*` refuses with `QueueFull`; the spine treats a refused park as "not parked this turn" (the holders are released) and a refused restore as a plain prefill. Restore destination blocks are counted against the pool: S2 refuses when `free_blocks < watermark_blocks + needed` (the scheduler's own watermark, aien-scheduler lib.rs:460-470), so restores can never starve admission. **Completion capacity** is reserved at submit: each accepted job reserves three completion slots (Completed, Published, terminal), so `poll` never overflows and the worker never blocks on a full queue. **Credit release** happens at the job's terminal completion as the store sees it, not when the caller later commits: a slow KV lock cannot hold store credits. **Bounded waiters**: at most `max_waiters_per_job` (default 16) per coalesced load and 1024 in total; the 17th request does a plain prefill instead of attaching. Holder metadata is bounded by the restore-block budget `AIEN_KV_RESTORE_MAX_BLOCKS`.

### 7.2 Fairness

Within the reserved shares, the worker dispatches by credits: 3 restores then 1 park when both queues are non-empty (restore latency matters to a waiting user; parks free resident memory and must not starve). Within a queue, FIFO by submission. The reservation of 7.1 guarantees admission fairness; the credits guarantee dispatch fairness. A PLANNED saturation test (C5) offers restores at the maximum rate for 10 000 operations and asserts parks are still admitted and dispatched at no less than their reserved share.

### 7.3 Coalescing (review finding 6)

Same-key loads coalesce at the **job and landing-buffer level**, never at the request-table level. One in-flight load per chunk key lands in that chunk's holder; later requests `attach_waiter` to the job and, on `Completed`, each request builds **its own table** with `assemble_table(seq_i, [holder_0 .. holder_k])` from the chunk holders it needs. The shared chunk blocks are immutable once `Completed` and shared by refcount; each request's suffix chunks are its own holders. A request commits (S3) only when all of its chunks are `Completed` and validated. `fork_sequence(first_seq, other_seq)` is never used for coalescing (it would share the first request's whole table and replace the other request's own). Same-key stores: `inflight_stores` set; duplicates are dropped at submit with `Ok(JobId)` of the existing job (the caller only needs "it will be published").

### 7.4 Eviction, reservation and thrash (review finding 7)

`submit_store` never evicts and never touches the filesystem; it runs under the spine's lock path and must stay O(1). Under the index mutex it only **reserves**: `reserved_bytes += payload_len + 4096`, admitted when `used + reserved + request <= quota + evictable_bytes` (evictable = unpinned, not in flight, not already evicting). If admitted but `used + reserved` is above quota, the job starts in state `Reclaiming`: the worker selects LRU victims under the index mutex (marking them `Evicting`, so `lookup` already misses them), releases the mutex, `unlink`s them, subtracts their bytes only after each unlink returns, and only when enough space is reclaimed proceeds to P1. If reclamation cannot free enough (everything left is pinned), the job ends `Failed(OutOfMemory)` and its reservation is released. Publication therefore always follows reclamation; no unlink or native slot invalidation ever runs under the spine or KV lock. Keys pinned by an assembling restore chain are never evicted, so a restore cannot be undone by the park it triggered. A sequence whose restore failed twice in a row within `AIEN_KV_RESTORE_RETRY_WINDOW_MS` is prefilled without lookup (counter per namespace+key chain head, bounded map of 1024 entries).

### 7.5 Shutdown drain (review finding 11)

Two separate promises. **Intake and drain deadline** (default 2000 ms, `shutdown(intake_deadline_ms)`): stop intake; cancel everything still `Submitted` or `Reclaiming` (temps unlinked); let jobs already past `fsync(file)` finish their rename; cancel all loads. **Eventual I/O termination** is not promised in time, because `fsync`, directory `fsync` and thread join have no bounded completion time on a stalled filesystem. Rules: (a) the worker owns an `Arc` clone of the KV manager and of every source and destination buffer, and keeps them until each job's terminal completion, so memory is never freed under a possibly running I/O (freeing it to force the deadline would violate ownership); (b) `shutdown` returns at the deadline with a `DrainReport` whose `unfinished` list names every job still running (key, bytes, stage) and the counters `kv_shutdown_unfinished_jobs` and `kv_shutdown_unfinished_bytes` are set; (c) the daemon logs the list and may exit; process exit reclaims all memory at once, and any temp left behind is deleted by the next recovery. Guarantee at exit: every job that reported `Published` is on disk or was reported `Indeterminate`; unfinished jobs are reported, never silently dropped; nothing else exists after restart (recovery deletes temps). An embedding that drops the store without exiting leaves the detached worker alive with its `Arc`s until the I/O ends.

## 8. Lock discipline

Locks, in the only allowed nesting order: spine `tokio::Mutex` (server.rs:27) -> KV `RwLock` (lib.rs:1685) -> store index `std::Mutex`. Rules:

1. No storage I/O, no `fsync`, no `unlink`, no sleep and no `poll` wait happens while the spine lock or the KV lock is held. The worker thread never takes the spine lock. `submit_store` performs no eviction (section 7.4); eviction, unlink and native slot invalidation run only on the worker.
2. S1, S2 and S3 run on the spine thread (S3 needs `&mut` scheduler access, which only the spine has: spine.rs:24, server.rs:204). The storage worker takes the KV lock only in S4, when a holder guard it owns is released, and never while holding its own state mutex or the index mutex (the worker drops released guards after its state lock, worker.rs `Shared::with`); it releases the index mutex before any I/O.
3. The spine takes the index mutex only through `submit_*`, `poll`, `cancel`, `lookup`, `attach_waiter`, which are O(1)/O(keys) and never block on I/O.
4. The KV lock is held across S1..S4 for O(blocks) bookkeeping; the existing scheduler already holds it for the same order of work (`complete_prefill`, `free_sequence`). S1 runs inside the scheduler's own finish transaction under the guard it already takes for `free_sequence`; it adds only `fork_range` and an outbox push and takes no other lock (the outbox mutex is a leaf lock held for one `Vec::push`).

## 9. Native mapping (aienos #367)

1. **Device contract kept**: every slot write and read is a sequence of `disk_write`/`disk_read` calls of at most `max_blocks_per_io` through the bounce buffer, one command at a time (disk.h:17-24, disk_nvme.h:17-19). The adapter owns no DMA memory: the caller's KV bytes are copied into the bounce buffer by the driver (bounce contract) so DMA visibility is the driver's, not the adapter's; the adapter only guarantees the source slice is unchanged until `disk_write` returns (same rule as `st_transact`, store_engine.h:146-149). Transfer splitting: a 4 MiB chunk is 32 commands of 128 KiB (calc:a81bdf6c; limit from disk_nvme.h:53-56). A faster command path is a profiled-only follow-up (section 15.7), not part of this contract.
2. **Completed vs durable**: `Completed` = payload units written and `disk_flush` returned; `Published` = header unit written; `Durable` = `disk_flush` after the header returned. Interruption between payload and header leaves a zero header (the writer zeroed and flushed it first) so recovery sees an empty slot: no partial record is ever exposed. A torn header unit is rejected by `unit_sha256`; `nvme_write_is_power_fail_atomic` is consulted at format time and recorded in the slab superblock as a documented limit, not a check.
3. **Bounded progress**: native has no asynchronous completion (context 15, 20). The adapter is therefore a **step-budgeted state machine**: `kv_slab_pump(budget_cmds)` issues at most `budget_cmds` disk commands (default 2, at most 262144 bytes = 256 KiB at the 128 KiB cap, calc:6d37d10d) and returns. Two callers (review finding 14): the between-steps hook, and an **idle pump** loop in the stage that owns the session, because `ck_infer_prompt` leaves its loop on `STEP_LAST` before its callback runs (infer.c:473-474, VERIFIED) and then consumes the request with `aienos_infer_session_cancel()` (infer.c:495-498, VERIFIED), so a park queued after the reply has no between-steps call to ride on. Tokens keep being produced while a slot fills over many steps, which is the bounded progress #367 asks for. **Limitation, reported, not hidden**: with a polled single-command driver on the same core, inference and I/O are interleaved, not overlapped. The evidence that distinguishes real interleaving from mere queueing is the per-step timestamp log: decode steps continue (token timestamps advance) while the slab command counter advances between them. True overlap would need a second core or interrupts and is out of scope.
4. **Separate quota and lifetime**: the slab region is a disjoint LBA range (section 2.5); `kv_disk_bind` refuses overlap; the slab is formatted and erased independently of the Store (`st_format` never touches it; wiping the slab never touches the Store or anchor). Cache eviction is slot overwrite inside the slab only. Existing Store/Cortex tests keep their regions and stay valid, and the explicit regression gates are named in CUTS.md (N2, N5).
5. **Encryption and authorization preserved**: the wire layout and the authenticated mapping from (namespace, chunk key, slot, incarnation, payload hash) into M5 are in section 2.5. `k_domain` exists only after the sealed Store is unlocked (store_boot.h), so a slot is ciphertext to anyone without the volume key: cache reuse cannot bypass session authorization. Decrypt follows the private scratchpad rule (AGENTS.md section 4): the tag is verified before any byte reaches the session's KV (m5_envelope.c:166-171, VERIFIED, uses a private 64 KiB scratch). The native `sharing_namespace` is the sealed Store uuid || identity class || the subject id, so a TEST-class Store never serves a PRODUCTION one.
6. **Native inference path** (review finding 13). `set_pos` alone cannot restore a session: `aienos_infer_session_prompt` resets the cache and sets `fed = 0` (session.rs:213-216, VERIFIED), and `aienos_infer_session_step` feeds `prompt[fed]` regardless of `DecodeState.pos` (session.rs:241-244, VERIFIED), so importing rows and setting only `pos` is either erased by the next prompt or repeats token zero at the restored position. New FFI in `aienos-infer-kernel` (no_std), in two parts so a failed restore can never leave a half-visible prefix:
   - `aienos_infer_session_kv_export(layer, pos0, n, out_k, out_v)` reads rows `[pos0, pos0+n)` of a layer; `aienos_infer_session_computed()` returns `DecodeState.pos`, the number of positions whose K/V exists (the last sampled token of a finished reply has none: `STEP_LAST` produces a token without running its forward pass, session.rs:253-258).
   - `aienos_infer_session_kv_import(layer, pos0, n, in_k, in_v)` writes rows only; it never touches `pos` or `fed`, so the rows are invisible until committed.
   - `aienos_infer_session_restore_commit(n, ids, n_ids)`, called **after** `aienos_infer_session_prompt` has prepared the request: all-or-nothing, it checks `s.fed == 0`, `n < prompt.len()`, `n <= kv_capacity`, `ids[..n] == prompt[..n]`, then sets `st.pos = n` and `fed = n`; the output list and `done` are untouched, logits come from the suffix steps as in a cold run. Any refusal changes nothing.
   Native **CPU** only; native GPU (#327) is not touched and not claimed.
7. **Park timing and source ownership** (review finding 14). The session's K/V buffers are reused by the next prompt (`reset` only sets `pos = 0`, decode.rs:93-96, VERIFIED), so a park job cannot keep reading them after the reply. At the park point (after `STEP_LAST`, before the consuming `aienos_infer_session_cancel`; `ck_infer_prompt` gains a `park_point` callback there) the stage **exports and seals the chunks synchronously into adapter-owned ciphertext buffers** (bounded: `KV_PARK_STAGE_BYTES`, section 10). From then on the slab job reads only those buffers; the session is free to reset, close or take the next prompt, and none of those events can change what the job writes. Jobs not terminal at a `ck_reset` simply vanish: their slots have a zero or torn header (section 2.5). The pump runs from the idle loop until the queue is empty; the stage's shutdown calls `kv_slab_drain(deadline)`.
8. **Overlap proof or limitation**: reported as limitation (item 3). Multiple outstanding commands are out of scope unless profiling justifies them (section 15.7); the pump budget is the only tuning knob and is not an acceptance claim.

## 10. Memory accounting

Reported separately, never conflated (names in METRICS_AND_SETTINGS.md):

- `kv_pool_allocated_bytes`: the pool allocation, constant after start (`KvPoolPlan.total_bytes`, shared_kv.rs:144). Freeing blocks never changes it, and the status report says so in its field name.
- `kv_pool_free_blocks`, `kv_pool_used_blocks` (existing `free_block_count`/`allocated_block_count`), `kv_pool_holder_blocks` (blocks referenced by storage holders), `kv_restore_inflight_blocks`.
- `kv_store_inflight_bytes`, `kv_store_inflight_ops` (queue occupancy), `kv_store_reserved_bytes`, `kv_store_persisted_bytes`, `kv_store_persisted_objects`, `kv_store_unindexed_bytes`, `kv_store_quota_bytes`.
- Host staging: `kv_host_tail_scratch_bytes` (the one-block scratch of section 2.2) and, when the `O_DIRECT` path is enabled, nothing else (direct I/O reads into the pool). Buffered mode adds no application buffer but uses the page cache, which is reported as `cache_state`, not as staging.
- Native staging, **each reported separately** (review finding 15): `kv_stage_export_bytes` (plaintext snapshot of the chunk being sealed), `kv_stage_cipher_bytes` (queued sealed envelopes, bounded by `KV_PARK_STAGE_BYTES`), `kv_stage_env_bytes` and `kv_stage_plain_bytes` (the contiguous envelope and the decrypt output of one restore chunk), `kv_crypto_scratch_bytes` (the private 64 KiB scratch inside `m5_envelope_open`, `scratch[M5_ENV_MAX_CHUNK]` at m5_envelope.c:170, VERIFIED, on the stack of the call), `kv_slab_bounce_bytes` (the driver's bounce size, a second copy of every transfer), `kv_stage_queued_bytes` (sum owned by queued jobs) and `kv_stage_peak_bytes` (high-water of all of the above together). The earlier text called the bounce buffer the only staging memory; that was wrong.
- Native adds `kv_slab_slots_total/used`.

## 11. Guarantee matrix

Every cell carries one label: PLANNED, TESTED(receipt), NOT CLAIMED or NOT_RUN. A TESTED receipt is the name of the test that proves the cell (host only, `pkv/integration`); `kv_restart_test.rs::guarantee_matrix_rows_are_labelled` checks the labels and that every named test exists. A cell stays PLANNED until its cut merges with a green run.

| scenario | host adapter | native adapter | claim |
|---|---|---|---|
| adapter recreation (same process, new `KvStore` over the same dir/slab) | TESTED(store_then_recreate_adapter_serves_same_bytes) (C4: recover, lookup, load, bytes equal) | PLANNED, host emulation over `disk_file` (N3) | objects published before `close` are served |
| process restart (real child process, `current_exe` pattern of `approved_crash_test.rs:80-89`) | TESTED(process_restart_rediscovers_and_restores_identical_bytes, greedy_continuation_after_restart_matches_uninterrupted_baseline) (C7: real child processes on a temp-dir host store; greedy on the CPU reference backend only) | PLANNED as emulator reboot on one image (N5) | greedy continuation equals baseline |
| host fault at each publication boundary (P1 to P5, section 2.4) | TESTED(fault_at_each_publication_boundary_leaves_consistent_outcome) (C4, `FsOps` fault seam) | NOT CLAIMED (no host publication boundaries natively; the slab write order belongs to cut N2) | consistent outcome at every boundary; `Indeterminate` reported, never silent |
| clean reboot | NOT CLAIMED on the host (no reboot test; fsync + rename + dir fsync is the standard recipe, INFERRED sufficient) | PLANNED in QEMU only (N5 is a reboot); physical Spark NOT_RUN (aienos#336) | emulator only |
| abrupt power loss | NOT CLAIMED on the host | PLANNED in emulation with the `disk_file` power-cut injector at every unit boundary (N2, same method as the store crash tests); physical NOT_RUN | no partial record is ever exposed; durability of `Published`-but-not-`Durable` slots is not claimed |
| real cold boot on hardware | NOT CLAIMED | PLANNED on a Raspberry Pi 5, labelled "native physical (Pi 5)" (cut P3, section 16); Spark-native stays NOT_RUN (aienos#336) | reboot through tryboot is a firmware restart, not a power cut |

## Consequences

- New crate `aien-kv-store` (identity, header codec, validation, trait, fake adapter, host adapter). `aien-kv-cache` gains `fork_range`, `assemble_table` and the reserved holder slot rule. `aien-scheduler` gains `SequencePhase::AwaitingRestore`, `admit_restored_prefix` and the `FinishHook`. `aien-runtime` gains the residency map, the park outbox drained after `step`, restore at submit, status fields. `aien-abi-core` gains `computed_prefix` and `restore_sequence` on the backend trait with defaults (`None` and `Err`).
- Objects are never shared across `block_size`, `chunk_tokens`, dtype, layout, backend, compute configuration or model; a model upgrade orphans the old objects, which still count toward quota until evicted (section 4.1, by design).
- The design does not add resident retention of parked chunks (an L1 of holders kept alive under low pressure). The ownership model allows it later with no format change.
- Native hosted and emulator results do not qualify physical hardware; every physical row stays NOT_RUN until an authorized run, and a Pi 5 result never stands in for a Spark result.

## Known limits

- Segment SHA-256 on load costs roughly 1 ms per MiB on one core (INFERRED from typical sha2 throughput, not measured); it runs on the worker and, with per-segment digests, on several cores in parallel (section 15.4), so it adds restore latency, not step latency. BLAKE3 is not adopted for format version 1 (section 15.4 gives the reason and the condition for revisiting it).
- The restorable prefix is capped at `prompt.len() - 1`; one token of prefill is always paid.
- Quota accounting counts foreign-namespace objects but does not read their payloads; an object with a valid header and a corrupt payload occupies quota until its first load deletes it or LRU evicts it.
- `max_objects` default 65536 and the index are per store directory; two daemons sharing one directory are not supported (STORE_ID is advisory, there is no lock file in this milestone).
- `arith_id` is a coarse label and `KV_ARITH_EPOCH` is a human-bumped counter backed by a pinned-bytes test; a numeric change the pinned test does not exercise could slip through (UNKNOWN residual risk; the test model should cover every op the production path uses).
- The host engine build id names the aien-runtime commit, so every new host build starts with an empty namespace; a build from a tree without git and without `AIEN_BUILD_COMMIT` gets the build id `+git:unknown` and then relies on the rounding class (backend name and `KV_ARITH_EPOCH`) alone. Uncommitted edits do not change the build id.
- `Indeterminate` publication means a power loss right after may lose the object; that is a cache miss, never a wrong answer.


## 12. C1 delta (exact changes to the implemented crate) (validation layers amended by 18.2)

C1 is implemented in `crates/aien-kv-store` at ce5d9eb (identity.rs, record.rs, validate.rs, error.rs, tests/golden/kv_record_v1.json, 4 vectors, `vectors_sha256 = 7c532a54...b89c` in tests/golden/README.md; all VERIFIED by reading the files). The delta below is the whole list; it lands as one red-first cut **C1b** (CUTS.md) before any later cut builds on C1. `format_version` stays 1 and the domain strings `AIEN-PKV-NS-V1\0`, `AIEN-PKV-CHUNK-V1\0` stay: no object has ever been written (the crate has no filesystem adapter, VERIFIED by listing `src/`) and the aienos pin does not exist yet (no `native/kv` in aienos at 5429602, VERIFIED), so no stored byte can be misread. The golden vectors are regenerated and the vectors hash changes.

### Identity changes

| # | change | file |
|---|---|---|
| I1 | `NamespaceFields` gains `compute_config_digest: [u8; 32]` | identity.rs |
| I2 | `namespace_id()` appends `compute_config_digest` (32 raw bytes) after field 13 (`sharing_namespace`, u16 length + bytes); every earlier field keeps its position and encoding | identity.rs |
| I3 | new `ComputeConfig { num_heads: u32, hidden_dim: u64, intermediate_dim: u64, vocab_size: u64, rms_norm_eps: f32, tie_word_embeddings: bool, hidden_act: HiddenAct, arith_id: &str (at most 255 bytes), arith_epoch: u32 }` and `ComputeConfig::digest() -> Result<[u8; 32], KvError>` = `SHA-256("AIEN-PKV-CFG-V1\0" || num_heads u32le || hidden_dim u64le || intermediate_dim u64le || vocab_size u64le || rms_norm_eps.to_bits() u32le || tie u8 || hidden_act u8 || arith_id u8-len + bytes || arith_epoch u32le)` | identity.rs |
| I4 | new `HiddenAct` enum, `Silu = 1` | identity.rs |
| I5 | `NamespaceFields` stays `Copy`; `rope_theta` encoding is unchanged | identity.rs |
| I6 | golden vectors: each of the 4 inputs gains the 9 `ComputeConfig` values and the digest; 2 new vectors: `rms-norm-eps-1e-6-only` (identical to a 1e-5 vector except `rms_norm_eps`, namespace differs) and `arith-epoch-2` | tests/golden/kv_record_v1.json |
| I7 | `vectors_sha256` re-pinned in `tests/golden/README.md` | tests/golden/README.md |

The conversion from `ModelConfig` is **not** in C1 (the crate has no dependency on `aien-abi-core`); it is `ComputeConfig::from(&ModelConfig)` in `aien-runtime`, exhaustive with no `..` (section 1.2.1), delivered in C6a/C6.

### Header and record byte changes

| # | change | offsets / sizes | file |
|---|---|---|---|
| H1 | the 256-byte header struct is **unchanged**: offsets 0 to 255, every field, `header_sha256 = SHA-256(bytes 0..224)`, `header_len = 256` | 0..256 | record.rs |
| H2 | a record file is `header region (4096 B) || payload`; new consts `HEADER_REGION_LEN = 4096`, `PAYLOAD_OFFSET = 4096`, `MAX_SEGMENTS = 120`; `HEADER_LEN` stays 256 | file layout | record.rs |
| H3 | bytes `256 .. 256 + 32 * n_seg` of the region hold the segment digest table (SHA-256 of each segment); all other bytes up to 4095 are zero | 256..4096 | record.rs |
| H4 | `payload_sha256` (offset 184) becomes the root `SHA-256("AIEN-PKV-PAYLOAD-V1\0" || seg_digest_0 || ... )`; new `payload_root(&[[u8;32]]) -> [u8;32]` | 184..216 meaning | record.rs |
| H5 | `RecordHeader::encode` keeps returning 256 bytes; new `encode_region(&self, seg_digests: &[[u8;32]]) -> Result<[u8; 4096], KvError>` | n/a | record.rs |
| H6 | `RecordHeader::decode` and `validate_record` take at least 4096 bytes (`Truncated` below that); file size rule becomes `4096 + payload_len` | n/a | record.rs, validate.rs |
| H7 | new `KvError` variants, in check order: `BadPadding` (nonzero byte in the region outside struct and table), `SegmentRootMismatch` (root recomputed from the table differs from offset 184), `TooManySegments` (`n_seg > 120`); inserted after `BadFlags`; `FileSizeMismatch` stays last | n/a | error.rs, validate.rs |
| H8 | new `segment_bytes(&NamespaceFields) -> u64` (host: `layout.block_stride`; native: `payload_len`, one segment) and `segment_count(...)`; `expected_payload_len` is unchanged | n/a | identity.rs |
| H9 | validation still performs zero heap allocations (the root is streamed through one `Sha256`) | n/a | validate.rs |

Red tests for C1b (PLANNED): `golden_vectors_include_compute_config_digest`, `rms_norm_eps_alone_changes_the_namespace`, `compute_config_digest_changes_with_each_field` (9 mutators), `namespace_changes_when_any_field_changes` (now 14 mutators), `header_struct_bytes_0_to_255_unchanged_by_the_region` (the old 256-byte vectors still match at offsets 0..255 except the re-derived namespace and root fields), `header_region_roundtrip_4096`, `validate_refuses_nonzero_padding`, `validate_refuses_segment_root_mismatch`, `validate_refuses_more_than_120_segments`, `file_size_is_4096_plus_payload`, `validate_refuses_before_any_allocation` (extended to the three new errors, still zero allocations).

## 13. C2 delta (KvStore trait and fake adapter)

The trait is not implemented yet, so this lists changes against the sketch of the first ADR revision; the sketch in section 3.2 is the revised one.

| # | change | finding |
|---|---|---|
| T1 | `submit_store` does no eviction and no I/O: it only reserves (`reserved_bytes += payload + 4096`) under the index mutex; reclamation is a worker state `Reclaiming` that precedes publication | 7 |
| T2 | `delete` only marks keys; the unlink runs on the worker; bytes are subtracted after the unlink returns | 7 |
| T3 | `Capacity` gains `state`, `reserved_bytes`, `evicting_bytes`, `unindexed_bytes`, `unindexed_objects`, per-class `restore_ops/park_ops` and caps, `max_waiters_per_job`; `inflight_ops/max_inflight_ops` are replaced by the per-class pair | 8, 9 |
| T4 | `recover(budget) -> RecoveryReport` becomes `recover_step(slice_ms) -> RecoveryProgress`; `lookup` returns misses and `submit_*` refuses with `QueueFull` while `state == Recovering` | 8 |
| T5 | `Completion` gains `terminal: Option<Terminal>` and `publish: PublishOutcome`; **exactly one terminal completion per accepted job**, including cancelled and failed ones | 5, 9 |
| T6 | `cancel` returns immediately but the destination is releasable only after the terminal completion is polled | 5 |
| T7 | publication semantics: rename succeeded and `fsync(dir)` failed yields `Published` with `PublishOutcome::Indeterminate`, the object is indexed and served, the adapter enters `Degraded` (no new stores until reopen); a failure before the rename publishes nothing | 10 |
| T8 | `attach_waiter(job)` with a per-job and a total bound | 9 |
| T9 | completion slots (three per job) are reserved at submit; `poll` never overflows | 9 |
| T10 | `shutdown(intake_deadline_ms) -> DrainReport { published, cancelled, unfinished }`; the worker keeps `Arc`s of the manager and buffers until each job is terminal; nothing is freed to meet the deadline | 11 |
| T11 | `PinnedSource` gains `tail_valid_tokens: u32` (rows at or above it in the last block are written as zero, section 2.2) | 3 |
| T12 | `StoreState { Recovering, Open, Degraded, Failed }` in `Capacity` | 8, 10 |

Fake adapter (`src/fake.rs`) additions: a deterministic virtual clock with an explicit `advance(dt)` that is the **only** place I/O and eviction happen (so a test can assert `submit_*` did zero I/O by counting calls inside the submit); scripted faults `rename_ok_dirsync_fail`, `fsync_file_fail`, `stall_fsync(duration)`, `stall_eviction(duration)`, `unlink_fail`; a **poison-on-free** destination check (every destination handed to a load is registered; after its holder is freed any write to it is recorded and the test fails); a recorder of which terminal completion each job produced. Red tests for C2 (PLANNED): `fake_submit_store_performs_no_io_and_no_eviction`, `fake_reservation_prevents_overcommit_by_concurrent_stores`, `fake_every_job_yields_exactly_one_terminal_completion` (including cancelled), `fake_rename_ok_dirsync_fail_reports_indeterminate_and_serves_object`, `fake_lookup_misses_and_submit_refused_while_recovering`, `fake_poll_never_overflows_at_max_inflight`, plus the cases already listed in CUTS.md.

## 14. Acceptance boxes the review found uncovered

| box | gap named by the review | closed by | red test (all PLANNED) |
|---|---|---|---|
| #484 A4 | host interruption at publication boundaries | C4 | `fault_at_each_publication_boundary_leaves_consistent_outcome` (P1 to P5, section 2.4) |
| #484 A6 | accounting after recovery limits | C4 | `recovery_keeps_admission_closed_until_walk_completes`, `deleting_one_indexed_object_does_not_reopen_admission_with_unscanned_remainder`, `used_bytes_counts_header_region_and_physical_blocks`, `inflight_temp_bytes_are_reserved` |
| #485 B3 | sibling-job cancellation case | C5, C6 | `restore_txn_holder_survives_sibling_failure_until_all_jobs_terminal`, `multi_chunk_restore_failure_does_not_free_destination_in_use` |
| #485 B4 | sustained admission starvation, completion credits | C5 | `sustained_restores_cannot_starve_parks`, `completion_queue_never_overflows_at_max_inflight`, `coalesced_waiters_are_bounded` |
| #485 B6 | completed-request counts, request-latency p50/p95, allocation and staging peaks | C9 | `bench_output_requires_completed_requests_latency_percentiles_and_peaks` |
| #367 D3 | explicit Cortex, boot, update, recovery regression gates | N2, N5 | existing gates re-run on an image that also holds a slab: `scripts/qemu_ck_store_test.sh`, `qemu_ck_lifecycle_test.sh`, `qemu_ck_continuity_test.sh`, `qemu_ck_boot_test.sh`, `qemu_ck_slot_loader_test.sh`, `qemu_ck_recovery_test.sh`, `qemu_ck_rollback_test.sh`, `qemu_ck_disk_layout_test.sh` (all exist, VERIFIED by `ls scripts`; QEMU gates run only through the heavy forge, per their headers); `n5_existing_gates_pass_with_slab_present` is the wrapper that runs them and records pass/fail |
| #367 D4 | session reset/close between pump calls | N4 | `park_survives_next_prompt_reset`, `session_close_with_queued_park_keeps_staging_until_job_terminal`, `reset_between_pump_calls_leaves_only_whole_slots` |
| #367 D6 | complete native staging accounting | N3, N4, N5 | `staging_report_lists_export_cipher_env_plain_scratch_bounce_peak_separately` |

## 15. Native performance track: beating the Linux path

Drake's rule (memory, 2026-10-13): the native design must outperform Linux and be differentiated, not match it. This section turns that into measurable work. Everything here is **PLANNED** or **UNKNOWN**; no ratio below has been measured and none is a promise. Each item names the cut that either earns the claim with a receipt or deletes it. The correctness contract of sections 1 to 11 is never traded for speed: every optimization below keeps the same validation order, the same integrity check over every payload byte, and the same ownership proofs.

### 15.1 The baseline to beat: the Linux host adapter, measured honestly

Cut **X1** builds the baseline first, because "beating Linux" is meaningless without a Linux number taken the same way. The host adapter (C4) is run through the C9 benchmark harness in four labelled modes, on the same machine, the same model and the same chunks:

| mode | how | label in the output |
|---|---|---|
| buffered, page-cache hit | `pread` after the file was just written or read; no cache drop | `cache_state = page_cache` |
| buffered, device read | `posix_fadvise(DONTNEED)` on the file, then (optional, recorded as done or not done) `echo 3 > /proc/sys/vm/drop_caches`; the harness verifies it by comparing `mincore` residency before the read (0 pages resident) | `cache_state = device` only when residency was verified, else `unknown` |
| `O_DIRECT`, device read | aligned reads (section 15.3) into pool memory; never served from the page cache | `cache_state = device` |
| `O_DIRECT` with a sector-aligned queue depth sweep (1, 2, 4, 8) | same, to expose the device's own parallelism | `cache_state = device`, `queue_depth` |

Page-cache hits and device reads are **never averaged together**; the harness refuses to emit a latency summary that mixes labels (red test `bench_refuses_mixed_cache_state_summary`, PLANNED). The Linux baseline for a native comparison is the best of the four modes for the same bytes on the same device, not the buffered default, so native cannot "win" against a deliberately slow Linux configuration. Reported per mode: p50 and p95 restore latency per chunk and per 2048-token restore, bytes per second, CPU time per MiB (user plus system, from `getrusage`), and peak resident memory. The device ceiling is measured separately with a raw sequential read of the same LBA span so every result can be stated as a fraction of what the device can do (UNKNOWN until X1 runs).

### 15.2 Zero-copy DMA into KV pool memory on GB10 unified memory, with an ownership proof (amended by 18.2)

The GB10 CPU and GPU share one physical memory pool (INFERRED from the Spark platform description; the exact coherency and IOMMU rules for a device writing into memory that the GPU later reads are **UNKNOWN** here and must be read from the NVIDIA documentation through the `spark-hardware-manual` skill and recorded with `docs-read` before cut X3 touches a driver, per the machine-debugging rule). Today the native disk contract copies through a driver-owned bounce buffer (section 9 item 1; native/disk/disk_nvme.h:17-19 VERIFIED), then through the envelope and decrypt scratch (section 10). The design goal is that restored bytes are written by the NVMe controller **directly into the destination KV memory**, with the decrypt/verify step reading them in place.

Ownership proof (the part that makes zero copy safe), as a state machine over a destination region `R`:

1. `Free`: nobody references `R`.
2. `DmaTarget(job)`: `R` is reserved to one restore job; the session cannot see it (rows are invisible until `restore_commit`, section 9 item 6); the allocator will not reuse it; the device may write it. The adapter holds the region until the **terminal completion** of every command that targets it (T5, T6 of section 13). A cancel or shutdown cannot release it earlier (D4 of #367).
3. `Verifying(job)`: all commands completed; the CPU reads `R` to verify the tag and digests; the device has no outstanding command on `R`.
4. `Committed`: `restore_commit` adopts `R`; the session owns it.
5. a failed verification returns `R` to `Free` after zeroing is not required (the contents were never visible), but the job is terminal exactly once.

A region is in at most one state; transitions are checked by a `region_state` table in the native adapter and by the fake in the host emulation. Red tests (PLANNED, cut X3): `dma_target_region_not_reused_until_terminal_completion`, `cancel_while_dma_pending_keeps_region_reserved`, `verify_failure_never_exposes_rows_to_session`, `region_state_transitions_are_exclusive` (a mutant that skips the state check must fail), and in the emulator `disk_file` with a poisoning check on release (the same method as C2's poison-on-free).

### 15.3 On-disk layout equal to the in-memory block layout, so restore is pure DMA (amended by 18.2)

Two layout facts decide how much copying is unavoidable.

- **Host**: the object payload is already the pool's own block layout (section 2.2: `[layer][K plane | V plane][token][head][dim]` bf16, one `block_stride_bytes` per block). Placing the payload at file offset 4096 (section 2.1) makes each segment (one block, `block_stride` = 524288 for the test geometry, calc:666b7ba6) a multiple of 4096, so an `O_DIRECT` read of a segment into the destination pool block needs no bounce and no reshaping. This is what X2 builds and what makes the Linux baseline of 15.1 the strongest possible one.
- **Native**: the payload is a strided gather from `[layer][capacity][kv_dim]` f32 arrays (section 2.3), so restore today needs a copy per layer. Pure DMA needs the native session KV to be stored **block-major** (a native paged KV: one contiguous region per chunk holding all layers' K and V rows for that token range) so the on-disk chunk and the in-memory chunk are the same bytes. This is a new `backend_format` value (3, "native paged f32"), not a change to format 2; it changes `aienos-infer-kernel` session memory layout, so it is its own cut **X4** with its own namespace, vectors and a measured cost on decode (a paged layout can slow attention; X4 refuses to merge if native decode throughput drops by more than 2 percent on the PI5-6 golden run, a threshold PLANNED here and to be revised only with data). Until X4 merges, native restore keeps the gather copy and the report states it (`kv_restore_copy_bytes`, METRICS_AND_SETTINGS.md).
- **Alignment rule** shared by both: payload offset 4096, segment length a multiple of 4096 (host segments are blocks; native one segment padded by the existing unit rule), destination addresses 4096-aligned. The adapter checks alignment at submit and falls back to the buffered path with a counter (`kv_odirect_fallback_total`) if the filesystem or the destination refuses `O_DIRECT`; the fallback is a correctness-preserving slow path, never an error.

Red tests (PLANNED): X2 `odirect_read_lands_in_pool_block_without_intermediate_copy` (counting allocator proves no staging buffer larger than the tail scratch), `misaligned_destination_falls_back_and_counts`; X4 `native_paged_chunk_bytes_equal_disk_bytes` (memcmp of the DMA'd region against the session rows) and `native_paged_decode_matches_strided_decode_bitwise` (golden tokens identical).

### 15.4 Integrity pipelined with transfer, not weakened (amended by 18.2)

The integrity rule does not change: every payload byte is covered by a digest verified before the bytes are visible to a session (section 1.4, 2.1). What changes is **when** the digest is computed.

- **Per-segment digests** (section 2.1) let verification of segment `i` overlap the transfer of segment `i+1`: the worker verifies a segment as soon as its command completes while the device fills the next one. The root over the table is checked last and gates `Published`/adoption. A flat payload hash cannot do this (it needs all bytes in order).
- **Hash function**: SHA-256 stays the format hash. Justification: it is already the identity hash everywhere (namespace, chunk key, header, M5 envelope on native uses `argus sha256`, section 2.5), one primitive means one set of vectors and one audited implementation on both platforms, and the Pi 5 CPU has the Arm SHA-2 instructions that make it fast (INFERRED from the Cortex-A76 feature set; to be confirmed by a measurement in X5). BLAKE3 is faster on large inputs and parallelizes inside one hash, but it would add a second hash primitive to a freestanding native tree that has none (no BLAKE3 anywhere in aienos `.c`, `.h` or `.rs` files, VERIFIED by grep), a second vector set, and no benefit once per-segment digests already give inter-segment parallelism. Per-block hashing is exactly the per-segment design on the host (segment = block). Decision: **SHA-256 over segments**; a different hash is considered only if X5 shows verification, not the device, is the bottleneck after pipelining, and would be a new `format_version`.
- **Native**: the AEAD tag already authenticates the envelope; the digest table is not stored on native (one segment). Pipelining there means decrypt-and-verify of chunk `i` overlapping the read of chunk `i+1` across pump calls, bounded by `KV_PARK_STAGE_BYTES` and the staging peaks of section 10.

Red tests (PLANNED, X5): `pipelined_verify_never_adopts_before_root_checked`, `corrupt_segment_k_detected_while_segment_k_plus_1_in_flight` (fault injection flips a byte in segment k, nothing is adopted), `pipelining_does_not_reorder_digest_checks_before_header_checks` (validation order of 1.4 holds), `verify_cpu_time_per_mib_reported`.

### 15.5 Predictive restore on application and Interplane signals

Today restore starts at `submit_request` (section 6.3). A native system knows more than a Linux page cache does: the application layer or Interplane (the AIEN message plane) can announce "conversation X is about to continue" before the user's text arrives. Design (cut **X6**): a `prefetch_hint(chain_head_key, expected_tokens)` call that is a **hint, never a command**: it may start a restore job into a holder with `class = Restore`, lowest priority, under the same per-class bounds (section 7.1), and it is cancelled and its holder freed if the real request does not arrive within `AIEN_KV_PREFETCH_TTL_MS` (default 30000, PLANNED) or memory pressure needs the blocks. A hint can only make a later restore faster; a wrong hint costs bounded I/O and holder blocks and is counted (`kv_prefetch_hint_total`, `kv_prefetch_hit_total`, `kv_prefetch_wasted_bytes_total`). Which signals Interplane actually emits is **UNKNOWN** (not read in this revision); X6 starts by reading the Interplane sources and records the real signal list, and only then writes the hook. Red tests: `prefetch_hint_never_blocks_step`, `unused_prefetch_expires_and_frees_holder`, `prefetch_respects_restore_bound`, `real_request_adopts_prefetched_prefix_without_second_read`.

### 15.6 Streaming park during decode

Today a park starts when the turn finishes (section 6.2). Full chunks of the prompt and of generated output are immutable the moment they are computed (their keys depend only on tokens and parents), so they can be parked **while decode continues**, hiding write latency and shrinking the work at `Finished` to the partial tail. Design (cut **X6b**, host first, native after X4): at the end of a step, for each running sequence whose computed prefix gained a full chunk, take a per-chunk holder with `fork_range` (section 5), submit a Park job, and release the holder at its terminal completion. Rules: the chunk is complete (all `chunk_tokens` computed, `computed_prefix` of section 6.1), the sequence is not in a state where its blocks can still be rewritten (rows below the computed prefix are never rewritten, INFERRED from append-only decode and COW on partial shared blocks, to be proven by the C3 test `owner_append_on_shared_partial_block_copies_not_holder`), and admission obeys the park class bound so streaming cannot starve restores. A sequence cancelled mid-decode leaves its already-published chunks in place (they are valid prefixes). Red tests (PLANNED): `streaming_park_publishes_full_chunks_before_finish`, `streaming_park_never_reads_rows_that_decode_still_writes` (poison the tail block, assert no streamed chunk includes it), `cancel_after_streamed_chunks_keeps_them_valid`, `streaming_park_respects_park_bound`. Metrics: `kv_park_streamed_chunks_total`, `kv_park_bytes_at_finish` (to show the finish-time work shrank).

### 15.7 Multi-command NVMe queue: only if profiling justifies it

The native contract is one command at a time through one bounce buffer (native/disk/disk_nvme.h:17-19 VERIFIED). Lifting it adds real risk: command ids, wraparound, DMA lifetime, reset and flush ordering. The rule is therefore: **no multi-command work until the X1/X3 profile shows the single-command contract is the bottleneck**, meaning in the native restore profile the device is idle for more than 30 percent of the restore wall time between commands (threshold PLANNED) while pump budget and verification are not limiting. If, and only if, that holds, cut **X7** implements a bounded submission window (`KV_QUEUE_DEPTH` is already the request bound, disk.h, section 9 item 7) and must pass, before any benchmark is run:

- `command_ids_are_unique_among_inflight_commands` and `command_id_wraparound_never_aliases_an_inflight_command` (drive the id counter across the wrap with commands pending);
- `dma_buffer_lifetime_covers_every_inflight_command` (region state of 15.2 stays `DmaTarget` until its own completion, with out-of-order completion in the emulator);
- `controller_reset_fails_every_inflight_job_terminal_once_and_releases_regions_after_reset_ack`;
- `cancel_with_commands_inflight_waits_for_their_completions_before_release`;
- `flush_ordering_header_flush_issued_only_after_all_payload_completions` (the header-last rule of section 2.5 and 9 item 2 holds under reordering; a mutant that issues the header write early must fail);
- the existing power-cut injector at every unit boundary (N2) re-run with more than one command in flight.

If the profile does not justify X7, the cut is closed as REJECTED with the profile as evidence and the single-command path remains.

### 15.8 Target ratios against the Linux baseline

These are design targets to be confirmed or revised by X1 and the later cuts; they are not results. Ratios are native divided by the best Linux mode of 15.1 for the same bytes on the same device, on the same machine class (Spark-native results stay NOT_RUN until aienos#336 is authorized; the emulator can only test logic).

| quantity | target | earned by | if missed |
|---|---|---|---|
| device-read restore throughput | at least 0.95 of the device ceiling (15.1) and at least 1.0 of `O_DIRECT` Linux | X3, X4, X5 | report the gap and its cause (copy count, verify, single command) |
| restore CPU time per MiB | at most 0.5 of Linux `O_DIRECT` (no kernel crossing, no page cache copy) | X3, X5 | report; target dropped if the measured Linux figure is already near the hash cost |
| restore bytes copied by the CPU | 0 for payload after X4 (only verify reads) versus 1 for Linux buffered | X3, X4 | the count is a direct measurement (`kv_restore_copy_bytes`), pass or fail |
| time to first token after a hint | restore fully overlapped with the user typing: 0 restore latency on the critical path when the hint arrives at least `restore_latency` earlier | X6 | `kv_prefetch_hit_total` shows the hit rate; no claim without it |
| work left at `Finished` | one partial chunk plus the last flush | X6b | `kv_park_bytes_at_finish` shows it |
| decode throughput during park | within 2 percent of no-park decode | X6b, N4 | the existing C8/N4 interleaving evidence |

A target met on the emulator or on the host is labelled `native_emulator` or `hosted`; only a run on the target hardware earns `native_physical` (METRICS_AND_SETTINGS.md `platform`).

### 15.9 Cuts for this track

X1 Linux baseline harness; X2 `O_DIRECT` host adapter path and alignment; X3 native zero-copy DMA ownership proof (emulator first); X4 native paged KV (`backend_format` 3) with pure-DMA restore; X5 pipelined integrity and the hash decision receipt; X6 predictive restore; X6b streaming park; X7 multi-command queue (conditional, closed as REJECTED unless the profile justifies it). Dependencies and tests are in `docs/persistent-kv/CUTS.md`.

## 16. Pi 5 physical lane

The Spark-native physical run is NOT_RUN and blocked on aienos#336. A Raspberry Pi 5 with a dedicated test card is available now (`~/handoffs/2026-10-13-pi5-roadmap-ledger.md`: PI5-6 inference ran on the Pi, #364 merged; pi5-agent #366 merged), so the first **real cold boot** evidence for native persistent KV can come from there. This lane is kept strictly separate from the Spark-native and the hosted results (section 16.5).

### 16.1 What exists (VERIFIED by reading the ledger, `pi5-run.sh` and aienos sources)

- The Pi boots AIENOS through a one-shot firmware setting: `pi5-run.sh` appends a `[tryboot]` block to the boot partition config and reboots with `reboot '0 tryboot'`; normal boots never read it (pi5-run.sh:11-15, 144-177). The AIENOS image on the card has an ESP and a raw model partition `p3` (784 MiB, `AIENOS-MODEL`) located through `MODEL.MAP` sectors (ledger line 60; `native/boot/model_map.h`).
- The Pi kernel writes only inside the AIENOS partition view (`native/kernel/dev/disk_part.h` header comment, lines 1 to 28). A raw range outside it cannot be written by the kernel.
- The Pi inference run proved 8 of 8 golden tokens ("The capital of France is Paris."), 17 prompt ids, 1.20 tok/s and a prefill of 17 tokens in 11.1 s (ledger line 60). One decode token is about 0.83 s, so prefill of the 17 tokens costs about 10.4 s more than a restore plus one suffix token would (calc:5a26db0a), which is the visible saving a restore must show.
- The pi5-agent stage persists the reply, the model hash and ids in the Store first, then loads the model (ledger line 62, #366).

### 16.2 Where the KV slab lives on the card (no repartitioning)

The test card is not repartitioned. The slab is carved from the **tail of the AIENOS partition's own Store region**, not from free space after `p3`, because the kernel cannot write outside its partition view (16.1). Concretely: `kv_disk_bind` receives the partition view and two disjoint ranges inside it (the Store region and anchor, and a 128 MiB slab = 32768 units of 4096 bytes, calc:9193e870). The Store region therefore shrinks by 128 MiB and the Store needs **one reformat** to take the smaller size (it is a test card; the Store holds only test facts and replies, so the cost is a re-run of the PI5-5 fact stage; this is recorded as the one irreversible step of the lane and needs the Store contents to be re-creatable, UNKNOWN until P1 checks what the card holds). The slab superblock records the region size; no Store byte is moved.

A new **`KV.MAP`** file on the ESP, written next to `MODEL.MAP` by the same tool path in `pi5-run.sh`, tells the kernel at boot that a slab exists, its first partition-relative unit, its unit count, and the sc ADR hash of the record format. `KV.MAP` is advisory: the kernel validates the slab superblock and refuses on disagreement, so a stale or missing map means "no KV slab", never a corrupt read. The state region question the review raised (a restore needs a durable record of "which chain head is the latest" across the cold boot): the agent writes a small **state record** (the latest chain head key and token count, sealed like a Store record) into the Store first, exactly as #366 persists the reply, and the slab holds the chunks; boot 2 reads the state record, finds the head, and restores.

### 16.3 Native CPU park and restore in the Pi agent session

The Pi runs the native CPU path (`aienos-infer-kernel`, NEON dotprod, VERIFIED by the ledger). The same N4 mechanics apply: `park_point` callback after `STEP_LAST` (section 9 item 7), synchronous export into adapter-owned buffers, pump between steps and an idle pump after the reply, then on boot 2 `kv_import`, `restore_commit`, and the suffix prefill. SD card speed is the constraint on Pi (own SD driver in PIO mode per the ledger); the park cost and restore cost are measured, not assumed, and a 1B-model KV of 17 tokens is small, so the lane also runs a longer prompt (KV 64, listed as a PI5-6 follow-up in the ledger) to make the saving visible.

### 16.4 The run, and what it proves

Cut **P3** is the run, in this order, each step logged by the stage:

1. Boot 1 (tryboot): prompt, reply, park chunks into the slab, flush, state record into the Store, print the park report. Golden continuation is the reply tokens already proven for this prompt.
2. **Real cold boot**: the hub reboots through the firmware (tryboot is a firmware restart that clears RAM; it is **not** a power cut, ledger line 52 says the real power-cut test needs Drake unplugging the Pi, and this lane does not claim power-cut behavior).
3. Boot 2: slab recovery (bounded header scan), state record read, restore the chunks, `restore_commit`, prefill only the suffix, generate. Pass condition: the generated ids equal the golden 8 ids **and** the stage reports restored tokens greater than zero **and** prefill time is lower than the cold prefill measured on the same boot by a control prompt run without restore.
4. Negative control: a second boot 2' with the slab superblock flipped by one byte must refuse the restore and fall back to plain prefill (proves the check is live).

The result is labelled **"native physical (Pi 5)"** in the evidence ledger, with the card, the commit of both repos and the cache-free nature of the device (an SD card has no host page cache in the native path). It proves: a restart of the AIENOS kernel preserves and correctly restores KV bytes on real hardware. It does **not** prove: power-loss safety, Spark behavior, GB10 DMA, NVMe performance, or any hosted result.

### 16.5 Separation of results

| label | meaning | may stand in for |
|---|---|---|
| `hosted` | Linux host adapter (C4 to C9) | nothing native |
| `native_emulator` | QEMU or `disk_file` (N2 to N5) | logic only, not hardware |
| `native physical (Pi 5)` | this lane (P3) | nothing on the Spark |
| `native_physical` (Spark) | NOT_RUN, aienos#336 | n/a |

The evidence ledger `docs/evidence/kv-native.md` keeps one row per label; a row with no receipt says NOT_RUN. The harness is shared with the Pi roadmap (the Pi 5 tryboot harness is owned by the Pi lane; a mind map message from the persistent-KV program already says it will ask to share it later, so P3 coordinates with that owner before using the Pi, and never runs while another lane has it).

### 16.6 Cuts for this lane

P1 (slab inside the partition view, `KV.MAP`, Store reformat plan, QEMU/aarch64 emulation of the card layout); P2 (native CPU park and restore in the pi5-agent session, emulator run with the golden ids); P3 (the physical run above, authorized by Drake because it reformats the Store on the test card and occupies the Pi). Dependencies are in CUTS.md.

## 17. Review disposition

The independent review (Codex GPT-6 Astra, 16 findings plus uncovered acceptance boxes) was checked finding by finding against source (sc-pkv for sovereign-core, aienos-pkv for native). Result: **all 16 findings were valid; none was rejected.** "Section" is where this ADR changed; "cut" and "red test" are where the proof will land (all tests PLANNED, no receipts yet).

| # | finding | verdict | design change | cut | named red test |
|---|---|---|---|---|---|
| 1 | `rms_norm_eps` and other compute config not bound by identity | RESOLVED | compute digest in namespace field 14 (1.2, 1.2.1, 12) | C1b, C6a | `rms_norm_eps_alone_changes_the_namespace`, `modelconfig_field_mutation_changes_compute_digest`, `arith_epoch_pinned_bytes` |
| 2 | scheduler frees KV before `Finished` reaches the spine | RESOLVED | park inside the finish transaction before `free_sequence` (6.1) | C6a | `finish_transaction_forks_holder_before_free` |
| 3 | last token has no KV; partial block tail not zero | RESOLVED | `computed_prefix`, tail zeroing (2.2, 6.1) | C6a, C4 | `computed_prefix_excludes_uncomputed_last_token`, `tail_rows_above_token_count_are_zero_in_object` |
| 4 | restored prefix could enter decode before the remaining prompt is prefilled | RESOLVED | `admit_restored_prefix` into Prefill phase (6.3) | C6a | `restored_prefix_does_not_enter_decode_before_remaining_prompt_prefilled` |
| 5 | restore job failure frees a destination another job still uses | RESOLVED | `RestoreTxn` holder with `Draining` until all jobs terminal (5.5) | C5, C6 | `restore_txn_holder_survives_sibling_failure_until_all_jobs_terminal` |
| 6 | per-chunk holders and coalescing cannot be built on `fork_sequence` | RESOLVED | `fork_range`, `assemble_table`, landing-buffer coalescing (5.2, 7.3) | C3, C5, C8 | `assemble_table_orders_holders_and_refcounts`, `coalesced_waiters_with_different_suffixes_get_independent_tables` |
| 7 | eviction and unlink under the spine lock or at submit | RESOLVED | worker-side `Reclaiming`, `submit_store` only reserves (7.4, 13) | C2, C4, C5 | `fake_submit_store_performs_no_io_and_no_eviction`, `eviction_runs_on_worker_not_in_submit` |
| 8 | recovery accounting reopens admission with an unscanned remainder | RESOLVED | resumable `recover_step`, admission closed until walk completes (4.1) | C4 | `recovery_keeps_admission_closed_until_walk_completes`, `deleting_one_indexed_object_does_not_reopen_admission_with_unscanned_remainder` |
| 9 | restores could starve parks; completion queue overflow; unbounded waiters | RESOLVED | per-class reservation, reserved completion slots, waiter bounds (7.1) | C5 | `sustained_restores_cannot_starve_parks`, `completion_queue_never_overflows_at_max_inflight`, `coalesced_waiters_are_bounded` |
| 10 | publication when rename succeeds but dir fsync fails | RESOLVED | P1 to P5 boundaries, `Indeterminate`, `Degraded` (2.4, 13) | C2, C4 | `fault_at_each_publication_boundary_leaves_consistent_outcome`, `fake_rename_ok_dirsync_fail_reports_indeterminate_and_serves_object` |
| 11 | shutdown with a stalled fsync cannot meet its deadline | RESOLVED | `shutdown(intake_deadline)` with `DrainReport`, buffers kept until terminal (7.5) | C5 | `shutdown_with_stalled_fsync_returns_by_deadline_and_reports_unfinished` |
| 12 | native slot layout unspecified; M5 mapping unproven | RESOLVED | slot wire layout and authenticated envelope mapping (2.5) | N1, N2, N3 | `slot_header_golden_vectors_match`, `slot_binding_wrong_slot_or_incarnation_fails_tag`, `slot_bounds_refused_before_decrypt` |
| 13 | `set_pos` alone cannot restore a native session | RESOLVED | `kv_import` rows only plus `restore_commit` after prompt (9 item 6) | N4 | `restore_commit_after_prompt_prefills_only_the_suffix`, `import_before_commit_is_invisible_to_step` |
| 14 | native park source is reused by the next prompt; no idle pump | RESOLVED | `park_point`, synchronous export and seal, idle pump (9 item 7, 3) | N4 | `park_snapshot_survives_next_prompt_reset`, `idle_pump_drains_without_new_prompt` |
| 15 | native staging accounting incomplete; "only staging memory" claim false | RESOLVED | separate staging counters (10), METRICS fixed | N3, N5 | `staging_report_lists_export_cipher_env_plain_scratch_bounce_peak_separately` |
| 16 | evidence labels and citations wrong (trait release line, OWN-30B pin, wording) | RESOLVED | citations re-verified by grep; PLANNED label; matrix labelled | doc, C7 | `guarantee_matrix_rows_are_labelled` |

Uncovered acceptance boxes: A4, A6, B3, B4, B6, D3, D4, D6 are closed in section 14.

What the review did not change: the chunk-key chain, the 256-byte header byte layout, SHA-256 as the format hash, "storage never blocks the step", and "a miss or a failed restore is plain prefill". What remains UNKNOWN: GB10 coherency for device DMA into the pool (15.2), Interplane's real signal list (15.5), what the Pi test card's Store holds (16.2), and every measured number in 15.8.

## 18. Review 2 disposition and implementation reviews (2026-10-13)

Codex re-review 2 (`d6669ed..b654f0a`, summary in the program handoff) and the Grok 4.7 reviews of C4 and C5. Each row was checked against source; "VERIFIED" means the cited line or test was read or run on `pkv/integration`, host only. Native rows were checked against aienos-pkv `8e34b63`.

### 18.1 Codex re-review 2

| item | verdict | where | evidence |
|---|---|---|---|
| partial 6: shared landing holder freed while another waiter still needs it | RESOLVED (C5) | 7.3 | a completed load hands the caller's guard back as `Completion::holder` (`LandingHolder`), one copy per waiter; the blocks are released when the last copy drops. Tests `a_restored_landing_holder_outlives_the_terminal_the_caller_sees`, `coalesced_waiters_share_one_landing_holder_until_the_last_drops_it` (VERIFIED) |
| partial 8: logical reservation vs physical charge | RESOLVED (host) | 7.4 | reservations round up to the objects directory's `st_blksize`; publication and recovery charge `st_blocks` (host_fs.rs `phys_bytes_of`). `st_blksize` as allocation unit is INFERRED for ext4/xfs. Test `host_fs_reservation_covers_the_physical_bytes_publication_charges` (red before, green after, VERIFIED). Native: fixed slots, does not arise |
| partial 9: completion capacity retained until polled | RESOLVED (C5) | 7.1 | worker admission counts unpolled completions + 3 per live job + extra-waiter slots (worker.rs `credit_check`, `try_attach`). Test `completion_queue_never_overflows_at_max_inflight` (VERIFIED) |
| partial 12 | split into new findings 1 and 4 | 2.5, 18.2 | below |
| new 1 (P1): native slot stride depends on payload | RESOLVED | 2.5 | format-time `slot_units`, per-record `sealed_units` bound; implemented by N2 (kv_slab.h:87, VERIFIED) |
| new 2 (P1): S3 on a worker that may not take the spine | RESOLVED (design; C6 implements) | 5.4 S3, 8 rule 2 | S3 runs on the spine thread after polling; the worker only reports completions |
| new 3 (P1): terminal failure treated as proof DMA stopped | RESOLVED (design; X3 implements) | 18.2 X3 | |
| new 4 (P2): one validation contract for host region and native slot | RESOLVED (design) | 18.2 | |
| new 5 (P2): X4 zero copy ignores encryption | RESOLVED (claim narrowed) | 18.2 X4 | |
| new 6 (P2): X5 overlap without an async path | RESOLVED (relabelled) | 18.2 X5 | |
| retained P1: `Incarnation(u64)` too small | RESOLVED (C2/C5) | 5.3 | `Incarnation { seq_id: u64, epoch: u32 }` (store.rs:39); per-waiter routing. Tests `incarnation_keeps_the_full_64_bit_sequence_id_and_the_epoch`, `stale_completion_with_reused_seq_id_but_new_epoch_is_rejected`, `coalesced_waiters_each_get_a_terminal_with_their_own_incarnation` (VERIFIED) |
| retained: `job_id as u32` holder ids wrap | RESOLVED | 5.2 | holder.rs `HolderGenerations` skips live ids and refuses on exhaustion; `generation_for_job` refuses ids that do not fit. Tests in holder_tests.rs (VERIFIED) |

### 18.2 Binding amendments (supersede the cited text where they differ)

- **Validation layers (new 4; amends 12 H6/H7).** Three validators with their own error order: (a) shared structural validation of the 256-byte header struct (H1, header hash, field checks, namespace), identical on host and native and the only part N1 mirrors; (b) host-region validation of bytes 256..4096 (H3 digest table, `BadPadding`, `SegmentRootMismatch`, `TooManySegments`) and the file size rule, host only; (c) native-slot validation of the slot unit (`unit_sha256`, slot fields, `sealed_units <= slot_units - 1`, then the M5 tag), native only. The M5 envelope at byte 256 of a native slot is nonzero by construction (m5_envelope.c, VERIFIED), so (b) never runs on a native slot. Each layer has its own vectors.
- **X3 DMA ownership (new 3; amends 15.2).** A terminal failure does not end `DmaTarget`. The NVMe timeout path only marks the controller failed (disk_nvme.c:86-88, 151, VERIFIED); the device may still write. A region leaves `DmaTarget` only on successful completion of every command that targets it, or after controller quiescence is acknowledged. Until then a region whose command failed or timed out is `Quarantined`: never reused, counted in `kv_quarantined_bytes`, released only after quiescence. The quiescence mechanism is DOCS NOT READ here; it must be read from the NVMe and GB10 documentation (`spark-hardware-manual`, `docs-read`) before X3.
- **X4 copies (new 5; amends 15.3).** Disk holds ciphertext with interleaved tags. The current open path wipes the output, authenticates each chunk into scratch and copies plaintext out (m5_envelope.c:152-153, 176-184, VERIFIED), so a paged layout alone cannot make restore copy-free. X4 claims only "ciphertext lands by DMA without the bounce buffer" and keeps reporting `kv_restore_copy_bytes` for the decrypt copy. An in-place authenticated decrypt is a separate future design, not part of X4.
- **X5 wording (new 6; amends 15.4).** The native disk driver keeps one command outstanding and waits for it before returning (disk_nvme.h:17-19, VERIFIED). Native verify-while-read is therefore **interleaving** across pump calls on one core, not overlap. True overlap needs an asynchronous submit/poll path or a second execution context, which is a dependency X5 does not have; until it exists, X5 reports interleaving.

### 18.3 Grok 4.7 implementation reviews

| review item | status | commit (pkv/integration) |
|---|---|---|
| C4 P1-1 failed store loses already-unlinked victims | FIXED: victims marked, unlinked only after the store is durable | 120b58d (red d22336c) |
| C4 P1-2 recovery memory unbounded, index lock across listing | FIXED: one streaming `ReadDir` under its own mutex | 120b58d |
| C4 P1-3 `read_dir().ok()` hides I/O errors; purge accounting | FIXED: listing/stat errors fail the store; purge subtracts only after unlink | 120b58d |
| C4 P2-4 foreign objects skip padding/root checks | FIXED | 120b58d |
| C4 P2-5 stores refused while unindexed files exist | FIXED: free headroom admits | 120b58d |
| C4 P2-6 `recover_step` done on a Failed store | FIXED (`recover_step_on_a_failed_store_is_not_done`) | 120b58d |
| C4 P2-7 `payload_len` narrowed to u32 | FIXED: u64 index entry | 120b58d |
| C4 P2-8 missing tests | FIXED: host_fs_hardening_tests, host_fs_stream_tests | d22336c |
| C5 P1-1 second store of a live key steals completions | FIXED: one live store job per key; by_inner never overwrites a live mapping | 00c8aab (red c1d4046) |
| C5 P1-2 landing holder freed before the caller sees Completed | FIXED: store gets a lease, worker keeps the guard, `Completion::holder` | 00c8aab |
| C5 P1-3 panic reports Failed while the store owns the buffer | FIXED: such jobs stay unfinished until the lease returns | 00c8aab |
| C5 P2-4 cancel/shutdown miss `Dispatching` | FIXED: shutdown/drop cancel handed jobs and mark in-submit jobs | 00c8aab |
| C5 P2-5 wall-clock test bounds | FIXED for the caller-latency test (store stalled indefinitely); deadline tests keep lower bounds tied to the deadline and 5 s upper bounds | 00c8aab |

Waiter default conflict (ADR 7.1 says 16, METRICS_AND_SETTINGS said 8): the ADR is binding; METRICS now says 16, matching `WorkerConfig::default` (worker.rs).

### 18.4 Park bound default (C9 bench finding)

The first C9 host run showed 8 conversations finishing in one step submit 24 chunk parks; with `max_park_ops = 16`, 8 were refused `QueueFull` (by design: not parked, the next turn prefills). The default is now 64 (worker and host adapter), enough for 8 simultaneous finishes of up to 8 chunk parks each. Test: `default_park_bound_admits_eight_conversations_finishing_together` (red at 16, green at 64; host).
Memory cost: the op bound is bookkeeping only (one job entry plus three reserved completion slots per op; completion capacity grows from 192 to 336 entries). Pool memory held by in-flight parks (forked holders keep finished turns' blocks referenced until written) stays bounded by the park byte limit, `max_inflight_bytes / 4` = 64 MiB by default (worker.rs `park_byte_limit`), which this change does not touch. With large real-model chunks the byte limit, not the op count, may refuse parks first (UNVERIFIED; to measure with OWN-30B geometry). The restore default stays 48: a refused restore falls back to prefill and loses nothing.

## 19. Host filesystem at-rest security (2026-10-13)

The host adapter (`host_fs.rs`) writes raw K/V blocks to disk. K/V is computed from user prompts, so an object file is sensitive user data. Before this section the adapter created directories `0755` and files `0644` (under the usual umask), accepted an existing root whatever its owner, mode or type, and followed symlinks on every open. Tests: `crates/aien-kv-store/tests/host_fs_security_tests.rs` (red before the fix, green after).

### 19.1 Threat model

- **In scope**: other local users on the same machine (other uids) reading, replacing or redirecting store files, including through symlinks or a shared parent directory.
- **Out of scope**: a process running as the same uid as the runtime (it can read the runtime's memory anyway) and root. Physical theft of the disk is covered only by encryption at rest (19.6), which the host adapter does not provide.

### 19.2 Modes

Every store directory (root, `objects/`, `tmp/`, all 256 shards) is created `0700` through `DirBuilder` with mode `0700` and then chmodded to `0700`, so a permissive umask cannot widen it and a narrow umask cannot lock the runtime out. Every store file (temps, `STORE_ID.part`, fault-injection writes) is created through one helper, `create_private_file`: mode `0600`, `O_NOFOLLOW | O_CLOEXEC`, then `fchmod` to `0600` if the result differs. Missing ancestors of the root, when the root is created recursively, get `0700` subject to the umask. VERIFIED by the umask tests, which re-run the creation test in a child process under umask `000` and `277`.

### 19.3 Ownership, symlink and refusal checks

`check_private_dir` uses `lstat` and requires: a real directory (not a symlink), owner = effective uid, no group or other permission bit. An existing directory that fails is refused with `PlatformError::AccessDenied`; the adapter never chmods an existing directory. Every existing directory (root, `tmp/`, `objects/`, every existing shard) is checked **before anything is created**, so a refused store is left exactly as it was (test: a shared last shard `ff` leaves no new entry).

### 19.4 Ancestor rule

The root's parent (the nearest existing ancestor when the root does not exist yet) must be a directory owned by the effective uid or by root, and if group or others can write it, it must carry the sticky bit (like `/tmp`, `1777`). Otherwise `AccessDenied`. Reason: whoever can write the parent can rename the root away and put their own directory in its place; the sticky bit stops other users renaming or removing entries they do not own. Only the immediate parent is checked; ancestors above it need not be `0700`.

### 19.5 Reads, foreign files and legacy objects

Every read (`STORE_ID`, the recovery header read, the load path) opens with `O_NOFOLLOW | O_CLOEXEC | O_NONBLOCK` and then checks the descriptor with `fstat`: regular file, owner = effective uid. `STORE_ID` must also be exactly 16 bytes; anything else puts the store in `Failed`, as an unreadable `STORE_ID` already did.

A symlink, a non-regular file or another owner's file inside a shard is a `foreign_file` under the existing section 4.1 rule: counted, **left alone, never read, never served, never deleted** (AIEN did not write it). The load path treats such a file as not found and drops it from the index. No new deletion policy is introduced.

**Legacy objects**: an object AIEN wrote before this section (owner = effective uid, mode `0644` or `0664`) is tightened to `0600` with `fchmod` on the descriptor during recovery, then validated and served as before. Rejecting them would throw away a valid cache for no security gain once the directories are private. If the `fchmod` fails, the file is counted as unreadable (kept, not served). `STORE_ID` is tightened the same way.

Crash recovery is unchanged: a leftover temp or `STORE_ID.part` belongs to AIEN in a verified private `tmp/`, so creation truncates it through `O_NOFOLLOW` instead of failing, and recovery still unlinks every `*.part` (4.1 step 2). A create that fails (for example on a planted symlink, `ELOOP`) leaves the path alone; the adapter only unlinks a temp its own successful create made.

### 19.6 Residual race (why no per-operation dirfd/openat)

Checks use paths, so a component could in principle be swapped between the check and the use. Once the root is verified as a real `0700` directory owned by the effective uid, under a trusted parent (19.4), no other non-root uid can create, rename or remove anything inside it, so only the same uid could perform the swap, and the same uid is out of scope (19.1). Final-component opens use `O_NOFOLLOW` and descriptor checks anyway. Per-operation `openat` on held directory descriptors is therefore not required; it becomes required if the threat model ever admits untrusted writers inside the root.

### 19.7 Encryption at rest

The host adapter does **not** encrypt. Object files are plaintext K/V protected only by file permissions; anyone with root, or with the raw disk, can read them. Encryption belongs in the record payload, as on the native path, where every slot is an M5 envelope bound to namespace, chunk key, slot and incarnation (section 2.5, section 9 item 5). A host equivalent (an encrypted payload format, `backend_format` other than 1, with a key that exists only while the runtime is unlocked) is **required before multi-tenant use** and before storing on media that leaves the operator's control. Encryption does not give tenant isolation by itself: one runtime key decrypts every tenant's objects; isolation needs per-tenant keys or namespaces (19.9) plus authorization at lookup.

### 19.8 Lifecycle deletion (current behaviour, VERIFIED in `host_fs.rs`)

- `delete`, eviction, purge of unindexed files and corrupt objects found by recovery or load: the file is unlinked on the worker.
- Shutdown, drop and process exit: published objects stay on disk by design (the store is persistent); recovery re-indexes them at the next open.
- Unlink only removes the name. The adapter does not overwrite or discard the blocks, so deleted K/V remains on the device until the filesystem reuses the space (and on SSDs possibly longer, because of wear levelling). Only encryption with key destruction (19.7) makes deletion final.
- Uninstall: no code removes `AIEN_KV_STORE_DIR`; the operator must delete the directory. UNKNOWN/TODO: an uninstall or `kv-store wipe` command.

### 19.9 Namespace isolation (known gap)

The namespace id has no tenant or user field. Its `sharing_namespace` field (1.2 field 13) comes on the host from one process-wide setting, `AIEN_KV_STORE_NAMESPACE`, so every request served by one runtime shares one namespace and can hit another request's cached prefix (a timing and content side channel between users of the same runtime). The native mapping (section 9) binds the subject id; the host path does not. TODO: a per-tenant or per-subject namespace component on the host, decided before any multi-tenant deployment. Until then the supported deployment is one trust domain per runtime and store directory.

## 20. Encryption at rest of saved KV files (#506, policy from the #498 resolution)

**In plain words.** Every saved conversation memory file is locked with a secret key before it touches the disk. Without the key, or if a single byte was changed, or if a file was moved to pose as another conversation, the file is thrown away and the conversation is simply recomputed. Nothing is ever restored from a file that does not prove itself.

**Policy (binding, locked by Drake in #498).** Authenticated encryption (AEAD), a fresh random nonce per file, the identity header bound as associated data, the key loaded from atlas-vault into process memory only (never written to disk or logs). On a decrypt failure, a wrong key, tampered bytes, a swapped identity header or a missing key: discard and recompute, never restore. Key rotation: old files are simply discarded.

**Cipher: AES-256-GCM** (ring 0.17, already in the lock file). Host micro-benchmark (`tests/seal_bench.rs`, ignored test, release build, in place, one thread, this Spark's Grace CPU cores Cortex-X925/A725 with the ARMv8 AES and PMULL extensions; which core type ran it is UNKNOWN):

| buffer | AES-256-GCM seal / open | ChaCha20-Poly1305 seal / open |
|---|---|---|
| 1 MiB | 7715 / 7847 MiB/s | 1956 / 1948 MiB/s |
| 16 MiB | 7694 / 7839 MiB/s | 1955 / 1949 MiB/s |

AES-256-GCM is about 4 times faster (VERIFIED on the host, 2026-10-13). What this adds to a real restore on the Spark under load is INFERRED (roughly 0.13 ms per MiB at the rate above) and is to be measured on the chip run, not claimed here.

**Format (host objects; amends 2.1 and 2.4).** The file stays `header region (4096 B) || payload`, same size, payload still at offset 4096. The payload bytes are ciphertext (GCM is length-preserving). The last 32 bytes of the region are the **seal slot**: offset 4064 magic `AKS`, 4067 sealed format version `2` (format 1 was the plaintext layout), 4068 nonce (12 bytes, from the OS random source per file), 4080 tag (16 bytes). Associated data is region bytes 0 to 4067: the 256-byte identity header (namespace id, chunk key, parent, geometry, payload root, header checksum), the segment digest table, the padding, and the magic and version. Consequences:

1. The digest table now ends at 4064, so **`MAX_SEGMENTS` is 119** on the host (`(4064 - 256) / 32`), not 120 (amends 2.1 item 3; the default chunk is 16 blocks). `BadPadding` covers bytes from the end of the table to 4063; the seal slot is checked by the seal module, not as padding.
2. The 256-byte header, its version (hashed into every namespace id, 1.2) and the golden vectors are unchanged, so native and host still share them; the version bump lives in the seal slot.
3. Recovery still reads only the region: a region without the sealed marker (an old plaintext file) is deleted as corrupt. Headers are not authenticated at recovery; an object sealed under another key, or tampered, is indexed and then deleted at its first load (same as the corrupt-payload limit above).
4. A load reads the whole payload, authenticates and decrypts it, and only then checks the segment digests and writes blocks to the destination. Nothing from a refused object ever reaches the pool (this replaces the segment-by-segment landing of 15.4 on the host; the per-segment digests remain as a second check).

**Key.** `AIEN_KV_STORE_KEY`, 64 hex digits (32 bytes), exactly; any other length or character is `key_malformed`. Started as `atlas-vault run AIEN_KV_STORE_KEY -- <daemon>`. `KvPersistSettings::from_env` reads it and removes it from the process environment (so child processes do not inherit it); `SealKey` holds it in a buffer zeroed on drop; its `Debug` prints `SealKey(redacted)` and refusals carry no key text. With `AIEN_KV_STORE_DIR` set and the key missing or malformed, persistence is refused: no store opened, nothing written, no park, no restore, plain prefill; the status reports `kv_store_state = refused`, `kv_store_refused_reason` and `kv_store_key_refused_total`. `HostFsStore::open` without a key refuses with `AccessDenied` before creating anything. With the store dir unset nothing changes. Loads refused by authentication count in `kv_store_seal_rejected_total`.

**Tests** (red dc34c99, then green): `seal_tests` (round trip, fresh nonce per file over 512 seals, wrong key, tampered ciphertext byte, tampered tag or nonce, swapped identity header, old plaintext format and other version, malformed and missing key, redacted Debug, environment removal), `host_fs_seal_tests` (the same refusals end to end through the store: the load fails `InvalidLayout`, the destination is untouched, the object is deleted; old plaintext file deleted at recovery; a header rewritten in place, self-consistent, refused at load; open without a key refused), `persistence_refused_without_a_valid_store_key_runs_plain_and_says_why` (runtime).

**Limits.**
- Metadata stays visible: file names (chunk keys), sizes, timestamps, the header and the segment digests of the plaintext. Someone who can read the store and guess a prompt can confirm the guess through the chunk key (1.3) as before; what is protected is the K/V content itself.
- Random 96-bit nonces under one key: the usual bound is about 2^32 files per key (NIST SP 800-38D); rotate the key long before that (a rotation discards the old files).
- The key schedule `ring` builds per seal or open is not zeroed by ring (UNKNOWN residue on the worker's stack). `remove_var` does not erase the original process environment block, which `/proc/<pid>/environ` still shows to the same user (INFERRED from Linux behaviour; not tested here). `remove_var` runs at settings read, at daemon start before the storage worker exists; other threads reading the environment at that moment would race (INFERRED low risk).
- File permissions, create-new temp files and no-follow opens are section 19 (#511); this section only covers encryption.
