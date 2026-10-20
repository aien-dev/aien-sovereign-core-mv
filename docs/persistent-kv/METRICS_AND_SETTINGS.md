# Persistent KV: metrics and settings

Companion to ADR 0003 (revised after the independent review). Names are final; values are defaults to be confirmed by C6/C7 (host) and N3 (native). "Where reported" names the exact surface. Every metric is PLANNED until its cut merges with a receipt; the host metrics implemented by C7 are listed under "Implemented on the host (C7)" below.

## Settings (host, environment variables, same style as `AIEN_KV_CONTEXT_TOKENS` in `crates/aien-runtime/src/shared_kv.rs:306`)

| name | type and unit | default | effect | parsed in |
|---|---|---|---|---|
| `AIEN_KV_STORE_DIR` | path | unset = persistence off | root of the host object store (ADR 2.4). Must be private: created 0700 if missing; an existing root, `tmp/`, `objects/` or shard that is a symlink, not ours, or reachable by group or others is refused (`AccessDenied`), and so is a root whose parent others can write unless the parent is sticky (ADR 19). Unset: no store, no park, no restore, zero behaviour change | `shared_kv.rs` |
| `AIEN_KV_STORE_KEY` | 64 hex digits (32 bytes) | unset = persistence refused when `AIEN_KV_STORE_DIR` is set | the key every saved KV file is sealed with (AES-256-GCM, ADR 0003 section 20). Pass it with `atlas-vault run AIEN_KV_STORE_KEY -- <daemon>`; it is read once, removed from the environment and never printed. Missing or malformed with the store dir set: fail closed (no store opened, nothing written, no park, no restore, plain prefill) and the status reports `kv_store_state = refused` with the reason. Ignored when the store dir is unset | `seal.rs`, `shared_kv.rs` |
| `AIEN_KV_STORE_NAMESPACE` | string, at most 64 bytes | `default` | sharing namespace (ADR 1.2 field 13) | `shared_kv.rs` |
| `AIEN_KV_STORE_QUOTA_BYTES` | bytes, decimal integer | 8 GiB | disk quota counting every object found at recovery | `shared_kv.rs` (checked by name, applied to `HostFsConfig`; default in `host_fs.rs`) |
| `AIEN_KV_STORE_MAX_OBJECTS` | count | 65536 | index cap at recovery and at runtime (ADR 4.1) | `shared_kv.rs` (checked by name, applied to `HostFsConfig`; default in `host_fs.rs`) |
| `AIEN_KV_STORE_MAX_RESTORE_OPS` | count | 48 | restore-class jobs between Submitted and terminal completion (per-class reservation, ADR 7.1); `restore_ops + park_ops` never exceeds the total cap | `worker.rs` |
| `AIEN_KV_STORE_MAX_PARK_OPS` | count | 64 | park-class jobs, reserved so restores cannot starve parks (ADR 7.1); 64 admits 8 conversations finishing together with up to 8 chunk parks each (ADR 18.4) | `worker.rs` |
| `AIEN_KV_STORE_MAX_WAITERS_PER_JOB` | count | 16 | coalesced waiters on one in-flight load; past it `attach_waiter` returns QueueFull (ADR 7.3) | `worker.rs` (`WorkerConfig::from_lookup`, enforced in `coalesce.rs`) |
| `AIEN_KV_STORE_MAX_INFLIGHT_BYTES` | bytes | 256 MiB | payload bytes between Submitted and Published | `worker.rs` |
| `AIEN_KV_CHUNK_BLOCKS` | blocks | 16 | chunk size in KV blocks (part of the namespace; changing it orphans objects) | `shared_kv.rs` |
| `AIEN_KV_RESTORE_MAX_BLOCKS` | blocks | pool / 4 | upper bound on holder blocks allocated for restores at once | `shared_kv.rs`, applied in `kv_residency.rs` |
| `AIEN_KV_RESTORE_RETRY_WINDOW_MS` | ms | 60000 | after two failed restores of one chain head within the window, prefill without lookup | `shared_kv.rs`, applied in `kv_residency.rs`. The chain head is the first chunk key of the prompt (C7 choice, ADR silent); a failure is a restore transaction that ended `Failed` (load failure or S3 refusal); at most 4096 heads are remembered |
| `AIEN_KV_STORE_RECOVERY_SLICE_MS` | ms | 50 | time slice of each resumable `recover_step`; admission stays closed until the walk is complete, however many slices it takes (ADR 4.1) | `shared_kv.rs` (checked by name, applied to `HostFsConfig`); the slices run on the storage worker thread, never on the step loop |
| `AIEN_KV_STORE_DRAIN_MS` | ms | 2000 | shutdown intake deadline; the return carries a `DrainReport`, nothing is freed to meet it (ADR 7.5) | `worker.rs`; the daemon drains after its step loop stops and logs the `DrainReport` (`server.rs`) |
| `AIEN_KV_ODIRECT` | `1`/`0` | `0` until X2 proves it | read restore payload with `O_DIRECT` into pool blocks (ADR 15.3); refusal falls back to buffered and counts | `host_fs.rs` |
| `AIEN_KV_PREFETCH_TTL_MS` | ms | 30000 | a predictive-restore hint that is not followed by a real request in this time is cancelled and its holder freed (ADR 15.5, cut X6) | `kv_residency.rs` |
| `AIEN_KV_PARK` | `1`/`0` | `1` when the store dir is set | park finished turns | `shared_kv.rs` |
| `AIEN_KV_RESTORE` | `1`/`0` | `1` when the store dir is set | restore at submit | `shared_kv.rs` |

All settings above are read once at daemon start (`KvPersistSettings::from_env`, called by `aien-cli` before the model pool is built). Every value is checked whether or not persistence is on, and a bad value stops the start with a message that names the setting (test `bad_persistent_kv_settings_fail_startup_with_the_setting_name`). With `AIEN_KV_STORE_DIR` unset nothing is created and nothing changes (test `persistence_off_when_store_dir_unset_changes_nothing`). With it set, the daemon also needs the weights digest of the loaded model; without one the start is refused.

Existing settings that this design reads and never redefines: `AIEN_KV_CONTEXT_TOKENS` (restorable prefix is capped at the operator context), `AIEN_ALLOW_UNCHECKED_KV_MEMORY` (untouched), `AIEN_STEP_LOG` (the step line gains the counters below), `AIEN_RUNTIME_STATE_DIR` (not used as a default store root: persistence must be opted into explicitly).

## Settings (native, compile-time or slab superblock)

| name | unit | default | where |
|---|---|---|---|
| `KV_SLAB_SLOTS` | count | sized at format from region units / slot_units | slab superblock |
| `KV_SLAB_SLOT_UNITS` | 4096-byte units | 1 + ceil(max_chunk_bytes / 4080) | slab superblock |
| `KV_PUMP_BUDGET_CMDS` | disk commands per pump call | 2 | `infer.c` (between-steps hook and the idle pump, ADR 9 item 3) |
| `KV_PARK_STAGE_BYTES` | bytes | one chunk of sealed envelopes plus export (to be sized in N3) | `kv_store.c` (bound on queued sealed data, ADR 10) |
| `KV_SLAB_PI_UNITS` | 4096-byte units | 32768 (128 MiB), inside the Aienux partition view | Pi superblock and `KV.MAP` (ADR 16.2) |
| `KV_QUEUE_DEPTH` | requests | `DISK_QUEUE_DEPTH` (32) | `disk.h` (reused, not redefined) |

## Metrics (host)

Reported in three places: `RuntimeStatusReport` (spine.rs `status_report`, serialized to the `status` IPC reply), the `AIEN_STEP_LOG=1` step line (counters only), and the benchmark JSON (C9). Units are in the name or the column.

| name | unit | kind | meaning | where |
|---|---|---|---|---|
| `kv_pool_allocated_bytes` | bytes | gauge, constant after start | the one pool allocation (`KvPoolPlan.total_bytes`); never changes when blocks are freed | status |
| `kv_pool_total_blocks`, `kv_pool_free_blocks`, `kv_pool_used_blocks` | blocks | gauge | existing counts, renamed to make the "free does not release RAM" point explicit | status (existing `free_kv_blocks`/`total_kv_blocks` kept as aliases) |
| `kv_pool_holder_blocks` | blocks | gauge | blocks referenced by storage holders (park sources, restore destinations) | status, step line |
| `kv_restore_inflight_blocks` | blocks | gauge | holder blocks of restores not yet adopted | status, step line |
| `kv_store_inflight_ops`, `kv_store_inflight_bytes` | count, bytes | gauge | queue occupancy between Submitted and terminal completion, with the per-class split (`kv_store_restore_ops`, `kv_store_park_ops`) | status, step line |
| `kv_store_reserved_bytes`, `kv_store_unindexed_bytes`, `kv_store_unindexed_objects` | bytes, count | gauge | submit-time reservation (payload plus 4096-byte header region); recovery remainder not yet indexed (ADR 4.1) | status |
| `kv_store_state` | enum `recovering`, `open`, `degraded`, `failed`, `refused` | gauge | `degraded` after an `Indeterminate` publication (ADR 2.4); `refused` when the store key is missing or malformed (ADR 20) | status |
| `kv_park_indeterminate_total`, `kv_store_shutdown_unfinished_jobs`, `kv_shutdown_unfinished_bytes` | count, bytes | counter, once | publication outcomes and the shutdown `DrainReport` (ADR 7.5) | status, shutdown log line |
| `kv_odirect_fallback_total` | count | counter | `O_DIRECT` reads that fell back to buffered (ADR 15.3) | status |
| `kv_prefetch_hint_total`, `kv_prefetch_hit_total`, `kv_prefetch_wasted_bytes_total` | count, bytes | counter | predictive restore (ADR 15.5, cut X6) | status |
| `kv_park_streamed_chunks_total`, `kv_park_bytes_at_finish` | count, bytes | counter, per turn | streaming park and the work left at finish (ADR 15.6, cut X6b) | status, bench |
| `kv_host_tail_scratch_bytes` | bytes | gauge, constant | the one-block tail scratch (ADR 2.2); the only host staging memory, and none at all for full blocks | status |
| `kv_verify_cpu_us_per_mib` | microseconds per MiB | per-run | segment verification CPU cost (ADR 15.4) | bench JSON |
| `kv_store_persisted_bytes`, `kv_store_persisted_objects` | bytes, count | gauge | index totals, including foreign-namespace objects | status |
| `kv_store_quota_bytes`, `kv_store_max_objects`, `kv_store_over_cap` | bytes, count, bool | gauge | limits and the over-cap flag | status |
| `kv_park_jobs_total`, `kv_park_bytes_total`, `kv_park_failed_total`, `kv_park_dedup_total` | count, bytes | counter | park outcomes | status, step line |
| `kv_park_latency_us_p50`, `kv_park_latency_us_p95` | microseconds | histogram summary (bounded reservoir of 1024) | Submitted to Published | status |
| `kv_restore_jobs_total`, `kv_restore_bytes_total`, `kv_restore_failed_total`, `kv_restore_coalesced_total`, `kv_restore_fallback_prefill_total` | count, bytes | counter | restore outcomes; fallback = miss or failure turned into plain prefill | status, step line |
| `kv_restore_retry_suppressed_total` | count | counter | requests prefilled without a lookup because their chain head failed twice within `AIEN_KV_RESTORE_RETRY_WINDOW_MS`; each is also a fallback | status, step line |
| `kv_restore_latency_us_p50`, `kv_restore_latency_us_p95` | microseconds | histogram summary | Submitted to adopted (S3) | status |
| `kv_restore_tokens_total`, `kv_prefill_tokens_total` | tokens | counter | tokens restored versus prefilled, so the saved fraction is derivable | status, bench |
| `kv_store_stale_completions_total`, `kv_store_cancelled_total`, `kv_store_queue_full_total`, `kv_store_disk_full_total`, `kv_store_evicted_bytes_total` | count, bytes | counter | ADR 5.3, 5.5, 7.1, 7.4 | status |
| `kv_store_refused_reason` | `key_missing`, `key_malformed`, empty | gauge | why persistence was refused at start (ADR 20) | status |
| `kv_store_key_refused_total`, `kv_store_seal_rejected_total` | count | counter | store-key refusals; loads whose authenticated decryption failed (wrong key, tampered bytes, swapped header), each object discarded and recomputed (ADR 20) | status |
| `kv_recovery_elapsed_us`, `kv_recovery_indexed`, `kv_recovery_corrupt_deleted`, `kv_recovery_tmp_deleted`, `kv_recovery_foreign_namespace` | microseconds, counts | gauge, set once at open | `RecoveryReport` | status, startup log line |
| `ttft_cold_us`, `ttft_restored_us` | microseconds | per-request measurement | first token latency with `kv_restore_tokens` = 0 versus > 0 | bench JSON, step line per request when `AIEN_STEP_LOG=1` |
| `decode_tokens_per_s` | tokens/second | per-run | aggregate decode throughput | bench JSON |
| `cache_state` | enum `page_cache`, `device`, `unknown` | label | how the restore bytes were served; `device` only after the drop-caches step ran and was recorded | bench JSON |
| `platform` | enum `hosted`, `native_emulator`, `native_physical_pi5`, `native_physical` | label | where the numbers came from; `native_physical_pi5` is printed "native physical (Pi 5)" and comes only from the Pi lane run (ADR 16, cut P3) and never stands in for Spark; `native_physical` (Spark) only from an authorized Spark run (NOT_RUN under aienos#336) | bench JSON, evidence ledgers |
| `queue_depth` | count | label | submission depth used by a baseline sweep (ADR 15.1) | bench JSON |
| `completed_requests`, `failed_requests`, `request_latency_us_p50`, `request_latency_us_p95` | count, microseconds | per-run | required by the benchmark schema (acceptance B6) | bench JSON |
| `peak_allocated_bytes`, `peak_staging_bytes`, `peak_rss_bytes` | bytes | per-run | memory peaks required by B6 | bench JSON |

### Implemented on the host (C7)

Reported now (host, `pkv/integration`), names exactly as in the table above:

- `status` reply (`RuntimeStatusReport`, built in spine.rs `status_report`): `kv_pool_allocated_bytes`, `kv_pool_total_blocks`, `kv_pool_free_blocks`, `kv_pool_used_blocks`, `kv_pool_holder_blocks` always; with persistence on, `kv_persist` carries `kv_store_state`, `kv_store_persisted_bytes`, `kv_store_persisted_objects` (indexed plus unindexed), `kv_store_quota_bytes`, `kv_store_max_objects`, `kv_store_over_cap`, `kv_store_reserved_bytes`, `kv_store_unindexed_bytes`, `kv_store_unindexed_objects`, `kv_store_inflight_ops`, `kv_store_inflight_bytes`, `kv_store_restore_ops`, `kv_store_park_ops`, `kv_restore_inflight_blocks`, the `kv_park_*` and `kv_restore_*` counters (jobs, bytes, failed, dedup, indeterminate, coalesced, fallback, retry suppressed, tokens) and `kv_store_stale_completions_total`, `kv_store_cancelled_total`, `kv_store_queue_full_total`, `kv_store_refused_reason`, `kv_store_key_refused_total`, `kv_store_seal_rejected_total`. With the store dir set and the key refused, `kv_persist` carries only `kv_store_state = refused`, the reason and the refusal counter (test `persistence_refused_without_a_valid_store_key_runs_plain_and_says_why`). Test `metrics_report_allocated_bytes_separately_from_free_blocks`.
- `AIEN_STEP_LOG=1` step line (server.rs `kv_step_log_fields`): `kv_park_jobs_total`, `kv_park_failed_total`, `kv_park_dedup_total`, `kv_park_indeterminate_total`, `kv_restore_jobs_total`, `kv_restore_failed_total`, `kv_restore_fallback_prefill_total`, `kv_restore_retry_suppressed_total`, `kv_store_stale_completions_total`, `kv_pool_holder_blocks`, `kv_restore_inflight_blocks`. Test `step_log_line_carries_kv_counters`.
- Shutdown log line (server.rs `kv_shutdown_log_line`): `kv_store_shutdown_unfinished_jobs`, `kv_shutdown_unfinished_bytes` and every unfinished job.
- `kv_park_indeterminate_total` counts every park terminal whose publication is `Indeterminate`, whatever the terminal (the fake reports it on a failed terminal). Test `indeterminate_park_publication_is_counted`.

Still PLANNED on the host: latency summaries, `kv_store_evicted_bytes_total`, `kv_store_disk_full_total`, the recovery report fields, `kv_odirect_fallback_total`, prefetch and streaming-park metrics, `kv_host_tail_scratch_bytes`, `kv_prefill_tokens_total`, and every bench JSON field (C9).

## Metrics (native)

Printed as one `kv:` line by the stage and recorded in `docs/evidence/kv-native.md` (aienos). Counters are `uint64_t` in the slab state, no clock dependency except `ck` microsecond timestamps where the kernel has them.

| name | unit | meaning |
|---|---|---|
| `kv_slab_slots_total`, `kv_slab_slots_used` | slots | capacity and occupancy after recovery and at report time |
| `kv_slab_used_bytes` | bytes | sum of payload_len of valid slots, any namespace |
| `kv_slab_bounce_bytes` | bytes | the driver's bounce buffer size: one staging copy among several, listed with the others below (the earlier claim that it was the only staging memory was wrong, ADR 10) |
| `kv_stage_export_bytes` | bytes | plaintext snapshot of the chunk being sealed at the park point (ADR 9 item 7) |
| `kv_stage_cipher_bytes` | bytes | queued sealed envelopes, bounded by `KV_PARK_STAGE_BYTES` |
| `kv_stage_env_bytes`, `kv_stage_plain_bytes` | bytes | contiguous envelope and decrypt output of one restore chunk |
| `kv_crypto_scratch_bytes` | bytes | the private 64 KiB scratch inside `m5_envelope_open` (stack of the call) |
| `kv_stage_queued_bytes`, `kv_stage_peak_bytes` | bytes | sum owned by queued jobs, and the high-water of all staging together; the report line prints each component separately and the peak |
| `kv_restore_copy_bytes` | bytes | payload bytes the CPU copied during restore; 1x today (gather), 0 for payload after X4 (ADR 15.3) |
| `kv_session_kv_bytes` | bytes | `2 * n_layer * capacity * kv_dim * 4` for the open session (allocated memory) |
| `kv_park_cmds`, `kv_restore_cmds` | disk commands | commands issued by the pump per job |
| `kv_park_us`, `kv_restore_us` | microseconds | Submitted to Durable (park) and Submitted to imported (restore) |
| `kv_pump_calls`, `kv_pump_cmds_max` | count | between-steps calls and the largest budget actually used |
| `kv_steps_during_park` | count | decode steps that completed while a park job was in flight (the interleaving evidence of ADR 9.3) |
| `kv_recovery_us`, `kv_recovery_slots_valid`, `kv_recovery_slots_empty`, `kv_recovery_slots_foreign` | microseconds, counts | the bounded header scan |
| `kv_fail_stop` | bool | the queue fail-stopped on a device error since open |
