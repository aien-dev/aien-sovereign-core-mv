# Persistent KV program: implementation cuts

Binding design: `docs/adr/0003-persistent-kv-identity-storage-contract-and-ownership.md` (ADR 0003, revised after the independent review, ADR section 17). Issues: sovereign-core #484, #485; aienos #367. Each cut is one reviewable PR with its own red-before-green tests. "Red" means the test is committed first and fails for the stated reason; "green" is the implementation commit in the same PR. Acceptance boxes are quoted by id from the tables at the end. Every test name below is **PLANNED** (designed, no receipt yet) unless a line says it exists. C1 exists (merged at ce5d9eb, red tests at 288c18c); everything else is not started.

## Plan corrections (against `~/handoffs/2026-10-13-persistent-kv-program.md`)

1. **The order "P2 all of #484, then P3 #485" is wrong at the runtime seam.** Parking and restoring in the runtime (#484 scope) cannot be wired without the bounded queue, the worker and the residency state machine (#485 scope): a park that blocks the spine or a restore with no incarnation check would be thrown away later. So the trait and the deterministic fake adapter (C2) come right after identity (C1), the worker and queues (C5) come before the runtime wiring (C6), and C6 is where #484 and #485 meet. The handoff's note that #485 "may start in parallel against a fake adapter" is kept and made concrete: C5 and C8 develop against C2's fake.
2. **The #484 "core" PR in the plan is too big to review** (identity + trait + adapter + recovery + park/restore + recompute + metrics). It is split into C1, C3, C4, C6, C7. Every cut stays under roughly 1500 changed lines including tests.
3. **The native pin goes in aienos, not in sc**, and it pins bytes, not a crate: the header codec and slot header test vectors (section "Pin").
4. **Recompute-on-miss is not a separate feature**: a miss or a failed restore leaves `prompt_tokens_prefilled = 0`, which is today's prefill path (ADR 0003 section 6.3). It is tested in C6, not built.
5. Benchmarks (C9) run on the host CPU backend by default; the GB10 leg is a separate, optional evidence run that must not take the GB10 lock away from the OWN-30B and 8B lanes named in the handoff.
6. **Review corrections (this revision).** (a) C1 as merged does not bind the compute configuration (finding 1) and has a 256-byte file header; **C1b** changes it before anything builds on it (ADR section 12). (b) The park cannot hook the spine's `Finished` branch because the scheduler frees KV first (finding 2); the scheduler and backend seam is its own cut, **C6a**, ahead of the runtime wiring in C6. (c) Holders need `fork_range` and `assemble_table`, not `fork_sequence` (finding 6), so C3 grows. (d) Native performance work (ADR section 15) and the Pi 5 physical lane (ADR section 16) are new cuts X1 to X7 and P1 to P3 at the end of this file. (e) The labelled guarantee matrix is a deliverable of C7, not prose.

## Dependency graph

```
sovereign-core (arrow means "must merge first"):
C1 -> C1b -> C2 -> C5 -> C6
C1b -> C3 -> C6a -> C6
C2 -> C4 -> C6
C6 -> C7, C6 -> C8 -> C9
C9 + C4 -> X1 -> X2 -> X5
C6 + C8 -> X6;  C6 + C3 -> X6b

aienos:
C1b -> N1 -> N2 -> N3 -> N4 -> N5
N4 -> X3 -> X4 -> X7 (conditional, needs X5 and a profile)
N2 + N4 -> P1 -> P2 -> P3 (Drake authorizes; Pi must be free)
```

Parallel lanes once C2 is merged: {C3, C4} and {C5} in parallel; {N1, N2} in parallel with C3..C5. C6a can start as soon as C3 lands (it needs `fork_range`) and in parallel with C4/C5. C8 and C9 after C6. N3..N5 need only C1b's vectors, not any later sc cut.

## sovereign-core cuts

### C1: identity, header codec, validation (#484) (MERGED at ce5d9eb, VERIFIED)

- Files: new crate `crates/aien-kv-store/` (`Cargo.toml`, `src/lib.rs`, `src/identity.rs`, `src/record.rs`, `src/validate.rs`, `src/error.rs`), `tests/golden/kv_record_v1.json`, `tests/golden/README.md`.
- Existing red tests: `identity_vectors_match_golden`, `namespace_changes_when_any_field_changes` (13 fields), `chunk_key_chain_depends_on_parent_and_position`, `header_roundtrip_256_bytes`, `validate_refuses_before_any_allocation`.
- Acceptance: A3 (identity half), D1 (bad length/checksum half, shared vectors).

### C1b: compute-config digest and 4096-byte header region (#484, review finding 1, ADR section 12)

- Files: `crates/aien-kv-store/src/{identity,record,validate,error}.rs`, regenerated `tests/golden/kv_record_v1.json` (4 inputs gain compute fields, 2 new vectors), `tests/golden/README.md` (new `vectors_sha256`). The exact byte and offset list is ADR section 12 (I1 to I7, H1 to H9).
- Red tests: `golden_vectors_include_compute_config_digest`, `rms_norm_eps_alone_changes_the_namespace`, `compute_config_digest_changes_with_each_field` (9 mutators), `namespace_changes_when_any_field_changes` (now 14 mutators), `header_struct_bytes_0_to_255_unchanged_by_the_region`, `header_region_roundtrip_4096`, `validate_refuses_nonzero_padding`, `validate_refuses_segment_root_mismatch`, `validate_refuses_more_than_120_segments`, `file_size_is_4096_plus_payload`, and `validate_refuses_before_any_allocation` extended to the three new errors.
- Acceptance: A3 (compute-config binding), D1 (shared vectors).

### C2: `KvStore` trait and deterministic fake adapter (#484, #485 shared; ADR section 13)

- Files: `crates/aien-kv-store/src/store.rs` (trait, `Completion`, `Terminal`, `PublishOutcome`, `StoreState`, `Capacity`, `RecoveryProgress`, `DrainReport`, stages), `src/fake.rs` (in-memory adapter on a virtual clock; faults: queue-full, disk-full, flush-fail, torn-write, delayed completion, duplicate completion, completion after cancel, `rename_ok_dirsync_fail`, `fsync_file_fail`, `stall_fsync`, `stall_eviction`, `unlink_fail`; poison-on-free destination check), `src/pin.rs` (`PinnedSource` with `tail_valid_tokens`, `PinnedDest` guards).
- Red tests: `fake_stages_are_ordered_submitted_completed_published_durable`, `fake_refuses_over_ops_and_over_bytes`, `fake_failed_store_leaves_index_unchanged`, `fake_cancel_before_complete_publishes_nothing`, `fake_cancel_after_complete_is_too_late`, `fake_same_key_store_dedups`, `fake_delete_skips_pinned`, `fake_submit_store_performs_no_io_and_no_eviction` (finding 7), `fake_reservation_prevents_overcommit_by_concurrent_stores`, `fake_every_job_yields_exactly_one_terminal_completion` (finding 5, 9), `fake_rename_ok_dirsync_fail_reports_indeterminate_and_serves_object` (finding 10), `fake_lookup_misses_and_submit_refused_while_recovering` (finding 8), `fake_poll_never_overflows_at_max_inflight` (finding 9), `fake_writes_to_a_freed_destination_fail_the_test`.
- Acceptance: B3 (queue-full, disk-full, cancel, duplicate at adapter level), A4 (adapter half).

### C3: kv-cache ownership primitives (#484, findings 5 and 6)

- Files: `crates/aien-kv-cache/src/lib.rs` (`fork_range(src, first_block, n_blocks, holder)`, `assemble_table(seq, &[holder], suffix_policy)`, `adopt_table(holder, seq)`, reserved holder slot `u32::MAX` rule and `holder_id(job)`, `holder_block_count()` for metrics; `KvBlock` unchanged), `crates/aien-kv-cache/tests/holder_ownership_tests.rs`.
- Red tests: `holder_fork_keeps_blocks_after_owner_free`, `holder_free_returns_blocks_only_at_zero_refcount`, `adopt_table_moves_table_and_keeps_refcounts`, `owner_append_on_shared_partial_block_copies_not_holder`, `shared_prefix_sibling_valid_after_holder_free`, `holder_id_never_collides_with_arena_ids`, `fork_range_forks_only_the_named_blocks_and_does_not_replace_a_live_child` (the property `fork_sequence` lacks), `assemble_table_orders_holders_and_refcounts`, `assemble_table_failure_leaves_every_holder_untouched`.
- Acceptance: A5, B2 (refcount half).

### C4: host filesystem adapter and recovery (#484; ADR sections 2.4, 4.1)

- Files: `crates/aien-kv-store/src/host_fs.rs` (feature `std`: precreated shard dirs with parent fsync, temp + fsync + rename + dir fsync with boundaries P1 to P5, worker-side reclaim, quota, resumable `recover_step`), `src/index.rs`, `crates/aien-kv-store/tests/host_fs_tests.rs`, `tests/host_fs_recovery_tests.rs`.
- Red tests: `store_then_recreate_adapter_serves_same_bytes` (A1 adapter-recreation row), `recovery_counts_foreign_namespace_objects_toward_quota`, `recovery_deletes_tmp_and_corrupt_headers_only`, `recovery_is_bounded_by_max_objects_and_sets_over_cap`, `recovery_reads_exactly_4096_bytes_per_object` (counting reader; the header region read), `enospc_mid_write_leaves_last_object_intact`, `fsync_failure_fails_job_and_publishes_nothing`, `truncated_payload_fails_on_load_and_is_deleted`, `lookup_is_index_only_no_io`, `fault_at_each_publication_boundary_leaves_consistent_outcome` (finding 10, A4), `recovery_keeps_admission_closed_until_walk_completes` (finding 8, A6), `deleting_one_indexed_object_does_not_reopen_admission_with_unscanned_remainder` (A6), `used_bytes_counts_header_region_and_physical_blocks`, `inflight_temp_bytes_are_reserved`, `tail_rows_above_token_count_are_zero_in_object` (finding 3), `eviction_runs_on_worker_not_in_submit` (finding 7).
- Acceptance: A1 (adapter recreation), A3 (payload half), A4, A6 (restart accounting), A7 (recreation row).

### C5: storage worker, bounded queues, fairness, coalescing, drain (#485; ADR section 7)

- Files: `crates/aien-kv-store/src/worker.rs` (thread, two intake queues, per-class admission reservation, 3:1 credit scheduler, completion slots reserved at submit, cancel flags checked per block, worker-side eviction, `shutdown(intake_deadline)`), `src/coalesce.rs` (landing-buffer coalescing, `attach_waiter` bounds), `crates/aien-kv-store/tests/worker_tests.rs` (against the fake).
- Red tests: `worker_bounds_ops_and_bytes_in_flight`, `worker_three_restores_then_one_park_when_both_waiting`, `worker_coalesces_same_key_loads_into_one_io`, `worker_cancel_removes_only_own_waiter`, `worker_stale_completion_is_dropped_by_incarnation`, `worker_never_touches_dest_after_cancel_ack`, `worker_backpressure_refuses_not_grows`, `sustained_restores_cannot_starve_parks` (finding 9, B4), `completion_queue_never_overflows_at_max_inflight` (B4), `coalesced_waiters_are_bounded` (B4), `shutdown_with_stalled_fsync_returns_by_deadline_and_reports_unfinished` (finding 11), `worker_shutdown_keeps_buffers_until_each_job_terminal`, `restore_txn_holder_survives_sibling_failure_until_all_jobs_terminal` (finding 5, B3).
- Acceptance: B3, B4 (bounded queues half), shutdown bullet of #485.

### C6a: scheduler and backend seam (#484, findings 2, 3, 4)

- Files: `crates/aien-scheduler/src/lib.rs` (`FinishHook` run inside the finish transaction before `free_sequence`; `computed_prefix(seq)`; `admit_restored_prefix(seq, n)` into `PrefillPending` with phase Prefill and counted in `running_sequences`; `AwaitingRestore`), `crates/aien-scheduler/src/sequence.rs`, `crates/aien-abi-core/src/lib.rs` (`restore_sequence` default `Err`), `crates/aien-inference-abi/src/transformer_backend.rs` (`restore_sequence`, same refusals as `fork_sequence`), `crates/aien-runtime/src/compute_config.rs` (`ComputeConfig::from(&ModelConfig)`, exhaustive destructuring, backend name and `KV_ARITH_EPOCH`), tests under `crates/aien-scheduler/tests/` and `crates/aien-runtime/tests/`.
- Red tests: `finish_transaction_forks_holder_before_free` (finding 2), `finish_hook_failure_still_frees_sequence`, `computed_prefix_excludes_uncomputed_last_token` (finding 3), `restored_prefix_does_not_enter_decode_before_remaining_prompt_prefilled` (finding 4), `restored_prefix_counts_in_running_sequences`, `awaiting_restore_is_not_admitted_until_released`, `modelconfig_field_mutation_changes_compute_digest` (finding 1: every `ModelConfig` field mutated, compiler-exhaustive struct), `arith_epoch_pinned_bytes` (finding 1).
- Acceptance: A2 (prerequisite), A3 (compute-config binding), A5.

### C6: runtime park at the finish transaction, restore at submit, residency states (#484 + #485 seam)

- Files: `crates/aien-runtime/src/kv_residency.rs` (states, epochs, lock rules of ADR 8), `crates/aien-runtime/src/spine.rs` (park via the C6a hook; restore in `submit_request`; `poll` of completions at the start of `step`; status fields), `crates/aien-runtime/src/shared_kv.rs` (store construction and settings), tests `crates/aien-runtime/tests/kv_park_restore_test.rs`.
- Red tests: `finished_turn_publishes_full_chunks_and_partial_tail`, `park_never_blocks_step`, `restore_caps_prefix_at_prompt_len_minus_one`, `restore_miss_falls_back_to_plain_prefill_with_no_refusal`, `restore_failure_frees_holder_and_prefills`, `stale_completion_cannot_touch_reused_sequence_id`, `awaiting_restore_is_not_admitted_until_all_chunks_completed`, `restore_refused_below_watermark_falls_back`, `multi_chunk_restore_failure_does_not_free_destination_in_use` (B3), `restore_txn_adopts_assembled_table_only_after_all_jobs_terminal`.
- Acceptance: A1 (restart row prerequisite), A2 (in-process half), A5, B1 (mechanism), B3 (stale completion through the runtime), B5 (routing).

### C7: real restart, greedy equality, metrics and settings, labelled matrix (#484)

- Files: `crates/aien-runtime/tests/kv_restart_test.rs` (child process via `std::env::current_exe`, pattern of `approved_crash_test.rs:80-89`), `crates/aien-runtime/src/spine.rs` (status fields), `crates/aien-runtime/src/server.rs` (step log counters), `docs/persistent-kv/METRICS_AND_SETTINGS.md` (kept in sync), `docs/FIRST_RUN.md` (one paragraph).
- Red tests: `process_restart_rediscovers_and_restores_identical_bytes` (A1), `greedy_continuation_after_restart_matches_uninterrupted_baseline` (A2), `wrong_model_digest_namespace_is_never_served_after_restart`, `different_rms_norm_eps_namespace_is_never_served_after_restart` (finding 1 end to end), `metrics_report_allocated_bytes_separately_from_free_blocks`, `guarantee_matrix_rows_are_labelled` (asserts every row of ADR section 11 carries one of PLANNED, TESTED(receipt id), NOT CLAIMED, NOT_RUN and that each TESTED row names a test that exists).
- Acceptance: A1, A2, A3, A6, A7 (restart row), A8.

### C8: concurrency proof through scheduler and backend (#485)

- Files: `crates/aien-runtime/tests/kv_concurrency_test.rs`, `crates/aien-scheduler/tests/kv_restore_admission_tests.rs`, fixtures under `crates/aien-runtime/tests/fixtures/kv/`.
- Red tests: `resident_sequence_emits_tokens_before_unrelated_restore_completes` (B1), `concurrent_park_and_restore_of_shared_prefix_preserve_bytes_and_refcounts` (B2), `fork_then_cancel_parent_during_park_keeps_child_valid` (B2), `duplicate_restore_requests_share_one_load` (B3), `coalesced_waiters_with_different_suffixes_get_independent_tables` (finding 6), `memory_pressure_restore_and_admission_make_progress_without_deadlock` (B4), `queue_full_and_disk_full_under_load_never_lose_resident_progress` (B3, B4).
- Acceptance: B1, B2, B3, B4, B5.

### C9: benchmark (#485)

- Files: `crates/aien-scheduler/src/bin/bench_persistent_kv.rs` (1, 2, 4, 8 conversations; 512 and 2048 token prompts; modes resident-only, cold-prefill, restore-from-page-cache, restore-from-device), `docs/persistent-kv/BENCH.md`, `benchmarks/data/` entry per ADR 0001.
- Output schema (B6): per mode and concurrency, `completed_requests`, `failed_requests`, request latency `p50_us` and `p95_us`, `ttft_us` p50 and p95, `decode_tokens_per_s`, `peak_allocated_bytes` (pool constant plus holders), `peak_staging_bytes`, `peak_rss_bytes`, `cache_state`, `platform`, `queue_depth`.
- Red tests: `bench_output_requires_completed_requests_latency_percentiles_and_peaks` (B6), `bench_binary_refuses_to_report_without_mode_and_cache_state_labels` (B7), `bench_refuses_mixed_cache_state_summary`, `bench_records_not_run_rows_for_native_physical` (B8).
- Acceptance: B6, B7, B8.
- Status (pkv/c9-bench, host only): red eca335c, green a2cfd1e and 5a07c36; the four red tests exist and pass in `crates/aien-kv-bench/tests/bench_persistent_kv_test.rs`. Deviation: the binary is `crates/aien-kv-bench/src/bin/bench_persistent_kv.rs` (its own crate), because it drives aien-runtime, which depends on aien-scheduler. One host run is recorded in `BENCH.md`; restore-from-device and the native physical rows are NOT_RUN.

## aienos cuts

### N1: native header codec, slot header, and golden vectors (the pin) (#367)

- Files: `native/kv/kv_record.h`, `native/kv/kv_record.c` (256-byte header encode/decode/validate, compute-config-aware namespace, key derivation with argus sha256, **slot header unit** per ADR 2.5), `native/kv/tests/kv_record_test.c`, `native/kv/tests/golden/kv_record_v1.json` (copied from sc C1b), `native/kv/tests/golden/kv_slot_v1.json` (slot header and envelope id/nonce vectors, generated in aienos with an independent cross-check), `native/kv/CONTRACT.lock`, `native/kv/README.md`, `native/kv/Makefile`.
- Red tests: every golden vector decodes and re-encodes byte for byte; every corruption case of sc C1b is refused with the mirrored error code; `slot_header_golden_vectors_match` (finding 12), `slot_bounds_refused_before_decrypt`; `nm -u` empty.
- Acceptance: D1 (bad length/checksum half).

### N2: slab format, header-last write order, bounded recovery (#367)

- Files: `native/kv/kv_slab.h`, `kv_slab.c` (`kv_disk_bind` with overlap refusal, `kv_slab_format`, `kv_slab_recover`, slot state machine), tests `native/kv/tests/kv_slab_test.c` over `disk_file`.
- Red tests: `bind_refuses_overlap_with_store_and_anchor`, `recover_reads_exactly_n_header_units`, `power_cut_at_every_unit_boundary_never_exposes_partial_record`, `torn_header_is_empty_slot`, `used_bytes_counts_foreign_namespace_slots`.
- Acceptance: D1 (partial writes, flush failure), D2 (format half), D3 (Store region untouched, below).
- **D3 gates**: `n2_existing_store_gates_pass_with_slab_present` runs `scripts/qemu_ck_store_test.sh`, `qemu_ck_lifecycle_test.sh`, `qemu_ck_continuity_test.sh`, `qemu_ck_boot_test.sh`, `qemu_ck_slot_loader_test.sh`, `qemu_ck_recovery_test.sh`, `qemu_ck_rollback_test.sh`, `qemu_ck_disk_layout_test.sh` on an image that also has a slab, through the heavy forge as their headers require, and records pass/fail per script. Red until the layout test accepts the slab region.

### N3: native adapter, sealing, fault injection (#367)

- Files: `native/kv/kv_store.c` (the `KvStore` state machine in C, M5 sealing with the authenticated mapping of ADR 2.5, 4080-byte chunks, scratchpad decrypt, separate staging counters), `native/kv/tests/kv_store_test.c`.
- Red tests: `stages_completed_published_durable_follow_flush_calls`, `flush_failure_fails_job_and_fail_stops`, `queue_exhausted_refuses_with_efull`, `wrong_key_class_cannot_open_slot`, `slot_binding_wrong_slot_or_incarnation_fails_tag` (finding 12), `disk_full_no_free_slot_keeps_existing_slots`, `decrypt_never_copies_before_tag_verify` (mutant GUARD), `staging_report_lists_export_cipher_env_plain_scratch_bounce_peak_separately` (finding 15, D6).
- Acceptance: D1 (all five fault classes), D4 (buffer retained until completion), D6 (staging half).

### N4: ownership, cancellation, pump between steps and idle, session restore (#367, findings 13 and 14)

- Files: `crates/aienos-infer-kernel/src/session.rs` (`kv_export`, `kv_import` rows only, `computed`, `restore_commit` FFI), `native/kernel/core/infer.c` (between-steps hook, `park_point` callback before the consuming cancel, idle pump), `native/kernel/include/ck.h`, tests `native/kv/tests/kv_pump_test.c`, `crates/aienos-infer-kernel/tests/kv_export_import.rs`.
- Red tests: `source_slice_unchanged_until_disk_write_returns`, `cancel_mid_slot_leaves_header_zero`, `shutdown_within_deadline_leaves_only_whole_slots`, `tokens_keep_flowing_while_slot_fills_across_steps`, `export_import_roundtrip_restores_identical_f32_rows`, `restore_commit_after_prompt_prefills_only_the_suffix` (finding 13), `import_before_commit_is_invisible_to_step`, `restore_commit_refusal_changes_nothing`, `park_snapshot_survives_next_prompt_reset` (finding 14, D4), `idle_pump_drains_without_new_prompt`, `session_close_with_queued_park_keeps_staging_until_job_terminal` (D4), `reset_between_pump_calls_leaves_only_whole_slots` (D4).
- Acceptance: D4, D5 (with the limitation documented), D6 (staging peaks).

### N5: QEMU store, reboot on the same image, restore, continuation (#367)

- Files: `scripts/qemu_ck_kv_restore_test.sh`, a `svc/kv_agent.c` stage or the `pi5_agent.c` path, `docs/adr/0021-native-persistent-kv-slab.md` (short aienos ADR pointing at sc ADR 0003), `docs/evidence/kv-native.md` (rows: built, emulator, hosted, native physical (Pi 5), Spark physical NOT_RUN).
- Red tests: the script fails until the boot-2 log shows the restored prefix and the golden continuation, and until `used_bytes` after reboot equals before; `n5_existing_gates_pass_with_slab_present` (D3 wrapper, see N2); `staging_peaks_reported_in_kv_line` (D6).
- Acceptance: D2, D3, D5, D6 (Spark physical rows NOT_RUN).

## Native performance track cuts (ADR section 15)

| cut | content | depends on | red tests |
|---|---|---|---|
| X1 | Linux baseline harness: four labelled modes, device ceiling, CPU time per MiB, peak RSS (15.1) | C4, C9 | `bench_refuses_mixed_cache_state_summary`, `cache_state_device_requires_verified_zero_residency`, `baseline_reports_fraction_of_device_ceiling` |
| X2 | `O_DIRECT` host path, payload at 4096, aligned reads straight into pool blocks, buffered fallback with counter (15.3) | X1, C1b | `odirect_read_lands_in_pool_block_without_intermediate_copy`, `misaligned_destination_falls_back_and_counts` |
| X3 | native region state machine and zero-copy DMA ownership proof, emulator first; GB10 rules read via `spark-hardware-manual` and `docs-read` recorded before any driver change (15.2) | N4 | `dma_target_region_not_reused_until_terminal_completion`, `cancel_while_dma_pending_keeps_region_reserved`, `verify_failure_never_exposes_rows_to_session`, `region_state_transitions_are_exclusive` |
| X4 | native paged KV, `backend_format` 3, on-disk chunk equals in-memory chunk; refuses to merge if decode throughput drops more than 2 percent on the PI5-6 golden run (15.3) | X3, N5 | `native_paged_chunk_bytes_equal_disk_bytes`, `native_paged_decode_matches_strided_decode_bitwise`, `kv_restore_copy_bytes_is_zero_for_payload` |
| X5 | pipelined per-segment verification and the SHA-256 vs BLAKE3 vs per-block receipt (15.4) | X2, N3 | `pipelined_verify_never_adopts_before_root_checked`, `corrupt_segment_k_detected_while_segment_k_plus_1_in_flight`, `pipelining_does_not_reorder_digest_checks_before_header_checks`, `verify_cpu_time_per_mib_reported` |
| X6 | predictive restore from application/Interplane hints; first reads the real Interplane signal list (15.5) | C6, C8 | `prefetch_hint_never_blocks_step`, `unused_prefetch_expires_and_frees_holder`, `prefetch_respects_restore_bound`, `real_request_adopts_prefetched_prefix_without_second_read` |
| X6b | streaming park of immutable full chunks during decode, host first, native after X4 (15.6) | C6, C3 | `streaming_park_publishes_full_chunks_before_finish`, `streaming_park_never_reads_rows_that_decode_still_writes`, `cancel_after_streamed_chunks_keeps_them_valid`, `streaming_park_respects_park_bound` |
| X7 | multi-command NVMe window; **only if** the profile shows the single-command contract is the bottleneck (15.7); otherwise closed REJECTED with the profile as evidence | X3, X5, profile from X1/N5 | `command_ids_are_unique_among_inflight_commands`, `command_id_wraparound_never_aliases_an_inflight_command`, `dma_buffer_lifetime_covers_every_inflight_command`, `controller_reset_fails_every_inflight_job_terminal_once_and_releases_regions_after_reset_ack`, `cancel_with_commands_inflight_waits_for_their_completions_before_release`, `flush_ordering_header_flush_issued_only_after_all_payload_completions`, power-cut injector re-run with more than one command in flight |

Target ratios and which cut earns each are in ADR 15.8; none is a result until its cut has a receipt.

## Pi 5 physical lane cuts (ADR section 16)

| cut | content | depends on | red tests |
|---|---|---|---|
| P1 | 128 MiB (32768 unit) slab inside the Aienux partition view, `KV.MAP` on the ESP written by the `pi5-run.sh` path, Store shrink and one reformat plan, state record; aarch64 emulation of the card layout | N2, N4 | `slab_lies_inside_partition_view`, `kv_map_stale_or_missing_means_no_slab`, `state_record_roundtrip_across_reformat_plan`, `existing_store_gates_pass_with_pi_layout` |
| P2 | native CPU park and restore in the pi5-agent session, emulator run reproducing the golden 17 ids and 8 output ids | P1, N4 | `pi_agent_parks_after_reply_and_restores_on_next_boot_emulated`, `pi_restore_prefills_only_suffix`, `pi_flipped_superblock_byte_refuses_restore` |
| P3 | the physical run: park, real cold boot via tryboot, restore, golden continuation, negative control; labelled "native physical (Pi 5)"; kept separate from Spark-native (NOT_RUN under aienos#336) and hosted results. Drake authorizes it because it reformats the test card's Store and occupies the Pi | P2, Pi free per the Pi lane owner | `golden_ids_equal_after_cold_boot`, `restored_tokens_greater_than_zero`, `restored_prefill_time_below_cold_control`, `corrupted_slab_falls_back_to_prefill`; receipt recorded in `docs/evidence/kv-native.md` |

## Pin: where it goes and what it pins

`native/kv/CONTRACT.lock` in aienos:

```
format_version = 1
sc_commit = <sha of the sc commit whose C1b generated the vectors>
sc_adr = docs/adr/0003-persistent-kv-identity-storage-contract-and-ownership.md
vectors_sha256 = <sha256 of tests/golden/kv_record_v1.json>
slot_vectors_sha256 = <sha256 of native/kv/tests/golden/kv_slot_v1.json>
```

It pins the 256-byte header layout, the 4096-byte region rule, the namespace field order including the compute-config digest, the key derivation strings, the validation error order, the native slot header and envelope mapping, and the test vectors. It does not pin a crate, a Rust version or any runtime behaviour. N1's test refuses to run if a vectors file hash differs from the lock. The sc side records the same hash in `crates/aien-kv-store/tests/golden/README.md` so a vector change is a deliberate two-repo bump. The pin is created by N1 only after C1b merges; the C1 vectors at ce5d9eb (`7c532a54...b89c`) are superseded by C1b and must not be pinned.

## Acceptance ids

#484: A1 store, real restart, rediscover, restore identical bytes. A2 greedy continuation equals baseline (stochastic separately specified: not claimed, ADR 6.4). A3 wrong model/layout/position/namespace and truncated/corrupt payloads cannot be reused. A4 disk-full, interruption, failed writes never discard the last usable KV/history. A5 shared-prefix owners remain valid on park/cancel. A6 resident, staging and disk budgets bounded including restart accounting. A7 tests distinguish adapter recreation, process restart, clean reboot, power loss. A8 metrics.

#485: B1 slow-I/O progress proof. B2 concurrent park/restore and fork/cancel preserve payloads and refcounts. B3 duplicate loads, stale completions, queue-full, disk-full, mid-transfer cancel. B4 memory pressure: bounded, progress, no deadlock. B5 real routing through scheduler/backend. B6 benchmark 1/2/4/8. B7 resident vs cold vs restore, page-cache vs device, hosted vs native. B8 NOT_RUN recorded.

#367: D1 host fault injection (partial writes, flush failure, bad length/checksum, disk-full, exhausted queues). D2 emulator store, reboot on the same disk, restore, continuation. D3 existing Store/Cortex/boot/update/recovery tests valid. D4 buffers/blocks retained until device completion incl. cancel/shutdown. D5 progress evidence distinguishing inference during I/O from queueing; native limitations documented. D6 report allocated memory, staging bytes, persisted bytes, transfer latency, restart recovery; physical only from an authorized run.

## Review coverage: boxes the review found uncovered

| box | closed by | test |
|---|---|---|
| A4 | C4 | `fault_at_each_publication_boundary_leaves_consistent_outcome` |
| A6 | C4 | `recovery_keeps_admission_closed_until_walk_completes`, `used_bytes_counts_header_region_and_physical_blocks` |
| B3 (sibling cancel) | C5, C6 | `restore_txn_holder_survives_sibling_failure_until_all_jobs_terminal`, `multi_chunk_restore_failure_does_not_free_destination_in_use` |
| B4 | C5 | `sustained_restores_cannot_starve_parks`, `completion_queue_never_overflows_at_max_inflight`, `coalesced_waiters_are_bounded` |
| B6 | C9 | `bench_output_requires_completed_requests_latency_percentiles_and_peaks` |
| D3 | N2, N5 | `n2_existing_store_gates_pass_with_slab_present`, `n5_existing_gates_pass_with_slab_present` |
| D4 | N4 | `park_snapshot_survives_next_prompt_reset`, `session_close_with_queued_park_keeps_staging_until_job_terminal`, `reset_between_pump_calls_leaves_only_whole_slots` |
| D6 | N3, N4, N5 | `staging_report_lists_export_cipher_env_plain_scratch_bounce_peak_separately`, `staging_peaks_reported_in_kv_line` |
