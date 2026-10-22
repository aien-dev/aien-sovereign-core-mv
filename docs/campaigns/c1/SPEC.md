# Campaign C1: Branch-Native Inference Correctness

```text
campaign_id  = "c1"
spec_version = 1
status       = DRAFT (frozen on merge of feat/c1-campaign-spec)
```

## 0. Question

> Does the native Agent State / scheduler / KV / transformer / GB10 architecture preserve correct inference under branching, scale, and resource pressure?

C1 answers that question and nothing else. It is a correctness campaign. Performance, placement optimization, AEGIS, Cortex, MCP, tool effects, multi-model support, INT4/INT8, and distributed inference are out of scope (Section 11).

Path under test:

```text
request -> aien-runtime -> aien-scheduler -> SequenceArena -> aien-kv-cache (physical KV/COW)
        -> native transformer -> aien-inference-abi (BlackwellGb10Backend + ReferenceCpuBackend)
        -> generated tokens
```

## 1. Evidence Model

Three roles produce three levels of evidence. They are never merged.

| Role | Does | Must not |
| ---- | ---- | -------- |
| Builder / operator | Builds the environment, runs the official harness scripts, fixes infrastructure failures | Decide PASS/FAIL, edit bundles |
| Verifier | Runs `aien-campaign-verifier` over a frozen bundle against this spec | Modify code under test, interpret results outside the acceptance files |
| Independent reproducer | Clones the pinned commit on another GB10 and reruns from published instructions | Receive help beyond the published instructions |

Evidence levels: "passed on my machine" (builder), "artifacts satisfy the predefined acceptance criteria" (verifier), "someone else reproduced it" (reproducer). C1 targets the second level. The third is a follow-on.

### 1.1 Freezing rules

1. This spec, `acceptance.toml`, `placement.toml`, `kv_contract.toml`, and the verifier are merged to `main` before the harness emits its first evidence bundle.
2. Any later change is a dated amendment in Section 14 that bumps `spec_version`, states its reason, and names exactly which gates and claims it supersedes. Results whose acceptance meaning did not change stay valid. Prose-only fixes supersede nothing.
3. Tolerances are set from the measured noise floor (Section 5.3) and committed before any COW equivalence run they govern.

### 1.2 Pass rules

1. Correctness invariants are binary. Every required trial passes, or the stage fails.
2. A failing seed or script becomes a permanent regression case.
3. A gate result is bound to one `main` commit SHA. Any change to code in the transitive path under test reruns every earlier gate. Documentation-only changes do not.
4. The campaign stops at the first failed gate. Later gates do not run against a commit that failed an earlier gate.
5. Passing gates may receive annotated tags `c1/gate<N>/pass/<short-sha>` whose annotation carries the bundle manifest digest. Tags are a convenience. The evidence identity is `code commit + bundle digest + spec digest + verifier digest`.

## 2. Identity Model

Losing KV costs computation, not identity.

```text
LogicalBranchId            durable semantic identity, constant for the life of a branch
  |-- TokenHistory         authoritative token sequence
  |-- fork lineage         parent LogicalBranchId, fork_position, fork_ordinal, PrefixSnapshotId
  |-- sampler coordinates  (LogicalBranchId; generation_step, sample_stream, draw_index)
  `-- durable journal      StepCommit / ForkCommit / cancel / eviction records

ExecutionLineage           a frozen software and model configuration
  `-- ExecutionFingerprint

SequenceId                 one runtime incarnation of a branch
  `-- { boot_epoch: u64, slot: u32, generation: u32 }   (opaque, 128 bits, no truncating conversion)

Physical KV                reconstructible execution state beneath all of the above
```

### 2.1 SequenceId

- Opaque. Equality compares all three fields. `to_u64()` and every other truncating conversion is removed.
- Every serialized reference (scheduler events, lineage, KV logs, receipts) carries the full 128-bit value.
- A branch may hold many `SequenceId`s over its life (preemption, restart). Each binding `SequenceId -> LogicalBranchId` is journaled per epoch.

### 2.2 LogicalBranchId

- 256-bit digest of `(run identity, parent LogicalBranchId, fork_position, fork_ordinal)`. Root branches derive from the run seed.
- Keys the sampler, `TokenHistory`, lineage, `PrefixSnapshot` leases, and every verifier identity check.

### 2.3 TokenHistory

Owned per branch: original input tokens, inherited prefix boundary, branch-specific input, every accepted generated token. A token is appended before it counts as committed. `rec.prompt` is not recovery state.

### 2.4 PrefixSnapshot

```text
PrefixSnapshot { snapshot_id, fork_position, block_ids[], partial_block_valid_tokens, digest }
```

Records the physical block IDs at fork time. It is never resolved later through a parent's mutable block table.

### 2.5 BranchCheckpoint

```text
BranchCheckpoint { logical_branch_id, parent_id, fork_position, prefix_snapshot_id,
                   prefix_lease: Held | Evicted, token_history_digest, recovery }
```

Preemption drops the private suffix and keeps the prefix lease. Readmission reuses the exact leased prefix and recomputes only the divergent suffix. If the lease was evicted, recovery is a full `TokenHistory` recompute.

Lease eviction is atomic: mark every affected suspended lease invalid, journal the eviction, release those suspended references, decrement physical refcounts, free only blocks that reach zero. Active references are never removed by a snapshot eviction. The scheduler reclaims private suffixes before suspended-only prefixes.

### 2.6 ExecutionFingerprint

Runtime binary sha256, runtime and scheduler schema versions, GPU kernel and library sha256, model manifest sha256, tokenizer sha256, `placement.toml` sha256, `kv_contract.toml` sha256, sampler algorithm and version, campaign spec version, relevant ABI versions.

## 3. Invariants

Every stage of every gate checks all applicable invariants. Only performance thresholds vary by stage.

| # | Invariant | Definition |
| - | --------- | ---------- |
| I1 | Identity | Parent/child lineage, `LogicalBranchId`, and `SequenceId` bindings survive every transition. No duplicate live `SequenceId`. No child references a nonexistent fork base. |
| I2 | Isolation | One branch's writes cannot change another branch's state. KV bytes outside legitimately COW-mutated pages are byte-for-byte identical. No write event ever targets a block with refcount > 1 or a leased snapshot block; any writer COWs first. |
| I3 | Equivalence | COW execution matches independent recomputation from `TokenHistory` within the frozen tolerance, at every generated step (teacher forced). |
| I4 | Conservation | Shared/private accounting equals actual allocations (Section 7). |
| I5 | Reclamation | Destroying or cancelling a branch returns every block it solely owned. Release is idempotent: repeated or racing release never double-decrements. |
| I6 | Parent survival | Parent inference is identical before and after child creation and destruction, except where the parent itself advances. |
| I7 | State continuity | Preemption, suspension, readmission, cancellation, and restart do not alter the logical or numerical identity of a surviving branch. |

### 3.1 Required branch behaviors

- At fork, children reference the parent's physical pages. No private copies exist until a write.
- After branch A diverges, only the affected pages become private to A. B, C, and P keep their original bytes.
- KV contents are checked, not only metadata: selected layers, tokens, heads, and K/V planes are hashed before fork, after fork, after divergence, and after child destruction.
- Forking into an existing sequence ID returns an error and leaves every refcount, free-list entry, block table, and existing child state byte-for-byte unchanged.
- Releasing an already released sequence returns an error without mutating allocator state.
- Fork positions tested (Gates 1, 2, 4): `1`, `bs - 1`, `bs`, `bs + 1`, `k*bs - 1`, `k*bs`, `k*bs + 1`, with `k` per profile in `acceptance.toml`. Every unaligned fork runs parent-writes-first, child-writes-first, parent-then-child, and child-then-parent.

## 4. Gates

| Gate | Purpose | Model | Preemption |
| ---- | ------- | ----- | ---------- |
| 0 | Measurement qualification: prove the measurement chain detects bad paths | mocks + TinyLlama | n/a |
| 1 | Structural correctness on real runtime components; capture known KV bugs failing, fix, rerun | TinyLlama architecture | off |
| 1.5 | Qwen qualification: checkpoint identity, 48-layer load, single-sequence reference parity | Qwen3-Coder-30B-A3B-Instruct-FP8 | off |
| 2 | Qwen branch equivalence at low scale (2, 8, 32 branches) | Qwen | off |
| 3 | Scale: 100 and 500 branches, fixed 40 GiB BF16 pool, max 512 private tokens per branch, idle teardown | Qwen | off (watermark 0) |
| 4 | Resilience: oversubscription, preemption, active cancellation, crash and restart, deterministic fault points | Qwen | on |
| 5 | Performance and placement changes. Separate campaign. | | |
| 6 | Whole-stack chaos and soak with AEGIS, Cortex, external capabilities. Separate campaign. | | |

Stage parameters (branch counts, context lengths, trial counts, fork positions) live in `acceptance.toml`.

### 4.1 Gate 0: measurement qualification

Test the tester. Each case has an expected verdict, and the failing cases stay in the evidence package.

| Case | Expected |
| ---- | -------- |
| Op required on GB10 runs on CPU | FAIL |
| Op permitted on CPU runs on CPU | PASS |
| Backend reports GPU, witness saw no kernel | FAIL |
| Witness saw a GPU kernel, backend reports CPU | FAIL |
| Kernel symbol outside the op's allowlist | FAIL |
| GPU call fails and falls back where fallback is prohibited | FAIL |
| Bundle file altered after `sha256sums.txt` was written | FAIL |
| Binary altered after hashing (fingerprint mismatch) | FAIL |
| Missing device at startup with `AIEN_REQUIRE_BLACKWELL=1` | abort |
| KV pool dtype differs from `kv_contract.toml` | abort |
| Event journal omits a release that the raw census reflects | FAIL |
| Mock backend that reports GPU while running the CPU reference | FAIL |

### 4.2 Gate 1: TinyLlama structural correctness

Order is mandatory:

1. Write the tests that trigger the known defects: duplicate child ID in `fork_sequence` (table overwrite, leaked refcounts) and `free_sequence` underflow (double free).
2. Run them. Archive the failing bundle.
3. Land the fix commit.
4. Rerun. Archive the passing bundle.

Evidence chain: `before-fix failing bundle -> fix commit -> after-fix passing bundle`.

Gate 1 also runs the adversarial script: fork 100 branches, mutate branch 73, destroy 20 arbitrary siblings, advance the parent, mutate branch 4, destroy branch 73, then verify every survivor against independent recompute. Scripts run in deterministic mode (Section 8).

### 4.3 Gate 1.5: Qwen qualification

Hard boundary, in order:

```text
checkpoint present
-> every shard, index, config, tokenizer file hashed individually
-> manifest digest over those hashes recorded
-> architecture and config parsed
-> all 48 layers loaded through the runtime (not the single-layer loader)
-> single-sequence parity with the frozen Hugging Face reference harness passes
-> only then Qwen enters Gate 2
```

Qwen enters `main` through ordinary reviewed and preflighted `feat/` PRs, not a worktree merge.

### 4.4 Gate 4: resilience scenarios (minimum set)

- Oversubscription: 500 branches x 1,000 private tokens against the 40 GiB pool (about 33,548 blocks required vs 27,306 available).
- Preempt and readmit with the prefix lease held: only the private suffix is recomputed.
- Parent terminated -> suspended children remain -> prefix becomes suspended-only -> prefix evicted -> child readmitted -> full `TokenHistory` recompute -> I3 passes.
- Cancellation during an active write at each named fault point.
- Release races: cancel twice, completion vs cancel, preemption vs cancel, parent teardown vs child cancel.
- Kill between step compute and `StepCommit` fsync; kill after fsync before token exposure.
- Restart with identical `ExecutionFingerprint`: every surviving branch recovers, with identical seeded token IDs.
- Restart with a deliberately mismatched fingerprint: restore refused.

## 5. Numerical Acceptance

### 5.1 Comparisons

For every compared step and every compared logit vector:

- max absolute error `|a - b|`
- max relative error `|a - b| / max(|ref|, epsilon)`, epsilon frozen in `acceptance.toml`
- top-1 agreement
- KL divergence of the softmax distributions

Top-1 disagreement is permitted only at steps where the reference top-2 margin is below the absolute tolerance. Every such step is listed in the report.

### 5.2 Exact comparisons

- KV bytes not legitimately COW-mutated: zero tolerance, byte equality.
- Seeded generation in Gate 4 continuity checks: identical token IDs. A token mismatch fails even if logits pass. Logs must show whether the cause was sampler state or a numerical boundary crossing.

### 5.3 Noise floor

Before any COW equivalence run, measure logit differences with no branching for the same sequence: run twice, with different batch neighbors, and prefill vs decode. Tolerance for each metric is `k x measured floor`, with `k` in `acceptance.toml`. The measured values are committed as an amendment before Gate 1 equivalence (TinyLlama) and before Gate 2 (Qwen).

### 5.4 External reference

Gate 1.5 compares AIEN against the Hugging Face reference on the same checkpoint and tokens (single sequence, no branching). I3 compares AIEN against itself and proves COW does not corrupt state. Both are required. Neither substitutes for the other.

## 6. Device Placement

"Zero fallback" is retired as a claim. Placement is enforced per op against `placement.toml`:

```text
AIEN_ENFORCE_PLACEMENT=docs/campaigns/c1/placement.toml
```

A call whose observed execution does not match the placement contract aborts the run. `AIEN_REQUIRE_BLACKWELL` keeps its existing meaning: a GB10 device must exist at startup. `fallback_count` is diagnostic telemetry only.

### 6.1 Two witnesses

1. AIEN dispatch record: every op call has an `op_call_id` and records requested and reported path.
2. External device witness: `aien-campaign-device-witness`, a CUPTI activity library loaded via `CUDA_INJECTION64_PATH`, records actual kernel launches and completions. AIEN brackets each op call with an NVTX range carrying its `op_call_id`. The verifier consumes the witness trace but does not link the witness library.

For an op required on GB10: AIEN reports GPU, the witness saw a kernel matching the op's allowlist inside the op's range, the kernel completed, and no CPU fallback event occurred. For an op permitted on CPU: AIEN reports CPU, `placement.toml` permits CPU, and the witness saw no kernel inside the op's range.

Kernel symbol allowlists are glob patterns bound to the frozen kernel and library digests. cuBLAS kernel names are internal and version-specific, so their patterns are frozen from the Gate 0 qualification trace by amendment.

## 7. KV Contract and Conservation

`KvDType::Bf16` is mandatory. The receipt records `kv_dtype`, element size, block size, bytes per block, total blocks, and configured pool bytes. A mismatch with `kv_contract.toml` aborts the run. This matters because some existing transformer constructors default to an FP32 pool.

The scheduler computes block demand only through a `KvPoolGeometry` value supplied by `aien-kv-cache` (block size, bytes per block, block count, dtype, layers, KV heads, head dim). The hardcoded `16` in the scheduler admission path is removed.

The verifier rebuilds state from three independent sources and requires all three to agree:

```text
1. mutation journal     kv_allocations.jsonl, branch_lineage.jsonl
2. raw final census     kv_census.json (sequence tables, free set, physical refcounts)
3. reported metrics     kv_metrics.json (AIEN's own counters)

replayed == census     replayed == reported
```

Conservation equations:

```text
allocated + free = pool_total
refcount(block)  = live references + suspended lease references reconstructed from lineage
no referenced block in free set
no unreferenced block outside free set
no duplicate live SequenceId
no child references a nonexistent fork base
```

Leak metric hierarchy on GB10 unified memory: primary is allocator block conservation; secondary are process RSS and total unified memory pressure.

## 8. Determinism

- Deterministic mode: the scheduler consumes a seeded event script at step boundaries. Seeds and scripts are recorded in the bundle.
- Named fault-injection points (`cow_copy_mid`, `fork_mid`, `release_mid`, `step_pre_commit`, `step_post_commit_pre_expose`) fire on the Nth hit. Used from Gate 4.
- Sampler: counter-based (Philox-style). Key is `LogicalBranchId`. Counter is `(generation_step, sample_stream, draw_index)`. No mutable sampler state survives across steps.
- Gates 1 to 3 run greedy decoding plus one frozen seeded-sampling configuration.
- Random chaos belongs to Gate 6 only.

## 9. Durability (Gate 4)

A model step is committed only when its `StepCommit` has been appended to the durable branch journal and synchronously persisted. No token is exposed externally before its commit is durable.

```text
StepCommit { logical_branch_id, step_number, previous_history_length, input_token,
             accepted_output_token, token_history_digest, sampler_coordinates,
             fork_base_id, logits_digest (campaign mode), execution_fingerprint }

ForkCommit { logical_child_id, logical_parent_id, fork_position, prefix_snapshot_id,
             token_history_digest }
```

- Journal: append-only, sequence-numbered, checksummed. A torn final record is discarded on replay (it was never exposed).
- One fsync covering every `StepCommit` of a scheduler step counts as synchronous. Modes that expose tokens before fsync are Gate 5 modes and never inherit the restart claim.
- A child does not exist externally until its `ForkCommit` is durable. Cancellation and prefix eviction are journaled.
- Boot epoch is the first record of each boot (`BootCommit`): acquire exclusive startup lock, read prior epoch, increment, append, fsync the journal, fsync its directory when the file was created or renamed, release lock, then accept requests.
- Processed-operation idempotency is derived from the journal. The separate `/tmp` processed-ops file is retired.
- After restart every lease is `Evicted` and recovery is full `TokenHistory` recompute.
- Physical KV is never persisted as the source of truth.
- Exactly-once external tool effects are out of scope (AEGIS campaign).

### 9.1 Restore vs migrate

Restore requires the persisted `ExecutionFingerprint` to equal the current one. Otherwise restore is refused. Migration (recompute durable history under a new fingerprint, new `ExecutionLineage`, `MigrationReceipt`) is out of scope for C1 except the refusal test.

## 10. Bundle and Receipt

Every run emits an immutable, content-addressed bundle stored outside Git:

```text
manifest.json            campaign_id, spec_commit, acceptance_sha256, placement_sha256,
                         kv_contract_sha256, kv_profile, verifier_commit, verifier_binary_sha256,
                         code_under_test_commit, dirty_tree, model_manifest_sha256,
                         execution_fingerprint, instrumentation_enabled, seeds
environment.json         driver, CUDA, Mojo, Rust toolchain, OS
hardware.json            device name, compute capability, memory
kv_contract.json         observed geometry and dtype
op_records.jsonl         AIEN dispatch record per op_call_id
witness.jsonl            device witness trace (hashed by the witness)
kv_allocations.jsonl     KV mutation journal
branch_lineage.jsonl     fork, teardown, lease, binding events
kv_census.json           raw final allocator state
kv_metrics.json          AIEN-reported counters
logits_reference/        recompute and HF reference logits
logits_cow/              COW-path logits
memory_samples.csv
latency_samples.csv
failures.jsonl
sha256sums.txt           over every file above
```

Only `bundle_sha256`, `manifest_sha256`, and `PASS/FAIL` are committed to the repository.

Instrumentation (KV hashing, witness, per-op records) is controlled by a runtime flag recorded in the manifest. Correctness gates run instrumented. Scale runs are repeated uninstrumented with end-to-end logit comparison. No timing number from an instrumented run supports a performance statement.

## 11. Scope

In scope: the path in Section 0 on one frozen model configuration with BF16 KV.

Cut from C1: Cortex, distillation, Harvester, MCP, tool use and external effects, cockpit UI, AEGIS policy, vLLM or other comparisons, multi-model, INT4/INT8 and exotic quantization, distributed inference, deadline scheduling (deadlines are stored but not enforced by `aien-scheduler`; nothing in C1 tests them).

Parallel track, not C1: the unrestricted `bash -c` path in `aien-cli/src/tools.rs` is a release-blocking security issue handled in the AEGIS security track.

## 12. Build Prerequisites

| Before | Required |
| ------ | -------- |
| Gate 0 | Per-op dispatch records; fail-closed placement enforcement; device witness; bundle and receipt schema with `ExecutionFingerprint`; `KvPoolGeometry` and KV contract check; raw state census API; this verifier |
| Gate 1 | 128-bit `SequenceId`; `LogicalBranchId`; `TokenHistory`; duplicate-fork rejection; idempotent release; deterministic scheduler scripts |
| Gate 1.5 | 48-layer Qwen runtime integration; checkpoint manifest; Hugging Face reference harness |
| Gate 3 | Per-branch teardown and cancel API sharing one idempotent release primitive |
| Gate 4 | `BranchCheckpoint` and `PrefixSnapshot` leases; durable journal with `BootCommit`, `StepCommit`, `ForkCommit`; named fault points; counter-based sampler; fingerprint-checked restore |

## 13. Claims

Frozen wording. Published claims are quoted verbatim with their evidence bundle digest.

- **G0**: "The measurement system detects every seeded violation in the qualification suite."
- **G1**: "On the TinyLlama architecture, AIEN's KV/COW, SequenceArena, and scheduler satisfy all seven invariants across the defined fork, mutation, preemption-free teardown, and reclamation scripts."
- **G1.5**: "AIEN loads the frozen Qwen3-Coder-30B-A3B-Instruct-FP8 checkpoint identified by manifest `<sha>` and matches the frozen Hugging Face reference harness within the specified numerical tolerances for the defined single-sequence qualification workload."
- **G2/G3**: "With the frozen device-placement specification and no disallowed execution paths, COW-branched inference on the frozen Qwen3-Coder checkpoint is numerically equivalent to independent recomputation at the tested branch counts through N branches with a 32K-token parent context."
- **G4**: "With the frozen device-placement specification and no disallowed execution paths, COW-branched inference remains numerically equivalent to independent recomputation and preserves branch state continuity under the defined oversubscription, preemption, cancellation, and restart fault scenarios."

Wording rules: C1 never claims "production". "Blackwell-native" and "zero fallback" may appear only in a later frozen claim that defines and proves them.

## 14. Amendments

None.
