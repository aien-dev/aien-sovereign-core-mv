# Adaptive weight flow: opt-in Ribbon experiment

Status: implemented and host-tested. Native GPU build succeeds. Hardware execution was **NOT RUN** because `spark-window` refused L689 under the existing GPU reservation. No speedup, real-model output equality, production release, canonical-motor integration or live-service deployment is claimed.

This branch extends Ribbon base `836a53446d3674e86cbfc435d814775a0b8503a8` (Sovereign Core PR #660). It is an experimental branch, **not a merge-ready replacement for current main**. Current-main reconciliation, canonical CUDApp/Omega attachment, native-artifact release qualification and the authorized hardware gate remain separate requirements. Do not merge the historical ancestor wholesale or restore retired workflows from it.

## Implemented

- `policy.om`: handwritten seeded Omega scheduling policy. It receives timings and bounds, never expert IDs or weight bytes.
- `weight_policy.rs`: explicit Rust host IPC adapter, with framed nonblocking transport, partial-I/O handling, a two-second deadline, failure latching and child cleanup. No policy fallback.
- `weight_flow.rs`: bounded known-demand preparation over the existing verified `ExpertPager`. The existing Rust host now orchestrates scoped preparation threads and one ordered compute consumer. This is **not an entirely Omega-native runtime**. Python files only build/test the native policy; Python is not on the inference path.
- `PagedExperts::set_flow`: opt-in FFN attachment. Whole expert pairs are pinned until gate/up, SwiGLU and down projection finish. Ordered expert outputs and reduction are unchanged. Original grouped FFN remains the default.
- Transactional FFN scratch: late compute/policy failure leaves caller output buffers unchanged. Error, cancellation and unwind release outstanding pins; pending preparations are joined. Existing read deadlines bound waiting; kernel reads are not forcibly cancelled.
- Benchmark switches in the existing `own30b_check`: synchronous, fixed two-buffer, adaptive, and a sweep including the old grouped baseline. Opted-in runs reset KV before each prompt and record per-step vocabulary-logit bit hashes as well as tokens. Sweep requires both to match and requires evidence that flow actually ran. Without the switch the original generation entry point is retained.

No speculative experts, weight compression, tensor tiling, numerical reassociation or replacement inference engine was added. The separate XOR/delta-storage suggestion is not implemented here.

## Bounds and adaptation

The caller configures at most 32 in-flight whole expert pairs; active capacity is also clamped to the pager's preallocated slot count. At most 32 unique known experts are accepted in one call. Synchronous depth is one. Fixed depth is two, clamped to capacity. Adaptive depth starts at two and is clamped to capacity.

The native controller grows depth when exposed preparation wait exceeds `compute / 8`; four wait-free observations reduce depth by one. Compute time greater than `previous + previous / 4`, or an explicit pressure bit, halves depth. Adaptive depth floors at two unless capacity is one. Integer division is truncating, thresholds are strict. Pressure input is implemented/tested in the native ABI, but the current host sends zero: this adapter does **not** claim live free-memory sensing or adaptive cache priorities. Memory safety comes from the fixed pool/queue bounds. Already-issued preparations are drained rather than revoked when desired depth shrinks.

The caller must serialize flow runs on a pager; `PagedExperts` does this. Lower-level direct users must obey that contract. Cancellation is supported by the core runner; the existing FFN entry point does not expose a new cancellation parameter.

## Native ABI

Little-endian request: ten u64 fields, 80 bytes:

`version mode cap depth wait_ns compute_ns previous_compute_ns calm pressure reserved`

Version=1; mode=0 synchronous, 1 fixed, 2 adaptive; cap=1..32; depth=1..cap; timings=0..i64::MAX; calm=0..3; pressure=0..1; reserved=0. Reply: two u64 fields, 16 bytes: `depth calm`. Invalid/truncated input exits 65; I/O errors exit 74. EOF at a frame boundary exits zero. Lifetime is explicitly bounded at one billion transactions. Actual stress evidence covers 1,000,001 consecutive requests, not the entire lifetime bound.

## Reproduce on Linux AArch64

Use the seed compiler whose SHA-256 is
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.
The builder verifies it and requires an absent output path; it validates nonempty AArch64 EXEC output. It does not silently use a different compiler.

```sh
OUT="$HOME/handoffs/weight-flow-replay"
mkdir -p "$OUT"
python3 native/weight-flow/build.py "$OMC" native/weight-flow/policy.om "$OUT/policy"
python3 native/weight-flow/test_policy.py "$OUT/policy"
python3 native/weight-flow/mutants.py "$OMC" native/weight-flow/policy.om "$OUT/mutants"
AIEN_WEIGHT_FLOW_POLICY="$OUT/policy" AIEN_FORCE_CPU_STUB=1 \
  CARGO_TARGET_DIR="$OUT/target" CARGO_BUILD_JOBS=3 \
  quiet-build cargo test -p aien-heartbeat -p aien-inference-abi
```

The native-policy integration tests deliberately require `AIEN_WEIGHT_FLOW_POLICY`; missing architecture/compiler prerequisites are not silently skipped. This branch is not yet a portable default-CI integration.

Hardware command, **only after admission by the reservation owner**, with a separately built native `own30b_check` and exact model/golden inputs:

```sh
own30b_check chip "$CHECKPOINT" "$GOLDEN" "$RECEIPTS" \
  --tokens-only --prompts 1,2 --new 16 --max-context 256 \
  --weights fp8 --attn fused --router device --lm-head gemv --prefill auto \
  --paged-experts 8 --manifest "$MANIFEST" \
  --weight-flow sweep --flow-policy "$POLICY" --flow-limit 8
```

The attempted native build used Omega `1549caaedfd73d7d885eab4363ab4fd36eb16feb` and Physics `6d7cf0d4d8eb2cda7b512100ff6058e25dbb3ddf`. Its GPU archive digest is not recorded in the inherited release lock, so successful linkage is not release qualification. No lock override or live-model change was made.

## Metrics and interpretation

`run.json` records emission timestamps, decode rate, physical read bytes, loaded bytes and page-wait counters. `last_layer_flow` records preparation/compute intervals, depth and peak in-flight occupancy for **the last FFN call only**, not a whole-prompt trace. Inter-token p95 can be calculated from successive emission timestamps, excluding the prefill-to-first-token interval; sample count must accompany it. The existing benchmark reports slot count, pool bytes and residency. No real benchmark result exists in this delivery.

`exposed_wait_ns` measures the serialized consumer's wait for preparation, not a device-wide GPU idle counter. Actual GPU idle time remains unmeasured. Per-step vocabulary SHA-256 covers the little-endian f32 bit patterns, not formatted decimal logits; hashes are compared across modes on identical prompts.

Sweep shares one model/pager allocation. With the proposed eight-slot, 48-layer workload, inter-layer eviction is intended to prevent cross-mode cache warmth; actual miss/hit/physical-byte equality must still be checked before timing comparisons. Larger-residency experiments need explicit warm/cold controls and alternating mode order. Shared-machine samples alone cannot establish a speedup.

## Evidence

See `../../docs/inference/evidence/weight-flow-L689/` for original failures, corrected host gates, native policy tests, compiled behavioral mutants, long-lived IPC probe, independent review and disposition, native build logs, hashes and the hardware-admission refusal. Raw failures are retained, not overwritten by later passing logs.

The host fixture exercises the same full tiny-model inputs across modes/capacities and compares exact outputs. It is not an OWN-30B or live Qwen3.6-35B hardware qualification. The production model, service and compiler owner's worktree were not modified.
