# Adaptive weight-flow implementation plan

> For agentic workers: REQUIRED SUB-SKILL: superpowers:executing-plans. Drake requested personal implementation; no implementation workers.

**Goal:** overlap verified known-demand expert loading with unchanged ordered expert computation, compare synchronous, fixed two-buffer and adaptive modes.
**Architecture:** extend the existing Ribbon pager and slot protocol at 836a53446d3674e86cbfc435d814775a0b8503a8. New policy is a seeded Omega executable; the existing Rust host adapter explicitly remains Rust and owns threads, read adapter and GPU calls. No replacement inference engine or claim of an entirely Omega-native runtime. Native policy never sees or selects expert IDs. One bounded queue of whole expert pairs, one serialized compute consumer, pins held until both projections complete. No compression, prediction, tensor tiling or floating-point reassociation.
**Tech stack:** pinned seed-built Omega policy, existing aien-heartbeat read/verify/pager and Omega slot ABI; Python only test/build orchestration.
**Spec:** user supplied adaptive weight-flow proposal, SC631/633 and Omega533 contracts. This is SC631 known-demand adaptation, not completion of speculative SC633 or CUDApp48.

## Global constraints
- No live server restart, model swap, driver change, shared worktree edits or third-party inference engine.
- Existing grouped FFN remains default and serves as the independent same-model reference.
- Three experimental modes execute experts individually in router order with identical arithmetic and output placement. Final FFN outputs publish only on whole-call success.
- At most 32 selected experts; reject duplicate/unknown IDs before launching. In-flight window never exceeds configured limit or actual slot capacity. Existing cache hits avoid retransfer.
- Native policy bounded integer nanoseconds; modes 0 sync, 1 fixed, 2 adaptive. Depth includes current expert. Adaptive starts at min(2,cap), grows by one on exposed wait above compute/8, halves on memory pressure or >25% compute degradation; after four zero-wait samples shrinks by one to floor2. Cap1 always serial. No speed gain assumed.
- Error/cancellation drains in-flight work and releases pins; already-issued kernel I/O cannot be forcibly interrupted. Read deadline remains explicit. Shutdown is deterministic only once kernel reads return, an inherited Linux limitation disclosed, not hidden.
- Actual host-wait intervals are not labeled GPU-idle measurements. No chip-speed claim without a properly locked hardware run.

## Review focus
Stale pins on early failure/panic; empty/duplicate/out-of-range requests; depth reduction with old requests in flight; short native protocol reads/child death; failure after earlier expert computations must leave caller outputs unchanged.

## Task 1: native policy
Files native/weight-flow/policy.om, build.py, test_policy.py.
- [ ] Compile a serial-only policy stub with pinned compiler; policy tests must fail behaviorally for fixed/adaptive widening.
- [ ] Implement framed fixed-size policy; reject invalid version/mode/cap/depth/booleans, saturated timings and truncated input; no response before complete validated frame.
- [ ] Verify independent literal boundary fixtures, fragmented I/O, repeated frames, compiled behavioral mutants.

## Task 2: bounded host integration
Files crates/aien-heartbeat/src/weight_flow.rs, lib.rs, tests/ribbon_store.rs.
Interfaces: FlowMode, FlowConfig, NativePolicy, ExpertPager::flow(layer,experts,config,policy,cancel,compute callback) -> FlowTrace. Callback receives ordinal and pinned slot. Prepared RAII ownership never leaks on callback failure. Existing pager used without bypassing digest checks.
- [ ] Run original heartbeat suite. Add serial scaffold and tests that prove next expert ready during current compute; observe RED.
- [ ] Implement ordered bounded overlap, pin RAII, cancellation/deadlines and native policy bridge with timeout.
- [ ] Verify real file bytes, forced overlap/non-overlap, capacity1, reuse, corruption/GPU faults, cancellation and panic cleanup.

## Task 3: FFN attachment and qualification
Files crates/aien-inference-abi/src/qwen3_moe_omega/ribbon.rs and native/weight-flow documentation/evidence.
- [ ] Add opt-in constructor/config method without changing default; wrap FFN individual-expert path in transactional output buffers.
- [ ] Compare full tiny-model logits against original grouped resident/paged path in all modes, pool bounds and both routing modes; injected errors preserve caller outputs.
- [ ] Run all heartbeat and inference ABI tests and exact-source replay, native mutants and final diff review.
- [ ] Publish source and real receipts. Hardware performance is a separate actual-run criterion, not inferred from host tests. Preserve full acceptance as open if required native/GPU integration cannot yet run safely; no invented tok/s.
