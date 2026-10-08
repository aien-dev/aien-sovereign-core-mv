# GB10 weight ownership and serving-time memory (design, for #277, not a fix)

## In plain words

On the GB10 the graphics chip has no memory of its own. It uses ordinary system memory that the
driver pins for it. Today the Qwen3-4B model takes that memory twice: a full 32-bit copy of every
weight stays in the program (about 15 GiB), and a second, smaller 16-bit copy is handed to the
driver (about 7.5 GiB). When the machine is busy and its "free" memory is low, the driver
sometimes cannot get the pinned memory it asks for, even though plenty of memory could be freed
by dropping cached files. That is the failure in #277 and omega#327. This note does two things.
First, it lists every place the program asks the driver for memory, and shows that some requests
happen while the model is serving, not only at start-up (so just opening the GPU earlier, as
draft PR #278 does, cannot cover them). Second, it proposes the longer-term fix: send each weight
straight from the model file to the GPU, never keep the big 32-bit copy, and ask the driver for
all serving memory once, at start-up. That would cut the machine's memory use from about 22 GiB to
about 9 GiB and remove the late requests. Nothing here changes runtime code, and nothing is
proven on the chip. The default refusal of Qwen3 on the GB10 stays in place.

## Status and evidence labels

- Source facts below were read at: omega main `01f6a74` (equal to `omega.lock`), sovereign-core
  main `cad36d9`, physics `6d7cf0d` (the repo that holds `nvrm/nvrm.c`). These are the commits read; both mains have since moved.
- OBSERVED = read in source or in a recorded receipt. COMPUTED = arithmetic from the model's
  `config.json` and the source formulas. INFERRED = follows from source, never run. UNVERIFIED =
  not checked.
- No GPU was used for this note. The #277 chip attempt is INVALID CONDITION (baseline passed in a
  scripted cache fill); the trigger of the failure is still UNKNOWN. This design assumes the
  working theory of #277 (RM system-memory allocations fail at low MemFree) and does not prove it.

Model numbers used (Qwen3-4B-Instruct-2507, `config.json`): hidden 2560, intermediate 9728, 36
layers, 32 query heads, 8 KV heads, head_dim 128, vocab 151936, tied embeddings, declared context
262144.


## Update 2026-10-07 (after omega#332 and omega#333; still no chip run)

- Cut B is done, on omega's fake GB10 layer (CPU only, no chip). omega#332 (merged 7894abb) counted
  458 driver allocations and 456 frees after warm-up over 428 tokens: the per-length attention pool
  reallocation and the matmul code-cache thrash (21 shapes, 8 slots) are real, OBSERVED on the fake
  layer. This confirms section 1(c): serving-time allocations exist, so #278 alone cannot fix #277.
- omega#333 (merged b980783) adds an opt-in up-front reservation, `omega_gpu_reserve_serving`
  with `OmegaGpuServingBounds`: attention pool sized from the KV block size, matmul scratch, and a
  deeper matmul kernel cache (32 slots of storage, 21 needed by Qwen3-4B). Opted in, the same test
  counts 0 allocations and 0 frees after warm-up; a request past the reservation is refused with a
  named error before any driver call. Not covered: elementwise scratch, the attention kernel cache
  (8 slots), prebuilding matmul shapes at reserve time.
- Wired in (draft PR "Part of #277", not yet proven on the chip): `omega.lock` is b980783 and the
  daemon, on the opt-in Qwen3 GB10 path only, calls the reservation once after the session opens and
  before the first request (`aien-inference-abi/src/gb10_serving.rs`), logging a
  `GB10_SERVING_RESERVATION` line and refusing to serve by name if omega refuses. Its chip attempt
  under the #277 protocol is still to be declared and run. Cuts A and C are not started. The default
  refusal of Qwen3 on the GB10 stays.
- The #277 chip attempts so far: attempt 1 (06:33Z) and attempt 2 (21:45Z) are both INVALID CONDITION:
  the baseline did not reproduce the failure, so the fix runs were skipped by the rule fixed in advance.
  In attempt 2 the low-memory state was present at launch, but the scripted cache fill reclaimed
  memory before the baseline ran. A protocol amendment that
  skips the fill when the trigger already holds is proposed on #277 and not applied.
## 1. Allocation audit

All GPU-side allocations go through `omega_gpu_session_alloc` (omega `src/omega_gpu_session.c:78-82`)
which calls `nvrm_alloc_gpu_uncached` (physics `nvrm/nvrm.c:473-475`, body `:477-`). The size is
rounded up to 4 KiB (`nvrm.c:480`), so any growth of even one page is a new request. A refused
request sets `rm->faulted = 1` (`nvrm.c:506-509`) and later requests then fail in the same
function (`nvrm.c:478`): one refused allocation ends the GPU session for the rest of the process.
OBSERVED.

### (a) Daemon and session start-up

| What | Size | When | Freed |
|---|---|---|---|
| Channel and its context buffers (`m16_native_create_channel`, session.c:125) | driver decides (the `kgrctxAllocMainCtxBuffer` that failed in #274) | first chip call, or the probe in `open_gpu_session_with_retry` (omega_backend.rs:968-) | at session close only; nothing in the daemon calls close |
| Session buffers: pushbuffer 64 KiB, cbank 4 KiB, marker 4 KiB, QMD 64 KiB (session.c:127-128) | 136 KiB | at open | at close |

Daemon order on main (commands.rs, `run_daemon_server`): weights loaded (`build_native_daemon_backend`,
:1332), KV pool built (:1508), then the session opened by the probe (:1528). Draft #278 moves the
open before the load. The session never reopens after a success.

### (b) First forward pass (warm-up turn, server.enable_warm_up, commands.rs:1554)

| What | Size formula | Notes |
|---|---|---|
| Resident weight matrix, one per matrix (`omega_gpu_tensor_upload_bf16`, matmul_api.c:210) | kp * np * 2 bytes, kp = k rounded to 16, np = n rounded to 8 | layers: 36 x (q 2560x4096, k 2560x1024, v 2560x1024, o 4096x2560, gate 2560x9728, up 2560x9728, down 9728x2560) = 3.63 B values = 6.76 GiB COMPUTED. Plus lm_head: `compute_logits` goes through `chip_matmul` with the tied embedding (omega_backend.rs:833-848), 151936x2560 = 0.72 GiB COMPUTED. Total resident 7.48 GiB. #277 quotes about 6.8 GiB, which matches the layers only; the lm_head copy is extra (INFERRED, check at the next run). |
| Matmul kernel code, one per (kp, np, grid_x) shape (matmul_api.c:163) | about one to a few pages each (code + 2048 B tail, rounded to 4 KiB, code_alloc.h) | see (c): the cache has 8 slots |
| Elementwise kernel code (rmsnorm, rope, swiglu) | pages each | elementwise_api.c:930, cache of 8 |
| Attention kernel code (one shape for Qwen3) | pages | attention_api.c:553, cache of 8 |
| Scratch first sizes | see (c) | all scratch starts at 0 and is created by the first call that needs it |

Today these resident uploads happen lazily on the first call that touches each weight, which is
the warm-up turn when a tokenizer is loaded (otherwise the first real request). Each upload also
makes transient host copies: a transposed f32 copy of the whole matrix (`wt`, omega_backend.rs:205-212),
a bf16 copy (matmul_api.c:`omega_gpu_tensor_upload_f32`), and a packing buffer (64 column tiles).
For lm_head that is 1.45 GiB + 0.72 GiB of ordinary memory at once.

### (c) Per-request serving

The key finding: scratch buffers are **grow-only with exact size, freed and re-requested from the
driver on every increase**. `omega_gpu_session_scratch` (session.c:97-105): if the request is
larger than the current capacity, free the old buffer and allocate a new one of exactly the new
high-water size. No headroom, no geometric growth, no pre-reservation. OBSERVED.

| Buffer | Size formula | Grows with | Reallocated mid-serving? | Freed |
|---|---|---|---|---|
| Attention staging pool `g_pool` (attention_api.c:600, run_launch :609) | f32 path: 2 * seq_len * kv_heads * head_dim * 4 = 8 KiB per token for Qwen3 (attention_api.c:711-714) | context length | **Yes.** Each new context length is 8 KiB more than the last, which is more than the 4 KiB rounding, so every generated token (and every prefill chunk) triggers one free plus one new driver allocation, first layer of the step only; the other 35 layers reuse it | at session close |
| Attention q / out | q_heads * head_dim * 4 = 16 KiB each, times number of sequences in the call | batch size | only when the number of sequences rises | at close |
| Attention table | a few bytes | batch | tiny | at close |
| Matmul activations `a_buf` | round16(m) * kp * 2 | rows per call m | when m or kp exceeds the previous maximum | at close |
| Matmul results `c_buf` | round16(m) * np * 4; logits step: m = number of decoding sequences, np = 151936 (9.3 MiB at m <= 16, 37 MiB at 64, about 148 MiB at 256) | concurrent decodes, prefill chunk (128 rows) | when m rises past the previous maximum, or a bigger np appears | at close |
| Elementwise a/b/c (`run_abc`, elementwise_api.c:978, :1074) | each launch is split to at most 64 CTAs (EW_MAX_CTAS), so at most 64 rows x 2560 x 4 = 640 KiB per buffer, independent of context | nothing past the first full-size call | no (after the first 64-row call) | at close |
| Kernel code, matmul (matmul_api.c:144-167) | pages | (kp, np, grid_x) | **INFERRED yes; since OBSERVED on the fake layer, see Update.** The key includes grid_x, which depends on rows (grid_x = budget / ceil(m/16), capped by np/8). With the daemon CTA budget of 256: decode (m = 1) and prefill chunk (m = 128) give 6 shapes each (12 distinct, k and v share one), and the cache has 8 slots, evicted round-robin (`cache_next`). Alternating prefill and decode, or different decode batch sizes, therefore frees and re-allocates code buffers during serving | at close |
| Pushbuffer, cbank, marker, QMD | fixed (session.c:127) | none | no (each launch rewrites the same buffers) | at close |
| Resident weights | fixed after (b) | none | no | when the `ResidentTensor` drops |
| KV cache pool (`UnifiedKvTensorPool`, Fp32, shared_kv.rs:44-57) | 288 KiB per token (36 layers x 8 heads x 128 x 2 x 4 B) COMPUTED; declared context 262144 would be 72 GiB, so `AIEN_KV_CONTEXT_TOKENS` must lower it | context | n/a: **host memory, not driver memory** | daemon exit |
| Per-call host gather of K and V for the f32 attention kernel (omega_backend.rs:464-490) | 2 * n * 4096 B host vectors | context | host only | per call |

Bound on the attention scratch: the f32 attention kernel refuses `seq_len > 4096`
(`OMEGA_GPU_ATTN_MAX_GQA_CTX`, attention_api.c:703, header :83). A longer context is a chip error,
which is a fatal fallback in a production build (strict.rs). So on this path the pool is at most
2 * 4096 * 8 * 128 * 4 = 32 MiB COMPUTED, and Qwen3 serving above 4096 tokens is not possible
natively today regardless of memory. That makes a pre-reserved envelope small and cheap.

### Answer to #277's open question

Yes. Anything else allocates driver memory per call later: the attention staging pool grows by
one request per generated token, matmul scratch grows with batch rows and decode batch size, and
matmul kernel code buffers are probably evicted and rebuilt when prefill and decode alternate.
OBSERVED for the scratch rule, INFERRED for the code-cache thrash. Opening the session first
(#278) only guarantees the channel context and the first allocations happen early. It cannot by
itself cover these later requests. Candidate 1 is therefore incomplete unless start-up also
reserves every scratch at its maximum, and only candidate 2 (smaller footprint) keeps MemFree high
for the whole run. The comment on #277 (source check at omega 8887454) reached the same
conclusion; this table re-checks it at 01f6a74 and adds the 4 KiB per-token rule, the 4096 token
bound, the lm_head resident copy and the code-cache thrash.

## 2. Ownership design (candidate 2)

### Goal

On the GB10 production path, the process never holds an f32 copy of a matmul weight. Each matrix
goes from the checkpoint's bf16 bytes to the GPU, and the host keeps only what the host really
reads.

### What the host still needs as f32

| Tensor | Used by | Plan |
|---|---|---|
| `embed_tokens` (1.45 GiB f32) | token embedding row gather (transformer_backend.rs:733, :1036) | keep as f32 in the first cut. Later cut: keep bf16 (0.72 GiB) and decode a row on demand |
| norm weights (`input_layernorm`, `post_attention_layernorm`, `final_norm`, `q_norm`, `k_norm`) | the rmsnorm ops take `&[f32]` and stage them to the GPU every call | keep as f32, about 1 M values, negligible |
| lm_head (tied) | `compute_logits` | resident only (shares the embedding's values, own transposed device copy) |
| q/k/v/o/gate/up/down | matmuls | resident only |

### Storage type

In `aien-inference-abi/src/weights.rs`:

```
pub enum MatrixWeight {
    Host(Vec<f32>),            // CPU reference, dev builds, tests, non-GB10 models
    Resident(ResidentWeight),  // GB10 production: owns the device copy, no host values
}
pub struct ResidentWeight { id: WeightId, k: usize, n: usize, tensor: ResidentTensor }
```

`TransformerLayerWeights` fields `q_proj .. down_proj` become `MatrixWeight`; `lm_head` becomes
`Option<MatrixWeight>`; `embed_tokens` and the norms stay `Vec<f32>`. `TensorBackend::matmul_vec`,
`matmul_batch` and `compute_logits` take `&MatrixWeight` instead of `&[f32]`. The 23 call sites in
transformer_backend.rs change mechanically. `ReferenceCpuBackend` accepts only `Host`; given a
`Resident` it returns a loud error (it has no host values to compute with), which is consistent
with strict.rs (a production build refuses CPU fallback anyway). The dev build with
`AIEN_DEV_FALLBACK` keeps working by loading `Host` (no residency) or by a documented
"resident plus host" load mode that keeps both for parity tests.

### Backend resident cache

`OmegaGb10Backend` today keys its cache on (pointer, length, in, out) plus a sampled fingerprint
(omega_backend.rs:52, :194-199) because it cannot tell when a borrowed slice changes. With
`ResidentWeight` the device copy is owned by the weight and immutable, so no cache and no
fingerprint is needed for it: the backend calls `matmul_f32` on the handle. A `Host` weight on the
Omega backend keeps the existing lazy path (tests, TinyLlama, Llama-3.2-1B continue to pass
unchanged until they are migrated). Dropping `TransformerWeights` drops each `ResidentTensor`,
which calls `omega_gpu_tensor_free` (aien-omega-gpu lib.rs:306-311). Ownership becomes one-way and
visible in the type.

### Upload straight from the checkpoint

- A new safe wrapper `ResidentTensor::upload_bf16(k, n, &[u16])` over the already declared FFI
  `omega_gpu_tensor_upload_bf16` (ffi.rs:107; omega matmul_api.h:88). A bf16 value widened to f32
  and rounded back by omega is bit-identical, so the resident data equals today's. COMPUTED from
  the conversion definition; the parity test checks it.
- Safetensors stores W as `[out][in]` (so `[n][k]`); the engine wants `[k][n]`. Cut 1 transposes in
  Rust into a bf16 buffer per matrix (largest 9728x2560x2 = 47.5 MiB; lm_head 0.72 GiB), then
  uploads. Option for a later omega change: accept `[n][k]` input and transpose while fragment
  packing, removing the transient (omega_matmul_frag_pack_b already walks the matrix in tiles).
- Loading becomes streaming: for each tensor, read from the shard (memory-mapped, so the pages
  are clean reclaimable page cache, not process memory), convert or upload, move on. The whole-file
  `fs::read` (checkpoint.rs:501, :581) that made the 7.49 GiB raw copy disappears for this path;
  the existing path stays for non-GB10 loads. Release is automatic: nothing holds the bytes after
  the loop.
- Order at daemon start: open the GPU session and run the probe first, then stream-load and
  upload. Every large driver allocation then happens during load while the host holds almost
  nothing, and the lazy-upload-on-first-prefill step disappears. This subsumes the ordering change
  in #278 (which then becomes a one-line no-op for GB10 Qwen3 and can stay for the other models).
- Non-bf16 checkpoints (f16 or f32 shards) go through a one-matrix-at-a-time convert to bf16 and
  upload; never a whole-model f32 copy.

### Serving-time reservation (no driver allocation after start-up)

Two parts, because scratch and code buffers behave differently.

1. Scratch, no omega change (first cut): after the session opens and the weights are resident,
   run a "serving envelope" warm-up that calls the backend once at the maximum of each dimension:
   attention at seq_len 4096 (32 MiB pool), matmul at the largest planned rows for each shape
   (prefill chunk 128 rows; decode rows = the scheduler's `max_batch_size`, 256 in the daemon,
   commands.rs:1478) including the logits shape, and elementwise at 64 rows. Because scratch only
   grows and never shrinks (session.c:97-105), every later call fits and no realloc happens.
   Envelope size COMPUTED: about 32 MiB attention + about 150 MiB matmul results at 256 decode rows
   + about 20 MiB matmul inputs + under 3 MiB elementwise, roughly 0.2 GiB. The envelope limits
   come from startup configuration (context cap, scheduler limits), logged, and a request beyond it
   is refused with a clear message instead of reaching the driver.
2. Kernel code, needs a small omega change: raise `CACHE_SLOTS` in matmul_api.c:28 from 8 to 16
   (covers 12 shapes with headroom), so no code buffer is evicted. Before relying on it, confirm
   the thrash with a CPU-only test on omega's fake native layer (`tests/fake_m16_native.c` already
   counts `uncached_calls` and `free_calls`): replay one prefill chunk, one decode and one more
   prefill and assert the allocation counters. If the thrash is not real, skip the change.
3. Guard (omega, optional): a "seal" call after warm-up so that any later allocation attempt
   returns a named error (`ENVELOPE_EXCEEDED`) instead of an opaque NV_ERR_NO_MEMORY, and a pair of
   counters (allocs, frees since seal) exposed through the Rust crate for the receipts.

### Peak memory estimate (COMPUTED, Qwen3-4B bf16, to be measured)

| | Today (measured / computed) | After cut 1 and 2 |
|---|---|---|
| Process memory at load peak | 22.48 GiB measured (f32 14.98 + raw bf16 7.49) | about 2.2 GiB (embedding 1.45 f32 + largest transient 0.72 for lm_head bf16 transpose); shard pages are reclaimable cache |
| Process memory in steady state | 14.98 GiB f32 | about 1.5 GiB (embedding + norms) |
| Pinned driver memory, steady | about 6.8 GiB quoted, 7.48 GiB computed | 7.48 GiB resident + about 0.2 GiB envelope + session |
| Total the machine must find | about 22.5 GiB | about 9.2 GiB |
| Transient at each upload | up to 2.2 GiB (lm_head f32 transpose + bf16) | up to 0.72 GiB |

Saved: about 13.5 GiB of ordinary memory for the layer matrices, 14.2 GiB if the embedding is
later kept as bf16. The KV pool (host) is unchanged and is set by `AIEN_KV_CONTEXT_TOKENS`.

### Failure modes and how the design handles them

- Driver refuses an allocation during load: the daemon fails to start with the stage text and
  MemFree, as in #239, before serving. Fine; this is the safest place to fail.
- Driver refuses an allocation during serving: should not happen inside the envelope. If it does,
  the session is dead (`faulted`), the strict build refuses to fall back, the request fails loudly.
  Not recoverable without a restart; the counters and the sealed guard make it visible.
- Request exceeds the envelope (context above 4096, batch above the declared maximum): refused with
  a message naming the envelope, before any chip call.
- Non-bf16 or non-Qwen3 models: keep `Host` path until migrated; the old behaviour is untouched.
- A `Resident` weight handed to the CPU reference backend: loud error by construction.
- Parity regression: guarded by the existing Qwen3 forward parity test (max abs logit difference
  bound 0.15, same top token) run on the new load path.
- Idle GPU memory is no longer reclaimable once pinned: unchanged from today, and the point of the
  budget.
- Memory fragmentation at start-up (UNKNOWN whether it matters, see #277 amendment 1): the design
  allocates large buffers first, early, which is the best available mitigation but is not a proof.

## 3. Declared test conditions and acceptance criteria

These follow the conventions of `~/workspace/hive/WQWEN/c277/run-277.sh` (sealed script, written
declaration before any run, one variable, base first, fixed run order, camera, no kills, receipts
with `SHA256SUMS`). They are written now and must be frozen in a DECLARATION file before the first
run.

Question: with the weight-ownership change, does the Qwen3-4B GB10 path pass in the organic
low-MemFree state where the unchanged baseline fails, without dropping the page cache?

- Condition (must hold, measured and recorded before every run, run refused if not met): organic
  state from normal use, `MemFree <= 25 GB` and `Cached >= 80 GB` (as #277 defines). No
  `drop_caches`, no setting change, and no scripted cache fill as the qualifying state (the
  #277 attempt shows a scripted fill did not reproduce the failure). The script reads
  `/proc/meminfo` and `/proc/buddyinfo` first and exits without taking the GPU hold if the numbers
  are not met, writing `STATE NOT MET` to the receipts. Waiting for the state is allowed; creating
  it artificially is not.
- Arms (one variable: weight ownership and reservation): base = unchanged main binary; fix = the
  implementation PR binary. Both built from sealed commits and checksummed in
  `receipts/binaries.txt`; same omega pin, same model revision (cdbee75f), same test
  (`real_qwen3_4b_instruct_2507_on_gb10_matches_transformers`) with
  `AIEN_GB10_QWEN3_DECLARED_ATTEMPT=1`.
- Order: base once, then fix three times (at least 3). The condition is valid only if base FAILS
  in the same organic state (expected signature: `RM_ALLOC ... status 0x51`, stage
  `m16_native_create_channel` or an upload). If base passes, the state did not reproduce the
  failure, the fix runs are skipped, nothing is claimed (INVALID CONDITION again).
- Camera: 1 s `/proc/meminfo` sampler for the whole run, `buddyinfo` and `dmesg | grep NVRM`
  captured before and after each run, process peak RSS (`/usr/bin/time -v` or `/proc/PID/status`
  VmHWM), session allocation and free counters before warm-up, after warm-up and at the end.
- One GPU hold under the quietlock procedure; no test is killed; the camera is stopped last.
- A serving leg, because the load-time leg alone would repeat #277's stated limit: after the
  parity prompt, decode at least 256 tokens in one sequence and run one second prompt of at least
  1024 tokens, with context reaching at least 2048, and one change of batch size if the harness
  can. Allocation counters must show zero driver allocations and zero frees between the end of
  warm-up and the end of the run.
- PASS = baseline FAIL in the same state, and fix 3 of 3 runs satisfy all of: test assertions
  (max abs logit difference <= 0.15, same top token), `chip_errors = 0`, `fallbacks = 0`, all ops
  native, serving-leg counters show 0 driver allocations after warm-up, process peak RSS below
  12 GiB (vs 22.5 GiB measured today; threshold set before any run), no `NVRM` allocation error in
  dmesg. Any fix failure is a FAIL; no reruns.
- Limits to state in the verdict: one model, one hardware unit, one driver (580.173.02), batch size
  coverage as run, context only up to what the serving leg reached (and never above 4096 on this
  path).
- The default refusal in `omega_model_refusal_with` (omega_backend.rs:543-) stays until the verdict
  is PASS and merged. Flipping it is a separate PR that cites the verdict.
- Do not drop caches to qualify. If the organic state never recurs, the answer is "not
  qualified", not a staged state.

## 4. CPU-side changes that are safe to land now (behind the existing refusal)

| Change | Diff estimate | Risk | Notes |
|---|---|---|---|
| MemFree, Cached and MemAvailable in the start-up log and in each session-open attempt line (#277 item 3): extend `read_mem_available` into a small meminfo reader, add a second closure or a struct to `open_gpu_session_with_retry` | about 60 lines plus 2 unit tests (parse, missing field) | none: logging only, stub-testable | touches the same function as #278; land after or rebase on it |
| Print the planned resident total and the planned serving envelope at start-up (computed from the config, like `KvPoolPlan`), and the expected pinned total next to MemFree | about 80 lines plus tests | none: no allocation behaviour changes | gives the operator the numbers the declaration needs |
| `OmegaGb10Backend::resident_bytes()` and a count of uploads | about 25 lines | none | feeds receipts |
| Safe wrapper `ResidentTensor::upload_bf16` and its stub/native tests | about 50 lines | low: not called by default | prepares cut 1 |
| Streaming checkpoint reader that yields one tensor at a time from a memory map (new function, old one unchanged) | about 120 lines plus tests | low: unused until wired | prepares cut 1 |

Not safe to land now without a chip run: the serving-envelope warm-up (it launches kernels at new
sizes on models that are accepted today, so it needs chip qualification), the `MatrixWeight` type
change (cross-crate, about 700 to 900 lines changed across abi, inference-runtime, runtime, cli
and tests; it needs its own reviewed PR and the parity test), and the omega `CACHE_SLOTS` / seal
changes (separate omega PR, then an `omega.lock` bump).

## 5. Recommended implementation order

1. Cut A (CPU, small, safe now): MemFree log plus resident and envelope plan lines plus resident
   byte counters. Gives the next declared attempt its numbers and costs no chip time.
2. Cut B (DONE, see Update: omega#332 measured, omega#333 reserves) (CPU test on omega's fake layer): confirm or refute the matmul code-cache thrash and the
   per-token pool reallocation with allocation counters, no GPU. This decides whether the omega
   changes are needed.
3. Cut C (the real change): `MatrixWeight`, streaming bf16 upload, session-first order, envelope
   warm-up; stub parity tests on CPU first, then one declared chip attempt as in section 3.
4. Cut D: flip the default refusal, only after a PASS verdict.

Open items I could not settle from source: why the baseline passed in the scripted state (trigger
UNKNOWN); whether the driver needs physically contiguous pages for the failing buffers (DOCS
SILENT per #277); how many rows a real mixed prefill and decode step puts into one matmul in the
daemon (the 256-row envelope is a bound from `max_batch_size`, not a measurement); the exact
resident total (7.48 GiB computed against about 6.8 GiB quoted; the gap is about the lm_head: 6.76 GiB is the layers alone, the lm_head adds 0.72 GiB, so the quote likely omits it).
