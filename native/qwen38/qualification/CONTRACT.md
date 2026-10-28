# Qwen3.8 qualification contract v2

Core #710 under #679. This is a bounded experimental contract, not a release
qualification. Frozen thresholds below precede any candidate measurement.
Changing a threshold or fixture requires a new contract version and new paired
runs. Status: **BLOCKED for model/hardware qualification**. The receipt checker
is native `.om` CPU code. Its synthetic tests are not inference measurements.

## Immutable inputs and entry gate

Model: `Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. The recorded 81-file upstream
manifest is retained with the evidence. Recheck every file before execution.
All 81 files were restored at the pinned revision on 2026-10-23 into
`/home/drakestapleton/models/qwen3.8-27b-fp8-017b9c7a`. Every download matched
the historical manifest; a second full SHA256 check passed. Raw recovery and
recheck receipts are in `evidence/qwen38-710-native-v2/`. This establishes
current checkpoint possession, not model loading or inference qualification.

Before running, bind the following in one immutable experiment manifest:

- Contract and fixture bytes; model, tokenizer, template and MTP head hashes.
- Exact input token arrays, expected prompt lengths and continuation boundaries.
- Core/Omega/CUDApp and transitive source pins, compiler seed and binary hashes,
  actual linked backend, executable hash, effective flags and tensor layouts.
- Device/driver identity; native allocation/upload/dispatch/wait/readback evidence.
- Reference outputs, logits, task answers and state checks, captured before W4 tuning.

An absent identity, CPU stub, different model, wrong linked backend, missing
full-model gate or silent fallback refuses entry. An archive's symbol name or
a branch name is insufficient proof. Existing CUDApp #33 standalone benchmark
inputs and gates remain separate and unchanged.

## Quality and state gates

1. Same-precision ordinary versus speculative greedy decoding: exact emitted
   token IDs and stop reason on every fixture. Ties use the lowest token ID.
   Report first divergence, reference margin and both logits. No tolerance in
   token identity, even when a finite-precision explanation is available.
2. Packed projections against scalar decoded-value reference: finite outputs,
   `abs(error) <= 1e-4 + 1e-3 * abs(reference)` for every element. This is a
   numerical acceptance threshold, not an observed error. Test all nibble
   values, sign/zero, scale extrema, tails/alignment and malformed layouts.
3. W4 versus frozen FP8: mean token NLL increase at most 0.05 nats separately
   for held-out code and prose; no individual fixture increase above 0.10.
   Publish perplexity ratio as well as raw NLL, top-1 agreement and logit margins.
   These limits must pass at each context length. No identity claim across precisions.
4. Every task with a deterministic answer must remain correct. Both FP8 and
   candidate must pass every frozen task check; a bad FP8 answer cannot excuse
   candidate failure. Report unscored prose separately. No quality promotion
   from LICENSE-256 or these small smoke fixtures alone: the held-out corpus
   must be frozen with per-token arrays and reference scores before optimization.
5. KV, all 48 GDN states, convolution tails, logical position and drafter input
   describe exactly the same consumed prefix. State ownership and token counts
   have zero tolerance. Restore-plus-suffix versus fresh prefill: exact tokens
   and stop reason; state elements obey the numerical tolerance in item 2.
   No state compression/precision change is covered by this contract.
6. Include each rejection position, all/zero acceptance, pending correction or
   bonus, EOS/stop/cancel, boundary capacity, second turn, independent sessions,
   repeated restore, permutation, padding, remapped slots and delayed producers.
   Compiled wrong-ancestor/missing-dependency/state-boundary mutants must fail.
7. Sampled generation uses ordinary decoding until exact acceptance/correction
   distribution and identical penalties have an independently reviewed test.
   For supported discrete test distributions, total variation <=0.01 over at
   least 100000 fixed-seed draws, alongside analytic probability checks within
   1e-6 absolute error. Statistical tests alone do not establish semantics.

## Workloads and timing

Four arms: 0 FP8, 1 FP8+MTP, 2 W4, 3 W4+MTP. Keep attention and recurrent
precision fixed. Qualify MLP first; head precision is a separate experiment.
Freeze packing/scales/rounding before W4 evaluation. Select actual packed format
from existing measured backend support; neither MXFP4 nor NVFP4 is presumed
implemented. Never call expanded FP8 weight traffic W4 acceleration.

For each frozen coding/prose input and context (short, 32768, 65536, 131072
within actual capacity), execute cold, warm-prefix and second-turn profiles.
Reserve suffix and output room within context capacity. At 128K, do not claim
131072 input tokens plus output fit a 131072-position cache. Record requested
and actual counts. Missing capacity is NOT_RUN, not a silently shortened input.
Use natural EOS. Stop/cancel/capacity tests belong to correctness; length-limited
outputs are insufficient samples for natural completed-task speed.

Exclude one warmup per arm. Five measured paired rounds minimum, identical
inputs and sampling parameters. Round 0 runs 0,1,2,3; odd measured rounds run
3,2,1,0 and even rounds 0,1,2,3. Do not remove inconvenient valid samples.
Retain rejected trials and reasons; replacements use a new explicit trial ID.

Report raw counts and microsecond durations for cold load, queue wait, prefill,
first visible token, ordinary decode, draft, verify, commit, and host overhead.
Exclusive phase durations reconcile exactly to completed-request elapsed time;
TTFT overlaps them and is not added twice. Host/device work that overlaps must
be reported as a separate trace, never summed twice into elapsed time.
Record peak unified resident memory and transient state bytes independently.
Report median and nearest-rank p95 of all metrics, all raw samples, and accepted
emitted tokens / total elapsed time. Expose queue-free service cost separately.
Include drafting, rejected work, verification and commit. A single kernel's
speed or tokens drafted is not useful-output speed. With five samples p95 is
the maximum; do not imply a precise tail estimate.

MTP starts at k=0,1,2,4. Adapt only from completed rounds, with bounded probing
and ordinary fallback when full elapsed cost loses. Compare complete-task cost
and quality before considering a tree; no tree is required by this contract.

## Contamination

Use the established quiet measurement scheduler and GPU lock, with build work
completed before acquisition. Never kill another job or alter the live driver.
Retain scheduling request/admission/release times and a read-only process trace.
Missing exclusive admission, any overlapping GPU job/build, any driver-error
increase, swap activity, thermal throttling, clock range outside the frozen
run envelope, incomplete telemetry or memory admission failure invalidates the
performance trial. Preserve the refusal and do independent work. Freeze the
machine-specific clock/temperature envelope during baseline calibration.
Do not infer an exclusive run from a lock file alone.

## Receipt arithmetic format

`receipt.om` reads exactly 576 unsigned decimal integers (24 rows x 24 fields),
separated by ASCII whitespace, <=65536 bytes. Every integer <=10^12. It refuses
trailing non-whitespace, missing fields, wrong order and numeric overflow.
One invocation covers one fixture/context/profile. There is no signature or
hash verification in this arithmetic layer. Assertions in fields 16,17,18,23
must be backed by the source-bound manifest and raw traces above. A malicious
producer can forge these values: exit 0 is **arithmetic consistency only**.

| Field | Meaning |
|---|---|
| 0,1 | arm 0..3, round 0..5 (0 warmup) |
| 2 | emitted accepted output tokens, including pending token |
| 3,4,5 | queue, cold load, prefill microseconds |
| 6 | request start to first visible token, microseconds |
| 7,8,9,10,11 | exclusive ordinary decode, draft, verify, commit, host microseconds |
| 12,13 | total completed-request microseconds, peak unified bytes |
| 14,15 | driver-error delta, contamination flag (both must be zero) |
| 16,17 | external quality gate passed, actual native backend verified (both 1) |
| 18 | weights remained packed through reads (1 only for W4 arms) |
| 19 | stop kind: 0 natural EOS, others refused for completed-task statistics |
| 20,21,22 | prompt count, consumed total prefix count, emitted pending token count 0/1 |
| 23 | external full-state boundary check passed (1) |

Output v2 emits 32 newline-delimited integers per arm (128 total): arm ID;
median then nearest-rank p95 for each input field 2 through 13 in field order;
median/p95 integer emitted tokens per total second; median/p95 queue-free
service microseconds; median/p95 integer emitted tokens per service second;
maximum peak bytes. Service is total minus queue and must be positive. Each
rate is computed per request before sorting, not from separately aggregated
counts and times. Zero service is refused before division. Warmup is excluded
from every statistic, including peak memory.

Version 2 changes the output layout, adds a zero-service refusal and extends
synthetic checks. Input layout and predeclared quality thresholds are unchanged.
No actual model trial existed under v1. Never combine output versions. Raw
phase traces, transient state memory, paired confidence intervals and
authenticated quality results remain required external acceptance work.

Exit 2 malformed/capacity; 3 invalid timing/order/state/length; 4 contaminated;
5 unqualified quality/backend/packing; 6 I/O. No summary is emitted until every
row validates. A partial output write is an I/O failure, not a valid summary.

## Reproduction of CPU-only checks

Consume Omega `migration/osc-to-seeded-omega` at
`9b072eb492e106d23f7eed984c5645251601adee`, `seed/out/omc`, SHA256
`01b1545b23e8ff40cb5691e20b5df910ea4acaefde3495f0d62d270465b98cd6`.
Compile each `.om` file with that executable on stdin and redirect stdout to
its own executable output, checking exit status before marking it executable.
`fixture.om` consumes one byte A..Y. A, W and Y emit valid **synthetic** data, with
and without a pending token; Y varies phase costs and output counts in shuffled
order. B..Q introduce specific bad fields, R extra data,
S missing data, T ordinary decoding with draft time, U context overflow and
V an impossible first-token time; X a queue-only request with zero service. Feed its output
to the compiled receipt checker. Exact replay commands and exit table are in
the evidence directory. Shell commands currently perform test invocation;
native process/test orchestration remains with the existing compiler owners.
No new shell/Rust/C/Python test implementation is included as a replacement.
