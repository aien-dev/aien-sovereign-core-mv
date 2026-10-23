# Independent review disposition

The original static report is preserved as `review.md`, with its reviewed-file hashes. The parent wrote all implementation; the reviewer made no code changes and ran no tests.

## Reported division by zero: refuted

The cited expression is prefix Omega: `gt wait div compute 8` means `wait > (compute / 8)`. The divisor is the literal eight, not `compute`. Likewise `div previous 4` divides by four. The source has no division by a timing value.

The exact reported input `[1,2,4,2,1,0,0,0,0,0]` was executed against the unchanged native executable and returned `[3,0]`, status zero, empty stderr. See `review-zero-compute.json`. The original 28-case native suite already contained the zero-compute case. A separate 1,000,001-request run returned every expected response, status zero, empty stderr (`policy-lifetime.json`). No code change or weakened assertion was needed. These receipts do not qualify all one billion supported transactions.

## Benchmark residency caveat: accepted

The sweep reuses one allocated model/pager. It must not be described as independently cold at arbitrary capacities. The attempted eight-slot/48-layer run was chosen to make inter-layer eviction dominate; actual hit/miss/read-byte equality still needs checking. Hardware admission refused before model load, so there are no hardware timings or equality results to reinterpret.

After the static review, the benchmark gained explicit per-prompt KV reset for opted-in flows, per-step f32-bit SHA-256 logit comparison, and preservation of the original `generate` entry point when flow is not requested. This CLI delta is parent-reviewed and build-tested, not covered by the original independent review hash. The scheduling policy, bridge and FFN runtime hashes can be checked against the original report and `SOURCE_SHA256SUMS`.

## Qualification limits

No concrete deadlock, pin leak or ordinary returned-error output-transactionality defect was identified in the static review. That is a bounded source-review result, not proof of GPU correctness. CPU-stub full-model parity, native policy behavior, fault/lifecycle tests, successful native linkage, hardware execution and live-service deployment are separate gates. The last two remain unrun; native release-digest qualification also remains outstanding.

The first bridge test failure was a test-double encoding error: the shell peer emitted the wrong bytes. Corrected peers emit the explicit 16-byte little-endian `[1,0]` response. `focused-final.log` preserves the failure; `focused-corrected.log` preserves the corrected gate, without weakening response bounds. The inherited full suite's ignored tests remain listed in the raw output and are not counted as passes.
