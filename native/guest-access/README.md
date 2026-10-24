# Guest admission for the two Spark models

This is the native Omega admission component for [core #724](https://github.com/aien-dev/aien-sovereign-core-mv/issues/724), ledger L792. It is not wired into an HTTP listener or deployed. The current local 35B service still does not enforce this policy. No operational credential is included or issued.

The module supports 16 guest entries and 32 active request reservations. Model bit 1 means the qualified 35B release; bit 2 means the qualified 27B release. Binding these bits to exact model identities is the serving adapter's responsibility. A request selects exactly one bit; an entry may allow both. This delegates inference admission only, never access to tools, shared boards, internal agents, files or administration.

## Integration contract

All calls and reads of `garesult` belong to one serialized server owner. This module is not thread-safe or a platform authority root. The existing listener, scheduler and credential-custody owners remain responsible for their boundaries. Array arguments are fixed-size local arrays or borrowed parameters, as required by the pinned seed compiler; global arrays cannot be passed as arguments.

1. Trusted provisioning calls `gaprovision(slot, key[32], cfg[6], now)`. The key must come from existing credential custody with 256 bits of cryptographic randomness. This module only checks its fixed length, nonzero content and absence from the current table; those checks do not establish entropy. Configuration fields are model mask, absolute expiry seconds, simultaneous-request cap, total context cap, requested-output cap and cumulative work budget. Configuration limits are 32 requests, 131072 context tokens, 16384 output tokens and 2^60 work tokens. Actual model limits may be smaller and must also be enforced by the serving adapter.
2. The existing HTTP parser accepts exactly one bearer header and decodes its wire representation without truncation or ambiguous duplicate headers. Call `gaauth(key[32], decoded_count, model, now)` before bounded preparation or tokenization. Only count 32 is accepted. Success sets `garesult` to `[0, 0, slot+1]`. It reserves no capacity and charges no work. Bind all subsequent preparation state to that authenticated guest and enforce the existing global preparation/connection limits.
3. After bounded preparation, call `gaadmit(key, decoded_count, model, prompt_tokens, requested_output_tokens, now)`. This authenticates again, so revocation or expiry during preparation is rejected. The prompt must be nonempty, output must be nonzero, and their sum must fit the context cap. Success returns zero and sets `garesult` to `[0, handle, slot+1]`; failures clear the handle and guest fields. Only then allocate model state or dispatch inference. Guest identity must scope all reusable state and prefixes; the admission component does not itself isolate KV, recurrent state or shared memory.
4. Before each bounded compute or output step, call `gacheck(handle, model, now)`. It rejects a revoked or expired entry, wrong model, stale handle or clock rollback. The integration must actually cancel work and stop output when this returns an error. A permission check alone cannot cancel an already submitted GPU operation.
5. On completion, cancellation, disconnect or a refusal from `gacheck`, call `gafinish(handle)` exactly once after releasing associated model/output resources. It works after expiry or revocation and frees the request slot. Double finish and late callbacks carrying an old handle are rejected. Handles are internal to one process lifetime, never client bearer credentials or durable capabilities.
6. `garevoke(slot)` immediately disables further authentication and continuation checks. It deliberately retains outstanding accounting until those requests finish. Replacement cannot occur while requests remain active. Rotation requires a different key; reinstalling a key still present in the table, including a revoked key, is refused. Historical key uniqueness across rotations/restarts remains credential custody's responsibility.

`now` is trusted wall-clock seconds, never request input. Observed time cannot decrease during the process lifetime. In particular, observing expiry advances the clock floor before denying access, preventing resurrection on clock rollback. A clock correction may temporarily refuse otherwise valid requests. Restart recovery and fresh key provisioning belong to the trusted supervisor; active handles must never survive a process restart.

Work budget is charged by `prompt_tokens + requested_output_tokens` at admission and is not refunded for early completion or cancellation. It is a cumulative reservation quota, not a time-based rate limiter or actual-output billing counter. Failure to admit does not spend budget or acquire capacity. Exhausted handle generations permanently retire their slot instead of wrapping to an earlier handle.

Status values: 0 success, 400 invalid trusted configuration or request size, 401 invalid/revoked/expired credential, 403 model denied, 409 conflicting rotation or stale handle, 429 capacity/budget exhaustion, 503 clock rollback or inconsistent internal accounting. `gaprovision`, `garevoke`, `gacheck` and `gafinish` return status directly; only `gaauth` and `gaadmit` publish `garesult`. `gareserve`, `gakeyvalid`, `gamatch`, `gaclock`, and `gahandle` are internal helpers, not public adapter entry points.

## Credential handling and remaining gates

Keys exist only in process memory; no function writes credentials or prompts. Lookup scans all 512 table bytes for a correctly shaped key, without early exit on a matching prefix. This is not a constant-time machine-code certification. Trusted provisioning and memory protection, transport TLS, token generation/delivery/storage, HTTP framing, per-guest connection and preparation limits, actual request cancellation and model/session isolation remain integration and review gates. The module cannot safely be exposed by itself.

No production code or native backend is replaced here. Existing work under core #710-#721, Omega #666 and CUDApp #63-#67 continues to own model correctness, speed, native GPU attachment and serving. Host admission tests do not establish any model performance or public release readiness.

## Reproduction

The tested compiler is `seed/out/omc` from immutable Omega commit `1dffceaeb0dd26fb69ea64d367a49a9950c25a62`, SHA256 `9f48f66a9ee9f6a166df615a7ab9c48bf5b06ac2ba544ac51e52e9a3d290a6c4`. That source is an open draft dependency, not merged-main qualification. Extract the compiler from that Git object and verify its digest; do not copy a mutable owner's build output. The first attempted local copy changed during ongoing owner work and was rejected by its digest before use.

On Linux AArch64, concatenate `admission.om` and `tests.om`, pass them to that compiler on standard input, save its standard output as an executable, and run only if compilation exits zero. The expected result is `GUEST_ADMISSION_PASS 939`, exit 0. Tests use public synthetic fixtures and never load models or open network ports. Executable hashes, native output and eight compiled negative controls are recorded in `evidence/guest-access-724/`. Each negative control compiles successfully and the same tests reject it with exit 1. No external runtime or foreign-language implementation is linked.

From this repository root, with `OMEGA_CHECKOUT` pointing to an existing Omega Git checkout containing the pinned commit:

```sh
set -eu
: "${OMEGA_CHECKOUT:?Set the existing Omega checkout path}"
guest_run=$(mktemp -d)
git -C "$OMEGA_CHECKOUT" show 1dffceaeb0dd26fb69ea64d367a49a9950c25a62:seed/out/omc > "$guest_run/omc"
test "$(sha256sum "$guest_run/omc" | cut -d ' ' -f 1)" = 9f48f66a9ee9f6a166df615a7ab9c48bf5b06ac2ba544ac51e52e9a3d290a6c4
chmod u+x "$guest_run/omc"
cat native/guest-access/admission.om native/guest-access/tests.om > "$guest_run/tests.om"
"$guest_run/omc" < "$guest_run/tests.om" > "$guest_run/tests.elf"
chmod u+x "$guest_run/tests.elf"
"$guest_run/tests.elf" > "$guest_run/result.txt"
test "$(cat "$guest_run/result.txt")" = 'GUEST_ADMISSION_PASS 939'
"$guest_run/omc" < "$guest_run/tests.om" > "$guest_run/rebuilt.elf"
cmp "$guest_run/tests.elf" "$guest_run/rebuilt.elf"
```

Independent final-source review, repository gates, HTTP/native model integration, external access tests and deployment remain required. The tests are author qualification, not independent approval.
