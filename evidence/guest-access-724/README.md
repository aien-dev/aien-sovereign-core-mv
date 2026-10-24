# Native guest-admission qualification

Scope: core #724, L792. Author qualification on Linux AArch64. This is an unconnected component, not a deployed authentication boundary, native model result or release approval.

- Base: core `a24076008dca189aee8d55c1193bcb8f6fe8aa71`.
- Compiler source: Omega `1dffceaeb0dd26fb69ea64d367a49a9950c25a62`, `seed/out/omc`, verified against the Git object. The dependency is draft PR676, not merged main.
- Compiler digest and two byte-identical executable builds: `artifacts.sha256`. Functional/test source hashes: `source.sha256`.
- Actual executable output: `native-test.txt`, exit 0, 939 checks. Native fixture coverage includes all 16 entries, every key byte, both model scopes, expiry and backward-clock behavior, live revocation, full 32-request capacity, quota exhaustion, cross-guest accounting, stale/double finish, wrong-model continuation, and handle-generation exhaustion without wrap.
- Eight separately compiled negative controls: `mutants.json`. All compile and execute; each reports `GUEST_ADMISSION_FAIL` and exits 1. The substitution strings specify the exact mutated source. Each variant appends unchanged `tests.om` and uses the same pinned compiler.
- `elf.txt`: native AArch64 executable, no interpreter or dynamic section. There is no foreign-language runtime behind the tests.
- `bash scripts/agent-preflight.sh`: exit 1. The browser screenshot fixture fails with a WebSocket read error, and cargo audit reports eight advisories. Exact excerpts are in `preflight-failures.txt`. `Cargo.lock` and the failing browser source are unchanged from the base; no same-base rerun or waiver is claimed. Full original output remains in the local L792 handoff with its digest in `preflight-log.sha256`.
- Crumb verification: both the workflow-pinned `94fffbcf88492c5a1c54211b5126c8a6c370a98d` tool and the installed gate shim report all 19 crumbs current. Regenerate after changes.

Reproduction uses the existing seed compiler interface: concatenate the two `.om` files into a temporary input, compile through the verified compiler's standard input/output, require compiler exit zero before executing, then require the exact PASS line and exit zero. Repeat compilation and compare executable bytes. For a negative control, apply its recorded single substitution to `admission.om`, append unchanged `tests.om`, compile successfully, and require the recorded failure line and exit 1. These are native Omega test programs; shell commands only coordinate files and execution.

The first copy from a mutable owner build produced digest `de528263e7e7901544ecb8f10867b93138b07362eee73241a4a431104d233e08`, so it was rejected before compilation and replaced with the immutable Git-object compiler. Earlier development builds were not counted as qualification. A compile-error iteration also showed why executable existence alone is insufficient: an empty redirected output is not a native test PASS. Every counted qualification and mutation required a successful compile followed by the explicit test sentinel.

Review and integration remain open: independent source review, credential custody and issuance, exact HTTP bearer parsing, pre-tokenization connection/preparation caps, TLS, real cancellation, tenant-scoped model state, 27B native model attachment, final speed trials, merged consumer pins and external guest tests. No live service, GPU state, credential or network exposure was changed.
