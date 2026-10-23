# Sealed KV nonce and recovery tests (#708)

NOT A MASTER PLAN. Scope: aien-kv-store host persistence under omega-mv #639. Base 754c07b118f41ad899607b60329bd851b1ccb13e, parent L691/storage L693. Existing PR #694 (byte corruption), #612 (host recovery followups) and #576 (folder lock) retain their branches and owners; this patch does not edit their files except the current main's owning host_fs module at the seal test include. No production RNG configuration, cipher migration, disk format change, live vault key, GPU or service action.

## How it works

Production always calls ring SystemRandom through private fresh_nonce(). The source override and its per-key field exist only under cfg(test); there is no environment-controlled fallback or deterministic production mode. Entropy is acquired before modifying the caller buffer. Existing host_fs error propagation already precedes temporary-file creation. Tests inject an error into that actual seal call, then assert no durable completion, lookup entry, extra write, final object or temporary file. No alternate filesystem implementation is tested.

Repeated-source test deliberately seals two different payloads with the same synthetic key/nonce: ring accepts them and ciphertext payload XOR equals plaintext XOR (tags excluded). That row is a demonstrated unsafe control, not a protected outcome. AAD tests mutate each of 4068 bound bytes independently and require refusal plus a zeroed output. Concurrency uses one injected sequence across eight recreated key objects; actual subprocess tests start two fresh test processes and restore/authenticate the older file. These prove the named fixture behavior, not OS entropy independence or uniqueness of random samples. Existing integration tests retain crash recovery, complete identity binding, old-key and corrupted-file discard/recompute behavior.

## Randomness, freshness and key budget

For independently uniform 96-bit nonces, the birthday union bound is q(q-1)/(2*2^96), aggregated across ALL files, processes and restores using one key, not per process. Wolfram independently evaluated q=2^24: bound 1.776356733521132e-15; q=2^32: 1.164153217998298e-10. This counts nonce collision only, not the complete AES-GCM forgery/security bound (payload/block counts and failed verification attempts also matter).

Operational recommendation for this bounded store: retire a key before 2^24 total sealing operations, sooner if a deployment's acceptable risk or GCM block/verification budget requires it. Recreated processes must share the aggregate accounting. **Implementation gap:** no durable cross-process per-key usage counter or enforced rekey budget exists today. This ticket does not pretend tests enforce that policy. OS RNG failure refuses writes; repeated RNG output is NOT detected. Disk rollback alone does not rewind the OS RNG; cloning a running VM/process and its entropy state may violate the independence assumption. Do not use shared keys across cloned entropy states. A valid restored ciphertext is authenticated, not fresh: an external monotonic authority is required to prove recency. Old-key files are discarded after rotation, not migrated silently.

## Reproduction and controls

`CARGO_TARGET_DIR=<private-target> cargo test --locked -p aien-kv-store`

`CARGO_TARGET_DIR=<private-target> python3 scripts/crypto_reuse_mutants.py <private-results>`

The Python driver is external serial test orchestration only. Four compiled mutants must fail a named assertion: swallowed entropy error, host publication after failed sealing, AAD removed from BOTH seal/open, and omitted failed-open zeroization. Builds, signals, timeouts and missing commands are INVALID, never detections. It restores source even after refusal. Do not run another test/build concurrently in this tree.

Evidence retains initial unconsumed-seam RED and no-publication RED. The first publication assertion incorrectly compared total lifetime writes with zero, which included pre-job recovery bookkeeping; the corrected fixture compares before/after the job and still requires no files or durable result. No production checks were weakened. All keys and payloads are synthetic.

## Acceptance

- [x] Deterministic error/repetition/concurrency/recreated-key and subprocess fixture tests.
- [x] Real host_fs sealing failure cannot publish; complete AAD mutation and payload-zero tests.
- [x] Four compiled behavioral mutants rejected by named assertions; existing crate suite passed.
- [x] Budget recommendation, actual enforcement gap and clone/rollback assumptions explicit.
- [ ] Exact committed-source final replay, independent review and sole-desk PR acceptance.
