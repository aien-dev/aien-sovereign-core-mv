# Evidence Receipt Adapter Specification for `aienos`

Scope: how `aienos` test setups emit machine readable qualification results
that `aien-proof` turns into sealed evidence receipts. This document is an
input contract only. Every JSON block marked EXAMPLE is a shape example, not
evidence. No gate has passed unless `aien-proof gate status` says PASS over
real stored receipts.

You never need to edit `aien-proof` or depend on its Rust internals. You need
only:

1. a built `aien-proof` binary on PATH (or `cargo run -p aien-proof --`),
2. a shared receipt store (the proof board dir, `AIEN_PROOF_DIR`),
3. an assertion file your test setup writes,
4. one `aien-proof receipt import` (alias: `aien-proof qualify`) call per run,
5. for Machine 1 runs, a `hold` lease around the run.

## 1. Assertion file contract

Your setup writes one JSON file per run. Prose is never parsed. The file
must match this shape exactly (unknown fields are refused):

```json
{
  "assertions": [
    {
      "id": "artifact_signature_valid",
      "expected": "valid",
      "observed": "valid",
      "result": "pass",
      "source": "scripts/qemu_seed0b.sh",
      "note": ""
    }
  ],
  "result": "PASS"
}
```

Rules:

- `id` is stable across runs (`secure_boot_unchanged`, `boot_order_unchanged`,
  `artifact_signature_valid`, `granted_subset_requested`,
  `w_x_never_simultaneous`, `unauthorized_write_denied`,
  `all_task_frames_reclaimed`, `store_previous_or_new_only`,
  `root_graph_valid`, `machine_returned_to_linux`).
- `result` per assertion is `pass` or `fail` (or boolean true/false).
- Top level `result` is optional. When absent it is derived (all pass gives
  PASS, else FAIL). A claimed PASS with any failing assertion is refused.
  A failing run can never be upgraded to PASS at import.
- At least one assertion is required for a PASS receipt.
- Assertion `source` is part of the canonical receipt identity. For hardware
  topology and ownership claims, use one stable provenance category as the
  source: `linux_observed`, `uefi_observed`, `native_aienos_observed`,
  `qemu_synthetic`, `derived`, or `unknown`. Put collector/tool identity and
  detail in `note` and bind its output as an artifact. A Linux-observed
  topology assertion cannot satisfy a gate that source-pins
  `native_aienos_observed`.

## 2. Import command contract

EXAMPLE (SEED-0B QEMU run):

```sh
aien-proof receipt import \
  --repo https://github.com/aien-dev/aienux \
  --commit 3fa4c8e2b1d04f7a9c6e5b8d2a1f4c7e9b0d3f21 \
  --gate seed-0b-qemu \
  --tier QEMU \
  --procedure "scripts/qemu_seed0b.sh" \
  --machine qemu-virt-aarch64 \
  --toolchain "$(rustc -vV)" \
  --assert /tmp/seed0b-assert.json \
  --artifact blake3:<hex-of-boot-image> \
  --output-file /tmp/seed0b-output.log \
  --dep <m3-receipt-id> \
  --dep <recovery-media-receipt-id>
```

Field notes:

- `--commit` is the exact 40 digit lowercase git SHA (64 digits accepted).
- `--tier` is one of `HOST_TEST QEMU QEMU_SECURITY MACHINE1_READ_ONLY
  MACHINE1_ATTENDED MACHINE1_MUTATING TEST_ONLY_TRUST PRODUCTION`.
  The environment label always equals the tier, so a QEMU run can never be
  filed as physical qualification.
- `--artifact` digests are `<algo>:<hex>` with algo `blake3`, `sha256`, or
  `sha512` (repeatable).
- `--output-file` is hashed to the receipt output digest. Prefer it over
  `--output-digest`.
- `--dep` lists prerequisite receipt IDs (repeatable). Missing deps stay
  missing: verification reports INCOMPLETE, never PASS.
- `--dirty` marks uncommitted source changes. Gates that require clean
  sources block dirty receipts.
- On success the command prints the receipt ID (the BLAKE3 of the canonical
  encoding) and stores `<store>/receipts/<id>.json`.

## 3. Machine 1 runs: the `hold` lease

Every `MACHINE1_*` receipt must bind an exclusive hold from the existing
coordination machinery. No second lock exists. Wrap the run:

```sh
HOLD=$(aien-proof hold --resource machine-1 --job seed-0b-machine1 -- \
  scripts/seed0b_machine1.sh 2>/dev/null | tail -n 1)
```

`hold` always runs (no fingerprint, stamp, or reuse), holds every
`--resource` key plus the holder record for the whole run, seals an `audit`
event into the Crumb ledger with the full output as payload, and stores
`holds/<key>.json`. The audit event hash printed on stdout is the hold ID.
Resource names are lowercase letters, digits, and dashes only. `run
--resource` takes the same keys, so test-board runs and holds exclude each
other on one machine.

Then import with:

```sh
  --tier MACHINE1_ATTENDED \
  --machine machine-1 \
  --procedure "scripts/seed0b_machine1.sh" \
  --declared-mutation REMOVABLE_MEDIA_ONLY \
  --lease-hold "$HOLD" \
  --lease-resource machine-1 \
  --output-file <hold-log-or-collector-output>
```

Rules:

- `--procedure` must equal the held command line exactly.
- `--machine` must equal the leased resource.
- `--output-file` (or digest) must equal the held run output digest.
- `--declared-mutation` is required for hardware tiers. Allowed classes:
  `NONE VOLATILE_ONLY REMOVABLE_MEDIA_ONLY ONE_TIME_BOOT_SELECTION
  BOUNDED_TEST_REGION_WRITE BOOT_CONFIGURATION_CHANGE TRUST_ROOT_CHANGE
  TPM_POLICY_CHANGE DESTRUCTIVE_STORAGE`.
- A read-only run (`MACHINE1_READ_ONLY`) cannot claim more than
  `REMOVABLE_MEDIA_ONLY`. Observed mutation beyond declared mutation fails.
- A Machine 1 receipt without a matching exclusive hold verifies as
  INCOMPLETE, never PASS.

## 4. EXAMPLE input contracts per gate input

All values below are placeholders. Replace them with measured values.

### 4.1 SEED-0B QEMU (owner: SEED-0B completion lane)

EXAMPLE assertion file:

```json
{
  "assertions": [
    {"id": "artifact_identity", "expected": "<image-digest>", "observed": "<image-digest>", "result": "pass", "source": "scripts/qemu_seed0b.sh"},
    {"id": "artifact_signature_valid", "expected": "valid", "observed": "valid", "result": "pass", "source": "scripts/qemu_seed0b.sh"},
    {"id": "granted_subset_requested", "expected": "subset", "observed": "subset", "result": "pass", "source": "admission-check"},
    {"id": "authorized_op_permitted", "expected": "permit", "observed": "permit", "result": "pass", "source": "admission-check"},
    {"id": "unauthorized_write_denied", "expected": "deny", "observed": "deny", "result": "pass", "source": "admission-check"},
    {"id": "all_task_frames_reclaimed", "expected": "0 live", "observed": "0 live", "result": "pass", "source": "cleanup-check"},
    {"id": "machine_returned_to_linux", "expected": "linux", "observed": "linux", "result": "pass", "source": "cleanup-check"}
  ]
}
```

Import with `--gate seed-0b-qemu --tier QEMU`, plus `--dep` on the M3
receipt and the recovery media receipt IDs. The hostile input matrix is a
separate import with `--gate seed-0b-hostile-matrix --tier QEMU`.

### 4.2 M0 rollback QEMU and Machine 1 are different facts (owner: M0 rollback lane)

EXAMPLE assertion file:

```json
{
  "assertions": [
    {"id": "rollback_harness_version_recorded", "expected": "version in procedure identity", "observed": "exact harness version", "result": "pass", "source": "rollback_harness"},
    {"id": "qemu_aavmf_environment_recorded", "expected": "QEMU and AAVMF identities", "observed": "exact QEMU/AAVMF versions and firmware digests", "result": "pass", "source": "qemu_environment_collector"},
    {"id": "rollback_normal_pass", "expected": "pass", "observed": "pass", "result": "pass", "source": "aavmf_serial_output"},
    {"id": "rollback_fault_pass", "expected": "pass", "observed": "pass", "result": "pass", "source": "aavmf_serial_output"},
    {"id": "rollback_timeout_pass", "expected": "pass", "observed": "pass", "result": "pass", "source": "aavmf_serial_output"},
    {"id": "rollback_rejected_pass", "expected": "pass", "observed": "pass", "result": "pass", "source": "aavmf_serial_output"},
    {"id": "rollback_absent_pass", "expected": "pass", "observed": "pass", "result": "pass", "source": "aavmf_serial_output"},
    {"id": "rollback_repeat_boot_pass", "expected": "pass", "observed": "pass", "result": "pass", "source": "aavmf_serial_output"},
    {"id": "bootnext_consumed", "expected": "unset in AAVMF NVRAM after one-time boot", "observed": "unset", "result": "pass", "source": "aavmf_variables"},
    {"id": "default_boot_unchanged", "expected": "default entry selected after candidate", "observed": "default entry selected", "result": "pass", "source": "aavmf_variables"},
    {"id": "m0_native_rollback_qemu_pass", "expected": "all required branches pass", "observed": "all required branches pass", "result": "pass", "source": "rollback_harness_summary"}
  ]
}
```

Import the QEMU run under `--gate m0-native-rollback-qemu --tier QEMU`.
Bind the exact `aienos` commit, harness version in `--procedure`, QEMU/AAVMF
identity in `--machine` and input artifact digests, and complete captured
output with `--output-file`. If the run also includes a pure state machine
mock, emit its observations with source `rollback_state_machine_mock`; firmware
variable observations must use source `aavmf_variables`. A mock result alone
cannot satisfy `bootnext_consumed`.

Do not import a Machine 1 rollback receipt until the owner trust chain permits
the attended run. That future receipt must prove Secure Boot stayed on,
permanent BootOrder stayed unchanged, BootNext was consumed, Linux/default boot
returned, root and ESP UUIDs stayed unchanged, expected mounts are healthy, and
the exact candidate digest and source commit were recorded. It must use
`MACHINE1_ATTENDED`, declare `ONE_TIME_BOOT_SELECTION`, and reference the
existing `aien-proof hold --resource machine-1 -- ...` event. QEMU rollback
evidence never satisfies this receipt.

### 4.3 System Store v1 crash consistency (owner: System Store lane)

EXAMPLE assertion file:

```json
{
  "assertions": [
    {"id": "store_previous_or_new_only", "expected": "no torn writes", "observed": "no torn writes", "result": "pass", "source": "crash-qualify.sh"},
    {"id": "store_graph_valid", "expected": "valid", "observed": "valid", "result": "pass", "source": "crash-qualify.sh"},
    {"id": "store_nospace_zero_writes", "expected": "zero writes", "observed": "zero writes", "result": "pass", "source": "crash-qualify.sh"},
    {"id": "store_io_error_not_fallback", "expected": "no silent fallback", "observed": "no silent fallback", "result": "pass", "source": "crash-qualify.sh"},
    {"id": "w_x_never_simultaneous", "expected": "never", "observed": "never", "result": "pass", "source": "crash-qualify.sh"}
  ]
}
```

Import with `--gate store-v1-crash --tier QEMU` (or the tier the lane
declares), with `--dep` on the ADR 0015 acceptance receipt once it exists.

The checked-in `P3_STORE_QEMU` manifest consumes separate Store format,
crash-consistency, and QEMU block-device receipts. Crash consistency must link
to the exact format qualification receipt. The block-device receipt must link
to both the exact format and crash-consistency receipts. Required structured
assertions are `store_previous_or_new_only`, `store_graph_valid`,
`store_nospace_zero_writes`, and `store_io_error_not_fallback`. This adapter
records observations; it does not define Store v1 semantics.

`P3_STORE_MACHINE1` additionally requires an exact bounded-region receipt,
exact-region provisioning authority, and an attended Machine 1 execution
receipt. The execution receipt must be `MACHINE1_MUTATING`, bind the existing
`machine-1` hold, depend on the target and provisioning receipts, and declare
and observe only `BOUNDED_TEST_REGION_WRITE`. A read-only Machine 1 receipt
cannot prove persistent writes.
Store v1 format tests, mount validation, and the QEMU block-device suite
follow the same shape with their own `--gate` kinds. Muse never defines
Store v1 wire or recovery behavior; the Store lane emits structured
observations and Muse verifies, chains, and evaluates them.

### 4.4 Admission Receipts (owner: SEED-0B admission lane)

Claude's Admission Receipt v0 stays where it is. It is the
subsystem-specific proof of artifact verification, admission, and execution.
It is not moved into `aien-proof`. An EvidenceReceipt references or digests
it without conflating the two layers:

- pass the admission receipt digest as an `--artifact` input digest, and/or
- record its location or identifier in `--external-ref`.

The EvidenceReceipt then proves the qualification procedure ran and its
required assertions and dependencies hold; the Admission Receipt proves what
the admission subsystem checked. Either layer can fail independently.

### 4.5 Provenance tags for characterization evidence

Characterization observations (for example GB10 topology facts) must carry
their provenance in the assertion `source` field using this vocabulary:

- `linux-observed`
- `uefi-observed`
- `native-aienos-observed`
- `qemu-synthetic`
- `derived`
- `unknown`

EXAMPLE source value: `gb10-topology:linux-observed`. Gate manifests pin the
required origin per assertion with `assertion_sources`, so Linux-observed
topology can never satisfy a native-AIENOS hardware-ownership claim and a
pure state-machine mock can never prove firmware variable behavior. When a
manifest is silent about origin, any recorded source is accepted, but the
source string is still part of the canonical receipt identity.

## 5. Verification commands (read only)

```sh
aien-proof receipt verify <receipt-file>
aien-proof verify-chain <receipt-file>
aien-proof gate status SEED-0B
aien-proof gate explain P3_DEVELOPMENT_QEMU_ENTRY
aien-proof gate explain P3_MACHINE1_WRITE_ENTRY
aien-proof gate explain STORE_V1_FORMAT_PR_MERGE
```

`gate explain` reports each checked-in manifest prerequisite. Missing ordinary
proof is `INCOMPLETE`. A known procedural blocker can declare `BLOCKED` and
name it explicitly. Example current physical state:

```text
P3_DEVELOPMENT_QEMU_ENTRY: INCOMPLETE
INCOMPLETE M0_NATIVE_ROLLBACK_QEMU ...
INCOMPLETE SEED0B_QEMU ...
INCOMPLETE ADR0015 ...
INCOMPLETE STRICT_MAIN ...

P3_MACHINE1_WRITE_ENTRY: BLOCKED
BLOCKED P3_DEVELOPMENT_QEMU_ENTRY ...
BLOCKED OWNER_SECURE_BOOT_TRUST_CHAIN MISSING: HARDWARE_QUALIFICATION_BLOCKED_BY_TRUST_CHAIN
BLOCKED M0_NATIVE_ROLLBACK_MACHINE1 MISSING: HARDWARE_QUALIFICATION_BLOCKED_BY_TRUST_CHAIN
INCOMPLETE BOUNDED_MACHINE1_STORE_REGION ...
```

`STORE_V1_FORMAT_PR_MERGE` is intentionally narrower. It requires ADR 0015,
pure Store v1 format/golden-vector assertions, and strict main. It can pass
without full SEED-0B completion. It does not authorize P3 QEMU qualification
or any Machine 1 mutation. `P3_DEVELOPMENT_QEMU_ENTRY` requires all four of
M0_NATIVE_ROLLBACK_QEMU, SEED0B_QEMU, ADR 0015, and strict main to pass.

The tier ordering is explicit: `TEST_ONLY_TRUST`, `HOST_TEST`, `QEMU`,
`QEMU_SECURITY`, `MACHINE1_READ_ONLY`, `MACHINE1_ATTENDED`,
`MACHINE1_MUTATING`, `PRODUCTION`. Test-only trust satisfies only itself.
Physical requirements reject all QEMU tiers. The read-only Machine 1 tier
cannot satisfy attended mutation requirements, and `PRODUCTION` cannot rely on
test-only trust ancestry.

Materialize the checked in example manifests with:

```sh
aien-proof gate examples --out <store>/gates
```

## 6. What the evidence lane will not do

- Change store, boot, artifact, admission, Secure Boot, TPM, trust root,
  native boot, or SEED-0B execution behavior.
- Infer completion from issue status or prose. If your run lacks a required
  input, the receipt schema or verification reports the gap instead.
- Upgrade failing runs, satisfy physical requirements with QEMU receipts,
  or close gates on BLOCKED runs.
- Move Claude's Admission Receipt format into `aien-proof`. An Admission
  Receipt proves artifact admission/execution inside its subsystem. An
  EvidenceReceipt proves a cross-repository qualification run and may bind an
  Admission Receipt by digest/reference.
