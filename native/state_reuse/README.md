# Complete-state reuse selection

This additive native Omega component advances Core #718. It chooses between
fresh prefill, a compatible RAM snapshot and a compatible disk snapshot by
estimated total completion time. It does not implement snapshots, allocate
memory, authenticate files, restore state or execute a model. Those remain
with Core #713, #484, #485 and the native serving owner. Nothing here is wired
into production. Hardware performance and full-model correctness are NOT_RUN.

Inspected baseline: Core main `d691a4dc`,
`crates/aien-serve-http/src/kv_persist.rs`, `PersistState` storage path.
Its `fetch(prompt, in_ram)` skips disk restoration when the disk prefix is no
longer than the RAM prefix. This component supplies a cost decision for the
future native complete-state owner; it does not change that existing path.

## Calling contract

Concatenate `policy.om` before a seeded `.om` caller. All values are unsigned
64-bit integers. Time estimates are microseconds. Calibration timestamps and
the current timestamp must use one monotonic clock in microseconds. There is
one caller-owned policy instance; concurrent calls need separate instances or
the existing scheduler's serialization. No locks or disk waits occur here.

`rcontrol` has these ten fields:

| Index | Meaning |
| --- | --- |
| 0 | Candidate count, 1 through 16; row zero is always fresh work |
| 1 | Previously selected stable snapshot ID; zero means fresh work |
| 2 | Current monotonic timestamp |
| 3 | Maximum calibration age |
| 4 | Minimum observed samples, 1 through 1,000,000 |
| 5 | Total allowed memory bytes, at most 2^50 |
| 6 | Already allocated backing and shared references, counted once |
| 7 | Absolute switching margin in microseconds, at most 10^12 |
| 8 | Relative switching margin in millionths, at most 1,000,000 |
| 9 | Current request token boundary, at most 131,072 |

Each candidate occupies 22 words in `rrows`:

| Index | Meaning |
| --- | --- |
| 0 | Enabled: exactly one admits the row; other values are skipped |
| 1 | Source kind: zero fresh, one RAM, two disk |
| 2 | Stable snapshot incarnation ID, unique among enabled rows; fresh is zero |
| 3 | Accepted, consumed prefix length; fresh is zero |
| 4 | Emitted but unconsumed token count, zero or one; fresh is zero |
| 5 | Complete-state qualification mask, exactly 31 for reuse |
| 6 | Observed calibration sample count, bounded at 1,000,000 |
| 7 | Last calibration timestamp |
| 8 through 14 | Lookup, read, integrity/decryption/validation, transfer, reconstruction, suffix prefill, and remaining generation time |
| 15 through 19 | Additional backing, metadata, staging, scratch and commit/output reservation bytes |
| 20 | Caller has verified payload integrity, exactly one for reuse |
| 21 | Caller still holds the snapshot and its generation, exactly one for reuse |

The five state bits mean attention, recurrent state, convolution tails,
position/pending-token/drafter state, and sampler/constraint continuation state
are all either present or explicitly qualified as reconstructible under the
same continuation contract. They are assertions from the qualified owner,
not a substitute for inspection or tests of those states. A model without one
of these state families still needs an explicit not-applicable qualification.

`rexpect` contains eight nonzero 256-bit identities, four words per identity:
model, tokenizer, template, precision/layout, backend, namespace, continuation
contract and cost-workload profile. The workload profile binds exact context
shape, output-work estimate and calibration environment. `rident` contains
the same 32-word tuple for each candidate. The caller validates the actual
prefix against request tokens separately, authenticates the source and binds
the tuple to it. This module compares all words; it does not create a digest
or interpret an untrusted ID as a capability.

The owner must include all retained shared backing in control field six, and
all incremental allocations and temporary reservations in each row. Count a
shared allocation once, and never subtract merely freed slots from allocated
backing. The selected reservation must succeed before copying or publishing
tokens. A failed reservation requires a fresh decision with updated accounting.

## Decisions and errors

`rselect` returns zero on a decision. `rresult` contains selected row, predicted
total time, predicted saving against fresh work, prediction-valid flag and
stale-or-missing-baseline fallback flag. Output is cleared before validation.
`rreason` gives each row's admission status; unexamined rows stay 77.

Status 65 means malformed configuration/row, 69 incompatible or missing
identity, 74 missing/stale/future calibration, 75 insufficient memory, 76 an
unqualified state boundary, and 77 disabled or unexamined. Invalid reuse rows
are skipped. An invalid baseline refuses the request, except that missing or
stale calibration selects fresh work with prediction-valid zero. In that
fallback, no predicted time or saving is asserted. The caller can run its
ordinary path only after its own admission checks.

Every phase is bounded at 10^12 microseconds; sums and margin arithmetic fit
64 bits. A zero total estimate is unknown. The cheapest admitted candidate
wins only when its saving strictly exceeds the absolute plus relative margin.
The reference is the admitted current snapshot if cheaper than fresh work,
otherwise fresh work. Exact ties retain that reference; other ties prefer RAM
to disk, then the smaller stable ID. Row ordering therefore does not determine
which of equally priced reusable snapshots wins. An invalid current snapshot
cannot be retained by the switching margin.

`rcalib(row, timestamp)` updates an already admitted profile from the seven
actual elapsed phases in `robs`. All fields are validated before any update.
Each estimate becomes `ceil((7 * old + observed) / 8)`; sample counts saturate.
This bounded smoothing is a policy choice, not a statistical confidence claim.
An expired or changed profile requires a new initial calibration from the
existing measurement owner. Do not feed predicted or fabricated measurements
back as observations. Record actual completion cost and the selected source
through the existing measurement owner, without exporting private prompts.

## Reproduction

The compiler is the existing independently reconstructed source-built Omega
compiler from omega-bridge `0fc7d8662cb799e7c206b17c18ea4c3ec2b8fea9`, built
with its existing `tools/bootstrap.sh --offline`. Required compiler SHA-256:
`191b776ef98821597c3e30655da6d680a9b66edbc9ceaca3c39c61ea6df098e9`.

Set `OMC` to that verified executable and `REUSE_OUT` to a private output
directory. Existing shell commands below are development orchestration only.
All new implementation and test logic is handwritten seeded `.om`.

```sh
cat native/state_reuse/policy.om native/state_reuse/tests.om > "$REUSE_OUT/tests.om"
quiet-build timeout 30 "$OMC" < "$REUSE_OUT/tests.om" > "$REUSE_OUT/tests.elf"
# Require successful compilation before executing the output.
chmod 700 "$REUSE_OUT/tests.elf"
quiet-build timeout 20 "$REUSE_OUT/tests.elf"
```

The test prints the check count and failure count, returning one on failure.
Its 8,000 synthetic cost triples independently enumerate the expected cheapest
fresh/RAM/disk choice. Other controls cover all identity words, all timing and
memory phases, stale/future profiles, state boundaries, hysteresis and atomic
calibration rejection. Synthetic timings establish policy behavior only.

Remaining #718 gates include actual snapshot retention/allocation, integration
with #713 and native serving, observed profiles on the qualified 27B artifact,
corrupt-file and cancellation recovery, storage overlap, actual RAM/page-cache/
NVMe comparisons, independent final-source review and normal repository gates.
No existing runtime or storage implementation is retired by this component.
