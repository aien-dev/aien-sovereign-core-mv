# aien-closure (`verify-closure`)

Verified Crumb, stage 5. This is a CI bridge: it makes today's Rust and C
code obey the rule "every internal dependency has verified evidence" before
Omega's compiler owns dependency resolution. When Omega's `import` resolves
dependencies through the verified library, this tool is retired or reduced to
a cross-check.

It reuses aien-proof for the receipt format. Receipt identity is
`aien_proof::evidence::receipt_id` (BLAKE3 over the canonical encoding), and
chains are checked with `aien_proof::chain::verify_chain`. Nothing here
re-implements the hash.

## What it checks

`verify-closure --root <repo> [--store <receipt dir>] [--write-lock]`

1. Rebuild the internal graph with `cargo metadata --offline --locked
   --no-deps`. Edges are path dependencies between workspace members (normal
   and build kinds; dev dependencies are not part of the built artifact).
2. Read `closure.toml` from each crate directory that has one.
3. Every edge must be declared in the manifest and pinned to the receipt id of
   the dependency, which must itself have a manifest with that receipt.
4. Each component receipt must exist in `<store>/receipts/<id>.json`, hash to
   its id, and the whole receipt chain must verify as PASS.
5. The receipt must have been taken at the source digest the manifest records,
   and the current source must still have that digest.
6. No cycles. No dirty or TEST_ONLY_TRUST receipts anywhere in the chain.
7. `closure.lock` must equal the closure rebuilt from the manifests.

If no manifest exists at all, the run fails (UNVERIFIED_DEPENDENCY). The store
defaults to `$AIEN_PROOF_DIR`, else `~/.local/state/aien-proof`.

## The nine codes

Output is one line per finding, `FINDING <CODE> component=<name> detail=<text>`,
then `SUMMARY status=<PASS|FAIL> findings=<n> components=<n>`. Exit code is 0
on PASS, 1 on an internal error, 2 on a usage error, and otherwise the code of
the first finding (sorted by the order below, then component).

| Exit | Code | Raised when |
| --- | --- | --- |
| 10 | UNVERIFIED_DEPENDENCY | a dependency has no manifest, a manifest is malformed, a declared dep is not a real edge, `closure.lock` pins a dependency the component does not have, the chain does not verify as PASS, the receipt tier is below the profile, or no manifest exists |
| 11 | MISSING_RECEIPT | a receipt id (component or any chain member) is not in the store |
| 12 | RECEIPT_HASH_MISMATCH | a receipt file does not hash to its id, or is not a valid receipt |
| 13 | DEPENDENCY_NOT_PINNED | a dep has no receipt pin, the pin differs from the dep's receipt, the receipt does not depend on the pinned receipt, or `closure.lock` is missing or has a missing or wrong line |
| 14 | DEPENDENCY_CYCLE | the internal graph or the receipt graph has a cycle |
| 15 | STALE_RECEIPT | current source digest differs from the manifest, or the receipt does not record the manifest's source digest |
| 16 | UNDECLARED_IMPORT | the graph has an edge the manifest does not declare (only this direction; an extra declaration is UNVERIFIED_DEPENDENCY) |
| 17 | TAINTED_ARTIFACT | a receipt in the chain was recorded from a dirty tree or is TEST_ONLY_TRUST |
| 18 | UNKNOWN_VERIFIER_PROFILE | the manifest names a profile this verifier does not know |

Profiles: `host-v1` needs tier HOST_TEST or better, `qemu-v1` needs QEMU or
better, `production-v1` needs PRODUCTION. Tier comparison is
`aien_proof::chain::tier_satisfies`, so a QEMU receipt never satisfies a
physical requirement.

## How a component gets a manifest

Put a `closure.toml` in the crate directory. Strict, line based, no comments,
no blank lines, no tabs, one trailing newline:

```text
closure-manifest v1
component aien-scheduler
profile host-v1
receipt <64 hex: receipt id that qualifies this component>
source blake3:<64 hex: source digest the receipt was taken at>
dep aien-kv-cache <64 hex: receipt id of aien-kv-cache>
dep aien-platform <64 hex>
```

Steps:

1. Qualify every dependency first (leaves up), each with its own manifest.
2. Compute the source digest of the crate. It is BLAKE3 over all regular files
   in the crate directory in sorted path order, excluding `target`, `.git`,
   `closure.toml` and `closure.lock` (see `src/digest.rs` for the exact byte
   layout). Symlinks in a crate directory are an error.
3. Record an aien-proof receipt whose `input_artifacts` contains
   `blake3:<source digest>`, whose `dependencies` contain the receipt ids of
   the dependencies, and whose result is PASS from a clean tree.
4. Write the manifest, then run `verify-closure --root . --write-lock` to
   produce `closure.lock`, and commit both.

Any source change makes the receipt stale on purpose. Re-qualify, then update
the manifest.

## C components (for Omega, not wired to CI yet)

`graph::scan_c_includes(src_dir, include_root)` builds a file level graph from
quoted `#include "..."` lines. An include resolves against the including
file's directory, then against `include_root` (the `-I` directory, `src` for
Omega). Unresolved quoted includes are returned, not dropped. Limitation: block
comments and `#if 0` regions are not understood, so a commented out include
still counts, which errs on the strict side. It is unit tested on a fixture
directory only.

## Tests

`cargo test -p aien-closure` seeds a fixture workspace (`tests/fixtures/seeded`)
into a temp directory with receipts built through aien-proof's own API, then
checks one PASS case and at least one failing case per code, each asserting the
exact finding and the binary's exit code. The committed seeded fixture (with
store) is what the SC-VERIFY-CLOSURE gate (`tests/gates/sc-verify-closure.gate`,
step `verify-closure` in `tests/gates/run.sh`) runs against on the Spark judge. To rewrite
it: `cargo test -p aien-closure -- --ignored regenerate_committed_fixture`.

## One meaning per declaration code

The meanings follow `aien-protocols` `specs/verified-crumb/SPEC.md` section 6.2
(contract 0.2.0), implemented in `src/declare.rs`:

| Situation | Code |
| --- | --- |
| an edge (or import) with no receipt pin, or a pin that is not the dependency's receipt | DEPENDENCY_NOT_PINNED |
| a `closure.lock` line for a dependency the component does not have | UNVERIFIED_DEPENDENCY |
| a declared dep that is not an actual edge | UNVERIFIED_DEPENDENCY |
| an actual edge the manifest does not declare | UNDECLARED_IMPORT |
| a VC's stored source imports a program not in `dependencies[]` (SPEC step 8) | UNDECLARED_IMPORT |
| a `dependencies[]` entry the VC's stored source does not import (SPEC step 8) | UNVERIFIED_DEPENDENCY |

This crate reads no VCs, so the last two rows are covered by a unit test of the
shared set comparison (`declare::compare`), not by the binary.
`tests/declaration_codes.rs` has one test per row plus a negative twin that
fails if the row's code is swapped for a neighbouring code.

## Receipt cross-check fixtures

`tests/fixtures/receipts/` holds receipts minted with aien-proof for cases the
Omega C reader's three fixtures do not cover. See the README there.
