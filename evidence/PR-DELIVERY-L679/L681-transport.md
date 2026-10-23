# L681 — GraphQL publication transport repair

## Outcome

Replaced all four publication `gh pr list/view --json ...baseRefOid` call sites (generic and website lookup/readback) with a shared GraphQL transport. It validates fresh exact head/base fields, decodes the exact-head commit/check context, queries the requested `headRefName` across OPEN+CLOSED (including merged) states, follows complete pagination up to a fixed 10-page bound, and refuses GraphQL errors, nonzero transport, malformed/truncated pages, repeated cursors, malformed rows, or duplicate PR numbers. Publication callers return held/pending without recording success when lookup/readback fails.

## Changed files

- `transport/publication_transport.py` — new GraphQL transport.
- `transport/test_publication_transport.py` — new transport regressions.
- `transport/delivery.py` — generic publication lookup/readback and website publication lookup/readback.
- `transport/renda-harness` — website publication lookup/readback, helper import, and policy-hash inclusion.
- `transport/test_delivery.py` — GraphQL fixtures and no-create-on-lookup-failure regression.
- `transport/test_site_policy.py` — GraphQL fixtures for website publication and readback.
- `transport/install.py`, `transport/renda-checked` — package and startup-test wiring.

Patch: `/Users/drakestapleton/.hermes/cache/scratch/pr-repair-L679/L681-transport.patch`

## Verification

- RED: `test_publication_transport` first failed to import the not-yet-implemented helper (`ModuleNotFoundError`).
- GREEN: all 7 package test scripts passed: reliability 7, local queue 26, review chunks 10, review integration 7, site policy 18, publication transport 6, delivery 26 (100 total).
- `py_compile` passed for modified/new Python files.
- The host default `/usr/bin/openssl` is LibreSSL 3.3.6 and lacks Ed25519; reran the complete suite with Homebrew OpenSSL 3 first on PATH, then all tests passed.
- Readonly Spark reproduction: installed `gh pr list ... --json ...baseRefOid` returned `Unknown JSON field: "baseRefOid"`.
- Real readonly lookup through the new helper over Spark's guarded `gh api graphql`: branch `desk/block-aien-sovereign-core-20261022T111633Z` returned PR #703, CLOSED, head `02ff6bcaa9859baf1269f96257a9a65dfc80b0cd`, base `main` at `1e6ae85c977b628fa26a8c84c8b16194e78e67e2`. Independent GraphQL readback of #703 returned the same exact pins.

No live install, PR write, merge, model call, or judge run was performed. The installed snapshot remains unchanged.