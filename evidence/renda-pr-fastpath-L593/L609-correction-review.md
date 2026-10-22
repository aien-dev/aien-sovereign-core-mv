# L609 corrective-delta independent review

**Verdict: APPROVE, bounded to cache-only modification/corruption. No remaining P1/P2 counterexample found.** Reviewed `3aae7e9c..303310729dd7ef6242dc1ec3cb63019beb811325` at the requested Spark worktree, against `tools/aien-local-ci/renda/L609-CACHE-INTEGRITY.md` and the prior independent review at `~/handoffs/pr-failure-triage-20261022/L609-review.md`.

## Findings

- HMAC-SHA256 covers the serialized cache payload, including per-chunk `decision`, `note`, exact chunk text/offsets/digests, and cache identity. `_valid_cache` requires valid authentication before reusing any result; cache edits invalidate it and a zero-budget pass stays incomplete rather than granting approval.
- The authentication root is outside queue/cache storage at `~/.config/renda/review-cache.key`. Initialization is flock-serialized; creation uses `O_NOFOLLOW`, mode 0600; the key must be regular, service-owned, single-link, private, and 32 bytes; its containing directory must be service-owned and not group/world writable. Tests patch `_key_path` to private temporary fixtures. The threat boundary is correctly limited: arbitrary execution as the service UID could replace the controller or read configuration and is not solved by HMAC.
- Verified boundaries: generic `Queue.review` requires an authenticated complete review and preserves an explicit `False`; generic READY delivery revalidates exact source/cache for both small and large diffs; website READY publication revalidates the authenticated cache. Tests include forged-boolean cache refusal at small generic publication and website publication, accepted authentic large-diff publication, and retained-refusal dominance. No positive path relies on a forged boolean.
- Prompt change clarifies artificial chunk segmentation without changing parsing or verdict semantics. The captured original engine `false` for an artificial-boundary truncation concern remains false; strict explicit-boolean parsing and terminal false handling remain intact. This review did not duplicate the separately assigned actual-engine conformance probe.

## Verification

On Spark, ran `RENDA_CANDIDATE=$PWD/renda-harness python3 -m unittest discover -v -p "test_*.py"` from `tools/aien-local-ci/renda`: **98 tests, all passed**. Also ran `git diff --check` (clean). The tests/probes are synthetic; no model, GPU, production key/cache, service, judge, or PR state was touched. No source files were modified.

**Scope limit:** approval covers the specified cache-only attacker and reviewed local code paths, not arbitrary service-UID code execution or model-review quality. No deployment or external publication is authorized by this review.