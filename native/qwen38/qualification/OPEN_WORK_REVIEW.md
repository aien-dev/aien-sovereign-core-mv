# Open-work review, 2026-10-23

Inventory: 225 open issues and 120 open PRs across Sovereign Core, Omega and
CUDApp. The complete title/link/update-time snapshot is
[`open-work.tsv`](../../../evidence/qwen38-710-native-v2/open-work.tsv).
All titles were triaged; this is not a source audit or approval of all 120 PRs.
Detailed inspection focused on the Qwen3.8 dependency path, effective consumer
pins and the failures encountered by #714. This is a delivery review under
Core #679, not a replacement architecture roadmap.

| Repository | Issues | PRs |
|---|---:|---:|
| Sovereign Core | 65 | 56 |
| Omega | 114 | 63 |
| CUDApp | 46 | 1 |

## Changes made during this review

- Restored all 81 original checkpoint files at the pinned Hugging Face
  revision, 30,890,049,597 bytes. Download-time hashes and a second complete
  read match the historical manifest. The missing-checkpoint blocker is
  resolved; model loading, tokenization and quality are still unqualified.
- Extended handwritten `.om` receipt arithmetic to emit median/p95 for all
  recorded phases, memory, output counts and useful-output rates, including
  queue-free service cost. Reject zero service before rate division.
- Verified 25 structured cases, five malformed inputs, exact shuffled-phase
  summaries, two compiled statistical mutants and old-versus-new zero-service
  behavior. These tests establish CPU arithmetic only.
- Corrected the ownership record: L419 is an unassigned migration slot, not
  proof of an active runtime implementer. Existing L408/L605 boundaries remain
  authoritative; no competing compiler or device work was started.

## Delivery order and exact gates

1. **Finish the reproducible baseline inputs, Core #710.** Checkpoint recovery
   is complete. Exact token arrays, held-out code/prose corpus, reference
   outputs and full native serving provenance remain open. The four smoke
   tasks do not qualify model quality. Keep numeric thresholds frozen while
   collecting reference data. #714 remains a draft.
2. **Close the native execution boundary through existing owners.**
   [Omega #636](https://github.com/aien-dev/omega-mv/issues/636) owns host
   facilities and [#638](https://github.com/aien-dev/omega-mv/issues/638) owns
   the seeded CPU-to-GPU path. L408 owns compiler facilities; L605 owns the
   device continuation. Reviewed open/ioctl operations do not establish
   allocation, upload, dispatch, completion and readback. L739's stopped
   native serving work is reusable source, not a running native forward pass.
   L720's immutable numeric work is reusable after its review gates; do not
   duplicate it or assume compiler floats are the only possible reference path.
3. **Keep correctness ahead of optimization.**
   [Omega #562](https://github.com/aien-dev/omega-mv/issues/562) records scratch
   lifetime, finite-value, tolerance, layer-order and failed-token recovery
   findings in the old backend. Each needs a checked disposition in its `.om`
   replacement. Open C implementation PRs #547 through #553 and #559 are
   references, not the required final implementation. Preserve
   [CUDApp #33](https://github.com/aien-dev/CUDApp/issues/33)'s separate
   qualification contract.
4. **Implement packed W4 and state-correct endpoint MTP, then compare.**
   Core #711 needs physical packed reads; the inspected FP8 re-expansion study
   is not that path. #713 must keep KV/GDN/convolution/drafter state at one
   consumed prefix before #712 can qualify rejection, correction and reuse.
   Benchmark all four arms only after those gates and exclusive admission.
   Tree speculation remains an unselected experiment.
5. **Delete superseded implementations only after replacement qualification.**
   The grouped retirement map in READINESS.md remains pending. No final
   file-by-file replacement/deletion audit or qualified serving artifact exists.

## Existing review and test-delivery blockers

| Existing work | Observed state | Required disposition |
|---|---|---|
| [Core #714](https://github.com/aien-dev/aien-sovereign-core-mv/pull/714) | Draft; previous head's hosted check failed because the root generated crumb is stale; Spark status failed | Keep raw failure. Existing advisory-crumb migration #698 owns the policy change. Native CPU checks do not waive repository gates |
| [Core #688](https://github.com/aien-dev/aien-sovereign-core-mv/pull/688) | GitHub reports conflicts and failed verification; its screenshot-test scope matches the local preflight failure | Existing owner must reconcile the branch and replay screenshot/process-cleanup checks. No independent source approval is claimed here |
| [Core #621](https://github.com/aien-dev/aien-sovereign-core-mv/pull/621) | Conflicts and failed Spark status; rustls-only proposal predates the complete current audit | Reconcile existing security work and re-audit the final lock. The current eight findings also include ctutils, rustls-webpki and both root-certificate packages |
| [Core #698](https://github.com/aien-dev/aien-sovereign-core-mv/pull/698) | Existing advisory-crumb proposal explicitly supersedes #605/#674 only when merged | Do not regenerate crumbs here or close predecessors prematurely. Its recorded workflow-push and coordinated deployment gates remain with the owner |
| Core #690/#691/#693/#695 | Existing single-judge, mirror-lock, memory-admission and build-before-GPU-lock work | Coordinate through the existing desk; no second scheduler or lock protocol. A lock file alone is not exclusive measurement evidence |

The broader inventory also contains persistent-KV, ALLEN, expert streaming,
language portability, graphics/vendor expansion and research experiments.
Those are not implicitly prerequisites for the scoped 27B delivery. CUDApp
#48 through #51's zero/rotation/factorization/delta studies remain separate
experiments. Similar titles are candidates for owner reconciliation, not
evidence that an issue is redundant or safe to close.

## Evidence and limits

Current receipt source and reproduction commands are in CONTRACT.md and
`evidence/qwen38-710-native-v2/README.md`. Previous full repository preflight
and dependency audit failures remain preserved in v1 evidence. Hosted failure
details are retained in v2. The live endpoint inspected is a different 35B
model; it was neither restarted nor used as the 27B baseline.

No issue was closed, no unqualified PR merged, and no hardware gate marked
passing. Achieved 27B speed and quality are still **NOT_MEASURED**.
