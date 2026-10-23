# L683 independent review: Renda delivery repair

Reviewed frozen source `c6e5a784ea9f8f736ecda5ae4ac75796934c97c7` against base `b1514c77f632bf3c036b20ba06a9b96213c43f00` in Spark worktree `~/workspace/hermes-desk/renda-delivery-L679`. Source remained untouched. Inspected the full 13-file `tools/aien-local-ci/renda` delta (572 insertions, 56 deletions) and ran the complete Python component suite from the isolated Mac candidate.

## Finding

**P1 – successor promotion incorrectly requires an existing PR's observed base SHA to equal the candidate's fresh main SHA.** `tools/aien-local-ci/renda/successor.py:363-370` rejects unless the existing PR's `base.sha == base`, where `base` is the successor's READY base. But `target` (`successor.py:336-341`) records only number/ref/expected head, not the existing PR's expected old base. Thus the code cannot distinguish an unchanged, explicitly approved older PR base from a moved/tampered target base. It blocks legitimate advancement even when the existing PR identity and old base are expected, while source qualification and `refs_match()` independently establish the candidate's exact current `main` base.

The authentic validation supplied for this review found PRs **#609 and #626** have GraphQL/REST `baseRefOid=12e71aad80fda23a428317d11b0dac02d1c777f1`, while live `refs/heads/main` and their qualified successor base are `754c07b118f41ad899607b60329bd851b1ccb13e`. PR #577's base matches current main. This candidate therefore safely holds #609/#626 today but does not deliver their intended update. Do not weaken the check to ignore base identity. Extend explicit READY successor target metadata with a full-SHA `expected_base` (the exact old PR base), require observed PR base ref/repository to remain `main`/same repository and observed base SHA to equal that lease, while continuing to require candidate `base` to match fresh `refs/heads/main` and the qualified source receipt. Recheck the old PR identity/base immediately at the write boundary; keep the existing expected-head branch lease. Never treat `expected_base` as candidate qualification base.

**Safe targeted negative reproduction:** in an isolated Python invocation against the frozen candidate, mocked target PR #609 as open, same-repo, correct ref/head, base `main`, but old base SHA `1*40`, while candidate base/current refs were `7*40` and all ownership/qualification/ref callbacks returned true. `promote_successor` returned `(False, 'k successor held: target changed, draft, closed, forked or malformed')`; one remote read, zero push writes. This confirms the unconditional base equality is the blocker; the fixture does not claim to validate live GitHub behavior.

## Verification

Full component suite command (Mac candidate, `RENDA_CANDIDATE` set to its harness, requested OpenSSL PATH and scratch TMPDIR):
`python3 -m unittest discover -s /Users/drakestapleton/.hermes/cache/scratch/pr-repair-L679/candidate -p 'test_*.py' -v`

Result: **117 tests passed**, 19.221 seconds, `OK`. Suite includes delivery, block-retirement recovery, successor, publication transport, installation, admission, queue, review, and site-policy tests.

## Deployment routing note

Per the supplied live validation, PR #577 is eligible to be routed after installation checks; #609/#626 need the explicit expected-old-base repair and a new qualification/review before their updates. This review made no deployment, PR, judge, model, queue, or source writes.
