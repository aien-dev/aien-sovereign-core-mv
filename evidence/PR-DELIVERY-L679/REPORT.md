# PR delivery repair L679

## Verified defects and corrections

- Failed integration batches could not be retired: the ordinary failure guard also refused closing them. Closure now has a distinct admission that requires an actual FAIL, the exact current block identity and Renda ownership, honors holds/pause, retains branches and evidence, and reads back CLOSED before changing local batch state. Merge/rebase failure refusal remains unchanged.
- Generic and website publishers requested `baseRefOid` from gh pr list/view JSON even though the installed gh does not expose it. They now use bounded complete GraphQL queries with exact head/base and check-context decoding. All three GraphQL states OPEN/CLOSED/MERGED are requested; parent caught the author's missing MERGED enum using the live schema and a failing regression.
- Existing-PR handoff packets were not consumable as explicit successor work. Intake now retains a provenance-bound target PR/ref/expected old head. A completely reviewed, signed-qualified candidate can advance ONLY that PR branch with an exact lease; moved/closed/forked targets refuse and failed readback cannot become published. No new PR is created for such an entry.
- Reacknowledged 23 authentic current registered READY candidates using immutable input manifests and the installed local-only API. Old entries and failed verdicts remain. This fixes intake, not qualification or scheduling.

## Verification

Configured Mac baseline: 93 component tests pass. Initial attempted baseline lacked RENDA_CANDIDATE and used system LibreSSL without Ed25519; original error log retained. Correct environment uses Homebrew OpenSSL3 and explicit candidate.

Parent closure and successor behavioral regressions failed before fixes. Integrated component suite: 117 tests pass on Mac and Spark. Existing failed-close test now patches the close-only admission boundary, retains original guard98 failure and unchanged-state assertions. No test skip, test waiver, model replay or judge retry.

Source c6e5a784 received independent review L683, which found an old-PR-base promotion blocker. Behavioral RED reproduced the issue. Corrective source 7dfcf3d10f178147ff9e494cce050c54d3fa332b explicitly binds the observed old PR base separately from the qualified candidate's fresh main. The expanded 119-test suite passed on Mac and Spark; independent corrective review L688 reported no blocking finding and independently passed all 119 tests. Immutable release L603-fbfa8c78dc482892 installed at 2026-10-23T10:51:11Z after its installer replayed every packaged suite. File hashes match the reviewed Mac and committed Spark sources. Backups and deployment receipt are retained. Renda1 was reloaded only while single-threaded, sleeping and childless; its replacement started at 10:52:11Z. Renda2 reload and actual completed-pass/progress verification are pending. Normal full-workspace/signed PR gates are not established by these component tests.

## Concrete remaining boundaries

- Omega0dbed9c1: M19 channel allocation RM_ALLOC0x51; R11 living hold refused; R12 graphics-seat heartbeat missing; R13 incumbent cost not confirmed. Existing owner fixes must be integrated and exactly qualified. FAILs are retained.
- Architecture294: six code checks pass but planner NOT_RUN/UNMAPPED_PATH68. Signed remote bundle absent. Neither top-level PASS nor rerunning initial missing-bundle hosted check resolves the unmapped planner.
- PR577/609/626 successors are now acknowledged by the installed local-only intake, each bound to its existing PR number/ref/expected head and observed old base. Exact manifest e489980d5b477d1bd79b0b157fc8458f85693f8dc8e7139476c96a3a6071767f. No competing PR or direct chat publication; real consumer review/qualification/promotion remains required.
- Restorable enrolled existing Renda lanes: synapse,aegis-runtime,spark-rsi. aienos legacy identity redirects to aien-dev/aienux; do not silently substitute kernel identity into old queue. CUDApp/aien-rig/aien-train have non-Spark policies not implemented by generic signed-Spark path. No new worker or fake qualification.
- Website lane remains deliberately publication-only, independent merge/deployment review required. Personal website explicit release holds are not lifted by this repair.
