# Browser661 recovery qualification

Functional candidate: `2f6b5ffeda7dd1aa68f26ab548e732de0de82ee8`.
Base: `754c07b118f41ad899607b60329bd851b1ccb13e`.
Existing PR688 only. Preserve original authors/recovered commits and merged page-target readiness fix #654. No competing PR, merge, deployment or issue closure is authorized by these receipts.

## Final results

- Linux real Chrome153 owning-crate suite: 142 passed, 0 failed, 21 ignored. Explicit inherited-ignored-SIGCHLD test passed separately, giving 143 distinct executed tests. Other 20 ignored integration/model cases remain unqualified.
- All-target Clippy with warnings denied, fmt check and screenshot binary build exited 0.
- Real synthetic data-URL screenshot: correct title, 1280x757 image, exact red/green/blue/yellow interior samples. No live website, user-browser session, model or GPU inference used.
- Trace of browser tests recorded successful pidfd SIGKILL operations. Numeric kills in that trace include controlled test-fixture neighbor cleanup; do not interpret aggregate numeric-call absence as a test result. Production Linux teardown is pidfd-only, independently reviewed statically.
- Final independent review `L676-final-review.md`: no new Critical/Important finding, source-only, no independent runtime replay.

## Findings and actual failures retained

L678 found tag-check/numeric-PID signaling race. L682 identified inherited auto-reaping undermining a waitable-leader assumption. Final Linux cleanup pins candidate identities before tag/private-profile revalidation and never sends numeric PID/group signals. Capability absence refuses browser spawn.

Tag-only and raw-argv-only pidfd variants timed out when Chrome scrubbed environment tags and rewrote argv into process titles. Failed commits and timeout logs remain historical failures, including the erroneous generated passing claim on b451a0d. Qualified correction 6ba9225 additionally recognizes exact private-profile flags in supported Chrome executable/title representations.

L692 found unconditional final wait could exceed cleanup deadline. Owned synthetic regression failed at about 8 seconds before correction. Final teardown returns false on timeout and retains profile, without an unbounded final wait. Original RED is retained in history.

## Input reconciliation

`final/state.json` records actual qualified input hashes, not a clean Linux Git checkout. Its executable input exactly matches functional candidate: browser_mirror.rs SHA256 `39631076d3601b327d5c3f7fa47e6c8492ce194c5844c698e0c0025a21502a1c`.

Final inventory caught an older documentation copy on Linux: self-improvement skill hash `ebeaaed023f6f1066f2e3430935a8629c18464019464de3c3ebbfa035dc6001f`, archived as `qualified-skill-before-sync.md`. The committed/reviewed skill only adds same-UID identity-boundary and timeout/profile-retention explanations. It was synced after tests, fmt check passed and final hash read back as `b0f8ccd64af3701c59d5aa352cbe74944f2f11a12dad8f0b481989503e17a879`. No executable input changed. Do not attribute the earlier documentation hash to the final candidate.

## Limits

Pidfd/syscall and fixture evidence is not forced OS PID-reuse testing. Cleanup timeout can leave undetected/unresponsive tasks and intentionally retains their profile. Exact tag/profile is a fixture discriminator, not authentication against same-UID processes copying it. Non-Linux detached-helper cleanup is not qualified. CPU stub results are not native compose/model/hardware qualification. Signed exact-head desk checks and integrated acceptance remain required.

Workflow inspection of candidate and freshly fetched main found verification on PR/main pushes, release drafts on v* tags and manual dispatch, and manual signed release publication. A feature-branch push does not trigger these release paths. No workflow edits or release dispatch were performed.
