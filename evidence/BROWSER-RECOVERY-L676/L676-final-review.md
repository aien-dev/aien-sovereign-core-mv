# L676 independent final corrective review — browser661

Reviewed `/Users/drakestapleton/.hermes/cache/scratch/loop24-browser661` at exact candidate `2f6b5ffeda7dd1aa68f26ab548e732de0de82ee8`, against base `754c07b118f41ad899607b60329bd851b1ccb13e`. Read the candidate diff, Linux process identity/teardown and Drop paths, tests, self-improvement documentation, and prior L678/L682/L692 reports. Checkout HEAD matched the requested candidate and was clean. Static review only: no builds/tests, browser launches, process signals, edits to source, or publication. Parent-run Chrome results were not replayed.

## Findings

**No new Critical or Important finding identified.** The L692 bounded-cleanup defect is corrected: `reap()` now returns `false` at `REAP_TIMEOUT` without a final blocking `Child::wait()`. Drop removes the profile only when reap confirms the direct child is reaped (including `ECHILD`) and the most recent ownership scan found no tagged/profile-matching process; on timeout it emits a diagnostic and retains the profile. The regression test exercises an intentionally undetected live child and checks that drop returns within the bound plus tolerance and retains the profile. I did not run that test.

The L682 inherited-SIGCHLD/auto-reap concern is addressed for Linux: cleanup no longer signals a numeric process group, and `ECHILD` marks the direct child as reaped. Candidate PIDs are pinned before identity revalidation and Linux signals use pidfds. I found no Linux numeric PID/PGID signal in teardown. The previous L678 PID-reuse signal race therefore remains resolved in this candidate.

## Remaining scope and limits

- The timeout is a bounded retry policy, not guaranteed successful termination: if ownership cannot be established or a pidfd signal fails, browser processes may remain after timeout. The implementation intentionally returns without blocking and retains the profile. This is fail-closed for profile deletion, not proof that all descendants terminate.
- The “no owned task remains” decision depends on `/proc` discovery and the supported Chrome tag/profile identity representations; static inspection cannot prove every Chrome helper retains one of them. Same-UID processes deliberately copying the exact tag/profile remain explicitly outside the authentication boundary.
- Non-Linux cleanup still uses a numeric process-group signal and detached-helper cleanup is documented as unqualified. I found no new Linux issue from that platform-specific path.
- No runtime, browser, race-injection, or test results are claimed here. In particular, this review does not independently verify Chrome 153 screenshot behavior, pidfd availability on the target, or the cleanup test's timing.
