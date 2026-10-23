# L682 independent browser661 cleanup identity review

Reviewed checkout `/Users/drakestapleton/.hermes/cache/scratch/loop24-browser661`, HEAD `7b5a74a2267235f2ed2a276ef49ccddd718b1aec`, against `754c07b118f41ad899607b60329bd851b1ccb13e`. Current `browser_mirror.rs` SHA-256 is `ca1e8e4298cb4c39cdc9c5ef9dc69dd02e95a371ffebab2d2c9951e14c5c18d8`, matching the file at HEAD. Reviewed the complete diff, cleanup lifecycle, tests, and prior L678 report. No source edits, builds, browser launches, process signals, service activity, or publication; no tests independently run.

## Important finding — group identity safety assumes the direct child remains waitable

`ChromeProcess::reap` signals `kill(-pgid, SIGKILL)` whenever `main_reaped` is false, then marks it true only when `Child::try_wait()` returns `Ok(Some(_))`. A `try_wait()` error leaves the flag false, and `child.wait().ok()` later discards any `ECHILD` result. Therefore, if the direct Chrome child is automatically reaped or reaped by a competing wait handler before `reap` signals the group, the direct-child PID no longer reserves the numeric PGID. In the timing where the original process group has also disappeared, that PGID can be reused; the one-shot group signal can then target a stranger's group. The pidfd-tag sweep does not protect this negative-PGID signal.

I found no `waitpid`, SIGCHLD handler, or subreaper setup in the repository's CLI code, so an in-process competing reaper is not part of the inspected normal Rust lifecycle. However, `SIGCHLD` ignored/`SA_NOCLDWAIT` dispositions can be inherited across `exec` and auto-reap children; the implementation does not establish or check the waitability assumption. This is a conditional identity-safety gap, not a claim that the normal documented invocation triggers it. A reproducer would run the CLI with SIGCHLD ignored (or an application-installed reaper), arrange for Chrome and its process group to exit before `reap`, and permit PGID reuse before drop cleanup. I did not run such a reproduction under the no-signals/no-build constraint.

## Resolved L678 finding

The previous tagged-PID read-then-numeric-kill race is closed in this variant: `TaggedProcess::open` pins a pidfd before reading `/proc/<pid>/environ`, validates the exact tag, and sends only through that descriptor. The one group signal occurs once before the owned `Child` is normally waited/reaped; thereafter helper cleanup uses exact-tag pidfd signaling. The group-PID reuse concern from L678 is materially addressed for the normal waitable-child lifecycle, but the above `ECHILD`/auto-reap case weakens the guard.

## Limits

- Static review only; no independent test or runtime claims. Parent-reported CLI/browser test results were not replayed.
- No repository-level external reaper was found. The finding depends on inherited ignored-SIGCHLD/auto-reap behavior or a competing reaper, which was not established as a supported deployment mode.
- Existing `try_wait` errors are silently ignored; the code neither records an error nor disables numeric group signaling when child identity cannot be confirmed.
