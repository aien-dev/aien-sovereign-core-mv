# Repository check repair

This candidate repairs the browser screenshot failure and eight dependency
advisories reproduced on canonical main
`6477e3ea03b962784e4b8c8c6b3185cbdb59cece` for issues #661 and #724.

## Source

The browser implementation is reused byte-for-byte from the existing reviewed
`qualification/661-browser` head
`32b30d4ed27101f11470e34ed6d0bab9d01f7b40` (functional change
`2f6b5ffeda7dd1aa68f26ab548e732de0de82ee8`). Existing PR #688 remains the
integration review. This branch is a current-main qualification candidate, not
a competing browser implementation or a replacement for its original history.
The import includes only browser_mirror.rs and its libc dependency.

The browser disables the software rasterizer that stalled screenshot capture.
Cleanup uses pidfds and the exact private profile/tag to avoid signaling reused
process identifiers. Timeout paths retain the profile when cleanup cannot be
proven complete. The browser file SHA256 is
`39631076d3601b327d5c3f7fa47e6c8492ce194c5844c698e0c0025a21502a1c`.

Cargo.lock updates ctutils to 0.4.4, rustls to 0.23.46, rustls-webpki to
0.103.16, and webpki-root-certs/webpki-roots to 1.0.10. No advisory exclusions,
test skips, or check-policy changes were added.

## Focused verification

- Baseline screenshot test: exit 101, screenshot WebSocket EAGAIN.
- Repaired browser module: 8 passed, 1 existing isolated test ignored by default.
- That isolated test: explicitly run with inherited SIGCHLD ignored and Chrome
  selected, 1 passed, 0 ignored.
- Dependency audit: 8 vulnerabilities before, 0 after. Six existing nonfatal
  advisory warnings remain; this is not a claim of zero warnings.
- `cargo clippy -p aien-cli --all-targets -- -D warnings`: pass.
- `cargo fmt --all --check` and `git diff --check`: pass.

The first whole-repository run uncovered a third failure: the checkout lacked
its required `forgejo` remote. The existing Spark mirror was independently
confirmed reachable with `git ls-remote`, then its connection was restored in
the shared checkout configuration. The exact failing test passed afterward.
No mirror content was pushed or changed. This is a local configuration repair,
so a fresh checkout must configure its actual approved mirror separately.

The isolated test must run directly from the compiled test executable with
`AIEN_CHROME=/usr/bin/google-chrome` and `trap "" CHLD` inherited through exec.
An initial ordinary cargo invocation refused its explicit setup precondition;
the corrected isolated invocation is recorded in browser-isolated.log.

inputs.sha256 binds the tested source and dependency lock. These repository
checks do not qualify live model throughput, GPU release behavior, public guest
authentication, or the unfinished dual-model deployment.

## Whole-repository result

`bash scripts/agent-preflight.sh` completed with exit 0 on 2026-10-24 UTC.
All five gates passed. The first run executed all 57 crate jobs, with only the
missing local mirror configuration failing. After restoring that connection,
the final run executed spark-adapters again and reused the other 56 passing
source-bound receipts: 57 crates, zero failures, 195.0 seconds.
`cargo check --workspace --all-targets` also completed with exit 0.

preflight-summary.log retains the gate results and per-crate dispositions.
raw-logs.sha256 binds the full local logs, including the failed first run;
the full logs remain in the operator handoff rather than being represented as
fresh runs where the proof board legitimately reused prior passing receipts.
Existing ignored or hardware-conditional tests retain their original behavior.
No claim of full hardware qualification follows from this result.
