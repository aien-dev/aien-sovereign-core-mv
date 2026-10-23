# Native IMAP recovery, issue707 and existing PR626

## Source and executed qualification

Functional source: `2ea9ee1c804b84c23dc74651c085c00949110e36`, current-main base `754c07b118f41ad899607b60329bd851b1ccb13e`. Recovery preserves the original four implementation/test/review commits and authors as `62e6a11`, `3e76245`, `a9b1765`, `75f8b1e`; obsolete generated root-note commit81760be was not carried. Parent corrections are61b9398 and2ea9ee1. Existing PR626 at81760bed remains the only integration target.

| Executed command | Mac | Linux/Spark |
| --- | --- | --- |
| `cargo test --locked -p spark-mail-rs` | 62 passed, 0 failed, 0 ignored | 62 passed, 0 failed, 0 ignored |
| `cargo clippy --locked -p spark-mail-rs --all-targets -- -D warnings` | exit0 | exit0 |
| `cargo fmt --all -- --check` | exit0 | exit0 |

These are the same62 tests executed on two platforms, not124 distinct tests. `qualified/` and `linux/` preserve full command output, source-input hashes and completion states. Linux ran in an owned worktree/per-head target, with explicit0700 temporary parent and `quiet-build`. No executable source changed after qualification/review.

The complete Linux workspace also passed `cargo check --locked --workspace` (exit0, `workspace-check.log` and `workspace-check.exit`). This is a workspace build check, not a whole-workspace test run.

`cargo metadata --locked --format-version 1` resolved528 packages across the workspace. None is lexical-core, imap or imap-proto (`final/dependency-check.json`). `initial/graph.log.gz` losslessly preserves the complete original graph output. Lock comparison against current main (`lock-delta.json`) proves no package added or version moved: only arrayvec0.5.2, bufstream0.1.4, imap2.4.1, imap-proto0.10.2, lexical-core0.7.6 and nom5.1.3 were removed. Alert1 GHSA-2326-pfpj-vx3h remains OPEN on merged main until the authorized publisher actually integrates and GitHub re-evaluates that graph; local removal does not close it.

## Actual failures and corrections

Initial recovered source passed53 tests and the existing static review L674 found no demonstrated Important/Critical defect. That review is preserved verbatim in `REVIEW-L674.md`, but its UID/body-selection assurance was disproved by concrete parent execution, not treated as approval.

A standalone probe compiled unmodified imap.rs with Rust2024 and returned UID10's body for requested UID9 because a quoted metadata string contained `UID 9` (`uid-probe-red.*`). The initial Rust2021 probe build failed on the source's existing edition2024 let-chain syntax; that setup failure is separately retained and is not the behavioral RED.

Four added real crate regressions then failed on the unmodified implementation (`fetch-regressions-red.log`):
- Standard ENVELOPE subject text containing UID9 selected actual message UID10.
- A nested BODY attribute supplied an unrelated literal instead of the requested top-level body.
- Malformed UID atom9x was accepted as UID9.
- An actual UID after a literal was shadowed by UID text in earlier nested metadata.

The correction streams over outer attribute/value pairs, never searches arbitrary substrings for UID or BODY. Quoted strings and literal payloads are opaque, nested lists are skipped iteratively with constant parser-state space, and literal-marker positions are cached once per fragment. No per-token vector or body clone is needed. UID atoms must parse completely; duplicates, unbalanced/trailing data and unsupported quoted body values fail closed. Valid unrelated attributes and BODY section selectors retain compatibility, including quoted delimiter characters.

Additional actual REDs and subsequent green coverage protect unrelated atoms/selectors (`opaque-selector-red.log`), FETCH-prefix impostors (`fetch-word-red.log`) and unparsed text before the attribute list (`fetch-list-prefix-red.log`). The structured fixture matrix checks varied nesting, UID placement before/after literals, opaque literal contents and one-byte fragmentation. Existing memory/response/search caps, LOGIN literal/escaping/refusal rules, TLS hostname/certificate checks and provider behavior were preserved. No unsafe cast, external parser dependency, ignored test, transport waiver or weaker assertion was added.

`REVIEW-L675.md` is the separate independent corrective review of exact final2ea9ee1 against current main: no demonstrated Important/Critical defect found. It read the full diff/parser/caller paths but ran no tests or live checks. Parent runtime execution is separate from that source-review verdict.

## Scope, publication and acceptance

All tests used synthetic protocol streams and isolated local fixture state. No real mail credential was obtained, Gandi endpoint accessed, mailbox changed, live SMTP submission made, production mail service restarted or model/GPU loaded. Live Gandi/TLS-server interoperability and operator mailbox acceptance are NOT_RUN. Passing scripted fixtures is not a live-service claim.

Candidate workflow files are identical to current main; `workflows.txt` preserves all four. Thin verification runs on PR/main; releases require v-tags/manual dispatch. No non-main branch-push automerge/deployment path was found. No workflow, trust-surface, default feature or unrelated dependency pin changed.

The final pinned root-note compile must follow the complete report/raw-evidence/hash inventory, then its exact published head is verified before delivery. Local pinned checks are not signed Spark acceptance. L673 alone updates existing PR626 after fresh expected-old-head/main checks, signed and hosted exact-head gates and integrated acceptance. No duplicate PR, resumed paused lane, merge, alert dismissal, issue closure or live-mail qualification follows merely from this packet.
