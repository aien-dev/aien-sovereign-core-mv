# Paired benchmark results (sc#587)

Measured against `RULE.md` (its commit, "paired benchmark rule, committed before measurement", precedes every bench run and result on this branch). Comparison is only against our
own pre-board flow (agent-mail + crumb whisper/sniff + scripted orchestrator relay).

- Counted run: `results/20261016T130132Z/` (3 pairs, order alternated: pair 1 mail then board,
  pair 2 board then mail, pair 3 mail then board). Model `claude-haiku-5-5` in every agent and
  relay step. Per-run receipts: `result.json`, `metrics.json`, `assert.json`, `task.json`,
  scrubbed transcripts, the commits each agent made.
- Failed attempt kept, not dropped: `results/20261016T123713Z-attempt1-FAIL/` (see below).

## Board rule, per board run

| Run | r1 same ids, gamma via hook as subscribed | r2 no relay, 2 board events | r3 no missed, no dup | r4 task | Pass |
| :- | :- | :- | :- | :- | :- |
| p1-board | true | true | true | true | **PASS** |
| p2-board | true | true | true | true | **PASS** |
| p3-board | true | true | true | true | **PASS** |
| attempt 1, p1-board | false | false | false | false | **FAIL** |
| attempt 1, p2-board | false | false | false | false | **FAIL** |

**Attempt 1 FAIL, reason.** The board root sat under the run dir
(`demo/bench-20261016T123713Z-p1-board/board`), which put the wake socket path at 108 bytes, one
over the 107-byte Unix socket limit. The board refused the watcher with `SocketPathTooLong`, so
beta's `swarm-bridge watch` exited with that error on both tries and beta never answered. Gamma
still received Q1 through its hook (hooks do not use wake sockets). demo-1 had passed only
because its run name was shorter. Fixes in the commit "watch polls when no wake can be armed": `watch` now falls back to polling the
cursor every 250 ms when no wake can be armed and says why (red-before-green test in
`tests/processes.rs`), and the harness keeps board roots short (`demo/b/<12 hex>`, length
checked). The attempt was stopped after its second board run; its partial pair 2 mail run was
never scored. In the counted run every `watch` reported `woken by a board notification`; the
polling fallback was not exercised.

## Side by side (n = 3 pairs, values in pair order p1, p2, p3)

| Metric | Mail (control) | Board | Board lower every pair |
| :- | :- | :- | :- |
| Ferrying model invocations (relay steps) | 4, 4, 4 | 0, 0, 0 | yes (by construction, see below) |
| Agent model invocations (alpha+beta+gamma) | 20, 18, 22 | 18, 20, 23 | **no** |
| All model invocations (agents + relay) | 24, 22, 26 | 18, 20, 23 | yes |
| Tool calls | 24, 21, 26 | 17, 19, 21 | yes |
| Waiting/polling tool calls | 5, 4, 5 | 3, 3, 3 | yes |
| Repeated context tokens (cache reads) | 680,758; 606,335; 752,552 | 524,308; 594,903; 718,977 | yes |
| Input-side tokens (input + cache create + cache read) | 798,559; 721,673; 871,026 | 601,307; 673,008; 798,880 | yes |
| CPU seconds (all claude processes) | 20.28, 19.64, 20.14 | 15.52, 15.35, 16.12 | yes |
| Coordination wall, Q1 sent to alpha holding A1 (s) | 32.5, 33.3, 32.0 | 4.0, 4.2, 3.9 | yes |
| Coordination wall, Q1 sent to gamma holding A1 (s) | 39.8, 32.9, 36.6 | 17.2, 18.1, 18.1 | yes |
| Missed updates | 0, 0, 0 | 0, 0, 0 | no (equal) |
| Duplicate updates | 1, 0, 1 | 0, 0, 0 | no (pair 2 equal) |
| Task result (note + gamma test rerun) | ok, ok, ok | ok, ok, ok | n/a |
| Joules | NOT_MEASURED | NOT_MEASURED | n/a |

The mail duplicates are gamma seeing Q1 twice (once relayed through `agent-mail`, once through
`crumb sniff`).

## Delivery latency (send to first appearance in the recipient's transcript)

| Arm | n | p50 (s) | p95 (s) |
| :- | :- | :- | :- |
| Mail | 12 | 13.93 | 22.20 |
| Board | 12 | 0.004 | 13.82 |

Per pair, board arm: Q1 to beta (watch) -0.005 to 0.024 s; Q1 to gamma (hook) -0.024 to 1.13 s;
A1 to alpha (watch) -0.042 to 0.004 s; A1 to gamma (hook) 13.2 to 14.2 s. The board's p95 is the
A1-to-gamma pair: gamma was inside a `sleep 15` and the hook delivers at the next tool boundary.
Mail arm pairs range 12.4 to 32.9 s (relay step plus polling).

Measurement caveat: times come from transcript timestamps written by each Claude Code process.
The sender's tool-result line can be stamped after the recipient's line, so near-zero board
latencies can come out slightly negative. Read them as "same instant within tens of ms", not as
exact values.

## What this does and does not show

- The board arm removed the orchestrator relay: 12 ferrying model invocations in the control
  across 3 runs, 0 in the board arm. The 0 holds by construction (the board arm runs no relay
  step); it is the cost of the relay flow, not a measured improvement of the agents.
- Agents in both arms still pay to read and reason about every message. Agent-only model
  invocations were **not** lower on every pair (board higher on pairs 2 and 3), so no saving is
  claimed there. The total-invocation saving comes from the relay steps alone.
- Lower tool calls, cache-read tokens, CPU and coordination wall time held on every pair. The
  wall-time gap is mostly the relay latency plus polling intervals in the control.
- n = 3 pairs on one machine, one task, one model. Energy was not metered.
