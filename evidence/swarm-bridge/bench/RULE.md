# Paired benchmark rule (sc#587), committed before any measurement

This file fixes what is measured, how, and what counts as success, before any run. Results go in
`evidence/swarm-bridge/bench/results/` and are compared only against this rule. Comparison is
against our own pre-board flow, never against outside systems.

## The task (identical in both arms)

Three headless Claude Code agents, model `claude-haiku-5-5`, each in its own fresh shallow clone
under `~/workspace/board83/demo/<run>/` on a scratch branch, with no pushes:

- **alpha** (synapse clone) documents how Synapse would re-verify a Crumbline ledger made by
  aien-sovereign-core's `crumbs::ledger::Ledger::verify`. It must ask beta one contract question
  ([Q1]: what `Ledger::verify` returns for an empty record slice, and what it returns for a broken
  chain), wait for the answer, write `docs/notes/ledger-verify-contract.md` citing it, and commit.
- **beta** (sc clone) owns `Ledger::verify` (`crates/crumbs/src/ledger.rs`). It waits for
  questions, reads the function, and answers once ([A1]).
- **gamma** (sc clone) follows the contract thread. It adds
  `crates/crumbs/tests/ledger_verify_empty.rs` asserting the empty-slice contract **as settled in
  the thread** (it must cite the answer it received), runs that test, and commits.

Same prompts apart from the messaging instructions, same model, same `--max-budget-usd` per agent,
same poll budget (at most 12 waits of 15 s, or one `watch` of at most 300 s), same permission
mode, same live user hooks (operator-owned `~/.claude/settings.json` is not changed). The crumbs
test binary is prebuilt into a shared target dir before both arms (not measured).

## The two arms

- **pre-board (control):** messages travel the way they do today: `agent-mail send` to the
  orchestrator inbox (which also drops a `crumb whisper`, with `CRUMB_COORD_ROOT` pointed at the
  run's own coordination root), and an **orchestrator relay step**: a scripted loop that, for each
  new orchestrator mail, starts one headless orchestrator model call (`claude -p`, same model)
  that reads the mail and reposts it with `agent-mail send` to the agents who need it (beta for
  the question; alpha and gamma for the answer). Workers wait by polling `agent-mail read <me>`
  and `crumb sniff`. Inbox names are prefixed per run so no real inbox is touched.
- **board (bridge):** the synapse#83 board plus `swarm-bridge` hooks (`demo-settings.json`).
  alpha posts `swarm-bridge question --to beta --thread contract/ledger-verify`; beta parks in
  `swarm-bridge watch` and answers with `swarm-bridge answer --parent <id>`; gamma is subscribed
  to `thread:contract/ledger-verify` and receives both events through its hook context at tool
  boundaries. No orchestrator step runs.

## What is counted (per arm, per run)

| Metric | Source | Definition |
| :- | :- | :- |
| Ferrying model invocations | orchestrator relay logs | API calls (distinct assistant message ids) made by relay steps, whose only job is moving a message. Board arm: relay steps are not run, so 0 by construction; stated as such, not as a measured saving |
| Agent model invocations | stream-json of each agent | distinct assistant message ids, per agent and total |
| Tool calls | stream-json | `tool_use` blocks, per agent and total; also the subset spent waiting/polling (sleep, agent-mail read, crumb sniff, swarm-bridge watch/read) |
| Repeated context tokens | `usage` of every API call | sum of `cache_read_input_tokens` (context re-sent and read from cache); total input-side tokens (`input + cache_creation + cache_read`) also reported |
| Coordination wall time | transcript timestamps | from alpha's question leaving (tool result of the send/post) to alpha holding the answer (first transcript line carrying [A1] / the answer id), and to gamma holding it |
| CPU seconds | `/usr/bin/time` around every claude process (agents and relay steps, children included) | user + system |
| Delivery latency | transcript timestamps | for each (message, recipient) pair: send time to first appearance in the recipient's transcript. p50 and p95 over all pairs of all runs, with n stated |
| Missed updates | transcripts | expected pairs: Q1 to beta, Q1 to gamma, A1 to alpha, A1 to gamma. A pair never seen by the end is missed |
| Duplicate updates | transcripts | a message shown to the same recipient more than once, not counting a bridge redelivery labelled with the same id |
| Task result | git in each clone + rerun by the bench script | alpha's note exists and quotes the answer; gamma's test exists, cites the answer and passes when the bench script reruns it |

Joules are not metered on this path and are reported as NOT_MEASURED.

## Success rule

The board arm **passes** a run when all of these hold:

1. alpha, beta and gamma transcripts all contain the same question event id; alpha and gamma
   contain the same answer event id; in gamma both reach it as subscribed (not addressed) events
   through `hook_additional_context` (hook delivery, not a Bash read).
2. No orchestrator or relay model call ran and nobody reposted a message (one board write per
   message: exactly 2 coordination events on the board for the task).
3. Zero missed and zero unlabelled duplicate updates.
4. Task result verified as above.

The **comparison** reports every metric for both arms side by side with n. A metric is called
better only when the board arm is lower on every paired run. No saving of LLM round trips is
claimed beyond the ferrying invocations actually counted in the control arm, and the doc states
that agents in both arms still pay to read and reason about every message. If the board arm fails
the rule on any run, that run is reported as FAIL with the reason, not dropped.
