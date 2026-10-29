# DGX Spark 35B local coding worker: existing endpoint qualification

Status: **operator runbook, not deployment approval** (rebased 2026-10-29 onto main; first draft 2026-10-24). Tracking: [core #730](https://github.com/aien-dev/aien-sovereign-core-mv/issues/730). No running service was changed or tested by the author of this document. This is a compatibility route to the *existing* Omega service, not a new serving implementation, inference engine, daemon, GPU wrapper, or public API.

## Evidence and boundaries

- [core #724](https://github.com/aien-dev/aien-sovereign-core-mv/issues/724) records a 2026-10-23 loopback smoke test at `127.0.0.1:18091` with a health-reported `omega-qwen36-batch` backend and `qwen3.6-35b-a3b-fp8` model. One answer is **not** coding quality or sustained serving qualification. Current branch and deployed binary are not necessarily identical to canonical main.
- `crates/aien-cli/src/client.rs` has a temporary compatibility seam: `AIEN_CHAT_BACKEND=remote`, `AIEN_MODEL_ENDPOINT` and `AIEN_MODEL_NAME`. Without explicit configuration it selects the native socket when that socket exists, otherwise the compatibility adapter at `AIEN_MODEL_ENDPOINT` or the default `http://127.0.0.1:18006/v1/chat/completions` (`atlas-lightning-omni`). The client logs its actual `chat_path` as `remote_adapter` or `native_runtime`; do not count adapter calls as native runtime evidence.
- `crates/aien-serve-http/src/server.rs` on **main** accepts structured `tools` and legacy `functions`, and folds `role=tool` into the prompt, in the Qwen3.6 qwen3_coder XML layout. A second generation while the engine is full returns 503 `server_busy`. The 2026-10-24 draft of this runbook said main rejected tools; that was true then and is false now. The live 35B executable on port 18091 may still be a different binary. Do not infer its tool or concurrency capabilities from this source or from a plain text answer. [#719](https://github.com/aien-dev/aien-sovereign-core-mv/issues/719) and [#720](https://github.com/aien-dev/aien-sovereign-core-mv/issues/720) own model/template and tool protocol qualification; [#716](https://github.com/aien-dev/aien-sovereign-core-mv/issues/716) owns bounded sessions.
- The existing CLI text `<tool_call>` executor (`crates/aien-cli/src/main.rs`) can run real effects. **Do not run a free-form agent with write, shell, vault, network, git push or deployment authority** just because the endpoint passes this runbook. First restrict it with the existing sandbox and effect-approval boundary and use a disposable repository checkout.
- The server is loopback-only. Guest/public serving, credential attachment and port exposure are not part of this runbook; #724 retains those release gates.

## A. Read-only discovery on the Spark

Run in a shell on the actual DGX Spark; do not stop jobs, change drivers, change binding, restart a daemon, or assume port ownership from the README.

```bash
set -euo pipefail
command -v curl >/dev/null
command -v jq >/dev/null

# Historical live 35B listener from #724; inspect before use.
PORT=18091
ss -ltnp | grep ":$PORT" || {
  echo "No listener on the documented port. Find the actual loopback 35B endpoint; do not start another one." >&2
  exit 1
}

BASE_URL="http://127.0.0.1:$PORT"
HEALTH=$(curl --fail-with-body --silent --show-error --max-time 5 "$BASE_URL/health")
MODELS=$(curl --fail-with-body --silent --show-error --max-time 5 "$BASE_URL/v1/models")
printf '%s\n' "$HEALTH" | jq .
printf '%s\n' "$MODELS" | jq .

MODEL_ID=$(jq -er 'if (.data | length) == 1 then .data[0].id else empty end' <<<"$MODELS")
ENGINE=$(jq -er '.engine' <<<"$HEALTH")
HEALTH_MODEL=$(jq -er '.model' <<<"$HEALTH")

# Fail closed: never silently drive the old 30B model, a stub or a CPU door.
test "$MODEL_ID" = "qwen3.6-35b-a3b-fp8"
test "$HEALTH_MODEL" = "$MODEL_ID"
test "$ENGINE" = "omega-qwen36-batch"
```

If any check fails, **do not set the model-seat environment yet**. Capture the `/health` and `/v1/models` responses after redacting sensitive identifiers, then reconcile with the currently deployed exact binary/configuration and #724. A different backend label may represent a legitimate newer implementation; it is not automatically qualified.

`/health` proves the service's **claimed** engine kind, not its actual linked GPU library, model weights or no-fallback execution. Production qualification additionally needs process executable digest, source/build pins, model/tokenizer hashes, hardware counters and negative controls under #724 and #730.

## B. A single inert model answer (no agent tools)

This is the first generation request. It changes no files, passes no tools, and stays on loopback. It still occupies GPU compute briefly, so coordinate a free window with the Spark job owner.

```bash
REQUEST=$(jq -n --arg model "$MODEL_ID" '{
  model: $model,
  messages: [
    {role: "system", content: "Reply with a concise explanation. Do not request tools."},
    {role: "user", content: "Explain why a small .om parser should reject an unbounded nested input before any device allocation."}
  ],
  max_tokens: 80,
  temperature: 0,
  stream: false
}')
REPLY=$(curl --fail-with-body --silent --show-error --max-time 180 \
  -H 'Content-Type: application/json' \
  --data-binary "$REQUEST" "$BASE_URL/v1/chat/completions")
jq -e '.choices[0].message.content | type == "string" and length > 0' <<<"$REPLY"
jq -r '.choices[0].message.content' <<<"$REPLY"
```

A nonempty answer is only a smoke test. If a different model answered, the GPU was unavailable, the call timed out, or the reply is incoherent, record FAIL and do not advance to agent execution.

## C. Opt in to the existing CLI adapter, without changing defaults

After A and B pass, these settings select the discovered local model for **this shell only**:

```bash
export AIEN_CHAT_BACKEND=remote
export AIEN_MODEL_ENDPOINT="$BASE_URL/v1/chat/completions"
export AIEN_MODEL_NAME="$MODEL_ID"
```

The `remote_adapter` trace in `aien-cli` is correct for this compatibility path, even though the destination is a local Omega inference process. Do not mislabel it as canonical native-runtime traffic.

For a manual **CLI-only** test, use an isolated, nonprivileged checkout with effect approvals and a sandbox already enforced. The CLI on main supplies tool instructions and may execute generated tool calls. When isolated appropriately, the entry point is:

```bash
cargo run --locked -q -p aien-cli -- --prompt \
  "Explain which source module should own bounded model selection; do not call tools, edit files or run commands."
```

Do **not** schedule `--auto`, open a public listener, issue guest tokens, or enable independent edits/pushes from this result. A prompt-level "do not call tools" instruction is **not** a security control; omission of a documented sandbox is a stop condition.

## D. Qualification before the local model becomes an autonomous developer

Core [#730](https://github.com/aien-dev/aien-sovereign-core-mv/issues/730) owns the missing consumer integration and task benchmark, reusing other owners:

1. Freeze 20-30 real, representative AIEN development cases and independent acceptance criteria before running any model.
2. Verify correct artifact/model/config/source identity and split measurements of prefill, first visible token, sustained decode, wall task time, context pressure and peak resident memory.
3. Qualify an actual bounded tool-request/response/continuation with inert tools under #719/#720, including parser and authorization negative tests. A text tag alone is not protocol equivalence.
4. Constrain tool effects, workspace, execution time and authorized context before attempting autonomous repo edits.
5. On repeat failures, invalid tools, out-of-budget context or high-risk work, assemble a minimal, redacted escalation package for **operator-approved** external review. The local model must still test and verify any returned proposal.
6. Publish PASS, FAIL, NOT_RUN and hardware contamination distinctly. Do not alter the live default, merge, or claim production readiness without coordinated Spark receipts and existing owner approval.

The desired outcome is **verified local coding tasks completed per hour**, not a headline tokens-per-second figure. This guide makes the first known good loopback path reproducible but **does not** finish the native .om migration, structured tool qualification against the live binary, independent assessment, or 35B production readiness.
