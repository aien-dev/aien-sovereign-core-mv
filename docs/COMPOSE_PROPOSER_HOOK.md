# Compose proposer hook (design note)

Scope: how an approved proposal from an upstream approval desk (INTERPLANE
`ComposeLedgerAuthority`, interplane#77) enters the production compose path.
Line numbers are for the commit that adds this note.

## Entry point
`ProposerHook::submit` (crates/aien-runtime/src/approved.rs:152) checks the
proposal (`verify`, approved.rs:109), refuses duplicates, then calls
`ComposeBridge::run_approved_task` (crates/aien-runtime/src/spine.rs:1362),
which is `run_task_inner` (spine.rs:1371), the body `RunComposeTask` runs
(server.rs:593 -> spine.rs:1351). The approved text is staged per task and
the registered model Skill (spine.rs:1224) returns it instead of calling the
`ComposeProposer` (spine.rs:1232); one attempt, the same template check, and edit
= None (spine.rs:1245): an approved proposal never goes through
`merge_edit_reply`, so an existing file is replaced byte-exact.

## Path to J-Space and World commit
`home.compose.run` (spine.rs:1412) -> `Compose::run`
(crates/aien-omega-compose/src/lib.rs:302) -> `rxc_host_run` in omega
`librx_compose.a` (omega.lock c0369e67, archive sha256 dfe0ffb8...): the Skill
result is staged on a J-Space branch, AEGIS calls the bridge's verify callback
(spine.rs:1273: result handle equals the Skill's text and it parses as one
file change), the winner commits through the World, Cortex records goal,
candidates, evidence and promotion (`cx_*` in `ComposeTaskReport`). The hook
returns Ok only if the run committed exactly the approved text.

## What the ledger hands over (`ApprovedProposal`)
request_id, trace_id, approval_id, approver, path, content,
approved_proposal_sha256, content_sha256, approval_mac. The hook returns
`ApprovedComposeReport`: state (COMMITTED | ALREADY_COMMITTED), the same
ids, compose_proposal_sha256, grant_links, desk_key_id, approval_key,
replay_claim and the compose `task` report (None for ALREADY_COMMITTED); or
`ApprovedRefusal` {refused_by, name, detail, request_id, trace_id,
approval_id, replay_claim}.

## Daemon command (#249 C)
`ControlCommand::ComposeApprovedProposal { proposal, workspace }` on the
daemon socket (same-user peer check, 0600 socket, like every compose
command). Answered `ComposeApprovedResult(report)` or
`ComposeApprovedRefused(refusal)`. It runs `ProposerHook::submit`; there is
no unauthenticated variant. It writes nothing to the workspace and mints no
grant: the effect still needs an `authorization` record, a
`ComposeEffectIntent` and a `ComposeEffectAck`, all naming
compose_proposal_sha256. `aien compose desk-key [--create 1]` prints the
desk key id and path (never the key) and creates the key if none exists.

## Authentication design (#249 A)
Reused primitive: none of the existing grant records can carry it. A
`ComposeNote` `authorization` grant is written by any same-user socket
caller, so binding the approval to one would be caller text in two steps;
the aien-mcp `ApprovalGrant` lives in the INTERPLANE process memory and the
daemon cannot check it. So the smallest verifiable evidence is used:
`approval_mac` = HMAC-SHA256 (RFC 2104 over the existing sha2 crate, no new
dependency) keyed by the approval desk key, over the canonical binding
(`approved_auth::binding_bytes`): compact JSON, keys sorted, of
approval_id, approved_proposal_sha256, approver, content_sha256,
desk_key_id, path, request_id, trace_id and `"v":"aien.approval.v1"`.
The daemon recomputes it with its copy of the key and compares in constant
time; anything else is `PROPOSAL_REFUSED Unauthenticated`.
- Key: 32 bytes from /dev/urandom, `<compose dir>/approval-desk.key`,
  64 lowercase hex, created 0600 with O_EXCL|O_NOFOLLOW. Every submission
  re-loads it: a symlink, a non-regular file, another owner, any group or
  other permission bit, or a malformed key is `NoDesk`. Only desk_key_id
  (first 16 hex of sha256 of the key) is ever printed or recorded.
- Confinement: the compose home and the submission's workspace must not
  overlap (either inside the other), else `Confinement`: the model-facing
  tools read inside the workspace, never the key or the journal.
- What it proves: the holder of the desk key approved exactly these fields.
  Not which human pressed approve: the desk (INTERPLANE host-only approval
  continuation) names the approver, and the MAC stops anyone without the key
  from changing it. Caller text can never carry a valid MAC.
- Rotation / loss: replacing the key refuses every MAC made with the old
  one (approvals issued and not yet submitted must be approved again);
  claims already recorded keep their keys. Without a key every submission
  is `NoDesk`.
- Limit: the OS user is still the outer boundary; a process of the
  daemon's user that can read the compose home can read the key.

## Replay (#249 B, crate::approved_replay, session 476ca4)
`approval_key` = sha256 of the binding above; the claim keys are the
approval key, the request id and the approval id, each on its own. Order in
`ProposerHook::submit`: verify -> reconcile gate -> desk key -> confinement
-> MAC -> `claim` (durable `accepted`) -> `mark_in_flight` -> one
`run_approved_task` -> `commit` | `fail` (ran, not committed) | `uncertain`
(run error, or committed but not the approved text). A refused claim is
returned with its state; a retry of the very same approval (all keys equal)
whose claim committed gets the original result back as ALREADY_COMMITTED,
never a second run. Daemon start runs `approved_replay::reconcile_at_start`
after the effect reconcile (accepted -> not_executed, in_flight ->
uncertain); a failure gates effect commands and approved proposals.

## Hash semantics (fix)
- approved_proposal_sha256 = sha256 of compact JSON {"content","path"} with
  sorted keys (interplane#77). Checked, then carried for provenance.
- compose_proposal_sha256 = sha256 of the committed text
  `filename: <path>\n<content>`. The authorization grant and the
  `ComposeEffectIntent` must name this value: `aien compose authorize` does
  (crates/aien-cli/src/compose.rs:252) and `Ledger::check_intent` compares it
  (crates/aien-runtime/src/effects.rs:341). Grant links = [cx_promotion,
  cx_evidence], as compose.rs:271.
- Content the template parser would normalise (no final newline, leading
  blank line, code fence) is refused, so content_sha256 is the committed one.

## Refusals (never silent)
`PROPOSAL_REFUSED` `Unverified` (hash, ids, path, template round trip),
`ReconcileFailed`, `NoDesk`, `Confinement`, `Unauthenticated` (nothing
appended for any of these); `REPLAY_REFUSED <Name>` from the claim
(AlreadyCommitted, AlreadyFailed, AlreadyConsumed, Uncertain, InFlight, ...);
after a claim: `ComposeError` and `Mismatch` (claim UNCERTAIN),
`NotCommitted` (claim FAILED).

## Deferred
- The effect receipt (`record_effect_receipt`) still has no request_id or
  trace_id (#76; see below). The replay `accepted` record carries
  `request_id` and `trace_id` (and `approval_id`, `approval_key`).
- Real model leg NOT_RUN; no GPU.

## Correlation for retained records (#76, design note only)
Today trace_id and request_id survive in: the command's report and refusal
(`request_id`, `trace_id`), the replay `accepted` record (Cortex host
record, kind effect, fields `request_id`, `trace_id`, `approval_id`,
`approval_key`), and the compose goal text (request only). The effect
grant/intent/ack records and `record_effect_receipt` carry neither.
Recommended: option B, bind rather than re-version: the effect grant written
after this command links (Cortex links) the replay claim record id
(`replay_claim`) next to [cx_promotion, cx_evidence], and the grant text
names `replay_claim`; a verifier walks ack -> intent -> grant -> claim and
reads trace_id/request_id from the immutable claim record. Receipt files
stay version 1. Tests to add: grant links the claim; walk from an ack
recovers trace_id and request_id; a grant naming a claim whose
compose_proposal_sha256 differs is refused at intent; a forged claim id
(not an `accepted` record) is refused.
