# Daemon generation record

Evidence that this daemon generated a piece of text from the model files it
loaded. It exists so a provenance chain (INTERPLANE, interplane#76) does not
have to take the run driver's word for "this text came from this model".

## What the daemon does

1. At start, after the model loads, the CLI holds the digest of the model
   weights (below: `model_sha256`, files streamed in 1 MiB chunks, hashed once)
   and the sha256 of the tokenizer file, with the path each came from. The
   daemon passes both to the server (`set_model_identity`). The same digests are
   in the "checkpoint loaded" log line; a sharded checkpoint also logs a
   `CHECKPOINT_SHARDS` line (below).
2. On every `StreamTurn` Finished event, and when the compose ledger is open,
   the daemon appends one record **before classifying completion**. Only
   `StopToken` sends `TurnFinished`, whose optional `generation_record` id is
   absent when no record was written. `LengthLimit`, `Aborted` and `Preempted`
   send the existing `Error` response with `max_tokens`, `aborted` or `preempted`
   in its text. Their records remain in the ledger and can be retrieved with
   `ComposeRecall`; the Error response does not contain a structured record id.
   No wire field was added: older clients already treat Error as failure.
   An unchanged older daemon can still misclassify these reasons; this source
   change is not a deployment claim.

## The record

An `effect`-class host note whose JSON has the marker field `generation`:

| field | meaning |
|---|---|
| `generation`, `v` | marker (1) and record version (1) |
| `model_sha256`, `model_path` | the digest of the loaded weights (form below) and the canonical path the daemon was given |
| `model_digest_kind` | which form `model_sha256` has: `file` or `index+shards` (below) |
| `tokenizer_sha256`, `tokenizer_path` | the loaded tokenizer file's digest and path |
| `prompt_ids_sha256` | `token_ids_sha256` of the submitted prompt token ids |
| `output_token_ids_sha256` | same hash over the generated token ids |
| `output_text_sha256` | sha256 of the exact accumulated text (returned in `TurnFinished` for a normal stop, or emitted as deltas before an incomplete-turn Error) |
| `total_tokens`, `output_tokens` | token count the engine reported; generated tokens |
| `finish_reason` | `eos`, `max_tokens`, `aborted` or `preempted` |
| `decoding` | how the backend chose the output tokens (below); ABSENT when the backend did not report it |
| `ops` | the tensor backend that ran the call and its op counters (below); ABSENT when the backend did not report them |
| `daemon` | `pid`, `start_ticks` (process start, clock ticks) and `started_unix_ms` |
| `request_id`, `operation_id` | CALLER-ASSERTED: the request envelope's ids, exactly as the client sent them |

The envelope ids are chosen by the client. They are recorded, not trusted (names kept to match the envelope fields).

### `model_sha256`: the digest of what was loaded (sc#338)

The daemon digests exactly the files the weights loader reads (the same
resolution: a `.safetensors` file, a `model.safetensors.index.json`, a model
directory holding either, or a shard file whose directory has an index, which
loads the whole sharded set).

- `model_digest_kind` = `file`: one safetensors file. `model_sha256` is that
  file's sha256 (`sha256sum model.safetensors`).
- `model_digest_kind` = `index+shards`: a sharded checkpoint. `model_sha256` is
  the sha256 of this UTF-8 manifest text, with `\n` line ends and a final `\n`:

  ```
  aien-checkpoint-digest v1
  index <sha256 of model.safetensors.index.json>
  <shard sha256>  <shard name>
  ...
  ```

  one shard line per file the index's `weight_map` names, de-duplicated and
  sorted by name (byte order), the name exactly as the index writes it. The
  shard lines are `sha256sum` lines, so a verifier can rebuild the manifest from
  `sha256sum` output in the model directory and hash it. A changed shard, a
  changed index, or a renamed shard changes `model_sha256`; the index alone does
  not determine it.
  The names are written raw: for a shard name holding a backslash or a line
  break, `sha256sum` escapes the line (leading `\`), so rebuild that line by
  hand. An index that names no shard is refused, and so is a directory given
  as the checkpoint path (as it was before sc#338).

The daemon also logs, on stdout after the "checkpoint loaded" line (which keeps
its pinned form, parsed by interplane#76), for a sharded checkpoint only:

```
CHECKPOINT_SHARDS model_sha256=<digest> model_digest_kind=index+shards index_sha256=<sha256> shards=[<name>:<sha256>,...]
```

Records written before sc#338 have no `model_digest_kind`; their
`model_sha256` is the sha256 of the file the daemon was given, which for a
sharded checkpoint was the index only.

### `decoding`: the decoding actually taken (sc#294)

The Qwen3 v4 review found greedy decoding documented but not observed: no
record said which decoding a run used. `decoding` is that observation. The
native backend (`NativeTransformerBackend`) counts every token it chooses
inside the branch that chose it (`sample_with_params_observed` for decode, the
prefill pick in `execute_step`), and the scheduler takes the count when the
sequence finishes (`take_decode_observation`) and puts it in
`CompletionEvent::Finished.decoding`. The daemon copies it into the record. It
is never derived from the request's `temperature` or from any config file.

| field | meaning |
|---|---|
| `mode` | `greedy` (every chosen token was the argmax), `sampled` (every one was a random draw), `mixed`, or `none` |
| `greedy_tokens`, `sampled_tokens` | how many chosen tokens took each branch |
| `temperature` | temperature of the sampled draws; only when a draw sampled |
| `top_p` | nucleus mass of the sampled decode draws; only when filtering ran (`0 < top_p < 1`) |
| `seed_request_id` | the backend request id the draws were seeded from; only when a draw sampled |

There is no top-k: `SamplingParams` has none. Counted tokens are the pick
after the final prefill chunk (a mid-prompt pick that is discarded is not
counted) and every decode pick, the stop token included, so the count can be
one more than `output_tokens`. A swarm branch counts only its own decode picks.
A preempted request finishes once at the preemption (the scheduler takes its
count then); after it resumes, counting starts again, so a later record covers
only the tokens chosen after the resume.

`decoding` is ABSENT, not `greedy`, when the backend does not observe its
decoding (the mock backend, a backend without the hook): an absent field is an
absent claim. A verifier that needs greedy decoding requires `decoding.mode ==
"greedy"` and `decoding.sampled_tokens == 0` on every record.

What it does not prove: like the rest of the record, it is the daemon's own
report, not signed. It covers the native backend's scheduler path
(`execute_step`); the standalone generation helpers (`generate_tokens*`) and
example binaries are not observed.

### `ops`: the backend and its fallbacks (sc#337)

"No silent fallback" used to be shown only by the absence of a crash. `ops`
is positive evidence. The native backend reads its tensor backend's op
counters (`TensorBackend::op_report`) when a sequence finishes
(`AienInferenceBackend::op_evidence`), the scheduler puts them in
`CompletionEvent::Finished.ops`, and the daemon copies them into the record.

| field | meaning |
|---|---|
| `backend` | the tensor backend's name (`TensorBackend::name`, e.g. the Omega GB10 engine or `ReferenceCpuBackend`) |
| `native_fallbacks` | runs of ops the backend claims native that ran on the reference CPU path |
| `reference_runs` | runs of ops on the reference CPU path by design (outside the backend's native mask) |
| `scope` | always `process`: the counts are totals since the daemon built the backend, NOT per call |
| `report` | the full `OP_REPORT native=[..] reference=[..] native_fallbacks=[..] reference_runs=[..]` line |
| `gpu_executions` | sc#402: operations the backend ran on its chip (the Omega GB10 engine's `chip_calls`), process total like the rest; ABSENT, not zero, for a backend with no chip (the CPU reference backend, the Omega stub). The `OP_REPORT` log line then carries ` gpu_executions=<n>` before ` backend=` |

Because the counts are process totals, a later record's counts include every
earlier call's; a per-call figure is the difference between two records of the
same daemon (`daemon.pid` and `daemon.start_ticks` equal), and only when no
other call ran in between. A production build panics on the first
claimed-native fallback (`STRICT_REAL_MODEL_VIOLATION`), so a record written by
a production daemon always has `native_fallbacks == 0`; a verifier that needs
a strict run requires `ops` to be present with `native_fallbacks == 0` and the
daemon's `STRICT strict=true dev_fallback_build=false ...` start line. `ops` is
ABSENT, not zero, when the backend does not account its ops (the mock backend
and the other `AienInferenceBackend` implementations; only
`NativeTransformerBackend`, the backend the daemon builds, reports them).

The daemon also logs, at the end of every model call (turn and compose), the
`report` line with ` backend=<name>` appended, on stderr. At start it prints
`STRICT strict=<bool> dev_fallback_build=<bool> require_checkpoint=<bool>
backend=<name>` on stdout with its other start lines: `strict` is the
effective value (false with `dev_fallback_build=false` means the
`AIEN_DEV_FALLBACK=1` opt-in), `dev_fallback_build` is the compile-time
`dev-fallback` feature, `require_checkpoint` is the effective checkpoint
policy.

## What it proves

The daemon process with that pid and start time generated text with that
digest, from prompt ids and output ids with those digests, while it held model
and tokenizer files with those digests. The record sits in the compose ledger, whose records carry digests the engine
re-checks on every read (`verified`), so an edited record is detected rather than
silently accepted.

## What it does not prove

- No capability, permission or authority. Nothing reads this record: no grant,
  intent, replay, commit or effect decision depends on it
  (`effects::generation_records_change_no_decision`).
- Not that the digest is of the very bytes the loader parsed. The CLI hashes the
  model files (for a sharded checkpoint, the index and every shard) and the
  tokenizer file at load, and the loaders then open the same paths again (the
  weights loader and tokenizer loader take paths, not bytes). The digest is of
  the files read at load time; a swap between the hash and the load is not
  detected.
- Not that the file on disk still has that digest later, only what was loaded.
- Not signed. The ledger's per-record digest detects edits to a record, not a
  forged append: a process of the same user that can append to the compose
  ledger could append a record that passes the digest check. Only the daemon
  writes one through the socket (ComposeNote refuses it), but the file is not
  protected against that user.
- Not that the text is correct, safe or what an operator approved.
- A caller cannot write one: `ComposeNote` refuses kind `generation` and any
  `effect` note carrying the `generation` field.

## Verifier guidance

Parse the note text strictly as one JSON object, require the top-level
`generation` marker (value 1) and `v` 1, require the record's links to be empty,
and compare the digests to the files and text you hold. Treat request_id and
operation_id as the client's claims.

## Timing

The record is written before either terminal response (`TurnFinished` or an
incomplete-turn `Error`), and the write waits for the compose home lock. A
running compose task holds that lock, so either response can be delayed until
the task releases it. There is no deadlock: the compose task's own model calls
(`generate_text`) write no record and take no home lock.

## No record means no claim

A normally stopped turn still succeeds, with `generation_record` absent, when:

- there is no compose ledger;
- the daemon has no model identity (a stub run, or the reference-weights
  fallback, or a tokenizer that did not load);
- the ledger append failed. The failure is logged
  (`generation record not written`); an id is never returned for a record that
  was not written.

A verifier must treat an absent record as an absent claim, not as a pass.

## Provenance link: which generation made a commit, and which ALLEN asked

The `StreamTurn` record above says what the daemon generated. It did not say
which commit that text became. The compose path (`aien compose propose`, the
model Skill) now closes that gap (arch#162, ALLEN end-to-end demo v1 steps S4
and S8).

### What the daemon does

1. When a compose task commits a model proposal, the daemon writes one
   generation record for the proposal attempt the commit came from (the
   `parsed` attempt), in the same format and with the same writer as above.
   It is written under the compose home lock the task already holds (the
   model call itself cannot take it). Extra fields on this record:
   `origin` = `compose_proposal`, `task` (the compose task id) and `attempt`
   (1-based). `request_id` is 0 and `operation_id` is `"0"`: the task makes
   this model call itself, there is no client envelope. `prompt_ids_sha256`
   and `output_token_ids_sha256` are the hashes of the ids the inference path
   returned for that attempt. `output_text_sha256` is the digest of the reply
   handed to the template parser (the assistant prefix plus the decoded
   tokens), the same digest the attempt records as `text_sha256`. In edit
   mode the committed proposal is that reply merged into the prior file, so
   the proposal digest is not this one; the link is by record id, below.
2. The id is returned in `ComposeTaskReport.generation_record` (optional
   field, absent in old reports) and written into the daemon's
   `compose_commit` record.
3. Every record of the commit chain carries one evidence object,
   `"provenance": {"generation_record": <id or null>, "allen_agent": "<64 hex or none>"}`:
   the `compose_commit` record, the minted grant (`ComposeAuthorize`), the
   intent, and the ack. Each is copied by the daemon from the record before it
   (grant from commit, intent from grant, ack from intent), never from the
   request. An approved grant (`ComposeApprovedProposal`) carries
   `generation_record` null (no model ran) and the ALLEN agent. A reconcile
   record is not stamped; its intent is.
4. `allen_agent` is the ALLEN LogicalAgentId (`aien_allen::Resolved::agent`,
   the 64 hex shown in the `ALLEN: engaged agent=...` line) of the daemon that
   ran the task, or the explicit string `none` when no ALLEN is attached. No
   second identity system.

A verifier follows the commit's records (`aien compose recall --ids ...`) to
`provenance.generation_record`, reads that record, and compares its
`model_sha256` with the weights it holds. `aien compose effects` does not
print provenance: the effect ledger deliberately does not parse it.

### What it does not change (separation from authorization)

Provenance is evidence, never an input. `Ledger`, `check_intent`,
`check_minted_backing`, `check_approved_backing`, the world checks and
reconcile never read it: their row types have no provenance field. The only
reader is `generation::Provenance::from_text`, called from one writer helper
(`effects::stamp`) after the writer's decisions are made. A damaged,
absent or wrong-typed field reads as `generation_record` null and
`allen_agent` `none`; it can never refuse or allow anything. Tests:
`effects::provenance_changes_no_decision` (ledgers with absent, valid, bogus
and hostile provenance give identical state and identical refusals),
`effects::provenance_is_read_by_no_decision_code` (a source scan: which
functions may mention provenance, and that no ledger type has the field).

### Not forgeable by a client

`ComposeNote` refuses any note, of any kind, whose JSON has a top-level
`provenance` field, and still refuses kind `generation` and any `effect` note
with the `generation` field. The grant, intent and ack requests carry only
ids and digests, so there is no field a client can fill. A record the daemon
writes that names a generation id is refused at write time unless that id is a
verified generation record of the same ledger (`write_compose_commit`:
`compose-commit not written: provenance names generation record #N ...`).
When the daemon copies provenance along the chain (grant, intent, ack), an id
that no longer names a verified generation record is copied as null instead of
refused, so copying can never change an authorization outcome.

### Old ledgers

Records written before this change have no `provenance` field. They open and
verify as before, and a grant, intent or ack the daemon writes from them
carries the explicit default (`generation_record` null, `allen_agent` `none`).
`ComposeTaskReport.generation_record` defaults to absent on old reports.
Test: `an_old_ledger_without_provenance_still_opens_and_runs`.

### No generation record means no claim

`generation_record` is null when the daemon has no model identity (stub run,
reference weights), the attempt did not expose its token ids, the proposal is
an approved one, or the ledger append failed (logged as
`generation record not written`). The commit still stands; a verifier treats
null as an absent claim, not a pass. The same limits as above apply: the
digest is of the file read at load time, the ledger is not signed, and the
record proves nothing about capability or authority.

### Tests

- `crates/aien-runtime/tests/provenance_link_test.rs`: the real compose
  library and the real daemon server, ordinary flow through the socket;
  asserts the commit's records give a generation id whose `model_sha256` is the
  loaded model's digest and the expected agent (an engaged ALLEN, and `none`).
- `crates/aien-cli/tests/provenance_link_live_test.rs`: the same with the real
  `aien daemon` process, SmolLM2-1.7B on CPU, the real CLI. Ignored (needs a
  prebuilt binary and the model).
