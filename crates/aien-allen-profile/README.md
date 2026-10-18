# aien-allen-profile

A small profile you can change for your ALLEN: a name, a tone, how much it says,
whether it keeps to plain language, and a few working preferences. It is tied to
your ALLEN identity, but it is not the identity. Renaming ALLEN never changes
who it is.

Tracks aien-dev/aien-architecture#159. This is version 1, host side only.

## What you can change

| Setting | Allowed values |
| --- | --- |
| name | 1 to 64 characters, one line. Default `ALLEN`. |
| tone | `neutral` (default), `warm`, `direct`, `formal` |
| verbosity | `brief`, `normal` (default), `detailed` |
| plain language | on or off (default off) |
| preferences | up to 32 of `key=value`, value up to 200 characters, scope `all`, `personal`, `work` or `project` |

Preference keys use lowercase letters, digits, `_`, `.` and `-` (up to 48
characters). A key that contains grant, permission, approve, capability, allow
or authori is refused: permissions are not profile fields, approvals go through
the approval desk.

## How to use it

The daemon owns the profile. You talk to it with `aien allen`. Each call prints
one JSON object and exits 1 on failure.

```text
aien allen status
aien allen show
aien allen set --expect 0 --name Nova --tone warm --pref units=metric@work
aien allen history
aien allen revert --to 1 --expect 3
aien allen reset --expect 4
```

Every change needs `--expect <revision you looked at>` (0 when nothing is saved
yet). If the profile moved since, the change is refused and nothing is written.
`set` also takes `--verbosity`, `--plain-language 0|1`, `--unset-pref KEY` and
`--note TEXT`. Undo is `revert`: it writes a new revision that copies an earlier
one, so history is never rewritten. `reset` writes a new revision with the
defaults. The identity is untouched by both.

If the daemon is not running, `aien allen` says so and calls it temporary. Your
identity and saved profile are not lost.

## What is enforced and what is advice

Enforced by code, with tests:

- The schema, its limits, and the rule that permission-like keys are refused.
- The binding to the engaged identity: a profile written for another agent or
  root is refused, not used and not changed. With ALLEN not engaged, every
  profile command refuses.
- Compare-and-swap: two writers racing, exactly one wins.
- Crash safety: each revision is written to a temporary file, flushed, hard
  linked into place, and the folder is flushed. A crash leaves the old head or
  the new head, never a mix.
- A damaged, oversize, unknown-schema, unknown-field, gapped or foreign file is
  a clear refusal. It is never silently replaced.
- No path from a model reply, a task or a workspace file to the profile writer.
  Only the operator commands write.
- Saving a profile does not authorize any write. A file write still needs the
  approval step (authorize, then execute).

Advice, not enforcement:

- How a model behaves with the style block. The block is text placed in front of
  the task prompt: at most 2048 bytes, preferences sorted by key, extra
  preferences dropped and counted. It says the preferences grant no permission
  and that the task wins on conflict. A model may still ignore it.

When the saved profile is damaged or foreign, tasks still run with the defaults
and the task report says `persona.state = "refused"` with the reason.

## Limits (v1)

- No avatar and no voice. `status` lists them as unsupported; there are no
  controls for them.
- No memory scoping. The `scope` on a preference is a label in the style block,
  not a wall around memory.
- No standing goals. That needs a native format decision (aienos ADR 0018) and
  is not in v1.
- Tests are fixture based. Identities come from the test-only subject encoder.
  The real-model run is NOT_RUN (the GPU lane belongs to another session).
- The store needs a filesystem that supports hard links.
- Timestamps are informational. Nothing orders or trusts them.
- Profile commands go through the compose home, so they wait while a task is
  running.
- If syncing the folder fails after a revision is linked in, the call reports an
  error even though the revision exists. A retry then says the profile moved
  on (StaleUpdate), so check `aien allen history` before trying again.
- Any unexpected file in the profile folder makes the store refuse it as
  damaged, and the persona falls back to defaults, reported as refused.
  Leftover temp files from a crash are ignored but are not cleaned up.
- Every request re-reads and re-checks the whole revision chain, and there is
  no cap on the number of revisions yet.

## Layout

The store is the folder `<compose home>.allen-profile/`, beside
`<home>.allen-binding`. Files are `r00000000000000000001.json`, `...2.json` and
so on, never edited in place. Each holds the sha256 of the previous file's
bytes. Leftover `.tmp-*` files from a crash are ignored.
