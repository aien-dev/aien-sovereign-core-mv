# What forget does, and what it does not do

Crate `aien-allen-memory`, tracking aien-dev/aien-architecture#159.

## The claim, stated narrowly

After `forget`, the item is **unreadable in the memory store**. That is the whole
claim. It is not a claim that the content is erased from the machine, from
other systems, or from anything that already received it.

## How it works

Each item version is encrypted (ChaCha20-Poly1305) under its own random 32-byte
key. The ciphertext lives in `log/` and stays there. The key lives in
`keys/<item>-v<version>.key` (file mode 0600, folder 0700).

Forget runs in this order:

1. Write an intent marker in `pending/` (item id or scope only).
2. Overwrite each key file with zeros, fsync, delete it, fsync the keys folder.
3. Append a Forget record to `log/` (item id, scope, time; no content).
4. Remove the marker.

A crash between steps leaves the item unreadable (key gone or marker present).
`inspect` reports it as `forgotten (key missing, record pending)`. The next open
finishes steps 2 to 4. A missing key on an item nobody asked to forget is
reported as `Unresolved: key missing`, never as live and never as forgotten.

A correction destroys the superseded version's key the same way.

## What forget keeps

- The Forget record keeps metadata: item id, scope, time. It never holds content.
- The Put and Correct records keep their ciphertext. Without the key it cannot
  be read, but it is still on disk.

## What forget does NOT reach

- Daemon logs and transcripts.
- The daemon generation record (the sovereign-core #301 ledger records of model turns).
- The agent chat archive.
- Persona and compose task reports.
- Model context that was already sent: a model that saw the text before the
  forget keeps whatever it kept.
- Cortex entries.
- Any copy of the plaintext the owner made, including `export` output.
- Backups or snapshots that include `keys/` taken before the forget.
- Filesystem journal blocks or SSD remapped (wear-levelled) blocks that may still
  hold the old key bytes. Zeroing a file does not promise the physical cells are
  overwritten.

## Restore and backups

- A restore from a backup taken AFTER the forget cannot bring the content back:
  the key was already gone.
- A backup taken BEFORE the forget that includes `keys/` CAN bring back the key.
  Recommendation: back up `log/` only and leave `keys/` out, or re-forget after
  the restore. The store helps: every open runs `reapply_forgets()`, which
  destroys the key of any item that has a Forget record, and of any superseded
  version. This only works while the log is intact and newer than the forget.
  A restore that also rolls `log/` back to before the forget removes the Forget
  record, so nothing remembers it; this crate cannot detect that (NOT_PROVED).
- Keys are not wrapped by a master key. Whoever can read `keys/` and `log/` can
  read live content. The protection is scope separation in the API and access
  to the folder, not secrecy from the machine's owner.

## Scopes

`personal`, `work`, `project:<name>`. Recall takes a `ScopeGrant` for exactly one
scope, built by the daemon from the request's authorised context. A scope named
inside a query string, a model reply or a task is only search text and never
widens a grant. `InspectAll` exists for the owner-facing inspect command.

## Standing goals

Items of kind `goal` are "host goal records, not subject intents". The host never
writes SubjectState. A real subject intent needs an aienos ADR 0018 amendment.

## Other limits

- Single writer per store is assumed. A second writer racing the same record
  number loses cleanly (`Conflict`); key files of a crashed put may be left
  behind and are harmless (no ciphertext refers to them).
- The store needs a filesystem that supports hard links.
- Timestamps are informational. Nothing orders or trusts them.
- Not wired into the daemon or CLI yet.
