# aien-cli

The `aien-cli` binary is AIEN's command line and the chat loop that Allen runs in on a host.

## Sessions: record first, then act

Every chat session keeps a journal in the sessions folder (`basecamp/sessions/<session-id>.journal.jsonl` under the home folder). Each message and each tool request is written to the journal and flushed to disk before the loop acts on it. This is the chat-loop form of the AIENOS continuity rule: advance the record, then act (sovereign-core #665).

- **Continue a session:** `aien-cli --resume <session-id>`. The id is printed when a session starts.
- **After a crash:** if the process stopped after a tool request was recorded but before its result was, the resumed session runs read-only requests (view, list, search) again. For anything else it tells Allen that it is not known whether the request happened, and it never runs that request a second time.
- **Damage:** a journal damaged anywhere except its last line is refused. A last line cut off mid-write is dropped.
- **Failed model calls:** each kind of failure gets one recovery per turn: one retry after a network error, one wait after a rate limit, and one compaction when the conversation is too long. A second failure of the same kind ends the turn with a named reason, so recovery cannot loop.
- **Cut-off tool calls:** a reply cut off inside a tool call is never run. It is asked again once with a higher output limit.

The plain `basecamp/sessions/<session-id>.json` file is still written at the end of each turn for readers that expect it.

The code is in `src/turn_journal.rs` (the journal) and `src/turn_recovery.rs` (failure classes). The mechanism follows Hermes Agent (Nous Research, MIT licence, commit 5480918). It was re-implemented from reading; no code was copied.

### Not covered yet

- `aien-cli goal auto` (the autonomous goal loop) still runs tools without the journal.
- Trying another credential or provider before a fallback model (part of #665) is not built: the loop has one model seat and no fallback chain.
