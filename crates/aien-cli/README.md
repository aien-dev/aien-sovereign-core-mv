# aien-cli

The `aien-cli` binary is AIEN's command line and the chat loop that Allen runs in on a host.

## Sessions: record first, then act

Every chat session keeps a journal in the sessions folder (`basecamp/sessions/<session-id>.journal.jsonl` under the home folder). Each message and each tool request is written to the journal and flushed to disk before the loop acts on it. This is the chat-loop form of the AIENOS continuity rule: advance the record, then act (sovereign-core #665).

- **Continue a session:** `aien-cli --resume <session-id>`. The id is printed when a session starts.
- **After a crash:** if the process stopped after a tool request was recorded but before its result was, the resumed session runs read-only requests (view, list, search, and the `list_directory` alias of list) again. For anything else it tells Allen that it is not known whether the request happened, and it never runs that request a second time.
- **Autonomous goals:** `aien-cli goal auto` uses the same journal. Each tool request is on disk before the tool runs. If that write fails, the goal stops.
- **Damage:** a journal damaged anywhere except its last line is refused. A last line cut off mid-write is dropped.
- **Failed model calls:** each kind of failure gets one recovery per turn: one retry after a network error, one wait after a rate limit, and one compaction when the conversation is too long. A second failure of the same kind ends the turn with a named reason, so recovery cannot loop.
- **Cut-off tool calls:** a reply cut off inside a tool call is never run. It is asked again once with a higher output limit.

The plain `basecamp/sessions/<session-id>.json` file is still written at the end of each turn for readers that expect it.

## Long conversations: the operator's words are kept

When a conversation gets too long for the model, older material is folded into a short recap so the session can go on. The operator's own messages are never folded or summarized: every one stays word for word, in order, for the whole session (sovereign-core #664). Only Allen's own material is folded: his replies, tool calls, tool output and earlier recaps. The recap sits after the operator messages it follows, and a short line in it marks where each operator message came.

- **One recap:** only the newest recap is kept. When it grows past 2,000 tokens, its oldest lines are dropped.
- **Nothing is lost:** every folded message is kept, byte for byte, in the session's compaction records, and the records rebuild the whole original conversation.
- **Recorded at once:** each compaction is written to the session journal as soon as it happens, so a resumed session loads the recap or the messages it replaced, never both. If that write fails, the turn stops. Resume then loads the history from before the compaction, which was already on disk.
- **Per-turn folding (opt-in):** `AIEN_MICRO_COMPACT=1` folds one exchange (Allen's part of one turn) after every turn, so the cost of each pass stays small. It is off by default because every pass changes the start of the prompt the model sees. A resumed session continues after the last recap and never folds an exchange twice.
- **The trade-off:** because the operator's messages are never folded, very long operator messages stay in the conversation. When that alone is over the limit, the loop says so instead of cutting them.

The code is in `src/turn_journal.rs` (the journal), `src/turn_recovery.rs` (failure classes) and `src/compaction.rs` (folding). The mechanisms follow Hermes Agent (Nous Research, MIT licence, commit 5480918). It was re-implemented from reading; no code was copied.

### Not covered yet

- The recap is built by rule (the first line of each message), not by a language model. A model-written summary, with its own failure pacing (skip an exchange after three failures, wait 60 s, then 300 s, then 900 s), is the next part of #664.
- Token counts are still estimated locally. Using the counts the model server reports (#664) is not built yet.
- `aien-cli goal auto` does not use the interactive loop's one-shot model-call recovery.
- Trying another credential or provider before a fallback model (part of #665) is not built: the loop has one model seat and no fallback chain.
