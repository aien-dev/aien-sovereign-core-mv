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

## Long conversations: the operator's words are kept

When a conversation gets too long for the model, older material is folded into a short recap so the session can go on. The operator's own messages are never folded or summarized: every one stays word for word, in order, for the whole session (sovereign-core #664). Only Allen's own material is folded: his replies, tool calls, tool output and earlier recaps. The recap sits after the operator messages it follows, and a short line in it marks where each operator message came.

- **One recap:** only the newest recap is kept. When it grows past 2,000 tokens, its oldest lines are dropped.
- **Nothing is lost:** every folded message is kept, byte for byte, in the session's compaction records, and the records rebuild the whole original conversation.
- **Recorded at once:** each compaction is written to the session journal as soon as it happens, so a resumed session loads the recap or the messages it replaced, never both.
- **Per-turn folding (opt-in):** `AIEN_MICRO_COMPACT=1` folds one exchange (Allen's part of one turn) after every turn, so the cost of each pass stays small. It is off by default because every pass changes the start of the prompt the model sees. A resumed session continues after the last recap and never folds an exchange twice.
- **The trade-off:** because the operator's messages are never folded, very long operator messages stay in the conversation. When that alone is over the limit, the loop says so instead of cutting them.

## Tools: side by side when safe, large output kept aside

When Allen asks for several tools in one reply, the loop plans them before running them (sovereign-core #666):

- **Reads run together:** viewing, listing and searching run side by side, up to eight at once, even on the same file.
- **Writes are kept apart:** a write never runs at the same time as another call on the same path or a folder that holds it.
- **Some tools always run alone:** shell commands, tools the planner does not know, and any call that would ask the operator for approval run alone, so two approval questions never share the screen.
- **Order is kept:** two calls that touch the same path run in the order Allen asked for them, and results come back in that order however they finish. Every request of a step is recorded in the journal before any of them runs.
- **Large output is saved to a file:** a result bigger than its share of the model's window (15 percent per result, 30 percent per reply, at least 8,000 and 16,000 characters, at most 100,000 and 200,000) is written whole to `basecamp/sessions/<session-id>-spill/`. Allen gets the start and the end of it (40 and 60 percent of 1,500 characters) and the file's path, so he can read the rest a page at a time instead of running the tool again. File reads are never saved aside this way, so a saved file can always be read back.

The code is in `src/turn_journal.rs` (the journal), `src/turn_recovery.rs` (failure classes), `src/compaction.rs` (folding), and `src/tool_plan.rs`, `src/tool_batch.rs` and `src/tool_spill.rs` (tool batches). The mechanisms follow Hermes Agent (Nous Research, MIT licence, commit 5480918). It was re-implemented from reading; no code was copied.

### Not covered yet

- The recap is built by rule (the first line of each message), not by a language model. A model-written summary, with its own failure pacing (skip an exchange after three failures, wait 60 s, then 300 s, then 900 s), is the next part of #664.
- Token counts are still estimated locally. Using the counts the model server reports (#664) is not built yet.
- `aien-cli goal auto` (the autonomous goal loop) still runs tools one at a time without the journal.
- Tool batches have no time limit yet: a tool that hangs holds its step. Paths are compared by spelling, so a symlink that gives one file two names is not detected.
- Trying another credential or provider before a fallback model (part of #665) is not built: the loop has one model seat and no fallback chain.
