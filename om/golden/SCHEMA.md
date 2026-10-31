# Golden-token file schemas

`golden.om` admits these two shapes and refuses anything else (unknown or missing keys, wrong types,
out-of-range values) with exit 4.

## Prompt set: `aien-golden-prompts/1`

```json
{"schema":"aien-golden-prompts/1","enable_thinking":false,"prompts":[
{"id":"p0","content":"..."}
]}
```

- `enable_thinking`: bool, sent as `chat_template_kwargs.enable_thinking` in every request.
- `prompts`: 1 to 1024 objects with exactly `id` (unique, 1 to 64 of `[A-Za-z0-9._-]`) and `content`
  (non-empty JSON string, copied into the request as the same JSON lexeme).
- The record pins the prompt set by `prompts_sha256`, the sha256 of the file's exact bytes.

Each request body is one line:

```json
{"model":"<model_repo>","messages":[{"role":"user","content":<content>}],"temperature":0,"top_p":1,"seed":0,"max_tokens":<N>,"stream":false,"chat_template_kwargs":{"enable_thinking":<bool>},"aien_return_token_ids":true}
```

An engine reply is refused (exit 5) when it carries `error`, lacks `usage.aien_token_ids` (an array
of u32), or has `usage.aien_sampling` that is not an object with `"mode":"greedy"`.

## Record: `aien-golden-record/1`

```json
{"schema":"aien-golden-record/1","label":"REAL","engine_sha256":"<64 hex>","model_repo":"org/name","model_rev":"<40 hex>","sampling":{"temperature":0,"top_p":1,"seed":0,"enable_thinking":false},"max_tokens":256,"prompts_sha256":"<64 hex>","host_id":"<id>","utc":"YYYY-MM-DDTHH:MM:SSZ","prompts":[
{"id":"p0","ids":[1,2,3]}
]}
```

| Key | Rule | Compared by `cmp` |
|---|---|---|
| `label` | `REAL` (recorded from an engine), `FAKE` (recorded from a test responder), `SYNTHETIC` (hand-written fixture) | yes |
| `engine_sha256` | 64 lowercase hex: sha256 of the serving binary | yes |
| `model_repo` | 1 to 96 of `[A-Za-z0-9._-/]` | yes |
| `model_rev` | 40 lowercase hex: the model repository commit | yes |
| `sampling.temperature`, `top_p`, `seed`, `enable_thinking` | integer 0, number, unsigned integer, bool | yes, as lexemes |
| `max_tokens` | 1 to 1000000 | yes |
| `prompts_sha256` | 64 lowercase hex | yes |
| `host_id` | 1 to 64 of `[A-Za-z0-9._-]` | no, printed |
| `utc` | recorder wall clock at record time | no, printed |
| `prompts` | 1 to 1024 `{id, ids}`; ids are u32 | ids per prompt, in order |

`cmp REF CAND` exits 3 (`NOTCOMPARABLE ...`) when any compared header field, the prompt count or a
prompt id differs, before looking at tokens. Otherwise it prints `MATCH <id> n=<N>` or
`DIVERGE <id> index=<K> ref=<X> cand=<Y>` per prompt, where K is the first differing position and
`end` marks a list that stopped early, and exits 1 when any prompt diverges.
