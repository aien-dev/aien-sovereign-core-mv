# Spark model gateway

`spark-model-gateway` is a compatibility and admission membrane around the current MAX/Nemotron server. Its provider boundary is replaceable, so a future native `aien-runtime` provider can use the same authentication, queue, budget, streaming, and metrics surface.

## Ports

- MAX/Nemotron binds privately to `127.0.0.1:18006`.
- The gateway binds to `127.0.0.1:18007`.
- OpenCode, AIEN workers, and background jobs use only `18007`.

## Admission invariant

The gateway loads the exact local production `tokenizer.json` and renders the configured Nemotron Qwen chat template before counting tokens. A request is admitted only when:

```text
assembled_prompt_tokens + requested_output_tokens + 4096 <= 49152
```

The configured output reserve is 8192 tokens and the maximum input budget is 36864 tokens. The 4096 token margin covers template and protocol additions and protects the model context boundary.

The waiting queue has four entries and one request may be running. A fifth waiting request receives HTTP 429, so the maximum admitted set is one running plus four waiting requests.

## Configuration

Provide tokens through the vault or service environment injection. The gateway does not write credentials:

```text
AIEN_GATEWAY_TOKEN_OPENCODE
AIEN_GATEWAY_TOKEN_AIEN_RUNTIME
AIEN_GATEWAY_TOKEN_RSI
AIEN_GATEWAY_TOKENIZER_PATH
AIEN_GATEWAY_UPSTREAM
AIEN_GATEWAY_CHAT_TEMPLATE=nemotron-qwen|plain
```

`/healthz` checks process liveness. `/readyz` checks the loaded tokenizer configuration and probes the upstream model endpoint. `/metrics` and model/completion routes require a bearer token. Metrics use bounded principal labels and never include bearer tokens, prompts, model names, or request IDs as labels.

The MAX launch retains `--max-batch-size 8`. Gateway `active slots` remains one until a Spark benchmark shows zero failures, no context violations, acceptable GPU memory, and queue-aware TTFT and service-time results at higher concurrency.
