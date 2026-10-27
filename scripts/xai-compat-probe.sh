#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Drake Stapleton (AIEN / aien-dev)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# xAI / OpenAI API compatibility probe for aien-serve-http (or any OpenAI-shaped server).
#
#   scripts/xai-compat-probe.sh <base-url> [model]
#     base-url  e.g. http://127.0.0.1:18291/v1 (a SIDE-PORT copy; port 18091 is refused)
#     model     model id to request; default: the first id from GET <base>/models
#   env XAI_COMPAT_BEARER   token to send as "Authorization: Bearer"; default "aien-compat-probe"
#   env XAI_COMPAT_OUT      keep raw responses in this directory; default: a temp dir, deleted
#   env XAI_COMPAT_ORIGIN   Origin sent by the CORS preflight; default "https://example.invalid"
#
# Sends request shapes taken from the xAI REST reference (docs.x.ai/developers/rest-api-reference,
# fetched 2026-10-26) and prints one line per check:
#   VERDICT<TAB>TIER<TAB>CHECK<TAB>HTTP<TAB>DETAIL
# VERDICT is PASS or FAIL. TIER says what a FAIL costs:
#   CORE    an OpenAI/xAI SDK pointed at this base_url breaks
#   EXT     a common optional field is ignored or refused
#   XAI     an xAI-only endpoint or shape is missing
#   LEGACY  an endpoint xAI itself lists as legacy or deprecated
# Every check is engine independent: it tests what the HTTP layer must do whatever the model says,
# so a stub engine is a valid target. Exit: 0 all PASS, 1 any FAIL, 2 refused or unreachable.
set -uo pipefail
base=${1:?usage: xai-compat-probe.sh <base-url> [model]}
base=${base%/}
case $base in
*:18091 | *:18091/*) echo "refused: port 18091 is the live server; probe a side-port copy" >&2; exit 2 ;;
esac
for t in curl jq; do command -v "$t" >/dev/null || { echo "missing $t" >&2; exit 2; }; done
tok=${XAI_COMPAT_BEARER:-aien-compat-probe}
origin=${XAI_COMPAT_ORIGIN:-https://example.invalid}
if [ -n "${XAI_COMPAT_OUT:-}" ]; then out=$XAI_COMPAT_OUT; mkdir -p "$out"; else
	out=$(mktemp -d); trap 'rm -rf "$out"' EXIT
fi
fails=0 passes=0

# req NAME METHOD PATH [BODY] [extra curl args...] -> $out/NAME.body, $out/NAME.hdr; prints status.
# PROBE_AUTH overrides the Authorization value for one call; PROBE_AUTH= sends none.
req() {
	local name=$1 method=$2 path=$3 body=${4:-} auth=${PROBE_AUTH-"Bearer $tok"}
	shift 4 2>/dev/null || shift $#
	local args=(-sS -m 30 -X "$method" -o "$out/$name.body" -D "$out/$name.hdr" -w '%{http_code}')
	[ -n "$auth" ] && args+=(-H "Authorization: $auth")
	[ -n "$body" ] && args+=(-H 'content-type: application/json' --data-binary "$body")
	curl "${args[@]}" "$@" "$base$path" 2>/dev/null || true
}
verdict() { # PASS|FAIL tier check http detail
	printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" "$5"
	if [ "$1" = PASS ]; then passes=$((passes + 1)); else fails=$((fails + 1)); fi
}
# check TIER NAME HTTP JQ-PREDICATE DETAIL: PASS when the body satisfies the predicate
check() {
	local tier=$1 name=$2 code=$3 pred=$4 detail=$5
	if jq -e "$pred" "$out/$name.body" >/dev/null 2>&1; then verdict PASS "$tier" "$name" "$code" "$detail"
	else verdict FAIL "$tier" "$name" "$code" "$detail; body: $(head -c 160 "$out/$name.body" 2>/dev/null | tr '\n\t' '  ')"; fi
}

# models
c=$(req models.list GET /models)
[ "$c" = 000 ] && { echo "unreachable: $base" >&2; exit 2; }
model=${2:-$(jq -r '.data[0].id // empty' "$out/models.list.body" 2>/dev/null)}
[ -n "$model" ] || model=unknown-model
check CORE models.list "$c" "$c == 200 and .object == \"list\" and (.data[0].id | type == \"string\")" \
	"GET /models lists ids"
c=$(req models.get GET "/models/$model")
check CORE models.get "$c" "$c == 200 and .id == \"$model\" and .object == \"model\"" "GET /models/{id}"
c=$(req models.language GET /language-models)
check XAI models.language "$c" "$c == 200 and (.models | type == \"array\")" "GET /language-models (xAI)"

user='{"role":"user","content":"Say hello."}'
sys='{"role":"system","content":"You are a test assistant."}'

# chat: the xAI quick-start request (OpenAI SDK with base_url swapped)
c=$(req chat.basic POST /chat/completions "{\"model\":\"$model\",\"messages\":[$sys,$user]}")
check CORE chat.basic "$c" '. as $r | '"$c"' == 200 and .object == "chat.completion" and (.id | type == "string")
	and .choices[0].message.role == "assistant" and (.choices[0].message.content | type == "string")
	and (.choices[0].finish_reason | IN("stop","length","tool_calls"))
	and (.usage.prompt_tokens | type == "number") and (.usage.completion_tokens | type == "number")' \
	"POST /chat/completions plain"

# xAI documents max_tokens as deprecated in favour of max_completion_tokens
c=$(req chat.max_completion_tokens POST /chat/completions \
	"{\"model\":\"$model\",\"messages\":[$user],\"max_completion_tokens\":1}")
check CORE chat.max_completion_tokens "$c" "$c == 200 and .usage.completion_tokens <= 1" "max_completion_tokens=1 bounds output"

# streaming: every data line is a chunk, text arrives, stream ends with [DONE]
c=$(req chat.stream POST /chat/completions "{\"model\":\"$model\",\"messages\":[$user],\"stream\":true}" -N)
if [ "$c" = 200 ] && [ "$(grep '^data: ' "$out/chat.stream.body" | tail -1)" = 'data: [DONE]' ] &&
	grep '^data: {' "$out/chat.stream.body" | sed 's/^data: //' |
	jq -se 'length > 0 and all(.object == "chat.completion.chunk") and any(.choices[0].delta.content // "" | length > 0)' >/dev/null 2>&1; then
	verdict PASS CORE chat.stream "$c" "SSE chunks, content deltas, data: [DONE]"
else verdict FAIL CORE chat.stream "$c" "SSE shape; first line: $(head -c 120 "$out/chat.stream.body" | tr '\n' ' ')"; fi

# stream_options.include_usage: xAI sends usage in an extra chunk before [DONE] (choices empty)
c=$(req chat.stream_usage POST /chat/completions \
	"{\"model\":\"$model\",\"messages\":[$user],\"stream\":true,\"stream_options\":{\"include_usage\":true}}" -N)
grep '^data: {' "$out/chat.stream_usage.body" | sed 's/^data: //' | jq -s . >"$out/chat.stream_usage.json" 2>/dev/null
if [ "$c" = 200 ] && jq -e 'any(.usage.total_tokens? | type == "number")' "$out/chat.stream_usage.json" >/dev/null 2>&1; then
	verdict PASS CORE chat.stream_usage "$c" "usage reaches the client in the stream"
else verdict FAIL CORE chat.stream_usage "$c" "no usage chunk in stream"; fi
if jq -e 'any(.usage != null and (.choices | length == 0))' "$out/chat.stream_usage.json" >/dev/null 2>&1; then
	verdict PASS XAI chat.stream_usage_chunk "$c" "usage in its own chunk with empty choices"
else verdict FAIL XAI chat.stream_usage_chunk "$c" "usage not sent as a separate choices:[] chunk"; fi

# tools: accepted with tool_choice auto; a tool_choice naming an unknown function must be refused
tools='[{"type":"function","function":{"name":"get_weather","description":"Weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'
c=$(req chat.tools POST /chat/completions "{\"model\":\"$model\",\"messages\":[$user],\"tools\":$tools,\"tool_choice\":\"auto\"}")
check CORE chat.tools "$c" "$c == 200 and .object == \"chat.completion\"" "tools + tool_choice auto accepted"
c=$(req chat.tool_choice_checked POST /chat/completions \
	"{\"model\":\"$model\",\"messages\":[$user],\"tools\":$tools,\"tool_choice\":{\"type\":\"function\",\"function\":{\"name\":\"no_such_tool\"}}}")
check EXT chat.tool_choice_checked "$c" "$c == 400 and (.error.message | type == \"string\")" \
	"tool_choice naming an absent function -> 400 (else tool_choice is ignored)"

# response_format: a bogus type must be refused if the field is read at all
c=$(req chat.response_format_checked POST /chat/completions \
	"{\"model\":\"$model\",\"messages\":[$user],\"response_format\":{\"type\":\"no_such_format\"}}")
check EXT chat.response_format_checked "$c" "$c == 400" "response_format type no_such_format -> 400 (else field is ignored)"

c=$(req chat.reasoning_effort POST /chat/completions "{\"model\":\"$model\",\"messages\":[$user],\"reasoning_effort\":\"low\"}")
check EXT chat.reasoning_effort "$c" "$c == 200" "reasoning_effort low accepted"
cp "$out/chat.basic.body" "$out/chat.usage_reasoning_tokens.body"
check XAI chat.usage_reasoning_tokens 200 '(.usage.completion_tokens_details.reasoning_tokens | type == "number")' \
	"usage.completion_tokens_details.reasoning_tokens present"
c=$(req chat.n2 POST /chat/completions "{\"model\":\"$model\",\"messages\":[$user],\"n\":2}")
check EXT chat.n2 "$c" "$c == 200 and (.choices | length == 2)" "n=2 returns two choices"
png='iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=='
c=$(req chat.image_part POST /chat/completions \
	"{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":[{\"type\":\"text\",\"text\":\"What is this?\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,$png\"}}]}]}")
check EXT chat.image_part "$c" "$c == 200" "image_url content part accepted"

# a client written for api.x.ai sends grok model ids
c=$(req chat.grok_model_id POST /chat/completions "{\"model\":\"grok-4.7\",\"messages\":[$user]}")
check XAI chat.grok_model_id "$c" "$c == 200" "model grok-4.7 accepted (alias)"

# errors: OpenAI/xAI error object
c=$(req error.empty_messages POST /chat/completions "{\"model\":\"$model\",\"messages\":[]}")
check CORE error.empty_messages "$c" "$c == 400 and (.error.message | type == \"string\")" "messages [] -> 400 {error:{message}}"
c=$(req error.unknown_route GET /no-such-route)
check CORE error.unknown_route "$c" "$c == 404 and (.error.message | type == \"string\")" "unknown route -> 404 {error:{message}}"

# auth: a missing or wrong bearer must be refused
c=$(PROBE_AUTH= req auth.missing POST /chat/completions "{\"model\":\"$model\",\"messages\":[$user]}")
check EXT auth.missing "$c" "$c == 401" "no Authorization -> 401"
c=$(PROBE_AUTH="Bearer wrong-$tok" req auth.wrong POST /chat/completions "{\"model\":\"$model\",\"messages\":[$user]}")
check EXT auth.wrong "$c" "$c == 401" "wrong bearer -> 401"

# CORS preflight, needed by browser clients
c=$(req cors.preflight OPTIONS /chat/completions '' -H "Origin: $origin" \
	-H 'Access-Control-Request-Method: POST' -H 'Access-Control-Request-Headers: authorization,content-type')
if [ "${c:0:1}" = 2 ] && grep -qi '^access-control-allow-origin:' "$out/cors.preflight.hdr"; then
	verdict PASS EXT cors.preflight "$c" "OPTIONS preflight answered"
else verdict FAIL EXT cors.preflight "$c" "no CORS preflight answer"; fi

# xAI Responses API (the primary xAI inference endpoint)
c=$(req responses.basic POST /responses "{\"model\":\"$model\",\"input\":\"Say hello.\"}")
check XAI responses.basic "$c" "$c == 200 and .object == \"response\" and
	any(.output[]?; .type == \"message\" and any(.content[]?; .type == \"output_text\" and (.text | type == \"string\")))" \
	"POST /responses input string"
c=$(req tokenize POST /tokenize-text "{\"model\":\"$model\",\"text\":\"Hello world\"}")
check XAI tokenize "$c" "$c == 200 and (.token_ids[0].token_id | type == \"number\")" "POST /tokenize-text"
c=$(req completions POST /completions "{\"model\":\"$model\",\"prompt\":\"Hello\",\"max_tokens\":4}")
check LEGACY completions "$c" "$c == 200 and .object == \"text_completion\" and (.choices[0].text | type == \"string\")" \
	"POST /completions (legacy)"
c=$(req messages POST /messages "{\"model\":\"$model\",\"max_tokens\":16,\"messages\":[$user]}" -H 'anthropic-version: 2023-06-01')
check LEGACY messages "$c" "$c == 200 and .type == \"message\" and (.content[0].text | type == \"string\")" \
	"POST /messages Anthropic shape (xAI: deprecated)"

printf 'SUMMARY\tbase=%s\tmodel=%s\tpass=%d\tfail=%d\n' "$base" "$model" "$passes" "$fails"
[ "$fails" -eq 0 ]
