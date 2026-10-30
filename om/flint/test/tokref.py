#!/usr/bin/env python3
"""Flint tokenizer and control-token fixture. TEST REFEREE ONLY, never an implementation dependency.

Loads the pinned tokenizer of RecursiveSI/flint@2a247be0 with the pinned reference runtime
(transformers a672eb5f AutoTokenizer, Rust tokenizers backend) and GenerationConfig, and writes the
text fixture that om/flint/tok.om checks against the tokenizer files on its own. check.sh reruns this
script and requires its output to equal the committed test/tok-fixture.txt byte for byte, so a
different transformers file, tokenizers build or source file shows up as a FAIL.

Usage: tokref.py SRC OUT
  SRC  the pinned source capture (tokenizer.json, tokenizer_config.json, chat_template.jinja,
       generation_config.json)
  OUT  the fixture file to write

Fixture: ASCII lines, fields separated by one space; lines starting with '#' are provenance only.
  size NV NA                  base vocabulary size and number of added tokens
  role NAME ID...             bos/eos as the tokenizer resolves them; genbos/geneos/genpad from
                              GenerationConfig (geneos is the ordered stop list)
  ctl NAME ID SPECIAL HEX     a control token: id, its added_tokens_decoder special flag, content bytes
  case NAME N ID*N NB HEX     ids the reference encodes for the NB bytes HEX (no special tokens added);
                              the reference must decode them back to the same bytes
"""
import hashlib
import inspect
import os
import sys

import tokenizers
import transformers
from transformers import AutoTokenizer, GenerationConfig

PINS = {
    "tokenizer.json": "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68",
    "tokenizer_config.json": "159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf",
    "chat_template.jinja": "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466",
    "generation_config.json": "a499c71895e008abd572558a5ad5904feaed92cf2a3b6e2cb0e9cabf03b96ed3",
}
BACKEND_SHA = "9ca05b9e41dbd73b53e2c6521eb28194f1f3ae55c153b9f3e8fd6b0ef450e1ea"  # a672eb5f

CONTROL = [
    ("begin-text", "<|begin_of_text|>"),
    ("end-text", "<|end_of_text|>"),
    ("system", "<|system|>"),
    ("user", "<|user|>"),
    ("assistant", "<|assistant|>"),
    ("observation", "<|observation|>"),
    ("developer", "<|developer|>"),
    ("think-open", "<think>"),
    ("think-close", "</think>"),
    ("tools-open", "<tools>"),
    ("tools-close", "</tools>"),
    ("call-open", "<tool_call>"),
    ("call-close", "</tool_call>"),
    ("argkey-open", "<arg_key>"),
    ("argkey-close", "</arg_key>"),
    ("argval-open", "<arg_value>"),
    ("argval-close", "</arg_value>"),
    ("resp-open", "<tool_response>"),
    ("resp-close", "</tool_response>"),
]


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


src, out = sys.argv[1:3]
for name, pin in PINS.items():
    got = sha(os.path.join(src, name))
    if got != pin:
        sys.exit("tokref: %s sha256 %s, pinned %s" % (name, got, pin))
tok = AutoTokenizer.from_pretrained(src)
gen = GenerationConfig.from_pretrained(src)
backend = inspect.getfile(type(tok))
if sha(backend) != BACKEND_SHA:
    sys.exit("tokref: %s sha256 %s, pinned %s" % (backend, sha(backend), BACKEND_SHA))
so = [f for f in os.listdir(os.path.dirname(tokenizers.__file__)) if f.endswith(".so")]
if len(so) != 1:
    sys.exit("tokref: expected one tokenizers extension, found %r" % so)
so_sha = sha(os.path.join(os.path.dirname(tokenizers.__file__), so[0]))

chat = [
    {"role": "system", "content": "You are Flint."},
    {"role": "user", "content": "What is 17*23?"},
    {"role": "assistant", "reasoning_content": "17*23 = 391, since 17*20 = 340 and 17*3 = 51.",
     "content": "391"},
]
tools = [{"type": "function", "function": {
    "name": "get_weather", "description": "Current weather for a city.",
    "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                   "required": ["city"]}}}]
chat_tool = [
    {"role": "user", "content": "Weather in Paris?"},
    {"role": "assistant", "reasoning_content": "Call the tool.", "content": "",
     "tool_calls": [{"type": "function", "function": {
         "name": "get_weather", "arguments": {"city": "Paris"}}}]},
    {"role": "tool", "content": "{\"temp_c\": 18}"},
]
CASES = [
    ("ascii", "Hello, world! The quick brown fox jumps over 13 lazy dogs; x=42."),
    ("utf8", "na\u00efve caf\u00e9 \u2014 \u65e5\u672c\u8a9e \U0001F642 \u041f\u0440\u0438\u0432\u0435\u0442, "
             "\u03ba\u1f79\u03c3\u03bc\u03b5! \u00bd \u2211 \U0001D518"),
    ("ws", "tab\tline\nCRLF\r\n  two  spaces\u00a0nbsp \x00nul\x7fdel\x1b[0m"),
    ("chat", tok.apply_chat_template(chat, tokenize=False)),
    ("chat-tool", tok.apply_chat_template(chat_tool, tools=tools, tokenize=False,
                                          add_generation_prompt=True)),
]

lines = [
    "# flint tokenizer fixture v1, written by om/flint/test/tokref.py",
    "# source RecursiveSI/flint@2a247be0c0aec895ccb0ac7ca92229ec6df37288",
    "# transformers %s backend %s sha256 %s" % (transformers.__version__,
                                                os.path.basename(backend), BACKEND_SHA),
    "# tokenizers %s %s sha256 %s" % (tokenizers.__version__, so[0], so_sha),
    "# tokenizer class %s" % type(tok).__name__,
]
for name, pin in PINS.items():
    lines.append("# %s sha256 %s" % (name, pin))

nv = tok.backend_tokenizer.get_vocab_size(with_added_tokens=False)
lines.append("size %d %d" % (nv, len(tok) - nv))


def ids(v):
    v = v if isinstance(v, list) else [v]
    if not v or not all(isinstance(i, int) and i >= 0 for i in v):
        sys.exit("tokref: not a token id list: %r" % (v,))
    return " ".join(str(i) for i in v)


lines.append("role bos " + ids(tok.bos_token_id))
lines.append("role eos " + ids(tok.eos_token_id))
lines.append("role genbos " + ids(gen.bos_token_id))
lines.append("role geneos " + ids(gen.eos_token_id))
lines.append("role genpad " + ids(gen.pad_token_id))
dec = tok.added_tokens_decoder
for name, content in CONTROL:
    i = tok.convert_tokens_to_ids(content)
    if tok.convert_ids_to_tokens(i) != content or dec[i].content != content:
        sys.exit("tokref: %s does not round-trip through id %d" % (content, i))
    lines.append("ctl %s %d %d %s" % (name, i, int(dec[i].special), content.encode().hex()))
for name, text in CASES:
    v = tok.encode(text, add_special_tokens=False)
    back = tok.decode(v, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if back != text:
        sys.exit("tokref: case %s does not decode back to its text: %r" % (name, back))
    raw = text.encode()
    lines.append("case %s %d %s %d %s" % (name, len(v), " ".join(map(str, v)), len(raw), raw.hex()))
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("tokref: %d control tokens, %d cases, tokenizers %s" % (len(CONTROL), len(CASES),
                                                           tokenizers.__version__))

# Optional: RAWJSON RAWOUT. RAWJSON is the pinned tokenizer.json with one added token's content
# changed by check.sh to hold characters outside the ByteLevel alphabet (space, tab), which no
# pinned token has; the decoder must then emit that content's own UTF-8 bytes.
if len(sys.argv) == 5:
    raw_json, raw_out = sys.argv[3:5]
    rt = tokenizers.Tokenizer.from_file(raw_json)
    content = "<|reserved 150143\t|>"
    if rt.token_to_id(content) != 150143:
        sys.exit("tokref: %s does not hold the derived added token" % raw_json)
    text = "Hello" + content + "world"
    v = rt.encode(text, add_special_tokens=False).ids
    if 150143 not in v or rt.decode(v, skip_special_tokens=False) != text:
        sys.exit("tokref: raw-added case does not round-trip: %r" % (v,))
    raw = text.encode()
    with open(raw_out, "w") as f:
        f.write("# derived tokenizer.json: id 150143 content %s\n" % content.encode().hex())
        f.write("case raw-added %d %s %d %s\n" % (len(v), " ".join(map(str, v)), len(raw), raw.hex()))
elif len(sys.argv) != 3:
    sys.exit("usage: tokref.py SRC OUT [RAWJSON RAWOUT]")
