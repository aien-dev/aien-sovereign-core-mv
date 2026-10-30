#!/usr/bin/env python3
"""Flint tokenizer and control-token fixture. TEST REFEREE ONLY, never an implementation dependency.

Loads the pinned tokenizer of RecursiveSI/flint@2a247be0 with the pinned reference runtime
(transformers a672eb5f AutoTokenizer, Rust tokenizers backend) and GenerationConfig, and writes the
text fixture that om/flint/tok.om checks against the tokenizer files on its own. check.sh reruns this
script and requires its output to equal the committed test/tok-fixture.txt byte for byte, so a
different transformers file, tokenizers build or source file shows up as a FAIL.

Usage: tokref.py SRC OUT [RAWJSON RAWOUT]
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
  enc NAME N ID*N NB HEX      ids the reference encodes for the UTF-8 text HEX (no special tokens
                              added); the hand cases of ENC_HAND, the CASES, then ENC_N strings
                              composed by random.Random(ENC_SEED) from the ENC_* fragment lists and
                              ENC_UN strings by random.Random(ENC_SEED + 1) that add non-ASCII fragments
  pre NAME N END*N NB HEX     for an enc text without added tokens: the byte end of each piece the
                              reference pre_tokenizer cuts HEX into
  ucd NAME NB HEX             the class of every code point, as test/ucdgen.py runs, read from the
                              reference's own regex engine (one Split per class over all code points)
"""
import hashlib
import inspect
import os
import random
import sys

import tokenizers
import transformers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ucdgen  # noqa: E402
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

# Encode corpus. It covers N runs longer than 3, the contractions in mixed case (and near misses),
# punctuation runs, whitespace and newline runs, inside and at the end of the text, control bytes,
# added tokens inside text, and outside ASCII: the CJK Split ranges and their edges, accented and
# decomposed Latin, combining marks (also first, after a space and after a capital, where a1/a2
# backtrack), emoji with variation selectors and ZWJ, flags, non-ASCII N (Arabic-Indic, fullwidth,
# Nl, No), every non-ASCII White_Space kind, Lt, Lm, other scripts, the fold of U+017F in a
# contraction, a Unicode 16 letter and a code point unassigned in Unicode 16.
ENC_SEED = 834
ENC_N = 240
ENC_HAND = [
    ("hello", "Hello"),
    ("digits", "12345678 1234 x99999y 0 007 3.14159 1e100"),
    ("ws-word", "a  b   c\t\td \t e"),
    ("trail-ws", "end   "),
    ("trail-nl", "line\n\n"),
    ("lead-ws", "   lead"),
    ("contract", "it's don't they're we've I'm you'll he'd"),
    ("contract-uc", "IT'S DON'T THEY'RE WE'VE I'M YOU'LL HE'D It'S wE'Ve"),
    ("contract-no", "rock'n'roll o'clock 'tis x' 'S ''s she'sn't"),
    ("punct", "Hi!!! ... ?!?! (a) [b] {c} <d> --e-- ~~f~~ a+b=c; $5 @x #y %z ^w `v` |u| \\t"),
    ("newline", "a\nb\r\nc\r\rd\n\n\ne \n f\t\n\tg"),
    ("punct-nl", "x.\r\n\r\ny!\n z?\n\n"),
    ("added", "x<think>y</think> z<|user|>hi<|assistant|>"),
    ("added-adj", "<tool_call><arg_key>k</arg_key><arg_value>v</arg_value></tool_call>"),
    ("added-part", "<thin <think <|user| <|user|>|> <<think>>"),
    ("ctl-bytes", "a\x00b\x01 c\x1c\x1fd\x7f e\x0bf\x0cg\x1b[0m"),
    ("caps", "HTTPServer iPhone ABCdef aBC XMLHttpRequest A B CD"),
    ("long", "supercalifragilisticexpialidocious " * 3 + "a" * 70 + " " * 40 + "9" * 25),
    ("cjk", "\u65e5\u672c\u8a9e\u306e\u30c6\u30ad\u30b9\u30c8\u3067\u3059\u3002\u4e2d\u6587\u6587\u672c\uff0c\u6d4b\u8bd5\u3002"),
    ("cjk-mix", "Tokyo\u6771\u4eacabc \u6771\u4eac 2024\u5e74 x\u3042y \u3042\u3044\u3046 a\u4e00b"),
    ("cjk-kata", " \u30ab\u30bf\u30ab\u30ca \u30fc\u30c9 a\u30a0b \u30fb"),
    ("cjk-edge", "\u4e00\u9fa5\u9fa6\u9fff\u3400\u3040\u309f\u30a0\u30ff\u3100 \uff76\uff80"),
    ("latin", "Caf\u00e9 d\u00e9j\u00e0 vu, na\u00efve r\u00e9sum\u00e9 \u00c9COLE \u00c9t\u00e9 \u00dfo \u0130stanbul \u0141\u00f3d\u017a"),
    ("decomp", "e\u0301te\u0301 Cafe\u0301 E\u0301COLE A\u0308BC a\u0308b"),
    ("marks", "\u0301abc \u0301x x \u0301 \u0301\u0302\u0303 A\u0301 \u20dd! '\u0301s .\u0301\u0301"),
    ("emoji", "I \u2764\ufe0f \U0001f355 and \U0001f468\u200d\U0001f469\u200d\U0001f467! \U0001f642\U0001f642\U0001f642 \U0001f1ef\U0001f1f5\U0001f1fa\U0001f1f8 \U0001f44d\U0001f3fd"),
    ("digits-u", "\u0663\u0664\u0665\u0666\u0667 \uff11\uff12\uff13\uff14 \u216b \u00bd \u00b2\u00b3 \u0967\u0968\u0969\u096a x\u0663y"),
    ("ws-u", "a\u00a0b\u3000c\u2028d\u0085e\u2029f\u1680g\u2009\u200ah \u3000\u3000"),
    ("ws-u-nl", "x \u00a0\n y\u3000\r\n\u2028z \u00a0"),
    ("not-ws", "a\u200bb\u200ec\ufeffd\u180ee"),
    ("long-s", "it'\u017f IT'S it'\u017ft '\u017f \u017fun"),
    ("lt-lm", "\u01c5emal \u01c8ubljana \u02b0\u02b2a x\u02b0 \u1fbc\u1f88"),
    ("scripts", "\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac \u041f\u0440\u0438\u0432\u0435\u0442, \u043c\u0438\u0440! "
                "\u0645\u0631\u062d\u0628\u0627 \u0628\u0627\u0644\u0639\u0627\u0644\u0645 \u0928\u092e\u0938\u094d\u0924\u0947 "
                "\u0e2a\u0e27\u0e31\u0e2a\u0e14\u0e35 \uc548\ub155\ud558\uc138\uc694 \u05e9\u05dc\u05d5\u05dd"),
    ("u16", "\U00010d50\U00010d70 \U00010940x \U0001e5d0"),
    ("punct-u", "\u00ab\u00bb \u201cq\u201d \u2014\u2013 \u2026 \u20ac5 \u00a9\u00ae \u2211\u221e \u00bf\u00a1"),
]
ENC_WORDS = ["the", "Hello", "WORLD", "x", "I", "a", "don", "CamelCase", "iOS", "HTTPServer",
             "snake_case", "ab", "Zz", "qux", "OK", "tokenizer", "Flint", "naive", "ZZTop"]
ENC_CONTR = ["'s", "'t", "'re", "'ve", "'m", "'ll", "'d", "'S", "'T", "'Re", "'rE", "'VE", "'M",
             "'Ll", "'lL", "'D", "'x", "'", "''", "'r", "'v", "'l"]
ENC_PUNCT = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
ENC_WS = [" ", "  ", "   ", "\t", "\n", "\n\n", "\r\n", " \n", "\t\n ", "\x0b", "\x0c", " \r\n  ",
          "\r", "\n ", "  \t"]
ENC_CTRL = ["\x00", "\x01", "\x08", "\x1b", "\x1c", "\x1f", "\x7f"]
ENC_ADDED = [dec[i].content for i in sorted(dec)]
rng = random.Random(ENC_SEED)


def enc_frag():
    k = rng.randrange(8)
    if k == 0:
        return rng.choice(ENC_WORDS)
    if k == 1:
        return rng.choice(ENC_WORDS) + rng.choice(ENC_CONTR)
    if k == 2:
        return "".join(rng.choice("0123456789") for _ in range(rng.randrange(1, 10)))
    if k == 3:
        return "".join(rng.choice(ENC_PUNCT) for _ in range(rng.randrange(1, 5)))
    if k == 4:
        return rng.choice(ENC_WS)
    if k == 5:
        return rng.choice(ENC_CTRL)
    if k == 6:
        return rng.choice(ENC_ADDED)
    return rng.choice(ENC_WORDS).upper()


enc = list(ENC_HAND) + list(CASES)
for k in range(ENC_N):
    parts = [enc_frag() for _ in range(rng.randrange(1, 13))]
    text = "".join(p + rng.choice(["", "", " "]) for p in parts)
    if rng.randrange(4) == 0:
        text += rng.choice(ENC_WS)
    enc.append(("r%03d" % k, text))
# Non-ASCII fragments for a second random corpus (its own generator, so r000.. stay as they were).
ENC_UN = 160
ENC_UFRAG = ["\u00e9", "\u00c9", "e\u0301", "\u0301", "\u0308", "\u00df", "\u0130", "\u017f", "\u01c5",
             "\u02b0", "\u65e5\u672c", "\u306e", "\u30ab\u30bf", "\u30fc", "\u3000", "\u00a0", "\u2028",
             "\u0085", "\u0663\u0664", "\uff11", "\u00bd", "\u216b", "\u2764\ufe0f", "\U0001f642",
             "\U0001f468\u200d\U0001f469", "\U0001f1ef\U0001f1f5", "\u041f\u0440\u0438", "\u0628\u0627",
             "\u0928\u092e\u0938\u094d", "\uc548\ub155", "\u2014", "\u201c", "\u20ac", "\u200b", "\U00010940"]
urng = random.Random(ENC_SEED + 1)
for k in range(ENC_UN):
    rng = urng
    parts = []
    for _ in range(urng.randrange(1, 13)):
        parts.append(urng.choice(ENC_UFRAG) if urng.randrange(2) == 0 else enc_frag())
    text = "".join(p + urng.choice(["", "", " "]) for p in parts)
    enc.append(("u%03d" % k, text))
# A text without added tokens also gets a pre record: the byte end of every piece the reference
# pre_tokenizer (the three Splits) cuts it into. BPE seldom merges across a piece boundary, so a
# wrong boundary (a mark class, a CJK range, a fold, backtracking) often leaves the ids unchanged;
# the pre record shows it.
npre = 0
for name, text in enc:
    if not 0 < len(text.encode()) <= 16384:
        sys.exit("tokref: enc %s is empty or longer than 16384 bytes" % name)
    v = tok.encode(text, add_special_tokens=False)
    if v != tok.backend_tokenizer.encode(text, add_special_tokens=False).ids:
        sys.exit("tokref: enc %s: transformers and tokenizers disagree" % name)
    if tok.decode(v, skip_special_tokens=False, clean_up_tokenization_spaces=False) != text:
        sys.exit("tokref: enc %s does not decode back to its text" % name)
    raw = text.encode()
    lines.append("enc %s %d %s %d %s" % (name, len(v), " ".join(map(str, v)), len(raw), raw.hex()))
    if any(a in text for a in ENC_ADDED):
        continue
    ends = []
    for _, (a, b) in tok.backend_tokenizer.pre_tokenizer.pre_tokenize_str(text):
        if a != (ends[-1] if ends else 0):
            sys.exit("tokref: pre %s: pieces do not cover the text" % name)
        ends.append(b)
    if ends[-1] != len(text):
        sys.exit("tokref: pre %s: pieces do not cover the text" % name)
    e = [len(text[:b].encode()) for b in ends]
    lines.append("pre %s %d %s %d %s" % (name, len(e), " ".join(map(str, e)), len(raw), raw.hex()))
    npre += 1

# Code point classes of the reference regex engine, as ucdgen.py runs: each class is one Split
# (behavior removed) over every code point but the surrogates (left in class a, as in the UCD
# build), whose removed spans are its members. The classes must not overlap. The (?i) contraction
# letters must fold as tok.om's te_low assumes: s also to U+017F, the others within ASCII.
CPS = [c for c in range(ucdgen.NCP) if not 0xD800 <= c <= 0xDFFF]
ALL = "".join(map(chr, CPS))


def members(rx):
    kept = bytearray(len(CPS))
    for _, (a, b) in tokenizers.pre_tokenizers.Split(tokenizers.Regex(rx), "removed").pre_tokenize_str(ALL):
        kept[a:b] = b"\x01" * (b - a)
    return [CPS[i] for i in range(len(CPS)) if not kept[i]]


cls = bytearray(b"a" * ucdgen.NCP)
for letter, rx, _ in ucdgen.CLASSES:
    for c in members(rx):
        if cls[c] != ord("a"):
            sys.exit("tokref: U+%04X is in %s and class %c" % (c, rx, cls[c]))
        cls[c] = ord(letter)
for x in "strvmld":
    want = {ord(x), ord(x.upper())} | ({0x17F} if x == "s" else set())
    if set(members("(?i:%s)" % x)) != want:
        sys.exit("tokref: (?i:%s) folds to %r" % (x, sorted(members("(?i:%s)" % x))))
u = ucdgen.runs(cls).encode()
lines.append("ucd unicode-%s %d %s" % (ucdgen.UCD_VERSION, len(u), u.hex()))
with open(out, "w") as f:
    f.write("\n".join(lines) + "\n")
print("tokref: %d control tokens, %d cases, %d enc (seed %d), %d pre, ucd %d runs, tokenizers %s" % (
    len(CONTROL), len(CASES), len(enc), ENC_SEED, npre,
    len(u) - len(u.translate(None, b"abcdefghijk")), tokenizers.__version__))

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
