#!/usr/bin/env python3
"""Read-only pinned Flint referee for ops.om generate (Core #834 item 5), not a runtime dependency.

Usage: genref.py SRC SHARDS OUTPUT
Renders PROMPT with the pinned chat template (add_generation_prompt), then greedy-generates N
tokens with the pinned transformers modules in f32 on CPU: one prefill over the prompt, then one
position per step through a DynamicCache, argmax of the last row's logits. Layers are built
layer by layer from the BF16 shards as in fwdref.py; routed experts load lazily.
Fixture (little-endian u32 words, f32 logits): header magic, op 9, 0, T (prompt length), N, 0,
21400 (torch 2.14.0), 0; then T prompt IDs; then per step: top-1 ID, top-1 logit, top-2 ID,
top-2 logit. Ambiguous steps follow test/TOLERANCE.md (greedy generation parity), printed here
before any .om run.
"""
import hashlib
import json
import os
import struct
import sys

import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
from transformers.models.flint import configuration_flint as cf
from transformers.models.flint import modeling_flint as mf

REF_SHA = "f0540832049f0d67eee9eadd2d8ed42e320388b6b3330e74c8821a5c4eac0c09"
CFG_SHA = "849eee3068846081077a7568663355802c8c4eec3741d92cd578d222bd144d00"
TOK_SHA = {  # PROFILE.md
    "tokenizer.json": "a263cce0b7e2b03f7ee07ce809783ef4b47f160c3f268a4ea1524f638da45b68",
    "tokenizer_config.json": "159ac88217e90ffd0a4a294726048bb302829e82e1958dc85bb203aa79138fbf",
    "chat_template.jinja": "7940f4ffa3275c36caa4a582f845aaad79da11018b3961e8fbcad6ae29733466",
}
PROMPT = "What is 2+2?"
PROMPT_IDS = [150003, 2387, 332, 220, 17, 10, 17, 30, 150004]
N = 16
for path, pin in ((mf.__file__, REF_SHA), (cf.__file__, CFG_SHA)):
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != pin:
        sys.exit("genref: reference source pin mismatch: " + path)
if torch.__version__ != "2.14.0+cu130":
    sys.exit("genref: torch pin mismatch: " + torch.__version__)
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.use_deterministic_algorithms(True)
src, shards, output = sys.argv[1:4]
for name, pin in TOK_SHA.items():
    if hashlib.sha256(open(os.path.join(src, name), "rb").read()).hexdigest() != pin:
        sys.exit("genref: tokenizer pin mismatch: " + name)
tok = AutoTokenizer.from_pretrained(src)
ids = tok.apply_chat_template([{"role": "user", "content": PROMPT}], tokenize=True, add_generation_prompt=True)
ids = list(ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids)
if ids != PROMPT_IDS:
    sys.exit("genref: rendered prompt IDs %s differ from the frozen %s" % (ids, PROMPT_IDS))
cfg = cf.FlintConfig(**json.load(open(os.path.join(src, "config.json"))))
cfg._attn_implementation = "eager"
cfg._experts_implementation = "eager"
wmap = json.load(open(os.path.join(src, "model.safetensors.index.json")))["weight_map"]
handles = {}


def shard(name):
    f = wmap[name]
    if f not in handles:
        handles[f] = safe_open(os.path.join(shards, f), framework="pt")
    return handles[f]


def tensor(name):
    t = shard(name).get_tensor(name)
    assert t.dtype == torch.bfloat16, (name, t.dtype)
    return t.float()


def load(module, prefix, rename=None):
    state = {k: tensor(prefix + (rename or {}).get(k, k)) for k in module.state_dict()}
    module.load_state_dict(state, strict=True)
    return module.float().eval()


def norm(prefix):
    return load(mf.FlintRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps), prefix)


class Lazy:
    def __init__(self, prefix, part):
        self.prefix, self.part = prefix, part

    def __getitem__(self, e):
        p = self.prefix + "experts.%d." % int(e)
        if self.part == "gate_up":
            return torch.cat([tensor(p + "gate_proj.weight"), tensor(p + "up_proj.weight")], dim=0)
        return tensor(p + "down_proj.weight")


def sparse(prefix):
    c = cf.FlintConfig(**cfg.to_dict())
    c._experts_implementation = "eager"
    c.num_experts = 1
    ex = mf.FlintExperts(c).float()
    del ex.gate_up_proj, ex.down_proj
    ex.gate_up_proj, ex.down_proj = Lazy(prefix, "gate_up"), Lazy(prefix, "down")
    ex.num_experts = cfg.num_experts
    blk = mf.FlintSparseMoeBlock.__new__(mf.FlintSparseMoeBlock)
    torch.nn.Module.__init__(blk)
    blk.gate = load(mf.FlintTopKRouter(cfg), prefix + "gate.")
    blk.experts = ex
    blk.shared_expert = load(mf.FlintMLP(cfg, cfg.shared_expert_intermediate_size), prefix + "shared_expert.")
    return blk


def layer(L):
    p = "model.layers.%d." % L
    m = mf.FlintDecoderLayer.__new__(mf.FlintDecoderLayer)
    torch.nn.Module.__init__(m)
    m.self_attn = load(mf.FlintAttention(cfg, L), p + "self_attn.", {"sinks": "softmax_offset"})
    m.input_layernorm = norm(p + "input_layernorm.")
    m.post_attention_layernorm = norm(p + "post_attention_layernorm.")
    m.mlp = load(mf.FlintMLP(cfg), p + "mlp.") if L < 2 else sparse(p + "mlp.")
    return m.eval()


layers = [layer(L) for L in range(cfg.num_hidden_layers)]
fnorm = norm("model.norm.")
rot = mf.FlintRotaryEmbedding(cfg)
esl = shard("model.embed_tokens.weight").get_slice("model.embed_tokens.weight")
lsl = shard("lm_head.weight").get_slice("lm_head.weight")
cache = DynamicCache()
g = 2048 * 2.0 ** -24 / (1 - 2048 * 2.0 ** -24)


def tau(x):
    return 8 * g * (1 + abs(x))


def step(tokens, p0):
    """Run positions p0 .. p0+len(tokens) through all layers; return the f32 logits of the last row."""
    x = torch.cat([esl[t:t + 1] for t in tokens]).float().unsqueeze(0)
    pos = torch.arange(p0, p0 + len(tokens)).unsqueeze(0)
    rope = rot(x, pos)
    n = len(tokens)
    mask = None
    if n > 1:
        mask = torch.triu(torch.full((n, n), torch.finfo(torch.float32).min), diagonal=1)[None, None]
    for m in layers:
        x = m(x, attention_mask=mask, position_ids=pos, past_key_values=cache, use_cache=True, position_embeddings=rope)
    h = fnorm(x)[0, -1:]
    # Stream the vocabulary in 2048-row chunks as Omega does.
    return torch.cat([torch.nn.functional.linear(h, lsl[f:f + 2048].float())[0]
                      for f in range(0, cfg.vocab_size, 2048)])


rows = []
logits = step(ids, 0)
for k in range(N):
    v, i = torch.topk(logits, 2)
    l1, l2 = float(v[0]), float(v[1])
    a1, a2 = int(i[0]), int(i[1])
    if a1 != int(torch.argmax(logits)):
        sys.exit("genref: topk and argmax disagree")
    amb = l1 - l2 <= tau(l1) + tau(l2)
    rows.append((a1, l1, a2, l2))
    print("genref step %d top1 %d %.6f top2 %d %.6f margin %.6f%s" % (
        k + 1, a1, l1, a2, l2, l1 - l2, " AMBIGUOUS" if amb else ""), flush=True)
    if k + 1 < N:
        logits = step([a1], len(ids) + k)
gen = [r[0] for r in rows]
with open(output, "wb") as f:
    f.write(struct.pack("<8I", 0x504F4C46, 9, 0, len(ids), N, 0, 21400, 0))
    f.write(struct.pack("<%dI" % len(ids), *ids))
    for a1, l1, a2, l2 in rows:
        f.write(struct.pack("<IfIf", a1, l1, a2, l2))
amb = [k + 1 for k, (a1, l1, a2, l2) in enumerate(rows) if l1 - l2 <= tau(l1) + tau(l2)]
print("genref prompt", ids, "generated", gen, json.dumps(tok.decode(gen)), flush=True)
print("genref ambiguous steps", amb if amb else "none", flush=True)
print("genref complete", hashlib.sha256(open(output, "rb").read()).hexdigest(), flush=True)
