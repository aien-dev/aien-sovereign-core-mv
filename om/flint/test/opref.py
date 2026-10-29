#!/usr/bin/env python3
"""Flint forward-operator fixtures. TEST REFEREE ONLY, never an implementation dependency.

Runs the pinned reference's own modules (transformers models/flint/modeling_flint.py and
configuration_flint.py, the files whose sha256 are REF_SHA and CFG_SHA below) in float32 on small
deterministic inputs drawn from the real BF16 weights of named layers, and writes one binary fixture
per operator for om/flint/ops.om to recompute. Before the q/k/v and attention fixtures it measures
the reference's RoPE powf, cosf and sinf against float64 and exits if any call is off by more than
ops.om's FULP ulps, the libm budget the RoPE bound assumes.

Usage: opref.py SRC SHARDS OUT
  SRC     the pinned source tree (config.json, model.safetensors.index.json)
  SHARDS  the 15 downloaded shards
  OUT     directory for the fixtures (created)

Fixture layout, all little-endian: 8 u32 header words, magic 0x504F4C46, op, layer, T, S, aux,
torch version (major * 10000 + minor * 100 + patch, so a torch change moves the fixtures sha256 that
check.sh pins), 0, then the op's arrays (u32 index arrays, f32 everything else) in the order ops.om
reads them:
  1 rmsnorm  T rows, aux 0 input_layernorm / 1 post_attention_layernorm: x[T,2048] y[T,2048]
  2 qkv      S rows: pos[S] x[S,2048] q[S,32,128] k[S,4,128] v[S,4,128] (after q/k norm and RoPE)
  3 attn     S query rows of T keys: pos[S] q[S,4096] k[T,512] v[T,512] o[S,4096] y[S,2048]
  4 router   T rows: x[T,2048] logits[T,256] idx[T,6] w[T,6]
  5 shared   T rows: x[T,2048] y[T,2048]
  6 expert   T rows, aux expert id: x[T,2048] y[T,2048] (routing weight 1)
  7 moe      T rows: x[T,2048] y[T,2048] (FlintSparseMoeBlock output)
Inputs are embedding rows of deterministic token ids passed through the named layer's real RMSNorm.
"""
import hashlib
import json
import os
import re
import struct
import sys

import torch
from safetensors import safe_open

from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask
from transformers.models.flint import modeling_flint as mf
from transformers.models.flint import configuration_flint as cf
from transformers.models.flint.configuration_flint import FlintConfig

REF_SHA = "f0540832049f0d67eee9eadd2d8ed42e320388b6b3330e74c8821a5c4eac0c09"  # transformers a672eb5f
CFG_SHA = "849eee3068846081077a7568663355802c8c4eec3741d92cd578d222bd144d00"  # transformers a672eb5f
MAGIC = 0x504F4C46
TORCH = [int(v) for v in torch.__version__.split("+")[0].split(".")[:3]]
TORCH_WORD = TORCH[0] * 10000 + TORCH[1] * 100 + TORCH[2]
torch.set_num_threads(1)
torch.use_deterministic_algorithms(True)
torch.set_grad_enabled(False)

src, shards, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for ref_file, pin in ((mf.__file__, REF_SHA), (cf.__file__, CFG_SHA)):
    got = hashlib.sha256(open(ref_file, "rb").read()).hexdigest()
    if got != pin:
        sys.exit("opref: %s sha256 %s, pinned %s" % (ref_file, got, pin))

cfg = FlintConfig(**json.load(open(os.path.join(src, "config.json"))))
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
    state = {}
    for k in module.state_dict():
        ck = (rename or {}).get(k, k)
        state[k] = tensor(prefix + ck)
    module.load_state_dict(state, strict=True)
    return module.float().eval()


def embed_rows(ids):
    name = "model.embed_tokens.weight"
    sl = shard(name).get_slice(name)
    return torch.cat([sl[i:i + 1] for i in ids]).float()


def ids(n, salt):
    return [1000 + (salt + 7919 * i) % 90000 for i in range(n)]


def norm(layer, kind, x):
    m = mf.FlintRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    load(m, "model.layers.%d.%s." % (layer, kind))
    return m(x)


def f32(a):
    a = a.detach().to(torch.float32).contiguous().reshape(-1)
    return a.numpy().astype("<f4").tobytes()


def u32(a):
    return struct.pack("<%dI" % len(a), *[int(v) for v in a])


def write(fname, op, layer, t, s, aux, *parts):
    with open(os.path.join(out, fname), "wb") as f:
        f.write(struct.pack("<8I", MAGIC, op, layer, t, s, aux, TORCH_WORD, 0))
        for p in parts:
            f.write(p)


# 1 RMSNorm, layer 2 input_layernorm, on raw embedding rows
x = embed_rows(ids(16, 1))
write("rmsnorm.bin", 1, 2, 16, 0, 0, f32(x), f32(norm(2, "input_layernorm", x)))

# 2/3 attention: layer 2 (sliding window 513, rotary) and layer 5 (full attention, no rotary)
T = 520
SEL = [0, 1, 2, 255, 511, 512, 513, 519]
pos_ids = torch.arange(T).unsqueeze(0)
cap = {}
orig_eager = mf.eager_attention_forward


def capture(module, query, key, value, attention_mask, scaling, **kw):
    o, w = orig_eager(module, query, key, value, attention_mask, scaling, **kw)
    cap.update(q=query, k=key, v=value, mask=attention_mask, o=o, scaling=scaling)
    return o, w


mf.eager_attention_forward = capture
rope = mf.FlintRotaryEmbedding(cfg)

# ops.om bounds RoPE for FULP ulps per powf, cosf and sinf call; measure the reference's own calls
# against float64 and refuse to write fixtures if any exceeds that budget.
FULP = int(re.search(r"^const FULP ulong (\d+)$",
                     open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ops.om")).read(),
                     re.M).group(1))


def ulps(got, exact):
    a = got.abs()
    ulp = (torch.nextafter(a, torch.full_like(a, float("inf"))) - a).double()
    return ((got.double() - exact).abs() / ulp).max().item()


theta = cfg.rope_parameters["rope_theta"]
expo = torch.arange(0, cfg.head_dim, 2, dtype=torch.float) / cfg.head_dim
pw = theta ** expo
assert torch.equal(1.0 / pw, rope.inv_freq), "inv_freq is 1 / theta ** (2i / d)"
cos, sin = rope(torch.zeros(1, T, 8), pos_ids)
ang = (pos_ids[..., None].float() * rope.inv_freq)[0]
libm = (ulps(pw, float(theta) ** expo.double()), ulps(cos[0, :, :64], ang.double().cos()),
        ulps(sin[0, :, :64], ang.double().sin()))
if max(libm) > FULP:
    sys.exit("opref: RoPE libm ulps pow %.3f cos %.3f sin %.3f exceed FULP %d" % (libm + (FULP,)))

for layer in (2, 5):
    kind = cfg.layer_types[layer]
    assert kind == ("sliding_attention" if layer == 2 else "full_attention"), (layer, kind)
    h = norm(layer, "input_layernorm", embed_rows(ids(T, 100 + layer))).unsqueeze(0)
    mk = {"config": cfg, "inputs_embeds": h, "attention_mask": None, "past_key_values": None,
          "position_ids": pos_ids}
    mask = create_sliding_window_causal_mask(**mk) if layer == 2 else create_causal_mask(**mk)
    allowed = torch.tril(torch.ones(T, T, dtype=torch.bool))
    if layer == 2:
        allowed &= torch.arange(T)[None, :] > torch.arange(T)[:, None] - cfg.sliding_window
    assert mask.shape == (1, 1, T, T) and torch.equal(mask[0, 0] == 0, allowed), "mask semantics"
    att = mf.FlintAttention(cfg, layer)
    load(att, "model.layers.%d.self_attn." % layer, {"sinks": "softmax_offset"})
    cap.clear()
    y, _ = att(h, position_embeddings=rope(h, pos_ids), attention_mask=mask)
    assert cap["scaling"] == cfg.head_dim ** -0.5
    q = cap["q"][0].transpose(0, 1)  # [T, 32, 128]
    k = cap["k"][0].transpose(0, 1)  # [T, 4, 128]
    v = cap["v"][0].transpose(0, 1)
    o = cap["o"][0]                  # [T, 32, 128] after transpose(1, 2)
    sel = torch.tensor(SEL)
    write("qkv-%d.bin" % layer, 2, layer, 0, len(SEL), 0, u32(SEL), f32(h[0][sel]), f32(q[sel]),
          f32(k[sel]), f32(v[sel]))
    write("attn-%d.bin" % layer, 3, layer, T, len(SEL), 0, u32(SEL), f32(q[sel]), f32(k), f32(v),
          f32(o[sel]), f32(y[0][sel]))
mf.eager_attention_forward = orig_eager

# 4 router, layer 24
L = 24
pre = "model.layers.%d.mlp." % L
x = norm(L, "post_attention_layernorm", embed_rows(ids(64, 300)))
router = load(mf.FlintTopKRouter(cfg), pre + "gate.")
logits, w, idx = router(x)
write("router.bin", 4, L, 64, 0, 0, f32(x), f32(logits), u32(idx.reshape(-1).tolist()), f32(w))

# comparator control: token 0 expects its rank-7 expert in place of its rank-6 one (clear margin)
top7 = torch.topk(logits[0].sigmoid(), 7).indices
bad = idx.clone()
bad[0, 5] = top7[6]
write("router-swap.bin", 4, L, 64, 0, 0, f32(x), f32(logits), u32(bad.reshape(-1).tolist()), f32(w))

# near tie: move token 0 along W[b] - W[a] until its rank-6 (a) and rank-7 (b) logits agree
W = router.weight.double()
x0 = x[0].double()
lg = W @ x0
order = torch.argsort(lg, descending=True)
a, b = order[5], order[6]
d = W[b] - W[a]
xt = (x0 + (lg[a] - lg[b]) / (d @ d) * d).float().unsqueeze(0)
lt, wt, it = router(xt)
write("router-tie.bin", 4, L, 1, 0, 0, f32(xt), f32(lt), u32(it.reshape(-1).tolist()), f32(wt))

# 5 shared expert, layer 24 (FlintMLP at shared_expert_intermediate_size)
x4 = x[:4]
sh = load(mf.FlintMLP(cfg, intermediate_size=cfg.shared_expert_intermediate_size), pre + "shared_expert.")
write("shared.bin", 5, L, 4, 0, 0, f32(x4), f32(sh(x4)))


class Lazy:
    """Expert-indexed view of the checkpoint: experts[e] fuses gate then up, like the conversion map."""

    def __init__(self, part):
        self.part = part

    def __getitem__(self, e):
        q = pre + "experts.%d." % int(e)
        if self.part == "gate_up":
            return torch.cat([tensor(q + "gate_proj.weight"), tensor(q + "up_proj.weight")], dim=0)
        return tensor(q + "down_proj.weight")


def experts():
    c = FlintConfig(**cfg.to_dict())
    c._experts_implementation = "eager"
    c.num_experts = 1  # placeholder parameters; the forward indexes the Lazy views below
    ex = mf.FlintExperts(c).float()
    del ex.gate_up_proj, ex.down_proj
    ex.gate_up_proj, ex.down_proj = Lazy("gate_up"), Lazy("down")
    ex.num_experts = cfg.num_experts
    return ex


# 6 one routed expert MLP: the expert token 0 ranks first, weight 1
e = int(idx[0, 0])
ones = torch.ones(4, 1)
ye = experts()(x4, torch.full((4, 1), e, dtype=torch.long), ones)
write("expert.bin", 6, L, 4, 0, e, f32(x4), f32(ye))

# 7 the whole sparse MoE block: router, six routed experts, shared expert added without a gate
blk = mf.FlintSparseMoeBlock.__new__(mf.FlintSparseMoeBlock)
torch.nn.Module.__init__(blk)
blk.gate, blk.experts, blk.shared_expert = router, experts(), sh
write("moe.bin", 7, L, 4, 0, 0, f32(x4), f32(blk(x4.unsqueeze(0))[0]))
print("opref ok ref %s cfg %s torch %s expert %d tie %d/%d rope ulps pow %.3f cos %.3f sin %.3f of %d"
      % ((REF_SHA[:16], CFG_SHA[:16], torch.__version__, e, int(a), int(b)) + libm + (FULP,)))
