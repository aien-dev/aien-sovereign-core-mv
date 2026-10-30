#!/usr/bin/env python3
"""Read-only pinned Flint referee for ops.om forward, not a runtime dependency.

Usage: fwdref.py SRC SHARDS OUTPUT [bf16]
Header matches opref.py, op 8 (f32 arm) or 9 (bf16 arm), T=2, then two token
IDs, embeddings, for each of 49 layers: input norm, attention output,
post-attention norm, MLP output, residual output. Finally final norm and all
vocabulary logits. All tensors are written as contiguous little-endian f32.
Weights stay BF16 on disk. No sampling or generation is claimed.

f32 arm: every module is upcast to float32, so the forward runs in f32 on CPU.

bf16 arm: the precision the checkpoint ships with (config.json torch_dtype
bfloat16). Every parameter is bfloat16 except the model's
_keep_in_fp32_modules_strict ("gate.weight", "sinks"), which stay float32;
activations are bfloat16. The pinned modules then round to bfloat16
(round to nearest even, from an f32 intermediate) at exactly these points
(test/TOLERANCE.md, bf16 forward arm): every nn.Linear output (q/k/v/o_proj,
gate/up/down_proj of the dense MLP, shared and routed experts, lm_head
logits); every FlintRMSNorm output (input, post-attention, q_norm, k_norm,
final); RoPE cos and sin, q*cos, rotate_half(q)*sin and their sum (same for
k); the softmax probabilities before P V and the P V output; silu(gate) and
silu(gate)*up; each routed expert output times its f32 routing weight; each
index_add_ into the bf16 accumulator (experts in ascending index order); the
routed sum plus the shared expert; both residual adds. Router logits,
sigmoid, top-k and weights, attention scores and softmax stay float32.
Every written bf16-arm value is therefore a bfloat16 (low 16 bits zero).
"""
import hashlib
import json
import os
import struct
import sys

import torch
from safetensors import safe_open
from transformers.models.flint import configuration_flint as cf
from transformers.models.flint import modeling_flint as mf

REF_SHA = "f0540832049f0d67eee9eadd2d8ed42e320388b6b3330e74c8821a5c4eac0c09"
CFG_SHA = "849eee3068846081077a7568663355802c8c4eec3741d92cd578d222bd144d00"
for path, pin in ((mf.__file__, REF_SHA), (cf.__file__, CFG_SHA)):
    if hashlib.sha256(open(path, "rb").read()).hexdigest() != pin:
        sys.exit("fwdref: reference source pin mismatch: " + path)
if mf.FlintPreTrainedModel._keep_in_fp32_modules_strict != ["gate.weight", "sinks"]:
    sys.exit("fwdref: fp32 module list changed")
if torch.__version__ != "2.14.0+cu130":
    sys.exit("fwdref: torch pin mismatch: " + torch.__version__)
torch.set_num_threads(1)
torch.set_grad_enabled(False)
torch.use_deterministic_algorithms(True)
src, shards, output = sys.argv[1:4]
BF16 = sys.argv[4:] == ["bf16"]
if not BF16 and sys.argv[4:]:
    sys.exit("fwdref: usage: fwdref.py SRC SHARDS OUTPUT [bf16]")
DT = torch.bfloat16 if BF16 else torch.float32
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
    return t.to(DT)


def load(module, prefix, rename=None):
    state = {k: tensor(prefix + (rename or {}).get(k, k)) for k in module.state_dict()}
    module.load_state_dict(state, strict=True)
    module.to(DT)
    # The shipped bf16 model keeps mf.FlintPreTrainedModel._keep_in_fp32_modules_strict in f32.
    for k, v in module.named_parameters():
        if isinstance(module, mf.FlintTopKRouter) and k == "weight" or k == "sinks":
            v.data = v.data.float()
    return module.eval()


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
    ex = mf.FlintExperts(c).to(DT)
    del ex.gate_up_proj, ex.down_proj
    ex.gate_up_proj, ex.down_proj = Lazy(prefix, "gate_up"), Lazy(prefix, "down")
    ex.num_experts = cfg.num_experts
    blk = mf.FlintSparseMoeBlock.__new__(mf.FlintSparseMoeBlock)
    torch.nn.Module.__init__(blk)
    blk.gate = load(mf.FlintTopKRouter(cfg), prefix + "gate.")
    blk.experts = ex
    blk.shared_expert = load(mf.FlintMLP(cfg, cfg.shared_expert_intermediate_size), prefix + "shared_expert.")
    return blk


def write(f, x):
    f.write(x.detach().float().contiguous().numpy().astype("<f4").tobytes())


ids = [1001, 8920]
sl = shard("model.embed_tokens.weight").get_slice("model.embed_tokens.weight")
x = torch.cat([sl[i:i + 1] for i in ids]).to(DT).unsqueeze(0)
pos = torch.arange(len(ids)).unsqueeze(0)
rope = mf.FlintRotaryEmbedding(cfg)(x, pos)
mask = torch.triu(torch.full((len(ids), len(ids)), torch.finfo(DT).min), diagonal=1)[None, None]
with open(output, "wb") as f:
    f.write(struct.pack("<8I", 0x504F4C46, 9 if BF16 else 8, 0, len(ids), 0, 0, 21400, 0))
    f.write(struct.pack("<2I", *ids))
    write(f, x)
    for L in range(cfg.num_hidden_layers):
        p = "model.layers.%d." % L
        layer = mf.FlintDecoderLayer.__new__(mf.FlintDecoderLayer)
        torch.nn.Module.__init__(layer)
        layer.self_attn = load(mf.FlintAttention(cfg, L), p + "self_attn.", {"sinks": "softmax_offset"})
        layer.input_layernorm = norm(p + "input_layernorm.")
        layer.post_attention_layernorm = norm(p + "post_attention_layernorm.")
        layer.mlp = load(mf.FlintMLP(cfg), p + "mlp.") if L < 2 else sparse(p + "mlp.")
        layer.eval()
        cap = {}

        def capture(name):
            def hook(module, args, out):
                cap[name] = out[0] if isinstance(out, tuple) else out
            return hook

        for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            getattr(layer, name).register_forward_hook(capture(name))
        x = layer(x, attention_mask=mask, position_ids=pos, position_embeddings=rope, use_cache=False)
        for name in ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp"):
            write(f, cap[name])
        write(f, x)
        print("fwdref layer", L, flush=True)
        del layer, cap
    x = norm("model.norm.")(x)
    write(f, x)
    name = "lm_head.weight"
    sl = shard(name).get_slice(name)
    # Stream the vocabulary in the same row chunks as Omega, never allocate all weights at once.
    for t in range(len(ids)):
        for first in range(0, cfg.vocab_size, 2048):
            w = sl[first:first + 2048].to(DT)
            write(f, torch.nn.functional.linear(x[0, t:t + 1], w))
print("fwdref complete", hashlib.sha256(open(output, "rb").read()).hexdigest(), flush=True)
