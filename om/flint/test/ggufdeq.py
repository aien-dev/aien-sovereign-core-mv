#!/usr/bin/env python3
"""Decode referee for `flint-gguf deq` (Core #834 item 7).

Usage: ggufdeq.py GGUF TSV

Recomputes the second line of `flint-gguf deq`,

    deq units U blocks B elements N decode_digest H

independently of gguf.om: names from its own map of GGUF to Hugging Face names, the TSV row index
of each unit (data lines, 0-based), the seeded sample (8 blocks per unit, LCG seed 834 continued
across units, BF16 units included), and each sampled NVFP4 block decoded by llama.cpp b11804's
gguf-py (`gguf.quants.dequantize`, the port of ggml's `dequantize_row_nvfp4`) times the tensor's
F32 `.scale` in f64. H is the digest gguf.om computes: h = 1469598103934665603, then for every
word v, h = (h xor v) * 1099511628211 mod 2^64 followed by h = h xor (h >> 29), over each sampled
block's TSV row index, sample row,
sample block and its 64 values as f64 bits (sub-block s, j < 8: value 16s+j, then 16s+j+8).
The gguf-py files used are checked against their sha256 at commit c7fc5094 first.
"""
import hashlib
import os
import struct
import sys

PIN = {
    "gguf_reader.py": "d0ea743200e19d7ef0a4c969edcc4a6dba8e3656a790b217e3bdc7b31170718e",
    "constants.py": "8d82aa2afea617f2ce1c9775fdad16dbb5a1dfa51261250d0d8918a78557bb3f",
    "quants.py": "5c30d83fda855283a3239e479a26f35ea002ddd7650fcc4d255e6210fc9e72f1",
}
M64 = (1 << 64) - 1
LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "attn_sinks.weight": "self_attn.softmax_offset",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_gate_inp.weight": "mlp.gate.weight",
    "ffn_gate_shexp.weight": "mlp.shared_expert.gate_proj.weight",
    "ffn_up_shexp.weight": "mlp.shared_expert.up_proj.weight",
    "ffn_down_shexp.weight": "mlp.shared_expert.down_proj.weight",
}
EXPS = {"ffn_gate_exps.weight": "gate_proj", "ffn_up_exps.weight": "up_proj", "ffn_down_exps.weight": "down_proj"}
TOP = {"token_embd.weight": "model.embed_tokens.weight", "output_norm.weight": "model.norm.weight",
       "output.weight": "lm_head.weight"}


def pinned():
    import gguf
    d = os.path.dirname(gguf.__file__)
    for f, want in PIN.items():
        got = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
        if got != want:
            sys.exit("ggufdeq: %s/%s sha256 %s is not the b11804 file" % (d, f, got))
    return gguf


def hf_names(name, ne2):
    if name in TOP:
        return [TOP[name]]
    _, layer, rest = name.split(".", 2)
    if rest in EXPS:
        return ["model.layers.%s.mlp.experts.%d.%s.weight" % (layer, e, EXPS[rest]) for e in range(ne2)]
    return ["model.layers.%s.%s" % (layer, LAYER[rest])]


class Sampler:
    def __init__(self):
        self.x = 834

    def pick(self, k, rows, bpr):
        if k == 0:
            return 0, 0
        if k == 1:
            return rows - 1, bpr - 1
        self.x = (self.x * 6364136223846793005 + 1442695040888963407) & M64
        return (self.x >> 33) % rows, (self.x >> 11) % bpr


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: ggufdeq.py GGUF TSV")
    gguf = pinned()
    import numpy as np
    from gguf.quants import dequantize
    nvfp4 = gguf.GGMLQuantizationType.NVFP4
    row = {}
    for i, ln in enumerate(open(sys.argv[2]).read().splitlines()[1:]):
        row[ln.split("\t", 1)[0]] = i
    reader = gguf.GGUFReader(sys.argv[1])
    byname = {t.name: t for t in reader.tensors}
    smp = Sampler()
    h = 1469598103934665603
    units = blocks = elements = 0
    for t in reader.tensors:
        if t.name.endswith(".scale"):
            continue
        ty = t.tensor_type.value
        ne = [int(v) for v in t.shape]
        names = hf_names(t.name, ne[2] if len(ne) == 3 else 1)
        units += len(names)
        if ty == 0:
            continue
        rows, bpr = ne[1], ne[0] // 64
        if ty == 30:
            for k in range(8):
                smp.pick(k, rows, bpr)
            continue
        if ty != nvfp4.value:
            sys.exit("ggufdeq: %s has type %d" % (t.name, ty))
        raw = np.asarray(t.data, dtype=np.uint8).reshape(-1)
        scale = np.asarray(byname[t.name[:-len(".weight")] + ".scale"].data, dtype=np.float32).reshape(-1)
        per = rows * bpr * 36
        for e, hn in enumerate(names):
            r = row[hn]
            sc = float(scale[e])
            for k in range(8):
                rr, bb = smp.pick(k, rows, bpr)
                o = e * per + (rr * bpr + bb) * 36
                v = dequantize(raw[o:o + 36], nvfp4).reshape(-1)
                words = [r, rr, bb]
                for s in range(4):
                    for j in range(8):
                        for x in (16 * s + j, 16 * s + j + 8):
                            words.append(struct.unpack("<Q", struct.pack("<d", float(v[x]) * sc))[0])
                for w in words:
                    h = ((h ^ w) * 1099511628211) & M64
                    h ^= h >> 29
                blocks += 1
                elements += 64
    print("deq units %d blocks %d elements %d decode_digest %d" % (units, blocks, elements, h))


if __name__ == "__main__":
    main()
