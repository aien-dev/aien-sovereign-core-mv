#!/usr/bin/env python3
"""Flint manifest fixture generator. TEST REFEREE ONLY, never an implementation dependency.

Builds a full Flint-shaped checkpoint from config.json alone: every tensor name and shape is derived
here independently of manifest.om. Shards are sparse: an 8-byte little-endian header length, the
header JSON, then ftruncate to the data length, so 79.5 GB of apparent size costs about 4 MB of disk.

Usage: gen.py CONFIG OUT MUTANT
Writes OUT/config.json, OUT/index.json, OUT/shards/*.safetensors and, for MUTANT base, OUT/expected.tsv.
Every other MUTANT applies exactly one defect; check.sh holds the exit code each one must produce.
"""
import json
import os
import struct
import sys

cfg_path, out, mut = sys.argv[1:4]
raw = open(cfg_path, "rb").read()
c = json.loads(raw)
H = c["hidden_size"]
V = c["vocab_size"]
HD = c["head_dim"]
NH = c["num_attention_heads"]
NKV = c["num_key_value_heads"]
NE = c["num_experts"]
MI = c["moe_intermediate_size"]
SI = c["shared_expert_intermediate_size"]
DI = c["intermediate_size"]
NL = c["num_hidden_layers"]
DENSE = set(c["mlp_only_layers"])


def tensors():
    t = [("model.embed_tokens.weight", [V, H]), ("model.norm.weight", [H]), ("lm_head.weight", [V, H])]
    for L in range(NL):
        p = "model.layers.%d." % L
        t += [
            (p + "input_layernorm.weight", [H]),
            (p + "post_attention_layernorm.weight", [H]),
            (p + "self_attn.q_proj.weight", [NH * HD, H]),
            (p + "self_attn.k_proj.weight", [NKV * HD, H]),
            (p + "self_attn.v_proj.weight", [NKV * HD, H]),
            (p + "self_attn.o_proj.weight", [H, NH * HD]),
            (p + "self_attn.q_norm.weight", [HD]),
            (p + "self_attn.k_norm.weight", [HD]),
            (p + "self_attn.softmax_offset", [NH]),
        ]
        if L in DENSE:
            t += [(p + "mlp.gate_proj.weight", [DI, H]), (p + "mlp.up_proj.weight", [DI, H]),
                  (p + "mlp.down_proj.weight", [H, DI])]
            continue
        t += [(p + "mlp.gate.weight", [NE, H]),
              (p + "mlp.shared_expert.gate_proj.weight", [SI, H]),
              (p + "mlp.shared_expert.up_proj.weight", [SI, H]),
              (p + "mlp.shared_expert.down_proj.weight", [H, SI])]
        for e in range(NE):
            q = p + "mlp.experts.%d." % e
            t += [(q + "gate_proj.weight", [MI, H]), (q + "up_proj.weight", [MI, H]),
                  (q + "down_proj.weight", [H, MI])]
    return t


def numel(shape):
    n = 1
    for d in shape:
        n *= d
    return n


NS = 8 if mut == "shard-cap" else 15
shape_of = dict(tensors())
names = sorted(shape_of)
sname = ["model-%05d-of-%05d.safetensors" % (i + 1, NS) for i in range(NS)]
# shard i: list of [name, dtype, shape, begin, end]; extras: unparsed text spliced into the header
shards = []
for i in range(NS):
    part = names[i * len(names) // NS:(i + 1) * len(names) // NS]
    shards.append([[n, "BF16", list(shape_of[n]), 0, 0] for n in part])
wmap = [[n, sname[i]] for i in range(NS) for n in (e[0] for e in shards[i])]


def pack(rows):
    o = 0
    for r in rows:
        r[3] = o
        o += 2 * numel(r[2])
        r[4] = o
    return o


for rows in shards:
    pack(rows)
total = sum(2 * numel(s) for s in shape_of.values())

# ---- one defect per mutant -------------------------------------------------------------------
s0 = shards[0]
cfg_text = raw
idx_tail = ""
extra_head = ""          # text placed before the first entry of shard 0
extra_entry = ""         # extra key inside the first entry of shard 0
dup_entry = False        # repeat the first entry of shard 0 at the end of its header
size_delta = 0           # bytes added to shard 0's data length
hl_mode = "le"
GATE = "model.layers.2.mlp.shared_expert_gate.weight"

if mut == "cfg-json":
    cfg_text = b"{"
elif mut == "cfg-field":
    cfg_text = raw.replace(b'"num_experts": 256', b'"num_experts": 128')
elif mut == "cfg-unknown":
    cfg_text = raw.replace(b'"vocab_size": 150144', b'"vocab_size": 150144,\n  "foo": 1')
elif mut == "cfg-dup":
    cfg_text = raw.replace(b'"vocab_size": 150144', b'"vocab_size": 150144,\n  "vocab_size": 150144')
elif mut == "cfg-missing":
    cfg_text = raw.replace(b',\n  "vocab_size": 150144', b"")
elif mut == "cfg-softmax":
    cfg_text = raw.replace(b'"router_score_function": "sigmoid"', b'"router_score_function": "softmax"')
elif mut == "cfg-window":
    cfg_text = raw.replace(b'"sliding_window": 513', b'"sliding_window": 4096')
elif mut == "cfg-tie":
    cfg_text = raw.replace(b'"tie_word_embeddings": false', b'"tie_word_embeddings": true')
elif mut == "idx-total":
    total += 2
elif mut == "idx-missing":
    del wmap[5]
elif mut == "idx-dup":
    wmap.insert(6, list(wmap[5]))
elif mut == "idx-unknown":
    wmap.append([GATE, sname[0]])
elif mut == "idx-shard-name":
    wmap[5][1] = "../" + sname[0]
elif mut == "idx-syntax":
    idx_tail = "x"
elif mut == "idx-size":
    idx_tail = " " * 4194304
elif mut == "hdr-dtype":
    s0[0][1] = "F16"
elif mut == "hdr-shape":
    r = next(r for r in s0 if r[0].endswith("q_proj.weight"))
    r[2] = [r[2][1], r[2][0]]
elif mut == "hdr-rank":
    r = next(r for r in s0 if r[0].endswith("q_norm.weight"))
    r[2] = r[2] + [1]
elif mut == "hdr-off-len":
    s0[-1][4] += 2
    size_delta = 2
elif mut == "hdr-off-overflow":
    s0[-1][4] = 2 ** 64
elif mut == "hdr-overlap":
    a, b = s0[0], s0[1]
    assert a[4] - a[3] == b[4] - b[3]
    b[3], b[4] = a[3], a[4]
elif mut == "hdr-gap":
    s0[-1][3] += 2
    s0[-1][4] += 2
    size_delta = 2
elif mut == "hdr-trailing":
    size_delta = 2
elif mut == "hdr-truncated":
    size_delta = -2
elif mut == "hdr-dup":
    dup_entry = True
elif mut == "hdr-mismatch":
    moved = s0.pop()
    shards[1].append(moved)
    pack(shards[1])
elif mut == "hdr-missing":
    s0.pop()
elif mut == "hdr-unknown":
    s0.append([GATE, "BF16", [1, H], 0, 0])
    pack(s0)
elif mut == "hdr-meta":
    extra_head = '"__metadata__":{"format":"pt"},'
elif mut == "hdr-entry":
    extra_entry = ',"extra":1'
elif mut not in ("base", "hdr-json", "hdr-len-be", "hdr-len-big", "shard-short", "shard-absent",
                 "shard-symlink", "shard-dir", "shard-cap"):
    sys.exit("gen.py: unknown mutant " + mut)
if cfg_text == raw and mut.startswith("cfg-") and mut != "cfg-json":
    sys.exit("gen.py: mutant " + mut + " did not change config.json")

# ---- write -------------------------------------------------------------------------------------
os.makedirs(out + "/shards")
with open(out + "/config.json", "wb") as f:
    f.write(cfg_text)

lines = ["{", '  "metadata": {', '    "total_size": %d' % total, "  },", '  "weight_map": {']
lines += ['    "%s": "%s"%s' % (n, s, "," if k + 1 < len(wmap) else "") for k, (n, s) in enumerate(wmap)]
lines += ["  }", "}"]
with open(out + "/index.json", "w") as f:
    f.write("\n".join(lines) + "\n" + idx_tail)


def entry(r, more=""):
    return '"%s":{"dtype":"%s","shape":[%s],"data_offsets":[%d,%d]%s}' % (
        r[0], r[1], ",".join(str(d) for d in r[2]), r[3], r[4], more)


for i, rows in enumerate(shards):
    path = out + "/shards/" + sname[i]
    ents = [entry(r, extra_entry if (i == 0 and k == 0) else "") for k, r in enumerate(rows)]
    if i == 0 and dup_entry:
        ents.append(entry(rows[0]))
    head = "{" + (extra_head if i == 0 else "") + ",".join(ents) + "}"
    if i == 0 and mut == "hdr-json":
        head = "{"
    head = head.encode()
    head += b" " * (-len(head) % 8)
    data = max((r[4] for r in rows if r[4] < 2 ** 63), default=0)
    if i == 0:
        data += size_delta
    hl = len(head)
    if i == 0 and mut == "hdr-len-big":
        hl = 8 + len(head) + data
    with open(path, "wb") as f:
        if i == 0 and mut == "shard-short":
            f.write(b"\x00" * 4)
            continue
        f.write(struct.pack(">Q" if (i == 0 and mut == "hdr-len-be") else "<Q", hl))
        f.write(head)
        f.truncate(8 + len(head) + data)

p0 = out + "/shards/" + sname[0]
if mut == "shard-absent":
    os.remove(p0)
elif mut == "shard-symlink":
    os.rename(p0, p0 + ".real")
    os.symlink(sname[0] + ".real", p0)
elif mut == "shard-dir":
    os.remove(p0)
    os.mkdir(p0)

if mut == "base":
    with open(out + "/expected.tsv", "w") as f:
        f.write("name\tshard\tdtype\tshape\tbegin\tend\n")
        for i, rows in enumerate(shards):
            for r in rows:
                f.write("%s\t%s\t%s\t%s\t%d\t%d\n" % (r[0], sname[i], r[1], "x".join(str(d) for d in r[2]), r[3], r[4]))
