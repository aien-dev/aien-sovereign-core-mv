#!/usr/bin/env python3
"""Synthetic GGUF v3 fixtures for om/flint/gguf.om (Core #834). Standard library only.

  ggufgen.py OUT CASE      writes the GGUF file OUT for CASE (see CASES)

"base" is a valid file: every scalar value type 0..12, arrays of six element types (one empty),
general.alignment, and four tensors (NVFP4, its F32 .scale, BF16, a rank-3 F32) laid out the way
ggml's gguf writer lays them out (data section aligned, each tensor padded to the alignment).
Every other case changes one thing so that exactly one guard of gguf.om must refuse it.
"""
import struct
import sys

ALIGN = 32
# ggml type: (block elements, block bytes) at llama.cpp b11804 (ggml.c type_traits)
TYPES = {0: (1, 4), 30: (1, 2), 40: (64, 36)}


def gstr(b):
    return struct.pack("<Q", len(b)) + b


def kv(key, vtype, payload):
    return gstr(key.encode()) + struct.pack("<I", vtype) + payload


def arr(etype, items):
    return struct.pack("<IQ", etype, len(items)) + b"".join(items)


def base_kvs(case):
    a = kv("general.alignment", 4, struct.pack("<I", ALIGN))
    if case == "align-pow2":
        a = kv("general.alignment", 4, struct.pack("<I", 48))
    if case == "align-type":
        a = kv("general.alignment", 10, struct.pack("<Q", ALIGN))
    kvs = [
        kv("general.architecture", 8, gstr(b"flint")),
        a,
        kv("general.name", 8, gstr(b"Synthetic GGUF fixture")),
        kv("general.file_type", 4, struct.pack("<I", 39)),
        kv("general.quantization_version", 4, struct.pack("<I", 2)),
        kv("t.u8", 0, struct.pack("<B", 200)),
        kv("t.i8", 1, struct.pack("<b", -5)),
        kv("t.u16", 2, struct.pack("<H", 60000)),
        kv("t.i16", 3, struct.pack("<h", -300)),
        kv("t.u32", 4, struct.pack("<I", 4000000000)),
        kv("t.i32", 5, struct.pack("<i", -70000)),
        kv("t.f32", 6, struct.pack("<f", 1.5)),
        kv("t.bool", 7, struct.pack("<B", 1)),
        kv("t.u64", 10, struct.pack("<Q", (1 << 40) + 3)),
        kv("t.i64", 11, struct.pack("<q", -(1 << 40))),
        kv("t.f64", 12, struct.pack("<d", 0.25)),
        kv("t.long", 8, gstr(b"x" * 100)),
        kv("t.tab", 8, gstr(b"a\tb")),
        kv("t.au8", 9, arr(0, [struct.pack("<B", v) for v in (1, 2, 3)])),
        kv("t.ai32", 9, arr(5, [struct.pack("<i", v) for v in (150001, -1, 7)])),
        kv("t.af32", 9, arr(6, [struct.pack("<f", v) for v in (1.0, 2.0)])),
        kv("t.abool", 9, arr(7, [struct.pack("<B", v) for v in (1, 0, 1)])),
        kv("t.astr", 9, arr(8, [gstr(b"ab"), gstr(b"cde")])),
        kv("t.ai64", 9, arr(11, [struct.pack("<q", v - 5) for v in range(10)])),
        kv("t.aempty", 9, arr(4, [])),
        kv("tokenizer.ggml.bos_token_id", 4, struct.pack("<I", 1)),
        kv("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
    ]
    if case == "kv-type":
        kvs.append(kv("t.bad", 13, struct.pack("<I", 0)))
    if case == "kv-nested":
        kvs.append(kv("t.nest", 9, arr(9, [])))
    if case == "kv-empty-key":
        kvs.append(kv("", 4, struct.pack("<I", 0)))
    if case == "kv-dup":
        kvs.append(kv("t.u8", 0, struct.pack("<B", 1)))
    return kvs


def tensors(case):
    # name, dims (innermost first, as GGUF stores them), ggml type
    ts = [
        ("a.weight", [128, 2], 40),
        ("a.scale", [1], 0),
        ("b.weight", [64, 3], 30),
        ("c.weight", [16, 2, 2], 0),
    ]
    if case == "ggml-type":
        ts[2] = ("b.weight", [64, 3], 43)
    if case == "ggml-removed":
        ts[2] = ("b.weight", [64, 3], 4)
    if case == "row-block":
        ts[0] = ("a.weight", [96, 2], 40)
    if case == "ndims":
        ts[3] = ("c.weight", [16, 2, 1, 1, 2], 0)
    if case == "tensor-dup":
        ts[3] = ("b.weight", [16, 2, 2], 0)
    return ts


def nbytes(dims, t):
    n = 1
    for d in dims:
        n *= d
    be, bb = TYPES.get(t, (1, 4))
    return n // be * bb


def pad(n):
    return (n + ALIGN - 1) // ALIGN * ALIGN


def build(case):
    kvs = base_kvs(case)
    ts = tensors(case)
    nt = len(ts) + (4997 if case == "cap-tensors" else 0)
    head = b"GGUF" if case != "magic" else b"GGUX"
    head += struct.pack("<I", 2 if case == "version" else 3)
    head += struct.pack("<QQ", nt, len(kvs))
    head += b"".join(kvs)
    infos = b""
    off = 0
    blobs = []
    for i, (name, dims, t) in enumerate(ts):
        o = off
        if case == "misaligned" and i == 2:
            o = off + 16
        if case == "gap" and i == 2:
            o = off + ALIGN
        if case == "past-eof" and i == 3:
            o = off + ALIGN * 1000
        infos += gstr(name.encode()) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<IQ", t, o)
        n = nbytes(dims, t)
        blobs.append(bytes((i * 37 + k * 11) & 255 for k in range(n)))
        off += pad(n)
    out = head + infos
    out += b"\0" * (pad(len(out)) - len(out))
    for blob in blobs:
        out += blob + b"\0" * (pad(len(blob)) - len(blob))
    if case == "truncated":
        out = out[: len(head) + len(infos) // 2]
    if case == "short-data":
        out = out[:-40]
    if case == "trailing":
        out += b"\0" * ALIGN
    return out


CASES = ["base", "truncated", "magic", "version", "kv-type", "kv-nested", "kv-empty-key", "kv-dup",
         "align-pow2", "align-type", "ndims", "row-block", "ggml-type", "ggml-removed", "tensor-dup",
         "misaligned", "past-eof", "short-data", "gap", "trailing", "cap-tensors"]

if __name__ == "__main__":
    out, case = sys.argv[1:3]
    if case not in CASES:
        sys.exit("unknown case " + case)
    with open(out, "wb") as f:
        f.write(build(case))
