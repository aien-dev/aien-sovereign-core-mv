#!/usr/bin/env python3
"""Referee for om/flint/gguf.om (Core #834): the report written from llama.cpp's own gguf-py.

  ggufref.py report GGUF   the header report gguf.om must reproduce byte for byte

Pinned referee: llama.cpp tag b11804 (commit c7fc5094179bf8a9c73c18593db714f8da41a402), gguf-py
0.19.0, imported from that checkout (PYTHONPATH=<checkout>/gguf-py). The two module files the report
depends on are checked against their sha256 at that commit before anything is read.
Format (one line each, file order for kv and tensor lines):
  gguf version V tensors T kvs K alignment A data D bytes F
  kv KEY u8|i8|u16|i16|u32|i32|u64|i64 VALUE | f32|f64 bits B | bool 0|1
  kv KEY string LEN [TEXT]               TEXT when 1 <= LEN <= 64 and every byte is 32..126
  kv KEY array ETYPE N bytes B           string elements: B = total string bytes
  kv KEY array ETYPE N sum S [values ..] numbers (float: their bits), values when N <= 8
  tensor NAME TYPE D0xD1.. OFFSET NBYTES dims innermost first, OFFSET within the data section
  type ID tensors N elements E bytes B   ascending ggml type id
  data used U padded P
  quant file_type X quantization_version Y      (none when absent)
  tokenizer bos X eos Y                         (none when absent)
"""
import hashlib
import os
import sys

PIN = {
    "gguf_reader.py": "d0ea743200e19d7ef0a4c969edcc4a6dba8e3656a790b217e3bdc7b31170718e",
    "constants.py": "8d82aa2afea617f2ce1c9775fdad16dbb5a1dfa51261250d0d8918a78557bb3f",
}
TNAME = ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool", "string", "array", "u64", "i64", "f64"]
SIGNED = {1, 3, 5, 11}
FLOAT = {6: "<u4", 12: "<u8"}


def pinned():
    import gguf
    d = os.path.dirname(gguf.__file__)
    for f, want in PIN.items():
        got = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
        if got != want:
            sys.exit("ggufref: %s/%s sha256 %s is not the b11804 file" % (d, f, got))
    return gguf


def num(part, t):
    if t in FLOAT:
        return int(part.view(FLOAT[t])[0])
    return int(part[0])


def text(b):
    b = bytes(b)
    if 1 <= len(b) <= 64 and all(32 <= c <= 126 for c in b):
        return " " + b.decode()
    return ""


def kvline(k, f):
    t = int(f.types[0])
    if t == 8:
        b = bytes(f.parts[-1])
        return "kv %s string %d%s" % (k, len(b), text(b))
    if t == 9:
        e = int(f.parts[3][0])  # f.types has no element type for an empty array
        n = len(f.data)
        s = "kv %s array %s %d" % (k, TNAME[e], n)
        if e == 8:
            return s + " bytes %d" % sum(len(f.parts[i]) for i in f.data)
        vals = [num(f.parts[i], e) for i in f.data]
        s += " sum %d" % sum(vals)
        if n <= 8 and n > 0:
            s += " values " + " ".join(str(v) for v in vals)
        return s
    v = num(f.parts[-1], t)
    if t in FLOAT:
        return "kv %s %s bits %d" % (k, TNAME[t], v)
    return "kv %s %s %d" % (k, TNAME[t], v)


def scalar(r, k):
    f = r.fields.get(k)
    if f is None or int(f.types[0]) in (8, 9):
        return "none"
    return str(num(f.parts[-1], int(f.types[0])))


def report(path):
    gguf = pinned()
    r = gguf.GGUFReader(path)
    a = r.alignment
    out = []
    ver = int(r.fields["GGUF.version"].parts[-1][0])
    kvs = [(k, f) for k, f in r.fields.items() if not k.startswith("GGUF.")]
    out.append("gguf version %d tensors %d kvs %d alignment %d data %d bytes %d"
               % (ver, len(r.tensors), len(kvs), a, r.data_offset, os.path.getsize(path)))
    for k, f in kvs:
        out.append(kvline(k, f))
    types = {}
    used = padded = 0
    for t in r.tensors:
        ty = int(t.tensor_type)
        nb = int(t.n_bytes)
        out.append("tensor %s %d %s %d %d" % (t.name, ty, "x".join(str(int(d)) for d in t.shape),
                                               int(t.data_offset) - r.data_offset, nb))
        c = types.setdefault(ty, [0, 0, 0])
        c[0] += 1
        c[1] += int(t.n_elements)
        c[2] += nb
        used += nb
        padded += (nb + a - 1) // a * a
    for ty in sorted(types):
        out.append("type %d tensors %d elements %d bytes %d" % (ty, *types[ty]))
    out.append("data used %d padded %d" % (used, padded))
    out.append("quant file_type %s quantization_version %s"
               % (scalar(r, "general.file_type"), scalar(r, "general.quantization_version")))
    out.append("tokenizer bos %s eos %s"
               % (scalar(r, "tokenizer.ggml.bos_token_id"), scalar(r, "tokenizer.ggml.eos_token_id")))
    sys.stdout.write("\n".join(out) + "\n")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "report":
        report(sys.argv[2])
    else:
        sys.exit("usage: ggufref.py report GGUF")
