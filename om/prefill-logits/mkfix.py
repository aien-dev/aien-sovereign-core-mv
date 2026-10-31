#!/usr/bin/env python3
"""Test equipment for om/prefill-logits/gate.sh: writes the synthetic logits fixtures.

Every file is 248320 little-endian binary32 values (993280 bytes), the layout qwen36_prefill_bench
writes for the last prompt token. Deterministic (fixed seeds); the judgment itself is made by
prefill-logits.om, never here.

  ref.f32    A: N(0, 2) background, top-1 id 1234 = 25.0, top-2 id 77 = 22.0, id 5000 = 0.5
  same.f32   copy of ref                                      -> PASS, max 0
  noise.f32  ref + N(0, 0.05) on every id                      -> PASS (green: the bound is not over-strict)
  edge.f32   ref with id 5000 = 1.5 (|d| = 1.0 exactly = TAU)  -> PASS (TAU is inclusive)
  over.f32   ref with id 5000 = nextafter(1.5, +inf)           -> FAIL (red mutant: one logit past TAU)
  refc.f32   ref with id 77 = 24.8 (close margin 0.2)
  flipc.f32  refc with id 1234 = 24.7, id 77 = 24.9            -> FAIL vs refc (top-1 flips, max |d| 0.3)
  refn.f32   ref with id 77 = 24.9375 (near tie, gap 1/16 below the 0.1 band)
  flipn.f32  refn with id 1234 = 24.9375, id 77 = 25.0        -> PASS vs refn (flip inside the band)
  nan.f32    ref with id 9 = NaN                               -> FAIL (nonfinite)
  other.f32  independent N(0, 2), top-1 id 42 = 25.0           -> FAIL (different prompt)
  short.f32  ref minus its last value (993276 bytes)          -> exit 3 (size)
"""
import random
import struct
import sys

NV = 248320


def background(seed):
    rng = random.Random(seed)
    return [rng.gauss(0.0, 2.0) for _ in range(NV)]


def f32(x):
    return struct.unpack("<f", struct.pack("<f", x))[0]


def write(path, vals, drop_last=False):
    if drop_last:
        vals = vals[:-1]
    with open(path, "wb") as f:
        f.write(struct.pack("<%df" % len(vals), *vals))


def main(out):
    ref = background(1)
    ref[1234] = 25.0
    ref[77] = 22.0
    ref[5000] = 0.5
    ref = [f32(v) for v in ref]
    write(f"{out}/ref.f32", ref)
    write(f"{out}/same.f32", ref)

    rng = random.Random(3)
    write(f"{out}/noise.f32", [v + rng.gauss(0.0, 0.05) for v in ref])

    edge = list(ref)
    edge[5000] = 1.5
    write(f"{out}/edge.f32", edge)
    over = list(ref)
    over[5000] = struct.unpack("<f", struct.pack("<I", struct.unpack("<I", struct.pack("<f", 1.5))[0] + 1))[0]
    write(f"{out}/over.f32", over)

    refc = list(ref)
    refc[77] = 24.8
    write(f"{out}/refc.f32", refc)
    flipc = list(refc)
    flipc[1234] = 24.7
    flipc[77] = 24.9
    write(f"{out}/flipc.f32", flipc)
    refn = list(ref)
    refn[77] = 24.9375
    write(f"{out}/refn.f32", refn)
    flipn = list(refn)
    flipn[1234] = 24.9375
    flipn[77] = 25.0
    write(f"{out}/flipn.f32", flipn)

    nan = list(ref)
    nan[9] = float("nan")
    write(f"{out}/nan.f32", nan)

    other = background(2)
    other[42] = 25.0
    write(f"{out}/other.f32", other)

    write(f"{out}/short.f32", ref, drop_last=True)


if __name__ == "__main__":
    main(sys.argv[1])
