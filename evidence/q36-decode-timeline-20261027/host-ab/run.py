#!/usr/bin/env python3
# THROWAWAY host A/B runner for evidence/q36-decode-timeline-20261027/PROTOCOL.md (#785).
# 2 warm-up rounds discarded, 20 rounds, arm order per round from random.Random(785),
# one process per arm per round on taskset -c 12 under nice -n 19.
import json, os, random, re, subprocess, sys

here = os.path.dirname(os.path.abspath(__file__))
exe = {a: open(os.path.join(here, f"exe-{a}.txt")).read().split()[0] for a in "AB"}
rng = random.Random(785)
out = {"exe": exe, "rounds": []}
for rnd in range(-2, 20):
    order = ["A", "B"]
    rng.shuffle(order)
    load = open("/proc/loadavg").read().split()[0]
    row = {"round": rnd, "order": "".join(order), "load1": load}
    for a in order:
        p = subprocess.run(
            ["nice", "-n", "19", "taskset", "-c", "12", exe[a], "--ignored", "--exact",
             "qwen36_batch::tests::bench_door_beat", "--nocapture", "--test-threads=1"],
            capture_output=True, text=True)
        m = re.search(r"BENCH tokens (\d+) ns (\d+) digest ([0-9a-f]+)", p.stdout)
        if p.returncode != 0 or not m:
            sys.exit(f"arm {a} round {rnd} failed rc={p.returncode}: {p.stdout[-500:]} {p.stderr[-500:]}")
        row[a] = {"tokens": int(m[1]), "ns": int(m[2]), "digest": m[3]}
    print(json.dumps(row), flush=True)
    out["rounds"].append(row)
json.dump(out, open(os.path.join(here, "raw.json"), "w"), indent=1)
kept = [r for r in out["rounds"] if r["round"] >= 0]
digests = {r[a]["digest"] for r in out["rounds"] for a in "AB"}
tokens = {r[a]["tokens"] for r in out["rounds"] for a in "AB"}
gate = "PASS" if len(digests) == 1 and tokens == {20000} else "VOID"
with open(os.path.join(here, "ab.wl"), "w") as f:
    f.write(f"(* correctness gate {gate}: digests {sorted(digests)}, tokens {sorted(tokens)} *)\n")
    f.write("nsA = {%s};\n" % ", ".join(str(r["A"]["ns"]) for r in kept))
    f.write("nsB = {%s};\n" % ", ".join(str(r["B"]["ns"]) for r in kept))
print("gate", gate)
