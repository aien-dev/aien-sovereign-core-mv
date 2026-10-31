# Oracle B free-running leg (design 2026-10-15 section 2.4 B): first greedy divergence per prompt,
# FP8 host golden (SMOKE-8B-FP8-v1) against the bf16 host golden recorded 2026-10-31 by the same
# binary. Teacher-forced agreement count is not implemented in own8b_check (NOT_RUN).
import json, sys
a = json.load(open(sys.argv[1])); b = json.load(open(sys.argv[2]))
for pa, pb in zip(a["prompts"], b["prompts"]):
    assert pa["prompt_tokens"] == pb["prompt_tokens"], pa["name"]
    ga, gb = pa["golden"], pb["golden"]
    d = next((i for i, (x, y) in enumerate(zip(ga, gb)) if x != y), min(len(ga), len(gb)))
    print(pa["name"], "new", len(ga), "first_divergence", d, "equal_positions", sum(x == y for x, y in zip(ga, gb)),
          "margin0_fp8 %.4f margin0_bf16 %.4f" % (pa["margins"][0], pb["margins"][0]))
