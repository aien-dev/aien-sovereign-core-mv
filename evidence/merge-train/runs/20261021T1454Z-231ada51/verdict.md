# merge-train verdict: 20261021T1454Z-231ada51

PASS/FAIL: identical / not identical to the reference. RED-OK/RED-FAIL: a control that must differ did / did not. NOT-ON-TRAIN: the lever (or its switch) is not in this candidate. SKIP: arm disabled in suite.json. ERROR: no verdict (no reply, 400, missing reference). -: the lever has no case there. A cell is the worst of the lever's arms.

| lever | name | VERDICT | golden-chat-max-24 | golden-chat-code | golden-chat-prose2 | golden-chat-chat | golden-chat-review | concurrent-32 | turn2 | tool-call | mutant-4-red | mutant-row-isolated | row-swap-red | board-gate |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L354 | batch door: many conversations per beat | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L312 | strokes v2: rows ride one graph | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L314 | exact two-pass head | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L374 | tool calling | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L369 | chunked prefill | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L375 | shared prompt prefix | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L371 | eight readers | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L355 | W4 experts store | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L313 | bf16 DeltaNet state | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | - |
| L376 | team board | **ERROR?** | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |

red controls (the comparison can fail): 0 RED-OK of 0 run, 3 in the suite

## cells (arm x case)

| arm | case | verdict | detail |
|---|---|---|---|
| plain | * | ERROR | candidate never served |
