# merge-train verdict: 20261022T0244Z-bfc41659

candidate bin-bfc41659; 1 load(s); load 91 s + suite 437 s = 528 s wall; reference build 5b726818

PASS/FAIL: identical / not identical to the reference. RED-OK/RED-FAIL: a control that must differ did / did not. NOT-ON-TRAIN: the lever (or its switch) is not in this candidate. SKIP: arm disabled in suite.json. ERROR: no verdict (no reply, 400, missing reference). -: the lever has no case there. A cell is the worst of the lever's arms.

| lever | name | VERDICT | golden-chat-max-24 | golden-chat-code | golden-chat-prose2 | golden-chat-chat | golden-chat-review | concurrent-32 | turn2 | tool-call | mutant-4-red | mutant-row-isolated | row-swap-red | board-gate | l369-long-prompts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| L354 | batch door: many conversations per beat | **PASS** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L312 | strokes v2: rows ride one graph | **PASS** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L314 | exact two-pass head | **PASS** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L374 | tool calling | **PASS** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L369 | chunked prefill | **PASS** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | PASS |
| L375 | shared prompt prefix | **NOT-ON-TRAIN** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L371 | eight readers | **NOT-ON-TRAIN** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L355 | W4 experts store | **NOT-ON-TRAIN** | PASS | PASS | PASS | PASS | PASS | PASS | PASS | PASS | RED-OK | RED-OK | RED-OK | - | - |
| L313 | bf16 DeltaNet state | **PASS** | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | RED-OK | RED-OK | RED-OK | - | - |
| L376 | team board | **NOT-ON-TRAIN** | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | SKIP | RED-OK | RED-OK | RED-OK | SKIP | - |

red controls (the comparison can fail): 3 RED-OK of 3 run, 3 in the suite

## cells (arm x case)

| arm | case | verdict | detail |
|---|---|---|---|
| plain | golden-chat-max-24 | PASS | ids identical to the golden (24 prompt tokens) |
| plain | golden-chat-code | PASS | ids identical to the golden (31 prompt tokens) |
| plain | golden-chat-prose2 | PASS | ids identical to the golden (37 prompt tokens) |
| plain | golden-chat-chat | PASS | ids identical to the golden (40 prompt tokens) |
| plain | golden-chat-review | PASS | ids identical to the golden (66 prompt tokens) |
| plain | concurrent-32 | PASS | 32/32 replies identical to the reference (alone) |
| plain | turn2 | PASS | turn 2 identical to alone |
| plain | tool-call | PASS | reply is a parsed tool call to get_weather ("{\"city\":\"Paris\"}") |
| plain | mutant-4-red | RED-OK | switched reply differs from the golden, as required |
| plain | mutant-row-isolated | RED-OK | rows 5 differ from alone as required (expected red [5], 7 others identical, 0 swap(s) logged) |
| plain | row-swap-red | RED-OK | rows 0 3 differ from alone as required (expected red [3], 6 others identical, 1 swap(s) logged) |
| plain | l369-long-prompts | PASS | owner's gate exit 0 |
| stroke-off | golden-chat-max-24 | PASS | ids identical to the golden (24 prompt tokens) |
| stroke-off | golden-chat-code | PASS | ids identical to the golden (31 prompt tokens) |
| stroke-off | golden-chat-prose2 | PASS | ids identical to the golden (37 prompt tokens) |
| stroke-off | golden-chat-chat | PASS | ids identical to the golden (40 prompt tokens) |
| stroke-off | golden-chat-review | PASS | ids identical to the golden (66 prompt tokens) |
| stroke-off | concurrent-32 | PASS | 32/32 replies identical to the reference (alone) |
| stroke-off | turn2 | PASS | turn 2 identical to alone |
| stroke-off | tool-call | PASS | reply is a parsed tool call to get_weather ("{\"city\":\"Paris\"}") |
| exact-head-off | golden-chat-max-24 | PASS | ids identical to the golden (24 prompt tokens) |
| exact-head-off | golden-chat-code | PASS | ids identical to the golden (31 prompt tokens) |
| exact-head-off | golden-chat-prose2 | PASS | ids identical to the golden (37 prompt tokens) |
| exact-head-off | golden-chat-chat | PASS | ids identical to the golden (40 prompt tokens) |
| exact-head-off | golden-chat-review | PASS | ids identical to the golden (66 prompt tokens) |
| exact-head-off | concurrent-32 | PASS | 32/32 replies identical to the reference (alone) |
| exact-head-off | turn2 | PASS | turn 2 identical to alone |
| exact-head-off | tool-call | PASS | reply is a parsed tool call to get_weather ("{\"city\":\"Paris\"}") |
| state-bf16 | golden-chat-max-24 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | golden-chat-code | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | golden-chat-prose2 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | golden-chat-chat | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | golden-chat-review | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | concurrent-32 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | turn2 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| state-bf16 | tool-call | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | golden-chat-max-24 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | golden-chat-code | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | golden-chat-prose2 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | golden-chat-chat | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | golden-chat-review | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | concurrent-32 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | turn2 | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | tool-call | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
| board | board-gate | SKIP | arm disabled in suite.json (its lever is not in the candidate yet) |
