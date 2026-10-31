# DRA-86: prompt cache for the 35B batch door (G4), 2026-10-31

Status: IMPLEMENTED and EXERCISED on host (cargo tests, mock door). Chip: NOT_RUN. Nothing here has run on the GB10.

## What the gate checks (G4)

| G4 item | step | judged by |
|---|---|---|
| switch default off | 1 `ref` | the door logs `prompt cache off (needs AIEN_Q36_PREFIX=1 …)` without the switch |
| red first | 2 `red` | `AIEN_Q36_PREFIX_MUTANT=1` (a restored row continues one position early) must change at least one turn 2 |
| token identity, set32 + l369-long | 3 `green` | 35 conversations x 2 turns all SAME vs step 1; every turn 2 restored, none refused; `budget 16.0 GiB` |
| stale snapshot refused (model/bin/env) | 4a/4b/4c | sealed store: 4b reloads and restores, 4c (another numeric env key = another namespace) discards every node and reads cold, SAME |
| budget enforced, resident bytes reported | 5 `budget` | `AIEN_Q36_PREFIX_POOL_GB=0.5`: every save line carries `conversation resident … MiB`, pool <= 512 MiB, evictions > 0, SAME |
| turn-2 TTFT 32k cached + 1k new | 6a `warm`, 6b `cold` | HTTP TTFT (client, first content delta) and the door's restore ms and prefill ms; RECORDED, compared with the model below |

Off-by-one turn, token mismatch, KV gap, KV/state length mismatch and model/lib tag: refused inside the pool before any
write (`crates/aien-serve-http/src/q36_prefix/tests.rs`: `a_node_restored_as_other_tokens_or_one_turn_off_is_refused`,
`a_kv_gap_or_a_kv_or_state_length_mismatch_is_refused`, `a_node_of_another_model_or_lib_tag_is_refused_and_dropped`;
door level `a_stale_or_short_pool_node_is_refused_and_the_prompt_read_cold`). A chip run cannot provoke them from
outside, so the chip gate covers the namespace and the mutant, and the host tests cover the rest. Shape refusal for the
device copy is in the CUDApp module (`src/omega_gpu_q36_prefix.c`, `tests/gpu_q36_prefix_test.c`).

## Files

- `window-dra86.sh`: `plan`, `run BIN OUT STEP…` (lock, lease, watchdog, receipts; from window-dra65.sh), probes `conv`, `time`.
- `set32.json` (copy of evidence/dra65-prefill-20261031/set32.json), `SHA256SUMS` (it plus l369 long-256/2048/8192).
- `hosttest/`: `mock86.py` (mock batch door with the door's log wording), `fake-serve.sh`, `runtests.sh`.

Host check: `nice -n 19 taskset -c 5 bash evidence/dra86-prefix-20261031/hosttest/runtests.sh` -> 13 `ok`,
`runtests: 0 failed` (plan; no window NOT_RUN; all 9 steps PASS on the exact mock; store key reaches 4a/4b/4c, one key,
in no receipt; red without effect FAIL; green after a failed red NOT_RUN; red without restore INCONCLUSIVE; green
without restore FAIL; stale namespace loaded FAIL; 4b alone NOT_RUN; green on an empty OUT NOT_RUN; budget probe at
16 GiB FAIL; no leftover process; lock free).

## Model (MODELED, WolframLanguageEvaluator)

Bytes per conversation (10 attention layers, K+V bf16, 2 x 512 per layer; 30 DeltaNet layers, state 2 MiB + conv tail 192 KiB):

- `kv = 10*2*2*32768*512; st = 30*2097152; tl = 30*196608; {kv, st, tl, kv+st+tl, N[(kv+st+tl)/2^20], Floor[16*2^30/(kv+st+tl)], 2*2*512*10}`
  -> `{671088640, 62914560, 5898240, 739901440, 705.625, 23, 20480}`
  (32k-token conversation 739,901,440 B = 705.6 MiB; 23 such in the 16 GiB default; K/V 20,480 B per token.)

Turn-2 TTFT (QMMA peak 237.5 TFLOPS, 4.87 GFLOP per token, attention 163,840 FLOP per (q,k) pair, bandwidth 273 GB/s,
33.6 GB weight pass overlapped with the GEMMs, 1,024 new tokens after 32,768 cached, restore = 2 x blob / bandwidth,
`e` = fraction of QMMA peak the prefill reaches):

- `peak = 237.5*10^12; bw = 273*10^9; f = 4.87*10^9; a = 163840; n = 1024; c = 32768; w = 33.6*10^9; snap = 739901440; gemm = n f/peak; pairs = n c + n (n + 1)/2; attn = pairs a/peak; wt = w/bw; rst = 2 snap/bw; t2[e_] := Max[gemm/e, wt] + attn/e + rst; m = n + c; cold = m f/peak + (m (m + 1)/2) a/peak; {gemm, pairs, attn, wt, rst, t2[1], cold, snap + 20480 n, FindRoot[t2[e] == 0.4, {e, 0.2}], t2[0.3]}`
  -> `{0.02099738947368421, 34079232, 0.02350964787738947, 0.12307692307692307, 578048/106640625, 0.15200709432427592, 1.0867966145158738, 760872960, {e -> 0.11279612850417702}, 0.20686293937151803}`
- warm turn 2 0.152 s at peak (restore 5.4 ms), cold 33k prompt 1.087 s, 760,872,960 B per conversation incl. the
  1k turn; G4's 0.4 s holds above 11.3 % of QMMA peak; 0.207 s at 30 %.
- Not met by today's prefill: the batch door's measured chunked prefill is ~250 tok/s, so 1,024 new tokens alone
  take ~4 s. G4's 0.4 s needs DRA-82's tensor-core prefill; step 6 records the split (restore ms, prefill ms).

Step estimates (s): `start=60; dec=0.0201; pf=250; pfl=120; t1set=756*dec+874*dec; t2setCold=(756+874+32*12)*dec+32*16*dec; t2setWarm=(32*12)*dec+32*16*dec+32*0.5; t1long=(256+2486+8210)/pf+72*dec; t2longCold=(256+2486+8210+72+36)/pf+48*dec; t2longWarm=(36)*dec+48*dec+3*0.5; ref=start+t1set+t2setCold+t1long+t2longCold; green=start+t1set+t2setWarm+t1long+t2longWarm; st4a=start+2486/pf+24*dec+20; st4=start+2486/pf+24*dec+(2486+60)/pf+16*dec; budget=start+3*((256+2486+8210)/3/pf)+2*(10952/pf)+200*dec; c=32840; n=1050; warm=start+c/pfl+16*dec+3*(1+n/pf+16*dec); cold=start+c/pfl+16*dec+(c+n)/pfl+16*dec; N[{ref,green,st4a,st4,budget,warm,cold}]`
-> `{233.9956, 175.2162, 90.4264, 80.932, 195.444, 350.553, 616.727}`; the script rounds up to 240/180/180/95/85/85/200/355/620.
Assumptions: batch server start 60 s; decode 0.0201 s/token (short set32 prompts stepped at the same rate); long
prompts 250 tok/s; the ~32.8k prompt a deliberately slow 120 tok/s; restore 0.5 s per turn 2; store flush 20 s.
