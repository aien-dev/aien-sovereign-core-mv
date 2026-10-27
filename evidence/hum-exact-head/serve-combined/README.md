# L314 + L356 combined served 35B: bin-409c8ffa (2026-10-21)

One build with all three quick levers ON by default:
- L314: exact two-pass head on greedy requests.
- L356: a|b beside mix.
- L356: greedy replies take the chip's token.

sc `hum/exact-head-levers` 409c8ffa merges `hum/exact-head-serve` e1f02b29 with `levers/q36-quick` 6b5c14ab. The
omega lib is the same as bin-e1f02b29 (`BUILD-bin-409c8ffa.txt`).

## Gate (`serve-gate.sh`, 18093, 03:54 to 03:57Z, the chip free: lock wait 0 on every OFF request)

- ON: gate2 PASS (`gate2-l314-combined-on.json`).
- MUTANT 5 (exact head MASK inverted): gate2 FAIL, as required.
- OFF (`AIEN_Q36_EXACT_HEAD=0`, the L356 levers still on): gate2 PASS.
- ON ids == OFF ids on 5 of 5 requests: 4 chat prompts x 128 greedy + 1 sampled (t 1.0, seed 7). VERDICT PASS
  (`serve-gate.txt`).

## Timing (server per-request line "graph chip ms/token"; single requests, no other chip work at the time)

| greedy request | ON | OFF (exact head off, L356 levers on) |
|---|---|---|
| gate2 (24 tok) | 21.9 | 23.8 |
| prompt 0 (128) | 21.8 | 24.0 |
| prompt 1 (128) | 21.8 | 24.0 |
| prompt 2 (128) | 21.8 | 24.0 |
| prompt 3 (128) | 21.6 | 24.2 |
| sampled, full head both | 24.6 | 24.6 |

Median greedy: ON 21.8 ms = 45.87 tok/s (calc:409d5856), OFF 24.0 ms = 41.67 tok/s (calc:b518c683). The exact head
saves 1.101x (calc:29cc659f). The memory-speed ceiling with this head is 68.1 to 78.0 tok/s (calc:c66bca93, c52efc2d,
9016f1ef), so the chip runs at about two thirds of it. The rest of the gap is outside the head. Sampled requests are
unchanged (24.6 both), as designed.

Timings were taken while other agents' jobs were stopped after the 03:44Z memory incident (L366). These are not
quiet-measure receipts.
