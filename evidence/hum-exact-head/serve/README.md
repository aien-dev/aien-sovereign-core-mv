# L314 exact two-pass head in the served 35B (2026-10-21)

Drake 02:25Z: "turn on the quick levers anyway even if it is not faster. Just turn it on." The speed gate is dropped and
the exact-words gate stays.

## Plain English

The served 35B now picks each word in two passes when the request is greedy (temperature 0). First a rough half-size
copy of the scoring table scores every word, with a proven worst-case error. Then the exact table rescores only the
words that could still win. The chosen word is the same as before, by construction. Sampled requests (temperature
above 0) still use the full table, because they need every score.

Result: the served words are identical to the golden and to the same binary with the lever off. A deliberately broken
build fails the test.

## Binary

`~/.local/state/aien-q36/bin-e1f02b29` (`BUILD-bin-e1f02b29.txt`, `SHA256SUMS-bin-e1f02b29`):
- sc `hum/exact-head-serve` e1f02b29, on motor/serve 06d1ce85, the base of the serving binary bin-5b726818.
- omega `hum/exact-head` 844509b5, on q36/serve 739c80e8. The lib is built against physics at physics.lock 6d7cf0d4,
  and only the 4 changed objects differ from bin-5b726818's lib.
- `AIEN_Q36_EXACT_HEAD=0` turns the lever off. `AIEN_Q36_TEST_MUTANT=5` is the red control: it inverts the MASK, so
  candidates get -inf and the other rows keep their logits.

## Gate (`serve-gate.sh`, port 18093, 18091 never touched)

| run | gate2-http.sh (24 ids vs host golden gen-chat-max-24) | extra requests |
|---|---|---|
| ON (default) | PASS (`gate2-l314-exact-on.json`) | 4 chat prompts x 128 greedy + 1 sampled (t 1.0, seed 7) |
| MUTANT 5 | FAIL, as required (`gate2-l314-exact-mutant5.json`) | |
| OFF (`AIEN_Q36_EXACT_HEAD=0`, same binary) | PASS (`gate2-l314-exact-off.json`) | the same 5 requests |

ON ids == OFF ids on 5 of 5 requests (`ask-on-*.json` vs `ask-off-*.json`). VERDICT PASS (`serve-gate.txt`).
Mutant 5 failing also shows that greedy requests really run the MASK path.

`attempt1-stale-physics/`: the first pin could not load (`gdn alloc: marker wait timed out`). Its omega lib had been
built with `PHYSICS_LOCK_CHECK=0` against a stale physics checkout that lacks #28 (GB10 GPFIFO/pushbuffer GPU-uncached).
Rebuilt at the pin, it loads. `server-*-fail*.log`: start attempts that hit memory or channel limits while other
agents' 35B jobs were loading (status 0x51). The script waits for memory and retries.

## Timing (first look, not a controlled measurement)

These numbers come from the server's own per-request line, "graph chip ms/token". They are single requests on a shared
chip, taken at different times, so other agents' jobs were running (the ON requests waited 40 to 88 s for the chip lock).

| greedy request | ON chip ms/token | OFF chip ms/token |
|---|---|---|
| gate2 (24 tok) | 20.6 | 23.8 |
| prompt 0 (128) | 20.5 | 23.7 |
| prompt 1 (128) | 21.6 | 23.9 |
| prompt 2 (128) | 21.6 | 24.1 |
| prompt 3 (128) | 22.0 | 23.9 |

Median: ON 21.6 ms = 46.3 tok/s (calc:f106255f), OFF 23.9 ms = 41.84 tok/s (calc:7da2444f), ratio 1.106
(calc:85294dbc). The memory-speed ceiling with this head is 68.1 to 78.0 tok/s (README above: calc:c66bca93, c52efc2d,
9016f1ef). A controlled run under quiet-measure is still to do.
