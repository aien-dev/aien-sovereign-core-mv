# L190 pre-registered prediction (written 2026-10-20, before any acceptance run)

Pair: Qwen3.8-27B FP8 (verifier, resident) + Qwen3-30B-A3B-Instruct-2507 FP8 (drafter, experts through the
heartbeat ring). Vocabularies differ (W1 receipt 03-vocab.json), so every round goes through the text
bridge (crates/aien-wormhole/src/align.rs). Texts: the three L080 sets (code-golden 30 words, prose 256,
codelong 256), verifier text = the 27B's own chip greedy output.

## Facts these numbers rest on (measured, not predicted)
- 30B word on the chip with every expert in memory: 1.554 GB dense (attention, router, norms, lm_head)
  + 8 of 128 experts of 28.99 GB = 3.366 GB (calc:f5aec75c) = 12.33 ms at the 27B's measured 273 GB/s
  (calc:bd2c2c9f). One expert = 4.719 MB (calc:2376bed3); a word that misses every expert reads
  48 x 8 of them from the SSD.
- 27B pass at the ceiling: 98.5 ms (docs/WORMHOLE.md section 6).
- Bridge cost per k=4 block on the host: 3.4 to 6.6 microseconds (03-vocab.json), under 0.01% of a pass.
- On these texts every 30B token boundary is also a 27B boundary (share 1.000); 94.3% of 27B boundaries
  are 30B boundaries. The 27B's chip ids equal a fresh cut of its own text on all three continuations.
- 27B exact multi-row verify (sc cascade/mtp-chain f21b8e88, 8 layers): m=1 21.7 ms, m=5 67.7 ms, decode
  graph one token 30.2 ms. A k=4 round (m=5) on that path costs 67.7/30.2 decode tokens, not 1.

## Prediction (guesses, to be scored)
Forward (30B drafts, 27B verifies), accepted / drafted in 27B tokens and words per round at k=4:
- code-golden: top-1 text agreement 85%, k=4 about 3.5 words per round.
- codelong: top-1 agreement 75%, k=4 about 3.0 words per round.
- prose: top-1 agreement 55%, k=4 about 2.0 words per round.
(Flash-Next, same family and vocabulary as the 27B, measured 97% / 89% / 71%; the 30B is an older
family with another tokenizer, so lower.)

Reverse (27B drafts, 30B verifies): the 30B's agreement with the 27B text about the same as forward.

Output identical to the verifier alone: 100% in both directions (the rule makes it so; a mismatch is a FAIL
to be found and fixed).

## Speed prediction by physics (verdicts W4 must confirm or refute)
- Forward, bandwidth model (a k-row check costs one pass): break-even E > 1.125 at k=1 (calc:f11ac297),
  E > 1.501 at k=4 (calc:e7a6e363). Predicted PASS on codelong and code-golden, marginal PASS on prose,
  only if every 30B expert is in memory.
- Forward, measured exact-rows cost (k=4 round = 67.7/30.2 decode tokens): break-even E > 2.74 at k=4
  (calc:f736a486).
  Predicted FAIL on prose, about even on codelong, PASS on code-golden.
- Forward with a partial expert cache: each miss reads 4.719 MB from the SSD at 10 GB/s (0.47 ms) and
  stalls (the router picks layer L's experts after layer L-1); predicted FAIL below about 0.9 hit rate.
- Reverse: every guess costs a whole 27B pass (98.5 ms) while the 30B alone costs 12.33 ms a word, so the
  reversed pair is slower than the 30B alone at any acceptance. Predicted FAIL by physics.
