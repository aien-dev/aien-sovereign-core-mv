# L356 lever 1 ON: a|b beside mix overlap on the served 35B (2026-10-21)
Drake 02:25Z: turn the quick levers on; no speed gate; exact-words gate stays.
- Build bin-0915000b (sc levers/q36-quick 0915000b, omega lib 678df626 unchanged): AIEN_OMEGA_Q36_OVERLAP_AB=1 by default.
- gate2-http.sh on 18093: ids identical to the host golden gen-chat-max-24 -> PASS (gate2-l356-overlap-ab-0915000b.json).
- Negative control, same binary with AIEN_Q36_TEST_MUTANT=4: ids DIFFER -> FAIL (gate2-l356-mutant4-0915000b.json). The gate can fail.
- Extra: 200-token greedy chat reply, live 18091 (bin-5b726818, overlap off) vs 18093 (bin-0915000b): token ids identical (long200-*.json).
- Speed: not measured yet (Drake: time it afterwards). One request in serve-18093.log reads chip 23.2 ms/token at 24 decode tokens,
  not a timing (chip shared with the live server, no window).

# L356 lever 2 ON: greedy replies take the chip's token (bin-d839a8ca, 2026-10-21 ~03:50Z)
- No logits copy (1 MB) and no host argmax per token on greedy replies; first token still from the prefill logits; AIEN_SERVE_CHIP_GREEDY=0 off.
- gate2-http.sh on 18096: PASS (gate2-l356-chip-greedy-d839a8ca.json). Mutant 4, same binary: FAIL (gate2-l356-mutant4-d839a8ca.json).
- 200-token greedy reply identical to the live bin-5b726818 reply (long200-bin-d839a8ca.json == long200-live-bin-5b726818.json).
- Sampled request (temperature 0.7) still served through the host sampler (log: chip greedy false).
- Host gap per token (wall - graph chip, one request each, NOT a timing): ~0.3 ms (23.3 - 23.0, 23.6 - 23.4) vs ~1.5 ms on bin-0915000b (24.7 - 23.2).
