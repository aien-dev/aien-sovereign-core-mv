# L356 lever 1 ON: a|b beside mix overlap on the served 35B (2026-10-21)
Drake 02:25Z: turn the quick levers on; no speed gate; exact-words gate stays.
- Build bin-0915000b (sc levers/q36-quick 0915000b, omega lib 678df626 unchanged): AIEN_OMEGA_Q36_OVERLAP_AB=1 by default.
- gate2-http.sh on 18093: ids identical to the host golden gen-chat-max-24 -> PASS (gate2-l356-overlap-ab-0915000b.json).
- Negative control, same binary with AIEN_Q36_TEST_MUTANT=4: ids DIFFER -> FAIL (gate2-l356-mutant4-0915000b.json). The gate can fail.
- Extra: 200-token greedy chat reply, live 18091 (bin-5b726818, overlap off) vs 18093 (bin-0915000b): token ids identical (long200-*.json).
- Speed: not measured yet (Drake: time it afterwards). One request in serve-18093.log reads chip 23.2 ms/token at 24 decode tokens,
  not a timing (chip shared with the live server, no window).
