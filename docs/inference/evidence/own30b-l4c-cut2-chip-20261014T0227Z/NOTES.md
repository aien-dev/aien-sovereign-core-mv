# W-L4c cut 2: suffix-index drafter on the chip (HARDWARE, GB10, 2026-10-14)
Build: sc own30b/spec-wp 9d89951, omega own30b/rc d0889a94 archive 6a6cb4df (dev override). Commands/hashes in *.start.
suffix.raw: --drafter suffix --suffix-global (past = other prompts' spec-off outputs) --runs 2. lookup.raw: prompt-lookup baseline, --runs 1.
Both: --verify both --wp-prefill, golden prompts p1-p4 at 64 tokens + p5-code-json (p5-code-json.txt, 506 prompt tokens, 256 new).
Both RESULT PASS; negative-control mutants caught; decode-rounding verify token-identical to spec-off in every run.

tok/s (spec-off -> on), tokens per round, acceptance; wp = whole-prompt verify, dec = exact decode-rounding verify:
| prompt | off | suffix wp | suffix dec | lookup wp | lookup dec | suffix tok/round, accept | lookup tok/round, accept |
| p1 | 37.7 | 47.5-47.9 (diverges @49) | 31.8-32.8 | 58.3 (diverges @14) | 25.8 | 2.27, 0.86 | 3.42, 0.35 (dec) |
| p2 | 36.7 | 36.3-36.5 | 24.2-24.6 | 25.6 | 11.2 | 1.50, 0.25 | 1.67, 0.08 |
| p3 | 36.2 | 38.7 (diverges @24) | 28.0-28.4 | 32.1 (diverges @7) | 12.2 | 1.79, 0.52 | 1.80, 0.11 |
| p4 | 35.9 | 34.0 (diverges @17) | 22.7 | 30.9 (diverges @11) | 11.1 | 1.20, 0.14 | 1.40, 0.05 |
| p5 code/JSON | 35.5 | 77.7-78.1 (IDENTICAL) | 35.5-35.7 | 79.1 (IDENTICAL) | 29.0 | 4.90, 0.86 | 6.10, 0.65 |
Verify ms per round (wp): suffix 32-47 (shorter, confident drafts), lookup 55-61 (always 8).
Whole-prompt identity vs spec-off: suffix 4/10, lookup 2/5 (cause: RC archive, see cut-1 NOTES).
Host replay (own30b_spec_sim, NOT hardware) reproduces the decode-path round/accept counts exactly; its tok/s projection is ~5% optimistic.
