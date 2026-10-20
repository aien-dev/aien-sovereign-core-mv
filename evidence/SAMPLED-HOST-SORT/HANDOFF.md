# L148 fix: sampled decode host sort (fixer handoff, 2026-10-19)

Branch `fix/sampled-host-sort` in aien-dev/aien-sovereign-core, worktree `~/workspace/fix-sampled-host-sort`.
Ledger L148. Brief `~/handoffs/briefs/L148-sampled-host-sort.md`. Evidence in the branch under `evidence/SAMPLED-HOST-SORT/`.

## Plain English
When the model is asked to pick words with some randomness (temperature above zero), the server was sorting
the whole 248,320-word vocabulary twice for every single word it produced, and doing a heavy math step on all
248,320 scores. Only 20 words are ever allowed to be picked. The fix looks at all the scores once to find the
20 best, then does the math and the random pick on those 20 only. The picked word and its probability are
exactly the same as before, bit for bit. Weird inputs (not-a-number or infinite scores) go through the old
code, so nothing changes for them.

## Symptom
Live 27B server: 28.18 ms/token outside the GPU wait when sampling (17% of wall) against 1.38 ms/token greedy
(omega `evidence/CHAIN-TO-GPU-1/host-gap.sh`).

## Cause
`top_k_mask` full-sorted 248,320 indices and copied a 248,320 vector, then `sample_with_params_traced` ran
`exp` in f64 over all 248,320 and full-sorted all of them again for the top_p cut. PR #646 replaced only the
first sort with a selection.

## Fix
- `sample_top_k_traced` in `crates/aien-inference-abi/src/transformer_backend.rs`: one O(V) scan keeps the k
  best (higher logit first, lower token id on ties), then softmax, top_p cut and draw over those k.
- The softmax sum adds the k terms in ascending token id, which is the order the old full pass added its
  non-zero terms (masked terms were exact zeros), so sum, probabilities, token, log-probability and trace are
  bit-identical.
- Returns `None` (nothing touched) for: NaN or +-inf maximum, a top_p cut that never reaches top_p, a draw that
  falls off the end, temperature <= 0.001, k = 0 or k >= vocab. `SamplingObserver::sample`
  (`crates/aien-serve-http/src/sampling.rs`) then uses #646's `top_k_mask` plus the old sampler.
- Not touched: the greedy path, the no-top_k path, the live server, the GPU.

## Stacking with #646
This branch is built ON TOP of `origin/perf/623-topk-select` (#646), one commit above it (a553cd4c). Land
#646 first, then this. It reuses #646's `top_k_mask` as the fallback and its `top_k_mask_full_sort` as the
reference in tests. Pushed to its own branch; the #646 branch was not modified.

## Evidence
- Tests (`evidence/SAMPLED-HOST-SORT/tests-green.txt`): 13 passed, 0 failed in `sampling::`. New:
  `fast_draw_matches_old_draw_at_full_vocabulary` (vocab 248,320, k 20, 3 logit shapes incl. a confident head,
  5 temperatures 0.002..2.5, 6 top_p values 0, 0.3, 0.9, 0.95, 0.9999, 1, 4 seeds incl. 0 and u64::MAX: token,
  log-probability bits, kept count, top_prob bits and the observation counters all equal to the old path),
  `fast_draw_matches_old_draw_random_ties_and_masks` (400 rounds of random, tie-heavy with +-0.0, -inf banned
  tokens, all-equal vectors, vocab 2..2048, every k), `fast_draw_declines_what_it_cannot_match`,
  `observer_with_nan_logits_keeps_the_old_answer`.
- Negative control (`tests-red-mutant.txt`): tie rule deliberately broken in the scan, same tests.
- Microbenchmark (`bench-release.txt`, release build, CPU only, one pinned core, box busy: load ~20 on 20
  cores, so absolute numbers are noisy; median of 101 calls, V=248,320, k=20, one call = mask + sampler):

  | T / top_p | old (as on main) | #646 (select + full sampler) | new |
  |---|---|---|---|
  | 1.0 / 0.95 | 31878.7 us | 7984.3 us | 165.6 us |
  | 0.7 / 1.0 | 28780.2 us | 6075.5 us | 165.4 us |
  | 0.7 / 0.8 | 32421.4 us | 8251.4 us | 163.9 us |

  The serve-log figure (28.18 ms/token) is the old row. Not measured: the live server end to end (forbidden
  by the brief); expect the host share to fall to the greedy figure plus about 0.17 ms, to be confirmed by
  re-running `host-gap.sh` after deploy.

## Limits
- Rebuilt-in-release tests passed on first run; the old path remains in the tree as the fallback and the
  reference, so the equality test keeps guarding it.
- Remaining per-call cost is one pass over 248,320 f32 logits (about 165 us here).
