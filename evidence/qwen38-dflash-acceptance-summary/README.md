# DFlash 2 drafter vs the MTP head: accepted words per verify pass (Qwen3.8-27B, exact verify path)

Date 2026-10-19/20 (UTC), host Spark (GB10), measure = words accepted per verify pass, which is a count, not a
timing, so the shared machine does not change it. Every run checks that the generated tokens are identical to
plain decoding (k = 0). Raw logs and commands are in the evidence folders named below, in this repo.

| prompt | n | DFlash 2 (replay) | DFlash 2 (live) | MTP chain k=7 | tokens identical | evidence |
|---|---|---|---|---|---|---|
| code | 256 | 6.000 | 6.000 | 5.800 | yes | qwen38-dflash-gen3-20261019T2108Z |
| prose | 256 | 2.286 | 2.286 | 2.168 | yes | qwen38-dflash-gen3-20261019T2108Z |
| tune | 256 | 3.507 | 3.507 | 3.480 | yes | qwen38-dflash-gen-20261019T1731Z |
| codelong | 256 | 4.414 | 4.414 | 4.145 | yes | qwen38-dflash-gen4-20261019T2331Z |

Claim strength: measured. Reading: on all four prompts DFlash 2 accepts slightly more words per pass than the MTP
head (0.03 to 0.27 more). The code prompt only had 5 verify rounds, so it is the weakest of the four numbers.
Single run each, no repeats: the counts are deterministic given the model, but the spread across prompts is not
tested beyond these four.

## Controls (a check that cannot go red proves nothing)
- conv-tap swap (--mutant convswap, code n128): words per pass 6.000 -> 1.208, tokens unchanged. Red as required.
- no-rollback (--norollback 1, code n128): live 4.417 vs replay 6.000, tool verdict FAIL. Red as intended.
- no-pairwise selector (--mutant nopair): code n128 stayed at 6.000 (no effect on that prompt, so that run proves
  nothing); tune n256 went 3.507 -> 3.368 (gen5 evidence). Red on tune only.

## Failed attempts, kept as evidence
Runs that hit the GPU out-of-memory error (status 0x51 = NV_ERR_NO_MEMORY) or the memory guard are kept as
`*.refused-tryN.log.txt` in the gen folders. They are not results.

## Not in this table
Chip backend parity (drafter GEMV and dynconv2 on the chip against the host reference): see
evidence qwen38-dflash-chip2-* when present. The dynconv2 kernel chip gate itself passed (omega
evidence/dflash-dynconv2-chip-20261019T2134Z: 18 checks, 0 failed).
